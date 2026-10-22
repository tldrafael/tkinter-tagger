#!/usr/bin/env python3
"""Web mask tagger: compare experiment masks, log a choice, or paint a composite.

Same job as ``compose_strong_label_v2.py``. Walk a list of images, compare the
masks derived by replacing ``/im/`` with ``/results/<experiment>/`` (extension
becomes ``.png``), and tag the winner. Tags are appended as
``<image path>,<label>`` and the next launch resumes after the last log line.
``odd``, ``all bad``, and ``Skip`` match the Tk tagger (Skip does not write a
log line). Compose picks a base mask and paints regions from a donor with
Replace / Union / Intersect, plus eraser and pencil. Saving writes an ``L``
PNG to ``/results/compose.dt<YYYYMMDD>/`` and logs that folder name.

The page is the local viewer from ``view_experiments.py``: one item at a time,
synced zoom and pan, background swatches, and show/hide for source, blend, and
mask. It is served on loopback only. The output directory and port stay fixed
across runs, so the URL can be bookmarked. A previous server on that port is
replaced.

The browser opens immediately. Tiles are rendered in the background, starting
at the resume image and then the next few, so tagging can start before the
rest of the list is ready. A truncated JPEG is decoded from the bytes it has
instead of aborting the run.

Usage:
    python3 compose_strong_label_view.py images.txt --experiments expA,expB
    python3 compose_strong_label_view.py --experiments expA,expB --images-file images.txt
    python3 compose_strong_label_view.py images.txt --experiments expA,expB --log labels.txt

Keys on the comparison page: ``1``–``9`` trigger the numbered buttons,
``C`` opens compose, arrows / ``n`` / ``p`` move between images (or pan when
zoomed). In the editor: drag paints, Space-drag or right-drag pans, ``E``
eraser, ``P`` pencil, ``S`` save, ``R`` reset, Cmd/Ctrl-Z undo, Esc cancels.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from datetime import date
from urllib.parse import parse_qs, quote, urlparse

from PIL import Image, ImageFile

GREEN = (0, 255, 0)
PORT_DEFAULT = 8766
OUT_DIR_DEFAULT = os.path.join(tempfile.gettempdir(), "bgrm_compose")
DEFAULT_LOG = "labels.txt"
_NAME_MAX = 255
_MAX_UNDO = 50
_MAX_EDGE = 20000
_MAX_BODY = _MAX_EDGE * _MAX_EDGE
_HIDDEN = {"/session.json", "/server.log"}
_BRUSH_MODES = ("Replace", "Union", "Intersect")
_FIXED_LABELS = ("odd", "all bad")
_LOOKAHEAD = 4

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow
    _LANCZOS = Image.LANCZOS

SESSION = None
PREPARER = None
_IO_LOCK = threading.Lock()


def mask_path_for(image_path: str, experiment: str) -> str:
    """``.../im/name.jpg`` -> ``.../results/<experiment>/name.png``."""
    replaced = image_path.replace("/im/", f"/results/{experiment}/")
    return replaced.rsplit(".", 1)[0] + ".png"


def compose_output_path(image_path: str, tag: str) -> str:
    """Absolute PNG path for a composed mask. Refuses to overwrite the source."""
    image_path = os.path.abspath(image_path)
    if not re.fullmatch(r"compose\.dt\d{8}", tag or ""):
        raise ValueError(f"invalid compose tag: {tag!r}")
    if "/im/" not in image_path:
        raise ValueError(f"image path has no /im/ segment: {image_path}")
    out = os.path.abspath(mask_path_for(image_path, tag))
    if out == image_path:
        raise ValueError("refusing to overwrite the source image")
    if f"/results/{tag}/" not in out:
        raise ValueError(f"unexpected compose output path: {out}")
    if os.path.basename(os.path.dirname(out)) != tag:
        raise ValueError(f"unexpected compose directory: {out}")
    return out


def make_compose_tag(date_str: str) -> str:
    if not re.fullmatch(r"\d{8}", date_str or ""):
        raise ValueError(f"compose date must be YYYYMMDD, got {date_str!r}")
    return f"compose.dt{date_str}"


def read_image_list(path: str) -> list:
    paths = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line)
    return paths


def read_log(log_file: str, items: list) -> tuple:
    """Return ``(labels, resume_index)`` aligned with ``items``.

    ``items`` entries need ``log`` (the string written to the file) and ``abs``.
    Resume is the image after the last recognized log line, clamped so a
    finished list reopens on its last image. Matches the Tk tagger.
    """
    labels = [None] * len(items)
    if not items:
        return labels, 0
    keys = {}
    for index, item in enumerate(items):
        keys[item["log"]] = index
        keys[item["abs"]] = index
    if not log_file or not os.path.exists(log_file):
        return labels, 0

    last_index = None
    with _IO_LOCK:
        with open(log_file, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                image, sep, label = line.partition(",")
                if not sep:
                    image, label = line, ""
                index = keys.get(image)
                if index is None and os.path.isabs(image):
                    index = keys.get(os.path.normpath(image))
                if index is None:
                    continue
                last_index = index
                labels[index] = label or None
    if last_index is None:
        return labels, 0
    return labels, min(last_index + 1, len(items) - 1)


def _item_name(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-")


def _cache_filename(stem: str, suffix: str) -> str:
    name = f"{stem}{suffix}"
    if len(name.encode("utf-8")) <= _NAME_MAX:
        return name
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
    tail = f"_{digest}{suffix}"
    budget = _NAME_MAX - len(tail.encode("utf-8"))
    prefix = stem.encode("utf-8")[:budget].decode("utf-8", "ignore").rstrip("-_")
    return f"{prefix}{tail}"


def _relative_url(path: str, out_dir: str) -> str:
    relative = os.path.relpath(os.path.abspath(path), os.path.abspath(out_dir))
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise ValueError(f"viewer asset escapes output directory: {path}")
    return quote(relative.replace(os.sep, "/"), safe="/")


def _prune_managed_assets(directory: str, keep_paths) -> None:
    keep = {os.path.abspath(path) for path in keep_paths}
    if not os.path.isdir(directory):
        return
    for entry in os.scandir(directory):
        path = os.path.abspath(entry.path)
        if path not in keep and (entry.is_file(follow_symlinks=False) or entry.is_symlink()):
            os.unlink(path)


def _load_image(path: str) -> Image.Image:
    """Open and decode an image.

    A truncated file is retried once with Pillow's partial decode so one short
    JPEG does not fail the render.
    """
    def _read() -> Image.Image:
        with Image.open(path) as image:
            image.load()
            return image.copy()

    try:
        return _read()
    except OSError as exc:
        if ImageFile.LOAD_TRUNCATED_IMAGES or "truncated" not in str(exc).lower():
            raise
        print(f"warning: truncated image, using partial decode: {path}", flush=True)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return _read()


def _render_cached(out_path: str, deps, render) -> bool:
    temporary = None
    try:
        if (os.path.exists(out_path) and not os.path.islink(out_path) and all(
                os.path.getmtime(out_path) >= os.path.getmtime(dep) for dep in deps)):
            return True
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        image = render()
        directory = os.path.dirname(out_path)
        fd, temporary = tempfile.mkstemp(prefix=".render-", suffix=".png", dir=directory)
        os.close(fd)
        image.save(temporary, format="PNG", compress_level=1)
        os.replace(temporary, out_path)
        temporary = None
        return True
    except Exception as exc:
        print(f"warning: failed to render {out_path}: {exc}")
        return False
    finally:
        if temporary and os.path.lexists(temporary):
            os.unlink(temporary)


def load_tagger_mask(mask_path: str, size: tuple) -> Image.Image:
    """Grayscale mask. RGBA/LA files contribute their alpha, same as the Tk tagger."""
    image = _load_image(mask_path)
    if image.mode in ("RGBA", "LA"):
        mask = image.getchannel("A")
    else:
        mask = image.convert("L")
    if mask.size != size:
        mask = mask.resize(size, _LANCZOS)
    return mask


def render_source(image_path: str) -> Image.Image:
    """RGB pixels in file order, so the browser cannot apply a different EXIF rotation."""
    return _load_image(image_path).convert("RGB")


def blend_on_green(image_path: str, mask_path: str) -> Image.Image:
    src = _load_image(image_path).convert("RGB")
    mask = load_tagger_mask(mask_path, src.size)
    background = Image.new("RGB", src.size, GREEN)
    return Image.composite(src, background, mask)


def _image_size(path: str) -> tuple:
    """Read the header size. A truncated file is retried once, without decoding pixels."""
    def _size() -> tuple:
        with Image.open(path) as image:
            return image.size

    try:
        return _size()
    except OSError as exc:
        if ImageFile.LOAD_TRUNCATED_IMAGES or "truncated" not in str(exc).lower():
            raise
        print(f"warning: truncated image, using partial decode: {path}", flush=True)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return _size()


def render_mask(image_path: str, mask_path: str) -> Image.Image:
    return load_tagger_mask(mask_path, _image_size(image_path))


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "--rgba needs OpenCV (cv2), which is not installed"
        ) from exc
    return cv2


def _read_rgb_alpha(mask_path: str):
    cv2 = _require_cv2()
    arr = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {mask_path}")
    if arr.ndim == 2:
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return rgb, arr
    if arr.shape[2] == 4:
        return arr[:, :, :3].copy(), arr[:, :, 3].copy()
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), None


def rgb_times_alpha(mask_path: str) -> Image.Image:
    rgb, alpha = _read_rgb_alpha(mask_path)
    if alpha is None:
        return Image.fromarray(rgb)
    fg = Image.fromarray(rgb)
    bg = Image.new("RGB", fg.size, (0, 0, 0))
    return Image.composite(fg, bg, Image.fromarray(alpha))


def alpha_of(mask_path: str) -> Image.Image:
    rgb, alpha = _read_rgb_alpha(mask_path)
    if alpha is None:
        return Image.fromarray(rgb).convert("L")
    return Image.fromarray(alpha)


def has_alpha(mask_path: str) -> bool:
    def _mode(image: Image.Image) -> bool:
        return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info

    try:
        with Image.open(mask_path) as image:
            return _mode(image)
    except OSError as exc:
        if ImageFile.LOAD_TRUNCATED_IMAGES or "truncated" not in str(exc).lower():
            raise
        print(f"warning: truncated image, using partial decode: {mask_path}", flush=True)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(mask_path) as image:
            return _mode(image)


def _pair_stem(image_path: str, experiment: str) -> str:
    name = _slug(_item_name(image_path)) or "image"
    exp = _slug(experiment) or "exp"
    digest = hashlib.sha1(experiment.encode("utf-8")).hexdigest()[:8]
    return f"{name}__{exp}_{digest}"


def _source_cache_path(source_dir: str, image_path: str) -> str:
    stem = _slug(_item_name(image_path)) or "image"
    digest = hashlib.sha1(os.path.abspath(image_path).encode("utf-8")).hexdigest()[:12]
    return os.path.join(source_dir, _cache_filename(f"{stem}_{digest}", ".png"))


def _json_for_script(obj) -> str:
    text = json.dumps(obj, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _list_token(items: list) -> str:
    digest = hashlib.sha1()
    for item in items:
        digest.update(item["log"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _blank_outs(experiments) -> list:
    return [{"m": experiment, "src": None, "mask": None} for experiment in experiments]


def prepare_image(item: dict, experiments, out_dir: str, rgba: bool, optional) -> tuple:
    """Render one list entry. A source that still will not decode stays in the list."""
    raw = item["log"]
    abs_path = item["abs"]
    name = _item_name(abs_path)
    keeps = {"source": set(), "blend": set(), "mask": set()}
    outs = _blank_outs(experiments)
    source_dir = os.path.join(out_dir, "source")
    source_path = _source_cache_path(source_dir, abs_path)
    if not _render_cached(source_path, [abs_path], lambda p=abs_path: render_source(p)):
        print(f"warning: unreadable image, leaving a placeholder: {raw}", flush=True)
        return {"source": None, "outs": outs, "error": "unreadable image"}, keeps
    keeps["source"].add(source_path)
    optional = set(optional or ())

    for index, experiment in enumerate(experiments):
        mask_file = mask_path_for(abs_path, experiment)
        stem = _pair_stem(abs_path, experiment)
        src_url = mask_url = None
        try:
            src_url, mask_url = _prepare_experiment(
                abs_path, name, experiment, mask_file, stem, out_dir, rgba, optional, keeps)
        except Exception as exc:
            print(f"warning: failed to prepare {name!r} / {experiment!r}: {exc}")
        outs[index] = {"m": experiment, "src": src_url, "mask": mask_url}
    return {
        "source": _relative_url(source_path, out_dir),
        "outs": outs,
        "error": None,
    }, keeps


def _prepare_experiment(abs_path, name, experiment, mask_file, stem, out_dir, rgba, optional, keeps):
    blend_dir = os.path.join(out_dir, "blend")
    mask_dir = os.path.join(out_dir, "mask")
    src_url = mask_url = None
    if not os.path.isfile(mask_file):
        if experiment not in optional:
            print(f"warning: no prediction for {name!r} / {experiment!r}, expected {mask_file}")
        return None, None
    if rgba and has_alpha(mask_file):
        rgb_path = os.path.join(blend_dir, _cache_filename(stem, "__rgbxa.png"))
        if _render_cached(rgb_path, [mask_file], lambda p=mask_file: rgb_times_alpha(p)):
            keeps["blend"].add(rgb_path)
            src_url = _relative_url(rgb_path, out_dir)
        alpha_path = os.path.join(mask_dir, _cache_filename(stem, ".png"))
        if _render_cached(alpha_path, [mask_file], lambda p=mask_file: alpha_of(p)):
            keeps["mask"].add(alpha_path)
            mask_url = _relative_url(alpha_path, out_dir)
        return src_url, mask_url
    if rgba:
        print(f"warning: {mask_file} has no alpha channel, "
              "falling back to blending the source image")
    blend_path = os.path.join(blend_dir, _cache_filename(stem, ".png"))
    if _render_cached(
            blend_path, [abs_path, mask_file],
            lambda img=abs_path, mask=mask_file: blend_on_green(img, mask)):
        keeps["blend"].add(blend_path)
        src_url = _relative_url(blend_path, out_dir)
    rendered_mask = os.path.join(mask_dir, _cache_filename(stem, ".png"))
    if _render_cached(
            rendered_mask, [abs_path, mask_file],
            lambda img=abs_path, mask=mask_file: render_mask(img, mask)):
        keeps["mask"].add(rendered_mask)
        mask_url = _relative_url(rendered_mask, out_dir)
    return src_url, mask_url


def write_viewer_shell(experiments, image_paths, *, out_dir=None, ncols=None,
                       labels=None, resume=0, compose_tag="", log_file="",
                       token="") -> str:
    """Write the page with every list entry and empty asset URLs.

    Rendering happens later, in the server process, so the browser can open
    before the tiles exist.
    """
    if not experiments:
        raise ValueError("need at least 1 experiment name")
    if not image_paths:
        raise ValueError("need at least 1 image path")
    if ncols is not None and ncols < 1:
        raise ValueError(f"ncols must be >= 1 or None, got {ncols!r}")

    out_dir = os.path.abspath(out_dir or OUT_DIR_DEFAULT)
    for directory in (out_dir,
                      os.path.join(out_dir, "blend"),
                      os.path.join(out_dir, "mask"),
                      os.path.join(out_dir, "source")):
        os.makedirs(directory, exist_ok=True)

    label_by_index = list(labels or [])
    items = []
    for index, item in enumerate(image_paths):
        label = label_by_index[index] if index < len(label_by_index) else None
        items.append({
            "name": _item_name(item["abs"]),
            "path": item["log"],
            "source": None,
            "label": label,
            "outs": _blank_outs(experiments),
            "error": None,
            "ready": False,
        })

    payload = {
        "items": items,
        "models": list(experiments),
        "expCols": ncols,
        "resume": resume,
        "composeTag": compose_tag,
        "token": token,
        "log": log_file,
    }
    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(HTML.replace("__DATA__", _json_for_script(payload)))
    print(f"wrote {out_path} with {len(items)} items; rendering continues in the server",
          flush=True)
    return out_dir


class CatalogPreparer:
    """Fill comparison tiles in the background, resume image first."""

    def __init__(self, out_dir: str):
        session = _session()
        self.out_dir = os.path.abspath(out_dir)
        self.experiments = list(session["experiments"])
        self.rgba = bool(session.get("rgba"))
        self.optional = set(session.get("optional") or ())
        self.images = session["images"]
        self.n = len(self.images)
        focus = int(session.get("resume") or 0)
        self.focus = max(0, min(focus, self.n - 1)) if self.n else 0
        self.ready = [False] * self.n
        self.results = [None] * self.n
        self.done = 0
        self._lock = threading.Lock()
        self._keeps = {"source": set(), "blend": set(), "mask": set()}

    def set_focus(self, index: int) -> None:
        with self._lock:
            if 0 <= index < self.n:
                self.focus = index

    def snapshot(self) -> dict:
        with self._lock:
            items = []
            for ready, result in zip(self.ready, self.results):
                if not ready or result is None:
                    items.append(None)
                else:
                    items.append({
                        "ready": True,
                        "source": result["source"],
                        "outs": result["outs"],
                        "error": result["error"],
                    })
            return {"ok": True, "done": self.done, "total": self.n, "items": items}

    def _next_index(self):
        with self._lock:
            if self.done >= self.n:
                return None
            focus = self.focus
            if not self.ready[focus]:
                return focus
            for offset in range(1, _LOOKAHEAD + 1):
                index = focus + offset
                if index < self.n and not self.ready[index]:
                    return index
            for index in list(range(focus, self.n)) + list(range(0, focus)):
                if not self.ready[index]:
                    return index
            return None

    def run(self) -> None:
        try:
            if self.rgba:
                _require_cv2()
            while True:
                index = self._next_index()
                if index is None:
                    break
                try:
                    entry, keeps = prepare_image(
                        self.images[index], self.experiments, self.out_dir,
                        self.rgba, self.optional,
                    )
                except Exception:
                    traceback.print_exc()
                    raw = self.images[index]["log"]
                    print(f"warning: unreadable image, leaving a placeholder: {raw}", flush=True)
                    entry = {
                        "source": None,
                        "outs": _blank_outs(self.experiments),
                        "error": "unreadable image",
                    }
                    keeps = {"source": set(), "blend": set(), "mask": set()}
                with self._lock:
                    if not self.ready[index]:
                        self.results[index] = entry
                        self.ready[index] = True
                        self.done += 1
                        for key, paths in keeps.items():
                            self._keeps[key].update(paths)
                    done = self.done
                print(f"prepared {done}/{self.n} images", flush=True)
            _prune_managed_assets(os.path.join(self.out_dir, "blend"), self._keeps["blend"])
            _prune_managed_assets(os.path.join(self.out_dir, "mask"), self._keeps["mask"])
            _prune_managed_assets(os.path.join(self.out_dir, "source"), self._keeps["source"])
            print("finished preparing", flush=True)
        except Exception:
            traceback.print_exc()


def start_preparer(out_dir: str) -> None:
    global PREPARER
    if _session().get("rgba"):
        _require_cv2()
    PREPARER = CatalogPreparer(out_dir)
    threading.Thread(target=PREPARER.run, name="bgrm-prepare", daemon=True).start()


def _canonical(image: str) -> dict:
    session = _session()
    item = session["by_log"].get(image)
    if item is None:
        raise ValueError("unknown image")
    return item


def _check_label(label: str) -> None:
    session = _session()
    if not isinstance(label, str) or label not in session["allowed"]:
        raise ValueError("label is not allowed")
    if any(char in label for char in "\r\n"):
        raise ValueError("invalid label")


def _append_log(image: str, label: str) -> None:
    log_file = _session()["log"]
    parent = os.path.dirname(log_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _IO_LOCK:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(f"{image},{label}\n")


def _session() -> dict:
    if not SESSION:
        raise ValueError("viewer session is not loaded")
    return SESSION


def activate_session(session: dict) -> None:
    global SESSION
    session["by_log"] = {item["log"]: item for item in session["images"]}
    session["allowed"] = set(session["allowed_labels"])
    SESSION = session


def write_session(out_dir: str, session: dict) -> None:
    path = os.path.join(out_dir, "session.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in session.items()
                   if key not in ("by_log", "allowed")}, handle)
    activate_session(session)


def load_session(out_dir: str) -> dict:
    path = os.path.join(out_dir, "session.json")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    activate_session(data)
    return data


def handle_catalog() -> dict:
    if PREPARER is None:
        return {"ok": True, "done": 0, "total": 0, "items": []}
    return PREPARER.snapshot()


def handle_focus(body: bytes) -> dict:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("expected an object")
    index = data.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("index is required")
    if PREPARER is not None:
        PREPARER.set_focus(index)
    return {"ok": True}


def handle_resume() -> dict:
    session = _session()
    labels, resume = read_log(session["log"], session["images"])
    return {"ok": True, "resume": resume, "labels": labels}


def handle_tag(body: bytes) -> dict:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("expected an object")
    image = data.get("image")
    label = data.get("label")
    if not isinstance(image, str) or not isinstance(label, str):
        raise ValueError("image and label are required")
    item = _canonical(image)
    _check_label(label)
    _append_log(item["log"], label)
    print(f"tagged {item['log']},{label}", flush=True)
    return {"ok": True, "label": label}


def handle_compose(body: bytes, query: str, headers) -> dict:
    params = parse_qs(query, keep_blank_values=False)
    if "image" not in params or not params["image"]:
        raise ValueError("missing image")
    item = _canonical(params["image"][0])
    try:
        width = int(headers["X-Width"])
        height = int(headers["X-Height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("X-Width and X-Height are required") from exc
    if width < 1 or height < 1 or width > _MAX_EDGE or height > _MAX_EDGE:
        raise ValueError(f"invalid mask size {width}x{height}")
    if len(body) != width * height:
        raise ValueError(f"expected {width * height} bytes, got {len(body)}")
    with Image.open(item["abs"]) as image:
        if image.size != (width, height):
            raise ValueError(
                f"mask is {width}x{height} but image is {image.size[0]}x{image.size[1]}"
            )
    tag = _session()["compose_tag"]
    out_path = compose_output_path(item["abs"], tag)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.frombytes("L", (width, height), body).save(out_path, format="PNG", compress_level=1)
    _append_log(item["log"], tag)
    print(f"saved {out_path}", flush=True)
    print(f"tagged {item['log']},{tag}", flush=True)
    return {"ok": True, "label": tag, "out": out_path}


class _ContainedHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files only when their resolved path remains within the root."""

    def send_head(self):
        parsed = urlparse(self.path)
        if parsed.path in _HIDDEN:
            self.send_error(404, "File not found")
            return None
        root = os.path.realpath(self.directory)
        requested = os.path.realpath(super().translate_path(self.path))
        try:
            contained = os.path.commonpath((root, requested)) == root
        except ValueError:
            contained = False
        if not contained:
            self.send_error(404, "File not found")
            return None
        return super().send_head()

    def end_headers(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html") or parsed.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()


class ComposerHandler(_ContainedHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/resume":
            try:
                self._send_json(200, handle_resume())
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"ok": False, "error": str(exc)})
            return
        if parsed.path == "/api/catalog":
            try:
                self._send_json(200, handle_catalog())
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"ok": False, "error": str(exc)})
            return
        return super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > _MAX_BODY:
            self.close_connection = True
            self._send_json(413, {"ok": False, "error": "payload too large"})
            return
        body = self.rfile.read(length) if length else b""
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/tag":
                result = handle_tag(body)
            elif parsed.path == "/api/compose":
                result = handle_compose(body, parsed.query, self.headers)
            elif parsed.path == "/api/focus":
                result = handle_focus(body)
            else:
                self._send_json(404, {"ok": False, "error": "not found"})
                return
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": str(exc)})
            return
        self._send_json(200, result)

    def _send_json(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _make_server(out_dir: str, port: int):
    handler = functools.partial(ComposerHandler, directory=os.path.abspath(out_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def _pidfile_path(port: int) -> str:
    return os.path.join(tempfile.gettempdir(), f"bgrm_compose_{port}.pid")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _pid_listening_on_port(port: int):
    """Best-effort Linux lookup of the process LISTENing on ``port``."""
    target_hex = f"{port:04X}"
    inodes = set()
    for proc_net in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_net) as handle:
                next(handle)
                for line in handle:
                    fields = line.split()
                    local_addr, state, inode = fields[1], fields[3], fields[9]
                    if state == "0A" and local_addr.split(":")[1] == target_hex:
                        inodes.add(inode)
        except (FileNotFoundError, OSError, IndexError):
            continue
    if not inodes:
        return None
    try:
        pids = [entry for entry in os.listdir("/proc") if entry.isdigit()]
    except OSError:
        return None
    for pid in pids:
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    link = os.readlink(os.path.join(fd_dir, fd))
                except OSError:
                    continue
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    return int(pid)
        except OSError:
            continue
    return None


def _kill_pid(pid: int, port: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        print(f"warning: could not kill previous server (pid {pid}) on port {port}: {exc}")
        return
    for _ in range(30):
        if not _pid_alive(pid):
            print(f"killed previous server (pid {pid}) on port {port}")
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"force-killed previous server (pid {pid}) on port {port}")
    except ProcessLookupError:
        pass


def _kill_pidfile(pidfile: str, port: int) -> None:
    if not os.path.exists(pidfile):
        return
    pid = None
    try:
        pid = int(open(pidfile).read().strip())
    except (ValueError, OSError):
        pid = None
    try:
        os.remove(pidfile)
    except OSError:
        pass
    if pid is not None and _pid_alive(pid):
        _kill_pid(pid, port)


def _kill_existing_server(port: int) -> None:
    """Free ``port`` if a previous compose viewer or experiment viewer holds it."""
    _kill_pidfile(_pidfile_path(port), port)
    _kill_pidfile(os.path.join(tempfile.gettempdir(), f"bgrm_view_{port}.pid"), port)
    fallback = _pid_listening_on_port(port)
    if fallback is not None and _pid_alive(fallback):
        _kill_pid(fallback, port)


def _spawn_background_server(out_dir: str, port: int, open_browser: bool) -> int:
    args = [sys.executable, os.path.abspath(__file__), "--serve-only", out_dir, "--port", str(port)]
    if not open_browser:
        args.append("--no-browser")
    log_path = os.path.join(out_dir, "server.log")
    log_file = open(log_path, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            args, stdout=log_file, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    finally:
        log_file.close()
    pidfile = _pidfile_path(port)
    for _ in range(30):
        if os.path.exists(pidfile):
            break
        if proc.poll() is not None:
            print(f"error: server exited immediately; see {log_path}", file=sys.stderr)
            break
        time.sleep(0.1)
    return proc.pid


def _serve_blocking(out_dir: str, port: int, open_browser: bool) -> None:
    start_preparer(out_dir)
    try:
        httpd = _make_server(out_dir, port)
    except OSError as exc:
        print(f"error: could not bind 127.0.0.1:{port}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    with open(_pidfile_path(port), "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    url = f"http://127.0.0.1:{port}/index.html"
    print(f"serving on {url}")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        try:
            os.remove(_pidfile_path(port))
        except OSError:
            pass


def serve_and_open(out_dir: str, *, port: int = PORT_DEFAULT, open_browser: bool = True,
                   background: bool = True) -> None:
    """Serve ``out_dir`` on loopback. Replaces any previous viewer on ``port``."""
    _kill_existing_server(port)
    if background:
        url = f"http://127.0.0.1:{port}/index.html"
        pid = _spawn_background_server(out_dir, port, open_browser)
        print(f"serving in background (pid {pid}) on {url}")
        return
    _serve_blocking(out_dir, port, open_browser)


def compose_and_view(experiments, image_paths, *, log_file: str = DEFAULT_LOG,
                     compose_date: str = None, port: int = PORT_DEFAULT,
                     open_browser: bool = True, out_dir: str = None,
                     background: bool = True, rgba: bool = False,
                     ncols=None, optional=()) -> str:
    """Build the tagger for ``image_paths`` and serve it. Returns the output dir."""
    if compose_date is None:
        compose_date = date.today().strftime("%Y%m%d")
    compose_tag = make_compose_tag(compose_date)
    optional = set(optional or ())
    log_file = os.path.abspath(log_file)

    items = []
    seen = set()
    for raw in image_paths:
        raw = raw.strip()
        if not raw:
            continue
        abs_path = os.path.normpath(os.path.abspath(raw))
        if not os.path.isfile(abs_path):
            raise SystemExit(f"Image not found: {raw}")
        if "/im/" not in abs_path:
            print(f"warning: {raw} has no /im/ segment; mask lookup and compose save expect one")
        if "," in raw:
            print(f"warning: image path contains a comma; log resume may not round-trip: {raw}")
        if raw in seen:
            print(f"warning: duplicate image path, keeping the first: {raw}")
            continue
        seen.add(raw)
        items.append({"log": raw, "abs": abs_path, "optional": optional})

    if not items:
        raise SystemExit("no image paths given")

    labels, resume = read_log(log_file, items)
    allowed = list(experiments) + list(_FIXED_LABELS) + [compose_tag]
    token = _list_token(items)
    if rgba:
        _require_cv2()
    out_dir = write_viewer_shell(
        experiments, items, out_dir=out_dir, ncols=ncols,
        labels=labels, resume=resume, compose_tag=compose_tag,
        log_file=log_file, token=token,
    )
    write_session(out_dir, {
        "log": log_file,
        "compose_tag": compose_tag,
        "images": [{"log": item["log"], "abs": item["abs"]} for item in items],
        "allowed_labels": allowed,
        "experiments": list(experiments),
        "rgba": bool(rgba),
        "optional": sorted(optional),
        "resume": resume,
    })
    print(f"log: {log_file}")
    print(f"compose tag: {compose_tag}")
    print(f"resume item: {resume + 1}/{len(items)}")
    serve_and_open(out_dir, port=port, open_browser=open_browser, background=background)
    return out_dir


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mask composer</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#272b34; --txt:#e6e8ec; --mut:#8b93a1; --accent:#4f8cff; }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body { margin:0; font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--txt); user-select:none;
         display:flex; flex-direction:column; height:100vh; overflow:hidden; }
  header, #tagbar, #pickbar, #editbar { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
           padding:8px 14px; background:var(--panel); border-bottom:1px solid var(--line); flex:0 0 auto; }
  header { position:sticky; top:0; z-index:5; }
  h1 { font-size:14px; margin:0; font-weight:600; }
  .sp { flex:1; }
  .pill { background:#222732; border:1px solid var(--line); border-radius:999px; padding:3px 10px;
          color:var(--mut); max-width:34vw; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .pill b { color:var(--txt); }
  button { background:#222732; color:var(--txt); border:1px solid var(--line);
           border-radius:7px; padding:6px 10px; cursor:pointer; font-size:13px; }
  button:hover { border-color:var(--accent); }
  button.active { border-color:var(--accent); color:var(--accent); }
  button.primary { background:#243656; border-color:var(--accent); font-weight:600; }
  button.warn { border-color:#8a6a30; }
  button.danger { border-color:#8a3a3a; }
  #tagbtns, #donors, #brushPresets { display:flex; flex-wrap:wrap; gap:4px; }
  #tagbtns button { max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  body.busy button { opacity:.55; pointer-events:none; }
  body.waiting #tagbtns button { opacity:.55; pointer-events:none; }
  .grp { display:flex; align-items:center; gap:4px; }
  .grp label { color:var(--mut); font-size:12px; }
  input[type="number"] { width:64px; background:#222732; color:var(--txt); border:1px solid var(--line);
                         border-radius:7px; padding:4px 6px; font-size:13px; }
  .sw { width:22px; height:22px; border-radius:5px; border:1px solid var(--line); cursor:pointer; padding:0; }
  .sw.on { outline:2px solid var(--accent); outline-offset:1px; }
  .sw.checker { background-color:#fff;
    background-image:linear-gradient(45deg,#ccc 25%,transparent 25%),linear-gradient(-45deg,#ccc 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ccc 75%),linear-gradient(-45deg,transparent 75%,#ccc 75%);
    background-size:10px 10px; background-position:0 0,0 5px,5px -5px,-5px 0; }
  main { padding:14px 16px; margin:0; flex:1 1 auto; min-height:0; overflow:hidden;
         display:flex; flex-direction:column; }
  .lightbox { position:fixed; inset:0; background:rgba(0,0,0,.92); display:none; align-items:center;
              justify-content:center; z-index:50; cursor:zoom-out; }
  .lightbox.on { display:flex; }
  .lightbox img { max-width:96vw; max-height:96vh; object-fit:contain; box-shadow:0 0 40px rgba(0,0,0,.6); }
  .lightbox .cap { position:fixed; top:14px; left:16px; color:#fff; font-size:13px; background:rgba(0,0,0,.5);
                   padding:3px 10px; border-radius:6px; }
  .grid { display:grid; grid-template-columns:repeat(var(--cols, 1), minmax(0,1fr));
          gap:10px; flex:1 1 auto; min-height:0; }
  .tile { background:var(--panel); border:2px solid var(--line); border-radius:10px; overflow:hidden;
          display:flex; flex-direction:column; height:100%; min-height:0; }
  .tile.orig { border-color:#4a4530; }
  .tile.orig .label { background:rgba(231,161,59,.85); color:#111; font-weight:700; }
  .ctrls { display:flex; align-items:center; justify-content:center; gap:6px; padding:7px;
           border-top:1px solid var(--line); flex:0 0 auto; color:var(--mut); }
  .canvas { position:relative; flex:1 1 auto; min-height:0; overflow:hidden; cursor:grab; background-color:#fff;
            touch-action:none;
            background-image:linear-gradient(45deg,#ccc 25%,transparent 25%),linear-gradient(-45deg,#ccc 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ccc 75%),linear-gradient(-45deg,transparent 75%,#ccc 75%);
            background-size:22px 22px; background-position:0 0,0 11px,11px -11px,-11px 0; }
  .canvas.grabbing, body.spacepan .canvas { cursor:grabbing; }
  body[data-mode="edit"] .canvas.paintable { cursor:crosshair; }
  .canvas img, .canvas canvas.stage { position:absolute; image-rendering:auto; }
  .canvas.missing { display:flex; align-items:center; justify-content:center; cursor:pointer; }
  .canvas .miss { color:var(--mut); font-size:13px; text-align:center; padding:8px; }
  .label { position:absolute; top:6px; right:8px; max-width:calc(100% - 16px); overflow:hidden;
           text-overflow:ellipsis; white-space:nowrap; background:rgba(0,0,0,.6); color:#fff;
           border-radius:6px; padding:1px 7px; font-size:12px; pointer-events:none; z-index:2; }
  .brush { position:absolute; border-radius:50%; outline:2px solid #ff3b3b; pointer-events:none;
           z-index:3; display:none; }
  .canvas[data-panel="donor"] .brush, .canvas[data-panel="blend"] .brush { outline-color:#2ee6e6; }
  .canvas.mask { display:none; border-top:1px solid var(--line); }
  #grid.show-mask .canvas.mask { display:block; }
  #grid.show-mask .canvas.mask.missing { display:flex; }
  #grid.hide-blend .canvas.blend { display:none; }
  #grid.hide-source .tile.orig { display:none; }
  footer { display:flex; gap:10px; align-items:center; padding:9px 14px; background:var(--panel);
           border-top:1px solid var(--line); flex-wrap:wrap; flex:0 0 auto; }
  .hint { color:var(--mut); font-size:12px; }
  #status { min-height:1.2em; }
  kbd { background:#222732; border:1px solid var(--line); border-radius:4px; padding:0 5px; }
  body[data-mode="browse"] .pick-only, body[data-mode="browse"] .edit-only,
  body[data-mode="pick"] .browse-only, body[data-mode="pick"] .edit-only,
  body[data-mode="edit"] .browse-only, body[data-mode="edit"] .pick-only { display:none !important; }
  body[data-mode="browse"] #pgrid, body[data-mode="browse"] #cgrid,
  body[data-mode="pick"] #grid, body[data-mode="pick"] #cgrid,
  body[data-mode="edit"] #grid, body[data-mode="edit"] #pgrid { display:none !important; }
</style>
</head>
<body data-mode="browse">
<header>
  <h1 id="title">Mask composer</h1>
  <span class="pill">Item <b id="idx">–</b>/<span id="total"></span></span>
  <span class="pill" id="name"></span>
  <span class="pill" id="tagpill"></span>
  <span class="pill" id="ctag"></span>
  <span class="pill" id="logpill"></span>
  <span class="grp browse-only"><label>bg</label><span id="swatches"></span>
    <input type="color" id="bgpick" value="#7f7f7f" title="custom background" class="sw"></span>
  <span class="grp"><label>zoom</label>
    <input type="range" id="zoom" min="1" max="36" step="0.25" value="1">
    <span id="zlabel" class="hint">1.0×</span>
    <button type="button" id="zreset">reset</button></span>
  <span class="grp browse-only">
    <button type="button" id="maskToggle" title="show/hide mask row">Hide mask</button>
    <button type="button" id="blendToggle" title="show/hide blend">Hide blend</button>
    <button type="button" id="sourceToggle" title="show/hide source tile">Hide source</button>
  </span>
  <span class="sp"></span>
</header>
<div id="tagbar" class="browse-only"><span id="tagbtns"></span></div>
<div id="pickbar" class="pick-only">
  <strong>Click a mask, or press 1–9, to use it as the base.</strong>
  <span class="sp"></span>
  <button type="button" id="cancelPick">Cancel</button>
</div>
<div id="editbar" class="edit-only">
  <span class="pill" id="baseName"></span>
  <span class="grp"><label>donor</label><span id="donors"></span></span>
  <span class="grp"><label>brush</label><span id="brushPresets"></span>
    <input id="brushSize" type="number" min="1" max="500" value="30" title="brush diameter in screen pixels"></span>
  <button type="button" id="eraserBtn">Eraser (E)</button>
  <button type="button" id="pencilBtn">Pencil (P)</button>
  <span class="grp"><label>mode</label>
    <button type="button" class="modeBtn active" data-mode="Replace">Replace</button>
    <button type="button" class="modeBtn" data-mode="Union">Union</button>
    <button type="button" class="modeBtn" data-mode="Intersect">Intersect</button>
  </span>
  <span class="sp"></span>
  <button type="button" id="undoBtn">Undo</button>
  <button type="button" id="resetBtn">Reset</button>
  <button type="button" id="cancelEdit">Cancel</button>
  <button type="button" id="saveBtn" class="primary">Save</button>
</div>
<main>
  <div class="grid" id="grid"></div>
  <div class="grid" id="pgrid"></div>
  <div class="grid" id="cgrid"></div>
</main>
<div class="lightbox" id="lightbox"><span class="cap" id="lightboxCap"></span><img id="lightboxImg" alt="source full"></div>
<footer>
  <button type="button" id="prev" class="browse-only">‹ Prev</button>
  <button type="button" id="next" class="browse-only">Next ›</button>
  <span id="status" class="hint"></span>
  <span class="sp"></span>
  <span class="hint browse-only"><kbd>1</kbd>–<kbd>9</kbd> tag · <kbd>C</kbd> compose ·
    <kbd>←</kbd><kbd>→</kbd> nav or pan · <kbd>n</kbd><kbd>p</kbd> prev/next ·
    wheel/<kbd>+</kbd><kbd>-</kbd> zoom · <kbd>0</kbd> reset · drag pan</span>
  <span class="hint pick-only"><kbd>1</kbd>–<kbd>9</kbd> choose base · wheel zoom · drag pan · <kbd>Esc</kbd> cancel</span>
  <span class="hint edit-only">drag paint · right-drag or <kbd>space</kbd>-drag pan · wheel zoom ·
    <kbd>E</kbd> eraser · <kbd>P</kbd> pencil · <kbd>S</kbd> save · <kbd>R</kbd> reset ·
    <kbd>⌘Z</kbd> undo · <kbd>Esc</kbd> cancel</span>
</footer>
<script>
"use strict";
const DATA = __DATA__;
const MAX_UNDO = 50;
const TITLES = {browse:"Mask composer", pick:"Choose base mask", edit:"Compose mask"};
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
let i = 0;
let viewMode = "browse";
let zoom = 1, focal = {x:0.5, y:0.5};
let busy = false;
let picking = false;
let loadGen = 0;
let spaceDown = false;
const BGS = [["checker",""],["#ffffff","#ffffff"],["#000000","#000000"],
             ["#7f7f7f","#7f7f7f"],["#ff00ff","#ff00ff"],["#00ff00","#00ff00"]];
let bg = localStorage.getItem("bgrm_compose_bg") || "checker";
const maskStored = localStorage.getItem("bgrm_compose_showmask");
let showMask = maskStored === null ? true : maskStored === "1";
let showBlend = localStorage.getItem("bgrm_compose_showblend") !== "0";
let showSource = localStorage.getItem("bgrm_compose_showsource") !== "0";
let brushScreen = clamp(parseInt(localStorage.getItem("bgrm_compose_brush") || "30", 10) || 30, 1, 500);
let brushMode = localStorage.getItem("bgrm_compose_bmode") || "Replace";
if (brushMode !== "Replace" && brushMode !== "Union" && brushMode !== "Intersect") brushMode = "Replace";
let eraser = false, pencil = false;
let donorIdx = -1, baseIdx = -1;
let imgW = 0, imgH = 0;
let composed = null, base = null, srcRGB = null, donors = [];
let undoStack = [];
let panels = {};
let stroking = false, lastPt = null, panning = false;
let dirty = null, rafId = 0;
let ACTIONS = [];
let holdStatus = false;
let pollTimer = 0;
DATA.prepareDone = 0;
DATA.prepareTotal = DATA.items.length;

function typingTarget(e){
  const t = e.target;
  return !!(t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable));
}
function setStatus(text, title){
  holdStatus = !!text;
  const el = document.getElementById("status");
  el.textContent = text || "";
  el.title = title || "";
}
function setProgress(text){
  if (holdStatus) return;
  const el = document.getElementById("status");
  el.textContent = text || "";
  el.title = "";
}
function syncWaiting(){
  const it = currentItem();
  const waiting = viewMode === "browse" && !!it && !it.ready;
  document.body.classList.toggle("waiting", waiting);
  const done = DATA.prepareDone || 0;
  const total = DATA.prepareTotal != null ? DATA.prepareTotal : DATA.items.length;
  if (waiting || (done < total && !holdStatus)) setProgress("Preparing " + done + "/" + total);
  else if (!holdStatus && done >= total) setProgress("");
}
function reportFocus(){
  fetch("/api/focus", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({index: i}),
  }).catch(() => {});
}
function applyCatalog(data){
  DATA.prepareDone = data.done || 0;
  DATA.prepareTotal = data.total != null ? data.total : DATA.items.length;
  let becameReady = false;
  (data.items || []).forEach((update, n) => {
    const item = DATA.items[n];
    if (!item || !update || item.ready || !update.ready) return;
    item.source = update.source;
    item.outs = update.outs || item.outs;
    item.error = update.error || null;
    item.ready = true;
    if (n === i) becameReady = true;
  });
  syncWaiting();
  if (becameReady && viewMode === "browse") render();
  if (DATA.prepareDone < DATA.prepareTotal) schedulePoll();
}
function schedulePoll(){
  if (pollTimer) return;
  pollTimer = setTimeout(pollCatalog, 500);
}
async function pollCatalog(){
  pollTimer = 0;
  try {
    applyCatalog(await api("/api/catalog"));
  } catch (err) {
    schedulePoll();
  }
}
function setBusy(on){ busy = on; document.body.classList.toggle("busy", on); }
function setIndex(n){
  i = clamp(n, 0, Math.max(0, DATA.items.length - 1));
  sessionStorage.setItem("bgrm_compose_idx", String(i));
  holdStatus = false;
  reportFocus();
  syncWaiting();
}
function viewOf(canvas){ return canvas.querySelector("img, canvas.stage"); }
function natSize(el){
  if (!el) return [0, 0];
  if (el.naturalWidth) return [el.naturalWidth, el.naturalHeight];
  return [el.width || 0, el.height || 0];
}
function sizeCanvas(canvas){
  const el = viewOf(canvas); if (!el) return;
  const [nw, nh] = natSize(el); if (!nw || !nh) return;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (cw < 2 || ch < 2) return;
  const baseScale = Math.min(cw / nw, ch / nh);
  const w = nw * baseScale * zoom, h = nh * baseScale * zoom;
  el.style.width = w + "px"; el.style.height = h + "px";
  el.style.left = (w <= cw ? (cw - w) / 2 : focal.x * (cw - w)) + "px";
  el.style.top = (h <= ch ? (ch - h) / 2 : focal.y * (ch - h)) + "px";
}
function applyZoom(){
  zoom = clamp(zoom, 1, 36);
  if (zoom === 1) focal = {x:0.5, y:0.5};
  document.querySelectorAll(".canvas").forEach(sizeCanvas);
  const slider = document.getElementById("zoom");
  slider.value = zoom;
  document.getElementById("zlabel").textContent = zoom.toFixed(1) + "×";
}
function layoutBrowse(){
  const grid = document.getElementById("grid");
  const srcEl = grid.querySelector(".tile.orig");
  const src = showSource ? srcEl : null;
  if (srcEl && !showSource) srcEl.style.gridRow = "";
  const n = grid.children.length - (srcEl && !showSource ? 1 : 0);
  if (!n) return;
  if (DATA.expCols != null) {
    const nExp = n - (src ? 1 : 0);
    const expCols = Math.min(DATA.expCols, Math.max(nExp, 1));
    const cols = (src ? 1 : 0) + (nExp ? expCols : 0);
    grid.style.setProperty("--cols", cols);
    if (src && nExp > expCols) src.style.gridRow = "span " + Math.ceil(nExp / expCols);
    else if (src) src.style.gridRow = "";
    return;
  }
  const GAP = 10, MIN = 280;
  const maxCols = Math.max(1, Math.floor((grid.clientWidth + GAP) / (MIN + GAP)));
  let cols;
  if (n <= maxCols) cols = n;
  else cols = Math.ceil(n / Math.ceil(n / maxCols));
  grid.style.setProperty("--cols", cols);
  const rows = Math.ceil(n / cols);
  if (src) {
    if (n > cols && n % cols !== 0) src.style.gridRow = "span " + rows;
    else src.style.gridRow = "";
  }
}
function layoutEqual(grid){
  const n = grid.children.length;
  if (!n) return;
  const GAP = 10, MIN = 240;
  const maxCols = Math.max(1, Math.floor((grid.clientWidth + GAP) / (MIN + GAP)));
  let cols = n <= maxCols ? n : Math.ceil(n / Math.ceil(n / maxCols));
  grid.style.setProperty("--cols", cols);
  grid.querySelectorAll(".tile").forEach(tile => { tile.style.gridRow = ""; });
}
function layoutActive(){
  if (viewMode === "edit") layoutEqual(document.getElementById("cgrid"));
  else if (viewMode === "pick") layoutEqual(document.getElementById("pgrid"));
  else layoutBrowse();
}
function applyBg(){
  document.querySelectorAll(".sw[data-bg]").forEach(sw => sw.classList.toggle("on", sw.dataset.bg === bg));
  document.querySelectorAll(".canvas:not(.missing)").forEach(canvas => {
    if (bg === "checker") { canvas.style.backgroundColor = "#fff"; canvas.style.backgroundImage = ""; }
    else { canvas.style.backgroundImage = "none"; canvas.style.backgroundColor = bg; }
  });
}
function applyMaskToggle(){
  document.getElementById("grid").classList.toggle("show-mask", showMask);
  const btn = document.getElementById("maskToggle");
  btn.classList.toggle("active", showMask);
  btn.textContent = showMask ? "Hide mask" : "Show mask";
  requestAnimationFrame(() => document.querySelectorAll(".canvas.mask").forEach(sizeCanvas));
}
function applyBlendToggle(){
  document.getElementById("grid").classList.toggle("hide-blend", !showBlend);
  const btn = document.getElementById("blendToggle");
  btn.classList.toggle("active", showBlend);
  btn.textContent = showBlend ? "Hide blend" : "Show blend";
  requestAnimationFrame(() => { layoutActive(); applyZoom(); });
}
function applySourceToggle(){
  document.getElementById("grid").classList.toggle("hide-source", !showSource);
  const btn = document.getElementById("sourceToggle");
  btn.classList.toggle("active", showSource);
  btn.textContent = showSource ? "Hide source" : "Show source";
  requestAnimationFrame(() => { layoutActive(); applyZoom(); });
}
function setMode(next){
  viewMode = next;
  document.body.dataset.mode = next;
  document.getElementById("title").textContent = TITLES[next] || TITLES.browse;
  layoutActive();
  applyZoom();
  requestAnimationFrame(() => { layoutActive(); applyZoom(); });
}
function wireZoom(canvas, useDbl){
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const el = viewOf(canvas); if (!el) return;
    const rect = canvas.getBoundingClientRect();
    const curLeft = parseFloat(el.style.left) || 0, curTop = parseFloat(el.style.top) || 0;
    const curW = parseFloat(el.style.width) || rect.width, curH = parseFloat(el.style.height) || rect.height;
    if (!(curW > 0) || !(curH > 0)) return;
    focal.x = clamp(((e.clientX - rect.left) - curLeft) / curW, 0, 1);
    focal.y = clamp(((e.clientY - rect.top) - curTop) / curH, 0, 1);
    zoom *= (e.deltaY < 0 ? 1.18 : 1 / 1.18);
    applyZoom();
  }, {passive:false});
  if (useDbl) canvas.addEventListener("dblclick", () => { zoom = 1; applyZoom(); });
}
function startPan(ev, canvas){
  if (zoom <= 1) return;
  ev.preventDefault();
  panning = true;
  canvas.classList.add("grabbing");
  const el = viewOf(canvas);
  const rect = canvas.getBoundingClientRect();
  let lx = ev.clientX, ly = ev.clientY, moved = 0;
  const move = e2 => {
    moved += Math.abs(e2.clientX - lx) + Math.abs(e2.clientY - ly);
    canvas._dragged = moved > 4;
    const w = parseFloat(el.style.width) || rect.width;
    const h = parseFloat(el.style.height) || rect.height;
    const dx = e2.clientX - lx, dy = e2.clientY - ly;
    if (w > rect.width) focal.x = clamp(focal.x + dx / (rect.width - w), 0, 1);
    if (h > rect.height) focal.y = clamp(focal.y + dy / (rect.height - h), 0, 1);
    lx = e2.clientX; ly = e2.clientY;
    applyZoom();
  };
  const up = () => {
    panning = false;
    canvas.classList.remove("grabbing");
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    setTimeout(() => { canvas._dragged = false; }, 0);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
}
function wireBrowsePan(canvas){
  canvas.addEventListener("pointerdown", ev => {
    if (ev.button !== 0 || spaceDown) {
      if (ev.button === 1 || ev.button === 2 || spaceDown) startPan(ev, canvas);
      return;
    }
    if (zoom <= 1) return;
    canvas._dragged = false;
    startPan(ev, canvas);
  });
}

function shortName(text){
  if (text.length <= 42) return text;
  return text.slice(0, 18) + "...." + text.slice(-18);
}
function currentItem(){ return DATA.items[i]; }
function updateHeader(){
  const it = currentItem();
  document.getElementById("idx").textContent = DATA.items.length ? (i + 1) : 0;
  document.getElementById("total").textContent = DATA.items.length;
  const name = document.getElementById("name");
  name.textContent = it ? it.name : "";
  name.title = it ? it.path : "";
  const pill = document.getElementById("tagpill");
  if (it && it.label) { pill.hidden = false; pill.textContent = it.label; pill.title = it.label; }
  else { pill.hidden = false; pill.textContent = "untagged"; pill.title = ""; }
}
function makeTileCanvas(className){
  const canvas = document.createElement("div");
  canvas.className = "canvas " + className;
  return canvas;
}
function addImage(canvas, url, alt){
  const img = document.createElement("img");
  img.draggable = false;
  img.alt = alt || "";
  img.onload = () => sizeCanvas(canvas);
  img.src = url;
  canvas.appendChild(img);
  if (img.complete) sizeCanvas(canvas);
  return img;
}
function openLightbox(){
  const it = currentItem(); if (!it || !it.source) return;
  document.getElementById("lightboxImg").src = it.source;
  document.getElementById("lightboxCap").textContent = "Source · " + it.name;
  document.getElementById("lightbox").classList.add("on");
}
function placeholderTile(title, message, orig){
  const tile = document.createElement("div");
  tile.className = orig ? "tile orig" : "tile";
  const canvas = makeTileCanvas("missing");
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = title;
  const miss = document.createElement("span");
  miss.className = "miss";
  miss.textContent = message;
  canvas.appendChild(label);
  canvas.appendChild(miss);
  tile.appendChild(canvas);
  return tile;
}
function render(){
  if (i < 0) i = 0;
  if (i >= DATA.items.length) i = DATA.items.length - 1;
  const it = currentItem();
  updateHeader();
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  if (!it) { layoutActive(); return; }
  if (!it.ready || it.error || !it.source) {
    const message = !it.ready ? "Preparing this image…" : (it.error || "unreadable image");
    grid.appendChild(placeholderTile("Source", message, true));
    layoutActive(); applyZoom(); applyBg();
    return;
  }

  const srcTile = document.createElement("div");
  srcTile.className = "tile orig";
  const srcCanvas = makeTileCanvas("");
  srcCanvas.dataset.orig = "1";
  const srcLabel = document.createElement("span");
  srcLabel.className = "label";
  srcLabel.textContent = "Source";
  srcCanvas.appendChild(srcLabel);
  addImage(srcCanvas, it.source, "source");
  wireZoom(srcCanvas, true);
  wireBrowsePan(srcCanvas);
  const ctrls = document.createElement("div");
  ctrls.className = "ctrls";
  const full = document.createElement("button");
  full.type = "button";
  full.textContent = "View full ⤢";
  full.onclick = openLightbox;
  ctrls.appendChild(full);
  srcTile.appendChild(srcCanvas);
  srcTile.appendChild(ctrls);
  grid.appendChild(srcTile);

  it.outs.forEach(out => {
    const tile = document.createElement("div");
    tile.className = "tile";
    if (!out.src) {
      const missing = makeTileCanvas("blend missing");
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = out.m;
      label.title = out.m;
      const miss = document.createElement("span");
      miss.className = "miss";
      miss.textContent = "no prediction";
      missing.appendChild(label);
      missing.appendChild(miss);
      tile.appendChild(missing);
      grid.appendChild(tile);
      return;
    }
    const blend = makeTileCanvas("blend");
    blend.dataset.m = out.m;
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = out.m;
    label.title = out.m;
    blend.appendChild(label);
    addImage(blend, out.src, out.m);
    wireZoom(blend, true);
    wireBrowsePan(blend);
    tile.appendChild(blend);
    if (out.mask) {
      const mask = makeTileCanvas("mask");
      const mlabel = document.createElement("span");
      mlabel.className = "label";
      mlabel.textContent = "mask";
      mask.appendChild(mlabel);
      addImage(mask, out.mask, out.m + " mask");
      wireZoom(mask, true);
      wireBrowsePan(mask);
      tile.appendChild(mask);
    } else {
      const mask = makeTileCanvas("mask missing");
      const mlabel = document.createElement("span");
      mlabel.className = "label";
      mlabel.textContent = "mask";
      const miss = document.createElement("span");
      miss.className = "miss";
      miss.textContent = "no mask";
      mask.appendChild(mlabel);
      mask.appendChild(miss);
      tile.appendChild(mask);
    }
    grid.appendChild(tile);
  });
  layoutActive(); applyZoom(); applyBg(); applyMaskToggle(); applyBlendToggle(); applySourceToggle();
  const next = DATA.items[i + 1];
  if (next && next.ready && next.source) {
    next.outs.forEach(out => {
      if (out.src) { const im = new Image(); im.src = out.src; }
      if (out.mask) { const im = new Image(); im.src = out.mask; }
    });
    const im = new Image(); im.src = next.source;
  }
}
function go(delta){
  if (viewMode !== "browse") return;
  setIndex(i + delta);
  render();
}
async function api(url, opts){
  const res = await fetch(url, opts);
  let data = {};
  try { data = await res.json(); } catch (err) { data = {}; }
  if (!res.ok || data.ok === false) throw new Error(data.error || ("HTTP " + res.status));
  return data;
}
async function tag(label){
  if (busy || viewMode !== "browse") return;
  const it = currentItem(); if (!it || !it.ready) return;
  setBusy(true);
  setStatus("Saving…");
  try {
    const data = await api("/api/tag", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({image: it.path, label}),
    });
    it.label = data.label;
    const last = i >= DATA.items.length - 1;
    if (!last) go(1);
    else render();
    setStatus(last ? ("Saved " + data.label + " · end of list") : ("Saved " + data.label));
  } catch (err) {
    setStatus(err.message || String(err));
  } finally {
    setBusy(false);
  }
}
function skip(){
  if (busy || viewMode !== "browse") return;
  const it = currentItem();
  if (it && !it.ready) return;
  if (i >= DATA.items.length - 1) { setStatus("End of list"); return; }
  go(1);
  setStatus("Skipped");
}
function buildActions(){
  ACTIONS = DATA.models.map(name => ({name, cls:"", run: () => tag(name)}));
  ACTIONS.push({name:"odd", cls:"warn", run: () => tag("odd")});
  ACTIONS.push({name:"all bad", cls:"danger", run: () => tag("all bad")});
  ACTIONS.push({name:"Skip", cls:"", run: skip});
  ACTIONS.push({name:"Compose", cls:"primary", run: openPicker, extra:"C"});
  const host = document.getElementById("tagbtns");
  host.innerHTML = "";
  ACTIONS.forEach((action, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    if (action.cls) btn.className = action.cls;
    const num = index < 9 ? (index + 1) + ". " : "";
    const extra = action.extra ? " (" + action.extra + ")" : "";
    btn.textContent = num + shortName(action.name) + extra;
    btn.title = action.name;
    btn.onclick = () => action.run();
    host.appendChild(btn);
  });
}

function loadHtmlImage(url){
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("failed to load image"));
    img.src = url;
  });
}
async function loadMask(url, w, h){
  const out = new Uint8Array(w * h);
  if (!url) return out;
  try {
    const img = await loadHtmlImage(url);
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d", {willReadFrequently:true});
    ctx.drawImage(img, 0, 0, w, h);
    const data = ctx.getImageData(0, 0, w, h).data;
    let useAlpha = false;
    for (let p = 3; p < data.length; p += 4) {
      if (data[p] !== 255) { useAlpha = true; break; }
    }
    const channel = useAlpha ? 3 : 0;
    for (let n = 0; n < out.length; n++) out[n] = data[n * 4 + channel];
  } catch (err) {
    console.warn(err);
  }
  return out;
}
function extractRGB(img, w, h){
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d", {willReadFrequently:true});
  ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const rgb = new Uint8Array(w * h * 3);
  for (let n = 0, p = 0; n < rgb.length; n += 3, p += 4) {
    rgb[n] = data[p]; rgb[n + 1] = data[p + 1]; rgb[n + 2] = data[p + 2];
  }
  return rgb;
}
function renderPicker(){
  const it = currentItem();
  const grid = document.getElementById("pgrid");
  grid.innerHTML = "";
  if (!it) return;
  it.outs.forEach((out, idx) => {
    const tile = document.createElement("div");
    tile.className = "tile";
    const canvas = makeTileCanvas(out.mask ? "" : "missing");
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = (idx + 1) + ". " + out.m;
    label.title = out.m;
    canvas.appendChild(label);
    if (out.mask) {
      addImage(canvas, out.mask, out.m);
      wireZoom(canvas, true);
      wireBrowsePan(canvas);
    } else {
      const miss = document.createElement("span");
      miss.className = "miss";
      miss.textContent = "no prediction — blank base";
      canvas.appendChild(miss);
    }
    canvas.addEventListener("click", () => {
      if (canvas._dragged) return;
      chooseBase(idx);
    });
    const ctrls = document.createElement("div");
    ctrls.className = "ctrls";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "primary";
    btn.textContent = "Use as base";
    btn.onclick = ev => { ev.stopPropagation(); chooseBase(idx); };
    ctrls.appendChild(btn);
    tile.appendChild(canvas);
    tile.appendChild(ctrls);
    grid.appendChild(tile);
  });
  layoutActive(); applyZoom(); applyBg();
}
function openPicker(){
  if (busy || viewMode !== "browse" || !currentItem()) return;
  const it = currentItem();
  if (!it.ready) { setStatus("Still preparing this image"); return; }
  if (it.error || !it.source) { setStatus(it.error || "unreadable image"); return; }
  donorIdx = -1;
  setMode("pick");
  renderPicker();
  setStatus("Pick the mask to start from");
}
function cancelCompose(){
  loadGen++;
  dropCompose();
  setMode("browse");
  setStatus("Compose cancelled");
}
function dropCompose(){
  composed = base = srcRGB = null;
  donors = [];
  undoStack = [];
  donorIdx = -1;
  stroking = false;
  lastPt = null;
  dirty = null;
  panels = {};
  document.getElementById("cgrid").replaceChildren();
}
async function chooseBase(idx){
  if (busy || picking || viewMode !== "pick") return;
  const it = currentItem();
  if (!it || idx < 0 || idx >= it.outs.length) return;
  const gen = ++loadGen;
  picking = true;
  setStatus("Loading masks…");
  try {
    const srcImg = await loadHtmlImage(it.source);
    if (gen !== loadGen || viewMode !== "pick") return;
    const w = srcImg.naturalWidth, h = srcImg.naturalHeight;
    if (!w || !h) throw new Error("source image has no pixels");
    const loaded = [];
    for (const out of it.outs) loaded.push(await loadMask(out.mask, w, h));
    if (gen !== loadGen || viewMode !== "pick") return;
    imgW = w; imgH = h;
    srcRGB = extractRGB(srcImg, w, h);
    donors = loaded;
    baseIdx = idx;
    base = donors[idx].slice();
    composed = base.slice();
    undoStack = [];
    donorIdx = -1;
    eraser = false; pencil = false;
    buildEditor();
    syncToolButtons();
    setMode("edit");
    setStatus("Paint donor regions onto the composed mask");
  } catch (err) {
    if (gen === loadGen) setStatus(err.message || String(err));
  } finally {
    picking = false;
  }
}
function panelLabel(name){
  if (name === "donor") return donorIdx < 0 ? "Donor — select one" : ("Donor · " + DATA.models[donorIdx]);
  if (name === "composed") return "Composed";
  if (name === "blend") return "Blended preview";
  return "Original";
}
function canPaint(name){
  if (name === "original" || name === "blend") return eraser || pencil;
  if (name === "donor" || name === "composed") return eraser || pencil || donorIdx >= 0;
  return false;
}
function showBrushFor(name){
  if (name === "original" || name === "blend") return eraser || pencil;
  return name === "donor" || name === "composed";
}
function refreshPaintability(){
  document.querySelectorAll("#cgrid .canvas").forEach(canvas => {
    canvas.classList.toggle("paintable", canPaint(canvas.dataset.panel));
    const label = canvas.querySelector(".label");
    if (label) {
      label.textContent = panelLabel(canvas.dataset.panel);
      label.title = label.textContent;
    }
  });
}
function buildEditor(){
  const grid = document.getElementById("cgrid");
  grid.replaceChildren();
  panels = {};
  ["donor", "composed", "blend", "original"].forEach(name => {
    const tile = document.createElement("div");
    tile.className = "tile";
    const canvas = makeTileCanvas("");
    canvas.dataset.panel = name;
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = panelLabel(name);
    const stage = document.createElement("canvas");
    stage.className = "stage";
    stage.width = imgW;
    stage.height = imgH;
    const brush = document.createElement("div");
    brush.className = "brush";
    canvas.appendChild(label);
    canvas.appendChild(stage);
    canvas.appendChild(brush);
    wireZoom(canvas, false);
    wireCompose(canvas);
    tile.appendChild(canvas);
    grid.appendChild(tile);
    panels[name] = {canvas, stage, ctx: stage.getContext("2d", {willReadFrequently:true}), brush};
  });
  document.getElementById("baseName").textContent = "Base: " + DATA.models[baseIdx];
  document.getElementById("baseName").title = DATA.models[baseIdx];
  const donorsHost = document.getElementById("donors");
  donorsHost.replaceChildren();
  DATA.models.forEach((name, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = shortName(name);
    btn.title = name;
    btn.dataset.donor = String(idx);
    btn.onclick = () => selectDonor(idx);
    donorsHost.appendChild(btn);
  });
  drawOriginal();
  drawDonor();
  redrawAll();
  refreshPaintability();
}
function selectDonor(idx){
  donorIdx = idx;
  document.querySelectorAll("#donors button").forEach(btn => {
    btn.classList.toggle("active", Number(btn.dataset.donor) === idx);
  });
  drawDonor();
  refreshPaintability();
}
function drawOriginal(){
  const ctx = panels.original.ctx;
  const canvas = document.createElement("canvas");
  canvas.width = imgW; canvas.height = imgH;
  const tmp = canvas.getContext("2d");
  const img = tmp.createImageData(imgW, imgH);
  const px = img.data;
  for (let n = 0, p = 0; n < srcRGB.length; n += 3, p += 4) {
    px[p] = srcRGB[n]; px[p + 1] = srcRGB[n + 1]; px[p + 2] = srcRGB[n + 2]; px[p + 3] = 255;
  }
  tmp.putImageData(img, 0, 0);
  ctx.drawImage(canvas, 0, 0);
}
function drawDonor(){
  const ctx = panels.donor.ctx;
  if (donorIdx < 0) {
    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(0, 0, imgW, imgH);
    return;
  }
  putGray(ctx, donors[donorIdx], 0, 0, imgW, imgH);
}
function putGray(ctx, src, x, y, w, h){
  const img = ctx.createImageData(w, h);
  const px = img.data;
  let p = 0;
  for (let row = 0; row < h; row++) {
    let si = (y + row) * imgW + x;
    for (let col = 0; col < w; col++, si++, p += 4) {
      const v = src[si];
      px[p] = px[p + 1] = px[p + 2] = v; px[p + 3] = 255;
    }
  }
  ctx.putImageData(img, x, y);
}
function scheduleFlush(){
  if (rafId) return;
  rafId = requestAnimationFrame(flushDirty);
}
function markDirty(x1, y1, x2, y2){
  if (!dirty) dirty = {x1, y1, x2, y2};
  else {
    dirty.x1 = Math.min(dirty.x1, x1);
    dirty.y1 = Math.min(dirty.y1, y1);
    dirty.x2 = Math.max(dirty.x2, x2);
    dirty.y2 = Math.max(dirty.y2, y2);
  }
  scheduleFlush();
}
function redrawAll(){
  dirty = {x1:0, y1:0, x2:imgW, y2:imgH};
  scheduleFlush();
}
function flushDirty(){
  rafId = 0;
  if (!dirty || !composed || !panels.composed) return;
  const x1 = dirty.x1, y1 = dirty.y1, x2 = dirty.x2, y2 = dirty.y2;
  dirty = null;
  const dw = x2 - x1, dh = y2 - y1;
  if (dw < 1 || dh < 1) return;
  const maskImg = panels.composed.ctx.createImageData(dw, dh);
  const blendImg = panels.blend.ctx.createImageData(dw, dh);
  const md = maskImg.data, bd = blendImg.data;
  let p = 0;
  for (let y = y1; y < y2; y++) {
    let si = y * imgW + x1;
    for (let x = 0; x < dw; x++, si++, p += 4) {
      const v = composed[si];
      const a = v / 255;
      const s = si * 3;
      md[p] = md[p + 1] = md[p + 2] = v; md[p + 3] = 255;
      bd[p] = Math.round(srcRGB[s] * a);
      bd[p + 1] = Math.round((1 - a) * 255 + srcRGB[s + 1] * a);
      bd[p + 2] = Math.round(srcRGB[s + 2] * a);
      bd[p + 3] = 255;
    }
  }
  panels.composed.ctx.putImageData(maskImg, x1, y1);
  panels.blend.ctx.putImageData(blendImg, x1, y1);
}
function pushUndo(){
  undoStack.push(composed.slice());
  if (undoStack.length > MAX_UNDO) undoStack.shift();
}
function undo(){
  if (viewMode !== "edit" || !undoStack.length) return;
  composed = undoStack.pop();
  redrawAll();
  setStatus("Undid last stroke");
}
function resetCompose(){
  if (viewMode !== "edit" || !base) return;
  undoStack = [];
  composed = base.slice();
  redrawAll();
  setStatus("Reset to " + DATA.models[baseIdx]);
}
function stamp(fx, fy, br){
  fx = Math.trunc(fx); fy = Math.trunc(fy); br = Math.trunc(br);
  if (br < 0) return;
  const r2 = br * br;
  const x1 = Math.max(0, fx - br), y1 = Math.max(0, fy - br);
  const x2 = Math.min(imgW, fx + br + 1), y2 = Math.min(imgH, fy + br + 1);
  if (x2 <= x1 || y2 <= y1) return;
  const donor = (!eraser && !pencil && donorIdx >= 0) ? donors[donorIdx] : null;
  for (let y = y1; y < y2; y++) {
    const dy = y - fy;
    let si = y * imgW + x1;
    for (let x = x1; x < x2; x++, si++) {
      const dx = x - fx;
      if (dx * dx + dy * dy > r2) continue;
      if (eraser) composed[si] = 0;
      else if (pencil) composed[si] = 255;
      else if (brushMode === "Union") composed[si] = Math.max(composed[si], donor[si]);
      else if (brushMode === "Intersect") composed[si] = Math.min(composed[si], donor[si]);
      else composed[si] = donor[si];
    }
  }
  markDirty(x1, y1, x2, y2);
}
function paintSegment(x0, y0, x1, y1, br){
  const dist = Math.hypot(x1 - x0, y1 - y0);
  const step = Math.max(1, br / 3);
  const n = Math.max(1, Math.ceil(dist / step));
  for (let s = 0; s <= n; s++) {
    const t = s / n;
    stamp(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, br);
  }
}
function eventToNative(canvas, ev){
  const stage = canvas.querySelector("canvas.stage");
  if (!stage) return null;
  const rect = stage.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return null;
  if (ev.clientX < rect.left || ev.clientY < rect.top || ev.clientX >= rect.right || ev.clientY >= rect.bottom) {
    return null;
  }
  return [
    (ev.clientX - rect.left) / rect.width * stage.width,
    (ev.clientY - rect.top) / rect.height * stage.height,
    stage,
  ];
}
function nativeRadius(stage){
  const rect = stage.getBoundingClientRect();
  if (rect.width < 1) return brushScreen;
  return brushScreen * (stage.width / rect.width);
}
function moveBrush(canvas, ev){
  const brush = canvas.querySelector(".brush");
  if (!brush) return;
  if (!showBrushFor(canvas.dataset.panel)) { brush.style.display = "none"; return; }
  const rect = canvas.getBoundingClientRect();
  const d = brushScreen * 2;
  brush.style.display = "block";
  brush.style.width = d + "px";
  brush.style.height = d + "px";
  brush.style.left = (ev.clientX - rect.left - brushScreen) + "px";
  brush.style.top = (ev.clientY - rect.top - brushScreen) + "px";
}
function wireCompose(canvas){
  canvas.addEventListener("contextmenu", ev => ev.preventDefault());
  canvas.addEventListener("pointerdown", ev => {
    if (ev.button === 1 || ev.button === 2 || spaceDown) { startPan(ev, canvas); return; }
    if (ev.button !== 0) return;
    const name = canvas.dataset.panel;
    if (!canPaint(name)) {
      setStatus(name === "original" || name === "blend"
        ? "Turn on eraser or pencil to paint here"
        : "Select a donor, or turn on eraser or pencil");
      return;
    }
    const pos = eventToNative(canvas, ev);
    if (!pos) return;
    ev.preventDefault();
    canvas.setPointerCapture(ev.pointerId);
    if (!stroking) { pushUndo(); stroking = true; }
    stamp(pos[0], pos[1], nativeRadius(pos[2]));
    lastPt = pos;
  });
  canvas.addEventListener("pointermove", ev => {
    if (panning) return;
    moveBrush(canvas, ev);
    if (!stroking || (ev.buttons & 1) === 0) return;
    const pos = eventToNative(canvas, ev);
    if (!pos) { lastPt = null; return; }
    const br = nativeRadius(pos[2]);
    if (lastPt) paintSegment(lastPt[0], lastPt[1], pos[0], pos[1], br);
    else stamp(pos[0], pos[1], br);
    lastPt = pos;
  });
  canvas.addEventListener("pointerup", ev => {
    if (ev.button === 0) { stroking = false; lastPt = null; }
  });
  canvas.addEventListener("pointerleave", () => {
    const brush = canvas.querySelector(".brush");
    if (brush && !stroking) brush.style.display = "none";
  });
}
function setBrush(size){
  const n = parseInt(size, 10);
  brushScreen = clamp(Number.isFinite(n) ? n : brushScreen, 1, 500);
  const input = document.getElementById("brushSize");
  if (document.activeElement !== input) input.value = String(brushScreen);
  localStorage.setItem("bgrm_compose_brush", String(brushScreen));
  document.querySelectorAll("#brushPresets button").forEach(btn => {
    btn.classList.toggle("active", Number(btn.dataset.size) === brushScreen);
  });
}
function setBrushMode(mode){
  brushMode = mode;
  localStorage.setItem("bgrm_compose_bmode", brushMode);
  syncToolButtons();
}
function setEraser(on){
  eraser = !!on;
  if (eraser) pencil = false;
  syncToolButtons();
}
function setPencil(on){
  pencil = !!on;
  if (pencil) eraser = false;
  syncToolButtons();
}
function syncToolButtons(){
  document.getElementById("eraserBtn").classList.toggle("active", eraser);
  document.getElementById("pencilBtn").classList.toggle("active", pencil);
  document.querySelectorAll(".modeBtn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === brushMode);
  });
  refreshPaintability();
}
async function saveCompose(){
  if (busy || viewMode !== "edit" || !composed) return;
  const it = currentItem(); if (!it) return;
  setBusy(true);
  setStatus("Saving mask…");
  try {
    const copy = composed.slice();
    const q = new URLSearchParams({image: it.path});
    const data = await api("/api/compose?" + q.toString(), {
      method:"POST",
      headers:{
        "Content-Type":"application/octet-stream",
        "X-Width": String(imgW),
        "X-Height": String(imgH),
      },
      body: copy,
    });
    it.label = data.label;
    const last = i >= DATA.items.length - 1;
    dropCompose();
    setMode("browse");
    if (!last) go(1);
    else render();
    setStatus(last ? ("Saved " + data.label + " · end of list") : ("Saved " + data.label), data.out || "");
  } catch (err) {
    setStatus(err.message || String(err));
  } finally {
    setBusy(false);
  }
}
function digitOf(e){
  if (e.key.length === 1 && e.key >= "0" && e.key <= "9") return e.key.charCodeAt(0) - 48;
  return null;
}

const swatches = document.getElementById("swatches");
BGS.forEach(([key, col]) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "sw" + (key === "checker" ? " checker" : "");
  btn.dataset.bg = key;
  if (key !== "checker") btn.style.background = col;
  btn.title = key;
  btn.onclick = () => { bg = key; localStorage.setItem("bgrm_compose_bg", bg); applyBg(); };
  swatches.appendChild(btn);
});
document.getElementById("bgpick").oninput = e => {
  bg = e.target.value; localStorage.setItem("bgrm_compose_bg", bg); applyBg();
};
document.getElementById("zoom").oninput = e => { zoom = parseFloat(e.target.value); applyZoom(); };
document.getElementById("zreset").onclick = () => { zoom = 1; applyZoom(); };
document.getElementById("maskToggle").onclick = () => {
  showMask = !showMask; localStorage.setItem("bgrm_compose_showmask", showMask ? "1" : "0"); applyMaskToggle();
};
document.getElementById("blendToggle").onclick = () => {
  showBlend = !showBlend; localStorage.setItem("bgrm_compose_showblend", showBlend ? "1" : "0"); applyBlendToggle();
};
document.getElementById("sourceToggle").onclick = () => {
  showSource = !showSource; localStorage.setItem("bgrm_compose_showsource", showSource ? "1" : "0"); applySourceToggle();
};
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(1);
document.getElementById("cancelPick").onclick = cancelCompose;
document.getElementById("cancelEdit").onclick = cancelCompose;
document.getElementById("undoBtn").onclick = undo;
document.getElementById("resetBtn").onclick = resetCompose;
document.getElementById("saveBtn").onclick = () => saveCompose();
document.getElementById("eraserBtn").onclick = () => setEraser(!eraser);
document.getElementById("pencilBtn").onclick = () => setPencil(!pencil);
document.querySelectorAll(".modeBtn").forEach(btn => {
  btn.onclick = () => setBrushMode(btn.dataset.mode);
});
const brushInput = document.getElementById("brushSize");
brushInput.value = String(brushScreen);
brushInput.addEventListener("input", () => {
  const n = parseInt(brushInput.value, 10);
  if (!Number.isFinite(n)) return;
  brushScreen = clamp(n, 1, 500);
  localStorage.setItem("bgrm_compose_brush", String(brushScreen));
  document.querySelectorAll("#brushPresets button").forEach(btn => {
    btn.classList.toggle("active", Number(btn.dataset.size) === brushScreen);
  });
});
brushInput.addEventListener("change", () => setBrush(brushInput.value));
[5, 15, 30, 50, 80].forEach(size => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = String(size);
  btn.dataset.size = String(size);
  btn.onclick = () => setBrush(size);
  document.getElementById("brushPresets").appendChild(btn);
});
setBrush(brushScreen);
syncToolButtons();
const lightbox = document.getElementById("lightbox");
lightbox.onclick = () => lightbox.classList.remove("on");
window.addEventListener("resize", () => { layoutActive(); applyZoom(); });
window.addEventListener("blur", () => { spaceDown = false; document.body.classList.remove("spacepan"); });
document.addEventListener("keydown", e => {
  if (e.code === "Space" && !typingTarget(e)) {
    e.preventDefault();
    spaceDown = true;
    document.body.classList.add("spacepan");
  }
  if (typingTarget(e)) return;
  if ((e.metaKey || e.ctrlKey) && (e.key === "z" || e.key === "Z")) {
    if (viewMode === "edit") { e.preventDefault(); undo(); }
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const zoomKey = e.key === "+" || e.key === "=" || e.key === "-" || e.key === "_";
  const arrow = e.key.startsWith("Arrow");
  if (e.repeat && !zoomKey && !arrow) return;
  if (busy) return;
  if (e.key === "Escape") {
    if (lightbox.classList.contains("on")) { lightbox.classList.remove("on"); return; }
    if (viewMode === "edit" || viewMode === "pick") cancelCompose();
    return;
  }
  if (e.key === "ArrowRight") { e.preventDefault(); if (zoom > 1) { focal.x = clamp(focal.x + 0.08, 0, 1); applyZoom(); } else if (viewMode === "browse") go(1); return; }
  if (e.key === "ArrowLeft") { e.preventDefault(); if (zoom > 1) { focal.x = clamp(focal.x - 0.08, 0, 1); applyZoom(); } else if (viewMode === "browse") go(-1); return; }
  if (e.key === "ArrowUp") { e.preventDefault(); if (zoom > 1) { focal.y = clamp(focal.y - 0.08, 0, 1); applyZoom(); } return; }
  if (e.key === "ArrowDown") { e.preventDefault(); if (zoom > 1) { focal.y = clamp(focal.y + 0.08, 0, 1); applyZoom(); } return; }
  if (zoomKey) {
    e.preventDefault();
    zoom *= (e.key === "+" || e.key === "=") ? 1.18 : 1 / 1.18;
    applyZoom();
    return;
  }
  const digit = digitOf(e);
  if (digit === 0) {
    if (viewMode === "edit" && (eraser || pencil)) setBrush(1);
    else { zoom = 1; applyZoom(); }
    return;
  }
  if (digit !== null) {
    if (viewMode === "browse" && digit <= ACTIONS.length) ACTIONS[digit - 1].run();
    else if (viewMode === "pick") chooseBase(digit - 1);
    else if (viewMode === "edit" && (eraser || pencil)) setBrush(digit);
    return;
  }
  if (viewMode === "edit" && (e.key === "e" || e.key === "E")) { setEraser(!eraser); return; }
  if (viewMode === "edit" && (e.key === "p" || e.key === "P")) { setPencil(!pencil); return; }
  if (viewMode === "edit" && (e.key === "s" || e.key === "S")) { saveCompose(); return; }
  if (viewMode === "edit" && (e.key === "r" || e.key === "R")) { resetCompose(); return; }
  if (viewMode === "browse" && (e.key === "n" || e.key === "N")) { go(1); return; }
  if (viewMode === "browse" && (e.key === "p" || e.key === "P" || e.key === "c" || e.key === "C")) {
    if (e.key === "c" || e.key === "C") openPicker();
    else go(-1);
  }
});
document.addEventListener("keyup", e => {
  if (e.code === "Space") {
    spaceDown = false;
    document.body.classList.remove("spacepan");
  }
});

document.getElementById("ctag").textContent = DATA.composeTag;
document.getElementById("ctag").title = "Composed masks save under results/" + DATA.composeTag + "/";
const logName = (DATA.log || "").split("/").pop();
document.getElementById("logpill").textContent = logName ? ("log " + logName) : "log";
document.getElementById("logpill").title = DATA.log || "";
document.getElementById("total").textContent = DATA.items.length;

async function init(){
  if (sessionStorage.getItem("bgrm_compose_token") !== DATA.token) {
    sessionStorage.removeItem("bgrm_compose_idx");
    sessionStorage.setItem("bgrm_compose_token", DATA.token);
  }
  let stored = parseInt(sessionStorage.getItem("bgrm_compose_idx") || "", 10);
  try {
    const data = await api("/api/resume");
    (data.labels || []).forEach((label, n) => { if (DATA.items[n]) DATA.items[n].label = label; });
    if (!Number.isFinite(stored)) stored = data.resume;
  } catch (err) {
    if (!Number.isFinite(stored)) stored = DATA.resume || 0;
    setStatus(err.message || String(err));
  }
  buildActions();
  setIndex(Number.isFinite(stored) ? stored : 0);
  setMode("browse");
  render();
  pollCatalog();
}
init();
</script>
</body>
</html>
"""


def _read_images_arg(args) -> list:
    paths = []
    if args.image_list:
        paths.extend(read_image_list(args.image_list))
    if args.images_file:
        paths.extend(read_image_list(args.images_file))
    if args.images:
        paths.extend(part.strip() for part in args.images.split(",") if part.strip())
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_list", nargs="?",
                        help="text file with one image path per line")
    parser.add_argument("--experiments",
                        help="comma-separated results folder names (the mask cases)")
    parser.add_argument("--images", help="comma-separated source image paths")
    parser.add_argument("--images-file",
                        help="text file, one image path per line")
    parser.add_argument("--log", default=DEFAULT_LOG,
                        help=f"tag log (default: {DEFAULT_LOG})")
    parser.add_argument("--compose-date", default=date.today().strftime("%Y%m%d"),
                        help="YYYYMMDD used in results/compose.dt<date>/ (default: today)")
    parser.add_argument("--optional", default="",
                        help="comma-separated experiments that may be missing without a warning")
    parser.add_argument("--out", default=None,
                        help=f"output dir (default: {OUT_DIR_DEFAULT})")
    parser.add_argument("--rgba", action="store_true",
                        help="show 4-channel predictions as RGB×alpha, and edit the alpha")
    parser.add_argument("--ncols", type=int, default=None,
                        help="experiment tiles per row (source tile is extra)")
    parser.add_argument("--port", type=int, default=PORT_DEFAULT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--foreground", action="store_true",
                        help="serve in this process until Ctrl+C, instead of a background process")
    parser.add_argument("--serve-only", metavar="OUT_DIR", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.serve_only:
        load_session(args.serve_only)
        _serve_blocking(args.serve_only, args.port, not args.no_browser)
        return 0

    if not args.experiments:
        parser.error("--experiments is required")
    experiments = [part.strip() for part in args.experiments.split(",") if part.strip()]
    if not experiments:
        parser.error("--experiments is required")
    image_paths = _read_images_arg(args)
    if not image_paths:
        parser.error("no images given: pass an image list, --images-file, and/or --images")
    if args.ncols is not None and args.ncols < 1:
        parser.error("--ncols must be >= 1")
    optional = [part.strip() for part in args.optional.split(",") if part.strip()]
    try:
        make_compose_tag(args.compose_date)
    except ValueError as exc:
        parser.error(str(exc))

    compose_and_view(
        experiments, image_paths, log_file=args.log, compose_date=args.compose_date,
        port=args.port, open_browser=not args.no_browser, out_dir=args.out,
        background=not args.foreground, rgba=args.rgba, ncols=args.ncols,
        optional=optional,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
