#!/usr/bin/env python3
"""Self-contained viewer for eyeballing prediction masks from different
experiments against a fixed set of source images — pure visualization, no
rating/annotation UI (see make_viewer.py / make_inspector.py for those).

Each experiment's prediction file is located from the source image path by a
fixed naming convention:

    predicted_path = image_path.replace("/im/", f"/results/{experiment}/")
                                .replace(".jpg", ".png")

and is by default a grayscale mask, so before display it's blended with the
source image onto a pure-green background. Pass `rgba=True` / `--rgba` when the
predictions are 4-channel RGB+alpha (as written by cv2.imwrite): the tile shows
the prediction's RGB multiplied by its alpha (``RGB * alpha``), while the mask
row shows the alpha channel extracted from it.

Usage (programmatic) — the main entry point:

    from view_experiments import view_experiments
    view_experiments(["expA", "expB"], ["/data/study/im/car_11.jpg", ...])
    view_experiments(["expA", "expB", "expC"], paths, ncols=2)  # 2 experiment cols; source is extra

This builds the page and stages its source assets in a fixed output directory,
then serves only that directory on http://127.0.0.1:8765/index.html. The server
is loopback-only and opens a browser tab automatically. The output directory
(and therefore the URL) is always the same across calls unless you pass an
explicit `out_dir=`/`port=`, so bookmarking the URL keeps working.

By default the server runs detached in the background and the call returns
immediately; pass `background=False` to block the current process instead
(Ctrl+C to stop). Either way, any previous viewer already running on the
same port is killed first and replaced by the new one.

Usage (CLI):

    python3 view_experiments.py --experiments expA,expB --images-file paths.txt
    python3 view_experiments.py --experiments expA,expB --images /a/im/1.jpg,/a/im/2.jpg
    python3 view_experiments.py --experiments expA,expB,expC --images-file paths.txt --ncols 2

The CLI also runs in the background by default (prints the URL and returns
immediately); pass --foreground to block with Ctrl+C to stop instead.
"""
import argparse
import functools
import hashlib
import http.server
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from urllib.parse import quote

from PIL import Image
import cv2

GREEN = (0, 255, 0)
PORT_DEFAULT = 8765
OUT_DIR_DEFAULT = os.path.join(tempfile.gettempdir(), "bgrm_view")
_NAME_MAX = 255  # Linux per-component limit (ENAMETOOLONG / errno 36)


def predicted_path(image_path: str, experiment: str) -> str:
    """Fixed naming convention linking a source image to one experiment's mask."""
    return image_path.replace("/im/", f"/results/{experiment}/").replace(".jpg", ".png")


def blend_mask_on_green(image_path: str, mask_path: str) -> Image.Image:
    """Composite the source image onto a pure-green background using the
    (grayscale) mask as alpha: mask=255 -> source shows, mask=0 -> green shows."""
    src = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != src.size:
        mask = mask.resize(src.size, Image.Resampling.LANCZOS)
    bg = Image.new("RGB", src.size, GREEN)
    return Image.composite(src, bg, mask)


def _read_rgb_alpha(mask_path):
    """Load a prediction PNG as (RGB uint8 HxWx3, alpha uint8 HxW).

    4-channel files from this repo are written with ``cv2.imwrite`` on RGB+A
    (OpenCV treats that as BGRA), so ``cv2.imread`` restores the original RGB
    in ``[:,:,:3]``. PIL would show R and B swapped.
    """
    arr = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {mask_path}")
    if arr.ndim == 2:
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        alpha = arr  # already a matte; treat as opaque-if-nonzero later
        return rgb, alpha
    if arr.shape[2] == 4:
        return arr[:, :, :3].copy(), arr[:, :, 3].copy()
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), None


def rgb_times_alpha(mask_path: str) -> Image.Image:
    """Opaque RGB of the prediction, multiplied by its alpha (``RGB * A/255``).

    Pixels with alpha 0 go to black; RGB is taken from the 4-channel file
    (cv2 channel order).
    """
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
    with Image.open(mask_path) as im:
        return im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info


def _render_cached(out_path: str, deps, render) -> bool:
    """Write render()'s image to out_path, skipping the (re)compute if it's
    already newer than every dep. Returns True on success, False if it
    couldn't be built."""
    try:
        if os.path.exists(out_path) and all(
                os.path.getmtime(out_path) >= os.path.getmtime(d) for d in deps):
            return True
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        render().save(out_path)
        return True
    except Exception as e:
        print(f"warning: failed to render {out_path}: {e}")
        return False


def _item_name(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-")


def _cache_filename(stem: str, suffix: str) -> str:
    """Build a cache filename that cannot exceed NAME_MAX bytes.

    ``suffix`` includes the extension (``.png`` or ``__rgbxa.png``).
    Short stems stay human-readable (preserves existing cache hits).
    Overlong stems keep a truncated prefix plus a sha1 of the original
    so two long names cannot collide after truncation.
    """
    name = f"{stem}{suffix}"
    if len(name.encode("utf-8")) <= _NAME_MAX:
        return name
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
    tail = f"_{digest}{suffix}"
    budget = _NAME_MAX - len(tail.encode("utf-8"))
    prefix = stem.encode("utf-8")[:budget].decode("utf-8", "ignore").rstrip("-_")
    return f"{prefix}{tail}"


def _asset_filename(path: str) -> str:
    """Return a readable, collision-safe filename for a staged input file."""
    absolute_path = os.path.abspath(path)
    basename = os.path.basename(absolute_path)
    stem, suffix = os.path.splitext(basename)
    readable_stem = _slug(stem) or "asset"
    digest = hashlib.sha1(absolute_path.encode("utf-8")).hexdigest()[:12]
    return _cache_filename(f"{readable_stem}_{digest}", suffix or ".bin")


def _stage_asset(source_path: str, destination_dir: str) -> str:
    """Hard-link an input into the viewer tree, copying when linking fails."""
    source_path = os.path.abspath(source_path)
    os.makedirs(destination_dir, exist_ok=True)
    destination = os.path.join(destination_dir, _asset_filename(source_path))

    # A prior hard link already reflects in-place source changes. Never accept
    # a symlink here because it could point outside the directory being served.
    if os.path.exists(destination) and not os.path.islink(destination):
        try:
            if os.path.samefile(source_path, destination):
                return destination
        except OSError:
            pass

    fd, temporary = tempfile.mkstemp(prefix=".stage-", dir=destination_dir)
    os.close(fd)
    os.unlink(temporary)
    try:
        try:
            os.link(source_path, temporary)
        except OSError:
            shutil.copy2(source_path, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return destination


def _relative_url(path: str, out_dir: str) -> str:
    """Convert a path inside ``out_dir`` to a quoted browser URL."""
    relative = os.path.relpath(os.path.abspath(path), os.path.abspath(out_dir))
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise ValueError(f"viewer asset escapes output directory: {path}")
    return quote(relative.replace(os.sep, "/"), safe="/")


def _prune_managed_assets(directory: str, keep_paths) -> None:
    """Remove files left by earlier viewer builds from a managed directory."""
    keep = {os.path.abspath(path) for path in keep_paths}
    if not os.path.isdir(directory):
        return
    for entry in os.scandir(directory):
        path = os.path.abspath(entry.path)
        if path not in keep and (entry.is_file(follow_symlinks=False) or entry.is_symlink()):
            os.unlink(path)


def build_viewer(experiments, image_paths, out_dir=None, rgba=False,
                 ncols=None) -> str:
    """Stage and blend every pair, then write ``out_dir/index.html``.

    With `rgba`, predictions are 4-channel RGB+alpha: the tile is
    ``RGB * alpha`` (cv2 channel order) and the mask row shows the alpha
    channel. `ncols` is the number of experiment tiles per row (the
    source tile is extra); ``None`` keeps the viewer's auto-fit layout.
    Every file referenced by the page is contained in ``out_dir``.
    Returns the output directory."""
    if not experiments:
        raise ValueError("need at least 1 experiment name")
    if not image_paths:
        raise ValueError("need at least 1 image path")
    if ncols is not None and ncols < 1:
        raise ValueError(f"ncols must be >= 1 or None, got {ncols!r}")

    out_dir = os.path.abspath(out_dir or OUT_DIR_DEFAULT)
    blend_dir = os.path.join(out_dir, "blend")
    alpha_dir = os.path.join(out_dir, "alpha")
    source_dir = os.path.join(out_dir, "assets", "source")
    mask_dir = os.path.join(out_dir, "assets", "mask")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(blend_dir, exist_ok=True)
    os.makedirs(alpha_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    items = []
    active_blends = set()
    active_alphas = set()
    active_sources = set()
    active_masks = set()
    for image_path in image_paths:
        image_path = os.path.abspath(image_path)
        name = _item_name(image_path)
        if not os.path.exists(image_path):
            print(f"warning: source image not found, skipping: {image_path}")
            continue
        staged_source = _stage_asset(image_path, source_dir)
        active_sources.add(staged_source)
        outs = []
        for exp in experiments:
            mask_path = predicted_path(image_path, exp)
            stem = f"{_slug(name)}__{_slug(exp)}"
            src = mask = None
            if not os.path.exists(mask_path):
                print(f"warning: no prediction for {name!r} / {exp!r}, expected {mask_path}")
            elif rgba and has_alpha(mask_path):
                rgb_path = os.path.join(blend_dir, _cache_filename(stem, "__rgbxa.png"))
                if _render_cached(rgb_path, [mask_path],
                                  lambda: rgb_times_alpha(mask_path)):
                    active_blends.add(rgb_path)
                    src = _relative_url(rgb_path, out_dir)
                alpha_path = os.path.join(alpha_dir, _cache_filename(stem, ".png"))
                if _render_cached(alpha_path, [mask_path], lambda: alpha_of(mask_path)):
                    active_alphas.add(alpha_path)
                    mask = _relative_url(alpha_path, out_dir)
            else:
                if rgba:
                    print(f"warning: {mask_path} has no alpha channel, "
                          "falling back to blending the source image")
                blend_path = os.path.join(blend_dir, _cache_filename(stem, ".png"))
                if _render_cached(blend_path, [image_path, mask_path],
                                  lambda: blend_mask_on_green(image_path, mask_path)):
                    active_blends.add(blend_path)
                    src = _relative_url(blend_path, out_dir)
                staged_mask = _stage_asset(mask_path, mask_dir)
                active_masks.add(staged_mask)
                mask = _relative_url(staged_mask, out_dir)
            outs.append({"m": exp, "src": src, "mask": mask})
        items.append({
            "name": name,
            "source": _relative_url(staged_source, out_dir),
            "outs": outs,
        })

    if not items:
        raise ValueError("no valid items to display — check the image paths / naming convention")

    _prune_managed_assets(blend_dir, active_blends)
    _prune_managed_assets(alpha_dir, active_alphas)
    _prune_managed_assets(source_dir, active_sources)
    _prune_managed_assets(mask_dir, active_masks)

    data_json = _json_dumps({"items": items, "models": experiments, "expCols": ncols})

    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w") as f:
        f.write(HTML.replace("__DATA__", data_json))
    print(f"wrote {out_path} with {len(items)} items, {len(experiments)} experiments")
    return out_dir


def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"))


def _pidfile_path(port: int) -> str:
    return os.path.join(tempfile.gettempdir(), f"bgrm_view_{port}.pid")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _pid_listening_on_port(port: int):
    """Best-effort (Linux-only) fallback: find the PID of whatever process is
    LISTENing on `port`, by cross-referencing /proc/net/tcp{,6} socket inodes
    against /proc/*/fd symlinks. Returns None if it can't be determined."""
    target_hex = f"{port:04X}"
    inodes = set()
    for proc_net in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_net) as f:
                next(f)  # header
                for line in f:
                    fields = line.split()
                    local_addr, state, inode = fields[1], fields[3], fields[9]
                    if state == "0A" and local_addr.split(":")[1] == target_hex:  # 0A = LISTEN
                        inodes.add(inode)
        except (FileNotFoundError, OSError):
            continue
    if not inodes:
        return None
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
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


def _kill_pid(pid: int, port: int):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as e:
        print(f"warning: could not kill previous viewer (pid {pid}) on port {port}: {e}")
        return
    for _ in range(30):
        if not _pid_alive(pid):
            print(f"killed previous viewer (pid {pid}) on port {port}")
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"force-killed previous viewer (pid {pid}) on port {port}")
    except ProcessLookupError:
        pass


def _kill_existing_viewer(port: int):
    """If a viewer (ours or otherwise) is already bound to `port`, kill it so
    the new one can take over the same address."""
    pidfile = _pidfile_path(port)
    pid = None
    if os.path.exists(pidfile):
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
    else:
        # pidfile missing/stale — best-effort fallback in case something else
        # (or an untracked previous run) is still listening on this port.
        fallback_pid = _pid_listening_on_port(port)
        if fallback_pid is not None:
            _kill_pid(fallback_pid, port)


def _spawn_background_server(out_dir: str, port: int, open_browser: bool) -> int:
    """Launch a fully detached child process that just serves `out_dir` on
    `port`, so it keeps running independently of the calling process/script.

    The child writes its own pidfile (via `_serve_blocking`) once it actually
    binds — callers here must NOT pre-write the pidfile, or a slow-starting
    child could later read its own pid back out of `_kill_existing_viewer`
    and terminate itself."""
    args = [sys.executable, os.path.abspath(__file__), "--serve-only", out_dir, "--port", str(port)]
    if not open_browser:
        args.append("--no-browser")
    log_path = os.path.join(out_dir, "server.log")
    log_file = open(log_path, "a")
    proc = subprocess.Popen(
        args, stdout=log_file, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    return proc.pid


class _ContainedHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files only when their resolved path remains within the root."""

    def send_head(self):
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


def _make_server(out_dir: str, port: int):
    """Create a loopback-only HTTP server rooted at ``out_dir``."""
    handler = functools.partial(
        _ContainedHTTPRequestHandler, directory=os.path.abspath(out_dir))
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


def _serve_blocking(out_dir: str, port: int, open_browser: bool):
    """Actually bind to `port` and serve `out_dir` in the current process,
    blocking until interrupted. Registers/cleans up the pidfile for `port`
    but does NOT check for or kill any pre-existing viewer — callers must
    have already done that (see `serve_and_open`)."""
    with open(_pidfile_path(port), "w") as f:
        f.write(str(os.getpid()))
    url = f"http://127.0.0.1:{port}/index.html"
    httpd = _make_server(out_dir, port)
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
                    background: bool = True):
    """Serve only ``out_dir`` on loopback and open the generated page.

    Any previous viewer already bound to `port` is killed first, so the new
    one always takes over the same address.

    If `background` (default), spawns a detached child process that keeps
    serving after this call returns (non-blocking). Otherwise blocks the
    current process until interrupted (Ctrl+C).
    """
    _kill_existing_viewer(port)

    if background:
        url = f"http://127.0.0.1:{port}/index.html"
        pid = _spawn_background_server(out_dir, port, open_browser)
        print(f"serving in background (pid {pid}) on {url}")
        return

    _serve_blocking(out_dir, port, open_browser)


def view_experiments(experiments, image_paths, *, port: int = PORT_DEFAULT,
                      open_browser: bool = True, out_dir: str = None,
                      background: bool = True, rgba: bool = False,
                      ncols=None) -> str:
    """Build the viewer for `image_paths` across `experiments` and serve it.

    experiments: ordered list of experiment names (the "H" in the results path).
    image_paths: source image paths (each expected to contain "/im/" and end in ".jpg").
    rgba: treat predictions as 4-channel RGB+alpha — show ``RGB * alpha``
    (cv2 channel order) instead of blending the source through a grayscale
    mask onto green.
    ncols: experiment tiles per row (source tile is extra). ``None``
    (default) keeps the auto-fit layout.
    Any previous viewer already running on `port` is killed and replaced.
    By default runs the server in a detached background process and returns
    immediately; pass background=False to block the current process until
    interrupted (Ctrl+C) instead. Returns the output dir.
    """
    out_dir = build_viewer(experiments, image_paths, out_dir=out_dir, rgba=rgba,
                           ncols=ncols)
    serve_and_open(out_dir, port=port, open_browser=open_browser, background=background)
    return out_dir


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment viewer</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#272b34; --txt:#e6e8ec; --mut:#8b93a1; --accent:#4f8cff; }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body { margin:0; font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--txt); user-select:none;
         display:flex; flex-direction:column; height:100vh; overflow:hidden; }
  header { position:sticky; top:0; z-index:5; display:flex; align-items:center; gap:10px;
           padding:9px 14px; background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header .sp { flex:1; }
  h1 { font-size:14px; margin:0; font-weight:600; }
  .pill { background:#222732; border:1px solid var(--line); border-radius:999px; padding:3px 10px; color:var(--mut); }
  .pill b { color:var(--txt); }
  button { background:#222732; color:var(--txt); border:1px solid var(--line);
           border-radius:7px; padding:6px 10px; cursor:pointer; font-size:13px; }
  button:hover { border-color:var(--accent); }
  button.active { border-color:var(--accent); color:var(--accent); }
  .grp { display:flex; align-items:center; gap:4px; }
  .grp label { color:var(--mut); font-size:12px; }
  .sw { width:22px; height:22px; border-radius:5px; border:1px solid var(--line); cursor:pointer; padding:0; }
  .sw.on { outline:2px solid var(--accent); outline-offset:1px; }
  .sw.checker { background-color:#fff;
    background-image:linear-gradient(45deg,#ccc 25%,transparent 25%),linear-gradient(-45deg,#ccc 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ccc 75%),linear-gradient(-45deg,transparent 75%,#ccc 75%);
    background-size:10px 10px; background-position:0 0,0 5px,5px -5px,-5px 0; }
  main { padding:14px 16px; max-width:none; margin:0; flex:1 1 auto; min-height:0;
         overflow:hidden; display:flex; flex-direction:column; }
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
  .ctrls.orig { justify-content:center; color:var(--mut); }
  .canvas { position:relative; flex:1 1 auto; min-height:0; overflow:hidden; cursor:grab; background-color:#fff;
            background-image:linear-gradient(45deg,#ccc 25%,transparent 25%),linear-gradient(-45deg,#ccc 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ccc 75%),linear-gradient(-45deg,transparent 75%,#ccc 75%);
            background-size:22px 22px; background-position:0 0,0 11px,11px -11px,-11px 0; }
  .canvas.grabbing { cursor:grabbing; }
  .canvas img { position:absolute; image-rendering:auto; }
  .canvas.missing { display:flex; align-items:center; justify-content:center; cursor:default; }
  .canvas .miss { color:var(--mut); font-size:13px; }
  .label { position:absolute; top:6px; right:8px; background:rgba(0,0,0,.6); color:#fff; border-radius:6px;
           padding:1px 7px; font-size:12px; pointer-events:none; z-index:1; }
  .canvas.mask { display:none; border-top:1px solid var(--line); }
  #grid.show-mask .canvas.mask { display:block; }
  #grid.show-mask .canvas.mask.missing { display:flex; }
  #grid.hide-blend .canvas.blend { display:none; }
  #grid.hide-source .tile.orig { display:none; }
  .ctrls { display:flex; align-items:center; gap:6px; padding:7px; border-top:1px solid var(--line); flex:0 0 auto; }
  footer { display:flex; gap:10px; align-items:center; padding:9px 14px;
           background:var(--panel); border-top:1px solid var(--line); flex-wrap:wrap; flex:0 0 auto; }
  .hint { color:var(--mut); font-size:12px; }
  kbd { background:#222732; border:1px solid var(--line); border-radius:4px; padding:0 5px; }
</style>
</head>
<body>
<header>
  <h1>Experiment viewer</h1>
  <span class="pill">Item <b id="idx">–</b>/<span id="total"></span></span>
  <span class="pill" id="name"></span>
  <span class="grp"><label>bg</label><span id="swatches"></span>
    <input type="color" id="bgpick" value="#7f7f7f" title="custom bg" class="sw"></span>
  <span class="grp"><label>zoom</label>
    <input type="range" id="zoom" min="1" max="36" step="0.25" value="1">
    <span id="zlabel" class="hint">1.0×</span>
    <button id="zreset">reset</button></span>
  <span class="grp">
    <button id="maskToggle" title="show/hide mask row below each blend">Show mask</button>
    <button id="blendToggle" title="show/hide blend (green composite / RGB×alpha)">Hide blend</button>
    <button id="sourceToggle" title="show/hide source image tile">Hide source</button>
  </span>
  <span class="sp"></span>
</header>
<main>
  <div class="grid" id="grid"></div>
</main>
<div class="lightbox" id="lightbox"><span class="cap" id="lightboxCap"></span><img id="lightboxImg" alt="source full"></div>
<footer>
  <button id="prev">‹ Prev</button>
  <button id="next">Next ›</button>
  <span class="hint"><kbd>←</kbd><kbd>→</kbd> nav (or pan when zoomed) · <kbd>n</kbd><kbd>p</kbd> next/prev ·
    <kbd>↑</kbd><kbd>↓</kbd> pan when zoomed ·
    wheel/<kbd>+</kbd><kbd>-</kbd>=zoom · <kbd>0</kbd>=reset zoom · drag=pan · dbl-click=reset tile zoom</span>
</footer>
<script>
const DATA = __DATA__;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
let i = parseInt(localStorage.getItem("bgrm_view_idx")||"0",10) || 0;
let zoom = 1, focal = {x:0.5, y:0.5};
const BGS = [["checker",""],["#ffffff","#ffffff"],["#000000","#000000"],
             ["#7f7f7f","#7f7f7f"],["#ff00ff","#ff00ff"],["#00ff00","#00ff00"]];
let bg = localStorage.getItem("bgrm_view_bg") || "checker";
let showMask = localStorage.getItem("bgrm_view_showmask") === "1";
let showBlend = localStorage.getItem("bgrm_view_showblend") !== "0";
let showSource = localStorage.getItem("bgrm_view_showsource") !== "0";

function sizeCanvas(canvas){
  const img=canvas.querySelector("img"); if(!img||!img.naturalWidth) return;
  const cw=canvas.clientWidth, ch=canvas.clientHeight;
  const base=Math.min(cw/img.naturalWidth, ch/img.naturalHeight);
  const w=img.naturalWidth*base*zoom, h=img.naturalHeight*base*zoom;
  img.style.width=w+"px"; img.style.height=h+"px";
  img.style.left=(w<=cw ? (cw-w)/2 : focal.x*(cw-w))+"px";
  img.style.top=(h<=ch ? (ch-h)/2 : focal.y*(ch-h))+"px";
}
function applyZoom(){
  zoom=clamp(zoom,1,36);
  if(zoom===1){ focal={x:0.5,y:0.5}; }
  document.querySelectorAll(".canvas").forEach(sizeCanvas);
  document.getElementById("zoom").value=zoom;
  document.getElementById("zlabel").textContent=zoom.toFixed(1)+"×";
}
function pan(dx,dy){ const step=0.08; focal.x=clamp(focal.x+dx*step,0,1); focal.y=clamp(focal.y+dy*step,0,1); applyZoom(); }
function applyBg(){ document.querySelectorAll(".sw[data-bg]").forEach(s=>s.classList.toggle("on", s.dataset.bg===bg));
  document.querySelectorAll(".canvas:not(.missing)").forEach(c=>{
    if(bg==="checker"){ c.style.backgroundColor="#fff"; c.style.backgroundImage=""; }
    else { c.style.backgroundImage="none"; c.style.backgroundColor=bg; } }); }
function applyMaskToggle(){
  document.getElementById("grid").classList.toggle("show-mask", showMask);
  const btn=document.getElementById("maskToggle");
  btn.classList.toggle("active", showMask);
  btn.textContent = showMask ? "Hide mask" : "Show mask";
  requestAnimationFrame(()=>document.querySelectorAll(".canvas.mask").forEach(sizeCanvas));
}
function applyBlendToggle(){
  document.getElementById("grid").classList.toggle("hide-blend", !showBlend);
  const btn=document.getElementById("blendToggle");
  btn.classList.toggle("active", showBlend);
  btn.textContent = showBlend ? "Hide blend" : "Show blend";
  requestAnimationFrame(()=>{ layoutGrid(); document.querySelectorAll(".canvas").forEach(sizeCanvas); });
}
function applySourceToggle(){
  document.getElementById("grid").classList.toggle("hide-source", !showSource);
  const btn=document.getElementById("sourceToggle");
  btn.classList.toggle("active", showSource);
  btn.textContent = showSource ? "Hide source" : "Show source";
  requestAnimationFrame(()=>{ layoutGrid(); document.querySelectorAll(".canvas").forEach(sizeCanvas); });
}

function layoutGrid(){
  const grid=document.getElementById("grid");
  const srcEl=grid.querySelector(".tile.orig");
  const src=showSource ? srcEl : null;
  if(srcEl && !showSource) srcEl.style.gridRow="";
  const n=grid.children.length-(srcEl && !showSource ? 1 : 0);
  if(!n) return;
  if(DATA.expCols!=null){
    const nExp=n-(src?1:0);
    const expCols=Math.min(DATA.expCols, Math.max(nExp,1));
    const cols=(src?1:0)+(nExp?expCols:0);
    grid.style.setProperty("--cols", cols);
    if(src && nExp>expCols) src.style.gridRow="span "+Math.ceil(nExp/expCols);
    else if(src) src.style.gridRow="";
    return;
  }
  const GAP=10, MIN=280;
  const maxCols=Math.max(1, Math.floor((grid.clientWidth+GAP)/(MIN+GAP)));
  let cols;
  if(n<=maxCols) cols=n;
  else {
    const fitRows=Math.ceil(n/maxCols);
    cols=Math.ceil(n/fitRows);
  }
  grid.style.setProperty("--cols", cols);
  const rows=Math.ceil(n/cols);
  if(src){
    if(n>cols && n%cols!==0) src.style.gridRow="span "+rows;
    else src.style.gridRow="";
  }
}

function wireInspect(canvas){
  canvas.addEventListener("wheel",e=>{ e.preventDefault(); const rect=canvas.getBoundingClientRect();
    const img=canvas.querySelector("img");
    const curLeft=parseFloat(img.style.left)||0, curTop=parseFloat(img.style.top)||0;
    const curW=parseFloat(img.style.width)||rect.width, curH=parseFloat(img.style.height)||rect.height;
    focal.x=clamp(((e.clientX-rect.left)-curLeft)/curW,0,1);
    focal.y=clamp(((e.clientY-rect.top)-curTop)/curH,0,1);
    zoom*=(e.deltaY<0?1.18:1/1.18); applyZoom(); }, {passive:false});
  canvas.addEventListener("dblclick",()=>{ zoom=1; applyZoom(); });
  canvas.addEventListener("mousedown",e=>{ if(zoom<=1) return; e.preventDefault(); canvas._dragged=false;
    canvas.classList.add("grabbing"); const rect=canvas.getBoundingClientRect(); let lx=e.clientX, ly=e.clientY;
    const move=ev=>{ canvas._dragged=true; const img=canvas.querySelector("img");
      const w=parseFloat(img.style.width)||rect.width, h=parseFloat(img.style.height)||rect.height;
      const dx=ev.clientX-lx, dy=ev.clientY-ly;
      if(w>rect.width) focal.x=clamp(focal.x+dx/(rect.width-w),0,1);
      if(h>rect.height) focal.y=clamp(focal.y+dy/(rect.height-h),0,1);
      lx=ev.clientX; ly=ev.clientY; applyZoom(); };
    const up=()=>{ canvas.classList.remove("grabbing"); setTimeout(()=>canvas._dragged=false,0);
      window.removeEventListener("mousemove",move); window.removeEventListener("mouseup",up); };
    window.addEventListener("mousemove",move); window.addEventListener("mouseup",up); });
}

function openLightbox(){ const it=DATA.items[i]; if(!it)return;
  document.getElementById("lightboxImg").src=it.source;
  document.getElementById("lightboxCap").textContent="Source · "+it.name;
  document.getElementById("lightbox").classList.add("on"); }

function render(){
  const v=DATA.items; if(i<0)i=0; if(i>=v.length)i=v.length-1;
  const it=v[i]; if(!it)return;
  document.getElementById("idx").textContent=i+1;
  document.getElementById("total").textContent=v.length;
  document.getElementById("name").textContent=it.name;
  const grid=document.getElementById("grid"); grid.innerHTML="";

  const srcTile=document.createElement("div");
  srcTile.className="tile orig";
  srcTile.innerHTML=`
    <div class="canvas" data-orig="1">
      <span class="label">Source</span>
      <img draggable="false" src="${it.source}" alt="source">
    </div>
    <div class="ctrls orig"><button class="fullBtn">View full ⤢</button></div>`;
  srcTile.querySelector(".fullBtn").onclick=openLightbox;
  const srcCv=srcTile.querySelector(".canvas");
  srcCv.querySelector("img").onload=()=>sizeCanvas(srcCv);
  wireInspect(srcCv);
  grid.appendChild(srcTile);

  it.outs.forEach(o=>{
    const tile=document.createElement("div");
    if(!o.src){
      tile.className="tile";
      tile.innerHTML=`<div class="canvas blend missing"><span class="label">${o.m}</span><span class="miss">no prediction</span></div>`;
      grid.appendChild(tile);
      return;
    }
    tile.className="tile";
    const maskHtml = o.mask
      ? `<div class="canvas mask" data-m="${o.m}-mask">
          <span class="label">mask</span>
          <img draggable="false" src="${o.mask}" alt="${o.m} mask">
        </div>`
      : `<div class="canvas mask missing"><span class="label">mask</span><span class="miss">no mask</span></div>`;
    tile.innerHTML=`
      <div class="canvas blend" data-m="${o.m}">
        <span class="label">${o.m}</span>
        <img draggable="false" src="${o.src}" alt="${o.m}">
      </div>
      ${maskHtml}`;
    const cv=tile.querySelector(".canvas:not(.mask)");
    cv.querySelector("img").onload=()=>sizeCanvas(cv);
    wireInspect(cv);
    const maskCv=tile.querySelector(".canvas.mask");
    if(!maskCv.classList.contains("missing")){
      maskCv.querySelector("img").onload=()=>sizeCanvas(maskCv);
      wireInspect(maskCv);
    }
    grid.appendChild(tile);
  });

  layoutGrid(); applyZoom(); applyBg(); applyMaskToggle(); applyBlendToggle(); applySourceToggle();
  const nx=v[i+1]; if(nx){ nx.outs.forEach(o=>{
    if(o.src){ const im=new Image(); im.src=o.src; }
    if(o.mask){ const im=new Image(); im.src=o.mask; }
  }); }
}

function go(d){ i=clamp(i+d,0,DATA.items.length-1); localStorage.setItem("bgrm_view_idx", i); render(); }

const sws=document.getElementById("swatches");
BGS.forEach(([key,col])=>{ const b=document.createElement("button");
  b.className="sw"+(key==="checker"?" checker":""); b.dataset.bg=key;
  if(key!=="checker") b.style.background=col; b.title=key;
  b.onclick=()=>{ bg=key; localStorage.setItem("bgrm_view_bg", bg); applyBg(); }; sws.appendChild(b); });
document.getElementById("bgpick").oninput=e=>{ bg=e.target.value; localStorage.setItem("bgrm_view_bg", bg); applyBg(); };
document.getElementById("zoom").oninput=e=>{ zoom=parseFloat(e.target.value); applyZoom(); };
document.getElementById("zreset").onclick=()=>{ zoom=1; applyZoom(); };
document.getElementById("maskToggle").onclick=()=>{
  showMask=!showMask; localStorage.setItem("bgrm_view_showmask", showMask?"1":"0"); applyMaskToggle();
};
document.getElementById("blendToggle").onclick=()=>{
  showBlend=!showBlend; localStorage.setItem("bgrm_view_showblend", showBlend?"1":"0"); applyBlendToggle();
};
document.getElementById("sourceToggle").onclick=()=>{
  showSource=!showSource; localStorage.setItem("bgrm_view_showsource", showSource?"1":"0"); applySourceToggle();
};
document.getElementById("prev").onclick=()=>go(-1);
document.getElementById("next").onclick=()=>go(1);
const lb=document.getElementById("lightbox");
lb.onclick=()=>lb.classList.remove("on");
window.addEventListener("resize", ()=>{ layoutGrid(); applyZoom(); });

document.addEventListener("keydown",e=>{
  if(e.key==="Escape"){ lb.classList.remove("on"); return; }
  if(e.key==="ArrowRight"){ e.preventDefault(); if(zoom>1) pan(1,0); else go(1); }
  else if(e.key==="ArrowLeft"){ e.preventDefault(); if(zoom>1) pan(-1,0); else go(-1); }
  else if(e.key==="ArrowUp"){ e.preventDefault(); if(zoom>1) pan(0,-1); }
  else if(e.key==="ArrowDown"){ e.preventDefault(); if(zoom>1) pan(0,1); }
  else if(e.key==="n"||e.key==="N"){ e.preventDefault(); go(1); }
  else if(e.key==="p"||e.key==="P"){ e.preventDefault(); go(-1); }
  else if(e.key==="0"){ zoom=1; applyZoom(); }
  else if(e.key==="+"||e.key==="="){ zoom*=1.18; applyZoom(); }
  else if(e.key==="-"||e.key==="_"){ zoom*=1/1.18; applyZoom(); }
});

document.getElementById("total").textContent = DATA.items.length;
render();
</script>
</body>
</html>
"""


def _read_images_arg(args):
    paths = []
    if args.images:
        paths.extend(p.strip() for p in args.images.split(",") if p.strip())
    if args.images_file:
        with open(args.images_file) as f:
            paths.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments",
                     help="comma-separated experiment names (the H in .../results/H/...)")
    ap.add_argument("--images", help="comma-separated source image paths")
    ap.add_argument("--images-file", help="text file, one image path per line (alternative/addition to --images)")
    ap.add_argument("--out", default=None,
                     help=f"output dir for index.html + blended pngs (default: fixed dir {OUT_DIR_DEFAULT!r}, "
                          "so the URL stays the same across runs)")
    ap.add_argument("--rgba", action="store_true",
                     help="treat predictions as 4-channel RGB+alpha: show RGB*alpha "
                          "(cv2 channel order) instead of blending the source onto green")
    ap.add_argument("--ncols", type=int, default=None,
                     help="experiment tiles per row (source tile is extra). "
                          "Default: auto-fit to the viewport")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    ap.add_argument("--foreground", action="store_true",
                     help="block in this process (Ctrl+C to stop) instead of the default "
                          "detached background server")
    ap.add_argument("--serve-only", metavar="OUT_DIR", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.serve_only:
        # internal: used by _spawn_background_server to run the detached child.
        # The parent already did the kill-existing check, so bind directly.
        _serve_blocking(args.serve_only, args.port, not args.no_browser)
        return

    if not args.experiments:
        ap.error("--experiments is required")
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    image_paths = _read_images_arg(args)
    if not image_paths:
        ap.error("no images given: pass --images and/or --images-file")
    if args.ncols is not None and args.ncols < 1:
        ap.error("--ncols must be >= 1")

    view_experiments(experiments, image_paths, port=args.port,
                      open_browser=not args.no_browser, out_dir=args.out,
                      background=not args.foreground, rgba=args.rgba,
                      ncols=args.ncols)


if __name__ == "__main__":
    main()
