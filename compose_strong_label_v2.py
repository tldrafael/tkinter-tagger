#!/usr/bin/env python3
"""
Mask Tagger – batch mask comparison and composition tool.

Receives a text file listing image paths (one per line) and derives four
alpha-grayscale masks for each.  Displays a two-row comparison grid
(blended overlays on top, raw masks on the bottom) and provides tagging
buttons that log to a text file then advance to the next image.
A "Compose" button opens a floating editor where a base mask can be
selectively overwritten with regions from other masks using a brush tool.

Usage:
    python tag_masks.py <image_list.txt> [--log log.txt]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
import tkinter as tk

from typing import List, Optional

import numpy as np
from PIL import Image, ImageTk


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_LOG = "labels.txt"
GREEN_BG = np.array([0.0, 1.0, 0.0], dtype=np.float32)

# MASK_CASES = [
#     "bgremoval3.dt20260201",
#     "20260106-dinov3-cnnVit-600k-L384-unfrzBB-algnCrnr-higherOS_L768_tuned_L1152-noIS_L1536_L1920C",
#     "bgremoval3.dinov3.dt20251028",
#     "BiRefNet_HRPP",
# ]

MASK_CASES = [
    # "../gt",
    # "20260329-from20260315-capEdge1152-algnCrnr-L1536_L1920-frzBB_L2304_matLossPP",
    # "20260413-from20260106-L768_tuned+expandedGS_L1152-noIS-bf16_cap1152-L1536_L1920-frzBBViT_L2304_matLossPP",
    "bgremoval3.singlearch.dt20260402",
    "bgremoval3.dt20260320",
    "bgremoval3.dinov3.dt20251028",
    # "bgremoval3.singlearch.dt20260611",
    "BiRefNet_HRPP",
    "20260529_fullrun-dinov3cnn-tipsv2_L384_L768_tuned_L1152-noIS-frzTIPS_L1536_L1920.bakPP",

]

OPTIONAL_CASES = {"../gt"}


def find_resume_index(log_file: str, image_paths: List[Path]) -> int:
    """Find the index to resume from by reading the last image path in the log."""
    log_path = Path(log_file)
    if not log_path.exists():
        return 0

    last_image = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if parts:
                last_image = parts[0]

    if last_image is None:
        return 0

    last_path = Path(last_image)
    for i, ip in enumerate(image_paths):
        if ip == last_path:
            return min(i + 1, len(image_paths) - 1)

    return 0


def derive_mask_paths(image_path: Path) -> List[Path]:
    """Derive the four mask paths from an image path.

    Replaces the extension .jpg -> .png and '/im/' -> '/results/{case}/'.
    """
    s = str(image_path)
    paths = []
    for case in MASK_CASES:
        mp = s.replace("/im/", f"/results/{case}/")
        mp = mp.rsplit(".", 1)[0] + ".png"
        paths.append(Path(mp))
    return paths


# ============================================================================
# Image helpers
# ============================================================================

def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path, fallback_size: Optional[tuple] = None) -> Image.Image:
    """Load a mask from *path*.

    If the file is missing or fails to load, returns a blank (all-zero)
    mask instead of raising, so a single missing mask doesn't abort the
    whole application. *fallback_size* controls the size of the blank
    mask when one is needed; defaults to a 1x1 placeholder.
    """
    try:
        img = Image.open(path)
    except (FileNotFoundError, OSError) as e:
        print(f"Warning: could not load mask {path} ({e}); using blank mask.")
        return Image.new("L", fallback_size or (1, 1), 0)
    if img.mode in ("RGBA", "LA"):
        return img.split()[-1]
    return img.convert("L")


def resize_to_fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    w = max(1, int(img.width * scale))
    h = max(1, int(img.height * scale))
    return img.resize((w, h), Image.LANCZOS)


def blend_on_green(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Composite *image* over a green background using *mask* as alpha."""
    base = np.asarray(image, dtype=np.float32) / 255.0
    alpha = np.asarray(mask, dtype=np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    result = (1.0 - alpha) * GREEN_BG + base * alpha
    return Image.fromarray((np.clip(result, 0.0, 1.0) * 255).astype(np.uint8))


def mask_to_rgb(mask: Image.Image) -> Image.Image:
    return Image.merge("RGB", [mask, mask, mask])


def compute_zoom_geometry(
    native_w: int, native_h: int, win_w: int, win_h: int,
    scale: float, center_x: float, center_y: float,
):
    """Compute the crop box (in native coords) and display size for a
    zoomed/panned view of a *native_w* x *native_h* image inside a
    *win_w* x *win_h* window at the given display *scale* (display px
    per native px), centered as close as possible to
    (*center_x*, *center_y*).

    The crop box is clamped to the image bounds, which is what makes
    panning work: as *scale* grows the crop shrinks below the full
    image and *center_x*/*center_y* pick which region is visible.

    Returns (crop_box, disp_w, disp_h, clamped_center_x, clamped_center_y).
    """
    disp_w = max(1, min(win_w, round(native_w * scale)))
    disp_h = max(1, min(win_h, round(native_h * scale)))
    crop_w = disp_w / scale
    crop_h = disp_h / scale

    if crop_w >= native_w:
        cx = native_w / 2.0
    else:
        cx = min(max(center_x, crop_w / 2.0), native_w - crop_w / 2.0)
    if crop_h >= native_h:
        cy = native_h / 2.0
    else:
        cy = min(max(center_y, crop_h / 2.0), native_h - crop_h / 2.0)

    x0, y0 = cx - crop_w / 2.0, cy - crop_h / 2.0
    crop_box = (x0, y0, x0 + crop_w, y0 + crop_h)
    return crop_box, disp_w, disp_h, cx, cy


def zoom_anchor_from_pointer(widget, crop_box, disp_w: int, disp_h: int):
    """Return the native-space point currently under the mouse pointer
    within *widget*'s window, given the current *crop_box*/display size.

    Falls back to the crop center if the pointer isn't over the
    displayed image (e.g. it's over a letterboxed margin).
    """
    win_w = widget.winfo_width()
    win_h = widget.winfo_height()
    px = widget.winfo_pointerx() - widget.winfo_rootx()
    py = widget.winfo_pointery() - widget.winfo_rooty()
    img_left = (win_w - disp_w) / 2.0
    img_top = (win_h - disp_h) / 2.0
    x0, y0, x1, y1 = crop_box

    if (img_left <= px < img_left + disp_w
            and img_top <= py < img_top + disp_h):
        sx = (x1 - x0) / disp_w
        sy = (y1 - y0) / disp_h
        return x0 + (px - img_left) * sx, y0 + (py - img_top) * sy
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


# ============================================================================
# Main window
# ============================================================================

class MaskTaggerApp:
    def __init__(self, root: tk.Tk, image_paths: List[Path],
                 log_file: str):
        self.root = root
        self.image_paths_list = image_paths
        self.current_index = find_resume_index(log_file, image_paths)
        self.log_file = log_file

        self._compose_win: Optional[ComposeWindow] = None
        self._photo_refs: list = []

        # Zoom state (single-tile fullscreen overlay)
        self._zoom_overlay: Optional[tk.Canvas] = None
        self._zoom_photo = None
        self._zoom_tile_idx: int = 0
        self._tile_info: List[tuple] = []   # (type, mask_index, title)
        self._tile_labels: List[tk.Label] = []
        self._zoom_hires: Optional[Image.Image] = None
        self._zoom_title: str = ""
        self._zoom_native_w: int = 1
        self._zoom_native_h: int = 1
        self._zoom_fit_scale: float = 1.0
        self._zoom_scale: float = 1.0
        self._zoom_center_x: float = 0.0
        self._zoom_center_y: float = 0.0
        self._zoom_crop: tuple = (0.0, 0.0, 1.0, 1.0)
        self._zoom_disp_w: int = 1
        self._zoom_disp_h: int = 1

        # Live zoom mode: a synced magnifier applied to every tile at once
        self._zoom_mode_active: bool = False
        self._zoom_factor: float = 3.0
        self._zoom_mode_var: Optional[tk.BooleanVar] = None
        self._zoom_factor_var: Optional[tk.StringVar] = None
        self._zoom_refresh_pending: bool = False
        self._pending_zoom_pos: Optional[tuple] = None
        self._live_zoom_refs: list = []
        self._mask_rgb_full: List[Image.Image] = []

        self._load_entry()
        self._build_ui()

    def _load_entry(self):
        """Load the current image and its masks from disk."""
        image_path = self.image_paths_list[self.current_index]
        self.image_path = image_path
        all_mask_paths = derive_mask_paths(image_path)

        n = len(self.image_paths_list)
        i = self.current_index + 1
        self.root.title(f"Mask Tagger – {image_path.name}  ({i}/{n})")

        self.original = load_image(image_path)
        self.masks: List[Image.Image] = []
        self.mask_paths: List[Path] = []
        self.active_cases: List[str] = []
        for case, p in zip(MASK_CASES, all_mask_paths):
            if not p.exists() and case in OPTIONAL_CASES:
                print(p)
                continue
            m = load_mask(p, fallback_size=self.original.size)
            if m.size != self.original.size:
                m = m.resize(self.original.size, Image.LANCZOS)
            self.masks.append(m)
            self.mask_paths.append(p)
            self.active_cases.append(case)

        self.blends = [blend_on_green(self.original, m) for m in self.masks]

    # ── layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self._screen_size = (sw, sh)

        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        for r in range(2):
            self.grid_frame.rowconfigure(r, weight=1)
        self.grid_frame.bind("<Leave>", self._on_grid_leave)

        self._populate_grid()

        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(fill=tk.X, padx=0, pady=0)
        self._create_buttons()

        self.root.bind("z", lambda _: self._toggle_zoom_mode())
        self.root.bind("<plus>", lambda _: self._on_plus_key())
        self.root.bind("<equal>", lambda _: self._on_plus_key())
        self.root.bind("<minus>", lambda _: self._on_minus_key())

        try:
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                self.root.geometry(f"{sw}x{sh}+0+0")

    def _populate_grid(self):
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self._photo_refs.clear()
        self._tile_info.clear()
        self._tile_labels.clear()
        self._live_zoom_refs = []
        self._pending_zoom_pos = None
        self._mask_rgb_full = [mask_to_rgb(m) for m in self.masks]

        sw, sh = self._screen_size
        cols = len(self.masks) + 1
        btn_reserve = 50
        tw = sw // cols
        th = (sh - btn_reserve) // 2
        self.tile_size = (tw, th)

        for c in range(cols + 5):
            self.grid_frame.columnconfigure(c, weight=(1 if c < cols else 0))

        # Row 0: original + 4 blended images
        self._add_tile(resize_to_fit(self.original, tw, th), 0, 0,
                       "Original", ("original", -1, "Original"))
        for i, bl in enumerate(self.blends):
            self._add_tile(resize_to_fit(bl, tw, th), 0, i + 1,
                           f"Blend {i + 1}", ("blend", i, f"Blend {i + 1}"))

        # Row 1, col 0: dark placeholder (no mask for original)
        ph_img = Image.new("RGB", (1, 1), (30, 30, 30))
        self._add_tile(resize_to_fit(ph_img, tw, th), 1, 0,
                       "", ("empty", -1, ""))

        # Row 1, cols 1-4: masks
        for i, m in enumerate(self.masks):
            self._add_tile(resize_to_fit(mask_to_rgb(m), tw, th), 1, i + 1,
                           f"Mask {i + 1}", ("mask", i, f"Mask {i + 1}"))

    def _add_tile(self, img: Image.Image, row: int, col: int, title: str,
                  info: tuple):
        tile_idx = len(self._tile_info)
        self._tile_info.append(info)

        photo = ImageTk.PhotoImage(img)
        self._photo_refs.append(photo)

        frame = tk.Frame(self.grid_frame)
        frame.grid(row=row, column=col, padx=0, pady=0, sticky="nsew")

        if title:
            tk.Label(frame, text=title, font=("Helvetica", 8), pady=0).pack(
                side=tk.TOP)

        lbl = tk.Label(frame, image=photo, cursor="hand2", padx=0, pady=0)
        lbl.image = photo
        lbl.pack(side=tk.TOP, padx=0, pady=0)
        lbl.bind("<Button-1>", lambda _, idx=tile_idx: self._on_tile_click(idx))
        lbl.bind("<Motion>", lambda e, idx=tile_idx: self._on_tile_motion(e, idx))
        lbl.bind("<MouseWheel>", self._on_zoom_wheel)
        lbl.bind("<Button-4>", lambda _: self._adjust_zoom_factor(1.2))
        lbl.bind("<Button-5>", lambda _: self._adjust_zoom_factor(1 / 1.2))
        self._tile_labels.append(lbl)

    # ── zoom overlay ──────────────────────────────────────────────────────

    def _load_hires(self, tile_type: str, mask_idx: int) -> Image.Image:
        """Reload from disk at full resolution to preserve fine details."""
        if tile_type == "original":
            return load_image(self.image_path)
        elif tile_type == "mask":
            m = load_mask(self.mask_paths[mask_idx], fallback_size=self.original.size)
            if m.size != self.original.size:
                m = m.resize(self.original.size, Image.LANCZOS)
            return mask_to_rgb(m)
        elif tile_type == "blend":
            orig = load_image(self.image_path)
            m = load_mask(self.mask_paths[mask_idx], fallback_size=orig.size)
            if m.size != orig.size:
                m = m.resize(orig.size, Image.LANCZOS)
            return blend_on_green(orig, m)
        return Image.new("RGB", (100, 100), (30, 30, 30))

    def _on_tile_click(self, tile_idx: int):
        tile_type, _, _ = self._tile_info[tile_idx]
        if tile_type == "empty":
            return
        if self._zoom_overlay is not None:
            return
        self._show_zoom(tile_idx)

    def _show_zoom(self, tile_idx: int):
        tile_type, mask_idx, title = self._tile_info[tile_idx]
        if tile_type == "empty":
            return
        # Only reset the zoom frame on a fresh open; navigating between
        # tiles while already zoomed in keeps the same scale/center so you
        # can compare the same region across tiles.
        reset_view = self._zoom_overlay is None
        self._zoom_tile_idx = tile_idx
        self._zoom_title = title

        self._zoom_hires = self._load_hires(tile_type, mask_idx)
        self._zoom_native_w, self._zoom_native_h = self._zoom_hires.size

        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()
        self._zoom_fit_scale = min(
            win_w / self._zoom_native_w, win_h / self._zoom_native_h, 1.0,
        )
        if reset_view:
            self._zoom_scale = self._zoom_fit_scale
            self._zoom_center_x = self._zoom_native_w / 2.0
            self._zoom_center_y = self._zoom_native_h / 2.0
        else:
            self._zoom_scale = max(
                self._zoom_fit_scale, min(1.0, self._zoom_scale),
            )

        self._render_tile_zoom(create=reset_view)

        if reset_view:
            self._zoom_overlay.bind("<Button-1>", lambda _: self._close_zoom())
            self.root.bind("<Escape>", lambda _: self._close_zoom())
            self.root.bind("<Left>", lambda _: self._zoom_navigate(-1))
            self.root.bind("<Right>", lambda _: self._zoom_navigate(1))

    def _render_tile_zoom(self, create: bool):
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()

        crop_box, disp_w, disp_h, cx, cy = compute_zoom_geometry(
            self._zoom_native_w, self._zoom_native_h, win_w, win_h,
            self._zoom_scale, self._zoom_center_x, self._zoom_center_y,
        )
        self._zoom_crop = crop_box
        self._zoom_disp_w, self._zoom_disp_h = disp_w, disp_h
        self._zoom_center_x, self._zoom_center_y = cx, cy

        x0, y0, x1, y1 = crop_box
        crop = self._zoom_hires.crop(
            (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
        )
        resample = Image.NEAREST if self._zoom_scale >= 0.999 else Image.LANCZOS
        disp = crop.resize((disp_w, disp_h), resample)

        pct = round(self._zoom_scale * 100)
        title = f"{self._zoom_title}  ({pct}%)"
        cx_screen, cy_screen = win_w // 2, win_h // 2
        self._zoom_photo = ImageTk.PhotoImage(disp)

        if create:
            self._zoom_overlay = tk.Canvas(
                self.root, highlightthickness=0,
            )
            self._zoom_overlay.place(x=0, y=0, relwidth=1, relheight=1)

            self._zoom_overlay.create_rectangle(
                0, 0, win_w, win_h, fill="black", stipple="gray50",
            )

            self._zoom_img_id = self._zoom_overlay.create_image(
                cx_screen, cy_screen, image=self._zoom_photo, anchor="center",
            )
            self._zoom_title_id = self._zoom_overlay.create_text(
                win_w // 2, 24, text=title,
                fill="white", font=("Helvetica", 16, "bold"), anchor="center",
            )
        else:
            self._zoom_overlay.itemconfig(
                self._zoom_img_id, image=self._zoom_photo)
            self._zoom_overlay.coords(
                self._zoom_img_id, cx_screen, cy_screen)
            self._zoom_overlay.itemconfig(self._zoom_title_id, text=title)
            self._zoom_overlay.coords(
                self._zoom_title_id, win_w // 2, 24)

    def _adjust_tile_zoom(self, mult: float):
        if self._zoom_overlay is None:
            return
        anchor_x, anchor_y = zoom_anchor_from_pointer(
            self.root, self._zoom_crop, self._zoom_disp_w, self._zoom_disp_h,
        )
        new_scale = max(self._zoom_fit_scale, min(1.0, self._zoom_scale * mult))
        if new_scale == self._zoom_scale:
            return
        self._zoom_scale = new_scale
        self._zoom_center_x, self._zoom_center_y = anchor_x, anchor_y
        self._render_tile_zoom(create=False)

    def _zoom_navigate(self, direction: int):
        if self._zoom_overlay is None:
            return
        idx = self._zoom_tile_idx + direction
        while 0 <= idx < len(self._tile_info):
            if self._tile_info[idx][0] != "empty":
                self._show_zoom(idx)
                return
            idx += direction

    def _close_zoom(self):
        if self._zoom_overlay is not None:
            self._zoom_overlay.destroy()
            self._zoom_overlay = None
            self._zoom_photo = None
        for key in ("<Escape>", "<Left>", "<Right>"):
            try:
                self.root.unbind(key)
            except Exception:
                pass

    # ── zoom mode (synced magnifier across every tile) ─────────────────────

    def _tile_source_image(self, tile_type: str, mask_idx: int) -> Optional[Image.Image]:
        """Full-resolution source image backing a given tile."""
        if tile_type == "original":
            return self.original
        if tile_type == "blend":
            return self.blends[mask_idx]
        if tile_type == "mask":
            return self._mask_rgb_full[mask_idx]
        return None

    def _toggle_zoom_mode(self):
        self._zoom_mode_active = not self._zoom_mode_active
        if self._zoom_mode_var is not None:
            self._zoom_mode_var.set(self._zoom_mode_active)
        if not self._zoom_mode_active:
            self._restore_thumbnails()

    def _adjust_zoom_factor(self, mult: float):
        self._zoom_factor = max(1.2, min(20.0, self._zoom_factor * mult))
        if self._zoom_factor_var is not None:
            self._zoom_factor_var.set(f"Zoom: {self._zoom_factor:.1f}x")
        if self._zoom_mode_active and self._pending_zoom_pos is not None:
            self._apply_live_zoom(*self._pending_zoom_pos)

    def _on_plus_key(self):
        """'+'/'=' zooms the fullscreen tile overlay if it's open,
        otherwise it adjusts the synced-magnifier zoom factor."""
        if self._zoom_overlay is not None:
            self._adjust_tile_zoom(1.25)
        else:
            self._adjust_zoom_factor(1.5)

    def _on_minus_key(self):
        if self._zoom_overlay is not None:
            self._adjust_tile_zoom(1 / 1.25)
        else:
            self._adjust_zoom_factor(1 / 1.5)

    def _on_zoom_wheel(self, event):
        if not self._zoom_mode_active:
            return
        mult = 1.2 if event.delta > 0 else 1 / 1.2
        self._adjust_zoom_factor(mult)

    def _on_tile_motion(self, event, tile_idx: int):
        if not self._zoom_mode_active:
            return
        tile_type, _, _ = self._tile_info[tile_idx]
        if tile_type == "empty":
            return
        lbl = self._tile_labels[tile_idx]
        w, h = lbl.winfo_width(), lbl.winfo_height()
        if w <= 1 or h <= 1:
            return
        rx = min(max(event.x / w, 0.0), 1.0)
        ry = min(max(event.y / h, 0.0), 1.0)
        self._pending_zoom_pos = (rx, ry)
        if not self._zoom_refresh_pending:
            self._zoom_refresh_pending = True
            self.root.after(20, self._do_live_zoom)

    def _on_grid_leave(self, event):
        if self._zoom_mode_active:
            self._restore_thumbnails()

    def _do_live_zoom(self):
        self._zoom_refresh_pending = False
        if not self._zoom_mode_active or self._pending_zoom_pos is None:
            return
        self._apply_live_zoom(*self._pending_zoom_pos)

    def _apply_live_zoom(self, rx: float, ry: float):
        """Show a magnified crop centered at fractional (rx, ry) in every tile."""
        orig_w, orig_h = self.original.size
        factor = self._zoom_factor
        crop_w = max(1, int(orig_w / factor))
        crop_h = max(1, int(orig_h / factor))
        cx = int(rx * orig_w)
        cy = int(ry * orig_h)
        x1 = min(max(cx - crop_w // 2, 0), max(orig_w - crop_w, 0))
        y1 = min(max(cy - crop_h // 2, 0), max(orig_h - crop_h, 0))
        x2 = min(x1 + crop_w, orig_w)
        y2 = min(y1 + crop_h, orig_h)

        refs = []
        for idx, (tile_type, mask_idx, _title) in enumerate(self._tile_info):
            if tile_type == "empty":
                continue
            src = self._tile_source_image(tile_type, mask_idx)
            if src is None:
                continue
            lbl = self._tile_labels[idx]
            disp_w, disp_h = lbl.winfo_width(), lbl.winfo_height()
            if disp_w <= 1 or disp_h <= 1:
                continue
            crop = src.crop((x1, y1, x2, y2)).resize(
                (disp_w, disp_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(crop)
            refs.append(photo)
            lbl.config(image=photo)
            lbl.image = photo
        self._live_zoom_refs = refs

    def _restore_thumbnails(self):
        self._pending_zoom_pos = None
        for idx, lbl in enumerate(self._tile_labels):
            if idx >= len(self._photo_refs):
                continue
            photo = self._photo_refs[idx]
            lbl.config(image=photo)
            lbl.image = photo
        self._live_zoom_refs = []

    # ── buttons ─────────────────────────────────────────────────────────────

    def _create_buttons(self):
        for w in self.btn_frame.winfo_children():
            w.destroy()
        for k in range(1, 10):
            try:
                self.root.unbind(str(k))
            except Exception:
                pass

        style = {"height": 1, "font": ("Helvetica", 8)}
        btn_idx = 1

        for i in range(len(self.active_cases)):
            case_name = self.active_cases[i]
            label = f"{btn_idx}. {case_name}"
            tk.Button(
                self.btn_frame, text=label,
                command=lambda name=case_name: self._tag(name), **style,
            ).pack(side=tk.LEFT, padx=2, pady=2)
            self.root.bind(str(btn_idx),
                           lambda _, name=case_name: self._tag(name))
            btn_idx += 1

        tk.Button(
            self.btn_frame, text=f"{btn_idx}. odd",
            command=lambda: self._tag("odd"), **style,
        ).pack(side=tk.LEFT, padx=2, pady=2)
        self.root.bind(str(btn_idx), lambda _: self._tag("odd"))
        btn_idx += 1

        tk.Button(
            self.btn_frame, text=f"{btn_idx}. all bad",
            command=lambda: self._tag("all bad"), **style,
        ).pack(side=tk.LEFT, padx=2, pady=2)
        self.root.bind(str(btn_idx), lambda _: self._tag("all bad"))
        btn_idx += 1

        tk.Button(
            self.btn_frame, text=f"{btn_idx}. Skip",
            command=self._advance,
            height=1, font=("Helvetica", 8),
        ).pack(side=tk.LEFT, padx=2, pady=2)
        self.root.bind(str(btn_idx), lambda _: self._advance())
        btn_idx += 1

        # spacer
        tk.Frame(self.btn_frame, width=20).pack(side=tk.LEFT)

        tk.Button(
            self.btn_frame, text=f"{btn_idx}. Compose",
            command=self._open_compose,
            height=1, font=("Helvetica", 8, "bold"),
        ).pack(side=tk.LEFT, padx=2, pady=2)
        self.root.bind(str(btn_idx), lambda _: self._open_compose())

        self.root.bind("q", lambda _: self.root.quit())

        # Zoom mode: a synced magnifier applied to every tile simultaneously
        self._zoom_mode_var = tk.BooleanVar(value=self._zoom_mode_active)
        tk.Checkbutton(
            self.btn_frame, text="Zoom Mode (Z)",
            variable=self._zoom_mode_var,
            command=self._toggle_zoom_mode,
            font=("Helvetica", 8, "bold"),
        ).pack(side=tk.RIGHT, padx=(2, 8), pady=2)

        self._zoom_factor_var = tk.StringVar(
            value=f"Zoom: {self._zoom_factor:.1f}x")
        tk.Label(
            self.btn_frame, textvariable=self._zoom_factor_var,
            font=("Helvetica", 8),
        ).pack(side=tk.RIGHT, padx=2, pady=2)

    def _tag(self, label: str):
        with open(self.log_file, "a") as f:
            f.write(f"{self.image_path},{label}\n")
        self._advance()

    def _advance(self):
        """Move to the next image, or quit if all images are processed."""
        self._close_zoom()
        if self._compose_win is not None:
            try:
                self._compose_win._on_close()
            except Exception:
                pass
            self._compose_win = None
        self.current_index += 1
        if self.current_index >= len(self.image_paths_list):
            self.root.quit()
            return
        self._load_entry()
        self._populate_grid()
        self._create_buttons()

    def _open_compose(self):
        if self._compose_win is not None:
            try:
                self._compose_win.window.lift()
                return
            except tk.TclError:
                self._compose_win = None

        self._compose_win = ComposeWindow(
            self.root, self.original, self.masks,
            self.mask_paths, self.image_path, app=self,
        )


# ============================================================================
# Compose window
# ============================================================================

class ComposeWindow:
    """Editor for composing a new mask from multiple sources."""

    BRUSH_SIZES = [5, 15, 30, 50, 80]
    BRUSH_MODES = ["Replace", "Union", "Intersect"]
    MAX_UNDO = 50

    def __init__(self, parent: tk.Tk, original: Image.Image,
                 masks: List[Image.Image], mask_paths: List[Path],
                 image_path: Path, app: 'MaskTaggerApp'):
        self.parent = parent
        self.original = original
        self.masks = masks
        self.mask_paths = mask_paths
        self.image_path = image_path
        self.app = app

        self.composed: Optional[np.ndarray] = None
        self.base_idx: Optional[int] = None
        self.donor_idx: int = -1
        self.brush_size = self.BRUSH_SIZES[2]
        self._brush_mode = "Replace"
        self._eraser_mode = False
        self._pencil_mode = False
        self._refresh_pending = False
        self._zoom_refresh_pending = False
        self._photo_refs: list = []
        self._undo_stack: list = []

        # Zoom state
        self._zoom_overlay: Optional[tk.Canvas] = None
        self._zoom_type: str = ""
        self._zoom_photo = None
        self._zoom_disp_w: int = 0
        self._zoom_disp_h: int = 0
        self._zoom_sx: float = 1.0
        self._zoom_sy: float = 1.0
        self._zoom_fit_scale: float = 1.0
        self._zoom_scale: float = 1.0
        self._zoom_center_x: float = 0.0
        self._zoom_center_y: float = 0.0
        self._zoom_crop: tuple = (0.0, 0.0, 1.0, 1.0)

        self.window = tk.Toplevel(parent)
        self.window.title("Compose Mask")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self._maximize_window()
        self._show_base_picker()

    def _maximize_window(self):
        self.window.update_idletasks()
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        self.window.geometry(f"{sw}x{sh}+0+0")

    # ── step 1: pick the base mask ──────────────────────────────────────────

    def _show_base_picker(self):
        for w in self.window.winfo_children():
            w.destroy()
        self._photo_refs.clear()

        tk.Label(
            self.window,
            text="Click a mask (or press 1–4) to use as the base:",
            font=("Helvetica", 14, "bold"),
        ).pack(pady=10)

        row = tk.Frame(self.window)
        row.pack(pady=4)

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        n = len(self.masks)
        thumb_w = (sw - 40 - n * 16) // n
        thumb_h = sh - 200
        for i, m in enumerate(self.masks):
            frm = tk.Frame(row, bd=2, relief=tk.RAISED, cursor="hand2")
            frm.grid(row=0, column=i, padx=8, pady=4)

            disp = resize_to_fit(mask_to_rgb(m), thumb_w, thumb_h)
            ph = ImageTk.PhotoImage(disp)
            self._photo_refs.append(ph)

            lbl = tk.Label(frm, image=ph)
            lbl.image = ph
            lbl.pack()
            lbl.bind("<Button-1>", lambda _, idx=i: self._pick_base(idx))
            frm.bind("<Button-1>", lambda _, idx=i: self._pick_base(idx))

            tk.Label(frm, text=f"{i + 1}. Mask {i + 1}",
                     font=("Helvetica", 11)).pack()

            self.window.bind(
                str(i + 1), lambda _, idx=i: self._pick_base(idx))

    def _pick_base(self, idx: int):
        for i in range(len(self.masks)):
            try:
                self.window.unbind(str(i + 1))
            except Exception:
                pass
        self.base_idx = idx
        self._undo_stack.clear()
        self.composed = np.asarray(self.masks[idx], dtype=np.uint8).copy()
        self._show_editor()

    # ── step 2: painting editor ─────────────────────────────────────────────

    def _show_editor(self):
        for w in self.window.winfo_children():
            w.destroy()
        self._photo_refs.clear()

        # ── control bar ──
        ctrl = tk.Frame(self.window)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        tk.Label(ctrl, text=f"Base: Mask {self.base_idx + 1}",
                 font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, padx=8)

        tk.Label(ctrl, text="Donor:",
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(20, 4))

        self._donor_var = tk.IntVar(value=-1)
        for i in range(len(self.masks)):
            tk.Radiobutton(
                ctrl, text=f"Mask {i + 1}",
                variable=self._donor_var, value=i,
                font=("Helvetica", 11),
                command=self._set_donor,
            ).pack(side=tk.LEFT, padx=2)

        tk.Label(ctrl, text="Brush:",
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(20, 4))

        self._brush_var = tk.IntVar(value=self.brush_size)
        self._brush_var.trace_add("write", lambda *_: self._set_brush())
        for bs in self.BRUSH_SIZES:
            tk.Radiobutton(
                ctrl, text=str(bs),
                variable=self._brush_var, value=bs,
                font=("Helvetica", 10),
                command=self._set_brush,
            ).pack(side=tk.LEFT, padx=1)

        self._brush_spinbox = tk.Spinbox(
            ctrl, from_=1, to=500, width=5,
            textvariable=self._brush_var,
            font=("Helvetica", 10),
            command=self._set_brush,
        )
        self._brush_spinbox.pack(side=tk.LEFT, padx=(4, 2))
        self._brush_spinbox.bind("<Return>", lambda _: self._set_brush())

        self._eraser_var = tk.BooleanVar(value=self._eraser_mode)
        self._eraser_cb = tk.Checkbutton(
            ctrl, text="Eraser (E)",
            variable=self._eraser_var,
            font=("Helvetica", 11),
            command=self._toggle_eraser,
        )
        self._eraser_cb.pack(side=tk.LEFT, padx=(20, 4))

        self._pencil_var = tk.BooleanVar(value=self._pencil_mode)
        self._pencil_cb = tk.Checkbutton(
            ctrl, text="Pencil (P)",
            variable=self._pencil_var,
            font=("Helvetica", 11),
            command=self._toggle_pencil,
        )
        self._pencil_cb.pack(side=tk.LEFT, padx=(4, 4))

        tk.Label(ctrl, text="Mode:",
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(20, 4))
        self._mode_var = tk.StringVar(value=self._brush_mode)
        for mode in self.BRUSH_MODES:
            tk.Radiobutton(
                ctrl, text=mode,
                variable=self._mode_var, value=mode,
                font=("Helvetica", 10),
                command=self._set_brush_mode,
            ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            ctrl, text="Save (S)", command=self._save,
            font=("Helvetica", 11, "bold"), width=8,
        ).pack(side=tk.RIGHT, padx=4)

        tk.Button(
            ctrl, text="Reset (R)", command=self._reset,
            font=("Helvetica", 11), width=8,
        ).pack(side=tk.RIGHT, padx=4)

        tk.Button(
            ctrl, text="Undo (⌘Z)", command=self._undo,
            font=("Helvetica", 11), width=10,
        ).pack(side=tk.RIGHT, padx=4)

        self.window.bind("s", lambda _: self._save())
        self.window.bind("r", lambda _: self._reset())
        self.window.bind("e", lambda _: self._toggle_eraser_key())
        self.window.bind("p", lambda _: self._toggle_pencil_key())
        self.window.bind("<Command-z>", lambda _: self._undo())
        self.window.bind("<Control-z>", lambda _: self._undo())
        self.window.bind("<plus>", lambda _: self._adjust_zoom_panel(1.25))
        self.window.bind("<equal>", lambda _: self._adjust_zoom_panel(1.25))
        self.window.bind("<minus>", lambda _: self._adjust_zoom_panel(1 / 1.25))

        for digit in range(10):
            self.window.bind(
                str(digit), lambda _, d=digit: self._digit_brush(d))

        # ── compute canvas dimensions (4 panels side-by-side) ──
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        max_cw = (sw - 40) // 4
        max_ch = sh - 140

        scale = min(max_cw / self.original.width,
                    max_ch / self.original.height, 1.0)
        self.cw = max(1, int(self.original.width * scale))
        self.ch = max(1, int(self.original.height * scale))
        self.sx = self.original.width / self.cw
        self.sy = self.original.height / self.ch

        self._disp_orig = self.original.resize((self.cw, self.ch),
                                               Image.LANCZOS)

        # ── canvases ──
        body = tk.Frame(self.window)
        body.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Left: donor mask (paintable – brush here also overwrites composed)
        df = tk.Frame(body)
        df.pack(side=tk.LEFT, padx=1)
        self._donor_label = tk.Label(
            df, text="Donor Mask  (select one)",
            font=("Helvetica", 11),
        )
        self._donor_label.pack()
        self.donor_canvas = tk.Canvas(
            df, width=self.cw, height=self.ch,
            bg="black", cursor="crosshair",
        )
        self.donor_canvas.pack()

        self.donor_canvas.bind("<B1-Motion>", self._on_paint)
        self.donor_canvas.bind("<Button-1>", self._on_paint_start)
        self.donor_canvas.bind("<Motion>", self._on_hover_donor)
        self.donor_canvas.bind("<Button-2>",
                               lambda _: self._open_zoom("donor"))
        self.donor_canvas.bind("<Button-3>",
                               lambda _: self._open_zoom("donor"))

        self._donor_brush_oval = self.donor_canvas.create_oval(
            0, 0, 0, 0, outline="cyan", width=2,
        )

        # Center: composed mask (paintable)
        cf = tk.Frame(body)
        cf.pack(side=tk.LEFT, padx=1)
        tk.Label(cf, text="Composed Mask  (paint here)",
                 font=("Helvetica", 11)).pack()
        self.mask_canvas = tk.Canvas(
            cf, width=self.cw, height=self.ch,
            bg="black", cursor="crosshair",
        )
        self.mask_canvas.pack()

        self.mask_canvas.bind("<B1-Motion>", self._on_paint)
        self.mask_canvas.bind("<Button-1>", self._on_paint_start)
        self.mask_canvas.bind("<Motion>", self._on_hover_composed)
        self.mask_canvas.bind("<Button-2>",
                              lambda _: self._open_zoom("composed"))
        self.mask_canvas.bind("<Button-3>",
                              lambda _: self._open_zoom("composed"))

        self._brush_oval = self.mask_canvas.create_oval(
            0, 0, 0, 0, outline="red", width=2,
        )

        # Blended preview (paintable with eraser/pencil)
        rf = tk.Frame(body)
        rf.pack(side=tk.LEFT, padx=1)
        tk.Label(rf, text="Blended Preview",
                 font=("Helvetica", 11)).pack()
        self.preview_canvas = tk.Canvas(
            rf, width=self.cw, height=self.ch,
            bg="black", cursor="crosshair",
        )
        self.preview_canvas.pack()
        self.preview_canvas.bind("<B1-Motion>", self._on_paint)
        self.preview_canvas.bind("<Button-1>", self._on_paint_start)
        self.preview_canvas.bind("<Motion>", self._on_hover_preview)
        self.preview_canvas.bind("<Button-2>",
                                 lambda _: self._open_zoom("blend"))
        self.preview_canvas.bind("<Button-3>",
                                 lambda _: self._open_zoom("blend"))

        self._preview_brush_oval = self.preview_canvas.create_oval(
            0, 0, 0, 0, outline="cyan", width=2,
        )

        # Right: original image (paintable with eraser/pencil)
        of = tk.Frame(body)
        of.pack(side=tk.LEFT, padx=1)
        tk.Label(of, text="Original Image",
                 font=("Helvetica", 11)).pack()
        self.original_canvas = tk.Canvas(
            of, width=self.cw, height=self.ch,
            bg="black", cursor="crosshair",
        )
        self.original_canvas.pack()
        self.original_canvas.bind("<B1-Motion>", self._on_paint)
        self.original_canvas.bind("<Button-1>", self._on_paint_start)
        self.original_canvas.bind("<Motion>", self._on_hover_original)
        self.original_canvas.bind("<Button-2>",
                                  lambda _: self._open_zoom("original"))
        self.original_canvas.bind("<Button-3>",
                                  lambda _: self._open_zoom("original"))

        self._original_brush_oval = self.original_canvas.create_oval(
            0, 0, 0, 0, outline="red", width=2,
        )

        self._refresh_display()

    # ── painting ────────────────────────────────────────────────────────────

    def _set_donor(self):
        self.donor_idx = self._donor_var.get()
        self._donor_label.config(
            text=f"Donor Mask {self.donor_idx + 1}  (paint here)",
        )
        self._refresh_display()

    def _set_brush(self):
        try:
            val = self._brush_var.get()
            self.brush_size = max(1, min(val, 500))
        except (tk.TclError, ValueError):
            pass

    def _digit_brush(self, digit):
        if not self._eraser_mode and not self._pencil_mode:
            return
        self._brush_var.set(max(1, digit))

    def _set_brush_mode(self):
        self._brush_mode = self._mode_var.get()

    def _toggle_eraser(self):
        self._eraser_mode = self._eraser_var.get()
        if self._eraser_mode:
            self._pencil_mode = False
            self._pencil_var.set(False)

    def _toggle_eraser_key(self):
        self._eraser_mode = not self._eraser_mode
        self._eraser_var.set(self._eraser_mode)
        if self._eraser_mode:
            self._pencil_mode = False
            self._pencil_var.set(False)

    def _toggle_pencil(self):
        self._pencil_mode = self._pencil_var.get()
        if self._pencil_mode:
            self._eraser_mode = False
            self._eraser_var.set(False)

    def _toggle_pencil_key(self):
        self._pencil_mode = not self._pencil_mode
        self._pencil_var.set(self._pencil_mode)
        if self._pencil_mode:
            self._eraser_mode = False
            self._eraser_var.set(False)

    def _on_hover_donor(self, event):
        r = self.brush_size
        self.donor_canvas.coords(
            self._donor_brush_oval,
            event.x - r, event.y - r, event.x + r, event.y + r,
        )

    def _on_hover_composed(self, event):
        r = self.brush_size
        self.mask_canvas.coords(
            self._brush_oval,
            event.x - r, event.y - r, event.x + r, event.y + r,
        )

    def _on_hover_preview(self, event):
        if not self._eraser_mode and not self._pencil_mode:
            self.preview_canvas.coords(self._preview_brush_oval, 0, 0, 0, 0)
            return
        r = self.brush_size
        self.preview_canvas.coords(
            self._preview_brush_oval,
            event.x - r, event.y - r, event.x + r, event.y + r,
        )

    def _on_hover_original(self, event):
        if not self._eraser_mode and not self._pencil_mode:
            self.original_canvas.coords(self._original_brush_oval, 0, 0, 0, 0)
            return
        r = self.brush_size
        self.original_canvas.coords(
            self._original_brush_oval,
            event.x - r, event.y - r, event.x + r, event.y + r,
        )

    def _save_undo_state(self):
        if self.composed is None:
            return
        if len(self._undo_stack) >= self.MAX_UNDO:
            self._undo_stack.pop(0)
        self._undo_stack.append(self.composed.copy())

    def _undo(self):
        if not self._undo_stack:
            return
        self.composed = self._undo_stack.pop()
        self._refresh_display()
        if self._zoom_overlay is not None:
            self._do_zoom_refresh()

    def _on_paint_start(self, event):
        is_imgview = event.widget in (self.original_canvas, self.preview_canvas)
        if is_imgview:
            if not self._eraser_mode and not self._pencil_mode:
                return
        elif not self._eraser_mode and not self._pencil_mode and self.donor_idx < 0:
            return
        self._save_undo_state()
        self._on_paint(event)

    def _on_zoom_paint_start(self, event):
        if self._zoom_type in ("original", "blend"):
            if not self._eraser_mode and not self._pencil_mode:
                return
        elif self._zoom_type in ("donor", "composed"):
            if not self._eraser_mode and not self._pencil_mode and self.donor_idx < 0:
                return
        else:
            return
        self._save_undo_state()
        self._on_zoom_paint(event)

    def _on_paint(self, event):
        is_imgview = event.widget in (self.original_canvas, self.preview_canvas)
        if is_imgview:
            if not self._eraser_mode and not self._pencil_mode:
                return
        elif not self._eraser_mode and not self._pencil_mode and self.donor_idx < 0:
            return

        r = self.brush_size

        if event.widget is self.donor_canvas:
            self.donor_canvas.coords(
                self._donor_brush_oval,
                event.x - r, event.y - r, event.x + r, event.y + r,
            )
        elif event.widget is self.mask_canvas:
            self.mask_canvas.coords(
                self._brush_oval,
                event.x - r, event.y - r, event.x + r, event.y + r,
            )
        elif event.widget is self.preview_canvas:
            self.preview_canvas.coords(
                self._preview_brush_oval,
                event.x - r, event.y - r, event.x + r, event.y + r,
            )
        elif event.widget is self.original_canvas:
            self.original_canvas.coords(
                self._original_brush_oval,
                event.x - r, event.y - r, event.x + r, event.y + r,
            )

        cx = max(0, min(event.x, self.cw - 1))
        cy = max(0, min(event.y, self.ch - 1))

        fx = int(cx * self.sx)
        fy = int(cy * self.sy)
        br = int(self.brush_size * max(self.sx, self.sy))

        h, w = self.composed.shape

        y1 = max(0, fy - br)
        y2 = min(h, fy + br + 1)
        x1 = max(0, fx - br)
        x2 = min(w, fx + br + 1)
        if y2 <= y1 or x2 <= x1:
            return

        yy, xx = np.ogrid[y1:y2, x1:x2]
        circle = ((xx - fx) ** 2 + (yy - fy) ** 2) <= br ** 2

        if self._eraser_mode:
            self.composed[y1:y2, x1:x2][circle] = 0
        elif self._pencil_mode:
            self.composed[y1:y2, x1:x2][circle] = 255
        else:
            donor = np.asarray(self.masks[self.donor_idx], dtype=np.uint8)
            patch = self.composed[y1:y2, x1:x2]
            dpatch = donor[y1:y2, x1:x2]
            if self._brush_mode == "Replace":
                patch[circle] = dpatch[circle]
            elif self._brush_mode == "Union":
                patch[circle] = np.maximum(patch[circle], dpatch[circle])
            elif self._brush_mode == "Intersect":
                patch[circle] = np.minimum(patch[circle], dpatch[circle])

        self._schedule_refresh()

    def _schedule_refresh(self):
        if not self._refresh_pending:
            self._refresh_pending = True
            self.window.after(30, self._do_refresh)

    def _do_refresh(self):
        self._refresh_pending = False
        self._refresh_display()

    def _refresh_display(self):
        # Donor mask canvas
        if self.donor_idx >= 0:
            donor_pil = self.masks[self.donor_idx]
            disp_donor = donor_pil.resize((self.cw, self.ch), Image.NEAREST)
            self._donor_ph = ImageTk.PhotoImage(mask_to_rgb(disp_donor))
        else:
            self._donor_ph = ImageTk.PhotoImage(
                Image.new("RGB", (self.cw, self.ch), (30, 30, 30)))
        self.donor_canvas.delete("img")
        self.donor_canvas.create_image(
            0, 0, anchor=tk.NW, image=self._donor_ph, tags="img",
        )
        self.donor_canvas.tag_raise(self._donor_brush_oval)

        # Composed mask canvas
        mask_pil = Image.fromarray(self.composed, mode="L")

        disp_mask = mask_pil.resize((self.cw, self.ch), Image.NEAREST)
        self._mask_ph = ImageTk.PhotoImage(mask_to_rgb(disp_mask))
        self.mask_canvas.delete("img")
        self.mask_canvas.create_image(
            0, 0, anchor=tk.NW, image=self._mask_ph, tags="img",
        )
        self.mask_canvas.tag_raise(self._brush_oval)

        # Blended preview canvas
        disp_alpha = mask_pil.resize((self.cw, self.ch), Image.BILINEAR)
        blend = blend_on_green(self._disp_orig, disp_alpha)
        self._blend_ph = ImageTk.PhotoImage(blend)
        self.preview_canvas.delete("img")
        self.preview_canvas.create_image(
            0, 0, anchor=tk.NW, image=self._blend_ph, tags="img",
        )
        self.preview_canvas.tag_raise(self._preview_brush_oval)

        # Original image canvas
        self._orig_ph = ImageTk.PhotoImage(self._disp_orig)
        self.original_canvas.delete("img")
        self.original_canvas.create_image(
            0, 0, anchor=tk.NW, image=self._orig_ph, tags="img",
        )
        self.original_canvas.tag_raise(self._original_brush_oval)

    # ── actions ─────────────────────────────────────────────────────────────

    def _reset(self):
        if self.base_idx is not None:
            self._undo_stack.clear()
            self.composed = np.asarray(
                self.masks[self.base_idx], dtype=np.uint8,
            ).copy()
            self._refresh_display()

    def _save(self):
        # date_tag = datetime.now().strftime("%Y%m%d")
        date_tag = "20260326" # Keep this manually updated
        out_str = str(self.image_path).replace(
            "/im/", f"/results/compose.dt{date_tag}/",
        )
        out_path = Path(out_str).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mask_pil = Image.fromarray(self.composed, mode="L")
        mask_pil.save(out_path)

        with open(self.app.log_file, "a") as f:
            f.write(f"{self.image_path},compose.dt{date_tag}\n")

        self.app._advance()

    # ── zoom overlay ──────────────────────────────────────────────────────

    _ZOOM_PANELS = ("donor", "composed", "blend", "original")

    def _zoom_get_hires(self, zoom_type: str) -> Image.Image:
        """Build a full-resolution RGB image for the requested panel."""
        if zoom_type == "donor" and self.donor_idx >= 0:
            return mask_to_rgb(self.masks[self.donor_idx])
        if zoom_type == "composed" and self.composed is not None:
            return mask_to_rgb(Image.fromarray(self.composed, mode="L"))
        if zoom_type == "blend" and self.composed is not None:
            mask_pil = Image.fromarray(self.composed, mode="L")
            return blend_on_green(self.original, mask_pil)
        if zoom_type == "original":
            return self.original.copy()
        return Image.new("RGB", (self.original.width, self.original.height),
                         (30, 30, 30))

    def _open_zoom(self, zoom_type: str):
        if self._zoom_overlay is not None:
            self._show_zoom_panel(zoom_type)
            return
        self._show_zoom_panel(zoom_type, create=True)

    def _show_zoom_panel(self, zoom_type: str, create: bool = False):
        # Only reset the zoom frame on a fresh open; navigating between
        # panels while already zoomed in keeps the same scale/center so you
        # can compare the same region across panels. All panels share the
        # original image's dimensions, so the fit scale is computed from
        # that directly (no need to build the panel's hires image here).
        reset_view = create
        self._zoom_type = zoom_type

        win_w = self.window.winfo_width()
        win_h = self.window.winfo_height()
        max_w = int(win_w * 0.92)
        max_h = int(win_h * 0.85)
        self._zoom_fit_scale = min(
            max_w / self.original.width, max_h / self.original.height, 1.0,
        )
        if reset_view:
            self._zoom_scale = self._zoom_fit_scale
            self._zoom_center_x = self.original.width / 2.0
            self._zoom_center_y = self.original.height / 2.0
        else:
            self._zoom_scale = max(
                self._zoom_fit_scale, min(1.0, self._zoom_scale),
            )

        self._render_zoom_panel(create=create)

        if create:
            self._zoom_overlay.bind("<B1-Motion>", self._on_zoom_paint)
            self._zoom_overlay.bind("<Button-1>", self._on_zoom_paint_start)
            self._zoom_overlay.bind("<Motion>", self._on_zoom_hover)
            self._zoom_overlay.bind(
                "<Button-2>", lambda _: self._close_zoom())
            self._zoom_overlay.bind(
                "<Button-3>", lambda _: self._close_zoom())
            self.window.bind("<Escape>", lambda _: self._close_zoom())
            self.window.bind("<Left>",
                             lambda _: self._zoom_navigate(-1))
            self.window.bind("<Right>",
                             lambda _: self._zoom_navigate(1))

    def _render_zoom_panel(self, create: bool, reset_brush: bool = True):
        win_w = self.window.winfo_width()
        win_h = self.window.winfo_height()

        hires = self._zoom_get_hires(self._zoom_type)

        crop_box, disp_w, disp_h, cx, cy = compute_zoom_geometry(
            hires.width, hires.height, win_w, win_h,
            self._zoom_scale, self._zoom_center_x, self._zoom_center_y,
        )
        self._zoom_crop = crop_box
        self._zoom_disp_w, self._zoom_disp_h = disp_w, disp_h
        self._zoom_center_x, self._zoom_center_y = cx, cy

        x0, y0, x1, y1 = crop_box
        crop = hires.crop(
            (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
        )
        self._zoom_sx = (x1 - x0) / disp_w
        self._zoom_sy = (y1 - y0) / disp_h

        resample = Image.NEAREST if self._zoom_scale >= 0.999 else Image.LANCZOS
        disp = crop.resize((disp_w, disp_h), resample)
        self._zoom_photo = ImageTk.PhotoImage(disp)
        cx_screen, cy_screen = win_w // 2, win_h // 2

        titles = {
            "donor": f"Donor Mask {self.donor_idx + 1}",
            "composed": "Composed Mask",
            "blend": "Blended Preview",
            "original": "Original Image",
        }
        paintable = self._zoom_type in ("donor", "composed") or (
            self._zoom_type in ("original", "blend")
            and (self._eraser_mode or self._pencil_mode)
        )
        hint = "  (right-click / Esc to close, paint with left-click)" \
            if paintable else "  (right-click / Esc to close)"
        pct = round(self._zoom_scale * 100)
        title = titles.get(self._zoom_type, "") + f"  [{pct}%]" + hint

        if create:
            self._zoom_overlay = tk.Canvas(
                self.window, highlightthickness=0,
            )
            self._zoom_overlay.place(x=0, y=0, relwidth=1, relheight=1)

            self._zoom_overlay.create_rectangle(
                0, 0, win_w, win_h, fill="black", stipple="gray50",
            )

            self._zoom_img_id = self._zoom_overlay.create_image(
                cx_screen, cy_screen, image=self._zoom_photo, anchor="center",
            )
            self._zoom_title_id = self._zoom_overlay.create_text(
                win_w // 2, 24, text=title,
                fill="white", font=("Helvetica", 14, "bold"),
                anchor="center",
            )

            self._zoom_brush_oval = self._zoom_overlay.create_oval(
                0, 0, 0, 0, outline="red", width=2,
            )
            self._zoom_overlay.tag_raise(self._zoom_brush_oval)
        else:
            self._zoom_overlay.itemconfig(
                self._zoom_img_id, image=self._zoom_photo)
            self._zoom_overlay.coords(self._zoom_img_id, cx_screen, cy_screen)
            self._zoom_overlay.itemconfig(
                self._zoom_title_id, text=title)
            self._zoom_overlay.coords(
                self._zoom_title_id, win_w // 2, 24)
            if reset_brush:
                self._zoom_overlay.coords(
                    self._zoom_brush_oval, 0, 0, 0, 0)

    def _adjust_zoom_panel(self, mult: float):
        if self._zoom_overlay is None:
            return
        anchor_x, anchor_y = zoom_anchor_from_pointer(
            self.window, self._zoom_crop, self._zoom_disp_w, self._zoom_disp_h,
        )
        new_scale = max(self._zoom_fit_scale, min(1.0, self._zoom_scale * mult))
        if new_scale == self._zoom_scale:
            return
        self._zoom_scale = new_scale
        self._zoom_center_x, self._zoom_center_y = anchor_x, anchor_y
        self._render_zoom_panel(create=False, reset_brush=False)

    def _zoom_navigate(self, direction: int):
        if self._zoom_overlay is None:
            return
        panels = self._ZOOM_PANELS
        try:
            idx = panels.index(self._zoom_type)
        except ValueError:
            return
        new_idx = idx + direction
        if 0 <= new_idx < len(panels):
            self._show_zoom_panel(panels[new_idx])

    def _on_zoom_hover(self, event):
        if self._zoom_type in ("original", "blend"):
            if not self._eraser_mode and not self._pencil_mode:
                self._zoom_overlay.coords(
                    self._zoom_brush_oval, 0, 0, 0, 0)
                return
        elif self._zoom_type not in ("donor", "composed"):
            return
        r = self.brush_size
        self._zoom_overlay.coords(
            self._zoom_brush_oval,
            event.x - r, event.y - r, event.x + r, event.y + r,
        )

    def _on_zoom_paint(self, event):
        if self._zoom_type in ("original", "blend"):
            if not self._eraser_mode and not self._pencil_mode:
                return
        elif self._zoom_type in ("donor", "composed"):
            if not self._eraser_mode and not self._pencil_mode and self.donor_idx < 0:
                return
        else:
            return

        r = self.brush_size
        self._zoom_overlay.coords(
            self._zoom_brush_oval,
            event.x - r, event.y - r, event.x + r, event.y + r,
        )

        win_w = self.window.winfo_width()
        win_h = self.window.winfo_height()
        img_left = (win_w - self._zoom_disp_w) / 2.0
        img_top = (win_h - self._zoom_disp_h) / 2.0

        rel_x = event.x - img_left
        rel_y = event.y - img_top
        if (rel_x < 0 or rel_y < 0
                or rel_x >= self._zoom_disp_w
                or rel_y >= self._zoom_disp_h):
            return

        crop_x0, crop_y0, _, _ = self._zoom_crop
        fx = int(crop_x0 + rel_x * self._zoom_sx)
        fy = int(crop_y0 + rel_y * self._zoom_sy)
        br = int(self.brush_size * max(self._zoom_sx, self._zoom_sy))

        h, w = self.composed.shape

        y1 = max(0, fy - br)
        y2 = min(h, fy + br + 1)
        x1 = max(0, fx - br)
        x2 = min(w, fx + br + 1)
        if y2 <= y1 or x2 <= x1:
            return

        yy, xx = np.ogrid[y1:y2, x1:x2]
        circle = ((xx - fx) ** 2 + (yy - fy) ** 2) <= br ** 2

        if self._eraser_mode:
            self.composed[y1:y2, x1:x2][circle] = 0
        elif self._pencil_mode:
            self.composed[y1:y2, x1:x2][circle] = 255
        else:
            donor = np.asarray(self.masks[self.donor_idx], dtype=np.uint8)
            patch = self.composed[y1:y2, x1:x2]
            dpatch = donor[y1:y2, x1:x2]
            if self._brush_mode == "Replace":
                patch[circle] = dpatch[circle]
            elif self._brush_mode == "Union":
                patch[circle] = np.maximum(patch[circle], dpatch[circle])
            elif self._brush_mode == "Intersect":
                patch[circle] = np.minimum(patch[circle], dpatch[circle])

        self._schedule_zoom_refresh()

    def _schedule_zoom_refresh(self):
        if not self._zoom_refresh_pending:
            self._zoom_refresh_pending = True
            self.window.after(30, self._do_zoom_refresh)

    def _do_zoom_refresh(self):
        self._zoom_refresh_pending = False
        if self._zoom_overlay is not None and self._zoom_type != "donor":
            self._render_zoom_panel(create=False, reset_brush=False)
            self._zoom_overlay.tag_raise(self._zoom_brush_oval)
        self._refresh_display()

    def _close_zoom(self):
        if self._zoom_overlay is not None:
            self._zoom_overlay.destroy()
            self._zoom_overlay = None
            self._zoom_photo = None
        for key in ("<Escape>", "<Left>", "<Right>"):
            try:
                self.window.unbind(key)
            except Exception:
                pass

    def _on_close(self):
        self._close_zoom()
        self.window.destroy()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mask Tagger – compare and compose segmentation masks.",
    )
    parser.add_argument("image_list",
                        help="Text file with one image path per line")
    parser.add_argument("--log", default=DEFAULT_LOG,
                        help=f"Log file path (default: {DEFAULT_LOG})")
    args = parser.parse_args()

    list_file = Path(args.image_list)
    if not list_file.exists():
        print(f"File not found: {list_file}", file=sys.stderr)
        return 1

    image_paths: List[Path] = []
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_paths.append(Path(line))

    if not image_paths:
        print("No image paths found in the list file.", file=sys.stderr)
        return 1

    errors = []
    for ip in image_paths:
        if not ip.exists():
            errors.append(f"Image not found: {ip}")
            continue
        for case, mp in zip(MASK_CASES, derive_mask_paths(ip)):
            if not mp.exists() and case not in OPTIONAL_CASES:
                print(f"Warning: mask not found (will use blank mask): {mp}",
                      file=sys.stderr)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    root = tk.Tk()
    try:
        root.state("zoomed")
    except Exception:
        try:
            root.attributes("-zoomed", True)
        except Exception:
            root.geometry(
                f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")

    if sys.platform == "darwin":
        try:
            from subprocess import Popen
            Popen(["osascript", "-e",
                   'tell application "System Events" to set frontmost '
                   'of the first process whose unix id is '
                   f'{__import__("os").getpid()} to true'])
        except Exception:
            pass
        root.lift()
        root.attributes("-topmost", True)
        root.after(100, lambda: root.attributes("-topmost", False))

    MaskTaggerApp(root, image_paths, args.log)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
