#!/usr/bin/env python3
"""
Strong Label – multi-case visual tagger for segmentation masks.

What it does
------------
- Loads a list of original image paths from a text file (one per line).
- For each "case" (prediction run name), finds the corresponding mask image by
  mapping the path from the original:
    * case == "gt":   /im/xxx.jpg -> /gt/xxx.png
    * otherwise:      /im/xxx.jpg -> /results/{case}/xxx.png   (via the GT path)
- Displays a responsive grid that always fits on screen:
    * 1 tile: original
    * 2 tiles per case: mask, overlay (base + green background where mask=0)
- Lets you label the current original image (keys 1..6 or buttons).
- Appends labels to a CSV-ish register file: "image_path,label"

Dynamic layout
--------------
The app detects your screen width/height, reserves space for the button bar,
then searches 1..N columns to find the largest square tile size so that all
tiles (original + 2*len(cases)) fit. Everything resizes automatically when
you relaunch with a different number of cases.

Metrics
-------
If the "gt" case is present and its file exists:
- For each other case, the overlay tile is annotated with:
    Δmag vs GT   : percent pixel difference across masks
    Δblobs vs GT : difference in small "gap" components count (see threshold below)
- The gt overlay tile shows:  blobs: N
(If "gt" is missing, metrics are omitted.)

Usage
-----
    python strong_label.py paths.txt --cases gt,runA,runB --register labels.txt

If --cases is omitted:
- If RUN_NAME is set in the environment: uses ["gt", RUN_NAME]
- Otherwise: ["gt"]

Note: Adjust the path mapping functions here if your dataset layout differs.

"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tkinter as tk
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk


# ---------------------------- Configuration ----------------------------------

REGISTER_FILE = "labels.txt"

# Fallback labels (unchanged from earlier tidy version)
LABELS: Tuple[str, ...] = (
    "GT better",
    "Predv2 better",
    "Predv3 better",
    "all bad",
    "odd",
)

# Fonts (portable fallbacks)
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-C.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


# ------------------------------ Image helpers --------------------------------

def pil_open(path: Path) -> Image.Image:
    img = Image.open(path)
    return img


def pil_open_resize(path: Path, size: int) -> Image.Image:
    img = pil_open(path)
    return img.resize((size, size), Image.BILINEAR)


def to_gray_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to grayscale ndarray (H, W) uint8 in [0,255]."""
    return np.asarray(img.convert("L"), dtype=np.uint8)


def pil_from_gray(arr: np.ndarray, size: int) -> Image.Image:
    """Given a uint8 grayscale array, resize square and return RGB PIL image."""
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def overlay_green_background(base_img: Image.Image, mask_img: Image.Image) -> Image.Image:
    """
    Where mask==1 (white), show base image; where mask==0, show green.
    Uses grayscale mask normalized to [0,1].
    """
    base = np.asarray(base_img.convert("RGB"), dtype=np.float32) / 255.0
    mask = to_gray_array(mask_img).astype(np.float32) / 255.0  # (H,W)
    if mask.ndim == 2:
        mask = mask[..., None]
    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    out = (1 - mask) * green + base * mask
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)


def load_font(size: int) -> ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate(img: Image.Image, lines: List[str], pos: Tuple[int, int], font_size: int) -> Image.Image:
    """Draw multi-line text onto a copy of the image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = load_font(font_size)
    y = pos[1]
    for line in lines:
        draw.text((pos[0], y), line, fill=(255, 0, 255), font=font)
        # simple line spacing proportional to font size
        y += int(font_size * 1.2)
    return out


# --------------------------- Mask & metrics helpers ---------------------------

def cv2_imread_any(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def remove_small_regions(
    mask: np.ndarray,
    min_area: int,
    max_area: Optional[int] = None,
    connectivity: int = 8,
) -> Tuple[np.ndarray, int, List[int]]:
    """
    Keep connected components with area in [min_area, max_area].
    Returns (cleaned_mask, num_blobs, list_of_areas).
    """
    if max_area is None:
        max_area = mask.shape[0] * mask.shape[1] + 1

    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=connectivity
    )

    cleaned = np.zeros_like(mask_u8)
    nblobs = 0
    areas: List[int] = []

    for label in range(1, num_labels):  # skip background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            cleaned[labels == label] = 255
            nblobs += 1
            areas.append(area)

    return cleaned, nblobs, areas


def count_gaps_components(mask_img_cv: np.ndarray, threshold: int = 10) -> int:
    """
    From a (possibly color) mask image (OpenCV BGR or GRAY), compute number of 'gap'
    components as in the original script:
      - pixels < threshold -> 255 (foreground), else 0
      - limit component size to <= 25% of image to avoid counting the whole image
    """
    if mask_img_cv.ndim == 3:
        mask_img_cv = cv2.cvtColor(mask_img_cv, cv2.COLOR_BGR2GRAY)

    gt_thrs_inv = np.where(mask_img_cv < threshold, 255, 0).astype(np.uint8)
    h, w = gt_thrs_inv.shape[:2]
    max_area = int(0.25 * h * w)
    _, nblobs, _ = remove_small_regions(gt_thrs_inv, min_area=1, max_area=max_area)
    return nblobs


def mask_diff_percent(gt_mask: Image.Image, pred_mask: Image.Image) -> float:
    """
    Percentage of pixels where the two binary masks differ.
    """
    gt = (to_gray_array(gt_mask).astype(np.float32) / 255.0)[..., None]
    pr = (to_gray_array(pred_mask).astype(np.float32) / 255.0)[..., None]
    diff = np.abs(gt - pr).sum()
    return 100.0 * diff / (gt.shape[0] * gt.shape[1] * gt.shape[2])


# ------------------------------- Path helpers --------------------------------

def gt_path_from_image_path(p: Path) -> Path:
    """Mapping: '/im/' -> '/gt/', '.jpg' -> '.png' """
    s = str(p)
    s = s.replace("/im/", "/gt/").replace(".jpg", ".png")
    return Path(s)


def mask_path_from_image_path(img_path: Path, case: str) -> Path:
    """
    For a given "case", derive mask path:
      - 'gt' -> GT mapping
      - else -> same as GT but replace '/gt/' with f'/results/{case}/'
    """
    pgt = gt_path_from_image_path(img_path)
    if case.lower() == "gt":
        return pgt
    return Path(str(pgt).replace("/gt/", f"/results/{case}/"))


# --------------------------- Layout / tiling helpers --------------------------

def choose_best_grid(
    n_tiles: int,
    screen_w: int,
    screen_h: int,
    reserved_h: int,
    margin: int = 16,
    gap: int = 8,
    max_cols: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Search the number of columns (1..max) that yields the largest square tile size
    given available space.

    Returns (cols, rows, tile_size).
    """
    avail_w = max(100, screen_w - 2 * margin)
    avail_h = max(100, screen_h - reserved_h - 2 * margin)
    if max_cols is None:
        max_cols = n_tiles

    best_cols, best_rows, best_tile = 1, n_tiles, 0
    for cols in range(1, max_cols + 1):
        rows = math.ceil(n_tiles / cols)
        tile_w = (avail_w - gap * (cols - 1)) // cols
        tile_h = (avail_h - gap * (rows - 1)) // rows
        tile = int(min(tile_w, tile_h))
        if tile > best_tile:
            best_cols, best_rows, best_tile = cols, rows, tile

    return best_cols, best_rows, best_tile


# -------------------------------- UI Application ------------------------------

class App:
    def __init__(self, root: tk.Tk, img_paths: Sequence[Path], cases: Sequence[str], register: Path):
        self.root = root
        self.root.title("Strong Label – multi-case")

        # State
        self.img_paths: List[Path] = [Path(p) for p in img_paths]
        self.cases: List[str] = list(cases)
        self.register_path = Path(register)
        self.preworked = self._read_register()
        self.index = 0

        # UI containers
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

        # Grid resizing config
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Buttons
        for i, lbl in enumerate(LABELS, start=1):
            action = self._make_action(lbl)
            b = tk.Button(self.btn_frame, text=f"{i}. {lbl}", command=action, width=14, height=2)
            b.grid(row=0, column=i - 1, padx=3, pady=3)
            self.root.bind(f"<Key-{i}>", lambda e, fn=action: fn())
        self.root.bind("<Key-q>", lambda e: self.root.quit())

        # Pre-calc button height for layout sizing
        self.root.update_idletasks()
        self.reserved_h = self.btn_frame.winfo_reqheight() + 24

        # Prepare first view
        self._advance_past_preworked()
        self._show_current_or_finish()

    # ---------- Register I/O ----------

    def _read_register(self) -> set[str]:
        if self.register_path.exists():
            text = self.register_path.read_text(encoding="utf-8", errors="ignore")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            done = {ln.split(",", 1)[0] for ln in lines if "," in ln}
            return done
        return set()

    def _append_label(self, label: str) -> None:
        self.register_path.parent.mkdir(parents=True, exist_ok=True)
        with self.register_path.open("a", encoding="utf-8") as f:
            f.write(f"{self.img_paths[self.index]},{label}\n")

    # ---------- Navigation / UI ----------

    def _make_action(self, label: str):
        def _action():
            self._append_label(label)
            self.index += 1
            self._advance_past_preworked()
            self._show_current_or_finish()
        return _action

    def _advance_past_preworked(self) -> None:
        while self.index < len(self.img_paths) and str(self.img_paths[self.index]) in self.preworked:
            self.index += 1

    def _show_current_or_finish(self) -> None:
        if self.index >= len(self.img_paths):
            for w in self.grid_frame.winfo_children():
                w.destroy()
            done_label = tk.Label(self.grid_frame, text="All images processed.", font=("Helvetica", 20))
            done_label.pack(padx=20, pady=20)
            self.root.after(600, self.root.quit)
            return
        self._show_current()

    def _show_current(self) -> None:
        # Compute optimal grid for all tiles: 1 (original) + 2 per case
        total_tiles = 2 + 2 * len(self.cases)

        # Screen size
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        sw, sh = 3200,1050
        cols, rows, tile = choose_best_grid(total_tiles, sw, sh, reserved_h=self.reserved_h)

        # Clear previous
        for w in self.grid_frame.winfo_children():
            w.destroy()

        # Load current base
        base_path = self.img_paths[self.index]
        title = f"Strong Label – {self.index + 1}/{len(self.img_paths)} – {Path(base_path).name}"
        self.root.title(title)

        try:
            base_img = pil_open_resize(base_path, tile)
        except Exception as ex:
            self._render_error_tile(str(ex), cols, rows, tile)
            return

        # Pre-load GT mask (if present among cases)
        gt_case_present = any(c.lower() == "gt" for c in self.cases)
        gt_mask_pil: Optional[Image.Image] = None
        gt_mask_cv: Optional[np.ndarray] = None

        if gt_case_present:
            p_gt = mask_path_from_image_path(base_path, "gt")
            if p_gt.exists():
                try:
                    gt_mask_pil = pil_open_resize(p_gt, tile)
                    gt_mask_cv = cv2_imread_any(p_gt)
                except Exception:
                    gt_mask_pil = None
                    gt_mask_cv = None
            else:
                gt_mask_pil = None
                gt_mask_cv = None

        # Build tile list: ("kind", case_name|None, PIL.Image)
        # kind in {"original", "mask", "overlay"}
        tiles: List[Tuple[str, Optional[str], Image.Image]] = []

        # Original tile
        orig_annot = annotate(base_img, [Path(base_path).name], pos=(10, 10), font_size=max(14, tile // 24))
        tiles.append(("original", None, orig_annot))

        # Case tiles
        for case in self.cases:
            mask_path = mask_path_from_image_path(base_path, case)
            mask_pil: Optional[Image.Image] = None
            overlay_pil: Optional[Image.Image] = None
            # Try to load mask
            if mask_path.exists():
                try:
                    mask_pil = pil_open_resize(mask_path, tile)
                except Exception:
                    mask_pil = None

            # Create tiles (mask & overlay), with annotations
            font_sz = max(14, tile // 28)

            # Mask tile (grayscale visualization)
            if mask_pil is None:
                mask_tile = self._placeholder_tile(tile, f"{case}\n(mask missing)", font_sz)
            else:
                # Ensure grayscale view of mask
                mask_arr = to_gray_array(mask_pil)
                mask_tile = pil_from_gray(mask_arr, tile)
                mask_tile = annotate(mask_tile, [f"{case} – mask"], pos=(10, 10), font_size=font_sz)
            tiles.append(("mask", case, mask_tile))

            # Overlay tile (with metrics)
            if mask_pil is None:
                overlay_tile = self._placeholder_tile(tile, f"{case}\n(overlay missing)", font_sz)
            else:
                overlay_tile = overlay_green_background(base_img, mask_pil)

                lines = [f"{case} – overlay"]
                if gt_case_present and gt_mask_pil is not None:
                    if case.lower() == "gt":
                        # count blobs for GT only
                        try:
                            n_gt = count_gaps_components(gt_mask_cv) if gt_mask_cv is not None else None
                        except Exception:
                            n_gt = None
                        if n_gt is not None:
                            lines.append(f"blobs: {n_gt}")
                    else:
                        try:
                            dmag = mask_diff_percent(gt_mask_pil, mask_pil)
                        except Exception:
                            dmag = None
                        # blob diff
                        try:
                            pred_cv = cv2_imread_any(mask_path)
                            n_gt = count_gaps_components(gt_mask_cv) if gt_mask_cv is not None else None
                            n_pr = count_gaps_components(pred_cv)
                            dblob = (abs(n_gt - n_pr) if (n_gt is not None) else None)
                        except Exception:
                            dblob = None
                        if dmag is not None:
                            lines.append(f"Δmag vs GT: {dmag:.2f}%")
                        if dblob is not None:
                            lines.append(f"Δblobs vs GT: {dblob}")
                overlay_tile = annotate(overlay_tile, lines, pos=(10, 10), font_size=font_sz)

            tiles.append(("overlay", case, overlay_tile))


        # Arrange tiles order (my code)
        im = tiles[0][2]
        w, h = im.size
        empty = Image.new("RGB", (w, h), (0,0,0))
        tiles = [("empty", None, empty)] + tiles[:]
        tiles = tiles[1::2] + tiles[::2]

        # Render tiles into a Tk grid (row-major)
        photos: List[ImageTk.PhotoImage] = []
        for idx, (_, _, img) in enumerate(tiles):
            r = idx // cols
            c = idx % cols
            ph = ImageTk.PhotoImage(img)
            photos.append(ph)
            lbl = tk.Label(self.grid_frame, image=ph)
            lbl.image = ph  # keep ref
            lbl.grid(row=r, column=c, padx=4, pady=4)

        # Keep references to prevent GC
        self._last_photos = photos
        self.root.update_idletasks()

    # ----- helpers for rendering -----

    def _placeholder_tile(self, size: int, text: str, font_sz: int) -> Image.Image:
        img = Image.new("RGB", (size, size), (40, 40, 40))
        draw = ImageDraw.Draw(img)
        font = load_font(font_sz)
        # Centered multi-line
        lines = text.split("\n")
        total_h = int(len(lines) * font_sz * 1.2)
        y = (size - total_h) // 2
        for line in lines:
            w, h = draw.textsize(line, font=font)
            x = (size - w) // 2
            draw.text((x, y), line, fill=(255, 100, 100), font=font)
            y += int(font_sz * 1.2)
        return img

    def _render_error_tile(self, err: str, cols: int, rows: int, tile: int) -> None:
        for w in self.grid_frame.winfo_children():
            w.destroy()
        img = self._placeholder_tile(tile, f"Error\n{err}", max(14, tile // 28))
        ph = ImageTk.PhotoImage(img)
        lbl = tk.Label(self.grid_frame, image=ph)
        lbl.image = ph
        lbl.grid(row=0, column=0, padx=4, pady=4)


# ---------------------------------- CLI --------------------------------------

def read_paths_file(file: Path) -> List[Path]:
    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [Path(ln.strip()) for ln in lines if ln.strip()]


def parse_cases(cases_arg: Optional[str]) -> List[str]:
    """
    Resolve the list of cases:
      - If --cases provided: split by comma
      - Else if RUN_NAME env set: ["gt", RUN_NAME]
      - Else: ["gt"]
    'gt' is treated as a normal case and may appear anywhere in the list.
    """
    if cases_arg and cases_arg.strip():
        cases = [c.strip() for c in cases_arg.split(",") if c.strip()]
        return cases
    env_run = os.getenv("RUN_NAME")
    if env_run:
        return ["gt", env_run]
    return ["gt"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strong Label (multi-case GT vs predictions).")
    parser.add_argument(
        "paths_file",
        help="Text file with one original image path per line.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help='Comma-separated list of cases. Example: "gt,2025_0910-tritonserver,runB". '
             'If omitted, uses ["gt", $RUN_NAME] if RUN_NAME is set, else ["gt"].',
    )
    parser.add_argument(
        "--register",
        default=REGISTER_FILE,
        help=f"Output CSV for labels (default: {REGISTER_FILE})",
    )
    args = parser.parse_args(argv)

    img_paths = read_paths_file(Path(args.paths_file))
    if not img_paths:
        print("No image paths found in the provided file.")
        return 1

    cases = parse_cases(args.cases)
    if not cases:
        print("No cases provided or resolved. Use --cases or set RUN_NAME.")
        return 1

    root = tk.Tk()

    # Try to maximize the window so the auto-sizing gets full screen
    try:
        # Windows
        root.state("zoomed")
    except Exception:
        try:
            # X11 (some WMs)
            root.attributes("-zoomed", True)
        except Exception:
            pass  # best effort

    App(root, img_paths, cases, Path(args.register))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

