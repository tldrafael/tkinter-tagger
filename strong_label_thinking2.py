#!/usr/bin/env python3
"""
Strong labeler (multi-prediction + proportional layout)

- Shows a composite with: large original (2x2), GT mask + overlay, and
  for each prediction root: pred mask + overlay.
- Auto-scales tile size to fit all columns into your screen width.
- Displays per-pred metrics (Δ% and Δ#blobs) on each overlay tile.
- Big image shows summary: best Δ% and which prediction.
- Logs labels to labels.txt; skips images already labeled.
- Keyboard shortcuts: 1..6 correspond to the 6 buttons.

Usage:
    python strong_label_multi.py /path/to/image_list.txt
    # with multiple predictions (repeatable):
    python strong_label_multi.py /path/to/image_list.txt \
       --pred /results/2025_0910-tritonserver/ \
       --pred /results/2025_0912-onnx/
    # with names:
    python strong_label_multi.py images.txt \
       --pred triton=/results/2025_0910-tritonserver/ \
       --pred onnx=/results/2025_0912-onnx/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import argparse
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2


# ---------------------------- Configuration ----------------------------------

BASE_SIZE = 250          # preferred tile size; we auto-shrink to fit the screen
MIN_TILE_SIZE = 140      # don't go smaller than this unless screen is tiny
LEFT_RIGHT_MARGIN = 64   # pixels reserved for window borders/margins
LABELS_FILE = Path("labels.txt")

# Labels and key bindings (1..n)
LABELS = [
    "GT better",
    "Pred better",
    "Both good",
    "Both bad",
    "odd",
    "other-class",
]

# Path infixes (tweak for your directory layout)
IM_ROOT = "/im/"
GT_ROOT = "/gt/"

# Multiple prediction roots: list of (name, root_path).
# You can also pass them via CLI: --pred name=/path or --pred /path
PRED_ROOTS: List[Tuple[str, str]] = [
    ("dinov2", "/results/20250613-tuning-afterstage2-gaploss-addMoreGS+PRODCKPT/"),  # <- adjust / add more
    ("dinov3", "/results/20250821_dinosegv3-384px-interpBi_long768_gapmap/"),  # <- adjust / add more
]

# Binary threshold for masks (grayscale 0..255). Pixels >= THRESHOLD are treated as 1
THRESHOLD = 10

# When counting blobs (holes/gaps), ignore components larger than this ratio of the image area
MAX_BLOB_AREA_RATIO = 0.25


# --------------------------- Image utilities ---------------------------------


def imread_cv(path: str | Path) -> np.ndarray:
    """Read image with OpenCV (unchanged)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def to_pil(img: np.ndarray) -> Image.Image:
    """Convert a CV/NumPy image to PIL in RGB."""
    if img.ndim == 2:
        return Image.fromarray(img)
    if img.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if img.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    raise ValueError("Unsupported image shape for conversion to PIL.")


def binarize_mask(mask_img: np.ndarray, threshold: int = THRESHOLD) -> np.ndarray:
    """
    Convert a grayscale mask (0..255) to uint8 binary {0,255} using threshold.
    Pixels >= threshold -> 255; else 0.
    """
    if mask_img.ndim == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(mask_img, threshold, 255, cv2.THRESH_BINARY)
    return bin_img.astype(np.uint8)


def remove_small_regions(
    mask: np.ndarray, min_area: int, max_area: int | None = None, connectivity: int = 8
) -> Tuple[np.ndarray, int, List[int]]:
    """
    Keep connected components with area in [min_area, max_area].
    Returns (cleaned_mask, n_kept, kept_areas).
    """
    if max_area is None:
        h, w = mask.shape[:2]
        max_area = h * w + 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )

    cleaned = np.zeros_like(mask)
    kept_areas: List[int] = []
    n_kept = 0

    for label in range(1, num_labels):  # 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            cleaned[labels == label] = 255
            n_kept += 1
            kept_areas.append(area)

    return cleaned, n_kept, kept_areas


def count_gaps(mask_gray: np.ndarray) -> int:
    """
    Count number of 'gaps' (connected components) in the INVERTED binary mask.
    Matches the original behavior that inverted the thresholded GT/Pred before counting.
    """
    bin_mask = binarize_mask(mask_gray)              # foreground = 255
    inv_mask = cv2.bitwise_not(bin_mask)             # gaps/background = 255
    h, w = inv_mask.shape[:2]
    max_area = int(MAX_BLOB_AREA_RATIO * h * w)
    _, nblobs, _ = remove_small_regions(inv_mask, min_area=1, max_area=max_area)
    return nblobs


def overlay_on_green(rgb: Image.Image, mask_gray: Image.Image, label: str | None = None) -> Image.Image:
    """
    Return an overlay where mask==1 shows the original image; mask==0 shows green.
    Optionally draw a small header label at the top-left.
    """
    rgb_np = np.array(rgb, dtype=np.float32) / 255.0  # (H,W,3)
    if mask_gray.mode != "L":
        mask_gray = mask_gray.convert("L")
    m = (np.array(mask_gray, dtype=np.float32) >= THRESHOLD).astype(np.float32)[..., None]  # (H,W,1)

    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, None, :]  # (1,1,3)
    out = (1 - m) * green + m * rgb_np
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(out)

    if label:
        draw = ImageDraw.Draw(im)
        font = safe_font(size=18)
        draw.text((8, 6), label, fill=(255, 255, 255), font=font)
    return im


def diff_percentage(gt_mask_gray: np.ndarray, pred_mask_gray: np.ndarray) -> float:
    """
    Percentage of pixels where the binary masks differ.
    """
    gt = (binarize_mask(gt_mask_gray) // 255).astype(np.uint8)
    pr = (binarize_mask(pred_mask_gray) // 255).astype(np.uint8)
    diff = np.abs(gt.astype(np.int16) - pr.astype(np.int16))
    return 100.0 * float(diff.mean())


def safe_font(size: int = 40) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Try to load a reasonably common TrueType font; fall back to default.
    """
    candidates = [
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-C.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------- Path helpers -----------------------------------


def gt_path_from_image_path(p: str) -> str:
    return p.replace(IM_ROOT, GT_ROOT).replace(".jpg", ".png")


def pred_paths_from_gt_path(pgt: str, pred_roots: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    For a given GT path, return list of (name, pred_path) across all pred roots.
    Only include predictions that actually exist on disk.
    """
    out: List[Tuple[str, str]] = []
    for name, root in pred_roots:
        pred_path = pgt.replace(GT_ROOT, root)
        if Path(pred_path).exists():
            out.append((name, pred_path))
    return out


# ----------------------------- Layout math -----------------------------------


def compute_tile_size(screen_w: int, n_preds: int) -> int:
    """
    The row is: [Original (2 tiles)] + [GT mask + GT overlay (2 tiles)] + [2 tiles per pred].
    Total columns = 2 + 2 + 2*n_preds = 4 + 2*n_preds.
    Fit to screen width with margins; cap at BASE_SIZE and floor at MIN_TILE_SIZE.
    """
    cols = 4 + 2 * n_preds
    usable = max(320, screen_w - LEFT_RIGHT_MARGIN)  # guard small screens
    t = usable // cols
    return max(min(t, BASE_SIZE), MIN_TILE_SIZE)


# ---------------------------- Compositing ------------------------------------


def draw_badge(im: Image.Image, text: str, at_xy: Tuple[int, int], font_size: int = 18) -> None:
    """Draw small text at position with a dark shadow for legibility."""
    draw = ImageDraw.Draw(im)
    font = safe_font(size=font_size)
    x, y = at_xy
    # shadow
    draw.text((x+1, y+1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def build_composite(
    p_im: str,
    p_gt: str,
    preds: List[Tuple[str, str]],
    tile: int,
) -> Tuple[ImageTk.PhotoImage, str]:
    """
    Create the composite canvas and return (PhotoImage, nice_title).
    """
    # Load originals
    im_cv = imread_cv(p_im)
    gt_cv = imread_cv(p_gt)

    # Convert to PIL RGB
    big = to_pil(im_cv).resize((2 * tile, 2 * tile), Image.BILINEAR)
    im_small = to_pil(im_cv).resize((tile, tile), Image.BILINEAR)
    gt_gray = to_pil(gt_cv).convert("L").resize((tile, tile), Image.NEAREST)

    # GT visuals
    gt_overlay = overlay_on_green(im_small, gt_gray, label="GT overlay")

    # Prepare predictions
    pred_items: List[Tuple[str, Image.Image, Image.Image, float, int]] = []
    # also track best for summary
    best_name = None
    best_delta = None

    for name, pp in preds:
        pr_cv = imread_cv(pp)
        pr_gray = to_pil(pr_cv).convert("L").resize((tile, tile), Image.NEAREST)
        pr_overlay = overlay_on_green(im_small, pr_gray, label=name)

        d_pct = diff_percentage(gt_cv, pr_cv)
        d_blobs = abs(count_gaps(gt_cv) - count_gaps(pr_cv))

        # annotate metrics on the overlay
        draw_badge(pr_overlay, f"Δ{d_pct:.2f}% | Δ# {d_blobs}", (8, tile - 26))

        pred_items.append((name, pr_gray, pr_overlay, d_pct, d_blobs))

        if best_delta is None or d_pct < best_delta:
            best_delta = d_pct
            best_name = name

    # Compose canvas (2 rows × (4 + 2*len(preds)) columns)
    cols = 4 + 2 * len(preds)
    canvas = Image.new("RGB", (cols * tile, 2 * tile), "black")

    # Place big original (2×2)
    canvas.paste(big, (0, 0))

    # Place GT mask + overlay
    canvas.paste(gt_gray, (2 * tile, 0))
    canvas.paste(gt_overlay, (3 * tile, 0))
    draw_badge(canvas, "GT mask", (2 * tile + 8, 8))

    # Place predictions (mask top row, overlay bottom row)
    x = 4 * tile
    for name, pr_gray, pr_overlay, _d, _b in pred_items:
        canvas.paste(pr_gray, (x, 0))
        canvas.paste(pr_overlay, (x, tile))
        draw_badge(canvas, f"{name} mask", (x + 8, 8))
        x += 2 * tile

    # Summary on the big image
    if best_name is not None and best_delta is not None:
        draw = ImageDraw.Draw(canvas)
        font = safe_font(size=22)
        txt = f"Best Δ: {best_delta:.2f}%  ({best_name}) | preds: {len(preds)}"
        draw.text((8, 2 * tile - 28), txt, fill=(255, 0, 255), font=font)

    title = Path(p_im).name.split("+")[0]
    return ImageTk.PhotoImage(canvas), title


# ------------------------------- UI ------------------------------------------


class App:
    def __init__(self, master: tk.Tk, imgpaths: List[str], pred_roots: List[Tuple[str, str]]) -> None:
        self.master = master
        self.pred_roots = pred_roots
        self.imgpaths = imgpaths
        self.gtpaths = [gt_path_from_image_path(p) for p in imgpaths]
        self.idx = 0

        # Lazily create labels file if missing; build "already done" set
        self.preworked = self._read_preworked()

        # Compute tile size from screen width and number of preds
        self.master.update_idletasks()
        screen_w = self.master.winfo_screenwidth()
        self.tile = compute_tile_size(screen_w, n_preds=len(self.pred_roots))

        # UI
        self.panel = tk.Label(master, compound="bottom", font=("Helvetica", 16))
        self.panel.grid(row=0, column=1, columnspan=17, padx=6, pady=6)

        self._build_buttons()
        self._bind_keys()
        self._skip_preworked()
        self._render()

    # --- controls

    def _build_buttons(self) -> None:
        for i, text in enumerate(LABELS):
            btn = tk.Button(self.master, text=text,
                            command=lambda t=text: self._on_label(t),
                            width=14, height=2)
            btn.grid(row=1, column=i + 2, padx=2, pady=4)

    def _bind_keys(self) -> None:
        for i, text in enumerate(LABELS, start=1):
            self.master.bind_all(str(i), lambda e, t=text: self._on_label(t))
        self.master.bind_all("<Escape>", lambda e: self.master.quit())

    # --- labeling flow

    def _read_preworked(self) -> set[str]:
        if LABELS_FILE.exists():
            lines = LABELS_FILE.read_text().splitlines()
            return {line.split(",")[0] for line in lines if line.strip()}
        return set()

    def _append_label(self, label: str) -> None:
        LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LABELS_FILE.open("a") as f:
            f.write(f"{self.imgpaths[self.idx]},{label}\n")

    def _skip_preworked(self) -> None:
        while self.idx < len(self.imgpaths) and self.imgpaths[self.idx] in self.preworked:
            self.idx += 1
        if self.idx >= len(self.imgpaths):
            self.master.quit()

    def _render(self) -> None:
        if self.idx >= len(self.imgpaths):
            return

        # Build list of per-image available predictions
        gt_path = self.gtpaths[self.idx]
        preds = pred_paths_from_gt_path(gt_path, self.pred_roots)

        photo, title = build_composite(
            self.imgpaths[self.idx],
            gt_path,
            preds,
            self.tile,
        )
        self.panel.img = photo  # keep a reference
        self.panel.config(image=photo, text=title)

    def _on_label(self, label: str) -> None:
        self._append_label(label)
        self.idx += 1
        self._skip_preworked()
        self._render()


# ------------------------------ Entrypoint -----------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("image_list", help="Path to text file with image paths")
    p.add_argument(
        "--pred",
        action="append",
        default=[],
        help="Prediction root as PATH or NAME=PATH; can be repeated",
    )
    return p.parse_args()


def normalize_pred_roots(cli_preds: List[str]) -> List[Tuple[str, str]]:
    """
    Turn --pred entries into list[(name, path)].
    If only PATH is given, name = basename(path.strip('/')).
    """
    if not cli_preds:
        return PRED_ROOTS
    out: List[Tuple[str, str]] = []
    for item in cli_preds:
        if "=" in item:
            name, path = item.split("=", 1)
            out.append((name.strip(), path.strip()))
        else:
            path = item.strip()
            name = Path(path.rstrip("/")).name or "pred"
            out.append((name, path))
    return out


def read_image_list(list_path: str | Path) -> List[str]:
    with open(list_path, "r") as f:
        items = [line.strip() for line in f if line.strip()]
    return items


def launch(imgpaths: List[str], pred_roots: List[Tuple[str, str]]) -> None:
    root = tk.Tk()
    root.title("Strong Labeler (Multi-Pred)")
    App(root, imgpaths, pred_roots)
    root.mainloop()


if __name__ == "__main__":
    args = parse_args()
    image_list = read_image_list(args.image_list)
    pred_roots = normalize_pred_roots(args.pred)
    launch(image_list, pred_roots)

