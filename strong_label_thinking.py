#!/usr/bin/env python3
"""
Strong labeler (tidy version)

- Shows a composite with: large original + GT, GT-overlay, Pred, Pred-overlay.
- Displays Diff-% and Diff-#blobs on the canvas.
- Logs labels to labels.txt; skips images already labeled.
- Keyboard shortcuts: 1..6 correspond to the 6 buttons.

Usage:
    python strong_label_tidy.py /path/to/image_list.txt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2


# ---------------------------- Configuration ----------------------------------

SIZE = 480  # base size for small tiles; original is shown at 2*SIZE
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
PRED_ROOT = "/results/20250613-tuning-afterstage2-gaploss-addMoreGS+PRODCKPT/"  # <- adjust as needed

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


def overlay_on_green(rgb: Image.Image, mask_gray: Image.Image) -> Image.Image:
    """
    Return an overlay where mask==1 shows the original image; mask==0 shows green.
    """
    rgb_np = np.array(rgb, dtype=np.float32) / 255.0  # (H,W,3)
    if mask_gray.mode != "L":
        mask_gray = mask_gray.convert("L")
    m = (np.array(mask_gray, dtype=np.float32) >= THRESHOLD).astype(np.float32)[..., None]  # (H,W,1)

    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, None, :]  # (1,1,3)
    out = (1 - m) * green + m * rgb_np
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


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


def pred_path_from_gt_path(pgt: str) -> str:
    return pgt.replace(GT_ROOT, PRED_ROOT)


# ---------------------------- Compositing ------------------------------------


def build_composite(p_im: str, p_gt: str, p_pred: str) -> Tuple[ImageTk.PhotoImage, str]:
    """
    Create the composite canvas and return (PhotoImage, nice_title).
    """
    # Load originals
    im_cv = imread_cv(p_im)
    gt_cv = imread_cv(p_gt)
    pr_cv = imread_cv(p_pred)

    # Convert to PIL RGB
    im_rgb = to_pil(im_cv).resize((2 * SIZE, 2 * SIZE), Image.BILINEAR)
    gt_gray = to_pil(gt_cv).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    pr_gray = to_pil(pr_cv).convert("L").resize((SIZE, SIZE), Image.NEAREST)

    # Small RGB for overlays
    im_small = to_pil(im_cv).resize((SIZE, SIZE), Image.BILINEAR)
    gt_overlay = overlay_on_green(im_small, gt_gray)
    pr_overlay = overlay_on_green(im_small, pr_gray)

    # Metrics
    diff_pct = diff_percentage(gt_cv, pr_cv)
    gt_blobs = count_gaps(gt_cv)
    pr_blobs = count_gaps(pr_cv)
    diff_blobs = abs(gt_blobs - pr_blobs)
    metrics = f"Diff-mag: {diff_pct:.2f}%\nDiff-blobs: {diff_blobs}"

    # Draw metrics on large original
    draw = ImageDraw.Draw(im_rgb)
    font = safe_font(size=40)
    draw.text((10, 2 * SIZE - 70), metrics, fill=(255, 0, 255), font=font)

    # Assemble 4x2 grid canvas
    canvas = Image.new("RGB", (4 * SIZE, 2 * SIZE))
    canvas.paste(im_rgb, (0, 0))                # big original (2x2)
    canvas.paste(gt_gray, (2 * SIZE, 0))
    canvas.paste(gt_overlay, (3 * SIZE, 0))
    canvas.paste(pr_gray, (2 * SIZE, SIZE))
    canvas.paste(pr_overlay, (3 * SIZE, SIZE))

    title = Path(p_im).name.split("+")[0]
    return ImageTk.PhotoImage(canvas), title


# ------------------------------- UI ------------------------------------------


class App:
    def __init__(self, master: tk.Tk, imgpaths: List[str]) -> None:
        self.master = master
        self.imgpaths = imgpaths
        self.gtpaths = [gt_path_from_image_path(p) for p in imgpaths]
        self.predpaths = [pred_path_from_gt_path(pgt) for pgt in self.gtpaths]
        self.idx = 0

        # Lazily create labels file if missing; build "already done" set
        self.preworked = self._read_preworked()

        # UI
        self.panel = tk.Label(master, compound="bottom", font=("Helvetica", 20))
        self.panel.grid(row=0, column=1, columnspan=17)

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
            btn.grid(row=1, column=i + 2)

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
        photo, title = build_composite(
            self.imgpaths[self.idx], self.gtpaths[self.idx], self.predpaths[self.idx]
        )
        self.panel.img = photo  # keep a reference
        self.panel.config(image=photo, text=title)

    def _on_label(self, label: str) -> None:
        self._append_label(label)
        self.idx += 1
        self._skip_preworked()
        self._render()


# ------------------------------ Entrypoint -----------------------------------


def launch(imgpaths: List[str]) -> None:
    root = tk.Tk()
    root.title("Strong Labeler")
    App(root, imgpaths)
    root.mainloop()


def read_image_list(list_path: str | Path) -> List[str]:
    with open(list_path, "r") as f:
        items = [line.strip() for line in f if line.strip()]
    return items


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python strong_label_tidy.py /path/to/image_list.txt")
        sys.exit(1)

    image_list = read_image_list(sys.argv[1])
    launch(image_list)

