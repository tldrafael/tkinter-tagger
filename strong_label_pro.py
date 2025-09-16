#!/usr/bin/env python3
"""
Strong Label – quick visual tagger for GT vs Pred segmentation masks.

- Shows a 2x2+panel composite: original (annotated), GT, GT-overlay, Pred, Pred-overlay.
- Lets you label the current image with one of a few categories (1..6 keys or buttons).
- Appends labels to `labels.txt` (image_path,label).

This is a tidy, clearer rewrite of the original script with the same behavior.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tkinter as tk
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk


# ---------------------------- Configuration ----------------------------------

TILE_SIZE = 450
REGISTER_FILE = "labels.txt"

# The prediction path is derived from the GT path by replacing '/gt/' with
# f'/results/{RUN_NAME}/' (kept from original script, now configurable).
RUN_NAME = os.getenv("RUN_NAME", "20250613-tuning-afterstage2-gaploss-addMoreGS+PRODCKPT")

LABELS: Tuple[str, ...] = (
    "GT better",
    "Pred better",
    "Both good",
    "Both bad",
    "odd",
    "other-class",
)

# Portable font fallback (we try a few common locations; otherwise PIL default).
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-C.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


# ------------------------------ Image helpers --------------------------------

def pil_open_resize(path: Path, size: int) -> Image.Image:
    """Open an image with PIL and resize to (size, size)."""
    img = Image.open(path)
    return img.resize((size, size), Image.BILINEAR)


def to_gray_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to grayscale ndarray (H, W) in [0, 255] uint8."""
    return np.asarray(img.convert("L"), dtype=np.uint8)


def overlay_green_background(base_img: Image.Image, mask_img: Image.Image) -> Image.Image:
    """
    Reproduce original overlay logic:
    - Where mask == 1 (white), show base image.
    - Where mask == 0 (black), paint green background.
    """
    base = np.asarray(base_img.convert("RGB"), dtype=np.float32) / 255.0
    mask = to_gray_array(mask_img).astype(np.float32) / 255.0  # (H, W)
    mask = mask[..., None]  # (H, W, 1)

    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # RGB
    out = (1 - mask) * green + base * mask
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)


def load_font(size: int = 40) -> ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate(img: Image.Image, text: str, pos: Tuple[int, int] = (10, 10)) -> Image.Image:
    """Draw multi-line text onto a copy of the image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.text(pos, text, fill=(255, 0, 255), font=load_font(40))
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
    max_area: int | None = None,
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
      - threshold < threshold -> 255, else 0
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
    Percentage of pixels where the two binary masks differ (original behavior).
    """
    gt = (to_gray_array(gt_mask).astype(np.float32) / 255.0)[..., None]
    pr = (to_gray_array(pred_mask).astype(np.float32) / 255.0)[..., None]
    diff = np.abs(gt - pr).sum()
    return 100.0 * diff / (gt.shape[0] * gt.shape[1] * gt.shape[2])


# ------------------------------- Path helpers --------------------------------

def gt_path_from_image_path(p: Path) -> Path:
    """
    Original mapping: '/im/' -> '/gt/', '.jpg' -> '.png'
    """
    s = str(p)
    s = s.replace("/im/", "/gt/").replace(".jpg", ".png")
    return Path(s)


def pred_path_from_gt_path(gt_path: Path, run_name: str = RUN_NAME) -> Path:
    """
    Original mapping: '/gt/' -> f'/results/{run_name}/'
    """
    return Path(str(gt_path).replace("/gt/", f"/results/{run_name}/"))


# ------------------------------ UI Application --------------------------------

class App:
    def __init__(self, root: tk.Tk, img_paths: Sequence[Path], register: Path):
        self.root = root
        self.root.title("Strong Label")

        self.img_paths: List[Path] = [Path(p) for p in img_paths]
        self.gt_paths: List[Path] = [gt_path_from_image_path(p) for p in self.img_paths]
        self.pred_paths: List[Path] = [pred_path_from_gt_path(p) for p in self.gt_paths]

        self.register_path = Path(register)
        self.preworked = self._read_register()
        self.index = 0

        # Layout
        self.panel = tk.Label(self.root)
        self.panel.grid(row=0, column=0, columnspan=len(LABELS) + 2, padx=8, pady=8)

        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=1, column=0, padx=8, pady=(0, 8))

        for i, lbl in enumerate(LABELS, start=1):
            action = self._make_action(lbl)
            b = tk.Button(btn_frame, text=f"{i}. {lbl}", command=action, width=14, height=2)
            b.grid(row=0, column=i - 1, padx=3, pady=3)
            # Bind numeric keys "<Key-1>", ..., "<Key-6>"
            self.root.bind(f"<Key-{i}>", lambda e, fn=action: fn())

        # Quit shortcut
        self.root.bind("<Key-q>", lambda e: self.root.quit())

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
            self.panel.config(text="All images processed.", image="")
            self.root.after(600, self.root.quit)
            return
        self._show_current()

    def _show_current(self) -> None:
        try:
            p = self.img_paths[self.index]
            pgt = self.gt_paths[self.index]
            ppred = self.pred_paths[self.index]

            # Load PIL tiles
            base = pil_open_resize(p, TILE_SIZE)
            gt = pil_open_resize(pgt, TILE_SIZE)
            pred = pil_open_resize(ppred, TILE_SIZE)

            # Overlays
            gt_overlay = overlay_green_background(base, gt)
            pred_overlay = overlay_green_background(base, pred)

            # Annotated original (2x size for legible text, matching original)
            orig_big = base.resize((2 * TILE_SIZE, 2 * TILE_SIZE), Image.BILINEAR)

            # Metrics
            diff_pct = mask_diff_percent(gt, pred)

            gt_cv = cv2_imread_any(pgt)
            pred_cv = cv2_imread_any(ppred)
            n_gt = count_gaps_components(gt_cv)
            n_pr = count_gaps_components(pred_cv)
            diff_blobs = abs(n_gt - n_pr)

            text = f"Diff-mag: {diff_pct:.2f}%\nDiff-blobs: {diff_blobs}"
            orig_annot = annotate(orig_big, text, pos=(10, 750))

            # Compose final 4*size x 2*size canvas
            canvas = Image.new("RGB", (4 * TILE_SIZE, 2 * TILE_SIZE))
            canvas.paste(orig_annot, (0, 0))
            canvas.paste(gt, (2 * TILE_SIZE, 0))
            canvas.paste(gt_overlay, (3 * TILE_SIZE, 0))
            canvas.paste(pred, (2 * TILE_SIZE, TILE_SIZE))
            canvas.paste(pred_overlay, (3 * TILE_SIZE, TILE_SIZE))

            photo = ImageTk.PhotoImage(canvas)

            bname = Path(p).name.split("+")[0]
            self.panel.configure(image=photo, text=bname, compound="bottom", font=("Helvetica", 20))
            # Keep a reference to avoid GC
            self.panel.image = photo

        except Exception as ex:
            self.panel.configure(text=f"Error: {ex}", image="")
            # Move on to the next image
            self.index += 1
            self._advance_past_preworked()
            self._show_current_or_finish()


# ---------------------------------- CLI --------------------------------------

def read_paths_file(file: Path) -> List[Path]:
    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [Path(ln.strip()) for ln in lines if ln.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strong Label (GT vs Pred tagger).")
    parser.add_argument(
        "paths_file",
        help="Text file with one image path per line (original image paths).",
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

    root = tk.Tk()
    App(root, img_paths, Path(args.register))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
