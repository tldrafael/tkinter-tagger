#!/usr/bin/env python3
"""
Strong Label – multi-case visual tagger for segmentation masks.

Loads image paths from a text file and displays them with their masks.
Allows labeling images with keyboard shortcuts or buttons.
Saves labels to a CSV file.
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


# ============================================================================
# Configuration
# ============================================================================

REGISTER_FILE = "labels.txt"

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-C.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


# ============================================================================
# Image Processing
# ============================================================================

def load_image(path: Path) -> Image.Image:
    """Load a PIL Image from file."""
    return Image.open(path)


def load_and_resize_image(path: Path, tile_size, print_=False) -> Image.Image:
    """Load image and resize preserving aspect ratio."""
    img = load_image(path)
    if print_:
        print(path, img.size)
    if isinstance(tile_size, tuple):
        max_w = tile_size[0]
        max_h = tile_size[1]
        # Keep the aspect ratio and find the greatest size for the image, such that any dimension is not larger than the tile size
        scales = [max_w / img.size[0], max_h / img.size[1]]
        scale = min(scales)
    else:
        scale = tile_size / max(img.size)
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    if print_:
        print(new_size)
    return img.resize(new_size, Image.BILINEAR)


def image_to_grayscale_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to grayscale numpy array."""
    return np.asarray(img.convert("L"), dtype=np.uint8)


def grayscale_array_to_image(arr: np.ndarray) -> Image.Image:
    """Convert grayscale numpy array to RGB PIL Image."""
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def create_overlay(base_img: Image.Image, mask_img: Image.Image) -> Image.Image:
    """
    Create overlay: where mask is white, show base image;
    where mask is black, show green background.
    """
    base = np.asarray(base_img.convert("RGB"), dtype=np.float32) / 255.0
    mask = image_to_grayscale_array(mask_img).astype(np.float32) / 255.0

    # Expand mask to 3 channels if needed
    if mask.ndim == 2:
        mask = mask[..., None]

    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    result = (1 - mask) * green + base * mask
    result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(result)


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a font from available candidates, or use default."""
    for font_path in FONT_CANDIDATES:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def add_text_to_image(img: Image.Image, text_lines: List[str],
                      position: Tuple[int, int], font_size: int) -> Image.Image:
    """Draw text lines onto an image copy."""
    result = img.copy()
    draw = ImageDraw.Draw(result)
    font = load_font(font_size)

    x, y = position
    for line in text_lines:
        draw.text((x, y), line, fill=(255, 0, 255), font=font)
        y += int(font_size * 1.2)

    return result


# ============================================================================
# Mask Analysis
# ============================================================================

def load_mask_cv2(path: Path) -> np.ndarray:
    """Load mask image using OpenCV."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def count_connected_components(mask: np.ndarray, min_area: int = 1,
                               max_area: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Count connected components in mask within area range.
    Returns (cleaned_mask, component_count).
    """
    if max_area is None:
        max_area = mask.shape[0] * mask.shape[1] + 1

    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    cleaned = np.zeros_like(mask_u8)
    count = 0

    for label in range(1, num_labels):  # Skip background (label 0)
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            cleaned[labels == label] = 255
            count += 1

    return cleaned, count


def count_gap_components(mask: np.ndarray, threshold: int = 10) -> int:
    """
    Count small 'gap' components in mask.
    Pixels below threshold are considered foreground.
    """
    if isinstance(mask, Image.Image):
        mask = np.asarray(mask, dtype=np.uint8)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Invert: pixels < threshold become foreground (255)
    foreground = np.where(mask < threshold, 255, 0).astype(np.uint8)

    # Limit component size to 25% of image to avoid counting large regions
    h, w = foreground.shape
    max_component_area = int(0.25 * h * w)

    _, count = count_connected_components(foreground, min_area=2, max_area=max_component_area)
    return count


def calculate_mask_difference(gt_mask: Image.Image, pred_mask: Image.Image) -> float:
    """Calculate percentage of pixels that differ between two masks."""
    gt = image_to_grayscale_array(gt_mask).astype(np.float32) / 255.0
    pred = image_to_grayscale_array(pred_mask).astype(np.float32) / 255.0

    diff = np.abs(gt - pred).sum()
    total_pixels = gt.shape[0] * gt.shape[1]

    return 100.0 * diff / total_pixels


# ============================================================================
# Path Mapping
# ============================================================================

def get_gt_path_from_image_path(image_path: Path) -> Path:
    """Convert image path to ground truth mask path."""
    path_str = str(image_path)
    path_str = path_str.replace("/im/", "/gt/").replace(".jpg", ".png")
    return Path(path_str)


def get_mask_path(image_path: Path, case: str) -> Path:
    """
    Get mask path for a given case.
    - 'gt' -> ground truth path
    - other -> results/{case}/ path
    """
    gt_path = get_gt_path_from_image_path(image_path)
    if case.lower() == "gt":
        return gt_path
    return Path(str(gt_path).replace("/gt/", f"/results/{case}/"))


# ============================================================================
# Layout Calculation
# ============================================================================

def calculate_grid_layout(
    num_tiles: int,
    screen_width: int,
    screen_height: int,
    reserved_height: int,
    margin: int = 8,
    gap: int = 8,
    max_cols: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Calculate optimal grid layout for tiles.
    Always uses 2 rows.
    Returns (columns, rows, tile_size).
    """
    # Frame padding: grid_frame has padx=8, pady=8 (16 pixels total on each side)
    frame_padding = 16  # 8 pixels on each side

    # Tile padding: each Label has padx=4, pady=4 (8 pixels total per tile)
    # For cols columns: cols * 8 pixels total horizontal padding
    # For rows rows: rows * 8 pixels total vertical padding
    tile_padding_per_tile = 8

    # Always use 2 rows
    rows = 2
    cols = math.ceil(num_tiles / rows)

    # screen_width = 1920
    # screen_height = 1080

    # Calculate available space accounting for all padding
    # Horizontal: window width - frame padding - all tile padding
    available_width = max(100, screen_width - frame_padding - cols * tile_padding_per_tile)

    # Vertical: window height - frame padding - all tile padding - reserved height for buttons
    available_height = max(100, screen_height - frame_padding - rows * tile_padding_per_tile - reserved_height)

    # Calculate maximum tile size that fits in width
    max_tile_width = available_width / cols

    # Calculate maximum tile size that fits in height
    max_tile_height = available_height / rows

    # Use the minimum to ensure it fits both dimensions - this gives us the maximum possible size
    # that doesn't exceed the window in either dimension
    tile_size = int(min(max_tile_width, max_tile_height))

    # Ensure tile_size is at least 1
    tile_size = max(1, tile_size)

    # return cols, rows, tile_size
    return cols, rows, (int(max_tile_width), int(max_tile_height))


# ============================================================================
# Main Application
# ============================================================================

class App:
    def __init__(self, root: tk.Tk, image_paths: Sequence[Path],
                 cases: Sequence[str], labels: Sequence[str], register_path: Path):
        self.root = root
        self.root.title("Strong Label – multi-case")

        # State
        self.image_paths = [Path(p) for p in image_paths]
        self.cases = list(cases)
        self.labels = list(labels)
        self.register_path = Path(register_path)
        self.processed_images = self._load_processed_images()
        self.current_index = 0

        # UI setup
        self._setup_ui()

        # Show first image
        self._skip_processed_images()
        self._display_current_image()

    def _setup_ui(self):
        """Initialize UI components."""
        # Main grid frame for images
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Button frame for labels
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

        # Configure grid resizing
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Create label buttons
        self._create_label_buttons()

        # Calculate reserved height for buttons
        self.root.update_idletasks()
        self.reserved_height = self.button_frame.winfo_reqheight() + 24

    def _create_label_buttons(self):
        """Create buttons for each label and bind keyboard shortcuts."""
        for i, label in enumerate(self.labels, start=1):
            action = self._make_label_action(label)
            button = tk.Button(
                self.button_frame,
                text=f"{i}. {label}",
                command=action,
                width=42,
                height=2
            )
            button.grid(row=0, column=i - 1, padx=3, pady=3)
            self.root.bind(f"<Key-{i}>", lambda e, fn=action: fn())

        self.root.bind("<Key-q>", lambda e: self.root.quit())

    def _make_label_action(self, label: str):
        """Create action function for a label."""
        def action():
            self._save_label(label)
            self.current_index += 1
            self._skip_processed_images()
            self._display_current_image()
        return action

    def _load_processed_images(self) -> set[str]:
        """Load set of already processed image paths from register file."""
        if not self.register_path.exists():
            return set()

        processed = set()
        try:
            text = self.register_path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                line = line.strip()
                if line and "," in line:
                    image_path = line.split(",", 1)[0]
                    processed.add(image_path)
        except Exception:
            pass

        return processed

    def _save_label(self, label: str):
        """Save label for current image to register file."""
        self.register_path.parent.mkdir(parents=True, exist_ok=True)
        with self.register_path.open("a", encoding="utf-8") as f:
            f.write(f"{self.image_paths[self.current_index]},{label}\n")

    def _skip_processed_images(self):
        """Skip images that have already been processed."""
        while (self.current_index < len(self.image_paths) and
               str(self.image_paths[self.current_index]) in self.processed_images):
            self.current_index += 1

    def _display_current_image(self):
        """Display current image with all masks and overlays."""
        if self.current_index >= len(self.image_paths):
            self._show_completion_message()
            return

        # Calculate layout
        total_tiles = 1 + 2 * len(self.cases)  # 1 original + 2 per case (mask + overlay)
        # Use actual window size instead of screen size
        self.root.update_idletasks()  # Ensure window is updated
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        # window_height = 1500

        # Fallback to screen size if window not yet sized
        if window_width <= 1 or window_height <= 1:
            window_width = self.root.winfo_screenwidth()
            window_height = self.root.winfo_screenheight()

        cols, rows, tile_size = calculate_grid_layout(
            total_tiles, window_width, window_height, self.reserved_height
        )

        # Clear previous display
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        # Update title
        current_path = self.image_paths[self.current_index]
        title = f"Strong Label – {self.current_index + 1}/{len(self.image_paths)} – {current_path.name}"
        self.root.title(title)

        # Load base image
        try:
            base_image = load_and_resize_image(current_path, tile_size, print_=True)
        except Exception as e:
            self._show_error(str(e), tile_size)
            return

        # Load GT mask if available
        gt_mask_pil = None
        gt_mask_cv = None
        if any(c.lower() == "gt" for c in self.cases):
            gt_path = get_mask_path(current_path, "gt")
            if gt_path.exists():
                try:
                    gt_mask_pil = load_and_resize_image(gt_path, tile_size)
                    gt_mask_cv = load_mask_cv2(gt_path)
                except Exception:
                    pass

        # Build tiles
        tiles = []

        # Original image tile
        tiles.append(("original", None, base_image))

        # Case tiles (mask + overlay for each case)
        for i, case in enumerate(self.cases):
            mask_path = get_mask_path(current_path, case)

            # Load mask
            mask_pil = None
            if mask_path.exists():
                try:
                    mask_pil = load_and_resize_image(mask_path, tile_size)
                    if i == 0:
                        reference_mask_pil = mask_pil.copy()
                except Exception:
                    pass

            # Mask tile
            if mask_pil is None:
                mask_tile = self._create_placeholder_tile(tile_size, f"{case}\n(mask missing)")
            else:
                mask_array = image_to_grayscale_array(mask_pil)
                mask_tile = grayscale_array_to_image(mask_array)
            tiles.append(("mask", case, mask_tile))

            # Overlay tile
            if mask_pil is None:
                overlay_tile = self._create_placeholder_tile(tile_size, f"{case}\n(overlay missing)")
            else:
                overlay_tile = create_overlay(base_image, mask_pil)

                # Add metrics if GT is available
                if i > 0:
                    reference = self.cases[0]
                    try:
                        diff_percent = calculate_mask_difference(reference_mask_pil, mask_pil)
                        diff_blobs = count_gap_components(mask_pil) - count_gap_components(reference_mask_pil)
                        txt = f"diff-mag: {diff_percent:.2f}%\ndiff-blob: {diff_blobs}"
                        overlay_tile = add_text_to_image(overlay_tile, [txt], (10, 10), 14)
                    except Exception:
                        pass

            tiles.append(("overlay", case, overlay_tile))

        # Custom tile arrangement: interleave original with cases
        if len(tiles) > 1:
            original_tile = tiles[0]
            case_tiles = tiles[1:]
            # Create empty placeholder
            empty = Image.new("RGB", original_tile[2].size, (0, 0, 0))
            # Interleave: empty, original, case1_mask, case1_overlay, case2_mask, ...
            arranged = [("empty", None, empty), original_tile] + case_tiles
            # Reorder: take every other starting from index 1, then from index 0
            tiles = arranged[1::2] + arranged[::2]

        # Display tiles in grid
        self._display_tiles(tiles, cols, tile_size)

    def _display_tiles(self, tiles: List[Tuple[str, Optional[str], Image.Image]],
                      cols: int, tile_size: int):
        """Display tiles in a grid layout."""
        photo_images = []

        for idx, (_, _, img) in enumerate(tiles):
            row = idx // cols
            col = idx % cols

            photo = ImageTk.PhotoImage(img)
            photo_images.append(photo)

            label = tk.Label(self.grid_frame, image=photo)
            label.image = photo  # Keep reference to prevent garbage collection
            label.grid(row=row, column=col, padx=4, pady=4)

        # Keep references to prevent garbage collection
        self._photo_images = photo_images
        self.root.update_idletasks()

    def _create_placeholder_tile(self, size, text: str) -> Image.Image:
        """Create a placeholder tile with text."""
        if isinstance(size, tuple):
            size = size[0]
        size = int(size)
        img = Image.new("RGB", (size, size), (40, 40, 40))
        draw = ImageDraw.Draw(img)
        font = load_font(max(14, size // 28))

        lines = text.split("\n")
        total_height = int(len(lines) * font.size * 1.2)
        y = (size - total_height) // 2

        for line in lines:
            # Get text size (PIL method varies by version)
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except AttributeError:
                text_width, _ = draw.textsize(line, font=font)

            x = (size - text_width) // 2
            draw.text((x, y), line, fill=(255, 100, 100), font=font)
            y += int(font.size * 1.2)

        return img

    def _show_error(self, error_message: str, tile_size):
        """Display error message."""
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        img = self._create_placeholder_tile(tile_size, f"Error\n{error_message}")
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(self.grid_frame, image=photo)
        label.image = photo
        label.grid(row=0, column=0, padx=4, pady=4)

    def _show_completion_message(self):
        """Show message when all images are processed."""
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        done_label = tk.Label(
            self.grid_frame,
            text="All images processed.",
            font=("Helvetica", 20)
        )
        done_label.pack(padx=20, pady=20)
        self.root.after(600, self.root.quit)


# ============================================================================
# Command Line Interface
# ============================================================================

def read_image_paths(file_path: Path) -> List[Path]:
    """Read image paths from text file (one per line)."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        paths = [Path(line.strip()) for line in text.splitlines() if line.strip()]
        return paths
    except Exception:
        return []


def parse_cases(cases_arg: Optional[str]) -> List[str]:
    """
    Parse cases from argument or environment.
    Returns list of case names.
    """
    if cases_arg and cases_arg.strip():
        return [c.strip() for c in cases_arg.split(",") if c.strip()]

    # Try environment variable
    env_run = os.getenv("RUN_NAME")
    if env_run:
        return ["gt", env_run]

    # Default
    return ["gt"]


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Strong Label – multi-case visual tagger for segmentation masks."
    )
    parser.add_argument(
        "paths_file",
        help="Text file with one original image path per line.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help='Comma-separated list of cases. Example: "gt,runA,runB". '
             'If omitted, uses ["gt", $RUN_NAME] if RUN_NAME is set, else ["gt"].',
    )
    parser.add_argument(
        "--register",
        default=REGISTER_FILE,
        help=f"Output CSV file for labels (default: {REGISTER_FILE})",
    )
    args = parser.parse_args(argv)

    # Read image paths
    image_paths = read_image_paths(Path(args.paths_file))
    if not image_paths:
        print("No image paths found in the provided file.")
        return 1

    # Parse cases
    cases = parse_cases(args.cases)
    if not cases:
        print("No cases provided. Use --cases or set RUN_NAME environment variable.")
        return 1

    # Build labels from cases plus "all bad" and "odd"
    labels = list(cases) + ["all bad", "odd"]

    # Create and run application
    root = tk.Tk()

    # Try to maximize window
    try:
        root.state("zoomed")  # Windows
    except Exception:
        try:
            root.attributes("-zoomed", True)  # X11
        except Exception:
            pass

    app = App(root, image_paths, cases, labels, Path(args.register))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
