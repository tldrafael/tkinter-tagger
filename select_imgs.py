import hashlib
import os
import sys
import threading
from collections import OrderedDict, deque
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QScrollArea, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QImageReader
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, QSize

# Import Pillow libraries for image processing.
from PIL import Image, ImageChops, ImageQt
import numpy as np


# Small JPEGs written after the first decode. Keyed by path, mtime, and size.
THUMB_CACHE_DIR = Path.home() / ".cache" / "select_imgs_thumbs"
N_LOADERS = 4


# In-memory LRU cache for thumbnails. Key: (img_path, size), value: QPixmap.
class ThumbnailCache:
    def __init__(self, max_size=800):
        self._cache = OrderedDict()
        self._max_size = max_size

    def get(self, img_path, size):
        key = (img_path, size)
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, img_path, size, pixmap):
        key = (img_path, size)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = pixmap
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


def _disk_thumb_path(img_path):
    st = os.stat(img_path)
    key = hashlib.sha1(os.path.abspath(img_path).encode()).hexdigest()
    return THUMB_CACHE_DIR / f"{key}_{st.st_mtime_ns}_{st.st_size}.jpg"


def load_thumbnail(img_path, size):
    """Decode a thumbnail, using a reduced JPEG size and the on-disk cache.

    QImage(path) decodes the full frame (hundreds of MB for the large photos
    in this list). QImageReader.setScaledSize asks the JPEG decoder for a
    smaller image, which is both faster and much lighter.
    """
    try:
        cached_path = _disk_thumb_path(img_path)
    except OSError:
        cached_path = None
    if cached_path is not None and cached_path.is_file():
        cached = QImage(str(cached_path))
        if not cached.isNull():
            return cached, False

    reader = QImageReader(img_path)
    reader.setAutoTransform(True)
    orig = reader.size()
    tw, th = size
    if orig.isValid() and orig.width() > 0 and orig.height() > 0:
        scale = min(tw / orig.width(), th / orig.height(), 1.0)
        if scale < 1.0:
            reader.setScaledSize(QSize(
                max(1, round(orig.width() * scale)),
                max(1, round(orig.height() * scale)),
            ))
    return reader.read(), True


def store_thumbnail(img_path, image):
    if image.isNull():
        return
    try:
        cached_path = _disk_thumb_path(img_path)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached_path.with_name(cached_path.name + ".tmp")
        if image.save(str(tmp), "JPEG", 80):
            os.replace(tmp, cached_path)
    except OSError:
        pass


class _LoadQueue:
    """Visible-cell jobs only. A new viewport replaces anything still pending."""

    def __init__(self):
        self._cv = threading.Condition()
        self._pending = deque()
        self._wanted = set()
        self._inflight = set()
        self._running = True

    def set_requests(self, jobs):
        with self._cv:
            self._wanted = set(jobs)
            self._pending.clear()
            for job in jobs:
                if job not in self._inflight:
                    self._pending.append(job)
            self._cv.notify_all()

    def pop(self):
        with self._cv:
            while self._running:
                while self._pending:
                    job = self._pending.popleft()
                    if job in self._wanted and job not in self._inflight:
                        self._inflight.add(job)
                        return job
                self._cv.wait(timeout=0.2)
            return None

    def done(self, job):
        with self._cv:
            self._inflight.discard(job)

    def stop(self):
        with self._cv:
            self._running = False
            self._pending.clear()
            self._cv.notify_all()


# Loads thumbnails off the GUI thread and emits when each is done.
class ThumbnailLoader(QThread):
    # QImage is thread-safe (unlike QPixmap); convert to QPixmap on the main thread.
    image_loaded = pyqtSignal(str, tuple, int, object)  # img_path, size, index, QImage

    def __init__(self, queue, parent=None):
        super().__init__(parent)
        self._queue = queue

    def run(self):
        while True:
            req = self._queue.pop()
            if req is None:
                return
            img_path, size, index = req
            fresh = False
            try:
                image, fresh = load_thumbnail(img_path, size)
            except Exception:
                image = QImage()
            self.image_loaded.emit(img_path, size, index, image)
            self._queue.done(req)
            if fresh:
                store_thumbnail(img_path, image)


# A clickable QLabel that emits a signal when clicked.
class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()


def make_mask_green(pil_im, pil_mask):
    """
    Replace pixels in pil_mask that are completely black (R=G=B=0)
    with green (0, 255, 0), keeping the original alpha.
    """
    fg = np.array(pil_im) / 255
    bg = np.array(pil_mask) / 255
    green = np.array([[[0, 1., 0, 1.]]])
    # blends = im/255*p/255 + bg*(1-p/255)

    data = fg * bg + (1 - bg) * green
    data = (data*255).astype(np.uint8)
    return Image.fromarray(data)


# Cell widget for the virtualized grid: shows thumbnail or placeholder, clickable, shows selection border.
THUMB_SIZE = 128*3


class VirtualizedCell(QWidget):
    clicked = pyqtSignal()
    _placeholder = None

    def __init__(self, size=(THUMB_SIZE, THUMB_SIZE), parent=None):
        super().__init__(parent)
        self.current_index = -1
        self._size = size
        self.setFixedSize(size[0] + 8, size[1] + 8)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.img_label = ClickableLabel()
        self.img_label.setFixedSize(size[0], size[1])
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.clicked.connect(self.clicked)
        layout.addWidget(self.img_label)
        self.set_placeholder()
        self.setStyleSheet("border: 1px solid gray;")

    def set_placeholder(self):
        if VirtualizedCell._placeholder is None:
            placeholder = QPixmap(self._size[0], self._size[1])
            placeholder.fill(Qt.gray)
            VirtualizedCell._placeholder = placeholder
        self.img_label.setPixmap(VirtualizedCell._placeholder)

    def set_pixmap(self, pixmap):
        if pixmap is None or pixmap.isNull():
            self.set_placeholder()
        else:
            scaled = pixmap.scaled(
                self._size[0], self._size[1],
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.img_label.setPixmap(scaled)

    def update_selection_style(self, is_selected):
        self.setStyleSheet("border: 5px solid blue;" if is_selected else "border: 1px solid gray;")


# The main gallery: virtualized grid with a fixed pool of cells, async thumbnails, debounced save.
class ImageGallery(QWidget):
    NUM_COLUMNS = 4
    POOL_ROWS = 8
    ROW_HEIGHT = THUMB_SIZE + 12
    CELL_WIDTH = THUMB_SIZE + 12

    def __init__(self, image_mask_pairs):
        super().__init__()
        self.selected_images = set()
        self.image_mask_pairs = image_mask_pairs
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_selection)
        self._thumbnail_cache = ThumbnailCache(max_size=800)
        self._load_queue = _LoadQueue()
        self._thumbnail_loaders = []
        for _ in range(N_LOADERS):
            loader = ThumbnailLoader(self._load_queue, self)
            loader.image_loaded.connect(self._on_image_loaded)
            loader.start()
            self._thumbnail_loaders.append(loader)
        self._last_first_row = None
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.timeout.connect(self._request_visible_loads)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        num_rows = (len(self.image_mask_pairs) + self.NUM_COLUMNS - 1) // self.NUM_COLUMNS
        content_height = max(1, num_rows) * self.ROW_HEIGHT
        self.content_widget = QWidget()
        self.content_widget.setFixedSize(
            self.NUM_COLUMNS * self.CELL_WIDTH,
            content_height
        )
        self.scroll_area.setWidget(self.content_widget)

        self.pool_cells = []
        thumb_size = (THUMB_SIZE, THUMB_SIZE)
        for r in range(self.POOL_ROWS):
            for c in range(self.NUM_COLUMNS):
                cell = VirtualizedCell(size=thumb_size, parent=self.content_widget)
                cell.clicked.connect(lambda w=cell: self._on_cell_clicked(w))
                self.pool_cells.append(cell)

        self.scroll_area.verticalScrollBar().valueChanged.connect(self._update_visible_cells)
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)
        self._update_visible_cells()

        first_row_paths = [
            self.image_mask_pairs[i][0]
            for i in range(min(self.NUM_COLUMNS, len(self.image_mask_pairs)))
        ]
        print("First row image paths:")
        for p in first_row_paths:
            print(f"  {p}")

    def _on_cell_clicked(self, cell):
        if cell.current_index < 0:
            return
        self.toggle_selection(cell.current_index, cell)

    def _on_image_loaded(self, img_path, size, index, qimage):
        pixmap = QPixmap.fromImage(qimage) if not qimage.isNull() else QPixmap()
        self._thumbnail_cache.put(img_path, size, pixmap)
        for cell in self.pool_cells:
            if cell.current_index == index:
                cell.set_pixmap(pixmap)
                break

    def _update_visible_cells(self):
        scroll_value = self.scroll_area.verticalScrollBar().value()
        first_row = scroll_value // self.ROW_HEIGHT
        # The scrollbar reports every pixel. The set of rows only changes once
        # per row, and re-queueing on each pixel was re-decoding the same JPEGs.
        if first_row == self._last_first_row:
            return
        self._last_first_row = first_row
        self._layout_pool(first_row)
        self._load_timer.start(60)

    def _layout_pool(self, first_row):
        thumb_size = (THUMB_SIZE, THUMB_SIZE)
        for i, cell in enumerate(self.pool_cells):
            r = i // self.NUM_COLUMNS
            c = i % self.NUM_COLUMNS
            index = (first_row + r) * self.NUM_COLUMNS + c
            if index >= len(self.image_mask_pairs):
                cell.current_index = -1
                cell.setVisible(False)
                continue
            cell.setVisible(True)
            cell.setGeometry(
                c * self.CELL_WIDTH,
                (first_row + r) * self.ROW_HEIGHT,
                self.CELL_WIDTH,
                self.ROW_HEIGHT
            )
            img_path, _mask_path = self.image_mask_pairs[index]
            cell.current_index = index
            cell.update_selection_style(index in self.selected_images)
            cached = self._thumbnail_cache.get(img_path, thumb_size)
            if cached is not None:
                cell.set_pixmap(cached)
            else:
                cell.set_placeholder()

    def _request_visible_loads(self):
        thumb_size = (THUMB_SIZE, THUMB_SIZE)
        pending = []
        for cell in self.pool_cells:
            index = cell.current_index
            if index < 0:
                continue
            img_path = self.image_mask_pairs[index][0]
            if self._thumbnail_cache.get(img_path, thumb_size) is None:
                pending.append((img_path, thumb_size, index))
        self._load_queue.set_requests(pending)

    def toggle_selection(self, idx, widget):
        if idx in self.selected_images:
            self.selected_images.remove(idx)
            widget.setStyleSheet("border: 1px solid gray;")
        else:
            self.selected_images.add(idx)
            widget.setStyleSheet("border: 5px solid blue;")

        img_path, _ = self.image_mask_pairs[idx]
        marker = "im+"
        pos = img_path.find(marker)
        if pos != -1:
            print(img_path[pos + len(marker):])
        else:
            print(img_path)

        print("Selected:", len(self.selected_images), "indices")
        self._save_timer.start(400)

    def save_selection(self):
        with open("out.txt", "w") as f:
            for idx in self.selected_images:
                img_path, _ = self.image_mask_pairs[idx]
                f.write(img_path + "\n")
        print("Saved selected image paths to out.txt")

    def closeEvent(self, event):
        self._load_queue.stop()
        for loader in self._thumbnail_loaders:
            loader.wait(1000)
        super().closeEvent(event)


import re
def get_gtpath(p):
    newp = re.sub(r'\.jpg$', '.png', p)
    newp = re.sub(r'\/im\/', '/gt/', newp)
    # newp = re.sub(r'\/im\/', '/results/20250718_tritonserver/', newp)
    return newp


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Example list of (image, mask) pairs. Replace these paths with your actual files.
    fpath = sys.argv[1]
    with open(fpath, 'r') as f:
        images = f.read().split('\n')[:-1]

    images = images[:]
    image_mask_pairs = [(p, get_gtpath(p)) for p in images]

    gallery = ImageGallery(image_mask_pairs)
    gallery.resize(800, 600)
    gallery.show()
    sys.exit(app.exec_())
