import io
import queue
import sys
from collections import OrderedDict

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QScrollArea, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread

# Import Pillow libraries for image processing.
from PIL import Image, ImageChops, ImageQt
import numpy as np


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


NUM_WORKERS = 4


class ThumbnailLoader(QThread):
    pixmap_loaded = pyqtSignal(str, tuple, int, str, object)  # cache_key, size, index, load_type, QPixmap

    def __init__(self, work_queue, parent=None):
        super().__init__(parent)
        self._queue = work_queue
        self._running = True

    def run(self):
        while self._running:
            try:
                req = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            img_path, mask_path, size, index, load_type = req
            try:
                if load_type == 'blend':
                    cache_key = "blend:" + img_path
                    pil_im = Image.open(img_path).convert('RGBA')
                    pil_mask = Image.open(mask_path).convert('RGBA')
                    pil_im.thumbnail(size, Image.LANCZOS)
                    pil_mask = pil_mask.resize(pil_im.size, resample=Image.LANCZOS)
                    blended = make_mask_green(pil_im, pil_mask)
                    buf = io.BytesIO()
                    blended.save(buf, format='PNG')
                    pixmap = QPixmap()
                    pixmap.loadFromData(buf.getvalue())
                else:
                    path = img_path if load_type == 'image' else mask_path
                    cache_key = path
                    pixmap = QPixmap(path)

                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        size[0], size[1],
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                self.pixmap_loaded.emit(cache_key, size, index, load_type, pixmap)
            except Exception:
                if load_type == 'blend':
                    cache_key = "blend:" + img_path
                elif load_type == 'mask':
                    cache_key = mask_path
                else:
                    cache_key = img_path
                self.pixmap_loaded.emit(cache_key, size, index, load_type, QPixmap())

    def stop(self):
        self._running = False


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
PLOT_MASKS = True


class VirtualizedCell(QWidget):
    clicked = pyqtSignal()

    def __init__(self, size=(THUMB_SIZE, THUMB_SIZE), parent=None):
        super().__init__(parent)
        self.current_index = -1
        self.current_type = 'image'
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
        placeholder = QPixmap(self._size[0], self._size[1])
        placeholder.fill(Qt.gray)
        self.img_label.setPixmap(placeholder)

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
    NUM_COLUMNS = 3
    POOL_ROWS = 8
    ROW_HEIGHT = THUMB_SIZE + 12
    CELL_WIDTH = THUMB_SIZE + 12
    ROW_TYPES = ['image', 'mask', 'blend'] if PLOT_MASKS else ['image']
    ROWS_PER_GROUP = len(ROW_TYPES)

    def __init__(self, image_mask_pairs):
        super().__init__()
        self.selected_images = set()
        self.image_mask_pairs = image_mask_pairs
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_selection)
        self._thumbnail_cache = ThumbnailCache(max_size=800)
        self._work_queue = queue.Queue()
        self._thumbnail_loaders = []
        for _ in range(NUM_WORKERS):
            loader = ThumbnailLoader(self._work_queue, self)
            loader.pixmap_loaded.connect(self._on_pixmap_loaded)
            loader.start()
            self._thumbnail_loaders.append(loader)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        num_item_rows = (len(self.image_mask_pairs) + self.NUM_COLUMNS - 1) // self.NUM_COLUMNS
        content_height = max(1, num_item_rows * self.ROWS_PER_GROUP) * self.ROW_HEIGHT
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

    def _on_cell_clicked(self, cell):
        if cell.current_index < 0:
            return
        self.toggle_selection(cell.current_index, cell)

    def _on_pixmap_loaded(self, cache_key, size, index, load_type, pixmap):
        self._thumbnail_cache.put(cache_key, size, pixmap)
        for cell in self.pool_cells:
            if cell.current_index == index and cell.current_type == load_type:
                cell.set_pixmap(pixmap)
                break

    def _update_visible_cells(self):
        while not self._work_queue.empty():
            try:
                self._work_queue.get_nowait()
            except queue.Empty:
                break
        scroll_value = self.scroll_area.verticalScrollBar().value()
        first_row = scroll_value // self.ROW_HEIGHT
        for i, cell in enumerate(self.pool_cells):
            r = i // self.NUM_COLUMNS
            c = i % self.NUM_COLUMNS
            visual_row = first_row + r
            item_group = visual_row // self.ROWS_PER_GROUP
            sub_row = visual_row % self.ROWS_PER_GROUP
            row_type = self.ROW_TYPES[sub_row]
            index = item_group * self.NUM_COLUMNS + c
            if index >= len(self.image_mask_pairs):
                cell.setVisible(False)
                continue
            cell.setVisible(True)
            cell.setGeometry(
                c * self.CELL_WIDTH,
                visual_row * self.ROW_HEIGHT,
                self.CELL_WIDTH,
                self.ROW_HEIGHT
            )
            cell.current_index = index
            cell.current_type = row_type
            img_path, mask_path = self.image_mask_pairs[index]
            cell.update_selection_style(index in self.selected_images)

            if row_type == 'image':
                cache_key = img_path
            elif row_type == 'mask':
                cache_key = mask_path
            else:
                cache_key = "blend:" + img_path

            cached = self._thumbnail_cache.get(cache_key, (THUMB_SIZE, THUMB_SIZE))
            if cached is not None:
                cell.set_pixmap(cached)
            else:
                cell.set_placeholder()
                self._work_queue.put((img_path, mask_path, (THUMB_SIZE, THUMB_SIZE), index, row_type))

    def toggle_selection(self, idx, widget):
        if idx in self.selected_images:
            self.selected_images.remove(idx)
        else:
            self.selected_images.add(idx)
        is_selected = idx in self.selected_images
        for cell in self.pool_cells:
            if cell.current_index == idx:
                cell.update_selection_style(is_selected)

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
        for loader in self._thumbnail_loaders:
            loader.stop()
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
