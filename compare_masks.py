import queue
import re
import sys
from collections import OrderedDict

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QScrollArea, QHBoxLayout, QVBoxLayout,
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread


class ThumbnailCache:
    def __init__(self, max_size=800):
        self._cache = OrderedDict()
        self._max_size = max_size

    def get(self, key, size):
        k = (key, size)
        if k not in self._cache:
            return None
        self._cache.move_to_end(k)
        return self._cache[k]

    def put(self, key, size, pixmap):
        k = (key, size)
        if k in self._cache:
            self._cache.move_to_end(k)
        self._cache[k] = pixmap
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


NUM_WORKERS = 4


class ThumbnailLoader(QThread):
    pixmap_loaded = pyqtSignal(str, tuple, int, int, object)

    def __init__(self, work_queue, parent=None):
        super().__init__(parent)
        self._queue = work_queue
        self._running = True

    def run(self):
        while self._running:
            try:
                path, size, row, col = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        size[0], size[1],
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                self.pixmap_loaded.emit(path, size, row, col, pixmap)
            except Exception:
                self.pixmap_loaded.emit(path, size, row, col, QPixmap())

    def stop(self):
        self._running = False


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()


THUMB_SIZE = 192


class VirtualizedCell(QWidget):
    clicked = pyqtSignal()

    def __init__(self, size=(THUMB_SIZE, THUMB_SIZE), parent=None):
        super().__init__(parent)
        self.current_row = -1
        self.current_col = -1
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
            self.img_label.setPixmap(pixmap)

    def update_selection_style(self, is_selected):
        self.setStyleSheet(
            "border: 5px solid blue;" if is_selected else "border: 1px solid gray;"
        )


def get_mask_path(img_path, hypothesis):
    """Replace /im/ with /results/{hypothesis}/ and .jpg with .png."""
    mask_path = re.sub(r"/im/", f"/results/{hypothesis}/", img_path)
    mask_path = re.sub(r"\.jpg$", ".png", mask_path)
    return mask_path


class MaskComparisonGallery(QWidget):
    POOL_ROWS = 8
    ROW_HEIGHT = THUMB_SIZE + 12
    CELL_WIDTH = THUMB_SIZE + 12

    def __init__(self, image_paths, hypotheses):
        super().__init__()
        self.image_paths = image_paths
        self.hypotheses = hypotheses
        self.num_columns = 1 + len(hypotheses)
        self.selected_images = set()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_selection)

        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(20)
        self._scroll_timer.timeout.connect(self._update_visible_cells)

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
        main_layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(0)
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(11)
        for i, name in enumerate(["image"] + self.hypotheses):
            label = QLabel(name)
            label.setFont(header_font)
            label.setFixedWidth(self.CELL_WIDTH)
            label.setFixedHeight(28)
            label.setAlignment(Qt.AlignCenter)
            header_layout.addWidget(label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        num_rows = len(self.image_paths)
        content_height = max(1, num_rows) * self.ROW_HEIGHT
        content_width = self.num_columns * self.CELL_WIDTH

        self.content_widget = QWidget()
        self.content_widget.setFixedSize(content_width, content_height)
        self.scroll_area.setWidget(self.content_widget)

        self.pool_cells = []
        thumb_size = (THUMB_SIZE, THUMB_SIZE)
        for _ in range(self.POOL_ROWS):
            for _ in range(self.num_columns):
                cell = VirtualizedCell(size=thumb_size, parent=self.content_widget)
                cell.clicked.connect(lambda w=cell: self._on_cell_clicked(w))
                self.pool_cells.append(cell)

        self.scroll_area.verticalScrollBar().valueChanged.connect(
            self._on_scroll
        )
        main_layout.addWidget(self.scroll_area)
        self._update_visible_cells()

    def _path_for_cell(self, row, col):
        img_path = self.image_paths[row]
        if col == 0:
            return img_path
        return get_mask_path(img_path, self.hypotheses[col - 1])

    def _on_scroll(self):
        if not self._scroll_timer.isActive():
            self._scroll_timer.start()

    def _on_cell_clicked(self, cell):
        if cell.current_row < 0:
            return
        self.toggle_selection(cell.current_row)

    def _on_pixmap_loaded(self, path, size, row, col, pixmap):
        self._thumbnail_cache.put(path, size, pixmap)
        for cell in self.pool_cells:
            if cell.current_row == row and cell.current_col == col:
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
            r = i // self.num_columns
            c = i % self.num_columns
            visual_row = first_row + r

            if visual_row >= len(self.image_paths):
                cell.setVisible(False)
                cell.current_row = -1
                cell.current_col = -1
                continue

            if cell.current_row == visual_row and cell.current_col == c:
                cell.setVisible(True)
                continue

            cell.setVisible(True)
            cell.setGeometry(
                c * self.CELL_WIDTH,
                visual_row * self.ROW_HEIGHT,
                self.CELL_WIDTH,
                self.ROW_HEIGHT,
            )
            cell.current_row = visual_row
            cell.current_col = c
            cell.update_selection_style(visual_row in self.selected_images)

            path = self._path_for_cell(visual_row, c)
            cached = self._thumbnail_cache.get(path, (THUMB_SIZE, THUMB_SIZE))
            if cached is not None:
                cell.set_pixmap(cached)
            else:
                cell.set_placeholder()
                self._work_queue.put(
                    (path, (THUMB_SIZE, THUMB_SIZE), visual_row, c)
                )

    def toggle_selection(self, row_idx):
        if row_idx in self.selected_images:
            self.selected_images.remove(row_idx)
        else:
            self.selected_images.add(row_idx)
        is_selected = row_idx in self.selected_images
        for cell in self.pool_cells:
            if cell.current_row == row_idx:
                cell.update_selection_style(is_selected)

        img_path = self.image_paths[row_idx]
        print(img_path)
        print("Selected:", len(self.selected_images), "images")
        self._save_timer.start(400)

    def save_selection(self):
        with open("out.txt", "w") as f:
            for idx in sorted(self.selected_images):
                f.write(self.image_paths[idx] + "\n")
        print("Saved selected image paths to out.txt")

    def closeEvent(self, event):
        for loader in self._thumbnail_loaders:
            loader.stop()
        for loader in self._thumbnail_loaders:
            loader.wait(1000)
        super().closeEvent(event)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_file> <H1> [H2] ...")
        sys.exit(1)

    fpath = sys.argv[1]
    # hypotheses = sys.argv[2:]
    hypotheses = [
        # '../gt',
        # '20260106-dinov3-cnnVit-600k-L384-unfrzBB-algnCrnr-higherOS_L768',
        '20260106-dinov3-cnnVit-600k-L384-unfrzBB-algnCrnr-higherOS_L768_tuned',
        '20260106-dinov3-cnnVit-600k-L384-unfrzBB-algnCrnr-higherOS_L768_tuned_L1152-noIS',
        '20260106-dinov3-cnnVit-600k-L384-unfrzBB-algnCrnr-higherOS_L768_tuned_L1152-noIS_L1536',
        '20260106-dinov3-cnnVit-600k-L384-unfrzBB-algnCrnr-higherOS_L768_tuned_L1152-noIS_L1536_L1920C'
    ]

    with open(fpath, "r") as f:
        image_paths = [line for line in f.read().splitlines() if line.strip()]

    app = QApplication(sys.argv)
    gallery = MaskComparisonGallery(image_paths, hypotheses)
    gallery.resize(800, 600)
    gallery.show()
    sys.exit(app.exec_())
