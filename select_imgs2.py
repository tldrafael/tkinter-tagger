import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QScrollArea, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

# Import Pillow libraries for image processing.
from PIL import Image, ImageChops, ImageQt
import numpy as np


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


# A composite widget that shows:
# - the main image
# - the mask image
# - the blended image (multiplication of main image and mask)
class ClickableImageWithMaskAndBlend(QWidget):
    clicked = pyqtSignal()

    size = 384
    def __init__(self, img_path, mask_path, size=(size,size)):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.size = size
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.img_label = ClickableLabel()
        pixmap = QPixmap(self.img_path)
        pixmap = pixmap.scaled(self.size[0], self.size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap = pixmap.scaled(self.size[0], self.size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pixmap)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(pixmap.size())

        self.img_label.clicked.connect(self.clicked)
        layout.addWidget(self.img_label)

        self.setLayout(layout)
        total_width = self.img_label.width()
        max_height = self.img_label.height()
        self.setFixedSize(total_width, max_height)

        # Default border style.
        self.setStyleSheet("border: 1px solid gray;")


# The main gallery that arranges these composite widgets in a scrollable grid.
class ImageGallery(QWidget):
    def __init__(self, image_mask_pairs):
        super().__init__()
        self.selected_images = set()  # Store indices of selected widgets.
        self.image_mask_pairs = image_mask_pairs  # List of tuples: (img_path, mask_path)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area_widget = QWidget()
        grid = QGridLayout(scroll_area_widget)
        num_columns = 4  # Adjust the number of columns as needed.
        row, col = 0, 0

        for idx, (img_path, mask_path) in enumerate(self.image_mask_pairs):
            widget = ClickableImageWithMaskAndBlend(img_path, mask_path)
            # When the composite widget is clicked, toggle its selection.
            widget.clicked.connect(lambda checked=False, idx=idx, widget=widget: self.toggle_selection(idx, widget))
            grid.addWidget(widget, row, col)
            col += 1
            if col >= num_columns:
                col = 0
                row += 1

        scroll_area.setWidget(scroll_area_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        self.setLayout(layout)

    def toggle_selection(self, idx, widget):
        # Toggle selection status.
        if idx in self.selected_images:
            self.selected_images.remove(idx)
            widget.setStyleSheet("border: 1px solid gray;")
        else:
            self.selected_images.add(idx)
            widget.setStyleSheet("border: 5px solid blue;")
        print("Selected image indices:", self.selected_images)
        self.save_selection()

    def save_selection(self):
        # Write the main image paths of the selected units to out.txt.
        with open("out.txt", "w") as f:
            for idx in self.selected_images:
                img_path, _ = self.image_mask_pairs[idx]
                f.write(img_path + "\n")
        print("Saved selected image paths to out.txt")


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
