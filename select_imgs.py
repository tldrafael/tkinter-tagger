import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QScrollArea, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

# Import Pillow libraries for image processing.
from PIL import Image, ImageChops, ImageQt
import numpy as np
import os


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

    # size = int(128*4.85) # fit the screen
    size = int(128*4)
    def __init__(self, img_path, mask_path, size=(size,size)):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.size = size
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main image label
        self.img_label = ClickableLabel()
        pixmap = QPixmap(self.img_path)
        pixmap = pixmap.scaled(self.size[0], self.size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap = pixmap.scaled(self.size[0], self.size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pixmap)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(pixmap.size())

        # Mask image label
        self.mask_label = ClickableLabel()
        mask_pixmap = QPixmap(self.mask_path)
        mask_pixmap = mask_pixmap.scaled(self.size[0], self.size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.mask_label.setPixmap(mask_pixmap)
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setFixedSize(pixmap.size())

        use_blend = True
        if use_blend:
            self.blend_label = ClickableLabel()
            # Load the images via Pillow, resize and compute the multiply blend.
            pil_img = Image.open(self.img_path).convert("RGBA")
            pil_mask = Image.open(self.mask_path).convert("RGBA")
            pil_mask = pil_mask.resize(pil_img.size, Image.LANCZOS)
            blended = make_mask_green(pil_img, pil_mask)
            # blended = ImageChops.multiply(pil_img, pil_mask)
            # qt_image = ImageQt.ImageQt(blended)
            # blend_pixmap = QPixmap.fromImage(qt_image)
            from PyQt5.QtGui import QImage
            w, h = blended.size
            raw = blended.tobytes("raw", "RGBA")
            qimg = QImage(raw, w, h, QImage.Format_RGBA8888)
            blend_pixmap = QPixmap.fromImage(qimg)
            blend_pixmap = blend_pixmap.scaled(self.size[0], self.size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.blend_label.setPixmap(blend_pixmap)
            self.blend_label.setAlignment(Qt.AlignCenter)
            self.blend_label.setFixedSize(pixmap.size())


        # Connect all labels' clicks to the composite widget's clicked signal.
        self.img_label.clicked.connect(self.clicked)
        layout.addWidget(self.img_label)

        self.mask_label.clicked.connect(self.clicked)
        layout.addWidget(self.mask_label)
        if use_blend:
            self.blend_label.clicked.connect(self.clicked)
            layout.addWidget(self.blend_label)

        self.setLayout(layout)
        if use_blend:
            total_width = self.img_label.width() + self.mask_label.width() + self.blend_label.width()
            max_height = max(self.img_label.height(), self.mask_label.height(), self.blend_label.height())
        else:
            total_width = self.img_label.width() + self.mask_label.width()
            max_height = max(self.img_label.height(), self.mask_label.height())

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
        num_columns = 1  # Adjust the number of columns as needed.
        row, col = 0, 0

        for idx, (img_path, mask_path) in enumerate(self.image_mask_pairs):
            name_bytes = os.path.basename(img_path).encode('utf-8')
            if len(name_bytes) > 254:
                print(f"Skipping {img_path} due to long name.")
                continue
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
        with open(fout, "w") as f:
            for idx in self.selected_images:
                img_path, _ = self.image_mask_pairs[idx]
                f.write(img_path + "\n")
        print("Saved selected image paths to out.txt")


import re
def get_gtpath(p):
    newp = re.sub(r'\.jpg$', '.png', p)
    newp = re.sub(r'\/im\/', '/gt/', newp)
    # newp = re.sub(r'\/im\/', '/results/20250812_production/', newp)
    newp = re.sub(r'\/im\/', '/results/20250616-refiner-gapmaps+PRODCKPT-from-20250905-expert-mannequin-afterstg2-sz3k6mixed22kPP/', newp)
    return newp


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Example list of (image, mask) pairs. Replace these paths with your actual files.
    fpath = sys.argv[1]
    with open(fpath, 'r') as f:
        images = f.read().split('\n')[:-1]

    i=int(sys.argv[2])
    fout = f'out.{i}.txt'
    if os.path.exists(fout):
        print(f"Output file {fout} already exists. Exiting to avoid overwriting.")
        sys.exit(0)
    step=1000
    images = images[i*step:(i+1)*step]
    image_mask_pairs = [(p, get_gtpath(p)) for p in images]

    gallery = ImageGallery(image_mask_pairs)
    gallery.resize(800, 600)
    gallery.show()
    sys.exit(app.exec_())
