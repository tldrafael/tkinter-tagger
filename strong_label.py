import tkinter as tk
from PIL import ImageTk, Image, ImageDraw, ImageFont
import numpy as np
import re
import sys
import os
import cv2


def remove_small_regions(mask, min_area, max_area=None, connectivity=8):
    if max_area is None:
        max_area = mask.shape[0]*mask.shape[1] + 1

    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )

    # Create output mask
    cleaned = np.zeros_like(mask)

    # stats is an array of shape (num_labels, 5):
    # columns: [CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA] # noqa
    nblobs = 0
    gaps_area = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area and area <= max_area:
            cleaned[labels == label] = 255
            nblobs += 1
            gaps_area.append(area)

    return cleaned, nblobs, gaps_area


def get_gtpath(p):
    pgt = p.replace('/im/', '/gt/').replace('.jpg', '.png')
    return pgt


def cv2_imread(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {p}")
    return img


def compute_number_of_gaps(p=None, pgt=None, gt=None):
    assert p is not None or pgt is not None

    if pgt is None and gt is None:
        pgt = get_gtpath(p)

    if gt is None:
        gt = cv2_imread(pgt)

    gt_thrs_inv = gt.copy()
    thrs = 10
    gt_thrs_inv[gt < thrs] = 255
    gt_thrs_inv[gt >= thrs] = 0

    max_area = int(.25*gt.shape[0]*gt.shape[1])
    # max_area=15000
    diff2, nblobs, sizes = remove_small_regions(
        gt_thrs_inv, min_area=1, max_area=max_area)

    #import matplotlib.pyplot as plt
    #plt.imshow(diff2); plt.show()
    return nblobs


def puthash_hexstring(x):
    return re.sub('^0x', '#', x[0])


class App:
    def __init__(self, master, imgpaths):
        self.imgpaths = imgpaths
        self.gtpaths = [p.replace('/im/', '/gt/').replace('.jpg', '.png') for p in imgpaths]

        self.predpaths = [p.replace('/gt/', '/results/2025_0910-tritonserver/') for p in self.gtpaths]
        # self.gtpaths2 = [p.replace('/gt/', '/results/20250616-refiner-gapmaps+PRODCKPT-from-20250911-expert-hanginghand-afterstg2-sz1k4mixed22k-gmap025PP/') for p in self.gtpaths]
        self.gtpaths = self.predpaths
        self.predpaths = self.gtpaths2

        self.id_ = 0
        self.register = 'labels.txt'
        self.preworked = self.read_register()

        self.master = master
        # self.labels = [['GT better', 'Pred better', 'Both good', 'Both bad', 'odd', 'other-class']]
        self.labels = [['general better', 'expert better', 'Both good', 'Both bad', 'odd', 'other-class']]
        self.run_photo()

    def load_photo(self):
        p = self.imgpaths[self.id_]
        pgt = self.gtpaths[self.id_]
        ppred = self.predpaths[self.id_]
        print(p)

        size = 480
        img = Image.open(p)
        img = img.resize((size, size), 2)

        gt = Image.open(pgt)
        gt = gt.resize((size, size), 2)
        green = np.array([[[0.,1.,0.]]])
        gt2 = np.array(gt)[..., None]/255
        img_fg = (1-gt2)*green
        img_fg += np.array(img)/255*gt2
        img_fg = (img_fg * 255).clip(0, 255).astype(np.uint8)
        img_fg = Image.fromarray(img_fg)

        pred = Image.open(ppred)
        pred = pred.resize((size, size), 2)
        pred2 = np.array(pred)[..., None]/255
        img_fg2 = (1-pred2)*green
        img_fg2 += np.array(img)/255*pred2
        img_fg2 = (img_fg2 * 255).clip(0, 255).astype(np.uint8)
        img_fg2 = Image.fromarray(img_fg2)



        img1 = Image.open(p)
        img1 = img1.resize((2*size, 2*size), 2)

        diff = (gt2 - pred2).__abs__().sum()
        diff = 100*diff / (gt2.shape[0] * gt2.shape[1] * gt2.shape[2])
        # print(f'Diff-mag: {diff:.2f}%')
        gt_nblobs = compute_number_of_gaps(pgt=pgt)
        # print(f'GT number of blobs: {gt_nblobs}')
        pred_nblobs = compute_number_of_gaps(pgt=ppred)
        # print(f'Pred number of blobs: {pred_nblobs}')
        diff_nblobs = np.abs(gt_nblobs - pred_nblobs)
        # print(f'Diff-blobs: {diff_nblobs}\n')

        text = f'Diff-mag: {diff:.2f}%\nDiff-blobs: {diff_nblobs}'
        draw = ImageDraw.Draw(img1)
        fontpath = '/usr/share/fonts/truetype/ubuntu/Ubuntu-C.ttf'
        font = ImageFont.truetype(fontpath, size=40)  # You can use any .ttf font
        position = (10, 750)  # X, Y coordinates
        color = (255, 0, 255)  # White
        draw.text(position, text, fill=color, font=font)


        im_total = Image.new('RGB', (4*size, 2*size))
        im_total.paste(img1, (0, 0))
        im_total.paste(gt, (2*size, 0))
        im_total.paste(img_fg, (3*size, 0))
        im_total.paste(pred, (2*size, size))
        im_total.paste(img_fg2, (3*size, size))

        return ImageTk.PhotoImage(im_total)

    def run_photo(self):
        panel = tk.Label(self.master)
        while self.imgpaths[self.id_] in self.preworked:
            self.id_ += 1
        panel.img = self.load_photo()
        bname = os.path.basename(self.imgpaths[self.id_]).split('+')[0]
        panel.config(
            image=panel.img, text=bname, compound='bottom',
            font=("Helvetica", 20),
        )

        panel.grid(row=0, column=1, rowspan=1, columnspan=17)
        for i in range(len(self.labels[0])):
            for j in range(1):
                self.create_button(i, j, self.labels[j][i])

    def create_button(self, i, j, l):
        action = self.button_action(l)
        btn = tk.Button(self.master, text=l, command=action, width=14, height=2)
        btn.grid(row=j + 1, column=i + 2)
        self.master.bind_all(str(i+1), action)
        return btn

    def read_register(self):
        if os.path.exists(self.register):
            with open(self.register, 'r') as f:
                lines = f.read().split('\n')
            return [l.split(',')[0] for l in lines][:-1]
        else:
            return []

    def write_label(self, l):
        with open(self.register, 'a') as f:
            f.write('{},{}\n'.format(self.imgpaths[self.id_], l))

    def button_action(self, l):
        def actions(*args, **kwargs):
            self.write_label(l)
            self.id_ += 1
            if self.id_ < len(self.imgpaths):
                self.run_photo()
            else:
                self.master.quit()
        return actions


def launchButtonApp(imgpaths):
    root = tk.Tk()
    App(root, imgpaths)
    tk.mainloop()


if __name__ == '__main__':
    # imgdir = sys.argv[1]
    # imgpaths = list(iglob(imgdir + '/*'))
    with open(sys.argv[1], 'r') as f:
        imgpaths = f.read().split('\n')

    imgpaths = imgpaths[:]
    launchButtonApp(imgpaths)
