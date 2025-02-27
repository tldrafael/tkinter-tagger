import tkinter as tk
from PIL import ImageTk, Image
from glob import iglob
import numpy as np
import re
import sys
import os
import cv2


def puthash_hexstring(x):
    return re.sub('^0x', '#', x[0])


class App:
    def __init__(self, master, imgpaths):
        self.imgpaths = imgpaths
        self.maskpaths = [p.replace('/im/', '/gt/').replace('.jpg', '.png') for p in imgpaths]
        # self.maskpaths2 = [p.replace('/im/', '/results/20240613-dinoseg/').replace('.jpg', '.png') for p in imgpaths]
        self.id_ = 0
        self.register = 'labels.txt'
        self.preworked = self.read_register()

        self.master = master
        # self.labels = [['OPAQUE', 'NONOPAQUE', 'OTHER', 'BAD', 'OPAQUE-SAME']]
        self.labels = [['OPAQUE', 'NONOPAQUE', 'OTHER', 'BAD', 'LETTERS', 'LETTERS-NONPAQUE', 'LETTERS-OPAQUE', 'OPAQUE-SAME', 'ANIME', 'ANIME-BAD']]
        self.run_photo()

    def load_photo(self):
        p = self.imgpaths[self.id_]
        print(p)
        im = cv2.imread(p)[..., ::-1]
        gt = cv2.imread(self.maskpaths[self.id_])
        blend = (im/255*gt/255*255).astype(np.uint8)
        tmp = np.concatenate([im, gt, blend], axis=1)
        h, w = im.shape[:2]
        factor = 512*2/max(h,w)
        newh = int(factor*h+.5)
        neww = int(factor*w+.5)
        tmp = cv2.resize(tmp, (neww, newh))
        res = int(512*2)
        # final = tmp
        final = np.zeros((res,res,3)).astype(np.uint8)
        final[:newh,:neww] = tmp
        final = Image.fromarray(final)
        return ImageTk.PhotoImage(final)

    def run_photo(self):
        panel = tk.Label(self.master)
        while self.imgpaths[self.id_] in self.preworked:
            self.id_ += 1
        panel.img = self.load_photo()
        panel.config(image=panel.img)
        panel.config(text=os.path.basename(self.imgpaths[self.id_]), compound='top')

        panel.grid(row=1, column=1, rowspan=16)
        for i in range(1):
            for j in range(len(self.labels[0])):
                self.create_button(i, j, self.labels[i][j])

    def create_button(self, i, j, l):
        action = self.button_action(l)
        btn = tk.Button(self.master, text=l, command=action, width=20, height=2)
        btn.grid(row=j + 1, column=i + 2)
        self.master.bind_all(str(j+1), action)
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

    launchButtonApp(imgpaths)
