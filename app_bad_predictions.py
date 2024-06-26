import tkinter as tk
from PIL import ImageTk, Image
from glob import iglob
import numpy as np
import re
import sys
import os
import pandas as pd


def puthash_hexstring(x):
    return re.sub('^0x', '#', x[0])


class App:
    def __init__(self, master, impaths):
        self.impaths = impaths
        self.predpaths = [p.replace('/im/', '/triton_refinerPP/').replace('.jpg', '.png') for p in impaths]
        self.id_ = 0
        self.register = 'labels.txt'
        self.preworked = self.read_register()

        self.master = master
        self.labels = [['design', 'design-problem', 'no-design', 'thrash']]
        # self.labels = [['NO', 'SOFT', 'HARD', 'DISCARD']]
        self.run_photo()

    def load_photo(self):
        im = Image.open(self.impaths[self.id_])
        pred = Image.open(self.predpaths[self.id_])
        blend = np.array(im) / 255 * np.array(pred)[..., None]/255
        # blend += 1. - np.array(pred)[..., None] / 255
        blend = Image.fromarray((blend * 255).astype(np.uint8))

        pics = [im, pred, blend]
        pics = [x.resize((450, 450), 2) for x in pics]

        widths, heights = zip(*(i.size for i in pics))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for pic in pics:
          new_im.paste(pic, (x_offset, 0))
          x_offset += pic.size[0]

        return ImageTk.PhotoImage(new_im)

    def run_photo(self):
        panel = tk.Label(self.master)
        while self.impaths[self.id_] in self.preworked:
            self.id_ += 1
        panel.im = self.load_photo()
        panel.config(image=panel.im)

        panel.grid(row=1, column=1, rowspan=6)
        for i in range(1):
            for j in range(len(self.labels[0])):
                self.create_button(i, j, self.labels[i][j])

    def create_button(self, i, j, l):
        action = self.button_action(l)
        btn = tk.Button(self.master, text=l, command=action, width=4, height=2)
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
            f.write('{},{}\n'.format(self.impaths[self.id_], l))

    def button_action(self, l):
        def actions(*args, **kwargs):
            self.write_label(l)
            self.id_ += 1
            if self.id_ < len(self.impaths):
                self.run_photo()
            else:
                self.master.quit()
        return actions


def launchButtonApp(impaths):
    root = tk.Tk()
    App(root, impaths)
    tk.mainloop()


if __name__ == '__main__':
    fpath = 'a.txt'
    with open(fpath, 'r') as f:
        impaths = f.read().split('\n')

    launchButtonApp(impaths)
