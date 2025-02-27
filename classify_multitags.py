import tkinter as tk
from PIL import ImageTk, Image
from glob import iglob
import numpy as np
import re
import sys
import os


def puthash_hexstring(x):
    return re.sub('^0x', '#', x[0])


class App:
    def __init__(self, master, imgpaths):
        self.imgpaths = imgpaths
        self.maskpaths = [p.replace('/im/', '/gt/').replace('.jpg', '.png') for p in imgpaths]
        self.id_ = 0
        self.register = 'labels.txt'
        self.preworked = self.read_register()

        self.master = master
        self.labels = [[
            'next', 'fur', 'hair', 'sod', 'multiple', 'grid', 'boundary', 'GS', 'transp',
            'car-transp', 'scene', 'detailed', 'camouflage', 'hard', 'ambiguity', 'card',
            'tissue-grid', 'shadow', 'hangers', 'mannequins', 'non-opaque', 'chains',
            'opaqueness',
            'bad-label'
            ]]
        self.run_photo()

    def load_photo(self):
        p = self.imgpaths[self.id_]
        img = Image.open(p)
        print(p, np.array(img).shape)
        img = img.resize((450, 450), 2)
        gt = Image.open(self.maskpaths[self.id_])
        gt = gt.resize((450, 450), 2)
        img_fg = np.array(gt)[..., None] / 255 * np.array(img) / 255
        img_fg = (img_fg * 255).clip(0, 255).astype(np.uint8)
        img_fg = Image.fromarray(img_fg)
        im_total = Image.new('RGB', (3 * 450, 450))
        im_total.paste(img, (0, 0))
        im_total.paste(gt, (450, 0))
        im_total.paste(img_fg, (2 * 450, 0))
        return ImageTk.PhotoImage(im_total)

    def run_photo(self):
        panel = tk.Label(self.master)
        while self.imgpaths[self.id_] in self.preworked:
            self.id_ += 1
        panel.img = self.load_photo()
        panel.config(image=panel.img)

        panel.grid(row=1, column=1, rowspan=6)
        for i in range(1):
            for j in range(len(self.labels[0])):
                self.create_button(i, j, self.labels[i][j])

    def create_button(self, i, j, l):
        action = self.button_action(l)
        btn = tk.Button(self.master, text=l, command=action, width=12, height=2)
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
            if l in ['nok', 'next']:
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
