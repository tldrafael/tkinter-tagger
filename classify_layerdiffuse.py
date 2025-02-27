import tkinter as tk
from PIL import ImageTk, Image
from glob import iglob
import numpy as np
import re
import sys
import os
from time import sleep


def puthash_hexstring(x):
    return re.sub('^0x', '#', x[0])


class App:
    def __init__(self, master, imgpaths):
        self.imgpaths = imgpaths
        self.maskpaths = [p.replace('/im/', '/gt/').replace('.jpg', '.png') for p in imgpaths]
        self.predpaths = [p.replace('/gt/', '/results/20240822_0026_dinolarge+378px/') for p in self.maskpaths]
        self.id_ = 0
        self.register = 'labels.txt'
        self.preworked = self.read_register()

        self.master = master
        # self.labels = [['good-and-transp', 'good-and-notransp', 'bad', 'ok']]
        # self.labels = [['verygood-and-transp', 'good-and-transp', 'good-and-notransp', 'bad', 'ok']]
        # self.labels = [['perfect-mask', 'almost-perfect-mask', 'perfect-mask-but-hallucination', 'almost-perfect-mask-but-hallucination', 'good-but-notransp', 'bad', 'bad-or-others']]
        # self.labels = [['opaque-good', 'opaque-bad', 'transp-good', 'transp-bad']]
        # self.labels = [['opaque-good', 'opaque-bad', 'nonopaque-good', 'nonopaque-bad']]
        self.labels = [['perfect', 'good-but-shadows', 'good-but-notransp', 'bad', 'interior-no-transp', 'interior']]

        self.run_photo()

    def load_photo(self):
        p = self.imgpaths[self.id_]
        img = Image.open(p)
        print(p, np.array(img).shape)

        # fl_pred = False
        fl_pred = True

        if fl_pred:
            res = 256*3//2
        else:
            res = 256*2

        img = img.resize((res, res), 2)
        gt = Image.open(self.maskpaths[self.id_])
        gt = gt.resize((res, res), 2)
        img_fg = np.array(gt)[..., None] / 255 * np.array(img) / 255
        img_fg = (img_fg * 255).clip(0, 255).astype(np.uint8)
        img_fg = Image.fromarray(img_fg)

        if fl_pred:
            pred = Image.open(self.predpaths[self.id_])
            pred = pred.resize((res, res), 2)
            im_total = Image.new('RGB', (4 * res, res))
        else:
            im_total = Image.new('RGB', (3 * res, res))

        im_total.paste(img, (0, 0))
        im_total.paste(gt, (res, 0))
        im_total.paste(img_fg, (2 * res, 0))

        if fl_pred:
            im_total.paste(pred, (3 * res, 0))

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

        btn = tk.Button(self.master, text='get-back', command=self.get_back, width=28, height=2)
        btn.grid(row=j+2, column=i+2)
        self.master.bind_all(str(j+2), self.get_back)

    def create_button(self, i, j, l):
        action = self.button_action(l)
        btn = tk.Button(self.master, text=l, command=action, width=28, height=2)
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
            sleep(.25)
            if True:
            # if l in ['next', 'none']:
                self.id_ += 1
                if self.id_ < len(self.imgpaths):
                    self.run_photo()
                else:
                    self.master.quit()
        return actions

    def get_back(self, *args, **kwargs):
        with open(self.register, 'r') as f:
            lines = f.read().split('\n')
            if len(lines[-1]) == 0:
                lines = lines[:-1]
            lines = lines[:-1]
            lines = '\n'.join(lines) + '\n'

        with open(self.register, 'w') as f:
            f.write(lines)

        self.id_ -= 1
        self.run_photo()

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
