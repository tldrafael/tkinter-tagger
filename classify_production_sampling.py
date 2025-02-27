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
        self.maskpaths = [p.replace('/im/', '/results/20240515_2209_refiner+tune-hangers-and-mannequins+tuneOnlyHangers_tmp/').replace('.jpg', '.png') for p in imgpaths]
        self.maskpaths2 = [p.replace('/im/', '/results/20240515_2209_refiner+tune-hangers-and-mannequins+tuneOnlyHangers_tmpPP/').replace('.jpg', '.png') for p in imgpaths]
        self.id_ = 0
        self.register = 'labels.txt'
        self.preworked = self.read_register()

        self.master = master
        self.labels = [[
            'skip',
            'partial-dummy-mannequin',
            'partial-human-mannequin',
            'dummy-mannequin-fail',
            'dummy-mannequin-fail-support',
            'dummy-mannequin-upper-support',
            'garment-closeup',
            'card-missing-letters',
            'card-letter-precision',
            'bad-shadows',
            'GS-non-opaqueness',
            'GS-missing-elements',
            'GS-missing-surround',
            'bad-boundaries',
            'thing-transparency',
            'hanger-missing',
            'hanger-incomplete',
            'hanger-with-hanger',
            'human-in-context',
            'screenshot-frame',
            'drawing',
            'thing-with-a-rule',
            'hard-sample',
            'hanged-sod',
            'laid-sod',
            'missing-detail',
            ]]

        self.labels2 = [[
            f'CORRECT-{x}' for x in self.labels[0]]]

        self.run_photo()

    def load_photo(self):
        p = self.imgpaths[self.id_]
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        res = 2*512
        pred1 = cv2.imread(self.maskpaths[self.id_])
        pred2 = cv2.imread(self.maskpaths2[self.id_])
        blend = im/255*pred2/255
        blend = (blend*255).round().clip(0,255).astype(np.uint8)
        total1 = np.concatenate([im, blend], axis=1)
        total2 = np.concatenate([pred1, pred2], axis=1)
        total = np.concatenate([total1, total2], axis=0)
        total = cv2.resize(total, (res,res))[..., ::-1]
        total = Image.fromarray(total)

        sad = (pred1/255 - pred2/255).__abs__().sum()
        sad /= 1000
        iou = ((pred1 > 128) * (pred2 > 128)).sum() / ((pred1 > 128) + (pred2 > 128)).sum()
        iou *= 100
        bp = os.path.basename(p)
        print(f'{bp}, sad: {sad:.0f}, iou: {iou:.0f}')

        return ImageTk.PhotoImage(total)

    def run_photo(self):
        panel = tk.Label(self.master)
        while self.imgpaths[self.id_] in self.preworked:
            self.id_ += 1
        panel.img = self.load_photo()
        panel.config(image=panel.img)

        panel.grid(row=1, column=1, rowspan=36, columnspan=1)
        for i in range(1):
            for j in range(len(self.labels[0])):
                self.create_button(i, j, self.labels[i][j])

        for j in range(len(self.labels2[0])):
            self.create_button(1, j, self.labels2[i][j])

    def create_button(self, i, j, l):
        action = self.button_action(l)
        btn = tk.Button(self.master, text=l, command=action, width=36, height=2)
        btn.grid(row=j + 1, column=i + 2)
        if i == 0:
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
            if True:
            # if l in ['next', 'none']:
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
