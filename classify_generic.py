import tkinter as tk
from PIL import ImageTk, Image
from glob import iglob
import numpy as np
import re
import sys
import os
from time import sleep
import argparse


# Default labels - can be overridden via command line
DEFAULT_LABELS = ['good', 'bad', 'skip']


def puthash_hexstring(x):
    return re.sub('^0x', '#', x[0])


class App:
    def __init__(self, master, imgpaths, labels, register_file='labels.txt', max_size=800):
        self.imgpaths = imgpaths
        self.id_ = 0
        self.register = register_file
        self.preworked = self.read_register()
        self.max_size = max_size  # Maximum size for image display

        self.master = master
        self.labels = labels

        # Configure grid weights for proper resizing
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        # Create main frame
        self.main_frame = tk.Frame(master)
        self.main_frame.grid(row=0, column=0, sticky='nsew')
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Create image panel
        self.panel = tk.Label(self.main_frame)
        self.panel.grid(row=0, column=0, padx=10, pady=10)

        # Create button frame below image
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, pady=10)

        self.run_photo()

    def load_photo(self):
        p = self.imgpaths[self.id_]
        img = Image.open(p)
        original_size = img.size
        print(f"{p} - Original size: {original_size}")

        # Scale image to fit within max_size while preserving aspect ratio
        img = self.fit_image(img, self.max_size)

        return ImageTk.PhotoImage(img)

    def fit_image(self, img, max_size):
        """Scale image to fit within max_size while preserving aspect ratio."""
        width, height = img.size

        # Calculate scaling factor to fit within max_size
        scale = min(max_size / width, max_size / height)

        if scale < 1:  # Only scale down, never up
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        return img

    def run_photo(self):
        # Skip already processed images
        while self.id_ < len(self.imgpaths) and self.imgpaths[self.id_] in self.preworked:
            self.id_ += 1

        if self.id_ >= len(self.imgpaths):
            print("All images processed!")
            self.master.quit()
            return

        # Update image
        self.panel.img = self.load_photo()
        self.panel.config(image=self.panel.img)

        # Update window title with progress
        self.master.title(f"Image {self.id_ + 1}/{len(self.imgpaths)}")

        # Clear and recreate buttons
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        # Create buttons in a horizontal row below the image
        for j, label in enumerate(self.labels):
            self.create_button(j, label)

    def get_shortcut_key(self, index):
        """Get shortcut key for a given index: 1-9 for first 9, then a-z."""
        if index < 9:
            return str(index + 1)  # 1-9
        else:
            # a-z for indices 9-34 (26 letters)
            letter_index = index - 9
            if letter_index < 26:
                return chr(ord('a') + letter_index)
        return None

    def create_button(self, j, label):
        action = self.button_action(label)
        # Show shortcut at the left of the label text
        shortcut = self.get_shortcut_key(j)
        button_text = f"{shortcut}. {label}" if shortcut else label
        btn = tk.Button(self.button_frame, text=button_text, command=action, width=14, height=2)
        btn.grid(row=0, column=j, padx=5, pady=5)

        # Bind shortcut key
        if shortcut:
            self.master.bind_all(shortcut, action)
        return btn

    def read_register(self):
        if os.path.exists(self.register):
            with open(self.register, 'r') as f:
                lines = f.read().split('\n')
            return [l.split(',')[0] for l in lines if l]
        else:
            return []

    def write_label(self, label):
        with open(self.register, 'a') as f:
            f.write('{},{}\n'.format(self.imgpaths[self.id_], label))

    def button_action(self, label):
        def actions(*args, **kwargs):
            self.write_label(label)
            print(f"[{self.id_ + 1}/{len(self.imgpaths)}] label: {label}")
            sleep(.15)
            self.id_ += 1
            if self.id_ < len(self.imgpaths):
                self.run_photo()
            else:
                print("All images processed!")
                self.master.quit()
        return actions


def launchButtonApp(imgpaths, labels, register_file='labels.txt', max_size=800):
    root = tk.Tk()
    root.title("Image Classifier")
    App(root, imgpaths, labels, register_file, max_size)
    tk.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description='Generic image classification tool')
    parser.add_argument('imglist', help='Path to file containing image paths (one per line)')
    parser.add_argument('--labels', '-l', nargs='+', default=DEFAULT_LABELS,
                        help='Labels for classification (default: good bad skip)')
    parser.add_argument('--output', '-o', default='labels.txt',
                        help='Output file for labels (default: labels.txt)')
    parser.add_argument('--max-size', '-s', type=int, default=800,
                        help='Maximum image size in pixels (default: 800)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.imglist, 'r') as f:
        imgpaths = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(imgpaths)} images")
    print(f"Labels: {args.labels}")
    print(f"Output file: {args.output}")
    print(f"Max image size: {args.max_size}px")
    print("Use keys 1-9 and a-z for quick selection")

    launchButtonApp(imgpaths, args.labels, args.output, args.max_size)
