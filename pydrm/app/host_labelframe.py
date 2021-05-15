from tkinter import ttk

class HostLF(ttk.LabelFrame):
    def __init__(self, width, height, text=''):
        ttk.LabelFrame.__init__(self, width=width, height=height, text=text)
        self.width = width
        self.height = height
        self.grid_propagate(0)
        