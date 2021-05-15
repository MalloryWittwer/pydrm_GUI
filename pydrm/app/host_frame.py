from tkinter import Frame

class HostF(Frame):
    def __init__(self, width, height):
        Frame.__init__(self, width=width, height=height)
        self.width = width
        self.height = height
        self.grid_propagate(0)