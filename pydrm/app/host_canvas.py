import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

class HostCanvas(tk.Canvas):
    def __init__(self, master, width, height):
        '''Initializes a blank canvas with a host image'''
        self.width = width
        self.height = height
        tk.Canvas.__init__(self, 
                           master=master,
                           width=width, 
                           height=height,
                           bg='white')
        self.menu_save = tk.Menu(self, tearoff=0)
        self.menu_save.add_command(label="Save", command=self.save_canvas_content)
        self.set_init_configuration()

    def set_init_configuration(self):
        '''Resets the host image'''
        self.delete('all')
        self.host_image = self.create_image(0, 0, anchor='nw', image=None)
        self.disable()
        
    def on_import_setup(self):
        self.enable()
        self.bind("<Button-3>", self.save_popup)
        
    def on_close_setup(self):
        self.unbind("<Button-3>")
        self.disable()
        
    def show_rgba(self, array, mode='RGBA', aspect_ratio=None):
        '''Displays input array as host image'''           
        array = Image.fromarray(array, mode=mode)
        if aspect_ratio:
            array = array.resize(aspect_ratio)
        else:
            array = array.resize((self.width, self.height))
        self.arrayIm = array # for image saving (temp)
        self.array = ImageTk.PhotoImage(array)
        self.itemconfig(self.host_image, image=self.array)
        
    def disable(self):
        self.configure(state=tk.DISABLED)
        
    def enable(self):
        self.configure(state=tk.NORMAL)

    def save_popup(self, event):
        try:
            self.menu_save.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu_save.grab_release()
    
    def save_canvas_content(self):
        try:
            with filedialog.asksaveasfile(mode='w', defaultextension='.png') as f:
                self.arrayIm.save(f.name)
        except:
            pass
            
    