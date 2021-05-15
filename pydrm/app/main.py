'''
Tkinter app intended for visualizing and manipulating DRM datasets.
'''
import os
import pathlib
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from .window_importer import ImporterWindow
from .window_help import HelpWindow
from .host_canvas import HostCanvas
from .canvas_fixDRP import FixDRPCan
from .canvas_movingDRP import MovDRPCan
from .canvas_microstructure import MicroDisplayCan
from .frame_mapping import MappingFrame
from .frame_resize import ResizeFrame
from .frame_scalers import FrameScalers

from .frame_segmentation import SegmentationFrame

class AppAnalysis(tk.Frame):
    '''
    Tkinter application for DRM analysis. Work in progress!
    '''
    def __init__(self, master=None, size=500):
        '''
        Instantiates the GUI application.
        '''
        # Initialize super() class --------------------------------------------
        tk.Frame.__init__(self, master)
        
        # Set window icon :)
        master.iconbitmap(
            os.path.join(pathlib.Path(__file__).parent.absolute(), 
                         'static/drm_icon.ico'))
        
        # Class variables -----------------------------------------------------
        self.master = master # The root Tk window
        self.ws = size # Determines widgets size
        
        # Menu ----------------------------------------------------------------
        menu = tk.Menu(self.master)
        self.master.configure(menu=menu)
        
        self.menufile = tk.Menu(menu, tearoff=0)
        self.menufile.add_command(label='Open dataset', 
                                  command=self.open_file)
        self.menufile.add_command(label='Import images', 
                                  command=self.launch_importer)
        self.menufile.add_command(label='Close dataset', 
                                  command=self.on_close_setup,
                                  state=tk.DISABLED)
        self.menufile.add_command(label='Save matrix', 
                                  command=self.launch_export, 
                                  state=tk.DISABLED)
        menu.add_cascade(label='File', menu=self.menufile)
        
        self.menutools = tk.Menu(menu, tearoff=0)
        self.menutools.add_command(label='Crop field of view',
                                   command=lambda:self.set_flag_release(1), 
                                   state=tk.DISABLED)
        menu.add_cascade(label='Tools', menu=self.menutools)
        
        menuhelp = tk.Menu(menu, tearoff=0)
        menuhelp.add_command(label='Documentation', 
                             command=self.launch_helpdoc)
        menu.add_cascade(label='Help', menu=menuhelp)
        
        # Global padx and pady ------------------------------------------------
        
        gpadx = 5
        gpady = 5
        
        # Progress bar --------------------------------------------------------
        self.pbar = ttk.Progressbar(self.master, orient=tk.HORIZONTAL,
                                    length=self.ws//4, mode='determinate')
        self.pbar.grid(row=0, column=1, sticky='n', padx=gpadx, pady=gpady)
        
        # Logging -------------------------------------------------------------
        self.tex = tk.Text(self.master, height=1, width=56)
        self.tex.grid(row=0, column=0, sticky='n', padx=gpadx, pady=gpady)

        # Canvas 0 (main display canvas) --------------------------------------
        self.canvas0 = MicroDisplayCan(
            self.master, self.ws//2+50, self.ws//2+50, self)
        self.canvas0.grid(row=1, column=0, rowspan=2, padx=gpadx, pady=gpady)
        
        # Canvas 1 (dynamic canvas) -------------------------------------------
        self.canvas1 = HostCanvas(
            self.master, width=self.ws//4, height=self.ws//4)
        self.canvas1.grid(row=1, column=1, padx=gpadx, pady=gpady)
        
        # Notebook canvas (static display) ------------------------------------
        self.nbk = ttk.Notebook(self.master)
        w = h = self.ws//4
        self.canvas2 = HostCanvas(self.nbk, w, h)
        self.nbk.add(self.canvas2, text='R-Pattern')
        self.canvas3 = HostCanvas(self.nbk, w, h)
        self.nbk.add(self.canvas3, text='Training')
        self.nbk.grid(row=2, column=1, padx=gpadx, pady=gpady)
        
        self.canvas4 = HostCanvas(self.nbk, w, h)
        self.nbk.add(self.canvas4, text='Grain size')
        
        # Live DRP ------------------------------------------------------------
        self.frame_movdrp = MovDRPCan(self.master, self.canvas1, self)
        
        # DRP fixed view ------------------------------------------------------       
        self.frame_fixdrp = FixDRPCan(
            self.master, self.ws, self.canvas2, self.canvas0, self)
        
        # Frame scalers -------------------------------------------------------
        self.frame_scalers = FrameScalers(
            self.master, self.ws//2, 100, self.canvas0, self.canvas1, self)
        self.frame_scalers.grid(row=3, column=0, padx=gpadx, pady=gpady)
        
        # Command pannel notebook ---------------------------------------------
        self.cmd_nbk = ttk.Notebook(self.master)
        w = self.ws//3
        h = self.ws//2+55
        self.frame_resize = ResizeFrame(
            self.cmd_nbk, w, h, self.canvas0, self.pbar, self)
        self.cmd_nbk.add(self.frame_resize, text='Downscaling')
        self.frame_segment = SegmentationFrame(
            self.cmd_nbk, w, h, self.canvas0, self.canvas4, self.pbar, self)
        self.cmd_nbk.add(self.frame_segment, text='Segmentation')
        self.frame_mapping = MappingFrame(
            self.cmd_nbk, w, h, self.canvas0, self.canvas3, self.pbar, self)
        self.cmd_nbk.add(self.frame_mapping, text='Machine Learning')
        self.cmd_nbk.grid(
            row=0, column=2, rowspan=3, sticky='ne', padx=gpadx, pady=gpady)
        
        ### SET GLOBAL INIT CONFIGURATION -------------------------------------
        self.set_init_configuration()
    
    def set_init_configuration(self):
        '''
        Sets initial GUI configuration.
        '''
        self.flag_release = 0 # flag_release: codes what to do when click-and-dragging.
        
        # Reset all widgets
        self.frame_resize.set_init_configuration()
        self.canvas0.set_init_configuration()
        self.frame_segment.set_init_configuration()
        self.frame_fixdrp.set_init_configuration()
        self.frame_movdrp.set_init_configuration()
        self.frame_scalers.set_init_configuration()
        self.frame_mapping.set_init_configuration()
        
        # Disable menu items
        self.menutools.entryconfig(0, state=tk.DISABLED)
        self.menufile.entryconfig(2, state=tk.DISABLED)
        self.menufile.entryconfig(3, state=tk.DISABLED)
        
        self.log_message('> Open a file to get started.')
        
    def on_import_setup(self, data_origin):
        '''
        On import setup.
        '''
        # Update data and origin data
        self.data_origin = data_origin
        self.rx, self.ry, self.s0, self.s1 = self.data_origin.shape
        self.rx_origin, self.ry_origin = self.rx, self.ry

        # Setup every widget on import
        self.frame_resize.on_import_setup()
        self.canvas0.on_import_setup()
        self.frame_segment.on_import_setup()
        self.frame_fixdrp.on_import_setup()
        self.frame_movdrp.on_import_setup()
        self.frame_scalers.on_import_setup()
        self.frame_mapping.on_import_setup()
        
        # Enable menu items
        self.menutools.entryconfig(0,state=tk.NORMAL)
        self.menutools.entryconfig(1,state=tk.NORMAL)
        self.menufile.entryconfig(2,state=tk.NORMAL)
        self.menufile.entryconfig(3,state=tk.NORMAL)
        
        self.log_message('> Loaded dataset!')
        
    def on_close_setup(self):
        '''
        On close setup.
        '''
        self.flag_release = 0 # flag_release: codes what to do when click-and-dragging.
        
        # Reset all widgets
        self.frame_resize.on_close_setup()
        self.canvas0.on_close_setup()
        self.frame_segment.on_close_setup()
        self.frame_fixdrp.on_close_setup()
        self.frame_movdrp.on_close_setup()
        self.frame_scalers.on_close_setup()
        self.frame_mapping.on_close_setup()

        # Disable menu items
        self.menutools.entryconfig(0, state=tk.DISABLED)
        self.menufile.entryconfig(2, state=tk.DISABLED)
        self.menufile.entryconfig(3, state=tk.DISABLED)
        
        # Delete loaded data (sketchy - rx doesn't even appear here!)
        del self.data, self.data_origin, self.rx, self.ry, self.xmax, self.ymax, self.sc
        
        self.log_message('> Open a file to get started.')
    
    # -------------------------------------------------------------------------
    # LOGGING -----------------------------------------------------------------
    def log_message(self, m):
        '''
        Displays a message in the text widget.
        '''
        self.tex.config(state=tk.NORMAL)
        self.tex.delete(1.0,'end')
        self.tex.insert(1.0, m)
        self.tex.config(state=tk.DISABLED)
    
    # LAUNCHERS ---------------------------------------------------------------
    def launch_importer(self):
        ImporterWindow(self.master, master_object=self)
        
    def launch_export(self):
        with filedialog.asksaveasfile(mode='w', defaultextension='.npy') as f:
            np.save(f.name, self.data)
        self.log_message('> Data saved!')
    
    def launch_helpdoc(self):
        HelpWindow(self)

    # DATA IMPORTS, OPEN AND CLOSE MANAGEMENT ---------------------------------
    def open_file(self):
        '''
        Opens file dialog, loads NPY data.
        '''
        filename = filedialog.askopenfilename(
                   initialdir=os.path.dirname(os.path.abspath(__file__)), 
                   title = "Select file",
                   filetypes = [('data files', '.npy .mat')])
        ext = os.path.splitext(os.path.abspath(filename))[1]
        if ext=='.npy':
            try:
                data = np.load(filename)
            except:
                self.log_message(f'> Error in: loading {ext} file.')
                return 0
        else:
            self.log_message('> Error: File is not NPY.')
            return 0
        
        self.on_import_setup(data)
    
    def set_flag_release(self, val):
        '''Changes action taken when click-and-dragging on the canvas.'''
        self.flag_release = val
        self.lift()
        
    def manage_button_release(self, sx, sy, ex, ey):
        '''Action taken when click-and-dragging on the canvas.'''
        
        data_slice = self.data[sy:ey,sx:ex]
        print('Data slice: ', data_slice.shape)
        
        if self.flag_release==1: # Crop field of view
            self.canvas0.delete_rectangle()
            self.on_import_setup(data_slice)
            self.flag_release = 0
            
        elif self.flag_release==2: # Select class C1
            self.frame_mapping.select_C1(data_slice, sx, sy, ex, ey)
        
        elif self.flag_release==3: # Select class C2
            self.frame_mapping.select_C2(data_slice, sx, sy, ex, ey)
        
        else: # Segment field of view
            self.frame_segment.update_starters(sx, sy)
            self.frame_segment.segmentation_pipeline(data_slice)
            # self.flag_release = 0