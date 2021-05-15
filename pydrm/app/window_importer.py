'''Class containing the importer toplevel window for the main app'''

import os
import re
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from pydrm.utils import show

class ImporterWindow(tk.Toplevel):
    '''
    Importing data and subtracting background.
    '''
    def __init__(self, master, width=200, height=300, master_object=None):
        tk.Toplevel.__init__(self, master, width=width, height=height)
        
        self.title('Import DRM data')
        self.resizable(False, False)
        
        self.master = master
        self.master_object = master_object
        
        self.img_chk = ImageTk.PhotoImage(Image.open(
            'pydrm/app/static/check_icon.png'))
        self.img_nop = ImageTk.PhotoImage(Image.open(
            'pydrm/app/static/orange_icon.png')) 
        
        padx = 10
        
        ### Progress bar
        self.pbar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=180, mode='determinate')
        self.pbar.grid(row=0, column=0, columnspan=2, padx=padx, pady=15, sticky='n')

        ### Igrey folder import
        btn_Igrey = ttk.Button(self, text='Sample', command=self.load_Igrey)
        btn_Igrey.grid(row=1, column=0, padx=padx, pady=5, sticky='we')
        
        # Little loading icon
        self.canvasIG = tk.Canvas(self, width=50, height=50)
        self.imoncanIG = self.canvasIG.create_image(2, 2, anchor='nw')
        self.canvasIG.itemconfig(self.imoncanIG, image=self.img_nop)
        self.canvasIG.grid(row=1, column=1, padx=padx, pady=5)
        
        ### Iback folder import (OPTIONAL)
        self.with_Iback = False
        btn_Iback = ttk.Button(self, text='Background', command=self.load_Iback)
        btn_Iback.grid(row=2, column=0, padx=padx, pady=5, sticky='we')
        
        # Little loading icon
        self.canvasIB = tk.Canvas(self, width=50, height=50)
        self.imoncanIB = self.canvasIB.create_image(2, 2, anchor='nw')
        self.canvasIB.itemconfig(self.imoncanIB, image=self.img_nop)
        self.canvasIB.grid(row=2, column=1, padx=padx, pady=5)
        
        ### Subtract background
        btn_load = ttk.Button(self, text='Process', command=self.process_data)
        btn_load.grid(row=3, column=0, columnspan=2, padx=padx, pady=15, sticky='we')
        
        self.image_types = ('*.jpg', '*.jpeg')
    
    def load_Igrey(self):
        '''
        Opens file dialog and loads images from Igrey folder.
        '''
        self.folderIgrey = filedialog.askdirectory(
            initialdir=os.path.dirname(os.path.abspath(__file__)), 
            title = "Select Igrey folder") + '/'
        
        grabbed_files = []
        for ext in self.image_types:
            grabbed_files.extend(glob.glob(os.path.join(self.folderIgrey, ext)))
        itIgrey = sorted(grabbed_files, key=lambda file:os.path.getctime(file))
        self.da_Igrey = self.glob_open(itIgrey)
        self.da_Iback = self.da_Igrey # just to avoid later problems
        
        phi_list = np.unique(self.da_Igrey.phi)
        phi_steps = phi_list[1] - phi_list[0]
        num_phi = int((self.da_Igrey.phi.max()-self.da_Igrey.phi.min())/phi_steps) + 1
        num_theta = int(len(self.da_Igrey.phi)/num_phi)
        
        self.shape = (num_phi, num_theta)
        
        self.canvasIG.itemconfig(self.imoncanIG, image=self.img_chk)
        self.lift()
    
    def load_Iback(self):
        '''
        Opens file dialog and loads images from Iback folder.
        '''
        self.with_Iback = True # Register that Iback is used.
        
        self.folderIback = filedialog.askdirectory(
            initialdir=os.path.dirname(os.path.abspath(__file__)), 
            title = "Select Iback folder") + '/'
        
        grabbed_files = []
        for ext in self.image_types:
            grabbed_files.extend(glob.glob(os.path.join(self.folderIback, ext)))
        itIback = sorted(grabbed_files, key=lambda file:os.path.getctime(file))
        self.da_Iback = self.glob_open(itIback)
        
        phi_list = np.unique(self.da_Iback.phi)
        phi_steps = phi_list[1] - phi_list[0]
        num_phi = int((self.da_Iback.phi.max()-self.da_Iback.phi.min())/phi_steps) + 1
        num_theta = int(len(self.da_Iback.phi)/num_phi)
        
        self.shape = (num_phi, num_theta)
        
        self.canvasIB.itemconfig(self.imoncanIB, image=self.img_chk)
        self.lift()
    
    def process_data(self):
        '''Reads images and background in parallel to perform normalization'''

        s0, s1 = self.shape
        num_images = s0*s1

        # Maybe check that the file is closed after that --
        fileOpen = lambda file:np.array(Image.open(file).convert('L')).astype(np.float32)
        
        bgSubtract = lambda im, bg : np.clip(im / (bg + 1.0), 0, 255)#.astype(np.uint8)
               
        self.data = None
        self.pbar.start()
        for k, (fnIgrey, fnIback) in enumerate(zip(self.da_Igrey.fname, self.da_Iback.fname)):
            im_Igrey = fileOpen(self.folderIgrey+fnIgrey)
            if self.data is None:
                rx, ry = im_Igrey.shape
                self.data = np.empty((num_images, rx, ry), dtype=np.uint8)
            
            if self.with_Iback:
                im_Iback = fileOpen(self.folderIback+fnIback)
                im_Igrey_norm = bgSubtract(im_Igrey, im_Iback)
                im_Igrey_norm = im_Igrey_norm - np.min(im_Igrey_norm)
                im_Igrey_norm = im_Igrey_norm / np.max(im_Igrey_norm)
                im_Igrey_norm = im_Igrey_norm * 255
                self.data[k] = im_Igrey_norm.astype(np.uint8)
            else:
                self.data[k] = im_Igrey
            
            if k%(num_images//10)==0: # 10% steps
                self.pbar.step(10)
                self.master.update_idletasks()
                
        self.pbar.stop()
        
        self.data = np.transpose(self.data, [1,2,0]).reshape((rx,ry,s0,s1))
        self.data = np.transpose(self.data, [0,1,3,2])
        
        print('> Loaded: ', self.data.shape, self.data.dtype)
        show(self.data[500,500])
        
        self.destroy()
        
        self.master_object.on_import_setup(self.data)
        
    def glob_open(self, iterator):    
        '''
        Opens all image files in a folder, from iterator and reg-exp.
        '''        
        fnames, thetas, phis = [], [], []
        for file in iterator:
            fname = os.path.basename(file)
            phi, theta = re.findall('[0-9]+', os.path.splitext(fname)[0])
            fnames.append(fname)
            phis.append(phi)
            thetas.append(theta)
        data_array = pd.DataFrame.from_dict({'fname':fnames, 
                                             'phi':phis, 
                                             'theta':thetas})
        data_array.phi = data_array.phi.astype('int')
        data_array.theta = data_array.theta.astype('int')
        data_array = data_array.sort_values(['phi', 'theta'])
        return data_array