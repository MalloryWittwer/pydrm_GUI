import time
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.decomposition import NMF, PCA
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.linear_model import LogisticRegression

from .host_frame import HostF
from pydrm.utils import (
    fit_lrc_model, 
    lrc_mrm_segmentation, 
    size_dist_plot, 
    shuffler,
    )

class DataCompressor():
    def __init__(self, compressor=None):
        '''Supported compressors: NMF() and PCA() objects'''
        self.compressor = compressor

    def fit(self, dataset, sampling_fraction=1.0, sample_size=1000):
        '''Samples the dataset and fits a compressor model'''
        data = dataset.get('data')
        self.s0, self.s1 = dataset.get('angular_resol')
        if self.compressor:
            # Extract a sample            
            if sample_size is None:
                sample_size = np.ceil(data.shape[0]*sampling_fraction).astype('int')
                print('sampled from fraction: ', sampling_fraction)
            
            data_extract, _ = shuffler(data, sample_size)
            
            # Fit the compressor
            self.compressor.fit(data_extract)

    def transform(self, data):
        '''Compresses the data in the input dataset'''
        if self.compressor:
            data_compressed = self.compressor.transform(data)
        else:
            data_compressed = data
        return data_compressed

class SegmentationFrame(HostF):
    def __init__(self, master, width, height, master_canvas, master_host_canvas, master_pbar, master_object):
        
        HostF.__init__(self, width, height)

        self.master = master
        self.host_canvas = master_host_canvas
        
        self.canvas = master_canvas
        self.pbar = master_pbar
        self.master_object = master_object
        
        self.segmentation = None
        
        # Text entries for segmentation parameters ----------------------------
        lbl = tk.Label(self, text='Components: ')
        lbl.grid(row=0, column=0, sticky='e', padx=5, pady=10)
        self.comp_Entry = tk.Entry(self, width=25)
        self.comp_Entry.grid(row=0, column=1, columnspan=2, sticky='w', padx=5, pady=10)
        lbl = tk.Label(self, text='Max sampling: ')
        lbl.grid(row=1, column=0, sticky='e', padx=5, pady=10)
        self.spl_Entry = tk.Entry(self, width=25)
        self.spl_Entry.grid(row=1, column=1, columnspan=2, sticky='w', padx=5, pady=10)
        lbl = tk.Label(self, text='Compression: ')
        lbl.grid(row=2, column=0, sticky='e', padx=5, pady=10)
        
        # Comrpessors radio buttons -------------------------------------------
        self.compressor_ID = tk.IntVar()
        radioNMF = tk.Radiobutton(self, text='NMF', 
                                  variable=self.compressor_ID, value=0)
        radioNMF.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        radioPCA = tk.Radiobutton(self, text='PCA', 
                                  variable=self.compressor_ID, value=1)
        radioPCA.grid(row=2, column=2, sticky='w', padx=5, pady=5)
        
        # Visibility check button ---------------------------------------------
        self.visibutton = tk.Checkbutton(self, text='Visible', 
                                         command=self.toggle_segmentation,
                                         state=tk.DISABLED)
        self.visibutton.grid(row=3, column=2, sticky='w', padx=5)
        
        # Segment All button --------------------------------------------------
        self.segAllbtn = ttk.Button(self, text='Segment All',
                               command=self.segment_all,
                               state=tk.DISABLED)
        self.segAllbtn.grid(row=3, column=0, columnspan=2, sticky='we', padx=5)
        
    def set_init_configuration(self):
        self.seg_visibility = True
        
        self.host_canvas.set_init_configuration()
        
        self.visibutton.configure(state=tk.DISABLED)
        self.segAllbtn.configure(state=tk.DISABLED)
        self.spl_Entry.delete(0, 'end')
        self.spl_Entry.insert(0,'5000')
        self.comp_Entry.delete(0, 'end')
        self.comp_Entry.insert(0,'50')
        self.visibutton.select()
    
    def on_import_setup(self):
        self.host_canvas.on_import_setup()
        self.fetch_parent_variables()
        self.visibutton.configure(state=tk.NORMAL)
        self.segAllbtn.configure(state=tk.NORMAL)
        
    def on_close_setup(self):
        self.set_init_configuration()
        self.host_canvas.on_close_setup()
        del self.rx, self.ry, self.s0, self.s1, self.sc
    
    def fetch_parent_variables(self):
        self.rx = self.master_object.rx
        self.ry = self.master_object.ry
        self.s0 = self.master_object.s0
        self.s1 = self.master_object.s1
        self.sc = self.master_object.sc

    # -------------------------------------------------------------------------
    def update_starters(self, startx, starty):
        self.startx, self.starty = startx, starty
        try:
            self.canvas.delete(self.imoncan_seg)
        except:
            pass

    def update_seg_params(self):
        '''Called by the Update button.'''
        self.fetch_parent_variables()
        try:
            ms = int(self.spl_Entry.get())
        except:
            self.master_object.log_message(('> Could not recognize sampling.'
                                           ' Will use default (all px).'))
            ms = self.rx*self.ry
        try:
            compressor_cps = int(self.comp_Entry.get())
            compressor_cps = np.clip(compressor_cps, 1, self.s0*self.s1)
            compressor = (
                DataCompressor(PCA(compressor_cps)) if self.compressor_ID.get()
                else DataCompressor(NMF(compressor_cps))
                )
        except:
            self.master_object.log_message('> Will not use data compression.')
            compressor = None
        return ms, compressor
            
    def toggle_segmentation(self, event=None):
        '''Toggles visibility of segmentation map'''
        self.seg_visibility = not self.seg_visibility
        if self.segmentation is not None:
            self.display_segmentation()
        
    def display_segmentation(self):
        '''Shows segmentation on the canvas'''
        if self.seg_visibility:
            # Make segmentation 50% opaque
            self.segmentation[:,:,3] = 128
            self.canvas.update_color('red')
        else:
            # Make segmentation fully transparent
            self.segmentation[:,:,3] = 0
            self.canvas.update_color('')
        
        hg, wg = self.segmentation.shape[0], self.segmentation.shape[1]
        self.segforshow = Image.fromarray(self.segmentation, mode='RGBA')
        self.segforshow = self.segforshow.resize((int(wg*self.sc), 
                                                  int(hg*self.sc)))
        self.segforshow = ImageTk.PhotoImage(self.segforshow)
        self.imoncan_seg = self.canvas.create_image(self.startx*self.sc,
                                                    self.starty*self.sc,
                                                    anchor='nw',
                                                    image=self.segforshow)
        
    def segment_all(self):
        '''Segments the entire area'''
        self.seg_visibility = True  
        self.update_starters(0, 0)
        self.canvas.delete_rectangle()
        self.segmentation_pipeline(self.master_object.data)
    
    def segmentation_pipeline(self, data_slice=None):
        '''Performs one round of segmentation on the provided data subset.'''
                
        # Collect segmentation parameters from pannel
        sample_size, compressor = self.update_seg_params()
        
        # Creating a dataset
        rx, ry, s0, s1 = data_slice.shape
        dataset = {
            'data':data_slice.reshape((rx*ry, s0*s1)), 
            'spatial_resol':(rx,ry),
            'angular_resol':(s0,s1),
            }
        
        segmentation, gbs = self.run_lrc_mrm(
            dataset, compressor, 
            sample_size = rx*ry if rx*ry <= sample_size else sample_size)
        
        ### GRAIN SIZE DISTRIBUTION PLOT
        ngrains = np.unique(segmentation).size
        fig = size_dist_plot(dataset, limdown=0, limup=1000, gsize=100)
        plt.title(f'Cumulative grain size (px)\nN = {ngrains}')
        plt.xlabel('')
        plt.ylabel('')
        plt.legend().remove()
        plt.tight_layout()
        plt.close()
        self.gsize = FigureCanvasAgg(fig)
        s, (width, height) = self.gsize.print_to_buffer()
        self.gsize = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        
        self.host_canvas.show_rgba(self.gsize)
        self.master_object.nbk.select(2)

        segmentation = plt.cm.jet(segmentation/np.max(segmentation))*255
        segmentation[gbs] *= 0
        self.segmentation = segmentation.astype(np.uint8)

        # Display the result
        self.seg_visibility = True
        self.visibutton.select()
        self.display_segmentation()
        
    def run_lrc_mrm(self, dataset_slice, compressor, sample_size):
        '''Runs the segmentation.'''
        t0 = time.time()
        
        # Start progress bar
        self.pbar.start()
        self.pbar.step(10)
        self.master.update_idletasks()
    
        # Data compression
        self.master_object.log_message('> Compressing data...')
    
        compressor.fit(dataset_slice, sample_size)
        compressed_dataset = compressor.transform(dataset_slice['data'])
        dataset_slice['data'] = compressed_dataset
        
        self.pbar.step(30)
        self.master.update_idletasks()
        # LRC model fitting
        self.master_object.log_message('> Fitting model...')
        
        dataset_slice = fit_lrc_model(
            dataset_slice,
            model=LogisticRegression(penalty='none', max_iter=2000), 
            training_set_size=sample_size,
        )
    
        self.pbar.step(30)
        self.master.update_idletasks()
        self.master_object.log_message('> Segmenting domain...')
    
        dataset_slice = lrc_mrm_segmentation(dataset_slice)
        
        # Stop progress bar
        self.pbar.step(30)
        self.master.update_idletasks()
        self.master_object.log_message(
            '> Domain segmented! ({:.2f} sec)'.format(time.time()-t0)
            )
        time.sleep(0.5)
        self.pbar.stop()
        
        rx, ry = dataset_slice.get('spatial_resol')
        segmentation = dataset_slice.get('segmentation').reshape((rx, ry))
        gbs = dataset_slice.get('boundaries').reshape((rx, ry))
        
        # Return segmentation map and grain boundary map
        return segmentation, gbs
