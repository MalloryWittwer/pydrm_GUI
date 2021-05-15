import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

from .window_model_params import ModelParameterWindow
from .window_model_train import ModelTrainerWindow
from .mappingCalculators import BiTexDiscriminator
from .host_frame import HostF
from pydrm.utils import shuffler

class MappingFrame(HostF):
    def __init__(self, master, width, height, master_canvas, host_canvas, master_pbar, master_object):

        HostF.__init__(self, width, height)
        
        self.master = master
        self.canvas = master_canvas
        self.host_canvas = host_canvas
        self.pbar = master_pbar
        self.master_object = master_object   

        pdx, pdy = 5, 4
        
        btn_width = 37
        btn_short = 17
        
        ### -------------------------------------------------------------------
        lbf = ttk.LabelFrame(self, text='Model')
        lbf.grid(row=0, column=0, sticky='we', padx=pdx, pady=pdy)        

        self.btn_create = ttk.Button(lbf, width=btn_width, text='Create', 
                                     command=self.launch_create_model)
        self.btn_create.grid(row=0, column=0, sticky='w', padx=pdx, pady=pdy)
        
        # self.btn_load_model = ttk.Button(lbf, width=btn_short, text='Load', 
        #                                  command=lambda:None, state=tk.DISABLED)
        # self.btn_load_model.grid(row=1, column=0, sticky='w', padx=pdx, pady=pdy)
        
        # self.btn_save = ttk.Button(lbf, width=btn_short, text='Save', 
        #                            command=lambda:None, state=tk.DISABLED)
        # self.btn_save.grid(row=1, column=1, sticky='w', padx=pdx, pady=pdy)
        
        self.tree_model = ttk.Treeview(lbf, columns=('one'), height=1, show="tree")
        self.tree_model.column("#0", width=70)
        self.tree_model.column('one', width=160)
        self.tree_model.insert('', 0, text='Loaded: ', values=('-', '-'))
        self.tree_model.grid(row=2, column=0, columnspan=2, padx=pdx, pady=pdy)
        
        ### -------------------------------------------------------------------
        lbf = ttk.LabelFrame(self, text='Training / Validation sets')
        lbf.grid(row=1, column=0, sticky='we', padx=pdx, pady=pdy)  
        
        self.btn_train_set = ttk.Button(lbf, width=btn_short, text='Select', 
                                    command=self.launch_set_selector)
        self.btn_train_set.grid(row=0, column=0, sticky='w', padx=pdx, pady=pdy)
 
        self.btn_reset = ttk.Button(lbf, width=btn_short, text='Reset', 
                                    command=self.reset_training_set)
        self.btn_reset.grid(row=0, column=1, sticky='w', padx=pdx, pady=pdy)
        
        self.tree_training = ttk.Treeview(lbf, columns=('one'), height=1, show="tree")
        self.tree_training.column("#0", width=70)
        self.tree_training.column('one', width=160)
        self.tree_training.insert('', 0, text='Loaded: ', values=('-', '-'))
        self.tree_training.grid(row=2, column=0, columnspan=2, padx=pdx, pady=pdy)
        
        ### -------------------------------------------------------------------
        lbf = ttk.LabelFrame(self, text='Training / Prediction')
        lbf.grid(row=2, column=0, sticky='we', padx=pdx, pady=pdy) 
        
        self.btn_train = ttk.Button(lbf, width=btn_short, text='Train', 
                                    command=self.launch_train_model)
        self.btn_train.grid(row=0, column=0, sticky='w', padx=pdx, pady=pdy)

        self.btn_predict = ttk.Button(lbf, width=btn_short, text='Predict', 
                                      command=self.predict_classes)
        self.btn_predict.grid(row=0, column=1, sticky='w', padx=pdx, pady=pdy)
        
        self.tree_valid = ttk.Treeview(lbf, columns=('one'), height=1, show="tree")
        self.tree_valid.column("#0", width=70)
        self.tree_valid.column('one', width=160)
        self.tree_valid.insert('', 0, text='Epochs: ', values=('-', '-'))
        self.tree_valid.grid(row=1, column=0, columnspan=2, padx=pdx, pady=pdy)
        
        ### -------------------------------------------------------------------
        lbf = ttk.LabelFrame(self, text='Display')
        lbf.grid(row=3, column=0, columnspan=2, sticky='we', padx=pdx, pady=pdy)
        
        self.cbb_map = ttk.Combobox(lbf, values=['Predictions', 'QR reading'])
        self.cbb_map.grid(row=0, column=0, sticky='w', padx=pdx, pady=pdy)
        self.cbb_map.bind("<<ComboboxSelected>>", self.show_predmap)
        
        self.predmap_visibility = tk.IntVar()
        self.visibutton = tk.Checkbutton(
            lbf, text='Visible', variable=self.predmap_visibility, onvalue=200, offvalue=0, command=self.show_predmap)
        self.visibutton.grid(row=0, column=1, sticky='w', padx=pdx, pady=pdy)
        
        self.btn_save_im = ttk.Button(lbf, width=btn_width, text='Save', 
                                      command=self.save_current_image)
        self.btn_save_im.grid(row=1, column=0, columnspan=2, sticky='w', padx=pdx, pady=pdy)

    def set_init_configuration(self):
        '''
        Set initial configuraiton.
        '''
        self.btn_create.configure(state=tk.DISABLED)
        self.btn_predict.configure(state=tk.DISABLED)
        self.btn_save_im.configure(state=tk.DISABLED)
        self.visibutton.configure(state=tk.DISABLED)
        self.cbb_map.configure(state=tk.DISABLED)
        self.btn_train_set.configure(state=tk.DISABLED)
        self.btn_reset.configure(state=tk.DISABLED)
        
        self.host_canvas.set_init_configuration()
        self.internal_model = None
        
        self.reset_training_set()
        
    def on_import_setup(self):
        '''
        On import setup.
        '''
        self.host_canvas.on_import_setup()
        self.fetch_parent_variables()

        init_im = np.ones((self.rx, self.ry, 4), dtype=np.uint8)*255
        self.dicomap = {0:[init_im, self.canvas.create_image(0, 0, anchor='nw')],
                        1:[init_im, self.canvas.create_image(0, 0, anchor='nw')]}
        self.cbb_map.current(0)
        self.visibutton.deselect()
        self.show_predmap()
        
        self.btn_create.configure(state=tk.NORMAL)
        self.btn_save_im.configure(state=tk.NORMAL)
        self.visibutton.configure(state=tk.NORMAL)
        self.cbb_map.configure(state=tk.NORMAL)
        
    def on_close_setup(self):
        '''
        On close setup.
        '''
        init_im = np.ones((self.rx, self.ry, 4), dtype=np.uint8)*255
        self.dicomap = {0:[init_im, self.canvas.create_image(0, 0, anchor='nw')],
                        1:[init_im, self.canvas.create_image(0, 0, anchor='nw')]}
        self.cbb_map.current(0)
        self.visibutton.deselect()
        self.show_predmap()
        
        self.host_canvas.on_close_setup()
        
        del self.data
        
        self.btn_create.configure(state=tk.DISABLED)
        self.btn_reset.configure(state=tk.DISABLED)
        
    def fetch_parent_variables(self):
        self.data = self.master_object.data
        self.rx = self.master_object.rx
        self.ry = self.master_object.ry
        self.s0 = self.master_object.s0
        self.s1 = self.master_object.s1
        self.sc = self.master_object.sc
    
    # -------------------------------------------------------------------------
    def reset_training_set(self):
        '''
        Reset training set.
        '''
        self.global_sets = {}
        self.global_count = 0
        
        self.training_areas = [] # Rectangles contraining training areas
        self.master_object.canvas0.define_training_areas(self.training_areas)
        
        self.tree_training.delete(self.tree_training.get_children()[0])
        self.tree_training.insert('', 0, text='Loaded: ', values='-')
        self.btn_train.configure(state=tk.DISABLED)
        self.master_object.log_message('> Reset training set!')
        
    # -------------------------------------------------------------------------
    def launch_create_model(self):
        '''
        Launches model parameter window, which calls self.set_model_pipleine.
        '''
        ModelParameterWindow(self.master, master_object=self)
    
    def set_model_pipeline(self, pipeline_params):
        '''
        Sets model pipeline.
        '''
        self.pipeline_params = pipeline_params

        self.internal_model = BiTexDiscriminator(
            self.master, self.pbar, self.master_object, self.host_canvas, self.canvas)

        sc = self.pipeline_params.get('scaler').__class__.__name__
        cp = self.pipeline_params.get('compressor').__class__.__name__
        parsed_model = f'{sc}-{cp}-ANN'
        self.tree_model.delete(self.tree_model.get_children()[0])
        self.tree_model.insert('', 0, text='Loaded: ', values=(parsed_model))

        self.master_object.log_message('> Model pipeline created!')
        
        # Enable selection of a training set and reset
        self.btn_predict.configure(state=tk.DISABLED)
        self.btn_train_set.configure(state=tk.NORMAL)
        self.btn_reset.configure(state=tk.NORMAL)
        
    # -------------------------------------------------------------------------
    def launch_set_selector(self):
        '''
        Launches tool to draw C1 and C2 selection areas.
        '''
        self.tlv = tk.Toplevel(self.master_object)
        self.tlv.title('Select C1 and C2')
        self.tlv.resizable(False, False)
        self.tlv.lift()
        
        btn = ttk.Button(self.tlv, text='C1', 
                         command=lambda:self.master_object.set_flag_release(2),
                         )
        btn.grid(row=0, column=0, padx=10, pady=10, sticky='we')
        
        btn = ttk.Button(self.tlv, text='C2', 
                         command=lambda:self.master_object.set_flag_release(3),
                         )
        btn.grid(row=0, column=1, padx=10, pady=10, sticky='we')
        
        btn = ttk.Button(self.tlv, text='Ok', command=self.set_selected_data)
        btn.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='we')
        
        self.wait_window(self.tlv)
        
        self.master_object.log_message('> Train/Test sets selected!')
        
    def select_C1(self, data_slice, sx, sy, ex, ey):
        '''Selects C1 data based on received data slice.'''
        rx, ry, s0, s1 = data_slice.shape
        self.training_areas.append(['C1', (sx, sy, ex, ey), '#0090ff',
                                    data_slice.reshape((rx*ry, s0*s1))])
        self.master_object.canvas0.define_training_areas(self.training_areas)
        self.tlv.lift()
        
    def select_C2(self, data_slice, sx, sy, ex, ey):
        '''Selects C2 data based on received data slice.'''
        rx, ry, s0, s1 = data_slice.shape
        self.training_areas.append(['C2', (sx, sy, ex, ey), '#ff6600', 
                                    data_slice.reshape((rx*ry, s0*s1))])
        self.master_object.canvas0.define_training_areas(self.training_areas)
        self.tlv.lift()
        
    def save_training_areas(self):
        '''Saves training areas.'''
        pass
    
    def load_training_areas(self):
        '''Loads training areas.'''
        pass
        
    def set_selected_data(self):
        '''
        Selects training and test data based on drawn training areas
        '''
        self.master_object.flag_release = 0 # stop selecting data
        
        data_C1 = np.empty((0, self.s0*self.s1))
        data_C2 = np.empty((0, self.s0*self.s1))
        
        for name, (x0, y0, x1, y1), col, data in self.training_areas:
            if name=='C1':
                data_C1 = np.vstack((data_C1, data))
                print('data C1: ', data_C1.shape)
            elif name=='C2':
                data_C2 = np.vstack((data_C2, data))
                print('data C2: ', data_C2.shape)
        
        total_slices = np.vstack((data_C1, data_C2))
        print('Total slice: ', total_slices.shape)
        
        zeros_array = np.zeros(len(data_C1))
        ones_array = np.ones(len(data_C2))
        
        total_labels = np.hstack((zeros_array, ones_array))
        
        train_size = int(0.8*total_slices.shape[0])
        valid_size = total_slices.shape[0]-train_size
        
        data_selection, idx = shuffler(total_slices, train_size+valid_size)
        target_selection = total_labels[idx]
        xtr = data_selection[:train_size]
        xte = data_selection[train_size:]
        ytr = target_selection[:train_size]
        yte = target_selection[train_size:]
        
        ytr = ytr.reshape((-1,1))
        yte = yte.reshape((-1,1))
        
        self.master_object.canvas0.delete_rectangle()
        
        ### Adds selected training and test sets to global training and test sets.
        self.global_count += 1
        self.global_sets[self.global_count] = {'xtr':xtr,'xte':xte,'ytr':ytr,'yte':yte}
    
        parsed_info = f'#{self.global_count}-tr:{ytr.shape[0]}/va:{yte.shape[0]}'
        self.tree_training.delete(self.tree_training.get_children()[0])
        self.tree_training.insert('', 0, text='Loaded: ', values=(parsed_info))
        
        # Enable training
        self.btn_train.configure(state=tk.NORMAL)
        self.tlv.destroy()
    
    # -------------------------------------------------------------------------
    def launch_train_model(self):
        '''
        Launches top window, which calls self.start_training.
        '''
        ModelTrainerWindow(self.master, master_object=self)
   
    def start_training(self, training_params):
        '''
        Starts training process.
        '''
        self.xtr = np.empty((0,self.s0*self.s1))
        self.xte = np.empty((0,self.s0*self.s1))
        self.ytr = np.empty((0,1))
        self.yte = np.empty((0,1))
        
        for key in self.global_sets.keys():
            self.xtr = np.vstack((self.xtr, self.global_sets[key]['xtr']))
            self.ytr = np.vstack((self.ytr, self.global_sets[key]['ytr']))
            self.xte = np.vstack((self.xte, self.global_sets[key]['xte']))
            self.yte = np.vstack((self.yte, self.global_sets[key]['yte']))
        
        self.ytr = np.squeeze(self.ytr)
        self.yte = np.squeeze(self.yte)
        
        print(f'> Training set: {(self.ytr==0).sum()} x C1 + {(self.ytr==1).sum()} x C2 = {len(self.xtr)} points.')
        
        # Shuffle training and test sets
        self.xtr, idx = shuffler(self.xtr, self.xtr.shape[0])
        self.ytr = self.ytr[idx]
        self.xte, idx = shuffler(self.xte, self.xte.shape[0])
        self.yte = self.yte[idx]
        
        # Set model pipleine
        self.internal_model.set_model_pipeline(
            self.pipeline_params, self.xtr, self.xte, self.ytr, self.yte)
        
        # Start training!
        self.internal_model.start_training(training_params)
        
        # Report training params
        self.tree_valid.delete(self.tree_valid.get_children()[0])
        self.tree_valid.insert('', 0, text='Epochs: ', 
                               values=(f'{self.internal_model.total_epochs}'))
        
        # Enable prediction
        self.btn_predict.configure(state=tk.NORMAL)
    
    def predict_classes(self):
        '''
        Maps the class prediction (C1 or C2) over the field of view.
        '''
        if self.internal_model is None:
            return 0
        self.master_object.log_message('> Predicting orientation...')
        im_classes, QR_output = self.internal_model.predict_classes()
        self.dicomap[0][0] = im_classes.copy()
        self.dicomap[1][0] = QR_output.copy()
        self.visibutton.select()
        self.show_predmap()
        self.master_object.log_message('> Orientation predicted!')
    
    def show_predmap(self, event=None):
        '''
        Gets map to show and shows it.
        '''
        idmap = self.cbb_map.current()
        self.dicomap[idmap][0][...,3] = self.predmap_visibility.get()
        orimap = self.dicomap[idmap][0]        
        hg, wg, _ = orimap.shape           
        im = Image.fromarray(orimap, mode='RGBA')
        im = im.resize((int(wg*self.sc), int(hg*self.sc)))
        self.im = ImageTk.PhotoImage(im)            
        self.canvas.itemconfig(self.dicomap[idmap][1], image=self.im)
    
    def save_current_image(self):
        '''
        Gets map to save and saves it.
        '''
        idmap = self.cbb_map.current()
        orimap = self.dicomap[idmap][0].copy()
        orimap[...,3] = 255
        hg, wg, _ = orimap.shape           
        im = Image.fromarray(orimap, mode='RGBA')
        with filedialog.asksaveasfile(mode='w', defaultextension='.png') as f:
            im.save(f.name)