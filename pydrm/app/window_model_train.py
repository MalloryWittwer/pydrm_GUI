import tkinter as tk
from tkinter import ttk
import tensorflow as tf

class ModelTrainerWindow(tk.Toplevel):
    '''
    Toplevel window for model parameter selection and preprocessing pipeline
    '''
    def __init__(self, master, master_object, width=250, height=500):
        tk.Toplevel.__init__(self,  master, width=width, height=height)
        
        self.title('Training Parameters')
        self.resizable(False, False)
        
        self.master_object = master_object
        
        pdx = pdy = 15
        entry_width = 5
        
        lbl = tk.Label(self, text='Epochs: ')
        lbl.grid(row=0, column=0, sticky='e', padx=pdx, pady=pdy)
        self.ep_Entry= tk.Entry(self, width=entry_width)
        self.ep_Entry.grid(row=0, column=1, sticky='we', padx=pdx, pady=pdy)
        self.ep_Entry.delete(0,'end')
        self.ep_Entry.insert(0,'10')
        
        lbl = tk.Label(self, text='Batch size: ')
        lbl.grid(row=1, column=0, sticky='e', padx=pdx, pady=pdy)
        self.bs_Entry= tk.Entry(self, width=entry_width)
        self.bs_Entry.grid(row=1, column=1, sticky='we', padx=pdx, pady=pdy)
        self.bs_Entry.delete(0,'end')
        self.bs_Entry.insert(0,'50')
        
        lbl = tk.Label(self, text='Learning rate: ')
        lbl.grid(row=2, column=0, sticky='e', padx=pdx, pady=pdy)
        self.lr_Entry= tk.Entry(self, width=entry_width)
        self.lr_Entry.grid(row=2, column=1, sticky='we', padx=pdx, pady=pdy)
        self.lr_Entry.delete(0,'end')
        self.lr_Entry.insert(0,'0.001')
        
        lbl = tk.Label(self, text='Optimizer: ')
        lbl.grid(row=3, column=0, sticky='e', padx=pdx, pady=pdy)
        self.cbb_optim = ttk.Combobox(self, values=['Adam', 'SGD', 'RMSprop'])
        self.cbb_optim.grid(row=3, column=1, sticky='w', padx=pdx, pady=pdy)
        self.cbb_optim.current(0)
        
        btn_start_training = ttk.Button(
            self, text='Train Model', command=self.train_model, width=15)
        btn_start_training.grid(
            row=4, column=0, columnspan=2, sticky='we', padx=15, pady=15)
        
    def train_model(self):
        lr = float(self.lr_Entry.get())
        optimizers_bank = {0:tf.keras.optimizers.Adam(lr),
                           1:tf.keras.optimizers.SGD(lr),
                           2:tf.keras.optimizers.RMSprop(lr),
        }
        training_params = {
            'epochs':int(self.ep_Entry.get()),
            'batch size':int(self.bs_Entry.get()),
            'optimizer':optimizers_bank[self.cbb_optim.current()],
        }
        self.destroy()
        self.master_object.start_training(training_params)