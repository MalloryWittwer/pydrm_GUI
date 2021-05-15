import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF

import tensorflow as tf
tf.keras.backend.set_floatx('float32')

class NoneScaler():
    def fit(self):
        pass
    def fit_transform(self, data):
        return data
    def transform(self, data):
        return data
    
class NoneCompressor():
    def fit(self):
        pass
    def fit_transform(self, data):
        return data
    def transform(self, data):
        return data

class ModelParameterWindow(tk.Toplevel):
    '''
    Toplevel window for model parameter selection and preprocessing pipeline
    '''
    def __init__(self, master, master_object, width=250, height=500):
        tk.Toplevel.__init__(self,  master, width=width, height=height)
        
        self.title('Model Constructor')
        self.resizable(False, False)
        
        self.master_object = master_object
        
        self.output_size = 2 # Output size of the ANN as determined by the task selected.
        
        pdx = pdy = 10, 5
        
        lbf1 = ttk.LabelFrame(self, text='Preprocessing')
        lbf1.grid(row=2, column=0, sticky='we', padx=pdx, pady=pdy)
        
        lbl = tk.Label(lbf1, text='Rescaling: ')
        lbl.grid(row=0, column=0, sticky='e', padx=pdx, pady=pdy)
        self.cbb_scale = ttk.Combobox(lbf1, values=['None', 'Normalize', 'Standardize'])
        self.cbb_scale.grid(row=0, column=1, sticky='w', padx=pdx, pady=pdy)
        self.cbb_scale.current(1)

        lbl = tk.Label(lbf1, text='Compression: ')
        lbl.grid(row=1, column=0, sticky='e', padx=pdx, pady=pdy)
        self.cbb_comp = ttk.Combobox(lbf1, values=['None', 'NMF (20)', 'PCA (20)'])
        self.cbb_comp.grid(row=1, column=1, sticky='w', padx=pdx, pady=pdy)
        self.cbb_comp.current(2)
        
        lbf2 = ttk.LabelFrame(self, text='ANN')
        lbf2.grid(row=3, column=0, sticky='we', padx=pdx, pady=pdy)
        
        lbl = tk.Label(lbf2, text='Layers: ')
        lbl.grid(row=0, column=0, sticky='e', padx=pdx, pady=pdy)
        self.layers_Entry= tk.Entry(lbf2, width=15)
        self.layers_Entry.grid(row=0, column=1, sticky='w', padx=pdx, pady=pdy)
        self.layers_Entry.delete(0,'end')
        self.layers_Entry.insert(0,'1')
        
        lbl = tk.Label(lbf2, text='Nodes: ')
        lbl.grid(row=1, column=0, sticky='e', padx=pdx, pady=pdy)
        self.neurons_Entry= tk.Entry(lbf2, width=15)
        self.neurons_Entry.grid(row=1, column=1, sticky='w', padx=pdx, pady=pdy)
        self.neurons_Entry.delete(0,'end')
        self.neurons_Entry.insert(0,'8')
        
        lbl = tk.Label(lbf2, text='L2 strength: ')
        lbl.grid(row=2, column=0, sticky='e', padx=pdx, pady=pdy)
        self.l2_Entry= tk.Entry(lbf2, width=15)
        self.l2_Entry.grid(row=2, column=1, sticky='w', padx=pdx, pady=pdy)
        self.l2_Entry.delete(0,'end')
        self.l2_Entry.insert(0,'1.0')
        
        lbl = tk.Label(lbf2, text='Initializer: ')
        lbl.grid(row=3, column=0, sticky='e', padx=pdx, pady=pdy)
        self.cbb_init = ttk.Combobox(lbf2, values=['Variance scaling', 
                                                   'Glorot uniform'])
        self.cbb_init.grid(row=3, column=1, sticky='w', padx=pdx, pady=pdy)
        self.cbb_init.current(0)
        
        lbl = tk.Label(lbf2, text='Activation: ')
        lbl.grid(row=4, column=0, sticky='e', padx=pdx, pady=pdy)
        self.cbb_activ = ttk.Combobox(lbf2, values=['ReLu', 'Sigmoid', 'Tanh'])
        self.cbb_activ.grid(row=4, column=1, sticky='w', padx=pdx, pady=pdy)
        self.cbb_activ.current(0)

        btn_get_eulers = ttk.Button(
            self, text='Create Model', command=self.create_model)
        btn_get_eulers.grid(
            row=4, column=0, columnspan=4, sticky='we', padx=15, pady=15)
    
    def ann_layer_constructor(self, n_nodes, activation, initializer, l2reg):
        return tf.keras.layers.Dense(n_nodes, activation=activation,
                                     kernel_initializer=initializer,
                                     kernel_regularizer=tf.keras.regularizers.l2(l2reg))
    
    def create_model(self):
        
        scaler_bank = {0:NoneScaler(), 1:MinMaxScaler(), 2:StandardScaler()}
        scaler = scaler_bank[self.cbb_scale.current()]
        
        compressor_bank = {0:NoneCompressor(), 1:NMF(20), 2:PCA(20)}
        compressor_ID = self.cbb_comp.current()
        compressor = compressor_bank[compressor_ID]
        dico_compressor_input_size = {0:self.master_object.s0*self.master_object.s1,
                                      1:20,2:20}
        input_size = dico_compressor_input_size[compressor_ID]
        
        initializer_bank = {0:tf.keras.initializers.VarianceScaling(0.01),
                            1:tf.keras.initializers.GlorotUniform()}
        initializer = initializer_bank[self.cbb_init.current()]
        
        activation_bank = {0:'relu', 1:'softmax', 2:'tanh'}
        activation = activation_bank[self.cbb_activ.current()]
        
        n_layers = int(self.layers_Entry.get())
        l2reg = float(self.l2_Entry.get())
        n_nodes = int(self.neurons_Entry.get())
        
        ann_model = tf.keras.Sequential()
        ann_model.add(self.ann_layer_constructor(input_size, activation, initializer, l2reg))
        ann_model.add(tf.keras.layers.BatchNormalization())
        for lay in range(n_layers):
            ann_model.add(self.ann_layer_constructor(n_nodes, activation, initializer, l2reg))
            ann_model.add(tf.keras.layers.BatchNormalization())
        ann_model.add(tf.keras.layers.Dense(self.output_size, activation=None, kernel_initializer=initializer))            

        pipeline_params = {
            'scaler':scaler,
            'compressor':compressor,
            'ann_model':ann_model,
        }
        
        self.master_object.set_model_pipeline(pipeline_params)
        
        self.destroy()
        
        