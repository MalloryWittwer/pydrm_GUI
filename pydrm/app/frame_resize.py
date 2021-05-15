import os
import numpy as np
import tkinter as tk
from tkinter import ttk
import cv2
import psutil

from .host_frame import HostF

class ResizeFrame(HostF):
    def __init__(self, master, width, height, master_canvas, master_pbar, master_object):
        
        HostF.__init__(self, width, height)

        self.master = master
        self.canvas = master_canvas
        self.pbar = master_pbar
        self.master_object = master_object
        
        lbl = tk.Label(self, text='Max size (px): ')
        lbl.grid(row=0, column=0, sticky='E', pady=15, padx=15)
        self.resol_Entry = tk.Entry(self, width=20)
        self.resol_Entry.grid(row=0, column=1, pady=15, padx=15)
        self.btn_resize = ttk.Button(self, text='Resize', 
                                     command=lambda:self.resize_working_data(
                                     self.data_origin),
                                     state=tk.DISABLED,)
        self.btn_resize.grid(row=1, column=0, columnspan=2, pady=0, padx=15, sticky='we')
        
        ### Treeview of loaded data
        self.tree = ttk.Treeview(self, columns=('one', 'three'), height=2)
        self.tree.column("#0", width=80)
        self.tree.column('one', width=110)
        self.tree.column('three', width=40)
        self.tree.heading('one', text='Matrix shape')
        self.tree.heading('three', text='RAM')
        self.tree.insert('', 0, text='Loaded', values=('-', '-'))
        self.tree.insert('', 1, text='Displayed', values=('-', '-'))
        self.tree.grid(row=2, column=0, columnspan=3, pady=10, padx=5)
        
    def set_init_configuration(self):
        self.tree.delete(self.tree.get_children()[0])
        self.tree.insert('', 0, text='Loaded', values=('-', '-', '-'))
        self.tree.delete(self.tree.get_children()[1])
        self.tree.insert('', 1, text='Displayed', values=('-', '-', '-'))
        self.resol_Entry.delete(0, 'end')
        self.btn_resize.configure(state=tk.DISABLED)

    def on_import_setup(self):
        self.fetch_parent_variables()
        self.resol_Entry.delete(0, 'end')
        self.resol_Entry.insert(0,f'{np.max((self.rx_origin, self.ry_origin))}')
        self.btn_resize.configure(state=tk.NORMAL)
        self.resize_working_data(self.data_origin)
    
    def on_close_setup(self):
        self.set_init_configuration()
        del self.data_origin, self.rx_origin, self.ry_origin
    
    def fetch_parent_variables(self):
        self.data_origin = self.master_object.data_origin
        self.rx_origin = self.master_object.rx_origin
        self.ry_origin = self.master_object.ry_origin
        
    # -------------------------------------------------------------------------
    def resize_working_data(self, origin_data):
        '''
        Resizes data stack. Defines rx, ry, data, sc, (xy)max
        '''        
        # Retreive the maximum pixel size
        try:
            sm = int(self.resol_Entry.get())
        except:
            self.master_object.log_message('>>> Error: resizing entry is not an integer!')
            return 0
        
        rx, ry = origin_data.shape[0], origin_data.shape[1]
        cw = self.canvas.width
        
        if rx <= ry:
            ymax = cw
            xmax = int(cw/ry*rx)
            if self.ry_origin >= sm:
                new_rx, new_ry = int(sm/ry*rx), sm
                data = self.resize_stack(origin_data, new_rx, new_ry)
            else:
                data = origin_data
        else:
            xmax = cw
            ymax = int(cw/rx*ry)
            if self.rx_origin >= sm:
                new_rx, new_ry = sm, int(sm/rx*ry)
                data = self.resize_stack(origin_data, new_rx, new_ry)
            else:
                data = origin_data
        
        rx, ry = data.shape[0], data.shape[1]
        sc = xmax/rx
        
        # Memory usage estimate
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_percent() 
        
        self.tree.delete(self.tree.get_children()[0])
        self.tree.insert('', 0, text='Loaded', 
                         values=(f'{self.data_origin.shape}', 
                                 '{:.0f}%'.format(mem_usage)))

        self.tree.delete(self.tree.get_children()[1])
        self.tree.insert('', 1, text='Displayed', 
                         values=(f'{data.shape}', '-'))
        
        self.master_object.sc = sc
        self.master_object.rx = rx
        self.master_object.ry = ry
        self.master_object.xmax = xmax
        self.master_object.ymax = ymax
        self.master_object.data = data
        
        self.master_object.frame_scalers.fetch_parent_variables()
        self.master_object.frame_fixdrp.fetch_parent_variables()
        self.master_object.frame_movdrp.fetch_parent_variables()
        self.master_object.canvas0.fetch_parent_variables()
        
        self.master_object.frame_scalers.update_canvas()
        
        self.master_object.log_message(f'Resized to: ({rx} x {ry} px)')
        
    def resize_stack(self, data, newx, newy):
        '''Uses cv2.resize function over a loop'''
        rx, ry, s0, s1 = data.shape
        data = data.reshape((rx, ry, s0*s1))
        data = np.transpose(data, [2,0,1])
        self.pbar.start()
        resized_data = np.empty((data.shape[0], newx, newy), dtype=np.uint8)
        for k, im in enumerate(data):
            resized_data[k] = cv2.resize(im, (newy, newx), 
                                          interpolation=cv2.INTER_CUBIC)
            if k%100==0:
                step = 100/data.shape[0]*100
                self.pbar.step(step)
                self.master.update_idletasks()
        self.pbar.stop()
        resized_data = np.transpose(resized_data, [1,2,0])
        resized_data = resized_data.reshape((newx, newy, s0, s1))
        
        return resized_data
        
        
