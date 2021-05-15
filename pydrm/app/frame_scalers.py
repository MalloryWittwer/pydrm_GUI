import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .host_frame import HostF

class FrameScalers(HostF):
    def __init__(self, master, width, height, master_canvas, master_host_canvas, master_object):
        
        HostF.__init__(self, width, height, )
        
        self.canvas = master_canvas
        self.host_canvas = master_host_canvas
        self.master_object = master_object
        
        # Theta scaler --------------------------------------------------------
        self.lbl_scale_theta = tk.Label(self, text='Elevation (00)')
        self.lbl_scale_theta.grid(row=0, column=0, sticky='e', padx=10)
        self.scaleTheta = tk.Scale(self, 
                                   from_=0, to_=0, resolution=1, 
                                   command=self.update_theta, 
                                   orient='horizontal', 
                                   width=25, 
                                   length=230,
                                   state=tk.DISABLED,
                                   showvalue=False,
                                   )
        self.scaleTheta.grid(row=0, column=1, sticky='w', padx=10)
        
        # Phi scaler ----------------------------------------------------------
        self.lbl_scale_phi = tk.Label(self, text='Azimuth (00)')
        self.lbl_scale_phi.grid(row=1, column=0, sticky='e', padx=10)
        self.scalePhi = tk.Scale(self, 
                                 from_=0, to_=0, resolution=1, 
                                 command=self.update_phi, 
                                 orient='horizontal', 
                                 width=25, 
                                 length=230,
                                 state=tk.DISABLED,
                                 showvalue=False,
                                 )
        self.scalePhi.grid(row=1, column=1, sticky='w', padx=10)

    def set_init_configuration(self):
        self.host_canvas.set_init_configuration()
        self.scaleTheta.configure(to_=1, state=tk.DISABLED)
        self.current_the = 0
        self.scalePhi.configure(to_=1, state=tk.DISABLED)
        self.current_phi = 0

    def on_import_setup(self):
        self.host_canvas.on_import_setup()
        self.fetch_parent_variables()
        self.scaleTheta.configure(to_=self.s0-1, state=tk.NORMAL)
        self.scalePhi.configure(to_=self.s1-1, state=tk.NORMAL)
        self.canvas.bind("<MouseWheel>", self.update_theta_mw)
        self.canvas.bind('<Shift-MouseWheel>', self.update_phi_mw)
        
    def on_close_setup(self):
        self.canvas.unbind("<MouseWheel>")
        self.canvas.unbind('<Shift-MouseWheel>')
        self.set_init_configuration()
        self.host_canvas.on_close_setup()
        del self.data, self.s0, self.s1, self.xmax, self.ymax
    
    def fetch_parent_variables(self):        
        self.data = self.master_object.data
        self.s0 = self.master_object.s0
        self.s1 = self.master_object.s1
        self.xmax = self.master_object.xmax
        self.ymax = self.master_object.ymax
    
    # -------------------------------------------------------------------------
    def update_phi(self, new_phi):
        self.current_phi = int(new_phi)
        self.lbl_scale_phi.configure(
            text='Azimuth ({:02d})'.format(self.current_phi))
        self.update_canvas()
    
    def update_theta(self, new_the):
        self.current_the = int(new_the)
        self.lbl_scale_theta.configure(
            text='Elevation ({:02d})'.format(self.current_the))
        self.update_canvas()
        
    def update_theta_mw(self, new_the):
        '''Mouse wheel update of theta, two-by-two (mw)'''
        self.current_the = np.clip(
            self.current_the + np.sign(new_the.delta)*2, 0, self.s0-1)
        self.scaleTheta.set(self.current_the)
        self.lbl_scale_theta.configure(
            text='Elevation ({:02d})'.format(self.current_the))
        self.update_canvas()
        
    def update_phi_mw(self, new_phi):
        '''Mouse wheel update of phi, two-by-two (shift + mw)'''
        self.current_phi = np.clip(
            self.current_phi + np.sign(new_phi.delta)*2, 0, self.s1-1)
        self.scalePhi.set(self.current_phi)
        self.lbl_scale_phi.configure(
            text='Azimuth ({:02d})'.format(self.current_phi))
        self.update_canvas()
        
    def update_canvas(self):
        '''Updateds micrograph and corresponding histogram'''
        try: # Not sure why but this gets called after deleting a dataset.
            self.data
        except:
            return 0
        
        ### Select micrograph to be displayed
        self.micrograph = self.data[:,:,self.current_the,self.current_phi]

        ### Update micrograph in canvas
        self.canvas.show_rgba(
            self.micrograph, mode='P', aspect_ratio=(self.ymax, self.xmax))
        
        ### Update histogram plot
        fig, ax = plt.subplots(figsize=(4,4), dpi=200)
        sns.distplot(self.micrograph.ravel(), kde=False, bins=16, ax=ax)
        ax.set_xlim(0,255)
        ax.set_yticks([])
        plt.close()
        
        histo = FigureCanvasAgg(fig)
        s, (width, height) = histo.print_to_buffer()
        histo = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        self.host_canvas.show_rgba(histo)
        