import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def plot3D(drp, s0, s1, mina=0, maxa=65):  
        fig, ax = plt.subplots(figsize=(6,6))
        u = np.arange(0, 2*np.pi*(1+1/s1), 2*np.pi/s1)
        a = np.pi/2*mina/90
        b = np.pi/2*maxa/90
        v = np.arange(a, b, (b-a)/(s0+1))
        x = np.outer(np.cos(u),np.cos(v))
        y = np.outer(np.sin(u),np.cos(v))
        ax.pcolormesh(x, y, drp, cmap=plt.cm.jet)        
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.axis('off')
        return fig

class MovDRPCan():
    def __init__(self, master, host_canvas, master_object):
        self.host_canvas = host_canvas
        self.master_object = master_object

    def set_init_configuration(self):
        self.host_canvas.set_init_configuration()
    
    def on_import_setup(self):
        self.host_canvas.on_import_setup()
        self.fetch_parent_variables()
    
    def on_close_setup(self):
        self.set_init_configuration()
        self.host_canvas.on_close_setup()
        del self.data, self.s0, self.s1
    
    def fetch_parent_variables(self):
        self.data = self.master_object.data
        self.s0 = self.master_object.s0
        self.s1 = self.master_object.s1
        
    # -------------------------------------------------------------------------
    def update_drp(self):
        x = self.master_object.canvas0.x
        y = self.master_object.canvas0.y
        drp = self.data[y, x].reshape((self.s0, self.s1)).T
        fig = plot3D(drp, self.s0, self.s1)
        plt.close()
        self.drpim = FigureCanvasAgg(fig)
        s, (width, height) = self.drpim.print_to_buffer()
        self.drpim = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        self.host_canvas.show_rgba(self.drpim)        
