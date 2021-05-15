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

class FixDRPCan():
    def __init__(self, master, ws, host_canvas, master_canvas, master_object):
        self.host_canvas = host_canvas
        self.canvas = master_canvas
        self.master_object = master_object

    def set_init_configuration(self):
        self.host_canvas.set_init_configuration()
        self.locator = self.canvas.create_oval(0,0,0,0,outline='')

    def on_import_setup(self):
        '''On import, bind print function and pass in relevant data'''
        self.host_canvas.on_import_setup()
        self.fetch_parent_variables()
        self.canvas.bind('<ButtonPress-3>', self.print_drp)
    
    def on_close_setup(self):
        self.canvas.unbind('<ButtonPress-3>')
        self.set_init_configuration()
        self.host_canvas.on_close_setup()
        del self.data, self.rx, self.ry, self.s0, self.s1, self.sc
        
    def fetch_parent_variables(self):
        self.data = self.master_object.data
        self.rx = self.master_object.rx
        self.ry = self.master_object.ry
        self.s0 = self.master_object.s0
        self.s1 = self.master_object.s1
        self.sc = self.master_object.sc

    # -------------------------------------------------------------------------
    def print_drp(self, event):
        '''
        Plots a red circle and shows current DRP.
        '''
        locX, locY = event.x, event.y
        
        x = np.clip(locX/self.sc, 0, self.ry-1).astype('int')
        y = np.clip(locY/self.sc, 0, self.rx-1).astype('int')
        
        # Display locator (red circle)
        self.canvas.delete(self.locator)
        self.locator = self.canvas.create_oval(locX-5, locY-5, locX+5, locY+5, outline='red', width=2,)
        
        # Print DRP in canvas
        drp = self.data[y, x].reshape((self.s0, self.s1)).T
        fig = plot3D(drp, self.s0, self.s1)
        plt.close()
        
        # ### OPTIONAL: PRINT DRP IN CONSOLE
        # fig, ax = plt.subplots(figsize=(8,8), dpi=200)
        # ax.imshow(drp.T, cmap=plt.cm.gray)
        # ax.axis('off')
        # plt.show()
        
        self.drpfixim = FigureCanvasAgg(fig)
        s, (width, height) = self.drpfixim.print_to_buffer()
        self.drpfixim = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        
        self.host_canvas.show_rgba(self.drpfixim)
        self.master_object.nbk.select(0)
