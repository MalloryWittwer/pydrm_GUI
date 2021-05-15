import numpy as np
from .host_canvas import HostCanvas

class MicroDisplayCan(HostCanvas):
    def __init__(self, master, width, height,master_object):
        HostCanvas.__init__(self, master=master, width=width, height=height)
        self.width = width
        self.height = height
        self.master_object = master_object
    
    def set_init_configuration(self):
        self.delete('all')
        self.host_image = self.create_image(0, 0, anchor='nw', image=None)
        self.training_areas = []
        self.disable()
    
    def on_import_setup(self):
        self.enable()
        self.fetch_parent_variables()
        self.rectangle = self.create_rectangle(0,0,0,0,outline='')
        self.bind('<Motion>', self.motion)
        self.bind('<ButtonPress-1>', self.on_button_press)
        self.bind('<ButtonRelease-1>', self.on_button_release)        
        self.bind('<B1-Motion>', self.update_rectangle)
        
    def on_close_setup(self):
        self.unbind('<Motion>')
        self.unbind('<ButtonPress-1>')
        self.unbind('<ButtonRelease-1>')        
        self.unbind('<B1-Motion>')
        self.set_init_configuration()
        del self.xmax, self.ymax, self.rx, self.ry, self.sc
    
    def fetch_parent_variables(self):
        self.xmax = self.master_object.xmax
        self.ymax = self.master_object.ymax
        self.rx = self.master_object.rx
        self.ry = self.master_object.ry
        self.sc = self.master_object.sc
    
    # -------------------------------------------------------------------------
    # UPDATING FUNCTIONS ------------------------------------------------------
    def update_color(self, color='red'):
        self.outline = color
    
    def update_rectangle(self, event=None):
        self.endx, self.endy = self.motion(event)
        self.draw_rectangle(self.endx, self.endy)
        
    def draw_rectangle(self, endx, endy):
        self.delete_rectangle()
        self.rectangle = self.create_rectangle(self.startx*self.sc, 
                                               self.starty*self.sc, 
                                               endx*self.sc, 
                                               endy*self.sc, 
                                               outline=self.outline,
                                               width=2)
    
    def delete_rectangle(self):
        self.delete(self.rectangle)

    # ON BUTTON PRESS AND RELEASE ---------------------------------------------
    def on_button_press(self, event):
        self.startx, self.starty = self.motion(event)
        self.endx, self.endy = self.startx, self.starty
        
        flag_release_colors = {0:'red', 1:'blue', 2:'#0090ff', 3:'#ff6600'}
        self.update_color(flag_release_colors[self.master_object.flag_release])
        self.delete_rectangle()
        if self.master_object.frame_segment.segmentation is not None:
            self.master_object.frame_segment.seg_visibility = False
            self.master_object.frame_segment.display_segmentation()
            self.update_color(flag_release_colors[self.master_object.flag_release])

    def on_button_release(self, event):
        self.startx, self.endx = np.sort([self.startx, self.endx])
        self.starty, self.endy = np.sort([self.starty, self.endy])
        if (self.endx-self.startx > 1) & (self.endy-self.starty > 1):
            self.master_object.manage_button_release(self.startx,
                                                     self.starty,
                                                     self.endx,
                                                     self.endy)

    # MOUSE MOVEMENT AND DRP UPDATING FUNCTIONS -------------------------------
    def motion(self, event):
        '''Returns coordinates of the mouse cursor on the canvas'''
        xraw, yraw = (event.x, event.y)
        
        xraw = np.floor(xraw/self.ymax*self.ry).astype('int')
        self.x = np.clip(xraw, 0, self.ry-1)
        
        yraw = np.floor(yraw/self.xmax*self.rx).astype('int')
        self.y = np.clip(yraw, 0, self.rx-1)
        
        if (xraw <= self.ry) & (yraw <= self.rx):
            self.master_object.frame_movdrp.update_drp()
        
        return self.x, self.y
    
    def define_training_areas(self, training_areas):
        '''
        Receives a list of training area coordinates; draws rectangles there.
        '''
        self.delete_training_areas()
        for name, (x0, y0, x1, y1), col, data in training_areas:
            rect = self.create_rectangle(x0*self.sc, 
                                         y0*self.sc, 
                                         x1*self.sc, 
                                         y1*self.sc, outline=col, width=2)
            self.training_areas.append(rect)
    
    def delete_training_areas(self):
        for rect in self.training_areas:
            self.delete(rect)
        self.training_areas = []