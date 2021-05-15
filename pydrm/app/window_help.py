import os
import pathlib
import tkinter as tk

class HelpWindow(tk.Toplevel):
    '''
    Toplevel window for displaying the help pannel
    '''
    def __init__(self, master):
        tk.Toplevel.__init__(self, 
                             master, 
                             # width=width,
                             # height=height,
                             )
        
        self.title('Documentation')
        self.resizable(False, False)
        
        doc_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            'static/doc.txt'
        )
        with open(doc_path, 'r') as f:
            doc_text = f.read()
        msg = tk.Message(self, text=doc_text, bg='white')
        msg.pack()  
        
        