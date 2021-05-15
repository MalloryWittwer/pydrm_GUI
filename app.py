from pydrm.app import AppAnalysis
import tkinter as tk

if __name__=='__main__':
    root = tk.Tk()
    root.title('@ AddMe Group -- v1.0')
    root.resizable(False, False)
    app = AppAnalysis(root, size=800)
    root.mainloop()