# importing tkinter module 
from tkinter import * 
from tkinter.ttk import *
  
# creating tkinter window 
root = Tk() 
  
# Progress bar widget 
progress = Progressbar(root, orient = HORIZONTAL, 
              length = 100, mode = 'determinate') 
  
# Function responsible for the updation 
# of the progress bar value 
def bar(): 
    import time
    for i in range(0, 100):
        progress['value'] = i
        root.update_idletasks()
        time.sleep(0.5)

  
  
progress.pack(pady = 10) 
  
bar() 
# infinite loop 
mainloop() 