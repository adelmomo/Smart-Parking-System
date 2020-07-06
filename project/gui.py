import tkinter as tk
import tkinter
from tkinter import Label,Tk,filedialog
from tkinter import IntVar
import cv2
from main import *
import matplotlib.pyplot as plt
from localization import *
import os
from PIL import Image
from PIL import ImageTk
import threading
import datetime
import imutils
import cv2
import os
import tensorflow as tf
import urllib3 as url
from tkinter import *
from threading import Thread
def recognize(path):
        plbw,flag=localize(path)
        if(flag!=-1):                
                ans=label_conn(plbw)
                return ans
        else:
                ans=""
                return ans

def openfile():
    var.set("")
    path=filedialog.askopenfilename(filetypes=[("Image File",'.jpg','.png')])
    if(not path):
        return
    else:
        ans=recognize(path)
        var.set(ans)
        plt.show()
window=Tk()
window.geometry("250x250")
window.title('License Plate Recognition')
var=tk.StringVar()
label_name=tk.Label(window,text='Plate Number:')
label_name.grid()
label=tk.Label(window,textvariable=var)
label.grid()
button=tk.Button(window,width=10,height=2,text="Choose",command=openfile)
button.place(relx=0.4, rely=0.4)
window.mainloop()
