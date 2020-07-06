import tkinter as tk
import tkinter
from tkinter import Label,Tk,filedialog
from tkinter import IntVar
import cv2
from main import *
import matplotlib.pyplot as plt
from localization import *
import os
import PIL
from PIL import Image
from PIL import ImageTk
import threading
import datetime
import imutils
import cv2
import os
import tensorflow as tf
import urllib3 as url
import tkinter.font as tkFont
from threading import Thread
import mysql.connector
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="sps"
)
mycursor = mydb.cursor()
def barplot(x_data, y_data, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Draw bars, position them in the center of the tick mark on the x-axis
    ax.bar(x_data, y_data, color = '#539caf', align = 'center')
    # Draw error bars to show standard deviation, set ls to 'none'
    # to remove line between points
   # ax.errorbar(x_data, y_data, yerr = error_data, color = '#297083', ls = 'none', lw = 2, capthick = 2)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
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
def show_pie():
        car_number=var.get()
        if(len(car_number)>0):
                _, ax = plt.subplots()
                freq=dict()
                freq['sat']=0
                freq['sun']=0
                freq['mon']=0
                freq['tus']=0
                freq['wen']=0
                freq['th']=0
                freq['fr']=0
                sql='select * from park_records where car_number=%s and park_id=2'
                val=(car_number,)
                mycursor.execute(sql,val)
                res=mycursor.fetchall()
                for i in res:
                        if(i[3].weekday()==0):
                                freq['mon']+=1
                        elif(i[3].weekday()==1):
                                freq['tus']+=1
                        elif(i[3].weekday()==2):
                                freq['wen']+=1
                        elif(i[3].weekday()==3):
                                freq['th']+=1
                        elif(i[3].weekday()==4):
                                freq['fr']+=1
                        elif(i[3].weekday()==5):
                                freq['sat']+=1
                        elif(i[3].weekday()==6):
                                freq['sun']+=1
                days=[]
                labels=[]
                if(freq['sat']>0):
                    days.append(freq['sat'])
                    labels.append('Saturday')
                if(freq['sun']>0):
                    days.append(freq['sun'])
                    labels.append('Sunday')
                if(freq['mon']>0):
                    days.append(freq['mon'])
                    labels.append('Monday')
                if(freq['tus']>0):
                    days.append(freq['tus'])
                    labels.append('Tuseday')
                if(freq['wen']>0):
                    days.append(freq['wen'])
                    labels.append('Wenesday')
                if(freq['th']>0):
                    days.append(freq['th'])
                    labels.append('Thursday')
                if(freq['fr']>0):
                    days.append(freq['fr'])
                    labels.append('Friday')
                ax.pie(days,labels=labels,autopct='%.1f%%')
                plt.title('The Number of Parking in Each Day in This Park')
                plt.show()
def gui():
        global window
        window=Tk()
        font = tkFont.Font(root=window, family='Helvetica', size=18)
        window.geometry("600x600")
        window.title('License Number Recognition')
        global var
        var=tk.StringVar()
        label_name=tk.Label(window,text='Plate Number:',font=font)
        label_name.pack()
        label_name1=tk.Label(window,textvariable=var,font=font)
        label_name1.pack()
        label_person=tk.Label(window,text='Person Name:',font=font)
        label_person.pack()
        global varname
        varname=tk.StringVar()
        label_person1=tk.Label(window,textvariable=varname,font=font)
        label_person1.pack()
        label_date=tk.Label(window,text='Birth_date:',font=font)
        label_date.pack()
        global date
        date=tk.StringVar()
        label_date1=tk.Label(window,textvariable=date,font=font)
        label_date1.pack()
        global cost
        cost=tk.StringVar()
        label_cost=tk.Label(window,text='Bill Cost:',font=font)
        label_cost.pack()
        label_cost1=tk.Label(window,textvariable=cost,font=font)
        label_cost1.pack()
        global can
        can=tkinter.Canvas(window,width=600,height=400)
        can.pack()
       # button=tk.Button(window,width=10,height=2,text="Visualize Date",command=show_pie)
       # button.place(relx=0.4, rely=0.4)
        window.mainloop()
thread=threading.Thread(target=gui,args=())
thread.start()
ur = 'http://192.168.43.1:8080/shot.jpg'
while True:
    http = url.PoolManager()
    r = http.request('GET',ur)
    imgNp = np.array(bytearray(r.data), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    cv2.imshow('Camera',cv2.resize(img,(600,400)))
    q = cv2.waitKey(1)
    if q == ord("q"):
           ans=recognize(img)
           var.set(ans)
           sql='select * from vehicles_info where car_number=%s'
           val=(ans,)
           mycursor.execute(sql,val)
           res=mycursor.fetchall()
           if(len(res)>0):
                   varname.set(res[0][1])
                   date.set(res[0][2])
                   sql='select * from park where id=2'
                   mycursor.execute(sql)
                   price=mycursor.fetchall()
                   price=float(price[0][2])
                   sql='select * from park_records where park_id=2 and cost is %s and car_number=%s'
                   val=(None,ans)
                   mycursor.execute(sql,val)
                   entry=mycursor.fetchall()
                   if(len(entry)>0):
                           var.set('')
                           varname.set('')
                           date.set('')
                           cost.set('')
                           entry=entry[0][3]
                           entry=datetime.datetime.strptime(str(entry),'%Y-%m-%d %H:%M:%S')
                           billcost=(datetime.datetime.now()-entry)
                           billcost=billcost.total_seconds()
                           billcost=(billcost/60)
                           billcost=(billcost/60)*price
                           billcost = float("{0:.2f}".format(billcost))
                           cost.set(str(billcost))
                           sql='update park_records set cost=%s,time_out=%s where car_number=%s and cost is %s'
                           val=(billcost,datetime.datetime.now(),ans,None)
                           mycursor.execute(sql,val)
                           mydb.commit()
                           var.set(ans)
                           varname.set(res[0][1])
                           date.set(res[0][2])
                   else:
                           entry=datetime.datetime.now()
                           sql='insert into park_records (id,park_id,car_number,entry,cost,time_out) values(%s,%s,%s,%s,%s,%s)'
                           val=('','2',ans,datetime.datetime.now(),None,None)
                           mycursor.execute(sql,val)
                           mydb.commit()
                           var.set(ans)
                           varname.set(res[0][1])
                           date.set(res[0][2])
                           cost.set('Unknown')
                   decoded = cv2.imdecode(np.frombuffer(res[0][3], np.uint8), -1)
                   decoded=cv2.resize(decoded,(600,400))
                   height, width, no_channels = decoded.shape
                   #can.delete('all')
                   #can = tkinter.Canvas(window, width = width, height = height)
                   photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(decoded))
                   img=can.create_image(30,30, image=photo, anchor=tkinter.NW)
                   #labels=['Saturday','Sunday','Monday','Tuesday','Wensday','Thursday','Friday']
                   #usages=[20,10,50,30,60,30,5]
                   #y=range(len(labels))
                   #bar(labels,usages,lables,usages,,'The Number Of Parking At Each Day For a Certain User')
                   show_pie()
           else:
                   can.delete('all')
                   var.set('')
                   varname.set('')
                   date.set('')
                   cost.set('')
                   
                   
           plt.show() 
