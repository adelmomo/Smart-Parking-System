import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from skimage.morphology import square
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
from plate import *
def localize(path):
        if (type(path) is str):
                im=cv2.imread(path)
        elif (type(path) is np.ndarray):
                im=path
        #im = cv2.blur(im,(5,5))
        #im = cv2.blur(im,(5,5))
        #kernel = np.ones((3,3),np.uint8)
        r=im[:,:,0]
        gr=im[:,:,1]
        b=im[:,:,2]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thresh, im_bw= cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #im_bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
         #  cv2.THRESH_BINARY,11,2)
        #mask=np.zeros((im_bw.shape[0]+2,im_bw.shape[1]+2),np.uint8)
        #cv2.floodFill(im_bw,mask,(0,0),0)    
        #gray = cv2.GaussianBlur(gray,(5,5),0)
        #fig, ax = plt.subplots()
        #ax.imshow(gray, cmap=plt.cm.gray)
        #x=(r>170)&(gr>170)&(b>170)
        #gray[x]=0
        selem=square(5)
        #rg=cv2.threshold(r,0.29*255, 255, cv2.THRESH_BINARY)
        #grg=cv2.threshold(gr,0.29*255, 255, cv2.THRESH_BINARY)
        #bg=cv2.threshold(b,0.28*255, 255, cv2.THRESH_BINARY)
        #bw=(rg[1]&grg[1]&bg[1])
        #bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,kernel,iterations=1)
        #bw=cv2.morphologyEx(bw, cv2.MORPH_CLOSE,kernel,iterations=1)
        #kernel = np.ones((3,3),np.uint8)
        #opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN,kernel,iterations=1000)
        #close=cv2.morphologyEx(opening, cv2.MORPH_CLOSE,kernel,iterations=3)
        #dilated = dilation(binary, selem)
        ret, label = cv2.connectedComponents(im_bw)
        #plt.imshow(dilated)
        fig=plt.subplot(3,3,1)
        plt.imshow(im_bw,cmap='gray')
        #cv2.imwrite('C:\\Users\\Adel\\Desktop\\report\\exp.jpg',~im_bw)
        plt.axis('off')
        ans=-1000000
        for region in regionprops(label):
            if(region.area>1000):
                        min_row, min_col, max_row, max_col = region.bbox
                        #rectBorder = patches.Rectangle(region.bbox, edgecolor="red", linewidth=2, fill=False)
                        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
                        fig.add_patch(rectBorder)
                        img=gray[min_row:max_row,min_col:max_col]
                        img=np.float32(img)
                        img = cv2.resize(img,(n,m))
                        img=img.reshape(n,m,1)
                        img=img/255
                        pro=model.predict_proba([[img]])
                        res=np.argmax(pro)
                        if(res==1):
                                if(ans<pro[0][1]):
                                        ans=pro[0][1]
                                        myplate=gray[min_row:max_row,min_col:max_col]
                                        rgb=im[min_row:max_row,min_col:max_col,:]
        #plt.show()
        if(ans!=-1000000):
                return rgb,1
        else:
                return 0,-1
