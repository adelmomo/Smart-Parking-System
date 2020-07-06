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
import skimage.morphology
from skimage.measure import regionprops
import matplotlib.patches as patches
from skimage.filters import threshold_otsu, threshold_adaptive
import digit_recog as dr
import digitdetection as dd
import scipy as sp
from skimage.filters import threshold_otsu, threshold_adaptive
from scipy import ndimage
n=100
m=100
c=1
def label_conn(im_b):
        im=im_b
        print(im.shape)
        plt.subplot(3,3,2)
        plt.imshow(im,cmap='gray')
        plt.axis('off')
        #if(im.shape[1]>=850):
        #       im=im[:,int(im.shape[1]/2):]
        if(im.shape[1]>=500):
                im=im[:,int(im.shape[1]/2):]
        #plt.subplot(3,3,2)
        #plt.imshow(im,cmap='gray')
        #plt.axis('off')
        im = cv2.blur(im,(5,5))
        kernel = np.ones((5,5),np.uint8)
        #im=cv2.resize(im,(1000,1000))
        #cv2.imwrite(os.path.join('','plate1.jpg'),im)
        r=im[:,:,0]
        gr=im[:,:,1]
        b=im[:,:,2]
        #rg=cv2.threshold(r,0.29*255, 255, cv2.THRESH_BINARY)
        #grg=cv2.threshold(gr,0.29*255, 255, cv2.THRESH_BINARY)
        #bg=cv2.threshold(b,0.28*255, 255, cv2.THRESH_BINARY)
        #im_bw=rg[1]&grg[1]&bg[1]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #im_bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
         #   cv2.THRESH_BINARY,11,2)
        thresh, im_bw= cv2.threshold(gray,0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       # im_bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #   cv2.THRESH_BINARY,11,2)
        #im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
        #im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
        #im_bw=cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel,iterations=1)
        #im_bw=cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel,iterations=1)
        #mask=np.zeros((im_bw.shape[0]+2,im_bw.shape[1]+2),np.uint8)
        #cv2.floodFill(im_bw,mask,(0,0),1)        
        #ax.imshow(gray, cmap=plt.cm.gray)
        #selem=square(5)
        ret, label = cv2.connectedComponents(~im_bw)
        fig=plt.subplot(3,3,3)
        plt.imshow(~im_bw.reshape(im_bw.shape[0],im_bw.shape[1]),cmap='gray')
        plt.axis('off')
        mostlydigits=[]
        w=80
        h=150
        for region in regionprops(label):
            if(region.area>50):
                        min_row, min_col, max_row, max_col = region.bbox
                        #rectBorder = patches.Rectangle(region.bbox, edgecolor="red", linewidth=2, fill=False)
                        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
                        fig.add_patch(rectBorder)
                        img=gray[min_row:max_row,min_col:max_col]
                        img = cv2.resize(img,(n,m))
                        img=img/255
                        img=img.reshape(n,m,1)
                        ans=dd.model.predict_proba([[img]])
                        res=ans[0][0]
                        ans=np.argmax(ans)
                        if(ans==0):
                            mostlydigits.append(tuple((res,tuple((min_col,img)))))
                            
        
        ans=""
        if(len(mostlydigits)>0):
                mostlydigits.sort()
                mostlydigits.reverse()
        sorteddigits=[]
        j=0
        for i in mostlydigits:
            if(j>=6):
                    break
            sorteddigits.append(i[1])
            j+=1
        if(len(sorteddigits)>0):
                sorteddigits.sort()
        j=4
        for i in sorteddigits:
                digit=i[1]    
                ans+=str(np.argmax(dr.model.predict_proba([[digit]])))
                plt.subplot(3,3,j)
                j+=1
                plt.imshow(i[1].reshape(i[1].shape[0],i[1].shape[1]), cmap="gray")
                plt.axis('off')
        return ans



