#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 04:52:08 2022

@author: piyushjain
"""

import cv2
import numpy as np
import pandas as pd


def CCL(image):
    img = image
    s = img.shape
    print(s)
    label = 0
    
    for i in range(0,s[0]-1):
        for j in range(0,s[1]-1):
            currpixel = img[i][j]
            #print(currpixel)
            if currpixel==255:
                V1 = img[i-1][j]
                #print(V1)
                V2 = img[i][j-1]
                if V2>0:   #if left already has a label
                    img[i][j]=V2
                    if V1>0 and V1!=V2: # if upper and left has label but different label i.e not same
                        if img[i][j]==V2:
                            img[i][j]=V1
                elif V1>0 and V2<0: # if left not has label but up has label
                    img[i][j]=V1
                else:
                    label=label+1;
                    img[i][j]=label
                    
    #print(V2)
    
    
    print(img.shape)
    l = np.unique(img)
    print(len(l))
N =k-1
for k in range[1:len(l)]:
    img(img[i][j]==l(k))= N

    
        
                   
                   
                   
return img
    
                
                



name = 'test_img.jpg'
test_img = cv2.imread('./'+name)
img_dim = test_img.shape
arr = np.ndarray(img_dim[0:2])
print(arr)
print(img_dim)
print(range(img_dim[0]-1))

for i in range(0,img_dim[0]-1):
    for j in range(0,img_dim[1]-1):
        
        if ((test_img[i][j][0] > 100) | (test_img[i][j][1] > 100) | (test_img[i][j][2] > 100)):
                arr[i][j] = 0
        else:
            arr[i][j] = 255
array_img = arr
print(array_img.shape)
cv2.imshow('Grayscale',array_img)
cv2.waitKey(0)
  

mustafa = CCL(array_img)