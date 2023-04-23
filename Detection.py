import argparse
import json
import os
import glob
import cv2 
import numpy as np
import matplotlib.pyplot as plt

#Creating a boundary of zeros on all sides
i1 = cv2.imread('./data/more_test_cases_for_project_1.rar', cv2.IMREAD_GRAYSCALE)
plt.imshow(i1)
k1=np.ones((3,3),np.float32)/9
#print(i1)
#print(k1)
#cv2.imshow('original',i1)
#cv2.waitKey(0)
s=i1.shape
#print(s)
k=k1.shape
#print(k)
r=s[0]+k[0]-1
c=s[1]+k[1]-1
z=np.zeros((r,c))
#print(z)
#print(z.shape)
for i in range(s[0]):
    for j in range(s[1]):
        z[i+np.int((k[0]-1)/2),j+np.int((k[1]-1)/2)]=i1[i][j]
    print(z)

#----------------------------------------------------------------------

L=0 #current label value

for i in z:
    for j in z:
        if i <= 10:
            z[i][j]=1
        else:
            z[i][j]=0
        
#ret, z=cv2.threshold(i1,127,255, cv2.IMREAD_GRAYSCALE)
#cv2.imshow('Grayscale Threshold', z)
#print(z)
#--------------------if --------------------------------------------------

#Raster Scan

for i in range(s[0]):
    for j in range(s[1]):
        cp=i1[i][j]
        if cp == 0:
            v1=i1[i-1][j] #upper side pixel
            v2=i1[i][j-1] #left side pixel
            if(v2==0):
                i1[i][j]=v2
                if v1==0 and v1!=v2:
                    i1[i][j]==v1 #{Change v2 and v1 acc to logic}       
            elif(v1==0 and v2>0):
                i1[i][j]=v1
            else:
                L=L+1;
                i1[i][j]=L
                print(L)
 
                    
#----------------------------------------------------------------------

i1=i1[2:s[0],2:s[1]]

#----------------------------------------------------------------------

#rint(img.shape)
'''
l = np.unique(i1)
print(len(l))
N = k-1
for k in range[1:len(l)]:
    i1(i1[i][j]=l(k))= N
'''

'''z[z==0]=0
values=np.unique(z)
for k in range[1:len(l)]:
    z[z==values(k)]=k-1
     
N=k-1
cell = []
x=cell(1,N)
for y in range(1,N):
    x=y.find(z==y)
'''

 