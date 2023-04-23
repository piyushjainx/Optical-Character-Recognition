#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 04:14:32 2022

@author: piyushjain
"""
import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
from turtle import delay
import numpy as np
import cv2
img = cv2.imread(r'/Users/piyushjain/Documents/CVIP Project 1/data/test_img.jpg')

def labeling(image_org) :
    # open file
    f = open('test.txt', mode='w')
    l = open('label.txt',mode='w')
    R = open('new_label.txt',mode='w')

    size = image_org.shape
    img_row = size[0]  # rows
    img_col = size[1]  # columns


    if len(image_org.shape) == 3 :
        binary = np.dot(image_org[...,:3], [0.299, 0.587, 0.114])
    else:
        binary = np.reshape(image_org,(img_row,img_col,1))

    #converting the image to binary
    threshold = 100
    for i in range(img_row):
        for j in range(img_col):
            if binary[i,j] > threshold:
                binary[i,j] = 0
                f.write(str(int(binary[i,j])))
            else:
                binary[i,j] = 1
                f.write(str(int(binary[i,j])))
        f.write('\n')
    image = binary

    # label matrix with same dimension as image to store the component labels
    label = np.ones([img_row,img_col])
    new = 0

    # link array
    link = []
    id = 0 # link index also present object number

    # first pass
    for row in range(img_row):
        for column in range(img_col):
            # for background
            if image[row,column] == [0] :
                label[row, column] = 0
                l.write(str(int(label[row,column])))

            # for foreground or detected characters/components
            else : # check neighbor label
                current_neighbors = [(label[row, column-1]),label[row-1, column]]

                # current is new label
                if current_neighbors == [0,0]:
                    new= new + 1
                    label[row, column] = new
                    l.write(str(int(label[row, column])))

                # when the neighbor already has label
                else :
                    # only one neighbor labeling => choose the large one (the only label)
                    if np.min(current_neighbors) == 0 or current_neighbors[0] == current_neighbors[1]:
                        label[row,column] = np.max(current_neighbors)
                        l.write(str(int(label[row, column])))

                    else:
                        label[row,column] = np.min(current_neighbors)
                        l.write(str(int(label[row, column])))
                        if id == 0:
                            link.append(current_neighbors)
                            id = id + 1

                        else:
                            check = 0
                            for k in range(id) :
                                tmp = set(link[k]).intersection(set(current_neighbors))
                                if len(tmp) != 0 :
                                    link[k] = set(link[k]).union(current_neighbors)
                                    np.array(link)
                                    check = check + 1
                            
                            if check == 0:
                                id = id +1
                                np.array(link)
                                link.append(set(current_neighbors))
        l.write('\n')


    # Second pass for the connected components labeling
    for row in range(img_row):
        for column in range(img_col):
            for x in range(id):
                if (label[row, column] in link[x]) and label[row, column] !=0 :
                    label[row, column] = min(link[x])

    for row in range(img_row):
        for column in range(img_col):
            for x in range(id):
                if (label[row, column] == min(link[x])):
                    label[row, column] = x+1
            R.write(str(int(label[row, column])))
        R.write('\n')
    
    #bounding box
    print(id)
    roix = []
    roiy = []
    for i in range(1,id+1):
        for r in range(img_row):
            for c in range(img_col):
                if label[r][c]==i:
                    roix.append(c)
                    roiy.append(r)
        xmin = min(roix)
        xmax = max(roix)
        ymin = min(roiy)
        ymax = max(roiy)

        print("for index",i, " x ", xmin, xmax)
        print("y ",ymin, ymax)
        

        cv2.rectangle(image_org,(xmin,ymin), (xmax,ymax), (255,0,0),1)

    return label,image_org,id #updated labels, image and link index= number of characters detected


label,image ,id= labeling(img)
print("out ===")
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
l = open('label.txt',mode='r')
# print(l.read())
R = open('new_label.txt',mode='r')
print(R.read())