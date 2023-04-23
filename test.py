#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 21:28:22 2022

@author: piyushjain
"""
import argparse
import json
import os
import glob
import cv2 
import numpy as np
import matplotlib.pyplot as plt



img1 = cv2.imread('./data/characters/dot.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/characters/c.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('./data/characters/a.jpg', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('./data/characters/2.jpg', cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread('./data/characters/e.jpg', cv2.IMREAD_GRAYSCALE)


test = []
image = [img1 , img2 , img3 , img4 , img5]
for i in range(len(image)):
  features = np.resize(image[i], (25*25))
  if features.all() >= 250:
      print(' ')
  else:
      print(features)  
  test.append(features)
  print(test)
  plt.imshow(image[i])
  plt.show()
  #print(image[i])




  

