#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 18:17:09 2022

@author: piyushjain
"""

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
'''
edges = cv2.Canny(img5,100,200)
plt.subplot(121),plt.imshow(img5,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
'''

test = []
image = [img1 , img2 , img3 , img4 , img5]
for i in range(len(image)):
  #features = np.resize(image[i], (25*25))
  edges = cv2.Canny(image[i],100,200)     
  test.append(edges)
  print(test)
  plt.imshow(edges)
  plt.show()
  #plt.imshow(image[i])
#  print(image[i])



'''import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg',0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()'''

  

