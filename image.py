#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:55:13 2022

@author: piyushjain
"""

import argparse
import json
import os
import glob
import cv2 
import numpy as np
import matplotlib.pyplot as plt

#Creating a boundary of zeros on all sides
i1 = cv2.imread('/Users/piyushjain/Documents/CVIP Project 1/data/test_img.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(i1)