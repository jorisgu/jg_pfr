#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 20:53:36 2017

@author: jogue
"""
import os
import caffe
import numpy as np
import yaml
from PIL import Image
import cPickle
import xml.etree.ElementTree as ET
import sklearn.metrics #for confusion matrix
import math #for view generation
from matplotlib import pyplot as plt

def prepareViews(img_w, img_h, ratio_for_stride=0.5, min_size_min=64, \
                 min_size_max=128, min_size_step=16, verbose=False):
    min_size_max +=1
    if verbose:
        print('img_size : ' + str(img_w) + ',' +str(img_h))
    list_min_size = range(min_size_min,min_size_max,min_size_step)
    if verbose:
        print('Looking for these sizes of patch : ' + str(list_min_size))

    views = []
    for min_size in list_min_size:
        stride = int(math.ceil(ratio_for_stride*min_size))
        i_w = 0
        i_h = 0
        if verbose:
            print('Looking for this size of patch : ' + str(min_size))
        while i_w + min_size-1 < img_w:
            i_h = 0
            while i_h + min_size-1 < img_h:
                if verbose:
                    print('Find : ' + str(i_w)+','+ str(i_h))
                # views.append([i_w, i_h, min_size, min_size]) # x y w h
                views.append([i_w, i_h, i_w + min_size-1, i_h + min_size-1]) #x1 y1 x2 y2
                i_h += stride
            i_w += stride
    if verbose:
        print('Number of patchs in this image : ' + str(len(views)))
    return views

    
img = Image.open('/home/jogue/workspace/datasets/ONERA.ROOM.extraits/data/d_raw_DHA_8bits/00011.png')
img = np.array(img, dtype=np.uint8)
print img.shape
plt.figure(1)
plt.imshow(img)
plt.title('img')
plt.show()


if len(img.shape)==3:
    img = img[:,:,::-1]
    img = img.transpose((2,0,1))
elif len(img.shape)==2:
    img = img[np.newaxis, ...]
else:
    print "Error : shape not accepted."
print img.shape
views = prepareViews(img.shape[2], img.shape[1], ratio_for_stride=0.5, \
                     min_size_min=128, min_size_max=128, \
                     min_size_step=64, verbose=False)
print "num of views :",len(views)
chosenView = 0
img_view = img[:,views[chosenView][1]:views[chosenView][3],views[chosenView][0]:views[chosenView][2]]
print img_view.shape
img_view = img_view.transpose((1,2,0))
img_view = img_view [:,:,::-1]
print img_view.shape
plt.figure(2)
plt.imshow(img_view)
plt.title('img view')
plt.show()
