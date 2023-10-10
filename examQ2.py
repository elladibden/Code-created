#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:58:29 2023

@author: elladibden
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
import skimage


JWST = plt.imread('JWST_DeepField.jpeg')

plt.imshow(JWST)

grayJWST = skimage.color.rgb2gray(JWST)
plt.figure()
plt.imshow(grayJWST, cmap=cm.gray)

fig1 = plt.figure()   
ax1 = fig1.gca()
partimage =JWST[240:320, 670:770]
graypimage = skimage.color.rgb2gray(partimage)
plt.imshow(graypimage, cmap=cm.gray)                                                    
ax1.set_axis_off()        

plt.figure()
plt.contour(graypimage,10) 
plt.axis('equal')
#%%
HST = plt.imread('HSTspaceslug.jpeg')
plt.figure()
plt.imshow(HST)

grayHST = skimage.color.rgb2gray(HST)
plt.imshow(grayHST, cmap=cm.gray)

plt.figure()
plt.contour(grayHST,10) 
plt.axis('equal')

#%%
fig = plt.figure(figsize=(12,10))
plt.subplots_adjust(wspace=0.4, hspace=0.4, top=0.85)
fig.suptitle('HST and JWST with contour maps', fontsize=18)
ax1 = fig.add_subplot(2, 2, 1) 
ax1.imshow(graypimage, cmap=cm.gray)
ax1.set_title('JWST image')
ax2 = fig.add_subplot(2, 2, 2) 
axc = ax2.imshow(grayHST, cmap=cm.gray)
ax2.set_title('HST image')
ax3 = fig.add_subplot(2, 2, 3) 
ax3.contour(graypimage,10)
ax3.invert_yaxis()
ax3.set_title('JWST 10 Contour map')
ax4 = fig.add_subplot(2, 2, 4) 
ax4.contour(grayHST,10)
ax4.invert_yaxis()
ax4.set_axis_off()
ax4.set_title('HST 10 Contour map')
ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
ax4.set_axis_off()
plt.savefig('examq2fiv.jpg')

