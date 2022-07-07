# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:43:40 2022

@author: HC05
"""
 

# simple colour balance correction taken from https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
import os, fnmatch
import cv2
import math
import numpy as np
import sys
import tkinter as tk
from math import *
from PIL import Image
import io

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        print ("Lowval: ", low_val)
        print ("Highval: ", high_val)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)


# apply this function if you want to view on screen to reduce the image size
def resizeImage(img):
    area = 0.50

    h, w = img.shape[:2]
    root = tk.Tk()
    screen_h = root.winfo_screenheight()
    screen_w = root.winfo_screenwidth()
    vector = sqrt(area)
    window_h = screen_h * vector
    window_w = screen_w * vector
    

    if h > window_h or w > window_w:
        if h / window_h >= w / window_w:
           multiplier = window_h / h
    else:
        multiplier = window_w / w
    img = cv2.resize(img, (0, 0), fx=multiplier, fy=multiplier)
    return (img)


def imgWBcorrection(src,ColourCorrect):
    
    for src, subdir, pho in os.walk(src):                                 # Finds the folders and subfolders
            for p in fnmatch.filter(pho,'*.jpg'):                         # Loops through subfolder
                    print ('Now processing ' + p)                         # Shows which image processing
                    path = os.path.join(src, p)                           # Create image file location by using folder + image
                    img = cv2.imread(path)                                # Read image in CV2 used  to manipulate images
                    corrected = simplest_cb(img, 1)                       # Correction of the white balance
                    path2save = os.path.join(ColourCorrect, p)            # Create path to save image
## CV2 doesnt keep metadata so read it using PIL and write it to the new image

                    imWithEXIF = Image.open(path)                         # Read your original image using PIL/Pillow
                    imWithEXIF.info['exif']                               # Read Exif data and atttach to image
                    corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)# CV2 reads it different to pil so change to match
                    OpenCVImageAsPIL = Image.fromarray(corrected)         # Convert OpenCV image to PIL Image
                    OpenCVImageAsPIL.save(path2save, format='JPEG', exif=imWithEXIF.info['exif']) # Encode newly-created image into memory as JPEG along with EXIF from other imag
                    
                    
                    

#### Use this snippet below to test images ####

# ## For testing 
# img = cv2.imread('C:/Users/hc05/Documents/MPA_IMAGE_PROCESSING/INPUT_IMG/WWBF_CEND0121_WWBF040_STN_159_A1_032.jpg')
# imgRS = resizeImage(img) 
# out = simplest_cb(imgRS, 1)
    
# cv2.imshow("before", imgRS)
# cv2.imshow("after", out)
# cv2.waitKey(0)



















