# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 17:48:56 2018

@author: Robert
"""
#Fahrzeugaufnahmen werden zugeschnitten und auf 200*200 Pixel
#verkleinert 



import cv2
import numpy as np
import os, os.path

imageDir = "C:/Users/Robert/Desktop/BA/Fahrtbilder 10.7/TEST/"
image_path_list = []

for file in os.listdir(imageDir):
    image_path_list.append(os.path.join(imageDir, file))
    

for imagePath in image_path_list:
    filename = imagePath
    image = cv2.imread(filename,0)
    print(filename)
    #if image is None:
    #print("Unable to open "+ filename)
    #exit(-1)
    cropped = image[0:240,256:496]
    dim = (200,200)
    resized = cv2.resize(cropped,dim,interpolation = cv2.INTER_AREA)
    cv2.imwrite(filename,resized)
   
