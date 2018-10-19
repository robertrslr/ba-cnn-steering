# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:53:37 2018

@author: RR
"""

import cv2
import numpy as np
import os, os.path
from matplotlib import pyplot as plt


#TODO hist_equal bekommt Ordner, berechnet hist ausgleich auf allen bildern in 
#dem ordner und speichert in ornder savepath, 2 verschiedene algorithmen wählbar
def histogram_equalization(image_path,savepath,equalization_type,plot):
    """
    """
    
    
    #imageDir = "C:/Users/user/Desktop/BA/BA/carolo_test_data"
    image_path_list = [] 
    #numb = 0
    
    for file in os.listdir(image_path):
        image_path_list.append(os.path.join(image_path, file))
        
   
    for imagePath in image_path_list:
        filename = imagePath
        image = cv2.imread(filename,0)
        
        
        
        #plt.hist(image.ravel(),256,[0,256])
        #plt.show()
        
        hist_img = cv2.equalizeHist(image)
        
        #plt.hist(equ.ravel(),256,[0,256])
        #plt.show()
        
        #path_hist= 'C:/Users/user/Desktop/BA/BA/histogram_equalization'
        
        
        
        #cv2.imwrite(os.path.join(path_hist,str(numb)+'.jpg'),equ)
   
        #numb = numb +1
        
        
        if equalization_type=="clahe":
            clahe = cv2.createCLAHE()
            hist_img = clahe.apply(image)
        
        #path_clahe = 'C:/Users/user/Desktop/BA/BA/clahe_equalization'
        
        #cv2.imwrite(os.path.join(path_clahe,str(numb)+'.jpg'),cl1)
        return hist_img
       
        