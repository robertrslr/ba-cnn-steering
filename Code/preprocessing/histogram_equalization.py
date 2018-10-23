# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:22:23 2018

@author: user
"""


import cv2
import numpy as np
import os, os.path
from matplotlib import pyplot as plt
















def main():
    
    imageDir = "C:/Users/user/Desktop/BA/BA/carolo_test_data_full"
    image_path_list = []
    
    #Value by which the greyscale brightness is adjusted, value 
    #is added to the current greyscale value 
    numb = 0
    
    for file in os.listdir(imageDir):
        image_path_list.append(os.path.join(imageDir, file))
        
   
    for imagePath in image_path_list:
        filename = imagePath
        image = cv2.imread(filename,0)
       
        
        split = filename.split('\\')
        
        
        
        #hist = cv2.calcHist([image],[0],None,[256],[0,256])
      
        #plt.hist(image.ravel(),256,[0,256])
        #plt.show()
        
        equ = cv2.equalizeHist(image)
        
        #plt.hist(equ.ravel(),256,[0,256])
        #plt.show()
        
        path_hist= 'C:/Users/user/Desktop/BA/BA/carolo_full_hist_equ'
        
        
        
        cv2.imwrite(os.path.join(path_hist,split[1]),equ)
   
        
        
        
        
        
        #clahe = cv2.createCLAHE()
        #cl1 = clahe.apply(image)
        
        path_clahe = 'C:/Users/user/Desktop/BA/BA/clahe_equalization'
        
        #cv2.imwrite(os.path.join(path_clahe,str(numb)),cl1)
        
        #plt.hist(cl1.ravel(),256,[0.256])
        #plt.show()
        
        #cv2.destroyAllWindows()
   
        
#def adjust_brightness():
    


if __name__ == "__main__":
    main()