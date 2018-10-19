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


def main():
    
    imageDir = "C:/Users/user/Desktop/BA/BA/clahe_equalization"
    image_path_list = []
    
    #Value by which the greyscale brightness is adjusted, value 
    #is added to the current greyscale value 
    value = 35
    numb = 0
    
    for file in os.listdir(imageDir):
        image_path_list.append(os.path.join(imageDir, file))
        
   
    for imagePath in image_path_list:
        filename = imagePath
        image = cv2.imread(filename,0)
        print(filename)
        
        
        img_brighter = np.where((255 - image) < value,255,image+value)
        #if image is None:
        #print("Unable to open "+ filename)
        #exit(-1)
        #cropped = image[0:240,200:550]
        #dim = (200,200)
        #resized = cv2.resize(cropped,dim,interpolation = cv2.INTER_AREA)
       
        #print(img_brighter)
        
       
      
        path_bright_adj= 'C:/Users/user/Desktop/BA/BAbrightness_test'
        
        
        
        cv2.imwrite(os.path.join(path_bright_adj,str(numb)+'.jpg'),img_brighter)
   
        numb = numb +1
#def adjust_brightness():
    


if __name__ == "__main__":
    main()