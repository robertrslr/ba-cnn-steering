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
    
    image_files = 'C:/Users/user/Desktop/BA/BA/carolo_test_data_full'
    file_dict = dict()
    for filename in os.listdir(image_files):
           
        filename_split = filename.split('_')
        file_dict[int(filename_split[1])] = filename
    print(file_dict)
        
    
def scale_steering_data(carolo_steering_value):
        """
        Carolo-Car steering values (1000 - 2000) are scaled to 
        DroNet steering values (-1,1)
        
        """
        return ((carolo_steering_value-1500)/500) 

if __name__ == "__main__":
    main()