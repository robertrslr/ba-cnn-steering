# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:40:26 2019

@author: Jan Robert Rösler
"""


import cv2
import numpy as np
import os, os.path
from Code import utilities
import seaborn as sns

def flip_images(filepath_images,filepath_save):
    
    
    for i,filename in enumerate(os.listdir(filepath_images)):
           
        image = cv2.imread(filepath_images+"/"+filename,0)
        #horinzontal fl
        flipped_image= cv2.flip(image,1)
        print("Filename: ",filename)
        filename_split = filename.split('_')
        
        framenumber = filename_split[1]
        print("Framenumber:",framenumber)
        steering_angle = int(filename_split[3])
        print("Steering Value: ",steering_angle)
        
        new_framenumber = str(framenumber)+str(i)
        
        if(steering_angle>1500):
            diff = steering_angle-1500
            flipped_angle = 1500-diff
            filename_split[1] = new_framenumber
            filename_split[3] = str(flipped_angle)
            new_filename = '_'.join(filename_split)
            cv2.imwrite(new_filename,flipped_image)
            print(filepath_images+"/"+new_filename," written")
            
        elif(steering_angle<1500):
            diff = 1500-steering_angle
            flipped_angle=1500+diff
            filename_split[1] = new_framenumber
            filename_split[3] = str(flipped_angle)
            new_filename = '_'.join(filename_split)
            cv2.imwrite(new_filename,flipped_image)
            print((filepath_images+"/"+new_filename)," written")
        elif steering_angle==1500:
            continue
        break
        
    
    

    
    
def plot_steering_angle_distribution(steering_file):
    """
    Receives a list of steering angles with associated frame id and plots the distribution over the interval -1<angle<1.
    """
    temp_steering = np.loadtxt(steering_file, delimiter='|||')
        
    steering_values = np.zeros((7000,),dtype=float)
        
    for i,tupel in enumerate(temp_steering):
          #before loading, the sign is inverted and the data is scaled
          steering_values[i] = utilities.switch_sign(utilities.scale_steering_data(tupel[1]))
    sns.distplot(steering_values,kde=False) 



    

def main():
    
    image_files = '../../../test_data/carolo_test_data_full'
  
    steering_file = '../../../carolo_experiment/carolo_images_OUT/steering_labels.txt'
        
    #plot_steering_angle_distribution(steering_file)
    flip_images(image_files,None)


if __name__ == "__main__":
    main()