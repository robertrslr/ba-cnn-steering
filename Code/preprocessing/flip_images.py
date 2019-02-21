# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:40:26 2019

@author: Jan Robert Rösler
"""


import cv2
import numpy as np
import os, os.path, shutil
from Code import utilities
import seaborn as sns
import random 

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
            cv2.imwrite(os.path.join(filepath_images,new_filename),flipped_image)
            print(os.path.join(filepath_images,new_filename)," written")
            
        elif(steering_angle<1500):
            diff = 1500-steering_angle
            flipped_angle=1500+diff
            filename_split[1] = new_framenumber
            filename_split[3] = str(flipped_angle)
            new_filename = '_'.join(filename_split)
            cv2.imwrite(os.path.join(filepath_images,new_filename),flipped_image)
            print(os.path.join(filepath_images,new_filename)," written")
        elif steering_angle==1500:
            continue
       
        
    
    

    
    
def plot_steering_angle_distribution(steering_file, sample_count):
    """
    Receives a list of steering angles with associated frame id and plots the distribution over the interval -1<angle<1.
    """
    temp_steering = np.loadtxt(steering_file, delimiter='|||')
        
    steering_values = np.zeros((sample_count,),dtype=float)
        
    for i,tupel in enumerate(temp_steering):
          #before loading, the sign is inverted and the data is scaled
          steering_values[i] = utilities.switch_sign(utilities.scale_steering_data(tupel[1]))
    sns.distplot(steering_values,kde=False) 


def super_dirty_val_train_split(image_path,copy_path,sample_count):
    """
    This will be deleted as sone as it has done what it should.
    """
    images = os.listdir(image_path)
    
    validation_ratio = 0.2
    
    validation_count = int(validation_ratio*sample_count)
    print("ValidationCount:",validation_count)
    
    s = set()
    
    n = validation_count + 200
    
    while n>0:
        s.add(random.randrange(0,sample_count))
        n = n - 1
    s = sorted(s)
    l = list(s)
    print(l)
    print(len(l))
    list_index = 0
    for i,image in enumerate(images):
        
        #print("for-index:",i," list-index:",list_index," list-value:",l[list_index])
        if i==l[list_index]:
            im_data = cv2.imread(os.path.join(image_path,image),0)
            cv2.imwrite(os.path.join(copy_path,image),im_data)
            os.remove(os.path.join(image_path,image))
            list_index=list_index+1
            
        
        
        
            
            
        
        
    
    

def main():
    
    image_files = '../../../testData/fullAndFlipped'
  
    steering_file = '../../../carolo_experiment/carolo_images_OUT'
        
    #plot_steering_angle_distribution(os.path.join(steering_file,'steering_labels.txt'),6000)
    #flip_images(image_files,None)
    
    super_dirty_val_train_split(image_files,'../../../testData/fullAndFlippedVal',11361)
    
    


if __name__ == "__main__":
    main()