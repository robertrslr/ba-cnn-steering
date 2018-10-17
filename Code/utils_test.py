# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:36:56 2018

@author: --
"""



import re
import os
import numpy as np
import tensorflow as tf
import json
import cv2

from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import Progbar
from keras.models import model_from_json








def modelToJson(model, json_model_path):
    """
    Serialize model into json.
    """
    model_json = model.to_json()

    with open(json_model_path,"w") as f:
        f.write(model_json)


def jsonToModel(json_model_path):
    """
    Serialize json into model.
    """
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    return model

def write_to_file(dictionary, fname):
    """
    Writes everything is in a dictionary in json model.
    """
    with open(fname, "w") as f:
        json.dump(dictionary,f)
        print("Written file {}".format(fname))
        
        
        
        
#############EIGENE FUKTIONEN#####################################

class CaroloDataGenerator(ImageDataGenerator):
    
    
    
    def flow_from_direcotry(self,directory):
        return NotImplementedError

    
    def load_img(self,file_path):
        
        #set grayscale erstmal immer auf true, da nur grayscale images
        grayscale = True
        
        img = cv2.imread(file_path)
        if grayscale:
            if len(img.shape) != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        if grayscale:
            img = img.reshape((img.shape[0], img.shape[1], 1))
    
    
        return np.asarray(img, dtype=np.float32)
    
    
    
    def loadCaroloData(self,img_path,steerings_path):
        """Laden der Bilddaten mit zugehörigen Steuerdaten (Labels) 
        
        """
        image_path_list = []
        
        all_img = []
        
        #load images first 
        for file in os.listdir(img_path):
            image_path_list.append(os.path.join(img_path, file))
        
        
        for index,image_path in enumerate(image_path_list):
            all_img.append(self.load_img(image_path))
            
        
        all_img = np.asarray(all_img)
        #now steerings
        steerings = self.get_scaled_steering_data_from_img(steerings_path) 
        
        return all_img, steerings
    
    
    
    
    
            
    # TODO: gleich ganzes array übergeben?
    def scale_steering_data(self,carolo_steering_value):
        """
        Carolo-Car steering values (1000 - 2000) are scaled to 
        DroNet steering values (-1,1)
        
        """
        return ((carolo_steering_value-1500)/500) 
    
    
    def get_scaled_steering_data_from_img(self,image_path):
        """
        
        """
        image_path_list = []
    
        for file in os.listdir(image_path):
            
            image_path_list.append(os.path.join(image_path, file))
        
        steering_data = []
        for imagePath in image_path_list:
            
            filename = imagePath
        
            #image = cv2.imread(filename,0)
            #print(filename)
            filename_split = filename.split('_')
            #print(filename.split('_'))
            scaled_steering = self.scale_steering_data(int(filename_split[5]))
        
            steering_data.append(scaled_steering)
                                 #Lenkwinkel
                                 
        return steering_data  
    
    #TODO Steuerdaten laden und in array speichern
    def load_steerings(steering_path):                        
        """
        """
        raise NotImplementedError


