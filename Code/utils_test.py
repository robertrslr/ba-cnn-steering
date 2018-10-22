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
    """
    Generate minibatches of images and labels with real-time augmentation.
    
    """
    def flow_from_directory(self, directory,
             color_mode='grayscale', batch_size=32,):
        return CaroloDataIterator(directory, self,
                color_mode=color_mode,
                batch_size=batch_size)
               
    
    
    
    
class CaroloDataIterator(Iterator):
    """
    Class for managing data loading.of images and labels

       directory: Path to the root directory to read data from.
       image_data_generator: Image Generator.
       target_size: tuple of integers, dimensions to resize input images to.
       crop_size: tuple of integers, dimensions to crop input images.
       color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
       batch_size: The desired batch size
       shuffle: Whether to shuffle data or not
       seed : numpy seed to shuffle data
       follow_links: Bool, whether to follow symbolic links or not

    # TODO: Add functionality to save images to have a look at the augmentation
    """
    def __init__(self, directory, image_data_generator,color_mode='grayscale',
                 batch_size=32):
       
        self.directory = directory
        self.image_data_generator = image_data_generator
      
        #Since all images from Carolo Car are png, hardcode this
        self.extension = 'png'
        
        if color_mode not in {'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "grayscale".')

        self.samples = 0
        
        
        # Idea = associate each filename with a corresponding steering or label
        self.filenames = dict()
        self.ground_truth = dict()

        self._load_carolo_data(directory)

        assert self.samples >0, 'Keine Daten gefunden'
        
        self.ground_truth = np.array(self.ground_truth, dtype = K.floatx())

       
        super(CaroloDataIterator, self).__init__(self.samples,
                batch_size, shuffle=None)
    
   
    
    
    def load_carolo_data(self,directory):
        """Laden der Bilddaten mit zugehörigen Steuerdaten (Labels) 
        
            Bilddaten werden in Unterordner carolo_test_data_full" erwartet.
            Steuerdaten werden in Unterordner steering_data" erwartet.
        
        """
    
        steerings_filename = os.path.join(directory, 
                                          "steering_data/steering_labels.txt")
        image_filename = os.path.join(directory,
                                      "carolo_test_data_full")

       
        #Steuerdaten laden und skalieren
        self.ground_truth = self.create_steering_dict(steerings_filename)
                    
        self.filenames = self.create_file_dict(image_filename)
        
        
    
    
    def create_file_dict(self,image_files):
        
        file_dict = dict()
        
        
        for filename in os.listdir(image_files):
           
            filename_split = filename.split('_')
        
            file_dict[int(filename_split[1])] = filename
                           #framenumber         
            self.samples += 1
        
        return file_dict
    
    def create_steering_dict(self,steering_file):
        """Creates a Dictionary of the steering values and their associated
           frame number
        """
        
        temp_steering = np.loadtxt(steering_file, delimiter='|||')
        
        steering_dict = dict()
        
        for tupel in temp_steering:
            steering_dict[int(tupel[0])] = scale_steering_data(tupel[1])
        
        return steering_dict
        
        
        
    
    def _get_batches_of_transformed_samples(self,index_array):
        """
        Generates a batch of data from images and associated steerings
       
        """
    
        current_batch_size = index_array.shape[0]
        # Image transformation is not under thread lock, so it can be done in
        # parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape,
                dtype=K.floatx())
        batch_steer = np.zeros((current_batch_size, 2,),
                dtype=K.floatx())
        
        grayscale = self.color_mode == 'grayscale'

        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = img_utils.load_img(os.path.join(self.directory, fname),
                    grayscale=grayscale,)

#            x = self.image_data_generator.random_transform(x)
#            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

         #Since images are not labelled with collision data,
         #we just load steering data
            batch_steer[i,0] =1.0
            batch_steer[i,1] = self.ground_truth[index_array[i]]
            batch_coll[i] = np.array([1.0, 0.0])


        batch_y = [batch_steer, batch_coll]
        return batch_x, batch_y

        
        
def generate_pred_and_gt(model, generator,steps):
    
    raise NotImplementedError
 






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


