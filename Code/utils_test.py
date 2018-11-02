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
        self.filenames = []
        self.ground_truth = []
        
        if not os.path.exists(os.path.join(directory,"steering_labels.txt")):
            extract_steering_values_from_img()
        
        self._load_carolo_data(directory)
        

        assert self.samples > 0, 'Keine Daten gefunden'
        
        #self.ground_truth = np.array(self.ground_truth, dtype = K.floatx())
               
        super(CaroloDataIterator, self).__init__(self.samples,
                batch_size, shuffle=None,seed = None)
    
       
    
    
    def _load_carolo_data(self,directory):
        """Laden der Bilddaten mit zugehörigen Steuerdaten (Labels) 
        
            Bilddaten werden in Unterordner carolo_test_data_full" erwartet.
            Steuerdaten werden in Unterordner steering_data" erwartet.
        
        """
        
        ground_truth = dict()
        filenames = dict()
        
        steerings_filename = os.path.join(directory, 
                                          "steering_labels.txt")
        image_filename = os.path.join(directory,
                                      "carolo_images")

        
        #Steuerdaten laden und skalieren
        ground_truth = self.create_steering_dict(steerings_filename)
                    
        filenames = self.create_file_dict(image_filename)
        
        #make dictionaries to lists while making sure,
        #that only ground_truth for existing images 
        #is taken
        for j,ID in enumerate(filenames):
           self.filenames.append(filenames[ID])
           self.ground_truth.append(ground_truth[ID])

    def create_file_dict(self,image_files):
        """Creates a Dictionary of the filenames and their associated 
            frame numbers
        """
        file_dict = dict()
        
        
        for filename in os.listdir(image_files):
           
            filename_split = filename.split('_')
        
            file_dict[int(filename_split[1])] = filename
                           #framenumber         
            self.samples += 1
        
        return file_dict
    
    def create_steering_dict(self,steering_files):
        """Creates a Dictionary of the steering values and their associated
           frame numbers
        """
        
        temp_steering = np.loadtxt(steering_files, delimiter='|||')
        
        steering_dict = dict()
        
        for tupel in temp_steering:
            steering_dict[int(tupel[0])] = scale_steering_data(tupel[1])
        
        return steering_dict
        
    
    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        
        return self._get_batches_of_transformed_samples(index_array)    
    
    
    
    
    def _get_batches_of_transformed_samples(self,index_array):
        """
        Generates a batch of data from images and associated steerings
       
        """
        
        image_dir = os.path.join(self.directory,"carolo_images")
        current_batch_size = index_array.shape[0]
        
                                                 #target size                                               
        batch_x = np.zeros((current_batch_size,)+(200,200,1),
                dtype=K.floatx())
        batch_steer = np.zeros((current_batch_size, 2,),
                dtype=K.floatx())
        batch_coll  = np.zeros((current_batch_size, 2,),
                dtype=K.floatx())
        
        #grayscale = self.color_mode == 'grayscale'
        #possible augmentation:
        
        
        
        
        
        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            
            x = load_img(os.path.join(image_dir, fname))
          
            #adjust brightness option
            x = adjust_brightness(x,100)
            
            #transform not necessary
            #x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

         #Since images are not labelled with collision data,
         #we just load steering data
            batch_steer[i,0] =1.0
            batch_steer[i,1] = self.ground_truth[index_array[i]]
            batch_coll[i] = np.array([1.0, 0.0])


        batch_y = [batch_steer, batch_coll]
        return batch_x, batch_y





def generate_pred_and_gt(model, generator,steps):
    """
    Predictions and associated ground truth are generated from the samples 
    yielded by the generator.
    
    The output of the generator should be in the same format that is accepted
    by the method predict_on_batch.
    """
    steps_done = 0
    all_outs = []
    all_labels = []
    all_ts = []
    
    progbar = Progbar(target = steps)
    
    while steps_done < steps:
        generator_output = next(generator)
    
        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, gt_lab = generator_output
            elif len(generator_output) == 3:
                x, gt_lab, _ = generator_output
            else:
                raise ValueError('output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))
        else:
            raise ValueError('Output not valid for current evaluation')
  
        outs = model.predict_on_batch(x)
        if not isinstance(outs,list):
            outs = [outs]
        if not isinstance(gt_lab,list):
            gt_lab = [gt_lab]
        
        if not all_outs:
            for out in outs:
                all_outs.append([])
        
        if not all_labels:
            for lab in gt_lab:
                all_labels.append([])
                all_ts.append([])
        
        for i, out in enumerate(outs):
            all_outs[i].append(out)
        for i, lab in enumerate(gt_lab):
            all_labels[i].append(lab[:,1])
            all_ts[i].append(lab[:,0])
        
        steps_done+=1
        progbar.update(steps_done)
    
    if steps_done == 1:
        return [out for out in all_outs],[lab for lab in all_labels],np.concatenate(all_ts[0])
    return np.squeeze(np.array([np.concatenate(out) for out in all_outs])).T, \
                          np.array([np.concatenate(lab) for lab in all_labels]).T, \
                          np.concatenate(all_ts[0])




def extract_steering_values_from_img():
    
    raise NotImplementedError

def load_img(file_path):
        
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
def scale_steering_data(carolo_steering_value):
        """
        Carolo-Car steering values (1000 - 2000) are scaled to 
        DroNet steering values (-1,1)
        
        """
        return ((carolo_steering_value-1500)/500) 
    
def get_scaled_steering_data_from_img(image_path):
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
            print(filename.split('_'))
            scaled_steering = scale_steering_data(int(filename_split[5]))
        
            steering_data.append(scaled_steering)
                                 #Lenkwinkel
                                 
        return steering_data  
    

def adjust_brightness(image, brightness_value):
    
    return np.where((255 - image) < brightness_value,255,image+brightness_value)


def switch_sign(value):
    """
    Takes a float and switches its sign.
    0.0 remains 0.0
  
    """
    if value == 0.0:
        return value
    elif value < 0.0:
        return -value
    elif value > 0.0:
        return -value
    
    return 
    


