# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:36:56 2018

@author: RR
"""


import os
import numpy as np
import json
import cv2
import tensorflow as tf


from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import Progbar
from keras.models import model_from_json

from Code import constants
from Code.preprocessing import noise_generator


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
        

def hard_mining_mse(k):
    """
    Compute MSE for steering evaluation and hard-mining for the current batch.
    # Arguments
        k: number of samples for hard-mining.
    # Returns
        custom_mse: average MSE for the current batch.
        
        !Function imported from DroNet Code.!
    """

    def custom_mse(y_true, y_pred):
        # Parameter t indicates the type of experiment
        t = y_true[:,0]

        # Number of steering samples
        samples_steer = tf.cast(tf.equal(t,1), tf.int32)
        n_samples_steer = tf.reduce_sum(samples_steer)

        if n_samples_steer == 0:
            return 0.0
        else:
            # Predicted and real steerings
            pred_steer = tf.squeeze(y_pred, squeeze_dims=-1)
            true_steer = y_true[:,1]

            # Steering loss
            l_steer = tf.multiply(t, K.square(pred_steer - true_steer))

            # Hard mining
            k_min = tf.minimum(k, n_samples_steer)
            _, indices = tf.nn.top_k(l_steer, k=k_min)
            max_l_steer = tf.gather(l_steer, indices)
            hard_l_steer = tf.divide(tf.reduce_sum(max_l_steer), tf.cast(k,tf.float32))

            return hard_l_steer

    return custom_mse
        
        
        
        
#############EIGENE FUKTIONEN#####################################

class CaroloDataGenerator(ImageDataGenerator):
    """
    Generate minibatches of images and labels with real-time augmentation.
    
    """
    def flow_from_directory(self, directory,
                            shuffle=False,
                            color_mode='grayscale',
                            batch_size=32,):
        return CaroloDataIterator(directory, self,
                                  color_mode=color_mode,
                                  batch_size=batch_size,
                                  shuffle=shuffle)


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
    def __init__(self, directory, image_data_generator, color_mode='grayscale',
                 batch_size=32, shuffle=True, noise=False):
       
        self.directory = directory
        self.image_data_generator = image_data_generator
      
        #Since all images from Carolo Car are png, hardcode this
        self.extension = 'png'
        
        if color_mode not in {'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "grayscale".')
        self.shuffle = shuffle

        if "training" in self.directory:
            self.noise = True
        elif "validation" in self.directory:
            self.noise = False

        self.samples = 0

        # Idea: associate each filename with a corresponding steering or label
        self.filenames = []
        self.ground_truth = []
   
        self._load_carolo_data(directory)
        

        assert self.samples > 0, 'Keine Daten gefunden'
        
        #self.ground_truth = np.array(self.ground_truth, dtype = K.floatx())
               
        super(CaroloDataIterator, self).__init__(self.samples,
                batch_size, self.shuffle, seed=None)
    
       
    
    
    def _load_carolo_data(self,directory):
        """
            Laden der Bilddaten mit zugehörigen Steuerdaten (Labels) 
        
            Bilddaten werden in Unterordner carolo_test_data_full" erwartet.
            Steuerdaten werden in Unterordner steering_data" erwartet.
        
        """
        
        ground_truth = dict()
        filenames = dict()
        
        image_filename = os.path.join(directory,
                                      "images")
        
        steerings_filename = os.path.join(image_filename, 
                                          "steering_labels.txt")
       
        if not os.path.exists(steerings_filename):
            make_steering_list(image_filename)

        
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
            
           Filenames are expected to be in Carolo file format. 
        """
        file_dict = dict()
        
        
        for filename in os.listdir(image_files):
            
            whatever,file_extension = os.path.splitext(filename)
            if file_extension ==".png":
           
                filename_split = filename.split('_')
                file_dict[int(filename_split[1])] = filename
                                   #framenumber       
                self.samples += 1
            else :
                continue
        return file_dict
    
    def create_steering_dict(self,steering_files):
        """Creates a Dictionary of the steering values and their associated
           frame numbers
           
           Filenames are expected to be in Carolo file format. 
        """
        
        temp_steering = np.loadtxt(steering_files, delimiter='|||')
        
        steering_dict = dict()
        
        for tupel in temp_steering:
            #before loading, the sign is inverted and the data is scaled
            steering_dict[int(tupel[0])] = switch_sign(scale_steering_data(tupel[1]))
            #print("scaled and inverted: ",steering_dict[int(tupel[0])])
        return steering_dict
        
    
    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        
        return self._get_batches_of_transformed_samples(index_array)    
    
    
    
    
    def _get_batches_of_transformed_samples(self,index_array):
        """
        Generates a batch of data from images and associated steerings
       
        """
        
        image_dir = os.path.join(self.directory,"images")
        current_batch_size = index_array.shape[0]
        
        #target size
        batch_x = np.zeros((current_batch_size,)+(200,200,1),
                dtype=K.floatx())
        batch_steer = np.zeros((current_batch_size, 2,),
                dtype=K.floatx())
        batch_coll  = np.zeros((current_batch_size, 2,),
                dtype=K.floatx())
    
        #possible augmentation:
        
        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            
            x = load_img(os.path.join(image_dir, fname), do_hist=False,raw_image=constants.RAW_IMAGE)
           
            
            #x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            # Noise only applied to training samples
            if self.noise is True:
                x = noise_generator.generate_random_noise(x)
            
            batch_x[i] = x

            #Since images are not labelled with collision data,
            #we just load steering data
            batch_steer[i,0] =1.0
            batch_steer[i,1] = self.ground_truth[index_array[i]]
            batch_coll[i] = np.array([1.0, 0.0])

        batch_y = batch_steer
        #batch_y = [batch_steer, batch_coll]
        
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


def make_steering_list(image_directory):
    """
    Extracts steering angles from images in carolo file format and 
    writes them to "steering_labels.txt" in the same directory.
    
    # Argument 
        image_directory : the directory containing the images 
        which are to included in the steering file
    
    """
    image_path_list = []
    for file in os.listdir(image_directory):
        image_path_list.append(os.path.join(image_directory, file))
    
    tupel =[]
    for imagePath in image_path_list:
            
        filename = imagePath
        
        SplittedFilename = filename.split('_')
        
        #TODO find a way to safely extract image number and steering angle every time
        tupel.append("%s|||%s" % (SplittedFilename[1],SplittedFilename[3]))
                                        #Bildnummer|||Lenkwinkel
                            
                            
    steering_file  = "\n".join(tupel)
    f = open(os.path.join(image_directory,"steering_labels.txt"), "w")
    f.write(steering_file)
    f.close() 
     

def load_img(file_path, do_hist=False, raw_image=False):
        
        #set grayscale erstmal immer auf true, da nur grayscale images
        grayscale = True

        img = cv2.imread(file_path)
        
        if grayscale:
            if len(img.shape) != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
        if raw_image:
            img = central_image_crop(img, constants.CROP_WIDTH, constants.CROP_HEIGHT)
        
        if do_hist:
            img = histogram_equalization(img)

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
        Retrieves scaled steering data from an Image.
        
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
    
    return np.where((255 - image) < brightness_value, 255, image+brightness_value)


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


def crop_image_height(img, crop_height):
    """
    Image img cropped in height, starting from the top.
    """
    
    img =img[crop_height:img.shape[0],img.shape[1]]
    
    return img


def crop_image_width(img, crop_width):
    """
    Image img cropped in width, starting from the centre to both sides.
    """
    half_the_width = int(img.shape[1] / 2)
    img = img[img.shape[0],
               half_the_width - int(crop_width / 2):
               half_the_width + int(crop_width / 2)]
               
    return img


def central_image_crop(img, crop_width=200, crop_heigth=200):
    """
    Crop the input image centered in width and starting from the top in height.
    
    # Arguments:
        crop_width: Width of the crop.
        crop_heigth: Height of the crop.
        
    # Returns:
        Cropped image.
    """
    half_the_width = int(img.shape[1] / 2)
    img = img[0: crop_heigth,
              half_the_width - int(crop_width / 2):
              half_the_width + int(crop_width / 2)]
    return img


def histogram_equalization(img, algorithm="normal"):
    """
    Normalises histogram of Image img.
    
    Two possible algorithms : "normal","clahe"
    
    """
    
    if algorithm == "normal":
        equ = cv2.equalizeHist(img)
    elif algorithm == "clahe":
        clahe = cv2.createCLAHE()
        equ = clahe.apply(img)
        
    return equ

