# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:45:21 2018

Constants (Paths, values and configuratiions)

@author: user
"""

#---------------------GENERAL PATHS----------------------

EXPERIMENT_DIRECTORY = "../../carolo_experiment"

TRAINING_DIRECTORY = "../../training"

VALIDATION_DIRECTORY =  "../../validation"

EVALUATION_PATH = "../../evaluation"

DRONET_MODEL_DIRECTORY = "../model_DroNet"

CAROLONET_MODEL_DIRECTORY = "../model_Carolo"

REPOSITORY_DIRECTORY ="../"

#---------------------NETWORK PATHS---------------------

DRONET_WEIGHTS_FILE = "/best_weights.h5"
DRONET_MODEL_FILE =  "/model_struct.json"

CAROLONET_WEIGHTS_FILE = ""
CAROLONET_MODEL_FILE = ""
#----------------NET WORK CONFIGURATIONS----------------
BATCH_SIZE = 32

INITIAL_EPOCH = 0

EPOCHS100 = 100

EPOCHS50 = 50

LOGGING_RATE = 10


RAW_IMAGE = True 

TEST_PHASE = 0
TRAIN_PHASE = 1


#-----------------IMAGE CONFIGURATIONS-----------------

# For carolo  images always the same
COLORMODE = 'grayscale'
IMG_CHANNELS = 1


#Restore previously trained model for further training/finetuning
RESTORE_MODEL = True

# Image values

ORIGINAL_IMG_HEIGHT = 480 
ORIGINAL_IMG_WIDTH = 752

CROP_WIDTH = 200
CROP_HEIGHT = 200


#value by which the brightness will be increased
# (if negative, image will be darker)
BRIGHTNESS_ADJUST = 100


