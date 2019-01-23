# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:45:21 2018

@author: user
"""

#Important constants

EXPERIMENT_DIRECTORY = "../../carolo_experiment"

#TODO
TRAINING_DIRECTORY = ""
#TODO
VALIDATION_DIRECTORY =  ""


EVALUATION_PATH = "../../evaluation"

BATCH_SIZE = 32

INITIAL_EPOCH = 0

EPOCHS100 = 100

EPOCHS50 = 50

LOGGING_RATE = 10


RAW_IMAGE = True 

# For Carolo always the same
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


TEST_PHASE = 0
TRAIN_PHASE = 1
