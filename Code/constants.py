# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:45:21 2018

@author: user
"""

#Important constants

EXPERIMENT_DIRECTORY = "../../carolo_experiment"


EVALUATION_PATH = "../../evaluation"

BATCH_SIZE = 32

INITIAL_EPOCH = 0

EPOCHS = 100

LOGGING_RATE = 10


RAW_IMAGE = True 


# Image values

CROP_WIDTH = 200
CROP_HEIGHT = 200


#value by which the brightness will be increased
# (if negative, image will be darker)
BRIGHTNESS_ADJUST = 100




# Input
#gflags.DEFINE_integer('img_width', 320, 'Target Image Width')
#gflags.DEFINE_integer('img_height', 240, 'Target Image Height')

#gflags.DEFINE_integer('crop_img_width', 200, 'Cropped image widht')
#gflags.DEFINE_integer('crop_img_height', 200, 'Cropped image height')



TEST_PHASE = 0
TRAIN_PHASE = 1
