"""
Dirty Bosch tryout for single picture live evaluation
"""
import numpy as np
import os
import sys
import glob

from pyueye import ueye

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import utils_test
from constants import TEST_PHASE

#------------------------------------------------------------------------------------------------
hCam = ueye.HIDS(0) 
sInfo = ueye.SENSORINFO()
cInfo = ueye.CAMINFO()
pcImageMemory = ueye.c_mem_p()
MemID = ueye.int()
rectAOI = ueye.IS_RECT()
pitch = ueye.INT()
nBitsPerPixel = ueye.INT(24)    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
channels = 3                    #3: channels for color mode(RGB); take 1 channel for monochrome
m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
bytes_per_pixel = int(nBitsPerPixel / 8)

#-----------------------------------------------------------------------------------------------

# Starts the driver and establishes the connection to the camera
nRet = ueye.is_InitCamera(hCam, None)
if nRet != ueye.IS_SUCCESS:
    print("is_InitCamera ERROR")


model = utils_test.jsonToModel('PATH TO MODEL')
    
try:
    model.load_weights('PATH TO WEIGHTS')
        #print("Loaded model from {}".format(weights_load_path))
except:
  print("Impossible to find weight path. Returning untrained model")
    
    
model.compile(loss='mse',optimizer='adam')
    
    
    #TODO Ueye picture loading 
    
picture = UEYEWHATEVER    
    
    
    
    
    
model.predict(picture,batchsize = 1)




    


    