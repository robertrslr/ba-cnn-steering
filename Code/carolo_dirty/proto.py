"""
Dirty Bosch tryout for single picture live evaluation
"""
import numpy as np
import os
import sys
import glob
import cv2

from pyueye import ueye

sys.path.append("../")
import utils_test

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



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

nRet = ueye.is_ResetToDefault( hCam)
if nRet != ueye.IS_SUCCESS:
    print("is_ResetToDefault ERROR")

nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)

#for monochrome camera models use Y8 mode
m_nColorMode = ueye.IS_CM_MONO8
nBitsPerPixel = ueye.INT(8)
bytes_per_pixel = int(nBitsPerPixel / 8)
print("Monochrome Mode")

#Area of interest can be set here
nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
if nRet != ueye.IS_SUCCESS:
    print("is_AOI ERROR")


width = rectAOI.s32Width
height = rectAOI.s32Height

#gain factor can be set here
#ueye.IS_SET_MASTER_GAIN_FACTOR(100)

# Prints out some information about the camera and the sensor
print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
print("Maximum image width:\t", width)
print("Maximum image height:\t", height)




# Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
if nRet != ueye.IS_SUCCESS:
    print("is_AllocImageMem ERROR")
else:
    # Makes the specified image memory the active memory
    nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_SetImageMem ERROR")
    else:
        # Set the desired color mode
        nRet = ueye.is_SetColorMode(hCam, m_nColorMode)


# Activates the camera's live video mode (free run mode)
nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
if nRet != ueye.IS_SUCCESS:
    print("is_CaptureVideo ERROR")


ueye.is_SetRopEffect(hCam,ueye.IS_SET_ROP_MIRROR_UPDOWN,1,0)
ueye.is_SetRopEffect(hCam,ueye.IS_SET_ROP_MIRROR_LEFTRIGHT,1,0)

# Enables the queue mode for existing image memory sequences
nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
if nRet != ueye.IS_SUCCESS:
    print("is_InquireImageMem ERROR")
else:
    print("Press q to leave the programm")

#--------------------------------------------------------------------------------------------------

#Load Weights and Model

#zero means test phase, trust me
K.set_learning_phase(0)

model = utils_test.jsonToModel('../../best_model_DroNet/model_struct.json')
    
try:
    model.load_weights('../../best_model_DroNet/best_weights.h5')
        #print("Loaded model from {}".format(weights_load_path))
except:
  print("Impossible to find weight path. Returning untrained model")
    

model.compile(loss='mse',optimizer='adam')


    
#model.predict(picture,batchsize = 1)

#--------------------------------------------------------------------------------------------------
#equ_conadj = np.zeros((height.value, width.value, 1))

# Continuous prediction and image display
while (nRet == ueye.IS_SUCCESS):

    # In order to display the image in an OpenCV window we need to...
    # ...extract the data of our image memory
    array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)

    #equ = cv2.equalizeHist(array)

    bytes_per_pixel = int(nBitsPerPixel / 8)

    # ...reshape it in an numpy array...
    frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
    #frame_equ = np.reshape(equ, (height.value, width.value, bytes_per_pixel))

    # ...resize the image by a half
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # frame_equ = cv2.resize(equ,(0,0),fx=0.5, fy = 0.5)

    # crop the image into 200*200 pixels, starting from the top in height and
    # from the middle in width
    equ_conadj_cut = frame
    half_the_width = int(equ_conadj_cut.shape[1] / 2)
    equ_conadj_cut = equ_conadj_cut[0: 200,
              half_the_width - int(200 / 2):
              half_the_width + int(200 / 2)]


    cv2.imshow("image", equ_conadj_cut)

    #now, lets do the actual magic
    #convert to numpy array
    working_img = np.asarray(equ_conadj_cut, dtype=np.float32)
    #normalise

    cv2.normalize(working_img,working_img_norm)

    # model.predict(picture,batchsize = 1)


    # Press q if you want to end the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# ---------------------------------------------------------------------------------------------------------------------------------------

# Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)

# Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
ueye.is_ExitCamera(hCam)

# Destroys the OpenCv windows
cv2.destroyAllWindows()

print()
print("END")

    


    