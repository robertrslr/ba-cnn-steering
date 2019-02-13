import numpy as np
import cv2

def central_crop(image, crop_heigth = 200, crop_width = 200):
    """
     Crops the image into 200*200 pixels, starting from the top in height and
     from the center in width.

    :param image: the image that will be cropped
    :param crop_height: height of the crop
    :param crop_width:  width of the crop
    :return: cropped image (200*200 pixels)
    """

    half_width = int(image.shape[1] / 2)
    cropped_image = image[0: crop_heigth,
                half_width - int(crop_width / 2):
                half_width + int(crop_width / 2)]
    return cropped_image

def prepare_raw_image(frame):
    """

    :param frame: the raw frame from the camera which needs to be prepared
    (as numpy array)
    :return: prepared image, ready for prediction with network (pixel values normalised
    to be in between 0 and 1
    """
    #make image smaller
    frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

    #compute histogram equalization
    #frame = cv2.equalizeHist(frame)
    
    #crop image to final size
    image = central_crop(frame)
    #reshape to have three dimensions (needed by Keras)
    image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    #normalise values to satisfy 0<=value<=1
    image = np.asarray(image, dtype=np.float32)
    image *= 1./255

    return image




