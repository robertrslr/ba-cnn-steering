# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:50:53 2019

@author: user
"""
import numpy as np
from matplotlib import pyplot as plt

from keras import backend as K

import cv2

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from vis.visualization import visualize_saliency, visualize_cam

from Code import utilities


def visualize_attention_on_image(raw_image, normalised_image, model, layer_index, filter_indices,type="saliency"):
    """
    Shows the saliency map of an image, overlayed over the input image

    ATTENTION: seems so to be influenced by the learning phase, learning phase has to be 1(for learning)

    :param image:
    :param model:
    :param layer_index:
    :param filter_indices:
    :return: Image with overlayed saliency map
    """
    titles = ['right steering', 'left steering', 'maintain steering']
    modifiers = [None, 'negate', 'small_values']

    for i, modifier in enumerate(modifiers):

        if type == "saliency":
            heatmap = visualize_saliency(model, layer_idx=layer_index, filter_indices=filter_indices,
                                         seed_input=normalised_image, grad_modifier=modifier)
        elif type == "cam":
            heatmap = visualize_cam(model, layer_idx=layer_index, filter_indices=filter_indices,
                                    seed_input=normalised_image, grad_modifier=modifier)
        else:
            print("Select 'saliency' or 'cam' as visualization type!")
            break

        plt.figure()
        plt.title(titles[i])

        plt.imshow(heatmap)
        
        #--------------------------------
        #Open CV (can be commented in)
        #cv2.imshow("heatmap", heatmap)
        #cv2.waitKey()
        #--------------------------------

        # alpha-blending heatmap into image
        #plt.imshow(overlay(raw_image, heatmap, alpha=0.7))
    plt.title('raw image')
    plt.imshow(raw_image)
    plt.show()

def preprocess_image(img):
    """
    Takes an image and returns it ready to be used as input for the network.
    """
    one_image_batch = np.zeros((1,) + (200, 200, 1),
                               dtype=K.floatx())
    # Pre Processing the image TODO modularisieren (ordentliches generisches pre pro)!
    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    equ_hist_img = utilities.histogram_equalization(greyscale_img)

    equ_hist_img_reshaped = np.reshape(equ_hist_img, (equ_hist_img.shape[0], equ_hist_img.shape[1], 1))

    # normalise values to satisfy 0<=value<=1
    normalised_image = np.asarray(equ_hist_img_reshaped, dtype=np.float32)
    normalised_image *= 1. / 255

    # model expects input as batch, so make a one-image-batch
    one_image_batch[0] = normalised_image
    
    return one_image_batch


def main():

    #------------------------------
    #seems to be necessary somehow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    #----------------------------------

    model = utilities.jsonToModel("../../model_DroNet/model_struct.json")
    model.load_weights("../../model_DroNet/best_weights.h5")
    
    #allow dynamic growth of memory on gpu, otherwise an error occurs
    

    img = cv2.imread("../../saliency/im_249417_286357.750000_1542_1540.png")
    
    preprocessed_one_image_batch = preprocess_image(img)

    visualize_attention_on_image(img,normalised_image=preprocessed_one_image_batch,
                                 model=model, layer_index=30, filter_indices=0, type="cam")


if __name__ == "__main__":
    main()
