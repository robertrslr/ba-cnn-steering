# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:50:53 2019

@author: user
"""
import numpy as np
from matplotlib import pyplot as plt
import os

from keras import backend as K

import cv2

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from skimage import data, color, io, img_as_float

from vis.visualization import visualize_saliency, visualize_cam, visualize_activation

from Code import utilities


def visualize_attention_on_image(raw_image, normalised_image,
                                 model, layer_index,
                                 filter_indices, type="saliency"):
    """
    Shows the saliency map of an image, overlayed over the input image

    ATTENTION: seems so to be influenced by the learning phase,
    learning phase has to be 1(for learning)

    :param image:
    :param model:
    :param layer_index:
    :param filter_indices:
    :return: Image with overlayed saliency map
    """
    titles = ['left steering', 'right steering']
    modifiers = [None, 'negate']

    for i, modifier in enumerate(modifiers):

        if type == "saliency":
            heatmap = visualize_saliency(model, layer_idx=layer_index,
                                         filter_indices=filter_indices,
                                         seed_input=normalised_image,
                                         grad_modifier=modifier)
        elif type == "cam":
            heatmap = visualize_cam(model, layer_idx=layer_index,
                                    filter_indices=filter_indices,
                                    seed_input=normalised_image,
                                    grad_modifier=modifier)
        elif type == "activation":
            heatmap = visualize_activation(model, layer_idx=layer_index,
                                           filter_indices=filter_indices,
                                           seed_input=normalised_image,
                                           input_range=(0, 1),
                                           grad_modifier=modifier)

        else:
            print("Select 'saliency' or 'cam' as visualization type!")
            break
        
        #plt.figure()
        #plt.title(titles[i])

        #plt.imshow(heatmap)

        overlay_colour_on_greyscale(heatmap, raw_image, titles[i])
        # --------------------------------
        # Open CV (can be commented in)
        # cv2.imshow("heatmap", heatmap)
        # cv2.waitKey()
        # -------------------------------

        # alpha-blending heatmap into image
        # plt.imshow(overlay(raw_image, heatmap, alpha=0.7))
    #plt.title('raw image')
    #plt.imshow(raw_image)
    #plt.show()


def preprocess_image(img):
    """
    Takes an image and returns it ready to be used as input for the network.
    """
    one_image_batch = np.zeros((1,) + (200, 200, 1),
                               dtype=K.floatx())
    # Pre Processing the image TODO modularisieren
    # (ordentliches generisches pre pro)!
    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    equ_hist_img = utilities.histogram_equalization(greyscale_img)

    equ_hist_img_reshaped = np.reshape(equ_hist_img,
                                       (equ_hist_img.shape[0], equ_hist_img.shape[1], 1))

    # normalise values to satisfy 0<=value<=1
    normalised_image = np.asarray(equ_hist_img_reshaped, dtype=np.float32)
    normalised_image *= 1. / 255

    # model expects input as batch, so make a one-image-batch
    one_image_batch[0] = normalised_image

    return one_image_batch, equ_hist_img


def overlay_colour_on_greyscale(image_color, image_greyscale, title="Overlay"):
    """

    Overlays image_colour on image_greyscale and returns the image

    """

    alpha = 1.5

    img = image_greyscale
    rows, cols = img.shape

    color_mask = image_color

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    color_mask = color.gray2rgb(image_color)

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_overlay = color.hsv2rgb(img_hsv)

    # Display the output
    f, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                      subplot_kw={'xticks': [], 'yticks': []},
                                      figsize=(10, 10))
    ax0.imshow(img, cmap=plt.cm.gray)
    ax1.imshow(color_mask)

    ax2.imshow(img_overlay)
    plt.title(title)
    plt.show()

    return img_overlay



def visualize_attention(image_folder_path, model):
    """
    Calls the visualize_attention_on_image method on every image in the folder specified
    by 'folder_path'.
    """
    
    for filename in os.listdir(image_folder_path):
        
        image = cv2.imread(image_folder_path+"/"+filename)
        
        preprocessed_one_image_batch, input_image = preprocess_image(image)

        visualize_attention_on_image(input_image,
                                     normalised_image=preprocessed_one_image_batch,
                                     model=model, layer_index=30,
                                     filter_indices=0, type="cam")

def main():
    # ------------------------------
    # seems to be necessary somehow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    # ----------------------------------

    model = utilities.jsonToModel("../../model_Carolo/model_struct.json")
    model.load_weights("../../model_Test/weights_197.h5")

    visualizationPath = '../../saliency/'
    visualize_attention(visualizationPath, model)

    #img = cv2.imread("../../saliency/im_186588_98375.835938_1437_1570.png")

    


if __name__ == "__main__":
    main()
