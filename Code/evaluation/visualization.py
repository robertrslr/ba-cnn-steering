# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:50:53 2019

@author: user
"""
import numpy as np
from matplotlib import pyplot as plt

from keras import backend as K

import cv2

from vis import utils
from vis.visualization import visualize_saliency, overlay

from Code import utilities


def visualize_saliency_on_image(raw_image, normalised_image, model, layer_index, filter_indices):
    """
    Shows the saliency map of an image, overlayed over the input image

    :param image:
    :param model:
    :param layer_index:
    :param filter_indices:
    :return: Image with overlayed saliency map
    """
    titles = ['right steering', 'left steering', 'maintain steering']
    modifiers = [None, 'negate', 'small_values']

    for i, modifier in enumerate(modifiers):
        heatmap = visualize_saliency(model, layer_idx=layer_index, filter_indices=filter_indices,
                                     seed_input=normalised_image, grad_modifier=modifier)
        plt.figure()
        plt.title(titles[i])
        plt.imshow(heatmap)
        # alpha-blending heatmap into image
        #plt.imshow(overlay(raw_image, heatmap, alpha=0.7),cmap='gray')
    plt.show()

    def visualize_cam_on_image(raw_image, normalised_image, model, layer_index, filter_indices):
        """
            Shows the grad-cam (Gradient-weighted Class Activation Map) map of an image, overlayed over the input image

            :param image:
            :param model:
            :param layer_index:
            :param filter_indices:
            :return: Image with overlayed saliency map
            """
        titles = ['right steering', 'left steering', 'maintain steering']
        modifiers = [None, 'negate', 'small_values']

        for i, modifier in enumerate(modifiers):
            heatmap = visualize_saliency(model, layer_idx=layer_index, filter_indices=filter_indices,
                                         seed_input=normalised_image, grad_modifier=modifier)
            plt.figure()
            plt.title(titles[i])
            plt.imshow(heatmap)
            # alpha-blending heatmap into image
            # plt.imshow(overlay(raw_image, heatmap, alpha=0.7),cmap='gray')
        plt.show()

def main():

    one_image_batch = np.zeros((1,) + (200, 200, 1),
                               dtype=K.floatx())

    model = utilities.jsonToModel("../../model_Carolo/model_struct.json")

    img = cv2.imread("../../saliency/im_211017_206746.640625_1515_1570.png")

    #Pre Processing the image TODO modularisieren!
    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    equ_hist_img = utilities.histogram_equalization(greyscale_img)

    equ_hist_img_reshaped = np.reshape(equ_hist_img, (equ_hist_img.shape[0], equ_hist_img.shape[1], 1))

    # normalise values to satisfy 0<=value<=1
    normalised_image = np.asarray(equ_hist_img_reshaped, dtype=np.float32)
    normalised_image *= 1. / 255

    #model expects input as batch, so make a one-image-batch
    one_image_batch[0] = normalised_image

    visualize_saliency_on_image(raw_image=equ_hist_img, normalised_image=one_image_batch,
                                model=model, layer_index=15, filter_indices=None)






if __name__ == "__main__":
    main()
