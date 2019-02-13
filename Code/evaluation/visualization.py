# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:50:53 2019

@author: user
"""
import numpy as np
from matplotlib import pyplot as plt

import cv2

from vis import utils
from vis.visualization import visualize_saliency, overlay

from Code import utilities


def plot_saliency_on_image(image,model,layer_index,filter_indices):
    """
    Shows the saliency map of an image, overlayed over the image

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
                                     seed_input=image, grad_modifier=modifier)
        plt.figure()
        plt.title(titles[i])
        # alpha blending heatmap into image
        plt.imshow(overlay(image, heatmap, alpha=0.7))


def main():

    model = utilities.jsonToModel("../../model_Carolo/model_struct.json")

    print(model.summary())
    for i, v in enumerate(model.layers):
        print(i, ":",  v, "\n")

    img = cv2.imread("../../saliency/im_211017_206746.640625_1515_1570.png")

    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    equ_hist_img = utilities.histogram_equalization(greyscale_img)

    #cv2.imshow("hist", equ_hist_img)
    #cv2.waitKey()

    plot_saliency_on_image(equ_hist_img, model, layer_index=30, filter_indices=None)






if __name__ == "__main__":
    main()
