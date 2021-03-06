# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:57:12 2019

@author: Jan Robert Rösler
"""



"""
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""

import numpy as np
import os
import cv2
from Code import utilities
from random import randint



def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row, col, ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.9
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch) 
        noisy = image + gauss 
        return np.clip(noisy, 0, 1)
    elif noise_typ =="speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)        
        noisy = image + image * gauss
        return np.clip(noisy, 0, 1)
  

def generate_random_noise(normalised_image):
    """
    Zufällige Noise Generierung für ein Bild.
    3/5 der Fälle bleibt das Bild wie es ist.
    1/5 Gauss-Noise.
    1/5 Speckle-Noise.
    """

    rand = randint(0, 4)
    if 2 <= rand <= 4:
        return normalised_image
    elif rand == 0 or rand == 1:
        return noisy("speckle", normalised_image)
    elif rand == 6:
        return noisy("gauss", normalised_image)
    else:
        return "Fehler"


def main():
    
    image = cv2.imread('../../saliency/im_211017_206746.640625_1515_1570.png')
    
    greyscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    equ_hist_img = utilities.histogram_equalization(greyscale_img)
    
    equ_hist_img_reshaped = np.reshape(equ_hist_img,
                            (equ_hist_img.shape[0], equ_hist_img.shape[1], 1))
    
    equ_hist_img_reshaped =equ_hist_img_reshaped*1./255
    
    cv2.imshow("original",equ_hist_img_reshaped)
    #ACHTUNG! imshow zieht werte zwischen 0 und 1 auf 0 bis 255!
    cv2.waitKey()
    cv2.imshow("gauss", noisy('gauss', equ_hist_img_reshaped))
    cv2.waitKey()
    cv2.imshow("poisson", noisy('poisson', equ_hist_img_reshaped))
    cv2.waitKey()
    cv2.imshow("speckle", noisy('speckle', equ_hist_img_reshaped))
    cv2.waitKey()
    cv2.destroyAllWindows()  
    
if __name__ == "__main__":
    main()
