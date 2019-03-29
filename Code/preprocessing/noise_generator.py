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
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch) 
        noisy = image + gauss 
        print(noisy)
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)        
        noisy = image + image * gauss
        print(noisy)
        return noisy
  

def generate_random_noise(normalised_image):

    rand = randint(0, 4)
    if 2 <= rand <= 4:
        return normalised_image
    elif rand == 0:
        return noisy("gauss", normalised_image)
    elif rand == 1:
        return noisy("speckle", normalised_image)
    else:
        return "Fehler"


def main():
    
    image = cv2.imread('../../saliency/im_186588_98375.835938_1437_1570.png')
    
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
