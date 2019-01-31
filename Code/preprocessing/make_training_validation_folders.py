# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30

@author: RR
"""

import cv2
import numpy as np
import os, os.path
from matplotlib import pyplot as plt


def histogram_equalization(images_path,training_path,validation_path):
    """
    Take images from a folder and split them into training and validation images (ratio 4/1)
    
    """
    
