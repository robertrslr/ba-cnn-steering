# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:50:40 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

    
def make_and_save_histograms(pred_steerings, real_steerings,
                             img_name = "histograms.png"):
    """
    Plot and save histograms from predicted steerings and real steerings.
    
    # Arguments
        pred_steerings: List of predicted steerings.
        real_steerings: List of real steerings.
        img_name: Name of the png file to save the figure.
        
        Function imported from DroNet Code.
    """
    pred_steerings = np.array(pred_steerings)
    real_steerings = np.array(real_steerings)
    max_h = np.maximum(np.max(pred_steerings), np.max(real_steerings))
    min_h = np.minimum(np.min(pred_steerings), np.min(real_steerings))
    bins = np.linspace(min_h, max_h, num=50)
    plt.hist(pred_steerings, bins=bins, alpha=0.5, label='Predicted', color='b')
    plt.hist(real_steerings, bins=bins, alpha=0.5, label='Real', color='r')
    plt.title('Steering angle')
    plt.legend(fontsize=10)
    plt.savefig(img_name, bbox_inches='tight')
    
    