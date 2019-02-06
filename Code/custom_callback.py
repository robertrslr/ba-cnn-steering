# -*- coding: utf-8 -*-
import numpy as np

import keras
from keras import backend as K



class MyCallback(keras.callbacks.Callback):
    """
    Customized callback class.
    
    # Arguments
       filepath: Path to save model.
       period: Frequency in epochs with which model is saved.
       batch_size: Number of images per batch.
    """
    
    def __init__(self, filepath, period, batch_size):
        self.filepath = filepath
        self.period = period
        self.batch_size = batch_size
        
    def on_epoch_end(self, epoch, logs={}):       

        # Hard mining
        sess = K.get_session()
        mse_function = self.batch_size-(self.batch_size-10)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0))))
        self.model.k_mse.load(int(np.round(mse_function)), sess)

