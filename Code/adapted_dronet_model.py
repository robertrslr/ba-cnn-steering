# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:36:50 2018

@author: user
"""

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers

#
# DroNet model "resnet8" architecture, which is used   
# for training with the carolo vehicle image data.
# 
# The first two residual Blocks are frozen from training,
# only the weights in the last block are optimized during training.
#
#
#

def partly_frozen_resnet8(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    
    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    
    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])
    
    #create seperate 
    frozen_model = Model(inputs=[img_input], outputs =[x5])
    
    for layer in frozen_model.layers:
        layer.trainable = False
        
    xconnect = frozen_model.output

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(xconnect)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(xconnect)
    x7 = add([x5, x6])

    x = Flatten()(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Steering channel
    steer = Dense(output_dim)(x)

    # Collision channel
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[frozen_model.input], outputs=[steer])
    
    print(model.summary())

    return model