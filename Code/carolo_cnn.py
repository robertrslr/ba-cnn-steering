# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:21:13 2018

@author: RR
"""

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
import os

from keras.callbacks import ModelCheckpoint
from keras import optimizers

from Code import utilities, constants,adapted_dronet_model,custom_callback

from tensorflow.python.client import device_lib


def gpu_dynamic_growth_activation():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

def getModel(img_width, img_height, img_channels, output_dim, weights_path):
    """
    Initialize model. For this, the original architecture is copied layer by layer.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       weights_path: Path to pre-trained model.

    # Returns
       model: A Model instance.
    """
    dronet_model = utilities.jsonToModel('C:/Users/user/Desktop/BA/BA/ba-cnn-steering/model_DroNet/model_struct.json')
    carolo_model = adapted_dronet_model.partly_frozen_resnet8(img_width, img_height, img_channels, output_dim)
    if weights_path:
        try:
            dronet_model.load_weights('C:/Users/user/Desktop/BA/BA/ba-cnn-steering/model_DroNet/best_weights.h5')
            for layerDronet,layerCarolonet in zip(dronet_model.layers,carolo_model.layers):
                layerCarolonet.set_weights(layerDronet.get_weights())
    
            print("Loaded model from {}".format(weights_path))
        except:
            print("Impossible to find weight path. Returning untrained model")

    return carolo_model
    
    
    
    
    
def trainModel(train_data_generator, val_data_generator, model, initial_epoch):
    """
    Model training.

    # Arguments
       train_data_generator: Training data generated batch by batch.
       val_data_generator: Validation data generated batch by batch.
       model: Target image channels.
       initial_epoch: Dimension of model output.
    """


    # Initialize loss weights
    #model.alpha = tf.Variable(1, trainable=False, name='alpha', dtype=tf.float32)
    #model.beta = tf.Variable(0, trainable=False, name='beta', dtype=tf.float32)

    # Initialize number of samples for hard-mining
    model.k_mse = tf.Variable(constants.BATCH_SIZE, trainable=False, name='k_mse', dtype=tf.int32)
    #model.k_entropy = tf.Variable(constants.batch_size, trainable=False, name='k_entropy', dtype=tf.int32)
    
    #COnfigure optimizer with small learning rate for fine tuning
    optimizer = optimizers.Adam(lr=0.0001, decay=1e-5)

    # Configure training process
    model.compile(loss=utilities.hard_mining_mse(model.k_mse),
                        optimizer=optimizer)

    # Save model with the lowest validation loss
    weights_path = os.path.join(constants.EXPERIMENT_DIRECTORY, 'weights_{epoch:03d}.h5')
    writeBestModel = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                     save_best_only=True, save_weights_only=True)

    # Save model every 'log_rate' epochs.
    # Save training and validation losses.
    
    #logz.configure_output_dir(constants.EXPERIMENT_DIRECTORY)
    saveModelAndLoss = custom_callback.MyCallback(filepath=constants.EXPERIMENT_DIRECTORY,
                                            period=10,
                                            batch_size=constants.BATCH_SIZE)

    # Train model
    steps_per_epoch = int(np.ceil(train_data_generator.samples / constants.BATCH_SIZE))
    validation_steps = int(np.ceil(val_data_generator.samples / constants.BATCH_SIZE))

    model.fit_generator(train_data_generator,
                        epochs=constants.EPOCHS150, steps_per_epoch = steps_per_epoch,
                        callbacks=[writeBestModel,saveModelAndLoss],
                        validation_data=val_data_generator,
                        validation_steps = validation_steps,
                        initial_epoch=initial_epoch)
    

    
def main():
    
    #print(device_lib.list_local_devices())
    gpu_dynamic_growth_activation()
    
    # Input image dimensions
    # img_width, img_height = constants.ORIGINAL_IMG_WIDTH, constants.ORIGINAL_IMG_HEIGHT

    # Cropped image dimensions
    crop_img_width, crop_img_height = constants.CROP_WIDTH, constants.CROP_HEIGHT

    # Output dimension (one for steering and one for collision)
    output_dim = 1
    
    img_channels = constants.IMG_CHANNELS

    # Generate training data with real-time augmentation
    train_datagen = utilities.CaroloDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(constants.TRAINING_DIRECTORY,
                                                        shuffle=True,
                                                        color_mode=constants.COLORMODE,
                                                        batch_size = constants.BATCH_SIZE)

    # Generate validation data with real-time augmentation
    val_datagen = utilities.CaroloDataGenerator(rescale=1./255)

    val_generator = val_datagen.flow_from_directory(constants.VALIDATION_DIRECTORY,
                                                    shuffle=True,
                                                    color_mode=constants.COLORMODE,
                                                    batch_size=constants.BATCH_SIZE)


    # Weights to restore
    weights_path = os.path.join(constants.CAROLONET_MODEL_DIRECTORY,
                                constants.CAROLONET_WEIGHTS_FILE)
    initial_epoch = 0
    if not constants.RESTORE_MODEL:
        # In this case weights will start from random
        weights_path = None
    else:
        # In this case weigths will start from the specified model
        initial_epoch = constants.INITIAL_EPOCH

    # Define model
    model = getModel(crop_img_width, crop_img_height, img_channels,
                        output_dim, weights_path)

    # Serialize model into json
    json_model_path = os.path.join(constants.CAROLONET_MODEL_DIRECTORY,
                                   constants.CAROLONET_MODEL_FILE)

    utilities.modelToJson(model, json_model_path)

    # Train model
    trainModel(train_generator, val_generator, model, initial_epoch)
    
    
if __name__ == "__main__":
    main()