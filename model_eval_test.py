"""
Created on Fri Oct  5 13:00:06 2018

@author: RR
"""

import gflags
import numpy as np
import os
import sys
import glob
from random import randint
from sklearn import metrics

from keras import backend as K

def _main():

   
    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)

    # Generate testing data
    test_datagen = utils_test.DroneDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(FLAGS.test_dir,
                          shuffle=False,
                          color_mode=FLAGS.img_mode,
                          target_size=(FLAGS.img_width, FLAGS.img_height),
                          crop_size=(FLAGS.crop_img_height, FLAGS.crop_img_width),
                          batch_size = FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils_test.jsonToModel(json_model_path)
    
    # Load weights
    weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")


    # Compile model
    model.compile(loss='mse', optimizer='adam')

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))

    predictions, ground_truth, t = utils_test.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose = 1)

    # Param t. t=1 steering, t=0 collision
    t_mask = t==1


    # ************************* Steering evaluation ***************************
    
    # Predicted and real steerings
    pred_steerings = predictions[t_mask,0]
    real_steerings = ground_truth[t_mask,0]

    # Compute random and constant baselines for steerings
    random_steerings = random_regression_baseline(real_steerings)
    constant_steerings = constant_baseline(real_steerings)

    # Create dictionary with filenames
    dict_fname = {'test_regression.json': pred_steerings,
                  'random_regression.json': random_steerings,
                  'constant_regression.json': constant_steerings}

    # Evaluate predictions: EVA, residuals, and highest errors
    for fname, pred in dict_fname.items():
        abs_fname = os.path.join(FLAGS.experiment_rootdir, fname)
        evaluate_regression(pred, real_steerings, abs_fname)

    # Write predicted and real steerings
    dict_test = {'pred_steerings': pred_steerings.tolist(),
                 'real_steerings': real_steerings.tolist()}
    utils_test.write_to_file(dict_test,os.path.join(FLAGS.experiment_rootdir,
                                               'predicted_and_real_steerings.json'))

