"""
Created on Fri Oct  5 13:00:06 2018

@author: RR
"""

import numpy as np
import os
import sys

from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from Code import utilities,constants,plot_evaluation


# Functions to evaluate steering prediction (DroNet)

def explained_variance_1d(ypred, y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression

    !Function imported from DroNet Code.!
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


def compute_explained_variance(predictions, real_values):
    """
    Computes the explained variance of prediction for each
    steering and the average of them
    
    !Function imported from DroNet Code.!
    """
    assert np.all(predictions.shape == real_values.shape)
    ex_variance = explained_variance_1d(predictions, real_values)
    print("EVA = {}".format(ex_variance))
    return ex_variance


def compute_sq_residuals(predictions, real_values):
    """
    !Function imported from DroNet Code.!
    """
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    sr = np.mean(sq_res, axis = -1)
    print("MSE = {}".format(sr))
    return sq_res


def compute_rmse(predictions, real_values):
    """
    !Function imported from DroNet Code.!
    """
    assert np.all(predictions.shape == real_values.shape)
    mse = np.mean(np.square(predictions - real_values))
    rmse = np.sqrt(mse)
    print("RMSE = {}".format(rmse))
    return rmse


def compute_highest_regression_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    
    !Function imported from DroNet Code.!
    """
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    highest_errors = sq_res.argsort()[-n_errors:][::-1]
    return highest_errors


def random_regression_baseline(real_values):
    """
    !Function imported from DroNet Code.!
    """
    mean = np.mean(real_values)
    std = np.std(real_values)
    return np.random.normal(loc=mean, scale=abs(std), size=real_values.shape)


def constant_baseline(real_values):
    """
    !Function imported from DroNet Code.!
    """
    mean = np.mean(real_values)
    return mean * np.ones_like(real_values)


def evaluate_regression(predictions, real_values, fname):
    """
    !Function imported from DroNet Code.!
    """
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    highest_errors = compute_highest_regression_errors(predictions, real_values,
            n_errors=20)
    dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(),
                  "highest_errors": highest_errors.tolist()}
    utilities.write_to_file(dictionary, fname)

#############################################################################################

def gpu_dynamic_growth_activation():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

def _main():
    
    # allow memory on gpu to grow dynamically
    gpu_dynamic_growth_activation()
    
    K.clear_session()
  
    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(constants.TEST_PHASE)
    
    test_datagen = utilities.CaroloDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
            constants.EXPERIMENT_DIRECTORY,
            color_mode='grayscale',
            batch_size=32
            )
    
    
    model = utilities.jsonToModel('C:/Users/user/Desktop/BA/BA/ba-cnn-steering/model_Carolo/model_struct.json')
    
    
    try:
        model.load_weights('C:/Users/user/Desktop/BA/BA/ba-cnn-steering/model_Test/weights_197.h5')
        #print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")
    
    
    model.compile(loss='mse',optimizer='adam')

    n_samples = test_generator.samples
 
    nb_batches = int(np.ceil(n_samples / 32))#batch size = 32

    # compute the predictions for all batches (nb_batches)
    predictions, ground_truth, t = utilities.generate_pred_and_gt(
            model, test_generator, nb_batches)
    
    pred_truth_compare =[]
    
    real_steerings = []
    predicted_steerings = []
    
    #Left/Right shift for Positive/Neagtive Values done while loading the images
    for data in ground_truth:
        real_steerings.append(data[0])
    #for i,data in enumerate(predictions):
    #    predicted_steerings.append(data[0])
     #   pred_truth_compare.append("%s|||%s" % (data[0],real_steerings[i]))
        
        
    joined = "\n".join(pred_truth_compare)
    #print(AlleTupel)
    
    plot_evaluation.make_and_save_histograms(predictions,real_steerings)
    
    #f = open("C:/Users/user/Desktop/BA/BA/test_compare.txt", "w")
    #f.write(joined)
    #f.close() 

    
    ###########################################################################

    # ************************* Steering evaluation ***************************

    predicted_steerings = np.asarray(predictions, dtype=np.float32)
    real_steerings = np.asarray(real_steerings, dtype=np.float32)


    # Compute random and constant baselines for steerings
    random_steerings = random_regression_baseline(real_steerings)
    constant_steerings= constant_baseline(real_steerings)

    # Create dictionary with filenames
    dict_fname = {'test_regression.json': predicted_steerings,
                  'random_regression.json': random_steerings,
                  'constant_regression.json': constant_steerings}

    # Evaluate predictions: EVA, residuals, and highest errors
    for fname, pred in dict_fname.items():
        abs_fname = os.path.join(constants.EXPERIMENT_DIRECTORY, fname)
        evaluate_regression(pred, real_steerings, abs_fname)

    # Write predicted and real steerings
    dict_test = {'pred_steerings': predicted_steerings.tolist(),
                 'real_steerings': real_steerings.tolist()}
    utilities.write_to_file(dict_test,os.path.join(constants.EXPERIMENT_DIRECTORY,
                                               'predicted_and_real_steerings.json'))
    
   ########################################################################################
    
def main(argv):
    # Utility main to load flags
#    try:
#      argv = FLAGS(argv)  # parse flags
#    except gflags.FlagsError:
#      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
#      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)

