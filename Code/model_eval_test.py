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

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import utils_test
from constants import TEST_PHASE
from flags import FLAGS






#############################################################################################

def gpu_dynamic_growth_activation():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras




def _main():
    
    #allow memory on gpu to grow dynamically
    gpu_dynamic_growth_activation()
    
    K.clear_session()
  
    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)
    
    test_datagen = utils_test.CaroloDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
            'C:/Users/user/Desktop/BA/BA',
            color_mode='grayscale',
            batch_size=32
            )
    
   # json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    
    model = utils_test.jsonToModel('C:/Users/user/Desktop/BA/BA/ba-cnn-steering/model_DroNet/model_struct.json')
    
    
    #weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    try:
        model.load_weights('C:/Users/user/Desktop/BA/BA/ba-cnn-steering/model_DroNet/best_weights.h5')
        #print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")
    
    
    model.compile(loss='mse',optimizer='adam')

    
    
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / 32))#batch size = 32

    #compute the predictions for all batches (nb_batches) 
    predictions, ground_truth, t = utils_test.generate_pred_and_gt(
            model, test_generator, nb_batches)
    
    pred_truth_compare =[]
    
    real_steerings = []
    predicted_steerings = []
    
    
    
#    DroNet yields negative values for right turn and positive for left turn.
#    For Carolo its the other way round.
#    So Carolo Values have to be adjusted for that.
    for data in ground_truth:
        real_steerings.append(utils_test.switch_sign(data[0]))
    for i,data in enumerate(predictions):
        predicted_steerings.append(data[0])
        pred_truth_compare.append("%s|||%s" % (data[0],real_steerings[i]))
        
        
    joined = "\n".join(pred_truth_compare)
    #print(AlleTupel)
    
    utils_test.make_and_save_histograms(predicted_steerings,real_steerings)
    
    f = open("C:/Users/user/Desktop/BA/BA/test_compare.txt", "w")
    f.write(joined)
    f.close() 

    
    ###########################################################################

    # ************************* Steering evaluation ***************************
#   
#    # Predicted and real steerings
#    pred_steerings = predictions[t_mask,0]
#    real_steerings = ground_truth[t_mask,0]
#
#    # Compute random and constant baselines for steerings
#    random_steerings = random_regression_baseline(real_steerings)
#    constant_steerings= constant_baseline(real_steerings)
#
#    # Create dictionary with filenames
#    dict_fname = {'test_regression.json': pred_steerings,
#                  'random_regression.json': random_steerings,
#                  'constant_regression.json': constant_steerings}
#
#    # Evaluate predictions: EVA, residuals, and highest errors
#    for fname, pred in dict_fname.items():
#        abs_fname = os.path.join(FLAGS.experiment_rootdir, fname)
#        evaluate_regression(pred, real_steerings, abs_fname)
#
#    # Write predicted and real steerings
#    dict_test = {'pred_steerings': pred_steerings.tolist(),
#                 'real_steerings': real_steerings.tolist()}
#    utils_test.write_to_file(dict_test,os.path.join(FLAGS.experiment_rootdir,
#                                               'predicted_and_real_steerings.json'))
    
   ########################################################################################
    
def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)

