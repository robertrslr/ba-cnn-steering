# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:32:50 2018

@author: RR
"""
import gflags 

FLAGS = gflags.FLAGS

# Files
gflags.DEFINE_string('experiment_rootdir', "./model", 'Folder '
                     ' containing all the logs, model weights and results')
gflags.DEFINE_string('train_dir', "../training", 'Folder containing'
                     ' training experiments')
gflags.DEFINE_string('val_dir', "../validation", 'Folder containing'
                     ' validation experiments')
gflags.DEFINE_string('test_dir', "../testing", 'Folder containing'
                     ' testing experiments')

#Model
gflags.DEFINE_string('weights_fname', "model_weights.h5", '(Relative) '
                                          'filename of model weights')

