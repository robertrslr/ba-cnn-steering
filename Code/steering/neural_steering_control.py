
import sys
from typing import Any, Union

import numpy as np
import cv2
import time
import sys
sys.path.append("../../")

from time import *

from Code import utilities, constants
from keras import backend as K
from Code.camera.ueye_cam import ueye_cam
from Code.camera import carolo_pre_pro as pp
from Code.ipc.uds_socket import uds_socket

from matplotlib import pyplot as plt

from vis.visualization import visualize_saliency, visualize_cam


"""
Initializes a convolutional model, gets a (preprocessed-)picture from the live video capture of the ueye 
camera, predicts a steering value and pushes it in the queue to send it to the cpp steering module.
"""

def load_model_and_weights():
    # zero means test phase, trust me

    #Important to activate before prediction on live images
    K.set_learning_phase(0)

    model = utilities.jsonToModel('../../model_Carolo/model_struct.json')

    try:
        model.load_weights('../../model_Test/weights_148.h5')
        # print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")

    model.compile(loss='mse', optimizer='adam')

    return model


def calc_framerate(current_time, time_last_round):
    """
    :param current_time: current time
    :param time_last_round:
    :return: framerate and the
    """
    framerate = 1 / (current_time - time_last_round)
    time_last_round = current_time

    return framerate, time_last_round

def main():

    # initialize model and weights
    model = load_model_and_weights()

    ueye = ueye_cam()

    ueye.set_pixel_clock(constants.PIXEL_CLOCK)

    ueye.set_frame_rate(constants.FRAMERATE)

    # socket initialisieren und Verbindung aufbauen
    socket = uds_socket()

    one_image_batch = np.zeros((1,) + (200, 200, 1),
                               dtype=K.floatx())

    time_last_round = 0
    while True:

        #get a camera frame fromt the live video capture
        frame = ueye.read()
        #preprocess image
        image = pp.prepare_raw_image(frame)

        #predict function needs image in array form, so we'll give it what it wants
        one_image_batch[0] = image

        #heatmap = visualize_cam(model=model, layer_idx=30, filter_indices=None, seed_input=one_image_batch, grad_modifier=None)

        #cv2.imshow("heatmap", heatmap)
        #cv2.imshow("image", image)

        #t1 = clock()  #----aktuelle Systemzeit = Anfangszeit
        prediction_st = model.predict(one_image_batch, batch_size=1)
        #t2 = clock()  #----aktuelle Systemzeit = Endzeit
        #dt = t2 - t1  #---Zeitdifferenz = Endzeit - Anfangszeit

        #prediction_st = prediction_st_col[0]
        current_time = clock()
        framerate, time_last_round = calc_framerate(current_time, time_last_round)

        print("Prediction:", prediction_st, "Framerate:", int(framerate), end='\r')

        #get data out of nested array structure
        for value in prediction_st:
             socket.send_data(utilities.switch_sign(value[0]))


        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped")
            break

if __name__ == "__main__":
    main()
