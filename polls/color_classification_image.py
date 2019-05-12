#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------

import cv2
from polls.color_recognition_api import color_histogram_feature_extraction
from polls.color_recognition_api import knn_classifier
import os
import os.path
import requests

def color(image_path):
    # read the test image
    source_image = cv2.imread(image_path)
    prediction = 'n.a.'

    # checking whether the training data is ready
    PATH = 'polls/training.data'

    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print ('training data is ready, classifier is loading...')
    else:
        print ('training data is being created...')
        open('polls/training.data', 'w')
        color_histogram_feature_extraction.training()
        print ('training data is ready, classifier is loading...')

    # get the prediction
    color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
    prediction = knn_classifier.main('polls/training.data', 'polls/test.data')
    
    cv2.putText(
        source_image,
        'Prediction: ' + prediction,
        (15, 45),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        200,
        )
    return prediction