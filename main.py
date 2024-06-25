import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info messages

import logging
logging.basicConfig(level=logging.ERROR)

import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import argparse
import time
import random
import csv
from datetime import datetime
import glob
from AngleBuffer import AngleBuffer
from detector import gaze_detection
from utils import detect_face, detect_blinks, find_blinkThreshold, plot_blink_threshold

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

RAW_PATH = '/Users/kendranoneman/OneDrive/ecog_data'
#OUT_PATH = '/Users/kendranoneman/OneDrive/eyeTracking_ECoG_results'

#RAW_PATH = '/ix1/pmayo/OneDrive/EyeMovement/Deidentified_DATA'

PATIENTS = sorted([directory for directory in os.listdir(RAW_PATH) if len(directory) == 2 and directory != '.DS_STORE'])

for patient in PATIENTS:
    for seizure in sorted([file for file in os.listdir(os.path.join(RAW_PATH, patient)) if file != '.DS_Store']):
        for folder in sorted([file for file in os.listdir(os.path.join(RAW_PATH, patient, seizure)) if file != '.DS_Store']):
            for video in sorted([file for file in os.listdir(os.path.join(RAW_PATH,patient,seizure,folder)) if file.endswith('.avi')]):
                #shift = [400,50,400,250] #X,Y,WIDTH,HEIGHT
                
                VID_NAME = os.path.join(RAW_PATH,patient,seizure,folder,video)
                #VID_NAME = '/Users/kendranoneman/Downloads/e947efd6-9296-48d5-a50b-f56278bc0946_0000.m4v'
                print(VID_NAME)

                #shift = detect_face(VID_NAME)
                SHIFT = [525, 105, 300, 300]
    
                blinks = find_blinkThreshold(VID_NAME, SHIFT=SHIFT, duration=20, showVideo=1)
                #detect_blinks(VID_NAME, SHIFT=SHIFT, BLINK_THRESHOLDS=[0.51,0.515,0.52,0.525], duration=20, showVideo=1)
                manual_blinks,start_frame = manual_detectBlinks(VID_NAME, SHIFT=SHIFT, duration=20)
                print(manual_blinks)

                plot_blink_threshold(thresholds, average_blinks_array)
                #BLINK_THRESHOLD = find_blinkThreshold(VID_NAME, SHIFT=SHIFT)
                
                #BLINK_THRESHOLD = 0.52
                print(blah)
                print(BLINK_THRESHOLD)

                gaze_detection(VID_NAME=VID_NAME,SHIFT=SHIFT,USER_FACE_WIDTH=150,BLINK_THRESHOLD=BLINK_THRESHOLD,MIN_DETECTION_CONFIDENCE=0.75,MIN_TRACKING_CONFIDENCE=0.75)
                print(blah)
