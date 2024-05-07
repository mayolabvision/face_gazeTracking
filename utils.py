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
import csv
from datetime import datetime
import random
from AngleBuffer import AngleBuffer
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

def detect_face(video_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Load the video
    cap = cv.VideoCapture(video_path)

    # Initialize MediaPipe face detection
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.95) as face_detection:

        # Initialize variables to store the best coordinates
        best_bbox = None
        best_confidence = 0

        # Iterate through each frame in the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Detect faces in the frame
            results = face_detection.process(frame_rgb)

            # If faces are detected
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    # Calculate detection confidence
                    confidence = detection.score[0]  # Accessing the first element of RepeatedScalarFieldContainer
                    # Update best coordinates if current detection confidence is higher
                    if confidence > best_confidence:
                        best_bbox = bbox
                        best_confidence = confidence

            # Display the frame with detected face
            cv.imshow('Face Detection', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture
        cap.release()
        cv.destroyAllWindows()

        # Return the best coordinates
        if best_bbox:
            shift = [best_bbox[0], best_bbox[1], best_bbox[2], best_bbox[3]]
            print("Best shift values (X, Y, WIDTH, HEIGHT):", shift)
            return shift
        else:
            print("No face detected in the video.")
            return None

def find_blinkThreshold(video_path, SHIFT=None, duration=10, showVideo=0):
    MIN_BLINK_THRESHOLD = 0.48  # Minimum blink threshold to start searching
    MAX_BLINK_THRESHOLD = 0.55  # Maximum blink threshold to start searching
    STEP_INITIAL = 0.01  # Step size for the initial search
    STEP_REFINEMENT = 0.002  # Step size for refining the search
    REPEATS_INITIAL = 3  # Number of times to repeat each threshold for initial search
    REPEATS_REFINEMENT = 10  # Number of times to repeat each threshold for refinement

    # First, perform an initial search to narrow down the range
    thresholds = np.arange(MIN_BLINK_THRESHOLD, MAX_BLINK_THRESHOLD + STEP_INITIAL, STEP_INITIAL)
    average_blinks_array = np.zeros(len(thresholds))

    for i, threshold in enumerate(thresholds):
        total_blinks_sum = 0 
        for _ in range(REPEATS_INITIAL):
            total_blinks_sum += detect_blinks(video_path, SHIFT, threshold, duration, showVideo)
        average_blinks = total_blinks_sum / REPEATS_INITIAL
        average_blinks_array[i] = average_blinks
    
    # Round thresholds to three significant figures
    thresholds = np.round(thresholds, decimals=3)
    
    print(average_blinks_array)
    # Find the min and max threshold values based on the initial search
    tolerance = 1e-6  # Tolerance level for float comparison
    non_zero_indices = np.where(np.abs(average_blinks_array) > tolerance)
    if len(non_zero_indices[0]) > 0:
        min_threshold = thresholds[non_zero_indices[0][0]]
        max_threshold = thresholds[non_zero_indices[0][-1]]
    else:
        min_threshold = 0.1 
        max_threshold = 0.9 

    print('-------------------------')
    print(f'Min Threshold: {min_threshold}')
    print(f'Max Threshold: {max_threshold}')
    print('-------------------------')

    # Refine the search within the narrowed range
    thresholds_refined = np.arange(min_threshold - (STEP_INITIAL+STEP_REFINEMENT), max_threshold + (STEP_INITIAL-STEP_REFINEMENT), STEP_REFINEMENT)
    thresholds_refined = np.round(thresholds_refined, decimals=3)
    average_blinks_array_refined = np.zeros(len(thresholds_refined))

    for i, threshold in enumerate(thresholds_refined):
        total_blinks_sum = 0
        for _ in range(REPEATS_REFINEMENT):
            total_blinks_sum += detect_blinks(video_path, SHIFT, threshold, duration, showVideo)
        average_blinks = total_blinks_sum / REPEATS_REFINEMENT
        average_blinks_array_refined[i] = average_blinks
    
        print('-------------------------')
        print(f'Threshold: {threshold} - Average Blinks: {average_blinks_array_refined[i]}')
        print('-------------------------')
        
    return thresholds_refined, average_blinks_array_refined


def plot_blink_threshold(thresholds, average_blinks_array):
    std_err = np.std(average_blinks_array) / np.sqrt(len(average_blinks_array))

    plt.errorbar(thresholds, average_blinks_array, yerr=std_err, fmt='-o', ecolor='black', color='black', capsize=5)
    plt.xlabel('MIN_BLINK_THRESHOLD')
    plt.ylabel('Average Blinks (in 10 sec snippet)')
    plt.title('Average Blinks vs Threshold')
    plt.grid(False)  # Disable grid lines
    plt.show()


def detect_blinks(video_path, SHIFT, BLINK_THRESHOLD, duration, showVideo):
    RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
    LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
    
    MOVING_AVERAGE_WINDOW = 1 # number of frames over which to calculate the moving average for smoothing angles.
    MIN_DETECTION_CONFIDENCE = 0.75
    MIN_TRACKING_CONFIDENCE = 0.75
    PRINT_DATA = 1
    EYE_AR_CONSEC_FRAMES = 2
    TOTAL_BLINKS = 0  # Tracks the total number of blinks detected
    EYES_BLINK_FRAME_COUNTER = (
        0  # Counts the number of consecutive frames with a potential blink
    )
    
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    
    cap = cv.VideoCapture(video_path)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration_frames = int(fps * duration)

    # Randomly select start frame
    start_frame = random.randint(0, max(0, num_frames - duration_frames))

    # Set starting frame position
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)  # Adjust size for smoothing
   
    if showVideo:
        cv.namedWindow("Eye Tracking", cv.WINDOW_NORMAL)

    for _ in range(duration_frames):
        ret, frame = cap.read()
        if SHIFT is not None:
            X, Y, WIDTH, HEIGHT = SHIFT[0], SHIFT[1], SHIFT[2], SHIFT[3]
            frame = frame[Y:Y+HEIGHT, X:X+WIDTH]
        else:
            pass

        if not ret:
            break

        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        l_eq = cv.equalizeHist(l)
        gamma = 1.5
        l_gamma = cv.pow(l_eq / 255.0, gamma) * 255.0
        enhanced_lab = cv.merge((l_gamma.astype(np.uint8), a, b))       
        rgb_frame = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR) 

        img_h, img_w = frame.shape[:2]
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )

            # create the camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
            if eyes_aspect_ratio <= BLINK_THRESHOLD:
                EYES_BLINK_FRAME_COUNTER += 1
                #print(f"BLINK: {eyes_aspect_ratio}")

            else:
                if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                    TOTAL_BLINKS += 1
                EYES_BLINK_FRAME_COUNTER = 0
            
        # Displaying the processed frame
        if showVideo:
            cv.imshow("Eye Tracking", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to quit
                break
        
    # Releasing camera and closing windows
    cap.release()
    if showVideo:
        cv.destroyAllWindows()
        
    return TOTAL_BLINKS

#######################################################################################################################################
def euclidean_distance_3D(points):                                                                                                                                                                                                        
    """Calculates the Euclidean distance between two points in 3D space.

    Args:
        points: A list of 3D points.

    Returns:
        The Euclidean distance between the two points.

        # Comment: This function calculates the Euclidean distance between two points in 3D space.
    """

# Get the three points.
    P0, P3, P4, P5, P8, P11, P12, P13 = points

# Calculate the numerator.
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )

# Calculate the denominator.
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3

# Calculate the distance.
    distance = numerator / denominator

    return distance

# This function calculates the blinking ratio of a person.
def blinking_ratio(landmarks):
    """Calculates the blinking ratio of a person.

    Args:
        landmarks: A facial landmarks in 3D normalized.

    Returns:
        The blinking ratio of the person, between 0 and 1, where 0 is fully open and 1 is fully closed.

    """

    # Get the right eye ratio.
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])

    # Get the left eye ratio.
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])

    # Calculate the blinking ratio.
    ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

    return ratio
