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

def convert_to_float(input_str):
    try:
        if input_str.lower() == 'nan':
            return np.nan
        else:
            return float(input_str)
    except ValueError:
        print("Invalid input, please enter a number or 'nan'")
        return None

def find_blinkThreshold(video_path, SHIFT=None, duration=10, showVideo=0):
    MIN_BLINK_THRESHOLD = 0.4  # Minimum blink threshold to start searching
    MAX_BLINK_THRESHOLD = 0.6  # Maximum blink threshold to start searching
    STEP1 = 0.015  # Step size for the initial search
    STEP2 = 0.002  # Step size for refining the search
    STEP3 = 0.001  # Step size for refining the search
    REPEATS1 = 1  # Number of times to repeat each threshold for initial search
    REPEATS2 = 3  # Number of times to repeat each threshold for initial search
    REPEATS3 = 1  # Number of times to repeat each threshold for initial search
    REPEATS_REFINEMENT = 10  # Number of times to repeat each threshold for refinement

    # First, perform an initial search to narrow down the range
    thresholds1 = np.arange(MIN_BLINK_THRESHOLD, MAX_BLINK_THRESHOLD + STEP1, STEP1)
    best_threshes = []
    for _ in range(REPEATS1):
        while True:
            total_blinks = detect_blinks(video_path, SHIFT=SHIFT, BLINK_THRESHOLDS=thresholds1, duration=10, showVideo=showVideo)
            best_thresh = input("Enter the closest blink threshold out of those options: ")
            best_thresh_float = convert_to_float(best_thresh)
            if not np.isnan(best_thresh_float):
                best_threshes.append(best_thresh_float)
                break
            else:
                print("NaN entered. Running detect_blinks again...")
    best_thresh = np.nanmean(best_threshes)
    
    thresholds2 = np.arange(best_thresh-(STEP1+STEP2), best_thresh+(STEP1-STEP2), STEP2)
    best_threshes = []
    for _ in range(REPEATS2):
        while True:
            total_blinks = detect_blinks(video_path, SHIFT=SHIFT, BLINK_THRESHOLDS=thresholds2, duration=10, showVideo=showVideo)
            best_thresh = input("Enter the closest blink threshold out of those options: ")
            best_thresh_float = convert_to_float(best_thresh)
            if not np.isnan(best_thresh_float):
                best_threshes.append(best_thresh_float)
                break
            else:
                print("NaN entered. Running detect_blinks again...")
    best_thresh = np.nanmean(best_threshes)

    thresholds3 = np.arange(best_thresh-(STEP2*2), best_thresh + (STEP2*2), STEP3)
    best_threshes = []
    for _ in range(REPEATS3):
        while True:
            total_blinks = detect_blinks(video_path, SHIFT=SHIFT, BLINK_THRESHOLDS=thresholds3, showVideo=showVideo)
            best_thresh = input("Enter the closest blink threshold out of those options: ")
            best_thresh_float = convert_to_float(best_thresh)
            if not np.isnan(best_thresh_float):
                best_threshes.append(best_thresh_float)
                break
            else:
                print("NaN entered. Running detect_blinks again...")
    best_thresh = np.nanmean(best_threshes)
   
    print(best_thresh)

    print(bt_round1)
    print(blahasdkjfkdl)
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


def detect_blinks(video_path, SHIFT=None, BLINK_THRESHOLDS=[0.5], duration=None, showVideo=0):
    RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
    LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

    MIN_DETECTION_CONFIDENCE = 0.75
    MIN_TRACKING_CONFIDENCE = 0.75
    EYE_AR_CONSEC_FRAMES = 2

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    caps = [cv.VideoCapture(video_path) for _ in BLINK_THRESHOLDS]
    frame_width = int(caps[0].get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv.CAP_PROP_FPS))
    num_frames = int(caps[0].get(cv.CAP_PROP_FRAME_COUNT))
    if duration is None:
        duration_frames = num_frames
    else:
        duration_frames = int(fps * duration)
        
        # Use the same start frame for all videos
        start_frame = random.randint(0, max(0, num_frames - duration_frames))
        for cap in caps:
            cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        
    if SHIFT is not None:
        X, Y, WIDTH, HEIGHT = SHIFT[0], SHIFT[1], SHIFT[2], SHIFT[3]
        new_frame_width = WIDTH
        new_frame_height = HEIGHT
    else:
        new_frame_width = frame_width
        new_frame_height = frame_height

    if showVideo:
        for i, BLINK_THRESHOLD in enumerate(BLINK_THRESHOLDS):
            window_name = f"Blink Threshold: {BLINK_THRESHOLD}"
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            if i<=4:
                cv.moveWindow(window_name, 10 + i * (new_frame_width + 15), 0)
            elif i>=5 and i<=9:
                cv.moveWindow(window_name, 10 + (i-5) * (new_frame_width + 15), 320)
            elif i>=10 and i<=14:
                cv.moveWindow(window_name, 10 + (i-10) * (new_frame_width + 15), 320+320)

    total_blinks_list = [0] * len(BLINK_THRESHOLDS)
    eyes_blink_frame_counters = [0] * len(BLINK_THRESHOLDS)

    for frame_num in range(duration_frames):
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                break
            if SHIFT is not None:
                X, Y, WIDTH, HEIGHT = SHIFT[0], SHIFT[1], SHIFT[2], SHIFT[3]
                frame = frame[Y:Y+HEIGHT, X:X+WIDTH]
            frames.append(frame)

        if len(frames) != len(BLINK_THRESHOLDS):
            break

        for i, (frame, BLINK_THRESHOLD) in enumerate(zip(frames, BLINK_THRESHOLDS)):
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
                    [np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                     for p in results.multi_face_landmarks[0].landmark]
                )

                mesh_points_3D = np.array(
                    [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
                )

                eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
                if eyes_aspect_ratio <= BLINK_THRESHOLD:
                    eyes_blink_frame_counters[i] += 1
                    # Print "BLINK" in the bottom right-hand corner in magenta color
                    cv.putText(frame, "BLINK", (img_w - 100, img_h - 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv.LINE_AA)
                else:
                    if eyes_blink_frame_counters[i] > EYE_AR_CONSEC_FRAMES:
                        total_blinks_list[i] += 1
                    eyes_blink_frame_counters[i] = 0

            # Displaying the processed frame
            if showVideo:
                cv.imshow(f"Blink Threshold: {BLINK_THRESHOLD}", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to quit
            break

    # Releasing camera and closing windows
    for cap in caps:
        cap.release()
    if showVideo:
        cv.destroyAllWindows()

    return total_blinks_list

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
