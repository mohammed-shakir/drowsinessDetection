import cv2
import numpy as np
import dlib
import time
from math import hypot
import playsound
import tensorflow as tf
import os
from skimage.transform import resize
from keras.utils import np_utils
from keras.preprocessing import image
import pandas as pd
import math
import matplotlib.pyplot as plt

# Use camera number one (webcam)
cap = cv2.VideoCapture(0)

# Libraries
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

eye_ratio = 0
blink = 0
counter = 0
font = cv2.FONT_HERSHEY_SIMPLEX
frameRate = cap.get(2)
imagesDirectory = r'C:\xampp\htdocs\github\gymnasiearbete\drowsiness_detection\image'
modelDirectory = r'C:\xampp\htdocs\github\gymnasiearbete\drowsiness_detection'
categories = ["eyes_closed", "eyes_open"]

# Load Model
model = tf.keras.models.load_model("closed_vs_open.h5")

# Start timer when face is not detected
last_time_face_detected = time.time()

# Start timer on first blink
first_blink = time.time()


def prepare(filepath):
    # Prepare image
    img_size = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


def prepare_all(dir):
    files = os.listdir(dir)
    x_test = []
    y_test = []
    for file in files:
        im = prepare(dir + file)
        x_test.append(im)
        if "closed" in file:
            y_test.append("eyes_closed")
        else:
            y_test.append("eyes_open")
    return x_test, y_test, files


def midpoint(p1, p2):
    # Create new facial landmarks
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def get_eye_ratio(eye_points, facial_landmarks):

    # Print facial landmarks on frame
    left_point = (facial_landmarks.part(
        eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(
        eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(
        eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(
        eye_points[5]), facial_landmarks.part(eye_points[4]))

    # Draw line between different facial landmarks
    cv2.line(frame, left_point, right_point, (0, 255, 0), 2)  # Horizontal line
    cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)  # Vertical line

    # Calculating length of the lines
    hor_line_lenght = hypot(
        (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot(
        (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_lenght/ver_line_lenght
    return ratio


while True:
    # Load frames from the camera
    ret, frame = cap.read()
    # Use gray frame for better tracking
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frameId = cap.get(1)

    faces = detector(gray)

    # If there is no face detected for more than 2 seconds: Print "Face not detected"
    if not faces:
        if ((time.time() - last_time_face_detected) > 2):
            # playsound.playsound('alarm.mp3', True)
            cv2.putText(frame, "Face not detected",
                        (150, 120), font, 1.2, (0, 0, 255))

    # Else, reset timer
    else:
        last_time_face_detected = time.time()

        if (ret != True):
            break

        if (frameId % math.floor(frameRate) == 0):
            os.chdir(imagesDirectory)
            filename = "Image.jpg"
            cv2.imwrite(filename, frame)

        os.chdir(modelDirectory)
        x_test, y_test, test_files = prepare_all("image/")
        for i in range(0, len(x_test)):
            prediction = model.predict(x_test[i])
            print("Pred:", prediction[0][0], "->",
                  categories[1 if prediction[0][0] > 0.5 else 0])

    # Print the face detection coordinates on the frame
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Eye ratio
        left_eye_ratio = get_eye_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_eye_ratio([42, 43, 44, 45, 46, 47], landmarks)
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

    # If the value of "eye_ratio" is less than 5 increase "counter" by 1
    if eye_ratio < 5:
        counter += 1

    else:
        # If the value of "counter" is greater than 0 increase "blink" by 1
        if counter > 0:
            blink += 1

        # Reset counter
        counter = 0

    # Start timer on first blink
    if blink > 0:
        # If the value of "blink" is grater than 20 after 1 minute: print "You are tired"
        if ((time.time() - first_blink) > 60):
            if blink > 20:
                cv2.putText(frame, "You are tired",
                            (190, 240), font, 1.2, (0, 0, 255))

            # Else, restart the timer and set "blink" value to 0
            else:
                first_blink = time.time()
                blink = 0

    # Show camera frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    # Press "c" to drink coffee
    if key == ord("c"):
        blink = 0

    # Press "Escape" to breake while loop
    if key == 27:
        break

# Turn off camera
cap.release()
cv2.destroyAllWindows
