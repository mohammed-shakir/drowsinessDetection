import cv2
import tensorflow as tf
import os
from skimage.transform import resize
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
import pandas as pd
import math
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(0)
frameRate = cap.get(2)
x = 1
imagesDirectory = r'C:\xampp\htdocs\github\gymnasiearbete\ml\image'
modelDirectory = r'C:\xampp\htdocs\github\gymnasiearbete\ml'
categories = ["eyes_closed", "eyes_open"]
model = tf.keras.models.load_model("closed_vs_open.h5")


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


while(cap.isOpened()):
    frameId = cap.get(1)
    ret, frame = cap.read()

    if (ret != True):
        break

    if (frameId % math.floor(frameRate) == 0):
        os.chdir(imagesDirectory)
        filename = "Image.jpg"
        cv2.imwrite(filename, frame)

    time.sleep(2)

    os.chdir(modelDirectory)
    x_test, y_test, test_files = prepare_all("image/")
    for i in range(0, len(x_test)):
        prediction = model.predict(x_test[i])
        print("Pred:", prediction[0][0], "->", categories[1 if prediction[0][0] > 0.5 else 0], ": Correct is",
              y_test[i], ": Image file is", test_files[i])

cap.release()
cv2.destroyAllWindows()
