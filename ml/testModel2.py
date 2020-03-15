import cv2
import tensorflow as tf
import os

# Categories
categories = ["eyes_closed", "eyes_open"]


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


# Load model
model = tf.keras.models.load_model("closed_vs_open.h5")

# Test model images
x_test, y_test, test_files = prepare_all("testImages/")
for i in range(0, len(x_test)):
    prediction = model.predict(x_test[i])
    print("Prediction: ", categories[1 if prediction[0][0] > 0.5 else 0], ": Correct is",
          y_test[i])
