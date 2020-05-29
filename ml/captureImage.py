import cv2
import os

cam = cv2.VideoCapture(0)

cv2.namedWindow("camera")

img_counter = 0

imagesDirectory = r'C:\xampp\htdocs\github\gymnasiearbete\ml\images'
originalDirectory = r'C:\xampp\htdocs\github\gymnasiearbete\ml'

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break

    k = cv2.waitKey(1)

    # Press ESC to close
    if k % 256 == 27:
        break

    elif k % 256 == 32:
        # Press Space to capture image
        os.chdir(imagesDirectory)
        img_name = "image{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        os.chdir(originalDirectory)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
