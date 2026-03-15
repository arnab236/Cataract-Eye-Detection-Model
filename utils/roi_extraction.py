import cv2
import numpy as np


def extract_roi(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray,(9,9),2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=40,
        maxRadius=120
    )

    if circles is not None:

        circles = np.uint16(np.around(circles))

        x, y, r = circles[0][0]

        roi = image[y-r:y+r, x-r:x+r]

        roi = cv2.resize(roi,(224,224))

        return roi

    return cv2.resize(image,(224,224))