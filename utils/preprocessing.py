import cv2
import numpy as np


# ---------- LOAD HAAR CASCADE ----------
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image could not be loaded")

    img = cv2.resize(img, (224,224))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5,5), 0)

    return img, gray


def is_eye_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # ---------- STEP 1 : HAAR EYE DETECTION ----------
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40,40)
    )

    if len(eyes) == 0:
        return False


    # ---------- STEP 2 : IRIS DETECTION ----------
    blurred = cv2.medianBlur(gray,5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=90
    )

    if circles is None:
        return False


    # ---------- STEP 3 : PUPIL DARKNESS CHECK ----------
    h, w = gray.shape

    center = gray[h//3:2*h//3, w//3:2*w//3]

    mean_intensity = np.mean(center)

    # pupil should be dark
    if mean_intensity > 120:
        return False


    return True