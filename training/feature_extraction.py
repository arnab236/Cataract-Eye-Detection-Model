import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_features(gray):

    # normalize image
    gray = cv2.equalizeHist(gray)

    # ---------- SIFT ----------

    sift = cv2.SIFT_create(nfeatures=150)

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        sift_features = np.zeros(128)
    else:
        sift_features = np.mean(descriptors, axis=0)

    # ---------- GLCM ----------

    glcm = graycomatrix(
        gray,
        distances=[1,2],
        angles=[0, np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm,'contrast').mean()
    homogeneity = graycoprops(glcm,'homogeneity').mean()
    energy = graycoprops(glcm,'energy').mean()
    correlation = graycoprops(glcm,'correlation').mean()

    texture_features = np.array([
        contrast,
        homogeneity,
        energy,
        correlation
    ])

    features = np.concatenate((sift_features, texture_features))

    return features