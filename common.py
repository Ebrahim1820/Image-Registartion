from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce
import numpy as np
import cv2
import imutils

# built-in modules
import os
import itertools as it
from contextlib import contextmanager

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

@contextmanager
def Timer(msg):
    print(msg, '...',)
    start = clock()
    try:
        yield
    finally:
        print('%.2f ms' %((clock()-start)*1000))


def load_and_preprocess_img(optic_path1, sar_path2):

    optic_img1 = cv2.imread(optic_path1, 0)  # load optic image
    sar_img2 = cv2.imread(sar_path2, 0)  # load sar image

    if optic_img1 is None:
        print('[INFO] Failed to load image:', optic_path1)
        sys.exit(1)

    if sar_img2 is None:
        print('[INFO] Failed to load image2:', sar_path2)
        sys.exit(1)

    # Blurring and Filtering images
    optic_img1 = cv2.GaussianBlur(optic_img1, (3, 3), 0)
    sar_img2 = cv2.GaussianBlur(sar_img2, (3, 3), 0)


    return  optic_img1, sar_img2