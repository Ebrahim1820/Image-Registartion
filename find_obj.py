# python 2/3 comatibiltiy
from __future__ import print_function

import numpy as np
import cv2

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6

def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.xfeatures2d_SIFT.create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d_SURF.create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(400)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'kaze':
        detector = cv2.KAZE_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                tabel_number=6,
                                key_size=12,
                                multi_probe_level=1)
        matcher = cv2.FlannBasedMatcher(flann_params, {})
    else:
        matcher = cv2.BFMatcher(norm)

    return detector, matcher

def transform_coordinates(x: np.float32, y: np.float32, matrix: np.ndarray) -> np.ndarray:
	"""
	Calculate coordinates transformation.
	:param x: x coordinate.
	:param y: y coordinate.
	:param matrix: transformation matrix. Can be obtained from cv2.findHomography().
	:return: transformed coordinates.
	"""
	return cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), matrix).reshape(2)

def inverse_transform_coordinates(x: np.float32, y: np.float32, matrix: np.ndarray) -> np.ndarray:
	"""
	Calculate coordinates inverse transformation.
	:param x: x coordinate.
	:param y: y coordinate.
	:param matrix: original (not inversed) transformation matrix. Can be obtained from cv2.findHomography().
	:return: inverse transformed coordinates.
	"""
	return cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), np.linalg.inv(matrix)).reshape(2)

def draw_point(x, y, r, g, b, img):
    """
    Draw circle on the image by x and y coordinate on the image.
    :param x: x coordinate.
    :param y: y coordinate.
    :param r-g-b: color value (255,255,255).
    :param matrix: original (not inversed) transformation matrix. Can be obtained from cv2.findHomography().
    :return: image with circle as feature  .
    """
    return cv2.circle(img, (int(x),int(y)), radius=3, color=(r, g, b), thickness=5)

def filter_matches(kp1, kp2, matches, ratio = 0.7):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])

    kp_pairs = zip(mkp1, mkp2)

    p3 = np.float32([kp.pt for kp in kp1])
    p4 = np.float32([kp.pt for kp in kp2])
    return p1, p2, p3, p4, list(kp_pairs)

def explor_match(win, img1, img2, kp_pairs, p3, p4, status=None, H=None):
    """
    :param img1:np.array(the first input image)
    :param img2:np.array(the second input image)
    :param kp_pairs: list(template image's and source image's matched keypoints)
    :param all_kp_pairs: list of all key points of both images
    :param H: 3x3 Homography matrix
    :param status: which is a mask which specifies the inlier points
    :return img_transform:image with transformed points, list(image of transformation points) and list(image of inverse transformation image points)
    """

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    transform_img = vis.copy()
    img_invers = vis.copy()


    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)

    p1, p2 = [],[]  # python2/3 change of zip unpaking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)

    for (x1, y1),(x2, y2), inlier in zip(p1, p2, status):

        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red, r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+1, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (y2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

    vis0 = vis.copy()
    for (x1, y1),(x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            img=cv2.line(vis, (x1, y1),(x2, y2), green)
            #img_invers = img.copy()


    xy_coordinates = []
    xy_tf_coordinates = []

    for (x1, y1), (x2, y2) in zip(p3, p4):
        x1_transformed, y1_transformed = np.round(transform_coordinates(x1, y1, H)).astype(int)

        #print('x1_transformed, y1_transformed',x2_transformed, y2_transformed)
        if (x1_transformed > 1 and y1_transformed > 1) and \
                x1_transformed < w2 and y1_transformed < h2:
            img_transform = draw_point(x1, y1, 0, 0, 255, transform_img)
            img_transform = draw_point(x1_transformed + w1, y1_transformed, 0, 0, 255, img_transform)
            img_transform = cv2.line(img_transform, (int(x1), int(y1)), (x1_transformed + w1, y1_transformed), (255, 0, 0))
            xy_coordinates.append((x1, y1))
            xy_tf_coordinates.append((x1_transformed + w1, y1_transformed))

			

        x2_transformed, y2_transformed = np.round(inverse_transform_coordinates(x2, y2, H)).astype(int)

        # extract ROI from each image for removing false matches points
        if (x2_transformed > 0 and y2_transformed > 0) and \
                x2_transformed < w1 and y2_transformed < h1:
            img_inv_transform = draw_point(int(x2 + w1), y2, 255, 0, 100, img_invers)

            img_inv_transform = draw_point(x2_transformed, y2_transformed, 255, 0, 0, img_inv_transform)

            img_inv_transform = cv2.line(img_inv_transform, (int(x2 + w1), int(y2)), (x2_transformed, y2_transformed), (0, 255, 255))


    return img_transform, img_inv_transform, xy_coordinates, xy_tf_coordinates
