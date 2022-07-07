import cv2
import numpy as np
import sys, getopt
from common import Timer
from multiprocessing.pool import ThreadPool
from find_obj import init_feature
from extract_kps import affine_detect
from common import load_and_preprocess_img
from find_obj import init_feature, filter_matches, explor_match, draw_point


def match_and_draw_kps(win, optic_img1, sar_img2, matcher, desc1, desc2, kp1, kp2):
    """"
    this function receives keypoints and descriptors of both images and return
    homography matrix and transform points from optic to sar and inverse
    """

    with Timer('matching'):
        raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
    p1, p2, p3, p4, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    if len(p1) >= 4:

        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

        print('%d / %d inliers / matched' % (np.sum(status), len(status)))

        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    img_tf_Optic2SAR, img_inv_tf_SAR2Optic, xy_coordinates_optic2SAR, xy_inv_tf_coordinates_SAR2Optic = explor_match(win,
                                                                                                    optic_img1,
                                                                                                    sar_img2, kp_pairs,
                                                                                                    p3, p4, None, H)

    return xy_coordinates_optic2SAR, xy_inv_tf_coordinates_SAR2Optic, H


def loadImage_calcHomography(sar_path, optic_path, feature='kaze'):

    print(__doc__)
    opts, args = getopt.getopt(sys.argv[1:], '',['feature='])
    opts = dict(opts)

    feature_name = opts.get('--feature', feature)


    optic_img1, sar_img2 = load_and_preprocess_img(optic_path, sar_path)

    detector, matcher = init_feature(feature_name)


    if detector is None:
        print('unknown feature: ', feature_name)
        sys.exit(1)
    print('[INFO] Using', feature_name, 'as Method')

    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    kp1, desc1 = affine_detect(detector, optic_img1, pool=pool)
    kp2, desc2 = affine_detect(detector, sar_img2, pool=pool)
    print('[INFO] optic_img1 - %d features, sar_img2 - %d features' % (len(kp1), len(kp2)))


    '''
    pass kps and descriptors and calculate homography matrix and transform points from
    Optic to Sar and inverse.
    It returns xy coordinates of optic2Sar and Sar2Optic also H homography matrix 
    '''
	
    win = 'Affine find_obj_match'
    xy_coordinates_optic2SAR, xy_inv_tf_coordinates_SAR2Optic, H = match_and_draw_kps(win, optic_img1,
                                                                                  sar_img2, matcher, desc1,
                                                                                  desc2, kp1, kp2)


    return xy_coordinates_optic2SAR, xy_inv_tf_coordinates_SAR2Optic, H, optic_img1, sar_img2