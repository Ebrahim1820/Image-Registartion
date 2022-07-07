# python 2/3 compotibility
from __future__ import print_function

# local madules
import numpy as np
from calculating_distance import (map_center,
                                  lonlat_to_xy,
                                  calculate_shift_in_meters,
                                  find_distance_between_points,
                                  xy_to_lonlat)

from match_draw_kps import match_and_draw_kps, loadImage_calcHomography

from find_obj import draw_point, inverse_transform_coordinates


def main(sar_coordinates, sar_path, optic_path,
        optic_center_location, scale):

    ''''
        This function takes images and methods for feature extraction.
        then calculate homography function and
        return transform coordinate from Optic2SAR
        and transform coordinates from SAR2Optic.
    '''
    xy_coordinates_optic2SAR, xy_inv_tf_coordinates_SAR2Optic, H, optic_img1,\
    sar_img2 = loadImage_calcHomography(sar_path,
                                        optic_path,
                                        feature='orb')

    # Calculate map coordinates using homography from sar coordinates
    x2_transformed, y2_transformed = np.round(inverse_transform_coordinates(sar_coordinates[0], sar_coordinates[1], H)).astype(int)
    transform_coordinates = (x2_transformed, y2_transformed)

    mapCenter = map_center(optic_img1)
    import cv2
    sar = draw_point(*sar_point_position, 0, 255, 0, sar_img2)
    im = draw_point(*mapCenter, 0, 255, 0, optic_img1)
    im = draw_point(*transform_coordinates, 0, 255, 0 , im)
    cv2.imshow('sar', sar)
    cv2.imshow('map', im)
    cv2.waitKey(0)
    filename1 = 'sar_point.jpg'
    filename2 = 'optic_point.jpg'
    # Using cv2.imwrite() method
    # Saving the image
    #cv2.imwrite(filename1, sar)
    #cv2.imwrite(filename2, im)


    #cv2.waitKey(0)


    ''''Find map coordinates shift in pixels relatively to map center
        Multiply shift in pixels by scale to calculate shift in meters
    '''
    dx, dy = calculate_shift_in_meters(mapCenter, transform_coordinates, scale)

    ''''Convert xy SAR position to Long and Lat
        Convert map center geolocation from degrees (WGS84) 
        to global meters (universal Mecrator projection)
    '''
    sar_location = xy_to_lonlat(optic_center_location, dx, dy)



    print('Sar estimated location', sar_location)

    return sar_location

from calculating_distance import find_distance
if __name__ == '__main__':
     # Define some manually verification constants just to have really tru values.
     optic_path = 'rotated_img_59.79894989561127_ 30.26867942082413.png'
     optic_size = 1188, 1623
     optic_center_position = 594, 811
     optic_center_location = 60.09269, 30.19904

     sar_path = 'rotated_img_59.79898_30.2684_SAR.png'
     sar_size = 931, 1272
     sar_point_position = 0, 0 #105, 205
     sar_point_location1 = 60.092229, 30.196719
     sar_point_location2 = 60.093101, 30.197189
     print('Real', find_distance(sar_point_location1, sar_point_location2))

     estimate_point1 = 60.092138, 30.196944
     estimate_point2 = 60.093033, 30.197047
     print('Estimated', find_distance(estimate_point1, estimate_point2))
     scale = 0.28
     sar_location = main(sar_point_position,
                    sar_path,
                    optic_path,
                    optic_center_location,
                    scale)

