"""
Tests Ebrahim module functions.
"""
import traceback

import math
from collections.abc import Callable
from pyproj import Geod


def test_main_function(function: Callable):
    """
    Test main module function that must generate homography transformation function f(optic_path,sar_path,optic_center_longitude,optic_center_latitude,scale)->H that must convert SAR point to location H(sar_x,sar_y)->(longitude,latitude).
    """
    try:
        homography = function(sar_path, optic_path, optic_center_location, scale)
        assert isinstance(homography, Callable), 'Main module function must return homography transformation function.'
        location = homography(sar_point_position)
        assert _is_near(location, sar_point_location), 'Calculated location of SAR point is wrong.'
        print('Main function is OK.')
    except Exception as ex:
        assert False, str(ex) 


# def test_locate_function(function: Callable, *args):
#     """
#     Test homography function that must convert SAR point to location H(sar_x,sar_y)->(longitude,latitude).
#     """
#     try:
#         location = function(*args)
#         assert _is_near(location, sar_point_location), 'Calculated location of SAR point is wrong.'
#         print('Locate function is OK.')
#     except Exception as ex:
#         traceback.print_exc()
#         print(str(ex))


def _geodistance(location1, location2) -> float:
    """
    Get distance between two locations in meters.
    """
#     # load Earth model WGS84
     model = Geod(ellps="WGS84")
     # calculate geodetic distance instead of euclidean
     return model.line_length([location1[0], location2[0]], [location1[1], location2[1]])


# def _is_near(location1, location2, confidence: float = 5) -> bool:
#     """
#     Check whether distance between two locations is not greater then confidence (in meters).
#     """
#     return _geodistance(location1, location2) <= confidence


#Define some manually verification constants just to have really tru values.
optic_path = 'optic.png'
optic_size = 1188, 1623
optic_center_position = 594, 811
optic_center_location = 60.09269, 30.19904

sar_path = 'sar.png'
sar_size = 931, 1272
sar_point_position = 750, 300
sar_point_location = 60.092874, 30.197645

# Calculate scale
optic_point_position = 669, 188
distance = math.sqrt(math.pow(optic_point_position[0] - optic_center_position[0], 2)\
                     + math.pow(optic_point_position[1] - optic_center_position[1], 2))

#geodistance = _geodistance(optic_center_location, sar_point_location)
#scale = geodistance / distance  # assume that vertical and horizontal scales are equal

scale = 0.28
print("scale", scale)
import sys
sys.path.append('..')
from main import main as locate
test_locate_function(locate, sar_point_position, sar_path, optic_path, optic_center_location, scale)




