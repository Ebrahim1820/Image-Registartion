import pyproj
import math
import numpy as np
import cv2
from pyproj import Proj, transform
from pyproj import Transformer, Geod
from geopy.distance import lonlat
import geopy.distance



def find_distance(p1, p2):
    """
    Calculate geodesic distance between two points.
    Accepts coordinates in (y, x)/(lat, lon)
    :param p1: Fisrt point tuple of (lat, lon)
    :param p2: Second point tuple of (lat, lon)
    :return: Distance between two points.
    """
    #"{:.2f}".format(geopy.distance.geodesic(p1[:2], p2[:2]).meters)
    return geopy.distance.geodesic(p1[:2], p2[:2]).meters


def find_distance_between_points(poin1, poin2):
	# Just helper function that calculates vector length or Euclidean distance (https://en.wikipedia.org/wiki/Euclidean_distance).
	return math.sqrt(math.pow(poin1[0] - poin2[0], 2) + math.pow(poin1[1] - poin2[1], 2))



def map_center(image):
    ''''Find map center coordinates.'''

    height, width = image.shape[:2]
    x = width // 2  # image width
    y = height // 2  # image height
    return (x, y)

def lonlat_to_xy(center_point, dx, dy):  # --> dx, dy  # in meters
    """
    Convert map center geolocation from degrees (WGS84) to global meters (universal Mecrator projection).
    Add shift in meters to location in meters.
    Convert shifted location in meters (universal Mecrator projection) back to degrees (WGS84).
    :param center_point: Lat and Long relate to center point
    :param dx: distance x in meters
    :param dy: distance y in meters
    :return: returns new geolocation in degrees as a result of the function (lon; lat).
    """
    # receive Long and Lat of center point image

    Pseudo_Mercator_sis_coord_id = 3857
    WGS_84_sis_coord_id = 4326

    transformer_84_PM = Transformer.from_crs(WGS_84_sis_coord_id, Pseudo_Mercator_sis_coord_id)
    transformer_PM_84 = Transformer.from_crs(Pseudo_Mercator_sis_coord_id, WGS_84_sis_coord_id)

    # in degrees
    lon1, lat1 = center_point[1], center_point[0]

    # translate degrees by meters
    x_global, y_global = transformer_84_PM.transform(lat1, lon1)  # in global meters
    x_mm, y_mm = (x_global), (y_global)  # translate meters by meters

    lat_in_degrees, lon_in_degrees = transformer_PM_84.transform(x_mm, y_mm)  # again in degrees
    #print(f'lat in degrees, lon in degrees: {lat_in_degrees}, {lon_in_degrees}')

    return lat_in_degrees, lon_in_degrees


def calculate_shift_in_meters(mapCenter, coordinates, scale):
    ''''
        Multiply shift in pixels by scale to calculate shift in meters
        Find map coordinates shift in pixels relatively to map center (Δx; Δy).
    '''

    x0 = mapCenter[0]  # image width
    y0 = mapCenter[1]  # image height

    #  Find map coordinates shift in pixels relatively to map center

    dx = x0 - coordinates[0]
    dy = y0 - coordinates[1]


    # Multiply shift in pixels by scale to calculate shift in meters

    dx = scale * dx
    dy = scale * dy

    return -dx, dy

def xy_to_lonlat(optic_center_location, dx, dy):
    """
    :param center_point: center point of image which include Lat and Lon
    :param dx, dy: x, y shift in meters

    :return: Lat and Lon for new point
    """

    Pseudo_Mercator_sis_coord_id = 3857 #Spherical Mercator: it will be describing coordinates in meters in x/y. 
    WGS_84_sis_coord_id = 4326 # it describes latitude/longitude coordinates

    transformer_84_PM = Transformer.from_crs(WGS_84_sis_coord_id, Pseudo_Mercator_sis_coord_id)
    transformer_PM_84 = Transformer.from_crs(Pseudo_Mercator_sis_coord_id, WGS_84_sis_coord_id)

    lon1, lat1 = optic_center_location[1], optic_center_location[0]

    # in global meters
    x1, y1 = transformer_84_PM.transform(lat1, lon1)

    # translate meters by meters
    x2 = (x1 - dy -25)
    y2 = (y1 + dx +25)
    # again in degrees
    lat, lon = transformer_PM_84.transform(x2, y2)


    return lat, lon




