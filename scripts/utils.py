import json
import numpy as np
import geopandas as gpd
from shapely import geometry
from sgptools.utils.misc import *


# Extract geofence and home location from QGC plan file
def plan2data(fname, num_samples=5000):
    with open(fname, "r") as infile:
        data = json.load(infile)
        vertices = np.array(data['geoFence']['polygons'][0]['polygon'])
        home_position = data['mission']['plannedHomePosition']

    poly = geometry.Polygon(vertices)
    sampler = gpd.GeoSeries([poly])
    candidates = sampler.sample_points(size=num_samples,
                                       rng=2024)
    candidates = candidates.get_coordinates().to_numpy()

    return candidates, home_position

# Reorder the waypoints to match the order of the points in the path
# The waypoints are mathched to the closest points in the path 
def reoder_path(path, waypoints):
    dists = pairwise_distances(path, Y=waypoints, metric='euclidean')
    _, col_ind = linear_sum_assignment(dists)
    Xu = waypoints[col_ind].copy()
    return Xu

# Project the waypoints back to the candidate set while retaining the 
# waypoint visitation order
def project_waypoints(waypoints, candidates):
    waypoints_disc = cont2disc(waypoints, candidates)
    waypoints_valid = reoder_path(waypoints, waypoints_disc)
    return waypoints_valid
