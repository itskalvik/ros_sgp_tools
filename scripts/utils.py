import json
import numpy as np
import geopandas as gpd
from shapely import geometry


# Extract geofence and home location from QGC plan file
def plan2data(fname, num_samples=5000):
    with open(fname, "r") as infile:
        data = json.load(infile)
        vertices = np.array(data['geoFence']['polygons'][0]['polygon'])
        # Swap lat/long
        vertices[:,[1,0]] = vertices[:,[0,1]]
        home_position = data['mission']['plannedHomePosition']

    poly = geometry.Polygon(vertices)
    sampler = gpd.GeoSeries([poly])
    candidates = sampler.sample_points(size=num_samples,
                                       rng=2024)
    candidates = candidates.get_coordinates().to_numpy()

    return candidates, home_position