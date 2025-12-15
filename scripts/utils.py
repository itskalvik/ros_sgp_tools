import utm
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2, PointField
except ImportError:
    pass

# Extract geofence, home location, and optionally waypoints from QGC plan file
def get_mission_plan(fname):
    with open(fname, "r") as infile:
        data = json.load(infile)

    vertices = np.array(data["geoFence"]["polygons"][0]["polygon"])
    home_position = data["mission"]["plannedHomePosition"]

    # Try to extract waypoints if they exist; otherwise return None for waypoints.
    waypoints = None
    try:
        items = data.get("mission", {}).get("items", [])
        for item in items:
            tsc = item.get("TransectStyleComplexItem")
            if not tsc:
                continue

            wp_list = []
            for wp in tsc.get("Items", []):
                if wp.get("command") == 16:
                    params = wp.get("params", [])
                    # QGC-style: params[4:7] are typically [lat, lon, alt]
                    if len(params) >= 7:
                        wp_list.append(params[4:7])

            if wp_list:
                waypoints = np.array(wp_list)[:, :2]
                break
    except (AttributeError, TypeError, KeyError):
        waypoints = None

    return vertices, home_position, waypoints


def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions.
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message

    Code source:
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
    """
    # In a PointCloud2 message, the point cloud is stored as an byte 
    # array. In order to unpack it, we also include some parameters 
    # which desribes the size of each individual point.
    points = np.array(points, dtype=np.float32)
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

    data = points.astype(dtype).tobytes() 

    # The fields specify what the bytes represents. The first 4 bytes 
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]

    # The PointCloud2 message also has a header which specifies which 
    # coordinate frame it is represented in. 
    header = Header(frame_id=parent_frame)

    return PointCloud2(
        header=header,
        height=1, 
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3), # Every point consists of three float32s.
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )

class LatLonStandardScaler(StandardScaler):
    def fit(self, X, y=None, sample_weight=None):
        # Map lat long to UTM points before normalization
        X = utm.from_latlon(X[:, 0], X[:, 1])
        self.encoding = X[2:]
        X = np.vstack([X[0], X[1]]).T

        # Fit normalization params
        super().fit(X, y=y, sample_weight=sample_weight)

        # Change variance/scale parameter to ensure all axis are scaled to the same value
        ind = np.argmax(self.var_)
        self.var_ = np.ones(X.shape[-1])*self.var_[ind]
        self.scale_ = np.ones(X.shape[-1])*self.scale_[ind]
        self.scale_ /= 10.0  # Scale to ensure an extent of ~10 units

        # Compute distance scale
        X1 = super().transform([np.copy(X[0])])
        X2 = super().transform([np.copy(X[0]) + [1.0, 0.0]])
        self.distance_scale = np.linalg.norm(X1-X2, axis=-1)

    def transform(self, X, copy=None):
        # Map lat long to UTM points before normalization
        X = utm.from_latlon(X[:, 0], X[:, 1])
        X = np.vstack([X[0], X[1]]).T
        return super().transform(X, copy=copy)
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def inverse_transform(self, X, copy=None):
        X = super().inverse_transform(X, copy=copy)

        # Map UTM to lat long points after de-normalization
        X = utm.to_latlon(X[:, 0], X[:, 1], 
                          self.encoding[0], self.encoding[1])
        X = np.vstack([X[0], X[1]]).T
        return X

    def meters2units(self, distance):
        # Map distance in meters to distance in normalized units
        return self.distance_scale*distance
    
    def units2meters(self, distance):
        # Map distance in normalized units to distance in meters
        return distance/self.distance_scale

class RunningStats:
    """Computes running mean and standard deviation
    Url: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
    Adapted from:
        *
        <http://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation>
        * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
        * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>
        
    Usage:
        rs = RunningStats()
        for i in range(10):
            rs += np.random.randn()
            print(rs)
        print(rs.mean, rs.std)
    """

    def __init__(self, n=0., m=None, s=None):
        self.n = n
        self.m = m
        self.s = s

    def clear(self):
        self.n = 0.

    def push(self, x, per_dim=False):
        x = np.array(x).copy().astype('float64')
        # process input
        if per_dim:
            self.update_params(x)
        else:
            for el in x.flatten():
                self.update_params(el)

    def update_params(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = 0.
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)
            
    def __add__(self, other):
        if isinstance(other, RunningStats):
            sum_ns = self.n + other.n
            prod_ns = self.n * other.n
            delta2 = (other.m - self.m) ** 2.
            return RunningStats(sum_ns,
                                (self.m * self.n + other.m * other.n) / sum_ns,
                                self.s + other.s + delta2 * prod_ns / sum_ns)
        else:
            self.push(other)
            return self

    @property
    def mean(self):
        return self.m if self.n else 0.0

    def variance(self):
        return self.s / (self.n - 1) if self.n else 0.0

    @property
    def std(self):
        return np.max([np.sqrt(self.variance()), np.ones_like(self.s)])
        
    def __repr__(self):
        return '<RunningMean(mean={: 2.4f}, std={: 2.4f}, n={: 2f}, m={: 2.4f}, s={: 2.4f})>'.format(self.mean, self.std, self.n, self.m, self.s)
        
    def __str__(self):
        return 'mean={: 2.4f}, std={: 2.4f}'.format(self.mean, self.std)

def haversine(pt1, pt2):
    """
    Calculate the great circle distance between two points
    on the earth's surface (specified in decimal degrees)
    https://stackoverflow.com/a/29546836

    Args:
        pt1 (ndarray, [n, 2]): Start location longitude and latitude 
        pt2 (ndarray, [n, 2]): End location longitude and latitude 
    """
    lon1, lat1 = pt1[:, 0], pt1[:, 1]
    lon2, lat2 = pt2[:, 0], pt2[:, 1]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    m = 6378.137 * c * 1000
    return m