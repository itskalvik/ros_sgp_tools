import utm
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2, PointField
except ImportError:
    pass

# Extract geofence and home location from QGC plan file
def get_mission_plan(fname):
    with open(fname, "r") as infile:
        data = json.load(infile)
        vertices = np.array(data['geoFence']['polygons'][0]['polygon'])
        home_position = data['mission']['plannedHomePosition']
    return vertices, home_position

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

class CustomStandardScaler(StandardScaler):
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