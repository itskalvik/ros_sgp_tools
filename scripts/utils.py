import utm
import json
import numpy as np
from sklearn.preprocessing import StandardScaler


# Extract geofence and home location from QGC plan file
def get_mission_plan(fname):
    with open(fname, "r") as infile:
        data = json.load(infile)
        vertices = np.array(data['geoFence']['polygons'][0]['polygon'])
        home_position = data['mission']['plannedHomePosition']
    return vertices, home_position

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