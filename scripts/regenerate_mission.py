import json
import h5py

# Paths
data_path = './launch/data/IPP-mission-2025-07-15-19-17-21'
mission_path = f"{data_path}/mission.plan"
log_path = f"{data_path}/mission-log.hdf5"

# Step 1: Load waypoints from HDF5
with h5py.File(log_path, "r") as f:
    waypoints = {}
    for key in f.keys():
        if "waypoints" in key:
            waypoints[key] = f[key][:].astype(float)

selected_wp_set = list(waypoints.values())[0]
new_waypoints = [(float(lat), float(lon)) for lat, lon in selected_wp_set]

# Step 2: Load existing mission plan
with open(mission_path, 'r') as f:
    mission_data = json.load(f)
items = mission_data['mission']['items']
for item in items:
    if 'TransectStyleComplexItem' in item:
        complex_item = item['TransectStyleComplexItem']
        break
else:
    raise ValueError("TransectStyleComplexItem not found in mission plan.")

# Step 3: Replace Items with new waypoints
new_items = []
base_id = 2

for idx, (lat, lon) in enumerate(new_waypoints):
    new_item = {
        "autoContinue": True,
        "command": 16, 
        "doJumpId": base_id + idx,
        "frame": 3,
        "params": [0, 0, 0, None, lat, lon, 50], 
        "type": "SimpleItem"
    }
    new_items.append(new_item)

complex_item["Items"] = new_items

# Step 4: Save updated mission plan
with open("mission_updated.plan", 'w') as f:
    json.dump(mission_data, f, indent=4)

print("Mission updated using waypoints from mission-log.hdf5.")