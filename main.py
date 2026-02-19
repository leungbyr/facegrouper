from sklearn.cluster import DBSCAN
import face_recognition
import os
import shutil
import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

def find_eps(encodings):
    # 1. Calculate distances to the nearest neighbor (k=2)
    # 'encodings' is your list of 128-d face vectors
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors_fit = neighbors.fit(encodings)
    distances, indices = neighbors_fit.kneighbors(encodings)

    # 2. Sort distances and find the "knee"
    sort_distances = np.sort(distances[:, 1], axis=0)
    kneedle = KneeLocator(range(len(sort_distances)), sort_distances, 
                        S=1.0, curve="convex", direction="increasing")

    optimal_eps = kneedle.knee_y
    return optimal_eps

# 1. Gather all face encodings from your folder
data = [] 
paths = [os.path.join("input_folder", f) for f in os.listdir("input_folder")]

for p in paths:
    img = face_recognition.load_image_file(p)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        # We take the first face found in the image
        data.append({"path": p, "encoding": encodings[0]})

# 2. Cluster the encodings
encodings_only = [d["encoding"] for d in data]
# 'eps' controls how strict the matching is (lower = stricter)
clt = DBSCAN(eps=0.5, min_samples=3, metric="euclidean")
clt.fit(encodings_only)

score = silhouette_score(encodings_only, clt.labels_) # might be wrong
print('silhouette score: ', score)

# 3. Move files into folders named by Cluster ID
for i, label_id in enumerate(clt.labels_):
    dest_dir = f"sorted_faces/person_{label_id}"
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(data[i]["path"], dest_dir)
