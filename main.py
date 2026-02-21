from sklearn.cluster import DBSCAN
import face_recognition
import os
import shutil
import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from collections import defaultdict


def find_eps(encodings):
    # 1. Calculate distances to the nearest neighbor (k=2)
    # 'encodings' is your list of 128-d face vectors
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors_fit = neighbors.fit(encodings)
    distances, indices = neighbors_fit.kneighbors(encodings)

    # 2. Sort distances and find the "knee"
    sort_distances = np.sort(distances[:, 1], axis=0)
    kneedle = KneeLocator(
        range(len(sort_distances)),
        sort_distances,
        S=1.0,
        curve="convex",
        direction="increasing",
    )

    optimal_eps = kneedle.knee_y
    return optimal_eps

def get_encodings(input_folder_path: str):
    image_files = [os.path.join(input_folder_path, f) for f in os.listdir(input_folder_path)]

    images_with_encodings = {}
    for image_file in image_files:
        face_recognition_image = face_recognition.load_image_file(image_file)
        faces_encodings = face_recognition.face_encodings(face_recognition_image)
        if faces_encodings:
            images_with_encodings.append(
                {"image_file": image_file, "encoding": faces_encodings[0]}  # Taking the first face
            )

    return images_with_encodings

def get_clusters(images_with_encodings: list):
    dbscan = DBSCAN(eps=0.5, min_samples=3, metric="euclidean")
    dbscan.fit([image_with_encoding["encoding"] for image_with_encoding in images_with_encodings])
    clusters = defaultdict(list)
    for i, label_id in enumerate(dbscan.labels_):
        clusters[label_id].append(images_with_encodings[i])
    return clusters

images_with_encodings = get_encodings("input_folder")
clusters = get_clusters(images_with_encodings=images_with_encodings)

#score = silhouette_score(encodings_only, clt.labels_)  # might be wrong
#print("silhouette score: ", score)
