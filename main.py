from sklearn.cluster import DBSCAN
import face_recognition
import os
import shutil
import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def get_encodings(input_folder_path: str):
    all_entries = [
        os.path.join(input_folder_path, f)
        for f in os.listdir(input_folder_path)
    ]
    image_files = [
        p for p in all_entries
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    ]
    total_images = len(image_files)
    logger.info(
        "Found %d images in %s, getting encodings...",
        total_images,
        input_folder_path,
    )

    images_with_encodings = []

    for index, image_file in enumerate(image_files, start=1):
        logger.info(
            "Encoding image %d/%d: %s",
            index,
            total_images,
            os.path.basename(image_file),
        )
        face_recognition_image = face_recognition.load_image_file(image_file)
        faces_encodings = face_recognition.face_encodings(face_recognition_image)
        if faces_encodings:
            images_with_encodings.append(
                {"image_file": image_file, "encoding": faces_encodings[0]}  # Taking the first face
            )

    return images_with_encodings

def get_clusters(images_with_encodings: list):
    logger.info("Finding optimal eps...")
    eps = find_eps([image_with_encoding["encoding"] for image_with_encoding in images_with_encodings])
    logger.info(f"Optimal eps: {eps}")
    dbscan = DBSCAN(eps=eps, min_samples=3, metric="euclidean")
    dbscan.fit([image_with_encoding["encoding"] for image_with_encoding in images_with_encodings])
    clusters = defaultdict(list)
    for i, label_id in enumerate(dbscan.labels_):
        clusters[label_id].append(images_with_encodings[i])
    
    score = silhouette_score([image_with_encoding["encoding"] for image_with_encoding in images_with_encodings], dbscan.labels_)
    return clusters, score

if __name__ == "__main__":
    input_folder = "input_folder"
    logger.info(f"Getting encodings for images in {input_folder}...")
    images_with_encodings = get_encodings(input_folder)
    logger.info("Getting clusters...")
    clusters, score = get_clusters(images_with_encodings=images_with_encodings)
    logger.info(f"Found {len(clusters)} clusters with silhouette score: {score}")

    logger.info("Copying images in each cluster to folders...");
    output_root = "output_clusters"
    os.makedirs(output_root, exist_ok=True)

    for cluster_id, cluster_images in clusters.items():
        cluster_folder = os.path.join(output_root, f"cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)

        for image_data in cluster_images:
            source_image = image_data["image_file"]
            destination_image = os.path.join(cluster_folder, os.path.basename(source_image))
            shutil.copy2(source_image, destination_image)
    logger.info("Complete")

