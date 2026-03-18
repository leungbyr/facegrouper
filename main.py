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

from constants import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def find_eps(encodings, metric="euclidean"):
    """Suggest eps from k-NN distance curve (k=2) using the same metric as DBSCAN."""
    encodings = np.asarray(encodings, dtype=np.float64)
    neighbors = NearestNeighbors(n_neighbors=2, metric=metric)
    neighbors_fit = neighbors.fit(encodings)
    distances, _ = neighbors_fit.kneighbors(encodings)
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

def get_encodings(
    input_folder_path: str,
    num_jitters: int = 1,
    number_of_times_to_upsample: int = 1,
):
    """
    Load images and compute face encodings.

    num_jitters: Higher (e.g. 10–100) averages over augmented crops for more
        robust encodings; slower. Default 1 for speed.
    number_of_times_to_upsample: 2 can help detect smaller faces; slightly slower.
    """
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
        face_locations = face_recognition.face_locations(
            face_recognition_image,
            number_of_times_to_upsample=number_of_times_to_upsample,
        )
        faces_encodings = face_recognition.face_encodings(
            face_recognition_image,
            known_face_locations=face_locations,
            num_jitters=num_jitters,
        )
        if faces_encodings:
            images_with_encodings.append(
                {"image_file": image_file, "encoding": faces_encodings[0]}  # Taking the first face
            )

    return images_with_encodings

def _reassign_noise_to_nearest_cluster(encodings, labels, eps, metric="euclidean"):
    """Assign noise points (-1) to nearest cluster if within eps."""
    labels = np.array(labels, dtype=int)
    noise_mask = labels == -1
    if not np.any(noise_mask):
        return labels
    encodings = np.asarray(encodings)
    if metric == "cosine":
        norms = np.linalg.norm(encodings, axis=1, keepdims=True)
        encodings = encodings / np.where(norms == 0, 1, norms)
        metric = "euclidean"
    unique_labels = np.setdiff1d(np.unique(labels), [-1])
    if len(unique_labels) == 0:
        return labels
    noise_encodings = encodings[noise_mask]
    cluster_centroids = np.array([
        encodings[labels == c].mean(axis=0) for c in unique_labels
    ])
    nn = NearestNeighbors(n_neighbors=1, metric=metric).fit(cluster_centroids)
    dists, idx = nn.kneighbors(noise_encodings)
    new_labels = labels.copy()
    assign = dists.ravel() <= eps
    new_labels[noise_mask] = np.where(assign, unique_labels[idx.ravel()], -1)
    return new_labels


def _merge_near_duplicate_clusters(encodings, labels, merge_threshold_ratio=0.5, metric="euclidean"):
    """
    Merge clusters whose centroids are very close (likely same identity).
    merge_threshold_ratio: merge if centroid distance < this fraction of eps; 0 to disable.
    """
    if merge_threshold_ratio <= 0:
        return labels
    labels = np.array(labels, dtype=int)
    unique = np.setdiff1d(np.unique(labels), [-1])
    if len(unique) < 2:
        return labels
    encodings = np.asarray(encodings)
    if metric == "cosine":
        norms = np.linalg.norm(encodings, axis=1, keepdims=True)
        encodings = encodings / np.where(norms == 0, 1, norms)
        metric = "euclidean"
    centroids = np.array([encodings[labels == c].mean(axis=0) for c in unique])
    nn = NearestNeighbors(metric=metric).fit(centroids)
    # Use a small k to get pairwise distances; we need eps in same scale
    d, _ = nn.kneighbors(centroids, n_neighbors=min(2, len(unique)))
    if d.shape[1] < 2:
        return labels
    second_dist = d[:, 1]
    eps_est = np.median(second_dist) * 2  # rough scale for "same cluster"
    merge_distance = eps_est * merge_threshold_ratio
    # Union-find merge clusters with centroid distance < merge_distance
    parent = {c: c for c in unique}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for i, c1 in enumerate(unique):
        for j, c2 in enumerate(unique):
            if i >= j:
                continue
            if np.linalg.norm(centroids[i] - centroids[j]) < merge_distance:
                parent[find(c1)] = find(c2)
    remap = {c: find(c) for c in unique}
    # Relabel to contiguous ids
    new_ids = {v: i for i, v in enumerate(sorted(set(remap.values())))}
    new_labels = np.array([new_ids.get(remap.get(l, l), l) if l >= 0 else -1 for l in labels])
    return new_labels


def get_clusters(
    images_with_encodings: list,
    metric: str = "euclidean",
    min_samples: int = 3,
    reassign_noise: bool = True,
    merge_near_duplicates: bool = True,
    merge_threshold_ratio: float = 0.5,
):
    """
    Cluster faces with DBSCAN.

    metric: "euclidean" (default) or "cosine". Cosine can help when embeddings
        are not L2-normalized or for angular separation.
    min_samples: minimum points to form a cluster; try 2 for few photos per person.
    reassign_noise: assign DBSCAN noise (-1) to nearest cluster if within eps.
    merge_near_duplicates: merge clusters with very close centroids (same identity).
    merge_threshold_ratio: merge if centroid distance < this fraction of estimated eps; 0 disables.
    """
    encodings = [x["encoding"] for x in images_with_encodings]
    logger.info("Finding optimal eps...")
    eps = find_eps(encodings, metric=metric)
    logger.info("Optimal eps: %s (metric=%s)", eps, metric)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(encodings)
    labels = dbscan.labels_.copy()

    if reassign_noise:
        labels = _reassign_noise_to_nearest_cluster(encodings, labels, eps, metric=metric)
        logger.info("Reassigned noise points to nearest cluster within eps")
    if merge_near_duplicates:
        labels = _merge_near_duplicate_clusters(
            encodings, labels,
            merge_threshold_ratio=merge_threshold_ratio,
            metric=metric,
        )
        logger.info("Merged near-duplicate clusters (ratio=%.2f)", merge_threshold_ratio)

    clusters = defaultdict(list)
    for i, label_id in enumerate(labels):
        if label_id >= 0:
            clusters[label_id].append(images_with_encodings[i])
    # Keep noise in a single group so we don't drop images
    noise = [images_with_encodings[i] for i in range(len(labels)) if labels[i] == -1]
    if noise:
        clusters[-1] = noise

    encodings_arr = np.asarray(encodings)
    if metric == "cosine":
        norms = np.linalg.norm(encodings_arr, axis=1, keepdims=True)
        encodings_arr = encodings_arr / np.where(norms == 0, 1, norms)
    labels_for_silhouette = np.where(labels >= 0, labels, -1)
    score = silhouette_score(encodings_arr, labels_for_silhouette)
    return clusters, score

if __name__ == "__main__":
    input_folder = "input_folder"
    output_root = "output_clusters"

    # Encoding quality: higher num_jitters = more robust encodings, slower (try 10 or 100)
    num_jitters = 1
    number_of_times_to_upsample = 2  # helps with smaller/distant faces

    logger.info("Getting encodings for images in %s...", input_folder)
    images_with_encodings = get_encodings(
        input_folder,
        num_jitters=num_jitters,
        number_of_times_to_upsample=number_of_times_to_upsample,
    )
    logger.info("Getting clusters...")
    clusters, score = get_clusters(
        images_with_encodings=images_with_encodings,
        metric="euclidean",  # or "cosine" if clusters still mix
        min_samples=3,       # try 2 if you have few photos per person
        reassign_noise=True,
        merge_near_duplicates=True,
        merge_threshold_ratio=0.5,
    )
    logger.info("Found %d clusters with silhouette score: %s", len(clusters), score)

    logger.info("Copying images in each cluster to folders...")
    os.makedirs(output_root, exist_ok=True)

    for cluster_id, cluster_images in clusters.items():
        cluster_folder = os.path.join(output_root, f"cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)

        for image_data in cluster_images:
            source_image = image_data["image_file"]
            destination_image = os.path.join(cluster_folder, os.path.basename(source_image))
            shutil.copy2(source_image, destination_image)
    logger.info("Complete")

