import os
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score

from constants import IMAGE_EXTENSIONS

if __name__ == "__main__":
    input_folder_path = "input_folder"
    output_folder_path = "output_clusters"
    all_input_entries = [
        os.path.join(input_folder_path, f)
        for f in os.listdir(input_folder_path)
    ]
    input_image_paths = [
        p for p in all_input_entries
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    ]
    y_true = [os.path.basename(image_file).split("_")[0] for image_file in input_image_paths]
    labels = []

    # Build hash map so we don't need to do n^2 to calculate labels
    cluster_map = {}
    for i, cluster_folder in enumerate(os.listdir(output_folder_path)):
        cluster_folder_path = os.path.join(output_folder_path, cluster_folder)
        if os.path.isdir(cluster_folder_path):
            for file in os.listdir(cluster_folder_path):
                cluster_map[file] = i

    for image_file in input_image_paths:
        labels.append(cluster_map[os.path.basename(image_file)])

    print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, labels)}")
    print(f"Adjusted Mutual Information: {adjusted_mutual_info_score(y_true, labels)}")
    print(f"V-Measure Score: {v_measure_score(y_true, labels)}")
