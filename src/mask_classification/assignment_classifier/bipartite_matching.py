import argparse
import os
import pickle
import torch
import json
from collections import defaultdict
import numpy as np
import scipy

from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import tqdm.contrib
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
import psutil
import time
import csv
import tqdm
import files_old

import opennerf.datasets.replica as replica
from bipartite_helpers import load_pickle_file, jaccard_index
import data_loading.scannet_processing as scannet_processing

def get_gpu_usage():
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024 ** 3  # Convert bytes to gigabytes

def get_cpu_memory_usage():
    return psutil.virtual_memory().used / 1024 ** 3

def load_masks(file_path):
    masks = load_pickle_file(file_path)
    if not isinstance(masks, torch.Tensor):
        masks = torch.tensor(masks, dtype=torch.int8)
    return masks

def main(saga_masks_dir, ovseg_masks_dir, dataset, scene, benchmark=False):
    nvmlInit()

    if benchmark:
        start_time = time.time()
        gpu_usages = []
        cpu_memory_usages = []
        combined_usages = []

    saga_files = sorted([f for f in os.listdir(saga_masks_dir) if f.endswith('.pkl')])
    ovseg_files = sorted([f for f in os.listdir(ovseg_masks_dir) if f.endswith('.pkl')])

    scene_name = os.path.basename(ovseg_masks_dir)
    masks_name = os.path.basename(os.path.dirname(ovseg_masks_dir)).split("_")[0]

    if len(saga_files) != len(ovseg_files):
        print(f"Mismatch in the number of files between saga_masks and ovseg_masks directories: {len(saga_files)} vs {len(ovseg_files)}")
        return

    cluster_votes = defaultdict(lambda: defaultdict(int))
    semantic_masks_saga = torch.empty([len(saga_files), files_old.IMG_HEIGHT, files_old.IMG_WIDTH])    

    for idx, (saga_file, ovseg_file) in enumerate(tqdm.contrib.tzip(saga_files, ovseg_files, desc="Matching clusters", unit='img', ncols=100)):
        saga_mask_path = os.path.join(saga_masks_dir, saga_file)
        ovseg_mask_path = os.path.join(ovseg_masks_dir, ovseg_file)

        saga_masks = load_masks(saga_mask_path)        
        ovseg_masks = load_masks(ovseg_mask_path)

        seg_height = ovseg_masks.shape[1]
        seg_width = ovseg_masks.shape[2]
        padding_seg = (0, int(files_old.IMG_WIDTH-seg_width), 0, int(files_old.IMG_HEIGHT-seg_height))
        ovseg_masks = F.pad(ovseg_masks, padding_seg, "constant", 0)


        saga_height = saga_masks.shape[1]
        saga_width = saga_masks.shape[2]
        padding_saga = (0, int(files_old.IMG_WIDTH-saga_width), 0, int(files_old.IMG_HEIGHT-saga_height))
        saga_masks = F.pad(saga_masks, padding_saga, "constant", 0)

        # argmax of saga_masks over all clusters for this image, needed to map back the cluster indices to the original class indices
        merged_masks = torch.argmax(saga_masks, dim=0)
        semantic_masks_saga[idx] = merged_masks
        
        jaccard_similarities = []

        for i in range(saga_masks.shape[0]):
            tensor1 = saga_masks[i].unsqueeze(0).repeat(ovseg_masks.shape[0], 1, 1)
            tensor2 = ovseg_masks

            jaccard_sim = jaccard_index(tensor1, tensor2, dim=(1, 2))
            jaccard_similarities.append(jaccard_sim)

        # Convert list of similarities to a tensor
        jaccard_similarities = torch.stack(jaccard_similarities)
        jaccard_similarities = torch.nan_to_num(jaccard_similarities, nan=0.).numpy()

        if benchmark:
            gpu_usage = get_gpu_usage()
            cpu_memory_usage = get_cpu_memory_usage()
            combined_usage = gpu_usage + cpu_memory_usage

            gpu_usages.append(gpu_usage)
            cpu_memory_usages.append(cpu_memory_usage)
            combined_usages.append(combined_usage)

        row_ind, col_ind = linear_sum_assignment(1 - jaccard_similarities)
        # cluster_labels = [replica.class_names_reduced[col_ind[j]] for j in range(len(col_ind))]

        if dataset == "replica":
            cluster_labels = [replica.class_names_reduced[col_ind[j]] if jaccard_similarities[row_ind[j], col_ind[j]] != 0 else '0' for j in range(len(col_ind))]
        elif dataset == "scannet":
            class_names = scannet_processing.extract_classnames()
            cluster_labels = [class_names[col_ind[j]] if jaccard_similarities[row_ind[j], col_ind[j]] != 0 else '0' for j in range(len(col_ind))]

        # Count the votes for each cluster
        for i, label in enumerate(cluster_labels):
            if label != '0':
                cluster_votes[i][label] += 1
            elif label == '0':
                cluster_votes[i][label] = 0

    # Determine the majority vote for each cluster
    final_cluster_labels = {}
    for cluster_idx, votes in cluster_votes.items():
        majority_label = max(votes, key=votes.get)
        final_cluster_labels[cluster_idx] = majority_label

    # Match the cluster labels back to the reduced class names
    semantic_class_labels = torch.empty_like(semantic_masks_saga, dtype=torch.int16)

    # Create a lookup table for the labels to indices mapping
    if dataset == "replica":
        label_to_index = {label: idx for idx, label in enumerate(replica.class_names_reduced)}
    elif dataset == "scannet":
        class_names = scannet_processing.extract_classnames()
        label_to_index = {label: idx for idx, label in enumerate(class_names)}

    # Convert the mapping to use indices instead of labels
    mapping_indices = {int(key): label_to_index[label] for key, label in final_cluster_labels.items()}

    # Convert semantic_masks_saga to a NumPy array if it's not already
    semantic_masks_saga_np = semantic_masks_saga.cpu().numpy() if isinstance(semantic_masks_saga, torch.Tensor) else semantic_masks_saga

    # Create an array for the mapped indices
    #mapped_indices = np.vectorize(mapping_indices.get)(semantic_masks_saga_np)
    mapped_indices = np.vectorize(lambda x: mapping_indices.get(x, 0))(semantic_masks_saga_np)

    # Convert back to a torch.Tensor if needed
    semantic_class_labels = torch.tensor(mapped_indices, device=semantic_masks_saga.device)

    # Apply padding
    #semantic_class_labels_padded = F.pad(semantic_class_labels, padding_saga, "constant", 0)

    if benchmark:
        end_time = time.time()
        elapsed_time = end_time - start_time

        avg_gpu_usage = sum(gpu_usages) / len(gpu_usages)
        max_gpu_usage = max(gpu_usages)
        avg_cpu_memory_usage = sum(cpu_memory_usages) / len(cpu_memory_usages)
        max_cpu_memory_usage = max(cpu_memory_usages)
        avg_combined_usage = sum(combined_usages) / len(combined_usages)
        max_combined_usage = max(combined_usages)

        output_file = os.path.join("results", "benchmarks", "train_benchmarks.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        file_exists = os.path.isfile(output_file)

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    'script', 'scene', 'time', 'avg_gpu_usage', 'max_gpu_usage', 'avg_cpu_memory_usage', 'max_cpu_memory_usage', 
                    'avg_combined_usage', 'max_combined_usage'
                ])
            writer.writerow([
                'ours', 'bipartite_matching.py', scene_name, '', '', '', elapsed_time, avg_gpu_usage, max_gpu_usage, avg_cpu_memory_usage, max_cpu_memory_usage, avg_combined_usage, max_combined_usage
            ])

    nvmlShutdown()

    # Write the final cluster labels to a JSON file
    json_path = files_old.CLASS_ASSIGNMENT_PATH.format("matching", scene_name, masks_name)
    with open(json_path, 'w') as json_file:
        json.dump(final_cluster_labels, json_file, indent=4)

    final_masks_filename = files_old.SEMANTIC_CLASS_LABELS_PATH.format("matching", scene_name, masks_name)
    with open(final_masks_filename, 'wb') as handle:
        pickle.dump(semantic_class_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SAGA and OVSEG masks')
    parser.add_argument('--masks_3dmodel_path', required=True, help='Directory containing SAGA masks')
    parser.add_argument('--masks_2dmodel_path', required=True, help='Directory containing OVSEG masks')
    parser.add_argument('--dataset', required=True, help='Path to the class names file')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--benchmark', action='store_true', help="Flag to enable benchmarking of resource usage.")

    args = parser.parse_args()

    main(args.masks_3dmodel_path, args.masks_2dmodel_path, args.dataset, args.scene, args.benchmark)
