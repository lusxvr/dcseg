import os
import gc
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import argparse
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import files

from bipartite_helpers import load_pickle_file, build_mask_graph
import class_names as classes

SAGA_CLUSTER_COUNT = 0

def load_mask_file(file_path):
    """Loads a mask from a .pkl or .pt file."""
    if file_path.endswith('.pkl'):
        mask = load_pickle_file(file_path)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
    elif file_path.endswith('.pt'):
        mask = torch.load(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return mask

def load_saga_masks(saga_masks_dir, saga_files, idx):
    file_path = os.path.join(saga_masks_dir, saga_files[idx])
    return load_mask_file(file_path)

def load_openseg_masks(openseg_masks_dir, openseg_files, idx):
    file_path = os.path.join(openseg_masks_dir, openseg_files[idx])
    return load_mask_file(file_path)

def load_ovseg_masks(ovseg_masks_dir, ovseg_files, idx):
    file_path = os.path.join(ovseg_masks_dir, ovseg_files[idx])
    return load_mask_file(file_path)

def majority_vote(d):
    result = {}
    for key, counts in d.items():
        majority_label = max(counts, key=counts.get)
        result[key] = majority_label
    return dict(result)

def pad_cluster_assignments(input_dict, saga_cluster_count):
    for i in range(saga_cluster_count):
        if i not in input_dict:
            input_dict[i] = '0'
    return input_dict

def main(saga_masks_dir, foundation_model_masks_dir, args):
    if args.dataset == "replica":
        INDEX_TO_LABEL = {idx: label for idx, label in enumerate(classes.replica_classes_reduced)}
    elif args.dataset == "scannet":
        INDEX_TO_LABEL = {idx: label for idx, label in enumerate(classes.scannet_classes_20)}

    scene_name = os.path.basename(saga_masks_dir)
    saga_files = sorted(os.listdir(saga_masks_dir), key=lambda x: int(x.split('_')[0]))
    foundation_model_files = sorted(os.listdir(foundation_model_masks_dir), key=lambda x: int(x.split('_')[0]))
    
    semantic_masks_saga = torch.empty([len(saga_files), files.IMG_HEIGHT, files.IMG_WIDTH])
    saga_label_counts = defaultdict(lambda: defaultdict(int))

    for idx in tqdm(range(len(saga_files)), desc="Assigning clusters", unit='img', ncols=100):
        saga_masks = load_saga_masks(saga_masks_dir, saga_files, idx).int().cpu()

        if saga_masks.is_sparse:
            saga_masks = saga_masks.to_dense()

        if args.openseg:
            foundation_model_masks = load_openseg_masks(foundation_model_masks_dir, foundation_model_files, idx).int().cpu()
        elif args.ovseg:
            foundation_model_masks = load_ovseg_masks(foundation_model_masks_dir, foundation_model_files, idx).int().cpu()

        if foundation_model_masks.is_sparse:
            foundation_model_masks = foundation_model_masks.to_dense()

        # apply padding to foundation model masks and saga masks
        saga_height = saga_masks.shape[1]
        saga_width = saga_masks.shape[2]

        padding = (0, int(files.IMG_WIDTH-saga_width), 0, int(files.IMG_HEIGHT-saga_height))
        saga_masks = F.pad(saga_masks, padding, "constant", 0)

        foundation_model_height = foundation_model_masks.shape[1]
        foundation_model_width = foundation_model_masks.shape[2]

        padding = (0, int(files.IMG_WIDTH-foundation_model_width), 0, int(files.IMG_HEIGHT-foundation_model_height))
        foundation_model_masks = F.pad(foundation_model_masks, padding, "constant", 0)

        saga_masks_merged = torch.argmax(saga_masks, dim=0)
        semantic_masks_saga[idx] = saga_masks_merged

        if idx == 0:
            SAGA_CLUSTER_COUNT = saga_masks.shape[0]

        B = build_mask_graph(foundation_model_masks, saga_masks)
        matching = nx.algorithms.matching.max_weight_matching(B, maxcardinality=True, weight='weight')
        
        for u, v in matching:
            if u.startswith('a') and v.startswith('b'):
                a_node = u
                b_node = v
            elif u.startswith('b') and v.startswith('a'):
                a_node = v
                b_node = u
            else:
                continue

            saga_index = int(b_node[1:])
            foundation_model_index = int(a_node.split('_')[0][1:])

            saga_label_counts[saga_index][INDEX_TO_LABEL[foundation_model_index]] += 1

        # Free the memory by deleting mask variables
        del saga_masks, foundation_model_masks, B, matching

        # Force garbage collection
        gc.collect()

    unpadded_results = majority_vote(saga_label_counts)
    results = pad_cluster_assignments(unpadded_results, SAGA_CLUSTER_COUNT)

    filename = files.CLASS_ASSIGNMENT_PATH.format('assignment', scene_name, 'openseg' if args.openseg else 'ovseg')
    with open(filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    semantic_class_labels = torch.empty_like(semantic_masks_saga, dtype=torch.int16)

    if args.dataset == "replica":
        label_to_index = {label: idx for idx, label in enumerate(classes.replica_classes_reduced)}
    elif args.dataset == "scannet":
        label_to_index = {label: idx for idx, label in enumerate(classes.scannet_classes_20)}

    mapping_indices = {int(key): label_to_index[label] for key, label in results.items()}
    semantic_masks_saga_np = semantic_masks_saga.cpu().numpy()
    mapped_indices = np.vectorize(mapping_indices.get)(semantic_masks_saga_np)
    semantic_class_labels = torch.tensor(mapped_indices, device=semantic_masks_saga.device)

    final_masks_filename = files.SEMANTIC_CLASS_LABELS_PATH.format('assignment', scene_name, 'openseg' if args.openseg else 'ovseg')
    with open(final_masks_filename, 'wb') as f:
        pickle.dump(semantic_class_labels, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process segmentation masks.")
    parser.add_argument('--openseg', action='store_true', help="Use openseg as foundation model")
    parser.add_argument('--ovseg', action='store_true', help="Use ovseg as foundation model")
    parser.add_argument('--masks_3dmodel_path', type=str, required=True, help="Path to the 3D model masks")
    parser.add_argument('--masks_2dmodel_path', type=str, required=True, help="Path to the 2D model masks")
    parser.add_argument('--dataset', required=True, help='Path to the class names file')
    parser.add_argument('--scene', required=True, help='Scene name')
    
    args = parser.parse_args()
    
    if not (args.openseg or args.ovseg):
        raise ValueError("Either --openseg or --ovseg must be specified")
    
    saga_masks_dir = args.masks_3dmodel_path
    foundation_model_masks_dir = args.masks_2dmodel_path
    
    # if args.openseg:
    #     foundation_model_masks_dir = args.masks_2dmodel_path
    #     foundation_model_files = sorted([f for f in os.listdir(foundation_model_masks_dir) if f.endswith('.pkl')])
    # elif args.ovseg:
    #     foundation_model_masks_dir = args.masks_2dmodel_path
    #     foundation_model_files = sorted([f for f in os.listdir(foundation_model_masks_dir) if f.endswith('.pkl')])
    
    main(saga_masks_dir, foundation_model_masks_dir, args)
