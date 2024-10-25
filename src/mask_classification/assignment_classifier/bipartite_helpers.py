import pickle
import networkx as nx
import torch
import cv2
import numpy as np

WEIGHT_THRESHOLD = 0.05
N_DUPLICATES_2D = 3

def calculate_weight(tensor_a, tensor_b):
    if tensor_a.is_sparse:
        tensor_a = tensor_a.to_dense()
    if tensor_b.is_sparse:
        tensor_b = tensor_b.to_dense()

    weight = jaccard_index(tensor_a, tensor_b, dim=(0, 1))
    return weight

def build_mask_graph(openseg_masks, saga_masks):
    B = nx.Graph()
    set_a = [f"a{i}_{k}" for i in range(openseg_masks.shape[0]) for k in range(N_DUPLICATES_2D)]
    set_b = [f"b{i}" for i in range(saga_masks.shape[0])]
    B.add_nodes_from(set_a, bipartite=0)
    B.add_nodes_from(set_b, bipartite=1)
    for i, openseg_tensor in enumerate(openseg_masks):
        for j, saga_tensor in enumerate(saga_masks):
            saga_tensor_cleaned = postprocess_mask(saga_tensor)
            if torch.all(saga_tensor_cleaned == 0) or torch.all(openseg_tensor == 0):
                continue
            weight = calculate_weight(openseg_tensor, saga_tensor_cleaned)
            if weight < WEIGHT_THRESHOLD:
                continue
            for k in range(N_DUPLICATES_2D): 
                B.add_edge(f"a{i}_{k}", f"b{j}", weight=weight)
    return B

def postprocess_mask(mask):
    if mask.is_sparse:
        saga_tensor = saga_tensor.to_dense()

    cleaned_mask = mask.numpy().astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    min_area = 1000
    cleaned_mask_np = np.zeros(cleaned_mask.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area: 
            cleaned_mask_np[labels == i] = 1
    return torch.tensor(cleaned_mask_np, dtype=torch.int)

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def jaccard_index(tensor1, tensor2, dim):
    # Only works for int tensors
    intersection = (tensor1 & tensor2).sum(dim=dim)
    union = (tensor1 | tensor2).sum(dim=dim)
    return intersection / union


