import numpy as np
import clip
import matplotlib.pyplot as plt
import scipy.ndimage
import pickle
from tqdm import tqdm
import argparse
import os
import sys
import torch
from pathlib import Path
import os.path as osp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../third_party/opennerf')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from opennerf.data.utils.openseg_dataloader import OpenSegDataloader
import datasets.replica as replica

import files
import data_processing.scannet.scannet_processing as scannet_processing

def main(image_dir, out_dir, dataset, scene_name):
    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))

    scene = os.path.basename(out_dir)
    image_paths = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".jpg"):
            image_paths.append(os.path.join(image_dir, filename))

    #NOTE: If this script is run multiple times, the cache_dir should be deleted before running it again, because it wont newly extract the features if there is a npy file in the cache dir
    cache_dir = f"{files.RESULT_PATH}openseg_features/{scene}"
    if not os.path.exists(os.path.dirname(cache_dir)):
        os.makedirs(os.path.dirname(cache_dir))

    openseg_cache_path = Path(osp.join(cache_dir, "openseg.npy"))
    OpenSegDataloader(
                    image_list=image_paths,
                    device=torch.device("cuda"),
                    cfg={"image_shape": list([files.IMG_HEIGHT, files.IMG_WIDTH])},
                    cache_path=openseg_cache_path,
                )
    
    if dataset == 'replica':
        class_names = replica.class_names_reduced
    elif dataset == 'scannet':
        class_names = scannet_processing.extract_classnames()
    
    clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cpu', jit=False)
    semantic_class_names_tokenized = clip.tokenize(class_names)  # tokenize
    text_features = clip_pretrained.encode_text(semantic_class_names_tokenized)  # encode
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # normalize
    text_features = text_features.detach().numpy()

    os_features = np.load(f"{files.RESULT_PATH}openseg_features/{scene}/openseg.npy")
    for idx in tqdm(range(os_features.shape[0])):
        img_features = os_features[idx]

        semantic_classes = np.argmax(img_features @ text_features.T, axis=-1)
        semantic_classes_upsample = scipy.ndimage.zoom(semantic_classes, 4, order=0)

        openseg_pred_binary = np.zeros([len(class_names), semantic_classes_upsample.shape[0], semantic_classes_upsample.shape[1]], dtype=np.int8)
        for i in range(len(class_names)):
            openseg_pred_binary[i][semantic_classes_upsample == i] = 1
        img_idx = str(idx+1).zfill(5)
        openseg_pred_binary = torch.tensor(openseg_pred_binary, dtype=torch.int8)
        with open(os.path.join(out_dir, f"frame_{img_idx}_pred.pkl"), 'wb') as f:
            pickle.dump(openseg_pred_binary, f)

if __name__ == "__main__":
    """
    Make sure to have the opennerf conda environment activated before running this script.

    Make sure to create the dir of the output path before running this script. Even though there is logic that should check if this dir exists and create it in case,
    due to some bug it is not working

    Example usage:
    python opennerf/get_openseg_masks.py --images <image_path> --out_path <output_path>
    """
    parser = argparse.ArgumentParser(description="Run Openseg predictions on a directory of images.")
    parser.add_argument('--images', required=True, help="Directory containing input images.")
    parser.add_argument('--dataset', required=True, help="Dataset from which the scene is.")
    parser.add_argument('--scene_name', required=True, help="Scene name.")
    parser.add_argument('--out_path', required=True, help="Directory to store output predictions.")

    args = parser.parse_args()
    main(args.images, args.out_path, args.dataset, args.scene_name)