import argparse
import os
import sys
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../third_party/segment_3d_gaussians')))

import arguments
import scene as scene_module
import gaussian_renderer
import hdbscan
import pickle
from tqdm import tqdm
import time

FEATURE_DIM = 32
FEATURE_GAUSSIAN_ITERATION = 10000

def get_combined_args(parser : ArgumentParser, model_path, target_cfg_file = None):
    """
    Combines command line arguments and configuration file arguments into a single namespace.

    Args:
        parser (ArgumentParser): The argument parser object.
        model_path (str): The path to the model.
        target_cfg_file (str, optional): The target configuration file. Defaults to None.

    Returns:
        Namespace: A namespace object containing the combined arguments.
    """
    cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    
    if target_cfg_file is None:
        if args_cmdline.target == 'seg':
            target_cfg_file = "seg_cfg_args"
        elif args_cmdline.target == 'scene' or args_cmdline.target == 'xyz':
            target_cfg_file = "cfg_args"
        elif args_cmdline.target == 'feature' or args_cmdline.target == 'coarse_seg_everything' or args_cmdline.target == 'contrastive_feature' :
            target_cfg_file = "feature_cfg_args"

    try:
        cfgfilepath = os.path.join(model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    return Namespace(**merged_dict)

def process_scene(model_path, scale_gate_path):
    """
    Process the scene by loading the model, scale gate, and setting up the dataset.

    Args:
        model_path (str): The path to the model file.
        scale_gate_path (str): The path to the scale gate file.

    Returns:
        tuple: A tuple containing the following elements:
            - scene (Scene): The scene object.
            - pipeline (PipelineParams): The pipeline parameters.
            - scene_gaussians (GaussianModel): The scene gaussians model.
            - feature_gaussians (FeatureGaussianModel): The feature gaussians model.
            - scale_gate (torch.nn.Sequential): The scale gate model.
            - args (argparse.Namespace): The command-line arguments.
    """
    scale_gate = torch.nn.Sequential(
      torch.nn.Linear(1, 32, bias=True),
      torch.nn.Sigmoid()
    )

    scale_gate.load_state_dict(torch.load(scale_gate_path))
    scale_gate = scale_gate.cuda()

    parser = ArgumentParser(description="Testing script parameters")
    model = arguments.ModelParams(parser, sentinel=True)
    pipeline = arguments.PipelineParams(parser)
    parser.add_argument('--target', default='scene', type=str)

    args = get_combined_args(parser, model_path)

    dataset = model.extract(args)

    # If use language-driven segmentation, load clip feature and original masks
    dataset.need_features = False

    # To obtain mask scales
    dataset.need_masks = True

    scene_gaussians = scene_module.GaussianModel(dataset.sh_degree)

    feature_gaussians = scene_module.FeatureGaussianModel(FEATURE_DIM)
    scene = scene_module.Scene(dataset, scene_gaussians, feature_gaussians, load_iteration=-1, feature_load_iteration=FEATURE_GAUSSIAN_ITERATION, shuffle=False, mode='eval', target='contrastive_feature')

    return scene, pipeline, scene_gaussians, feature_gaussians, scale_gate, args

def preprocess_point_features(feature_gaussians, gates):
    """
    Preprocesses the point features by normalizing and sampling them.

    Args:
        feature_gaussians (Tensor): The input feature gaussians.
        gates (Tensor): The gates for conditioning the point features.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the normalized point features and the normalized sampled point features.
    """
    point_features = feature_gaussians.get_point_features
    scale_conditioned_point_features = torch.nn.functional.normalize(point_features, dim=-1, p=2) * gates.unsqueeze(0)
    normed_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim=-1, p=2)
    sampled_point_features = scale_conditioned_point_features[torch.rand(scale_conditioned_point_features.shape[0]) > 0.98]
    normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim=-1, keepdim=True)

    return normed_point_features, normed_sampled_point_features

def extract_cluster(cluster_label, point_colors, seg_score):
    """
    Extracts the points belonging to a specified cluster and replaces points not in the cluster with black.

    Args:
        cluster_label (int): The label of the cluster to extract.
        point_colors (numpy.ndarray): Array of colors for each point.
        seg_score (torch.Tensor): Segmentation scores for each point.

    Returns:
        numpy.ndarray: Array of colors with points not in the specified cluster replaced with black.
    """
    # Create a copy of point_colors
    cluster_colors = np.copy(point_colors)
    
    # Identify points belonging to the specified cluster
    cluster_indices = seg_score.argmax(dim=-1).cpu().numpy() == cluster_label
    
    # Replace points not in the specified cluster with black
    cluster_colors[~cluster_indices] = (0, 0, 0)
    
    return cluster_colors

def cluster_gaussians(scene_gaussians, normed_sampled_point_features, normed_point_features):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01)
    cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())
    cluster_centers = torch.zeros(len(np.unique(cluster_labels)) - 1, normed_sampled_point_features.shape[-1])

    for i in range(1, len(np.unique(cluster_labels))):
        cluster_centers[i - 1] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels == i - 1].mean(dim=0), dim=-1)

    # segmenting with all labels
    seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), normed_point_features.cpu())

    #TODO: Cann we use the same color for the same cluster every time? OpenNerf has the SCANNET Colorlist, maybe this is a start? Lets get rid of the randomness
    np.random.seed(12)
    label_to_color = np.random.rand(250, 3)

    point_colors = label_to_color[seg_score.argmax(dim=-1).cpu().numpy()] #
    point_colors[seg_score.max(dim=-1)[0].detach().cpu().numpy() < 0.5] = (0, 0, 0)

    try:
        scene_gaussians.roll_back()
    except:
        pass

    cluster_ids = np.unique(cluster_labels)
    return cluster_ids, point_colors, seg_score

def get_masks(cluster_ids, point_colors, seg_score, scene_gaussians, pipeline, cameras, img_index, args):
    view = deepcopy(cameras[img_index])
    img = view.original_image * 255
    img = img.permute([1,2,0]).detach().cpu().numpy().astype(np.uint8)

    cluster_masks = torch.empty([len(cluster_ids), img.shape[0], img.shape[1]], dtype=torch.int8)

    for id in cluster_ids:
        new_points = extract_cluster(id, point_colors, seg_score)
        bg_color = [0 for i in range(FEATURE_DIM)]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        rendered_seg_map = gaussian_renderer.render(cameras[img_index], scene_gaussians, pipeline.extract(args), background, override_color=torch.from_numpy(new_points).cuda().float())['render']
        img_seg = rendered_seg_map.permute([1,2,0]).detach().cpu()
        images_sum = img_seg.sum(axis=2)
        #TODO: This 0.5 is a magic number atm (we check if the sum over all three channels is over 0.5), we should find a better way to determine the threshold
        masks = torch.where(images_sum > 0.5, 1.0, 0.)
        cluster_masks[id] = masks

    return cluster_masks

def main(splatting_model, images_folder, out_path):

    # Define necessary paths
    model_path = splatting_model
    scale_gate_path = os.path.join(model_path, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')

    # Obtain features
    scene, pipeline, scene_gaussians, feature_gaussians, scale_gate, args = process_scene(model_path, scale_gate_path)
    cameras = scene.getTrainCameras()

    with torch.no_grad():
        scale = torch.tensor([0.8]).cuda()
        gates = scale_gate(scale)

    normed_point_features, normed_sampled_point_features = preprocess_point_features(feature_gaussians, gates)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    cluster_ids, point_colors, seg_score = cluster_gaussians(scene_gaussians, normed_sampled_point_features, normed_point_features)

    image_files = sorted(os.listdir(images_folder))

    for img_index in tqdm(range(len(cameras)), desc="Extracting Saga Masks...", unit='img', ncols=100):
        img_name = image_files[img_index].split('.')[0]

        image_aligned_masks = get_masks(cluster_ids, point_colors, seg_score, scene_gaussians, pipeline, cameras, img_index, args)

        if image_aligned_masks.dtype != torch.int8:
            image_aligned_masks = torch.tensor(image_aligned_masks, dtype=torch.int8)

        # Convert to a sparse tensor
        sparse_mask = image_aligned_masks.to_sparse()
        
        # Save the sparse tensor as a .pt file
        torch.save(sparse_mask, os.path.join(out_path, f"{img_name}_pred.pt"))

if __name__ == "__main__":
    """
    This script extracts SAGA masks based on a Gaussian Splatting model for a given scene.

    Example usage:
    python get_saga_masks.py --splatting_model <GS model path> --images <images folder> --out_path <output folder> --benchmark
    """
    parser = argparse.ArgumentParser(description="Run SAGA predictions on a directory of images.")
    parser.add_argument('--splatting_model', required=True, help="Path to the Gaussian Splatting model")
    parser.add_argument('--images', required=True, help="Directory containing input images corresponding to the GS model's scene.")
    parser.add_argument('--out_path', required=True, help="Directory to store output predictions.")

    args = parser.parse_args()
    main(args.splatting_model, args.images, args.out_path)