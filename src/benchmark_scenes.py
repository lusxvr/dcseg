import argparse
import files

from mask_proposal.saga_seg.saga_proposer import SagaMaskProposer
from mask_classification.assignment_classifier.assignment_classifier import AssignmentClassifier
from eval.evaluator import Evaluator
from utils.types import AwarePredictionType

import data_processing.scannet.scannet_processing as scannet_processing

DOWNSAMPLING_FACTOR = 3

def process_scene(dataset, scene_name):

    # Define paths
    colmap_path = files.COLMAP_PATH.format(dataset, scene_name)
    image_path = files.SCANNET_IMAGES_PATH.format(scene_name)
    gs_out_path = files.GS_OUT_PATH.format(scene_name)

    ovseg_masks_out_path = files.OVSEG_MASKS_OUT_PATH.format(scene_name)
    saga_masks_out_path = files.SAGA_MASKS_OUT_PATH.format(scene_name)

    # download scannet data and convert to compatible format
    if dataset == "scannet":
        scannet_processing.preprocess_scene_data(scene_name, downsampling_factor=DOWNSAMPLING_FACTOR)
    elif dataset == "replica":
        pass # If the preprocessing script for Replica was run, everything should be in the correct format

    proposer = SagaMaskProposer(
        image_source=image_path, 
        agnostic_2dmask_dir=saga_masks_out_path, 
        gs_path=gs_out_path,
        colmap_path=colmap_path
    )

    proposer.get_sam_masks()
    proposer.extract_features()
    proposer.propose_masks()

    mask_classifier = AssignmentClassifier(
        image_source=image_path, 
        agnostic_mask_directory=saga_masks_out_path, 
        aware_mask_directory=ovseg_masks_out_path
    )

    mask_classifier.predict_aware_masks(dataset, prediction_type=AwarePredictionType.OVSEG)
    mask_classifier.get_assignments(dataset=dataset, scene_id=scene_name, prediction_type=AwarePredictionType.OVSEG)

    # To utilize the NeRF Studio Visualization, we need to bring the data into the format it expects, this is easily done by running the opennerf pipeline, 
    # but If you are not interested in the visualization, you can skip this step
    # Additionally, the evaluation scripts also depend on the back-projection of the predicions to the original given point_clouds, so this is also done through the NERF Studio pipeline,
    # but if you are not interested in the evaluation (e.g. you dont have GT), you can skip this step
    # Note that our pipeline is already completely finished after the previous setep, so you can skip this step if you are not interested in the visualization or evaluation and still
    # utilize the predictions for your own purposes
    evaluator = Evaluator(dataset, scene_name)
    evaluator.run_opennerf()
    evaluator.eval_scene()
    evaluator.eval_semantics()
    
    
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Process 3D segmentation for multiple scenes')
    parser.add_argument('--scene_names', nargs='+', required=True, help='List of scenes to process')
    parser.add_argument('--dataset', required=True, choices=['scannet', 'replica'], help='Dataset to use (scannet or replica)')

    args = parser.parse_args()

    # Check if the dataset is valid
    if args.dataset not in ['scannet', 'replica']:
        print("Error: Dataset must be either 'scannet' or 'replica'.")
        exit(1)

    # Iterate over each scene
    for scene_name in args.scene_names:
        process_scene(args.dataset, scene_name)
