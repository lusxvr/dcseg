import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../third_party/ov-seg')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from open_vocab_seg.utils.predictor import OVSegPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

import class_names as classes
import files

from tqdm import tqdm
import torch
import numpy as np

def setup_cfg(cfg_path, model_path):
    """Sets up the configuration for the model.

    Args:
        cfg_path (str): Path to the configuration file.
        model_path (str): Path to the model weights.

    Returns:
        cfg: Configured model.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(["MODEL.WEIGHTS", model_path])
    cfg.freeze()
    return cfg

def predict_and_save(model, image_path, out_path, class_names):
    """Runs prediction on a single image and saves the result as a pickle file.

    Args:
        model (OVSegPredictor): The model for prediction.
        image_path (str): Path to the input image.
        out_path (str): Directory where the output will be saved.
        class_names (list): List of class names for prediction.
    """
    img = read_image(image_path, format="RGB")
    pred = model(img, class_names=class_names)
    predictions = pred['sem_seg']
    image_name = os.path.basename(image_path).split('.')[0]

    #TODO: Another magic number as the threshold
    # ovseg_masks_binary = (predictions > 0.5).int().cpu()

    # Getting new ovseg masks with softmax
    sm = torch.nn.Softmax(dim=0)
    sm_pred = sm(predictions).detach().cpu().numpy()
    max_category = np.argmax(sm_pred, axis=0)
    sm_pred_binary = np.zeros_like(sm_pred, dtype=np.int8)
    for i in range(predictions.shape[0]):
        sm_pred_binary[i][max_category == i] = 1
    sm_pred_binary = torch.tensor(sm_pred_binary, dtype=torch.int8)

    # Convert to a sparse tensor
    sparse_sm_pred_binary = sm_pred_binary.to_sparse()
    torch.save(sparse_sm_pred_binary, os.path.join(out_path, f"{image_name}_pred.pt"))

def main(images_folder, ovseg_model, dataset, out_path):
    """Main function to process images and save predictions.

    Args:
        ovseg_model (str): Path to the model weights.
        images_folder (str): Directory containing images.
        out_path (str): Directory where outputs will be stored.
    """

    setup_logger()
    cfg_path = f"{files.BASE_PATH}src/third_party/ov-seg/configs/ovseg_swinB_vitL_demo.yaml"
    cfg = setup_cfg(cfg_path, ovseg_model)
    model = OVSegPredictor(cfg)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_name in tqdm(sorted(os.listdir(images_folder))):
        image_path = os.path.join(images_folder, image_name)

        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            if dataset == 'replica':
                class_names = classes.replica_classes_reduced
            elif dataset == 'scannet':
                class_names = classes.scannet_classes_20
            predict_and_save(model, image_path, out_path, class_names)


if __name__ == "__main__":
    """
    Make sure to have the ovseg conda environment activated before running this script.

    Example usage:
    python get_ovseg_masks.py --images <image_path> --class_names --out_path <output_path>
    """
    parser = argparse.ArgumentParser(description="Run OVSeg predictions on a directory of images.")
    parser.add_argument('--images', required=True, help="Directory containing input images.")
    parser.add_argument('--ovseg_model', required=True, help="Path to ovseg checkpoint.")
    parser.add_argument('--dataset', required=True, help="Used Dataset.")
    parser.add_argument('--out_path', required=True, help="Directory to store output predictions.")

    args = parser.parse_args()
    main(args.images, args.ovseg_model, args.dataset, args.out_path)
