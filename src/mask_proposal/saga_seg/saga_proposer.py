import os
import torch

from ..agnostic_proposer import AgnosticProposer
from utils.processing import run_command
import files

import os

class SagaMaskProposer(AgnosticProposer):
    def __init__(self, image_source: str, agnostic_2dmask_dir: str, gs_path: str, colmap_path: str):
        super().__init__(image_source, agnostic_2dmask_dir)
        self.gs_path = gs_path
        self.colmap_path = colmap_path

        # Check if the Gaussian Splatting path exists
        if os.path.exists(gs_path):
            print("[INFO] Gaussian Splatting Reconstruction exists. Using the existing model.")
        else:
            if colmap_path is None:
                raise ValueError("colmap_path must be specified if Gaussian Splatting Reconstruction does not exist.")
            print("[INFO] Gaussian Splatting Reconstruction does not exist. Performing Gaussian Splatting 3D Reconstruction...")
            run_command(f'{files.DCSEG_ENV_PYTHON} {files.BASE_PATH}src/third_party/segment_3d_gaussians/train_scene.py -s {colmap_path} -m {gs_path}')

    def extract_features(self, skip_scale=False, skip_contrastive=False):
        if not skip_scale:
            if self.colmap_path is None:
                raise ValueError("colmap_path must be specified to extract features.")
            
            run_command(f'{files.SAGA_ENV_PYTHON} {files.BASE_PATH}src/third_party/segment_3d_gaussians/get_scale.py --image_root {self.colmap_path} --model_path {self.gs_path}')
        
        if not skip_contrastive:
            run_command(f'{files.SAGA_ENV_PYTHON} {files.BASE_PATH}src/third_party/segment_3d_gaussians/train_contrastive_feature.py -m {self.gs_path} --iterations 10000 --num_sampled_rays 1000')

    def get_sam_masks(self, downscale=2):
        if self.colmap_path is None:
                raise ValueError("colmap_path must be specified to extract SAM masks.")
        run_command(f'{files.SAGA_ENV_PYTHON} {files.BASE_PATH}src/third_party/segment_3d_gaussians/extract_segment_everything_masks.py --image_root {self.colmap_path} --sam_checkpoint_path {files.SAM_CKPT} --downsample {downscale}')

    def propose_masks(self):
        # Placeholder implementation, to be defined as per the use case
        run_command(f'{files.SAGA_ENV_PYTHON} {files.BASE_PATH}src/mask_proposal/saga_seg/get_saga_masks.py --splatting_model {self.gs_path} --images {self.colmap_path + "/images"} --out_path {self.agnostic_2dmask_dir}')

    def save_mask(self, mask):
        # Check if the mask is a 2D tensor
        if not isinstance(mask, torch.Tensor):
            raise TypeError("The mask must be a PyTorch tensor.")
        
        if len(mask.shape) != 2:
            raise ValueError("The mask must be a 2D tensor with shape (h, w).")
        
        # Define the output file path for the mask
        mask_file_path = os.path.join(self.agnostic_2dmask_directory, "mask.pt")
        
        # Save the mask as a .pt file (PyTorch tensor)
        torch.save(mask, mask_file_path)
        print(f"Mask saved to {mask_file_path}")
