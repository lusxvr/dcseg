import os
import shutil

# base path
BASE_PATH = ".../dcseg/" # needs to be adjusted when cloning this repo
DATA_PATH = f"{BASE_PATH}data/"

OPENNERF_PATH = f"{BASE_PATH}src/third_party/opennerf/"

# env python paths
SAGA_ENV_PYTHON = ".../anaconda3/envs/gaussian_splatting/bin/python" # needs to be adjusted when cloning this repo
OPENNERF_ENV_PYTHON = ".../anaconda3/envs/opennerf/bin/python" # needs to be adjusted when cloning this repo
OVSEG_ENV_PYTHON = ".../anaconda3/envs/ovseg/bin/python" # needs to be adjusted when cloning this repo
SCANNET_ENV_PYTHON = ".../anaconda3/envs/scannet/bin/python" # need to be adjusted when cloning this repo
DCSEG_ENV_PYTHON = ".../anaconda3/envs/dcseg/bin/python" # need to be adjusted when cloning this repo

# model checkpoints
SAM_CKPT = f'{BASE_PATH}models/sam_ckpt/sam_checkpoint.pth'
CLIP_CKPT = f'{BASE_PATH}models/clip_ckpt/ViT-B-16-laion2b-s34b_b88k.bin'
OVSEG_CKPT = f'{BASE_PATH}models/ovseg_ckpt/ovseg_swinbase_vitL14_ft_mpt.pth'

# Nerfstudio Data
NERFSTUDIO_PATH = f'{DATA_PATH}nerfstudio/{{}}_{{}}/'
NERFSTUDIO_IMAGE_PATH = f'{NERFSTUDIO_PATH}color/'
NERFSTUDIO_DEPTH_PATH = f'{NERFSTUDIO_PATH}depth/'
NERFSTUDIO_INTRINSIC_PATH = f'{NERFSTUDIO_PATH}intrinsic/'
NERFSTUDIO_POSE_PATH = f'{NERFSTUDIO_PATH}pose/'

COLMAP_PATH = f'{BASE_PATH}data/{{}}/{{}}'

# Scannet Data
SCANNET_RAW_PATH = f'{DATA_PATH}raw/scannet/{{}}/'
SCANNET_PATH = f'{DATA_PATH}scannet/{{}}'
SCANNET_INPUT_PATH = f'{DATA_PATH}scannet/{{}}/input'
SCANNET_IMAGES_PATH = f'{DATA_PATH}scannet/{{}}/images'

SCANNET_CLASSES_PATH = f"{BASE_PATH}scannet.py" #f'{SCANNET_PATH}/{{}}.aggregation.json'

# outputs
RESULT_PATH = f'{BASE_PATH}results/'

GS_OUT_PATH = f'{RESULT_PATH}splatting_models/{{}}'
SAGA_MASKS_OUT_PATH = f'{RESULT_PATH}saga_masks/{{}}'
OPENSEG_FEATURES_OUT_PATH = f'{RESULT_PATH}openseg_features/{{}}'
OPENSEG_MASKS_OUT_PATH = f'{RESULT_PATH}openseg_masks/{{}}'
OVSEG_MASKS_OUT_PATH = f'{RESULT_PATH}ovseg_masks/{{}}'
OPENNERF_OUTPUT_PATH = f'{RESULT_PATH}opennerf_outputs/{{}}_{{}}'
CLASS_ASSIGNMENT_PATH = f"{RESULT_PATH}class_assignments/{{}}_{{}}_{{}}.json" # scene, model_2d
SEMANTIC_CLASS_LABELS_PATH = f"{RESULT_PATH}semantic_labels/{{}}_{{}}_{{}}.pickle" # scene, model_2d

# image dimensions
IMG_WIDTH = 1296
IMG_HEIGHT = 968

# clean folders
def clean_data_folder():
    subdirectories_to_clean = [
        'replica_saga'
    ]

    for subdirectory in subdirectories_to_clean:
        full_path = os.path.join(BASE_PATH, 'data', subdirectory)
        
        if os.path.exists(full_path):
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"Cleaned: {full_path}")
        else:
            print(f"Directory not found: {full_path}")

def clean_results_folder():    
    subdirectories_to_clean = [
        'splatting_models',
        'opennerf_outputs',
        'class_assignments',
        'openseg_features',
        'openseg_masks',
        'ovseg_masks',
        'saga_masks',
        'semantic_labels'
    ]
    
    for subdirectory in subdirectories_to_clean:
        full_path = os.path.join(RESULT_PATH, subdirectory)
        
        if os.path.exists(full_path):
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"Cleaned: {full_path}")
        else:
            print(f"Directory not found: {full_path}")