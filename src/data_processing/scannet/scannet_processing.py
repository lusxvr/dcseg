import files
import subprocess
import shutil
import os
from . import constants
from .download_scannet import download_scan, FILETYPES
from .sensor_data import SensorData

def run_command(command):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")

def is_directory_empty(path):
    if not os.path.exists(path):
        return True  # Treat non-existing directory as empty
    return not os.listdir(path)

def preprocess_scene_data(scene, downsampling_factor=1):
    # Define some paths
    scannet_raw_path = files.SCANNET_RAW_PATH.format(scene)
    scannet_raw_images_path = f"{scannet_raw_path}color/"
    scannet_raw_depth_path = f"{scannet_raw_path}depth/"
    scannet_raw_pose_path = f"{scannet_raw_path}pose/"
    scannet_raw_intrinsic_path = f"{scannet_raw_path}intrinsic/"
    scannet_sens_data_path = f"{scannet_raw_path}{scene}.sens"
    scannet_input_path = files.SCANNET_INPUT_PATH.format(scene)

    # Download scannet .sens data and ground-truth meshes
    if all(not is_directory_empty(p) for p in [scannet_raw_images_path, scannet_raw_depth_path, scannet_raw_pose_path, scannet_raw_intrinsic_path]):
        print("[INFO] Scannet raw paths are not empty. Skipping data extraction from .sens file.")
    else:
        download_scan(scene, scannet_raw_path, file_types=['.sens', '_vh_clean.ply', '_vh_clean_2.ply'], use_v1_sens=True)

        # Obtain images, depth, pose and intrinsic data from .sens file
        sensor_data = SensorData(scannet_sens_data_path, downsampling_factor)
        sensor_data.export_color_images(scannet_raw_images_path)
        sensor_data.export_depth_images(scannet_raw_depth_path)
        sensor_data.export_poses(scannet_raw_pose_path)
        sensor_data.export_intrinsics(scannet_raw_intrinsic_path)

    # Copy images from raw path to colmap input path
    if not os.path.exists(scannet_input_path):
        os.makedirs(scannet_input_path)
    
    for file_name in os.listdir(scannet_raw_images_path):
        full_file_name = os.path.join(scannet_raw_images_path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, scannet_input_path)
    
    # Ensure the image path exists before running COLMAP
    if not os.path.exists(scannet_input_path) or not os.listdir(scannet_input_path):
        raise FileNotFoundError(f"Image path '{scannet_input_path}' does not exist or is empty.")

    # Check if the colmap path contains non-empty folders
    colmap_path = files.COLMAP_PATH.format("scannet", scene)
    colmap_folders = ["distorted", "images", "images_2", "images_4", "images_8", "input", "sparse", "stereo"]
    if all(not is_directory_empty(os.path.join(colmap_path, folder)) for folder in colmap_folders):
        print("[INFO] Colmap folders are not empty. Skipping convert.py command.")
    else:
        run_command(f'{files.DCSEG_ENV_PYTHON} {files.BASE_PATH}src/mask_proposal/saga_seg/convert.py -s {colmap_path} --resize')
    
    # Create the Nerfstudio paths and check if they are empty
    nerfstudio_image_path = files.NERFSTUDIO_IMAGE_PATH.format("scannet", scene)
    nerfstudio_depth_path = files.NERFSTUDIO_DEPTH_PATH.format("scannet", scene)
    nerfstudio_intrinsic_path = files.NERFSTUDIO_INTRINSIC_PATH.format("scannet", scene)
    nerfstudio_pose_path = files.NERFSTUDIO_POSE_PATH.format("scannet", scene)

    paths_to_check = [
        (nerfstudio_image_path, "images"),
        (nerfstudio_depth_path, "depth"),
        (nerfstudio_intrinsic_path, "intrinsics"),
        (nerfstudio_pose_path, "poses")
    ]

    # Check each Nerfstudio path and skip copying if not empty
    for path, description in paths_to_check:
        if not is_directory_empty(path):
            print(f"[INFO] Nerfstudio {description} path is not empty. Skipping copy process for {description}.")
        else:
            os.makedirs(path, exist_ok=True)
    
    # Copy the images from SAGA to Nerfstudio image path
    if is_directory_empty(nerfstudio_image_path):
        for file_name in os.listdir(files.SCANNET_IMAGES_PATH.format(scene)):
            full_file_name = os.path.join(scannet_input_path, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, nerfstudio_image_path)
    
    # Copy depth images only for corresponding images
    if is_directory_empty(nerfstudio_depth_path):
        depth_src_path = f"{files.SCANNET_RAW_PATH.format(scene)}depth/"
        for image_file in os.listdir(nerfstudio_image_path):
            image_id = os.path.splitext(image_file)[0]
            depth_file = f"{image_id}.png"
            depth_full_file_name = os.path.join(depth_src_path, depth_file)
            if os.path.isfile(depth_full_file_name):
                shutil.copy(depth_full_file_name, nerfstudio_depth_path)
    
    # Copy the whole intrinsics folder
    if is_directory_empty(nerfstudio_intrinsic_path):
        intrinsic_src_path = f"{files.SCANNET_RAW_PATH.format(scene)}intrinsic/"
        shutil.copytree(intrinsic_src_path, nerfstudio_intrinsic_path)
    
    # Copy poses only for corresponding images
    if is_directory_empty(nerfstudio_pose_path):
        pose_src_path = f"{files.SCANNET_RAW_PATH.format(scene)}pose/"
        for image_file in os.listdir(nerfstudio_image_path):
            image_id = os.path.splitext(image_file)[0]
            pose_file = f"{image_id}.txt"
            pose_full_file_name = os.path.join(pose_src_path, pose_file)
            if os.path.isfile(pose_full_file_name):
                shutil.copy(pose_full_file_name, nerfstudio_pose_path)

