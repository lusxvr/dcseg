import os

from mask_classification.mask_classifier import MaskClassifier
from utils.types import AwarePredictionType
from utils.processing import run_command

import files
    
class AssignmentClassifier(MaskClassifier):
    def __init__(self, image_source: str, agnostic_mask_directory: str, aware_mask_directory: str):
        super().__init__(agnostic_mask_directory)
        self.aware_mask_directory = aware_mask_directory
        self.image_source = image_source

        # check if aware_mask_directory exists and create it if not
        if not os.path.exists(self.aware_mask_directory):
            os.makedirs(self.aware_mask_directory)

    def predict_aware_masks(self, dataset, prediction_type: AwarePredictionType):

        if prediction_type == AwarePredictionType.OVSEG:
            run_command(f'{files.OVSEG_ENV_PYTHON} {files.BASE_PATH}src/mask_classification/assignment_classifier/get_ovseg_masks.py --dataset {dataset} --ovseg_model {files.OVSEG_CKPT} --images {self.image_source} --out_path {self.aware_mask_directory}')    
        elif prediction_type == AwarePredictionType.OPENSEG:
            pass

    def get_assignments(self, dataset, scene_id, prediction_type: AwarePredictionType):
        if prediction_type == AwarePredictionType.OVSEG:
            run_command(f'{files.DCSEG_ENV_PYTHON} {files.BASE_PATH}src/mask_classification/assignment_classifier/bipartite_assignment.py --dataset {dataset} --scene {scene_id} --ovseg --masks_3dmodel_path {self.agnostic_mask_directory} --masks_2dmodel_path {self.aware_mask_directory}')
        elif prediction_type == AwarePredictionType.OPENSEG:
            pass
