from abc import ABC, abstractmethod
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.processing import run_command
import files

class Evaluator(ABC):
    def __init__(self, dataset: str, scene: str, masks: str, pairing: str):
        super().__init__()
        self.dataset = dataset
        self.scene = scene
        self.masks = masks
        self.pairing = pairing
        self.EXPERIMENT_NAME = "benchmark"

    def run_opennerf(self):
        run_command(f'{files.OPENNERF_ENV_PYTHON} {files.OPENNERF_PATH}scripts/train_eval_semantics.py --dataset {self.dataset} --scene {self.scene}')
    
    def eval_scene(self):
        cmd = [f"{files.OPENNERF_ENV_PYTHON}",
        f"{files.BASE_PATH}src/eval/{self.dataset}_semantics.py",
        f"interpolate",
        f"--interpolation-steps=1",
        f"--pose_source=train",
        f"--load-config={files.OPENNERF_PATH}outputs/{self.dataset}_{self.scene}/opennerf/{self.EXPERIMENT_NAME}/config.yml",
        f"--colormap-options.colormap=pca",
        f"--output_path={files.OPENNERF_PATH}outputs/{self.dataset}_{self.scene}/opennerf/{self.EXPERIMENT_NAME}/",
        f"--rendered-output-names=rgb",
        f"--eval-num-rays-per-chunk=500",
        f"--downscale-factor=1",
        f"--semantics_input_path={files.RESULT_PATH}semantic_labels/{self.pairing}_{self.scene}_{self.masks}.pickle",
        f"--semantics_output_path={files.RESULT_PATH}pr_semantics/semantics_{self.pairing}_{self.scene}_{self.masks}.txt",]
        run_command(cmd)
    
    def eval_semantics(self):
        if self.dataset == "replica":
            run_command(f'{files.OPENNERF_ENV_PYTHON} /home/luca_luis/adl4cv/dcseg/src/eval/evaluate_replica.py --scene {self.scene} --masks {self.masks} --pairing {self.pairing}')
        elif self.dataset == "scannet":
            run_command(f'{files.SCANNET_ENV_PYTHON} /home/luca_luis/adl4cv/dcseg/src/eval/evaluate_scannet.py --scene {self.scene} --masks {self.masks} --pairing {self.pairing}')