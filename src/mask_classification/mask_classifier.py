from abc import ABC, abstractmethod
import os
import torch

class MaskClassifier(ABC):
    def __init__(self, agnostic_mask_directory: str):
        self.agnostic_mask_directory = agnostic_mask_directory

    def load_agnostic_masks(self, filename: str):
        # Construct the full path to the .pt file
        file_path = os.path.join(self.agnostic_mask_directory, filename)
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Load the tensor from the .pt file
        agnostic_masks = torch.load(file_path)
        
        # Ensure the loaded object is a tensor
        if not isinstance(agnostic_masks, torch.Tensor):
            raise TypeError(f"Loaded object is not a torch.Tensor. Found type: {type(agnostic_masks)}")
        
        return agnostic_masks