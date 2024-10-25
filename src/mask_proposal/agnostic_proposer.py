import os
from abc import ABC, abstractmethod

class AgnosticProposer(ABC):
    def __init__(self, image_source: str, agnostic_2dmask_dir: str):
        self.image_source = image_source
        self.agnostic_2dmask_dir = agnostic_2dmask_dir

        # Check if the image source path exists
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"The image source directory '{image_source}' does not exist.")

        # Ensure the agnostic_2dmask_directory exists
        os.makedirs(agnostic_2dmask_dir, exist_ok=True)

    @abstractmethod
    def propose_masks(self):
        pass

    @abstractmethod
    def save_mask(self, mask):
        pass
