import os
import pathlib
import torch
from typing import Tuple, Dict, List
from torch.utils.data import Dataset
from PIL import Image

class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None, target_transform=None) -> None:
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*/*.png"))  # Adjust for different file types as needed
        # Setup transforms
        self.transform = transform
        self.target_transform = target_transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = self.find_classes(targ_dir)
        # Extract labels for all images
        self.labels = [self.class_to_idx[path.parent.parent.name] for path in self.paths]

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        image = Image.open(image_path).convert("RGB")
        return image

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        image = self.load_image(index)
        class_name = self.paths[index].parent.parent.name
        class_idx = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            class_idx = self.target_transform(class_idx)

        return image, class_idx