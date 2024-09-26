import os
import pathlib
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class BinaryClassificationDataset(Dataset):
    def __init__(self, targ_dir: str, transform=None, target_transform=None) -> None:
        self.paths = list(pathlib.Path(targ_dir).glob("*/*/*.png"))
        self.transform = transform
        self.target_transform = target_transform
        self.classes, self.class_to_idx = self.find_classes(targ_dir)
        
        self.non_distracted_classes = {'sitting_still', 'entering_car', 'exiting_car'}
        self.class_to_idx_binary = {cls_name: 0 if cls_name in self.non_distracted_classes else 1 for cls_name in self.classes}
        
        # Map binary labels to class names
        self.binary_label_to_class_name = {0: 'non_distracted', 1: 'distracted'}

        # Attribute for all binary labels of the dataset
        self.all_binary_labels = [self.class_to_idx_binary[path.parent.parent.name] for path in self.paths]

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        image = Image.open(image_path).convert("RGB")
        return image

    def __len__(self) -> int:
        return len(self.paths)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        image = self.load_image(index)
        class_name = self.paths[index].parent.parent.name
        class_idx = self.class_to_idx[class_name]
        class_idx_binary = self.class_to_idx_binary[class_name]
        
        # Convert binary label to its corresponding class name
        class_name_binary = self.binary_label_to_class_name[class_idx_binary]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            class_idx_binary = self.target_transform(class_idx_binary)
        
        # Return the image, binary class index, and binary class name
        return image, class_idx_binary, class_idx, class_name_binary
    

class BinaryClassificationDatasetAnalysis(BinaryClassificationDataset):
    def __init__(self, targ_dir: str, dataset_name='Dataset', dataset_split="split_0", transform=None, target_transform=None) -> None:
        super().__init__(targ_dir, transform, target_transform)
        # Attribute to store the name of the dataset & its split
        self.dataset_name = dataset_name 
        self.dataset_split_name = dataset_split

    def count_samples_per_class(self):
        # Count the number of samples in each class
        counts = {'non_distracted': 0, 'distracted': 0}
        for label in self.all_binary_labels:
            class_name = self.binary_label_to_class_name[label]
            counts[class_name] += 1
        return counts

    def plot_class_distribution(self):
        counts = self.count_samples_per_class()
        
        # Plotting the class distribution
        plt.figure(figsize=(8, 5))
        bars = plt.bar(counts.keys(), counts.values())
        plt.xlabel('Class')
        plt.ylabel('Number of samples')
        plt.title(f'Class Distribution in {self.dataset_name} ({self.dataset_split_name})')
        
        # Adding count labels above each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def calculate_class_ratios(self):
        counts = self.count_samples_per_class()
        total_samples = sum(counts.values())
        ratios = {class_name: count / total_samples for class_name, count in counts.items()}
        return ratios

    def plot_class_ratios(self):
        ratios = self.calculate_class_ratios()
        
        # Plotting the class ratios
        plt.figure(figsize=(6, 6))  # Use a square figure to make the pie chart circular
        plt.pie(ratios.values(), labels=ratios.keys(), autopct='%1.1f%%', startangle=140)
        plt.title(f'Class Ratios in {self.dataset_name} ({self.dataset_split_name})')
        
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        plt.show()
