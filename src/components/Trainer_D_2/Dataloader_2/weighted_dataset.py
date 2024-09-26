import torch
from torch.utils.data import Dataset
from PIL import Image

class WeightedImageDataset(Dataset):
    def __init__(self, image_paths_list, weights_list, labels_list, transforms=None):
        # Flatten the nested lists
        self.image_paths = [path for sublist in image_paths_list for path in sublist]
        self.weights = [weight for sublist in weights_list for weight in sublist]
        self.labels = [label for sublist in labels_list for label in sublist]

        # if None, no transformations will be applied
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, image_path):
        try:
            image = Image.open(image_path)
            if self.transforms:
                image = self.transforms(image)  # Apply the model specific transformations
            return image
        except Exception as e:
            print(f"Error loading image from {image_path}: {str(e)}")
            return None

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        weight = self.weights[idx]
        label = self.labels[idx]

        image = self.load_image(image_path)
        return image, weight, label, image_path
