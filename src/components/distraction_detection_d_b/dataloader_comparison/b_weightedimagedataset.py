import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler

class WeightedImageDataset(Dataset):
    def __init__(self, image_paths_list, weights_list, labels_list):
        # Flatten the nested lists
        self.image_paths = [path for sublist in image_paths_list for path in sublist]
        self.weights = [weight for sublist in weights_list for weight in sublist]
        self.labels = [label for sublist in labels_list for label in sublist]

        # Define a transformation to preprocess the loaded image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to a fixed size (e.g., 224x224)
            transforms.ToTensor(),          # Convert the image to a PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize the image channels (mean)
                std=[0.229, 0.224, 0.225]    # Normalize the image channels (std)
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = self.transform(image)
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
