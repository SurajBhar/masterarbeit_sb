"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
from src.components.dataset import ImageFolderCustom
from torchvision import transforms
from torch.utils.data import DataLoader

# NUM_WORKERS = os.cpu_count()
# num_workers: int=NUM_WORKERS

def create_dataloaders(
    train_dir: str, 
    val_dir : str,
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int,
    prefetch_factor):
    """Creates training and validation DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to validation directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, val_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, val_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                val_dir=path/to/val_dir,
                                transform=some_transform,
                                batch_size=16,
                                num_workers=4)
    """
    # Use ImageFolder to create dataset(s)
    train_dataset = ImageFolderCustom(train_dir, transform=transform)
    val_dataset = ImageFolderCustom(val_dir, transform=transform)

    # Get class names
    class_names = train_dataset.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last = False,
                                  prefetch_factor = prefetch_factor)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last = False,
                                prefetch_factor = prefetch_factor)

    return train_dataloader, val_dataloader, class_names


def create_test_dataloader(
    test_dir: str,
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int,
    prefetch_factor: int):
    """Creates testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        test_dataloader, class_names = \
            = create_test_dataloader(test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=16,
                                num_workers=4)
    """
    # Use ImageFolder to create dataset(s)
    test_dataset = ImageFolderCustom(test_dir, transform=transform)

    # Get class names
    class_names = test_dataset.classes

    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last = False,
                                 prefetch_factor = prefetch_factor)

    return test_dataloader, class_names