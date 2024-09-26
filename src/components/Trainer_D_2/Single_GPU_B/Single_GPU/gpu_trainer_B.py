"""
This file contains single GPU Implementation for the Dataloader B Supervised Baseline experiment.

"""
import os
import sys
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm

# Local Imports
sys.path.append('/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline')

from src.components.dataset import ImageFolderCustom
from src.components import utils
from src.components.Trainer_D_2.Dataloader_2.weighted_dataset import WeightedImageDataset
from src.components.Trainer_D_2.Dataloader_2.dataloader_2 import DataLoaderB
import src.components.Trainer_D_2.Single_GPU_B.Single_GPU.configs.cfg_ddp_baseline as cfg_loader
import src.components.Trainer_D_2.Single_GPU_B.Single_GPU.jobs_server as server_file

# Step 2. Write the DDP compatible Trainer Class

class Trainer:
    def __init__(self,
                 model: torch.nn.Module, 
                 train_dataloader, 
                 val_dataloader, 
                 test_dataloader, 
                 optimizer_choice, 
                 scheduler_choice, 
                 lr, 
                 momentum, 
                 weight_decay, 
                 gpu_id, 
                 num_classes,
                 num_epochs, 
                 log_dir, 
                 exp_name, 
                 save_every):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.log_dir = log_dir
        self.experiment_name = exp_name
        self.save_every = save_every

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.logger = self.configure_logger()
        self.optimizer = self.configure_optimizer(optimizer_choice, lr, momentum, weight_decay)
        self.lr_scheduler = self.configure_scheduler(scheduler_choice, lr)

    def configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        log_file_path = os.path.join(self.log_dir, "training_log.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        return logger

    def calculate_balanced_accuracy(self, y_pred, y_true, num_classes):
        """
        Calculates the balanced accuracy score using PyTorch operations.
        (y_pred == c): Creates a boolean tensor where each element is True 
        if the predicted label equals class c, and False otherwise.

        (y_true == c): Creates another boolean tensor where each element is True 
        if the true label equals class c, and False otherwise.

        &: Performs a logical AND operation between the two boolean tensors. 
        The result is a tensor where each element is True only if both conditions 
        are met: the predicted label is class c, and the true label is also class c. 
        This effectively filters out the true positives for class c.

        .sum(): Sums up the True values in the resultant tensor, which corresponds
        to the count of true positive predictions for class c.

        Args:
            y_pred (torch.Tensor): Tensor of predicted class labels( No Logits & Probabilities, only labels).
            y_true (torch.Tensor): Tensor of true class labels.
            num_classes (int): Number of classes.

        Returns:
            float: The balanced accuracy score.
        """
        correct_per_class = torch.zeros(num_classes, device=y_pred.device)
        total_per_class = torch.zeros(num_classes, device=y_pred.device)

        for c in range(num_classes):
            # The number of true positive predictions for class c. 
            # True positives are instances that are correctly identified as 
            # belonging to class c by the classifier.
            true_positives = ((y_pred == c) & (y_true == c)).sum()
            # Condition Positive: total number of instances that actually belong to class c, 
            # regardless of whether they were correctly identified by the classifier or not.
            condition_positives = (y_true == c).sum()
            
            correct_per_class[c] = true_positives.float()
            total_per_class[c] = condition_positives.float()

        # .clamp(min=1) function ensures that no value in the total_per_class tensor is less than 1
        recall_per_class = correct_per_class / total_per_class.clamp(min=1)
        balanced_accuracy = recall_per_class.mean().item()  # Convert to Python scalar for compatibility

        return balanced_accuracy

    def configure_optimizer(self, optimizer_choice, initial_lr, momentum, weight_decay):
        if optimizer_choice.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        elif optimizer_choice.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError("Invalid optimizer choice. Choose 'adam' or 'sgd'.")
        return optimizer

    def configure_scheduler(self, scheduler_choice, initial_lr):
        if scheduler_choice.lower() == 'cosineannealinglr':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)  # T_max=80, adjusted to "num_epochs"
        elif scheduler_choice.lower() == 'lambdalr':
            lr_lambda = lambda epoch: 0.1 ** (epoch // 20) #reduces LR every 20 epochs
            # lr_lambda = lambda epoch: 1 # Keeps LR constant throughout training
            lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError("Invalid scheduler choice. Choose 'cosineannealinglr' or 'lambdalr'.")
        return lr_scheduler

    def _train_epoch(self):
        self.model.train()
        running_loss, num_samples = 0, 0
        y_pred_all, y_all = [], []
        for batch, (X, weight, y, image_path) in enumerate(self.train_dataloader):
            X, y = X.to(self.gpu_id), y.to(self.gpu_id)
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            running_loss += loss.item() * X.size(0)
            num_samples += X.size(0)
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            y_pred_all.append(y_pred_class)
            y_all.append(y)

        # Flatten the lists of tensors into single tensors
        y_pred_all_gathered = torch.cat(y_pred_all)
        y_all_gathered = torch.cat(y_all)
        balanced_accuracy = self.calculate_balanced_accuracy(y_pred_all_gathered, y_all_gathered, self.num_classes)

        # Calculate the average loss on the root GPU
        average_loss = running_loss / num_samples

        return balanced_accuracy, average_loss

    def _validation_epoch(self):
        self.model.eval()
        running_loss, num_samples = 0, 0
        y_pred_all, y_all = [], []

        with torch.no_grad():
            for batch, (X, y) in enumerate(self.val_dataloader):
                X, y = X.to(self.gpu_id), y.to(self.gpu_id)
                y_pred = self.model(X)
                loss = F.cross_entropy(y_pred, y)
                running_loss += loss.item() * X.size(0)
                num_samples += X.size(0)
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                # Concatenate predictions and labels locally on each GPU
                y_pred_all.append(y_pred_class)
                y_all.append(y)

        # Flatten the lists of tensors into single tensors
        y_pred_all_gathered = torch.cat(y_pred_all)
        y_all_gathered = torch.cat(y_all)
        balanced_accuracy = self.calculate_balanced_accuracy(y_pred_all_gathered, y_all_gathered, self.num_classes)

        # Calculate the average loss on the root GPU
        average_loss = running_loss / num_samples

        return balanced_accuracy, average_loss

    def _test_epoch(self):
        self.model.eval()
        running_loss, num_samples = 0, 0
        y_pred_all, y_all = [], []

        with torch.no_grad():
            for batch, (X, y) in enumerate(self.test_dataloader):
                X, y = X.to(self.gpu_id), y.to(self.gpu_id)
                y_pred = self.model(X)
                loss = F.cross_entropy(y_pred, y)
                running_loss += loss.item() * X.size(0)
                num_samples += X.size(0)
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                # Concatenate predictions and labels locally on each GPU
                y_pred_all.append(y_pred_class)
                y_all.append(y)

        # Flatten the lists of tensors into single tensors
        y_pred_all_gathered = torch.cat(y_pred_all)
        y_all_gathered = torch.cat(y_all)
        balanced_accuracy = self.calculate_balanced_accuracy(y_pred_all_gathered, y_all_gathered, self.num_classes)

        # Calculate the average loss on the root GPU
        average_loss = running_loss / num_samples

        return balanced_accuracy, average_loss

    def train_val_test(self, max_epochs, resume=False, checkpoint_path=None):
        total_start_time = time.time()
        start_epoch = 0
        if resume:
            c_epoch = self.load_checkpoint(checkpoint_path)
            self._log(f"Resuming training from epoch {c_epoch}")
            start_epoch = c_epoch + 1

        for epoch in tqdm(range(start_epoch, max_epochs)):
            server_file.setup_ccname()
            epoch_start_time = time.time()
            train_balanced_accuracy, train_loss = self._train_epoch()
            total_epoch_duration = time.time() - epoch_start_time
            
            val_epoch_start_time = time.time()
            val_balanced_accuracy, val_loss = self._validation_epoch()
            total_epoch_val_duration = time.time() - val_epoch_start_time

            test_epoch_start_time = time.time()
            test_balanced_accuracy, test_loss = self._test_epoch()
            total_epoch_test_duration = time.time() - test_epoch_start_time
            
            self.lr_scheduler.step()
            
            # Log metrics to TensorBoard and console
            self.writer.add_scalar("Train/Loss", train_loss, epoch)
            self.writer.add_scalar("Train/Balanced_Accuracy", train_balanced_accuracy, epoch)
            self.writer.add_scalar("Validation/Loss", val_loss, epoch)
            self.writer.add_scalar("Validation/Balanced_Accuracy", val_balanced_accuracy, epoch)
            self.writer.add_scalar("Test/Loss", test_loss, epoch)
            self.writer.add_scalar("Test/Balanced_Accuracy", test_balanced_accuracy, epoch)

            self._log(f"Epoch: {epoch} |Total training time: {self._format_time(total_epoch_duration)} | Train Loss: {train_loss:.4f} | Train Balanced Accuracy: {train_balanced_accuracy * 100:.4f} %")
            self._log(f"Epoch: {epoch} |Total validation time:{self._format_time(total_epoch_val_duration)} | Validation Loss: {val_loss:.4f} | Validation Balanced Accuracy: {val_balanced_accuracy * 100:.4f} % ")
            self._log(f"Epoch: {epoch} |Total testing time: {self._format_time(total_epoch_test_duration)} | Test Loss: {test_loss:.4f} | Test Balanced Accuracy: {test_balanced_accuracy * 100:.4f} %")

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, train_balanced_accuracy, train_loss, val_balanced_accuracy, val_loss,  test_balanced_accuracy, test_loss)

        total_duration = time.time() - total_start_time
        self._log(f"Total training, validation and testing time: {self._format_time(total_duration)}.")
        self.writer.close()

    def _save_checkpoint(self, epoch, train_balanced_accuracy, train_loss, val_balanced_accuracy, val_loss, test_balanced_accuracy, test_loss):
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{self.experiment_name}_epoch_{epoch}.pth")
        
        # Prepare the checkpoint dictionary
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'train_balanced_accuracy': train_balanced_accuracy,
            'val_balanced_accuracy': val_balanced_accuracy,
            'test_balanced_accuracy': test_balanced_accuracy,
            # Saving the current learning rate (from the first param group)
            'current_lr': self.optimizer.param_groups[0]['lr']
        }
        
        # If a learning rate scheduler is used, save its state as well
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            checkpoint_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint_dict, checkpoint_path)
        self._log(f"Saved checkpoint at epoch {epoch}: {checkpoint_path}")

    def _log(self, message):
        self.logger.info(message)

    def _format_time(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h:{int(minutes)}m:{int(seconds)}s"
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load the current learning rate back into the optimizer, if it was saved
        if 'current_lr' in checkpoint:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = checkpoint['current_lr']
        
        # If a learning rate scheduler state was saved, load it as well
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int, prefetch_factor: int, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,  # Set shuffle to False when using a sampler
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

def main(device, config):
    utils.set_seeds(1)
    # batch size per GPU
    batch_size_per_gpu = config.batch_size

    log_dir = os.path.join(os.path.dirname(__file__), "experiments", config.experiment_name, "runs")
    os.makedirs(log_dir, exist_ok=True)

    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

    # Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Change the classifier head
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=34)
    pretrained_vit_transforms = pretrained_vit_weights.transforms()

    # Make sure the head parameters are trainable: two print statements for two parameters : (Weight & Biases)
    for param in pretrained_vit.heads.parameters():
        print(f" The parameters of the head layer in model requires gradient or are trainable ? Answer: {param.requires_grad}")

    # Here is the Dataloader B strategy for train dataloader
    # The sole purpose of this strategy is to see its impact on model learning behaviour & to utilise unsupervised learning
    feature_file_path = config.features_path
    label_file_path = config.labels_path 
    img_path_file_path = config.img_path
    
    data_loader_instance = DataLoaderB(feature_file_path, label_file_path, img_path_file_path, num_categories=34)
    img_paths_list = data_loader_instance.img_paths_list
    gt_labels_loader, weights_list, all_labels, all_cluster_counts = data_loader_instance.initialize_dataloaders()
    
    # Creating an instance of the weigtedimagedataset class as a training dataset
    dataset_b = WeightedImageDataset(img_paths_list, weights_list, gt_labels_loader, transforms=pretrained_vit_transforms)
    # Utilising weighted random sampler for sampling of samples from train dataset based on calculated weights
    sampler_b = WeightedRandomSampler(dataset_b.weights, num_samples=len(dataset_b.weights), replacement=True)
    
    train_dataloader = prepare_dataloader(
        dataset_b,                 
        batch_size=batch_size_per_gpu, 
        sampler=sampler_b, 
        num_workers=config.num_workers, 
        prefetch_factor=config.prefetch_factor
    )
    # Use ImageFolderCustom to create Val and test dataset(s)
    val_dataset = ImageFolderCustom(config.val_dir, transform=pretrained_vit_transforms)
    test_dataset = ImageFolderCustom(config.test_dir, transform=pretrained_vit_transforms)
    
    # Pass the adjusted batch size here
    val_dataloader = prepare_dataloader(
        val_dataset, 
        batch_size_per_gpu, 
        num_workers= config.num_workers, 
        prefetch_factor=config.prefetch_factor
    )
    test_dataloader = prepare_dataloader(
        test_dataset, 
        batch_size_per_gpu, 
        num_workers= config.num_workers, 
        prefetch_factor=config.prefetch_factor
    )

    trainer = Trainer(
        model=pretrained_vit,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer_choice=config.optimizer,
        scheduler_choice=config.scheduler,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        gpu_id = device,
        num_classes=config.num_classes,
        num_epochs=config.num_epochs,
        log_dir=log_dir,
        exp_name=config.experiment_name,
        save_every=config.save_every
    )

    trainer.train_val_test(max_epochs=config.num_epochs, resume=config.resume, checkpoint_path=config.checkpoint_path)

if __name__ == "__main__":

    config = cfg_loader.get_config()
    print(f"Experiment Name: {config.experiment_name}")
    print(f"Experiment Running on GPU No: {config.device}")
    print(f"Features directory split_0 DAA: {config.features_path}")
    print(f"Labels directory split_0 DAA: {config.labels_path}")
    print(f"Image paths directory split_0 DAA: {config.img_path}")
    print(f"Val directory: {config.val_dir}")
    print(f"Test directory: {config.test_dir}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Number of Workers: {config.num_workers}")
    print(f"Number of Epochs: {config.num_epochs}")
    print(f"Number of Classes in DAA: {config.num_classes}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Scheduler: {config.scheduler}")
    print(f"Learning Rate (lr): {config.lr}")
    print(f"Momentum: {config.momentum}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(config.device)
    else:
        device = torch.device('cpu')
        print('No GPU avaialable, Using CPU')

    main(device=device, config=config)
