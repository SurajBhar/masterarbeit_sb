# Standard Library Imports
import os
import sys
import time
import logging
import getpass
from glob import glob
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional

# Third-Party Library Imports
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Local Imports
sys.path.append('/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline')
from src.components import data_setup
from src.components.dataset import ImageFolderCustom
from src.components import utils
from src.components.config_manager_baseline import get_config

def setup_ccname():
    user=getpass.getuser()
    # check if k5start is running, exit otherwise
    try:
        pid=open("/tmp/k5pid_"+user).read().strip()
        os.kill(int(pid), 0)
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nk5start not running!\n")
        sys.exit(1)
    try:
        ccname=open("/tmp/kccache_"+user).read().split("=")[1].strip()
        os.environ['KRB5CCNAME']=ccname
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nmaybe k5start not running?\n")
        sys.exit(1)

def format_time(seconds):
    """Converts time in seconds to hours, minutes, and seconds format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

def configure_logger(log_dir, experiment_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a file handler for logging
    log_file_path = os.path.join(log_dir, "training_log.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Create a console handler for real-time progress
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

def calculate_balanced_accuracy(y_pred, y_true, num_classes, epsilon=1e-9):
    """
    Calculates the balanced accuracy score.
    
    Args:
        y_pred (torch.Tensor): Predicted labels.
        y_true (torch.Tensor): True labels.
        num_classes (int): Number of classes in the dataset.
        epsilon (float): A small value to add to denominators to prevent division by zero.
        
    Returns:
        float: Balanced accuracy score.
    """
    # Create confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_pred.device)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    # Calculate recall for each class, adding epsilon to avoid division by zero
    # Recall =  dividing the true positives by the sum of the true positive and false negative for each class
    # Recall = (diagonal elements of the confusion matrix) /  (the sum of elements in each row of the confusion matrix + epsilon)
    recall = torch.diag(confusion_matrix) / (confusion_matrix.sum(1) + epsilon)

    # balanced_accuracy_per_class = recall  # This line is technically not needed but added for clarity

    # Calculate balanced accuracy
    balanced_accuracy = recall.mean().item()

    return balanced_accuracy

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        gpu_id: int,
        num_classes: int,
        log_dir: str,
        exp_name: str,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.num_classes = num_classes
        self.model = model.to(gpu_id)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_dir = log_dir
        self.experiment_name = exp_name
        self.save_every = save_every
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.logger = configure_logger(self.log_dir, self.experiment_name)

    def train_step(self):
        self.model.train()
        running_train_loss, train_acc, num_samples = 0, 0, 0
        print(f"In the beginning the value of running train loss is: {running_train_loss} , train accuracy is : {train_acc}, number of samples is: {num_samples}")
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.gpu_id), y.to(self.gpu_id)
            print(f"The shape of the image batch: {batch} , is : {X.size()}")
            print(f"The shape of the ground truth batch : {batch} , is : {y.size()}")
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            print(f"The shape of the predictions for batch: {batch} , is: {y_pred.size()}")
            loss = F.cross_entropy(y_pred, y)
            print(f"The calculated Loss value for batch: {batch} , is : {y_pred.size()}")
            loss.backward()
            self.optimizer.step()
            # F.cross_entropy returns the mean loss per batch, 
            # and we need the total loss to calculate the average loss over all samples after the loop.
            running_train_loss += loss.item() * X.size(0)
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).type(torch.float).sum().item()
            num_samples += X.size(0)
            print(f"After batch {batch} the value of running train loss is: {running_train_loss} , train accuracy is : {train_acc}, number of samples is: {num_samples}")
        avg_loss = running_train_loss / num_samples
        # Average accuracy = Summation of Accuracy over all batches / Number of samples
        avg_acc = train_acc / num_samples
        return avg_loss, avg_acc

    def validation_step(self):
        self.model.eval()
        val_loss, val_acc, num_samples = 0, 0, 0
        with torch.no_grad():
            for X, y in self.val_dataloader:
                X, y = X.to(self.gpu_id), y.to(self.gpu_id)
                y_pred = self.model(X)
                loss = F.cross_entropy(y_pred, y)
                
                val_loss += loss.item() * X.size(0)
                val_acc += (y_pred.argmax(1) == y).type(torch.float).sum().item()
                num_samples += X.size(0)
        avg_loss = val_loss / num_samples
        avg_acc = val_acc / num_samples
        return avg_loss, avg_acc

    def training_validation(self, max_epochs: int, resume: bool = False, checkpoint_path=None):
        total_start_time = time.time()
        start_epoch = 0
        if resume:
            start_epoch, _, _, _, _ = self.load_checkpoint(checkpoint_path)
            self._log(f"Resuming training from epoch {start_epoch}")

        for epoch in tqdm(range(start_epoch, max_epochs)):
            setup_ccname()
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.validation_step()

            # Log metrics to TensorBoard and console
            self.writer.add_scalar("Train/Loss", train_loss, epoch)
            self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
            self.writer.add_scalar("Validation/Loss", val_loss, epoch)
            self.writer.add_scalar("Validation/Accuracy", val_acc, epoch)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("Learning Rate", current_lr, epoch)

            self._log(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")

            # Adjust learning rate
            self.lr_scheduler.step()

            # Save checkpoints
            if (epoch + 1) % self.save_every == 0:
                checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.save_checkpoint(epoch, checkpoint_dir, train_loss, val_loss, train_acc, val_acc)

        total_end_time = time.time()
        total_duration = format_time(total_end_time - total_start_time)
        self._log(f"Total training and validation time: {total_duration}.")
        self.writer.close()
    
    def save_checkpoint(self, epoch, checkpoint_dir, train_loss, val_loss, train_bal_acc, val_bal_acc):
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_{self.experiment_name}_epoch_{epoch}.pth"
        )
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_bal_acc': train_bal_acc,
                'val_bal_acc': val_bal_acc,
            },
            checkpoint_path,
        )
        self._log(f"Saved Trainining checkpoint at epoch {epoch}: {checkpoint_path}")

    def _log(self, message):
        self.logger.info(message)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss'], checkpoint['train_bal_acc'], checkpoint['val_bal_acc']

    def training_validation(self, max_epochs: int, resume: bool = False, checkpoint_path=None):
        total_start_time = time.time()  # Start time for the entire training and validation process
        try:
            start_epoch = 0

            if resume:
                start_epoch, _, _, _, _ = self.load_checkpoint(checkpoint_path)
                self._log(f"Resuming training from epoch {start_epoch}")

            for epoch in tqdm(range(start_epoch, max_epochs)):
                setup_ccname()
                try:
                    ############### Training & Validation step Here ##########################
                    train_loss, train_acc, train_bal_acc= self.train_step(epoch)
                    val_loss, val_acc, val_bal_acc = self.validation_step(epoch)

                    self.writer.add_scalar("Train/Loss", train_loss, epoch)
                    self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
                    self.writer.add_scalar("Train/Balanced Accuracy", train_bal_acc, epoch)
                    self.writer.add_scalar("Validation/Loss", val_loss, epoch)
                    self.writer.add_scalar("Validation/Accuracy", val_acc, epoch)
                    self.writer.add_scalar("Validation/Balanced Accuracy", val_bal_acc, epoch)

                    exp_path = os.path.join(os.path.dirname(__file__), f"experiments/{self.experiment_name}")
                    checkpoint_dir = os.path.join(exp_path, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    if (epoch+1) % self.save_every == 0:
                        # Save the checkpoint
                        self.save_checkpoint(epoch, checkpoint_dir, train_loss, val_loss, train_bal_acc, val_bal_acc)

                except RuntimeError as e:
                    self._log(f"Runtime error occurred in epoch {epoch}: {e}")
                    continue

        except Exception as e:
            self._log(f"An unexpected error occurred: {e}")

        finally:
            total_end_time = time.time()  # End time for the entire training and validation process
            total_duration = total_end_time - total_start_time
            formatted_duration = format_time(total_duration)
            self._log(f"Total training and validation time: {formatted_duration}.")
            self.writer.close()

def configure_logger(log_dir, experiment_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a file handler for logging
    log_file_path = os.path.join(log_dir, "training_log.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Create a console handler for real-time progress
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

def calculate_balanced_accuracy(y_pred, y_true, num_classes, epsilon=1e-9):
    """
    Calculates the balanced accuracy score.
    
    Args:
        y_pred (torch.Tensor): Predicted labels.
        y_true (torch.Tensor): True labels.
        num_classes (int): Number of classes in the dataset.
        epsilon (float): A small value to add to denominators to prevent division by zero.
        
    Returns:
        float: Balanced accuracy score.
    """
    # Create confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_pred.device)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    # Calculate recall for each class, adding epsilon to avoid division by zero
    # Recall =  dividing the true positives by the sum of the true positive and false negative for each class
    # Recall = (diagonal elements of the confusion matrix) /  (the sum of elements in each row of the confusion matrix + epsilon)
    recall = torch.diag(confusion_matrix) / (confusion_matrix.sum(1) + epsilon)

    # balanced_accuracy_per_class = recall  # This line is technically not needed but added for clarity

    # Calculate balanced accuracy
    balanced_accuracy = recall.mean().item()

    return balanced_accuracy

def load_train_objs(model, num_epochs, optimizer_choice, scheduler_choice, initial_lr, momentum, weight_decay_adam, wd_sgd):
    # Setup the optimizer
    if optimizer_choice == 'ADAM':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=initial_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay_adam
        )
    elif optimizer_choice == 'SGD':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=initial_lr,
            momentum=momentum,
            weight_decay=wd_sgd
        )
    else:
        raise ValueError("Invalid optimizer choice. Choose 'adam' or 'sgd'.")

    # Define the lambda function for learning rate scheduling
    def lr_lambda(epoch):
        # Decrease the learning rate by a factor of 10 every 30 epochs
        return 0.1 ** (epoch // 30)

    # Setup the learning rate scheduler
    if scheduler_choice == 'CosineAnnealingLR':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )
    elif scheduler_choice == 'LambdaLR':
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda  # Used the custom lambda function
        )
    else:
        raise ValueError("Invalid scheduler choice. Choose 'LambdaLR' or 'CosineAnnealingLR'")
    
    return optimizer, lr_scheduler


def log_dir_fxn(experiment_name):
    exp_path = os.path.dirname(__file__)
    log_dir_path = os.path.join(exp_path, "experiments", experiment_name, "runs")
    try:
        os.makedirs(log_dir_path, exist_ok=True)
        print("Log directory for Tensorboard Events/Logs created")
    except FileExistsError:
        print("Log directory already exists")
    return log_dir_path

def setup_and_create_dataloaders(batch_size, train_dir, val_dir, num_workers, prefetch_factor):
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

    # Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Change the classifier head
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=34)
    pretrained_vit_transforms = pretrained_vit_weights.transforms()

    train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        transform=pretrained_vit_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

    return train_dataloader, val_dataloader, class_names , pretrained_vit

# EXP: 1: Batch_size = 1024, SGD+CosineAnnealing, Split_0 , train + validation dataset

def main():
    config = get_config()
    print(f"Experiment Name: {config.experiment_name}")
    print(f"Train directory: {config.train_dir}")
    print(f"Val directory: {config.val_dir}")
    print(f"GPU/ Device Number: {config.device}")
    print(f"Number of workers: {config.num_workers}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Pre-Fetch Factor: {config.prefetch_factor}")
    print(f"Number of Epochs: {config.num_epochs}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Scheduler: {config.scheduler}")
    print(f"Learning Rate (lr): {config.lr}")
    print(f"Momentum: {config.momentum}")
    # print(f"Weight Decay for Adam (w_decay_adam): {config.w_decay_adam}")
    print(f"Weight Decay: {config.weight_decay}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(config.device)
    else:
        device = torch.device('cpu')
        print('No GPU avaialable, Using CPU')

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.set_seeds(1)
    log_dir = log_dir_fxn(config.experiment_name)
    num_workers = os.cpu_count()
    train_dataloader, val_dataloader, _, model_vit = setup_and_create_dataloaders(config.batch_size, 
                                                                                  config.train_dir, 
                                                                                  config.val_dir, 
                                                                                  num_workers, 
                                                                                  config.prefetch_factor,
                                                                                )
    optimizer, lr_scheduler = load_train_objs(model_vit, 
                                              config.num_epochs, 
                                              config.optimizer, 
                                              config.scheduler, 
                                              config.lr, 
                                              config.momentum, 
                                              config.w_decay_adam, 
                                              config.weight_decay
                                            )
    
    trainer = Trainer(model=model_vit, 
                      train_dataloader=train_dataloader, 
                      val_dataloader=val_dataloader,
                      optimizer=optimizer, 
                      lr_scheduler=lr_scheduler, 
                      gpu_id=device,
                      num_classes=config.num_classes,
                      log_dir=log_dir, 
                      exp_name=config.experiment_name,
                      save_every=config.save_every)

    trainer.training_validation(max_epochs=config.num_epochs, resume=config.resume, checkpoint_path=config.checkpoint_path)


if __name__ == "__main__":
    main()


# Example Use: 
# python /src/components/base_d1_train_val.py --config /src/components/config_manager/experiment_1_split_0.txt