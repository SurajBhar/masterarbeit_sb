"""
This file contains DDP Implementation for the Feature Extractor Supervised Models used for Baseline.

"""
import os
import sys
import time
import math
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Local Imports
sys.path.append('/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline')

from src.components.distraction_detection_d_a.binary_dataset import BinaryClassificationDataset
import src.components.distraction_detection_d_a.cfg_binary_baseline as cfg_loader
import src.components.distraction_detection_d_a.jobs_server as server_file

# Step 1. Initialise the distributed Process Group
  
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "38755"
    os.environ['CUDA_VISIBLE_DEVICES'] = "3,4,5,6"
    dist.init_process_group(backend = "nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def launch(main_fn, cfg, world_size):
    mp.spawn(main_fn, args=(world_size, cfg), nprocs=world_size, join=True)

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
                 end_lr, 
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
        self.model = DDP(self.model, device_ids = [self.gpu_id])
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.log_dir = log_dir
        self.experiment_name = exp_name
        self.save_every = save_every
        # Create an instance of Tensorboar writer to add data to tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.optimizer = self.configure_optimizer(optimizer_choice, lr, momentum, weight_decay)
        self.lr_scheduler = self.configure_scheduler(scheduler_choice, lr, end_lr)

    def calculate_balanced_accuracy(self, y_pred, y_true, num_classes):
        correct_per_class = torch.zeros(num_classes, device=y_pred.device)
        total_per_class = torch.zeros(num_classes, device=y_pred.device)

        for c in range(num_classes):
            true_positives = ((y_pred == c) & (y_true == c)).sum()
            condition_positives = (y_true == c).sum()
            
            correct_per_class[c] = true_positives.float()
            total_per_class[c] = condition_positives.float()

        # .clamp(min=1) function ensures that no value in the total_per_class tensor is less than 1
        recall_per_class = correct_per_class / total_per_class.clamp(min=1)
        balanced_accuracy = recall_per_class.mean().item()  # Convert to Python scalar for compatibility

        return balanced_accuracy

    def configure_optimizer(self, optimizer_choice, initial_lr, momentum, weight_decay):
        if optimizer_choice.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError("Invalid optimizer choice. Choose 'sgd'.")
        return optimizer
    
    def configure_scheduler(self, scheduler_choice, initial_lr, end_lr):
        if scheduler_choice.lower() == 'linearinterpolationlr':
            last_epoch_index = self.num_epochs - 1
            lr_lambda = lambda epoch: (1 - float(epoch) / float(last_epoch_index)) + (float(epoch) / float(last_epoch_index))* end_lr * (1/initial_lr)
            lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError("Invalid scheduler choice. Choose 'linearinterpolationlr'")
        return lr_scheduler

    def _train_epoch(self):
        self.model.train()
        running_loss, num_samples = torch.tensor(0.0).to(self.gpu_id), torch.tensor(0).to(self.gpu_id)
        y_pred_all, y_all = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        y_pred_all = y_pred_all.to(self.gpu_id)
        y_all = y_all.to(self.gpu_id)

        for batch, (X, y,_,_) in enumerate(self.train_dataloader):
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
            # Concatenate predictions and labels locally on each GPU
            y_pred_all = torch.cat((y_pred_all, y_pred_class))
            y_all = torch.cat((y_all, y))

        # Convert to correct data type for all_reduce
        running_loss = running_loss.float()
        num_samples = num_samples.float()

        # Sum running_loss and num_samples across all GPUs
        torch.distributed.all_reduce(running_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(num_samples, op=torch.distributed.ReduceOp.SUM)

        # Gather all predictions and labels on the root GPU
        y_pred_all_gathered = [torch.zeros_like(y_pred_all) for _ in range(torch.distributed.get_world_size())]
        y_all_gathered = [torch.zeros_like(y_all) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(y_pred_all_gathered, y_pred_all)  # Gathering predictions
        torch.distributed.all_gather(y_all_gathered, y_all)  # Gathering labels

        if torch.distributed.get_rank() == 0:
            # Flatten the lists of tensors into single tensors
            y_pred_all_gathered = torch.cat(y_pred_all_gathered)
            y_all_gathered = torch.cat(y_all_gathered)
            balanced_accuracy = self.calculate_balanced_accuracy(y_pred_all_gathered, y_all_gathered, self.num_classes)

            # Calculate the average loss on the root GPU
            average_loss = running_loss.item() / num_samples.item()
        else:
            balanced_accuracy = None  # Non-root GPUs do not compute metrics
            average_loss = None  # Only computed on the root GPU

        return balanced_accuracy, average_loss

    def _validation_epoch(self):
        self.model.eval()
        running_loss, num_samples = torch.tensor(0.0).to(self.gpu_id), torch.tensor(0).to(self.gpu_id)
        y_pred_all, y_all = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        y_pred_all = y_pred_all.to(self.gpu_id)
        y_all = y_all.to(self.gpu_id)

        with torch.no_grad():
            for batch, (X, y,_,_) in enumerate(self.val_dataloader):
                X, y = X.to(self.gpu_id), y.to(self.gpu_id)
                y_pred = self.model(X)
                loss = F.cross_entropy(y_pred, y)
                running_loss += loss.item() * X.size(0)
                num_samples += X.size(0)
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                # Concatenate predictions and labels locally on each GPU
                y_pred_all = torch.cat((y_pred_all, y_pred_class))
                y_all = torch.cat((y_all, y))

        # Convert to correct data type for all_reduce
        running_loss = running_loss.float()
        num_samples = num_samples.float()

        # Sum running_loss and num_samples across all GPUs
        torch.distributed.all_reduce(running_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(num_samples, op=torch.distributed.ReduceOp.SUM)

        # Gather all predictions and labels on the root GPU
        y_pred_all_gathered = [torch.zeros_like(y_pred_all) for _ in range(torch.distributed.get_world_size())]
        y_all_gathered = [torch.zeros_like(y_all) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(y_pred_all_gathered, y_pred_all)  # Gathering predictions
        torch.distributed.all_gather(y_all_gathered, y_all)  # Gathering labels

        if torch.distributed.get_rank() == 0:
            # Flatten the lists of tensors into single tensors
            y_pred_all_gathered = torch.cat(y_pred_all_gathered)
            y_all_gathered = torch.cat(y_all_gathered)
            balanced_accuracy = self.calculate_balanced_accuracy(y_pred_all_gathered, y_all_gathered, self.num_classes)

            # Calculate the average loss on the root GPU
            average_loss = running_loss.item() / num_samples.item()
        else:
            balanced_accuracy = None  # Non-root GPUs do not compute metrics
            average_loss = None  # Only computed on the root GPU

        return balanced_accuracy, average_loss

    def _test_epoch(self):
        self.model.eval()
        running_loss, num_samples = torch.tensor(0.0).to(self.gpu_id), torch.tensor(0).to(self.gpu_id)
        y_pred_all, y_all = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        y_pred_all = y_pred_all.to(self.gpu_id)
        y_all = y_all.to(self.gpu_id)

        with torch.no_grad():
            for batch, (X, y,_,_) in enumerate(self.test_dataloader):
                X, y = X.to(self.gpu_id), y.to(self.gpu_id)
                y_pred = self.model(X)
                loss = F.cross_entropy(y_pred, y)
                running_loss += loss.item() * X.size(0)
                num_samples += X.size(0)
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                # Concatenate predictions and labels locally on each GPU
                y_pred_all = torch.cat((y_pred_all, y_pred_class))
                y_all = torch.cat((y_all, y))

        # Convert to correct data type for all_reduce
        running_loss = running_loss.float()
        num_samples = num_samples.float()

        # Sum running_loss and num_samples across all GPUs
        torch.distributed.all_reduce(running_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(num_samples, op=torch.distributed.ReduceOp.SUM)

        # Gather all predictions and labels on the root GPU
        y_pred_all_gathered = [torch.zeros_like(y_pred_all) for _ in range(torch.distributed.get_world_size())]
        y_all_gathered = [torch.zeros_like(y_all) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(y_pred_all_gathered, y_pred_all)  # Gathering predictions
        torch.distributed.all_gather(y_all_gathered, y_all)  # Gathering labels

        if torch.distributed.get_rank() == 0:
            # Flatten the lists of tensors into single tensors
            y_pred_all_gathered = torch.cat(y_pred_all_gathered)
            y_all_gathered = torch.cat(y_all_gathered)
            balanced_accuracy = self.calculate_balanced_accuracy(y_pred_all_gathered, y_all_gathered, self.num_classes)

            # Calculate the average loss on the root GPU
            average_loss = running_loss.item() / num_samples.item()
        else:
            balanced_accuracy = None  # Non-root GPUs do not compute metrics
            average_loss = None  # Only computed on the root GPU

        return balanced_accuracy, average_loss

    def train_val_test(self, max_epochs, resume=False, checkpoint_path=None):
        start_epoch = 0
        if resume:
            last_completed_epoch = self.load_checkpoint(checkpoint_path)
            start_epoch = last_completed_epoch + 1
            print(f"Resuming training from epoch {start_epoch}")
            # Checkpoint saves lr for last completed epoch so update it
            self.lr_scheduler.step()

        for epoch in tqdm(range(start_epoch, max_epochs)):
            server_file.setup_ccname()
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            train_balanced_accuracy, train_loss = self._train_epoch()
            
            if isinstance(self.val_dataloader.sampler, DistributedSampler):
                self.val_dataloader.sampler.set_epoch(epoch)

            val_balanced_accuracy, val_loss = self._validation_epoch()

            if isinstance(self.test_dataloader.sampler, DistributedSampler):
                self.test_dataloader.sampler.set_epoch(epoch)

            test_balanced_accuracy, test_loss = self._test_epoch()

            # Log the learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if self.gpu_id == 0:
                # Log metrics to TensorBoard and console
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)
                self.writer.add_scalar("Train/Loss", train_loss, epoch)
                self.writer.add_scalar("Train/Balanced_Accuracy", train_balanced_accuracy, epoch)
                self.writer.add_scalar("Validation/Loss", val_loss, epoch)
                self.writer.add_scalar("Validation/Balanced_Accuracy", val_balanced_accuracy, epoch)
                self.writer.add_scalar("Test/Loss", test_loss, epoch)
                self.writer.add_scalar("Test/Balanced_Accuracy", test_balanced_accuracy, epoch)

            if self.gpu_id == 0 and (epoch+1) % self.save_every == 0:
                self._save_checkpoint(epoch)
            # Updating the learning rate schedule
            self.lr_scheduler.step()
        self.writer.close()

    def _save_checkpoint(self, epoch):
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{self.experiment_name}_epoch_{epoch}.pth")
        
        # Prepare the checkpoint dictionary
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Saving the current learning rate (from the first param group)
            'current_lr': self.optimizer.param_groups[0]['lr']
        }
        
        # If a learning rate scheduler is used, save its state as well
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            checkpoint_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint_dict, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch}: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Loads a checkpoint, restoring the model state, optimizer state, and the current learning rate.
        Optionally restores the learning rate scheduler state.
        """
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(self.gpu_id))
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load the current learning rate back into the optimizer, if it was saved
        if 'current_lr' in checkpoint:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = checkpoint['current_lr']
        
        # If a learning rate scheduler state was saved, load it as well
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int, prefetch_factor: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # Set shuffle to False when using a sampler
        sampler=DistributedSampler(dataset),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

def main(rank: int, world_size: int, config):
    seed_val = 42
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    ddp_setup(rank, world_size)

    # Adjust batch size per GPU
    batch_size_per_gpu = config.batch_size // world_size

    log_dir = os.path.join(os.path.dirname(__file__), "vit_baseline_4gpus_experiments", config.experiment_name, "runs")
    os.makedirs(log_dir, exist_ok=True)

    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

    # Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Change the classifier head to match with binary classification:
    # {distracted_driver, non_distracted_driver}
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=2)
    pretrained_vit_transforms = pretrained_vit_weights.transforms()

    # Use ImageFolder to create dataset(s)
    train_dataset = BinaryClassificationDataset(config.train_dir, transform=pretrained_vit_transforms)
    val_dataset = BinaryClassificationDataset(config.val_dir, transform=pretrained_vit_transforms)
    test_dataset = BinaryClassificationDataset(config.test_dir, transform=pretrained_vit_transforms)

    # Pass the adjusted batch size here
    train_dataloader = prepare_dataloader(train_dataset, batch_size_per_gpu, num_workers= config.num_workers, prefetch_factor=config.prefetch_factor)
    val_dataloader = prepare_dataloader(val_dataset, batch_size_per_gpu, num_workers= config.num_workers, prefetch_factor=config.prefetch_factor)
    test_dataloader = prepare_dataloader(test_dataset, batch_size_per_gpu, num_workers= config.num_workers, prefetch_factor=config.prefetch_factor)

    trainer = Trainer(
        model=pretrained_vit,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer_choice=config.optimizer,
        scheduler_choice=config.scheduler,
        lr=config.lr,
        end_lr=config.end_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        gpu_id = rank, # Rank => DDP
        num_classes=config.num_classes,
        num_epochs=config.num_epochs,
        log_dir=log_dir,
        exp_name=config.experiment_name,
        save_every=config.save_every
    )

    trainer.train_val_test(max_epochs=config.num_epochs, resume=config.resume, checkpoint_path=config.checkpoint_path)
    cleanup()

if __name__ == "__main__":

    config = cfg_loader.get_config()
    print(f"Experiment Name: {config.experiment_name}")
    print(f"Train directory: {config.train_dir}")
    print(f"Val directory: {config.val_dir}")
    print(f"Test directory: {config.test_dir}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Number of Workers: {config.num_workers}")
    print(f"Number of Epochs: {config.num_epochs}")
    print(f"Number of Classes in DAA: {config.num_classes}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Scheduler: {config.scheduler}")
    print(f"Initial Learning Rate (lr): {config.lr}")
    print(f"End Learning Rate (lr): {config.end_lr}")
    print(f"Momentum: {config.momentum}")
    world_size = 4
    launch(main, config, world_size)
