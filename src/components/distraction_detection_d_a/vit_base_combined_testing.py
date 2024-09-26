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

import logging
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Local Imports
sys.path.append('/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline')

from torchvision.datasets import ImageFolder
import src.components.distraction_detection_d_a.Baseline_VIT_A_RGB_No_Aug.test_configs.combined_test_config_vit_binary_baseline as cfg_loader
import src.components.distraction_detection_d_a.jobs_server as server_file

# Step 1. Initialise the distributed Process Group  
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "40581"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
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
                 kir_test_dataloader, 
                 nir_test_dataloader,  
                 gpu_id, 
                 num_classes,
                 log_dir, 
                 exp_name):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids = [self.gpu_id])

        self.num_classes = num_classes
        self.kir_test_dataloader = kir_test_dataloader
        self.nir_test_dataloader = nir_test_dataloader
        self.log_dir = log_dir
        self.experiment_name = exp_name
        self.logger = self.configure_logger()

    def configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        log_file_path = os.path.join(self.log_dir, "vit_baseline_combined_testing_d_a_no_aug_log.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        return logger

    def calculate_balanced_accuracy(self, y_pred, y_true, num_classes):
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

    def _test_kir_epoch(self):
        self.model.eval()
        running_loss, num_samples = torch.tensor(0.0).to(self.gpu_id), torch.tensor(0).to(self.gpu_id)
        y_pred_all, y_all = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        y_pred_all = y_pred_all.to(self.gpu_id)
        y_all = y_all.to(self.gpu_id)

        with torch.no_grad():
            for batch, (X, y) in enumerate(self.kir_test_dataloader):
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
            # self._log(f"Epoch {epoch}, Average Loss: {average_loss}, Balanced Accuracy: {balanced_accuracy}")
        else:
            balanced_accuracy = None  # Non-root GPUs do not compute metrics
            average_loss = None  # Only computed on the root GPU

        return balanced_accuracy, average_loss
    
    def _test_nir_epoch(self):
        self.model.eval()
        running_loss, num_samples = torch.tensor(0.0).to(self.gpu_id), torch.tensor(0).to(self.gpu_id)
        y_pred_all, y_all = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        y_pred_all = y_pred_all.to(self.gpu_id)
        y_all = y_all.to(self.gpu_id)

        with torch.no_grad():
            for batch, (X, y) in enumerate(self.nir_test_dataloader):
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
            # self._log(f"Epoch {epoch}, Average Loss: {average_loss}, Balanced Accuracy: {balanced_accuracy}")
        else:
            balanced_accuracy = None  # Non-root GPUs do not compute metrics
            average_loss = None  # Only computed on the root GPU

        return balanced_accuracy, average_loss

    def combined_test(self, checkpoint_path=None):
        total_start_time = time.time()
        start_epoch = 0
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(self.gpu_id))
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        best_model_epoch = checkpoint['epoch']
        self._log(f"Testing best ViT B 16 model tarined on Kinect RGB Right Top DAA dataset for {best_model_epoch + 1} epochs.")

        for epoch in tqdm(range(start_epoch, 1)):
            server_file.setup_ccname() 
            if isinstance(self.kir_test_dataloader.sampler, DistributedSampler):
                self.kir_test_dataloader.sampler.set_epoch(epoch)
            
            kir_epoch_start_time = time.time()
            kir_test_balanced_accuracy, kir_test_loss = self._test_kir_epoch()
            kir_epoch_duration = time.time() - kir_epoch_start_time
    
            if isinstance(self.nir_test_dataloader.sampler, DistributedSampler):
                self.nir_test_dataloader.sampler.set_epoch(epoch)
            
            nir_epoch_start_time = time.time()
            nir_test_balanced_accuracy, nir_test_loss = self._test_nir_epoch()
            nir_epoch_duration = time.time() - nir_epoch_start_time

            if self.gpu_id == 0:
                self._log(f"This file contains logs for Generalisation for: {self.experiment_name}")
                self._log(f"Epoch: {epoch} |Total KIR testing time: {self._format_time(kir_epoch_duration)} | KIR Test Loss: {kir_test_loss:.4f} | KIR Test Balanced Accuracy: {kir_test_balanced_accuracy * 100:.4f} %")
                self._log(f"Epoch: {epoch} |Total NIR testing time: {self._format_time(nir_epoch_duration)} | NIR Test Loss: {nir_test_loss:.4f} | NIR Test Balanced Accuracy: {nir_test_balanced_accuracy * 100:.4f} %")
            
        total_duration = time.time() - total_start_time
        self._log(f"Total combined testing time: {self._format_time(total_duration)}.")

    def _log(self, message):
        if self.gpu_id == 0:  # Logging only from the master process
        # print(message) # Uncomment if you want to print it
            self.logger.info(message)

    def _format_time(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h:{int(minutes)}m:{int(seconds)}s"

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

    log_dir = os.path.join(os.path.dirname(__file__), "vit_baseline_01_combined_testing_d_a_with_aug_experiments", config.experiment_name, "runs")
    os.makedirs(log_dir, exist_ok=True)

    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

    # Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Change the classifier head to match with binary classification:
    # {0: _non_distracted_driver, 1: distracted_driver}
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=2)
    pretrained_vit_transforms = pretrained_vit_weights.transforms()

    # Use ImageFolder to create dataset(s)
    kir_test_dataset = ImageFolder(config.kir_test_dir, transform=pretrained_vit_transforms)
    nir_test_dataset = ImageFolder(config.nir_test_dir, transform=pretrained_vit_transforms)

    kir_test_dataloader = prepare_dataloader(kir_test_dataset, batch_size_per_gpu, num_workers= config.num_workers, prefetch_factor=config.prefetch_factor)
    nir_test_dataloader = prepare_dataloader(nir_test_dataset, batch_size_per_gpu, num_workers= config.num_workers, prefetch_factor=config.prefetch_factor)

    trainer = Trainer(
        model=pretrained_vit,
        kir_test_dataloader=kir_test_dataloader,
        nir_test_dataloader=nir_test_dataloader,
        gpu_id = rank, # Rank => DDP
        num_classes=config.num_classes,
        log_dir=log_dir,
        exp_name=config.experiment_name
    )

    trainer.combined_test(checkpoint_path=config.checkpoint_path)
    cleanup()

if __name__ == "__main__":

    config = cfg_loader.get_config()
    print(f"Experiment Name: {config.experiment_name}")
    print(f"Kir Test directory: {config.kir_test_dir}")
    print(f"NIR Test directory: {config.nir_test_dir}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Number of Workers: {config.num_workers}")
    print(f"Number of Classes in DAA: {config.num_classes}")

    world_size = 2
    launch(main, config, world_size)
