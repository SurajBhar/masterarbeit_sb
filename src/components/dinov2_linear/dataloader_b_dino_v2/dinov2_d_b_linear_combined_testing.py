"""
This file contains ClusteredFeatureWeightingDataloader implementation and Training pipeline.
"""
# Libraries and requiredd packages
import os
import sys
import time
import math
import random
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder
# Local Imports
sys.path.append('/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline')
import src.components.dinov2_linear.configs.dinov2_d_b_no_aug.test_configs.test_configs_no_aug.combined_test_config_dinov2_vit_binary as cfg_loader
import src.components.dinov2_linear.utils.jobs_server as server_file
from src.components.dinov2_linear.data.transforms import make_classification_eval_transform

######################### Trainer Class #######################################
class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 kir_test_dataloader, 
                 nir_test_dataloader,   
                 gpu_id, 
                 num_classes, 
                 log_dir, 
                 exp_name, 
                ):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
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
        log_file_path = os.path.join(self.log_dir, "dinov2_d_b_testing_log.log")
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
        running_loss, num_samples = 0.0, 0
        y_pred_all = []
        y_all = []

        with torch.no_grad():
            # Here traditional dataloader is used
            for batch, (X, y) in enumerate(self.kir_test_dataloader):
                X, y = X.to(self.gpu_id), y.to(self.gpu_id)
                y_pred = self.model(X)
                loss = F.cross_entropy(y_pred, y)
                running_loss += loss.item() * X.size(0)
                num_samples += X.size(0)
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                y_pred_all.append(y_pred_class)
                y_all.append(y)

        average_loss = running_loss / num_samples
        balanced_accuracy = self.calculate_balanced_accuracy(torch.cat(y_pred_all), torch.cat(y_all), self.num_classes)
        return balanced_accuracy, average_loss

    def _test_nir_epoch(self):
        self.model.eval()
        running_loss, num_samples = 0.0, 0
        y_pred_all = []
        y_all = []

        with torch.no_grad():
            # Here traditional dataloader is used
            for batch, (X, y) in enumerate(self.nir_test_dataloader):
                X, y = X.to(self.gpu_id), y.to(self.gpu_id)
                y_pred = self.model(X)
                loss = F.cross_entropy(y_pred, y)
                running_loss += loss.item() * X.size(0)
                num_samples += X.size(0)
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                y_pred_all.append(y_pred_class)
                y_all.append(y)

        average_loss = running_loss / num_samples
        balanced_accuracy = self.calculate_balanced_accuracy(torch.cat(y_pred_all), torch.cat(y_all), self.num_classes)
        return balanced_accuracy, average_loss

    def combined_test(self, checkpoint_path=None):
        total_start_time = time.time()
        start_epoch = 0
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        best_model_epoch = checkpoint['epoch']
        self._log(f"Testing best DINOV2_ViT_B_14 model tarined on Kinect RGB Right Top DAA dataset for {best_model_epoch + 1} epochs.")

        for epoch in tqdm(range(start_epoch, 1)):
            server_file.setup_ccname()

            kir_epoch_start_time = time.time()
            kir_test_balanced_accuracy, kir_test_loss = self._test_kir_epoch()
            kir_epoch_duration = time.time() - kir_epoch_start_time

            nir_epoch_start_time = time.time()
            nir_test_balanced_accuracy, nir_test_loss = self._test_nir_epoch()
            nir_epoch_duration = time.time() - nir_epoch_start_time

            self._log(f"This file contains logs for Generalisation for: {self.experiment_name}")
            self._log(f"Epoch: {epoch} |Total KIR testing time: {self._format_time(kir_epoch_duration)} | KIR Test Loss: {kir_test_loss:.4f} | KIR Test Balanced Accuracy: {kir_test_balanced_accuracy * 100:.4f} %")
            self._log(f"Epoch: {epoch} |Total NIR testing time: {self._format_time(nir_epoch_duration)} | NIR Test Loss: {nir_test_loss:.4f} | NIR Test Balanced Accuracy: {nir_test_balanced_accuracy * 100:.4f} %")
            
        total_duration = time.time() - total_start_time
        self._log(f"Total combined testing time: {self._format_time(total_duration)}.")

    def _log(self, message):
        self.logger.info(message)

    def _format_time(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h:{int(minutes)}m:{int(seconds)}s"
    
######## Linear Classifier #######################
###################################################

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, backbone, num_features=768, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.linear = nn.Linear(num_features, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        features = self.backbone(x)
        output = self.linear(features)
        return output
#####################################################
############### Dataloader Function #################

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int, prefetch_factor: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # Set shuffle to False for evaluation
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

########## Main Function #######################################
#################################################################
def main():
    config = cfg_loader.get_config()
    print(f"Experiment Name: {config.experiment_name}")
    print(f"Kir Test directory: {config.kir_test_dir}")
    print(f"NIR Test directory: {config.nir_test_dir}")
    print(f"GPU device: {config.device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Number of Workers: {config.num_workers}")
    print(f"Number of Classes in DAA: {config.num_classes}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(config.device)
    else:
        device = torch.device('cpu')
        print('No GPU avaialable, Using CPU')

    seed_val = 42
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    log_dir = os.path.join(os.path.dirname(__file__), "dinov2_daa_d_b_combined_testing", config.experiment_name, "runs")
    os.makedirs(log_dir, exist_ok=True)

    dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2_model = LinearClassifier(backbone=dinov2_vitb14)
    # Freeze the parameters of the DINOv2_vitb14_encoder
    for param in dinov2_model.backbone.parameters():
        param.requires_grad = False

    eval_transform = make_classification_eval_transform()
    # Use ImageFolder to create dataset(s)
    kir_test_dataset = ImageFolder(config.kir_test_dir, transform= eval_transform)
    nir_test_dataset = ImageFolder(config.nir_test_dir, transform= eval_transform)

    kir_test_dataloader = prepare_dataloader(kir_test_dataset, config.batch_size, num_workers= config.num_workers, prefetch_factor=config.prefetch_factor)
    nir_test_dataloader = prepare_dataloader(nir_test_dataset, config.batch_size, num_workers= config.num_workers, prefetch_factor=config.prefetch_factor)

    trainer = Trainer(
        model=dinov2_model,
        kir_test_dataloader=kir_test_dataloader,
        nir_test_dataloader=nir_test_dataloader,
        gpu_id = device,
        num_classes=config.num_classes,
        log_dir=log_dir,
        exp_name=config.experiment_name
    )

    trainer.combined_test(checkpoint_path=config.checkpoint_path)

if __name__ == "__main__":
    main()
