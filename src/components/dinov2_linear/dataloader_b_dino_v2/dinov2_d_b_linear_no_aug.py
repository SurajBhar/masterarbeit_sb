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
import pickle
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import hdbscan
from torchvision.datasets import ImageFolder
from sklearn.metrics.pairwise import cosine_similarity
# Local Imports
sys.path.append('/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline')
import src.components.dinov2_linear.configs.cfg_d_b_binary as cfg_loader
import src.components.dinov2_linear.utils.jobs_server as server_file
from src.components.dinov2_linear.dataloader_b_dino_v2.b_weightedimagedataset import WeightedImageDataset
from src.components.dinov2_linear.data.transforms import make_classification_eval_transform, make_classification_train_transform
################# Dataloader Class ##################################################
class ClusteredFeatureWeightingDataloader:
    def __init__(self, feature_file_path, label_file_path, img_path_file_path, num_categories, batch_size, train_transform, num_workers, prefetch_factor):
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.num_categories = num_categories
        self.features, self.labels, self.img_paths_list = self.load_data(feature_file_path, label_file_path, img_path_file_path)
        self.image_paths_all = [path for sublist in self.img_paths_list for path in sublist]
        self.dataloader_b, self.weights_list, self.all_labels, self.all_cluster_counts = self.initialize_dataloaders()

    def load_data(self, feature_file_path, label_file_path, img_path_file_path):
        with open(feature_file_path, 'rb') as file:
            features = pickle.load(file)
        with open(label_file_path, 'rb') as file:
            labels = pickle.load(file)
        with open(img_path_file_path, 'rb') as file:
            img_paths = pickle.load(file)
        return features, labels, img_paths
    
    def initialize_dataloaders(self):
        # Batch conversion of precomputed features, Batches = 254, Batch Size 1024, Feature size: [1,1280]
        # For flexbility the hardcoded 1024 can be replaced with a batch size variable
        # features_loader = [self.features[i:i+self.batch_size] for i in range(0, len(self.features), self.batch_size)]
        features_loader = [self.features[i:i+1024] for i in range(0, len(self.features), 1024)]
        gt_labels_loader = [self.labels[i:i+1024] for i in range(0, len(self.labels), 1024)]

        # Get the weights, predicted clustering labels and total cluster counts
        weights_list, all_labels, all_cluster_counts = self.process_batches(features_loader)
        # Here gt_labels_loader corresponds to the true class labels of each image
        dataset_b = WeightedImageDataset(self.img_paths_list, weights_list, gt_labels_loader, transform=self.train_transform)
        sampler_b = WeightedRandomSampler(dataset_b.weights, num_samples=len(dataset_b.weights), replacement=True)
        dataloader_b = DataLoader(dataset_b, batch_size=self.batch_size, sampler=sampler_b, num_workers=self.num_workers, pin_memory=True, prefetch_factor=self.prefetch_factor)
        # This function returns dataloader_b, weights, predicted clustering labels & cluster counts
        # dataloader_b is going to be used directly, while other can be used for further analysis
        return dataloader_b, weights_list, all_labels, all_cluster_counts
    
    def compute_weights_cosine_dist(self, features):
        cosine_dist_matrix = 1 - cosine_similarity(features).astype(np.float64)
        # Using Updated HDBSCAN for clustering with tuned Hyperparameters
        clusterer = hdbscan.HDBSCAN(min_cluster_size=25, 
                                    min_samples=1, 
                                    cluster_selection_epsilon=0.0, 
                                    metric='precomputed', 
                                    cluster_selection_method='eom', 
                                    allow_single_cluster=False)
        labels = clusterer.fit_predict(cosine_dist_matrix)

        weights = np.zeros_like(labels, dtype=float)
        unique_labels = np.unique(labels)
        noise_label = -1
        # Initialize variables for managing the new outlier clusters
        max_label = labels.max()
        current_outlier_cluster_label = max_label + 1
        outlier_cluster_count = 0

        for label in unique_labels:
            if label == noise_label:
                # Process each noise point
                for noise_index in np.where(labels == noise_label)[0]:
                    # Assign it to the current outlier cluster
                    labels[noise_index] = current_outlier_cluster_label
                    outlier_cluster_count += 1
                    weights[noise_index] = 0.001  # Assign weight as 0.001 Exp9

                    # If the outlier cluster reaches its max size, move to a new one
                    if outlier_cluster_count >= 50:
                        current_outlier_cluster_label += 1
                        outlier_cluster_count = 0
            else:
                # For non-noise points, distribute weights evenly within clusters
                indices = np.where(labels == label)[0]
                weights[indices] = 1.0 / len(indices)

        total_clusters = len(np.unique(labels)) - 1  # Exclude the original noise label

        return weights, labels, total_clusters
    
    def process_batches(self, dataloader):
        all_weights = []
        all_labels = []
        all_cluster_counts = []
        for batch_features in dataloader:
            weights, labels, total_clusters = self.compute_weights_cosine_dist(batch_features)
            all_weights.append(weights)
            all_labels.append(labels)
            all_cluster_counts.append(total_clusters)

        return all_weights, all_labels, all_cluster_counts

######################### Trainer Class #######################################
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
        self.logger = self.configure_logger()
        self.optimizer = self.configure_optimizer(optimizer_choice, lr, momentum, weight_decay)
        self.lr_scheduler = self.configure_scheduler(scheduler_choice, lr, end_lr)

    def configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        log_file_path = os.path.join(self.log_dir, "dinov2_training_log.log")
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
    
    def configure_scheduler(self, scheduler_choice, initial_lr, end_lr, last_epoch=-1):
        if scheduler_choice.lower() == 'cosineannealinglr':
            # T_max specifies the number of training epochs until the learning rate completes its 
            # first cycle and reaches the minimum value (eta_min). After this point, 
            # the cycle restarts from the maximum learning rate again.
            # T_max adjusted to: "num_epochs" to complete a single cycle over the training course
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, last_epoch=last_epoch)
        elif scheduler_choice.lower() == 'cosineannealing':
            # T_max should be (10) for 10 cosine cycles
            # After every 10th epoch the cycle starts again
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0, last_epoch=last_epoch)
        elif scheduler_choice.lower() == 'cosineannealingwarmrestarts':
            # T_max should be (10) for 10 cosine cycles
            # After every 10th epoch the cycle starts again
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=0.0, last_epoch=last_epoch)
        elif scheduler_choice.lower() == 'lambdalr':
            lr_lambda = lambda epoch: 0.1 ** (epoch // 20)  # reduces LR every 20 epochs (Step Decay)
            # lr = {0.03,0.003,0.0003,0.00003,0.000003}
            # ex: from 0 to 19 th epoch lr = 0.03 , after 20th epoch lr = 0.03 * (0.01^1) = 0.003
            # lr_lambda = lambda epoch: 1 # Keeps LR constant throughout training
            lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif scheduler_choice.lower() == 'linearinterpolationlr':
            # index of last epoch
            last_epoch_index = self.num_epochs - 1
            # Lambda function for linear interpolation between initial_lr and end_lr
            # epoch = current iteration index
            # total_epochs = total number of steps
            # lr_new = (1-epoch/total_epochs) * initial_lr + (epoch/total_epochs) * end_lr
            # Lambda_LR = initial_lr * (Lambda function return value)
            # lr_Lambda_new = (1-epoch/total_epochs) + (epoch/total_epochs)* (end_lr) * (1/initial_lr)
            lr_lambda = lambda epoch: (1 - float(epoch) / float(last_epoch_index)) + (float(epoch) / float(last_epoch_index))* end_lr * (1/initial_lr)
            lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif scheduler_choice.lower() == 'exponentialdecayexp':
            # Solution for (dN/dt) = -exponential decay constant x N
            # N(t) = N_0 exp(-exponential decay constant x t)
            # lr(t)=initial_lr × exp**(−decay_rate × epoch)
            decay_rate = 0.01
            lr_lambda = lambda epoch: math.exp(-decay_rate * epoch)
            lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError("Invalid scheduler choice. Choose among 'cosineannealingwarmrestarts',cosineannealingtencycles', 'cosineannealinglr', 'lambdalr', 'linearinterpolationlr','exponentialdecayexp'.")
        return lr_scheduler

    def _train_epoch(self):
        self.model.train()
        running_loss, num_samples = 0.0, 0
        y_pred_all = []
        y_all = []
        # Here the weighted image dataset and dataloader_b is used
        for batch, (X,_, y,_) in enumerate(self.train_dataloader):
            X, y = X.to(self.gpu_id), y.to(self.gpu_id)
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            # Applying gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            running_loss += loss.item() * X.size(0)
            num_samples += X.size(0)
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            y_pred_all.append(y_pred_class)
            y_all.append(y)

        average_loss = running_loss / num_samples
        balanced_accuracy = self.calculate_balanced_accuracy(torch.cat(y_pred_all), torch.cat(y_all), self.num_classes)

        return balanced_accuracy, average_loss

    def _validation_epoch(self):
        self.model.eval()
        running_loss, num_samples = 0.0, 0
        y_pred_all = []
        y_all = []

        with torch.no_grad():
            # Here traditional dataloader is used
            for batch, (X, y) in enumerate(self.val_dataloader):
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

    def _test_epoch(self):
        self.model.eval()
        running_loss, num_samples = 0.0, 0
        y_pred_all = []
        y_all = []

        with torch.no_grad():
            # Here traditional dataloader is used
            for batch, (X, y) in enumerate(self.test_dataloader):
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

    def train_val_test(self, max_epochs, resume=False, checkpoint_path=None):
        total_start_time = time.time()
        start_epoch = 0
        if resume:
            last_completed_epoch = self.load_checkpoint(checkpoint_path)
            start_epoch = last_completed_epoch + 1
            self._log(f"Resuming training from epoch {start_epoch}")
            # Checkpoint saves lr for last completed epoch so update it
            self.lr_scheduler.step()

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

            # Log the learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics to TensorBoard and console
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)
            self.writer.add_scalar("Train/Loss", train_loss, epoch)
            self.writer.add_scalar("Train/Balanced_Accuracy", train_balanced_accuracy, epoch)
            self.writer.add_scalar("Validation/Loss", val_loss, epoch)
            self.writer.add_scalar("Validation/Balanced_Accuracy", val_balanced_accuracy, epoch)
            self.writer.add_scalar("Test/Loss", test_loss, epoch)
            self.writer.add_scalar("Test/Balanced_Accuracy", test_balanced_accuracy, epoch)
            self._log(f"This file contains logs for: {self.experiment_name}")
            self._log(f"Epoch: {epoch} |Total training time: {self._format_time(total_epoch_duration)} | Train Loss: {train_loss:.4f} | Train Balanced Accuracy: {train_balanced_accuracy * 100:.4f} %")
            self._log(f"Epoch: {epoch} |Total validation time:{self._format_time(total_epoch_val_duration)} | Validation Loss: {val_loss:.4f} | Validation Balanced Accuracy: {val_balanced_accuracy * 100:.4f} % ")
            self._log(f"Epoch: {epoch} |Total testing time: {self._format_time(total_epoch_test_duration)} | Test Loss: {test_loss:.4f} | Test Balanced Accuracy: {test_balanced_accuracy * 100:.4f} %")

            if (epoch+1) % self.save_every == 0:
                self._save_checkpoint(epoch, train_balanced_accuracy, train_loss, val_balanced_accuracy, val_loss,  test_balanced_accuracy, test_loss)
            # Updating the learning rate schedule
            self.lr_scheduler.step()
            
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
        """
        Loads a checkpoint, restoring the model state, optimizer state, and the current learning rate.
        Optionally restores the learning rate scheduler state.
        """
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
    print(f"GPU device: {config.device}")
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

    log_dir = os.path.join(os.path.dirname(__file__), "dinov2_daa_d_b_experiments", config.experiment_name, "runs")
    os.makedirs(log_dir, exist_ok=True)

    dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2_model = LinearClassifier(backbone=dinov2_vitb14)
    # Freeze the parameters of the DINOv2_vitb14_encoder
    for param in dinov2_model.backbone.parameters():
        param.requires_grad = False

    train_transform = make_classification_train_transform()
    eval_transform = make_classification_eval_transform()

    # Use dataloader class for loading train data
    # Created an instance of DataLoaderEpochsCalculation and performed the comparison
    train_dataloader_b = ClusteredFeatureWeightingDataloader(
        config.feature_file_path, 
        config.label_file_path, 
        config.img_path_file_path, 
        config.num_classes, 
        config.batch_size, 
        train_transform=train_transform,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
    )
    train_dataloader = train_dataloader_b.dataloader_b

    # Use ImageFolder to create Validation and Test dataset(s)
    val_dataset = ImageFolder(config.val_dir, transform=eval_transform)
    test_dataset = ImageFolder(config.test_dir, transform=eval_transform)

    # Pass the adjusted batch size here
    val_dataloader = prepare_dataloader(val_dataset, config.batch_size, num_workers= config.num_workers, prefetch_factor=config.prefetch_factor)
    test_dataloader = prepare_dataloader(test_dataset, config.batch_size, num_workers= config.num_workers, prefetch_factor=config.prefetch_factor)

    trainer = Trainer(
        model=dinov2_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer_choice=config.optimizer,
        scheduler_choice=config.scheduler,
        lr=config.lr,
        end_lr=config.end_lr,
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
    main()
