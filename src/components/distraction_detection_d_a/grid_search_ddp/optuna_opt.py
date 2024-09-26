import optuna
import argparse
from functools import partial
from optuna.trial import TrialState
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.utils.data

import os
import sys
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


# Local Imports
sys.path.append('/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline')

from src.components.distraction_detection_d_a.binary_dataset import BinaryClassificationDataset
import src.components.distraction_detection_d_a.cfg_binary_baseline as cfg_loader
from src.components import utils
import src.components.distraction_detection_d_a.jobs_server as server_file

# Configurations for the experiment

DIR = os.getcwd()
EPOCHS = 50
N_TRIALS = 20
BATCHSIZE = 1024
CLASSES = 2

train_dir = "/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_0/train"
val_dir = "/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_0/val"
test_dir = "/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_0/test"

def get_stratified_indices(labels, test_size=0.1):
    """
    The get_stratified_indices function:
    It is a utility for generating a stratified subset of a dataset in PyTorch using indices, 
    leveraging sklearn's train_test_split for stratification. 
    This ensures that the class distribution in the subset matches that of the original dataset, 
    which is important for maintaining the integrity of machine learning models, 
    especially when dealing with imbalanced classes as in DAA.
    """
    # Generate indices for a stratified split
    # X_train, X_test, y_train, y_test
    _, stratified_idx, _, _ = train_test_split(
        range(len(labels)), labels, test_size=test_size, stratify=labels, random_state=42)
    
    return stratified_idx

def calculate_balanced_accuracy(y_pred, y_true, num_classes=2):
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

# Objective Function Here
def objective(single_trial, device_id):
    trial = optuna.integration.TorchDistributedTrial(single_trial)

    # Model definition here
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

    # Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Change the classifier head
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=2)
    pretrained_vit_transforms = pretrained_vit_weights.transforms()

    train_dataset = BinaryClassificationDataset(train_dir, transform=pretrained_vit_transforms)
    val_dataset = BinaryClassificationDataset(val_dir, transform=pretrained_vit_transforms)

    train_indices = get_stratified_indices(train_dataset.all_binary_labels)
    val_indices = get_stratified_indices(val_dataset.all_binary_labels)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_subset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=val_subset, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        sampler=train_sampler,
        batch_size=BATCHSIZE,
        shuffle=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        val_subset,
        sampler=valid_sampler,
        batch_size=BATCHSIZE,
        shuffle=False,
    )

    # Generate the model.
    model = DDP(pretrained_vit.to(device_id), device_ids=None if device_id == "cpu" else [device_id])

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    bal_accuracy = 0
    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        # Shuffle train dataset.
        train_loader.sampler.set_epoch(epoch)
        for X, y, _,_ in train_loader:
            X, y = X.to(device_id), y.to(device_id)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # Validation of the model.
        model.eval()
        y_pred_all, y_all = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        y_pred_all = y_pred_all.to(device_id)
        y_all = y_all.to(device_id)
        with torch.no_grad():
            for X, y, _,_ in valid_loader:
                X, y = X.to(device_id), y.to(device_id)
                y_pred = model(X)
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                y_pred_all = torch.cat((y_pred_all, y_pred_class))
                y_all = torch.cat((y_all, y))

        # Gather all predictions and labels on the root GPU
        y_pred_all_gathered = [torch.zeros_like(y_pred_all) for _ in range(torch.distributed.get_world_size())]
        y_all_gathered = [torch.zeros_like(y_all) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(y_pred_all_gathered, y_pred_all)  # Gathering predictions
        torch.distributed.all_gather(y_all_gathered, y_all)  # Gathering labels

        # Flatten the lists of tensors into single tensors
        y_pred_all_gathered = torch.cat(y_pred_all_gathered)
        y_all_gathered = torch.cat(y_all_gathered)
        val_balanced_accuracy = calculate_balanced_accuracy(y_pred_all_gathered, y_all_gathered, num_classes=2)
        
        trial.report(val_balanced_accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned(f"Trial was pruned at epoch {epoch}.")

    return val_balanced_accuracy

def setup(backend, rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Using {backend} backend.")

def cleanup():
    dist.destroy_process_group()

def run_optimize(rank, world_size, device_ids, return_dict, master_port):
    device = "cpu" if len(device_ids) == 0 else device_ids[rank]
    print(f"Running basic DDP example on rank {rank} device {device}.")

    # Set environmental variables required by torch.distributed.
    backend = "gloo"
    if torch.distributed.is_nccl_available():
        if device != "cpu":
            backend = "nccl"
    setup(backend, rank, world_size, master_port)

    if rank == 0:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            partial(objective, device_id=device),
            n_trials=N_TRIALS,
            timeout=300,
        )
        return_dict["study"] = study
    else:
        for _ in range(N_TRIALS):
            try:
                objective(None, device)
            except optuna.TrialPruned:
                pass

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch distributed data-parallel training with spawn example."
    )
    parser.add_argument(
        "--device-ids",
        "-d",
        nargs="+",
        type=int,
        default=[0],
        help="Specify device_ids if using GPUs.",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable CUDA training."
    )
    parser.add_argument("--master-port", type=str, default="12355", help="Specify port number.")
    args = parser.parse_args()
    if args.no_cuda:
        device_ids = []
    else:
        device_ids = args.device_ids

    world_size = max(len(device_ids), 1)
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        run_optimize,
        args=(world_size, device_ids, return_dict, args.master_port),
        nprocs=world_size,
        join=True,
    )
    study = return_dict["study"]

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

