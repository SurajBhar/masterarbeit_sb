# Standard Library Imports
import os
import sys
import argparse
import logging
import json
from glob import glob
from typing import Dict, List, Tuple, Optional

# Third-Party Library Imports
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Local Imports
sys.path.append('/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline')
from src.components import data_setup
from src.components import utils

def configure_logger(log_dir, experiment_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a file handler for logging
    log_file_path = os.path.join(log_dir, "testing_log.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Create a console handler for real-time progress
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

######### Class for Testing the best available model on test dataset splits  ######################
class Testing:
    def __init__(
        self,
        model: torch.nn.Module,
        test_dataloader: DataLoader,
        gpu_id: int,
        log_dir:str,
        json_file_path_bal_acc,
        exp_name:str, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.test_data = test_dataloader
        self.log_dir = log_dir
        self.json_file_path_bal_acc = json_file_path_bal_acc
        self.experiment_name = exp_name
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Configure logging
        self.logger = configure_logger(self.log_dir, self.experiment_name)

    def test_step(self,
                  model: torch.nn.Module, 
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device) -> Tuple[float, float, list, list]:
        """
        Tests a PyTorch model for a single epoch.

        Turns a target PyTorch model to "eval" mode and then performs
        a forward pass on a testing dataset.

        Args:
            model: A PyTorch model to be tested.
            dataloader: A DataLoader instance for the model to be tested on.
            device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
            A tuple of testing loss and testing accuracy and test balanced accuracy metrics lists.
            In the form (test_loss, test_accuracy, pred, labels). For example:

            (0.0223, 0.8985, list, list)
        """
        # Put model in eval mode
        model.eval() 

        # Setup test loss and test accuracy values
        test_loss, test_acc= 0, 0

        test_y_pred_all = []
        test_y_all = []

        # Turn on inference context manager
        with torch.no_grad():
            # Loop through DataLoader batches
            for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing", leave=False):
                # Send data to target device
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                test_pred_logits = model(X)

                # 2. Calculate and accumulate loss
                loss = F.cross_entropy(test_pred_logits, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

                # Accumulate the prediction per batch
                test_y_pred_all.append(test_pred_labels)
                test_y_all.append(y)


        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

        # Concatenate all the predictions and ground truths per epoch
        test_y_pred_all = torch.concatenate(test_y_pred_all)
        test_y_all = torch.concatenate(test_y_all)

        return test_loss, test_acc, test_y_pred_all, test_y_all

    def _log(self, message):
        self.logger.info(message)

    def test(self):
        try:
            # Specify the dedicated path for the text/json data
            json_path = self.json_file_path_bal_acc
            json_filename_test = "test_data_bal_acc.json"
            full_json_path_test = os.path.join(json_path, json_filename_test)

            test_loss, test_acc, test_y_pred_all, test_y_all= self.test_step(model=self.model,
                                                                dataloader=self.test_data,
                                                                device=self.gpu_id)
            # Prepare the data as a dictionary
            epoch_data_test = {"predictions": test_y_pred_all.cpu().detach().numpy(), "labels": test_y_all.cpu().detach().numpy()}

            # At the end of test step, save the data to the text file
            # storing the tensor data as NumPy arrays within the JSON file.
            with open(full_json_path_test, mode='a') as json_file:
                json.dump(epoch_data_test, json_file, default=lambda x: x.tolist())
                json_file.write('\n')  # Add a newline to separate entries

            self._log(f" Testing the best Model on unseen test dataset| Test Loss on test dataset: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

        except Exception as e:
            self._log(f"An unexpected error occurred: {e}")

def create_log_and_json_dirs(experiment_name):
    exp_path = os.path.dirname(__file__)
    log_dir_path = os.path.join(exp_path, "experiments", experiment_name, "runs")
    json_file_path_bal_acc = os.path.join(exp_path, "experiments", experiment_name, "accumulated_pred_gt")

    try:
        os.makedirs(log_dir_path, exist_ok=True)
        print("Log directory for Tensorboard Events/Logs created")
        os.makedirs(json_file_path_bal_acc, exist_ok=True)
        print("JSON file directory created")
    except FileExistsError:
        print("Log directory already exists")

    return log_dir_path, json_file_path_bal_acc

def setup_and_create_dataloaders(batch_size, test_dir):
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

    # Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Change the classifier head
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=34)
    pretrained_vit_transforms = pretrained_vit_weights.transforms()

    test_dataloader, class_names = data_setup.create_test_dataloader(
        test_dir=test_dir,
        transform=pretrained_vit_transforms,
        batch_size=batch_size,
        num_workers=8
    )

    return test_dataloader, class_names , pretrained_vit


def main():
    parser = argparse.ArgumentParser(description="Training script for Vision Transformer")
    parser.add_argument("--experiment_name", type=str, default="First_Experiment", help="Name of the experiment")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--test_dir", type=str, default="/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_0/test", help="Path to the test data directory")
    ###################### Attention Here ##########################################
    # Update the Checkpoint path everytime you want to test the best available model
    parser.add_argument("--checkpoint_path", type=str, default="/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/src/components/experiments/Fourth_SGD/checkpoints/checkpoint_Fourth_SGD_epoch_89.pth", help="Path to the checkpoint directory")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(3)
    else:
        device = torch.device('cpu')
        print('No GPU avaialable, Using CPU')

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.set_seeds(1)
    log_dir, json_file_path_bal_acc = create_log_and_json_dirs(args.experiment_name)

    test_dataloader, _, model_vit = setup_and_create_dataloaders(args.batch_size, args.test_dir)
    
    # Load the Best model checkpoint
    checkpoint_path = args.checkpoint_path  # Update this with your checkpoint path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_vit.load_state_dict(checkpoint['model_state_dict'])
    
    tester = Testing(model=model_vit, 
                      test_dataloader=test_dataloader, 
                      gpu_id=device, 
                      log_dir=log_dir, 
                      json_file_path_bal_acc=json_file_path_bal_acc, 
                      exp_name=args.experiment_name)
    tester.test()

# The if __name__ == "__main__": guard is used to ensure 
# that the code inside it only runs when the script is executed directly,
#  not when it's imported as a module.
if __name__ == "__main__":
    main()
