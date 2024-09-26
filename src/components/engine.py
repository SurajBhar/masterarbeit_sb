"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
               device: torch.device) -> Tuple[float, float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        scheduler: A PyTorch scheduler to help the scheduling of learning rate.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Setup the Balanced Accuracy Score values
    train_balanced_acc = 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        lr_scheduler.step()

        # Calculate and accumulate accuracy and balanced accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
        y_pred_class_cpu = y_pred_class.detach().cpu()
        y_numpy = y.numpy() if isinstance(y, torch.Tensor) else y
        balanced_acc = balanced_accuracy_score(y_numpy, y_pred_class_cpu.numpy())
        train_balanced_acc += balanced_acc

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_balanced_acc = train_balanced_acc/ len(dataloader)
    return train_loss, train_acc, train_balanced_acc


def validation_step(model: torch.nn.Module,
                    dataloader:torch.utils.data.DataLoader,
                    loss_fn: torch.nn.Module, 
                    device: torch.device)-> Tuple[float, float, float]:
    model.eval()
    val_loss, val_acc, val_balanced_acc = 0, 0, 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            val_acc += (y_pred_class == y).sum().item() / len(y_pred)

            y_pred_class_cpu = y_pred_class.detach().cpu()
            y_numpy = y.numpy() if isinstance(y, torch.Tensor) else y
            balanced_acc = balanced_accuracy_score(y_numpy, y_pred_class_cpu.numpy())
            val_balanced_acc += balanced_acc

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    val_balanced_acc = val_balanced_acc / len(dataloader)
    return val_loss, val_acc, val_balanced_acc



def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc, test_balanced_acc = 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

            y_pred_class_cpu = test_pred_labels.detach().cpu()
            y_numpy = y.numpy() if isinstance(y, torch.Tensor) else y
            balanced_acc = balanced_accuracy_score(y_numpy, y_pred_class_cpu.numpy())
            test_balanced_acc += balanced_acc

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    test_balanced_acc = test_balanced_acc / len(dataloader)
    return test_loss, test_acc, test_balanced_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and validate a PyTorch model.

    Passes a target PyTorch models through train_step() and validation_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_balanced_acc = train_step(model=model,
                                                               dataloader=train_dataloader,
                                                               loss_fn=loss_fn,
                                                               optimizer=optimizer,
                                                               lr_scheduler = lr_scheduler,
                                                               device=device)
        val_loss, val_acc, val_balanced_acc = validation_step(model=model,
                                                              dataloader=test_dataloader,
                                                              loss_fn=loss_fn,
                                                              device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"train_acc: {train_balanced_acc:.4f} | "
            f"test_loss: {val_loss:.4f} | "
            f"test_acc: {val_acc:.4f} |"
            f"test_loss: {val_balanced_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_acc"].append(train_balanced_acc)
        results["test_loss"].append(val_loss)
        results["test_acc"].append(val_acc)
        results["train_acc"].append(val_balanced_acc)

    # Return the filled results at the end of the epochs
    return results