import os
from typing import Tuple
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from visualize_results import generate_graphs


def train_model(model,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: torch.optim,
                criterion, num_epochs,
                device: torch.device,
                save_model_and_dataloader: bool = False,
                save_name: str = None,
                plot_name: Tuple = None):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    print(f"\nRunning training for {num_epochs} epochs\n")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc='Training', unit='batch', leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_accuracy = 100 * correct / total
        train_acc_history.append(train_accuracy)

        print(f'\nEpoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%\n')

        # VALIDATION
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation', unit='batch', leave=False):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total

        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)

        print(f'\nEpoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n')
    if plot_name:
        generate_graphs(save_name=plot_name,
                        train_loss_acc=(train_loss_history, train_acc_history),
                        val_loss_acc=(val_loss_history, val_acc_history)
                        )
    if save_model_and_dataloader:
        torch.save(model.state_dict(), os.path.join("models", f"model_{save_name}"))

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history
