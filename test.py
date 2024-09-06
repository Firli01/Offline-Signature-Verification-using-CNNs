import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def test_model(model, test_loader: DataLoader, device: torch.device):
    model.eval()
    predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # print(f"Predictions: {predictions}")

    return accuracy