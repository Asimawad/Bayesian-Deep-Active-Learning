import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from Models.bayesianModel import create_model_optimizer, mc_dropout, compute_performance_metrics

# Global configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 1024
EPOCHS = 10
MC_SAMPLES = 32
LEARNING_RATE = 0.001

# Data loaders for MNIST
def get_mnist_loaders(batch_size, eval_batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_loader, test_loader

# Training function
def train_model(model, optimizer, train_loader, test_loader, epochs, device):
    """
    Train a Bayesian Neural Network on MNIST and evaluate test accuracy and NLL.

    Args:
        model (nn.Module): Bayesian Neural Network.
        optimizer (Optimizer): Optimizer for training.
        train_loader (DataLoader): DataLoader for training set.
        test_loader (DataLoader): DataLoader for test set.
        epochs (int): Number of training epochs.
        device (str): Device to use for training.

    Returns:
        dict: Training history containing losses and accuracies.
    """
    criterion = nn.NLLLoss()
    model.to(device)
    
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            log_probs, _ = model(inputs)
            loss = criterion(log_probs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = log_probs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        # Evaluate with MC Dropout
        model.train()  # Ensure dropout is enabled
        log_probs = mc_dropout(model, test_loader, n=MC_SAMPLES, device=device)
        test_accuracy, test_nll = compute_performance_metrics(log_probs, test_loader)
        history["test_loss"].append(test_nll)
        history["test_accuracy"].append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy * 100:.2f}% | "
              f"Test Loss: {test_nll:.4f} | Test Accuracy: {test_accuracy * 100:.2f}%")

    return history

# Plot training curves
def plot_training_curves(history):
    """
    Plot training and test accuracy and NLL curves.

    Args:
        history (dict): Training history containing losses and accuracies.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["test_accuracy"], label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid()

    # Loss (NLL)
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train NLL")
    plt.plot(history["test_loss"], label="Test NLL")
    plt.xlabel("Epochs")
    plt.ylabel("NLL")
    plt.title("Negative Log Likelihood (NLL) vs Epochs")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE, EVAL_BATCH_SIZE)

    # Initialize model and optimizer
    seed = 42
    model, optimizer = create_model_optimizer(seed, LEARNING_RATE)

    # Train model
    history = train_model(model, optimizer, train_loader, test_loader, EPOCHS, DEVICE)

    # Plot training curves
    plot_training_curves(history)
