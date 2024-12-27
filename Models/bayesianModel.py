import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianMNISTCNN(nn.Module):
    """
    Bayesian Neural Network for MNIST using MC Dropout.
    """
    def __init__(self, dropout_rate=0.2):
        super(BayesianMNISTCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for Bayesian inference
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Input shape: (batch_size, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # Shape: (batch_size, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # Shape: (batch_size, 64, 7, 7)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64*7*7)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        log_probs = F.log_softmax(self.fc2(x), dim=1)  # Log probabilities for classes

        return log_probs, x  # Return embeddings for uncertainty calculation

def create_model_optimizer(seed, learning_rate=0.001, dropout_rate=0.2):
    """
    Initialize the BayesianMNISTCNN model and its optimizer.

    Args:
        seed (int): Random seed for reproducibility.
        learning_rate (float): Learning rate for the optimizer.
        dropout_rate (float): Dropout rate for the BNN.

    Returns:
        Tuple[nn.Module, torch.optim.Optimizer]: Model and optimizer.
    """
    torch.manual_seed(seed)
    model = BayesianMNISTCNN(dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

@torch.inference_mode()
def mc_dropout(model, data_loader, n=32, device='cuda'):
    """
    Perform MC Dropout sampling to compute uncertainties.

    Args:
        model (nn.Module): Bayesian model with dropout.
        data_loader (DataLoader): DataLoader for the dataset.
        n (int): Number of MC dropout samples.
        device (str): Device to use.

    Returns:
        torch.Tensor: Log probabilities of shape (n_samples, batch_size, num_classes).
    """
    model.to(device)
    model.train()  # Enable dropout for MC sampling

    all_outputs = []
    for data, _ in data_loader:
        data = data.to(device)
        outputs = []
        for _ in range(n):
            log_probs, _ = model(data)
            outputs.append(log_probs)
        all_outputs.append(torch.stack(outputs, dim=0))  # Stack along sample axis

    model.eval()
    return torch.cat(all_outputs, dim=1)  # Shape: (n_samples, batch_size, num_classes)

@torch.inference_mode()
def compute_performance_metrics(log_probs, data_loader):
    """
    Compute accuracy and NLL for a given set of predictions.

    Args:
        log_probs (torch.Tensor): Log probabilities of shape (n_samples, batch_size, num_classes).
        data_loader (DataLoader): DataLoader with the ground-truth labels.

    Returns:
        Tuple[float, float]: Accuracy and NLL.
    """
    mean_log_probs = torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(log_probs.shape[0]))
    predictions = mean_log_probs.argmax(dim=1)

    all_labels = torch.cat([labels for _, labels in data_loader]).to(log_probs.device)
    accuracy = (predictions == all_labels).float().mean().item()
    nll = -mean_log_probs[range(len(all_labels)), all_labels].mean().item()

    return accuracy, nll
