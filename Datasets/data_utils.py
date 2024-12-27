import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_data_loaders(batch_size=64, eval_batch_size=1024, device="cuda"):
    """
    Prepare train/test loaders for MNIST, Dirty-MNIST, and Fashion-MNIST datasets.

    Args:
        batch_size (int): Batch size for training loaders.
        eval_batch_size (int): Batch size for evaluation loaders.
        device (str): Device to use for dataset preparation.

    Returns:
        Tuple of DataLoaders: (train_loader_mnist, train_loader_dirty_mnist, 
                               test_loader_mnist, test_loader_dirty_mnist, test_loader_fashion)
    """
    # MNIST Dataset
    train_dataset_mnist = datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    )
    test_dataset_mnist = datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    )

    # Fashion-MNIST Dataset
    test_dataset_fashion = datasets.FashionMNIST(
        './data', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    )

    # Dirty-MNIST Dataset (Placeholder)
    # Replace with the actual Dirty-MNIST dataset if available
    train_dataset_dirty_mnist = train_dataset_mnist  # Replace with actual dataset
    test_dataset_dirty_mnist = test_dataset_mnist  # Replace with actual dataset

    # Create DataLoaders 
    train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=batch_size, shuffle=True)
    train_loader_dirty_mnist = DataLoader(train_dataset_dirty_mnist, batch_size=batch_size, shuffle=True)
    test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=eval_batch_size, shuffle=False)
    test_loader_dirty_mnist = DataLoader(test_dataset_dirty_mnist, batch_size=eval_batch_size, shuffle=False)
    test_loader_fashion = DataLoader(test_dataset_fashion, batch_size=eval_batch_size, shuffle=False)

    return (train_loader_mnist, train_loader_dirty_mnist, 
            test_loader_mnist, test_loader_dirty_mnist, test_loader_fashion)

def select_k(k, pool, labels):
    """
    Randomly sample `k` labeled examples per class from the pool.

    Args:
        k (int): Number of samples to select per class.
        pool (list): List of indices grouped by class.
        labels (np.ndarray): Array of labels for the dataset.

    Returns:
        np.ndarray: Indices of selected samples.
    """
    choices = []
    for c in range(10):  # Assuming 10 classes
        class_indices = pool[c]
        sampled_indices = np.random.choice(class_indices, k, replace=False)
        choices.append(sampled_indices)

        # Remove selected indices from the pool
        pool[c] = np.setdiff1d(class_indices, sampled_indices)

    return np.concatenate(choices)

def create_pool(dataset):
    """
    Create a pool of indices grouped by class for active learning.

    Args:
        dataset (torchvision.datasets): Dataset object.

    Returns:
        list: List of indices grouped by class.
    """
    data, labels = dataset.data, dataset.targets
    labels = labels.numpy() if torch.is_tensor(labels) else np.array(labels)
    pool = [np.where(labels == c)[0] for c in range(10)]  # Group indices by class
    return pool

def subset_loader(dataset, indices, batch_size=32):
    """
    Create a DataLoader for a given subset of the dataset.

    Args:
        dataset (torchvision.datasets): Dataset object.
        indices (list): Indices for the subset.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the subset.
    """
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)
