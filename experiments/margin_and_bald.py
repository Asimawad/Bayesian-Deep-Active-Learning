import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from train.train_mnist import train_labeled_model, evaluate_model
from Models.bayesianModel import create_model_optimizer, mc_dropout, compute_uncertainties
from Datasets.data_utils import get_data_loaders
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def select_random_indices(pool, num_samples):
    """
    Randomly select indices from the unlabeled pool.

    Args:
        pool (list): List of available indices per class.
        num_samples (int): Number of samples to acquire.

    Returns:
        selected_indices (list): Selected indices.
        updated_pool (list): Updated pool after selection.
    """
    selected_indices = []
    for class_pool in pool:
        selected = np.random.choice(class_pool, size=num_samples, replace=False)
        selected_indices.extend(selected)
        updated_class_pool = np.setdiff1d(class_pool, selected)
    return np.array(selected_indices), updated_class_pool

def get_mnist_loaders(batch_size, eval_batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_loader, test_loader


def margin_sampling(model, unlabeled_pool, train_dataset, k):
    """
    Perform margin sampling to select k samples with the smallest margin.
    """
    margins = []
    indices = []
    subset = Subset(train_dataset, indices=unlabeled_pool)
    subset_loader = DataLoader(subset, batch_size=64, shuffle=False)

    model.eval()
    with torch.no_grad():
        for data, _ in subset_loader:
            data = data.to(DEVICE)
            log_probs = mc_dropout(model, data.unsqueeze(0), n=32)
            probs = log_probs.exp()
            sorted_probs, _ = torch.sort(probs, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            margins.extend(margin.cpu().numpy())

    # Select top k samples with the smallest margins
    selected_indices = np.argsort(margins)[:k]
    return [unlabeled_pool[i] for i in selected_indices]

def bald_sampling(model, unlabeled_pool, train_dataset, k):
    """
    Perform BALD sampling to select k samples with the highest epistemic uncertainty.
    """
    epistemic_uncertainties = []
    subset = Subset(train_dataset, indices=unlabeled_pool)
    subset_loader = DataLoader(subset, batch_size=64, shuffle=False)

    model.eval()
    with torch.no_grad():
        for data, _ in subset_loader:
            data = data.to(DEVICE)
            log_probs = mc_dropout(model, data.unsqueeze(0), n=32)
            _, _, _, epistemic_uncertainty = compute_uncertainties(log_probs)
            epistemic_uncertainties.extend(epistemic_uncertainty.cpu().numpy())

    # Select top k samples with the highest epistemic uncertainty
    selected_indices = np.argsort(epistemic_uncertainties)[-k:]
    return [unlabeled_pool[i] for i in selected_indices]

def active_learning_experiment(acquisition_function, train_dataset, test_loader, num_classes=10, trials=10):
    """
    Conduct an active learning experiment using a specified acquisition function.
    """
    labeled_per_class = [2, 4, 8, 16, 32, 64, 128]
    results = []

    for trial in range(trials):
        np.random.seed(42 + trial)
        unlabeled_pool = np.arange(len(train_dataset))
        labeled_indices = select_random_indices(train_dataset, 2, num_classes)
        trial_results = []

        for examples_per_class in labeled_per_class:
            # Train and evaluate
            model, optimizer = create_model_optimizer(seed=42 + trial)
            subset = Subset(train_dataset, indices=labeled_indices)
            subset_loader = DataLoader(subset, batch_size=32, shuffle=True)

            train_labeled_model(model, optimizer, subset_loader, test_loader, epochs=10)
            accuracy, nll = evaluate_model(model, test_loader)

            trial_results.append((len(labeled_indices), accuracy, nll))

            # Acquire more samples
            selected_indices = acquisition_function(model, unlabeled_pool, train_dataset, examples_per_class)
            labeled_indices.extend(selected_indices)
            unlabeled_pool = np.setdiff1d(unlabeled_pool, selected_indices)

        results.append(trial_results)

    return results

def plot_learning_curves(results, title="Active Learning Performance"):
    """
    Plot learning curves with confidence intervals.
    """
    num_examples = [r[0] for r in results[0]]
    mean_acc = np.mean([[r[1] for r in trial] for trial in results], axis=0)
    std_acc = np.std([[r[1] for r in trial] for trial in results], axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(num_examples, mean_acc, label='Accuracy')
    plt.fill_between(num_examples, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, label='Confidence Interval')
    plt.xscale('log')
    plt.xlabel('Number of Labeled Examples (log scale)')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load data
    train_loader, test_loader = get_mnist_loaders(batch_size=64, eval_batch_size=1024)
    train_dataset = train_loader.dataset

    # Run Margin Sampling
    margin_results = active_learning_experiment(margin_sampling, train_dataset, test_loader)
    plot_learning_curves(margin_results, title="Margin Sampling Performance")

    # Run BALD Sampling
    bald_results = active_learning_experiment(bald_sampling, train_dataset, test_loader)
    plot_learning_curves(bald_results, title="BALD Sampling Performance")
