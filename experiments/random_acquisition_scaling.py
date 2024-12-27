import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from Models.bayesianModel import create_model_optimizer, mc_dropout, compute_performance_metrics
from train.train_mnist import get_mnist_loaders, train_labeled_model, evaluate_model
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def random_acquisition_scaling_epochs(train_dataset, test_loader, labeled_indices, epochs_list, num_classes=10):
    """
    Train the BNN with varying epochs and evaluate test accuracy and NLL.
    """
    test_accuracies = []
    test_nlls = []

    for epochs in epochs_list:
        print(f"Training for {epochs} epochs...")
        model, optimizer = create_model_optimizer(seed=42)
        subset = Subset(train_dataset, indices=labeled_indices)
        subset_loader = DataLoader(subset, batch_size=32, shuffle=True)

        # Train and evaluate
        train_labeled_model(model, optimizer, subset_loader, test_loader, epochs)
        log_probs = mc_dropout(model, test_loader, n=32)
        accuracy, nll = compute_performance_metrics(log_probs, test_loader)

        test_accuracies.append(accuracy * 100)
        test_nlls.append(nll)

        print(f"Epochs: {epochs} | Test Accuracy: {accuracy * 100:.2f}% | NLL: {nll:.4f}")

    return test_accuracies, test_nlls

def random_acquisition_scaling_samples(train_dataset, test_loader, examples_list, epochs=10, num_classes=10):
    """
    Train the BNN with varying labeled examples and evaluate test accuracy and NLL.
    """
    test_accuracies = []
    test_nlls = []

    for examples_per_class in examples_list:
        print(f"Training with {examples_per_class} labeled examples per class...")
        labeled_indices = select_random_indices(train_dataset, examples_per_class, num_classes)

        model, optimizer = create_model_optimizer(seed=42)
        subset = Subset(train_dataset, indices=labeled_indices)
        subset_loader = DataLoader(subset, batch_size=32, shuffle=True)

        # Train and evaluate
        train_labeled_model(model, optimizer, subset_loader, test_loader, epochs)
        log_probs = mc_dropout(model, test_loader, n=32)
        accuracy, nll = compute_performance_metrics(log_probs, test_loader)

        test_accuracies.append(accuracy * 100)
        test_nlls.append(nll)

        print(f"Labeled Examples: {examples_per_class * num_classes} | Test Accuracy: {accuracy * 100:.2f}% | NLL: {nll:.4f}")

    return test_accuracies, test_nlls

def heuristic_epochs(labeled_pool_size):
    """
    Define a heuristic for the number of epochs based on the labeled pool size.
    """
    if labeled_pool_size < 100:
        return 200
    elif labeled_pool_size < 400:
        return 100
    elif labeled_pool_size < 1000:
        return 50
    else:
        return 10

def plot_scaling_behavior(x_values, accuracies, nlls, x_label, title):
    """
    Plot test accuracy and NLL against the provided x_values.
    """
    plt.figure(figsize=(12, 6))

    # Plot Test Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(x_values, accuracies, marker='o', label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel(x_label)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Test Accuracy vs. {x_label} ({title})')
    plt.grid(True)
    plt.legend()

    # Plot NLL
    plt.subplot(1, 2, 2)
    plt.plot(x_values, nlls, marker='o', color='red', label='Test NLL')
    plt.xscale('log')
    plt.xlabel(x_label)
    plt.ylabel('Negative Log Likelihood (NLL)')
    plt.title(f'Test NLL vs. {x_label} ({title})')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data
    train_loader, test_loader = get_mnist_loaders(batch_size=64, eval_batch_size=1024)
    train_dataset = train_loader.dataset

    # Subpart 1: Scaling with epochs
    labeled_indices = select_random_indices(train_dataset, 5, 10)
    epochs_list = np.geomspace(start=1, stop=2048, num=12, dtype=int)
    acc_epochs, nll_epochs = random_acquisition_scaling_epochs(train_dataset, test_loader, labeled_indices, epochs_list)

    plot_scaling_behavior(epochs_list, acc_epochs, nll_epochs, 'Number of Epochs (log scale)', 'Scaling with Epochs')

    # Subpart 2: Scaling with labeled examples
    examples_list = [2, 4, 8, 16, 32, 64, 128]
    acc_samples, nll_samples = random_acquisition_scaling_samples(train_dataset, test_loader, examples_list)

    plot_scaling_behavior([e * 10 for e in examples_list], acc_samples, nll_samples, 'Number of Labeled Examples (log scale)', 'Scaling with Examples')

    # Subpart 3: Analysis and heuristic-based training
    print("Analyzing the effects of epochs vs. labeled examples...")

    # Subpart 4: Random acquisition with heuristic
    labeled_indices = select_random_indices(train_dataset, 2, 10)
    labeled_pool = list(labeled_indices)

    heuristic_train_accuracies = []
    heuristic_test_accuracies = []

    while len(labeled_pool) < 1020:
        epochs = heuristic_epochs(len(labeled_pool))
        model, optimizer = create_model_optimizer(seed=42)
        train_subset = Subset(train_dataset, labeled_pool)
        train_loader_subset = DataLoader(train_subset, batch_size=32, shuffle=True)

        train_labeled_model(model, optimizer, train_loader_subset, test_loader, epochs)
        test_acc = evaluate_model(model, test_loader)

        heuristic_train_accuracies.append((len(labeled_pool), train_acc))
        heuristic_test_accuracies.append((len(labeled_pool), test_acc))

        # Acquire more samples
        acquired_indices, unlabeled_pool = select_random_indices(unlabeled_pool, BATCH_SIZE)
        labeled_pool.extend(acquired_indices)

    plot_learning_curve(heuristic_test_accuracies, "Random Acquisition with Heuristic")
