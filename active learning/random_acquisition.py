import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from Models.bayesianModel import create_model_optimizer, mc_dropout, compute_performance_metrics
from train.train_mnist import get_mnist_loaders, plot_training_curves
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
MC_SAMPLES = 32
NUM_TRIALS = 10
INIT_SAMPLES_PER_CLASS = 2
BATCH_SIZE = 10
FINAL_SAMPLES_PER_CLASS = 102

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

def active_learning_random_acquisition(model, optimizer, train_loader, test_loader, num_classes=10):
    """
    Perform active learning with random acquisition.

    Args:
        model (nn.Module): Bayesian Neural Network.
        optimizer (Optimizer): Optimizer for training.
        train_loader (DataLoader): DataLoader for training set.
        test_loader (DataLoader): DataLoader for test set.

    Returns:
        train_accuracies (list): Train accuracies over trials.
        test_accuracies (list): Test accuracies over trials.
    """
    train_accuracies = []
    test_accuracies = []

    # Initialize pool of labeled and unlabeled data
    train_dataset = train_loader.dataset
    all_indices = np.arange(len(train_dataset))
    train_labels = np.array([train_dataset[i][1] for i in all_indices])

    labeled_pool = []
    unlabeled_pool = [np.where(train_labels == c)[0] for c in range(num_classes)]

    # Select initial samples
    for c in range(num_classes):
        initial_samples, unlabeled_pool[c] = select_random_indices(unlabeled_pool[c], INIT_SAMPLES_PER_CLASS)
        labeled_pool.extend(initial_samples)

    labeled_pool = np.array(labeled_pool)

    # Active learning loop
    while len(labeled_pool) < FINAL_SAMPLES_PER_CLASS * num_classes:
        train_indices = labeled_pool

        # Train the model
        train_subset = Subset(train_dataset, train_indices)
        train_loader_subset = DataLoader(train_subset, batch_size=32, shuffle=True)

        model, train_acc = train_labeled_model(model, optimizer, train_loader_subset, train_loader, EPOCHS)
        test_acc = evaluate_model(model, test_loader)

        train_accuracies.append((len(labeled_pool), train_acc))
        test_accuracies.append((len(labeled_pool), test_acc))

        # Acquire new samples randomly
        acquired_samples, unlabeled_pool = select_random_indices(unlabeled_pool, BATCH_SIZE)
        labeled_pool = np.concatenate((labeled_pool, acquired_samples))

        print(f"Labeled samples: {len(labeled_pool)} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    return train_accuracies, test_accuracies

def plot_learning_curve(accuracies, title="Learning Curve"):
    """
    Plot the learning curve for active learning.

    Args:
        accuracies (list): Accuracies over trials.
        title (str): Title for the plot.
    """
    num_samples = [x[0] for x in accuracies]
    mean_accuracies = [np.mean(x[1]) for x in accuracies]
    std_accuracies = [np.std(x[1]) for x in accuracies]

    plt.figure(figsize=(10, 6))
    plt.plot(num_samples, mean_accuracies, label="Mean Accuracy")
    plt.fill_between(
        num_samples,
        np.array(mean_accuracies) - np.array(std_accuracies),
        np.array(mean_accuracies) + np.array(std_accuracies),
        alpha=0.2,
        label="Confidence Interval"
    )
    plt.xlabel("Number of Labeled Examples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load data
    train_loader, test_loader = get_mnist_loaders(batch_size=64, eval_batch_size=1024)

    # Run active learning trials
    all_train_accuracies = []
    all_test_accuracies = []

    for trial in range(NUM_TRIALS):
        seed = 42 + trial
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, optimizer = create_model_optimizer(seed)
        train_acc, test_acc = active_learning_random_acquisition(model, optimizer, train_loader, test_loader)
        all_train_accuracies.append(train_acc)
        all_test_accuracies.append(test_acc)

    # Plot learning curves
    plot_learning_curve(all_test_accuracies, "Random Acquisition Learning Curve")
