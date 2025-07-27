import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# load and preprocess data
def load_data(data_dir, train_ratio=0.7, val_ratio=0.1):
    # define a transform, resize images and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # load images from the folder with their labels
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    total_size = len(full_dataset)

    # split sizes for train, validation, and test
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # randomly split the dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # create DataLoaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset.classes


# extract data from DataLoader into numpy arrays
def extract_data(loader):
    images = []
    labels = []
    for batch in loader:
        X, y = batch
        # flatten images and convert to numpy
        X = X.view(X.size(0), -1).numpy()
        images.extend(X)
        labels.extend(y.numpy())
    return np.array(images), np.array(labels)


# train KNN model
def train_knn_model(X_train, y_train, k=5):
    # create and train a K Nearest Neighbors classifier
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


# evaluate and visualize
def evaluate_knn_model(model, X_test, y_test, class_names):
    # predict labels for test set
    y_pred = model.predict(X_test)

    # compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # print metrics
    print("KNN Model Metrics on Testing Data:")
    print(f" - Accuracy: {accuracy:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall: {recall:.4f}")
    print(f" - F1 Score: {f1:.4f}")

    # create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title("Confusion Matrix")

    # plot bar chart of metrics
    metrics = [accuracy, precision, recall, f1]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    axes[1].bar(labels, metrics, color='lightskyblue', alpha=0.7)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("KNN Metrics on Test Set")
    axes[1].set_ylabel("Score")

    plt.tight_layout()
    plt.show()

    return accuracy, precision, recall, f1

def main():
    data_dir = r"C:\deep learning\weather status\data"
    # load and split the dataset
    train_loader, val_loader, test_loader, class_names = load_data(data_dir)

    # convert image batches to numpy arrays
    X_train, y_train = extract_data(train_loader)
    X_val, y_val = extract_data(val_loader)
    X_test, y_test = extract_data(test_loader)

    # normalize pixel values to range [0,1]
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    # train KNN model with k=5
    knn_model = train_knn_model(X_train, y_train, k=5)

    # evaluate model performance
    evaluate_knn_model(knn_model, X_test, y_test, class_names)


if __name__ == "__main__":
    main()
