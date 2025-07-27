import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# load and preprocess data
def load_data(data_dir, train_ratio=0.7, val_ratio=0.1):
    # define image transformations, resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    total_size = len(full_dataset)

    # calculate sizes for training, validation, and testing sets
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # randomly split the dataset into train, validation, and test
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # create DataLoaders for batch loading
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset.classes


# extract flat pixel data from loader
def extract_data(loader):
    images = []
    labels = []
    for batch in loader:
        X, y = batch
        # flatten image tensors and convert to numpy arrays
        X = X.view(X.size(0), -1).numpy()
        images.extend(X)
        labels.extend(y.numpy())
    return np.array(images), np.array(labels)


# apply PCA for dimensionality reduction
def apply_pca(X_train, X_val, X_test, n_components=100):
    # use PCA to reduce dimensionality of the feature space
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca


# train SVM model
def train_svm_model(X_train, y_train):
    # train a Support Vector Machine with RBF kernel
    model = SVC(kernel='rbf', class_weight='balanced')
    model.fit(X_train, y_train)
    return model


# evaluate and visualize results
def evaluate_svm_model(model, X_test, y_test, class_names):
    # predict the test labels
    y_pred = model.predict(X_test)

    # calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # print metrics
    print("SVM model metrics on testing data:")
    print(f" - Accuracy: {accuracy:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall: {recall:.4f}")
    print(f" - F1 Score: {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # confusion Matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0], cmap='Greens', colorbar=False)
    axes[0].set_title("Confusion Matrix")

    # comparison bar chart vs KNN
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    knn_metrics = [0.4134, 0.4898, 0.4282, 0.3631]
    svm_metrics = [accuracy, precision, recall, f1]
    x = np.arange(len(labels))
    width = 0.35

    axes[1].bar(x - width/2, knn_metrics, width, label='KNN', color='lightskyblue')
    axes[1].bar(x + width/2, svm_metrics, width, label='SVM', color='yellowgreen')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Score")
    axes[1].set_title("KNN vs SVM Performance")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return accuracy, precision, recall, f1

def main():
    data_dir = r"C:\deep learning\weather status\data"

    # load dataset and split into train, validation, test
    train_loader, val_loader, test_loader, class_names = load_data(data_dir)

    # extract flat feature arrays and labels
    X_train, y_train = extract_data(train_loader)
    X_val, y_val = extract_data(val_loader)
    X_test, y_test = extract_data(test_loader)

    # normalize image pixel values to range [0,1]
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    # apply PCA to reduce dimensionality to 100 components
    X_train_pca, X_val_pca, X_test_pca = apply_pca(X_train, X_val, X_test, n_components=100)

    # train SVM model on training data
    svm_model = train_svm_model(X_train_pca, y_train)

    # evaluate model on test data
    evaluate_svm_model(svm_model, X_test_pca, y_test, class_names)

if __name__ == "__main__":
    main()
