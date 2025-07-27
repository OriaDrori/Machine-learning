import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from collections import Counter

# load dataset and apply transformations
def load_data(data_dir, train_ratio=0.7, val_ratio=0.1):
    # define image preprocessing, resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # split dataset into train, validation, and test
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, dataset.classes, dataset

# compute class weights to handle class imbalance
def calculate_class_weights(dataset, device):
    class_counts = Counter(dataset.targets)
    total_samples = len(dataset)
    num_classes = len(dataset.classes)

    weights = []
    for i in range(num_classes):
        weight = total_samples / (num_classes * class_counts[i])
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    return weights_tensor

# train the model over several epochs and record loss values
def train_cnn_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        # training loop
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

# evaluate model performance on the test set
def evaluate_cnn_model(model, data_loader, device, train_losses, val_losses):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # calculate evaluation metrics
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("CNN model metrics on testing data:")
    print(f" - Accuracy: {accuracy:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall: {recall:.4f}")
    print(f" - F1 Score: {f1:.4f}")

    # compare CNN to KNN, SVM, and RF using bar chart
    knn_metrics = [0.4314, 0.4898, 0.4282, 0.3631]
    svm_metrics = [0.6312, 0.5681, 0.6378, 0.5897]
    rf_metrics = [0.6359, 0.6954, 0.4778, 0.4997]
    cnn_metrics = [accuracy, precision, recall, f1]
    labels_text = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(labels_text))
    width = 0.2

    cm = confusion_matrix(all_labels, all_preds)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(c) for c in sorted(set(all_labels))])
    disp.plot(ax=axes[0], cmap='Reds', colorbar=False)
    axes[0].set_title("Confusion Matrix - CNN")

    # bar chart comparison of models
    axes[1].bar(x - 1.5 * width, knn_metrics, width, label='KNN', color='lightskyblue')
    axes[1].bar(x - 0.5 * width, svm_metrics, width, label='SVM', color='yellowgreen')
    axes[1].bar(x + 0.5 * width, rf_metrics, width, label='RF', color='darkorange')
    axes[1].bar(x + 1.5 * width, cnn_metrics, width, label='CNN', color='lightcoral')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_text)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Model Comparison")
    axes[1].legend()

    # plot training and validation losses
    axes[2].plot(train_losses, label='Train Loss', color='lightcoral')
    axes[2].plot(val_losses, label='Validation Loss', color='firebrick')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Train vs Validation Loss")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    return accuracy, precision, recall, f1

def main():
    data_dir = r"C:\deep learning\weather_status"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data and class names
    train_loader, val_loader, test_loader, class_names, full_dataset = load_data(data_dir)

    # compute class weights
    weights_tensor = calculate_class_weights(full_dataset, device)

    # load pretrained ResNet18 model and freeze its layers
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    # replace the final layer to fit our classification task (5 classes)
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # define loss, optimizer and learning rate scheduler
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # train and evaluate the model
    train_losses, val_losses = train_cnn_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
    evaluate_cnn_model(model, test_loader, device, train_losses, val_losses)

if __name__ == "__main__":
    main()
