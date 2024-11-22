import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time  # Import time module to track epoch duration

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn thư mục
data_dir = 'CNN_MultiClass_data'

# Phép biến đổi hình ảnh
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize ảnh về kích thước 299x299
    transforms.ToTensor(),  # Chuyển ảnh về dạng tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Chuẩn hóa ảnh theo mean và std của ImageNet
])

# Tạo dataset và dataloader
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'animals'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Tạo mô hình CNN truyền thống
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Các lớp Convolutional và Pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Lớp Conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Lớp Conv2
        self.pool = nn.MaxPool2d(2, 2)  # Lớp Pooling
        self.fc1 = nn.Linear(64 * 74 * 74, 512)  # Lớp Fully Connected
        self.fc2 = nn.Linear(512, 3)  # Lớp Fully Connected cuối cùng với 3 lớp phân loại

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Convolution 1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Convolution 2 + ReLU + Pooling
        x = x.view(-1, 64 * 74 * 74)  # Chuyển đổi thành vector 1 chiều
        x = torch.relu(self.fc1(x))  # Fully Connected 1 + ReLU
        x = self.fc2(x)  # Fully Connected 2 (output)
        return x

# Khởi tạo mô hình
model = CNNModel().to(device)

# In kiến trúc của mô hình để kiểm tra
print('Model architecture')
print(model)

# Huấn luyện mô hình
def train_test_cnn(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()  # Sử dụng CrossEntropyLoss cho phân loại
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Sử dụng Adam optimizer

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    for epoch in range(epochs):
        start_time = time.time()  # Start timer for the epoch

        model.train()  # Chế độ huấn luyện
        train_loss, train_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)

        model.eval()  # Chế độ kiểm tra
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc.item())
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())

        # Calculate and print the time taken for this epoch
        end_time = time.time()  # End timer for the epoch
        epoch_time = end_time - start_time  # Calculate elapsed time for the epoch

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Epoch Time: {epoch_time:.2f} seconds")  # Print epoch time

    # Vẽ đồ thị
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()


# Huấn luyện và kiểm tra mô hình
train_test_cnn(model, train_loader, val_loader, epochs=10)
