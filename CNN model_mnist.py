import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from collections import Counter
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

## Paths
train_path = r".\mnist_png\training"
test_path = r".\mnist_png\testing"
actualtest_path = r".\mnist_png\actualtest"
result_file_path = f"mnist_result.txt"

## Parameters
total_classes = 10
folder_level = 1
num_epochs = 20

## Mode change
mode = "test"

## Loading data
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, folder_level=1):

        self.img_dir = img_dir
        self.transform = transform
        
        self.img_files = []
        self.labels = []
        

        if folder_level == 1:
            class_names = sorted(os.listdir(img_dir))
            for label, class_name in enumerate(class_names):
                class_dir = os.path.join(img_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.png'):
                            self.img_files.append(os.path.join(class_dir, img_name))
                            self.labels.append(label)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)

        return image, label


# transformations
# mnist
mean = [0.1307]
std = [0.3081]

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
                
# datasets, dataloaders
train_dataset = CustomImageDataset(img_dir=train_path, transform=train_transform, folder_level=folder_level)
test_dataset = CustomImageDataset(img_dir=test_path, transform=test_transform, folder_level=folder_level)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

#device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition (smaller ResNet for MNIST)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = self.relu(out)
        return out


class CustomResNet(nn.Module):
    """A small ResNet suitable for MNIST (grayscale 28x28).

    This uses BasicBlock and a much smaller channel count than the CIFAR/Imagenet-style
    bottleneck version in the original file.
    """
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()

        # initial conv: keep stride=1 so 28x28 stays 28x28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # layers: each layer consists of a small number of BasicBlocks
        self.layer1 = self._make_layer(16, 16, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(16, 32, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(32, 64, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Model, Loss, Optimizer, Scheduler
    custom_resnet_model = CustomResNet(total_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(custom_resnet_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # Training loop
    if mode == "train":
        best_val_accuracy = 0.0
        for epoch in range(num_epochs):
            custom_resnet_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for step, (images, labels) in enumerate(train_loader, 1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = custom_resnet_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            scheduler.step()
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100. * correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

            # Validation  
            custom_resnet_model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = custom_resnet_model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss /= len(test_loader)
            val_accuracy = 100. * correct / total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            # save
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(custom_resnet_model.state_dict(), "mnist_best_model.pth")
                print(f"New best accuracy: {best_val_accuracy:.2f}%, model saved.")
            else:
                print(f"No improvement (Best: {best_val_accuracy:.2f}%)")

    elif mode == "test":
        custom_resnet_model.load_state_dict(torch.load("mnist_best_model.pth"))
        custom_resnet_model.eval()

        class_names = sorted(os.listdir(train_path))

        predictions = []

        with torch.no_grad():
            for file_name in sorted(os.listdir(actualtest_path)):
                if not file_name.endswith(".png"):
                    continue

                file_path = os.path.join(actualtest_path, file_name)

                img = Image.open(file_path).convert("L")
                img = test_transform(img)
                img = img.unsqueeze(0).to(device)

                outputs = custom_resnet_model(img)
                _, predicted = outputs.max(1)
                predicted_idx = predicted.item()

                predicted_label = class_names[predicted_idx]

                img_id = os.path.splitext(file_name)[0]
                predictions.append((img_id, predicted_label))

        # 寫入結果
        with open(result_file_path, "w") as f:
            for img_id, label in predictions:
                f.write(f"{img_id} {label}\n")

        print(f"Predictions saved to {result_file_path}")



    
