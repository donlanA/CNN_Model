import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

## Paths
# train_path = r".\mnist_png\training"
# test_path = r".\mnist_png\testing"
# actualtest_path = r".\mnist_png\actualtest"
# result_file_path = f"mnist_result.txt"

# train_path = r".\cifar10\train"
# test_path = r".\cifar10\test"
# actualtest_path = r".\cifar10\actualtest"
# result_file_path = f"cifar10_result.txt"

# train_path = r".\CIFAR100\TRAIN"
# test_path = r".\CIFAR100\TEST"
# actualtest_path = r".\CIFAR100\ACTUALTEST"
# result_file_path = f"cifar100_result.txt"

train_path = r".\CIFAR100_2levels\TRAIN"
test_path = r".\CIFAR100_2levels\TEST"
actualtest_path = r".\CIFAR100_2levels\ACTUALTEST"

result_file_path = f"cifar100_2levels_result.txt"

## Parameters
total_classes = 100
folder_level = 2

## Mode change
mode = "test"

## Loading data
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, folder_level=1):
        """
        folder_level=1 -> one level folder: img_dir/class_name/*.png
        folder_level=2 -> two levels folder: img_dir/super_class/sub_class/*.png
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = []
        self.labels = []
        self.class_names = []

        if folder_level == 1:
            self.class_names = sorted([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])
            class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

            for class_name in self.class_names:
                class_path = os.path.join(img_dir, class_name)
                for img_name in os.listdir(class_path):
                    if img_name.endswith('.png'):
                        self.img_files.append(os.path.join(class_path, img_name))
                        self.labels.append(class_to_idx[class_name])

        elif folder_level == 2:
            super_classes = sorted(os.listdir(img_dir))
            subclass_set = set()

            for super_class in super_classes:
                super_path = os.path.join(img_dir, super_class)
                if not os.path.isdir(super_path):
                    continue
                for sub_class in sorted(os.listdir(super_path)):
                    sub_path = os.path.join(super_path, sub_class)
                    if os.path.isdir(sub_path):
                        subclass_set.add(sub_class)

            self.class_names = sorted(list(subclass_set))
            class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

            for super_class in super_classes:
                super_path = os.path.join(img_dir, super_class)
                if not os.path.isdir(super_path):
                    continue
                for sub_class in sorted(os.listdir(super_path)):
                    sub_path = os.path.join(super_path, sub_class)
                    if os.path.isdir(sub_path):
                        label = class_to_idx[sub_class]
                        for img_name in os.listdir(sub_path):
                            if img_name.endswith('.png'):
                                self.img_files.append(os.path.join(sub_path, img_name))
                                self.labels.append(label)
        else:
            raise ValueError("folder_level must be 1 or 2")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


# transformations
# cifar10
# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2023, 0.1994, 0.2010]
# cifar100
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
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

# Model definition
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        ) 
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)
        return out

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(ResidualBlock, 64, 2)
        
        self.layer2_down = DownsampleBlock(64, 128)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2)
        self.layer3_down = DownsampleBlock(128, 256)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2)
        self.layer4_down = DownsampleBlock(256, 512)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, channels, num_blocks):
        layers = [block(channels) for _ in range(num_blocks)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2_down(x)
        x = self.layer2(x)
        x = self.layer3_down(x)
        x = self.layer3(x)
        x = self.layer4_down(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Model, Loss, Optimizer, Scheduler
    custom_resnet_model = CustomCNN(total_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(custom_resnet_model.parameters(), lr=0.05, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # Training loop
    if mode == "train":
        num_epochs = 10
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

            if (epoch + 1) % 10 == 0:
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
                    torch.save(custom_resnet_model.state_dict(), "cifar100_best_model.pth")
                    print(f"New best accuracy: {best_val_accuracy:.2f}%, model saved.")
                else:
                    print(f"No improvement (Best: {best_val_accuracy:.2f}%)")

    elif mode == "test":
        custom_resnet_model.load_state_dict(torch.load("cifar100_best_model.pth"))
        custom_resnet_model.eval()

        predictions = []

        #  class_names in train_dataset
        class_names = train_dataset.class_names

        with torch.no_grad():
            for file_name in os.listdir(actualtest_path):
                file_path = os.path.join(actualtest_path, file_name)
                if not file_name.endswith(".png"):
                    continue

                img = Image.open(file_path).convert("RGB")
                img = test_transform(img)
                img = img.unsqueeze(0).to(device)

                outputs = custom_resnet_model(img)
                _, predicted = outputs.max(1)
                predicted_idx = predicted.item()

                predicted_label = class_names[predicted_idx]

                img_id = os.path.splitext(file_name)[0]

                predictions.append((img_id, predicted_label))

        with open(result_file_path, "w") as f:
            for img_id, label in predictions:
                f.write(f"{img_id} {label}\n")

        print(f"Predictions saved to {result_file_path}")


    
