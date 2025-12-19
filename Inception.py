from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

train = datasets.CIFAR10(root="./data", train=True, download=True,
                         transform=transform)

test = datasets.CIFAR10(root="./data", train=False, download=True,
                        transform=transform)

train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader  = DataLoader(test,  batch_size=64, shuffle=False)

img, label = train[0]
print("image dim", img.shape)
print("label:", label)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

class Inception(nn.Module):
  def __init__(self, in_depth):
    super().__init__()
    self.layer1 = nn.Sequential(nn.Conv2d(in_depth, 64, kernel_size=1, padding=0, stride=1))
    self.layer2 = nn.Sequential(nn.Conv2d(in_depth, 96, kernel_size=1, padding=0, stride=1), nn.Conv2d(in_depth, 128, kernel_size=5, padding=2, stride=1))
    self.layer3 = nn.Sequential(nn.Conv2d(in_depth, 96, kernel_size=1, padding=0, stride=1), nn.Conv2d(in_depth, 32, kernel_size=3, padding=1, stride=1))
    self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, padding=1, stride=1), nn.Conv2d(in_depth, 32, kernel_size=1, padding=0, stride=1))
  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer1(x)
    out3 = self.layer1(x)
    out4 = self.layer1(x)
    return torch.concat([out1, out2, out3, out4], dim=1)

class Main(nn.Module):
  def __init__(self):
    super().__init__()
    self.front = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 112 → 56
      nn.Conv2d(64, 64, kernel_size=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 192, kernel_size=3, padding=1),
      nn.BatchNorm2d(192),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.inception1 = Inception(192)
    self.inception2 = Inception(256)
    self.inception3 = Inception(256)
    self.fc = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),                  
      nn.Linear(256, 10)
    )
  def forward(self, x):
    x = self.front(x)
    x = self.inception3(self.inception2(self.inception1(x)))
    x = self.fc(x)
    return x

model = Main().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(3):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")
