import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformasyonlar
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Veri setini yükle
train_dataset = datasets.ImageFolder(root='Bean_Dataset', transform=transform)
valid_dataset = datasets.ImageFolder(root='Bean_Dataset', transform=transform)

# Veri setindeki sınıf isimlerini ve sayısını kontrol et
num_classes = len(train_dataset.classes)
print(f"Number of classes in the training set: {num_classes}")
print(f"Classes: {train_dataset.classes}")

# Veri yükleyiciler
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Sınıf dağılımlarını kontrol et
train_class_counts = [0] * num_classes
for _, label in train_dataset:
    train_class_counts[label] += 1
print(f"Training set class distribution: {train_class_counts}")

valid_class_counts = [0] * num_classes
for _, label in valid_dataset:
    valid_class_counts[label] += 1
print(f"Validation set class distribution: {valid_class_counts}")

# Konvolüsyon bloğu 
def ConvBlock(in_channels, out_channels, pool=False): 
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
               nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    if pool: 
        layers.append(nn.MaxPool2d(2)) # 2x2 Pooling 
    return nn.Sequential(*layers)

# Model tanımı
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Model ve hiperparametreler
model = ResNet9(in_channels=3, num_classes=num_classes)

# Eğitim parametreleri ve kayıp fonksiyonu
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim döngüsü
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # Doğrulama döngüsü
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Loss: {val_loss/len(valid_loader)}, Accuracy: {100 * correct / total}%')

# Modeli kaydet
torch.save(model.state_dict(), 'resnet9_model.pth')
print("Model kaydedildi!")

print("Eğitim tamamlandı!")
