import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from io import BytesIO

# Konvolüsyon bloğu 
def ConvBlock(in_channels, out_channels, pool=False): 
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    if pool: 
        layers.append(nn.MaxPool2d(2))  # 2x2 Pooling
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

# Modeli yükle
model = ResNet9(in_channels=3, num_classes=3)
model.load_state_dict(torch.load('resnet9_model.pth', map_location=torch.device('cpu')))
model.eval()

# Sınıf isimleri
class_names = ['angular_leaf_spot', 'bean_rust', 'healthy']

# Görüntü ön işleme fonksiyonu
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)  # Batch boyutuna dönüştür
    return img

# PNG'yi JPG'ye çeviren yardımcı fonksiyon
def convert_png_to_jpg(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    output = BytesIO()
    img.save(output, format='JPEG')
    return Image.open(output)

# Tahmin yapma fonksiyonu
def predict_image(img):
    img = preprocess_image(np.array(img))
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

# Streamlit arayüzü
st.title("Fasulye Hastalığı Tespit Uygulaması")

camera_input = st.camera_input('Kameradan resim çek')
gallery_input = st.file_uploader('VEYA Fasulye Fotoğrafı Ekleyin', accept_multiple_files=False)

if camera_input is not None:
    img_bytes = camera_input.getvalue()
    img = Image.open(BytesIO(img_bytes))
    # Resmi PNG'den JPG'ye çevir
    if img.format == 'PNG':
        img = convert_png_to_jpg(img)

    predicted_class, confidence = predict_image(img)
    st.write(f"Tahmin Edilen Sınıf: {class_names[predicted_class]}")
    st.write(f"İnanılırlık Yüzdesi: {confidence*100:.2f}%")

elif gallery_input is not None:
    img_bytes = gallery_input.getvalue()
    img = Image.open(BytesIO(img_bytes))
    # Resmi PNG'den JPG'ye çevir
    if img.format == 'PNG':
        img = convert_png_to_jpg(img)

    predicted_class, confidence = predict_image(img)
    st.write(f"Tahmin Edilen Sınıf: {class_names[predicted_class]}")
    st.write(f"İnanılırlık Yüzdesi: {confidence*100:.2f}%")

else:
    st.write("Lütfen bir resim yükleyin veya kamera kullanarak bir resim çekin.")
