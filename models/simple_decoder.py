import torch
import torch.nn as nn

info_string = """
class SimpleFaceDetector(nn.Module):
    def __init__(self):
        super(SimpleFaceDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Пример для размера изображения 256x256 после двух max pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)  # 4 выхода: x1, y1, x2, y2 (координаты bounding box)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
        
model = SimpleFaceDetector()
"""

class SimpleFaceDetector(nn.Module):
    
    
    def __init__(self):
        super(SimpleFaceDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Пример для размера изображения 256x256 после двух max pooling
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_box = nn.Linear(128, 4)  # 4 выхода: x1, y1, x2, y2 (координаты bounding box)
        self.sig = nn.Sigmoid()
        self.fc_prob = nn.Linear(128, 1) # Вероятность наличия лица

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        
        box_coords = self.fc_box(x)
        face_probability = self.sig(self.fc_prob(x))  # Сигмоида для вероятности
        
        return box_coords, face_probability