import torch
import torch.nn as nn
import torch.nn.functional as F

info_string = """
class FaceLocalizationModel(nn.Module):
    def __init__(self):
        super(FaceLocalizationModel, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) # Output: (48, 48) -> Pooling -> (24, 24)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Output: (24, 24) -> Pooling -> (12, 12)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Output: (12, 12) -> Pooling -> (6, 6)

        # Define pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Input size adjusted for 48x48 input
        self.fc2 = nn.Linear(128, 64)
        self.fc_bbox = nn.Linear(64, 4)  # Bounding box coordinates (x1, y1, x2, y2)
        self.fc_confidence = nn.Linear(64, 1)  # Confidence score

        # Dropout for regularization (optional)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the feature maps
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 64 * 6 * 6)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Optional Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x) # Optional Dropout

        # Output bounding box coordinates and confidence
        bbox_coords = self.fc_bbox(x)  # (batch_size, 4)
        confidence = torch.sigmoid(self.fc_confidence(x))  # (batch_size, 1), sigmoid for confidence

        return bbox_coords, confidence
        
model = FaceLocalizationModel()
"""

class FaceLocalizationModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()

        # 1. CONV BLOCK 1 (Downsample, Feature Extraction) - Output: 96x96 -> 48x48
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding='same', bias=False)  # More channels
        self.bn1 = nn.BatchNorm2d(32)

        # 2. CONV BLOCK 2 (Downsample, Feature Extraction) - Output: 48x48 -> 24x24
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same', bias=False)  # More channels
        self.bn2 = nn.BatchNorm2d(64)

        # 3. CONV BLOCK 3 (Downsample, Feature Extraction) - Output: 24x24 -> 12x12
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False)  # More channels
        self.bn3 = nn.BatchNorm2d(128)

        # 4. FEATURE EXTRACTION (No Downsample) - Output: 12x12
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding='same', groups=16, bias=False) # Grouped Conv, more efficient
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding='same', groups=16, bias=False) # Grouped Conv, more efficient
        self.bn5 = nn.BatchNorm2d(128)

        # 5. CONV BLOCK 4 (Downsample) - Output 12x12 -> 6x6
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False)
        self.bn6 = nn.BatchNorm2d(128)

        # Adjusting output size for the fully connected layers (6x6 after 4 max pools)

        self.fc1 = nn.Linear(128 * 6 * 6, 256, bias=False)  # More neurons, no bias
        self.bn7 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128, bias=False) # More neurons, no bias
        self.bn8 = nn.BatchNorm1d(128)
        self.fc_bbox = nn.Linear(128, 4)
        self.fc_confidence = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout_rate) # Reduced Dropout

        self.sig = nn.Sigmoid()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True) # inplace=True saves memory

        # Initialize weights (Optional) - Kaiming He
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # Batchnorm layers have learnable parameters, so we don't need to initialize them

    def forward(self, x):
        # 1. Convolutional blocks with Max Pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # conv1 -> bn1 -> relu -> pool (48x48)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # conv2 -> bn2 -> relu -> pool (24x24)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # conv3 -> bn3 -> relu -> pool (12x12)

        # 2.  Deeper Feature Extraction Blocks
        x = self.relu(self.bn4(self.conv4(x))) # conv4 -> bn4 -> relu (12x12)
        x = self.relu(self.bn5(self.conv5(x))) # conv5 -> bn5 -> relu (12x12)

        # 3. Add final pooling before flatten to have 6x6 feature map (12x12 -> 6x6)
        x = self.pool(self.relu(self.bn6(self.conv6(x))))

        # 4. Flatten and Fully Connected Layers
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.bn7(self.fc1(x))))
        x = self.dropout(self.relu(self.bn8(self.fc2(x))))

        # 5. Output layers
        bbox_coords = self.fc_bbox(x) # No activation
        confidence = self.sig(self.fc_confidence(x)) # Sigmoid for confidence score

        return bbox_coords, confidence