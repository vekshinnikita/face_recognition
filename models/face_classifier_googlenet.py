import torchvision.models as models
import torch.nn as nn

info_string = '''
class FaceGoogLeNet(nn.Module):
    
    
    def __init__(self, embedding_size = 128, pretrained=True, freeze_features=True):
        """
        Инициализация модели FaceGoogLeNet.

        Args:
            embedding_size (int): Размерность embedding лица.
            pretrained (bool): Использовать ли предобученную модель GoogLeNet.
            freeze_features (bool): Заморозить ли параметры сверточных слоев.
        """
        super(FaceGoogLeNet, self).__init__()

        
        # Загрузка предобученной модели GoogLeNet
        self.googlenet = models.googlenet(pretrained=pretrained)

        # Заморозка параметров сверточных слоев (опционально)
        if freeze_features:
            for param in self.googlenet.parameters():
                param.requires_grad = False

        # Замена последнего полносвязного слоя
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, embedding_size)

        # Добавление L2-нормализации (важно для функций потерь, таких как Triplet Loss)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        """
        Прямой проход через модель.

        Args:
            x (torch.Tensor): Входной тензор (изображение).

        Returns:
            torch.Tensor: Embedding лица.
        """
        x = self.googlenet(x)
        x = self.layer_norm(x)  # Нормализация после embedding слоя
        return x
        
model = FaceGoogLeNet(embedding_size=128, pretrained=True)
'''

class FaceGoogLeNet(nn.Module):
    
    
    def __init__(self, embedding_size = 128, pretrained=True, freeze_features=False):
        """
        Инициализация модели FaceGoogLeNet.

        Args:
            embedding_size (int): Размерность embedding лица.
            pretrained (bool): Использовать ли предобученную модель GoogLeNet.
            freeze_features (bool): Заморозить ли параметры сверточных слоев.
        """
        super(FaceGoogLeNet, self).__init__()

        
        # Загрузка предобученной модели GoogLeNet
        self.googlenet = models.googlenet(pretrained=pretrained)

        # Заморозка параметров сверточных слоев (опционально)
        if freeze_features:
            for param in self.googlenet.parameters():
                param.requires_grad = False

        # Замена последнего полносвязного слоя
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, embedding_size)

        # Добавление L2-нормализации (важно для функций потерь, таких как Triplet Loss)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        """
        Прямой проход через модель.

        Args:
            x (torch.Tensor): Входной тензор (изображение).

        Returns:
            torch.Tensor: Embedding лица.
        """
        x = self.googlenet(x)
        x = self.layer_norm(x)  # Нормализация после embedding слоя
        return x