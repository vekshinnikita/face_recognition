import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        """
        Инициализация ArcFace Loss.

        Args:
            in_features (int): Размерность входных признаков (embedding).
            out_features (int): Количество классов (количество идентификаций лиц).
            s (float): Scale factor.
            m (float): Margin.
            easy_margin (bool): Использовать ли easy margin.
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # Инициализация весов
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        """
        Прямой проход через ArcFace Loss.

        Args:
            input (torch.Tensor): Входные признаки (embedding) размера (batch_size, in_features).
            label (torch.Tensor): Метки классов размера (batch_size).

        Returns:
            torch.Tensor: Loss.
        """
        # Нормализация признаков и весов
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = F.cross_entropy(output, label)  # Используем CrossEntropyLoss
        return loss