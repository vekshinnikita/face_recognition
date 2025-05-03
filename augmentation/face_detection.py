import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from PIL import Image
import numpy as np

class FaceDetectorAugmentation:
    def __init__(self, rotate_degrees=(30,50), scale_range=(0.5, 1.5), translate_range=(0.2, 0.2)):
        self.rotate_degrees = rotate_degrees
        self.scale_range = scale_range
        self.translate_range = translate_range

        self.transform = A.Compose([
            A.Affine(
                rotate=self.rotate_degrees,
                scale=self.scale_range, 
                translate_percent=self.translate_range, 
                interpolation=cv2.INTER_LINEAR,
                p=1,
                keep_ratio=True,
            ),
            A.RandomBrightnessContrast(
                p=0.5
            ),
            A.GaussNoise(std_range = (0.01, 0.1), p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])) # Указываем формат BBox

    def __call__(self, image, bbox):
        # Albumentations требует входные данные в формате numpy
        np_image = np.array(image)
        # bbox - это список [x_min, y_min, x_max, y_max]
        # category_ids - список с идентификаторами категорий (для детекции лиц - 1)
        transformed = self.transform(image=np_image, mask= np_image, bboxes=[bbox], category_ids=[1])

        if transformed is None:
            return image, bbox

        transformed_image = transformed['image']  # Теперь это numpy array
        transformed_bboxes = transformed['bboxes']

        if len(transformed_bboxes) == 0: # Если BBox был обрезан
            return image, bbox

        transformed_bbox = transformed_bboxes[0]
        # Конвертируем bbox в torch.Tensor
        transformed_bbox = torch.tensor(transformed_bbox, dtype=torch.float32)

        # Преобразуем numpy array обратно в PIL Image
        transformed_image = Image.fromarray(transformed_image)

        return transformed_image, transformed_bbox