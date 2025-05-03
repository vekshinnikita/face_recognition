import albumentations as A
from PIL import Image
import numpy as np

class FaceClassifierAugmentation:
    def __init__(self, rotate_degrees=(30,50)):
        self.rotate_degrees = rotate_degrees

        self.transform = A.Compose([
            A.Rotate(rotate_degrees, p=0.01),
            A.Blur(p=0.01, blur_limit=(3, 7)),
            A.RandomBrightnessContrast(
                p=0.01
            ),
            A.GaussNoise(std_range = (0.01, 0.1), p=0.01),
        ])

    def __call__(self, image):
        np_image = np.array(image)
        transformed = self.transform(image=np_image, mask=np_image, category_ids=[1])

        if transformed is None:
            return image

        transformed_image = transformed['image']

        return Image.fromarray(transformed_image)