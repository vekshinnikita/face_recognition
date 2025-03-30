import torch
from typing import Iterable

def transform_bbox(
    bbox: Iterable, 
    original_image_size: Iterable,
    new_image_size: Iterable,
  ):
    # Преобразование bbox
    original_height, original_width = original_image_size
    new_height, new_width = new_image_size

    x_min, y_min, x_max, y_max = bbox

    # Масштабирование bbox
    x_min = int((x_min / original_width) * new_width)
    y_min = int((y_min / original_height) * new_height)
    x_max = int((x_max / original_width) * new_width)
    y_max = int((y_max / original_height) * new_height)

    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
  
def reverse_transform_decorator(reverse_transform_image,  image_size: Iterable):
    
    def wrapper(image, bbox):
      if reverse_transform_image is not None:
        new_image = reverse_transform_image(image, image_size)
      else:
        new_image = image
      new_bbox = transform_bbox(
        bbox, 
        image.shape[1:],
        new_image.shape[1:],
      )
      
      return new_image, new_bbox
    
    return wrapper