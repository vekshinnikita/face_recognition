import os
import json
import random
from typing import Dict, Iterable, List


import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import v2

from augmentation.face_detection import FaceDetectorAugmentation
from utils.iou import calculate_rectangle_area
from utils.transform import transform_bbox


class VGGFace2DetectorDataset(Dataset):
  transform_image_to_tensor = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
  ])
  
  def __init__(
    self, 
    images_path, 
    labels_path, 
    transform_image=None, 
    reverse_transform_image=None
  ):
    self.images_path = images_path
    self.labels_path = labels_path
    self.transform_image = transform_image
    self.reverse_transform_image = reverse_transform_image
    
    self.augmentation = FaceDetectorAugmentation(
      rotate_degrees=(-70,70),
      scale_range=(0.2, 1),
      translate_range=(-0.4, 0.4),
    )

    self.len_dataset = 0
    self.data_list = self._get_data_list(self.images_path)

    
  def _validate_image(self, image_path):
    if '.jpg' not in image_path:
      return False
    
    label_path = self._get_annotation_path(image_path)
    return os.path.exists(label_path)
        
  def _get_data_list(self, images_path):
    data_list = []
    for path_dir, dir_list, file_list in os.walk(images_path):
      if path_dir == images_path:
        continue

      for file_path in file_list:
        image_path = os.path.join(path_dir, file_path)
        
        if self._validate_image(image_path):
          data_list.append(image_path)
      
    return data_list
    
  
  def transform(self, image, bbox):
    if self.transform_image is not None:
      new_image = self.transform_image(image)
    else:
      new_image = self.transform_image_to_tensor(image)
      
    new_bbox = transform_bbox(
      bbox, 
      image.size[::-1],
      new_image.shape[1:],
    )
    
    return new_image, new_bbox
  
  def _get_annotation_path(self, image_path):
    relative_label_path = os.path.sep.join(image_path.split(os.path.sep)[-2:]).replace('.jpg', '.json')
    return os.path.join(self.labels_path, relative_label_path)
  
  def _make_valid_bbox(self, bbox, height:int, width:int):
    bbox = np.array(bbox) 
    
    bbox[0] = np.clip(bbox[0], 0, width)   # x_min
    bbox[1] = np.clip(bbox[1], 0, height)  # y_min
    bbox[2] = np.clip(bbox[2], 0, width)   # x_max
    bbox[3] = np.clip(bbox[3], 0, height)  # y_max
    
    return bbox.tolist()
  
  def _get_annotation_largest_bbox(self, annotations: List, height:int, width:int):
    if len(annotations) == 0:
      return [0,0,0,0]
    
    if len(annotations) == 1:
      return self._make_valid_bbox(annotations[0]['bbox'], height, width)
    
    index = 0
    max_area = 0
    for idx, annotation in enumerate(annotations):
      area = calculate_rectangle_area(*annotation['bbox'])
      if area > max_area:
        index = idx
        max_area = area
    
    return self._make_valid_bbox(annotations[index]['bbox'], height, width)
    
  
  def __getitem__(self, index):
    image_path = self.data_list[index]
    
    label_path = self._get_annotation_path(image_path)
    
    with open(label_path, 'r') as f:
      label = json.loads(f.read(), strict=False)
    
    original_image = Image.open(image_path)
    image_size = original_image.size[::-1]
    original_bbox = self._get_annotation_largest_bbox(label['annotations'], image_size[0], image_size[1])
    
    confidence = 0 if original_bbox == [0,0,0,0] else 1
    
    augmented_image = original_image
    augmented_bbox = original_bbox
    if random.random() < 0.5 and confidence == 1:
      augmented_image, augmented_bbox = self.augmentation(original_image, original_bbox)
    
    new_image, new_bbox = self.transform(augmented_image, augmented_bbox)
    
    return new_image, new_bbox, torch.tensor([confidence], dtype=torch.float32), original_image.size[::-1]
  
  def __len__(self):
    return len(self.data_list)