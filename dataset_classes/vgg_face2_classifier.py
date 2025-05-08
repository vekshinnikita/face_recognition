from collections import defaultdict
import os
import json
import random
from typing import List, Tuple


import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import v2

from augmentation.face_classifier import FaceClassifierAugmentation
from utils.dict import traverse
from utils.system import get_last_dirname


class VGGFace2ClassifierDataset(Dataset):
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
  ):
    self.images_path = images_path
    self.labels_path = labels_path
    self.transform_image = transform_image
  

    self.path_by_index, self.lists_map = self._get_lists_map(self.images_path)

  def _get_lists_map(self, images_path):
    path_by_index = [] 
    lists_map = defaultdict(list)
    for path_dir, dir_list, file_list in os.walk(images_path):
      if path_dir == images_path:
        continue

      for file_path in file_list:
        image_path = os.path.join(path_dir, file_path)
        class_name = get_last_dirname(path_dir)
        
        if self._validate_image(image_path):
          
          class_list = lists_map[class_name]
          class_list.append(image_path)
          
          path_by_index.append((class_name, len(class_list)-1))
          
    return path_by_index, lists_map
  
  def _get_image_path_by_index(self, index):
    path = self.path_by_index[index]
    return traverse(self.lists_map, *path)
  
  def _get_random_positive_image_path(self, index):
    path = self.path_by_index[index]
    class_name = path[0]
    exclude_index = path[1]
    
    class_list = self.lists_map[class_name]
    positive_index = random.randint(0, len(class_list)-1)
    if positive_index == exclude_index:
      positive_index = positive_index - 1 if positive_index == len(class_list)-1 else positive_index + 1
    
    return class_list[positive_index]
    
  def _get_random_negative_image_path(self, index):
    class_name = self.path_by_index[index][0]
    
    filtered_class_names = [ 
      cn
      for cn in self.lists_map.keys()
      if class_name != cn
    ]
    
    negative_class_name = random.choice(filtered_class_names)
    return random.choice(self.lists_map[negative_class_name])

  def _validate_image(self, image_path):
    if '.jpg' not in image_path:
      return False
    
    label_path = self._get_annotation_path(image_path)
    return os.path.exists(label_path)
  
  def _get_annotation_path(self, image_path):
    relative_label_path = os.path.sep.join(image_path.split(os.path.sep)[-2:]).replace('.jpg', '.json')
    return os.path.join(self.labels_path, relative_label_path)
  
  def _make_valid_bbox(self, bbox, height:int, width:int):
    bbox = np.array(bbox) 
    
    bbox[0] = float(np.clip(bbox[0], 0, width) )  # x_min
    bbox[1] = np.clip(bbox[1], 0, height)  # y_min
    bbox[2] = np.clip(bbox[2], 0, width)   # x_max
    bbox[3] = np.clip(bbox[3], 0, height)  # y_max
    
    return tuple(bbox.tolist())
  
  def _get_annotation_bbox(self, annotations: List, height:int, width:int) -> Tuple[float,float, float, float]:
    if len(annotations) == 0:
      return (0,0,0,0)
    
    return self._make_valid_bbox(annotations[0]['bbox'], height, width)
    
  def transform(self, image):
    if self.transform_image is not None:
      new_image = self.transform_image(image)
    else:
      new_image = self.transform_image_to_tensor(image)
    
    return new_image
  
  def _get_face_image(self, image_path):
    label_path = self._get_annotation_path(image_path)
    with open(label_path, 'r') as f:
      label = json.loads(f.read(), strict=False)
    
    image = Image.open(image_path)
    image_size = image.size[::-1]
    
    bbox = self._get_annotation_bbox(label['annotations'], image_size[0], image_size[1])
    
    cropped_image = image.crop(bbox)
    
    return self.transform(cropped_image)
    
  def __getitem__(self, index):
    anchor_path = self._get_image_path_by_index(index)
    positive_path = self._get_random_positive_image_path(index)
    negative_path = self._get_random_negative_image_path(index)
    
    anchor = self._get_face_image(anchor_path)
    positive = self._get_face_image(positive_path)
    negative = self._get_face_image(negative_path)
    
    return anchor, positive, negative
  
  def __len__(self):
    return len(self.path_by_index)
  
  
  
class VGGFace2ClassifierArcDataset(Dataset):
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
    
    self.augmentation = FaceClassifierAugmentation(
      rotate_degrees=(-70,70),
    )

    (
      self.class_idx_map, 
      self.targets, 
      self.path_by_index, 
      self.lists_map
     ) = self._get_lists_map(self.images_path)
    

  def _get_lists_map(self, images_path):
    targets = []
    class_idx_map = {}
    path_by_index = [] 
    lists_map = defaultdict(list)
    
    for path_dir, dir_list, file_list in os.walk(images_path):
      if path_dir == images_path:
        counter = 0
        for dir_name in dir_list:
          class_idx_map[dir_name] = counter
          counter += 1
        
        continue

      class_name = get_last_dirname(path_dir)
      class_idx = class_idx_map[class_name]
      class_list = lists_map[class_idx]
      for file_path in file_list:
        image_path = os.path.join(path_dir, file_path)
        if self._validate_image(image_path):
          class_list.append(image_path)
          path_by_index.append((class_idx, len(class_list)-1))
          targets.append(class_idx)
          
    return class_idx_map, targets, path_by_index, lists_map
  
  def _get_image_path_by_index(self, index):
    path = self.path_by_index[index]
    return traverse(self.lists_map, *path)
  

  def _validate_image(self, image_path):
    if '.jpg' not in image_path:
      return False
    
    label_path = self._get_annotation_path(image_path)
    return os.path.exists(label_path)
  
  def _get_annotation_path(self, image_path):
    relative_label_path = os.path.sep.join(image_path.split(os.path.sep)[-2:]).replace('.jpg', '.json')
    return os.path.join(self.labels_path, relative_label_path)
  
  def _make_valid_bbox(self, bbox, height:int, width:int):
    bbox = np.array(bbox) 
    
    bbox[0] = float(np.clip(bbox[0], 0, width) )  # x_min
    bbox[1] = np.clip(bbox[1], 0, height)  # y_min
    bbox[2] = np.clip(bbox[2], 0, width)   # x_max
    bbox[3] = np.clip(bbox[3], 0, height)  # y_max
    
    return tuple(bbox.tolist())
  
  def _get_annotation_bbox(self, annotations: List, height:int, width:int) -> Tuple[float,float, float, float]:
    if len(annotations) == 0:
      return (0,0,0,0)
    
    return self._make_valid_bbox(annotations[0]['bbox'], height, width)
    
  def transform(self, image):
    if self.transform_image is not None:
      new_image = self.transform_image(image)
    else:
      new_image = self.transform_image_to_tensor(image)
    
    return new_image
  
  def _get_face_image(self, image_path):
    label_path = self._get_annotation_path(image_path)
    with open(label_path, 'r') as f:
      label = json.loads(f.read(), strict=False)
    
    image = Image.open(image_path)
    image_size = image.size[::-1]
    
    bbox = self._get_annotation_bbox(label['annotations'], image_size[0], image_size[1])
    
    cropped_image = image.crop(bbox)
    augmented_image = self.augmentation(cropped_image)
    
    return self.transform(augmented_image)
    
  def __getitem__(self, index):
    anchor_path = self._get_image_path_by_index(index)
    anchor = self._get_face_image(anchor_path)
    target_class_idx = self.targets[index]
    
    return anchor, torch.tensor(target_class_idx, dtype=torch.long)
  
  def __len__(self):
    return len(self.path_by_index)