import os
import json


import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import v2

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

    self.len_dataset = 0
    self.data_list = self._get_data_list(self.images_path)

    
  def _validate_image(self, image_path):
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
  
  def __getitem__(self, index):
    image_path = self.data_list[index]
    
    label_path = self._get_annotation_path(image_path)
    
    with open(label_path, 'r') as f:
      label = json.loads(f.read(), strict=False)

    original_bbox = label['annotations'][0]['bbox'] if label['annotations'] else [0,0,0,0]
    original_image = Image.open(image_path)
    
    new_image, new_bbox = self.transform(original_image, original_bbox)
    
    return new_image, new_bbox, original_image.size[::-1]
  
  def __len__(self):
    return len(self.data_list)