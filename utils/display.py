from typing import Any, Iterable
import torch
from torchvision.transforms import v2

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_image_with_bboxes(
    image: torch.Tensor | Any, 
    bboxes: torch.Tensor | Any = None
):
    if bboxes is None:
        bboxes = []
        
    if isinstance(image, torch.Tensor):
        image = v2.ToPILImage()(image)
    if isinstance(bboxes, torch.Tensor):    
        bboxes = bboxes.detach().numpy()
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox in bboxes:
        # Добавляем ограничивающую рамку.
        rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()