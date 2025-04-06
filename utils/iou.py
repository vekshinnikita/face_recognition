

from typing import Any, List
import torch


def calculate_rectangle_area(x1, y1, x2, y2):
    """
    Вычисляет площадь прямоугольника по координатам двух диагональных точек.

    Args:
        x1: x-координата первой точки.
        y1: y-координата первой точки.
        x2: x-координата второй точки.
        y2: y-координата второй точки.

    Returns:
        Площадь прямоугольника.
    """
    length = abs(x2 - x1)
    width = abs(y2 - y1)
    area = length * width
    return area


def iou(box1, box2):
    """
    Ограничивающие рамки в формате (x1, y1, x2, y2).
    
    Parameters:
        box1: torch.Tensor, size (4,) 
        box2: torch.Tensor, size (4,)
    
    Returns:
        iou: torch.Tensor(scalar)
    """

    b1x1, b1y1, b1x2, b1y2 = box1
    b2x1, b2y1, b2x2, b2y2 = box2
    
    areal = calculate_rectangle_area(b1x1, b1y1, b1x2, b1y2)
    area2 = calculate_rectangle_area(b2x1, b2y1, b2x2, b2y2)
    
    x_left = torch.max(b1x1, b2x1)
    y_top = torch.max(b1y1, b2y1)
    x_right = torch.min (b1x2, b2x2)
    y_bottom = torch.min(b1y2, b2y2)
    
    if x_right < x_left or y_bottom < y_top:
        return torch.tensor(0, dtype=torch.float)
    
    w = x_right - x_left
    h = y_bottom - y_top
    
    inter = w*h
    union = areal + area2 - inter
    
    if union == 0:
        return 0
    
    iou = inter / union
    
    return iou

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
        Ограничивающие рамки в формате (x1, у1, x2, у2).
        
        Parameters:
            boxes1: torch. Tensor, size (N, 4) 
            boxes2: torch. Tensor, size (M, 4)
        
        Returns:
            iou: torch. Tensor, size (N, M)
    """

    area1 = (boxes1[:, 2] - boxes1 [:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # size (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # size (N, M, 2)
    
    wh = (rb - lt).clamp(min=0) # size (N, M, 2)
    
    inter = wh[..., 0] * wh[..., 1] # size (N, M)
    union = area1[:, None] + area2 - inter # size (N, M)
    
    iou = inter / (union + 1e-9) # size (N, M)
    return iou
    