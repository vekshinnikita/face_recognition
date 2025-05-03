import random
import string
import torch
from utils.iou import box_iou, iou
from utils.system import ensure_directory_exists_os

try:
    from google.colab import files as google_files
    RUNNING_IN_COLAB = True
except ImportError:
    google_files = None
    RUNNING_IN_COLAB = False
    

def generate_sequence(length):
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choice(alphabet) for _ in range(length))

def save_model(
    file_path: str,
    model,
    opt,
    start_datetime,
    lr_scheduler,
    train_loss,
    val_loss,
    best_loss,
    train_precision,
    val_precision,
    train_recall,
    train_f1,
    val_recall,
    val_f1,
    train_iou_avg,
    val_iou_avg,
    lr_list,
    EPOCHS,
    epoch,
    str_info: str = ''
):
    checkpoint = {
        'info': str_info,
        'start_datetime': start_datetime,
        'state_model': model.state_dict(),
        'state_opt': opt.state_dict(),
        'state_lr_scheduler': lr_scheduler.state_dict(),
        'loss': {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': best_loss
        },
        
        'metric': {
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'train_iou_avg': train_iou_avg,
            'val_iou_avg': val_iou_avg
        },
        'lr_list': lr_list,
        'epoch': {
            'EPOCHS' : EPOCHS,
            'save_epoch': epoch
        }
    }
    
    file_path = file_path.format(epoch=epoch, start_datetime=start_datetime)
    
    ensure_directory_exists_os(file_path)
    
    torch.save(checkpoint, file_path)
    
    if RUNNING_IN_COLAB:
        google_files.download(file_path)
    
def calc_batch_metrics(
    pred_bboxes, 
    target_bboxes, 
    pred_confidences, 
    target_confidences, 
    iou_threshold = 0.5
):
    iou_batch_sum = 0
    num_predicted_bbox = 0
    
    tp = 0
    fp = 0
    fn = 0
    
    for pred_bbox, target_bbox, pred_conf, target_conf in zip(
        pred_bboxes, 
        target_bboxes,
        pred_confidences, 
        target_confidences
    ):
        pred_conf = 1 if pred_conf >= 0.5 else 0
        if pred_conf == target_conf and target_conf == 1:
            iou_value = iou(pred_bbox, target_bbox)
            iou_batch_sum += iou_value.item()
            num_predicted_bbox += 1
            
            if iou_value > iou_threshold:
                tp += 1
            else:
                fp += 1
                
        elif pred_conf == target_conf and target_conf == 0:
            tp += 1
            
        else:
            fn += 1

    return torch.tensor([
        tp,
        fp,
        fn,
        iou_batch_sum,
        num_predicted_bbox
    ], dtype=torch.float32)
    
def calc_metrics(
    tp,
    fp,
    fn,
    iou_batch_sum,
    num_predicted_bbox
):
    precision = tp / (tp + fp)
    
    recall = tp / (tp + fn)
    
    f1 = 2 * (precision * recall) / (precision + recall) 
    
    iou_avg = iou_batch_sum/num_predicted_bbox
    
    return precision, recall, f1, iou_avg
    
    
    
    