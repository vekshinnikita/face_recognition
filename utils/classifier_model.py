import random
import string
import torch
from utils.iou import box_iou, iou
from utils.system import ensure_directory_exists_os
import torch.nn.functional as F

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
    train_accuracy,
    train_frr,
    train_far,
    val_accuracy,
    val_frr,
    val_far,
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
            'train_accuracy': train_accuracy,
            'train_frr': train_frr,
            'train_far': train_far,
            'val_accuracy': val_accuracy,
            'val_frr': val_frr,
            'val_far': val_far,
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


def evaluate_recognition_batch(anchor_embeddings, positive_embeddings, negative_embeddings, threshold=0.7):
    fp = 0
    fn = 0
    tp = 0
    tn = 0

    for anchor, positive in zip(anchor_embeddings, positive_embeddings):
        distance = F.pairwise_distance(anchor, positive)
        
        if distance <= threshold:
            tp += 1
        else:
            fn += 1
        
    for anchor, negative in zip(anchor_embeddings, negative_embeddings):
        distance = F.pairwise_distance(anchor, negative)
        
        if distance <= threshold:
            fp += 1
        else:
            tn += 1


    return {
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
    }


def aggregate_metrics(batch_results):
    fp = sum(result["fp"] for result in batch_results)
    fn = sum(result["fn"] for result in batch_results)
    tp = sum(result["tp"] for result in batch_results)
    tn = sum(result["tn"] for result in batch_results)


    total = (tp + tn + fp + fn)
    
    accuracy = (tp + tn) / total
    far = fp / total
    frr = fn / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    f1 = (2*recall*precision)/(recall+precision)
    

    return {
        "accuracy": accuracy, 
        "far": far, 
        "frr": frr, 
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



# Пример использования (в цикле оценки):
def evaluate_in_batches(model, dataloader, device, threshold):
    """
    Оценивает модель батчами.
    """
    model.eval()
    batch_results = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)

            batch_metrics = evaluate_recognition_batch(embeddings, labels, threshold)
            batch_results.append(batch_metrics)

    # Агрегируем результаты по всем батчам
    aggregated_metrics = aggregate_metrics(batch_results)
    return aggregated_metrics