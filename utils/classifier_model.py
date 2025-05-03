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


def evaluate_recognition_batch(embeddings_batch, labels_batch, threshold=0.5):
    """
    Оценивает модель распознавания лиц для одного батча, вычисляя Accuracy, FAR, FRR.
    Возвращает промежуточные результаты для объединения.

    Args:
        embeddings_batch (torch.Tensor): Embedding лиц размера (batch_size, embedding_size).
        labels_batch (torch.Tensor): Метки классов размера (batch_size).
        threshold (float): Пороговое значение.

    Returns:
        dict: Словарь с промежуточными результатами (correct, false_accepts, false_rejects, total_positives, total_negatives).
    """
    n = len(embeddings_batch)
    correct = 0
    false_accepts = 0
    false_rejects = 0
    total_positives = 0
    total_negatives = 0

    # Вычисление попарных расстояний (косинусное расстояние)
    distances = 1 - torch.nn.functional.cosine_similarity(embeddings_batch.unsqueeze(1), embeddings_batch.unsqueeze(0), dim=2) # Косинусное расстояние

    # Создание матрицы соответствия (метки одинаковые = 1, разные = 0)
    ground_truth = labels_batch.unsqueeze(0) == labels_batch.unsqueeze(1)


    for i in range(n):
        for j in range(i + 1, n): # Итерируем только по верхней треугольной матрице
            distance = distances[i, j].item()
            same_person = ground_truth[i, j].item()


            if same_person:
                total_positives += 1
                if distance > threshold:  # Ошибка: расстояние больше порога, но это один и тот же человек
                    false_rejects += 1
                    correct += 0 # Явно указываем, что ответ неверный
                else:
                    correct += 1 # Явно указываем, что ответ верный

            else:
                total_negatives += 1
                if distance <= threshold:  # Ошибка: расстояние меньше или равно порогу, но это разные люди
                    false_accepts += 1
                    correct += 0  # Явно указываем, что ответ неверный
                else:
                    correct += 1  # Явно указываем, что ответ верный

    return {
        "correct": correct,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "total_positives": total_positives,
        "total_negatives": total_negatives,
    }


def aggregate_metrics(batch_results):
    """
    Агрегирует результаты оценки по батчам в общие метрики.

    Args:
        batch_results (list): Список словарей с промежуточными результатами для каждого батча.

    Returns:
        dict: Словарь с общими метриками (accuracy, far, frr).
    """
    total_correct = sum(result["correct"] for result in batch_results)
    total_false_accepts = sum(result["false_accepts"] for result in batch_results)
    total_false_rejects = sum(result["false_rejects"] for result in batch_results)
    total_total_positives = sum(result["total_positives"] for result in batch_results)
    total_total_negatives = sum(result["total_negatives"] for result in batch_results)

    accuracy = total_correct / (total_total_positives + total_total_negatives) if (total_total_positives + total_total_negatives) > 0 else 0
    far = total_false_accepts / total_total_negatives if total_total_negatives > 0 else 0
    frr = total_false_rejects / total_total_positives if total_total_positives > 0 else 0

    return {"accuracy": accuracy, "far": far, "frr": frr}



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