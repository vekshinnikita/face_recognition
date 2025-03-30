import random
import string
import torch
from utils.system import ensure_directory_exists_os

def generate_sequence(length):
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choice(alphabet) for _ in range(length))

def save_model(
    file_path: str,
    model,
    model_id,
    opt,
    lr_scheduler,
    train_loss,
    val_loss,
    best_loss,
    train_acc,
    val_acc,
    lr_list,
    EPOCHS,
    epoch,
    str_info: str = ''
):
    checkpoint = {
        'info': str_info,
        'model_id': model_id,
        'state_model': model.state_dict(),
        'state_opt': opt.state_dict(),
        'state_lr_scheduler': lr_scheduler.state_dict(),
        'loss': {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': best_loss
        },
        
        'metric': {
            'train_acc': train_acc,
            'val_acc': val_acc,
        },
        'lr_list': lr_list,
        'epoch': {
            'EPOCHS' : EPOCHS,
            'save_epoch': epoch
        }
    }
    
    file_path = file_path.format(epoch=epoch, id=model_id)
    
    ensure_directory_exists_os(file_path)
    
    torch.save(checkpoint, file_path)
    