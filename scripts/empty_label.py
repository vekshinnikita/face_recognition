import json
import os

from utils.system import ensure_directory_exists_os

def make_empty_label_file(dir_path: str, file_path: str):
    last_dir = dir_path.split(os.path.sep)[-2]
    
    json_template = {
        "file_name": file_path,
        "annotations": [],
        "file_path": os.path.join(last_dir, file_path)
    }
    
    return json.dumps(json_template)


empty_images_path = './dataset/images/test/empty/'

empty_labels_path = './dataset/labels/test/empty/'

ensure_directory_exists_os(empty_labels_path)
for file_path in os.listdir(empty_images_path):
    if not file_path.startswith('.'):
        new_file_name = file_path.replace('.jpg', '.json')
        
        json_str = make_empty_label_file(empty_images_path, new_file_name)
        with open(os.path.join(empty_labels_path, new_file_name), 'w', encoding="utf-8") as f:
            f.write(json_str)
        
    
    