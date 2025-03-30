import os

import shutil


dir_path = './dataset/images/test/n000000'

count = 1
for root, directories, files in os.walk(dir_path):

    for file_path in files:
        if not file_path.startswith('.'):
            source_path = os.path.join(root, file_path)
            destination_path = os.path.join(dir_path, f'{count:05d}.jpg')
            shutil.move(source_path, destination_path)
            count += 1
        
    
    