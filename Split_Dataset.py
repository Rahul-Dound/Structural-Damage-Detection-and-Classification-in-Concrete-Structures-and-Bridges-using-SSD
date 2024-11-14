import os
import shutil
import math

# Define paths
dataset_path = 'C:\\EDI 7th SEM\\10kNew\\train'
img_folder = os.path.join(dataset_path, 'img')
ann_folder = os.path.join(dataset_path, 'ann')
output_folder = 'C:\\EDI 7th SEM\\Split datasets'

# Get all files from img and ann folders
img_files = sorted(os.listdir(img_folder))
ann_files = sorted(os.listdir(ann_folder))

# Ensure both folders have the same number of files
assert len(img_files) == len(ann_files), "Mismatch in number of images and annotations"

# Calculate split size
total_files = len(img_files)
split_size = math.ceil(0.05 * total_files)

# Split and copy files
split_num = 1
for i in range(0, total_files, split_size):
    # Create split folder
    split_folder = os.path.join(output_folder, f'split_{split_num}')
    img_split_folder = os.path.join(split_folder, 'img')
    ann_split_folder = os.path.join(split_folder, 'ann')
    os.makedirs(img_split_folder, exist_ok=True)
    os.makedirs(ann_split_folder, exist_ok=True)
    
    # Copy files
    img_batch = img_files[i:i + split_size]
    ann_batch = ann_files[i:i + split_size]
    
    for img_file, ann_file in zip(img_batch, ann_batch):
        shutil.copy(os.path.join(img_folder, img_file), os.path.join(img_split_folder, img_file))
        shutil.copy(os.path.join(ann_folder, ann_file), os.path.join(ann_split_folder, ann_file))
    
    split_num += 1

print(f"Dataset split into {split_num-1} parts.")
