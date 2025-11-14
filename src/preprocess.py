# preprocess.py
"""
Beginner-friendly script for loading coronary CT images, resizing, normalizing, converting to RGB,
and splitting into train/val/test folders (if not already split).
"""
import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob

# Set your image directory paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../data')
IMG_SIZE = (600, 600)

# Output split (if dataset is not already split)
def split_data(source_dir, train_dir, val_dir, test_dir, val_size=0.2, test_size=0.2, seed=42):
    images = glob(os.path.join(source_dir, '*.png')) + glob(os.path.join(source_dir, '*.jpg'))
    train_img, temp_img = train_test_split(images, test_size=(val_size+test_size), random_state=seed)
    val_img, test_img = train_test_split(temp_img, test_size=0.5, random_state=seed)
    splits = [(train_dir, train_img), (val_dir, val_img), (test_dir, test_img)]
    for out_dir, imgs in splits:
        os.makedirs(out_dir, exist_ok=True)
        for img_f in imgs:
            shutil.copy(img_f, out_dir)
    print(f"Split: {len(train_img)} train, {len(val_img)} val, {len(test_img)} test.")

def preprocess_and_save(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    images = glob(os.path.join(src_folder, '*.png')) + glob(os.path.join(src_folder, '*.jpg'))
    for img_f in images:
        img = cv2.imread(img_f, cv2.IMREAD_UNCHANGED)
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        img = cv2.resize(img, IMG_SIZE)
        # Normalize
        img = img / 255.0
        # Save as numpy array for speed (optional, else save as PNG)
        base_name = os.path.basename(img_f)
        np.save(os.path.join(dst_folder, base_name + '.npy'), img.astype(np.float32))
    print(f"Processed {len(images)} images in {src_folder}")

if __name__ == "__main__":
    normal_dir = os.path.join(DATA_DIR, 'CAD')
    disease_dir = os.path.join(DATA_DIR, 'NonCAD')
    # Optional: split_data() and preprocess_and_save() for each class
    # Example:
    # split_data(normal_dir, <train_path>, <val_path>, <test_path>)
    # preprocess_and_save(<your_input_folder>, <your_output_folder>)
    print("Edit src/preprocess.py as needed. See comments!")
