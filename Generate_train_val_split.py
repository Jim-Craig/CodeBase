import os
import numpy as np
import shutil

if __name__ == "__main__":
    #partition the dataset into train and validation sets with a split of 80% for training and 20% for validation. 
    # The images and their corresponding binary masks should be moved to separate folders for training and validation. 


    images_folder = "/home/godwinkhalko/ISRO/ISRO_COMPONENTS"
    annotations_folder = "/home/godwinkhalko/ISRO/ISRO_COMPONENTS_BINARY"

    train_images_folder = "/home/godwinkhalko/ISRO/VOID_DATA/train_images"
    train_annotations_folder = "/home/godwinkhalko/ISRO/VOID_DATA/train_annotations"
    val_images_folder = "/home/godwinkhalko/ISRO/VOID_DATA/val_images"
    val_annotations_folder = "/home/godwinkhalko/ISRO/VOID_DATA/val_annotations"

    # Create folders
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_annotations_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_annotations_folder, exist_ok=True)

    # Load paths
    images = sorted([
        os.path.join(images_folder, f)
        for f in os.listdir(images_folder)
    ])

    annotations = sorted([
        os.path.join(annotations_folder, f)
        for f in os.listdir(annotations_folder)
    ])

    # Pair and shuffle
    combined = list(zip(images, annotations))
    np.random.shuffle(combined)
    images, annotations = zip(*combined)

    # Split
    split_index = int(len(images) * 0.8)

    train_pairs = list(zip(images[:split_index], annotations[:split_index]))
    val_pairs = list(zip(images[split_index:], annotations[split_index:]))

    # Copy files
    for img_path, mask_path in train_pairs:
        shutil.copy(img_path, os.path.join(train_images_folder, os.path.basename(img_path)))
        shutil.copy(mask_path, os.path.join(train_annotations_folder, os.path.basename(mask_path)))

    for img_path, mask_path in val_pairs:
        shutil.copy(img_path, os.path.join(val_images_folder, os.path.basename(img_path)))
        shutil.copy(mask_path, os.path.join(val_annotations_folder, os.path.basename(mask_path)))