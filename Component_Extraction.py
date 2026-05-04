import argparse

import cv2
import numpy as np
import os
import os
import random
import torchvision
import torchvision.transforms.functional as TF
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from scipy import ndimage as ndi


# Component cropping for a single image
def component_cropping(input_folder, filename, output_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):
        
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
        if image is None:
            print(f"Skipping unreadable file: {filename}")
            return
    # -----------------------
    # Threshold (Otsu)
    # -----------------------
    _, thresh = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # If objects appear black, invert
    # thresh = cv2.bitwise_not(thresh)

    # -----------------------
    # Morphological cleaning
    # -----------------------
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # -----------------------
    # Connected Components
    # -----------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    # Collect bounding boxes
    boxes = []

    for i in range(1, num_labels):  # skip background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter small noise
        if area > 2000:   # adjust threshold if needed
            boxes.append((x, y, w, h))
    # -----------------------
    # Sort into grid structure
    # -----------------------
    # Sort by Y (top to bottom)
    boxes = sorted(boxes, key=lambda b: b[1])

    # Group into rows (3 rows expected)
    rows = np.array_split(boxes, 3)

    sorted_boxes = []

    for row in rows:
        # Sort each row by X (left to right)
        row = sorted(row, key=lambda b: b[0])
        sorted_boxes.extend(row)
    # Crop and Save as filename_{object_number}.tiff
    os.makedirs(output_path, exist_ok=True)

    for idx, (x, y, w, h) in enumerate(sorted_boxes):
        crop = image[y:y+h, x:x+w]
        #save as tiff format in 16 bit format
        if crop.dtype != np.uint16:
            crop = crop.astype(np.uint16) * 256   # scale 8-bit → 16-bit
        cv2.imwrite(os.path.join(output_path, f"{os.path.splitext(filename)[0]}_object_{idx+1:02}.png"), crop)
    print(f"Saved {len(sorted_boxes)} cropped objects from {filename}.")
    return len(sorted_boxes) not in [13, 15]  # return True if count is unexpected


#Component cropping for all images in a folder
def folder_component_cropping(input_folder, output_path):
    error = []
    for filename in os.listdir(input_folder):
        if component_cropping(input_folder, filename, output_path):
            error.append(filename)

    if error:
        print(f"Errors found in files: {error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #take command line arguments for hyperparameters and paths
    parser.add_argument('--input_folder', type=str, default="/home/godwinkhalko/ISRO/ISRO_DATASET", help='Path to the folder containing the images to be processed')
    parser.add_argument('--output_folder', type=str, default='/home/godwinkhalko/ISRO/ISRO_COMPONENTS', help='Path to save the cropped component images')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    folder_component_cropping(input_folder, output_folder)