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
import argparse

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
    return sorted_boxes

#Component cropping for all images in a folder
def folder_component_cropping(input_folder, output_path):
    for filename in os.listdir(input_folder):
        component_cropping(input_folder, filename, output_path)


def init_model():
    checkpoint = "/home/godwinkhalko/ISRO/UNetPP/isro_unetplusplus_resnet34.pth"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",   # backbone
        encoder_weights="imagenet",  # pretrained
        in_channels=1,  # X-ray = grayscale
        classes=1       # binary segmentation
        )
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.to(device)
    model.eval()  # Set to evaluation mode
    #iterate through validation images and predict masks
    model.eval()

    return model, device

def get_defect_description(index, image, pred_mask, output_path):
    binary_mask = np.array(pred_mask) > 0

    # Label connected components
    labeled, num = ndi.label(binary_mask)
    sizes = ndi.sum(binary_mask, labeled, range(1, num + 1))

    # Filter out noise below threshold
    min_pixels = 5

    # Find bounding boxes for all labeled objects
    objects = ndi.find_objects(labeled)

    # Count and print only valid (non-noise) objects
    valid_objects = [(i, obj) for i, obj in enumerate(objects) if sizes[i] > min_pixels]
    #Write onto a .txt file
    with open(f"{output_path}/defect_description_{index}.txt", "w") as f:
        f.write(f"Number of voids in component {index}: {len(valid_objects)}\n")
        for rank, (i, obj) in enumerate(valid_objects, start=1):
            # Convert slices to readable pixel coordinates
            row_start = obj[0].start
            row_end   = obj[0].stop
            col_start = obj[1].start
            col_end   = obj[1].stop
            width     = col_end - col_start
            height    = row_end - row_start

            f.write(f"\void {rank} in component {index}:")
            f.write(f"  Bounding box -> Row: {row_start} to {row_end}, Col: {col_start} to {col_end}")
            f.write(f"  Width: {width}px, Height: {height}px")
            f.write(f"  Area (white pixels): {int(sizes[i])} pixels")
    #save the description in a text file
    f.close()
    #plot the original image and the predicted mask overlaid on top of original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask Overlay")
    plt.imshow(image, cmap='gray')
    plt.imshow(pred_mask, cmap='jet', alpha=0.5)
    #Write the plot to the defect description file
    plt.savefig(f"{output_path}/defect_visualization_component_{index}.png")
    plt.close()

            
def predict_and_describe(model, device, input_folder, output_path,threshold=0.15):
    pred_masks = []
    resize = torchvision.transforms.Resize((256, 256))

    for i, filename in enumerate(os.listdir(input_folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):
            continue
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        arr = np.array(image, dtype=np.float32)

        # Stretch actual range (24320–64000) to full 0–255
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        image = Image.fromarray(arr.astype(np.uint8), mode='L')
        image = resize(image)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=[0.485], std=[0.229])
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.sigmoid(output).cpu()  # keep as tensor
            pred_mask = (pred_mask > threshold).float()
            kernel = np.ones((3,3), np.uint8)
            pred_mask = cv2.dilate(pred_mask.numpy().squeeze(), kernel)
            pred_mask = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float()  # back to tensor with shape [1, 1, H, W]
            get_defect_description(i, image, pred_mask.squeeze().cpu().numpy(), output_path)
            pred_masks.append(pred_mask)
    return pred_masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISRO Defect Detection and Description")
    parser.add_argument('--input_folder', type=str, default="/home/godwinkhalko/ISRO/ISRO_DATASET", help='Path to the folder containing the images to be processed')
    parser.add_argument('--component_output_folder', type=str, default='/home/godwinkhalko/ISRO/ISRO_COMPONENTS', help='Path to save the cropped component images')
    parser.add_argument('--prediction_output_folder', type=str, default='/home/godwinkhalko/ISRO/output_prediction', help='Path to save the prediction outputs')
    parser.add_argument('--filename', type=str, default="Batch  no_ 2023-13-16---551 to 567-BC shot.tiff", help='Filename of the image to be processed for component extraction')
    parser.add_argument('--component_extraction', type=bool, default=True, help='Flag to check if component extraction is needed before prediction')
    args = parser.parse_args()

    if args.component_extraction:
        input_folder = args.input_folder
        component_output_folder = args.component_output_folder
        filename = args.filename

        sorted_boxes = component_cropping(input_folder, filename, component_output_folder)
    prediction_output_path = args.prediction_output_folder
    model, device = init_model()
    pred_mask = predict_and_describe(model, device,component_output_folder, prediction_output_path)