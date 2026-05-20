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
from ultralytics import YOLO

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


def init_model(device = "cuda:1"):
    checkpoint = "/home/godwinkhalko/ISRO/UNetPP/isro_unetplusplus_resnet34.pth"

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
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


def init_ROI_model(device):
    checkpoint = "/home/godwinkhalko/ISRO/CodeBase/ROI_Detector/runs/detect/runs/isro/exp1/weights/best.pt"
    model = YOLO(checkpoint)
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model, device

# def get_defect_description(index, image, pred_mask, output_path, SGP, LGP):
#     binary_mask = np.array(pred_mask) > 0

#     # Label connected components
#     labeled, num = ndi.label(binary_mask)
#     sizes = ndi.sum(binary_mask, labeled, range(1, num + 1))

#     # Filter out noise below threshold
#     min_pixels = 5

#     # Find bounding boxes for all labeled objects
#     objects = ndi.find_objects(labeled)

#     # Count and print only valid (non-noise) objects
#     valid_objects = [(i, obj) for i, obj in enumerate(objects) if sizes[i] > min_pixels]
#     #Write onto a .txt file
#     with open(f"{output_path}/defect_description_{index}.txt", "w") as f:
#         f.write(f"Number of voids in component {index}: {len(valid_objects)}\n")
#         for rank, (i, obj) in enumerate(valid_objects, start=1):
#             # Convert slices to readable pixel coordinates
#             row_start = obj[0].start
#             row_end   = obj[0].stop
#             col_start = obj[1].start
#             col_end   = obj[1].stop
#             width     = col_end - col_start
#             height    = row_end - row_start

#             f.write(f"\void {rank} in component {index}:")
#             f.write(f"  Bounding box -> Row: {row_start} to {row_end}, Col: {col_start} to {col_end}")
#             f.write(f"  Width: {width}px, Height: {height}px")
#             f.write(f"  Area (white pixels): {int(sizes[i])} pixels")
#     #save the description in a text file
#     f.close()
#     #plot the original image and the predicted mask overlaid on top of original image
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(image, cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.title("Predicted Mask Overlay")
#     plt.imshow(image, cmap='gray')
#     plt.imshow(pred_mask, cmap='jet', alpha=0.5)
#     #Write the plot to the defect description file
#     plt.savefig(f"{output_path}/defect_visualization_component_{index}.png")
#     plt.close()

def get_defect_description(index, image, pred_mask, output_path, SGP, LGP):
    binary_mask = np.array(pred_mask) > 0

    # Label connected components
    labeled, num = ndi.label(binary_mask)
    sizes = ndi.sum(binary_mask, labeled, range(1, num + 1))

    min_pixels = 5
    objects = ndi.find_objects(labeled)
    valid_objects = [(i, obj) for i, obj in enumerate(objects) if sizes[i] > min_pixels]

    def compute_overlap_area(obj, box):
        """Compute pixel overlap between a defect bounding box and an ROI box."""
        row_start, row_end = obj[0].start, obj[0].stop
        col_start, col_end = obj[1].start, obj[1].stop
        bx1, by1, bx2, by2 = box.astype(int)

        # Intersection
        ix1 = max(col_start, bx1)
        iy1 = max(row_start, by1)
        ix2 = min(col_end,   bx2)
        iy2 = min(row_end,   by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0  # No overlap
        return (ix2 - ix1) * (iy2 - iy1)

    def assign_object_to_region(obj, SGP, LGP):
        """Assign defect to SGP or LGP based on max overlap."""
        best_region = None
        best_overlap = 0

        
        overlap = compute_overlap_area(obj, SGP)
        if overlap > best_overlap:
            best_overlap = overlap
            best_region = 'SGP'

        
        overlap = compute_overlap_area(obj, LGP)
        if overlap > best_overlap:
            best_overlap = overlap
            best_region = 'LGP'

        return best_region  # None if no overlap with any box

    # Group objects
    sgp_objects, lgp_objects, unassigned_objects = [], [], []

    for i, obj in valid_objects:
        region = assign_object_to_region(obj, SGP, LGP)
        if region == 'SGP':
            sgp_objects.append((i, obj))
        elif region == 'LGP':
            lgp_objects.append((i, obj))
        else:
            unassigned_objects.append((i, obj))

    def write_object_details(f, rank, i, obj, sizes):
        row_start, row_end = obj[0].start, obj[0].stop
        col_start, col_end = obj[1].start, obj[1].stop
        width  = col_end - col_start
        height = row_end - row_start
        f.write(f"\n  Void {rank}:")
        f.write(f"\n    Bounding box -> Row: {row_start} to {row_end}, Col: {col_start} to {col_end}")
        f.write(f"\n    Width: {width}px, Height: {height}px")
        f.write(f"\n    Area (white pixels): {int(sizes[i])} pixels")

    with open(f"{output_path}/defect_description_{index}.txt", "w") as f:
        f.write(f"Number of voids in component {index}: {len(valid_objects)}\n")
        f.write(f"  - Inside SGP regions: {len(sgp_objects)}\n")
        f.write(f"  - Inside LGP regions: {len(lgp_objects)}\n")
        f.write(f"  - Outside all ROI regions: {len(unassigned_objects)}\n")

        # SGP objects
        f.write(f"\n--- Voids in SGP Region ({len(sgp_objects)} total) ---")
        for rank, (i, obj) in enumerate(sgp_objects, start=1):
            write_object_details(f, rank, i, obj, sizes)

        # LGP objects
        f.write(f"\n\n--- Voids in LGP Region ({len(lgp_objects)} total) ---")
        for rank, (i, obj) in enumerate(lgp_objects, start=1):
            write_object_details(f, rank, i, obj, sizes)

        # Unassigned
        if unassigned_objects:
            f.write(f"\n\n--- Voids Outside ROI Regions ({len(unassigned_objects)} total) ---")
            for rank, (i, obj) in enumerate(unassigned_objects, start=1):
                write_object_details(f, rank, i, obj, sizes)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask Overlay")
    plt.imshow(image, cmap='gray')
    plt.imshow(pred_mask, cmap='jet', alpha=0.5)
    plt.savefig(f"{output_path}/defect_visualization_component_{index}.png")
    plt.close()


def run_roi_inference(image_path, roi_model, device):
    # Load and normalize to 0-255 uint8
    image = Image.open(image_path)
    arr = np.array(image, dtype=np.float32)

    # Stretch to 0-255
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    arr_uint8 = arr.astype(np.uint8)

    # Convert grayscale to RGB (YOLO needs 3 channels)
    image_rgb = Image.fromarray(arr_uint8, mode='L').convert('RGB')
    image_rgb = image_rgb.resize((640, 640))  # YOLO default size

    # Pass directly to YOLO — no manual tensor conversion
    roi_results = roi_model(
        np.array(image_rgb),
        device=device,
        verbose=False
    )

    # Extract from result[0] (first image in batch)
    boxes = roi_results[0].boxes
    scale = 256/640
    boxes = (boxes * scale)
    SGP = boxes[0].xyxy.cpu().numpy()   
    LGP = boxes[1].conf.cpu().numpy()  

    return SGP, LGP
            
def predict_and_describe(model, roi_model, device, input_folder, output_path,threshold=0.15):
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
            SGP, LGP = run_roi_inference(image_path, roi_model, device)
            if len(SGP) == 0 or len(LGP) == 0 :
                print(f"No ROI detected in component {i}. Skipping defect description.")
                continue

            output = model(image_tensor)
            pred_mask = torch.sigmoid(output).cpu()  # keep as tensor
            pred_mask = (pred_mask > threshold).float()
            kernel = np.ones((3,3), np.uint8)
            pred_mask = cv2.dilate(pred_mask.numpy().squeeze(), kernel)
            pred_mask = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float()  # back to tensor with shape [1, 1, H, W]
            get_defect_description(i, image, pred_mask.squeeze().cpu().numpy(), output_path, SGP, LGP)
            pred_masks.append(pred_mask)
    return pred_masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISRO Defect Detection and Description")
    parser.add_argument('--input_folder', type=str, default="/home/godwinkhalko/ISRO/ISRO_DATASET", help='Path to the folder containing the images to be processed')
    parser.add_argument('--component_output_folder', type=str, default='/home/godwinkhalko/ISRO/output', help='Path to save the cropped component images')
    parser.add_argument('--prediction_output_folder', type=str, default='/home/godwinkhalko/ISRO/output_prediction', help='Path to save the prediction outputs')
    parser.add_argument('--filename', type=str, default="Batch  no; 2023-13-11---401 to 415-A shot processed.tiff", help='Filename of the image to be processed for component extraction')
    parser.add_argument('--component_extraction', type=bool, default=False, help='Flag to check if component extraction is needed before prediction')
    args = parser.parse_args()

    component_output_folder = args.component_output_folder
    if args.component_extraction:
        input_folder = args.input_folder
        filename = args.filename
        sorted_boxes = component_cropping(input_folder, filename, component_output_folder)

    prediction_output_path = args.prediction_output_folder
    os.makedirs(prediction_output_path, exist_ok=True)
    model, device = init_model()
    roi_model, device = init_ROI_model(device)
    try:
        pred_mask = predict_and_describe(model, roi_model, device, component_output_folder, prediction_output_path)
    except Exception as e:
        print(f"Error during prediction and description: {e}")
    finally:
        del model
        del roi_model
        del device
