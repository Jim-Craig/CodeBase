import os
import cv2
import numpy as np
# Component cropping for a single image


# I wouldn't want to change the component cropping code since it is working perfectly and is not a bottleneck in the pipeline.
# However if you feel like the components aren't being cropped correctly, you can adjust the threshold in the code below.
# A larger threshold will result in fewer, larger components (potentially merging nearby objects), 
# while a smaller threshold will result in more, smaller components (potentially splitting single objects into multiple parts).
def component_cropping(input_folder, filename, output_path, threshold=1500):
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
        if area > threshold:   # adjust threshold if needed
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
def folder_component_cropping(input_folder, output_path, threshold=1500):
    for filename in os.listdir(input_folder):
        component_cropping(input_folder, filename, output_path, threshold)

