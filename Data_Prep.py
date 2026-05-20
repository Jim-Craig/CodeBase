import os
import cv2
import numpy as np
from PIL import Image
if __name__ == "__main__":
    images_folder = "/home/godwinkhalko/ISRO/ISRO_COMPONENTS"
    masks_folder = "/home/godwinkhalko/ISRO/ISRO_COMPONENTS_ANNOTATION"
    binary_folder = "/home/godwinkhalko/ISRO/ISRO_COMPONENTS_BINARY"
    binary_mask_value = 1

    os.makedirs(binary_folder, exist_ok=True)

    for img_name in os.listdir(images_folder):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".tiff")):
            continue

        img_path = os.path.join(images_folder, img_name)
        mask_path = os.path.join(masks_folder, img_name.rsplit(".", 1)[0] + ".png")
        binary_mask_path = os.path.join(binary_folder, img_name.rsplit(".", 1)[0] + ".png")

        # Load original image to get dimensions
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        # Create blank binary mask
        binary_mask = np.zeros((h, w), dtype=np.uint8)

        if os.path.exists(mask_path):
            # Load the annotation mask (assuming it's a grayscale image where objects are white and background is black)
            annotation_mask = np.array(Image.open(mask_path).convert("L"))

            # Create binary mask where object pixels are set to binary_mask_value and background is 0
            binary_mask[annotation_mask > 0] = binary_mask_value

        # Save the binary mask as a PNG image in the binary folder
        cv2.imwrite(binary_mask_path, binary_mask)