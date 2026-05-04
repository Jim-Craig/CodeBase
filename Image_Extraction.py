import argparse
import os
import cv2

def load_images_from_folder(folder, extracted_images_path):
    if not os.path.exists(extracted_images_path):
        os.makedirs(extracted_images_path)

    folders = os.listdir(folder)
    #iterate though the folders
    print(f"Found folders: {folders}")
    for f in folders:
        #Collect the subfolders in the folder
        subfolders = os.listdir(os.path.join(folder, f))
        print(f"Found subfolders in {f}: {subfolders}")
        #iterate through the subfolders
        for sf in subfolders:
            sf_path = os.path.join(folder, f, sf)
            if os.path.isdir(sf_path):
                #load imges inthis subfolder and convert to numpy arrays
                for img in os.listdir(sf_path):
                    img_path = os.path.join(sf_path, img)
                    img_array = cv2.imread(img_path)
                    if img_array is None:
                        print(f"Could not read image {img} from path: {img_path}")
                        continue
                    fileName, _ = os.path.splitext(img)
                    save_path = os.path.join(extracted_images_path, f"{fileName}.tiff")
                    cv2.imwrite(save_path, img_array)
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #take command line arguments for hyperparameters and paths
    parser.add_argument('--dataset_path', type=str, default='/home/godwinkhalko/ISRO/VSSC Processed Images', help='path to the VSSC dataset')
    parser.add_argument('--train_mask_dir', type=str, default='/home/godwinkhalko/ISRO/ISRO_DATASET', help='path to save the extracted images')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    extracted_images_path = args.train_mask_dir
    #load the images onto a image array

    load_images_from_folder(dataset_path, extracted_images_path)
