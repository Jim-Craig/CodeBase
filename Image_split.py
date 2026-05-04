import os
import argparse
import re
import shutil

from importlib_metadata import files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--components_folder', type=str, default="/home/godwinkhalko/ISRO/ISRO_COMPONENTS", help='Path to the folder containing the components')
    parser.add_argument('--text_file', type=str, default="/home/godwinkhalko/ISRO/CodeBase/half1.txt", help='Path to the text file containing the image names')

    args = parser.parse_args()

    output_path = os.path.join("/home/godwinkhalko/ISRO", os.path.basename(args.text_file).split('.')[0])
    os.makedirs(output_path, exist_ok=True)

    with open(args.text_file, 'r') as file:
        image_names = file.readlines()
        image_names = [name.lower().split('.')[0].strip() for name in image_names]
        #iterate through the image_names and add the files in components_folder whose first half matches the image names into output_path
        for f in os.listdir(args.components_folder):
            try:
                spl = f.lower().split(' processed')
                new_f = (spl[0] + spl[1]).split('.')[0].strip()
            except IndexError:
                print(f"File {f} does not match the expected format")
                break
            if new_f in image_names:
                src_path = os.path.join(args.components_folder, f)
                dst_path = os.path.join(output_path, f)
                #copy the files into src_path
                shutil.copy(src_path, dst_path)
    print(f"Files have been copied to {output_path}")
