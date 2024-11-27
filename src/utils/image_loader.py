import cv2
import os
import glob
import numpy as np

def load_images_from_folder(folder_path):
    """Load all images from the specified folder."""
    image_files = glob.glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) + \
                  glob.glob(os.path.join(folder_path, '*.[pP][nN][gG]'))
    images = []
    filenames = []

    for image_path in sorted(image_files):
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
            filenames.append(os.path.basename(image_path))
            print(f"Loaded: {os.path.basename(image_path)}")
        else:
            print(f"Failed to load: {image_path}")

    return images, filenames