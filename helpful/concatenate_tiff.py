import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import fast


tiff_folder = '/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/export_cores'
destination_folder = '/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/export_triplets'

# Initialize a dictionary to hold the grouped file names
grouped_files = {}

# Iterate over all files in the specified folder
for file in os.listdir(tiff_folder):
    if file.endswith('.tiff') or file.endswith('.tif'):
        # Extract the numeric prefix from the filename
        prefix = file.split('_')[0]
        # Add the file to the corresponding group in the dictionary
        if prefix not in grouped_files:
            grouped_files[prefix] = []
        grouped_files[prefix].append(file)

# Iterate over each group of files and concatenate them
for prefix, files in grouped_files.items():
    # List to hold image objects
    images = []

    # Load each image and append to the list
    for file in files:
        image_path = os.path.join(tiff_folder, file)
        image = Image.open(image_path)
        images.append(np.array(image))

    max_height = max(img.shape[0] for img in images)
    padded_images = []

    for img in images:
        height, width, channels = img.shape
        # Create a new array with the maximum height
        padded_img = np.zeros((max_height, width, channels), dtype=img.dtype)
        # Copy the original image into the padded array
        padded_img[:height, :] = img
        padded_images.append(padded_img)


    concatenated_array = np.concatenate(padded_images, axis=1)

    # plt.imshow(concatenated_array)
    # plt.axis('off')
    # plt.show()

    output_path = os.path.join(destination_folder, f'{prefix}.tiff')

    concatenated_image = Image.fromarray(concatenated_array)
    Image.MAX_IMAGE_PIXELS = None
    concatenated_image.save(output_path, format='TIFF')

    importer_annotated = fast.ImageImporter.create(output_path)
    image = importer_annotated.runAndGetOutputData()

    print(f'Saved {output_path}')