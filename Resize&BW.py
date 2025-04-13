import os
from PIL import Image

# Path to the folder containing images
input_folder_path = r'C:\Users\shrey\OneDrive\Desktop\Project_Photo'
# Path to the folder where processed images will be saved
output_folder_path = r'C:\Users\shrey\OneDrive\Desktop\Project_Photo_BW'

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Standard size for the images
standard_size = (256, 256)

# Loop through the files in the input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        try:
            # Open the image
            img_path = os.path.join(input_folder_path, filename)
            img = Image.open(img_path)

            # Convert to black and white
            img_bw = img.convert('L')

            # Resize to standard size
            img_resized = img_bw.resize(standard_size)

            # Save the processed image to the output folder
            output_path = os.path.join(output_folder_path, filename)
            img_resized.save(output_path)

            print(f'Processed {filename} and saved to {output_path}')
        except Exception as e:
            print(f'Error processing {filename}: {e}')

print('All images have been processed.')
