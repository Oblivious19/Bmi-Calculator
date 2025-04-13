import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image

# Disable oneDNN custom operations to avoid log messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Path to the folder containing black and white images
input_folder_path = r'C:\Users\shrey\OneDrive\Desktop\Project_Photo'
# Path to the folder where augmented images will be saved
output_folder_path = r'C:\Users\shrey\OneDrive\Desktop\Project_Photo_Augmented'

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Initialize the ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to create new filename
def create_new_filename(original_filename, aug_num, image_count):
    # Remove the file extension
    name_without_ext = os.path.splitext(original_filename)[0]
    # Split the filename
    parts = name_without_ext.split('_')
    # Reconstruct the filename
    new_name = f"{parts[0]}_{parts[1]}_{aug_num:03d}_{(image_count-1)*15 + aug_num}.jpg"
    return new_name

# Count to keep track of generated images and processed original images
generated_count = 0
image_count = 0
augmentations_per_image = 15
target_count = 7000

# Loop through the files in the input folder
for filename in sorted(os.listdir(input_folder_path)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            image_count += 1
            img_path = os.path.join(input_folder_path, filename)
            img = load_img(img_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Generate augmented images
            for i in range(augmentations_per_image):
                aug_img = datagen.flow(x, batch_size=1)[0][0].astype(np.uint8)
                aug_filename = create_new_filename(filename, i+1, image_count)
                aug_path = os.path.join(output_folder_path, aug_filename)
                Image.fromarray(aug_img).save(aug_path)
                generated_count += 1

                if generated_count >= target_count:
                    break

        except Exception as e:
            print(f'Error processing {filename}: {e}')

        if generated_count >= target_count:
            break

print(f'Total {generated_count} augmented images generated and saved to {output_folder_path}')