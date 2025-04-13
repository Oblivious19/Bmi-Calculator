import os
import csv

# Path to the folder containing augmented images
augmented_folder_path = r'C:\Users\shrey\OneDrive\Desktop\Project_Photo_Augmented'

# Function to convert height from the custom format to cm
def height_to_cm(height):
    feet = height // 100
    inches = height % 100
    return feet * 30.48 + inches * 2.54

# Function to convert weight from pounds to kg
def weight_to_kg(weight):
    return weight * 0.453592

# Initialize lists to store data
data = []

# Loop through the augmented images folder
for filename in os.listdir(augmented_folder_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        try:
            print(f"Processing file: {filename}")

            # Extract information from filename
            height_weight = filename.split('_')[0]  # Extract the height-weight part
            name = filename.split('_')[1]  # Extract the person's name
            augmented_number = filename.split('_')[-1].split('.')[0]  # Extract the augmented number

            print(f"Extracted height-weight: {height_weight}")
            print(f"Extracted name: {name}")
            print(f"Extracted augmented number: {augmented_number}")

            # Extract height and weight
            height = int(height_weight.split('-')[0])  # Extract height (inches)
            weight = int(height_weight.split('-')[1])  # Extract weight (pounds)

            print(f"Extracted height (inches): {height}")
            print(f"Extracted weight (pounds): {weight}")

            # Convert height to cm and weight to kg using the correct functions
            height_cm = round(height_to_cm(height), 2)
            weight_kg = round(weight_to_kg(weight), 2)

            print(f"Converted height (cm): {height_cm}")
            print(f"Converted weight (kg): {weight_kg}")

            # Construct row data
            row = {
                'Filename': filename,
                'Name': name,
                'Height': height,
                'Weight': weight,
                'Height_cm': height_cm,
                'Weight_kg': weight_kg,
                'Augmented_Number': augmented_number
            }

            # Append row to data list
            data.append(row)

        except Exception as e:
            print(f'Error processing {filename}: {e}')

# Define CSV file path
csv_file_path = r'C:\Users\shrey\OneDrive\Desktop\augmented_images_metadata.csv'

# Write data to CSV file
with open(csv_file_path, mode='w', newline='') as file:
    fieldnames = ['Filename', 'Name', 'Height', 'Weight', 'Height_cm', 'Weight_kg', 'Augmented_Number']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f'CSV file saved with augmented images metadata: {csv_file_path}')
