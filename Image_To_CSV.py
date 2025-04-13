import os
import pandas as pd

# Function to convert height in feet-inches to centimeters
def height_to_cm(height):
    try:
        feet = int(height[:-2])  # Extract feet part from the string
        inches = int(height[-2:])  # Extract inches part from the string
        return (feet * 30.48) + (inches * 2.54)
    except ValueError:
        return None

# Function to convert weight in pounds to kilograms
def weight_to_kg(weight):
    return weight * 0.453592

# Path to the folder containing images
folder_path = r'C:\Users\shrey\OneDrive\Desktop\Project_Photo'

# List to store image data
image_data = []

# Loop through the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Extract height and weight from filename
        height, rest = filename.split('_')[0].split('-')
        weight = rest.split('_')[0]

        # Convert height and weight
        height_cm = height_to_cm(height)
        weight_kg = weight_to_kg(int(weight))

        # Append data to list
        image_data.append([filename, f'{height} ft-{rest} in', weight, height_cm, weight_kg])

# Create a DataFrame
df = pd.DataFrame(image_data, columns=['Image Name', 'Height (ft-in)', 'Weight (lbs)', 'Height (cm)', 'Weight (kg)'])

# Save the DataFrame to a CSV file
output_csv = os.path.join(folder_path, 'image_data.csv')
df.to_csv(output_csv, index=False)

print(f'Data successfully written to {output_csv}')
