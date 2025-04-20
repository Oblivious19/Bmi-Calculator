import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the directory where the model is saved
output_dir = r'C:\Users\shrey\OneDrive\Desktop\BMI_Output'
model_path = os.path.join(output_dir, 'custom_cnn_bmi_model_final.keras')

# Load the pre-trained model
try:
    model = load_model(model_path)
    print(f"Model successfully loaded from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to preprocess the input image
def preprocess_image(image_path):
    print(f"Loading image from: {image_path}")
    
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    
    try:
        img = load_img(image_path, target_size=(224, 224))  # Resize to model's expected input size
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        raise IOError(f"Error loading or processing image: {e}")

# Function to calculate BMI from height and weight
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    return bmi

# Function to save and print results
def save_and_print_results(predicted_height, predicted_weight, predicted_bmi, output_file):
    with open(output_file, 'w') as file:
        file.write(f"Predicted Height: {predicted_height:.2f} cm\n")
        file.write(f"Predicted Weight: {predicted_weight:.2f} kg\n")
        file.write(f"Predicted BMI: {predicted_bmi:.2f}\n")
    
    print(f"Predicted Height: {predicted_height:.2f} cm")
    print(f"Predicted Weight: {predicted_weight:.2f} kg")
    print(f"Predicted BMI: {predicted_bmi:.2f}")
    print(f"Results saved to: {output_file}")

# Main function to process an image and save the results
def main(image_path, output_file):
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path)
        
        # Predict height and weight using the model
        prediction = model.predict(img_array)
        print("Raw prediction output:", prediction)  # Print prediction output for debugging
        
        if prediction.shape[1] != 2:
            raise ValueError("Model output shape is not as expected. Check the model configuration.")
        
        predicted_height_cm, predicted_weight_kg = prediction[0]
        
        # Calculate BMI
        predicted_bmi = calculate_bmi(predicted_weight_kg, predicted_height_cm)
        
        # Save and print results
        save_and_print_results(predicted_height_cm, predicted_weight_kg, predicted_bmi, output_file)
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
new_image_path = r'C:\Users\shrey\OneDrive\Desktop\410-90_Katherine_L1.jpg'
output_file_path = r'C:\Users\shrey\OneDrive\Desktop\BMI_Calculator\prediction_results.txt'
# Check the model summary to understand the architecture
model.summary()

main(new_image_path, output_file_path)
