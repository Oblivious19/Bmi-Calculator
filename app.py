from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Set path to the model
output_dir = r'C:\Users\shrey\OneDrive\Desktop\BMI_Output'
model_path = os.path.join(output_dir, 'custom_cnn_bmi_model_final.keras')

# Directory to temporarily save images
TEMP_DIR = 'temp'

# Ensure the temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Load the model
try:
    model = load_model(model_path)
    print(f"Model successfully loaded from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to preprocess the input image
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))  # Resize to model's expected input size
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        raise IOError(f"Error loading or processing image: {e}")

# Function to calculate BMI from height and weight
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100.0
    bmi = (weight_kg / (height_m * height_m) / 10000)
    return bmi

# Main route for rendering the HTML form
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the templates folder

# Route to serve temporary images
@app.route('/temp/<filename>')
def send_temp_image(filename):
    return send_from_directory(TEMP_DIR, filename)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded image temporarily
        image_path = os.path.join(TEMP_DIR, image_file.filename)
        image_file.save(image_path)

        # Preprocess the image
        img_array = preprocess_image(image_path)

        # Predict height and weight using the model
        prediction = model.predict(img_array)

        if prediction.shape[1] != 2:
            raise ValueError("Model output shape is not as expected. Check the model configuration.")

        predicted_height_cm, predicted_weight_kg = prediction[0]

        # Convert numpy float32 to native Python float
        predicted_height_cm = float(predicted_height_cm)
        predicted_weight_kg = float(predicted_weight_kg)

        # Calculate BMI
        predicted_bmi = calculate_bmi(predicted_weight_kg, predicted_height_cm)

        # Return results as JSON including the image URL
        return jsonify({
            "predicted_height_cm": round(predicted_height_cm, 2),
            "predicted_weight_kg": round(predicted_weight_kg, 2),
            "predicted_bmi": round(predicted_bmi, 2),
            "image_url": f"/temp/{os.path.basename(image_path)}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
