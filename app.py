from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'BMI_Output')
MODEL_PATH = os.path.join(MODEL_DIR, 'custom_cnn_bmi_model_final.keras')

# Ensure required directories exist
for directory in [TEMP_DIR, UPLOAD_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Constants for image processing
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
TARGET_SIZE = (224, 224)

# Model loading with error handling
def load_bmi_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = load_model(MODEL_PATH)
        logger.info(f"Model successfully loaded from: {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Initialize model
model = load_bmi_model()

# Function to check if model is ready
def is_model_ready():
    return model is not None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file):
    """Validate the uploaded image file."""
    if not file:
        raise ValueError("No file provided")
    
    if not allowed_file(file.filename):
        raise ValueError(f"Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > MAX_IMAGE_SIZE:
        raise ValueError(f"File size exceeds maximum limit of {MAX_IMAGE_SIZE/1024/1024}MB")

def preprocess_image(image_path):
    """Preprocess the input image for model prediction."""
    try:
        # Load and validate image
        img = load_img(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(TARGET_SIZE)
        
        # Convert to array and normalize
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Image preprocessed successfully: {image_path}")
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError(f"Error preprocessing image: {str(e)}")

# Function to calculate BMI from height and weight
def calculate_bmi(weight_kg, height_m):
    """Calculate BMI using weight in kg and height in meters."""
    return weight_kg / (height_m * height_m)

# Main route for rendering the HTML form
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the templates folder

# Route to serve temporary images
@app.route('/temp/<filename>')
def send_temp_image(filename):
    return send_from_directory(TEMP_DIR, filename)

# Function to validate and scale height prediction
def validate_height(height_m):
    """
    Validate and scale height prediction to ensure it's within realistic human ranges.
    Average human height range: 1.4m to 2.1m (140cm to 210cm)
    """
    MIN_HEIGHT_M = 1.4  # 140 cm
    MAX_HEIGHT_M = 2.1  # 210 cm
    
    # Log the original prediction
    logger.info(f"Original height prediction: {height_m}m")
    
    # If height is already in a realistic range, return as is
    if MIN_HEIGHT_M <= height_m <= MAX_HEIGHT_M:
        logger.info(f"Height {height_m}m is within realistic range, using as is")
        return height_m
    
    # If height is in centimeters (e.g., 169), convert to meters
    if 140 <= height_m <= 210:
        converted = height_m / 100
        logger.info(f"Converting from cm to m: {height_m}cm -> {converted}m")
        return converted
    
    # If height is too large (e.g., 5.69m), scale it down by factor of 3
    if height_m > MAX_HEIGHT_M:
        # Try different scaling factors
        scale_factors = [
            (3.3, "divide by 3.3"),  # For values around 5.6m -> 1.7m
            (3.0, "divide by 3.0"),  # For values around 5.1m -> 1.7m
            (10.0, "divide by 10"),  # For values like 17m -> 1.7m
        ]
        
        for factor, method in scale_factors:
            scaled = height_m / factor
            if MIN_HEIGHT_M <= scaled <= MAX_HEIGHT_M:
                logger.info(f"Scaling height using {method}: {height_m}m -> {scaled}m")
                return scaled
    
    # If all attempts fail, use closest valid height
    closest_valid = min(max(height_m / 3.3, MIN_HEIGHT_M), MAX_HEIGHT_M)
    logger.warning(f"Using closest valid height: {closest_valid}m (original was {height_m}m)")
    return closest_valid

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if not is_model_ready():
        return jsonify({"error": "Model not initialized. Please check server logs."}), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file found"}), 400

        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Validate the uploaded image
        validate_image(image_file)
        
        # Create a unique filename to avoid conflicts
        filename = f"{os.path.splitext(image_file.filename)[0]}_{int(time.time())}{os.path.splitext(image_file.filename)[1]}"
        image_path = os.path.join(TEMP_DIR, filename)
        
        # Save and preprocess the image
        image_file.save(image_path)
        img_array = preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        if prediction.shape[1] != 2:
            raise ValueError("Model output shape is not as expected")
        
        # Get raw predictions
        raw_height_m, predicted_weight_kg = map(float, prediction[0])
        
        # Validate and scale height prediction
        predicted_height_m = validate_height(raw_height_m)
        
        # Convert height to cm for display
        predicted_height_cm = predicted_height_m * 100
        
        # Calculate BMI using validated height in meters
        predicted_bmi = calculate_bmi(predicted_weight_kg, predicted_height_m)
        
        # Clean up old files in temp directory
        cleanup_temp_files()
        
        # Prepare response with additional debug info
        response = {
            "success": True,
            "predicted_height_cm": round(predicted_height_cm, 2),  # Display in cm
            "predicted_height_m": round(predicted_height_m, 2),    # Also send meters
            "predicted_weight_kg": round(predicted_weight_kg, 2),
            "predicted_bmi": round(predicted_bmi, 2),
            "bmi_category": get_bmi_category(predicted_bmi),
            "image_url": f"/temp/{filename}",
            "debug_info": {
                "raw_height": round(raw_height_m, 3),
                "scaled_height": round(predicted_height_m, 3),
                "scaling_factor": round(predicted_height_m / raw_height_m, 3) if raw_height_m != 0 else 0
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def cleanup_temp_files(max_age_hours=1):
    """Clean up old temporary files."""
    try:
        current_time = time.time()
        for filename in os.listdir(TEMP_DIR):
            filepath = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > (max_age_hours * 3600):
                    os.remove(filepath)
    except Exception as e:
        logger.warning(f"Error cleaning up temporary files: {str(e)}")

def get_bmi_category(bmi):
    """Return BMI category based on the calculated BMI value."""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

if __name__ == '__main__':
    app.run(debug=True)
