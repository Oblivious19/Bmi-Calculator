import os
import logging
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# Load the model
def load_model():
    try:
        # Try multiple possible paths for the model
        possible_paths = [
            "C:/Users/shrey/OneDrive/Desktop/Working Projects/BMI/BMI_Output/custom_cnn_bmi_model_final.keras",
            os.path.join(os.path.dirname(__file__), "model", "custom_cnn_bmi_model_final.keras"),
            os.path.join(os.path.dirname(__file__), "custom_cnn_bmi_model_final.keras"),
            "model/custom_cnn_bmi_model_final.keras",
            "custom_cnn_bmi_model_final.keras"
        ]
        
        model = None
        for model_path in possible_paths:
            if os.path.exists(model_path):
                logger.info(f"Attempting to load model from: {model_path}")
                model = tf.keras.models.load_model(model_path)
                logger.info(f"Model successfully loaded from: {model_path}")
                break
        
        if model is None:
            # If no model is found, create a dummy model for testing
            logger.warning("No model file found. Creating a dummy model for testing.")
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(2, input_shape=(224, 224, 3))
            ])
            logger.info("Dummy model created successfully.")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Initialize model
model = load_model()

def preprocess_image(image_bytes):
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def validate_height(height):
    """Validate and adjust height predictions to realistic ranges"""
    MIN_HEIGHT = 1.4  # 140cm
    MAX_HEIGHT = 2.1  # 210cm
    
    logger.info(f"Original height prediction: {height}m")
    
    # If height is already in a realistic range, return as is
    if MIN_HEIGHT <= height <= MAX_HEIGHT:
        return height
    
    # If height is in centimeters, convert to meters
    if height > 10:  # Likely in centimeters
        height = height / 100
        logger.info(f"Converted from cm to m: {height}m")
    
    # If still too large, try scaling down
    if height > MAX_HEIGHT:
        # Try different scaling factors
        scaling_factors = [3.3, 3.0, 10.0]
        for factor in scaling_factors:
            scaled_height = height / factor
            if MIN_HEIGHT <= scaled_height <= MAX_HEIGHT:
                logger.info(f"Scaled height by factor {factor}: {scaled_height}m")
                return scaled_height
        
        # If all scaling attempts fail, return the closest valid height
        logger.warning(f"Could not scale height to valid range, using closest valid height")
        return min(max(height, MIN_HEIGHT), MAX_HEIGHT)
    
    return height

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please try again later.'
            }), 500

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
        
        # Read and preprocess the image
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        if processed_image is None:
            return jsonify({'success': False, 'error': 'Error processing image'})
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Extract height and weight
        predicted_height = float(prediction[0][0])
        predicted_weight = float(prediction[0][1])
        
        # Validate height
        validated_height = validate_height(predicted_height)

        # Calculate BMI
        bmi = predicted_weight / (validated_height ** 2)
        
        # Determine BMI category
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
        elif 25 <= bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
        
        return jsonify({
            'success': True,
            'predicted_height_m': round(validated_height, 2),
            'predicted_height_cm': round(validated_height * 100, 1),
            'predicted_weight_kg': round(predicted_weight, 1),
            'predicted_bmi': round(bmi, 1),
            'bmi_category': category,
            'debug_info': {
                'raw_height': round(predicted_height, 2),
                'scaled_height': round(validated_height, 2),
                'scaling_factor': round(predicted_height / validated_height, 2) if predicted_height != validated_height else 1.0
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error during prediction: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
