# AI-Powered BMI Calculator

![BMI Calculator](https://img.shields.io/badge/BMI-Calculator-blue)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Flask](https://img.shields.io/badge/Flask-2.0.1-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)

An intelligent BMI calculator that uses AI to predict height and weight from full-body photos, providing instant BMI calculations and health insights.

## 🌟 Features

- **AI-Powered Predictions**: Automatically predicts height and weight from full-body photos
- **Real-time BMI Calculation**: Instantly calculates BMI based on predicted measurements
- **Health Category Classification**: Categorizes BMI into standard health categories
- **Modern UI**: Sleek, responsive interface with dark theme
- **Drag & Drop Upload**: Easy image upload with drag & drop functionality
- **Mobile Responsive**: Works seamlessly on all devices

## 🚀 Live Demo

Visit the live application at: [BMI Calculator](https://bmi-calculator.onrender.com)

## 📋 Prerequisites

- Python 3.9+
- TensorFlow 2.15.0
- Flask 2.0.1
- Other dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Oblivious19/Bmi-Calculator.git
   cd Bmi-Calculator
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the model file:

   Due to file size limitations, the model file is not included in the repository. You can download it from [Google Drive](https://drive.google.com/file/d/18Qv7ipbhfAer0XxfnBxPc2vUNedEbJhj/view?usp=sharing) and place it in the root directory of the project with the name `custom_cnn_bmi_model_final.keras`.

5. Run the application:

   ```bash
   python app.py
   ```

6. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## 💻 Usage

1. Upload a full-body photo using the drag & drop interface or click to select a file
2. Wait for the AI to process the image and predict height and weight
3. View your predicted BMI and health category
4. Check the detailed metrics and explanations

## 🔧 Technical Details

### Model Architecture

The application uses a custom CNN model trained to predict height and weight from full-body images. The model processes images with the following steps:

1. Image preprocessing (resize to 224x224, normalization)
2. Feature extraction through convolutional layers
3. Prediction of height and weight values
4. Post-processing to validate and adjust predictions to realistic ranges

### Height Validation

The application includes sophisticated height validation to ensure predictions are within realistic human ranges:

- Minimum height: 1.4m (140cm)
- Maximum height: 2.1m (210cm)
- Automatic conversion from centimeters to meters
- Scaling of unrealistic predictions using multiple factors

### BMI Categories

- Underweight: BMI < 18.5
- Normal weight: 18.5 ≤ BMI < 25
- Overweight: 25 ≤ BMI < 30
- Obese: BMI ≥ 30

## 🚀 Deployment

This application is configured for deployment on Render:

1. Fork this repository
2. Create a new Web Service on Render
3. Connect your forked repository
4. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Environment: Python 3.9

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- TensorFlow team for the deep learning framework
- Flask team for the web framework
- All contributors and users of this application

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.

---

Made with ❤️ by [Your Name]
