import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler

# Create a directory to save the model
output_dir = r'C:\Users\shrey\OneDrive\Desktop\BMI_Output'
os.makedirs(output_dir, exist_ok=True)

# Load CSV data
csv_file = r"C:\Users\shrey\OneDrive\Desktop\augmented_images_metadata.csv"
df = pd.read_csv(csv_file)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the height and weight columns
df[['Height_cm', 'Weight_kg']] = scaler.fit_transform(df[['Height_cm', 'Weight_kg']])

# Create a data generator for training and validation sets
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=r'C:\Users\shrey\OneDrive\Desktop\Project_Photo_Augmented',
    x_col='Filename',
    y_col=['Height_cm', 'Weight_kg'],
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    subset='training'
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=r'C:\Users\shrey\OneDrive\Desktop\Project_Photo_Augmented',
    x_col='Filename',
    y_col=['Height_cm', 'Weight_kg'],
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    subset='validation'
)

# Configure TensorFlow to use only the CPU
tf.config.set_visible_devices([], 'GPU')  # Hides all GPUs from TensorFlow

# Build an enhanced CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(2)  # Output layer for height and weight
])

# Compile the model with updated optimizer and learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Setup callbacks
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(output_dir, 'best_custom_cnn_model.keras'), 
    save_best_only=True, 
    monitor='val_loss', 
    mode='min', 
    verbose=1
)

early_stopping_cb = EarlyStopping(
    monitor='val_loss', 
    patience=8, 
    restore_best_weights=True, 
    verbose=1
)

# Learning rate reduction on plateau
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)

# Train the model with the updated architecture and callbacks
model.fit(
    train_generator, 
    validation_data=val_generator, 
    epochs=20, 
    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
)

# Evaluate and save the final model
val_loss, val_mae = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss}, Validation MAE: {val_mae}')

model_save_path = os.path.join(output_dir, 'enhanced_custom_cnn_bmi_model_final.keras')
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Function to preprocess a single image for prediction
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale pixel values
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

# Load the trained model
model = tf.keras.models.load_model(model_save_path)

# Predict height and weight from an image
image_path = r'C:\Users\shrey\OneDrive\Desktop\BMI_Calculator\temp\410-100_Cece_L1.jpg'  # Replace with your test image path
img_array = preprocess_image(image_path)
height_cm, weight_kg = model.predict(img_array)[0]

# Calculate BMI
height_m = height_cm / 100  # Convert cm to m
bmi = weight_kg / (height_m ** 2)

print(f"Predicted Height: {height_cm:.2f} cm, Predicted Weight: {weight_kg:.2f} kg, Calculated BMI: {bmi:.2f}")
