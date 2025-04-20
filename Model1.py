import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Create a directory to save the model in the current directory
output_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Model will be saved in: {output_dir}")

# Load CSV data
csv_file = r"C:\Users\shrey\OneDrive\Desktop\Working Projects\BMI\augmented_images_metadata.csv"
df = pd.read_csv(csv_file)

# Convert height in the format HHH (e.g., 410 for 4 feet 10 inches) to meters
def convert_height_to_meters(height):
    feet = height // 100  # Get the hundredth digit (feet)
    inches = height % 100  # Get the rest (inches)
    total_inches = feet * 12 + inches
    return total_inches * 0.0254  # Convert inches to meters

# Convert weight from pounds to kilograms
def convert_weight_to_kg(weight):
    return weight * 0.453592

# Calculate BMI using the formula: weight (kg) / height (m)^2
df['Height_m'] = df['Height'].apply(convert_height_to_meters)
df['Weight_kg'] = df['Weight'].apply(convert_weight_to_kg)
df['BMI'] = df['Weight_kg'] / (df['Height_m'] ** 2)

# Create a data generator for training and validation sets
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=r'C:\Users\shrey\OneDrive\Desktop\Working Projects\BMI\Project_Photo_Augmented',
    x_col='Filename',
    y_col=['Height_m', 'Weight_kg'],  # Use height and weight as target variables
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    subset='training'
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=r'C:\Users\shrey\OneDrive\Desktop\Working Projects\BMI\Project_Photo_Augmented',
    x_col='Filename',
    y_col=['Height_m', 'Weight_kg'],  # Use height and weight as target variables
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    subset='validation'
)

# Build a custom CNN model with Dropout, Batch Normalization, and L2 regularization
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    
    Dense(2)  # Output layer for height and weight
])

# Compile the model with Adam optimizer with momentum
model.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss='mse', metrics=['mae'])

# Setup callbacks
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(output_dir, 'best_custom_cnn_model.keras'),  # Save the best model during training
    save_best_only=True, 
    monitor='val_loss', 
    mode='min', 
    verbose=1
)

early_stopping_cb = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True, 
    verbose=1
)

# Train the model
model.fit(
    train_generator, 
    validation_data=val_generator, 
    epochs=10, 
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# Evaluate and save the final model
val_loss, val_mae = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss}, Validation MAE: {val_mae}')

# Predict on validation data
val_predictions = model.predict(val_generator)
val_true = val_generator.labels

# Calculate MAE and RMSE
mae = mean_absolute_error(val_true, val_predictions)
rmse = np.sqrt(mean_squared_error(val_true, val_predictions))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the model
model_save_path = os.path.join(output_dir, 'custom_cnn_bmi_model_final.keras')
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
