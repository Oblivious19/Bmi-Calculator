import os
import shutil

# Source model file
source_model = r"C:\Users\shrey\OneDrive\Desktop\Working Projects\BMI\BMI_Output\custom_cnn_bmi_model_final.keras"

# Destination (current directory)
dest_dir = os.path.dirname(os.path.abspath(__file__))
dest_model = os.path.join(dest_dir, "custom_cnn_bmi_model_final.keras")

# Copy the file
if os.path.exists(source_model):
    print(f"Copying model from {source_model} to {dest_model}")
    shutil.copy2(source_model, dest_model)
    print("Model copied successfully!")
else:
    print(f"Error: Source model file not found at {source_model}")
    
    # Try the enhanced model
    source_model = r"C:\Users\shrey\OneDrive\Desktop\Working Projects\BMI\BMI_Output\enhanced_custom_cnn_bmi_model_final.keras"
    if os.path.exists(source_model):
        print(f"Copying enhanced model from {source_model} to {dest_model}")
        shutil.copy2(source_model, dest_model)
        print("Enhanced model copied successfully!")
    else:
        print(f"Error: Enhanced model file not found at {source_model}") 