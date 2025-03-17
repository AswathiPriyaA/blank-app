import numpy as np
from PIL import Image
import os
import random

# Example function to load model (replace with actual model)
def load_model():
    print("Model loaded!")
    return None  # Replace with actual model loading

# Example function to get recommendations (replace with your logic)
def get_recommendations(img):
    img_array = np.array(img)  # Convert to array if needed
    # Example: Use model to predict similar images
    # recommendations = model.predict(img_array)

    # For demo: return random images from the 'images' folder
    image_dir = 'images'
    all_images = os.listdir(image_dir)
    recommendations = random.sample(all_images, 3)
    
    return [f"{image_dir}/{img}" for img in recommendations]

# Load the model
model = load_model()
