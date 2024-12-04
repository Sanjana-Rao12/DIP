from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = os.path.join('models', 'resnet50_weather_classification.h5')  # Adjust the path if necessary
model = load_model(MODEL_PATH)

# Define class labels
class_names = ['Shine', 'fogsmog', 'lighting', 'rain', 'sandstorm', 'snow']

# Image preprocessing function for prediction
def prepare_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to apply Unsharp Masking and Guided Filtering
def apply_enhancement(image):
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Resize image for processing
    processed_image = cv2.resize(img_array, (224, 224))

    # Step 1: Unsharp Masking
    blurred = cv2.GaussianBlur(processed_image, (5, 5), 1.0)
    unsharp_mask = cv2.addWeighted(processed_image, 1.5, blurred, -0.5, 0)

    # Step 2: Guided Filtering
    try:
        unsharp_mask_float = unsharp_mask.astype(np.float32) / 255.0
        guided_filtered = cv2.ximgproc.guidedFilter(
            guide=unsharp_mask_float, src=unsharp_mask_float, radius=8, eps=0.01
        )
        guided_filtered = (guided_filtered * 255).astype(np.uint8)
    except AttributeError:
        # Fallback to bilateral filter if guided filtering is unavailable
        guided_filtered = cv2.bilateralFilter(unsharp_mask, d=9, sigmaColor=75, sigmaSpace=75)

    return unsharp_mask, guided_filtered

# Function to convert image to base64
def img_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load the uploaded image
        image = Image.open(file)

        # Apply Unsharp Masking and Guided Filtering
        unsharp_mask, guided_filtered = apply_enhancement(image)

        # Convert processed images to base64 for displaying on the web
        unsharp_mask_b64 = img_to_base64(Image.fromarray(cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2RGB)))
        guided_filtered_b64 = img_to_base64(Image.fromarray(cv2.cvtColor(guided_filtered, cv2.COLOR_BGR2RGB)))

        # Preprocess the image for prediction
        prepared_image = prepare_image(image)

        # Predict the output
        predictions = model.predict(prepared_image)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]

        # Get the confidence score
        confidence = float(np.max(predictions))

        # Return the prediction result along with processed images
        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence,
            'unsharp_mask': unsharp_mask_b64,
            'guided_filtered': guided_filtered_b64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
