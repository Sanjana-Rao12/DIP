from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import cv2  # OpenCV for DIP techniques
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = os.path.join('models', 'resnet50_weather_classification.h5')  # Adjust path if necessary
model = load_model(MODEL_PATH)

# Define class labels
class_names = ['Shine', 'fogsmog', 'lighting', 'rain', 'sandstorm', 'snow']

# Image preprocessing function
def prepare_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to apply DIP techniques
def apply_dip_techniques(image):
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    processed_image = cv2.resize(img_array, (224, 224))
    
    # Fourier Transform
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    fourier_transform = np.fft.fft2(gray_image)
    fourier_transform_shifted = np.fft.fftshift(fourier_transform)
    magnitude_spectrum = 20 * np.log(np.abs(fourier_transform_shifted))
    
    # Sobel Edge Detection
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    
    # Histogram Equalization
    equalized_image = cv2.equalizeHist(gray_image)
    
    return magnitude_spectrum, sobel_edges, equalized_image

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
        
        # Apply DIP techniques
        magnitude_spectrum, sobel_edges, equalized_image = apply_dip_techniques(image)

        # Convert processed images to base64 for displaying on the web
        magnitude_spectrum_b64 = img_to_base64(Image.fromarray(np.uint8(magnitude_spectrum)))
        sobel_edges_b64 = img_to_base64(Image.fromarray(np.uint8(sobel_edges)))
        equalized_image_b64 = img_to_base64(Image.fromarray(np.uint8(equalized_image)))

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
            'magnitude_spectrum': magnitude_spectrum_b64,
            'sobel_edges': sobel_edges_b64,
            'equalized_image': equalized_image_b64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
