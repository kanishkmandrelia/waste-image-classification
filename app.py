import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

# Load the saved Keras model
model = load_model("model_CNN.h5")  # Use .h5 format

# Print expected input shape for debugging
print(f"Expected input shape: {model.input_shape}")

# Class labels
class_labels = ['Organic', 'Recyclable']  # Adjust as needed

st.title("Waste Classification App")
st.write("Upload an image, and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB format
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = np.array(image)
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    print(f"Processed image shape: {image.shape}")  # Debugging step
    # image = image.reshape(1, -1)

    # Make prediction
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_name = class_labels[class_index]

    st.write(f"### Prediction: {class_name}")
