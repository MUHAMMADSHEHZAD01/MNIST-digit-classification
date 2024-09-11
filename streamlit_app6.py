import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import io

# Load your Keras model (replace 'your_model.h5' with the path to your model file)
model = load_model('model.h5')

# Function to generate predictions and visualize results
def predict_and_plot(image):
    # Preprocess the image (assuming the model expects 224x224 RGB images)
    image = tf.image.resize(image, [224, 224])
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize

    # Make predictions
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Create a plot
    fig, ax = plt.subplots()
    ax.imshow(image[0])
    ax.set_title(f'Predicted Class: {predicted_class}')
    ax.axis('off')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Streamlit UI
st.title("Keras Model Prediction")

st.write("Upload an image to get a prediction from the model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = tf.io.decode_image(uploaded_file.read(), channels=3)
    
    # Get prediction and plot
    buf = predict_and_plot(image)
    
    # Display the image and plot
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.image(buf, caption='Model Prediction Visualization.', use_column_width=True)