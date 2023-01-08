import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model
import cv2

st.title("Proyecto Patacotrón")
st.markdown("por github.com/frncscp")

# Load the image classification model
model = load_model("ptctrn_v1.7.h5")

# Set the image dimensions
IMAGE_WIDTH = IMAGE_HEIGHT = 224

# Create a file uploader widget
uploaded_file = st.file_uploader("Elige una imagen...", type= ['jpg','png', 'jpeg', 'jfif', 'webp', 'heic'])

if uploaded_file is not None:
    # Load the image and resize it to the required dimensions
    img = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Convert the image to RGB and preprocess it for the model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255

    # Pass the image to the model and get the prediction
    y_gorrito = model.predict(np.expand_dims(img, 0))

    # Show the image
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.write(f'La probabilidad de que la imagen tenga un patacón es del: {round(float(y_gorrito), 2)*100}%')
        st.image(img)

    with col3:
        st.write(' ')

#streamlit run d:/ptctrn/app.py  --server.fileWatcherType none