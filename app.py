import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model
import cv2

st.set_page_config(
    page_title = 'Patacotrón',
    layout= 'wide',
    initial_sidebar_state = 'collapsed',
    menu_items = {
        "About" : 'Proyecto ideado para la investigación de "Clasificación de imágenes de una sola clase con algortimos de Inteligencia Artificial".',
        "Report a Bug" : 'https://docs.google.com/forms/d/e/1FAIpQLScH0ZxAV8aSqs7TPYi86u0nkxvQG3iuHCStWNB-BoQnSW2V0g/viewform?usp=sf_link'
    }
)


col_a, col_b, col_c = st.columns(3)

with col_a:
        st.write(' ')

with col_b:
        st.title("Patacotrón")
        st.markdown("Los modelos no están en orden de eficacia, sino en orden de creación.\nLos modelos 1.5, 1.6 y 1.8 tienden a dar mejores resultados.")

with col_c:
        st.write(' ')

model_list = []
for i in range(9):
    model_list.append(f'models/ptctrn_v1.{i+1}.h5')

# Create a dropdown menu to select the model
model_choice = st.selectbox("Seleccione un modelo de clasificación", model_list)

# Load the selected image classification model
model = load_model(model_choice)

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
        st.write('Si los resultados no fueron los esperados, por favor, despliga la barra lateral y entra al botón "Report a Bug"')
        st.image(img)

    with col3:
        st.write(' ')

