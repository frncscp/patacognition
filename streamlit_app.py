import streamlit as st
import cv2
from PIL import Image

# Load the image classification model
model = load_model("ptctrn_v1.7.h5")

# Create a button to access the user's camera
camera_button = st.button("Abrir cámara")

if camera_button:
    # Use OpenCV to access the user's camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the camera
        _, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show the frame in the Streamlit app
        st.image(image=gray, caption="Capturando imagen...", use_column_width=True)

        # Check if the user clicked the 'Capture' button
        capture_button = st.button("Capturar")
        if capture_button:
            # Resize the image to the required dimensions (224x224)
            image = cv2.resize(gray, (224, 224))

            # Convert the image to a PIL Image object
            image = Image.fromarray(image)

            # Pass the image to the model
            prediction = model.predict(image)

            # Show the prediction in the Streamlit app
            st.write("Predicción:", prediction)

            # Break the loop
            break

# Release the camera
cap.release()
