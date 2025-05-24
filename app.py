from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

model = YOLO(r"C:\Users\debra\Desktop\Final Code\New folder\FlipGears\runs\detect\train5\weights\best.pt")  # use raw string or forward slashes

st.title("Weld Defect Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Run detection
    results = model.predict(image)

    # Plot result
    result_image = results[0].plot()  # returns a numpy image with boxes
    st.image(result_image, caption='Detected Defects', use_column_width=True)
