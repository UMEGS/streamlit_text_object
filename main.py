import streamlit as st
import numpy as np
from image_detector import detect_image, annotate_image, object_detection
from text_classification import classifications
from PIL import Image

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Object Detection", "Text Classification"])

if app_mode == "Object Detection":
    st.title("Object Detection")

    detection_mode = st.selectbox("Detection mode", ["Image", "Video"])

    if detection_mode == "Image":
        st.write("This is the Image detection page")

        image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if image_file is not None:
            image = Image.open(image_file)
            image = np.array(image)
            results = detect_image(image)
            annotated_image = annotate_image(image, results)
            st.image(annotated_image)
        else:
            st.write("Please upload an image")
    else:
        st.write("This is the Video detection page")

        image_element = st.image([], channels="BGR")
        frame_generator = object_detection(0)

        while True:
            try:
                frame = next(frame_generator)
                image_element.image(frame)
            except StopIteration:
                break


elif app_mode == "Text Classification":
    st.title("Text Classification")

    texts = st.text_area("Enter text here", height=100)

    if texts:
        res = classifications(texts)
        st.write(res)
        print(res)


else:
    st.error("Something has gone terribly wrong.")
