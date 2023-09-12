import streamlit as st

st.write("Hello world")

st.number_input("Enter a number", min_value=0, max_value=100, value=50, step=5)

image_path = st.text_input("Enter image path")

if image_path:
    st.image(image_path, width=300)


