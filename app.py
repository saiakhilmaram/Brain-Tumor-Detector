import streamlit as st

import PIL.Image

st.title("Image Classification using CNN Architecture")

st.header("Brain Tumor Prediction")

st.text("Upload a brain MRI scan image to classify as Brain Tumor or No Tumor(Healthy)")

st.write('Find some MRI images here : https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection')

from img_classification import classification

uploaded_file = st.file_uploader("Upload a brain MRI scan", type="jpg")

if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)

        st.image(image, caption='Uploaded MRI scan', use_column_width=True)

        st.write("")

        st.write("Classifying...")

        label = classification(image, 'best_model.h5')

        if label == 0:

            st.write("The MRI scan has a brain tumor")

        else:

            st.write("The MRI scan is of healthy brain")
