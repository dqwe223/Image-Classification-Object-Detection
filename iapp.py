from transformers import ViTForImageClassification
from PIL import Image
import requests
import streamlit as st
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from transformers import BeitImageProcessor, BeitForImageClassification
import torch

st.sidebar.title("Image Classification & Object Detection App")
st.sidebar.write("Simply upload your image, and in just a few minutes, receive accurate and insightful results Display. This Application can do Image Classification.")

st.sidebar.title("Application Instructions:")
st.sidebar.write("1.Click the Browse Files button to upload your desired image. \n\n2.Press the Classification or Object Detection button . \n\n3.Wait a few minutes while our system analyzes your image. \n\n4.View the results and insights provided by Web App.")

st.subheader('Image Classification and Object Detection Web Application')
uploaded_file = st.file_uploader("Choose an Your Image...", type='jpeg')
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(uploaded_file)

b1 = st.button('Classification', key = "1")
b2 = st.button('Object Detection', key = "2")



if b1:
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    
    predicted_class_idx = logits.argmax(-1).item()
    

    st.subheader('Image Prediction:')
    st.write(model.config.id2label[predicted_class_idx])

    st.balloons()

if b2:
    processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    
    predicted_class_idx = logits.argmax(-1).item()
    print("Prediction class:", model.config.id2label[predicted_class_idx])
    st.write(model.config.id2label[predicted_class_idx])

