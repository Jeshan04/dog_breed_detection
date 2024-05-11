import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('dog_breed.h5')

class_list = ['Scottish Deerhound', 'Bernese Mountain Dog','Maltese Dog']

st.title("Dog Breed Predictor")

dog_img = st.file_uploader("Choose img", type="png")
submit = st.button('Predict')

if submit:
    if dog_img is not None:
        bytes = np.asarray(bytearray(dog_img.read()), dtype=np.uint8)
        opencv = cv2.imdecode(bytes,1)

        st.image(opencv,channels="BGR")
        opencv = cv2.resize(opencv,(224,224))
        opencv.shape=(1,224,224,3)
        Y_pred = model.predict(opencv)
        st.title(str("Dog breed is"+class_list[np.argmax(Y_pred)]))