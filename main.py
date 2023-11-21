import streamlit as st
import torch
from PIL import Image
import io
import cv2
import numpy as np


@st.cache
def load_model():
    @~Tanisha load the model ehre
    return custom_yolov7_model


def get_prediction(img_bytes, model):
    img = Image.open(io.BytesIO(img_bytes))
    results = model(img, size=640)
    return results

# Analyze the image and display predictions
def analyse_image(img, model):
    img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
    result = get_prediction(img_bytes, model)
    result.render()

    for img in result.imgs:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_arr = cv2.imencode('.jpg', RGB_img)[1]
        st.image(im_arr.tobytes(), channels="RGB")

    result_list = list((result.pandas().xyxy[0])["name"])
    return result_list

# Display letters to form the final word
def display_letters(letters_list):
    word = ''.join(letters_list)
    path_file = "/workspace/word_file.txt"
    with open(path_file, "a") as f:
        f.write(word)
    return path_file

def main():
    st.title("Real-time Sign Language Prediction App")

    
    model = load_model()

    
    cap = cv2.VideoCapture(0)

    st.header("Real-time Sign Language Prediction")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("Error capturing video stream.")
            break

        # Resize the frame for prediction
        resized_frame = cv2.resize(frame, (640, 480))
        result_list = analyse_image(resized_frame, model)

        # Display the recognized letters
        st.write("Recognized Letters: ", result_list)

        # Display the original webcam feed
        st.image(frame, channels="BGR")

        # Check for a stop signal
        if st.button("Stop"):
            break

    # Release the webcam and close the streamlit app
    cap.release()

if _name_ == "_main_":
    main()