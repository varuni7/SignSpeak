import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the action labels
labels_dict = {0: '1', 1: '2', 2: '3'}

# Function to perform hand action detection
def detect_hand_action(frame):
    data_aux = []
    x_ = []
    y_ = []

    # Process the frame with MediaPipe Hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calculate bounding box coordinates
        x1 = int(min(x_) * frame.shape[1]) - 10
        y1 = int(min(y_) * frame.shape[0]) - 10
        x2 = int(max(x_) * frame.shape[1]) - 10
        y2 = int(max(y_) * frame.shape[0]) - 10

        # Make prediction using the model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box and predicted action on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    return frame

# Streamlit app
st.title("Real-time Hand Action Detection")

# Open the webcam
cap = cv2.VideoCapture(0)

# Display the webcam feed and continuously update the processed frames in Streamlit
while st.checkbox("Capture Video"):
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame.")
        break

    processed_frame = detect_hand_action(frame)

    # Display the processed frame in Streamlit
    st.image(processed_frame, channels="BGR", use_column_width=True, caption="Hand Action Detection")

# Release the webcam and close Streamlit app when done
cap.release()
cv2.destroyAllWindows()
