import cv2
import mediapipe as mp
import pickle
import numpy as np
import string

# Create a list of labels for 1 to 9 and A to Z

labels_dict={0:'1',1:'2',2:'3'}

# Create the labels_dict
# labels_dict = {i: label for i, label in enumerate(labels_list)}
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    data_aux = []
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame if needed
    # frame_rgb = cv2.resize(frame_rgb, (width, height))

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

        # Check if hands are detected before trying to predict
        if data_aux:
            # Reshape the input to match the expected number of features
            input_data = np.asarray(data_aux).reshape(1, -1)
            

            # Get the number of features from the input data shape
            #num_features = input_data.shape[1]
            
            # Ensure the input data has the correct shape
            #if num_features == model_dict.get('num_features', None):
                # Predict the class (or use predict_proba for probability estimates)
            prediction = model.predict(input_data)
            predicted_character = labels_dict[int(prediction[0])]
            print(f"Predicted class: {predicted_character}")
        else:
            print("mismatch")
           
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
