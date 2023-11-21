import cv2
import mediapipe as mp
import pickle
import numpy as np
import string
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

data = []
labels = []
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
DATA_DIR = r'C:\Proj\data'
width, height = 224, 224  # Specify the desired resolution

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        
        # Resize the image to a specific resolution
        img_rgb = cv2.resize(img, (width, height))
        
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            data.append(data_aux)
            labels.append(dir_)
            print(f"Processed image: {os.path.join(DATA_DIR, dir_, img_path)}")

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

data_dict = pickle.load(open('./data.pickle', 'rb'))
print(data_dict.keys())

# Extract features (data) and labels from the loaded data
data = data_dict['data']
labels = data_dict['labels']

# Find the maximum number of features among all data points
max_features = max(len(point) for point in data)

# Pad or truncate each data point to have the same length
data = np.array([point + [0] * (max_features - len(point)) for point in data])

# Convert string labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and fit the XGBoost model
model = XGBClassifier()
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Reverse transformation to get original labels for accuracy calculation (if needed)
y_test_original = label_encoder.inverse_transform(y_test)
y_pred_original = label_encoder.inverse_transform(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)



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
