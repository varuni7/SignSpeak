import pickle
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
