import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('eeg_eog_semg_dataset_large.csv')

# Preprocess the data
# Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Standardize the signal data
scaler = StandardScaler()
signals = df[['Signal1', 'Signal2', 'Signal3']]
signals_scaled = scaler.fit_transform(signals)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    signals_scaled, df['Label'], test_size=0.2, random_state=42
)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Check the unique classes in y_test and y_pred
unique_y_test = np.unique(y_test)
unique_y_pred = np.unique(y_pred)

# Ensure the target_names correspond to the classes in y_test
target_names = [label_encoder.inverse_transform([i])[0] for i in unique_y_test]

# Detailed classification report
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    labels=unique_y_test,
    target_names=target_names,
    zero_division=1  # Handle any cases where no samples exist for a class
))

# Output results
for i, (true_label, predicted_label) in enumerate(zip(y_test, y_pred)):
    true_label_name = label_encoder.inverse_transform([true_label])[0]
    predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]

    if true_label == predicted_label:
        print(f"Test Sample {i + 1}: Signals matched. Access granted. (True Label: {true_label_name}, Predicted Label: {predicted_label_name})")
    else:
        print(f"Test Sample {i + 1}: Signals did not match. Access denied. (True Label: {true_label_name}, Predicted Label: {predicted_label_name})")

# Visual feedback (optional)
plt.figure(figsize=(10, 6))
plt.plot(X_test[:50], label='Test Signals (First 50 Samples)')
plt.title('Test Signals')
plt.xlabel('Sample Index')
plt.ylabel('Signal Value (Standardized)')
plt.legend()
plt.show()
