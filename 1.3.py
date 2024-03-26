import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

# Load dataset
print("Loading dataset...")
data = np.load('baccarat_data.npy')
X = data[:, :2]  # Player's hand and banker's hand
y = data[:, 2]   # Outcome (0: player win, 1: banker win, 2: tie)

# Split data into train and test sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=10000, random_state=42)

# Train the model until it achieves a certain accuracy threshold or reaches a maximum number of iterations
accuracy_threshold = 0.80
max_iterations = 10000
iteration = 0

while iteration < max_iterations:
    # Train the model with partial_fit
    model.partial_fit(X_train, y_train, classes=np.unique(y))
    iteration += 1

    # Evaluate the model on the test set
    accuracy = model.score(X_test, y_test)
    print(f"Iteration {iteration}: Accuracy = {accuracy:.2f}")

    # Check if the accuracy threshold is reached
    if accuracy >= accuracy_threshold:
        print("Model trained successfully with an accuracy of", accuracy)
        break

# Save the trained model
print("Saving the trained model...")
joblib.dump(model, "baccarat_model.pkl")
print("Model saved successfully.")
