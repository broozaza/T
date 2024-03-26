import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to simulate a single baccarat game
def simulate_baccarat_game():
    player_hand = np.random.randint(0, 10)
    banker_hand = np.random.randint(0, 10)
    
    if player_hand > banker_hand:
        outcome = 0  # Player win
    elif player_hand < banker_hand:
        outcome = 1  # Banker win
    else:
        outcome = 2  # Tie
    
    return player_hand, banker_hand, outcome

# Simulate multiple baccarat games to generate data
def simulate_baccarat_data(num_games):
    print("Simulating baccarat games...")
    data = []
    for i in range(num_games):
        if i % 1000000 == 0:
            print(f"Simulated {i} games out of {num_games}...")
        player_hand, banker_hand, outcome = simulate_baccarat_game()
        data.append([player_hand, banker_hand, outcome])
    print("Simulation completed.")
    return np.array(data)

# Generate synthetic baccarat data
num_games = 5000000  # Adjust the number of games as needed
synthetic_data = simulate_baccarat_data(num_games)

# Save the synthetic data to a .npy file
print("Saving synthetic baccarat data to file...")
np.save("baccarat_data.npy", synthetic_data)
print("Data saved successfully.")

# Load baccarat data
baccarat_data = np.load("baccarat_data.npy", allow_pickle=True)

# Assuming baccarat_data is a numpy array with shape (num_samples, num_features)
# Split the data into features and labels
X = baccarat_data[:, :-1]  # Features (player_hand, banker_hand)
y = baccarat_data[:, -1]   # Labels (outcome)

# Function to train the model and check win rate
def train_and_check(X_train, X_test, y_train, y_test, threshold=0.8):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Main training loop
best_models = []
best_accuracy = 0.0
threshold_count = 0
while threshold_count < 3:  # Check if win rate exceeds 80% three times
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model, accuracy = train_and_check(X_train, X_test, y_train, y_test)
    if accuracy > 0.8:
        threshold_count += 1
        best_models.append((model, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy

# Save the best models
for i, (model, accuracy) in enumerate(best_models):
    with open(f"baccarat_model_{i}_accuracy_{accuracy:.2f}.pkl", "wb") as f:
        pickle.dump(model, f)
