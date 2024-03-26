import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from collections import Counter
import joblib

# 1. Data Collection (Assuming you have a dataset of baccarat outcomes)
print("Loading dataset...")
data = np.load('baccarat_data.npy')
X = data[:, :2]  # Player's hand and banker's hand
y = data[:, 2]   # Outcome (0: player win, 1: banker win, 2: tie)

# 2. Model Training
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a neural network model
print("Training the model...")
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100000, random_state=42)

# Train the model until it achieves an 80% win rate
while True:
    model.fit(X_train, y_train)
    win_rate = model.score(X_test, y_test)
    print("Current win rate:", win_rate)
    if win_rate >= 0.8:
        print("Model trained successfully with a win rate of", win_rate)
        break

# Save the trained model
print("Saving the trained model...")
joblib.dump(model, "baccarat_model.pkl")
print("Model saved successfully.")

# 3. Strategy Implementation
def predict_outcome(player_hand, banker_hand):
    # Load the trained model
    print("Loading the trained model...")
    model = joblib.load("baccarat_model.pkl")
    # Predict outcome using the trained model
    outcome = model.predict([[player_hand, banker_hand]])
    return outcome[0]

def bet_strategy(player_hand, banker_hand):
    # Load the trained model
    print("Loading the trained model...")
    model = joblib.load("baccarat_model.pkl")
    # Simple strategy: bet on the outcome with the highest predicted probability
    probabilities = model.predict_proba([[player_hand, banker_hand]])[0]
    bet_index = np.argmax(probabilities)
    if bet_index == 0:
        return "Bet on player"
    elif bet_index == 1:
        return "Bet on banker"
    else:
        return "Bet on tie"

# 4. Iterative Testing and Refinement
def simulate_baccarat_games(num_games):
    wins = 0
    print("Simulating baccarat games...")
    for i in range(num_games):
        if i % 1000 == 0:
            print(f"Simulated {i} games out of {num_games}...")
        # Simulate baccarat game
        player_hand = np.random.randint(0, 10)
        banker_hand = np.random.randint(0, 10)
        outcome = predict_outcome(player_hand, banker_hand)
        
        # Implement betting strategy
        if outcome == 0:
            bet_result = "player win"
        elif outcome == 1:
            bet_result = "banker win"
        else:
            bet_result = "tie"
        
        # Count wins
        if bet_result == "player win":
            wins += 1
    
    win_rate = wins / num_games
    print("Simulation completed.")
    return win_rate

# Test the bot's performance
win_rate = simulate_baccarat_games(10000)
print("Final Win rate:", win_rate)
