import numpy as np
import joblib

# Load the trained model
model = joblib.load("baccarat_model_2_accuracy_1.00.pkl")

# Function to predict baccarat outcome
def predict_outcome(player_hand, banker_hand):
    # Predict outcome using the trained model
    outcome = model.predict([[player_hand, banker_hand]])
    if outcome == 0:
        return "Player win"
    elif outcome == 1:
        return "Banker win"
    else:
        return "Tie"

# Function to implement a betting strategy based on the last 10 game results
def bet_strategy(last_10_results):
    # Load the last 10 game results
    last_results_numerical = [1 if r == 'B' else 0 for r in last_10_results]
    
    # Randomly generate player and banker hands
    player_hand = np.random.randint(0, 10)
    banker_hand = np.random.randint(0, 10)
    
    # Predict outcome using the learned bot
    outcome = predict_outcome(player_hand, banker_hand)
    
    # Implement betting strategy based on the predicted outcome
    if outcome == "Player win":
        bet = "B" if model.predict([[player_hand, banker_hand]])[0] == 0 else "P"
    elif outcome == "Banker win":
        bet = "P" if model.predict([[player_hand, banker_hand]])[0] == 1 else "B"
    else:
        bet = last_10_results[-1]  # Bet same as the last result in case of tie
    
    return bet

# Main function to play the game
def play_game():
    # Ask users to input the last 10 results
    last_10_results = input("Enter the last 10 game results (B/P/T): ").upper()
    
    # Initialize total wins and losses
    total_wins = 0
    total_losses = 0
    
    # Main loop to allow the user to make betting decisions
    while True:
        # Implement betting strategy
        bet = bet_strategy(last_10_results)
        print("Betting on:", bet)
        
        # Ask user for the next digit
        next_digit = input("Enter the next digit (B/P/T): ").upper()
        
        # Validate user input
        if next_digit not in ['B', 'P', 'T']:
            print("Invalid input. Please enter 'B', 'P', or 'T'.")
            continue
        
        # Update last 10 results with the new digit and remove the oldest digit
        last_10_results = last_10_results[1:] + next_digit
        
        # Check if the user won or lost
        if next_digit == bet:
            total_wins += 1
            print("You won!")
        else:
            total_losses += 1
            print("You lost.")
        
        # Print total wins and losses
        print("Total wins:", total_wins)
        print("Total losses:", total_losses)

# Start the game
play_game()
