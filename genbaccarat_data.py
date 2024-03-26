import numpy as np

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
num_games = 500000  # Adjust the number of games as needed
synthetic_data = simulate_baccarat_data(num_games)

# Save the synthetic data to a .npy file
print("Saving synthetic baccarat data to file...")
np.save("baccarat_data.npy", synthetic_data)
print("Data saved successfully.")
