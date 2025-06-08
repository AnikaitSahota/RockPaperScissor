import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# ANSI color codes
RESET = "\033[0m"  # Reset to default color
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

# Text colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

player_choices = []
choices = ['rock', 'paper', 'scissors']

# Encoding maps
move_to_int = {'rock': 0, 'paper': 1, 'scissors': 2}
int_to_move = {0: 'rock', 1: 'paper', 2: 'scissors'}

# Neural network definition
class RPSPredictor(nn.Module):
    def __init__(self, input_size=3, embedding_dim=10, hidden_dim=16, output_size=3):
        super(RPSPredictor, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # only last time step
        return out

# Preprocess opponent history into sequences
def preprocess_data(history, seq_length=3):
    X, y = [], []
    for i in range(len(history) - seq_length):
        seq = history[i:i+seq_length]
        label = history[i+seq_length]
        X.append([move_to_int[m] for m in seq])
        y.append(move_to_int[label])
    return torch.tensor(X), torch.tensor(y)

# Train and predict function
def predict_next_move(history, seq_length=3, epochs=50):
    if len(history) <= seq_length:
        return random.choice(['rock', 'paper', 'scissors'])

    # Prepare data
    X, y = preprocess_data(history, seq_length)

    model = RPSPredictor()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Predict next move
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor([[move_to_int[m] for m in history[-seq_length:]]])
        output = model(last_seq)
        predicted_move = torch.argmax(output).item()

    # Return the counter move
    counter_moves = {0: 1, 1: 2, 2: 0}  # rock→paper, paper→scissors, scissors→rock
    return int_to_move[counter_moves[predicted_move]]

def get_computer_choice_v1():
    """Randomly selects rock, paper, or scissors for the computer."""
    return random.choice(choices)

def get_computer_choice_v2(player_history):
    """Computes the computer's choice based on the player's previous choices."""
    
    # If the player has a history, try to counter the most common previous choice
    if player_history:
        most_common_player_choice = max(set(player_history), key=player_history.count)
        
        # Counter strategy (the computer picks the option that beats the player's most frequent choice)
        if most_common_player_choice == 'rock':
            return 'paper'
        elif most_common_player_choice == 'paper':
            return 'scissors'
        elif most_common_player_choice == 'scissors':
            return 'rock'
    
    # If there's no history, pick randomly
    return random.choice(choices)

def get_computer_choice_v3() :
    return predict_next_move(opponent_history)
    

def get_computer_choice(player_history = []) :
    return get_computer_choice_v3(player_choices)

def get_player_choice():
    """Prompts the player to input their choice."""
    while True:
        choice = input("Enter your choice (rock, paper, or scissors): ").lower()
        if choice in choices:
            return choice
        else:
            print("Invalid choice. Please choose either rock, paper, or scissors.")

# def determine_winner(player_choice, computer_choice):
#     """Determines the winner based on the player's and computer's choices."""
#     if player_choice == computer_choice:
#         return "It's a tie!"
    
#     # Rock beats Scissors, Scissors beats Paper, Paper beats Rock
#     if (player_choice == 'rock' and computer_choice == 'scissors') or \
#        (player_choice == 'scissors' and computer_choice == 'paper') or \
#        (player_choice == 'paper' and computer_choice == 'rock'):
#         return "You win!"
    
#     return "Computer wins!"
def determine_winner(player_choice, computer_choice):
    """Determines the winner based on the player's and computer's choices."""
    if player_choice == computer_choice:
        return f"{YELLOW}It's a tie!{RESET}"
    
    # Rock beats Scissors, Scissors beats Paper, Paper beats Rock
    if (player_choice == 'rock' and computer_choice == 'scissors') or \
       (player_choice == 'scissors' and computer_choice == 'paper') or \
       (player_choice == 'paper' and computer_choice == 'rock'):
        return f"{GREEN}You win!{RESET}"
    
    return f"{RED}Computer wins!{RESET}"

def play_game():
    """Plays a single round of rock-paper-scissors."""
    print("Welcome to Rock, Paper, Scissors!")
    
    player_choice = get_computer_choice_v1() #get_player_choice()
    player_choices.append(player_choice)
    computer_choice = get_computer_choice()
    
    print(f"You chose {player_choice}, Computer chose {computer_choice}.")
    
    result = determine_winner(player_choice, computer_choice)
    print("##############" , result , "##############")
    return result

if __name__ == "__main__":
    computer_score = 0  # Counter to keep track of the number of times the computer wins
    total_games = 0
    while(total_games < 100) :
        try:
            result = play_game()
            if "Computer wins!" in result:
                computer_score += 2
            elif "tie" in result :
                computer_score += 1
            total_games += 1
            print("------------------")
        # except KeyboardInterrupt:  # Handle Ctrl+C gracefully
        except Exception as e:  # Catch any kind of exception
            print("\nGame interrupted by the user.")
            break  # Exit the while loop when Ctrl+C is pressed
    print(f"Computer Score: {computer_score}/{total_games*2} ({computer_score/total_games*50})%")