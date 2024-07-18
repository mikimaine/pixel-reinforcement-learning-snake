import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Check if GPU is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQN(nn.Module):
    """Deep Q-Network model"""

    def __init__(self, input_state_size, hidden_feature_state_size, output_action_size):
        super(DQN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_state_size, hidden_feature_state_size),
            nn.ReLU(),
            nn.Linear(hidden_feature_state_size, output_action_size)
        )

    def forward(self, x):
        """Forward pass through the network"""
        return self.linear_relu_stack(x)

    def save(self, path):
        """Save the model to a file"""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load the model from a file if it exists"""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            print(f'Model loaded from {path}')
        else:
            print(f'No model file found at {path}, starting with a new model')


class BrainTrainer:
    """Trainer for the DQN model"""

    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        """Train the model with a batch of experiences"""

        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)

        # Handle single state case by adding batch dimension
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            done = (done,)

        # Forward pass: Compute predicted Q-values
        predicted_q_values = self.model(state)

        # Clone the predictions to avoid modifying them during the update
        target_q_values = predicted_q_values.clone()

        for i in range(len(done)):
            # Compute the target Q-value
            q_update = reward[i]
            if not done[i]:
                q_update = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target_q_values[i][torch.argmax(action[i]).item()] = q_update

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Compute the loss
        loss = self.loss_fn(target_q_values, predicted_q_values)

        # Backward pass: Compute gradients
        loss.backward()

        # Update model parameters
        self.optimizer.step()

        return loss.item()
