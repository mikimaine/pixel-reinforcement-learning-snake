import torch
from torch import nn, optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DQN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer_size)
        self.layer2 = nn.Linear(hidden_layer_size, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x

    def save(self, path):
        pass

class BrainTrainer:
    def __init__(self, model, learning_rate, gamma, epsilon_decay=0, epsilon_min=0 ):
        self.lr = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, amsgrad=True)
        self.loss = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        prediction = self.model(state)
        copy = prediction.clone()
        for i in range(len(done)):
            QNew = reward[i]
            if not done[i]:
                QNew = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            copy[i][torch.argmax(action[i])] = QNew

        self.optimizer.zero_grad()
        loss = self.loss(copy, prediction)
        loss.backward()
        self.optimizer.step()