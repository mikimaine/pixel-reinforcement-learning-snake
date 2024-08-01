from collections import deque
import torch

from brain import DQN, BrainTrainer, device
from game import Game
from trainingagent import get_state
from util import GameMode
from settings import GameSettings

class TestAgent:
    def __init__(self, settings: GameSettings):
        self.num_games = 0
        self.epsilon = 0
        self.gamma = settings.gamma
        self.memory = deque(maxlen=settings.max_memory)
        self.model = DQN(settings.state_size, settings.hidden_layer_size, settings.action_size)
        self.model.to(device=device)
        self.trainer = BrainTrainer(self.model, learning_rate=settings.learning_rate, gamma=self.gamma)
        self.epsilon_decay = settings.epsilon_decay
        self.batch_size = settings.batch_size
        self.position_history = deque(maxlen=100)  # Track the last 100 positions to detect spirals
        self.losses = []

    def get_action(self, state):
        action = [0, 0, 0]
        state_tensor = torch.tensor(state, dtype=torch.float, device=device)
        prediction = self.model(state_tensor)
        move = torch.argmax(prediction).item()
        action[move] = 1
        return action


def test(agent_settings):

    print(agent_settings)

    agent = TestAgent(agent_settings)
    game = Game(agent_settings)
    game.game_mode = GameMode.TEST

    # load the saved model
    agent.model.load('model.pth')
    agent.model.eval()
    while True:
        # get current state
        state = get_state(game)
        # play the game
        action = agent.get_action(state)
        reward, game_over, score = game.play_step(action)
        # ani = setup_plot(scores, average_rewards, agent.losses, epsilons)
        # if it finds something update the policy (learn)
        if game_over:
            # rect = pygame.Rect(25, 25, 100, 50)
            # sub = screen.subsurface(rect)
            # pygame.image.save(sub, "screenshot.jpg")
            print("Game Over!")
            break
