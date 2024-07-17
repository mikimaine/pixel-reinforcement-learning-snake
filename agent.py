import random
from collections import deque

import numpy as np
import torch

from brain import DQN, BrainTrainer, device
from game import Game, Point, BLOCK_SIZE, Direction
from real_time_plot import setup_plot

class Agent:
    def __init__(self, settings):
        self.num_games = 0
        self.epsilon = 0
        self.gamma = settings['gamma']
        self.memory = deque(maxlen=settings['max_memory'])
        self.model = DQN(settings['state_size'], settings['hidden_layer_size'], settings['action_size'])
        self.model.to(device=device)
        self.trainer = BrainTrainer(self.model, learning_rate=settings['learning_rate'], gamma=self.gamma)
        self.epsilon_decay = settings['epsilon_decay']
        self.batch_size = settings['batch_size']
        self.position_history = deque(maxlen=100)  # Track the last 100 positions to detect spirals

    def get_state(self, game):
        head = game.snake[0]

        # Possible future positions
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        # Current direction
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_right and game.is_collision(point_right)) or
            (dir_left and game.is_collision(point_left)) or
            (dir_up and game.is_collision(point_up)) or
            (dir_down and game.is_collision(point_down)),

            # Danger right
            (dir_up and game.is_collision(point_right)) or
            (dir_down and game.is_collision(point_left)) or
            (dir_left and game.is_collision(point_up)) or
            (dir_right and game.is_collision(point_down)),

            # Danger left
            (dir_down and game.is_collision(point_right)) or
            (dir_up and game.is_collision(point_left)) or
            (dir_right and game.is_collision(point_up)) or
            (dir_left and game.is_collision(point_down)),

            # Move direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            # Food location
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y  # Food down
        ]

        return np.array(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_batch = random.sample(self.memory, self.batch_size)
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(0, self.epsilon_decay - self.num_games)
        action = [0, 0, 0]  # [straight, left, right]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float, device=device)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            action[move] = 1
        return action

    def detect_spiral(self, position):
        """Detect if the snake is in a spiral by checking if it revisits the same positions."""
        if position in self.position_history:
            return True
        self.position_history.append(position)
        return False

def train():
    # Agent settings
    agent_settings = {
        'state_size': 11,
        'hidden_layer_size': 256,
        'action_size': 3,
        'max_memory': 100_000,
        'epsilon_decay': 80,
        'learning_rate': 0.001,
        'batch_size': 1000,
        'gamma': 0.9
    }

    agent = Agent(agent_settings)
    game = Game()
    high_score = 0
    total_score = 0
    scores = []
    average_rewards = []
    losses = []
    epsilons = []
    episode_rewards = []

    agent.model.load('model.pth')
    ani = setup_plot(scores, average_rewards, losses, epsilons)

    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)

        next_state = agent.get_state(game)

        # Detect spiral and apply penalty
        if agent.detect_spiral(game.snake[0]):
            reward -= 5  # Apply a penalty for getting into a spiral

        agent.train_short_memory(state, action, reward, next_state, done)
        agent.remember(state, action, reward, next_state, done)

        episode_rewards.append(reward)

        if done:
            game.reset_game()
            agent.num_games += 1
            agent.train_long_memory()

            if score > high_score:
                high_score = score
                agent.model.save('model.pth')

            avg_reward = np.mean(episode_rewards)
            scores.append(score)
            average_rewards.append(avg_reward)
            epsilons.append(agent.epsilon)

            episode_rewards = []  # Reset episode rewards

            print(f'Game: {agent.num_games}, Score: {score}, High Score: {high_score}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}')