import random
from collections import deque

import numpy as np
import torch

from brain import DQN, BrainTrainer, device
from game import Game, Point, BLOCK_SIZE, Direction

MAX_MEMORY = 1000
EPSILON_GAMES = 80
STATE = 11
HIDDEN_LAYER_NEURON = 200
OUTPUT = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 1024

class Agent:
    def __init__(self):
        self.no_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(STATE, HIDDEN_LAYER_NEURON, OUTPUT)
        self.model.to(device=device)
        self.trainer = BrainTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        # current snake head position
        head = game.snake[0]

        # future head position of snake
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # direction is relative to the direction of the head of snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # is there a collision going Straight
            (dir_r and game.collision(point_r)) or
            (dir_l and game.collision(point_l)) or
            (dir_u and game.collision(point_u)) or
            (dir_d and game.collision(point_d)),
            # is there a collision going Right
            (dir_u and game.collision(point_r)) or
            (dir_d and game.collision(point_l)) or
            (dir_l and game.collision(point_u)) or
            (dir_r and game.collision(point_d)),
            # is there a collision going Left
            (dir_d and game.collision(point_r)) or
            (dir_u and game.collision(point_l)) or
            (dir_r and game.collision(point_u)) or
            (dir_l and game.collision(point_d)),

            # possible move
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Where is the food?
            game.food.x < game.head.x,  # is it left?
            game.food.x > game.head.x,  # is it right?
            game.food.y < game.head.y,  # is it up?
            game.food.y > game.head.y  # is it down?
        ]

        return np.array(state)

    def long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory
        state, move, reward, new_state, done = zip(*sample)  # [[s,s,s,s,s], [m,m,m,m,m] ...]
        self.trainer.train(state, move, reward, new_state, done)

    def short_memory(self, state, move, reward, new_state, done):
        self.trainer.train(state, move, reward, new_state, done)

    def remember(self, state, move, reward, new_state, done):
        self.memory.append((state, move, reward, new_state, done))  # [[s,m,...], [s,m,...], [s,m,...], [s,m,...]]

    # Choose next move using epsilon-greedy strategy
    def get_action(self, state):
        self.epsilon = EPSILON_GAMES - self.no_games
        move = [0, 0, 0]  # Straight, LEFT, RIGHT
        if random.randint(0, 200) < self.epsilon:
            m = random.randint(0, 2)
            move[m] = 1
        else:
            state_ = torch.tensor(state, dtype=torch.float, device=device)
            pred = self.model(state_)
            m = torch.argmax(pred).item()
            move[m] = 1
        return move


def train():
    agent = Agent()
    game = Game()
    high_score = 0
    total_score = 0
    while True:
        state = agent.get_state(game)
        move = agent.get_action(state)
        reward, done, score = game.play(move)

        new_state = agent.get_state(game)
        agent.short_memory(state, move, reward, new_state, done)
        agent.remember(state, move, reward, new_state, done)

        if done:
            game.reset()
            agent.no_games += 1
            agent.long_memory()
            if score > high_score:
                high_score = score

            print('info', agent.no_games, score, )
            total_score += score
