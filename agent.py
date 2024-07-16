import random
from collections import deque

import numpy as np
import torch

from game import Game, Point, BLOCK_SIZE, Direction

MAX_MEMORY = 1000
EPSILON_GAMES = 80

class Agent:
    def __init__(self):
        self.no_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None
        self.trainer = None

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
            game.food.x < game.head.x, # is it left?
            game.food.x > game.head.x, # is it right?
            game.food.y < game.head.y, # is it up?
            game.food.y > game.head.y  # is it down?
        ]

        return np.array(state)

    def long_memory(self):
        pass
    def short_memory(self):
        pass

    # Choose next move using epsilon-greedy strategy
    def get_action(self, state):
        self.epsilon = EPSILON_GAMES - self.no_games
        move = [0, 0, 0] # Straight, LEFT, RIGHT
        if random.random(0, 200) < self.epsilon:
            m = random.randint(0, 2)
            move[m] = 1
        else:
            state_ = torch.tensor(state, dtype=torch.float)
            # pred =
        return move

def train():
    agent = Agent()
    game = Game()

    while True:
        state = agent.get_state(game)
        move = agent.get_action(state)
        reward, done, score = game.move(move)

        new_state = agent.get_state(game)



