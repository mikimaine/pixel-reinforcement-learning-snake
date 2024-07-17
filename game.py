import sys
from collections import namedtuple
from enum import Enum
import random

import numpy as np
import pygame

pygame.init()

font = pygame.font.SysFont('Arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GREEN_LIGHT = (0, 255, 255)

GAME_SPEED = 100
BLOCK_SIZE = 20

REWARD = 10


class Game:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Game')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score = 0
        self.food = None
        self.set_food()
        self.current_round = 0

    def set_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE ) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE ) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self.set_food()

    def play(self, action):
        self.current_round += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        # game over
        if self.collision() or self.current_round > 100*len(self.snake):
            game_over = True
            reward = -REWARD
            return reward, game_over, self.score

        # print('debug',self.head == self.food, self.head, self.food)
        if self.head == self.food:
            self.score += 1
            print(f'Score: {self.score}')
            reward = REWARD
            self.set_food()
        else:
            self.snake.pop()

        self.render()
        self.clock.tick(GAME_SPEED)
        return reward, game_over, self.score

    def move(self, action):  # [1,0,0]
        d = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]  # clock wise
        idx = d.index(self.direction)  # current direction of the snake

        if np.array_equal(action, [1, 0, 0]):  # move straight
            next_idx = idx
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # right r -> d -> l ->u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4  # left r->u->l-d

        self.direction = d[next_idx]  # new direction from index

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(int(x), int(y))

    def collision(self, pt=None):
        if pt is None:
            pt = self.head
        # snake collision border
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # snake collision snake
        if pt in self.snake[1:]:
            return True

        return False

    def render(self):
        self.display.fill((0, 0, 0))
        for point in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN_LIGHT,
                             pygame.Rect(point.x + 4, point.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

#%%
