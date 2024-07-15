import sys
from collections import namedtuple
from enum import Enum
import random

import pygame

pygame.init()

font = pygame.font.SysFont('Arial', 25)


class Direction(Enum):
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4


Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GREEN_LIGHT = (0, 255, 255)

GAME_SPEED = 20
BLOCK_SIZE = 20


class Game:
    def __init__(self, w=800, h=600):
        self.h = h
        self.w = w

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Game')
        self.clock = pygame.time.Clock()
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

    def set_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self.set_food()

    def play(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        self.move(self.direction)
        self.snake.insert(0, self.head)
        game_over = False
        # game over
        if self.collision():
            game_over = True
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            self.set_food()
        else:
            self.snake.pop()

        self.render()
        self.clock.tick(GAME_SPEED)
        return game_over, self.score

    def move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def collision(self):
        if self.head.x > self.w or self.head.x < 0 or self.head.y > self.h or self.head.y < 0:
            return True
        if self.head in self.snake[1:]:
            return True

        return False

    def render(self):
        self.display.fill((0, 0, 0))
        for point in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN_LIGHT,
                             pygame.Rect(point.x + 2, point.y + 2, BLOCK_SIZE - 8, BLOCK_SIZE - 8))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


if __name__ == '__main__':
    game = Game()
    while True:
        game_over, score = game.play()
        if game_over:
            break

    pygame.quit()

#%%
