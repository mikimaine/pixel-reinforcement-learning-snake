from collections import namedtuple
from enum import Enum

import pygame

# Define a point structure for positions
Coordinate = namedtuple('Coordinate', 'x, y')
# Define colors
WHITE = (255, 255, 255)
BACKGROUND_COLOR_TOP = (30, 30, 30)  # Dark grey
BACKGROUND_COLOR_BOTTOM = (50, 50, 50)  # Slightly lighter grey
INFO_BOX_COLOR = (50, 50, 50, 180)  # Semi-transparent dark grey

BLOCK_SIZE = 20

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


def get_direction(start, end):
    """Get direction from start point to end point."""
    if start.x < end.x:
        return Direction.RIGHT
    elif start.x > end.x:
        return Direction.LEFT
    elif start.y < end.y:
        return Direction.DOWN
    elif start.y > end.y:
        return Direction.UP


def load_assets():
    """Load all the game assets and resize them to match BLOCK_SIZE."""

    def load_and_scale_image(path):
        image = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(image, (BLOCK_SIZE, BLOCK_SIZE))

    assets = {
        'head_up': load_and_scale_image('assets/head_up.png'),
        'head_down': load_and_scale_image('assets/head_down.png'),
        'head_left': load_and_scale_image('assets/head_left.png'),
        'head_right': load_and_scale_image('assets/head_right.png'),
        'body_horizontal': load_and_scale_image('assets/body_horizontal.png'),
        'body_vertical': load_and_scale_image('assets/body_vertical.png'),
        'body_topleft': load_and_scale_image('assets/body_topleft.png'),
        'body_topright': load_and_scale_image('assets/body_topright.png'),
        'body_bottomleft': load_and_scale_image('assets/body_bottomleft.png'),
        'body_bottomright': load_and_scale_image('assets/body_bottomright.png'),
        'tail_up': load_and_scale_image('assets/tail_up.png'),
        'tail_down': load_and_scale_image('assets/tail_down.png'),
        'tail_left': load_and_scale_image('assets/tail_left.png'),
        'tail_right': load_and_scale_image('assets/tail_right.png'),
        'apple': load_and_scale_image('assets/apple.png')
    }
    return assets
