import pygame
import random
import sys
import numpy as np
from settings import GameSettings

from util import Direction, load_assets, get_direction, BLOCK_SIZE, BACKGROUND_COLOR_TOP, BACKGROUND_COLOR_BOTTOM, \
    WHITE, Coordinate, GameMode

pygame.init()

font_small = pygame.font.SysFont('Arial', 14, bold=False)

# Game settings
REWARD = 10


class Game:
    def __init__(self, settings: GameSettings):
        self.speed = settings.speed
        self.width = settings.width
        self.height = settings.height
        self.grass_background = None

        # Set up display 
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset_game()
        self.high_score = 0

        # Load assets
        self.assets = load_assets()

    def place_food(self):
        """Place food randomly on the game board."""
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Coordinate(x, y)
        if self.food in self.snake:
            self.place_food()

    def play_step(self, action):
        """Play one step of the game."""
        self.frame_iteration += 1

        # Handle game events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.move_snake(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        # Check for collisions
        if self.is_collision() or (self.game_mode == GameMode.TRAINING and self.frame_iteration > 100 * len(self.snake)):
            game_over = True
            reward = -REWARD
            if self.score > self.high_score:
                self.high_score = self.score
            return reward, game_over, self.score

        # Check if snake has eaten the food
        if self.head == self.food:
            self.score += 1
            print(f'Score: {self.score}')
            reward = REWARD
            self.place_food()
        else:
            self.snake.pop()

        self.render()
        self.clock.tick(self.speed)
        return reward, game_over, self.score

    def move_snake(self, action):
        """Move the snake in the specified direction."""
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_direction_index = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  # Move straight
            next_direction_index = current_direction_index
        elif np.array_equal(action, [0, 1, 0]):  # Turn right
            next_direction_index = (current_direction_index + 1) % 4
        else:  # Turn left
            next_direction_index = (current_direction_index - 1) % 4

        self.direction = directions[next_direction_index]

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

        self.head = Coordinate(int(x), int(y))

    def is_collision(self, point=None):
        """Check for collisions with walls or itself."""
        if point is None:
            point = self.head

        # Check collision with walls
        if point.x >= self.width or point.x < 0 or point.y >= self.height or point.y < 0:
            return True

        # Check collision with itself
        if point in self.snake[1:]:
            return True

        return False

    def render(self):
        """Update the game's user interface."""
        self.display.blit(self.grass_background, (0, 0))  # Blit the persistent grass background

        for i, point in enumerate(self.snake):
            if i == 0:
                self.display.blit(self.get_snake_asset('head'), (point.x, point.y))
            elif i == len(self.snake) - 1:
                self.display.blit(self.get_snake_asset('tail', self.snake[i - 1], point), (point.x, point.y))
            else:
                self.display.blit(self.get_snake_asset('body', self.snake[i - 1], point, self.snake[i + 1]),
                                  (point.x, point.y))

        self.display.blit(self.assets['apple'], (self.food.x, self.food.y))

        self.display_info_box()

        pygame.display.flip()

    def display_gradient_background(self):
        """Create a gradient background effect."""
        for i in range(self.height):
            color = (
                BACKGROUND_COLOR_TOP[0] + (BACKGROUND_COLOR_BOTTOM[0] - BACKGROUND_COLOR_TOP[0]) * i // self.height,
                BACKGROUND_COLOR_TOP[1] + (BACKGROUND_COLOR_BOTTOM[1] - BACKGROUND_COLOR_TOP[1]) * i // self.height,
                BACKGROUND_COLOR_TOP[2] + (BACKGROUND_COLOR_BOTTOM[2] - BACKGROUND_COLOR_TOP[2]) * i // self.height
            )
            pygame.draw.line(self.display, color, (0, i), (self.width, i))

    def display_info_box(self):
        """Display the score and other game information in a modern info box."""
        info_surface_width = 180
        info_surface_height = 80
        info_surface = pygame.Surface((info_surface_width, info_surface_height), pygame.SRCALPHA)

        # Create a semi-transparent background with a slightly darker border
        background_color = (50, 50, 50, 180)  # More transparent background
        border_color = (50, 50, 50, 200)

        pygame.draw.rect(info_surface, background_color, (0, 0, info_surface_width, info_surface_height),
                         border_radius=10)
        pygame.draw.rect(info_surface, border_color, (0, 0, info_surface_width, info_surface_height), 1,
                         border_radius=10)

        # Render the text
        score_text = font_small.render(f"Score: {self.score}", True, WHITE)
        high_score_text = font_small.render(f"High Score: {self.high_score}", True, WHITE)
        iteration_text = font_small.render(f"Frame: {self.frame_iteration}", True, WHITE)
        snake_length_text = font_small.render(f"Length: {len(self.snake)}", True, WHITE)

        padding = 5
        info_surface.blit(score_text, (padding, padding))
        info_surface.blit(high_score_text, (padding, padding + 18))
        info_surface.blit(iteration_text, (padding, padding + 36))
        info_surface.blit(snake_length_text, (padding, padding + 54))

        # Adding subtle shadow for the info box
        shadow_surface = pygame.Surface((info_surface_width, info_surface_height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 50), shadow_surface.get_rect(), border_radius=10)
        self.display.blit(shadow_surface, (12, 12))
        self.display.blit(info_surface, (10, 10))

    def get_snake_asset(self, part, previous=None, current=None, next=None):
        """Get the appropriate snake asset based on the part and surrounding segments."""
        if part == 'head':
            if self.direction == Direction.UP:
                return self.assets['head_up']
            elif self.direction == Direction.DOWN:
                return self.assets['head_down']
            elif self.direction == Direction.LEFT:
                return self.assets['head_left']
            elif self.direction == Direction.RIGHT:
                return self.assets['head_right']
        elif part == 'tail':
            if previous is None:
                return self.assets['tail_up']  # Default to UP
            tail_direction = get_direction(previous, current)
            if tail_direction == Direction.UP:
                return self.assets['tail_up']
            elif tail_direction == Direction.DOWN:
                return self.assets['tail_down']
            elif tail_direction == Direction.LEFT:
                return self.assets['tail_left']
            elif tail_direction == Direction.RIGHT:
                return self.assets['tail_right']
        elif part == 'body':
            if previous is None or next is None:
                return self.assets['body_horizontal']  # Default to horizontal
            prev_direction = get_direction(previous, current)
            next_direction = get_direction(current, next)
            if prev_direction == Direction.UP and next_direction == Direction.UP or \
                    prev_direction == Direction.DOWN and next_direction == Direction.DOWN:
                return self.assets['body_vertical']
            elif prev_direction == Direction.LEFT and next_direction == Direction.LEFT or \
                    prev_direction == Direction.RIGHT and next_direction == Direction.RIGHT:
                return self.assets['body_horizontal']
            elif (prev_direction == Direction.DOWN and next_direction == Direction.LEFT) or \
                    (prev_direction == Direction.RIGHT and next_direction == Direction.UP):
                return self.assets['body_topleft']
            elif (prev_direction == Direction.DOWN and next_direction == Direction.RIGHT) or \
                    (prev_direction == Direction.LEFT and next_direction == Direction.UP):
                return self.assets['body_topright']
            elif (prev_direction == Direction.UP and next_direction == Direction.LEFT) or \
                    (prev_direction == Direction.RIGHT and next_direction == Direction.DOWN):
                return self.assets['body_bottomleft']
            elif (prev_direction == Direction.UP and next_direction == Direction.RIGHT) or \
                    (prev_direction == Direction.LEFT and next_direction == Direction.DOWN):
                return self.assets['body_bottomright']

    def create_grass_background(self):
        """Create a persistent grass-like background with random dark dots of varying sizes."""
        self.grass_background = pygame.Surface((self.width, self.height))
        # Background settings
        light_green = (170, 215, 81)  # light green
        dark_green = (122, 169, 53)  # Darker green for dots
        dot_density = 25 # lower numbers more dots
        max_dot_size = 2
        # Fill the background surface with light green
        self.grass_background.fill(light_green)


        # Randomly place darker green dots of varying sizes
        number_of_dots = (self.width * self.height) // dot_density


        for _ in range(number_of_dots):
            dot_size = random.randint(1, max_dot_size)
            x = random.randint(0, self.width - dot_size)
            y = random.randint(0, self.height - dot_size)
            pygame.draw.ellipse(self.grass_background, dark_green, (x, y, dot_size, dot_size))

    def reset_game(self):
        """Reset the game state."""

        self.direction = Direction.RIGHT
        self.head = Coordinate(self.width / 2, self.height / 2)
        self.snake = [
            self.head,
            Coordinate(self.head.x - BLOCK_SIZE, self.head.y),
            Coordinate(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        self.game_mode = GameMode.TRAINING

        # Create a persistent grass background
        self.create_grass_background()