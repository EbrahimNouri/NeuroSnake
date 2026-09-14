import random

import numpy as np

from config import GRID_SIZE


class SnakeGame:

    def __init__(self):
        self.reset()

    def reset(self):
        center = GRID_SIZE // 2

        self.snake = [
            (center, center),
            (center - 1, center),
            (center - 2, center),
        ]

        self.direction = (1, 0)
        self.food = self._spawn_food()

        self.score = 0
        self.steps = 0

        return self.get_observation()

    def _spawn_food(self):
        while True:
            food = (
                random.randint(0, GRID_SIZE - 1),
                random.randint(0, GRID_SIZE - 1),
            )

            if food not in self.snake:
                return food

    def _get_direction_vectors(self):
        dx, dy = self.direction

        left = (-dy, dx)
        right = (dy, -dx)

        return left, right

    def _is_collision(self, position):
        x, y = position

        if x < 0 or x >= GRID_SIZE:
            return True

        if y < 0 or y >= GRID_SIZE:
            return True

        return position in self.snake

    def get_observation(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        dx, dy = self.direction

        left_direction, right_direction = self._get_direction_vectors()

        straight_position = (
            head_x + dx,
            head_y + dy,
        )

        left_position = (
            head_x + left_direction[0],
            head_y + left_direction[1],
        )

        right_position = (
            head_x + right_direction[0],
            head_y + right_direction[1],
        )

        danger_straight = float(self._is_collision(straight_position))
        danger_right = float(self._is_collision(right_position))
        danger_left = float(self._is_collision(left_position))

        direction_up = float(dy == -1)
        direction_down = float(dy == 1)
        direction_left = float(dx == -1)
        direction_right = float(dx == 1)

        food_left = float(food_x < head_x)
        food_right = float(food_x > head_x)
        food_up = float(food_y < head_y)
        food_down = float(food_y > head_y)

        return np.array(
            [
                danger_straight,
                danger_right,
                danger_left,
                direction_up,
                direction_down,
                direction_left,
                direction_right,
                food_left,
                food_right,
                food_up,
                food_down,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        previous_distance = self._food_distance()

        directions = [
            (0, -1),  # UP
            (0, 1),   # DOWN
            (-1, 0),  # LEFT
            (1, 0),   # RIGHT
        ]

        new_direction = directions[action]

        opposite_direction = (
            -self.direction[0],
            -self.direction[1],
        )

        if new_direction != opposite_direction:
            self.direction = new_direction

        head_x, head_y = self.snake[0]

        new_head = (
            head_x + self.direction[0],
            head_y + self.direction[1],
        )

        self.steps += 1

        if self._is_collision(new_head):
            return self.get_observation(), -10.0, True

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()

            reward = 10.0
        else:
            self.snake.pop()

            current_distance = self._food_distance()

            reward = -0.02

            if current_distance < previous_distance:
                reward += 0.10
            else:
                reward -= 0.10

        if self.steps >= 500:
            return self.get_observation(), reward, True

        return self.get_observation(), reward, False

    def _food_distance(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        return abs(head_x - food_x) + abs(head_y - food_y)

    def render(self, screen, pygame):
        screen.fill((20, 20, 20))

        for x, y in self.snake:
            pygame.draw.rect(
                screen,
                (0, 200, 0),
                (
                    x * 25,
                    y * 25,
                    25,
                    25,
                ),
            )

        food_x, food_y = self.food

        pygame.draw.rect(
            screen,
            (200, 0, 0),
            (
                food_x * 25,
                food_y * 25,
                25,
                25,
            ),
        )
