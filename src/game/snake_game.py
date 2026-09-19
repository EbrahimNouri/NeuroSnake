import random

import numpy as np

from src.config import (
    GRID_SIZE,
    MAX_STEPS,
    REWARD_FOOD,
    REWARD_DEATH,
    REWARD_STEP,
    REWARD_CLOSER_FOOD,
    REWARD_FARTHER_FOOD,
    REWARD_TRUNCATED,
)


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

        left = (-dy, dx)
        right = (dy, -dx)

        obs = []

        for d in [self.direction, right, left]:
            nx, ny = head_x + d[0], head_y + d[1]
            obs.append(float(self._is_collision((nx, ny))))

        obs.extend([
            float(dy == -1), float(dy == 1),
            float(dx == -1), float(dx == 1),
        ])

        obs.extend([
            float(food_x < head_x), float(food_x > head_x),
            float(food_y < head_y), float(food_y > head_y),
        ])

        obs.extend([
            head_x / GRID_SIZE,
            (GRID_SIZE - 1 - head_x) / GRID_SIZE,
            head_y / GRID_SIZE,
            (GRID_SIZE - 1 - head_y) / GRID_SIZE,
        ])

        dist = abs(head_x - food_x) + abs(head_y - food_y)
        obs.append(dist / (2 * GRID_SIZE))

        for d in [self.direction, right, left]:
            for step in [1, 2, 3]:
                nx = head_x + d[0] * step
                ny = head_y + d[1] * step
                obs.append(float(self._is_collision((nx, ny))))

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        previous_distance = self._food_distance()

        directions = [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
        ]

        new_direction = directions[action]
        opposite = (-self.direction[0], -self.direction[1])

        if new_direction != opposite:
            self.direction = new_direction

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        self.steps += 1

        if self._is_collision(new_head):
            return self.get_observation(), REWARD_DEATH, True, False

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
            reward = REWARD_FOOD
        else:
            self.snake.pop()
            current_distance = self._food_distance()
            reward = REWARD_STEP

            if current_distance < previous_distance:
                reward += REWARD_CLOSER_FOOD
            else:
                reward += REWARD_FARTHER_FOOD

        if self.steps >= MAX_STEPS:
            return self.get_observation(), REWARD_TRUNCATED, False, True

        return self.get_observation(), reward, False, False

    def _food_distance(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return abs(head_x - food_x) + abs(head_y - food_y)

    def render(self, screen, pygame, cell_size):
        screen.fill((20, 20, 20))

        for x, y in self.snake:
            pygame.draw.rect(
                screen, (0, 200, 0),
                (x * cell_size, y * cell_size, cell_size, cell_size),
            )

        food_x, food_y = self.food
        pygame.draw.rect(
            screen, (200, 0, 0),
            (food_x * cell_size, food_y * cell_size, cell_size, cell_size),
        )