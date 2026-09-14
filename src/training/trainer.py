import os

import numpy as np
import pygame
import torch

from src.agent.fly_agent import FlyAgent
from src.config import (
    CELL_SIZE,
    DEVICE,
    EPISODES,
    GRID_SIZE,
    MAX_STEPS,
    MODEL_PATH,
    PRINT_EVERY,
)
from src.game.snake_game import SnakeGame


class Trainer:

    def __init__(self, load_checkpoint=True):
        self.game = SnakeGame()
        self.agent = FlyAgent()

        self.start_episode = 1
        self.best_score = 0

        if load_checkpoint:
            self._load_checkpoint()

    def _load_checkpoint(self):
        episode, best_score = self.agent.load_checkpoint()

        if episode > 0:
            self.start_episode = episode + 1
            self.best_score = best_score

            print(
                f"Checkpoint loaded. "
                f"Continuing from episode {self.start_episode}"
            )
        else:
            print("Starting training from scratch.")

    def train(self):
        scores = []

        if self.start_episode > EPISODES:
            print(
                f"Training already reached episode "
                f"{self.start_episode - 1}."
            )
            return self.agent

        for episode in range(
            self.start_episode,
            EPISODES + 1,
        ):
            state = self.game.reset()

            total_reward = 0.0
            losses = []

            for _ in range(MAX_STEPS):
                action = self.agent.choose_action(
                    state,
                    training=True,
                )

                next_state, reward, done = self.game.step(
                    action
                )

                self.agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                loss = self.agent.train_step()

                if loss is not None:
                    losses.append(loss)

                state = next_state
                total_reward += reward

                if done:
                    break

            score = self.game.score
            scores.append(score)

            if score > self.best_score:
                self.best_score = score

            if episode % PRINT_EVERY == 0:
                average_score = np.mean(
                    scores[-PRINT_EVERY:]
                )

                average_loss = (
                    np.mean(losses)
                    if losses
                    else 0.0
                )

                print(
                    f"Episode {episode} | "
                    f"Score {score} | "
                    f"Avg {average_score:.2f} | "
                    f"Best {self.best_score} | "
                    f"Reward {total_reward:.2f} | "
                    f"Loss {average_loss:.4f} | "
                    f"Epsilon {self.agent.epsilon:.4f}"
                )

                self.agent.save_checkpoint(
                    episode,
                    self.best_score,
                )

        self.agent.save_checkpoint(
            EPISODES,
            self.best_score,
        )

        self._save_model()

        print(
            f"Training finished. "
            f"Best score: {self.best_score}"
        )

        return self.agent

    def _save_model(self):
        directory = os.path.dirname(MODEL_PATH)

        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(
            self.agent.brain.state_dict(),
            MODEL_PATH,
        )

        print(
            f"Model saved to: {MODEL_PATH}"
        )

    def play(self):
        if not self.agent.load_model():
            print(
                f"Model not found: {MODEL_PATH}"
            )
            return

        pygame.init()

        screen = pygame.display.set_mode(
            (
                GRID_SIZE * CELL_SIZE,
                GRID_SIZE * CELL_SIZE,
            )
        )

        pygame.display.set_caption(
            "Snake - FlyBrain"
        )

        clock = pygame.time.Clock()

        state = self.game.reset()

        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = self.agent.choose_action(
                state,
                training=False,
            )

            next_state, reward, done = self.game.step(
                action
            )

            state = next_state

            screen.fill((20, 20, 20))

            for x, y in self.game.snake:
                pygame.draw.rect(
                    screen,
                    (0, 200, 0),
                    (
                        x * CELL_SIZE,
                        y * CELL_SIZE,
                        CELL_SIZE,
                        CELL_SIZE,
                    ),
                )

            food_x, food_y = self.game.food

            pygame.draw.rect(
                screen,
                (200, 0, 0),
                (
                    food_x * CELL_SIZE,
                    food_y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ),
            )

            pygame.display.flip()

            if done:
                print(
                    f"Game over | "
                    f"Score: {self.game.score}"
                )

                state = self.game.reset()

            clock.tick(10)

        pygame.quit()

