import os

import numpy as np
import pygame
import torch

from src.config import (
    CELL_SIZE,
    EPISODES,
    GRID_SIZE,
    MAX_STEPS,
    MODEL_PATH,
    PRINT_EVERY,
    SPEED_PLAY,
)
from src.game.snake_game import SnakeGame
from src.agent.fly_agent import FlyAgent
from src.config.logger import Logger


class Trainer:

    def __init__(self, load_checkpoint=True, load_pretrained=True):
        self.logger = Logger()
        self.game = SnakeGame()
        self.agent = FlyAgent()

        self.start_episode = 1
        self.best_score = 0

        if load_checkpoint:
            self._load_checkpoint()

        if self.start_episode == 1 and load_pretrained:
            self.agent.load_pretrained()

    def _load_checkpoint(self):
        episode, best_score = self.agent.load_checkpoint()

        if episode > 0:
            self.start_episode = episode + 1
            self.best_score = best_score
            self.logger.log(
                f"Checkpoint loaded. Continuing from episode {self.start_episode}"
            )
        else:
            self.logger.log("Starting training from scratch.")

    def train(self):
        scores = []

        if self.start_episode > EPISODES:
            self.logger.log(
                f"Training already reached episode {self.start_episode - 1}."
            )
            return self.agent

        for episode in range(self.start_episode, EPISODES + 1):
            state = self.game.reset()

            total_reward = 0.0
            losses = []

            for _ in range(MAX_STEPS):
                action = self.agent.choose_action(state, training=True)
                next_state, reward, done, truncated = self.game.step(action)

                self.agent.remember(state, action, reward, next_state, done)

                loss = self.agent.train_step()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                total_reward += reward

                if done or truncated:
                    break

            score = self.game.score
            scores.append(score)

            if score > self.best_score:
                self.best_score = score

            if episode % PRINT_EVERY == 0:
                average_score = np.mean(scores[-PRINT_EVERY:])
                average_loss = np.mean(losses) if losses else 0.0

                self.logger.episode(
                    episode=episode,
                    score=score,
                    avg_score=average_score,
                    best_score=self.best_score,
                    reward=total_reward,
                    loss=average_loss,
                    epsilon=self.agent.epsilon,
                    steps=self.game.steps,
                )

                self.agent.save_checkpoint(episode, self.best_score)

        self.agent.save_checkpoint(EPISODES, self.best_score)
        self._save_model()
        self.logger.log(f"Training finished. Best score: {self.best_score}")

        return self.agent

    def _save_model(self):
        directory = os.path.dirname(MODEL_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(self.agent.brain.state_dict(), MODEL_PATH)
        self.logger.log(f"Model saved to: {MODEL_PATH}")

    def play(self):
        if not self.agent.load_model():
            self.logger.log(f"Model not found: {MODEL_PATH}")
            return

        pygame.init()
        screen = pygame.display.set_mode(
            (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
        )
        pygame.display.set_caption("Snake - FlyBrain")
        clock = pygame.time.Clock()

        state = self.game.reset()
        self.logger.log("Play mode started.")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = self.agent.choose_action(state, training=False)
            next_state, reward, done, truncated = self.game.step(action)
            state = next_state

            self.game.render(screen, pygame, CELL_SIZE)
            pygame.display.flip()

            if done or truncated:
                self.logger.log(f"Game over | Score: {self.game.score}")
                state = self.game.reset()

            clock.tick(SPEED_PLAY)

        pygame.quit()
        self.logger.log("Play mode stopped.")