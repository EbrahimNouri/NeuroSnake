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
    PARALLEL_ENVIRONMENTS,
    PRINT_EVERY,
    SPEED_PLAY,
)
from src.config.logger import Logger
from src.game.snake_game import SnakeGame


class Trainer:

    def __init__(self, load_checkpoint=True):
        self.logger = Logger()

        self.env_count = PARALLEL_ENVIRONMENTS

        self.games = [
            SnakeGame()
            for _ in range(self.env_count)
        ]

        self.agent = FlyAgent()

        self.states = [
            game.get_observation()
            for game in self.games
        ]

        self.game = self.games[0]

        self.start_episode = 1
        self.best_score = 0

        if load_checkpoint:
            self._load_checkpoint()

    def _load_checkpoint(self):
        episode, best_score = self.agent.load_checkpoint()

        if episode > 0:
            self.start_episode = episode + 1
            self.best_score = best_score

            self.logger.log(
                f"Checkpoint loaded. "
                f"Continuing from episode {self.start_episode}"
            )
        else:
            self.logger.log(
                "Starting training from scratch."
            )

    def train(self):
        scores = []

        if self.start_episode > EPISODES:
            self.logger.log(
                f"Training already reached episode "
                f"{self.start_episode - 1}."
            )
            return self.agent

        for episode in range(
            self.start_episode,
            EPISODES + 1,
        ):
            active = np.ones(
                self.env_count,
                dtype=bool,
            )

            episode_scores = []

            total_rewards = np.zeros(
                self.env_count,
                dtype=np.float64,
            )

            episode_steps = np.zeros(
                self.env_count,
                dtype=np.int64,
            )

            losses = []

            for _ in range(MAX_STEPS):
                indices = np.flatnonzero(active)

                if indices.size == 0:
                    break

                batch_states = np.stack(
                    [
                        self.states[index]
                        for index in indices
                    ]
                )

                actions = self.agent.choose_action_batch(
                    batch_states,
                    training=True,
                )

                for index, action in zip(indices, actions):
                    game = self.games[index]
                    state = self.states[index]

                    next_state, reward, done = game.step(
                        int(action)
                    )

                    self.agent.remember(
                        state,
                        int(action),
                        reward,
                        next_state,
                        done,
                    )

                    total_rewards[index] += reward
                    episode_steps[index] += 1

                    if done:
                        episode_scores.append(game.score)

                        if game.score > self.best_score:
                            self.best_score = game.score

                        active[index] = False

                        self.states[index] = game.reset()
                    else:
                        self.states[index] = next_state

                if self.agent.train_step():
                    losses.append(
                        self.agent.mean_loss()
                    )

            scores.extend(episode_scores)

            if episode_scores:
                episode_score = int(
                    np.mean(episode_scores)
                )
            else:
                episode_score = 0

            # Update epsilon once per episode
            self.agent.update_epsilon()

            if episode % PRINT_EVERY == 0:
                average_score = np.mean(
                    scores[-PRINT_EVERY:]
                ) if scores else 0.0

                average_loss = (
                    np.mean(losses)
                    if losses
                    else 0.0
                )

                self.logger.episode(
                    episode=episode,
                    score=episode_score,
                    avg_score=average_score,
                    best_score=self.best_score,
                    reward=float(
                        np.mean(total_rewards)
                    ),
                    loss=average_loss,
                    epsilon=self.agent.epsilon,
                    steps=int(np.max(episode_steps)),
                    envs=self.env_count,
                    episodes_finished=len(episode_scores),
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

        self.logger.log(
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

        self.logger.log(
            f"Model saved to: {MODEL_PATH}"
        )

    def play(self):
        if not self.agent.load_model():
            self.logger.log(
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

        self.logger.log("Play mode started.")

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
                self.logger.log(
                    f"Game over | "
                    f"Score: {self.game.score}"
                )

                state = self.game.reset()

            clock.tick(SPEED_PLAY)

        pygame.quit()

        self.logger.log("Play mode stopped.")

