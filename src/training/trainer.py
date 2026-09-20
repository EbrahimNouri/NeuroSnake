import os

import numpy as np
import pygame
import torch

from src.agent.fly_agent import FlyAgent
from src.config.config import (
    CELL_SIZE,
    DEVICE,
    EPISODES,
    GRID_SIZE,
    MAX_STEPS,
    MAX_STEPS_PLAY,
    MODEL_PATH,
    PRINT_EVERY,
    SAFETY_NET_ENABLED,
    SPEED_PLAY,
    STUCK_STEPS,
)
from src.config.logger import Logger
from src.game.snake_game import SnakeGame
from src.game.snake_planner import SnakePlanner


class Trainer:

    def __init__(self, load_checkpoint=True):
        self.logger = Logger()

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

        # A full-board run needs far more steps than training allows.
        self.game.max_steps = MAX_STEPS_PLAY

        planner = SnakePlanner()

        pygame.init()

        screen = pygame.display.set_mode(
            (
                GRID_SIZE * CELL_SIZE,
                GRID_SIZE * CELL_SIZE,
            )
        )

        pygame.display.set_caption(
            "Snake - FlyBrain + Safety Net"
        )

        clock = pygame.time.Clock()

        state = self.game.reset()

        # Hybrid-controller state.
        brain_moves = 0
        net_moves = 0
        steps_since_food = 0
        planner_driving = False

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

            override = False

            if SAFETY_NET_ENABLED:
                if planner_driving:
                    action, _ = planner.guide(self.game)
                    override = True
                else:
                    action, override = planner.decide(
                        self.game,
                        action,
                    )

                    steps_since_food += 1

                    if steps_since_food >= STUCK_STEPS:
                        # Brain stalled: let the planner drive until
                        # the next food is eaten.
                        planner_driving = True
                        action, _ = planner.guide(self.game)
                        override = True

            if override:
                net_moves += 1
            else:
                brain_moves += 1

            next_state, reward, done = self.game.step(
                action
            )

            state = next_state

            screen.fill((20, 20, 20))

            for index, (x, y) in enumerate(
                self.game.snake
            ):
                color = (
                    (80, 160, 255)
                    if (index == 0 and override)
                    else (0, 200, 0)
                )

                pygame.draw.rect(
                    screen,
                    color,
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

            if reward >= 10.0:
                # Food eaten: reset stuck tracking and hand control
                # back to the brain.
                steps_since_food = 0
                planner_driving = False

            if done:
                if self.game.won:
                    self.logger.log(
                        f"BOARD FILLED! | "
                        f"Score: {self.game.score} | "
                        f"Brain moves: {brain_moves} | "
                        f"Safety net moves: {net_moves}"
                    )
                else:
                    self.logger.log(
                        f"Game over | "
                        f"Score: {self.game.score} | "
                        f"Brain moves: {brain_moves} | "
                        f"Safety net moves: {net_moves}"
                    )

                brain_moves = 0
                net_moves = 0
                steps_since_food = 0
                planner_driving = False

                state = self.game.reset()

            clock.tick(SPEED_PLAY)

        pygame.quit()

        self.logger.log("Play mode stopped.")

