import os

import numpy as np
import torch

from src.agent.fly_agent import FlyAgent
from src.config import (
    DEVICE,
    EPISODES,
    MAX_STEPS,
    MODEL_PATH,
    PRINT_EVERY,
)
from src.game.snake_game import SnakeGame


class Trainer:

    def __init__(self):
        self.game = SnakeGame()
        self.agent = FlyAgent()

        self.start_episode = 1
        self.best_score = 0

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

                if losses:
                    average_loss = np.mean(losses)
                else:
                    average_loss = 0.0

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

    def evaluate(self, episodes=10):
        scores = []

        for episode in range(episodes):
            state = self.game.reset()
            total_reward = 0.0

            for _ in range(MAX_STEPS):
                action = self.agent.choose_action(
                    state,
                    training=False,
                )

                next_state, reward, done = self.game.step(
                    action
                )

                state = next_state
                total_reward += reward

                if done:
                    break

            scores.append(self.game.score)

            print(
                f"Evaluation {episode + 1} | "
                f"Score {self.game.score} | "
                f"Reward {total_reward:.2f}"
            )

        print(
            f"Evaluation average: "
            f"{np.mean(scores):.2f}"
        )

        return scores

