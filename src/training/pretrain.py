import os
import time

import torch
import torch.nn.functional as F

from src.brain.fly_brain import FlyBrain
from src.config import DEVICE, GRID_SIZE, PRETRAINED_PATH
from src.game.snake_game import SnakeGame
from src.game.pathfinder import choose_safe_action


def pretrain(episodes=3000, save_path=PRETRAINED_PATH):
    brain = FlyBrain().to(DEVICE)
    optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)
    game = SnakeGame()

    best_score = 0
    start_episode = 0

    if os.path.exists(save_path):
        try:
            checkpoint = torch.load(save_path, map_location=DEVICE)
            brain.load_state_dict(checkpoint["brain_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_episode = checkpoint.get("episode", 0) + 1
            best_score = checkpoint.get("best_score", 0)
            print(f"Resuming pretrain from episode {start_episode}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    print("Starting pretrain loop...")

    for episode in range(start_episode, episodes):
        game.reset()
        losses = []
        episode_start = time.time()

        for step in range(1000):
            action = choose_safe_action(
                game.snake, game.food, GRID_SIZE, game.direction,
            )

            state = game.get_observation()

            # ⚠️ استفاده از as_tensor به جای tensor
            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=DEVICE,
            ).unsqueeze(0)

            q = brain(state_t)
            target = torch.tensor([action], device=DEVICE, dtype=torch.long)
            loss = F.cross_entropy(q, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            _, _, done, truncated = game.step(action)
            if done or truncated:
                break

        if game.score > best_score:
            best_score = game.score

        episode_time = time.time() - episode_start

        # ⚠️ چاپ هر episode، نه هر ۱۰۰
        avg_loss = sum(losses) / max(len(losses), 1)
        print(
            f"Episode {episode} | Loss {avg_loss:.4f} | "
            f"Score {game.score} | Best {best_score} | "
            f"Steps {game.steps} | Time {episode_time:.2f}s",
            flush=True,
        )

        if episode % 100 == 0 and episode > 0:
            torch.save(
                {
                    "episode": episode,
                    "brain_state": brain.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score": best_score,
                },
                save_path,
            )

    torch.save(brain.state_dict(), save_path)
    print(f"Pretrained model saved to: {save_path}")


if __name__ == "__main__":
    pretrain(episodes=3000)