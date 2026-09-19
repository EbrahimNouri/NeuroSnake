import os
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.brain.fly_brain import FlyBrain
from src.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DEVICE,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    LEARNING_RATE,
    MEMORY_SIZE,
    MODEL_PATH,
    N_STEP,
    PRETRAINED_PATH,
    TARGET_UPDATE,
)
from src.rl.replay_memory import NStepReplayMemory


class FlyAgent:

    def __init__(self):
        self.brain = FlyBrain().to(DEVICE)

        self.target_brain = FlyBrain().to(DEVICE)
        self.target_brain.load_state_dict(self.brain.state_dict())
        self.target_brain.eval()

        self.optimizer = optim.Adam(
            self.brain.parameters(),
            lr=LEARNING_RATE,
        )

        self.memory = NStepReplayMemory(
            MEMORY_SIZE, N_STEP, GAMMA,
        )

        self.gamma = GAMMA
        self.n_step = N_STEP
        self.epsilon = EPSILON_START
        self.training_steps = 0

    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(4)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.brain(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(BATCH_SIZE)

        states = torch.tensor(
            states, dtype=torch.float32, device=DEVICE,
        )
        actions = torch.tensor(
            actions, dtype=torch.long, device=DEVICE,
        )
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=DEVICE,
        )
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=DEVICE,
        )
        dones = torch.tensor(
            dones, dtype=torch.float32, device=DEVICE,
        )

        current_q = self.brain(states).gather(
            1, actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_actions = self.brain(next_states).argmax(
                dim=1, keepdim=True,
            )

            next_q = self.target_brain(next_states).gather(
                1, next_actions,
            ).squeeze(1)

            target_q = rewards + (
                (self.gamma ** self.n_step)
                * next_q
                * (1.0 - dones)
            )

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.brain.parameters(), 10.0,
        )
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % TARGET_UPDATE == 0:
            self.target_brain.load_state_dict(
                self.brain.state_dict()
            )

        self.epsilon = max(
            EPSILON_END,
            self.epsilon * EPSILON_DECAY,
        )

        return loss.item()

    def save_checkpoint(self, episode, best_score):
        directory = os.path.dirname(CHECKPOINT_PATH)

        if directory:
            os.makedirs(directory, exist_ok=True)

        checkpoint = {
            "episode": episode,
            "brain_state": self.brain.state_dict(),
            "target_brain_state": self.target_brain.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "best_score": best_score,
        }

        torch.save(checkpoint, CHECKPOINT_PATH)

    def load_checkpoint(self):
        if not os.path.exists(CHECKPOINT_PATH):
            return 0, 0

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )

        self.brain.load_state_dict(checkpoint["brain_state"])
        self.target_brain.load_state_dict(
            checkpoint["target_brain_state"]
        )
        self.optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]

        episode = checkpoint.get("episode", 0)
        best_score = checkpoint.get("best_score", 0)

        return episode, best_score

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            return False

        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        self.brain.load_state_dict(state_dict)
        self.target_brain.load_state_dict(
            self.brain.state_dict()
        )

        print(f"Model loaded from: {MODEL_PATH}")

        return True

    def load_pretrained(self):
        if not os.path.exists(PRETRAINED_PATH):
            print(f"Pretrained model not found: {PRETRAINED_PATH}")
            return False

        checkpoint = torch.load(
            PRETRAINED_PATH,
            map_location=DEVICE,
            weights_only=False,
        )

        # اگه checkpoint یه dict با کلید brain_state بود
        if isinstance(checkpoint, dict) and "brain_state" in checkpoint:
            state_dict = checkpoint["brain_state"]
            print(f"Loaded pretrained at episode {checkpoint.get('episode', '?')} "
                  f"| best_score {checkpoint.get('best_score', '?')}")
        else:
            state_dict = checkpoint

        self.brain.load_state_dict(state_dict)
        self.target_brain.load_state_dict(state_dict)

        print(f"Pretrained model loaded from: {PRETRAINED_PATH}")

        return True