import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.brain.fly_brain import FlyBrain

from src.config import (
    ACTION_COUNT,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    COMPILE_BRAIN,
    COMPILE_MODE,
    DEVICE,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    LEARNING_RATE,
    MODEL_PATH,
    PARALLEL_ENVIRONMENTS,
    TARGET_UPDATE,
)
from src.rl.replay_memory import ReplayMemory


class FlyAgent:

    def __init__(self):
        self.brain = FlyBrain().to(DEVICE)

        self.target_brain = FlyBrain().to(DEVICE)
        self.target_brain.load_state_dict(
            self.brain.state_dict()
        )
        self.target_brain.eval()

        self.optimizer = optim.Adam(
            self.brain.parameters(),
            lr=LEARNING_RATE,
        )

        self.memory = ReplayMemory()

        self.gamma = GAMMA

        self.epsilon = EPSILON_START

        self.training_steps = 0

        self.loss_sum = torch.zeros((), device=DEVICE)
        self.loss_count = 0

        # A CUDA graph can only replay one shape, so inference is always
        # padded to this width instead of shrinking as environments die.
        self.inference_batch = PARALLEL_ENVIRONMENTS

        self.compiled = bool(COMPILE_BRAIN)
        self.compile_failed = False

        self._infer_fn = self._compile(self.brain)
        self._loss_fn = self._compile(self._td_loss)

    def _compile(self, fn):
        if not COMPILE_BRAIN:
            return fn

        try:
            return torch.compile(fn, mode=COMPILE_MODE)
        except Exception:
            self.compiled = False
            self.compile_failed = True
            return fn

    def _disable_compile(self, error):
        if not self.compile_failed:
            self.compile_failed = True
            self.compiled = False

            print(
                f"torch.compile disabled "
                f"({type(error).__name__}: {error}). "
                f"Falling back to eager, training will be slower."
            )

        self._infer_fn = self.brain
        self._loss_fn = self._td_loss

    def _td_loss(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
    ):
        current_q = self.brain(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_actions = self.brain(
                next_states
            ).argmax(
                dim=1,
                keepdim=True,
            )

            next_q = self.target_brain(
                next_states
            ).gather(
                1,
                next_actions,
            ).squeeze(1)

            target_q = rewards + (
                self.gamma
                * next_q
                * (1.0 - dones)
            )

        loss = F.smooth_l1_loss(
            current_q,
            target_q,
        )

        return loss, current_q, target_q

    def choose_action(self, state, training=True):
        action = self.choose_action_batch(
            np.asarray(
                state,
                dtype=np.float32,
            ).reshape(1, -1),
            training=training,
        )

        return int(action[0])

    def choose_action_batch(self, states, training=True):
        states = np.asarray(
            states,
            dtype=np.float32,
        )

        if states.ndim == 1:
            states = states.reshape(1, -1)

        env_count = states.shape[0]

        if env_count > self.inference_batch:
            chunks = [
                self._infer(
                    states[start:start + self.inference_batch],
                    training,
                )
                for start in range(
                    0,
                    env_count,
                    self.inference_batch,
                )
            ]

            return np.concatenate(chunks)

        actions = self._infer(states, training)

        return actions[:env_count]

    def _infer(self, states, training):
        width = self.inference_batch

        if states.shape[0] < width:
            padding = np.zeros(
                (width - states.shape[0], states.shape[1]),
                dtype=np.float32,
            )
            states = np.concatenate(
                (states, padding),
                axis=0,
            )

        states_tensor = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=DEVICE,
        )

        try:
            with torch.no_grad():
                q_values = self._infer_fn(states_tensor)
        except Exception as error:
            self._disable_compile(error)

            with torch.no_grad():
                q_values = self.brain(states_tensor)

        actions = q_values.argmax(dim=1)

        if training and self.epsilon > 0.0:
            explore = (
                torch.rand(
                    width,
                    device=DEVICE,
                )
                < self.epsilon
            )

            random_actions = torch.randint(
                0,
                ACTION_COUNT,
                (width,),
                device=DEVICE,
            )

            actions = torch.where(
                explore,
                random_actions,
                actions,
            )

        return actions.cpu().numpy()

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done,
        )

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
            states,
            dtype=torch.float32,
            device=DEVICE,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=DEVICE,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=DEVICE,
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=DEVICE,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=DEVICE,
        )

        try:
            loss, current_q, target_q = self._loss_fn(
                states,
                actions,
                rewards,
                next_states,
                dones,
            )
        except Exception as error:
            self._disable_compile(error)

            loss, current_q, target_q = self._td_loss(
                states,
                actions,
                rewards,
                next_states,
                dones,
            )

        # CUDA-graph outputs are recycled on the next replay, so snapshot the
        # three scalars we still need after backward() and the debug print.
        loss_value = loss.detach().clone()
        current_q = current_q.detach().clone()
        target_q = target_q.detach().clone()
        td_error = (target_q - current_q).abs()

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.brain.parameters(),
            5.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        # Accumulating on-device avoids a GPU sync on every single step.
        self.loss_sum = self.loss_sum + loss_value.mean()
        self.loss_count += 1

        if self.training_steps % 500 == 0:
            average = (
                self.loss_sum / max(self.loss_count, 1)
            ).item()

            print(
                f"Q={current_q.mean().item():.2f} "
                f"Target={target_q.mean().item():.2f} "
                f"TD={td_error.abs().mean().item():.2f} "
                f"Loss={average:.4f}"
            )

            self.loss_sum.zero_()
            self.loss_count = 0

        if self.training_steps % TARGET_UPDATE == 0:
            self.target_brain.load_state_dict(
                self.brain.state_dict()
            )

        return True

    def mean_loss(self):
        if self.loss_count == 0:
            return 0.0

        return (
            self.loss_sum / self.loss_count
        ).item()

    def update_epsilon(self):
        self.epsilon *= EPSILON_DECAY

        if self.epsilon < EPSILON_END:
            self.epsilon = EPSILON_END

    def save_checkpoint(self, episode, best_score):
        directory = os.path.dirname(
            CHECKPOINT_PATH
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        checkpoint = {
            "episode": episode,
            "brain_state": self.brain.state_dict(),
            "target_brain_state": (
                self.target_brain.state_dict()
            ),
            "optimizer_state": (
                self.optimizer.state_dict()
            ),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "best_score": best_score,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH,
        )

    def load_checkpoint(self):
        if not os.path.exists(
            CHECKPOINT_PATH
        ):
            return 0, 0

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )

        self.brain.load_state_dict(
            checkpoint["brain_state"]
        )

        self.target_brain.load_state_dict(
            checkpoint["target_brain_state"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        self.epsilon = checkpoint["epsilon"]

        self.training_steps = checkpoint[
            "training_steps"
        ]

        episode = checkpoint.get(
            "episode",
            0,
        )

        best_score = checkpoint.get(
            "best_score",
            0,
        )

        return episode, best_score

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            return False

        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        self.brain.load_state_dict(
            state_dict
        )

        self.target_brain.load_state_dict(
            self.brain.state_dict()
        )

        print(
            f"Model loaded from: {MODEL_PATH}"
        )

        return True
