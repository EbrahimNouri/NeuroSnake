import random
from collections import deque

import numpy as np


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.memory)


class NStepReplayMemory:

    def __init__(self, capacity, n_step, gamma):
        self.memory = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def _get_n_step_transition(self):
        reward = 0.0

        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * r

        state, action = self.n_step_buffer[0][:2]
        next_state, done = self.n_step_buffer[-1][3:]

        return state, action, reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append(
            (state, action, reward, next_state, done)
        )

        if len(self.n_step_buffer) < self.n_step:
            if done:
                self.n_step_buffer.clear()
            return

        self.memory.append(self._get_n_step_transition())

        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.memory)