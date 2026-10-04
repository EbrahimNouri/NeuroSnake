import numpy as np

from src.config import (
    INPUT_SIZE,
    MEMORY_SIZE,
    REPLAY_DTYPE,
    REPLAY_INITIAL_CAPACITY,
)

ACTION_COUNT = 4


class ReplayMemory:

    def __init__(
        self,
        capacity=MEMORY_SIZE,
        state_size=INPUT_SIZE,
        dtype=REPLAY_DTYPE,
    ):
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.dtype = np.dtype(dtype)

        self._allocate(
            min(self.capacity, REPLAY_INITIAL_CAPACITY)
        )

        self.position = 0
        self.size = 0

    def _allocate(self, length):
        self.states = np.empty(
            (length, self.state_size),
            dtype=self.dtype,
        )

        self.next_states = np.empty(
            (length, self.state_size),
            dtype=self.dtype,
        )

        self.actions = np.empty(
            length,
            dtype=np.int8,
        )

        self.rewards = np.empty(
            length,
            dtype=np.float32,
        )

        self.dones = np.empty(
            length,
            dtype=bool,
        )

        self._length = length

    def _grow(self):
        length = min(self._length * 2, self.capacity)

        if length == self._length:
            return False

        previous_size = self.size

        old_states = self.states
        old_next_states = self.next_states
        old_actions = self.actions
        old_rewards = self.rewards
        old_dones = self.dones

        self._allocate(length)

        index = np.arange(previous_size)

        self.states[index] = old_states[index]
        self.next_states[index] = old_next_states[index]
        self.actions[index] = old_actions[index]
        self.rewards[index] = old_rewards[index]
        self.dones[index] = old_dones[index]

        # The buffer was full, so every old entry is copied and the next
        # write continues right after them.
        self.position = previous_size
        self.size = previous_size

        return True

    def push(self, state, action, reward, next_state, done):
        index = self.position

        self.states[index] = state
        self.next_states[index] = next_state

        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done

        self.position = index + 1

        if self.position == self._length:
            self.position = 0

        if self.size < self._length:
            self.size += 1

        if self.size == self._length:
            self._grow()

    def sample(self, batch_size):
        if batch_size >= self.size:
            index = np.arange(self.size)
        else:
            # Without replacement, matching random.sample on the old deque.
            index = np.random.choice(
                self.size,
                size=batch_size,
                replace=False,
            )

        return (
            np.asarray(
                self.states[index],
                dtype=np.float32,
            ),
            np.asarray(
                self.actions[index],
                dtype=np.int64,
            ),
            self.rewards[index],
            np.asarray(
                self.next_states[index],
                dtype=np.float32,
            ),
            self.dones[index].astype(np.float32),
        )

    def nbytes(self):
        return int(
            self.states.nbytes
            + self.next_states.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.dones.nbytes
        )

    def __len__(self):
        return self.size