# NeuroSnake

A reinforcement learning experiment where a neural network learns to play Snake through trial and error.

The project started as a simple DQN-based Snake agent and is being developed as a foundation for experimenting with neural decision-making, reward-driven learning, and eventually more biologically inspired neural models.

## Overview

NeuroSnake uses **Deep Q-Learning (DQN)** to learn how to play Snake without being explicitly programmed with a strategy.

The agent observes the current game state, chooses an action, receives a reward or penalty, and gradually learns which actions lead to better long-term outcomes.

The current version uses a compact neural network with a recurrent, LIF-inspired neural processing mechanism.

## How It Works

The Snake agent receives an 11-dimensional state representation:

* Danger directly ahead
* Danger to the right
* Danger to the left
* Current movement direction
* Food position relative to the head

The network produces four Q-values corresponding to the possible actions:

```text
UP
DOWN
LEFT
RIGHT
```

The action with the highest estimated Q-value is selected during exploitation.

During training, the agent uses an epsilon-greedy policy to balance exploration and exploitation.

## Neural Model

The current model is intentionally small:

```text
Input:  14
Hidden: 64
Output: 4
```

The hidden layer uses a simplified LIF-inspired mechanism with:

* Membrane potential
* Voltage decay
* Spike generation
* Voltage reset
* Multiple neural ticks per decision

This is not intended to be a biological simulation of a fly brain. It is an experimental neural architecture inspired by spiking neural networks.

## Reinforcement Learning

The agent uses:

* Deep Q-Learning
* Experience Replay
* Target Network
* Epsilon-Greedy Exploration
* Huber Loss
* Gradient Clipping

Training experiences are stored in a replay buffer and sampled randomly to improve learning stability.

The model also supports checkpoints, allowing training to continue after the program or system is stopped.

## Reward System

The current reward function encourages the agent to:

| Event                  | Reward |
| ---------------------- | -----: |
| Eat food               |    +10 |
| Move closer to food    |  +0.10 |
| Move farther from food |  -0.10 |
| Normal movement        |  -0.02 |
| Collision              |    -10 |

The reward system is intentionally simple and will likely evolve as the agent becomes more capable.

## Project Structure

```text
NeuroSnake/
│
├── src/
│   ├── agent/
│   │   └── fly_agent.py
│   │
│   ├── brain/
│   │   └── fly_brain.py
│   │
│   ├── game/
│   │   └── snake_game.py
│   │
│   ├── rl/
│   │   └── replay_memory.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   ├── config.py
│   └── main.py
│
├── models/
├── requirements.txt
└── README.md
```

## Requirements

* Python 3.10+
* PyTorch
* NumPy
* Pygame

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Start training with:

```bash
python main.py train
```

If a checkpoint exists, training automatically continues from the latest saved checkpoint.

The training configuration can be changed in:

```text
src/config.py
```

For example:

```python
EPISODES = 10000
MAX_STEPS = 5000
BATCH_SIZE = 128
HIDDEN_SIZE = 64
NEURAL_TICKS = 6
```

## Playing

After training:

```bash
python main.py play
```

The trained model is loaded and Snake is controlled by the neural network.

## Checkpoints

Two model files are used:

```text
models/snake_fly_checkpoint.pt
models/snake_fly_dqn.pt
```

### Checkpoint

The checkpoint contains the training state, including:

* Neural network weights
* Target network weights
* Optimizer state
* Exploration state
* Training step
* Episode number
* Best score

This allows interrupted training to continue without starting over.

### Model

The final model contains the trained neural network weights and is used for playing/evaluation.

## Current Status

The agent is capable of learning non-trivial Snake behavior and has achieved scores above 50 during training.

However, the current agent still has important limitations. It can learn to reach food effectively but may eventually create its own dead ends by trapping itself inside its own body.

This behavior is expected at the current stage and is part of the experiment.

## Roadmap

The project is intentionally being developed incrementally.

Possible future directions include:

* Improving state representation
* Better survival and spatial-awareness strategies
* More biologically inspired spiking networks
* Larger neural architectures
* Alternative reinforcement learning algorithms
* Visualization of neural activity
* Experiments with real neural connectome data

The long-term goal is to use this type of environment as a controlled testbed for experimenting with biologically inspired neural computation.

## License

This project is an experimental research and learning project.
