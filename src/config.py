import torch


GRID_SIZE = 20
CELL_SIZE = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 10000

LEARNING_RATE = 0.001
GAMMA = 0.90

HIDDEN_SIZE = 64
NEURAL_TICKS = 6

BATCH_SIZE = 128
MEMORY_SIZE = 100_000

TARGET_UPDATE = 200

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

MAX_STEPS = 5000

MODEL_PATH = "models/snake_fly_dqn.pt"
CHECKPOINT_PATH = "models/snake_fly_checkpoint.pt"

PRINT_EVERY = 25