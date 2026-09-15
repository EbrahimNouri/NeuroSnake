import torch


GRID_SIZE = 20
CELL_SIZE = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 10_000

LEARNING_RATE = 0.001
GAMMA = 0.90

HIDDEN_SIZE: int = 128
NEURAL_TICKS = 6
LOOKAHEAD_STEPS = 12

# HIDDEN : 11 =>
# Immediate collision danger:       straight, right, left, danger_straight, danger_right, danger_left
# Current movement direction:       up, down, left, right, direction_up, direction_down, direction_left, direction_right
# Food position relative to the snake's head: food_left, food_right, food_up, food_down
INPUT_SIZE = 11 + (LOOKAHEAD_STEPS * 3)

BATCH_SIZE = 128
MEMORY_SIZE = 100_000

TARGET_UPDATE = 200

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

MAX_STEPS = 10_000

MODEL_PATH = "models/snake_fly_dqn.pt"
CHECKPOINT_PATH = "models/snake_fly_checkpoint.pt"

PRINT_EVERY = 25

SPEED_PLAY = 50