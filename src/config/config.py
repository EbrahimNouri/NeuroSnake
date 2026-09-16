import torch

GRID_SIZE: int  = 20
CELL_SIZE: int  = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES: int  = 10_000

LEARNING_RATE: float = 0.001
GAMMA: float = 0.90

HIDDEN_SIZE: int = 128
NEURAL_TICKS: int = 6
LOOKAHEAD_STEPS: int = 20

# HIDDEN : 11 =>
# Immediate collision danger:       straight, right, left, danger_straight, danger_right, danger_left
# Current movement direction:       up, down, left, right, direction_up, direction_down, direction_left, direction_right
# Food position relative to the snake's head: food_left, food_right, food_up, food_down
INPUT_SIZE: int  = 11 + (LOOKAHEAD_STEPS * 3)

BATCH_SIZE: int  = 128
MEMORY_SIZE: int  = 100_000

TARGET_UPDATE: int  = 200

EPSILON_START: float  = 1.0
EPSILON_END: float = 0.01
EPSILON_DECAY: float = 0.995

MAX_STEPS: int = 10_000

MODEL_PATH: str = "models/snake_fly_dqn.pt"
CHECKPOINT_PATH: str = "models/snake_fly_checkpoint.pt"

PRINT_EVERY: int = 25

SPEED_PLAY: int = 20

# Reward settings
REWARD_FOOD: float =10.0
REWARD_DEATH: float = -12.0

REWARD_STEP: float = -0.02
REWARD_CLOSER_FOOD: float = 0.10
REWARD_FARTHER_FOOD: float = -0.10