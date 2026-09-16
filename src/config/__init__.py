import torch
def __init__():
  pass

GRID_SIZE: int = 20
CELL_SIZE: int = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES: int = 2_000

LEARNING_RATE: float = 0.001
GAMMA: float = 0.90

HIDDEN_SIZE: int = 128
NEURAL_TICKS: int = 6
LOOKAHEAD_STEPS: int = 20

# HIDDEN : 11 =>
# Immediate collision danger:       straight, right, left, danger_straight, danger_right, danger_left
# Current movement direction:       up, down, left, right, direction_up, direction_down, direction_left, direction_right
# Food position relative to the snake's head: food_left, food_right, food_up, food_down
INPUT_SIZE: int = 11 + (LOOKAHEAD_STEPS * 3)

BATCH_SIZE: int = 256
MEMORY_SIZE: int = 20_000

TARGET_UPDATE: int = 200

EPSILON_START: float = 1.0
EPSILON_END: float = 0.005
EPSILON_DECAY: float = 0.955

MAX_STEPS: int = 10_000

MODEL_PATH: str = "models/snake_fly_dqn.pt"
CHECKPOINT_PATH: str = "models/snake_fly_checkpoint.pt"

PRINT_EVERY: int = 5

SPEED_PLAY: int = 20

# Reward settings
REWARD_FOOD: float = 11.0
REWARD_DEATH: float = -13.0

REWARD_STEP: float = -0.02
REWARD_CLOSER_FOOD: float = 0.10
REWARD_FARTHER_FOOD: float = -0.09


def to_string():
  print(f"""
  GRID_SIZE: int  = {GRID_SIZE}
  CELL_SIZE: int  = {CELL_SIZE}
  DEVICE = {DEVICE}
  EPISODES: int  = {EPISODES}
  LEARNING_RATE: float = {LEARNING_RATE}
  GAMMA: float = {GAMMA}
  HIDDEN_SIZE: int = {HIDDEN_SIZE}
  NEURAL_TICKS: int = {NEURAL_TICKS}
  LOOKAHEAD_STEPS: int = {LOOKAHEAD_STEPS}
  INPUT_SIZE: int  = {INPUT_SIZE}
  BATCH_SIZE: int  = {BATCH_SIZE}
  MEMORY_SIZE: int  = {MEMORY_SIZE}
  TARGET_UPDATE: int  = {TARGET_UPDATE}
  EPSILON_START: float  = {EPSILON_START}
  EPSILON_END: float = {EPSILON_END}
  EPSILON_DECAY: float = {EPSILON_DECAY}
  MAX_STEPS: int = {MAX_STEPS}
  MODEL_PATH: str = {MODEL_PATH}
  CHECKPOINT_PATH: str = {CHECKPOINT_PATH}
  PRINT_EVERY: int = {PRINT_EVERY}
  SPEED_PLAY: int = {SPEED_PLAY}
  REWARD_FOOD: float = {REWARD_FOOD}
  REWARD_DEATH: float = {REWARD_DEATH}
  REWARD_STEP: float = {REWARD_STEP}
  REWARD_CLOSER_FOOD: float = {REWARD_CLOSER_FOOD}
  REWARD_FARTHER_FOOD: float = {REWARD_FARTHER_FOOD}
  """)
