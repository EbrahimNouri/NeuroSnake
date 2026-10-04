import torch
def __init__():
  pass

GRID_SIZE = 20
CELL_SIZE = GRID_SIZE

ACTION_COUNT: int = 4
OUTPUT_SIZE: int = ACTION_COUNT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE == torch.device("cuda"):
  torch.set_float32_matmul_precision("high")

EPISODES: int = 1_000

LEARNING_RATE: float = 0.001
GAMMA: float = 0.95

HIDDEN_SIZE: int = 128
NEURAL_TICKS = 200
LOOKAHEAD_STEPS: int = 20

# HIDDEN : 11 =>
# Immediate collision danger:       straight, right, left, danger_straight, danger_right, danger_left
# Current movement direction:       up, down, left, right, direction_up, direction_down, direction_left, direction_right
# Food position relative to the snake's head: food_left, food_right, food_up, food_down
INPUT_SIZE: int = 11 + (LOOKAHEAD_STEPS * 3)

BATCH_SIZE: int = 4096
MEMORY_SIZE: int = 10_000_000

PARALLEL_ENVIRONMENTS: int = 15

# Replay memory
# Transitions are stored in preallocated numpy arrays instead of a deque of
# python objects. float32 is exact; float16 halves RAM again (~2.8GB at 10M)
# and is safe here because observations are all in [0, 1].
REPLAY_DTYPE = "float32"
REPLAY_INITIAL_CAPACITY: int = 250_000

# The brain runs NEURAL_TICKS sequential steps, which means thousands of tiny
# kernels per forward. The model is far too small to fill the GPU, so it is
# launch-bound, not compute-bound. reduce-overhead replays the whole tick loop
# from a CUDA graph, which removes almost all of that dispatch cost.
# First run pays a one-off inductor compile (~2-4 min, then disk-cached).
COMPILE_BRAIN: bool = True
COMPILE_MODE: str = "reduce-overhead"

TARGET_UPDATE: int = 1000

EPSILON_START: float = 0.1
EPSILON_END: float = 0.0001
EPSILON_DECAY: float = 0.0381

MAX_STEPS: int = 50_000

MODEL_PATH: str = "models/snake_fly_dqn.pt"
CHECKPOINT_PATH: str = "models/snake_fly_checkpoint.pt"

PRINT_EVERY: int = 5

SPEED_PLAY: int = 100

# Reward settings
REWARD_FOOD: float = 10.0
REWARD_DEATH: float = -12.0

REWARD_STEP: float = -0.02
REWARD_CLOSER_FOOD: float = 0.1
REWARD_FARTHER_FOOD: float = -0.1


SAFETY_NET_ENABLED = True
STUCK_STEPS = 300
MAX_STEPS_PLAY = 100_000

def to_string():
  print(f"""
  GRID_SIZE: int  = {GRID_SIZE}
  CELL_SIZE: int  = {CELL_SIZE}
  ACTION_COUNT: int = {ACTION_COUNT}
  DEVICE = {DEVICE}
  EPISODES: int  = {EPISODES}
  LEARNING_RATE: float = {LEARNING_RATE}
  GAMMA: float = {GAMMA}
  HIDDEN_SIZE: int = {HIDDEN_SIZE}
  NEURAL_TICKS: int = {NEURAL_TICKS}
  LOOKAHEAD_STEPS: int = {LOOKAHEAD_STEPS}
  INPUT_SIZE: int  = {INPUT_SIZE}
  BATCH_SIZE: int  = {BATCH_SIZE}
  MEMORY_SIZE: int = {MEMORY_SIZE}
  REPLAY_DTYPE = {REPLAY_DTYPE}
  PARALLEL_ENVIRONMENTS: int = {PARALLEL_ENVIRONMENTS}
  COMPILE_BRAIN: bool = {COMPILE_BRAIN}
  COMPILE_MODE: str = {COMPILE_MODE}
  TARGET_UPDATE: int = {TARGET_UPDATE}
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
  PARALLEL_ENVIRONMENTS: int = {PARALLEL_ENVIRONMENTS}
  SAFETY_NET_ENABLED: int = {SAFETY_NET_ENABLED}
  STUCK_STEPS: int = {STUCK_STEPS}
  MAX_STEPS_PLAY: int = {MAX_STEPS_PLAY}
  """)
