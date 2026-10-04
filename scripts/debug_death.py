"""Debug: dissect the state where the planner failed to rescue."""

import sys

sys.path.insert(0, ".")

from src.agent.fly_agent import FlyAgent
from src.config.config import MAX_STEPS_PLAY, STUCK_STEPS
from src.game.snake_game import SnakeGame
from src.game.snake_planner import (
    SnakePlanner,
    VirtualSnake,
    DIRECTIONS,
)


class Shim:
    """Minimal game-like object for planner diagnostics."""

    def __init__(self, snake, food, direction):
        self.snake = snake
        self.food = food
        self.direction = direction


agent = FlyAgent()
agent.load_model()

planner = SnakePlanner()
game = SnakeGame(max_steps=MAX_STEPS_PLAY)

state = game.reset()
steps_since_food = 0
planner_driving = False

history = []

while True:
    action = agent.choose_action(state, training=False)

    if planner_driving:
        action, _ = planner.guide(game)
        override = True
    else:
        action, override = planner.decide(game, action)
        steps_since_food += 1

        if steps_since_food >= STUCK_STEPS:
            planner_driving = True
            action, _ = planner.guide(game)
            override = True

    history.append(
        {
            "snake": list(game.snake),
            "food": game.food,
            "direction": game.direction,
            "action": action,
            "override": override,
            "driving": planner_driving,
        }
    )

    state, reward, done = game.step(action)

    if reward >= 10.0:
        steps_since_food = 0
        planner_driving = False

    if done or game.steps >= 60_000:
        break

    if game.steps % 500 == 0:
        print(f"step {game.steps}: score {game.score}", flush=True)

print(f"Ended: score={game.score} steps={game.steps} won={game.won}")
print(f"Planner stats: {planner.stats}")

target = None
target_index = None

for index in range(len(history) - 1, -1, -1):
    entry = history[index]

    if entry["override"]:
        target = entry
        target_index = index
        break

print(f"Last overridden decision: step {target_index + 1}")

snake = target["snake"]
food = target["food"]
direction = target["direction"]

print(f"  head={snake[0]} dir={direction} len={len(snake)} "
      f"chosen={target['action']}")

wrapped = Shim(snake, food, direction)

print(f"  beeline_action: {planner._beeline_action(wrapped)}")
print(f"  tail_chase_action: {planner._tail_chase_action(wrapped)}")
print(f"  survival_action: {planner._survival_action(wrapped)}")
print(f"  any_surviving: {planner._any_surviving_action(wrapped)}")

virtual = VirtualSnake(snake, food, direction)
tail = virtual.snake[-1]
print(f"  tail={tail}")

for candidate in DIRECTIONS:
    probe = VirtualSnake(snake, food, direction)
    lethal = probe.step(candidate)

    if lethal:
        print(f"  action={candidate}: lethal")
        continue

    tail_ok = planner._tail_reachable(probe)
    space = planner._flood_fill(probe)
    print(f"  action={candidate}: tail_ok={tail_ok}, space={space}")
