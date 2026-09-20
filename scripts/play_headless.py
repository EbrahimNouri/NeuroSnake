"""Headless verification of the NeuroSnake play-mode safety net.

Runs N games with the same hybrid controller as `python main.py play`
(neural brain + pathfinding safety net) but without pygame, and prints
the board-fill rate, score statistics, and brain-vs-net move counts.

Usage:
    python scripts/play_headless.py [num_games]
"""

import sys
import time

sys.path.insert(0, ".")

from src.agent.fly_agent import FlyAgent
from src.config.config import (
    MAX_STEPS_PLAY,
    MODEL_PATH,
    SAFETY_NET_ENABLED,
    STUCK_STEPS,
)
from src.game.snake_game import SnakeGame
from src.game.snake_planner import SnakePlanner


def run_game(agent, planner, game):
    state = game.reset()

    brain_moves = 0
    net_moves = 0
    steps_since_food = 0
    planner_driving = False

    while True:
        action = agent.choose_action(
            state,
            training=False,
        )

        override = False

        if SAFETY_NET_ENABLED:
            if planner_driving:
                action, _ = planner.guide(game)
                override = True
            else:
                action, override = planner.decide(
                    game,
                    action,
                )

                steps_since_food += 1

                if steps_since_food >= STUCK_STEPS:
                    planner_driving = True
                    action, _ = planner.guide(game)
                    override = True

        if override:
            net_moves += 1
        else:
            brain_moves += 1

        state, reward, done = game.step(action)

        if reward >= 10.0:
            # Food eaten: reset stuck tracking and hand control
            # back to the brain.
            steps_since_food = 0
            planner_driving = False

        if done:
            return {
                "score": game.score,
                "won": game.won,
                "brain_moves": brain_moves,
                "net_moves": net_moves,
                "steps": game.steps,
            }


def main():
    num_games = 10

    if len(sys.argv) > 1:
        num_games = int(sys.argv[1])

    agent = FlyAgent()

    if not agent.load_model():
        print(f"Model not found: {MODEL_PATH}")
        return

    planner = SnakePlanner()
    game = SnakeGame(max_steps=MAX_STEPS_PLAY)

    scores = []
    wins = 0
    total_brain = 0
    total_net = 0

    start = time.time()

    for index in range(1, num_games + 1):
        result = run_game(agent, planner, game)

        scores.append(result["score"])
        total_brain += result["brain_moves"]
        total_net += result["net_moves"]

        if result["won"]:
            wins += 1

        status = (
            "FILLED"
            if result["won"]
            else f"ended (steps={result['steps']})"
        )

        print(
            f"Game {index:2d}: score={result['score']:3d} "
            f"{status} | brain={result['brain_moves']} "
            f"net={result['net_moves']}"
        )

    elapsed = time.time() - start

    max_score = 20 * 20 - 3
    average = sum(scores) / len(scores)

    print()
    print(f"Games:      {num_games}")
    print(f"Board fill: {wins}/{num_games} "
          f"({100.0 * wins / num_games:.1f}%)")
    print(f"Avg score:  {average:.1f} / {max_score}")
    print(f"Max score:  {max(scores)}")
    print(f"Brain moves: {total_brain} | Net moves: {total_net}")
    print(f"Elapsed:    {elapsed:.1f}s")


if __name__ == "__main__":
    main()
