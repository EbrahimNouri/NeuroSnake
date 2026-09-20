"""Play-mode safety net for NeuroSnake.

The trained neural brain proposes moves, but this planner vets them:
a move is only accepted if it is non-lethal AND leaves the snake's
tail reachable (flood-fill check), so the snake can never trap itself.

When the brain's move is unsafe, the planner rescues with, in order:
  1. BFS beeline to the food (if the whole path stays safe)
  2. BFS chase of its own tail (classic survival strategy)
  3. Move that maximizes reachable free space
  4. Any non-lethal move (last resort)

All search runs on virtual copies; the real game is never mutated.
"""

from collections import deque

from src.config.config import GRID_SIZE

# Action indices, matching SnakeGame.step.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
VECTORS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}


def _neighbors(cell):
    x, y = cell
    return (
        (x, y - 1),
        (x, y + 1),
        (x - 1, y),
        (x + 1, y),
    )


def _in_bounds(cell):
    x, y = cell
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def _action_from_delta(delta):
    for action, vector in VECTORS.items():
        if vector == delta:
            return action

    return None


class VirtualSnake:
    """Copy of a SnakeGame state used for harmless simulation."""

    __slots__ = ("snake", "food", "direction")

    def __init__(self, snake, food, direction):
        self.snake = list(snake)
        self.food = food
        self.direction = direction

    @classmethod
    def from_game(cls, game):
        return cls(game.snake, game.food, game.direction)

    def copy(self):
        return VirtualSnake(
            list(self.snake),
            self.food,
            self.direction,
        )

    def effective_direction(self, action):
        dx, dy = self.direction
        opposite = (-dx, -dy)

        if VECTORS[action] == opposite:
            # The engine ignores reverse commands: snake keeps going.
            return self.direction

        return VECTORS[action]

    def step(self, action):
        """Apply one move in place. Returns True if the move is lethal."""
        self.direction = self.effective_direction(action)

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if (
            not _in_bounds(new_head)
            or new_head in self.snake
        ):
            return True

        self.snake.insert(0, new_head)

        if new_head == self.food:
            # Ate: body grows (no tail pop this move).
            self.food = None
        else:
            self.snake.pop()

        return False

    def is_occupied(self, cell):
        if cell == self.food:
            return False

        return cell in self.snake


class SnakePlanner:
    """Pathfinding safety net that assists the neural brain in play mode."""

    def __init__(self):
        # Decision-branch counters for verification runs.
        self.stats = {
            "beeline": 0,
            "tail_chase": 0,
            "survival": 0,
            "any": 0,
            "failed": 0,
        }

    def decide(self, game, brain_action):
        """Return (action, overrode) for the current game state."""
        virtual = VirtualSnake.from_game(game)

        if virtual.step(brain_action):
            # Brain move kills the snake directly: must override.
            return self._rescue(game, brain_action)

        if not self._is_safe(virtual):
            # Brain move survives now but leads into a trap: override.
            return self._rescue(game, brain_action)

        return brain_action, False

    def guide(self, game):
        """Planner-driven move, used when the brain is stuck.

        Returns (action, overrode); action is None only in a truly
        dead position.
        """
        action = self._beeline_action(game)

        if action is not None:
            self.stats["beeline"] += 1
            return action, True

        action = self._tail_chase_action(game)

        if action is not None:
            self.stats["tail_chase"] += 1
            return action, True

        action = self._survival_action(game)

        if action is not None:
            self.stats["survival"] += 1
            return action, True

        action = self._any_surviving_action(game)

        if action is not None:
            self.stats["any"] += 1
            return action, True

        self.stats["failed"] += 1
        return None, True

    def _rescue(self, game, brain_action):
        action, _ = self.guide(game)

        if action is None:
            # Truly dead position: keep the brain's choice.
            return brain_action, True

        return action, True

    def _beeline_action(self, game):
        """First step of a shortest BFS path to the food, but only if
        walking the whole path (and eating) stays alive and untrapped."""
        virtual = VirtualSnake.from_game(game)

        path = self._bfs_path(
            virtual,
            start=virtual.snake[0],
            goal=virtual.food,
        )

        if path is None:
            return None

        return self._verified_path_action(game, path)

    def _tail_chase_action(self, game):
        """First step of a BFS path to a cell adjacent to the tail.

        Chasing the tail keeps the snake alive and coiled until a
        beeline becomes safe again. Every candidate step is still
        verified with a full path simulation.
        """
        virtual = VirtualSnake.from_game(game)

        tail = virtual.snake[-1]

        for target in _neighbors(tail):
            if not _in_bounds(target):
                continue

            if virtual.is_occupied(target):
                continue

            path = self._bfs_path(
                virtual,
                start=virtual.snake[0],
                goal=target,
            )

            if path is None:
                continue

            action = self._verified_path_action(game, path)

            if action is not None:
                return action

        return None

    def _survival_action(self, game):
        """Safe move that maximizes reachable free space."""
        best_action = None
        best_space = -1

        for action in DIRECTIONS:
            virtual = VirtualSnake.from_game(game)

            if virtual.step(action):
                continue

            if not self._is_safe(virtual):
                continue

            space = self._flood_fill(virtual)

            if space > best_space:
                best_space = space
                best_action = action

        return best_action

    def _any_surviving_action(self, game):
        """Last resort: non-lethal move maximizing open space."""
        best_action = None
        best_space = -1

        for action in DIRECTIONS:
            virtual = VirtualSnake.from_game(game)

            if virtual.step(action):
                continue

            space = self._flood_fill(virtual)

            if space > best_space:
                best_space = space
                best_action = action

        return best_action

    def _verified_path_action(self, game, path):
        """Simulate a full BFS path (with eating) and return the first
        action only if the snake stays alive and untrapped throughout."""
        if len(path) < 2:
            return None

        head_x, head_y = game.snake[0]
        first_x, first_y = path[1]

        first_action = _action_from_delta(
            (first_x - head_x, first_y - head_y)
        )

        if first_action is None:
            return None

        simulated = VirtualSnake.from_game(game)

        for cell in path[1:]:
            dx = cell[0] - simulated.snake[0][0]
            dy = cell[1] - simulated.snake[0][1]

            action = _action_from_delta((dx, dy))

            if action is None:
                return None

            if simulated.step(action):
                return None

        if not self._is_safe(simulated):
            return None

        return first_action

    def _bfs_path(self, virtual, start, goal):
        """Shortest path from start to goal over free cells.

        The goal cell is always enterable (food is never occupied; a
        tail-adjacent target was already checked by the caller).
        """
        if goal is None:
            return None

        if start == goal:
            return [start]

        came_from = {start: None}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            if current == goal:
                path = []
                node = current

                while node is not None:
                    path.append(node)
                    node = came_from[node]

                path.reverse()
                return path

            for neighbor in _neighbors(current):
                if not _in_bounds(neighbor):
                    continue

                if neighbor in came_from:
                    continue

                if neighbor != goal and virtual.is_occupied(neighbor):
                    continue

                came_from[neighbor] = current
                queue.append(neighbor)

        return None

    def _is_safe(self, virtual):
        """True if the state can survive indefinitely.

        Requires tail access plus the tail-lap invariant: the snake
        can walk a complete BFS path to its (current) tail cell and
        still reach the tail afterwards. A state that passes this can
        keep chasing its tail forever, so no accepted move ever leads
        into a shrinking pocket.
        """
        if not self._tail_reachable(virtual):
            return False

        return self._tail_lap_survivable(virtual)

    def _tail_lap_survivable(self, virtual):
        """Simulate one full lap: follow the shortest BFS path to the
        current tail cell, then confirm the tail is still reachable.

        The walk is collision-free by construction: the path only
        crosses cells that are free at planning time and the body only
        vacates cells as it advances (the tail cell itself was vacated
        on the first move). If the head eats mid-lap the body grows,
        so the path is then recomputed from the new state.
        """
        probe = virtual.copy()

        guard = GRID_SIZE * GRID_SIZE * 2

        while guard > 0:
            guard -= 1

            path = self._bfs_path(
                probe,
                start=probe.snake[0],
                goal=probe.snake[-1],
            )

            if path is None or len(path) < 2:
                return False

            for cell in path[1:]:
                dx = cell[0] - probe.snake[0][0]
                dy = cell[1] - probe.snake[0][1]

                action = _action_from_delta((dx, dy))

                if action is None:
                    return False

                if probe.step(action):
                    return False

                if probe.food is None:
                    # Ate mid-lap: body grew, recompute the lap path.
                    probe.food = cell
                    break

            else:
                # Completed the lap without eating.
                return self._tail_reachable(probe)

        return False

    def _tail_reachable(self, virtual):
        """True if the snake's tail cell is reachable from the head.

        The tail cell frees up as the snake moves, so it is treated
        as passable for the flood fill.
        """
        blocked = set(virtual.snake[1:-1])

        start = virtual.snake[0]
        goal = virtual.snake[-1]

        if start == goal:
            return True

        visited = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for neighbor in _neighbors(current):
                if not _in_bounds(neighbor):
                    continue

                if neighbor in visited:
                    continue

                if neighbor != goal and neighbor in blocked:
                    continue

                visited.add(neighbor)
                queue.append(neighbor)

        return goal in visited

    def _flood_fill(self, virtual):
        """Number of free cells reachable from the head."""
        start = virtual.snake[0]

        blocked = set(virtual.snake[1:])

        visited = {start}
        queue = deque([start])
        count = 0

        while queue:
            current = queue.popleft()

            for neighbor in _neighbors(current):
                if not _in_bounds(neighbor):
                    continue

                if neighbor in visited or neighbor in blocked:
                    continue

                visited.add(neighbor)
                queue.append(neighbor)
                count += 1

        return count
