import heapq


def a_star(snake, food, grid_size):
    start = snake[0]
    blocked = set(snake[1:])

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(heuristic(start, food), start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in closed:
            continue

        if current == food:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        closed.add(current)

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < grid_size
                    and 0 <= neighbor[1] < grid_size):
                continue

            if neighbor in blocked:
                continue

            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor, food)
                heapq.heappush(open_set, (f, neighbor))

    return None

def path_to_action(snake, path):
    if not path:
        return None

    head = snake[0]
    next_pos = path[0]

    dx = next_pos[0] - head[0]
    dy = next_pos[1] - head[1]

    if (dx, dy) == (0, -1):
        return 0
    elif (dx, dy) == (0, 1):
        return 1
    elif (dx, dy) == (-1, 0):
        return 2
    elif (dx, dy) == (1, 0):
        return 3

    return None

def choose_safe_action(snake, food, grid_size, current_direction):
    path = a_star(snake, food, grid_size)
    action = path_to_action(snake, path)

    if action is not None:
        dx, dy = current_direction

        if (dx, dy) == (0, -1) and action == 1:
            action = None
        elif (dx, dy) == (0, 1) and action == 0:
            action = None
        elif (dx, dy) == (-1, 0) and action == 3:
            action = None
        elif (dx, dy) == (1, 0) and action == 2:
            action = None

    if action is not None:
        return action

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    head = snake[0]
    dx, dy = current_direction

    safe_actions = []
    for i, (ddx, ddy) in enumerate(directions):
        if (ddx, ddy) == (-dx, -dy):
            continue

        nx, ny = head[0] + ddx, head[1] + ddy
        if (0 <= nx < grid_size and 0 <= ny < grid_size
                and (nx, ny) not in snake):
            safe_actions.append(i)

    if safe_actions:
        return safe_actions[0]

    return 0