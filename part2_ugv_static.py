"""
Part 2: UGV Navigation — Static Obstacles (A* / Dijkstra on 70×70 km Grid)
============================================================================
Models a 70×70 km battlefield as a grid.
Each cell = 1 km².  Three obstacle-density levels: low / medium / high.
Uses A* (heuristic = Euclidean distance) which is the optimal generalisation
of Dijkstra for grids with known obstacles.

Usage:
    python part2_ugv_static.py              # interactive
    python part2_ugv_static.py demo         # non-interactive demo
"""

import heapq
import random
import math
import time
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
GRID_SIZE   = 70          # 70 × 70 cells  (each cell = 1 km²)
MOVE_COST_STRAIGHT = 1.0  # km
MOVE_COST_DIAGONAL = math.sqrt(2)   # ≈ 1.414 km

OBSTACLE_DENSITY = {
    "low":    0.10,   # 10 % of cells are blocked
    "medium": 0.25,
    "high":   0.40,
}

# Direction vectors (8-connected grid)
DIRECTIONS = [
    (0,  1,  MOVE_COST_STRAIGHT),   # E
    (0, -1,  MOVE_COST_STRAIGHT),   # W
    (1,  0,  MOVE_COST_STRAIGHT),   # S
    (-1, 0,  MOVE_COST_STRAIGHT),   # N
    (1,  1,  MOVE_COST_DIAGONAL),   # SE
    (1, -1,  MOVE_COST_DIAGONAL),   # SW
    (-1, 1,  MOVE_COST_DIAGONAL),   # NE
    (-1,-1,  MOVE_COST_DIAGONAL),   # NW
]


# ─────────────────────────────────────────────────────────────────────────────
# Grid generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_grid(size, density_level, start, goal, seed=42):
    """
    Create a binary grid: 0 = free, 1 = obstacle.
    Guarantees start and goal cells are free.
    """
    random.seed(seed)
    density = OBSTACLE_DENSITY[density_level]
    grid = [[0] * size for _ in range(size)]

    for r in range(size):
        for c in range(size):
            if random.random() < density:
                grid[r][c] = 1

    # Clear start and goal
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]]   = 0
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Heuristics
# ─────────────────────────────────────────────────────────────────────────────

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def octile(a, b):
    """Octile distance — admissible heuristic for 8-connected grids."""
    dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
    return MOVE_COST_STRAIGHT * max(dx, dy) + (MOVE_COST_DIAGONAL - MOVE_COST_STRAIGHT) * min(dx, dy)


# ─────────────────────────────────────────────────────────────────────────────
# A* Search
# ─────────────────────────────────────────────────────────────────────────────

def astar(grid, start, goal):
    """
    A* on an 8-connected grid.

    Returns:
        path            : list of (row, col) from start → goal, or []
        g_cost          : actual path cost (km)
        nodes_expanded  : number of nodes popped from the open list
        nodes_generated : total nodes pushed onto the open list
        elapsed_ms      : search time in milliseconds
    """
    size = len(grid)
    t0   = time.time()

    g = {start: 0.0}
    parent = {start: None}
    open_list = [(octile(start, goal), 0.0, start)]
    closed = set()
    nodes_expanded  = 0
    nodes_generated = 1

    while open_list:
        f, cost, node = heapq.heappop(open_list)

        if node in closed:
            continue
        closed.add(node)
        nodes_expanded += 1

        if node == goal:
            # Reconstruct path
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            elapsed = (time.time() - t0) * 1000
            return path, cost, nodes_expanded, nodes_generated, elapsed

        r, c = node
        for dr, dc, move_cost in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == 0:
                neighbor = (nr, nc)
                new_g = cost + move_cost
                if neighbor not in g or new_g < g[neighbor]:
                    g[neighbor] = new_g
                    parent[neighbor] = node
                    h = octile(neighbor, goal)
                    heapq.heappush(open_list, (new_g + h, new_g, neighbor))
                    nodes_generated += 1

    elapsed = (time.time() - t0) * 1000
    return [], float('inf'), nodes_expanded, nodes_generated, elapsed   # no path


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_map(grid, path, start, goal, max_display=50):
    """
    ASCII render of the grid (capped at max_display × max_display for readability).
    Legend:
        S = Start    G = Goal    * = Path
        # = Obstacle   . = Free
    """
    size  = len(grid)
    scale = max(1, size // max_display)   # down-sample factor
    disp  = max_display

    path_set = set(path)

    print(f"\n  Map ({size}×{size} km grid, displayed at 1:{scale} scale — {disp}×{disp} characters)")
    print("  " + "┌" + "─" * disp + "┐")

    for r in range(0, size, scale):
        row_str = "│"
        for c in range(0, size, scale):
            cell_r = r // scale
            cell_c = c // scale
            if cell_r >= disp or cell_c >= disp:
                continue
            actual_r, actual_c = r, c
            if (actual_r, actual_c) == start:
                row_str += "S"
            elif (actual_r, actual_c) == goal:
                row_str += "G"
            elif (actual_r, actual_c) in path_set:
                row_str += "*"
            elif grid[actual_r][actual_c] == 1:
                row_str += "█"
            else:
                row_str += "·"
        print("  " + row_str[:disp+1] + "│")

    print("  " + "└" + "─" * disp + "┘")
    print("  Legend:  S=Start  G=Goal  *=Path  █=Obstacle  ·=Free")


# ─────────────────────────────────────────────────────────────────────────────
# Measures of Effectiveness
# ─────────────────────────────────────────────────────────────────────────────

def print_moe(path, path_cost, nodes_expanded, nodes_generated, elapsed_ms,
              grid, start, goal, density_level):
    size = len(grid)
    total_cells  = size * size
    obstacle_cnt = sum(grid[r][c] for r in range(size) for c in range(size))
    free_cnt     = total_cells - obstacle_cnt
    straight_line = euclidean(start, goal)
    path_efficiency = (straight_line / path_cost * 100) if path_cost > 0 else 0

    print("\n" + "═" * 55)
    print("  MEASURES OF EFFECTIVENESS (MoE)")
    print("═" * 55)
    print(f"  {'Grid size':<35}: {size}×{size} km ({total_cells:,} cells)")
    print(f"  {'Obstacle density level':<35}: {density_level}  ({OBSTACLE_DENSITY[density_level]*100:.0f}%)")
    print(f"  {'Obstacle cells':<35}: {obstacle_cnt:,} / {total_cells:,}  ({obstacle_cnt/total_cells*100:.1f}%)")
    print(f"  {'Free cells':<35}: {free_cnt:,}")
    print(f"  {'Start → Goal':<35}: {start} → {goal}")
    print("─" * 55)
    if path:
        print(f"  {'Path found':<35}: YES")
        print(f"  {'Path length (nodes)':<35}: {len(path)}")
        print(f"  {'Path cost (km — actual road)':<35}: {path_cost:.3f} km")
        print(f"  {'Straight-line distance (km)':<35}: {straight_line:.3f} km")
        print(f"  {'Path efficiency':<35}: {path_efficiency:.1f}%  (straight/actual × 100)")
        print(f"  {'Detour ratio':<35}: {path_cost/straight_line:.3f}  (actual / straight)")
    else:
        print(f"  {'Path found':<35}: NO (goal unreachable)")
    print("─" * 55)
    print(f"  {'Nodes expanded (closed set)':<35}: {nodes_expanded:,}")
    print(f"  {'Nodes generated (open list)':<35}: {nodes_generated:,}")
    print(f"  {'Search efficiency':<35}: {nodes_expanded/nodes_generated*100:.1f}%")
    print(f"  {'Search time':<35}: {elapsed_ms:.2f} ms")
    print("═" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(start, goal, density_level, seed=42):
    print(f"\n{'─'*55}")
    print(f"  Scenario: {start} → {goal}  |  Density: {density_level.upper()}")

    grid = generate_grid(GRID_SIZE, density_level, start, goal, seed=seed)
    path, path_cost, n_exp, n_gen, elapsed = astar(grid, start, goal)

    render_map(grid, path, start, goal)
    print_moe(path, path_cost, n_exp, n_gen, elapsed,
              grid, start, goal, density_level)
    return path


def demo():
    print("=" * 55)
    print("  PART 2: UGV NAVIGATION — STATIC OBSTACLES (A*)")
    print("=" * 55)

    scenarios = [
        ((0, 0),  (69, 69), "low"),
        ((5, 5),  (65, 60), "medium"),
        ((10, 10),(60, 60), "high"),
    ]
    for start, goal, density in scenarios:
        run_scenario(start, goal, density)


def interactive():
    print("=" * 55)
    print("  PART 2: UGV NAVIGATION — STATIC OBSTACLES (A*)")
    print("=" * 55)
    print(f"  Grid: {GRID_SIZE}×{GRID_SIZE} km  |  8-connected  |  Algorithm: A*")

    while True:
        print("\n" + "─" * 55)
        print("  1. Run a scenario")
        print("  2. Run all density demos")
        print("  3. Exit")
        choice = input("  Choice: ").strip()

        if choice == "1":
            try:
                sr = int(input(f"  Start row (0–{GRID_SIZE-1}): "))
                sc = int(input(f"  Start col (0–{GRID_SIZE-1}): "))
                gr = int(input(f"  Goal  row (0–{GRID_SIZE-1}): "))
                gc = int(input(f"  Goal  col (0–{GRID_SIZE-1}): "))
                dl = input("  Density (low/medium/high): ").strip().lower()
                seed = int(input("  Random seed (e.g. 42): ") or "42")
                if dl not in OBSTACLE_DENSITY:
                    print("  Invalid density."); continue
                run_scenario((sr, sc), (gr, gc), dl, seed)
            except ValueError:
                print("  Invalid input.")
        elif choice == "2":
            demo()
        elif choice == "3":
            break
        else:
            print("  Unknown choice.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    elif not sys.stdin.isatty():
        demo()
    else:
        interactive()
