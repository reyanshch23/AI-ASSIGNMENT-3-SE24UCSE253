"""
Part 3: UGV Navigation — Dynamic Obstacles (D* Lite)
======================================================
Extends Part 2 by relaxing the static-obstacle assumption.

Algorithm : D* Lite (Koenig & Likhachev, 2002)
             — the standard algorithm for replanning in partially-known
               and dynamically-changing environments.

Simulation:
  • UGV starts navigating along the initially-planned A* path.
  • After every N steps, a batch of NEW obstacles appears (simulating
    enemy vehicles, debris, etc. revealed by sensors at close range).
  • D* Lite replans from the UGV's current position in O(changes) time,
    far cheaper than a full A* re-run from scratch.

Usage:
    python part3_ugv_dynamic.py              # interactive
    python part3_ugv_dynamic.py demo         # non-interactive demo
"""

import heapq
import random
import math
import time
import sys
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
GRID_SIZE          = 70
SENSOR_RADIUS      = 8          # UGV can sense obstacles within 8 km
OBSTACLE_DENSITY   = 0.10       # initial known obstacle density
DYNAMIC_OBSTACLE_BATCHES = 4    # how many times new obstacles pop up
NEW_OBSTACLES_PER_BATCH  = 30   # obstacles revealed per batch

MOVE_COST_STRAIGHT = 1.0
MOVE_COST_DIAGONAL = math.sqrt(2)

DIRECTIONS = [
    (0,  1,  MOVE_COST_STRAIGHT),
    (0, -1,  MOVE_COST_STRAIGHT),
    (1,  0,  MOVE_COST_STRAIGHT),
    (-1, 0,  MOVE_COST_STRAIGHT),
    (1,  1,  MOVE_COST_DIAGONAL),
    (1, -1,  MOVE_COST_DIAGONAL),
    (-1, 1,  MOVE_COST_DIAGONAL),
    (-1,-1,  MOVE_COST_DIAGONAL),
]


# ─────────────────────────────────────────────────────────────────────────────
# D* Lite implementation
# Reference: Koenig & Likhachev (2002) "D* Lite"
# ─────────────────────────────────────────────────────────────────────────────

INF = float('inf')

class DStarLite:
    """
    D* Lite searches from GOAL → START (reversed), maintaining g and rhs values.
    After initialisation, call compute_shortest_path() to get the first plan.
    When the map changes, call update_vertex() for each changed cell, then
    call compute_shortest_path() again for the replan — only affected nodes
    are re-expanded.
    """

    def __init__(self, grid, start, goal):
        self.size   = len(grid)
        self.grid   = [row[:] for row in grid]   # mutable copy
        self.start  = start
        self.goal   = goal
        self.k_m    = 0          # key modifier for moved start

        self.g   = defaultdict(lambda: INF)
        self.rhs = defaultdict(lambda: INF)
        self.rhs[goal] = 0.0

        self.open_list = []       # (key_tuple, node)
        self.open_set  = {}       # node → key_tuple (for membership check)
        self._insert(goal, self._calculate_key(goal))

        self.replans           = 0
        self.total_nodes_exp   = 0

    # ── Key calculation ──────────────────────────────────────────────────────

    def _h(self, a, b=None):
        if b is None: b = self.start
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _calculate_key(self, u):
        g_rhs = min(self.g[u], self.rhs[u])
        return (g_rhs + self._h(u) + self.k_m, g_rhs)

    # ── Priority queue helpers ────────────────────────────────────────────────

    def _insert(self, u, key):
        heapq.heappush(self.open_list, (key, u))
        self.open_set[u] = key

    def _top_key(self):
        while self.open_list:
            key, u = self.open_list[0]
            if self.open_set.get(u) == key:
                return key
            heapq.heappop(self.open_list)
        return (INF, INF)

    def _pop(self):
        while self.open_list:
            key, u = heapq.heappop(self.open_list)
            if self.open_set.get(u) == key:
                del self.open_set[u]
                return u
        return None

    # ── Neighbour / cost ─────────────────────────────────────────────────────

    def _neighbours(self, u):
        r, c = u
        for dr, dc, cost in DIRECTIONS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield (nr, nc), cost

    def _c(self, u, v):
        """Edge cost: INF if either endpoint is an obstacle."""
        if self.grid[u[0]][u[1]] == 1 or self.grid[v[0]][v[1]] == 1:
            return INF
        # find the move cost for this direction
        dr, dc = v[0]-u[0], v[1]-u[1]
        return MOVE_COST_DIAGONAL if abs(dr) == 1 and abs(dc) == 1 else MOVE_COST_STRAIGHT

    # ── Core D* Lite ─────────────────────────────────────────────────────────

    def _update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min(
                self._c(u, v) + self.g[v]
                for v, _ in self._neighbours(u)
            )
        if u in self.open_set:
            del self.open_set[u]
        if self.g[u] != self.rhs[u]:
            self._insert(u, self._calculate_key(u))

    def compute_shortest_path(self):
        nodes_expanded = 0
        while (self._top_key() < self._calculate_key(self.start)
               or self.rhs[self.start] != self.g[self.start]):
            k_old = self._top_key()
            u = self._pop()
            if u is None:
                break
            nodes_expanded += 1
            k_new = self._calculate_key(u)
            if k_old < k_new:
                self._insert(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for v, _ in self._neighbours(u):
                    self._update_vertex(v)
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for v, _ in self._neighbours(u):
                    self._update_vertex(v)
        self.total_nodes_exp += nodes_expanded
        return nodes_expanded

    def update_obstacle(self, cell, is_obstacle):
        """
        Call this when a cell's status changes.
        is_obstacle = True  → cell became blocked
        is_obstacle = False → cell was cleared
        """
        self.grid[cell[0]][cell[1]] = 1 if is_obstacle else 0
        for v, _ in self._neighbours(cell):
            self._update_vertex(v)
        self._update_vertex(cell)
        self.replans += 1

    def get_path(self):
        """Trace the greedy path from start → goal through g values."""
        path = [self.start]
        current = self.start
        visited = {current}
        for _ in range(self.size * self.size):
            if current == self.goal:
                break
            best, best_cost = None, INF
            for v, c in self._neighbours(current):
                if v not in visited:
                    total = self._c(current, v) + self.g[v]
                    if total < best_cost:
                        best_cost, best = total, v
            if best is None:
                return []       # no path
            path.append(best)
            visited.add(best)
            current = best
        return path if current == self.goal else []

    def move_start(self, new_start):
        """Update D* Lite when the UGV moves to a new cell."""
        self.k_m   += self._h(self.start, new_start)
        self.start  = new_start


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def octile(a, b):
    dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
    return MOVE_COST_STRAIGHT * max(dx, dy) + (MOVE_COST_DIAGONAL - MOVE_COST_STRAIGHT) * min(dx, dy)


def generate_initial_grid(size, density, start, goal, seed):
    random.seed(seed)
    grid = [[0]*size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if random.random() < density:
                grid[r][c] = 1
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]]   = 0
    return grid


def reveal_dynamic_obstacles(dstar, free_cells, ugv_pos, n_new, seed_offset):
    """Add random new obstacles that are NOT adjacent to the UGV."""
    random.seed(seed_offset)
    random.shuffle(free_cells)
    added = []
    for cell in free_cells:
        if len(added) >= n_new:
            break
        r, c = cell
        # Don't block the UGV's current cell or goal
        if cell == dstar.start or cell == dstar.goal:
            continue
        # Keep a safety bubble around UGV so it's never instantly trapped
        if octile(cell, ugv_pos) < 3:
            continue
        dstar.grid[r][c] = 1
        dstar.update_obstacle(cell, True)
        added.append(cell)
    # Remove added cells from free_cells
    for cell in added:
        if cell in free_cells:
            free_cells.remove(cell)
    return added


def render_map(grid, path, start, goal, ugv_pos, dynamic_obs, max_display=50):
    size  = len(grid)
    scale = max(1, size // max_display)
    disp  = max_display

    path_set    = set(path)
    dynamic_set = set(dynamic_obs)

    print(f"\n  Map ({size}×{size} km grid, 1:{scale} scale)")
    print("  ┌" + "─" * disp + "┐")
    for r in range(0, size, scale):
        row_str = "│"
        for c in range(0, size, scale):
            if len(row_str) - 1 >= disp:
                break
            actual = (r, c)
            if actual == start:
                row_str += "S"
            elif actual == goal:
                row_str += "G"
            elif actual == ugv_pos:
                row_str += "U"
            elif actual in path_set:
                row_str += "·"
            elif actual in dynamic_set:
                row_str += "X"
            elif grid[r][c] == 1:
                row_str += "█"
            else:
                row_str += " "
        print("  " + row_str[:disp+1] + "│")
    print("  └" + "─" * disp + "┘")
    print("  S=Start  G=Goal  U=UGV  ·=Planned path  X=New obstacle  █=Static obstacle")


# ─────────────────────────────────────────────────────────────────────────────
# Measures of Effectiveness
# ─────────────────────────────────────────────────────────────────────────────

def print_moe(stats):
    print("\n" + "═" * 58)
    print("  MEASURES OF EFFECTIVENESS — DYNAMIC OBSTACLE NAVIGATION")
    print("═" * 58)
    for key, val in stats.items():
        print(f"  {key:<40}: {val}")
    print("═" * 58)


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(start, goal, seed=42, verbose=True):
    print(f"\n{'─'*58}")
    print(f"  D* Lite Simulation  |  {start} → {goal}  |  seed={seed}")

    t_total = time.time()

    # 1. Build initial (partially known) grid
    grid = generate_initial_grid(GRID_SIZE, OBSTACLE_DENSITY, start, goal, seed)
    free_cells = [
        (r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
        if grid[r][c] == 0 and (r, c) != start and (r, c) != goal
    ]

    # 2. Initialise D* Lite and compute first plan
    dstar = DStarLite(grid, start, goal)
    n_initial = dstar.compute_shortest_path()

    initial_path = dstar.get_path()
    if not initial_path:
        print("  ✗ Initial plan: No path found (try a different seed).")
        return

    if verbose:
        print(f"\n  Initial plan found — {len(initial_path)} nodes, "
              f"{n_initial} nodes expanded")
        render_map(dstar.grid, initial_path, start, goal, start, [])

    # 3. Simulate UGV movement with dynamic obstacle reveals
    ugv_pos          = start
    full_trajectory  = [ugv_pos]
    all_dynamic_obs  = []
    replans          = 0
    replan_times_ms  = []
    steps_moved      = 0
    path_cost        = 0.0

    # Walk along the path; reveal new obstacles periodically
    current_path = list(initial_path)
    step_idx     = 0
    batch_interval = max(1, len(current_path) // (DYNAMIC_OBSTACLE_BATCHES + 1))

    while ugv_pos != goal:
        # Reveal a batch of new obstacles?
        if step_idx > 0 and step_idx % batch_interval == 0:
            new_obs = reveal_dynamic_obstacles(
                dstar, free_cells, ugv_pos,
                NEW_OBSTACLES_PER_BATCH, seed + step_idx
            )
            all_dynamic_obs.extend(new_obs)
            if new_obs:
                replans += 1
                t0 = time.time()
                dstar.compute_shortest_path()
                replan_times_ms.append((time.time() - t0) * 1000)
                current_path = dstar.get_path()
                if not current_path:
                    print(f"\n  ✗ Goal became unreachable after obstacle batch {replans}.")
                    break
                if verbose:
                    print(f"\n  [Batch {replans}] {len(new_obs)} new obstacles revealed at step {step_idx}.")
                    print(f"  Replanned: new path length = {len(current_path)} nodes")
                    render_map(dstar.grid, current_path, start, goal, ugv_pos, all_dynamic_obs)

        # Move one step
        if step_idx < len(current_path) - 1:
            next_pos = current_path[step_idx + 1]
            move_c   = MOVE_COST_DIAGONAL if abs(next_pos[0]-ugv_pos[0]) == 1 and \
                                             abs(next_pos[1]-ugv_pos[1]) == 1 else MOVE_COST_STRAIGHT
            path_cost  += move_c
            ugv_pos     = next_pos
            full_trajectory.append(ugv_pos)
            steps_moved += 1
            dstar.move_start(ugv_pos)
            step_idx   += 1
        else:
            # Ran off the end of the current path — replan
            dstar.compute_shortest_path()
            current_path = dstar.get_path()
            if not current_path:
                print("  ✗ No path to goal from current position.")
                break
            step_idx = 0

        if steps_moved > GRID_SIZE * GRID_SIZE:
            print("  ✗ Exceeded maximum steps — aborting.")
            break

    t_elapsed = (time.time() - t_total) * 1000
    reached   = (ugv_pos == goal)
    sl_dist   = math.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2)

    stats = {
        "Grid size"                        : f"{GRID_SIZE}×{GRID_SIZE} km",
        "Initial obstacle density"         : f"{OBSTACLE_DENSITY*100:.0f}%",
        "Dynamic obstacle batches revealed": DYNAMIC_OBSTACLE_BATCHES,
        "New obstacles per batch"          : NEW_OBSTACLES_PER_BATCH,
        "Total new obstacles"              : len(all_dynamic_obs),
        "Start → Goal"                     : f"{start} → {goal}",
        "Goal reached"                     : "YES ✓" if reached else "NO ✗",
        "Steps taken"                      : steps_moved,
        "Actual path cost (km)"            : f"{path_cost:.3f}",
        "Straight-line distance (km)"      : f"{sl_dist:.3f}",
        "Path efficiency"                  : f"{sl_dist/path_cost*100:.1f}%" if path_cost else "N/A",
        "Number of replans"                : replans,
        "Avg replan time (ms)"             : f"{sum(replan_times_ms)/len(replan_times_ms):.2f}" if replan_times_ms else "N/A",
        "Total nodes expanded"             : dstar.total_nodes_exp,
        "Total simulation time (ms)"       : f"{t_elapsed:.2f}",
    }
    print_moe(stats)
    return stats


def demo():
    print("=" * 58)
    print("  PART 3: UGV NAVIGATION — DYNAMIC OBSTACLES (D* Lite)")
    print("=" * 58)

    scenarios = [
        ((0, 0),  (69, 69), 42),
        ((5, 10), (64, 58), 7),
    ]
    for start, goal, seed in scenarios:
        run_simulation(start, goal, seed=seed, verbose=True)


def interactive():
    print("=" * 58)
    print("  PART 3: UGV NAVIGATION — DYNAMIC OBSTACLES (D* Lite)")
    print("=" * 58)

    while True:
        print("\n" + "─" * 58)
        print("  1. Run simulation")
        print("  2. Run demo scenarios")
        print("  3. Exit")
        choice = input("  Choice: ").strip()

        if choice == "1":
            try:
                sr   = int(input(f"  Start row (0–{GRID_SIZE-1}): "))
                sc   = int(input(f"  Start col (0–{GRID_SIZE-1}): "))
                gr   = int(input(f"  Goal  row (0–{GRID_SIZE-1}): "))
                gc   = int(input(f"  Goal  col (0–{GRID_SIZE-1}): "))
                seed = int(input("  Random seed (e.g. 42): ") or "42")
                run_simulation((sr, sc), (gr, gc), seed=seed, verbose=True)
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
