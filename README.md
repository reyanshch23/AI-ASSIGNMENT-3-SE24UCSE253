# AI Search Algorithms — Programming Assignment

Three standalone Python programs covering:

| File | Algorithm | Problem |
|---|---|---|
| `part1_dijkstra_india.py` | Dijkstra / Uniform-Cost Search | Shortest road path between Indian cities |
| `part2_ugv_static.py` | A\* (octile heuristic) | UGV navigation with static known obstacles |
| `part3_ugv_dynamic.py` | D\* Lite | UGV navigation with dynamic unknown obstacles |

---

## Requirements

- Python 3.8+
- No external libraries — only the standard library (`heapq`, `math`, `random`, `collections`)

---

## Part 1 — Dijkstra's Algorithm: Indian Cities

Finds the shortest **road distance** between any two major Indian cities using Dijkstra's algorithm (identical to Uniform-Cost Search in AI terminology).

**Graph:** 70+ cities connected by ~200 road edges (distances in km, sourced from Google Maps).

```bash
# Interactive menu
python part1_dijkstra_india.py

# Direct query
python part1_dijkstra_india.py Delhi Chennai

# All distances from one city
python part1_dijkstra_india.py Mumbai
```

**Sample output:**
```
  Source         : Delhi
  Destination    : Chennai
  Distance       : 2,179 km
  Nodes Expanded : 48
  Hops           : 7

  Path: Delhi → Agra → Gwalior → ... → Chennai
```

---

## Part 2 — UGV Static Obstacles (A\*)

Navigates a **70×70 km battlefield grid** from a user-specified start to goal.  
Obstacles are randomly seeded at three density levels and known **a priori**.

**Algorithm:** A\* with the octile-distance heuristic (optimal for 8-connected grids).

```bash
python part2_ugv_static.py          # interactive
python part2_ugv_static.py demo     # non-interactive demo (3 scenarios)
```

**Density levels:**

| Level | % Blocked |
|-------|-----------|
| low | 10% |
| medium | 25% |
| high | 40% |

**Measures of Effectiveness reported:**
- Path cost (km) vs straight-line distance
- Path efficiency & detour ratio
- Nodes expanded / generated
- Search time (ms)

---

## Part 3 — UGV Dynamic Obstacles (D\* Lite)

Extends Part 2. Obstacles are **not fully known a priori** — new ones are revealed
as the UGV moves (simulating real-time sensor discovery).

**Algorithm:** **D\* Lite** (Koenig & Likhachev, 2002).  
After each obstacle batch is revealed, D\* Lite *repairs* only the affected
portion of the search graph — dramatically faster than re-running A\* from scratch.

```bash
python part3_ugv_dynamic.py         # interactive
python part3_ugv_dynamic.py demo    # non-interactive demo
```

**Simulation loop:**
1. Initial partial map → D\* Lite computes first plan
2. UGV moves along plan
3. Every N steps → new obstacle batch revealed by sensors
4. D\* Lite replans from current position (incremental)
5. Repeat until goal reached or path exhausted

**Measures of Effectiveness reported:**
- Actual vs straight-line path cost
- Number of replans triggered
- Average replan time (ms)
- Total nodes expanded across all replans

---

## Algorithm Comparison

| Property | Dijkstra | A\* | D\* Lite |
|---|---|---|---|
| Heuristic | None | Yes (octile) | Yes (reverse) |
| Optimal | Yes | Yes | Yes |
| Dynamic replanning | No | No (full restart) | **Yes (incremental)** |
| Use case | Known static graph | Known static grid | Unknown/changing grid |

---

## References

- Russell & Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed.
- Koenig & Likhachev (2002). *D\* Lite.* AAAI-02.
- Dijkstra (1959). *A note on two problems in connexion with graphs.*
