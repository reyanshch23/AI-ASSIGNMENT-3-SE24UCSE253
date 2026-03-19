"""
Part 1: Dijkstra's Algorithm - Shortest Road Distances Between Indian Cities
============================================================================
Implements Dijkstra's (Uniform-Cost Search) algorithm to find the shortest
road path between any two major Indian cities.

Usage:
    python part1_dijkstra_india.py
"""

import heapq
import sys
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH DATA: Major Indian cities with approximate road distances (in km)
# Sources: Google Maps road distances (approximate)
# ─────────────────────────────────────────────────────────────────────────────

INDIA_ROAD_GRAPH = {
    # Format: "City": [("Neighbor", distance_km), ...]
    "Delhi": [
        ("Jaipur", 281), ("Agra", 233), ("Chandigarh", 275),
        ("Lucknow", 555), ("Amritsar", 455), ("Dehradun", 302),
        ("Haridwar", 310), ("Mathura", 183),
    ],
    "Mumbai": [
        ("Pune", 149), ("Ahmedabad", 524), ("Nashik", 167),
        ("Surat", 284), ("Goa", 597), ("Aurangabad", 335),
        ("Nagpur", 835),
    ],
    "Bangalore": [
        ("Chennai", 346), ("Hyderabad", 575), ("Mysuru", 143),
        ("Mangaluru", 352), ("Coimbatore", 365), ("Pune", 838),
        ("Goa", 561),
    ],
    "Chennai": [
        ("Bangalore", 346), ("Hyderabad", 626), ("Coimbatore", 491),
        ("Madurai", 460), ("Tirupati", 138), ("Vijayawada", 430),
        ("Pondicherry", 170),
    ],
    "Hyderabad": [
        ("Bangalore", 575), ("Chennai", 626), ("Pune", 559),
        ("Nagpur", 502), ("Vijayawada", 275), ("Warangal", 148),
        ("Mumbai", 711),
    ],
    "Kolkata": [
        ("Bhubaneswar", 440), ("Patna", 585), ("Siliguri", 569),
        ("Dhanbad", 260), ("Asansol", 200), ("Guwahati", 1017),
    ],
    "Jaipur": [
        ("Delhi", 281), ("Jodhpur", 335), ("Ajmer", 132),
        ("Agra", 240), ("Udaipur", 393), ("Kota", 247),
        ("Bikaner", 333),
    ],
    "Ahmedabad": [
        ("Mumbai", 524), ("Surat", 264), ("Vadodara", 111),
        ("Rajkot", 216), ("Jodhpur", 460), ("Jaipur", 670),
    ],
    "Pune": [
        ("Mumbai", 149), ("Hyderabad", 559), ("Nashik", 211),
        ("Aurangabad", 234), ("Bangalore", 838), ("Solapur", 257),
        ("Kolhapur", 228),
    ],
    "Lucknow": [
        ("Delhi", 555), ("Agra", 375), ("Kanpur", 83),
        ("Varanasi", 296), ("Allahabad", 200), ("Gorakhpur", 273),
        ("Patna", 571),
    ],
    "Agra": [
        ("Delhi", 233), ("Jaipur", 240), ("Lucknow", 375),
        ("Gwalior", 118), ("Mathura", 58), ("Kanpur", 295),
    ],
    "Chandigarh": [
        ("Delhi", 275), ("Amritsar", 229), ("Shimla", 118),
        ("Dehradun", 178), ("Ludhiana", 95), ("Jammu", 294),
    ],
    "Amritsar": [
        ("Chandigarh", 229), ("Ludhiana", 141), ("Jammu", 206),
        ("Delhi", 455),
    ],
    "Varanasi": [
        ("Lucknow", 296), ("Allahabad", 121), ("Patna", 246),
        ("Gaya", 247), ("Gorakhpur", 213),
    ],
    "Patna": [
        ("Varanasi", 246), ("Lucknow", 571), ("Kolkata", 585),
        ("Gaya", 100), ("Muzaffarpur", 77),
    ],
    "Bhubaneswar": [
        ("Kolkata", 440), ("Cuttack", 27), ("Visakhapatnam", 420),
        ("Puri", 60),
    ],
    "Visakhapatnam": [
        ("Bhubaneswar", 420), ("Vijayawada", 345), ("Chennai", 780),
        ("Hyderabad", 625),
    ],
    "Vijayawada": [
        ("Hyderabad", 275), ("Chennai", 430), ("Visakhapatnam", 345),
        ("Tirupati", 375),
    ],
    "Nagpur": [
        ("Hyderabad", 502), ("Mumbai", 835), ("Pune", 703),
        ("Raipur", 290), ("Jabalpur", 330), ("Aurangabad", 519),
    ],
    "Goa": [
        ("Mumbai", 597), ("Bangalore", 561), ("Hubli", 157),
        ("Mangaluru", 352),
    ],
    "Coimbatore": [
        ("Chennai", 491), ("Bangalore", 365), ("Madurai", 210),
        ("Kochi", 182), ("Trichy", 210),
    ],
    "Kochi": [
        ("Coimbatore", 182), ("Trivandrum", 214), ("Madurai", 318),
        ("Mangaluru", 386),
    ],
    "Trivandrum": [
        ("Kochi", 214), ("Madurai", 243), ("Coimbatore", 353),
    ],
    "Madurai": [
        ("Chennai", 460), ("Coimbatore", 210), ("Trivandrum", 243),
        ("Kochi", 318), ("Trichy", 137),
    ],
    "Trichy": [
        ("Chennai", 330), ("Madurai", 137), ("Coimbatore", 210),
        ("Pondicherry", 195),
    ],
    "Surat": [
        ("Mumbai", 284), ("Ahmedabad", 264), ("Vadodara", 152),
        ("Nashik", 262),
    ],
    "Vadodara": [
        ("Ahmedabad", 111), ("Surat", 152), ("Mumbai", 400),
        ("Udaipur", 261),
    ],
    "Rajkot": [
        ("Ahmedabad", 216), ("Surat", 342), ("Junagadh", 102),
    ],
    "Jodhpur": [
        ("Jaipur", 335), ("Ahmedabad", 460), ("Udaipur", 253),
        ("Bikaner", 251), ("Ajmer", 209),
    ],
    "Udaipur": [
        ("Jodhpur", 253), ("Jaipur", 393), ("Ahmedabad", 263),
        ("Ajmer", 274),
    ],
    "Kota": [
        ("Jaipur", 247), ("Jodhpur", 334), ("Ajmer", 200),
        ("Gwalior", 460),
    ],
    "Ajmer": [
        ("Jaipur", 132), ("Jodhpur", 209), ("Kota", 200),
        ("Udaipur", 274),
    ],
    "Bikaner": [
        ("Jaipur", 333), ("Jodhpur", 251), ("Delhi", 436),
    ],
    "Gwalior": [
        ("Agra", 118), ("Jaipur", 330), ("Bhopal", 414),
        ("Kota", 460),
    ],
    "Bhopal": [
        ("Gwalior", 414), ("Nagpur", 345), ("Indore", 192),
        ("Jabalpur", 297), ("Aurangabad", 500),
    ],
    "Indore": [
        ("Bhopal", 192), ("Ujjain", 55), ("Ahmedabad", 394),
        ("Nashik", 561),
    ],
    "Aurangabad": [
        ("Mumbai", 335), ("Pune", 234), ("Nagpur", 519),
        ("Nashik", 188), ("Bhopal", 500),
    ],
    "Nashik": [
        ("Mumbai", 167), ("Pune", 211), ("Aurangabad", 188),
        ("Surat", 262), ("Indore", 561),
    ],
    "Raipur": [
        ("Nagpur", 290), ("Bhubaneswar", 650), ("Jabalpur", 450),
    ],
    "Jabalpur": [
        ("Nagpur", 330), ("Bhopal", 297), ("Raipur", 450),
    ],
    "Dehradun": [
        ("Delhi", 302), ("Chandigarh", 178), ("Haridwar", 52),
        ("Rishikesh", 43),
    ],
    "Haridwar": [
        ("Delhi", 310), ("Dehradun", 52), ("Rishikesh", 24),
    ],
    "Rishikesh": [
        ("Haridwar", 24), ("Dehradun", 43),
    ],
    "Shimla": [
        ("Chandigarh", 118), ("Delhi", 370),
    ],
    "Jammu": [
        ("Chandigarh", 294), ("Amritsar", 206), ("Srinagar", 258),
    ],
    "Srinagar": [
        ("Jammu", 258),
    ],
    "Ludhiana": [
        ("Chandigarh", 95), ("Amritsar", 141), ("Delhi", 320),
    ],
    "Siliguri": [
        ("Kolkata", 569), ("Guwahati", 468),
    ],
    "Guwahati": [
        ("Kolkata", 1017), ("Siliguri", 468),
    ],
    "Allahabad": [
        ("Lucknow", 200), ("Varanasi", 121), ("Kanpur", 191),
    ],
    "Kanpur": [
        ("Lucknow", 83), ("Agra", 295), ("Allahabad", 191),
    ],
    "Gorakhpur": [
        ("Lucknow", 273), ("Varanasi", 213), ("Patna", 214),
    ],
    "Gaya": [
        ("Patna", 100), ("Varanasi", 247),
    ],
    "Muzaffarpur": [
        ("Patna", 77),
    ],
    "Dhanbad": [
        ("Kolkata", 260), ("Asansol", 56), ("Patna", 330),
    ],
    "Asansol": [
        ("Kolkata", 200), ("Dhanbad", 56),
    ],
    "Cuttack": [
        ("Bhubaneswar", 27),
    ],
    "Puri": [
        ("Bhubaneswar", 60),
    ],
    "Tirupati": [
        ("Chennai", 138), ("Vijayawada", 375),
    ],
    "Pondicherry": [
        ("Chennai", 170), ("Trichy", 195),
    ],
    "Warangal": [
        ("Hyderabad", 148),
    ],
    "Hubli": [
        ("Goa", 157), ("Bangalore", 415), ("Mangaluru", 250),
    ],
    "Mangaluru": [
        ("Bangalore", 352), ("Goa", 352), ("Kochi", 386),
        ("Hubli", 250),
    ],
    "Mysuru": [
        ("Bangalore", 143), ("Coimbatore", 210), ("Mangaluru", 232),
    ],
    "Solapur": [
        ("Pune", 257), ("Hyderabad", 324), ("Aurangabad", 384),
    ],
    "Kolhapur": [
        ("Pune", 228), ("Goa", 232),
    ],
    "Junagadh": [
        ("Rajkot", 102), ("Ahmedabad", 317),
    ],
    "Ujjain": [
        ("Indore", 55), ("Bhopal", 184),
    ],
    "Mathura": [
        ("Delhi", 183), ("Agra", 58),
    ],
}


def build_undirected_graph(graph_data):
    """Build an undirected adjacency list from the one-sided definition above."""
    g = defaultdict(list)
    for city, neighbors in graph_data.items():
        for neighbor, dist in neighbors:
            g[city].append((neighbor, dist))
            g[neighbor].append((city, dist))          # reverse edge
    # deduplicate
    for city in g:
        seen = {}
        for nb, d in g[city]:
            if nb not in seen or seen[nb] > d:
                seen[nb] = d
        g[city] = [(nb, d) for nb, d in seen.items()]
    return g


def dijkstra(graph, start, goal=None):
    """
    Dijkstra's / Uniform-Cost Search.

    Args:
        graph : adjacency dict  {city: [(neighbor, cost), ...]}
        start : source city
        goal  : destination city (if None, explores all reachable cities)

    Returns:
        distances : {city: shortest_distance}
        parents   : {city: previous_city}  (for path reconstruction)
        nodes_expanded : count of nodes popped from the priority queue
    """
    distances = {city: float('inf') for city in graph}
    distances[start] = 0
    parents = {start: None}
    visited = set()
    nodes_expanded = 0

    # Priority queue entries: (cost, city)
    pq = [(0, start)]

    while pq:
        cost, city = heapq.heappop(pq)

        if city in visited:
            continue
        visited.add(city)
        nodes_expanded += 1

        # Early exit if we only need the path to the goal
        if goal and city == goal:
            break

        for neighbor, edge_cost in graph[city]:
            new_cost = cost + edge_cost
            if new_cost < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_cost
                parents[neighbor] = city
                heapq.heappush(pq, (new_cost, neighbor))

    return distances, parents, nodes_expanded


def reconstruct_path(parents, start, goal):
    """Trace back from goal to start using the parent map."""
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents.get(current)
    path.reverse()
    if path[0] != start:
        return []          # no path found
    return path


def print_banner():
    print("=" * 65)
    print("   DIJKSTRA'S ALGORITHM — INDIAN CITIES ROAD NETWORK")
    print("=" * 65)


def print_all_from_source(graph, source):
    """Print shortest distances from source to every reachable city."""
    distances, parents, expanded = dijkstra(graph, source)

    reachable = sorted(
        [(d, c) for c, d in distances.items() if d < float('inf') and c != source]
    )

    print(f"\n  Source: {source}  |  Nodes expanded: {expanded}")
    print(f"  {'Destination':<22} {'Distance (km)':>14}  Path")
    print("  " + "-" * 61)
    for dist, city in reachable:
        path = reconstruct_path(parents, source, city)
        path_str = " → ".join(path)
        if len(path_str) > 45:
            path_str = path_str[:42] + "..."
        print(f"  {city:<22} {dist:>14,.0f}  {path_str}")


def find_path(graph, source, destination):
    """Find and display the shortest path between two cities."""
    if source not in graph:
        print(f"  ✗ '{source}' not found in the graph.")
        return
    if destination not in graph:
        print(f"  ✗ '{destination}' not found in the graph.")
        return

    distances, parents, expanded = dijkstra(graph, source, goal=destination)

    if distances[destination] == float('inf'):
        print(f"  No road path found between {source} and {destination}.")
        return

    path = reconstruct_path(parents, source, destination)

    print(f"\n  {'Source':<15}: {source}")
    print(f"  {'Destination':<15}: {destination}")
    print(f"  {'Distance':<15}: {distances[destination]:,} km")
    print(f"  {'Nodes Expanded':<15}: {expanded}")
    print(f"  {'Hops':<15}: {len(path) - 1}")
    print(f"\n  Path: {' → '.join(path)}")

    # Print leg-by-leg breakdown
    print(f"\n  {'Step':<6} {'From':<22} {'To':<22} {'Leg (km)':>10} {'Total (km)':>11}")
    print("  " + "-" * 74)
    cumulative = 0
    for i in range(len(path) - 1):
        frm, to = path[i], path[i + 1]
        leg = next(d for nb, d in graph[frm] if nb == to)
        cumulative += leg
        print(f"  {i+1:<6} {frm:<22} {to:<22} {leg:>10,} {cumulative:>11,}")


def interactive_menu(graph):
    cities = sorted(graph.keys())
    print("\n  Available cities:")
    for i, c in enumerate(cities, 1):
        print(f"    {i:>3}. {c}")

    while True:
        print("\n" + "─" * 65)
        print("  OPTIONS")
        print("  1. Find shortest path between two cities")
        print("  2. Show all distances from a source city")
        print("  3. List all cities")
        print("  4. Exit")
        choice = input("\n  Enter choice (1-4): ").strip()

        if choice == "1":
            src = input("  Enter source city: ").strip().title()
            dst = input("  Enter destination city: ").strip().title()
            find_path(graph, src, dst)

        elif choice == "2":
            src = input("  Enter source city: ").strip().title()
            if src not in graph:
                print(f"  ✗ '{src}' not found.")
            else:
                print_all_from_source(graph, src)

        elif choice == "3":
            print("\n  Cities in graph:")
            for i, c in enumerate(cities, 1):
                print(f"    {i:>3}. {c}")

        elif choice == "4":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid choice. Try again.")


def demo_run(graph):
    """Run a few fixed demos without user interaction."""
    demos = [
        ("Delhi", "Mumbai"),
        ("Chennai", "Amritsar"),
        ("Kolkata", "Kochi"),
        ("Jaipur", "Bangalore"),
    ]
    for src, dst in demos:
        print("\n" + "─" * 65)
        find_path(graph, src, dst)


if __name__ == "__main__":
    print_banner()
    graph = build_undirected_graph(INDIA_ROAD_GRAPH)
    total_cities = len(graph)
    total_edges  = sum(len(v) for v in graph.values()) // 2
    print(f"\n  Graph loaded: {total_cities} cities, {total_edges} road connections")

    # If run non-interactively (e.g. in a CI / GitHub viewer), show demos
    if not sys.stdin.isatty():
        demo_run(graph)
    else:
        # Check for command-line arguments
        if len(sys.argv) == 3:
            src, dst = sys.argv[1].title(), sys.argv[2].title()
            find_path(graph, src, dst)
        elif len(sys.argv) == 2:
            src = sys.argv[1].title()
            print_all_from_source(graph, src)
        else:
            # Show a quick demo then open the interactive menu
            print("\n  === Quick Demo ===")
            find_path(graph, "Delhi", "Chennai")
            interactive_menu(graph)
