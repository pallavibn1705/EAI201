import heapq
import math
import time


DIRS_4 = [(0,1),(1,0),(-1,0),(0,-1)]
DIRS_8 = DIRS_4 + [(1,1),(1,-1),(-1,1),(-1,-1)]

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def diagonal(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

def greedy_best_first(start, goal, grid, heuristic=manhattan, dirs=DIRS_4):
    pq = [(heuristic(start, goal), start)]
    visited = set()
    parent = {}
    explored = 0

    while pq:
        _, current = heapq.heappop(pq)
        explored += 1
        if current == goal:
            break
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in dirs:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != 1:
                if (nx, ny) not in visited:
                    parent[(nx, ny)] = current
                    heapq.heappush(pq, (heuristic((nx,ny), goal), (nx,ny)))
    return reconstruct_path(parent, start, goal), explored

def astar(start, goal, grid, heuristic=manhattan, dirs=DIRS_4):
    pq = [(heuristic(start, goal), 0, start)]
    g = {start: 0}
    parent = {}
    explored = 0

    while pq:
        f, cost, current = heapq.heappop(pq)
        explored += 1
        if current == goal:
            break

        for dx, dy in dirs:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != 1:
                new_cost = g[current] + (grid[nx][ny] if grid[nx][ny] > 0 else 1)
                if (nx, ny) not in g or new_cost < g[(nx, ny)]:
                    g[(nx, ny)] = new_cost
                    f = new_cost + heuristic((nx,ny), goal)
                    parent[(nx, ny)] = current
                    heapq.heappush(pq, (f, new_cost, (nx,ny)))
    return reconstruct_path(parent, start, goal), explored

def reconstruct_path(parent, start, goal):
    if goal not in parent:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


grid = [
    ['S',0,0,1,0],
    [1,1,0,1,'G'],
    [0,0,0,1,0],
    [1,1,0,1,1],
    [0,0,0,0,0]
]


start = (0,0)
goal = (1,4)
grid_numeric = [[0 if cell==0 or cell in ['S','G'] else 1 for cell in row] for row in grid]


for algo in [greedy_best_first, astar]:
    for h in [manhattan, euclidean, diagonal]:
        t0 = time.time()
        path, explored = algo(start, goal, grid_numeric, heuristic=h, dirs=DIRS_4)
        t1 = time.time()
        print(f"{algo.__name__} with {h.__name__}:")
        print("Path:", path)
        print("Path length:", len(path))
        print("Nodes explored:", explored)
        print("Time: %.6f s" % (t1-t0))
        print()
