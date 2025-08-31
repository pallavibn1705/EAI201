import math
import heapq
from collections import deque

class PipeNetwork:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)] 
        self.coordinates = [(0,0)] * n

    def add_pipe(self, u, v, cost):
        self.graph[u].append((v, cost))
        self.graph[v].append((u, cost))

    def set_coordinates(self, junction, x, y):
        self.coordinates[junction] = (x, y)

    def straight_line_distance(self, a, b):
        x1, y1 = self.coordinates[a]
        x2, y2 = self.coordinates[b]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # DFS
    def dfs(self, start, target):
        visited = [False] * self.n
        path = []
        count_visited = 0
        found = False

        def dfs_visit(u):
            nonlocal found, count_visited
            if found:
                return
            visited[u] = True
            count_visited += 1
            path.append(u)
            if u == target:
                found = True
                return
            for v, _ in self.graph[u]:
                if not visited[v]:
                    dfs_visit(v)
                    if found:
                        return
            if not found:
                path.pop()

        dfs_visit(start)
        return path, count_visited

    # BFS 
    def bfs(self, start, target):
        visited = [False] * self.n
        parent = [-1] * self.n
        queue = deque([start])
        visited[start] = True
        count_visited = 1

        while queue:
            u = queue.popleft()
            if u == target:
                break
            for v, _ in self.graph[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    count_visited += 1

        if not visited[target]:
            return [], count_visited

        path = []
        node = target
        while node != -1:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path, count_visited

    # UCS
    def ucs(self, start, target):
        heap = [(0, start, [-1])]  # (cost, node, path_tracker)
        visited = set()
        count_visited = 0

        while heap:
            cost, u, path = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            count_visited += 1

            new_path = path + [u]
            if u == target:
                return new_path[1:], cost, count_visited  # remove dummy -1

            for v, edge_cost in self.graph[u]:
                if v not in visited:
                    heapq.heappush(heap, (cost + edge_cost, v, new_path))

        return [], math.inf, count_visited

    # A* algorithm 
    def a_star(self, start, target):
        dist = [math.inf] * self.n
        parent = [-1] * self.n
        dist[start] = 0
        heap = [(self.straight_line_distance(start, target), 0, start)]
        count_visited = 0
        visited = [False] * self.n

        while heap:
            est_total, cur_dist, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            count_visited += 1
            if u == target:
                break
            for v, cost in self.graph[u]:
                new_dist = cur_dist + cost
                if dist[v] > new_dist:
                    dist[v] = new_dist
                    parent[v] = u
                    est = new_dist + self.straight_line_distance(v, target)
                    heapq.heappush(heap, (est, new_dist, v))

        if dist[target] == math.inf:
            return [], math.inf, count_visited

        path = []
        node = target
        while node != -1:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path, dist[target], count_visited


def main():
    n, m = map(int, input("Number of junctions and pipes: ").split())
    network = PipeNetwork(n)

    print("Enter pipes (junction1 junction2 cost):")
    for _ in range(m):
        u, v, cost = map(int, input().split())
        network.add_pipe(u, v, cost)

    print("Enter coordinates of each junction (x y):")
    for i in range(n):
        x, y = map(float, input().split())
        network.set_coordinates(i, x, y)

    start, target = map(int, input("Starting and target junction: ").split())

    print("\nDFS path and cost :")
    path_dfs, visited_dfs = network.dfs(start, target)
    cost_dfs = sum(network.graph[path_dfs[i]][[v for v,c in network.graph[path_dfs[i]]].index(path_dfs[i+1])][1] for i in range(len(path_dfs)-1)) if path_dfs else math.inf
    print("Path:", path_dfs)
    print("Total cost:", cost_dfs)
    print("Junctions visited:", visited_dfs)

    print("\nBFS path and cost :")
    path_bfs, visited_bfs = network.bfs(start, target)
    cost_bfs = sum(network.graph[path_bfs[i]][[v for v,c in network.graph[path_bfs[i]]].index(path_bfs[i+1])][1] for i in range(len(path_bfs)-1)) if path_bfs else math.inf
    print("Path:", path_bfs)
    print("Total cost:", cost_bfs)
    print("Junctions visited:", visited_bfs)

    print("\nUniform Cost Search (UCS) :")
    path_ucs, cost_ucs, visited_ucs = network.ucs(start, target)
    print("Path:", path_ucs)
    print("Total cost:", cost_ucs)
    print("Junctions visited:", visited_ucs)

    print("\nA* algorithm :")
    path_astar, cost_astar, visited_astar = network.a_star(start, target)
    print("Path:", path_astar)
    print("Total cost:", cost_astar)
    print("Junctions visited:", visited_astar)


if __name__ == "__main__":
    main()

