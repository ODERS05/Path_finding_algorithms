import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
from queue import Queue, LifoQueue
from heapq import heappop, heappush

class Maze:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[1] * cols for _ in range(rows)]  # Initialize grid with walls
        self.source = (1, 1)  # Default source position
        self.destination = (rows - 2, cols - 2)  # Default destination position

    def generate_maze(self):
        self._recursive_backtracking(0, 0)  # Start maze generation from cell (0, 0)
        self.grid[self.source[0]][self.source[1]] = 0  # Set source cell as passage
        self.grid[self.destination[0]][self.destination[1]] = 0  # Set destination cell as passage

    def _recursive_backtracking(self, row, col):
        self.grid[row][col] = 0  # Mark current cell as visited

        # Randomly shuffle the directions (up, down, left, right)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        np.random.shuffle(directions)

        for dr, dc in directions:
            new_row, new_col = row + 2 * dr, col + 2 * dc  # Neighbor cell
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.grid[new_row][new_col]:
                self.grid[row + dr][col + dc] = 0  # Remove wall between current and neighbor cell
                self._recursive_backtracking(new_row, new_col)  # Recursively call on neighbor cell

    def visualize_maze(self, paths=None, times=None, titles=None):
        num_algorithms = len(paths)
        fig, axes = plt.subplots(1, num_algorithms, figsize=(self.cols * num_algorithms / 3, self.rows / 3))
        if num_algorithms == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(self.grid, cmap='binary', interpolation='none')
            ax.scatter(*self.source[::-1], color='blue', s=100, label='Source')
            ax.scatter(*self.destination[::-1], color='red', s=100, label='Destination')
            if paths[i]:
                path_row, path_col = zip(*paths[i])
                ax.plot(path_col, path_row, color='green', linewidth=2, label='Path')
            ax.legend()
            ax.set_title(f"{titles[i]}\nTime taken: {times[i]:.7f} seconds")
            ax.set_xticks([]), ax.set_yticks([])  # Hide axis ticks
        plt.show()

class PathfindingAlgorithms:
    def __init__(self, maze):
        self.maze = maze

    def breadth_first_search(self):
        start = self.maze.source
        end = self.maze.destination
        visited = [[False] * self.maze.cols for _ in range(self.maze.rows)]
        parent = [[None] * self.maze.cols for _ in range(self.maze.rows)]

        start_time = time.time()  # Record start time

        queue = Queue()
        queue.put(start)
        visited[start[0]][start[1]] = True

        while not queue.empty():
            current_row, current_col = queue.get()

            if (current_row, current_col) == end:
                # Build path from destination to source
                path = []
                while (current_row, current_col) != start:
                    path.append((current_row, current_col))
                    current_row, current_col = parent[current_row][current_col]
                path.append(start)
                end_time = time.time()  # Record end time
                time_taken = end_time - start_time
                return path[::-1], time_taken  # Reverse the path to get source to destination, return time taken

            # Explore neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_row, new_col = current_row + dr, current_col + dc
                if 0 <= new_row < self.maze.rows and 0 <= new_col < self.maze.cols and self.maze.grid[new_row][new_col] == 0 and not visited[new_row][new_col]:
                    visited[new_row][new_col] = True
                    parent[new_row][new_col] = (current_row, current_col)
                    queue.put((new_row, new_col))

        return [], 0  # No path found, return 0 time

    def depth_first_search(self):
        start = self.maze.source
        end = self.maze.destination
        visited = [[False] * self.maze.cols for _ in range(self.maze.rows)]
        parent = [[None] * self.maze.cols for _ in range(self.maze.rows)]

        start_time = time.time()  # Record start time

        stack = LifoQueue()
        stack.put(start)
        visited[start[0]][start[1]] = True

        while not stack.empty():
            current_row, current_col = stack.get()

            if (current_row, current_col) == end:
                # Build path from destination to source
                path = []
                while (current_row, current_col) != start:
                    path.append((current_row, current_col))
                    current_row, current_col = parent[current_row][current_col]
                path.append(start)
                end_time = time.time()  # Record end time
                time_taken = end_time - start_time
                return path[::-1], time_taken  # Reverse the path to get source to destination, return time taken

            # Explore neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_row, new_col = current_row + dr, current_col + dc
                if 0 <= new_row < self.maze.rows and 0 <= new_col < self.maze.cols and self.maze.grid[new_row][new_col] == 0 and not visited[new_row][new_col]:
                    visited[new_row][new_col] = True
                    parent[new_row][new_col] = (current_row, current_col)
                    stack.put((new_row, new_col))

        return [], 0  # No path found, return 0 time

    def a_star_search(self):
        start = self.maze.source
        end = self.maze.destination
        open_list = []
        heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}

        start_time = time.time()  # Record start time

        while open_list:
            _, current = heappop(open_list)

            if current == end:
                # Build path from destination to source
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                end_time = time.time()  # Record end time
                time_taken = end_time - start_time
                return path[::-1], time_taken  # Reverse the path to get source to destination, return time taken

            # Explore neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if 0 <= neighbor[0] < self.maze.rows and 0 <= neighbor[1] < self.maze.cols and self.maze.grid[neighbor[0]][neighbor[1]] == 0:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                        heappush(open_list, (f_score[neighbor], neighbor))

        return [], 0  # No path found, return 0 time

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

def track_memory_usage():
    """Function to track memory usage."""
    process = psutil.Process()
    return process.memory_info().rss

def main():
    sizes = [11, 21, 31, 41, 51]
    bfs_times = []
    dfs_times = []
    astar_times = []
    bfs_memory = []
    dfs_memory = []
    astar_memory = []

    for size in sizes:
        total_bfs_time = 0
        total_dfs_time = 0
        total_astar_time = 0
        total_bfs_memory = 0
        total_dfs_memory = 0
        total_astar_memory = 0
        num_tests = 5
        for _ in range(num_tests):
            maze = Maze(rows=size, cols=size)
            maze.generate_maze()
            pathfinding = PathfindingAlgorithms(maze)

            # Track memory usage before running each algorithm
            initial_memory = track_memory_usage()

            # Run Breadth-First Search algorithm and measure time
            bfs_path, bfs_time = pathfinding.breadth_first_search()
            total_bfs_time += bfs_time
            total_bfs_memory += track_memory_usage() - initial_memory

            # Track memory usage before running each algorithm
            initial_memory = track_memory_usage()

            # Run Depth-First Search algorithm and measure time
            dfs_path, dfs_time = pathfinding.depth_first_search()
            total_dfs_time += dfs_time
            total_dfs_memory += track_memory_usage() - initial_memory

            # Track memory usage before running each algorithm
            initial_memory = track_memory_usage()

            # Run A* Search algorithm and measure time
            astar_path, astar_time = pathfinding.a_star_search()
            total_astar_time += astar_time
            total_astar_memory += track_memory_usage() - initial_memory

            # Visualize the maze with paths
            maze.visualize_maze(paths=[bfs_path, dfs_path, astar_path],
                                 times=[bfs_time, dfs_time, astar_time],
                                 titles=['Breadth-First Search', 'Depth-First Search', 'A* Search'])

        bfs_times.append(total_bfs_time / num_tests)
        dfs_times.append(total_dfs_time / num_tests)
        astar_times.append(total_astar_time / num_tests)
        bfs_memory.append(total_bfs_memory / num_tests)
        dfs_memory.append(total_dfs_memory / num_tests)
        astar_memory.append(total_astar_memory / num_tests)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, bfs_times, label='BFS')
    plt.plot(sizes, dfs_times, label='DFS')
    plt.plot(sizes, astar_times, label='A*')
    plt.xlabel('Size of Maze')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison of Pathfinding Algorithms')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(sizes, bfs_memory, label='BFS')
    plt.plot(sizes, dfs_memory, label='DFS')
    plt.plot(sizes, astar_memory, label='A*')
    plt.xlabel('Size of Maze')
    plt.ylabel('Memory Usage (KB)')
    plt.title('Memory Usage Comparison of Pathfinding Algorithms')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


