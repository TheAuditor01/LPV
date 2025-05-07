/*
 * Parallel Breadth-First Search (BFS) Implementation using OpenMP
 * ==============================================================
 *
 * INSTALLATION INSTRUCTIONS:
 *
 * For Ubuntu:
 * -----------
 * sudo apt-get update
 * sudo apt-get install g++
 * sudo apt-get install libomp-dev
 *
 * For macOS:
 * ----------
 * brew install libomp
 *
 * COMPILATION & EXECUTION:
 *
 * For Ubuntu:
 * -----------
 * g++ -fopenmp -o bfs bfs.cpp
 * ./bfs
 *
 * For macOS:
 * ----------
 * g++ -Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include" -L"$(brew --prefix libomp)/lib" -lomp -o bfs bfs.cpp
 * ./bfs
 *
 * THEORETICAL CONCEPTS:
 *
 * Breadth-First Search (BFS):
 * --------------------------
 * - BFS is a graph traversal algorithm that explores all vertices at the current depth before moving to vertices at the next depth level
 * - It uses a queue data structure to keep track of vertices to be explored
 * - BFS finds the shortest path in unweighted graphs
 * - Time Complexity: O(V + E) where V is number of vertices and E is number of edges
 * - Space Complexity: O(V) for the queue and visited array
 *
 * OpenMP Parallelization:
 * ----------------------
 * - This implementation parallelizes BFS using OpenMP's #pragma omp parallel for directive
 * - The parallel section is the loop that processes neighbors of the current vertex
 * - Multiple threads explore different neighbors concurrently
 * - shared(adj_list, visited, q) indicates these structures are shared among threads
 * - schedule(dynamic) allows dynamic load balancing among threads
 *
 * Implementation Analysis:
 * ----------------------
 * - The parallelization may have race conditions when multiple threads access the queue simultaneously
 * - Without proper synchronization, nodes might be visited multiple times
 * - The queue operations (push/pop) are still sequential, limiting overall parallelism
 *
 * Performance Considerations:
 * --------------------------
 * - BFS is more challenging to parallelize effectively than DFS due to its level-by-level approach
 * - The queue becomes a synchronization bottleneck in parallel implementations
 * - Best performance gains are seen with wide, shallow graphs with high branching factor
 *
 * Potential Improvements:
 * ---------------------
 * - Use thread-safe queue or concurrent data structures
 * - Implement level-synchronous BFS for better parallelism
 * - Use atomic operations or locks to protect shared data structures
 *
 * SAMPLE INPUT/OUTPUT:
 * ------------------
 *
 * Example 1:
 * Input:
 *   6 7 0    (6 vertices, 7 edges, start from vertex 0)
 *   0 1      (edge between vertices 0 and 1)
 *   0 2      (edge between vertices 0 and 2)
 *   1 3      (edge between vertices 1 and 3)
 *   1 4      (edge between vertices 1 and 4)
 *   2 4      (edge between vertices 2 and 4)
 *   3 5      (edge between vertices 3 and 5)
 *   4 5      (edge between vertices 4 and 5)
 *
 * Output:
 *   0 1 2 3 4 5  (BFS traversal order from vertex 0)
 *
 * Example 2:
 * Input:
 *   5 4 2    (5 vertices, 4 edges, start from vertex 2)
 *   0 1      (edge between vertices 0 and 1)
 *   1 2      (edge between vertices 1 and 2)
 *   2 3      (edge between vertices 2 and 3)
 *   3 4      (edge between vertices 3 and 4)
 *
 * Output:
 *   2 1 3 0 4  (BFS traversal order from vertex 2)
 */

#include <iostream>
#include <queue>
#include <vector>
#include <omp.h>
using namespace std;
int main()
{
    int num_vertices, num_edges, source;
    cin >> num_vertices >> num_edges >> source;
    vector<vector<int>> adj_list(num_vertices + 1);
    for (int i = 0; i < num_edges; i++)
    {
        int u, v;
        cin >> u >> v;
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    queue<int> q;
    vector<bool> visited(num_vertices + 1, false);
    q.push(source);
    visited[source] = true;
    while (!q.empty())
    {
        int curr_vertex = q.front();
        q.pop();
        cout << curr_vertex << " ";
#pragma omp parallel for shared(adj_list, visited, q) schedule(dynamic)
        for (int i = 0; i < adj_list[curr_vertex].size(); i++)
        {
            int neighbour = adj_list[curr_vertex][i];
            if (!visited[neighbour])
            {
                visited[neighbour] = true;
                q.push(neighbour);
            }
        }
    }
    return 0;
}