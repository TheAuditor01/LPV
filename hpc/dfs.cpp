/*
 * Parallel Depth-First Search (DFS) Implementation using OpenMP
 * ============================================================
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
 * g++ -fopenmp -o dfs dfs.cpp
 * ./dfs
 *
 * For macOS:
 * ----------
 * g++ -Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include" -L"$(brew --prefix libomp)/lib" -lomp -o dfs dfs.cpp
 * ./dfs
 *
 * THEORETICAL CONCEPTS:
 *
 * Depth-First Search (DFS):
 * -------------------------
 * - DFS is a graph traversal algorithm that explores as far as possible along each branch before backtracking
 * - It uses a stack (implicitly via recursion in this implementation) to keep track of nodes to visit
 * - Time Complexity: O(V + E) where V is number of vertices and E is number of edges
 * - Space Complexity: O(V) for the visited array and recursion stack
 *
 * OpenMP Parallelization:
 * ----------------------
 * - This implementation parallelizes DFS using OpenMP's #pragma omp parallel for directive
 * - The parallel section is the loop that processes adjacency list of each node
 * - Each thread explores different neighbors of the current node independently
 *
 * Implementation Analysis:
 * ----------------------
 * - The parallelization could lead to race conditions since multiple threads might visit the same nodes
 * - The 'visited' array is accessed without synchronization, which may cause issues
 * - True parallelism is limited due to the recursive nature of DFS - deeper recursive calls are still sequential
 *
 * Performance Considerations:
 * --------------------------
 * - For small graphs, the overhead of thread creation may outweigh performance benefits
 * - Ideal for large graphs with high branching factor where multiple threads can explore different branches
 * - Memory access patterns affect cache efficiency and overall performance
 *
 * Potential Improvements:
 * ---------------------
 * - Adding proper synchronization for shared data structures
 * - Using task-based parallelism instead of loop-based parallelism
 * - Implementing work stealing to balance load among threads
 *
 * SAMPLE INPUT/OUTPUT:
 * ------------------
 *
 * Example 1:
 * Input:
 *   6 6     (6 nodes and 6 edges)
 *   0 1     (edge between nodes 0 and 1)
 *   0 2     (edge between nodes 0 and 2)
 *   1 3     (edge between nodes 1 and 3)
 *   2 4     (edge between nodes 2 and 4)
 *   3 5     (edge between nodes 3 and 5)
 *   4 5     (edge between nodes 4 and 5)
 *   0       (start DFS from node 0)
 *
 * Output:
 *   1 2 3 4 5  (nodes visited from node 0)
 *
 * Example 2:
 * Input:
 *   4 3     (4 nodes and 3 edges)
 *   0 1     (edge between nodes 0 and 1)
 *   1 2     (edge between nodes 1 and 2)
 *   2 3     (edge between nodes 2 and 3)
 *   1       (start DFS from node 1)
 *
 * Output:
 *   0 2 3    (nodes visited from node 1)
 */

#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;
const int MAXN = 1e5;
vector<int> adj[MAXN + 5]; // adjacency list
bool visited[MAXN + 5];    // mark visited nodes
void dfs(int node)
{
    visited[node] = true;
#pragma omp parallel for
    for (int i = 0; i < adj[node].size(); i++)
    {
        int next_node = adj[node][i];
        if (!visited[next_node])
        {
            dfs(next_node);
        }
    }
}
int main()
{
    cout << "Please enter nodes and edges ";
    int n, m; // number of nodes and edges
    cin >> n >> m;
    for (int i = 1; i <= m; i++)
    {
        int u, v; // edge between u and v
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    int start_node; // start node of DFS
    cin >> start_node;
    dfs(start_node);
    // Print visited nodes
    for (int i = 0; i <= n; i++)
    {
        if (visited[i])
        {
            cout << i << " ";
        }
    }
    cout << endl;
    return 0;
}