#include <omp.h>
#include <iostream>
#include <queue>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

class Node
{
public:
    int value;
    Node *left;
    Node *right;
    
    Node(int value)
    {
        this->value = value;
        this->left = NULL;
        this->right = NULL;
    }
};

// Generates a complete binary tree with specified number of nodes
Node *generateTree(int numNodes)
{
    if (numNodes <= 0)
        return NULL;
    
    // Create nodes with sequential values
    vector<Node*> nodes(numNodes);
    for (int i = 0; i < numNodes; i++)
    {
        nodes[i] = new Node(i + 1);
    }
    
    // Connect nodes in a complete binary tree structure
    for (int i = 0; i < numNodes; i++)
    {
        int leftIndex = 2 * i + 1;
        int rightIndex = 2 * i + 2;
        
        if (leftIndex < numNodes)
            nodes[i]->left = nodes[leftIndex];
        
        if (rightIndex < numNodes)
            nodes[i]->right = nodes[rightIndex];
    }
    
    return nodes[0]; // Return the root node
}

// Sequential BFS
void bfs_sequential(Node *root)
{
    if (root == NULL)
        return;
    
    queue<Node *> q;
    q.push(root);
    
    while (!q.empty())
    {
        Node *node = q.front();
        q.pop();
        
        // Process node (just print for demo)
        //cout << node->value << " ";
        
        if (node->left != NULL)
            q.push(node->left);
        if (node->right != NULL)
            q.push(node->right);
    }
}

// Parallel BFS using OpenMP
void bfs_parallel(Node *root)
{
    if (root == NULL)
        return;
    
    queue<Node *> q;
    q.push(root);
    vector<Node *> currentLevel;
    
    while (!q.empty())
    {
        // Get all nodes at current level
        currentLevel.clear();
        int levelSize = q.size();
        
        for (int i = 0; i < levelSize; i++)
        {
            currentLevel.push_back(q.front());
            q.pop();
        }
        
        // Process current level nodes in parallel
        #pragma omp parallel for
        for (int i = 0; i < currentLevel.size(); i++)
        {
            Node *node = currentLevel[i];
            
            // Process node (critical section for printing)
            #pragma omp critical
            {
                //cout << node->value << " ";
            }
            
            // Add children to queue (needs to be synchronized)
            #pragma omp critical
            {
                if (node->left != NULL)
                    q.push(node->left);
                if (node->right != NULL)
                    q.push(node->right);
            }
        }
    }
}

// Sequential DFS (recursive pre-order traversal)
void dfs_sequential(Node *root)
{
    if (root == NULL)
        return;
    
    // Process node
    //cout << root->value << " ";
    
    // Visit children
    dfs_sequential(root->left);
    dfs_sequential(root->right);
}

// Parallel DFS using OpenMP
void dfs_parallel(Node *root)
{
    if (root == NULL)
        return;
    
    // Process current node (synchronized with critical section)
    #pragma omp critical
    {
        //cout << root->value << " ";
    }
    
    // Process children in parallel
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            dfs_parallel(root->left);
        }
        
        #pragma omp section
        {
            dfs_parallel(root->right);
        }
    }
}

// Clean up tree memory
void cleanupTree(Node *root)
{
    if (root == NULL)
        return;
    
    cleanupTree(root->left);
    cleanupTree(root->right);
    delete root;
}

int main(int argc, char *argv[])
{
    // Default to 15 nodes if not specified
    int numNodes = 15;
    
    // Check if size was provided as command-line argument
    if (argc > 1) {
        numNodes = atoi(argv[1]);
    }
    
    cout << "Generating tree with " << numNodes << " nodes..." << endl;
    Node *root = generateTree(numNodes);
    
    // For large trees, don't print traversal results
    bool printResults = (numNodes <= 20);
    
    // Measure execution time for Sequential BFS
    auto start_bfs_seq = high_resolution_clock::now();
    if (printResults) cout << "Sequential BFS: ";
    bfs_sequential(root);
    auto stop_bfs_seq = high_resolution_clock::now();
    auto duration_bfs_seq = duration_cast<microseconds>(stop_bfs_seq - start_bfs_seq);
    
    if (printResults) cout << endl;
    cout << "Execution time for Sequential BFS: " << duration_bfs_seq.count() << " microseconds" << endl;
    
    // Measure execution time for Parallel BFS
    auto start_bfs_par = high_resolution_clock::now();
    if (printResults) cout << "Parallel BFS: ";
    bfs_parallel(root);
    auto stop_bfs_par = high_resolution_clock::now();
    auto duration_bfs_par = duration_cast<microseconds>(stop_bfs_par - start_bfs_par);
    
    if (printResults) cout << endl;
    cout << "Execution time for Parallel BFS: " << duration_bfs_par.count() << " microseconds" << endl;
    
    // Measure execution time for Sequential DFS
    auto start_dfs_seq = high_resolution_clock::now();
    if (printResults) cout << "Sequential DFS: ";
    dfs_sequential(root);
    auto stop_dfs_seq = high_resolution_clock::now();
    auto duration_dfs_seq = duration_cast<microseconds>(stop_dfs_seq - start_dfs_seq);
    
    if (printResults) cout << endl;
    cout << "Execution time for Sequential DFS: " << duration_dfs_seq.count() << " microseconds" << endl;
    
    // Measure execution time for Parallel DFS
    auto start_dfs_par = high_resolution_clock::now();
    if (printResults) cout << "Parallel DFS: ";
    dfs_parallel(root);
    auto stop_dfs_par = high_resolution_clock::now();
    auto duration_dfs_par = duration_cast<microseconds>(stop_dfs_par - start_dfs_par);
    
    if (printResults) cout << endl;
    cout << "Execution time for Parallel DFS: " << duration_dfs_par.count() << " microseconds" << endl;
    
    // Print speedup comparisons
    cout << "\n--- Performance Comparison ---" << endl;
    cout << "BFS Speedup (Sequential vs Parallel): " 
         << (float)duration_bfs_seq.count() / duration_bfs_par.count() << "x" << endl;
    cout << "DFS Speedup (Sequential vs Parallel): " 
         << (float)duration_dfs_seq.count() / duration_dfs_par.count() << "x" << endl;
    
    // Clean up memory
    cleanupTree(root);
    
    return 0;
}