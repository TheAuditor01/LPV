#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class Graph {
    int V; // Number of vertices
    vector<vector<int>> adj; // Adjacency list

public:
    Graph(int V) : V(V), adj(V) {}

    // Add an edge to the undirected graph
    void addEdge(int v, int w) {
        adj[v].push_back(w);
        adj[w].push_back(v);
    }

    // Print the adjacency list
    void printGraph() {
        cout << "\nGraph Adjacency List:\n";
        for (int i = 0; i < V; i++) {
            cout << "Vertex " << i << ": ";
            for (int neighbor : adj[i]) {
                cout << neighbor << " ";
            }
            cout << endl;
        }
    }

    // Parallel Breadth-First Search
    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        double start_time = omp_get_wtime();
        
        visited[start] = true;
        q.push(start);

        cout << "\nParallel BFS starting from vertex " << start << ":\n";

        while (!q.empty()) {
            #pragma omp parallel
            {
                #pragma omp single
                {
                    int level_size = q.size();
                    for (int i = 0; i < level_size; i++) {
                        int v = q.front();
                        q.pop();
                        cout << v << " ";

                        // Process neighbors in parallel
                        #pragma omp task firstprivate(v) shared(visited, q)
                        {
                            for (int neighbor : adj[v]) {
                                #pragma omp critical
                                {
                                    if (!visited[neighbor]) {
                                        visited[neighbor] = true;
                                        q.push(neighbor);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        double end_time = omp_get_wtime();
        cout << "\n\nParallel BFS completed in " << (end_time - start_time) * 1000 << " milliseconds\n";
    }

    // Parallel Depth-First Search (using iterative approach)
    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        stack<int> s;

        double start_time = omp_get_wtime();
        
        s.push(start);
        visited[start] = true;

        cout << "\nParallel DFS starting from vertex " << start << ":\n";

        while (!s.empty()) {
            int v;
            #pragma omp critical
            {
                v = s.top();
                s.pop();
            }
            
            cout << v << " ";

            // Process neighbors in parallel
            #pragma omp parallel for
            for (int i = 0; i < adj[v].size(); i++) {
                int neighbor = adj[v][i];
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        s.push(neighbor);
                    }
                }
            }
        }
        
        double end_time = omp_get_wtime();
        cout << "\n\nParallel DFS completed in " << (end_time - start_time) * 1000 << " milliseconds\n";
    }
};

int main() {
    int V, E, start_vertex;
    int num_threads;
    
    cout << "Enter number of vertices: ";
    cin >> V;
    cout << "Enter number of edges: ";
    cin >> E;
    
    Graph g(V);
    
    cout << "Enter edges (vertex pairs, 0-based indexing):\n";
    for (int i = 0; i < E; i++) {
        int v, w;
        cin >> v >> w;
        if (v >= V || w >= V || v < 0 || w < 0) {
            cout << "Invalid vertex! Vertices must be between 0 and " << V-1 << endl;
            i--; // Retry this edge
            continue;
        }
        g.addEdge(v, w);
    }
    
    cout << "Enter starting vertex for BFS/DFS (0-" << V-1 << "): ";
    cin >> start_vertex;
    if (start_vertex < 0 || start_vertex >= V) {
        cout << "Invalid starting vertex! Using 0 as default.\n";
        start_vertex = 0;
    }
    
    cout << "Enter number of threads to use: ";
    cin >> num_threads;
    omp_set_num_threads(num_threads);
    
    g.printGraph();
    
    // Run parallel BFS
    g.parallelBFS(start_vertex);
    
    // Run parallel DFS
    g.parallelDFS(start_vertex);
    
    return 0;
}



# include <iostream>
# include <vector>
# include <stack>
# include <queue>
# include <omp.h>

using namespace std;

class Graph
{
    private:
    int NumVertices;
    vector<vector<int>> adj;

    public:
    Graph(int vertices)
    {
        NumVertices=vertices;
        adj.resize(vertices);
    }

    void addEdge(int src, int dest)
    {
        adj[src].push_back(dest);
        adj[dest].push_back(src);
    }

    void viewGraph()
    {
        cout<<"Graph : "<<endl;
        for (int i=0;i<NumVertices;i++)
        {
            cout<<"Vertex : "<<i<<" ";
            for (int neighbor :adj[i])
            {
                cout<<neighbor<<" ";
            }
            cout<<endl;
        }
    }

    void bfs(int StartVertex)
    {
        vector<bool> visited (NumVertices,false);
        queue<int> q;

        visited[StartVertex]=true;
        q.push(StartVertex);

        cout<<"Parallel BFS Output : "<<endl;

        while (!q.empty())
        {
            int current_vertex=q.front();
            q.pop();

            #pragma omp critical
            {
                cout<<current_vertex<<endl;
            }

            vector<int> neighbors=adj[current_vertex];

            #pragma omp parallel for
            for(int i=0;i<neighbors.size();i++)
            {
                int neighbor=neighbors[i];
                int thread_id=omp_get_thread_num();

                bool shouldVisit=false;

                #pragma omp critical
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor]=true;
                        q.push(neighbor);
                        shouldVisit=true;
                    }
                }

                #pragma omp critical
                {
                    cout<<"Thread "<<thread_id<<" visiting Vertex : "<<neighbor<<endl;
                }

            }
        }
    }

    void dfs(int StartVertex)
    {
        vector<bool> visited(NumVertices,false);
        stack<int> s;

        visited[StartVertex]=true;
        s.push(StartVertex);

        cout<<"Parallel DFS Output : "<<endl;

        while(!s.empty())
        {
            int current_vertex=s.top();
            s.pop();

            #pragma omp critical
            {
                cout<<"Vertex : "<<current_vertex<<endl;
            }

            vector<int> neighbors=adj[current_vertex];

            #pragma omp parallel for
            for (int i=0;i<neighbors.size();i++)
            {
                int neighbor=neighbors[i];
                int thread_id=omp_get_thread_num();

                bool shouldVisit=false;

                #pragma omp critical
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor]=true;
                        s.push(neighbor);

                        shouldVisit=true;
                    }
                }

                #pragma omp critical
                {
                    cout<<"Thread "<<thread_id<<" is visiting vertex "<<neighbor<<endl;
                }
            }
        }

    }
};

int main()
{
    int NumVertices,NumEdges;

    cout<<"Enter the no. of vertices of the graph : ";
    cin>>NumVertices;

    Graph graph(NumVertices);

    cout<<"Enter the no. of edges : ";
    cin>>NumEdges;

    int src,dest;
    cout<<"Enter the edge as source dest"<<endl;
    for (int i=0;i<NumEdges;i++)
    {
        cin>>src>>dest;
        graph.addEdge(src,dest);
    }
    cout<<endl;

    graph.viewGraph();

    cout<<"Enter the starting vertex : ";
    int startvertex;
    cin>>startvertex;

    double start,end;

    start=omp_get_wtime();
    graph.bfs(startvertex);
    end=omp_get_wtime();

    cout<<"Time taken for parallel BFS is : "<<end-start<<" seconds"<<endl;

    start=omp_get_wtime();
    graph.dfs(startvertex);
    end=omp_get_wtime();

    cout<<"Time taken for parallel DFS is : "<<end-start<<" seconds"<<endl;
}