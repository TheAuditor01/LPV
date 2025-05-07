/*
Compilation Instructions:

For macOS:
1. Install OpenMP: brew install libomp
2. Compile: clang++ -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp parallel_reduction.cpp -o parallel_reduction

For Ubuntu:
1. Install OpenMP: sudo apt-get update && sudo apt-get install libomp-dev
2. Compile: g++ -fopenmp parallel_reduction.cpp -o parallel_reduction

Run: ./parallel_reduction
*/

// Min, Max, Sum and Avg using parallel reduction

#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>
using namespace std;
void min_reduction(vector<int> &arr)
{
  int min_value = INT_MAX;
#pragma omp parallel for reduction(min : min_value)
  for (int i = 0; i < arr.size(); i++)
  {
    if (arr[i] < min_value)
    {
      min_value = arr[i];
    }
  }
  cout << "Minimum value: " << min_value << endl;
}
void max_reduction(vector<int> &arr)
{
  int max_value = INT_MIN;
#pragma omp parallel for reduction(max : max_value)
  for (int i = 0; i < arr.size(); i++)
  {
    if (arr[i] > max_value)
    {
      max_value = arr[i];
    }
  }
  cout << "Maximum value: " << max_value << endl;
}
void sum_reduction(vector<int> &arr)
{
  int sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < arr.size(); i++)
  {
    sum += arr[i];
  }
  cout << "Sum: " << sum << endl;
}
void average_reduction(vector<int> &arr)
{
  int sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < arr.size(); i++)
  {
    sum += arr[i];
  }
  cout << "Average: " << (double)sum / arr.size() << endl;
}
int main()
{
  vector<int> arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};
  min_reduction(arr);
  max_reduction(arr);
  sum_reduction(arr);
  average_reduction(arr);
}

/*
Theoretical Concepts:

1. Parallel Reduction Fundamentals:
   - Reduction is a fundamental parallel programming pattern
   - Combines multiple values into a single result using an associative operator
   - Follows the principle of divide-and-conquer
   - Time complexity: O(log n) with n processors, O(n/p + log p) with p processors

2. OpenMP Architecture:
   - Fork-Join Model:
     * Master thread creates team of worker threads (fork)
     * Threads execute parallel region
     * Threads synchronize and merge results (join)
   - Thread Management:
     * Dynamic thread creation and destruction
     * Thread pool for better performance
     * Automatic load balancing

3. Memory Model:
   - Shared Memory Architecture:
     * All threads share the same address space
     * Direct access to shared variables
     * Need for synchronization mechanisms
   - Memory Consistency:
     * Flush operations for memory synchronization
     * Atomic operations for thread safety
     * Memory barriers for ordering

4. Parallel Reduction Algorithm:
   - Step 1: Data Partitioning
     * Divide input data among threads
     * Each thread processes its portion independently
   - Step 2: Local Computation
     * Each thread performs reduction on its portion
     * Creates partial results
   - Step 3: Global Reduction
     * Combine partial results using reduction operator
     * Tree-based combination for efficiency
     * Final result in shared variable

5. Performance Considerations:
   - Scalability:
     * Strong scaling: fixed problem size, increasing processors
     * Weak scaling: problem size grows with processors
   - Overhead Factors:
     * Thread creation and management
     * Synchronization costs
     * Memory access patterns
   - Optimization Techniques:
     * Cache utilization
     * False sharing avoidance
     * Load balancing
     * Memory alignment

6. Mathematical Properties:
   - Associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
   - Commutativity: a ⊕ b = b ⊕ a
   - Identity Element: a ⊕ e = a
   - Required for parallel reduction to work correctly

7. Common Reduction Operations:
   - Arithmetic: sum, product, average
   - Logical: AND, OR, XOR
   - Statistical: min, max, variance
   - Custom: user-defined reduction operators

8. Error Handling:
   - Race Conditions:
     * Multiple threads accessing shared data
     * Need for proper synchronization
   - Deadlocks:
     * Circular dependencies in synchronization
     * Proper lock ordering
   - Load Imbalance:
     * Uneven work distribution
     * Dynamic scheduling solutions
*/
