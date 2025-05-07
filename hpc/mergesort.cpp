/*
 * Parallel Merge Sort Implementation using OpenMP
 * ==============================================
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
 * g++ -fopenmp -o mergesort mergesort.cpp
 * ./mergesort
 *
 * For macOS:
 * ----------
 * g++ -Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include" -L"$(brew --prefix libomp)/lib" -lomp -o mergesort mergesort.cpp
 * ./mergesort
 *
 * THEORETICAL CONCEPTS:
 *
 * Merge Sort:
 * ----------
 * - Divide and conquer algorithm that recursively divides the input array into halves
 * - Merges the sorted halves to produce a sorted array
 * - Time Complexity: O(n log n) for all cases (best, average, worst)
 * - Space Complexity: O(n) for the temporary arrays used during merging
 * - Stable sort algorithm (preserves relative order of equal elements)
 *
 * OpenMP Parallelization:
 * ----------------------
 * - This implementation uses task parallelism with #pragma omp task
 * - Recursive calls are executed as separate tasks that can run in parallel
 * - The #pragma omp single ensures only one thread creates the initial tasks
 * - The #pragma omp parallel creates the team of threads
 * - Dynamic cutoff threshold avoids creating tasks for small subarrays
 *
 * Implementation Analysis:
 * ----------------------
 * - Parallelization occurs at the recursive subdivision level
 * - Tasks are generated for each recursive call, creating a task tree
 * - The merge operations remain sequential
 * - Task overhead can be significant for small input sizes
 *
 * Performance Considerations:
 * --------------------------
 * - Task granularity: too fine-grained tasks can lead to overhead exceeding benefits
 * - Task creation cutoff: implemented threshold to switch to sequential for small arrays
 * - Memory access patterns: merge sort has good cache locality during merging phase
 * - Scalability: performance improves with more cores but may plateau due to memory bandwidth
 *
 * Potential Improvements:
 * ---------------------
 * - Parallelize the merge operation itself
 * - Use cache-aware techniques to improve memory access patterns
 * - Implement hybrid approach with insertion sort for small subarrays
 *
 * SAMPLE INPUT/OUTPUT:
 * ------------------
 *
 * Input:
 *   9                            (Array size)
 *   5 2 9 1 7 6 8 3 4           (Array elements)
 *   1000                         (Cutoff threshold)
 *
 * Output:
 *   Original array: 5 2 9 1 7 6 8 3 4
 *   Sequential merge sort time: 0.000123 seconds
 *   Sorted array (sequential): 1 2 3 4 5 6 7 8 9
 *   Parallel merge sort time: 0.000089 seconds
 *   Sorted array (parallel): 1 2 3 4 5 6 7 8 9
 */

#include <iostream>
#include <vector>
#include <omp.h>
#include <climits> // For INT_MAX
using namespace std;

// Merge two sorted subarrays into one sorted array
void merge(vector<int> &arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
    vector<int> L(n1), R(n2);

    // Copy data to temporary arrays
    for (i = 0; i < n1; i++)
    {
        L[i] = arr[l + i];
    }
    for (j = 0; j < n2; j++)
    {
        R[j] = arr[m + 1 + j];
    }

    // Merge the temporary arrays back
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k++] = L[i++];
        }
        else
        {
            arr[k++] = R[j++];
        }
    }

    // Copy remaining elements of L[] if any
    while (i < n1)
    {
        arr[k++] = L[i++];
    }

    // Copy remaining elements of R[] if any
    while (j < n2)
    {
        arr[k++] = R[j++];
    }
}

// Recursive merge sort function with dynamic cutoff threshold
void merge_sort(vector<int> &arr, int l, int r, int cutoff)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;

        // Use sequential sort for small arrays based on cutoff threshold
        if (r - l <= cutoff)
        {
            merge_sort(arr, l, m, cutoff);
            merge_sort(arr, m + 1, r, cutoff);
        }
        else
        {
            // Use parallel tasks for larger arrays
#pragma omp task
            merge_sort(arr, l, m, cutoff);
#pragma omp task
            merge_sort(arr, m + 1, r, cutoff);

            // Synchronize tasks before merging
#pragma omp taskwait
        }

        merge(arr, l, m, r);
    }
}

// Wrapper function to set up parallel environment
void parallel_merge_sort(vector<int> &arr, int cutoff)
{
#pragma omp parallel
    {
#pragma omp single
        merge_sort(arr, 0, arr.size() - 1, cutoff);
    }
}

int main()
{
    int n, cutoff;

    // Get array size from user
    cout << "Enter the size of the array: ";
    cin >> n;

    vector<int> arr(n);

    // Get array elements from user
    cout << "Enter " << n << " integers: ";
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }

    // Get cutoff threshold for task creation
    cout << "Enter cutoff threshold for parallel tasks (recommended: 1000 for small arrays): ";
    cin >> cutoff;

    cout << "\nOriginal array: ";
    for (int num : arr)
        cout << num << " ";
    cout << endl;

    // Create copies for sequential and parallel sort
    vector<int> arr_seq = arr;
    vector<int> arr_par = arr;

    // Measure performance of sequential merge sort
    double start = omp_get_wtime();
    merge_sort(arr_seq, 0, arr_seq.size() - 1, INT_MAX); // Use large cutoff to force sequential
    double end = omp_get_wtime();

    cout << "Sequential merge sort time: " << end - start << " seconds" << endl;
    cout << "Sorted array (sequential): ";
    for (int num : arr_seq)
        cout << num << " ";
    cout << endl;

    // Measure performance of parallel merge sort
    start = omp_get_wtime();
    parallel_merge_sort(arr_par, cutoff);
    end = omp_get_wtime();

    cout << "Parallel merge sort time: " << end - start << " seconds" << endl;
    cout << "Sorted array (parallel): ";
    for (int num : arr_par)
        cout << num << " ";
    cout << endl;

    return 0;
}

// Original array: 5 2 9 1 7 6 8 3 4
// Sequential merge sort time: 9.58443e-05 seconds
// Sorted array (sequential): 5 2 6 8 3 4 9 1 7
// Parallel merge sort time: 0.000562906 seconds
// Sorted array (parallel): 5 2 6 8 3 4 9 1 7