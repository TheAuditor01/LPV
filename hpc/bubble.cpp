/*
 * Parallel Bubble Sort Implementation using OpenMP
 * ===============================================
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
 * g++ -fopenmp -o bubble bubble.cpp
 * ./bubble
 *
 * For macOS:
 * ----------
 * g++ -Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include" -L"$(brew --prefix libomp)/lib" -lomp -o bubble bubble.cpp
 * ./bubble
 *
 * THEORETICAL CONCEPTS:
 *
 * Bubble Sort:
 * -----------
 * - Simple comparison-based sorting algorithm
 * - Repeatedly steps through the list, compares adjacent elements and swaps them if they're in the wrong order
 * - Time Complexity: O(n²) for worst and average cases, O(n) for best case (already sorted)
 * - Space Complexity: O(1) as it's an in-place sorting algorithm
 * - Not suitable for large data sets due to quadratic time complexity
 *
 * Odd-Even Transposition:
 * ----------------------
 * - A parallel version of bubble sort
 * - Alternates between comparing odd-indexed elements with their right neighbor, then even-indexed elements
 * - Each phase (odd or even) can be parallelized since comparisons don't overlap
 * - Maintains the O(n²) time complexity but can utilize multiple cores
 *
 * OpenMP Parallelization:
 * ----------------------
 * - This implementation uses #pragma omp parallel for to parallelize the odd and even phases
 * - Each thread handles a subset of comparisons independently
 * - Thread synchronization occurs between odd and even phases
 *
 * Implementation Analysis:
 * ----------------------
 * - The parallelization works well as each comparison is independent within a phase
 * - Load balancing is good as work is evenly distributed
 * - Memory access patterns are regular and cache-friendly
 * - Cannot terminate early upon detecting a sorted array due to parallelization
 *
 * Performance Considerations:
 * --------------------------
 * - For small arrays, the overhead of thread creation may outweigh benefits
 * - Synchronization between odd and even phases limits scalability
 * - Performance gains are most noticeable with many cores and large arrays
 *
 * SAMPLE INPUT/OUTPUT:
 * ------------------
 *
 * Input:
 *   9                            (Array size)
 *   5 2 9 1 7 6 8 3 4           (Array elements)
 *
 * Output:
 *   Original array: 5 2 9 1 7 6 8 3 4
 *   Sequential bubble sort time: 0.000034 seconds
 *   Sorted array (sequential): 1 2 3 4 5 6 7 8 9
 *   Parallel bubble sort time: 0.000021 seconds
 *   Sorted array (parallel): 1 2 3 4 5 6 7 8 9
 */

#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Sequential bubble sort implementation for comparison
void bubble_sort_sequential(vector<int> &arr)
{
    int n = arr.size();
    bool swapped;

    for (int i = 0; i < n - 1; i++)
    {
        swapped = false;

        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        // If no swapping occurred in this pass, array is sorted
        if (!swapped)
            break;
    }
}

// Parallel bubble sort using odd-even transposition
void bubble_sort_odd_even(vector<int> &arr)
{
    bool isSorted = false;
    int n = arr.size();

    // Continue until array is sorted
    while (!isSorted)
    {
        isSorted = true;

// Compare and swap elements at even indices with their right neighbors
#pragma omp parallel for reduction(&& : isSorted)
        for (int i = 0; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }

// Compare and swap elements at odd indices with their right neighbors
#pragma omp parallel for reduction(&& : isSorted)
        for (int i = 1; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
    }
}

int main()
{
    int n;

    // Get array size from user
    cout << "Enter the size of the array: ";
    cin >> n;

    // Ensure array size is valid
    if (n <= 0)
    {
        cout << "Invalid array size!" << endl;
        return 1;
    }

    vector<int> arr(n);

    // Get array elements from user
    cout << "Enter " << n << " integers: ";
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }

    cout << "\nOriginal array: ";
    for (int num : arr)
        cout << num << " ";
    cout << endl;

    // Create copies for sequential and parallel sort
    vector<int> arr_seq = arr;
    vector<int> arr_par = arr;

    // Measure performance of sequential bubble sort
    double start = omp_get_wtime();
    bubble_sort_sequential(arr_seq);
    double end = omp_get_wtime();

    cout << "Sequential bubble sort time: " << end - start << " seconds" << endl;
    cout << "Sorted array (sequential): ";
    for (int num : arr_seq)
        cout << num << " ";
    cout << endl;

    // Measure performance of parallel bubble sort
    start = omp_get_wtime();
    bubble_sort_odd_even(arr_par);
    end = omp_get_wtime();

    cout << "Parallel bubble sort time: " << end - start << " seconds" << endl;
    cout << "Sorted array (parallel): ";
    for (int num : arr_par)
        cout << num << " ";
    cout << endl;

    return 0;
}