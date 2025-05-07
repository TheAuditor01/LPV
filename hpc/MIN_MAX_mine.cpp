#include <omp.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
using namespace std;
using namespace std::chrono;

// Display Array (only first 20 elements for large arrays)
void displayArray(int nums[], int length)
{
    cout << "Nums (first 20): [";
    for (int i = 0; i < min(length, 20); i++) {
        cout << nums[i];
        if (i != min(length, 20) - 1)
            cout << ", ";
    }
    if (length > 20) cout << ", ...";
    cout << "]" << endl;
}

// Sequential Min, Max, Sum, Avg functions
int seqMin(int nums[], int length)
{
    int minVal = nums[0];
    for (int i = 1; i < length; i++)
        if (nums[i] < minVal)
            minVal = nums[i];
    return minVal;
}

int seqMax(int nums[], int length)
{
    int maxVal = nums[0];
    for (int i = 1; i < length; i++)
        if (nums[i] > maxVal)
            maxVal = nums[i];
    return maxVal;
}

int seqSum(int nums[], int length)
{
    int sum = 0;
    for (int i = 0; i < length; i++)
        sum += nums[i];
    return sum;
}

float seqAvg(int nums[], int length)
{
    int sum = seqSum(nums, length);
    return static_cast<float>(sum) / length;
}

// Parallel Min, Max, Sum, Avg functions
int parallelMin(int nums[], int length)
{
    int minVal = nums[0];
#pragma omp parallel for reduction(min : minVal)
    for (int i = 0; i < length; i++)
        minVal = (nums[i] < minVal) ? nums[i] : minVal;
    return minVal;
}

int parallelMax(int nums[], int length)
{
    int maxVal = nums[0];
#pragma omp parallel for reduction(max : maxVal)
    for (int i = 0; i < length; i++)
        maxVal = (nums[i] > maxVal) ? nums[i] : maxVal;
    return maxVal;
}

int parallelSum(int nums[], int length)
{
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < length; i++)
        sum += nums[i];
    return sum;
}

float parallelAvg(int nums[], int length)
{
    int sum = parallelSum(nums, length);
    return static_cast<float>(sum) / length;
}

// Main
int main()
{
    srand(time(0)); // Seed random number generator

    int length;
    cout << "Enter number of elements: ";
    cin >> length;

    int* nums = new int[length];
    for (int i = 0; i < length; i++) {
        nums[i] = rand() % length + 1;  // Random values from 1 to 1000
    }

    displayArray(nums, length);

    // Sequential
    auto startSeq = high_resolution_clock::now();
    int sMin = seqMin(nums, length);
    int sMax = seqMax(nums, length);
    int sSum = seqSum(nums, length);
    float sAvg = seqAvg(nums, length);
    auto endSeq = high_resolution_clock::now();

    cout << "\nSequential Results:" << endl;
    cout << "Min: " << sMin << "\nMax: " << sMax << "\nSum: " << sSum << "\nAverage: " << sAvg << endl;
    cout << "Sequential execution time: " << duration_cast<microseconds>(endSeq - startSeq).count() << " microseconds" << endl;

    // Parallel
    auto startPar = high_resolution_clock::now();
    int pMin = parallelMin(nums, length);
    int pMax = parallelMax(nums, length);
    int pSum = parallelSum(nums, length);
    float pAvg = parallelAvg(nums, length);
    auto endPar = high_resolution_clock::now();

    cout << "\nParallel Results:" << endl;
    cout << "Min: " << pMin << "\nMax: " << pMax << "\nSum: " << pSum << "\nAverage: " << pAvg << endl;
    cout << "Parallel execution time: " << duration_cast<microseconds>(endPar - startPar).count() << " microseconds" << endl;

    delete[] nums;
    return 0;
}



// The time difference happens due to several factors:

// 1. **Overhead of Parallelization:** When you use parallelism, the system needs to manage multiple threads. This involves overhead like dividing the work, scheduling tasks, and synchronizing between threads. For smaller datasets, this overhead can actually make parallel execution slower than the sequential version.

// 2. **Thread Management Costs:** Spawning, synchronizing, and managing threads take time, especially for a relatively small task. With 10,000 elements, the CPU may not be fully utilized, leading to threads waiting for each other, which adds delay.

// 3. **Diminishing Returns:** Parallelization shows significant benefits when dealing with larger datasets or computationally intensive tasks. For smaller datasets, the time it takes to split the task across threads can exceed the time saved by parallel execution, leading to slower performance.

// In summary, for small datasets (like 10,000 elements), parallelizing doesn’t offer performance improvements. The system spends more time managing the threads than processing the data. For larger datasets, parallelization becomes more efficient as the task size grows.
