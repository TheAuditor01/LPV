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

// Sequential Min
int seqMin(int nums[], int length)
{
    int minVal = nums[0];
    for (int i = 1; i < length; i++)
        if (nums[i] < minVal)
            minVal = nums[i];
    return minVal;
}

// Sequential Max
int seqMax(int nums[], int length)
{
    int maxVal = nums[0];
    for (int i = 1; i < length; i++)
        if (nums[i] > maxVal)
            maxVal = nums[i];
    return maxVal;
}

// Sequential Sum
int seqSum(int nums[], int length)
{
    int sum = 0;
    for (int i = 0; i < length; i++)
        sum += nums[i];
    return sum;
}

// Parallel Min
int parallelMin(int nums[], int length)
{
    int minVal = nums[0];
#pragma omp parallel for reduction(min : minVal)
    for (int i = 0; i < length; i++)
        minVal = (nums[i] < minVal) ? nums[i] : minVal;
    return minVal;
}

// Parallel Max
int parallelMax(int nums[], int length)
{
    int maxVal = nums[0];
#pragma omp parallel for reduction(max : maxVal)
    for (int i = 0; i < length; i++)
        maxVal = (nums[i] > maxVal) ? nums[i] : maxVal;
    return maxVal;
}

// Parallel Sum
int parallelSum(int nums[], int length)
{
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < length; i++)
        sum += nums[i];
    return sum;
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
        nums[i] = rand() % 1000 + 1;  // Random values from 1 to 1000
    }

    displayArray(nums, length);

    // Sequential
    auto startSeq = high_resolution_clock::now();
    int sMin = seqMin(nums, length);
    int sMax = seqMax(nums, length);
    int sSum = seqSum(nums, length);
    float sAvg = static_cast<float>(sSum) / length;
    auto endSeq = high_resolution_clock::now();

    cout << "\nSequential Results:" << endl;
    cout << "Min: " << sMin << "\nMax: " << sMax << "\nSum: " << sSum << "\nAverage: " << sAvg << endl;
    cout << "Sequential execution time: " << duration_cast<microseconds>(endSeq - startSeq).count() << " microseconds" << endl;

    // Parallel
    auto startPar = high_resolution_clock::now();
    int pMin = parallelMin(nums, length);
    int pMax = parallelMax(nums, length);
    int pSum = parallelSum(nums, length);
    float pAvg = static_cast<float>(pSum) / length;
    auto endPar = high_resolution_clock::now();

    cout << "\nParallel Results:" << endl;
    cout << "Min: " << pMin << "\nMax: " << pMax << "\nSum: " << pSum << "\nAverage: " << pAvg << endl;
    cout << "Parallel execution time: " << duration_cast<microseconds>(endPar - startPar).count() << " microseconds" << endl;

    delete[] nums;
    return 0;
}
