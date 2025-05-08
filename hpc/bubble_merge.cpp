#include <iostream>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <chrono>
 
using namespace std;
using namespace std::chrono;
 
void swap(int &a, int &b)
{
    int test = a;
    a = b;
    b = test;
}

void bubbleSort(int arr[], int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void parallelBubbleSort(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {   
        int first = i % 2;  
        #pragma omp parallel for shared(a, first)
        for (int j = first; j < n - 1; j += 2)
        {   
            if (a[j] > a[j + 1])
            {   
                swap(a[j], a[j + 1]);
            }   
        }   
    }
}
 
void merge(int arr[], int l, int m, int r)
{
    int n1 = m - l + 1;
    int n2 = r - m;
 
    int *L = new int[n1];
    int *R = new int[n2];
 
    for (int i = 0; i < n1; ++i)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[m + 1 + j];
 
    int i = 0, j = 0, k = l;
 
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
 
    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
 
    delete[] L;
    delete[] R;
}
 
void mergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(arr, l, m);
            #pragma omp section
            mergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}
 
void printArray(int arr[], int size)
{
    for (int i = 0; i < size; ++i)
        cout << arr[i] << " ";
    cout << endl;
}
 
int main()
{
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;

    srand(time(0));  // Seed for random numbers

    int *arr = new int[n];
    int *arr_copy = new int[n];

    // Fill the array with random numbers
    for (int i = 0; i < n; ++i)
    {
        arr[i] = rand() % 1000;  // Random number between 0-999
        arr_copy[i] = arr[i];
    }

    cout << "\nOriginal array (first 20 elements max): ";
    for (int i = 0; i < min(n, 20); ++i)
        cout << arr[i] << " ";
    cout << "\n";

    auto start = high_resolution_clock::now();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    // Sequential Bubble Sort
    int *arr1 = new int[n];
    copy(arr_copy, arr_copy + n, arr1);
    start = high_resolution_clock::now();
    bubbleSort(arr1, n);
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    cout << "\nSequential Bubble Sort - Time: " << duration.count() << " µs\n";

    // Parallel Bubble Sort
    int *arr2 = new int[n];
    copy(arr_copy, arr_copy + n, arr2);
    start = high_resolution_clock::now();
    parallelBubbleSort(arr2, n);
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    cout << "Parallel Bubble Sort - Time: " << duration.count() << " µs\n";

    // Sequential Merge Sort
    int *arr3 = new int[n];
    copy(arr_copy, arr_copy + n, arr3);
    start = high_resolution_clock::now();
    mergeSort(arr3, 0, n - 1);
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    cout << "Sequential Merge Sort - Time: " << duration.count() << " µs\n";

    // Parallel Merge Sort
    int *arr4 = new int[n];
    copy(arr_copy, arr_copy + n, arr4);
    start = high_resolution_clock::now();
    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(arr4, 0, n - 1);
    }
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    cout << "Parallel Merge Sort - Time: " << duration.count() << " µs\n";

    // Clean up
    delete[] arr;
    delete[] arr_copy;
    delete[] arr1;
    delete[] arr2;
    delete[] arr3;
    delete[] arr4;

    return 0;
}
