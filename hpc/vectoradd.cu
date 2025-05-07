/*
 * Vector Addition Program using CUDA
 * ==================================
 *
 * This program demonstrates a simple vector addition using CUDA.
 * It initializes two vectors A and B on the host (CPU), copies them to the device (GPU),
 * performs the addition on the GPU using a custom CUDA kernel, and then copies the result
 * vector C back to the host to print.
 *
 * COMPILATION & EXECUTION (Linux/macOS with NVIDIA CUDA Toolkit):
 * ---------------------------------------------------------------
 * 1. Ensure NVIDIA CUDA Toolkit is installed.
 * 2. Save the code as vectoradd.cu (or vectoradd.cpp if your nvcc is configured for it).
 * 3. Compile:
 *    nvcc vectoradd.cu -o vectoradd
 * 4. Execute:
 *    ./vectoradd
 *
 * THEORETICAL CONCEPTS:
 *
 * CUDA (Compute Unified Device Architecture):
 * ------------------------------------------
 * - A parallel computing platform and programming model created by NVIDIA.
 * - Allows software developers to use a CUDA-enabled graphics processing unit (GPU) for
 *   general purpose processing (an approach known as GPGPU).
 *
 * Key CUDA Concepts Used:
 * -----------------------
 * - `__global__` function (Kernel): A function that runs on the GPU and can be called
 *   from host code. In this program, `add` is the kernel.
 * - `blockIdx`, `blockDim`, `threadIdx`: Built-in CUDA variables that specify the unique
 *   ID of a thread within the GPU grid structure.
 *   - `blockIdx.x`: The x-dimension index of the current thread block within the grid.
 *   - `blockDim.x`: The number of threads in the x-dimension of a block.
 *   - `threadIdx.x`: The x-dimension index of the current thread within its block.
 * - `cudaMalloc()`: Allocates memory on the GPU.
 * - `cudaMemcpy()`: Copies data between host (CPU) memory and device (GPU) memory.
 *   - `cudaMemcpyHostToDevice`: Host to Device.
 *   - `cudaMemcpyDeviceToHost`: Device to Host.
 * - Kernel Launch (`add<<<blocksPerGrid, threadsPerBlock>>>`): Syntax to execute a
 *   `__global__` function on the GPU.
 *   - `blocksPerGrid`: The number of thread blocks in the grid.
 *   - `threadsPerBlock`: The number of threads in each block.
 * - `cudaFree()`: Frees memory on the GPU.
 *
 * Host Code vs. Device Code:
 * --------------------------
 * - Host code runs on the CPU (e.g., `main` function, memory allocation with `new`).
 * - Device code runs on the GPU (e.g., the `add` kernel).
 *
 * Workflow:
 * ---------
 * 1. Initialize data on the host.
 * 2. Allocate memory on the device for this data.
 * 3. Copy data from host to device.
 * 4. Execute the kernel on the device to process the data.
 * 5. Copy results from device back to host.
 * 6. Free device memory.
 * 7. Free host memory.
 *
 * Error Handling (Simplified):
 * ----------------------------
 * - This basic example does not include explicit CUDA error checking (e.g., checking return
 *   values of `cudaMalloc`, `cudaMemcpy`, `cudaGetLastError`). In real-world applications,
 *   robust error handling is crucial.
 *
 * Sample Output (for N=4, random numbers):
 * ----------------------------------------
 * Vector A: 3 6 7 5
 * Vector B: 9 2 4 1
 * Addition: 12 8 11 6
 *
 * (Note: The actual numbers will vary due to `rand() % 10`)
 */
#include <iostream>
using namespace std;
__global__ void add(int *A, int *B, int *C, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        C[tid] = A[tid] + B[tid];
    }
}
void initialize(int *vector, int size)
{
    for (int i = 0; i < size; i++)
    {
        vector[i] = rand() % 10;
    }
}
void print(int *vector, int size)
{
    for (int i = 0; i < size; i++)
    {
        cout << vector[i] << " ";
    }
    cout << endl;
}
int main()
{
    int N = 4;
    int *A, *B, *C;
    int vectorSize = N;
    size_t vectorBytes = vectorSize * sizeof(int);
    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];
    initialize(A, vectorSize);
    initialize(B, vectorSize);
    cout << "Vector A: ";
    print(A, N);
    cout << "Vector B: ";
    print(B, N);
    int *X, *Y, *Z;
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);
    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);
    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);
    cout << "Addition: ";
    print(C, N);
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);
    return 0;
}

/*
 * ## Theoretical Concepts for CUDA Vector Addition ##
 *
 * This program implements vector addition (C = A + B) using NVIDIA's CUDA framework.
 * The core idea is to leverage the parallel processing capabilities of a GPU.
 *
 * 1. Vector Addition:
 *    - Given two vectors A and B of the same size n, their sum C is a vector where each
 *      element C[i] = A[i] + B[i].
 *    - This operation is inherently parallel because each element C[i] can be computed
 *      independently of the others.
 *
 * 2. CUDA Parallelism Model:
 *    - Kernel (`__global__` function): A function written in CUDA C/C++ that runs on the GPU.
 *      In this program, `add` is the kernel.
 *    - Threads: The basic unit of execution on the GPU. Many threads run the same kernel code
 *      in parallel.
 *    - Thread Blocks: Threads are grouped into blocks. Threads within a block can cooperate by
 *      sharing data through shared memory and synchronizing their execution.
 *    - Grid: Blocks are organized into a grid. All threads in a grid execute the same kernel.
 *    - Thread Hierarchy (Built-in Variables):
 *        - `threadIdx.x`: The index of a thread within its block (1D in this case).
 *        - `blockDim.x`: The number of threads in a block (1D).
 *        - `blockIdx.x`: The index of a block within the grid (1D).
 *    - Global Thread ID: A unique ID for each thread across the entire grid can be calculated.
 *      For a 1D grid of 1D blocks (as used here):
 *      `int tid = blockIdx.x * blockDim.x + threadIdx.x;`
 *      This `tid` is then used to map each thread to a specific element of the vectors.
 *
 * 3. Memory Spaces:
 *    - Host Memory: CPU's RAM. Vectors A, B, and C are initially created here.
 *    - Device Memory: GPU's RAM. Vectors X, Y, and Z (corresponding to A, B, C) are stored here
 *      during the GPU computation.
 *    - `cudaMalloc()`: Allocates memory on the device.
 *    - `cudaMemcpy()`: Transfers data between host and device memory.
 *        - `cudaMemcpyHostToDevice`: Copies data from CPU to GPU.
 *        - `cudaMemcpyDeviceToHost`: Copies data from GPU to CPU.
 *    - `cudaFree()`: Deallocates memory on the device.
 *
 * 4. Program Workflow:
 *    a. Initialization (Host): Vectors A and B are created and filled with values on the CPU.
 *       Memory for C is also allocated on the host.
 *    b. Device Memory Allocation (Host calls CUDA API): Memory is allocated on the GPU for X, Y, Z.
 *    c. Data Transfer (Host to Device): Contents of A and B are copied to X and Y on the GPU.
 *    d. Kernel Launch (Host calls Kernel): The `add` kernel is launched on the GPU.
 *       - `add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);`
 *       - `blocksPerGrid`: Specifies the number of thread blocks in the grid.
 *       - `threadsPerBlock`: Specifies the number of threads in each block.
 *       - The host code calculates these values to ensure one thread per vector element.
 *    e. Kernel Execution (Device): Each GPU thread executes the `add` kernel. Using its unique
 *       `tid`, it computes `Z[tid] = X[tid] + Y[tid]`.
 *    f. Data Transfer (Device to Host): The resulting vector Z is copied from the GPU to vector C on the CPU.
 *    g. Cleanup (Host calls CUDA API & C++): Memory allocated on the device (`cudaFree`) and on the
 *       host (`delete[]`) is freed.
 *
 * 5. Scalability:
 *    - The performance benefit of CUDA comes from executing thousands of threads in parallel.
 *    - For vector addition, as the vector size `N` increases, the GPU can often perform the
 *      additions much faster than a CPU executing a sequential loop, provided `N` is large
 *      enough to overcome the overhead of data transfers and kernel launch.
 *
 * 6. Error Handling (Important Note):
 *    - This program omits explicit CUDA error checking (e.g., checking the return values of
 *      `cudaMalloc`, `cudaMemcpy`, and using `cudaGetLastError()` after kernel launches).
 *    - In production code, robust error handling is essential for diagnosing issues related
 *      to GPU operations or memory.
 */
