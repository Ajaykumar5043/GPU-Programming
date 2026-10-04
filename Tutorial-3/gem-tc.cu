#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace wmma;

// Tensor Core tile dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
// Initialize matrices
__global__ void init(half *A, half *B, int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < total)
    {
        A[tid] = __float2half((float)tid);
        B[tid] = __float2half((float)tid);
    }
}

// ============================================================
// Tensor Core Matrix Multiplication
//
// A : M x K
// B : K x N
// C : M x N
//
// One warp computes one 16x16 output tile.
//
// For 64x64:
//     4 x 4 = 16 output tiles
//     16 warps
//     4 MMA operations per output tile
// ============================================================

__global__ void tensorCoreGemmKernel(half *A,half *B,float *C,int M,int N,int K)
{
    // One warp = 32 threads
    int warpId = threadIdx.x / 32;

    // Map warp to a 4x4 grid of 16x16 output tiles
    int tileRow = warpId / 4;
    int tileCol = warpId % 4;

    // Starting row and column of this output tile
    int mTile = tileRow * WMMA_M;
    int nTile = tileCol * WMMA_N;

    if (mTile >= M || nTile >= N)
        return;


    // --------------------------------------------------------
    // Declare WMMA fragments
    // --------------------------------------------------------

    fragment<matrix_a,WMMA_M, WMMA_N,WMMA_K,half,row_major> aFrag;

    fragment<matrix_b,WMMA_M,WMMA_N,WMMA_K,half,row_major> bFrag;

    fragment<accumulator,WMMA_M,WMMA_N,WMMA_K,float> cFrag;


    // Initialize accumulator to zero
    fill_fragment(cFrag, 0.0f);


    // --------------------------------------------------------
    // K = 64
    // Each Tensor Core operation handles K = 16
    //
    // 64 / 16 = 4 MMA operations
    // --------------------------------------------------------

    for (int i = 0; i < 4; i++)
    {
        // A tile:
        //
        // A[mTile : mTile+15]
        //  [i*16 : i*16+15]
        //
        // Starting address:
        int aind = mTile * K + i * WMMA_K;


        // B tile:
        //
        // B[i*16 : i*16+15]
        //  [nTile : nTile+15]
        //
        // Starting address:
        int bind = i * WMMA_K * N + nTile;


        // Load 16x16 tile of A
        load_matrix_sync(aFrag,A + aind,K);


        // Load 16x16 tile of B
        load_matrix_sync(bFrag,B + bind,N);


        // Tensor Core matrix multiply-accumulate:
        //
        // C = A * B + C
        //
        mma_sync(cFrag,aFrag,bFrag,cFrag);
    }


    // Store this warp's 16x16 output tile

    float *cTile = C + mTile * N + nTile;

    store_matrix_sync(cTile,cFrag,N,mem_row_major);
}


// ============================================================
// CPU reference matrix multiplication
//
// Used only for correctness verification.
//
// A and B are stored as half.
// Accumulation is performed in float.
// ============================================================

void cpuMatrixMultiply(
    half *A,
    half *B,
    float *C,
    int M,
    int N,
    int K)
{
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; k++)
            {
                float a = __half2float(A[row * K + k]);
                float b = __half2float(B[k * N + col]);

                sum += a * b;
            }

            C[row * N + col] = sum;
        }
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    // Required assignment dimensions
    int M = 64;
    int N = 64;
    int K = 64;


    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    half *hostA;
    half *hostB;

    float *hostGPU_C;
    float *hostCPU_C;

    hostA = (half *)malloc(M * K * sizeof(half));

    hostB = (half *)malloc(K * N * sizeof(half));

    hostGPU_C = (float *)malloc(M * N * sizeof(float));

    hostCPU_C = (float *)malloc(M * N * sizeof(float));


    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    half *devA;
    half *devB;
    float *devC;

    cudaError_t err;
    err = cudaMalloc(&devA,M * K * sizeof(half));

    if (err != cudaSuccess)
    {
        printf("cudaMalloc devA failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }


    err = cudaMalloc(&devB,K * N * sizeof(half));

    if (err != cudaSuccess)
    {
        printf("cudaMalloc devB failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }


    err = cudaMalloc(
        &devC,
        M * N * sizeof(float)
    );

    if (err != cudaSuccess)
    {
        printf("cudaMalloc devC failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }


    // --------------------------------------------------------
    // Initialize matrices on GPU
    //
    // 64 x 64 = 4096 elements
    // --------------------------------------------------------

    int total = M * K;

    int initThreads = 256;
    int initBlocks = (total + initThreads - 1) / initThreads;

    init<<<initBlocks, initThreads>>>(devA, devB, total);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf(
            "INIT LAUNCH ERROR: %s\n",
            cudaGetErrorString(err)
        );
        return 1;
    }


    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf(
            "INIT EXECUTION ERROR: %s\n",
            cudaGetErrorString(err)
        );
        return 1;
    }


    // --------------------------------------------------------
    // Copy initialized matrices back to CPU
    //
    // Needed for CPU reference computation.
    // --------------------------------------------------------

    cudaMemcpy(hostA,devA, M * K * sizeof(half),cudaMemcpyDeviceToHost);

    cudaMemcpy(hostB,devB,K * N * sizeof(half),cudaMemcpyDeviceToHost);

    // --------------------------------------------------------
    // CPU reference result
    // --------------------------------------------------------

    cpuMatrixMultiply(hostA,hostB,hostCPU_C,M,N,K);
    // Initialize GPU output
    cudaMemset(
        devC,
        0,
        M * N * sizeof(float)
    );
    // --------------------------------------------------------
    // CUDA timing events
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsedTime;


    // --------------------------------------------------------
    // Launch Tensor Core kernel
    //
    // 512 threads
    // = 16 warps
    // = 16 output tiles
    // --------------------------------------------------------

    cudaEventRecord(start, 0);

    tensorCoreGemmKernel<<<1, 512>>>(devA,devB,devC,M,N,K);


    // Check kernel launch
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf(
            "GEMM LAUNCH ERROR: %s\n",
            cudaGetErrorString(err)
        );
        return 1;
    }


    // Wait for kernel to finish
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf(
            "GEMM EXECUTION ERROR: %s\n",
            cudaGetErrorString(err)
        );
        return 1;
    }


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(
        &elapsedTime,
        start,
        stop
    );


    printf(
        "Kernel execution time: %f milli seconds\n",
        elapsedTime
    );


    // --------------------------------------------------------
    // Copy GPU result back to CPU
    // --------------------------------------------------------

    cudaMemcpy(hostGPU_C,devC,M * N * sizeof(float),cudaMemcpyDeviceToHost);
    // --------------------------------------------------------
    // Verify GPU result
    // --------------------------------------------------------

    float maxError = 0.0f;
    float maxRelativeError = 0.0f;
    int errorCount = 0;

    for (int i = 0; i < M * N; i++)
    {
        float gpu = hostGPU_C[i];
        float cpu = hostCPU_C[i];

        float error = fabs(gpu - cpu);

        float relativeError =
            error / fmaxf(1.0f, fabs(cpu));

        if (error > maxError)
            maxError = error;

        if (relativeError > maxRelativeError)
            maxRelativeError = relativeError;

        // Allow small floating-point differences
        if (relativeError > 1e-5f)
        {
            errorCount++;
        }
    }


    // --------------------------------------------------------
    // Print sample results
    // --------------------------------------------------------

    printf("\n");
    printf("Matrix size: %dx%d\n", M, N);
    printf("Tensor Core tile: %dx%d\n",
           WMMA_M, WMMA_N);

    printf("Number of output tiles: %d\n",
           (M / WMMA_M) * (N / WMMA_N));

    printf("Number of warps: %d\n",
           512 / 32);

    printf("MMA operations per output tile: %d\n",
           K / WMMA_K);

    printf("\n");

    printf(
        "C[0][0]       GPU = %f   CPU = %f\n",
        hostGPU_C[0],
        hostCPU_C[0]
    );

    printf(
        "C[15][15]     GPU = %f   CPU = %f\n",
        hostGPU_C[15 * N + 15],
        hostCPU_C[15 * N + 15]
    );

    printf(
        "C[32][32]     GPU = %f   CPU = %f\n",
        hostGPU_C[32 * N + 32],
        hostCPU_C[32 * N + 32]
    );

    printf(
        "C[63][63]     GPU = %f   CPU = %f\n",
        hostGPU_C[63 * N + 63],
        hostCPU_C[63 * N + 63]
    );


    printf("Maximum absolute error: %f\n", maxError);
    printf("Maximum relative error: %e\n", maxRelativeError);
    printf("Elements outside tolerance: %d\n", errorCount);


    if (errorCount == 0)
    {
        printf("\n");
        printf("=====================================\n");
        printf("        CORRECTNESS: PASS\n");
        printf("=====================================\n");
    }
    else
    {
        printf("\n");
        printf("=====================================\n");
        printf("        CORRECTNESS: FAIL\n");
        printf("=====================================\n");
    }


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    free(hostA);
    free(hostB);
    free(hostGPU_C);
    free(hostCPU_C);

    return 0;
}