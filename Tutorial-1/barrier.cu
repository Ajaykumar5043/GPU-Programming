#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void barrierKernel(int *counter){
    int tid = threadIdx.x;
    printf("Thread %d reached the barrier\n", tid);
    atomicAdd(counter,1);
    __syncthreads();
    printf("Thread %d passed the barrier\n", tid);
}

int main(){
    const int N = 128;
    int counter = 0;

    int *d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice);

    barrierKernel<<<32,N>>>(d_counter);

    cudaDeviceSynchronize();
    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Total threads reached barrier: %d\n", result);

    cudaFree(d_counter);
    return 0;
}
