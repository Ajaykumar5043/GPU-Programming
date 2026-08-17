#include <iostream>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor){
    switch(major){
        case 2:
            return (minor == 1) ? 48 : 32;
        case 3:
            return 192;
        case 5:
            return 128;
        case 6:
            if (minor == 1 || minor == 2)
                return 128;
            return 64;
        case 7:
            return 64;
        case 8:
            if(minor == 6 || minor == 9)
                return 128;
            return 64;
        case 9:
            return 128;
        default:
            return 128;
    }
}

int main(){
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if(error != cudaSuccess){
        std::cerr << "CUDA Error: "
                  << cudaGetErrorString(error)
                  << std::endl;
        return 1;
    }
    std::cout << "Found "
              << deviceCount
              << " CUDA device(s).\n\n";
    
    for(int i = 0; i < deviceCount; ++i){
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        if(error != cudaSuccess){
            std::cerr << "Error getting device properties: "
                      << cudaGetErrorString(error)
                      << std::endl;
            continue;
        }
        int coresPerSM = getCoresPerSM(prop.major, prop.minor);

        std::cout << "============================================\n";
        std::cout << "          CUDA DEVICE " << i << "\n";
        std::cout << "============================================\n";

        std::cout << "Device Name:              "
                  << prop.name << "\n";
        std::cout << "Compute Capability        "
                  << prop.major << "." << prop.minor << "\n";

        std::cout << "Total Global Memory:      "
                  << prop.totalGlobalMem / (1024 * 1024)
                  << "MB\n";
        
        std::cout << "Shared Memory Per Block   "
                  << prop.sharedMemPerBlock / 1024
                  << "KB\n";
        
        std::cout << "Constant Memory           "
                  << prop.totalConstMem / 1024
                  << "KB\n";

        std::cout << "L2 Cache:                 "
                  << prop.l2CacheSize / 1024
                  << "KB\n";
        
        std::cout << "Streaming Multiprocessors:"
                  << prop.multiProcessorCount << "\n";

        std::cout << "Cores per SM:             "
                  << coresPerSM << "\n";
                
        std::cout << "Approx. Total CUDA Cores: "
                  << prop.multiProcessorCount * coresPerSM
                  << "\n";

        std::cout << "Warp Size:                "
                  << prop.warpSize << "\n";
                
        std::cout << "Max Threads Per Block:    "
                  << prop.maxThreadsPerBlock << "\n";
                
        std::cout << "Max Threads Per SM:       "
                  << prop.maxThreadsPerMultiProcessor << "\n";
        
        std::cout << "Max Threads Dimension:    "
                  << prop.maxThreadsDim[0] << " x "
                  << prop.maxThreadsDim[1] << " x "
                  << prop.maxThreadsDim[2] << "\n";
                
        std::cout << "Registers Per Block:      "
                  << prop.regsPerBlock << "\n";



        int clockRate;
int memoryClockRate;

cudaDeviceGetAttribute(
    &clockRate,
    cudaDevAttrClockRate,
    i
);

cudaDeviceGetAttribute(
    &memoryClockRate,
    cudaDevAttrMemoryClockRate,
    i
);
        std::cout << "GPU Clock rate:           "
                  << clockRate / 1000
                  << "MHz\n";

        std::cout << "Memory Clock Rate:        "
                  <<  memoryClockRate / 1000
                  << "MHz\n";

        std::cout << "Memory Bus Width:         "
                  << prop.memoryBusWidth
                  << " bits\n";

        std::cout << "Concurrent Kernels:       "
                  << (prop.concurrentKernels ? "Yes" : "No")
                  << "\n";
        
        std::cout << "Unified Addresssing:      "
                  << (prop.unifiedAddressing ? "Yes" : "No")
                  << "\n";

        std::cout << "ECC Support:              "
                  << (prop.ECCEnabled ? "Yes" : "No")
                  << "\n";

        std::cout << "===========================================\n\n";
    }
    return 0;
}