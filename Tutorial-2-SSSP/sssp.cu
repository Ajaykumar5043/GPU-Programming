#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <limits>
#include <chrono>
#define CUDA_CHECK(call)                                      \
do                                                           \
{                                                            \
    cudaError_t error = (call);                              \
    if (error != cudaSuccess)                                \
    {                                                        \
        cerr << "CUDA Error: " << cudaGetErrorString(error)  \
             << "\n";                                       \
        return 1;                                            \
    }                                                        \
} while (0)

using namespace std;
struct Edge{
    int src;
    int dst;
    int weight;
};

//CPU Bellman-FORD using CSR
vector<int> bellmanFordCPU(int V, int source, 
    const vector<int>& row_offsets,
    const vector<int>& col_indices,
    const vector<int>& weights){
    
    const int INF = numeric_limits<int>::max();

    //initialize distances
    vector<int> dist(V,INF);
    dist[source] = 0;

    //Bellman-Ford
    for(int iter = 0; iter<V-1;iter++){
        bool changed = false;
        //visit every vertex
        for(int u = 0; u < V; u++){
            if(dist[u] == INF){
                continue;
            }
            //Find all outgoing edges of u using CSR
            int start = row_offsets[u];
            int end = row_offsets[u+1];
            for(int i = start;i<end;i++){
                int v = col_indices[i];
                int w = weights[i];
                //Relax edge u->v
                int new_distance = dist[u] + w;
                if(new_distance < dist[v]){
                    dist[v] = new_distance;
                    changed = true;
                }
            }
        }
        //If no distance changed during this iteration,
        // the shortest paths have already converged.
        if(!changed){
            cout << "CPU Bellman-Ford converged after "
                 << iter + 1
                 << " iteration(s).\n";
            break;
        }
    }
    return dist;
}

//GPU relaxation kernel
__global__ void relaxKernel(const int* row_offsets, const int* col_indices, const int* weights, const int* dist, int* new_dist,int* changed, int V){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if( u >= V){
        return;
    }
    const int INF = 2147483647;
    if(dist[u] == INF){
        return;
    }
    int start = row_offsets[u];
    int end = row_offsets[u+1];

    for(int i = start; i<end; i++){
        int v = col_indices[i];
        int w = weights[i];

        int new_distance = dist[u] + w;

        int old_distance = atomicMin(&new_dist[v], new_distance);
        if(new_distance < old_distance){
            atomicExch(changed, 1);
        }
    }
}


int main(int argc, char* argv[]){
    //read graph from file 
    const char* filename = "Graph.txt";
    if(argc > 1){
        filename = argv[1];
    }

    ifstream file(filename);
    cout << "Input graph: " << filename << "\n\n";
    if(!file){
        cerr << "Error: Could not open\n"
             << filename
             << "\n";
        return 1;
    }
    int V,E;

    file >> V >> E;
    if(V <= 0 || E < 0)
    {
        cerr << "Error: Invalid graph dimensions.\n";
        return 1;
    }
    cout << "Number of vertices: " << V << '\n';
    cout << "Number of edges: "   << E << "\n\n";


    //Read all edges
    vector<Edge> edges(E);
    for(int i = 0; i < E; i++)
    {
        file >> edges[i].src
            >> edges[i].dst
            >> edges[i].weight;

        if(edges[i].src < 0 || edges[i].src >= V ||
        edges[i].dst < 0 || edges[i].dst >= V)
        {
            cerr << "Error: Invalid edge at line "
                << i + 2
                << ".\n";

            return 1;
        }
    }
    file.close();

    //Build csr
    vector<int> row_offsets(V+1, 0);
    vector<int> col_indices(E);
    vector<int> weights(E);

    //Count outgoing edges from every vertex
    for(const Edge& edge: edges){
        row_offsets[edge.src+1]++;
    }
    //convert counts into offsets
    for(int i = 1; i <= V;i++){
        row_offsets[i] += row_offsets[i-1];
    }
    //Temporary position array
    vector<int> position = row_offsets;

    //Fill column indices and weights
    for(const Edge& edge : edges){
        int index = position[edge.src]++;
        col_indices[index] = edge.dst;
        weights[index] = edge.weight;
    }

    //CPU SSSP
    int source = 0;
    cout <<"\nSource vertex: "<< source << "\n";
    // CPU PERFORMANCE MEASUREMENT
    auto cpu_start = chrono::high_resolution_clock::now();

    vector<int> cpu_dist =
        bellmanFordCPU(V,source,row_offsets,col_indices,weights);

    auto cpu_end = chrono::high_resolution_clock::now();

    double cpu_time_ms =
        chrono::duration<double, milli>(cpu_end - cpu_start).count();

    cout << "\nCPU Bellman-Ford Time: "
        << cpu_time_ms
        << " ms\n";

    //CPU Result sample
    cout << "\nCPU SSSP Sample\n";
    cout << "===============\n";
    cout << "dist[0] = " << cpu_dist[0] << '\n';

    if(V > 1)
        cout << "dist[1] = " << cpu_dist[1] << '\n';

    if(V > 100)
        cout << "dist[100] = " << cpu_dist[100] << '\n';

    if(V > 1000)
        cout << "dist[1000] = " << cpu_dist[1000] << '\n';

    if(V > 4999)
        cout << "dist[4999] = " << cpu_dist[4999] << '\n';

    //Allocate CSR arrays on GPU
    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
    int* d_weights = nullptr;

    int* d_dist = nullptr;
    int* d_new_dist = nullptr;
    int* d_changed = nullptr;
    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;

    cudaEvent_t gpu_total_start;
    cudaEvent_t gpu_total_stop;

    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));

    CUDA_CHECK(cudaEventCreate(&gpu_total_start));
    CUDA_CHECK(cudaEventCreate(&gpu_total_stop));

    CUDA_CHECK(cudaMalloc(&d_row_offsets,(V + 1)*sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_col_indices,E * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_weights,E * sizeof(int)));

    //Allocate distance arrays
    CUDA_CHECK(cudaMalloc(&d_dist,V*sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_changed,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_dist,V * sizeof(int)));
    CUDA_CHECK(cudaEventRecord(gpu_total_start));
    //Copy CSR data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_row_offsets,row_offsets.data(),(V+1)*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices,col_indices.data(),E * sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights,weights.data(),E * sizeof(int),cudaMemcpyHostToDevice));
    //Initialize GPU distances
    const int INF = numeric_limits<int>::max();
    vector<int> initial_dist(V, INF);
    initial_dist[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_dist, initial_dist.data(), V*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new_dist,d_dist, V * sizeof(int), cudaMemcpyDeviceToDevice));

    //Launch GPU
    int threadsPerBlock = 512;
    int blocks = (V+threadsPerBlock - 1)/threadsPerBlock;
    cout << "\nLaunching GPU relaxation kernel...\n";
    cout << "Threads Per Block: "
         << threadsPerBlock
         << '\n';

    cout << "Number of Blocks: "
         << blocks
         << "\n";
    
    cout << "\nGPU Bellman-Ford\n";
    cout << "==============================\n";
    CUDA_CHECK(cudaEventRecord(gpu_start));

    for(int iter = 0; iter < V - 1; iter++)
    {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_changed,&zero,sizeof(int),cudaMemcpyHostToDevice));
        // Copy current distances to new_dist
        CUDA_CHECK(cudaMemcpy(d_new_dist,d_dist,V * sizeof(int),cudaMemcpyDeviceToDevice));

        // Launch GPU relaxation kernel
        relaxKernel<<<blocks, threadsPerBlock>>>(d_row_offsets,d_col_indices,d_weights,d_dist,d_new_dist,d_changed,V);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // New distances become current distances
        swap(d_dist, d_new_dist);
        int changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed,d_changed,sizeof(int),cudaMemcpyDeviceToHost));
        if (iter == 0 ||iter == V - 2){
        cout << "GPU iteration "
            << iter + 1
            << " completed.\n";
        }
        if(changed == 0) {
            cout << "GPU Bellman-Ford converged after "
                << iter + 1
                << " iteration(s).\n";
            break;
        }
    }
    CUDA_CHECK(cudaEventRecord(gpu_stop));
    CUDA_CHECK(cudaEventSynchronize(gpu_stop));

    float gpu_time_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms,gpu_start,gpu_stop));

    cout << "GPU Kernel Time: "
        << gpu_time_ms
        << " ms\n";
    double kernel_speedup = cpu_time_ms / gpu_time_ms;
    cout << "CPU / GPU Kernel Speedup: "
        << kernel_speedup
        << "x\n";
    
    // Copy final GPU result back to CPU
    vector<int> gpu_dist(V);

    CUDA_CHECK(cudaMemcpy(gpu_dist.data(),d_dist,V * sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(gpu_total_stop));
    CUDA_CHECK(cudaEventSynchronize(gpu_total_stop));

    float gpu_total_time_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&gpu_total_time_ms,gpu_total_start,gpu_total_stop));

    cout << "GPU End-to-End Time: "
        << gpu_total_time_ms
        << " ms\n";

    double end_to_end_speedup = cpu_time_ms / gpu_total_time_ms;

    cout << "CPU / GPU End-to-End Speedup: "
        << end_to_end_speedup
        << "x\n";

    // GPU SSSP sample
    cout << "\nGPU SSSP Sample\n";
    cout << "===============\n";

    cout << "dist[0] = " << gpu_dist[0] << '\n';

    if (V > 1)
        cout << "dist[1] = " << gpu_dist[1] << '\n';

    if (V > 100)
        cout << "dist[100] = " << gpu_dist[100] << '\n';

    if (V > 1000)
        cout << "dist[1000] = " << gpu_dist[1000] << '\n';

    if (V > 4999)
        cout << "dist[4999] = " << gpu_dist[4999] << '\n';


    // ========================================================
    // CPU vs GPU VALIDATION
    // ========================================================

    cout << "\nResults - Verification\n";
    cout << "========================\n";

    if (gpu_dist == cpu_dist)
    {
        cout << "CPU vs GPU SSSP: PASS\n";
    }
    else
    {
        cout << "CPU vs GPU SSSP: FAIL\n";
    }


    vector<int> test_row_offsets(V + 1);
    vector<int> test_col_indices(E);
    vector<int> test_weights(E);


    //Copy databack from GPU to CPU for verification
    CUDA_CHECK(cudaMemcpy(test_row_offsets.data(),d_row_offsets,(V + 1)*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(test_col_indices.data(),d_col_indices,E*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(test_weights.data(),d_weights,E*sizeof(int),cudaMemcpyDeviceToHost));

    //Verify GPU data
    bool transferCorrect = true;
    if(test_row_offsets != row_offsets){
        transferCorrect = false;
    }
    if(test_col_indices != col_indices){
        transferCorrect = false;
    }
    if(test_weights != weights){
        transferCorrect = false;
    }
    if(transferCorrect){
        cout << "GPU CSR transfer verification: PASS\n";
    }
    else{
        cout << "GPU CSR transfer verification: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_new_dist));
    CUDA_CHECK(cudaFree(d_changed));
    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));
    CUDA_CHECK(cudaEventDestroy(gpu_total_start));
    CUDA_CHECK(cudaEventDestroy(gpu_total_stop));

    cout << "\nGPU memory freed successfully.\n";
    return 0;

}