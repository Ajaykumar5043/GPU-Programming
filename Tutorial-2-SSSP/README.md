# Tutorial-2: Single Source Shortest Path (SSSP) on GPU

## 1. Project Overview

This project implements the **Single Source Shortest Path (SSSP)** problem on both the CPU and GPU.

The graph is stored in **Compressed Sparse Row (CSR)** format and processed using the **Bellman-Ford algorithm**.

The project demonstrates how a graph algorithm can be converted from a serial CPU implementation into a parallel CUDA implementation.

The implementation includes:

- Graph input from a text file
- Conversion of the graph into CSR format
- Serial Bellman-Ford implementation on CPU
- Parallel Bellman-Ford relaxation on GPU using CUDA
- GPU atomic operations for safe concurrent updates
- GPU double buffering
- CPU vs GPU result verification
- GPU CSR data-transfer verification
- CPU and GPU performance measurement
- GPU kernel timing
- GPU end-to-end timing
- Threads-per-block performance experiments
- Testing with negative edge weights
- Benchmarking on large graphs

---

## 2. Objective

The objective of this tutorial is to implement SSSP using a graph stored in CSR format and compare a serial CPU implementation with a CUDA GPU implementation.

The main goals are:

1. Read a graph from hard disk into CPU memory.
2. Convert the graph into CSR representation.
3. Implement Bellman-Ford on the CPU.
4. Use the CPU implementation as a reference for correctness.
5. Transfer the CSR graph to GPU memory.
6. Implement parallel Bellman-Ford relaxation using CUDA.
7. Use CUDA atomic operations to handle concurrent distance updates.
8. Compare GPU results against CPU results.
9. Measure CPU and GPU execution times.
10. Study the effect of CUDA threads-per-block configuration on performance.

---

## 3. Why Bellman-Ford?

The Bellman-Ford algorithm was selected for this implementation because it supports:

- Positive edge weights
- Zero-weight edges
- Negative edge weights
- Shortest-path computation from a single source

More importantly for this project, the edge relaxation operation can be parallelized.

For every vertex `u`, a CUDA thread processes its outgoing edges stored in CSR format.

The basic relaxation operation is:

```c
if (dist[u] + weight(u,v) < dist[v])
    dist[v] = dist[u] + weight(u,v);
```

On the GPU, multiple threads may attempt to update the same destination vertex. Therefore, the implementation uses CUDA's `atomicMin()` to safely perform concurrent distance updates.

---

## 4. Hardware and CUDA Environment

The GPU implementation was developed and tested on the following system.

### 4.1 GPU

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **GPU Memory:** 6 GB
- **Compute Capability:** 8.6

The RTX 3050 is a CUDA-capable NVIDIA GPU and provides many parallel CUDA cores capable of executing thousands of GPU threads.

The GPU architecture allows the SSSP relaxation work to be distributed across multiple CUDA blocks and warps.

### 4.2 CPU

The CPU implementation runs on the host processor and is used as the reference implementation for correctness verification.

The system processor is:

- **CPU:** 13th Gen Intel Core i5-13450HX
- **Physical Cores:** 10
- **Logical Processors:** 16

The CPU implementation uses a serial Bellman-Ford algorithm, meaning the relaxation work is performed sequentially.

### 4.3 Software Environment

- **Operating System:** Windows 11
- **CUDA Toolkit:** 13.3
- **CUDA Compiler:** nvcc 13.3.73
- **C++ Compiler:** Microsoft Visual C++

The CUDA program is compiled using:

```bash
nvcc sssp.cu -o sssp.exe
```

---

## 5. System Architecture

The overall implementation consists of two execution paths.

```
                    Input Graph
                         |
                         v
                Graph stored in CPU RAM
                         |
                         v
                    Build CSR
                         |
            +------------+------------+
            |                         |
            v                         v
        CPU Path                 GPU Path
            |                         |
            v                         v
  Serial Bellman-Ford          Copy CSR to GPU
            |                         |
            v                         v
 CPU shortest paths          CUDA initialization
                                      |
                                      v
                         Parallel Bellman-Ford
                                      |
                                      v
                              GPU distances
                                      |
                                      v
                          Copy result to CPU
            |                         |
            +------------+------------+
                         |
                         v
                  Compare Results
                         |
                         v
                   PASS / FAIL
```

The CPU implementation therefore acts as the reference implementation, while the GPU implementation is evaluated against it.

---

## 6. CPU vs GPU Computation Model

### CPU

The CPU processes the graph sequentially:

```
Vertex 0
   ↓
Vertex 1
   ↓
Vertex 2
   ↓
...
```

For each vertex, its outgoing edges are relaxed one after another.

### GPU

The GPU distributes vertices among CUDA threads:

```
Thread 0  → Vertex 0
Thread 1  → Vertex 1
Thread 2  → Vertex 2
Thread 3  → Vertex 3
...
Thread N  → Vertex N
```

Each CUDA thread processes the outgoing CSR edges of its assigned vertex.

This allows many vertices to be processed concurrently.

---

## 7. GPU Parallelization Strategy

The CUDA kernel uses:

```c
int u = blockIdx.x * blockDim.x + threadIdx.x;
```

to map a CUDA thread to a graph vertex.

For example, with **Threads Per Block = 256**, the first CUDA block contains threads 0–255 and processes vertices 0–255. The second block processes approximately vertices 256–511, and so on.

The number of blocks is calculated using:

```c
int blocks = (V + threadsPerBlock - 1) / threadsPerBlock;
```

---

## 8. GPU Hardware Parallelism

CUDA organizes execution hierarchically:

```
GPU
 └── Streaming Multiprocessors (SMs)
      └── CUDA Blocks
           └── Warps
                └── 32 Threads
```

A CUDA warp contains **32 threads**. The tested block configurations correspond to:

| Threads/Block | Warps/Block |
|--------------:|------------:|
| 64            | 2           |
| 128           | 4           |
| 256           | 8           |
| 512           | 16          |
| 1024          | 32          |

---

## 9. Threads Per Block Experiment

To investigate how CUDA execution configuration affects performance, the program was tested using different numbers of threads per block.

The following configurations were tested: **64, 128, 256, 512, 1024**.

The benchmark graph was `GraphBenchmark3.txt` with:

- **Vertices =** 25,000
- **Edges =** 149,999

For 25,000 vertices, the corresponding block counts are:

| Threads/Block | Number of Blocks |
|--------------:|-----------------:|
| 64            | 391              |
| 128           | 196              |
| 256           | 98               |
| 512           | 49               |
| 1024          | 25               |

This demonstrates:

```
Threads Per Block ↑
        ↓
Number of Blocks ↓
```

---

## 10. Threads Per Block Performance Results

The observed GPU kernel timings were:

| Threads/Block | Blocks | GPU Kernel Time |
|--------------:|-------:|----------------:|
| 64            | 391    | 8.04138 ms      |
| 128           | 196    | 10.6626 ms      |
| 256           | 98     | 9.05622 ms      |
| 512           | 49     | 6.22086 ms      |
| 1024          | 25     | 7.26346 ms      |

For these particular runs, **512 threads per block produced the lowest observed GPU kernel time**.

However, the result does not imply that 512 threads per block is universally optimal. CUDA performance depends on:

- GPU resource utilization
- Occupancy
- Memory access pattern
- Kernel resource requirements
- Graph topology
- Number of active warps
- Scheduling
- System workload

Therefore, the experiment was used to study the effect of block configuration rather than to claim a universally optimal configuration.

---

## 11. Benchmark Results

The main benchmark graph used for testing was `GraphBenchmark3.txt`.

- **Number of vertices:** 25,000
- **Number of edges:** 149,999
- **Source vertex:** 0

A representative run using 512 threads per block produced:

```
CPU Bellman-Ford Time: 13.939 ms

Threads Per Block: 512
Number of Blocks: 49

GPU Kernel Time: 6.22086 ms

CPU / GPU Kernel Speedup: 2.24069x

GPU End-to-End Time: 20.3031 ms

CPU / GPU End-to-End Speedup: 0.686544x
```

The GPU produced the same shortest-path sample:

| Vertex | Distance |
|-------:|---------:|
| 0      | 0        |
| 1      | 1        |
| 100    | 78       |
| 1000   | 43       |
| 4999   | 14       |

**Verification:**

- CPU vs GPU SSSP: **PASS**
- GPU CSR transfer verification: **PASS**

---

## 12. Interpretation of Benchmark Results

The benchmark demonstrates an important distinction between GPU kernel performance and GPU end-to-end performance.

For the representative 512-thread run:

- **CPU Time** = 13.939 ms
- **GPU Kernel Time** = 6.22086 ms

Therefore:

```
13.939 / 6.22086 ≈ 2.24x
```

The GPU kernel was approximately **2.24× faster** than the measured CPU Bellman-Ford computation for that run.

However, the **GPU End-to-End Time** = 20.3031 ms, which includes GPU-related overhead. Therefore:

```
13.939 / 20.3031 ≈ 0.687x
```

The complete GPU execution was slower than the CPU measurement in this run.

This occurs because GPU execution includes operations beyond kernel computation, including:

- Memory allocation
- Host → Device transfers
- Kernel launches
- Synchronization
- Device → Host transfer

For relatively small or moderately sized graphs, these overheads can become significant compared with the actual computation.

---

## 13. CPU and GPU Iteration Behavior

An important observation from the benchmark is:

- **CPU Bellman-Ford:** 2 iterations
- **GPU Bellman-Ford:** 121 iterations

This difference is caused by the different update models.

The CPU implementation performs **in-place relaxation**. When a distance is updated, a later vertex in the same iteration can immediately use that updated value.

The GPU implementation uses **double buffering** (`d_dist` and `d_new_dist`):

- The current iteration reads from `d_dist` while updates are written into `d_new_dist`.
- The updated buffer becomes the input for the next iteration.

Therefore, shortest-path information may propagate more gradually on the GPU.

This difference does not indicate incorrectness because the final distances are identical.

---

## 14. Correctness Verification

Correctness is checked in two stages.

### 14.1 CPU vs GPU SSSP Verification

The final GPU distances are compared against the CPU distances.

**Expected output:**

```
CPU vs GPU SSSP: PASS
```

This verifies that both implementations produce the same shortest-path results.

### 14.2 CSR Transfer Verification

The CSR arrays transferred from CPU memory to GPU memory are also verified.

The arrays include:

- `row_offsets`
- `col_indices`
- `weights`

**Expected output:**

```
GPU CSR transfer verification: PASS
```

This verifies that the graph representation was transferred correctly to GPU memory.

---

## 15. Negative Edge Weight Testing

Bellman-Ford was also tested using a graph containing negative edge weights.

**Example:**

```
5 6
0 1 4
0 2 2
2 1 -1
1 3 2
2 3 5
3 4 1
```

The expected shortest distances from source 0 are:

| Vertex | Distance |
|-------:|---------:|
| 0      | 0        |
| 1      | 1        |
| 2      | 2        |
| 3      | 3        |
| 4      | 4        |

The CPU and GPU implementations produced matching results.

This test confirms that the implementation can process graphs containing negative edges when no negative cycle is present.

---

## 16. Performance Metrics

The program reports three important timing values.

### CPU Bellman-Ford Time

Time required by the CPU implementation to compute SSSP.

### GPU Kernel Time

Time spent executing the CUDA relaxation kernel.

This is the most direct measurement of the GPU computation itself.

### GPU End-to-End Time

Time associated with the complete GPU execution path, including GPU setup and memory-transfer overhead.

Therefore, `GPU Kernel Time` and `GPU End-to-End Time` should not be interpreted as the same metric.

---

## 17. Speedup Calculation

Speedup is calculated as:

```
Speedup = CPU Time / GPU Time
```

For **kernel speedup**:

```
CPU / GPU Kernel Speedup = CPU Bellman-Ford Time / GPU Kernel Time
```

For **end-to-end speedup**:

```
CPU / GPU End-to-End Speedup = CPU Bellman-Ford Time / GPU End-to-End Time
```

**Interpretation:**

| Speedup | Meaning                  |
|--------:|--------------------------|
| > 1     | GPU is faster            |
| = 1     | Approximately equal      |
| < 1     | CPU is faster            |

---

## 18. Project Structure

```
Tutorial-2-SSSP/
│
├── sssp.cu
│
├── Graph.txt
├── Graph2.txt
├── Graph3.txt
├── Graph4.txt
├── GraphLarge.txt
│
├── NegativeWeightGraph.txt
├── NegativeLargeGraph.txt
│
├── GraphBenchmark1.txt
├── GraphBenchmark2.txt
├── GraphBenchmark3.txt
└── GraphBenchmark4.txt
```

The main CUDA implementation is `sssp.cu`.

---

## 19. Compilation


Compile:

```bash
nvcc sssp.cu -o sssp.exe
```

---

## 20. Running

Run a graph using:

```bash
.\sssp.exe Graph.txt
```

Benchmark:

```bash
.\sssp.exe GraphBenchmark3.txt
```

Negative-edge test:

```bash
.\sssp.exe NegativeWeightGraph.txt
```

---

## 21. Algorithm Complexity

For Bellman-Ford:

- **Time Complexity:** O(V·E)
- **Space Complexity:** O(V + E)

CSR storage requires:

- `row_offsets` → O(V)
- `col_indices` → O(E)
- `weights` → O(E)

Therefore, graph storage is **O(V + E)**.

The GPU implementation parallelizes the relaxation work, but actual performance depends on graph structure and GPU execution characteristics.

---

## 22. Key Technical Concepts Demonstrated

This project demonstrates the following GPU programming concepts:

| Category                 | Concept / Tool                        |
|--------------------------|---------------------------------------|
| Graph Representation     | Compressed Sparse Row (CSR)            |
| Parallel Algorithm       | Parallel Bellman-Ford relaxation      |
| CUDA Thread Mapping      | One CUDA thread → One graph vertex    |
| Atomic Operation         | `atomicMin()`                         |
| GPU Synchronization      | `cudaDeviceSynchronize()`             |
| Double Buffering         | `d_dist`, `d_new_dist`                |
| CUDA Execution Config    | Threads per block, blocks per grid, warps |
| Performance Analysis     | CPU time, GPU kernel time, GPU end-to-end time, speedup |
| Correctness              | CPU vs GPU verification, CSR transfer verification |

---

## 23. Final Conclusion

This project implements and evaluates a CUDA-based solution to the Single Source Shortest Path problem using the Bellman-Ford algorithm and Compressed Sparse Row graph representation.

The CPU implementation provides a reference solution, while the GPU implementation parallelizes vertex-level edge relaxation using CUDA.

The GPU implementation uses `atomicMin()` to safely handle concurrent updates to destination vertices and double buffering to maintain synchronous relaxation iterations.

The implementation was successfully tested on multiple graphs, including graphs containing negative edge weights. CPU and GPU shortest-path results matched in the verification tests.

Performance experiments were also conducted by varying the CUDA threads-per-block configuration from 64 to 1024. For the tested `GraphBenchmark3.txt`, **512 threads per block produced the lowest observed GPU kernel time** among the tested configurations.

The benchmark also demonstrated that **GPU kernel acceleration does not necessarily translate directly into end-to-end application speedup**, because memory transfers, allocation, kernel launches, and synchronization introduce additional overhead.

Overall, the project demonstrates the complete workflow of implementing a graph algorithm on CUDA:

```
Graph Input
    ↓
CSR Construction
    ↓
CPU Reference Implementation
    ↓
GPU Memory Transfer
    ↓
CUDA Parallelization
    ↓
Atomic Relaxation
    ↓
GPU Result
    ↓
CPU/GPU Verification
    ↓
Performance Analysis
```

## 25. Full Experimental Results Table


| Graph              |   Vertices |     Edges | CPU Time | GPU Kernel | GPU E2E | Kernel Speedup | E2E Speedup |
|--------------------|-----------:|----------:|---------:|-----------:|--------:|---------------:|------------:|
| GraphBenchmark1    |      5,000 |    29,999 | 0.633ms  | 4.81978ms  |5.54928ms| 0.131334x      | 0.114069x   |
| GraphBenchmark2    |     10,000 |    59,999 | 1.4007ms | 5.16954ms  |6.17299ms| 0.270953x      | 0.226908x   |
| GraphBenchmark3    |     25,000 |   149,999 | 29.1257ms| 8.1816ms   |10.8588ms| 3.5599x        | 2.68222x    |
| GraphBenchmark4    |     50,000 |   299,999 | 16.9855ms| 18.3292ms  |20.5481ms| 0926692x       | 0.826622x   |

