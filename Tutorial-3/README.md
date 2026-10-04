# Tutorial 3 – Tensor Core Matrix Multiplication

## Objective

Perform matrix multiplication of two **64×64 matrices** using NVIDIA Tensor Cores with **16×16 tiles**.

## Implementation

- Matrix size: `64 × 64`
- Tensor Core tile size: `16 × 16`
- Output tiles: `4 × 4 = 16`
- One warp computes one `16 × 16` output tile
- Total warps: `16`
- K dimension: `64`
- MMA operations per output tile: `64 / 16 = 4`
- WMMA API is used for Tensor Core matrix multiplication.
- Accumulator type: `float`
- Input matrices: `half`

## GPU Used

The program was tested on a **NVIDIA Tesla T4 (Compute Capability 7.5)** using CUDA 13.0. in Google Collab 

Compilation:

```bash
nvcc -arch=sm_75 gem-tc.cu -o gem-tc