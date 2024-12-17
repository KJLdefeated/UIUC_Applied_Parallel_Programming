#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define N_STREAMS 4

__global__ void matrix_multiplication_with_built_in_unrolling(const float *device_input,
                                        const float* device_mask, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K, const int Map_out) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    __shared__ half2 As[TILE_WIDTH][TILE_WIDTH];
    __shared__ half2 Bs[TILE_WIDTH][TILE_WIDTH];

    #define in_4d(i3, i2, i1, i0) device_input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;
    
    int baseM = by * TILE_WIDTH;
    int baseN = bx * TILE_WIDTH;  

    int numARows = Map_out;
    int numAColumns = Channel * K * K;
    int numBRows = Channel * K * K;
    int numBColumns = Batch * Height_out * Width_out;
    int numCRows = Map_out;
    int numCColumns = Batch * Height_out * Width_out;
    half2 val = __float2half2_rn(0.0f);

    for (int tile_idx = 0; tile_idx < (numAColumns + TILE_WIDTH - 1)/TILE_WIDTH; tile_idx++) {
        // Load tiles into shared memory
        int k = tile_idx * TILE_WIDTH;
        if (baseM + ty < numARows && k + tx < numAColumns) {
            As[ty][tx] = __float2half2_rn(device_mask[size_t (baseM + ty) * numAColumns + k + tx]);
        } else {
            As[ty][tx] = __float2half2_rn(0.0f);
        }
        if (baseN + tx < numBColumns && k + ty < numBRows) {
            size_t row = k + ty;
            size_t col = baseN + tx;
            size_t b = col / (Height_out * Width_out);
            size_t c = (row / (K * K));
            size_t i = (col % (Width_out * Height_out)) / Width_out;
            size_t j = (col % (Width_out * Height_out)) % Width_out;
            size_t i_off = (row % (K * K)) / K;
            size_t j_off = (row % (K * K)) % K;
            Bs[ty][tx] = __float2half2_rn(in_4d(b, c, i + i_off, j + j_off));
        } else {
            Bs[ty][tx] = __float2half2_rn(0.0f);
        }
        
        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        if (by * TILE_WIDTH + ty  < numCRows && bx * TILE_WIDTH + tx < numCColumns) {
            // for (int i = 0; i < TILE_WIDTH; i++) {
            //     val += As[ty][i] * Bs[i][tx];
            // }
            val += As[ty][0] * Bs[0][tx];
            val += As[ty][1] * Bs[1][tx];
            val += As[ty][2] * Bs[2][tx];
            val += As[ty][3] * Bs[3][tx];
            val += As[ty][4] * Bs[4][tx];
            val += As[ty][5] * Bs[5][tx];
            val += As[ty][6] * Bs[6][tx];
            val += As[ty][7] * Bs[7][tx];
            val += As[ty][8] * Bs[8][tx];
            val += As[ty][9] * Bs[9][tx];
            val += As[ty][10] * Bs[10][tx];
            val += As[ty][11] * Bs[11][tx];
            val += As[ty][12] * Bs[12][tx];
            val += As[ty][13] * Bs[13][tx];
            val += As[ty][14] * Bs[14][tx];
            val += As[ty][15] * Bs[15][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    int row = baseM + ty;
    int col = baseN + tx;

    if (row < numCRows && col < numCColumns) {
        int m = (row * numCColumns + col) / (Batch * Height_out * Width_out);
        int x = (row * numCColumns + col) % (Height_out * Width_out);
        int b = (row * numCColumns + col) % (Batch * Height_out * Width_out) / (Height_out * Width_out);
        output[(size_t)(b * Map_out * Width_out * Height_out + m * Width_out * Height_out + x)] = __low2float(val);
    }

    #undef in_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    cudaMalloc((void**)device_output_ptr, (size_t) Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));
    cudaMalloc((void**)device_input_ptr, (size_t) Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, (size_t) Map_out * Channel * K * K * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, (size_t) Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, (size_t) Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    dim3 kernel_fuse_grid_dim((Batch * Height_out * Width_out - 1) / TILE_WIDTH + 1, (Map_out - 1) / TILE_WIDTH + 1, 1);
    matrix_multiplication_with_built_in_unrolling<<<kernel_fuse_grid_dim, dim3(TILE_WIDTH, TILE_WIDTH, 1)>>>(
        device_input, device_mask, device_output, Batch, Channel, Height, Width, K, Map_out
    );
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}