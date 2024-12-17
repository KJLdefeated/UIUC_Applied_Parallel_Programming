#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

using namespace nvcuda;

#define TILE_WIDTH 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define N_STREAMS 2

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
    int b_x = blockIdx.x;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int b_y = blockIdx.y;
    int b_z = blockIdx.z;
    
    size_t row = t_x + b_x * TILE_WIDTH;
    size_t col = b_z * (Height_out * Width_out) + t_y + b_y * TILE_WIDTH;

    size_t b = col / (Height_out * Width_out);
    size_t c = (row / (K * K));
    size_t i = (col % (Width_out * Height_out)) / Width_out;
    size_t j = (col % (Width_out * Height_out)) % Width_out;
    size_t i_off = (row % (K * K)) / K;
    size_t j_off = (row % (K * K)) % K;

    if (row < (size_t) Channel * K * K && col < (size_t) Batch * Height_out * Width_out) {
        output[row * ((size_t) Batch * Height_out * Width_out) + col] = in_4d(b, c, i + i_off, j + j_off);
    }

    #undef in_4d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{	
	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
	
	wmma::fill_fragment(c_frag, 0.0f);

	// Shared memory for the tiles
    __shared__ half As[TILE_WIDTH*TILE_WIDTH];
    __shared__ half Bs[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Cs[TILE_WIDTH*TILE_WIDTH];

    // Calculate thread block position
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate the starting positions
    int baseM = by * WMMA_M;
    int baseN = bx * WMMA_M;

    // Loop over tiles
    for (int tile_idx = 0; tile_idx < (numAColumns + TILE_WIDTH - 1)/TILE_WIDTH; tile_idx++) {
        // Load tiles into shared memory
        int k = tile_idx * WMMA_K;
        if (baseM + ty < numARows && k + tx < numAColumns) {
            As[ty*TILE_WIDTH + tx] = __float2half(A[size_t (baseM + ty) * numAColumns + k + tx]);
        } else {
            As[ty*TILE_WIDTH + tx] = __float2half(0.0f);
        }
        if (baseN + tx < numBColumns && k + ty < numBRows) {
            Bs[tx * TILE_WIDTH + ty] = __float2half(B[size_t (k + ty) * numBColumns + baseN + tx]);
        } else {
            Bs[tx * TILE_WIDTH + ty] = __float2half(0.0f);
        }
        
        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Load the matrices from shared memory into fragments
        wmma::load_matrix_sync(a_frag, (half *)As, TILE_WIDTH);
        wmma::load_matrix_sync(b_frag, (half *)Bs, TILE_WIDTH);

        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Store the result
    int c = baseM + ty;
    int d = baseN + tx;
    wmma::store_matrix_sync(Cs, c_frag, WMMA_M, wmma::mem_row_major);
    if (c < numCRows && d < numCColumns) {
        C[(size_t) c * numCColumns + d] = Cs[ty*TILE_WIDTH + tx];
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    
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
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    dim3 unrolling_kernel_grid_dim((Channel * K * K - 1) / TILE_WIDTH + 1, (Height_out * Width_out - 1) / TILE_WIDTH + 1, Batch);
    matrix_unrolling_kernel<<<unrolling_kernel_grid_dim, dim3(TILE_WIDTH, TILE_WIDTH, 1)>>>(
        device_input, unrolled_matrix, Batch, Channel, Height, Width, K
    );

    // TODO: Set the kernel dimensions and call the matmul kernel
    dim3 matmul_kernel_grid_dim((Width_unrolled - 1) / TILE_WIDTH + 1, (Map_out - 1) / TILE_WIDTH + 1, 1);
    matrixMultiplyShared<<<matmul_kernel_grid_dim, dim3(TILE_WIDTH, TILE_WIDTH, 1)>>>(
        device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled
    );

    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
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