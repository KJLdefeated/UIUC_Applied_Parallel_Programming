// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *block_sums, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[2 * BLOCK_SIZE];
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  XY[threadIdx.x] = (i < len) ? input[i] : 0;
  XY[threadIdx.x + blockDim.x] = (i + blockDim.x < len) ? input[i + blockDim.x] : 0;
  
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE) {
      XY[index] += XY[index - stride];
    }
    stride *= 2;
  }
  
  if (threadIdx.x == 0 && block_sums != NULL) {
    block_sums[blockIdx.x] = XY[2 * BLOCK_SIZE - 1];
  }
  if (threadIdx.x == 0) {
    XY[2 * BLOCK_SIZE - 1] = 0;
  }

  stride = BLOCK_SIZE;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE) {
      float temp = XY[index];
      XY[index] += XY[index - stride];
      XY[index - stride] = temp;
    }
    stride /= 2;
  }

  __syncthreads();
  if (i < len) {
    output[i] = XY[threadIdx.x];
  }
  if (i + blockDim.x < len) {
    output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
  }
}

__global__ void add_block_sums(float *output, float *block_sums, int len) {
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (blockIdx.x > 0 && i < len) {
    output[i] += block_sums[blockIdx.x - 1];
  }
  if (blockIdx.x > 0 && i + blockDim.x < len) {
    output[i + blockDim.x] += block_sums[blockIdx.x - 1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, (numElements+1) * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, (numElements+1) * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  int numBlocks = (numElements + 1 + (2 * BLOCK_SIZE - 1)) / (2 * BLOCK_SIZE);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  float *deviceBlockSums;
  float *deviceBlockSumsOutput;
  cudaMalloc((void **)&deviceBlockSums, BLOCK_SIZE * 2 * sizeof(float));
  cudaMalloc((void **)&deviceBlockSumsOutput, BLOCK_SIZE * 2 * sizeof(float));
  scan<<<numBlocks, BLOCK_SIZE>>>(deviceInput, deviceOutput, deviceBlockSums, numElements);

  if(numBlocks > 1) {
    scan<<<1, BLOCK_SIZE>>>(deviceBlockSums, deviceBlockSumsOutput, NULL, numBlocks);
    cudaDeviceSynchronize();
    add_block_sums<<<numBlocks, BLOCK_SIZE>>>(deviceOutput, deviceBlockSumsOutput, numElements);
    cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();

  cudaFree(deviceBlockSumsOutput);
  cudaFree(deviceBlockSums);

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

