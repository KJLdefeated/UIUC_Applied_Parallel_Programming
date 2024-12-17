#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define MASK_RADIUS MASK_WIDTH / 2
#define TILE_DIM_X 8
#define TILE_DIM_Y 8
#define TILE_DIM_Z 8

//@@ Define constant memory for device kernel here
__constant__ float deviceMask[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int x = blockIdx.x * TILE_DIM_X + threadIdx.x;
  int y = blockIdx.y * TILE_DIM_Y + threadIdx.y;
  int z = blockIdx.z * TILE_DIM_Z + threadIdx.z;

  __shared__ float N_ds[TILE_DIM_Z + MASK_WIDTH - 1][TILE_DIM_Y + MASK_WIDTH - 1][TILE_DIM_X + MASK_WIDTH - 1];

  for(int tz = threadIdx.z; tz < TILE_DIM_Z + MASK_WIDTH - 1; tz++){
    for(int ty = threadIdx.y; ty < TILE_DIM_Y + MASK_WIDTH - 1; ty++){
      for(int tx = threadIdx.x; tx < TILE_DIM_X + MASK_WIDTH - 1; tx++){
        int x_in = x + tx - MASK_RADIUS;
        int y_in = y + ty - MASK_RADIUS;
        int z_in = z + tz - MASK_RADIUS;
        if(x_in >= 0 && x_in < x_size && y_in >= 0 && y_in < y_size && z_in >= 0 && z_in < z_size){
          N_ds[tz][ty][tx] = input[z_in * y_size * x_size + y_in * x_size + x_in];
        } else {
          N_ds[tz][ty][tx] = 0.0f;
        }
      }
    }
  }

  __syncthreads();

  if (x < x_size && y < y_size && z < z_size) {
    float Pvalue = 0.0f;
    for (int i = 0; i < MASK_WIDTH; i++) {
      for (int j = 0; j < MASK_WIDTH; j++) {
        for (int k = 0; k < MASK_WIDTH; k++) {
          Pvalue += deviceMask[i][j][k] * N_ds[threadIdx.z + i][threadIdx.y + j][threadIdx.x + k];
        }
      }
    }
    output[z * y_size * x_size + y * x_size + x] = Pvalue;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float));


  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceMask, hostKernel, kernelLength * sizeof(float));



  //@@ Initialize grid and block dimensions here
  dim3 gridDim((x_size + TILE_DIM_X - 1)/TILE_DIM_X,
               (y_size + TILE_DIM_Y - 1)/TILE_DIM_Y,
               (z_size + TILE_DIM_Z - 1)/TILE_DIM_Z);
  dim3 blockDim(TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z);

  //@@ Launch the GPU kernel here
  conv3d<<<gridDim, blockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

