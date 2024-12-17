#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  float Cvalue = 0;
  for (int i=0; i < (numAColumns - 1) / TILE_WIDTH + 1; i++) {
    // Load the matrices from device memory to shared memory
    if (row < numARows && i * TILE_WIDTH + threadIdx.x < numAColumns) {
      ds_A[threadIdx.y][threadIdx.x] = A[row * numAColumns + i * TILE_WIDTH + threadIdx.x];
    } 
    else {
      ds_A[threadIdx.y][threadIdx.x] = 0.0;
    }
    if (i * TILE_WIDTH + threadIdx.y < numBRows && col < numBColumns) {
      ds_B[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * numBColumns + col];
    } 
    else {
      ds_B[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    for (int j=0; j<TILE_WIDTH; j++) {
      Cvalue += ds_A[threadIdx.y][j] * ds_B[j][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < numCRows && col < numCColumns) {
    C[row * numCColumns + col] = Cvalue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;


  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  

  //@@ Allocate GPU memory here
  float *dA;
  float *dB;
  float *dC;
  cudaMalloc((void **)&dA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **)&dB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **)&dC, numCRows * numCColumns * sizeof(float));


  //@@ Copy memory to the GPU here
  cudaMemcpy(dA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);


  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(dA, dB, dC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, dC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  

  //@@ Free the GPU memory here
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);


  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);

  return 0;
}
