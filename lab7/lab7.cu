// Histogram Equalization

#include <wb.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 128
#define TILE_SIZE 16

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

//@@ insert code here
__global__ void floar2uchar(float *input, unsigned char *output, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = threadIdx.z;

  if (x < width && y < height) {
    int idx = (y * width + x) * channels + z;
    output[idx] = (unsigned char)(255 * input[idx]);
  }
}

__global__ void rgb2gray(unsigned char *input, unsigned char *output, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = (y * width + x) * channels;
    output[y * width + x] = (unsigned char)(0.21 * input[idx] + 0.71 * input[idx + 1] + 0.07 * input[idx + 2]);
  }
}

__global__ void histogramming(unsigned int* output, unsigned char* input, int width, int height) {
  __shared__ unsigned int histo[HISTOGRAM_LENGTH];
  int index = threadIdx.x + threadIdx.y * blockDim.x;
  if (index < HISTOGRAM_LENGTH) {
    histo[index] = 0;
  }
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < width && j < height) {
    atomicAdd(&(histo[input[j * width + i]]), 1);
  }
  __syncthreads();
  if (index < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[index]), histo[index]);
  }
}

__global__ void scanCDF(float* output, unsigned int* input, int size) {
  __shared__ float temp[HISTOGRAM_LENGTH];
  int index = threadIdx.x;
  int start = 2 * blockIdx.x * blockDim.x;
  temp[2 * index] = (float) input[start + 2 * index] / size;
  temp[2 * index + 1] = (float) input[start + 2 * index + 1] / size;
  int stride = 1;
  while (stride < HISTOGRAM_LENGTH) {
    __syncthreads();
    int idx = (index + 1) * stride * 2 - 1;
    if (idx < HISTOGRAM_LENGTH && idx - stride >= 0) {
      temp[idx] += temp[idx - stride];
    }
    stride = stride * 2;
  }
  stride = HISTOGRAM_LENGTH / 4;
  while (stride > 0) {
    __syncthreads();
    int idx = (index + 1) * stride * 2 - 1;
    if (idx + stride < HISTOGRAM_LENGTH) {
      temp[idx + stride] += temp[idx];
    }
    stride = stride / 2;
  }
  __syncthreads();
  output[start + 2 * index] = temp[2 * index];
  output[start + 2 * index + 1] = temp[2 * index + 1];
}

__global__ void correctcolor(unsigned char* output, float* input, unsigned char* uchar, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = threadIdx.z;
  if (x < width && y < height) {
    int index = (y * width + x) * channels + z;
    int val = (int)uchar[index];
    output[index] = min(max(255.0f*(input[val] - input[0])/(1.0f - input[0]), 0.0f), 255.0f);
  }
}

__global__ void uchar2float(float* output, unsigned char* input, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = threadIdx.z;
  if (x < width && y < height) {
    int idx = (y * width + x) * channels + z;
    output[idx] = (float)input[idx] / 255.0;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char *deviceUcharImageData;
  unsigned char *deviceGrayImageData;
  unsigned char *deviceCorrectedColorImageData;
  unsigned int *deviceHistogram;
  float *deviceCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);


  //@@ insert code here
  int totalSize = imageWidth * imageHeight * imageChannels * sizeof(float);
  int pixelSize = imageWidth * imageHeight * sizeof(unsigned char);
  cudaMalloc((void **)&deviceInputImageData, totalSize);
  cudaMalloc((void **)&deviceOutputImageData, totalSize);
  cudaMalloc((void **)&deviceUcharImageData, pixelSize * imageChannels);
  cudaMalloc((void **)&deviceGrayImageData, pixelSize);
  cudaMalloc((void **)&deviceCorrectedColorImageData, pixelSize * imageChannels);
  cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));

  cudaMemcpy(deviceInputImageData, hostInputImageData, totalSize, cudaMemcpyHostToDevice);

  // Kernel Dimension
  dim3 gridconversion((imageWidth - 1) / TILE_SIZE + 1, (imageHeight - 1) / TILE_SIZE + 1, 1);
  dim3 blockconversion(TILE_SIZE, TILE_SIZE, imageChannels);
  dim3 gridscan(1,1,1);
  dim3 blockscan(HISTOGRAM_LENGTH, 1, 1);
  dim3 gridDim((imageWidth - 1) / TILE_SIZE + 1, (imageHeight - 1) / TILE_SIZE + 1, 1);
  dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);

  // launch kernel
  floar2uchar<<<gridconversion, blockconversion>>>(deviceInputImageData, deviceUcharImageData, imageWidth, imageHeight, imageChannels);
  checkCUDAErrorWithLine("floar2uchar failed");
  rgb2gray<<<gridDim, blockDim>>>(deviceUcharImageData, deviceGrayImageData, imageWidth, imageHeight, imageChannels);
  checkCUDAErrorWithLine("rgb2gray failed");
  histogramming<<<gridDim, blockDim>>>(deviceHistogram, deviceGrayImageData, imageWidth, imageHeight);
  checkCUDAErrorWithLine("histogramming failed");
  
  // int *host_hists;
  // host_hists = (int *)malloc(HISTOGRAM_LENGTH * sizeof(int));
  // cudaMemcpy(host_hists, deviceHistogram, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < HISTOGRAM_LENGTH; i++)
  //   std::cout << "hist[" << i << "]: " << host_hists[i] << std::endl;
  
  scanCDF<<<gridscan, blockscan>>>(deviceCDF, deviceHistogram, imageWidth * imageHeight);
  checkCUDAErrorWithLine("scanCDF failed");

  correctcolor<<<gridconversion, blockconversion>>>(deviceCorrectedColorImageData, deviceCDF, deviceUcharImageData, imageWidth, imageHeight, imageChannels);
  checkCUDAErrorWithLine("correctcolor failed");

  uchar2float<<<gridconversion, blockconversion>>>(deviceOutputImageData, deviceCorrectedColorImageData, imageWidth, imageHeight, imageChannels);
  checkCUDAErrorWithLine("uchar2float failed");

  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, totalSize, cudaMemcpyDeviceToHost);

  // // output debugging
  // for (int i = 0; i < totalSize / sizeof(float); i++) {
  //   std::cout << "hostOutputImageData[" << i << "]: " << hostOutputImageData[i] << std::endl;
  // }

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceUcharImageData);
  cudaFree(deviceGrayImageData);
  cudaFree(deviceCorrectedColorImageData);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);

  return 0;
}

