// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32

//@@ insert code here

__global__ void float2Uchar(float* input, unsigned char* output, int width, int height){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int ii = row * width + col + width * height * blockIdx.z;
  if(col < width && row < height)
    output[ii] = (unsigned char)(255 * input[ii]);
}

__global__ void RGB2Greyscale(unsigned char* input, unsigned char* output, int width, int height){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = row * width + col;

  if(col < width && row < height){
    unsigned char r = input[3*idx];
    unsigned char g = input[3*idx + 1];
    unsigned char b = input[3*idx + 2];
    output[idx] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void histo_compute(unsigned char* input, int* output, int width, int height){
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

  int block_idx = threadIdx.x + threadIdx.y * blockDim.x;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int image_idx = row * width + col;

  if (block_idx < HISTOGRAM_LENGTH)
    histo_private[block_idx] = 0;
  __syncthreads();

  if(col < width && row < height){
    atomicAdd(&(histo_private[input[image_idx]]), 1);
  }
  __syncthreads();

  if (block_idx < HISTOGRAM_LENGTH)
    atomicAdd(&(output[block_idx]), histo_private[block_idx]);
}

__global__ void cdf_compute(int* input, float* output, int image_size){
  __shared__ float T[HISTOGRAM_LENGTH];
  int t = threadIdx.x;
  T[t] = input[2 * blockDim.x * blockIdx.x  + t];
  T[t + blockDim.x] = input[2 * blockDim.x * blockIdx.x + t + blockDim.x];

  int stride = 1;
  while (stride < 2 * HISTOGRAM_LENGTH) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < 2 * blockDim.x && (index - stride) >= 0)
      T[index] += T[index - stride];
    stride = stride * 2;
  }

  stride = blockDim.x / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index + stride) < 2 * blockDim.x)
      T[index + stride] += T[index];
    stride = stride / 2;
  }
  __syncthreads();
  output[2 * (blockDim.x * blockIdx.x) + t] = T[t] / ((float)(image_size));
  output[2 * (blockDim.x * blockIdx.x) + t + blockDim.x] = T[t + blockDim.x] / ((float)(image_size));
}

__global__ void histo_equalization(unsigned char* image, float* cdf, int width, int height){
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int idx = row * width + col + width * height * blockIdx.z;
  float cdfMin = cdf[0];

  if(col < width && row < height)
    image[idx] = min(max(255*(cdf[image[idx]] - cdfMin)/(1.0-cdfMin), 0.0), 255.0);
}

__global__ void uCharToFloat(unsigned char* input, float* output, int width, int height){
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int idx = (blockIdx.z * width * height) + (row * width) + col;
  if(col < width && row < height)
    output[idx] = (float)(input[idx]/255.0);
}

__global__ void uchar2Float(unsigned char* input, float* output, int width, int height){
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int ii = row * width + col + width * height * blockIdx.z;
  if(col < width && row < height)
    output[ii] = (float)(input[ii]/255.0);
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

  float *device_inputImage, *device_outputImage;
  unsigned char *device_ucharImage, *device_grayImage;
  int *device_histogram;
  float *device_cdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int input_image_size = imageWidth * imageHeight;
  cudaMalloc((void**)&device_inputImage, sizeof(float) * input_image_size * imageChannels);
  cudaMalloc((void**)&device_ucharImage, sizeof(unsigned char) * input_image_size * imageChannels);
  cudaMalloc((void**)&device_grayImage, sizeof(unsigned char) * input_image_size);
  cudaMalloc((void**)&device_histogram, sizeof(int) * HISTOGRAM_LENGTH);
  cudaMalloc((void**)&device_cdf, sizeof(float) * HISTOGRAM_LENGTH);
  cudaMalloc((void**)&device_outputImage, sizeof(float) * input_image_size * imageChannels);

  cudaMemcpy(device_inputImage, hostInputImageData, sizeof(float) * input_image_size * imageChannels, cudaMemcpyHostToDevice);
  cudaMemset((void*)device_histogram, 0, sizeof(int) * HISTOGRAM_LENGTH);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 dimGrid_C3(ceil((float)imageWidth/BLOCK_SIZE), ceil((float)imageHeight/BLOCK_SIZE), imageChannels);
  dim3 dimGrid_C1(ceil((float)imageWidth/BLOCK_SIZE), ceil((float)imageHeight/BLOCK_SIZE), 1);
  
  float2Uchar<<<dimGrid_C3, dimBlock>>>(device_inputImage, device_ucharImage, imageWidth, imageHeight);
  RGB2Greyscale<<<dimGrid_C1, dimBlock>>>(device_ucharImage, device_grayImage, imageWidth, imageHeight);
  histo_compute<<<dimGrid_C1, dimBlock>>>(device_grayImage, device_histogram, imageWidth, imageHeight);
  cdf_compute<<<1, HISTOGRAM_LENGTH>>>(device_histogram, device_cdf, input_image_size);
  histo_equalization<<<dimGrid_C3, dimBlock>>>(device_ucharImage, device_cdf, imageWidth, imageHeight);
  uchar2Float<<<dimGrid_C3, dimBlock>>>(device_ucharImage, device_outputImage, imageWidth, imageHeight);

  cudaMemcpy(hostOutputImageData, device_outputImage, sizeof(float) * input_image_size * imageChannels, cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);
  //@@ insert code here
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
