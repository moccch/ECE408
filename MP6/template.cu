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

__global__ void scan(float *input, float *output, int len, float *aux_array, int aux_enabled) {
    // Allocate shared memory array of size 2*BLOCK_SIZE
    __shared__ float T[BLOCK_SIZE * 2];
    int t = threadIdx.x;
    if (2 * (blockDim.x * blockIdx.x) + t < len) T[t] = input[2 * blockDim.x * blockIdx.x  + t];
    if (2 * (blockDim.x * blockIdx.x) + t + BLOCK_SIZE < len) T[t + BLOCK_SIZE] = input[2 * blockDim.x * blockIdx.x + t + BLOCK_SIZE];

    int stride = 1;
    while (stride < 2 * BLOCK_SIZE) {
      __syncthreads();
      int index = (threadIdx.x + 1) * stride * 2 - 1;
      if (index < 2 * BLOCK_SIZE && (index - stride) >= 0)
        T[index] += T[index - stride];
      stride = stride * 2;
    }

    stride = BLOCK_SIZE / 2;
    while (stride > 0) {
      __syncthreads();
      int index = (threadIdx.x + 1) * stride * 2 - 1;
      if ((index + stride) < 2 * BLOCK_SIZE)
        T[index + stride] += T[index];
      stride = stride / 2;
    }
    __syncthreads();
    if (2 * blockDim.x * blockIdx.x + t < len) output[2 * (blockDim.x * blockIdx.x) + t] = T[t];
    if (2 * (blockDim.x * blockIdx.x) + t + BLOCK_SIZE < len) output[2 * (blockDim.x * blockIdx.x) + t + BLOCK_SIZE] = T[t + BLOCK_SIZE];

    if (aux_enabled == 1 && t == BLOCK_SIZE - 1) aux_array[blockIdx.x] = T[(BLOCK_SIZE * 2) - 1];
}

__global__ void add(float *input, float *output, int len, float *aux_array) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len - blockDim.x) {
    output[i + blockDim.x] = input[i + blockDim.x] + aux_array[blockIdx.x];
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

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  float *aux_array;
  int num_blocks = (numElements - 1 + BLOCK_SIZE << 1) / (BLOCK_SIZE << 1);
  cudaMalloc((void **)&aux_array, num_blocks * sizeof(float));
  scan<<<num_blocks, BLOCK_SIZE>>>(deviceInput, deviceOutput, numElements, aux_array, 1);
  scan<<<1, BLOCK_SIZE>>>(aux_array, aux_array, num_blocks, aux_array, 0);
  add<<<num_blocks - 1, BLOCK_SIZE * 2>>>(deviceOutput, deviceOutput, numElements, aux_array);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
