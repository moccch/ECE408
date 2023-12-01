#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"

#define TILE_WIDTH 16

__constant__ float MASK[4096];

__global__ void conv_forward_kernel(half *output, const half *input, const half *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    extern __shared__ half SM[];
    int shared_width = S * (TILE_WIDTH - 1) + K;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) MASK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define sm_3d(i2, i1, i0) SM[(i2) * (shared_width * shared_width) + (i1) * shared_width + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int W_grid = ceil((float)(W_out)/TILE_WIDTH); 
    int H_grid = ceil((float)(H_out)/TILE_WIDTH); 
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + tx;
    half temp = 0.0;
    int left_h = h - ty;
    int left_w = w - tx;

    for (int c = 0; c < C; c++){
        for(int i = ty; i < shared_width; i += TILE_WIDTH){
            for(int j = tx; j < shared_width; j += TILE_WIDTH){
                if (left_h * S + i < H && left_w * S + j < W)
                    sm_3d(c, i, j) = in_4d(b, c, S * left_h + i, S * left_w + j);
                else
                    sm_3d(c, i, j) = 0.0;    
            }
        }
    }
    __syncthreads();

    if(h < H_out && w < W_out) {
        for(int c = 0; c < C; c++){
            for(int p = 0; p < K; p++){
                for(int q = 0; q < K; q++) {
                    // temp += sm_3d(c, ty * S + p, tx * S + q) * mask_4d(m, c, p, q);  
                    temp = __hadd(temp, __hmul(sm_3d(c, ty * S + p, tx * S + q), __float2half(mask_4d(m, c, p, q))));              
                }	
            }
        }
        out_4d(b, m, h, w) = temp;
    }


    // if ((h < H_out) && (w < W_out)){
    //   half temp = 0.0;
    //   for (int c = 0; c < C; c++){
    //     for (int p = 0; p < K; p++){
    //       for (int q = 0; q < K; q++){
    //         temp = __hadd(temp, __hmul(in_4d(b, c, S * h + p, S * w + q), __float2half(mask_4d(m, c, p, q))));
    //       }
    //     }
    //   }
    //   out_4d(b, m, h, w) = temp;
    // }  

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void float2half(const float* in_float, half* out_half, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < size; i += TILE_WIDTH * 1024){
    out_half[i] = __float2half(in_float[i]);
  }
}

__global__ void half2float(float* out_float, half* in_half, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < size; i += TILE_WIDTH * 1024){
    out_float[i] = __half2float(in_half[i]);
  }
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    cudaMalloc((void**)device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void**)device_output_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, M * C * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(MASK, host_mask, M * C * K * K * sizeof(float));
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int W_grid = ceil((float)(W_out)/TILE_WIDTH); 
    int H_grid = ceil((float)(H_out)/TILE_WIDTH); 
    int Y = H_grid * W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // output tile for untiled code
    dim3 gridDim(B, M, Y);

    int size_Input = H * W * B * C;
    int size_Kernel = K * K * M * C;
    int size_Output = H_out * W_out * B * M;

    half* half_device_input;
    half* half_device_output;
    half* half_device_mask;

    cudaMalloc(&half_device_input, H * W * B * C * sizeof(half));
    cudaMalloc(&half_device_mask, K * K * M * C * sizeof(half));
    cudaMalloc(&half_device_output, H_out * W_out * B * M * sizeof(half));

    dim3 gridHalf(TILE_WIDTH, 1, 1);
    dim3 blockHalf(1024, 1, 1);

    float2half <<< gridHalf, blockHalf >>> (device_input, half_device_input, H * W * B * C);
    cudaDeviceSynchronize();
    float2half <<< gridHalf, blockHalf >>> (device_mask, half_device_mask, K * K * M * C);
    cudaDeviceSynchronize();


    conv_forward_kernel <<< gridDim, blockDim >>> (half_device_output, half_device_input, half_device_mask, B, M, C, H, W, K, S);
    cudaDeviceSynchronize();

    half2float <<< gridHalf, blockHalf >>> (device_output, half_device_output, size_Output);
    cudaDeviceSynchronize();

    cudaFree(half_device_input);
    cudaFree(half_device_mask);
    cudaFree(half_device_output);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
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
