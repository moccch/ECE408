#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define MASK_WIDTH 5
#define MASK_RADIUS (MASK_WIDTH / 2)
#define WIDTH 512
#define HEIGHT 512

__global__ void conv2D(float* input, float* output, float* mask, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - MASK_RADIUS;
    int col_i = col_o - MASK_RADIUS;

    __shared__ float N_ds[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

    // Loading input tile into shared memory
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        N_ds[ty][tx] = input[row_i * width + col_i];
    }
    else {
        N_ds[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Only compute for inner tiles, avoiding halos
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        float output_value = 0.0f;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                output_value += mask[i * MASK_WIDTH + j] * N_ds[i + ty][j + tx];
            }
        }

        // Writing the output
        if (row_o < height && col_o < width) {
            output[row_o * width + col_o] = output_value;
        }
    }
}

int main() {
    // Initialize data
    float* h_input = new float[WIDTH * HEIGHT];
    float* h_output = new float[WIDTH * HEIGHT];
    float h_mask[MASK_WIDTH * MASK_WIDTH] = {
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
    };

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_input, *d_output, *d_mask;
    cudaMalloc((void**)&d_input, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&d_output, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&d_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));

    cudaMemcpy(d_input, h_input, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH, (HEIGHT + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);

    conv2D<<<dimGrid, dimBlock>>>(d_input, d_output, d_mask, WIDTH, HEIGHT);

    cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

    // Now h_output contains the convolution result, you can print it or write it to a file if needed

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);

    return 0;
}
