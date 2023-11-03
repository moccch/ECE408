#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void MatrixMulKernel(float *M, float *N, float *P, int widthM, int widthN, int heightM, int heightN);

int main()
{
	const int widthM = 2;
	const int heightM = 3;
	const int widthN = 4;
	const int heightN = 2;

	if (widthM != heightN)
	{
		std::cerr << "Matrix dimensions do not match for multiplication" << std::endl;
		return -1;
	}

	const int sizeM = widthM * heightM * sizeof(float);
	const int sizeN = widthN * heightN * sizeof(float);
	const int sizeP = heightM * widthN * sizeof(float);

	float h_M[6] = {1, 2, 3, 4, 5, 6};						// 3x2 matrix
	float h_N[8] = {7, 8, 9, 10, 11, 12, 13, 14}; // 2x4 matrix
	float *h_P = new float[heightM * widthN];
	std::cout << "Matrix M:" << std::endl;
	for (int i = 0; i < heightM; ++i)
	{
		for (int j = 0; j < widthM; ++j)
		{
			std::cout << h_M[i * widthM + j] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << "Matrix N:" << std::endl;
	for (int i = 0; i < heightN; ++i)
	{
		for (int j = 0; j < widthN; ++j)
		{
			std::cout << h_N[i * widthN + j] << " ";
		}
		std::cout << std::endl;
	}

	float *d_M, *d_N, *d_P;
	cudaMalloc((void **)&d_M, sizeM);
	cudaMalloc((void **)&d_N, sizeN);
	cudaMalloc((void **)&d_P, sizeP);

	cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice);

	dim3 dimGrid((widthN + TILE_WIDTH - 1) / TILE_WIDTH, (heightM + TILE_WIDTH - 1) / TILE_WIDTH);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, widthM, widthN, heightM, heightN);

	cudaMemcpy(h_P, d_P, sizeP, cudaMemcpyDeviceToHost);

	// Print the result (or do something else with it)
	std::cout << "Matrix Result:" << std::endl;
	for (int i = 0; i < heightM; ++i)
	{
		for (int j = 0; j < widthN; ++j)
		{
			std::cout << h_P[i * widthN + j] << " ";
		}
		std::cout << std::endl;
	}

	delete[] h_P;

	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);

	return 0;
}

__global__ void MatrixMulKernel(float *M, float *N, float *P, int widthM, int widthN, int heightM, int heightN)
{
	__shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;

	for (int q = 0; q < (widthM + TILE_WIDTH - 1) / TILE_WIDTH; ++q)
	{
		if (Row < heightM && q * TILE_WIDTH + tx < widthM)
		{
			subTileM[ty][tx] = M[Row * widthM + q * TILE_WIDTH + tx];
		}
		else
		{
			subTileM[ty][tx] = 0.0;
		}

		if (q * TILE_WIDTH + ty < heightN && Col < widthN)
		{
			subTileN[ty][tx] = N[(q * TILE_WIDTH + ty) * widthN + Col];
		}
		else
		{
			subTileN[ty][tx] = 0.0;
		}

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k)
		{
			Pvalue += subTileM[ty][k] * subTileN[k][tx];
		}

		__syncthreads();
	}

	if (Row < heightM && Col < widthN)
	{
		P[Row * widthN + Col] = Pvalue;
	}
}
