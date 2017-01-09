
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> // rand
#include <time.h>

#define WIDTH 128
#define TILE_WIDTH 32


cudaError_t multiplyWithCuda(int *c, int *a, int *b, unsigned int size, double &tt);
int multiplyWithCPU(int *c, int *a, int *b, unsigned int w);
bool checkResult(int *c2, int *c1, unsigned int w);
void print_results(int *m);
void error_handeling(int *d_a, int *d_b, int *d_c);

__global__ void multiplyKernel(int *c, int *a, int *b)
{
	__shared__ int shared_a[TILE_WIDTH*TILE_WIDTH];
	__shared__ int shared_b[TILE_WIDTH*TILE_WIDTH];

	// Target position in c to be filled
	int row = blockIdx.y*TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	int n_phases = WIDTH / TILE_WIDTH;
	int r_load, c_load;
	int temp = 0;
	
	for (int k = 0; k < n_phases; k++)
	{
		r_load = row;
		c_load = k*TILE_WIDTH + threadIdx.x;
		shared_a[threadIdx.y*TILE_WIDTH + threadIdx.x] = a[r_load*WIDTH + c_load];
		r_load = k*TILE_WIDTH + threadIdx.y;
		c_load = col;
		shared_b[threadIdx.y*TILE_WIDTH + threadIdx.x] = b[r_load*WIDTH + c_load];
		__syncthreads();

		for (int j = 0; j < TILE_WIDTH; j++)
		{
			temp += shared_a[threadIdx.y*TILE_WIDTH + j] * shared_b[j*TILE_WIDTH + threadIdx.x];
			//temp += 1;
		}
		__syncthreads();
	}
	
	c[row*WIDTH + col] = temp;
	
}

int main()
{
	srand(time(NULL));

	int a[WIDTH*WIDTH];
	int b[WIDTH*WIDTH];

	// Allocate output matrix
	int c1[WIDTH*WIDTH];
	int c2[WIDTH*WIDTH];

	int iter = 100;
	clock_t begin, end;

	// == CUDA version ==
	begin = clock();
	double gpu_comput_time = 0;
	for (int i = 0; i < iter; i++)
	{
		// Initialize input matrices
		for (int j = 0; j < WIDTH*WIDTH; j++)
		{
			a[j] = rand() % 30; // i%30
			b[j] = 15 - rand() % 30; // 15 - i % 30
		}
		double tt;
		cudaError_t cudaStatus = multiplyWithCuda(c1, a, b, WIDTH, tt);
		gpu_comput_time += tt;
	}
	end = clock();
	double time_spent_gpu = (double)(end - begin) / CLOCKS_PER_SEC;
	/*
	// == CPU version ==
	begin = clock();
	for (int i = 0; i < iter; i++)
	{
		for (int j = 0; j < WIDTH*WIDTH; j++)
		{
			a[j] = rand() % 30; // i%30
			b[j] = 15 - rand() % 30; // 15 - i % 30
		}
		int cpuStatus = multiplyWithCPU(c2, a, b, WIDTH);
	}
	end = clock();
	double time_spent_cpu = (double)(end - begin) / CLOCKS_PER_SEC;

	// == Check results ==
	bool pass = checkResult(c2, c1, WIDTH);
	if (pass)
		printf("Result: PASS\n");
	else
		printf("Result: FAIL\n");
	*/

	printf("GPU compute time = %d usec\n", int(gpu_comput_time * 1e6));
	printf("GPU wall time = %d usec\n", int(time_spent_gpu * 1e6));
	//printf("CPU wall time = %d usec\n", int(time_spent_cpu * 1e6));

	//print_results(a);
	//print_results(b);

    return 0;
}

cudaError_t multiplyWithCuda(int *c, int *a, int *b, unsigned int w, double &tt)
{
	int sz = w*w*sizeof(int);
	int *d_a = 0;
	int *d_b = 0;
	int *d_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		error_handeling(d_a, d_b, d_c);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&d_a, sz);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		error_handeling(d_a, d_b, d_c);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&d_b, sz);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		error_handeling(d_a, d_b, d_c);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&d_c, sz);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		error_handeling(d_a, d_b, d_c);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_a, a, sz, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyHostToDevice failed!");
		error_handeling(d_a, d_b, d_c);
		return cudaStatus;
	}
	
	cudaStatus = cudaMemcpy(d_b, b, sz, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyHostToDevice failed!");
		error_handeling(d_a, d_b, d_c);
		return cudaStatus;
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 numBlocks(WIDTH/threadsPerBlock.x, WIDTH/threadsPerBlock.y);
	clock_t begin, end;
	begin = clock();
	multiplyKernel <<<numBlocks, threadsPerBlock >>> (d_c, d_a, d_b);
	end = clock();
	tt = (double)(end - begin) / CLOCKS_PER_SEC;

	cudaStatus = cudaMemcpy(c, d_c, sz, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
		error_handeling(d_a, d_b, d_c);
		return cudaStatus;
	}

	return cudaStatus;

}

void error_handeling(int *d_a, int *d_b, int *d_c)
{
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return;
}

int multiplyWithCPU(int *c, int *a, int *b, unsigned int w)
{
	for (unsigned int row = 0; row < w; row++)
	{
		for (unsigned int col = 0; col < w; col++)
		{
			int temp = 0;
			for (unsigned int k = 0; k < w; k++)
				temp += a[row*w + k] * b[k*w + col];
			
			c[row*w + col] = temp;
		}
	}
	return 0;
}

bool checkResult(int *c2, int *c1, unsigned int w)
{
	bool pass = true;
	for (unsigned int row = 0; row < w; row++)
	{
		for (unsigned int col = 0; col < w; col++)
		{
			if (c1[row*w + col] != c2[row*w + col])
			{
				pass = false;
				return pass;
			}
		}
	}

	return pass;
}

void print_results(int *m)
{
	printf("Result = \n");
	printf("======\n");
	for (int r = 0; r < WIDTH; r++)
	{
		for (int c = 0; c < WIDTH; c++)
		{
			printf("%4d ", m[r*WIDTH + c]);
		}
		printf("\n");
	}
	printf("======\n");
}