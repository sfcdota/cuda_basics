#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "sm_35_atomic_functions.h"
#include <iostream>
#include <math.h>
#include <time.h>
#include "curand_kernel.h"
#define BLOCKS_LIMIT 127
#define EXPERIMENTS_LIMIT 100000
#define PI_KEK 3.141592653589793238462643383279502884

__global__ void setStates(curandState* states, long long dots, time_t seed)
{
	int id = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
	curand_init(seed, id, 0, &states[id]);
	curand_init(seed + 1, id, 0, &states[id + 1]);

}

__device__ float rand_float(curandState *state)
{
	return curand_uniform(state);
}

__global__ void ineffectiveMonteCarloCall( unsigned long long *hits, curandState *states, int experiments)
{
	int id = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
	for (int i = 0; i < experiments; i++)
		if (pow(rand_float(&states[id]), 2) + pow(rand_float(&states[id + 1]), 2) <= 1.0)
			atomicAdd(hits, 1ull);
}


__global__ void effectiveMonteCarloCall(unsigned long long* hits, curandState* states, int experiments)
{
	int id = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
	int res = 0;
	for (int i = 0; i < experiments; i++)
		if (pow(rand_float(&states[id]), 2) + pow(rand_float(&states[id + 1]), 2) <= 1.0)
			++res;
	atomicAdd(&hits[blockIdx.x], res);
}

double GetPI(bool isEffective, unsigned long long& hits, int &dots, int &experiments, int &blocks)
{
	unsigned long long res = 0;
	unsigned long long*res_arr;
	if (!isEffective)
		cudaMemcpy(&res, &hits, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	else
	{
		cudaMallocHost(&res_arr, sizeof(unsigned long long) * blocks);
		cudaMemcpy(res_arr, &hits, sizeof(unsigned long long) * blocks, cudaMemcpyDeviceToHost);
	}
	for (int i = 0; isEffective && i < blocks; i++)
		res += res_arr[i];
	if (isEffective)
		cudaFreeHost(res_arr);
	return static_cast<double>(res) * 4 / dots / experiments;
}



void MonteCarlo(bool isEffective, int experiments, int &blocks, int &threads, int &dots)
{
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaStream_t stream;
	time_t seed = time(NULL);
	cudaError_t error;
	curandState* states;
	unsigned long long* hits;
	double res;
	if (!isEffective)
	{
		cudaMalloc(&hits, sizeof(unsigned long long));
		cudaMemset(hits, 0, sizeof(unsigned long long));
	}
	else
	{
		cudaMalloc(&hits, sizeof(unsigned long long) * blocks);
		cudaMemset(hits, 0, sizeof(unsigned long long) * blocks);
	}
	cudaMalloc(&states, sizeof(curandState) * dots * 2);
	setStates << <blocks, threads >> > (states, dots, seed);
	if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess)
		std::cout << cudaGetErrorString(error) << std::endl << cudaGetErrorName(error) << std::endl;
	cudaStreamCreate(&stream);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, stream);
	if (!isEffective)
		ineffectiveMonteCarloCall <<<blocks, threads>>> (hits, states, experiments);
	else
		effectiveMonteCarloCall <<<blocks, threads>>> (hits, states, experiments);
	if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess)
		std::cout << cudaGetErrorString(error) << std::endl << cudaGetErrorName(error) << std::endl;
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaStreamDestroy(stream);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("ineffective call elapsed time = %f\tPI = %f\tDifference = %f\n",
		elapsedTime, res = GetPI(isEffective, *hits, dots, experiments, blocks), abs(res - PI_KEK));
	cudaFree(states);
	cudaFree(hits);
}


int main()
{
	int dots;
	int slices;
	int experiments;

	std::cout << "Enter experiments count to do not more than 100000: ";
	std::cin >> experiments;
	std::cout << std::endl;
	std::cout << "Enter slices to divided a grid with values from 0 to 127(value will be adjuted to the nearest x64 multiply number): ";
	std::cin >> slices;
	std::cout << std::endl;
	slices = slices > BLOCKS_LIMIT ? BLOCKS_LIMIT - 1 : slices;
	experiments = experiments > EXPERIMENTS_LIMIT ? EXPERIMENTS_LIMIT : experiments;
	int blocks = slices / 64 + 1;
	int threads = blocks * 64;
	dots = threads * blocks;
	std::cout << "blocks = " << blocks << "\t threads = " << threads << "\texperiments = " << experiments << std::endl;
	MonteCarlo(false, experiments, blocks, threads, dots);
	cudaDeviceSynchronize();
	MonteCarlo(true, experiments, blocks, threads, dots);
	return (0);
}