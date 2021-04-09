#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <stdio.h>

/// <summary>
/// 
/// </summary>
/// <param name="info_volume_multiplier"></param>
/// <param name="kind">: Host to device = 1;
/// device to host = 2;</param>
/// <returns></returns>
float copySpeed (int info_volume_multiplier, int step, int kind)
{
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaStream_t stream;

	int *a_host, *a_dev;
	cudaMalloc(&a_dev, sizeof(int) * info_volume_multiplier);
	cudaMallocHost (&a_host, sizeof (int) * info_volume_multiplier);
	for (int i = 0; i < info_volume_multiplier; i++)
		a_host[i] = INT_MAX;
	cudaMemcpy (a_dev, a_host, sizeof (int) * info_volume_multiplier, cudaMemcpyHostToDevice);
	cudaStreamCreate(&stream);
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord(start, stream);
	if (kind == 1)
		cudaMemcpy(a_dev, a_host, sizeof(int) * info_volume_multiplier, cudaMemcpyHostToDevice);
	else if (kind == 2)
		cudaMemcpy (a_host, a_dev, sizeof (int) * info_volume_multiplier, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaStreamDestroy(stream);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFreeHost(a_host);
	cudaFree (a_dev);
	return (elapsedTime);
}

__global__ void calculatePI(const int segments, const float start, const float end, const float section_length, float *pi)
{
	float mid_x = start + section_length * (blockIdx.x * blockDim.x + threadIdx.x + 0.5);
	pi[blockIdx.x * 1024 + threadIdx.x] = sqrtf(1 - mid_x * mid_x);
}

double get_pi(const int& segments, const float& start, const float& end)
{
	cudaError_t error;
	cudaDeviceProp deviceProp;
	float* pi_dev;
	float* pi_host;
	int seg1024;
	int blocks;
	int threads;
	float seg_length;

	cudaGetDeviceProperties (&deviceProp, 0);
	seg1024 = segments > deviceProp.maxThreadsDim[0] ? segments + 1024 - segments % 1024 : segments;
	blocks = seg1024 > deviceProp.maxThreadsDim[0] ? seg1024 / 1024 : 1;
	threads = seg1024 / blocks;
	seg_length = (end - start) / seg1024;
	cudaMallocHost (&pi_host, sizeof (float) * seg1024);
	cudaMalloc(&pi_dev, sizeof(float) * seg1024);
	printf("adjusted segments = %i\nblocks = %i\n"
		"threads = %i\nsegment length = %f\n", seg1024, blocks, threads, seg_length);
	calculatePI <<<blocks, threads >>> (seg1024, start, end, seg_length, pi_dev);
	error = cudaGetLastError();
	if (error != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess)
		std::cout << "There was an error in cuda device calculation" << std::endl
		<< "Error description: " << cudaGetErrorString(error) << std::endl << cudaGetErrorName(error) << std::endl;
	cudaMemcpy(pi_host, pi_dev, sizeof(float) * seg1024, cudaMemcpyDeviceToHost);
	error = cudaGetLastError ();
	if (error != cudaSuccess || (error = cudaDeviceSynchronize ()) != cudaSuccess)
		std::cout << "There was an error in cuda device calculation" << std::endl
		<< "Error description: " << cudaGetErrorString (error) << std::endl << cudaGetErrorName (error) << std::endl;
	cudaFree(pi_dev);
	error = cudaGetLastError ();
	if (error != cudaSuccess || (error = cudaDeviceSynchronize ()) != cudaSuccess)
		std::cout << "There was an error in cuda device calculation" << std::endl
		<< "Error description: " << cudaGetErrorString (error) << std::endl << cudaGetErrorName (error) << std::endl;
	float sum = 0.0;
	for (int i = 0; i < seg1024; i++)
		sum += pi_host[i];
	cudaFreeHost(pi_host);
	return 4 * sum *seg_length;
}


__global__ void calculateDzeta(float *res, float real_part)
{
	res[blockIdx.x * 1024 + threadIdx.x] =  powf(blockIdx.x * 1024 + threadIdx.x + 1, -real_part);
}

float dzeta_f(const int& segments, const float real_part)
{
	cudaError_t error;
	cudaDeviceProp deviceProp;
	float *res_host;
	float* res_dev;
	int seg1024 = segments;
	int blocks;
	int threads;

	cudaGetDeviceProperties (&deviceProp, 0);
	if (segments > deviceProp.maxThreadsDim[0])
		seg1024 += 1024 - segments % 1024;
	blocks = seg1024 > deviceProp.maxThreadsDim[0] ? seg1024 / 1024 : 1;
	threads = seg1024 / blocks;
	cudaMalloc(&res_dev, sizeof(float) * seg1024);
	cudaMallocHost (&res_host, sizeof (float) * seg1024);
	calculateDzeta << <blocks, threads >> > (res_dev, real_part);
	error = cudaGetLastError();
	if (error != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess)
		std::cout << "There was an error in cuda device calculation" << std::endl
		<< "Error description: " << cudaGetErrorString(error) << std::endl << cudaGetErrorName(error) << std::endl;
	cudaMemcpy(res_host, res_dev, sizeof(float) * seg1024, cudaMemcpyDeviceToHost);
	cudaFree(res_dev);
	float sum = 0;
	for (int i = 0; i < seg1024; i++)
		sum += res_host[i];
	cudaFreeHost(res_host);
	return sum;
}

void speedTests (int count, int step)
{
	float *devToHost = new float[count];
	float* hostToDev = new float[count];
	for (int i = 0; i < count; i++)
	{
		devToHost[i] = copySpeed ((i + 1) * step, step, 2);
		hostToDev[i] = copySpeed ((i + 1) * step, step, 1);
	}

	for (int i = 0; i < count; i++)
		printf ("elem count = %d\tdevToHost=%f\thostToDev=%f\n", (i + 1) * step, devToHost[i], hostToDev[i]);
}

int main ()
{
	std::string selection;
	const float START = 0;
	const float END = 1;
	int segments;
	float argument;
	int tests;
	while (1)
	{
		std::cout << "Select exercise (1 or 2 or 3), or enter exit to quit, other input discrads: ";
		std::cin >> selection;
		if (selection == "1")
		{
			std::cout << std::endl << "Enter tests count: ";
			std::cin >> tests;
			std::cout << std::endl;
			std::cout << "Enter step: ";
			std::cin >> segments;
			std::cout << std::endl;
			speedTests (tests, segments);

		}
		else if (selection == "2")
		{
			std::cout << "Enter segments count (precision by segments of x axis): ";
			std::cin >> segments;
			std::cout << std::endl << "segments = " << segments << std::endl;
			std::cout << std::endl << "PI = " << get_pi(segments, START, END) << std::endl;
		}
		else if (selection == "3")
		{
			std::cout << "Enter elements count (precision like): ";
			std::cin >> segments;
			std::cout << std::endl << "Enter real part of argument: ";
			std::cin >> argument;
			std::cout << std::endl << "segments = " << segments << std::endl
				<< "argument = " << argument << std::endl;
			std::cout << std::endl << "res = " << dzeta_f(segments, argument) << std::endl;
		}
		else if (selection == "exit")
			break ;
		std::cout << std::endl;
	}
	return 0;
}