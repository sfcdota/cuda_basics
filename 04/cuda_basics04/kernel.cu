
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
//#define ARRAY_SIZE 1000000000
//#define BLOCK_SIZE 1000000
//#define THREAD_SIZE 1000
#define ARRAY_SIZE 10000
#define BLOCK_SIZE 10
#define THREAD_SIZE 10
#define PI 3.141592653589793238462643383279502884
#define TYPE float
#define MODULE 30 + 1
#define WIDTH 35
#define VECTOR_SIZE 5
#define VECTOR_ANGLE 0.54353

template <typename T>
T Error(int elements, T* arr, T (*f)(T), std::string const& func, float elapsedTime)
{
	T res = 0;
	for (int i = 0; i < elements; i++)
		res += abs(f(static_cast<T>(i % MODULE)) - arr[i]);
	std::cout << func << " completed in " << elapsedTime << "ms." <<  " RESULT ERROR = " << res / elements << std::endl;
	return res / elements;
}

template<typename T, typename TFuncDevice>
__global__ void unified(T* arr, TFuncDevice f)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	arr[id] = f(static_cast<T>(id % MODULE));

}

template <typename T, typename TFuncDevice>
void test_func(int elements, TFuncDevice f, T func(T), std::string const& name)
{
	T* arr_host, * arr_dev;
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc(&arr_dev, sizeof(T) * elements);
	cudaMallocHost(&arr_host, sizeof(T) * elements);
	cudaEventRecord(start, 0);
	unified << <BLOCK_SIZE, THREAD_SIZE >> > (arr_dev, f);
	cudaDeviceSynchronize();
	cudaMemcpy(arr_host, arr_dev, sizeof(T) * elements, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	Error(elements, arr_host, func, name, elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(arr_dev);
	cudaFreeHost(arr_host);
}

//capture address of a device functions
__device__ float (*logf_dev)(float) = __logf;
__device__ float (*log10f_dev)(float) = __log10f;
__device__ float (*log2f_dev)(float) = __log2f;

void ex1()
{
	//copying address of a device functions to device pointers, defined on host for futher using in templates
	float (*log2f_d)(float), (*logf_d)(float), (*log10f_d)(float);
	cudaMemcpyFromSymbol(&log2f_d, log2f_dev, sizeof(void*));
	cudaMemcpyFromSymbol(&logf_d, logf_dev, sizeof(void*));
	cudaMemcpyFromSymbol(&log10f_d, log10f_dev, sizeof(void*));


	test_func<float>(ARRAY_SIZE, log2f_d, log2f, "log2f");
	test_func<float>(ARRAY_SIZE, logf_d, logf, "logf");
	test_func<float>(ARRAY_SIZE, log10f_d, log10f, "log10f");
	test_func<double>(ARRAY_SIZE, log2f_d, log2, "log2");
	test_func<double>(ARRAY_SIZE, logf_d, log, "log");
	test_func<double>(ARRAY_SIZE, log10f_d, log10, "log10");
}


__global__ void multiply(float* a_d, float* b_d, float* c_d)
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < VECTOR_SIZE)
		atomicAdd(c_d, a_d[id] * b_d[id]);
}


void print_vector(float* a)
{
	for (int i = 0; i < VECTOR_SIZE; i++)
		std::cout << std::setw(14) << std::right << a[i] << (i == VECTOR_SIZE - 1 ? "" : ", ");
	std::cout << std::endl;
}

float scalar(float** a_h, float** a_d, float** b_h, float**b_d, float* c_h, float* c_d)
{
	cudaMemcpy(*a_d, *a_h, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(*b_d, *b_h, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice);
	multiply << <BLOCK_SIZE, THREAD_SIZE >> > (*a_d, *b_d, c_d);
	cudaMemcpy(c_h, c_d, sizeof(float), cudaMemcpyDeviceToHost);
	return *c_h;
}

void ex2()
{
	float* a_h, * b_h, * c_h, * a_d, * b_d, * c_d;
	cudaMallocHost(&a_h, sizeof(float) * VECTOR_SIZE);
	cudaMallocHost(&b_h, sizeof(float) * VECTOR_SIZE);
	cudaMallocHost(&c_h, sizeof(float));
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		a_h[i] = i + VECTOR_SIZE;
		b_h[i] = -2 * i + VECTOR_SIZE;
	}
	cudaMalloc(&a_d, sizeof(float) * VECTOR_SIZE);
	cudaMalloc(&b_d, sizeof(float) * VECTOR_SIZE);
	cudaMalloc(&c_d, sizeof(float));
	std::cout << std::endl << std::endl << "vector a = ";
	print_vector(a_h);
	std::cout << "vector b = ";
	print_vector(b_h);
	std::cout << "scalar multiply a * b = " << scalar (&a_h, &a_d, &b_h, &b_d, c_h, c_d) << std::endl << std::endl << std::endl;
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	cudaFreeHost(a_h);
	cudaFreeHost(b_h);
	cudaFreeHost(c_h);
}


float host_multiply(float* a, float* b)
{
	float res = 0;
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		res += a[i] * b[i];
	}
	return res;
}

__global__ void iter(float** vectors_d, int current, int start, float *t1, float*t2)
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

	*t1 = 0; *t2 = 0;
	__syncthreads();

	if (id < VECTOR_SIZE)
	{
		atomicAdd(t1, vectors_d[current][id] * vectors_d[start][id]);
		atomicAdd(t2, vectors_d[start][id] * vectors_d[start][id]);
		__syncthreads();
		vectors_d[current][id] -= *t1 / *t2 * vectors_d[start][id];
	}
}


void ex3()
{
	float** vectors_h, **vectors_d, *t1, *t2;
	cudaMallocHost(&vectors_h, sizeof(float*) * VECTOR_SIZE);
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		cudaMallocHost(&vectors_h[i], sizeof(float));
		for (int j = 0; j < VECTOR_SIZE; j++)
			vectors_h[i][j] = j < i ? 0 : 1;
	}
	cudaMalloc(&vectors_d, sizeof(float*) * VECTOR_SIZE);
	cudaMalloc(&t1, sizeof(float));
	cudaMalloc(&t2, sizeof(float));
	cudaMemcpy(vectors_d, vectors_h, sizeof(float*) * VECTOR_SIZE, cudaMemcpyHostToDevice);







	for (int current = 1; current < VECTOR_SIZE; current++)
	{
		for (int i = 0; i < current; i++)
		{

			iter << <BLOCK_SIZE, THREAD_SIZE >> > (vectors_d, current, i, t1, t2);
			cudaDeviceSynchronize();
		}
	}






	cudaMemcpy(vectors_h, vectors_d, sizeof(float*) * VECTOR_SIZE, cudaMemcpyDeviceToHost);
	std::cout << std::setw(60) << "orthogonal vectors" << std::endl << std::endl;
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		std::cout << "vector #" << i + 1 << ": ";
		print_vector(vectors_h[i]);
	}
	std::cout << std::endl << std::endl << std::setw(75) << "Multiplication of orthogonal vectors" << std::endl << std::endl;
	std::cout << std::setw (12) << "";
	for (int i = 0; i < VECTOR_SIZE; i++)
		std::cout << std::setw (15 + i) << std::right << i + 1;
	std::cout << std::endl << std::endl;
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		std::cout << std::setw (16) << std::left << i + 1;
		for (int j = 0; j < VECTOR_SIZE; j++)
			std::cout << std::setw (15 + j) << std::right << host_multiply (vectors_h[i], vectors_h[j]);
		std::cout << std::endl;
	}
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		cudaFreeHost(&vectors_h[i]);
		cudaFree(&vectors_d[i]);
	}
	cudaFree(t1);
	cudaFree(t2);
	cudaFreeHost(vectors_h);
	cudaFree(vectors_d);
}

int main()
{
	ex1();
	ex2();
	ex3();
	return 0;
}
