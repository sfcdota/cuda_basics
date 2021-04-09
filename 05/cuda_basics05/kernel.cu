
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <iostream>
#define THREADS 512
#define N1 2
#define M1 3
#define N2 3
#define M2 2

template <typename T>
__global__ void is_orthogonal_int(T* a, bool *t, int n1, int m1, int n2, int m2)
{
	unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i >= n1 || j >= m2)
		return;
	T temp = 0;
	for (int p = 0; p < M1; p++)
	{
		temp += a[i * M1 + p] * a[p + j * N2];
		//if (i == 1 && j == 1)
		//printf("i = %li i_elem = %i\tj = %li j_elem = %i\tval1 = %f\tval2 = %f\n", i, i * M1 + p, j, p + j * N2, a[i * M1 + p], a[p + j * N2]);
	}
	//printf("i = %i\t j = %i\t res = %i\n", i, j, static_cast<int>(temp));
	if ((i != j && temp != 0) || (i == j && temp != 1))
		*t = false;
}

template <typename T>
__global__ void multiply(T* a, T* b, T *c, int n1, int m1, int n2, int m2)
{
	unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i >= n1 || j >= m2)
		return;
	T temp = 0;
	for (int p = 0; p < m1; p++)
	{
		temp += a[i * m1 + p] * b[p * m2 + j];
		//if (i == 0)
		//printf("i = %li i_elem = %i\tj = %li j_elem = %i\tval1 = %f\tval2 = %f\n", i, i * m1 + p, j, p * m2 + j, a[i * m1 + p], b[p * m2 + j]);
	}
	c[i * n1 + j] = temp;
}

template <typename T>
void print_matrix(T* matrix, int n, int m, std::string const & msg)
{
	printf("%s", msg.c_str());
	for (int i = 0; i < n; i++)
	{
		printf("| ");
		for (int j = 0; j < m; j++)
			printf("%f ", matrix[i * m + j]);
		printf("|\n");
	}
	printf("\n");
}

template <typename T>
void fill_matrix(T* matrix, int n, int m)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			matrix[i * m + j] = static_cast<T>(i * n - j);
}

template <typename T>
void fill_matrix2(T* matrix, int n, int m)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			matrix[i * m + j] = static_cast<T>(i * n + j);
}

template <typename T>
void fille_matrix(T* matrix, int n, int m)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			matrix[i * m + j] = i == j ? 1 : 0;
}

template <typename T>
T* transpose(T* a, int n, int m)
{
	T* t;
	cudaMallocHost(&t, sizeof(T) * n * m);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			t[i * n + j] = a[j * m + i];
	return t;
}

template <typename T>
T* mult(T* a_h, T* b_h, int n1, int m1, int n2, int m2)
{
	cudaError_t error;
	T * c_h, * a_d, * b_d, * c_d;

	if (m1 != n2)
	{
		printf("multiplication is impossible\n");
		return nullptr;
	}
	cudaMallocHost(&c_h, sizeof(T) * n1 * m2);

	cudaMalloc(&a_d, sizeof(T) * n1 * m1);
	cudaMalloc(&b_d, sizeof(T) * n2 * m2);
	cudaMalloc(&c_d, sizeof(T) * n1 * m2);


	//print_matrix(a_h, n1, m1);
	//print_matrix(b_h, n2, m2);

	cudaMemcpy(a_d, a_h, sizeof(T) * n1 * m1, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, sizeof(T) * n2 * m2, cudaMemcpyHostToDevice);
	dim3 threads(THREADS, THREADS);
	dim3 blocks(static_cast<unsigned>((ceil(n1 * 1.0 / THREADS))), static_cast<unsigned>((ceil(m2 * 1.0 / THREADS))));
	multiply << <blocks, threads >> > (a_d, b_d, c_d, n1, m1, n2, m2);
	if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess)
		std::cout << "Cuda error in matrix multiplication" << std::endl
		<< "Error description: " << cudaGetErrorString(error) << std::endl << cudaGetErrorName(error) << std::endl;
	cudaMemcpy(c_h, c_d, sizeof(T) * n1 * m2, cudaMemcpyDeviceToHost);








	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	return c_h;
}

void test_multiply()
{
	float* a_h, * b_h;
	cudaMallocHost(&a_h, sizeof(float) * N1 * M1);
	cudaMallocHost(&b_h, sizeof(float) * N2 * M2);
	fill_matrix(a_h, N1, M1);
	fill_matrix(b_h, N2, M2);
	print_matrix(a_h, N1, M1, "Matrix a\n");
	print_matrix(b_h, N2, M2, "Matrix b\n");
	float *c_h = mult(a_h, b_h, N1, M1, N2, M2);
	print_matrix(c_h, N1, M2, "MULTIPLICATION RESULT\n");

	cudaFreeHost(a_h);
	cudaFreeHost(b_h);
	if (c_h) cudaFreeHost(c_h);
}

void ex0()
{
	cudaError_t error;
	bool* t_h, * t_d;
	float* a_h, *a_d;
	cudaMallocHost(&t_h, sizeof(bool));
	cudaMallocHost(&a_h, sizeof(float) * N1 * M1);
	cudaMalloc(&a_d, sizeof(float) * N1 * M1);
	cudaMalloc(&t_d, sizeof(bool));


	*t_h = true;
	fill_matrix(a_h, N1, M1);
	print_matrix(a_h, N1, M1, "Matrix a\n");

	cudaMemcpy(t_d, t_h, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(a_d, a_h, sizeof(float) * N1 * M1, cudaMemcpyHostToDevice);


	dim3 threads(THREADS, THREADS);
	dim3 blocks(static_cast<unsigned>((ceil(N1 * 1.0 / THREADS))), static_cast<unsigned>((ceil(M1 * 1.0 / THREADS))));
	is_orthogonal_int<float> << <blocks, threads >> > (a_d, t_d, N1, M1, M1, N1);
	if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess)
		std::cout << "Cuda error in orthogonalization" << std::endl
		<< "Error description: " << cudaGetErrorString(error) << std::endl << cudaGetErrorName(error) << std::endl;
	cudaMemcpy(t_h, t_d, sizeof(bool), cudaMemcpyDeviceToHost);
	printf("ex 1: input matrix is %s", *t_h ? "orthogonal\n" : "NOT orthogonal\n");

	float* at_h = transpose(a_h, N1, M1);
	print_matrix(at_h, M1, N1, "transpose matrix a\n");
	float* ata_mult = mult(a_h, at_h, N1, M1, M1, N1);
	print_matrix(ata_mult, N1, N1, "aT * a multiplication\n");
	cudaFreeHost(a_h);
	cudaFreeHost(at_h);
	cudaFreeHost(ata_mult);
	cudaFreeHost(t_h);
	cudaFree(t_d);
	cudaFree(a_d);
}


void ex1(int n)
{
	float* a_h, * b_h;
	cudaMallocHost(&a_h, sizeof(float) * n * n);
	cudaMallocHost(&b_h, sizeof(float) * n * n);
	fill_matrix(a_h, n, n);
	fill_matrix2(b_h, n, n);
	print_matrix(a_h, n, n, "Matrix a\n");
	print_matrix(b_h, n, n, "Matrix b\n");

	float* ab_h = mult(a_h, b_h, n, n, n, n);
	float* ba_h = mult(b_h, a_h, n, n, n, n);
	bool ok = true;
	for (int i = 0; i < n && ok; i++)
	{

		for (int j = 0; j < n && ok; j++)
			if (ab_h[i * n + j] != ba_h[i * n + j])
				ok = false;
	}
	printf("matrix a and matrix b are %s\n", ok ? "commutable" : "not commutable");
	print_matrix(ab_h, n, n, "Matrix a * b\n");
	print_matrix(ba_h, n, n, "Matrix b * a\n");
	cudaFreeHost(a_h);
	cudaFreeHost(b_h);
	cudaFreeHost(ab_h);
	cudaFreeHost(ba_h);

}

template <typename T>
void fillv(T a[][M1], int n, int m)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			a[i][j] = static_cast<T>(i * n - j);
}

template <typename T>
void print_matrixv(T a[][M1], int n, int m, std::string const & msg)
{
	printf("%s", msg.c_str());
	for (int i = 0; i < n; i++)
	{
		printf("| ");
		for (int j = 0; j < m; j++)
			printf("%f ", a[i][j]);
		printf("|\n");
	}
	printf("\n");
}

template <typename T>
__global__ void sum(T* a, T* b, T* c, int n1, int m1, int n2, int m2)
{
	unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i >= n1 || j >= m2)
		return;
	c[i * m1 + j] = a[i * m1 + j] + b[i * m1 + j];
}

void ex2()
{
	cudaError_t error;
	float a_h[N1][M1], b_h[N1][M1], *a_d, *b_d, *c_d, *c_h;
	fillv(a_h, N1, M1);
	fillv(b_h, N1, M1);


	cudaMalloc(&a_d, sizeof(float) * N1 * M1);
	cudaMalloc(&b_d, sizeof(float) * N1 * M1);
	cudaMalloc(&c_d, sizeof(float) * N1 * M1);
	cudaMallocHost(&c_h, sizeof(float) * N1 * M1);


	print_matrixv(a_h, N1, M1, "MATRIX A\n");
	print_matrixv(b_h, N1, M1, "MATRIX B\n");

	cudaMemcpy(a_d, a_h, sizeof(float) * N1 * M1, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, sizeof(float) * N1 * M1, cudaMemcpyHostToDevice);
	dim3 threads(THREADS, THREADS);
	dim3 blocks(static_cast<unsigned>((ceil(N1 * 1.0 / THREADS))), static_cast<unsigned>((ceil(M1 * 1.0 / THREADS))));
	sum << <blocks, threads >> > (a_d, b_d, c_d, N1, M1, N1, M1);
	if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess)
		std::cout << "Cuda error in matrix sum" << std::endl
		<< "Error description: " << cudaGetErrorString(error) << std::endl << cudaGetErrorName(error) << std::endl;
	cudaMemcpy(c_h, c_d, sizeof(float) * N1 * M1, cudaMemcpyDeviceToHost);
	print_matrix(c_h, N1, M1, "SUM\n");







	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	cudaFree(c_h);
}

int main()
{
	//test_multiply();
	ex0();
	ex1(2);
	ex2();
	return 0;
}
