#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
__global__ void sum (int a, int b)
{
	printf ("sum of a + b = %d\n", a + b);
}
int main ()
{
	int a, b;
	std::cout << "Enter num a:";
	std::cin >> a;
	std::cout << "Enter num b:";
	std::cin >> b;
	sum <<<1, 1>>>(a, b);
	getchar ();
	return 0;
}