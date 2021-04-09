#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>


void cudaDevicesInfo ()
{
	int deviceCount;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount (&deviceCount);
	for (int device = 0; device < deviceCount; ++device) {
		printf ("Device #%d:\n\n", device);
		cudaGetDeviceProperties (&deviceProp, device);
		printf ("Device name : %s\n", deviceProp.name);
		printf ("Total global memory : %llu MB\n", deviceProp.totalGlobalMem / 1024 / 1024);
		printf ("Total constant memory: %llu\n", deviceProp.totalConstMem);
		printf ("Shared memory per block : %llu\n", deviceProp.sharedMemPerBlock);
		printf ("Registers per block : %d\n", deviceProp.regsPerBlock);
		printf ("Warp size : %d\n", deviceProp.warpSize);
		printf ("Max threads per block : %d\n", deviceProp.maxThreadsPerBlock);
		printf ("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf ("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
		printf ("Clock rate: %d\n", deviceProp.clockRate);
		printf ("Memory clock rate: %d\n", deviceProp.memoryClockRate);
		printf ("L2 cache size: %d\n", deviceProp.l2CacheSize);
		printf ("Memory bus width: %d\n", deviceProp.memoryBusWidth);
		printf ("Max dimension of a block in grid: %d x %d x %d\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf ("Max dimension of a grid:  %d x %d x %d\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

	}
}


int main () {
	cudaDevicesInfo ();
}