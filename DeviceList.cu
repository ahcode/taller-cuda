#include <cuda_runtime.h>

#include <stdio.h>

int main(){
	int deviceCount; cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d (%s) has compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
	}
}
