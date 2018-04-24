#include <cuda_runtime.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#define DIMBLOCK_X 65535 //2^16
#define DIMBLOCK_Y 32 //2^5
#define DIMTHREAD_X 1024 //2^10

//Total 2^31

__device__ char found(0);

__global__ void searchFactor(unsigned long int * number, unsigned int * factor){
	if (found) return;

	unsigned int block = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int n = block * blockDim.x + threadIdx.x;
	n = (n + 1) * 2 + 1;

	if (*number % n == 0){
		*factor = n;
		found = 1;
	}
}


int main(){
	struct timeval t1, t2;
	unsigned long int h_number;
	unsigned int h_prime1;
	unsigned long int h_prime2;
	unsigned long int *d_number;
	unsigned int *d_factor;

	h_number = 742312722905005279;

	cudaMalloc((void **)&d_number, sizeof(unsigned long int));
	cudaMalloc((void **)&d_factor, sizeof(unsigned int));

	cudaMemcpy(d_number, &h_number, sizeof(unsigned long int), cudaMemcpyHostToDevice);

	dim3 blocks(DIMBLOCK_X, DIMBLOCK_Y);

	gettimeofday(&t1, 0);

	searchFactor<<<blocks, DIMTHREAD_X>>>(d_number, d_factor);

	cudaDeviceSynchronize();
	gettimeofday(&t2, 0);

	cudaMemcpy(&h_prime1, d_factor, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	printf("Primo 1 = %d\n", h_prime1);
	h_prime2 = h_number / h_prime1;
	printf("Primo 2 = %ld\n", h_prime2);

	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Tiempo: %f ms\n", time);

	/*
	 * p = 976250239;
	 * q = 760371361;
	*/

	cudaFree(d_number);
	cudaFree(d_factor);
}
