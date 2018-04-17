#include <cuda_runtime.h>

#include <stdio.h>

__global__ void cudahello(){
	int thread[2] = {threadIdx.x, threadIdx.y};
	int block[2] = {blockIdx.x, blockIdx.y};
	printf("Hola Mundo! Soy el hilo %d,%d del bloque %d,%d\n", thread[0], thread[1], block[0], block[1]);
}

int main(){
	cudahello<<<dim3(2,2),dim3(2,2)>>>();
	cudaDeviceSynchronize();
}
