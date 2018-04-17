#include <cuda_runtime.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 8388608
#define THREADS_PER_BLOCK 1024

//Kernel
__global__ void add(int * A, int * B, int * C){
	int thread = blockIdx.x*blockDim.x + threadIdx.x;
	C[thread] = A[thread] + B[thread];
}


int main(){
	struct timeval t1, t2;
	int *hA, *hB, *hC; //Host Arrays
	int *dA, *dB, *dC; //Device Arrays

	//Reserva de memoria Host
	hA = (int*)malloc(N*sizeof(int));
	hB = (int*)malloc(N*sizeof(int));
	hC = (int*)malloc(N*sizeof(int));

	//Inicialización de vectores
	srand(time(NULL));
	for (int i = 0; i < N; i++){
		hA[i] = rand();
		hB[i] = rand();
	}

	//Reserva de memoria Device
	cudaMalloc((void **)&dA, N*sizeof(int));
	cudaMalloc((void **)&dB, N*sizeof(int));
	cudaMalloc((void **)&dC, N*sizeof(int));

	//Copia de memoria Host->Device
	cudaMemcpy(dA, hA, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, N*sizeof(int), cudaMemcpyHostToDevice);

	int nblocks = N / THREADS_PER_BLOCK;

	gettimeofday(&t1, 0);

	//Función Kernel
	add<<<nblocks, THREADS_PER_BLOCK>>>(dA, dB, dC);

	cudaDeviceSynchronize();
	gettimeofday(&t2, 0);

	//Copia de memoria Device->Host
	cudaMemcpy(hC, dC, N*sizeof(int), cudaMemcpyDeviceToHost);

	//Comprobación de errores
	bool error = false;
	for(int i = 0; i < N; i++){
		if(hC[i] != hA[i] + hB[i]){
			error = true;
			break;
		}
	}

	if(error)
		printf("La suma de vectores ha fallado.\n");
	else
		printf("Suma de vectores correcta.\n");

	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Tiempo: %f ms\n", time);

	//Liberar memoria Host y Device
	free(hA);
	free(hB);
	free(hC);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}
