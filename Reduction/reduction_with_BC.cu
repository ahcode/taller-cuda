#include <cuda_runtime.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 8388608
#define BLOCK_DIM 256

//Kernel
__global__ void reduction(int * in, int * out){
	int globalid = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ int s_array[BLOCK_DIM];

	s_array[threadIdx.x] = in[globalid];
	__syncthreads();

	for (int i = 1; i < blockDim.x; i *= 2){
		if (threadIdx.x % (2*i) == 0){
			s_array[threadIdx.x] += s_array[threadIdx.x+i];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		out[blockIdx.x] = s_array[0];
}

int main(){
	struct timeval t1, t2;
	int *hArray;
	int hReduction;
	int *dIn, *dOut; //Device Arrays

	//Reserva de memoria Host
	hArray = (int*)malloc(N*sizeof(int));

	//Inicialización del vector
	srand(time(NULL));
	for (int i = 0; i < N; i++){
		hArray[i] = ((float)rand()/RAND_MAX)*200 - 100;
	}

	//Reserva de memoria Device
	cudaMalloc((void **)&dIn, N*sizeof(int));
	cudaMalloc((void **)&dOut, (N/BLOCK_DIM)*sizeof(int));

	//Copia de memoria Host->Device
	cudaMemcpy(dIn, hArray, N*sizeof(int), cudaMemcpyHostToDevice);

	int *aux;
	int block_dim_stage = BLOCK_DIM;
	int blocks;

	gettimeofday(&t1, 0);

	//Reducción por etapas
	for(int left = N; left > 1; left /= block_dim_stage){
		if(left < block_dim_stage)
			block_dim_stage = left;
		blocks = left / block_dim_stage;
		cudaDeviceSynchronize();
		reduction<<<blocks, block_dim_stage>>>(dIn, dOut);
		aux = dIn;
		dIn = dOut;
		dOut = aux;
	}

	cudaDeviceSynchronize();
	gettimeofday(&t2, 0);

	//Copia de memoria Device->Host
	cudaMemcpy(&hReduction, dIn, sizeof(int), cudaMemcpyDeviceToHost);

	//Comprobación de errores
	int hReduction2 = 0;
	for(int i = 0; i < N; i++){
		hReduction2 += hArray[i];
	}

	if(hReduction != hReduction2)
		printf("Error\n");
	else
		printf("Correcto\n");

	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Tiempo: %f ms\n", time);
	printf("Reducción = %d\n", hReduction);

	//Liberar memoria Host y Device
	free(hArray);
	cudaFree(dIn);
	cudaFree(dOut);
}
