#include <cuda_runtime.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 2048
#define THREADS_PER_BLOCK 256

//Kernel
__global__ void marks(float * media, int * final){
	int thread = blockIdx.x*blockDim.x + threadIdx.x;
	//TODO -> Calcular nota final según los criterios establecidos
}


int main(){
	struct timeval t1, t2;
	float *hMedia, *dMedia;
	int *hFinal, *dFinal;

	//Reserva de memoria Host
	hMedia = (float*)malloc(N*sizeof(int));
	hFinal = (int*)malloc(N*sizeof(int));

	//Inicialización de vectores
	srand(time(NULL));
	for (int i = 0; i < N; i++){
		hMedia[i] = ((float)rand()/RAND_MAX)*10;
	}

	//Reserva de memoria Device
	cudaMalloc((void **)&dMedia, N*sizeof(int));
	cudaMalloc((void **)&dFinal, N*sizeof(int));

	//Copia de memoria Host->Device
	cudaMemcpy(dMedia, hMedia, N*sizeof(int), cudaMemcpyHostToDevice);

	int nblocks = N / THREADS_PER_BLOCK;

	gettimeofday(&t1, 0);

	//Función Kernel
	marks<<<nblocks, THREADS_PER_BLOCK>>>(dMedia, dFinal);

	cudaDeviceSynchronize();
	gettimeofday(&t2, 0);

	//Copia de memoria Device->Host
	cudaMemcpy(hFinal, dFinal, N*sizeof(int), cudaMemcpyDeviceToHost);

	//Comprobación de errores
	bool error = false;
	for(int i = 0; i < N; i++){
		if (hMedia[i] == (int)hMedia[i]){
			if(hFinal[i] != (int)hMedia[i]){
				error = true;
				printf("Media[%d] = %f -> Final[%d] = %d\n", i, hMedia[i], i, hFinal[i]);
				break;
			}
		}else if (hMedia[i] > 4 && hMedia[i] < 5){
			if (hFinal[i] != 4){
				error = true;
				printf("Media[%d] = %f -> Final[%d] = %d\n", i, hMedia[i], i, hFinal[i]);
				break;
			}
		}else if(hMedia[i] > 9){
			if (hFinal[i] != 9){
				error = true;
				printf("Media[%d] = %f -> Final[%d] = %d\n", i, hMedia[i], i, hFinal[i]);
				break;
			}
		}else if(hFinal[i] != (int)hMedia[i] + 1){
			error = true;
			printf("Media[%d] = %f -> Final[%d] = %d\n", i, hMedia[i], i, hFinal[i]);
			break;
		}
	}

	if(error)
		printf("La nota final no se ha calculado correctamente :(\n");
	else
		printf("La nota final se ha calculado correctamente! :D\n");

	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Tiempo: %f ms\n", time);

	//Liberar memoria Host y Device
	free(hMedia);
	free(hFinal);
	cudaFree(dMedia);
	cudaFree(dFinal);
}
