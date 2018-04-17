#include <cuda_runtime.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

//Tamaño de matrices (cuadradas)
#define N 1024

//Kernel
__global__ void mul(int * A, int * B, int * C){
	int i = blockIdx.x;
	int j = threadIdx.x;
	//TODO -> Calcular elemento C(i,j)
}


int main(){
	struct timeval t1, t2;
	int *hA, *hB, *hC, *hC2; //Host Matrix
	int *dA, *dB, *dC; //Device Matrix

	//Reserva de memoria Host
	hA = (int*)malloc(N*N*sizeof(int));
	hB = (int*)malloc(N*N*sizeof(int));
	hC = (int*)malloc(N*N*sizeof(int));
	hC2 = (int*)malloc(N*N*sizeof(int));

	//Inicialización de matrices
	srand(time(NULL));
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			hA[i*N+j] = rand();
			hB[i*N+j] = rand();
		}
	}

	//Reserva de memoria GPU
	//TODO -> Reservar Memoria (dA, dB, dC)

	//Copia Host -> GPU
	//TODO -> Copiar hA a dA y hB a dB

	gettimeofday(&t1, 0);

	//Ejecución Kernel
	mul<<<N, N>>>(dA, dB, dC);

	cudaDeviceSynchronize();
	gettimeofday(&t2, 0);

	//Copia Device -> Host
	//TODO -> Copiar dC a hC

	//Multiplicación en Host
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			hC2[i*N + j] = 0;
			for(int k = 0; k < N; k++){
				hC2[i*N + j] += hA[i*N + k] * hB[k*N + j];
			}
		}
	}

	//Comprobación de errores
	bool error = false;
	for(int i = 0; i < N*N; i++){
		if(hC[i] != hC2[i]){
			error = true;
			break;
		}
	}

	if(error)
		printf("La multiplicación de matrices ha fallado.\n");
	else
		printf("Multiplicación de matrices correcta.\n");

	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Tiempo: %f ms\n", time);

	//Liberar memoria
	free(hA);
	free(hB);
	free(hC);
	//TODO -> Liberar dA, dB y dC
}
