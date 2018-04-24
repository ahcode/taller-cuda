#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#define N 1048576
#define TAMSIMPLESORT 1024
#define TPB 1024

#define DEBUG false
#define PRINTARRAY false

typedef float dato; //Tipo de dato a ordenar

__device__ void simple_sort(dato *data, int left, int right) {
	for (int i = left; i <= right; ++i) {
		dato min_val = data[i];
		int min_idx = i;

		// Find the smallest value in the range [left, right].
		for (int j = i + 1; j <= right; ++j) {
			unsigned val_j = data[j];

			if (val_j < min_val) {
				min_idx = j;
				min_val = val_j;
			}
		}

		// Swap the values.
		if (i != min_idx) {
			data[min_idx] = data[i];
			data[i] = min_val;
		}
	}
}

__global__ void run_simple_sort(dato * clasificado, int * indices) {
	int thread = blockIdx.x*blockDim.x + threadIdx.x;
	int min = thread * TAMSIMPLESORT * 2;
	int max = (thread * TAMSIMPLESORT * 2) + indices[thread] - 1;
	if (DEBUG) printf("RSS Hilo %d - min:%d - max:%d\n", thread, min, max);
	simple_sort(clasificado, min, max);
}

__global__ void clasificar(dato * v, dato * fronteras, int * indices, int bloques, dato * clasificado) {
	int thread = blockIdx.x*blockDim.x + threadIdx.x;
	int ind, i;
	for (i = 0; i <= bloques; i++) {
		if (v[thread] <= fronteras[i]) {
			ind = atomicAdd(indices + i, 1);
			break;
		}
	}
	clasificado[2 * TAMSIMPLESORT*i + ind] = v[thread];
}

__global__ void rellenarIndFront(dato min, dato paso, int * indices, dato * fronteras) {
	int thread = blockIdx.x*blockDim.x + threadIdx.x;
	indices[thread] = 0;
	fronteras[thread] = min + paso*(thread+1);
	if (DEBUG) printf("RELLENAR hilo:%d frontera:%f min:%f paso:%f\n", thread, fronteras[thread], min, paso);
}

void runsort(dato * v_H, int l) {
	dato * v_D;
	cudaMalloc((void **)&v_D, sizeof(dato)*l);
	cudaMemcpy(v_D, v_H, sizeof(dato)*l, cudaMemcpyHostToDevice);

	dato min = v_H[0];
	dato max = v_H[0];


	for (int i = 0; i < l; i++) {
		if (v_H[i] < min) {
			min = v_H[i];
		}
		if (v_H[i] > max) {
			max = v_H[i];
		}
	}

	int bloques = l / TAMSIMPLESORT;
	if (l % TAMSIMPLESORT != 0) bloques++;

	dato * fronteras;
	cudaMalloc((void**)&fronteras, bloques * sizeof(dato));
	int * indices;
	cudaMalloc((void**)&indices, bloques * sizeof(int));
	int cudablocks = bloques / TPB;
	if (bloques % TPB != 0) cudablocks++;
	int cudathreads = bloques / cudablocks;
	if (DEBUG) printf("RUNSORT RELLENAR blocks:%d threads:%d min:%f, max:%f\n", cudablocks, cudathreads, min, max);
	rellenarIndFront <<<cudablocks, cudathreads>>> (min, (max-min) / (dato)bloques, indices, fronteras);
	cudablocks = l / TPB;
	if (l % TPB != 0) cudablocks++;
	cudathreads = l / cudablocks;

	dato * destino;
	cudaMalloc((void**)&destino, l * 2 * sizeof(dato));

	clasificar <<<cudablocks, cudathreads >>> (v_D, fronteras, indices, bloques, destino);
	cudaFree(v_D);

	cudablocks = bloques / TPB;
	if (bloques % TPB != 0) cudablocks++;
	cudathreads = bloques / cudablocks;
	run_simple_sort <<<cudablocks, cudathreads >>> (destino, indices);

	int * H_indices = (int *) malloc(sizeof(int)*bloques);
	cudaMemcpy(H_indices, indices, sizeof(int)*bloques, cudaMemcpyDeviceToHost);
	int offset = 0;
	for (int i = 0; i < bloques; i++) {
		if (DEBUG) printf("COPY Dest: %d Origen: %d Long: %d\n", offset, i*TAMSIMPLESORT * 2, H_indices[i]+1);
		cudaMemcpy(v_H + offset, destino + i*TAMSIMPLESORT * 2, sizeof(dato)*(H_indices[i]+1), cudaMemcpyDeviceToHost);
		offset += H_indices[i];
	}
	if (DEBUG) printf("NUM ELEMENTOS = %d\n", offset);
	cudaFree(v_D);
	cudaFree(destino);
	cudaFree(fronteras);
	cudaFree(indices);
}

int main()
{
	dato *vector;
	vector = (dato *)malloc(N * sizeof(dato));

	srand(time(NULL));
	for (int i = 0; i<N; i++)
		vector[i] = rand();
	bool correcto = true;

	struct timeval t1, t2;

	printf("Ordenando vector de %d elementos...\n", N);

	gettimeofday(&t1, 0);

	runsort(vector, N);

	gettimeofday(&t2, 0);

	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Ordenado en %f ms \n", time);

	dato aux = 0;
	for (int i = 0; i<N; i++) {
		if (aux > vector[i]) {
			correcto = false;
		}
		aux = vector[i];
		if (PRINTARRAY) printf("%d - %f\n", i, aux);
	}

	if (correcto) {
		printf("El vector se ha ordenado correctamente.\n\n");
	}
	else {
		printf("Ha fallado la ordenacion del vector.\n\n");
	}
	return 0;
}
