#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 8388608

int main(){
	struct timeval t1, t2;
	int *hA, *hB, *hC; //Host Arrays

	//Reserva de memoria
	hA = (int*)malloc(N*sizeof(int));
	hB = (int*)malloc(N*sizeof(int));
	hC = (int*)malloc(N*sizeof(int));

	//Inicialización
	srand(time(NULL));
	for (int i = 0; i < N; i++){
		hA[i] = rand();
		hB[i] = rand();
	}

	gettimeofday(&t1, 0);

	//Suma de vectores
	for(int i = 0; i < N; i++){
		hC[i] = hA[i] + hB[i];
	}

	gettimeofday(&t2, 0);

	//Comprobación de errores
	int error = 0;
	for(int i = 0; i < N; i++){
		if(hC[i] != hA[i] + hB[i]){
			error = 1;
			break;
		}
	}

	if(error)
		printf("La suma de vectores ha fallado.\n");
	else
		printf("Suma de vectores correcta.\n");

	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Tiempo: %f ms\n", time);

	free(hA);
	free(hB);
	free(hC);
}
