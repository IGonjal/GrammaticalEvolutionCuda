#include "Ordenacion.cuh"

/*
	Checks if a value is valid. The n=n comparation is because the NaN numbers
	return false, the other two comparations proove a positive or negative
	infinite number.
*/
__device__ bool isValid(float n){
	return !(isnan(n) || isinf(n) || n != n || n <= -FLT_MAX || n >= FLT_MAX);

}

/*
	This method of sorting will take a logarithmic time runing on parallel,
	the ordenation uses a "bitonic sorting" algorithm.
	
	Sorts an array that represents a matrix taking the quality of each row. 
	The array size must be NUMCROMOSOMAS * SIZECROMOSOMAS, two constants 
	included in "Constantes.cuh".

	"Orden" will be an empty array which should messure NUMCROMOSOMAS. It will
	return the positions before the sorting. "fitnes" will be the array which
	will messure the quality of the solutions, and will be the fild which will
	mark the order. "poblacion" will be sorted following the fitness.
*/
__global__ void ordenar(int * orden, float * fitness, int *poblacion){
	int i = 0;
	int j = 0;
	float f = 0.0;
	int aux = 0;

	//initial orden registered (1, 2, 3...)
	orden[threadIdx.x] = threadIdx.x;
	//Logarithm on base 2 of numElements
	for (i = 2; i <= NUMCROMOSOMAS; i = i * 2){
		// descending from i reducing to half each iteration
		for (j = i; j >= 2; j = j / 2){
			__syncthreads();
			if (threadIdx.x % j  < j / 2){
				// ascending or descending consideration using (threadIdx.x % (i*2) < i) 
				if ((threadIdx.x % (i * 2) < i) && (fitness[threadIdx.x] >  fitness[threadIdx.x + j / 2] || !isValid(fitness[threadIdx.x])) ||
					((threadIdx.x % (i * 2) >= i) && (fitness[threadIdx.x] <= fitness[threadIdx.x + j / 2] || !isValid(fitness[threadIdx.x + j / 2])))){
					aux = orden[threadIdx.x];
					orden[threadIdx.x] = orden[threadIdx.x + j / 2];
					orden[threadIdx.x + j / 2] = aux;
					//fitness reubicated
					f = fitness[threadIdx.x];
					fitness[threadIdx.x] = fitness[threadIdx.x + j / 2];
					fitness[threadIdx.x + j / 2] = f;
				}
			}
			__syncthreads();
		}
	}

	//Here the sorting happens. The marked chromosomes to change, take place
	for (int i = 0; i < SIZECROMOSOMAS;i++){
		__syncthreads();
		aux = poblacion[orden[threadIdx.x] * SIZECROMOSOMAS + i];
		__syncthreads();
		poblacion[threadIdx.x*SIZECROMOSOMAS + i] = aux;
	}
	
} 

/*
	This algorithm represents a perfect shuffle alforitm that will execute
	"rondas" times. it will receive an empty "orden" array which will be used 
	as an auxiliar array and that will return the new position of every chromosome.
*/
__global__ void barajar(int * orden, float * fitness, int *poblacion, int rondas){
	

	if (threadIdx.x < NUMELITISTAS || threadIdx.x >=NUMCROMOSOMAS - NUMELITISTAS) return;

	int aux = 0;
	

	orden[threadIdx.x] = threadIdx.x;
	for (int i = 0; i<rondas; i++){
		if (threadIdx.x > NUMELITISTAS && threadIdx.x < NUMCROMOSOMAS - NUMELITISTAS){
			if (threadIdx.x < NUMCROMOSOMAS / 2){

				aux = threadIdx.x * 2 - 1;

				if (aux > NUMCROMOSOMAS - NUMELITISTAS){
					aux = NUMELITISTAS + aux - (NUMCROMOSOMAS - NUMELITISTAS);
				}
				orden[threadIdx.x] = aux;

			}
			else{

				aux = ((threadIdx.x % (NUMCROMOSOMAS / 2)) * 2) + 1;

				if (aux > NUMCROMOSOMAS - NUMELITISTAS){
					aux = NUMCROMOSOMAS - NUMELITISTAS - (NUMELITISTAS - aux);
				}
				orden[threadIdx.x] = aux;

			}

		}
	}
	float a;
	
	a = fitness[orden[threadIdx.x]];
	__syncthreads();
	fitness[threadIdx.x] = a;

	for (int i = 0; i < SIZECROMOSOMAS; i++){
	
		aux = poblacion[orden[threadIdx.x] * SIZECROMOSOMAS + i];
		__syncthreads();
		poblacion[threadIdx.x*SIZECROMOSOMAS + i] = aux;
		
	}
}
