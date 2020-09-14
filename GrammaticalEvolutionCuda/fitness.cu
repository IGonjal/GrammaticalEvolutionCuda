#include "fitness.cuh"

/*
	 Checks if a value is valid. The n=n comparation is because the NaN numbers
	 return false, the other two comparations proove a positive or negative
	 infinite number.
*/
__device__ bool valorValido(float n){
	return ( n == n && n >= -FLT_MAX && n <= FLT_MAX);

}

/*

  This function will take "variablesEntrada" and "valores obtenidos" 
  and will compare them, returning the values into the fitness input.

  ValoresObtenidos
		NUMCROMOSOMAS
		|
		|___ CANTIDADMEDICIONES


	variablesEntrada
		CANTIDADMEDICIONES
		 |
		 |___ CANTIDADENTRADAS
	
  <<<1,NUMCROMOSOMAS>>>

  This execution paralelizes the chromosomes despite the remainder
  functions paralelize the meditions. The way used to calculate the 
  fitness is the sumatory of the absolute diference between the 
  obtained and the expected values.
*/
__global__ void calculaFitness(float * variablesEntrada, float * valoresObtenidos, float* fitness, int numMediciones){
	fitness[threadIdx.x] = 0;
	for (int medicion = 0; medicion < numMediciones; medicion++){

		fitness[threadIdx.x] = fitness[threadIdx.x] + (fabs(variablesEntrada[medicion*NUMENTRADAS] - valoresObtenidos[medicion + threadIdx.x * numMediciones]) / numMediciones);
	}

	if (!valorValido(fitness[threadIdx.x])) fitness[threadIdx.x] = FLT_MAX;


}

