#include "Constantes.cuh"

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
__global__ void calculaFitness(float * valoresEsperados, float * valoresObtenidos, float* fitness, int numMediciones);