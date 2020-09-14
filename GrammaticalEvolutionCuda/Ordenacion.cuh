#include "Constantes.cuh"

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
__global__ void ordenar(int * orden, float * fitness, int *poblacion);

/*
	This algorithm represents a perfect shuffle alforitm that will execute
	"rondas" times. it will receive an empty "orden" array which will be used
	as an auxiliar array and that will return the new position of every chromosome.
*/
__global__ void barajar(int * orden, float * fitness, int *poblacion, int rondas);