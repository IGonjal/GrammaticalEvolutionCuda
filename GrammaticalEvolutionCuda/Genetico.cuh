#include "Constantes.cuh"

/*
	@value poblacion the population we will crossover

	@value cruzamos an empty array which will indicate after
	the execution which cromosoma will cross with this

	@value ptCruce an empty array which will indicate after the
	execution the crossover points chosen (despite the
	chromosome would not cross will have points)

	@value state a curandState array filled and initialized

	@value avance an integer which must be between 1 and
	NUMCROMOSOMAS/2. Will indicate the chromosome
	which will cross together

	@value cruza_muta this array wil register if the chromosome
	actually had crossing

	<<<1, NUMCROMOSOMAS >>>

	This method recieves a population of chromosomes and will
	return into the same variable the population after the
	chromosomes have crossed with another chromosome from the
	same generation.
*/
__global__ void cruce(int * poblacion, int * cruzamos, int * ptCruce, curandState* state, int avance, int * cruza_muta, int sizeCromosomas, int numCromosomas, int numElitistas, int tasaCruce);

/*
	This method receives a population, a array of curandState and
	a "cruza_muta" array which will store if it mutates

	<<<1, NUMCROMOSOMAS >>>

	The way of working is picking every chromosome, getting a random
	number and if it is below the mutate treshold, it will mutate in
	NUMMUTACIONES random points
*/
__global__ void mutacion(int * poblacion, curandState* state, int * cruza_muta, int numCromosomas, int tasaMutacion, int numElitistas, int numMutaciones, int sizeCromosomas, int maxValorCromosoma);

/*
	Will receive a population, the fitness asociated to every chromosome and
	a curandState will initialized.

	<<<1, NUMCROMOSOMAS >>>

	this method will face every chromosome to another one. If it's fitness is
	better, the best one will replace the worst one. There is an small posibility
	of the bad one will survive, and it has a percentage of PROBABILIDAD_SUPERVIVENCIA_DEBIL
*/
__global__ void seleccion(int* poblacion, float * fitness, curandState* state, int probabilidadSupervivenciaDebil, int numCromosomas, int numElitistas, int sizeCromosomas);

