#include "Genetico.cuh"
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
__global__ void cruce(int * poblacion, int * cruzamos, int * ptCruce, curandState* state, int avance, int * cruza_muta, int sizeCromosomas, int numCromosomas, int numElitistas, int tasaCruce){

	curandState localState = state[threadIdx.x];
	int aux = 0;

	//Points of crossover defined and ordered
	int pt1 = curand(&localState) % sizeCromosomas;
	int pt2 = curand(&localState) % sizeCromosomas;

	if (pt1 > pt2){
		aux = pt1;
		pt2 = pt1;
		pt1 = aux;
	}

	//number of elements on pair crossing block
	int lim = numCromosomas / avance;
	lim = lim % 2 == 0 ? lim : lim - 1;

	//beyond the pair overcrossing point 
	int ini = avance*lim;
	int fin = numCromosomas;
	int mitad = (fin - ini) / 2;


	//one of each two threads before the "lim" point is who stablishes if and where the cross happens
	if (threadIdx.x <numElitistas || threadIdx.x < ini){
		if (threadIdx.x % (2 * avance) < avance){

			ptCruce[threadIdx.x] = pt1;
			ptCruce[threadIdx.x + avance] = pt1;
			ptCruce[threadIdx.x + numCromosomas] = pt2;
			ptCruce[threadIdx.x + avance + numCromosomas] = pt2;

			//Not crossing
			if (curand(&localState) % 100 >= tasaCruce){
				cruzamos[threadIdx.x] = threadIdx.x;
				cruzamos[threadIdx.x + avance] = threadIdx.x + avance;
				cruza_muta[threadIdx.x] = 0;
				cruza_muta[threadIdx.x + avance] = 0;
			}
			//Crossing
			else{
				cruza_muta[threadIdx.x] = 1;
				cruza_muta[threadIdx.x + avance] = 1;
				cruzamos[threadIdx.x] = threadIdx.x + avance;
				cruzamos[threadIdx.x + avance] = threadIdx.x;
			}

		}
		
	}
	
	else{
		
		//Beyond the "lim" point, one of each two threads stablishes if and where the cross happens
		if (threadIdx.x < ini + mitad){

			ptCruce[threadIdx.x] = pt1;
			ptCruce[threadIdx.x + avance] = pt1;
			ptCruce[threadIdx.x + numCromosomas] = pt2;
			ptCruce[threadIdx.x + avance + numCromosomas] = pt2;
			//Not crossing
			if (curand(&localState) % 100 >= tasaCruce){
				cruzamos[threadIdx.x] = threadIdx.x;
				cruzamos[threadIdx.x + mitad] = threadIdx.x + mitad;
				cruza_muta[threadIdx.x] = 0;
				cruza_muta[threadIdx.x + mitad] = 0;
			}
			//Crossing
			else{
				cruza_muta[threadIdx.x] = 1;
				cruza_muta[threadIdx.x + mitad] = 1;
				cruzamos[threadIdx.x] = threadIdx.x + mitad;
				cruzamos[threadIdx.x + mitad] = threadIdx.x;
			}
		}
	}
	//Elitism
	if (threadIdx.x < numElitistas){
		cruza_muta[threadIdx.x] = 0;
		cruzamos[threadIdx.x] = threadIdx.x;
	}
	
	//Crossover really happens here
	for (int i = 0; i < sizeCromosomas; i++){

		aux = poblacion[threadIdx.x*sizeCromosomas + i];

		//if on crossover point and the chromosome must cross, it does
		if (i >= ptCruce[threadIdx.x] && i <= ptCruce[threadIdx.x + numCromosomas])
			aux = poblacion[cruzamos[threadIdx.x]*sizeCromosomas + i];
		//thread sync in order to not overwrite the crossing chromosome befor it
		//could has been read
		__syncthreads();
		poblacion[threadIdx.x*sizeCromosomas + i] = aux;
	}
}

/*
	This method receives a population, a array of curandState and
	a "cruza_muta" array which will store if it mutates

	<<<1, NUMCROMOSOMAS >>>

	The way of working is picking every chromosome, getting a random
	number and if it is below the mutate treshold, it will mutate in
	NUMMUTACIONES random points
*/
__global__ void mutacion(int * poblacion, curandState* state, int * cruza_muta, int numCromosomas, int tasaMutacion, int numElitistas, int numMutaciones, int sizeCromosomas, int maxValorCromosoma){
	curandState localState = state[threadIdx.x];

	cruza_muta[threadIdx.x+numCromosomas] = 0;

	if (curand(&localState) % 100 < tasaMutacion && threadIdx.x >= numElitistas){
		cruza_muta[threadIdx.x + numCromosomas] = 1;
		for (int i = 0; i < numMutaciones; i++){
			poblacion[(curand(&localState) % sizeCromosomas) + threadIdx.x * sizeCromosomas] = curand(&localState) % maxValorCromosoma;
		}
	}
}

/*
	Will receive a population, the fitness asociated to every chromosome and
	a curandState will initialized.

	<<<1, NUMCROMOSOMAS >>>

	this method will face every chromosome to another one. If it's fitness is
	better, the best one will replace the worst one. There is an small posibility
	of the bad one will survive, and it has a percentage of PROBABILIDAD_SUPERVIVENCIA_DEBIL
*/
__global__ void seleccion(int* poblacion, float * fitness, curandState* state, int probabilidadSupervivenciaDebil,int numCromosomas,int numElitistas, int sizeCromosomas){
	curandState localState = state[threadIdx.x];	

	int ganaPeor = (curand(&localState) % 100 < probabilidadSupervivenciaDebil);

	int yo = threadIdx.x;
	int otro = curand(&localState) % numCromosomas;

	//The another randomly choosed chromosome is who will survive
	if ((threadIdx.x >= numElitistas) &&
		((!ganaPeor && fitness[yo] > fitness[otro]) ||
		(ganaPeor && fitness[yo] <= fitness[otro])))
	{
		yo = otro;
	}
	//This chromosome is who survive
	else
	{
		otro = yo;
	}

	//Note that despite doing the swap ever, if this chromosome
	//is who survive, yo == otro and there will not be a real swap
		fitness[yo] = fitness[otro];

		yo = yo * sizeCromosomas;
		otro = otro * sizeCromosomas;

		for (int i = 0; i < sizeCromosomas; i++){
			__syncthreads();
			poblacion[yo + i] = poblacion[otro + i];	
		}
	

}