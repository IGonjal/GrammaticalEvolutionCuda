#include "Constantes.cuh"

/*
	This method will receive a population of chromosomes, a quantity of
	variables and al empty but allocated in memory array. The utility of
	this function is to decrypt the phenotype of the chromosomes, storing
	the result on "valoresObtenidos". The variables used into the decodification
	step are received via "variables" and the population decrypted will be
	"poblacion".

	<<<1, NUMMEDICIONES >>>

	We'll iterate on the chromosomes and not over the meditions, because the
	threads with every medition will run the same recursive callings in order
	to decrypt, so we'll can execute every medition of a chromosome being
	parallel, so the loop will run over every chromosome, decrypting all it's
	meditions at once.
*/
__global__ void descifra(int * poblacion, float * variables, float * valoresObtenidos, int sizeCromosomas, int numCromosomas, int numMediciones);

/*
	This method will decrypt a chromosome into human language
	without actually executing the operations.
*/
__host__ void traduceGramaticaAHumano(int*adn, char* cad);