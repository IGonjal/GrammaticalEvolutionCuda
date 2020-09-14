
#include "Constantes.cuh"
#include "Ordenacion.cuh"
#include "Genetico.cuh"
#include "Gramatica.cuh"
#include "fitness.cuh"
#include "LeeArchivos.cuh"
#include "argumentsParser.cuh"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



/*
	This method will populate the initial chromosomes.

	<<<1, NUMCROMOSOMAS >>>
*/
__global__ void poblar(int * adn, curandState * state, float * fitnessDevice, int maxValorCromosoma, int sizeCromosomas){

	curandState localState = state[threadIdx.x];
	for (int i = 0; i < sizeCromosomas; i++){
		adn[(threadIdx.x*sizeCromosomas) + i] = curand(&localState) % maxValorCromosoma;
	}
	fitnessDevice[threadIdx.x] = FLT_MAX;
}

/*
	This method will refresh some chromosomes by generating random numbers,
	if the rand is below a certain number "PROBABILIDADDEREFRESCO" it will
	be substituted by a new one.

	<<<1, NUMCROMOSOMAS >>>
*/
__global__ void rePoblar(int * adn, curandState * state, int maxValorCromosomas, int numElitistas, int probabilidadRefresco, int sizeCromosomas){

	curandState localState = state[threadIdx.x];

	if (curand(&localState) % 100 < probabilidadRefresco && threadIdx.x >= numElitistas){
		for (int i = 0; i < sizeCromosomas; i++)
			adn[(threadIdx.x*sizeCromosomas) + i] = curand(&localState) % maxValorCromosomas;
	}
}

#ifndef LEE_DESDE_ARCHIVO
/*
	If "LEE_DESDE_ARCHIVO" is undefined, the input variables will be automatically
	generated by this method

 <<<1, CANTIDADMEDICIONES>>>
*/
__global__ void generaEntradas(float * entradasDevice, curandState* state){

	curandState localState = state[threadIdx.x];


	
		
//		 entradasDevice representa una matriz tridimensional, las coordenadas son las siguientes
//
//			CANTIDADENTRADAS
//			   |
//			   |___  CANTIDADMEDICIONES


		

		//t es tiempo, c es carbohidratos, ins es insulina


		float x = threadIdx.x /10 - 0.5;
		

		float z = curand(&localState) % MAXVALORCARBOHIDRATOS;
		

		float w = curand(&localState) % MAXVALORINSULINA;
		entradasDevice[threadIdx.x                    ] = x;
		entradasDevice[threadIdx.x +     NUMMEDICIONES] = z;
		entradasDevice[threadIdx.x + 2 * NUMMEDICIONES] = w;


		float y;
		switch (FUNCIONUSADA % NUMFUNCIONES){
		case 0:
			y = x*x + x;
			break;
		case 1:
			y = x + x + w;
			break;
		case 2:
			y = x*x+ x + 12;
			break;
		case 3:
			y = w*x + z + 10;
			break;
		case 4:
			y = x + 7;
			break;
		}
		entradasDevice[threadIdx.x + 3 * NUMMEDICIONES] = y;

	
}
#endif
/*
	This method initializes the curandState, in order to generate
	well seeded, parallel and independent random numbers.
	<<<1, NUMCROMOSOMAS>>>
*/
__global__ void setup_cuRand(curandState * state, unsigned long seed)
{
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

/*
	Each generation will be represented here. The operations that happens 
	are the following:
		-selection
		-crossover
		-mutation
		-transcroption+decrypt
		-evaluation
		-Sorting
		-shuffle (except the elitist)
		-copy the best individual

	From this method almost all the parallel methods are called
*/
__host__ float generacion (	int * adnDevice, float *inDevice, float * fitnessDevice, curandState* state, int * cruceAux, 
							int * ptCruce, float * valoresObtenidos, int * orden, int * cruza_muta, char * mejorGramatica, float mejorFitnessAnterior
						  ){

	//char cad[1000];
	char * cad = (char *)malloc(sizeof(char)*longitudMaxCadena);
	cad[longitudMaxCadena - 1] = '\0';
	//Elitist Selection
	seleccion << <1, numCromosomas >> >(adnDevice, fitnessDevice, state, probabilidadSupervivenciaDebil, numCromosomas, numElitistas, sizeCromosomas);
	//Elitist Crossover
	int avance = (rand() % (numCromosomas / 2) -2) + 1 ;
	cruce << <1, numCromosomas >> >(adnDevice, cruceAux, ptCruce, state, avance, cruza_muta, sizeCromosomas, numCromosomas, numElitistas, tasaCruce);

	//Elitist mutation
	mutacion << <1, numCromosomas >> >(adnDevice, state, cruza_muta, numCromosomas, tasaMutacion, numElitistas, numMutaciones, sizeCromosomas, maxValorCromosoma);

	//Translate chromosomes+get values
	descifra << <1, numMediciones >> >(adnDevice, inDevice, valoresObtenidos, sizeCromosomas, numCromosomas, numMediciones);
	

	//Use values to calculate fitness
	calculaFitness << <1, numCromosomas >> >(inDevice, valoresObtenidos, fitnessDevice, numMediciones);

	//Sorting
	ordenar << <1, numCromosomas >> >(orden, fitnessDevice, adnDevice);
	
	



	//Elitist shuffle (does not shuffle the elitist chromosomes)
	avance = 1;
	barajar << <1, numCromosomas >> >(orden, fitnessDevice, adnDevice, avance);

	//Copying the best individual
	//int i[SIZECROMOSOMAS];
	int * i = (int*)malloc(sizeof(int)*sizeCromosomas);
	cudaMemcpy(i, adnDevice, sizeCromosomas*sizeof(int), cudaMemcpyDeviceToHost);
	
	//Translating the best result
	traduceGramaticaAHumano(i, cad);
	cad[longitudMaxCadena - 1] = '\0';

	strcpy(mejorGramatica, cad);

	//Copying best fitness
	float b;
	cudaMemcpy(&b, &(fitnessDevice[0]), sizeof(float), cudaMemcpyDeviceToHost);
	
	free(cad);
	free(i);
	return b;
	//return 0.0;

}

int main(int argc, char ** argv)
{


	char * cad = (char*)malloc(sizeof(char)*longitudMaxCadena);


	parse(argc, argv);


	//La poblacion de adn en CPU y en dispositivo
	//int adnCPU[NUMCROMOSOMAS * SIZECROMOSOMAS];
	int * adnDevice;
	//Los valores de entrada
	//float inCPU [NUMENTRADAS*NUMMEDICIONES];
	float * inCPU = (float *)malloc(sizeof(float)*NUMENTRADAS*numMediciones);

	float * inDevice;
	//Los fitness obtenidos
	float * fitnessDevice;
	//Las mediciones al paciente, representadas con f�rmulas
	//float * valoresEsperados;
	//Los valores obtenidos por la gram�tica
	float * valoresObtenidos;
	//Un array que servir� como array auxiliar para la ordenaci�n
	int * orden;
	//Necesario para compartir informaci�n sobre el cruce entre hilos
	int * cruceAux;
	int * ptCruce;
	//Las variables necesarias para calcular el tiempo
	clock_t begin, end; 
	float time_spent =0;
	int * cruza_muta;
	//int mejorAdnCPU[SIZECROMOSOMAS];
	int * mejorAdnCPU = (int*)malloc(sizeof(int)*sizeCromosomas);
	curandState * randState;
	cudaError_t error;
	//char cad[LONGITUDMAXCADENA];

	error = cudaGetLastError();
	//Setting the device
	error = cudaSetDevice(0);
	if (error != cudaSuccess) {
		if (cudaDeviceReset() != cudaSuccess && cudaSetDevice(0) != cudaSuccess) {
			cudaDeviceReset();
			free(inCPU);
			return 1;
		}
	}

	//cruza_muta
	error = cudaMalloc(&cruza_muta, numCromosomas * 2 * sizeof(int));
	if (error == cudaErrorMemoryAllocation){// cudaSuccess){
		cudaDeviceReset();
		return 1;
	}
	//randState
	error = cudaMalloc(&randState, numCromosomas * sizeof(curandState));
	if (error == cudaErrorMemoryAllocation){
		cudaDeviceReset();
		return 1;
	}
	//adnDevice (poblacion)
	error = cudaMalloc((void **)&adnDevice, numCromosomas * sizeCromosomas * sizeof(int));
	if (error == cudaErrorMemoryAllocation){// cudaSuccess){
		cudaDeviceReset();
		return 1;
	}
	//inDevice (entradas)
	error = cudaMalloc((void **)&inDevice, numMediciones*NUMENTRADAS*sizeof(int));
	if (error == cudaErrorMemoryAllocation){// cudaSuccess){
		cudaDeviceReset();
		return 1;
	}
	//fitnessDevice (el fitness que se obtendr�)
	error = cudaMalloc((void **)&fitnessDevice, numCromosomas*sizeof(float));
	if (error == cudaErrorMemoryAllocation){// cudaSuccess){
		cudaDeviceReset();
		return 1;
	}
	//Los valores que se obtendr�n al comparar las gram�ticas
	error = cudaMalloc((void **)&valoresObtenidos, numMediciones*numCromosomas*sizeof(float));
	if (error == cudaErrorMemoryAllocation){// cudaSuccess){
		cudaDeviceReset();
		return 1;
	}
	//Compartir� si dos cromosomas se cruzan
	error = cudaMalloc((void **)&cruceAux, numCromosomas*sizeof(int));
	if (error == cudaErrorMemoryAllocation){// cudaSuccess){
		cudaDeviceReset();
		return 1;
	}
	//Compartir� los 2 puntos de cruce de un cromosoma
	error = cudaMalloc((void **)&ptCruce, numCromosomas*2*sizeof(int));
	if (error == cudaErrorMemoryAllocation){// cudaSuccess){
		cudaDeviceReset();
		return 1;
	}
	//Orden
	error = cudaMalloc((void **)&orden, numCromosomas * sizeof(int));
	if (error == cudaErrorMemoryAllocation){// cudaSuccess){
		cudaDeviceReset();
		return 1;
	}




	/*POBLANDO VALORES INICIALES*/

		// inicializaci�n cuRand
		setup_cuRand <<<1, numCromosomas>>> (randState, unsigned(time(NULL)));
		// inicializaci�n adnDevice, randState, fitnessDevice(todo a 0)
		poblar <<<1, numCromosomas >>> (adnDevice, randState, fitnessDevice,maxValorCromosoma, sizeCromosomas);
		// Genera las entradas

#ifdef LEE_DESDE_ARCHIVO

		//for (int i = 0; i < NUMDIAS;i++)
			leeArch(inCPU, currArch,NUMVAR);

		cudaMemcpy(inDevice, inCPU, NUMENTRADAS*numMediciones*sizeof(float), cudaMemcpyHostToDevice);
#else

		generaEntradas << <1, NUMMEDICIONES >> >(inDevice, randState);

#endif

	/* VALORES INICIALES POBLADOS */
		time_t t = time(NULL);
		char * nombreArchivo = (char *)malloc(sizeof(char)*longitudMaxCadena);
		struct tm tm = *localtime(&t);
		//a�omesdia_horaminutosegundo_Gramatica_MaxNumGeneraciones_NumCrom
		sprintf(nombreArchivo, ".\\results\\%d%d%d_%d%d%d_%i_%i_%i.csv", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,funcionUsada,maximoNumeroGeneraciones,numCromosomas);

		FILE *pFile;
		pFile = fopen(nombreArchivo, "w+");
		if (pFile == NULL){
			cudaDeviceReset();
			return 1;
		}
		//fprintf(pFile, mensajes);
		printf("\nOptimization Running, please wait...");
		fprintf(pFile, "#filename: ");
		fprintf(pFile, currArch);
		fprintf(pFile, "\n");
		//Imprimimos solo datos. Generaci�n ; tiempo; mejor fitness ; indiv�duo
		fprintf(pFile, "Generation ; time; best fitness ; chromosome\n");

		printf("\a\n");
	/*INICIO DEL RELOJ*/
	begin = clock();
	/******************/

	//La mejor gram�tica de la generaci�n
	//char mejorGramatica[LONGITUDMAXCADENA] = "";
	char * mejorGramatica = (char*)malloc(sizeof(char)*longitudMaxCadena);
	mejorGramatica[0] = '\0';

	/* INICIO DEL BUCLE PRINCIPAL*/
	float curr = FLT_MAX;
	int i = 0;
	while ((curr > fitnessEsperado || curr < 0) && i < maximoNumeroGeneraciones){

		

		curr = generacion(adnDevice, inDevice, fitnessDevice, /*valoresEsperados,*/ randState, cruceAux, ptCruce, valoresObtenidos, orden, cruza_muta, mejorGramatica, curr);

		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

		i += 1;
		//Imprimimos solo datos. Generaci�n ; tiempo; mejor fitness ; indiv�duo
		fprintf(pFile, "%i ; %f ; %.9f ;", i, time_spent, curr);
		fprintf(pFile, mejorGramatica);
		fprintf(pFile, "\n");

		if (i% numGeneracionesRefresco == 0){
			rePoblar << <1, numCromosomas >> >(adnDevice, randState, maxValorCromosoma, numElitistas, probabilidadRefresco, sizeCromosomas);
		}
		// */
		
	}
	/*****************************/


	error = cudaMemcpy(mejorAdnCPU, adnDevice, sizeCromosomas * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		cudaDeviceReset();
		return 1;
	}


	/*TIEMPO TRANSCURRIDO DESDE EL INICIO DEL RELOJ*/
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	/***********************************************/

	fclose(pFile);
	error = cudaDeviceReset();
	if (error != cudaSuccess) {
		cudaDeviceReset();
		return 1;
	}
	
	//while (1);
	printf("End of execution");
	printf("\a\n");

	free(inCPU);
	free(mejorAdnCPU);
	free(mejorGramatica);
	free(cad);
	free(nombreArchivo);

	return 0;
}