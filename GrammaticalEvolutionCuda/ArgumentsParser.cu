#include "ArgumentsParser.cuh"

//Grammar
int funcionUsada;
//Refresh of chromosomes
int probabilidadRefresco;
int numGeneracionesRefresco;
//Name of the input file
char currArch[512];
//Chromosome values
int sizeCromosomas;
int numCromosomas;
int maxValorCromosoma;
//Points represented
int numMediciones;
//Minimum Fitness
float fitnessEsperado;
//Elitism
int numElitistas;
//Generations
int maximoNumeroGeneraciones;
//Crossover
int tasaCruce;
//Mutation
int tasaMutacion;
int numMutaciones;
int probabilidadSupervivenciaDebil;
int longitudMaxCadena;


void parse(int argc, char ** argv){


	funcionUsada = FUNCIONUSADA;
	probabilidadRefresco = PROBABILIDADREFRESCO;
	numGeneracionesRefresco = NUMGENERACIONESREFRESCO;
	strcpy(currArch, CURRARCH);
	sizeCromosomas = SIZECROMOSOMAS;
	numCromosomas = NUMCROMOSOMAS;
	maxValorCromosoma = MAXVALORCROMOSOMA;
	numMediciones = NUMMEDICIONES;
	fitnessEsperado = FITNESSESPERADO;
	numElitistas = NUMELITISTAS;
	maximoNumeroGeneraciones = MAXIMONUMEROGENERACIONES;
	tasaCruce = TASACRUCE;
	tasaMutacion = TASAMUTACION;
	numMutaciones = NUMMUTACIONES;
	probabilidadSupervivenciaDebil = PROBABILIDAD_SUPERVIVENCIA_DEBIL;
	longitudMaxCadena = LONGITUDMAXCADENA;
	
	char *currArg;

	int i = 1;
	while (i < argc){
		if (strcmp(argv[i], "--help") == 0){
			printf("\nIMPLEMENTACIÓN DE GRAMÁTICA EVOLUTIVA\n");
			printf("Algoritmo de programación genética por Ismael Gonjal Montero, licencia MIT, este programa buscará una función comparando con un archivo de entrada con formato igual a los disponibles en \\Tablas Entrada\n\n");

			printf("Si no se introduce ningún argumento, el programa se ejecutará con los valores por defecto, que pueden consultarse en el archivo Constantes.cuh\n\n\n");

			printf("ARGUMENTOS\n\n");
			
			printf("   FUNCION POR DEFECTO										\nUso: -fu [entero], --funcionPorDefecto [entero]\nModifica la función por defecto utilizada, se representa con un número del 1 al 4 Las funciones por defecto disponibles son: \n   1-  y=x+7\n   2-  x^2+x-3\n   3-  x^3+2x^2-x-2\n   4-  sin(x+2)\n\n");
			printf("   PROBABILIDAD DE REFRESCO									\nUso: -pr [entero], --probabilidadRefresco [entero]\nEl número debe ser un entero entre 0 y 100 e indica la probabilidad de que se refresquen los cromosomas en cada generacion\n\n");
			printf("   NUMERO DE GENERACIONES DE REFRESCO						\nUso: -gr [entero], --numeroGeneracionesRefresco [entero]\nEl número debe ser un entero entre 0 y MAXINT e indica cada cuantas generaciones se produce un refresco en los cromosomas.\n\n");
			printf("   ARCHIVO DE ENTRADA										\nUso: -ae [cadeba], --archivoEntrada [cadeba]\nEl archivo que se va a leer. El archivo por defecto es %s\n\n", CURRARCH);
			printf("   TAMAÑO DE CROMOSOMAS										\nUso: -t [entero],  --tamanioCrom [entero]\nLongitud indica la cantidad de genes de cada cromosoma, será un número entero que debería ser mayor que 30\n\n");
			printf("   TAMAÑO DE LA POBLACION									\nUso: -p [entero],  --tamPob [entero]\nEl tamaño de la población, en CPU el límite es el tamaño maximo de la pila, en GPU la maxima cantidad de hilos ejecutable, por defecto es %i. EN GPU DEBE SER POTENCIA DE DOS\n\n", NUMCROMOSOMAS);
			printf("   MAXIMO VALOR DE CROMOSOMA								\nUso: -ma [entero],  --maxValCrom [entero]\nLongitud indica el valor máximo de un gen del cromosoma. Debe siempre ser mayor que 5 y para tener buenos fines estadísticos un valor mayor, por defecto es %i\n\n",MAXVALORCROMOSOMA);
			printf("   NUMERO DE MEDICIONES										\nUso: -me [entero],  --mediciones [entero]\nCantidad de puntos que se tomarán del archivo. Si el archivo tiene menos o el número es negativo el programa no funcionará correctamente. La cifra por defecto es %i.\n\n", NUMCROMOSOMAS);
			printf("   FITNESS ESPERADO											\nUso: -fi [double], --fitness [double]\nCantidad de puntos que se tomarán del archivo. Si el archivo tiene menos o el número es negativo el programa no funcionará correctamente. La cifra por defecto es %d.\n\n", FITNESSESPERADO);
			printf("   CANTIDAD ELITISTAS										\nUso: -el [entero], --elitistas [entero]\nCantidad de cromosomas elitistas. La cifra por defecto es %i. Los elitistas en GPU deben ser potencia de 2.\n\n", NUMELITISTAS);
			printf("   MAXIMO NUMERO DE GENERACIONES							\nUso: -ge [entero], --generaciones [entero]\nCantidad de generaciones. La cifra por defecto es %i.\n\n", MAXIMONUMEROGENERACIONES);
			printf("   TASA DE CRUCE											\nUso: -cr [entero], --probabilidadCruce [entero]\nProbabilidad de crucamiento. La cifra debe ser un entero entre 0 y 100, por defecto es %i.\n\n", TASACRUCE);
			printf("   TASA DE MUTACIONES										\nUso: -mu [entero], --probabilidadMutacion [entero]\nProbabilidad de mutación. La cifra debe ser un entero entre 0 y 100, por defecto es %i.\n\n", TASAMUTACION);
			printf("   NUMERO DE MUTACIONES										\nUso: -nm [entero], --numeroMutaciones [entero]\nCantidad de genes que mutan en un cromosoma. La cifra debe ser un entero mayor que 0, por defecto es %i.\n\n", NUMMUTACIONES);
			printf("   PROBABILIDAD DE SUPERVIVENCIA DEL MENOS APTO				\nUso: -s [entero],  --supervivenciaDebil [entero]\nEn la selección, probabilidad de que el cromosoma menos apto sobreviva. La cifra debe ser un entero mayor que 0 y menor que 100, debería ser una cifra baja o la evolución se invertirá, por defecto es %i.\n\n", PROBABILIDAD_SUPERVIVENCIA_DEBIL);
			printf("   LONGITUD MAXIMA DE CADENA								\nUso: -l [entero],  --longitudCadena [entero]\nLongitud máxima de las cadenas, incluidas las de la gramática. La cifra por defecto será %i, si no es suficiente, desbordará la cadena y se producirá comportamiento no deseado.\n\n", LONGITUDMAXCADENA);
			printf("Press ENTER to exit");
			printf("\a\n");
			getchar();
			exit(0);
		}
		if (strcmp(argv[i], "-fu") == 0 || strcmp(argv[i], "--funcionPorDefecto") == 0){
			i++;
			sscanf(argv[i], "%i", &funcionUsada);
			i++;
		}
		else if (strcmp(argv[i], "-pr") == 0 || strcmp(argv[i], "--probabilidadRefresco") == 0){
			i++;
			sscanf(argv[i], "%i", &probabilidadRefresco);
			i++;
		}
		else if (strcmp(argv[i], "-gr") == 0 || strcmp(argv[i], "--numeroGeneracionesRefresco") == 0){
			i++;
			sscanf(argv[i], "%i", &numGeneracionesRefresco);
			i++;
		}
		else if (strcmp(argv[i], "-ae") == 0 || strcmp(argv[i], "--archivoEntrada") == 0){
			i++;
			strcpy(currArch,argv[i]);
			//sscanf((char*)argv[i], "%s", currArch);
			i++;
		}
		else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tamanioCrom") == 0){
			i++;
			sscanf(argv[i], "%i", &sizeCromosomas);
			i++;
		}
		else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--tamPob") == 0){
			i++;
			sscanf(argv[i], "%i", &numCromosomas);
			i++;
		}
		else if (strcmp(argv[i], "-ma") == 0 || strcmp(argv[i], "--maxValCrom") == 0){
			i++;
			sscanf(argv[i], "%i", &maxValorCromosoma);
			i++;
		}
		else if (strcmp(argv[i], "-me") == 0 || strcmp(argv[i], "--mediciones") == 0){
			i++;
			sscanf(argv[i], "%i", &numMediciones);
			i++;
		}
		else if (strcmp(argv[i], "-fi") == 0 || strcmp(argv[i], "--fitness") == 0){
			i++;
			sscanf(argv[i], "%lf", &fitnessEsperado);
			i++;
		}
		else if (strcmp(argv[i], "-el") == 0 || strcmp(argv[i], "--elitistas") == 0){
			i++;
			sscanf(argv[i], "%i", &numElitistas);
			i++;
		}
		else if (strcmp(argv[i], "-ge") == 0 || strcmp(argv[i], "--generaciones") == 0){
			i++;
			sscanf((char*)argv[i], "%i", &maximoNumeroGeneraciones);
			i++;
		}
		else if (strcmp(argv[i], "-cr") == 0 || strcmp(argv[i], "--probabilidadCruce") == 0){
			i++;
			sscanf((char*)argv[i], "%i", &tasaCruce);
			i++;
		}
		else if (strcmp(argv[i], "-mu") == 0 || strcmp(argv[i], "--probabilidadMutacion") == 0){
			i++;
			sscanf(argv[i], "%i", &tasaMutacion);
			i++;
		}
		else if (strcmp(argv[i], "-nm") == 0 || strcmp(argv[i], "--numeroMutaciones") == 0){
			i++;
			sscanf(argv[i], "%i", &numMutaciones);
			i++;
		}
		else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--supervivenciaDebil") == 0){
			i++;
			sscanf(argv[i], "%i", &probabilidadSupervivenciaDebil);
			i++;
		}
		else if (strcmp((char*)argv[i], "-l") == 0 || strcmp((char*)argv[i], "--longitudCadena") == 0){
			i++;
			sscanf((char*)argv[i], "%i", &longitudMaxCadena);
			i++;
		}
		else{
			i++;
		}
	}

	printf("Los parámetros en los que está inicializado el programa son:\n");
	printf("Funcion Usada=%i\n",funcionUsada);
	printf("Probabilidad de refresco=%i\n",probabilidadRefresco);
	printf("Númer de generaciones de refresco=%i\n",numGeneracionesRefresco);
	printf("Archivo de entrada="); printf(currArch); printf("\n");
	printf("Tamaño de los cromosomas=%i\n", sizeCromosomas);
	printf("Numero de cromosomas=%i\n",numCromosomas);
	printf("Maximo valor de los cromosomas=%i\n", maxValorCromosoma);
	printf("Numero de mediciones=%i\n",numMediciones);
	printf("Fitness esperado=%lf\n", fitnessEsperado);
	printf("Numero de elitistas=%i\n",numElitistas);
	printf("Maximo numero de generaciones=%i\n",maximoNumeroGeneraciones);
	printf("Tasa de cruce=%i\n",tasaCruce);
	printf("Tasa de mutacion=%i\n",tasaMutacion);
	printf("Numero de mutaciones=%i\n", numMutaciones);
	printf("Probabilidad de supervivencia del más débil=%i\n",probabilidadSupervivenciaDebil);
	printf("Longitud máxima de las cadenas=%i\n", longitudMaxCadena);


}