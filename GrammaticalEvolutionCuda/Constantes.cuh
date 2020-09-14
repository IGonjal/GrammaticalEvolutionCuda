//Cuda
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

//Standard C
#include <stdio.h>
#include <cfloat>
#include <ctime>

//This value must be defined, and with a number between 1 and 4
#define FUNCIONUSADA 5
//If this value is undefined, the lecture will be generated randomly
#define LEE_DESDE_ARCHIVO

#ifndef LEE_DESDE_ARCHIVO

#define FUNCIONUSADA 0
#define NUMFUNCIONES 5
#endif


#define ARCH1 ".\\Tablas entrada\\y=x+7.csv"
#define ARCH2 ".\\Tablas entrada\\x^2+x-3.csv"
#define ARCH3 ".\\Tablas entrada\\x^3+2x^2-x-2.csv"
#define ARCH4 ".\\Tablas entrada\\sin(x+2).csv"
#define ARCH5 ".\\Tablas entrada\\x^4+x^3+2x^2+x.csv"



//Selection
#define PROBABILIDAD_SUPERVIVENCIA_DEBIL 2


#if (FUNCIONUSADA==1)
#define CURRARCH ARCH1
#define MAXIMONUMEROGENERACIONES 1000
#elif(FUNCIONUSADA==2)
#define CURRARCH ARCH2
#define MAXIMONUMEROGENERACIONES 1000
#elif(FUNCIONUSADA==3)
#define MAXIMONUMEROGENERACIONES 10000
#define CURRARCH ARCH3
#elif(FUNCIONUSADA==4)
#define MAXIMONUMEROGENERACIONES 1000
#define CURRARCH ARCH4
#else
#define MAXIMONUMEROGENERACIONES 3000
#define CURRARCH ARCH5
#endif

//Chromosome
#define NUMCROMOSOMAS 512
#define SIZECROMOSOMAS 256
#define MAXVALORCROMOSOMA 255

//Meditions
#define NUMENTRADAS 2 // y, x
#define NUMMEDICIONES 288
#define MAXVALORCARBOHIDRATOS 400
#define MAXVALORINSULINA 400

//Fitness
#define FITNESSESPERADO 0.0

//Elitism
#define NUMELITISTAS 4


//Refreshing Generation values
#define PROBABILIDADREFRESCO 10
#define NUMGENERACIONESREFRESCO 100

//Crossover
#define TASACRUCE 65

//Mutation
#define TASAMUTACION 5
#define NUMMUTACIONES 7

//Selection
#define PROBABILIDAD_SUPERVIVENCIA_DEBIL 2

//Grammar

#define NUMOPS 3
#define NUMVAR (NUMENTRADAS-1)
#define NUMUNOPS 4
#define NUMEXP 3

//Shuffle
#define MAXRRONDASBARAJADO 100

//String max size
#define LONGITUDMAXCADENA 500


//Grammar
extern int funcionUsada;

//Refresh of chromosomes
extern int probabilidadRefresco;
extern int numGeneracionesRefresco;

//Name of the input file
extern char currArch[512];

//Chromosome values
extern int sizeCromosomas;
extern int numCromosomas;
extern int maxValorCromosoma;

//Points represented
extern int numMediciones;

//Minimum Fitness
extern float fitnessEsperado;

//Elitism
extern int numElitistas;

//Generations
extern int maximoNumeroGeneraciones;

//Crossover
extern int tasaCruce;

//Mutation
extern int tasaMutacion;
extern int numMutaciones;

extern int probabilidadSupervivenciaDebil;

extern int longitudMaxCadena;

