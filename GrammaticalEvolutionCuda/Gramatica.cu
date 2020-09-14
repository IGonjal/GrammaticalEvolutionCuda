#include "Gramatica.cuh"
#include "assert.h"

/*
Checks if a value is valid. The n=n comparation is because the NaN numbers
return false, the other two comparations proove a positive or negative
infinite number.
*/
__device__ bool esValido(float a){
	bool ret = true;
	if (a >= FLT_MAX || a <= FLT_MIN || a!=a)
		ret = false;;

	return ret;
}

/*
	This method represents the following grammaar

	<program> ::=		<expression>

	<expression> ::=	<expression> <op> <expression>
					|	<unop> <expr>
					|	<var>

	<op> ::=			+
					|	-
					|	*

	
	<unop> ::=			sin
					|	cos
					|	exp
					|   log

	<var> ::=			x


			
	So the method given an array will take the content on their positions
	and will calculate the grammatical evolution and, simultaneously and
	recursively, the result of the equation it has. because of that, it 
	needs to receive a "variables" array which will have NUMVAR variables.
*/
__device__ float expresion(int * poblacion, int * i, float * variables, int currCrom, int sizeCromosomas){

	if (*i >= sizeCromosomas)
		return FLT_MAX;

	int gen = poblacion[currCrom + (*i)] % NUMEXP;
	*i += 1;

	switch (gen){
		
	case 0: // OPERADOR BINARIO
		gen = poblacion[currCrom + (*i)] % NUMOPS;
		*i += 1;

		float a = expresion(poblacion, i, variables, currCrom,sizeCromosomas);
		float b = expresion(poblacion, i, variables, currCrom, sizeCromosomas);
		if (a >= FLT_MAX || b >= FLT_MAX) return FLT_MAX;

		switch (gen){
		case 0:
			return  a * b;
		case 1:
			return a + b;
		case 2:
			return a - b;

		}
		
		
	case   1: //VARIABLE
		gen = poblacion[currCrom + (*i)] % NUMVAR;
		*i += 1;

		switch (gen){
		case 0:
			return variables[0];
		case 1:
			return variables[1];
		case 2:
			return variables[2];
		}
		
	case 2: //OPERADOR UNARIO
		gen = poblacion[currCrom + (*i)] %NUMUNOPS;
		*i += 1;

		float c = expresion(poblacion, i, variables, currCrom, sizeCromosomas);
		if (c >= FLT_MAX) return FLT_MAX;

		switch (gen){
			case 0: //SIN(EXP)
				return sinf(c);
			case 1: //COS(EXP)
				return cosf(c);
			case 2:
				return exp(c);
			case 3:
				return log(c);
		}
	}
	//Control line, it will never show, if it does, the conditions must be reviewed
	printf("*****************************************************************************************************************************************************************************");
}




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
__global__ void descifra(int * poblacion, float * variables, float * valoresObtenidos, int sizeCromosomas, int numCromosomas, int numMediciones){

	int indiceAdn = 0;
	float var[NUMVAR] = { 0 };

	
		for (int cromosoma = 0; cromosoma < NUMCROMOSOMAS; cromosoma++){
			indiceAdn = 0;
			
			
			
				var[0] = variables[1 + NUMENTRADAS*threadIdx.x];
			if (NUMVAR > 1)
				var[1] = variables[2 + NUMENTRADAS*threadIdx.x];
			if (NUMVAR > 2)
				var[2] = variables[3 + NUMENTRADAS*threadIdx.x];

			/*

			NUMMEDICIONES
			  |
			  |_____ NUMCROMOSOMAS


			*/
			valoresObtenidos[threadIdx.x + numMediciones*cromosoma] = expresion(poblacion, &indiceAdn, var, cromosoma*sizeCromosomas, sizeCromosomas);
		}
		
	

}

/*
	This method does the same that te method  "expresion" but without 
	executing the operations. The idea behind that is being able to 
	translate the productions used to human language.
*/
__host__ void traduceAux(int *adn, char * c, int* i){
	if (*i >= sizeCromosomas){
		strcat(c, "INDICE EXCEDIDO");
		return;
	}
	int gen = adn[*i] % NUMEXP;
	*i += 1;



	switch (gen){
		//<expr><op><expr>
	case 0:
		gen = adn[*i] % NUMOPS;
		*i += 1;
		strcat(c, "(");
		traduceAux(adn, c, i);
		strcat(c, (gen == 0 ? "*" : (gen == 1 ? "+" :  "-" )));
		traduceAux(adn, c, i);
		strcat(c, ")");
		break;

		//<var>
	case 1:
		gen = adn[*i] % NUMVAR;
		*i += 1;
		strcat(c, (gen == 0 ? "x" : (gen == 1 ? "z" : "w")));
		break;
		//<unop>
	case 2:
		gen = adn[*i] % NUMUNOPS;
		*i += 1;
		strcat(c, (gen == 0 ? "sin(" :(gen ==1)? "cos(" :(gen==2)? "exp(": "log("));
		traduceAux(adn, c, i);
		strcat(c, ")");
		break;
	}

}
/*
	This method will decrypt a chromosome into human language
	without actually executing the operations.
*/
__host__ void traduceGramaticaAHumano(int*adn, char* cad){
	int i = 0;
	strcpy(cad, "");
	traduceAux(adn, cad, &i);
}