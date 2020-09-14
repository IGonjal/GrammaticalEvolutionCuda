#include "LeeArchivos.cuh"

/*
	This method receive "nombre", which will be the name of a file in
	relative path, numvar variables, which will be the number of variables
	that will be read from each line, being NUMMEDICIONES the number of
	lines that will be on the file. After being read, the data will be
	copyied on the "tabla". The first line of the file will not be read
	but skipped because of the ".csv" extension usualli uses it in order
	to name each column.
*/
void leeArch(float * tabla, char * nombre, int numVar)
{

	std::ifstream myfile;
	//nombre = ".\\patient4_Month_01_Day_01.csv";
	myfile.open(nombre , std::ios::in);

	//Saltamos la primera línea
	myfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');



	char ch;
	//int dummy;

	float y;
	float x;
	float z;
	float w;

	int i = 0;

	//float tabla[CANTIDADMEDICIONES * 4];

	//variablesEntrada[medicion * NUMENTRADAS]

	while (i < NUMMEDICIONES){
		switch (numVar){
		case 1:
			myfile >> y >> ch >> x;
			break;
		case 2:
			myfile >> y >> ch >> x >> ch >> z;
			break;
		case 3:
			myfile >> y >> ch >> x >> ch >> z >> ch >> w;
		}
		switch (numVar){
		case 3:
			tabla[3 + i*NUMENTRADAS] = w;
			
		case 2:
			tabla[2 + i*NUMENTRADAS] = z;
		case 1:
			tabla[1 + i*NUMENTRADAS] = x;
			tabla[    i*NUMENTRADAS] = y;
	
		}

			//dummy >> ch, dummy, ch, dummy >> ch >> ch >> dummy >> ch >> dummy >> ch;

		
		
		

		myfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		/*
		printf("%i, %i, %f, %i ----", t, gluc, ins, carb);
		printf("%f, %f, %f, %f", tabla[i], tabla[i + CANTIDADMEDICIONES], tabla[i + 2 * CANTIDADMEDICIONES], tabla[i + 3 * CANTIDADMEDICIONES]);
		printf("\n");
		*/
		i++;
	}


	myfile.close();
}