#include <stdio.h>


#include <iostream>
#include <fstream>
#include "Constantes.cuh"

/*
This method receive "nombre", which will be the name of a file in
relative path, numvar variables, which will be the number of variables
that will be read from each line, being NUMMEDICIONES the number of
lines that will be on the file. After being read, the data will be
copyied on the "tabla". The first line of the file will not be read
but skipped because of the ".csv" extension usualli uses it in order
to name each column.
*/
void leeArch(float * tabla, char * nombre, int numvar);