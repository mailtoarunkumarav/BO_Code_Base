
//
// Hypervolume functions - debugging
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _hyper_debug_h
#define _hyper_debug_h


void pspace(int d);
void printX(double **X, double *y,             int numVects, int dim, int d              );
void printX(double **X, double *y, int *kappa, int numVects, int dim, int d              );
void printX(double **X, double *y,             int numVects, int dim, int d, int *axisind);


#endif
