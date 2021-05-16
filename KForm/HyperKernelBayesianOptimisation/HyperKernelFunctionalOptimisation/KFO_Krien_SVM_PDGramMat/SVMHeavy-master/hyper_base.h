
//
// Hypervolume functions - basic forms
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <math.h>
#include <stddef.h>
#include <iostream>

#ifndef _hyper_base_h
#define _hyper_base_h

// Calculates hypervolume, hypervolume increase and expected hypervolume
// increase as per AISTATS paper.
//
// No cacheing, no optimisation.
//
// h: calculate hypervolume enclosed by X (in positive quadrant)
// hi: calculates hypervolume increase when y is added to X
// ehi: calculates expected hypervolume increase when y has mean mu, var s
//      (note that s = sigma^2)
//
// numVects (M): number of vectors
// dim (n): dimension of vectors
// X: an array of M points to n dimensional double arrays
//
// xmax: is defined, X[j][i] is replaced by max(X[j][i],xmax[i]).
// axisind: if defined X[j][i] -> X[j][axisind[i]], likewise xmax.
// iind: an integer array with M elements (pre-allocate for speed)

double  h (double **X, int numVects, int dim,                        int *iind = NULL);
double  hi(double **X, int numVects, int dim, double *y,             int *iind = NULL);
double ehi(double **X, int numVects, int dim, double *mu, double *s, int *iind = NULL);

double h(double **X, double *xmax, int numVects, int dim, int *axisind, int *iind = NULL);

// Relevant constants
//
// NUMBASE_PI          pi
// NUMBASE_SQRT1ON2:   1/srqt(2)
// NUMBASE_1ONSQRT2PI: 1/srqt(2.pi)

#define NUMBASE_PI          3.14159265358979323846264338328
#define NUMBASE_SQRT1ON2    0.70710678118654752440084436210
#define NUMBASE_1ONSQRT2PI  0.39894228040143267793994605993

#endif
