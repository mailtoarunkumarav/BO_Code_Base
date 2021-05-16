
//
// Hypervolume functions - Prune, Sort, Cluster
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _hyper_psc_h
#define _hyper_psc_h

// Functions related to pruning, sorting and clustering.
//
//
// xdomy(x,y,dim)
// xdomy(x,y,xmax,dim,axisind)
//
// Return 1 if x dominates y (that is, x[i] >= y[i] for all i).
// Return 0 otherwise.
// dim: dimension of x.
// xmax: is defined, x[i] is replaced by max(x[i],xmax[i]).  Likewise y.
// axisind: if defined x[i] -> x[axisind[i]], likewise xmax and y.
//
//
//
//
// dom(X,numVects,dim)
// dom(X,kappa,numVects,dim)
// dom(X,xmax,numVects,dim,axisind)
//
// Sorts X so that the dominant set is 0->Mret-1, rest Mret->numVects-1.
// If kappa is not present then pointers X[i] are sorted.
// If kappa is present then X[i] -> X[kappa[i]] and kappa is sorted.
// numVects is number of vectors in X.
// Mret is returned (number of vectors in dominant set).
// dim: dimension of X[i] for all i.
// xmax,axisind: see xdomy.
//
//
//
//
// axissort(X,numVects,dim)
// axissort(X,kappa,numVects,dim)
// axissort(X,xmax,numVects,dim,axisind)
//
// Sorts X from smallest to largest on n-1 axis.
// If kappa is not present then pointers X[i] are sorted.
// If kappa is present then X[i] -> X[kappa[i]] and kappa is sorted.
// See above for arguments.
//
//
//
//
// cluster(X,iind,numVects,dim)
// cluster(X,kappa,iind,numVects,dim)
// cluster(X,xmax,iind,numVects,dim,axisind)
//
// Sets iind to the form:
//
// [ 0 ... 0 1 ... 1 2 ... 2 ... m-1 ... m-1 ]
//
// where:
//
// for all i0_k such that iind[i0_k] = 0, X[i0_0][n-1] = X[i0_1][n-1] = ...
// for all i1_k such that iind[i1_k] = 1, X[i1_0][n-1] = X[i1_1][n-1] = ...
// ...
// x[i0_0][n-1] < x[i1_0][n-1] < ...
//
// It is assumed that X[i] is sorted such that X[0][n-1] <= X[1][n-1] <= ...
// prior to calling.  Uses either pointer sorting or indexed depending on
// whether kappa is present.  All arguments as for previous functions.
// Return value is number of clusters m.

int xdomy(double *x, double *y,               int dim              );
int xdomy(double *x, double *y, double *xmax, int dim, int *axisind);

int dom(double **X,               int numVects, int dim              );
int dom(double **X, int *kappa,   int numVects, int dim              );
int dom(double **X, double *xmax, int numVects, int dim, int *axisind);

void axissort(double **X,               int numVects, int dim              );
void axissort(double **X, int *kappa,   int numVects, int dim              );
void axissort(double **X, double *xmax, int numVects, int dim, int *axisind);

int cluster(double **X,               int *iind, int numVects, int dim              );
int cluster(double **X, int *kappa,   int *iind, int numVects, int dim              );
int cluster(double **X, double *xmax, int *iind, int numVects, int dim, int *axisind);


#endif
