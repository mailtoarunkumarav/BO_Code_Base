
//
// Hypervolume functions - alternative methods
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _hyper_alt_h
#define _hyper_alt_h

// Operation essentially the same as hyper_base.  Methods are:
//
// ehi_hup: based on:
//
//    1 Hupkens et al 2015, Faster exact algorithms for computing expected
//      hypervolume improvement. In International Conference on Evolutionary
//      Multi-Criterion Optimization, pg 65-79, Springer.
//
// ehi_cou: based on:
//
//    2 Couckuyt et al 2014, Fast Calculation of multiobjective probability
//      of improvement and expected improvement criteria for pareto
//      optimization.  Journal of Global Optimization, 60(3):575-594

double ehi_hup(double **X, int numVects, int dim, double *mu, double *s);
double ehi_cou(double **X, int numVects, int dim, double *mu, double *s);

void ehi_hup(double *res, int Nres, double **X, int numVects, int dim, double **mu, double **s);
void ehi_cou(double *res, int Nres, double **X, int numVects, int dim, double **mu, double **s);

#endif
