
//
// Hypervolume functions - optimised form
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <math.h>
#include <stddef.h>
#include <iostream>

#ifndef _hyper_opt_h
#define _hyper_opt_h

class hyper_cache;
class hyper_cache
{
public:
    // isbase: 1 if base node, 0 otherwise
    // dim: dimension at this layer
    // m: see paper
    // numVects: number of vectors
    //
    // E: erf/exp cache (all point back to base node)
    // kappal: index of l vectors (-1 is zero, m+1 dim)
    // kappau: index of u vectors (-1 is zero, m+1 dim)
    // next: next level hypercaches
    //
    // NB: if n is negative then this is a partial cache with dimension
    //     equal to -n.  A partial cache has E and Estat caches set but
    //     does not pre-compute kappal, kappau etc.

    int isbase;
    int dim;
    int m;
    int numVects;

    double **E;
    int **Estat;
    int ecnt;
    int *kappal;
    int *kappau;
    hyper_cache **next;

    double **X;
};

// Cache construction and destruction functions
//
// Set n = -n to only allocate E cache
// delX: 0 don't delete X, 1 do delete X

hyper_cache *make_cache(double **X, int numVects, int dim);
void del_cache(hyper_cache *hc);

// quick-swap function

inline void qswap(hyper_cache *&a, hyper_cache *&b);
inline void qswap(hyper_cache *&a, hyper_cache *&b)
{
    hyper_cache *c;

    c = a;
    a = b;
    b = c;

    return;
}

// Optimised version using full kappa and E cache

double ehi(double *mu, double *s, hyper_cache *hc);

// Optimised batch version.  Works by:
//
// Generate cache on X
// Run tests
// Delete cache

void ehi(double *res, int Nres, double **X, int numVects, int dim, double **mu, double **s);

#endif
