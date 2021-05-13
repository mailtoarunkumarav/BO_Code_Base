
//
// Hypervolume functions
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "hyper_base.h"
#include "hyper_psc.h"
#include "hyper_debug.h"
#include "basefn.h"
#include "numbase.h"
#include <math.h>
#include <stddef.h>
#include <iostream>



// Internal versions
//
// d: used by debugging.  Defines level of recursion

double sS(double **X, int M, int n, int *iind, int d = 0);
double dS(double **X, int M, int n, double *y, int *iind, int d = 0);
double dS(double **X, int M, int n, double *mu, double *s, int *iind, int d = 0);

double sS(double **X, double *xmax, int M, int n, int *axisind, int *iind, int d = 0);







// External functions.

double h(double **X, int M, int n, int *iind)
{
    NiceAssert( X );
    NiceAssert( M >= 0 );
    NiceAssert( n >= 1 );

    int lociind = iind ? 0 : 1;

    if ( lociind )
    {
        MEMNEWARRAY(iind,int,M);
    }

    NiceAssert( iind );

    double res = sS(X,M,n,iind);

    if ( lociind )
    {
        MEMDELARRAY(iind);
    }

    return res;
}

double hi(double **X, int M, int n, double *y, int *iind)
{
    NiceAssert( X );
    NiceAssert( M >= 0 );
    NiceAssert( n >= 1 );
    NiceAssert( y );

    int lociind = iind ? 0 : 1;

    if ( lociind )
    {
        MEMNEWARRAY(iind,int,M);
    }

    NiceAssert( iind );

    double res = dS(X,M,n,y,iind);

    if ( lociind )
    {
        MEMDELARRAY(iind);
    }

    return res;
}

double ehi(double **X, int M, int n, double *mu, double *s, int *iind)
{
    NiceAssert( X );
    NiceAssert( M >= 0 );
    NiceAssert( n >= 1 );
    NiceAssert( mu );
    NiceAssert( s );

    int lociind = iind ? 0 : 1;

    if ( lociind )
    {
        MEMNEWARRAY(iind,int,M);
    }

    NiceAssert( iind );

    double res = dS(X,M,n,mu,s,iind);

    if ( lociind )
    {
        MEMDELARRAY(iind);
    }

    return res;
}

double h(double **X, double *xmax, int M, int n, int *axisind, int *iind)
{
    NiceAssert( X );
    NiceAssert( xmax );
    NiceAssert( M >= 0 );
    NiceAssert( n >= 1 );
    NiceAssert( axisind );

    int lociind = iind ? 0 : 1;

    if ( lociind )
    {
        MEMNEWARRAY(iind,int,M);
    }

    NiceAssert( iind );

    double res = sS(X,xmax,M,n,axisind,iind);

    if ( lociind )
    {
        MEMDELARRAY(iind);
    }

    return res;
}







































// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------

// Actual code start here
//
// Notes:
//
//  -  we use n-1 axis here so that we can just keep passing the same
//     pointers.  If we recursed on 0 as per the paper then we would
//     either need to realloc X at each recursion (memory hit, slow)
//     or do some funky pointer arithmetic X[i] = X[i]++ (harder to
//     read, not needed, slight speed hit).
//
//  -  the order of recursion has been reversed.  We
//     justify this with the observation that the order of data in
//     X_j can be shuffled provided that we don't then need to access
//     X_{j+1}, X_{j+2}, ....  Reversed order recursion means that
//     this condition will be met when recursion is needed, so we can
//     just pass in the pointers and not worry.  Otherwise we would
//     need to allocate fresh memory and copy, which would be slow and
//     use up way too much memory.
//
//  -  some variable names are a bit mixed up compared to the paper.



double sS(double **X, int M, int n, int *iind, int d)
{
    double res = 0;

    // X -> dom(X), M = |dom(X)|

    M = dom(X,M,n);

    // Sort on n-1 axis, smallest to largest

    axissort(X,M,n);

    // Cluster at slice edges (sets iind, see previous), set m = number of
    // slices.

    int m = cluster(X,iind,M,n);

    if ( n == 1 )
    {
        // Base case

        // Calculate L.  If M > 0 then m = 1 and L is the length, otherwise
        // L = 0.

        int j;

        if ( M )
        {
            for ( j = 0 ; j < M ; j++ )
            {
                if ( X[j][n-1] > res )
                {
                    res = X[j][n-1];
                }
            }
        }
    }

    else
    {
        // General case

        // Calculate startpoint (endpoint in paper) for recursion.

        int p = m-1;

        // Recursive calculation:

        int i = M-1; // this will index the end of i^(p-1)
        int j;
        double l = 0;
        double u = 0;
        double L = 0;
        double V = 0;

        if ( p >= 0 )
        {
            for ( j = p ; j >= 0 ; j-- )
            {
                if ( !j )
                {
                    i = -1;
                }

                else
                {
                    while ( iind[i] >= j )
                    {
                        i--;

                        NiceAssert( i >= 0 );
                    }
                }

                l = ( i >= 0  ) ? X[i  ][n-1] : 0;
                u = ( i < M-1 ) ? X[i+1][n-1] : 0;

                L = u-l;

                NiceAssert( L >= 0 );

                V = sS(X+(i+1),M-(i+1),n-1,iind+(i+1),d+1);
                res += L*V;
            }
        }
    }

    return res;
}

double dS(double **X, int M, int n, double *y, int *iind, int d)
{
    double res = 0;

    // X -> dom(X), M = |dom(X)|

    M = dom(X,M,n);

    // Sort on n-1 axis, smallest to largest

    axissort(X,M,n);

    // Cluster at slice edges (sets iind, see previous), set m = number of
    // slices.

    int m = cluster(X,iind,M,n);

    if ( n == 1 )
    {
        // Base case

        // Result is increase in length

        int j;

        if ( M )
        {
            for ( j = 0 ; j < M ; j++ )
            {
                if ( X[j][n-1] > res )
                {
                    res = X[j][n-1];
                }
            }
        }

        res = ( y[n-1] > res ) ? y[n-1]-res : 0;
    }

    else
    {
        int j;
        double l = 0;
        double u = 0;
        double L = 0;
        double V = 0;

        // Calculate startpoint (endpoint in paper) for recursion.
        // Need to locate y for HI

        j = 0;

        if ( M )
        {
            for ( j = 0 ; j < M ; j++ )
            {
                if ( y[n-1] < X[j][n-1] )
                {
                    break;
                }
            }
        }

        int p = ( j < M ) ? iind[j] : m;

        // Recursive calculation:

        int i = M-1; // this will index the end of i^(p-1)

        if ( p >= 0 )
        {
            for ( j = p ; j >= 0 ; j-- )
            {
                if ( !j )
                {
                    i = -1;
                }

                else
                {
                    while ( iind[i] >= j )
                    {
                        i--;

                        NiceAssert( i >= 0 );
                    }
                }

                l = ( i >= 0  ) ? X[i  ][n-1] : 0;
                u = ( i < M-1 ) ? X[i+1][n-1] : 0;

                L = u-l;

                if ( j == p )
                {
                    L = y[n-1]-l;
                }

                NiceAssert( L >= 0 );

                V = dS(X+(i+1),M-(i+1),n-1,y,iind+(i+1),d+1);
                res += L*V;
            }
        }
    }

    return res;
}

double dS(double **X, int M, int n, double *mu, double *s, int *iind, int d)
{
    double res = 0;
    double varadj = sqrt(s[n-1]);

    // X -> dom(X), M = |dom(X)|

    M = dom(X,M,n);

    // Sort on n-1 axis, smallest to largest

    axissort(X,M,n);

    // Cluster at slice edges (sets iind, see previous), set m = number of
    // slices.

    int m = cluster(X,iind,M,n);

    if ( n == 1 )
    {
        // Base case

        // Result is expected increase in length

        int j;

        double l = 0;
        double lsqmod = 0;

        if ( M )
        {
            for ( j = 0 ; j < M ; j++ )
            {
                if ( X[j][n-1] > l )
                {
                    l = X[j][n-1];
                }
            }
        }

        l -= mu[n-1];
        lsqmod = NUMBASE_SQRT1ON2*l/varadj;

        res = (NUMBASE_1ONSQRT2PI*varadj*exp(-lsqmod*lsqmod))
            + ((l/2)*(erf(lsqmod)-1));
    }

    else
    {
        int j;
        double L = 0;
        double V = 0;
        double l = 0;
        double u = 0;
        double lsqmod,usqmod;

        // Calculate startpoint (endpoint in paper) for recursion.
        // Iteration length is m+1 for ehi

        int p = m;

        // Recursive calculation:

        int i = M-1; // this will index the end of i^(p-1)

        if ( p >= 0 )
        {
            for ( j = p ; j >= 0 ; j-- )
            {
                if ( !j )
                {
                    i = -1;
                }

                else
                {
                    while ( iind[i] >= j )
                    {
                        i--;

                        NiceAssert( i >= 0 );
                    }
                }

                l = ( i >= 0  ) ? X[i  ][n-1] : 0;
                u = ( i < M-1 ) ? X[i+1][n-1] : 0;

                V = dS(X+(i+1),M-(i+1),n-1,mu,s,iind+(i+1),d+1);

                if ( j == m )
                {
                    l -= mu[n-1];
                    lsqmod = NUMBASE_SQRT1ON2*l/varadj;

                    L = (NUMBASE_1ONSQRT2PI*varadj*exp(-lsqmod*lsqmod))
                      + ((l/2)*(erf(lsqmod)-1));
                }

                else
                {
                    l -= mu[n-1];
                    lsqmod = NUMBASE_SQRT1ON2*l/varadj;

                    u -= mu[n-1];
                    usqmod = NUMBASE_SQRT1ON2*u/varadj;

                    L = (NUMBASE_1ONSQRT2PI*varadj*exp(-lsqmod*lsqmod))
                      + ((l/2)*(erf(lsqmod)-1))
                      - (NUMBASE_1ONSQRT2PI*varadj*exp(-usqmod*usqmod))
                      - ((u/2)*(erf(usqmod)-1));
                }

                NiceAssert( L >= 0 );

                res += L*V;
            }
        }
    }

    return res;
}


double sS(double **X, double *xmax, int M, int n, int *axisind, int *iind, int d)
{
    double res = 0;

    // X -> dom(X), M = |dom(X)|

    M = dom(X,xmax,M,n,axisind);

    // Sort on n-1 axis, smallest to largest

    axissort(X,xmax,M,n,axisind);

    // Cluster at slice edges (sets iind, see previous), set m = number of
    // slices.

    int m = cluster(X,xmax,iind,M,n,axisind);

    if ( n == 1 )
    {
        // Base case

        // Calculate L.  If M > 0 then m = 1 and L is the length, otherwise
        // L = 0.

        int j;

        if ( M )
        {
            for ( j = 0 ; j < M ; j++ )
            {
                if ( ( ( X[j][axisind[n-1]] < xmax[axisind[n-1]] ) ? X[j][axisind[n-1]] : xmax[axisind[n-1]] ) > res )
                {
                    res = ( X[j][axisind[n-1]] < xmax[axisind[n-1]] ) ? X[j][axisind[n-1]] : xmax[axisind[n-1]];
                }
            }
        }
    }

    else
    {
        // General case

        // Calculate startpoint (endpoint in paper) for recursion.

        int p = m-1;

        // Recursive calculation:

        int i = M-1; // this will index the end of i^(p-1)
        int j;
        double l = 0;
        double u = 0;
        double L = 0;
        double V = 0;

        if ( p >= 0 )
        {
            for ( j = p ; j >= 0 ; j-- )
            {
                if ( !j )
                {
                    i = -1;
                }

                else
                {
                    while ( iind[i] >= j )
                    {
                        i--;

                        NiceAssert( i >= 0 );
                    }
                }

                l = ( i >= 0  ) ? ( ( X[i  ][axisind[n-1]] < xmax[axisind[n-1]] ) ? X[i  ][axisind[n-1]] : xmax[axisind[n-1]] ) : 0;
                u = ( i < M-1 ) ? ( ( X[i+1][axisind[n-1]] < xmax[axisind[n-1]] ) ? X[i+1][axisind[n-1]] : xmax[axisind[n-1]] ) : 0;

                L = u-l;

                NiceAssert( L >= 0 );

                V = sS(X+(i+1),xmax,M-(i+1),n-1,axisind,iind+(i+1),d+1);
                res += L*V;
            }
        }
    }

    return res;
}

