
//
// Hypervolume functions - Prune, Sort, Cluster
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "hyper_psc.h"
#include "basefn.h"
#include <math.h>
#include <stddef.h>
#include <iostream>

// Dominance test.

int xdomy(double *x, double *y, int n)
{
    int i;
    int res = 1;

    if ( n )
    {
        for ( i = 0 ; i < n ; i++ )
        {
            if ( x[i] < y[i] )
            {
                res = 0;
                break;
            }
        }
    }

    return res;
}

int xdomy(double *x, double *y, double *xmax, int n, int *axisind)
{
    int i;
    int res = 1;

    if ( n )
    {
        for ( i = 0 ; i < n ; i++ )
        {
            if ( ( ( x[axisind[i]] < xmax[axisind[i]] ) ? x[axisind[i]] : xmax[axisind[i]] ) < ( ( y[axisind[i]] < xmax[axisind[i]] ) ? y[axisind[i]] : xmax[axisind[i]] ) )
            {
                res = 0;
                break;
            }
        }
    }

    return res;
}













// Pruning function.

int dom(double **X, int M, int n)
{
    int i,j,k;
    double *xtemp;

    // NB: we just switch pointers here, putting the dominated points at
    // the end.  DO NOT just overwrite pointers as it is important that
    // they be kept (they might still be used by the function that called
    // this one).

    if ( M > 1 )
    {
        for ( i = 0 ; i < M ; i++ )
        {
            // Reversed inner count gives small speedup as it slightly
            // increases the change that j == m in multiple swaps, in
            // which case we avoid the need to switch pointers.

            for ( j = M-1 ; j >= 0 ; j-- )
            {
                if ( i < j )
                {
                    if ( xdomy(X[i],X[j],n) )
                    {
                        // Decrement vector count, switch to X[j] to end

                        M--;

                        if ( j < M )
                        {
                            xtemp = X[j];
                            X[j] = X[M];
                            X[M] = xtemp;
                        }
                    }
                }

                else if ( i > j )
                {
                    if ( xdomy(X[i],X[j],n) )
                    {
                        // Decrement vector count, switch to X[j] to end

                        i--;
                        M--;

                        xtemp = X[j];

                        for ( k = j ; k < M ; k++ )
                        {
                            X[k] = X[k+1];
                        }

                        X[M] = xtemp;
                    }
                }
            }
        }
    }

    return M;
}

int dom(double **X, int *kappa, int M, int n)
{
    int i,j,k,l;

    if ( M > 1 )
    {
        for ( i = 0 ; i < M ; i++ )
        {
            for ( j = M-1 ; j >= 0 ; j-- )
            {
                if ( i < j )
                {
                    if ( xdomy(X[kappa[i]],X[kappa[j]],n) )
                    {
                        M--;

                        if ( j < M )
                        {
                            l = kappa[j];
                            kappa[j] = kappa[M];
                            kappa[M] = l;
                        }
                    }
                }

                else if ( i > j )
                {
                    if ( xdomy(X[kappa[i]],X[kappa[j]],n) )
                    {
                        i--;
                        M--;

                        l = kappa[j];

                        for ( k = j ; k < M ; k++ )
                        {
                            kappa[k] = kappa[k+1];
                        }

                        kappa[M] = l;
                    }
                }
            }
        }
    }

    return M;
}

int dom(double **X, double *xmax, int M, int n, int *axisind)
{
    int i,j,k;
    double *xtemp;

    if ( M > 1 )
    {
        for ( i = 0 ; i < M ; i++ )
        {
            for ( j = M-1 ; j >= 0 ; j-- )
            {
                if ( i < j )
                {
                    if ( xdomy(X[i],X[j],xmax,n,axisind) )
                    {
                        M--;

                        if ( j < M )
                        {
                            xtemp = X[j];
                            X[j] = X[M];
                            X[M] = xtemp;
                        }
                    }
                }

                else if ( i > j )
                {
                    if ( xdomy(X[i],X[j],xmax,n,axisind) )
                    {
                        // Decrement vector count, switch to X[j] to end

                        i--;
                        M--;

                        xtemp = X[j];

                        for ( k = j ; k < M ; k++ )
                        {
                            X[k] = X[k+1];
                        }

                        X[M] = xtemp;
                    }
                }
            }
        }
    }

    return M;
}


















// Sorting function

void axissort(double **X, int M, int n)
{
    int i,j;
    double *xtemp;

    if ( n && ( M > 1 ) )
    {
        for ( i = 0 ; i < M-1 ; i++ )
        {
            for ( j = i+1 ; j < M ; j++ )
            {
                if ( X[j][n-1] < X[i][n-1] )
                {
                    xtemp = X[i];
                    X[i] = X[j];
                    X[j] = xtemp;
                }
            }
        }
    }

    return;
}

void axissort(double **X, int *kappa, int M, int n)
{
    int i,j,k;

    if ( n && ( M > 1 ) )
    {
        for ( i = 0 ; i < M-1 ; i++ )
        {
            for ( j = i+1 ; j < M ; j++ )
            {
                if ( X[kappa[j]][n-1] < X[kappa[i]][n-1] )
                {
                    k = kappa[i];
                    kappa[i] = kappa[j];
                    kappa[j] = k;
                }
            }
        }
    }

    return;
}

void axissort(double **X, double *xmax, int M, int n, int *axisind)
{
    NiceAssert( n );

    int i,j;
    double *xtemp;

    if ( n && ( M > 1 ) )
    {
        for ( i = 0 ; i < M-1 ; i++ )
        {
            for ( j = i+1 ; j < M ; j++ )
            {
                if ( ( ( X[j][axisind[n-1]] < xmax[axisind[n-1]] ) ? X[j][axisind[n-1]] : xmax[axisind[n-1]] ) < ( ( X[i][axisind[n-1]] < xmax[axisind[n-1]] ) ? X[i][axisind[n-1]] : xmax[axisind[n-1]] ) )
                {
                    xtemp = X[i];
                    X[i] = X[j];
                    X[j] = xtemp;
                }
            }
        }
    }

    return;
}






















// Clustering function.

int cluster(double **X, int *iind, int M, int n)
{
    NiceAssert( n );

    int i;
    int m = 0;

    if ( M )
    {
        for ( i = 0 ; i < M ; i++ )
        {
            if ( !i )
            {
                m++;
            }

            else
            {
                if ( X[i][n-1] > X[i-1][n-1] )
                {
                    m++;
                }
            }

            iind[i] = m-1;
        }
    }

    return m;
}

int cluster(double **X, int *kappa, int *iind, int M, int n)
{
    NiceAssert( n );

    int i;
    int m = 0;

    if ( M )
    {
        for ( i = 0 ; i < M ; i++ )
        {
            if ( !i )
            {
                m++;
            }

            else
            {
                if ( X[kappa[i]][n-1] > X[kappa[i-1]][n-1] )
                {
                    m++;
                }
            }

            iind[i] = m-1;
        }
    }
    return m;
}

int cluster(double **X, double *xmax, int *iind, int M, int n, int *axisind)
{
    NiceAssert( n );

    int i;
    int m = 0;

    if ( M )
    {
        for ( i = 0 ; i < M ; i++ )
        {
            if ( !i )
            {
                m++;
            }

            else
            {
                if ( ( ( X[i][axisind[n-1]] < xmax[axisind[n-1]] ) ? X[i][axisind[n-1]] : xmax[axisind[n-1]] ) > ( ( X[i-1][axisind[n-1]] < xmax[axisind[n-1]] ) ? X[i-1][axisind[n-1]] : xmax[axisind[n-1]] ) )
                {
                    m++;
                }
            }

            iind[i] = m-1;
        }
    }

    return m;
}
