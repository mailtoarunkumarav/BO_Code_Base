
//
// Hypervolume functions
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "hyper_debug.h"
#include "basefn.h"
#include <math.h>
#include <stddef.h>
#include <iostream>

void pspace(int d)
{
    while ( d > 0 )
    {
        errstream() << "    ";
        d--;
    }

    return;
}

void printX(double **X, double *y, int M, int n, int d)
{
    if ( M )
    {
        int i;

        for ( i = 0 ; i < M ; i++ )
        {
            pspace(d);
            errstream() << "X[" << i << "] = [ ";

            if ( n )
            {
                int j;

                for ( j = 0 ; j < n ; j++ )
                {
                    errstream() << "\t" << X[i][j];
                }
            }

            errstream() << " ]\n";
        }
    }

    if ( y )
    {
        pspace(d);
        errstream() << "y = [";

        if ( n )
        {
            int j;

            for ( j = 0 ; j < n ; j++ )
            {
                errstream() << "\t" << y[j];
            }
        }

        errstream() << " ]\n";
    }

    return;
}

void printX(double **X, double *y, int *kappa, int M, int n, int d)
{
    if ( M )
    {
        int i;

        for ( i = 0 ; i < M ; i++ )
        {
            pspace(d);
            errstream() << "X[" << i << "] = [ ";

            if ( n )
            {
                int j;

                for ( j = 0 ; j < n ; j++ )
                {
                    errstream() << "\t" << X[kappa[i]][j];
                }
            }

            errstream() << " ]\n";
        }
    }

    if ( y )
    {
        pspace(d);
        errstream() << "y = [";

        if ( n )
        {
            int j;

            for ( j = 0 ; j < n ; j++ )
            {
                errstream() << "\t" << y[kappa[j]];
            }
        }

        errstream() << " ]\n";
    }

    if ( kappa )
    {
        pspace(d);
        errstream() << "kappa = [";

        if ( M )
        {
            int j;

            for ( j = 0 ; j < M ; j++ )
            {
                errstream() << "\t" << kappa[j];
            }
        }

        errstream() << " ]\n";
    }

    return;
}

void printX(double **X, double *y, int M, int n, int d, int *axisind)
{
    if ( M )
    {
        int i;

        for ( i = 0 ; i < M ; i++ )
        {
            pspace(d);
            errstream() << "X[" << i << "] = [ ";

            if ( n )
            {
                int j;

                for ( j = 0 ; j < n ; j++ )
                {
                    errstream() << "\t" << X[i][axisind[j]];
                }
            }

            errstream() << " ]\n";
        }
    }

    if ( y )
    {
        pspace(d);
        errstream() << "y = [";

        if ( n )
        {
            int j;

            for ( j = 0 ; j < n ; j++ )
            {
                errstream() << "\t" << y[axisind[j]];
            }
        }

        errstream() << " ]\n";
    }

    return;
}



