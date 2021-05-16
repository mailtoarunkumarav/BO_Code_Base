
//
// Hypervolume functions
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "hyper_opt.h"
#include "hyper_base.h"
#include "hyper_psc.h"
#include "basefn.h"
#include "numbase.h"
#include <math.h>
#include <stddef.h>
#include <iostream>


hyper_cache *make_cache_int(double **X, int *kappa, int M, int n, double **E, int **Estat, int ecnt, int *iind, int d);
double calcE(int i, int j, double **X, double *mu, double *s, double **E, int **Estat, int ecnt);
double dS(double **X, double *mu, double *s, hyper_cache *hc, int ecnt);
double dS(double **X, int *kappa, int M, int n, double *mu, double *s, double **E, int **Estat, int ecnt, int *iind, int d = 0);





hyper_cache *make_cache(double **X, int M, int n)
{
    NiceAssert( X );
    NiceAssert( M >= 0 );
    NiceAssert( n );

    hyper_cache *res = NULL;

    if ( n > 0 )
    {
        int *kappa;
        int *iind;
        double **E;
        int **Estat;

        MEMNEWARRAY(kappa,int,M+1);
        MEMNEWARRAY(iind,int,M);
        MEMNEWARRAY(E,double *,M+1);
        MEMNEWARRAY(Estat,int *,M+1);

        NiceAssert( kappa );
        NiceAssert( iind );
        NiceAssert( E );
        NiceAssert( Estat );

        int i,j;

        for ( i = 0 ; i <= M ; i++ )
        {
            kappa[i] = i;
            MEMNEWARRAY(E[i],double,n);
            MEMNEWARRAY(Estat[i],int,n);

            NiceAssert( E[i] );
            NiceAssert( Estat[i] );

            for ( j = 0 ; j < n ; j++ )
            {
                Estat[i][j] = 0;
            }
        }

        res = make_cache_int(X,kappa,M,n,E,Estat,0,iind,0);

        MEMDELARRAY(kappa);
        MEMDELARRAY(iind);
    }

    else
    {
        MEMNEW(res,hyper_cache);

        res->isbase   = 1;
        res->dim      = n;
        res->numVects = M;
        res->ecnt     = 0;

        res->X = X;

        n *= -1;

        MEMNEWARRAY(res->E,double *,M+1);
        MEMNEWARRAY(res->Estat,int *,M+1);

        NiceAssert( res->E );
        NiceAssert( res->Estat );

        int i,j;

        for ( i = 0 ; i <= M ; i++ )
        {
            MEMNEWARRAY((res->E)[i],double,n);
            MEMNEWARRAY((res->Estat)[i],int,n);

            NiceAssert( (res->E)[i] );
            NiceAssert( (res->Estat)[i] );

            for ( j = 0 ; j < n ; j++ )
            {
                (res->Estat)[i][j] = 0;
            }
        }

        MEMNEWARRAY(res->kappal,int,M+1); // we'll use this as kappa
        MEMNEWARRAY(res->kappau,int,M+1); // we'll use this as iind
    }

    return res;
}

void del_cache(hyper_cache *hc)
{
    NiceAssert( hc );

    int j;

    if ( hc->isbase )
    {
        for ( j = 0 ; j <= hc->numVects ; j++ )
        {
            MEMDELARRAY((hc->E)[j]);
            MEMDELARRAY((hc->Estat)[j]);
        }

        MEMDELARRAY(hc->E);
        MEMDELARRAY(hc->Estat);
    }

    if ( hc->dim > 0 )
    {
        if ( hc->dim > 1 )
        {
            for ( j = 0 ; j <= hc->m ; j++ )
            {
                del_cache((hc->next)[j]);
            }
        }

        MEMDEL(hc->next);
    }

    MEMDELARRAY(hc->kappal);
    MEMDELARRAY(hc->kappau);

    MEMDEL(hc);

    return;
}

double ehi(double *mu, double *s, hyper_cache *hc)
{
    NiceAssert( mu );
    NiceAssert( s );
    NiceAssert( hc );

    (hc->ecnt)++;

    double res = 0;

    if ( hc->dim > 0 )
    {
        res = dS(hc->X,mu,s,hc,hc->ecnt);
    }

    else
    {
        int i;

        for ( i = 0 ; i <= hc->numVects ; i++ )
        {
            (hc->kappal)[i] = i;
        }

        res = dS(hc->X,hc->kappal,hc->numVects,-(hc->dim),mu,s,hc->E,hc->Estat,hc->ecnt,hc->kappau);
    }

    return res;
}

void ehi(double *res, int Nres, double **X, int M, int n, double **mu, double **s)
{
    NiceAssert( res );
    NiceAssert( Nres >= 0 );
    NiceAssert( M >= 0 );
    NiceAssert( n );
    NiceAssert( mu );
    NiceAssert( s );

    if ( Nres )
    {
        int j;

        hyper_cache *hc = make_cache(X,M,n);

        for ( j = 0 ; j < Nres ; j++ )
        {
            res[j] = ehi(mu[j],s[j],hc);
        }

        del_cache(hc);
    }

    return;
}





































double dS(double **X, double *mu, double *s, hyper_cache *hc, int ecnt)
{
    double res = 0;
    double V,L;

    if ( hc->dim == 1 )
    {
        res = calcE((hc->kappal)[hc->m],(hc->dim)-1,X,mu,s,hc->E,hc->Estat,ecnt);
    }

    else
    {
        int j;

        for ( j = hc->m ; j >= 0 ; j-- )
        {
            V = dS(X,mu,s,(hc->next)[j],ecnt);

            if ( j == hc->m )
            {
                L = calcE((hc->kappal)[j],(hc->dim)-1,X,mu,s,hc->E,hc->Estat,ecnt);
            }

            else
            {
                L = calcE((hc->kappal)[j],(hc->dim)-1,X,mu,s,hc->E,hc->Estat,ecnt)
                  - calcE((hc->kappau)[j],(hc->dim)-1,X,mu,s,hc->E,hc->Estat,ecnt);
            }

            //NiceAssert( L >= 0 );

            res += L*V;
        }
    }

    return res;
}

double dS(double **X, int *kappa, int M, int n, double *mu, double *s, double **E, int **Estat, int ecnt, int *iind, int d)
{
    double res = 0;

    // X -> dom(X), M = |dom(X)|

    M = dom(X,kappa,M,n);

    // Sort on n-1 axis, smallest to largest

    axissort(X,kappa,M,n);

    // Cluster at slice edges (sets iind, see previous), set m = number of
    // slices.

    int m = cluster(X,kappa,iind,M,n);

    if ( n == 1 )
    {
        // Base case

        NiceAssert( ( m == 0 ) || ( m == 1 ) );

        int j;
        int i = M-1;

        j = m;
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

            int il = ( i >= 0   ) ? kappa[i  ] : -1;
            //int iu = ( i <  M-1 ) ? kappa[i+1] : -1;

            res = calcE(il,n-1,X,mu,s,E,Estat,ecnt);
        }
    }

    else
    {
        // Documentation - see hyper_base

        int j;
        int i = M-1;
        double L,V;

        for ( j = m ; j >= 0 ; j-- )
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

            int il = ( i >= 0   ) ? kappa[i  ] : -1;
            int iu = ( i <  M-1 ) ? kappa[i+1] : -1;

            if ( j == m )
            {
                L = calcE(il,n-1,X,mu,s,E,Estat,ecnt);
            }

            else
            {
                L = calcE(il,n-1,X,mu,s,E,Estat,ecnt)
                  - calcE(iu,n-1,X,mu,s,E,Estat,ecnt);
            }

            NiceAssert( L >= 0 );

            V = dS(X,kappa+(i+1),M-(i+1),n-1,mu,s,E,Estat,ecnt,iind+(i+1),d+1);
            res += L*V;
        }
    }

    return res;
}

hyper_cache *make_cache_int(double **X, int *kappa, int M, int n, double **E, int **Estat, int ecnt, int *iind, int d)
{
    hyper_cache *res;

    MEMNEW(res,hyper_cache);

    res->numVects = M; // Must be size of X (only matters as isbase level)
    res->X        = X;

    // X -> dom(X), M = |dom(X)|

    M = dom(X,kappa,M,n);

    // Sort on n-1 axis, smallest to largest

    axissort(X,kappa,M,n);

    // Cluster at slice edges (sets iind, see previous), set m = number of
    // slices.

    int m = cluster(X,kappa,iind,M,n);

    res->isbase = ( d == 0 ) ? 1 : 0;
    res->dim = n;
    res->m   = m;

    res->E      = E;
    res->Estat  = Estat;
    res->ecnt   = ecnt;
    MEMNEWARRAY(res->kappal,int,m+1);
    MEMNEWARRAY(res->kappau,int,m+1);
    MEMNEWARRAY(res->next,hyper_cache *,m+1);

    if ( n == 1 )
    {
        // Base case

        NiceAssert( ( m == 0 ) || ( m == 1 ) );

        int j;
        int i = M-1;

        for ( j = m ; j >= 0 ; j-- )
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

            (res->kappal)[j] = ( i >= 0   ) ? kappa[i  ] : -1;
            (res->kappau)[j] = ( i <  M-1 ) ? kappa[i+1] : -1;
            (res->next)[j] = NULL;
        }
    }

    else
    {
        // Documentation - see hyper_base

        int j;
        int i = M-1;

        for ( j = m ; j >= 0 ; j-- )
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

            (res->kappal)[j] = ( i >= 0   ) ? kappa[i  ] : -1;
            (res->kappau)[j] = ( i <  M-1 ) ? kappa[i+1] : -1;
            (res->next)[j] = make_cache_int(X,kappa+(i+1),M-(i+1),n-1,E,Estat,ecnt,iind+(i+1),d+1);
        }
    }

    return res;
}

double calcE(int i, int j, double **X, double *mu, double *s, double **E, int **Estat, int ecnt)
{
    if ( Estat[i+1][j] < ecnt )
    {
        double varadj = sqrt(s[j]);
        double l = ( ( i >= 0 ) ? X[i][j] : 0 ) - mu[j];
        double lsqmod = NUMBASE_SQRT1ON2*l/varadj;

        Estat[i+1][j] = ecnt;
        E[i+1][j] = (NUMBASE_1ONSQRT2PI*varadj*exp(-lsqmod*lsqmod))
                  + ((l/2)*(erf(lsqmod)-1));

//        NiceAssert( E[i+1][j] >= -1e-6 );

        if ( E[i+1][j] < 0 )
        {
            E[i+1][j] = 0;
        }
    }

    return E[i+1][j];
}
