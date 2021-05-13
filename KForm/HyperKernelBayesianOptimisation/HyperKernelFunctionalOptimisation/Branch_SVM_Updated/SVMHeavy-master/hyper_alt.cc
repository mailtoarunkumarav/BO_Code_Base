
//
// Hypervolume functions
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "hyper_alt.h"
#include "hyper_opt.h"
#include "hyper_base.h"
#include "hyper_psc.h"
#include "hyper_debug.h"
#include "basefn.h"
#include "numbase.h"
#include <math.h>
#include <stddef.h>
#include <iostream>

//#define DEBUG 1

double sS(double **X, double *xmax, int M, int n, int *axisind, int *iind = NULL);
double sS(double **X, double *xmax, int M, int n, int *axisind, int *iind)
{
    return h(X,xmax,M,n,axisind,iind);
}

double ehi_hup(double **X, int M, int n, double *mu, double *s)
{
    double res;
    double *xres = &res;

    ehi_hup(xres,1,X,M,n,&mu,&s);

    return res;
}

double ehi_cou(double **X, int M, int n, double *mu, double *s)
{
    double res;
    double *xres = &res;

    ehi_cou(xres,1,X,M,n,&mu,&s);

    return res;
}

void ehi_hup(double *res, int Nres, double **X, int M, int n, double **mu, double **s)
{
    if ( !Nres )
    {
        return;
    }

    int frank;
    int i,j,k,h;

    for ( frank = 0 ; frank < Nres ; frank++ )
    {
        res[frank] = 0;
    }

    // X -> dom(X), M = |dom(X)|

#ifdef DEBUG
errstream() << "X pre dom sort\n";
printX(X,NULL,M,n,1);
#endif
    M = dom(X,M,n);

#ifdef DEBUG
errstream() << "X post dom sort\n";
printX(X,NULL,M,n,1);
#endif
    // Need to sort along all axis
    //
    // This starts to look somewhat obtuse but the basic idea is not too
    // complex.  Each axis i has the following variables:
    //
    // Xsort[i]: an array of pointers to the vectors.  This will be sorted
    //           smallest to largest based on axis i
    // iind[i]:  like the iind vector in the AISTATS method, but there is
    //           a separate one for each axis.  It represents the clustering
    //           of points along axis i.  It is an increasing vectors where
    //           each number represents one value of Xsort[i] on the axis.
    // m[i]:     the number of clusters on axis i.

    double ***Xsort;
    int **iind;
    int *m;

    MEMNEWARRAY(Xsort,double **,n);
    MEMNEWARRAY(iind,int *,n);
    MEMNEWARRAY(m,int,n);

    NiceAssert( Xsort );
    NiceAssert( iind );
    NiceAssert( m );

    for ( i = 0 ; i < n ; i++ )
    {
        MEMNEWARRAY(Xsort[i],double *,M);
        MEMNEWARRAY(iind[i],int,M);

        NiceAssert( Xsort[i] );
        NiceAssert( iind[i] );

        for ( j = 0 ; j < M ; j++ )
        {
            Xsort[i][j] = X[j];
        }

        // Sort on i axis, smallest to largest

        axissort(Xsort[i],M,i+1);

        // Grouping (sets iind, see previous), set m.

        m[i] = cluster(Xsort[i],iind[i],M,i+1);

        NiceAssert( m[i] >= 0 );
    }
#ifdef DEBUG
int qqqi,qqqj,qqqk;
for ( qqqk = 0 ; qqqk < n ; qqqk++ )
{
errstream() << "Sort on axis " << qqqk << "] = ";
for ( qqqi = 0 ; qqqi < M ; qqqi++ )
{
errstream() << "Xsort[ " << qqqk << "] = ";
for ( qqqj = 0 ; qqqj < n ; qqqj++ )
{
errstream() << Xsort[qqqk][qqqi][qqqj] << "\t";
}
errstream() << "\n";
}
errstream() << "\n";
}
#endif

    // Complicated loopy thing:
    //
    // (should probably do this using a class with an increment operator).
    // The complexity comes from the fact that we can't just have n loops
    // when n isn't known beforehand and we're not using recursion.

    int isdone = 0;
    int isdoneb = 0;
    int locbreak = 0;
    int udomX = 1;
    double dV;
    double a,b,e,f,varadj;
    int isedge = 0;
    int Malt;
    int isval;
    int *ii;
    int *jj;
    double *l;
    double *u;
    int *idummy;
    int *kk;
    double **Xalt;
    double woop;

    MEMNEWARRAY(ii,int,n);
    MEMNEWARRAY(jj,int,n);
    MEMNEWARRAY(l,double,n);
    MEMNEWARRAY(u,double,n);
    MEMNEWARRAY(idummy,int,M);
    MEMNEWARRAY(kk,int,n);
    MEMNEWARRAY(Xalt,double *,M);

    NiceAssert( ii );
    NiceAssert( jj );
    NiceAssert( l );
    NiceAssert( u );
    NiceAssert( idummy );
    NiceAssert( kk );
    NiceAssert( Xalt );

    isedge = 0;

    for ( i = 0 ; i < n ; i++ )
    {
        ii[i] = M-1;
        jj[i] = m[i]; //-1;

        if ( !jj[i] )
        {
            ii[i] = -1;
        }

        else
        {
            while ( iind[i][ii[i]] >= jj[i] )
            {
                ii[i]--;

                NiceAssert( ii[i] >= 0 );
            }
        }

        l[i] = ( ii[i] >= 0  ) ? Xsort[i][ii[i]  ][i] : 0;
        u[i] = ( ii[i] < M-1 ) ? Xsort[i][ii[i]+1][i] : 0;

        isedge |= ( ii[i] == M-1 );
    }

    // Because the loop is complicated we just use a while and set isdone
    // on exit.

    while ( !isdone )
    {
#ifdef DEBUG
errstream() << "l = "; for ( j = 0 ; j < n ; j++ ) { errstream() << l[j] << ", "; } errstream() << "\n";
errstream() << "u = "; for ( j = 0 ; j < n ; j++ ) { errstream() << u[j] << ", "; } errstream() << "\n";
#endif
        // Hypercube is bounded by l,u
        //
        // The first step is to see if the hypercube is already in the
        // dominated subset.  If it is then there is no contribution so
        // we can skip it.

        udomX = 1;

        for ( j = 0 ; j < M ; j++ )
        {
            if ( !isedge )
            {
                if ( xdomy(X[j],u,n) )
                {
                    udomX = 0;
                    break; // Only need to find a single dominating point.
                }
            }
        }

        if ( udomX )
        {
#ifdef DEBUG
errstream() << "Relevant cube found!\n";
#endif
            // Hypercube is not dominated, so go ahead and calculate the
            // contribution.  Note that because of the way that the bounding
            // box [l,u) is constructed we can safely assume that the box
            // is completely within the non-dominated set.

            // We need to loop over all combinations of axis - that is,
            // {}, {i_j}, {i_j,i_k},... where i_j,i_k \in \mathbb{Z}_n and
            // there are no repeats.  Each element will contribute to the
            // hypervolume.  See below for details on this.  Note that the
            // algorithm used will enumerate only valid sets, so there is
            // no need for expensive testing if sets are valid.

            for ( i = 0 ; i <= n ; i++ )
            {
#ifdef DEBUG
errstream() << "Working on C cardinality " << i << "\n";
#endif
                for ( j = 0 ; j < n ; j++ )
                {
                    kk[j] = j;
                }

                isdoneb = 0;

                while ( !isdoneb )
                {
                    {
#ifdef DEBUG
errstream() << "C = ";  if ( i > 0 ) { for ( j = 0 ; j < i ; j++ ) { errstream() << kk[j] << ", "; } } errstream() << "\n";
errstream() << "~C = "; if ( i < n ) { for ( j = i ; j < n ; j++ ) { errstream() << kk[j] << ", "; } } errstream() << "\n";
#endif

                        // Need to calculate contribution of hypercube
                        // [l,u) for axis subset {kk[0],kk[1],...,kk[i-1]}
                        // (or empty set if i = 0).  This is C in Emmerich's
                        // paper.

                        // First calculate ICconst.  Note pointer arithmetic
                        // to ensure the D\C is used and range of loop.

                        dV = 1;

                        if ( i < n )
                        {
                            // Loop for factors:
                            //
                            // ld-rd (where rd = 0 by assumption here)

                            for ( j = i ; j < n ; j++ )
                            {
                                dV *= l[kk[j]];
                            }

#ifdef DEBUG
errstream() << "dV vol = " << dV << "\n";
#endif
                            // Minus dominated restricted sub-hypervolume

                            Malt = 0;

                            for ( k = 0 ; k < M ; k++ )
                            {
                                isval = 1;

                                if ( i )
                                {
                                    for ( j = 0 ; j < i ; j++ )
                                    {
                                        if ( X[k][kk[j]] <= l[kk[j]] )
                                        {
                                            isval = 0;
                                            break;
                                        }
                                    }
                                }

                                if ( isval )
                                {
                                    Xalt[Malt] = X[k];
                                    Malt++;
                                }
                            }

                            dV -= sS(Xalt,l,Malt,n-i,kk+i,idummy);
                        }
#ifdef DEBUG
errstream() << "dV ICconst = " << dV << "\n";
#endif

                        for ( frank = 0 ; frank < Nres ; frank++ )
                        {
                            // Next incorporate the factors:
                            //
                            // psi(ld,ld,mud,sigmad) - psi(ld,ud,mud,sigmad)
                            //
                            // psi(l,s,mu,si) = sqrt(si).phi(s-mu/sqrt(si))
                            //                + ((l-mu)/2).(1+erf(s-mu/sqrt(2.si)))
                            //
                            // (and yada yada our si is really si^2 etc)
                            //
                            // psi(l,s,mu,si) = (sqrt(si)/sqrt(2pi)).exp(-(s-mu)^2/(2si))
                            //                + ((l-mu)/2).(1+erf(s-mu/sqrt(2.si)))
                            //
                            // So:
                            //
                            // psi(l,l,mu,si) = (sqrt(si)/sqrt(2pi)).exp(-(l-mu)^2/(2si))
                            //                + ((l-mu)/2).(1+erf(l-mu/sqrt(2.si)))
                            // psi(l,u,mu,si) = (sqrt(si)/sqrt(2pi)).exp(-(u-mu)^2/(2si))
                            //                + ((l-mu)/2).(1+erf(u-mu/sqrt(2.si)))
                            //
                            // psi(ld,ld,mud,sigmad) - psi(ld,ud,mud,sigmad)
                            //
                            // = (si/sqrt(2pi)).( exp(-(l-mu)^2/(2si)) - exp(-(u-mu)^2/(2si)) )
                            // + ((l-mu)/2).( erf(l-mu/sqrt(2.si)) - erf(u-mu/sqrt(2.si)) )
                            //
                            // Let: g = sqrt(2.si)
                            //      e = l-mu
                            //      f = u-mu
                            //      a = (l-mu)/sqrt(2.si) = e/g
                            //      b = (u-mu)/sqrt(2.si) = f/g
                            //      c = (l-mu)^2/(2si) = a^2
                            //      d = (u-mu)^2/(2si) = b^2
                            //
                            // psi(ld,ld,mud,sigmad) - psi(ld,ud,mud,sigmad)
                            //
                            // = (si/sqrt(2pi)).( exp(-c) - exp(-d) )
                            // + (e/2).( erf(a) - erf(b) )

                            woop = 1;

                            if ( i > 0 )
                            {
                                for ( j = 0 ; j < i ; j++ )
                                {
                                    varadj = sqrt(s[frank][kk[j]]);
                                    e = l[kk[j]]-mu[frank][kk[j]];
                                    f = u[kk[j]]-mu[frank][kk[j]];
                                    a = NUMBASE_SQRT1ON2*e/varadj;
                                    b = NUMBASE_SQRT1ON2*f/varadj;

                                    if ( ii[kk[j]] < M-1 )
                                    {
                                        woop *= ( (varadj*NUMBASE_1ONSQRT2PI*(exp(-a*a)-exp(-b*b))) + ((e/2)*(erf(a)-erf(b))) );
                                    }

                                    else
                                    {
                                        woop *= ( (varadj*NUMBASE_1ONSQRT2PI*(exp(-a*a))) + ((e/2)*(erf(a)-1)) );
                                    }
                                }
                            }
#ifdef DEBUG
errstream() << "dV ICconst*C factors = " << dV << "\n";
#endif

                            // Finally incorporate the factors:
                            //
                            // Phi_{mud,sigmad}(ud) - Phi_{mud,sigmad}(ld)
                            //
                            // Phi_{mud,sigmad}(s) = 1/2 ( 1 + erf((s-mud)/sqrt(.2sigmad)) )
                            //
                            // (and yada yada our sigmad is really sigmad^2 etc)
                            //
                            // So:
                            //
                            // Phi_{mud,sigmad}(ud) - Phi_{mud,sigmad}(ld)
                            // = 1/2 ( erf((ud-mud)/sqrt(2sigmad)) - erf((ld-mud)/sqrt(2sigmad)) )

                            if ( i < n )
                            {
                                for ( j = i ; j < n ; j++ )
                                {
                                    varadj = sqrt(s[frank][kk[j]]);
                                    e = l[kk[j]]-mu[frank][kk[j]];
                                    f = u[kk[j]]-mu[frank][kk[j]];
                                    a = NUMBASE_SQRT1ON2*e/varadj;
                                    b = NUMBASE_SQRT1ON2*f/varadj;

                                    if ( ii[kk[j]] < M-1 )
                                    {
                                        woop *= ( erf(b) - erf(a) )/2;
                                    }

                                    else
                                    {
                                        woop *= ( 1 - erf(a) )/2;
                                    }
                                }
                            }

#ifdef DEBUG
errstream() << "End dV = " << dV << "\n";
#endif
                            res[frank] += woop*dV;
                        }
                    }

                    // Increment set
                    //
                    // Note pattern: eg n = 6
                    //
                    // 0 1 2 3
                    // 0 1 2 4
                    // 0 1 2 5
                    // 0 1 3 4
                    // 0 1 3 5
                    // 0 1 4 5
                    // 0 2 3 4
                    // 0 2 3 5
                    // 0 2 4 5
                    // 0 3 4 5
                    // 1 2 3 4
                    // 1 2 3 5
                    // 1 2 4 5
                    // 1 3 4 5
                    // 2 3 4 5
                    //
                    // That is, increment last element in set if less
                    // than n-1, otherwise look at previous element.  If
                    // previous element less than n-2 increment and set
                    // elements after this increasing by 1, otherwise
                    // move to element before that... less than n-3 etc.

                    locbreak = 0;

                    if ( i )
                    {
                        for ( j = i-1 ; j >= 0 ; j-- )
                        {
                            if ( kk[j] < n-(i-j) )
                            {
                                kk[j]++;

                                if ( j < i-1 )
                                {
                                    for ( k = j+1 ; k < i ; k++ )
                                    {
                                        kk[k] = kk[k-1]+1;
                                    }
                                }

                                locbreak = 1;
                                break;
                            }
                        }
                    }

                    // If j < 0 then could not continue with process and
                    // we are done with this cardinality.

                    if ( !locbreak )
                    {
                        isdoneb = 1;
                    }

                    else if ( i < n )
                    {
                        // We need to update the D\C set, which is just
                        // those not in C ordered from smallest to largest.

                        j = i;
                        k = 0;
                        h = 0;

                        while ( j < n )
                        {
                            if ( k >= i )
                            {
                                kk[j] = h;
                                j++;
                                h++;
                            }

                            else if ( kk[k] > h )
                            {
                                kk[j] = h;
                                j++;
                                h++;
                            }

                            else if ( kk[k] == h )
                            {
                                h++;
                                k++;
                            }

                            else
                            {
                                // kk[k] < h

                                k++;
                            }
                        }
                    }
                }
            }
        }

        else
        {
            // Speedup: we are moving in the direction of
            // decreasing on axis 0.  If !udomX then we can skip the rest
            // of the tests on axis 0 and move straight on to next axis.
            // This will automatically enforce the optimised grid search in
            // the Hupkens 2-d case and something similar in n>2 dim.

            jj[0] = 0;
            ii[0] = -1;
        }

        // Increment.  Not as complicated as it may first appear.

        locbreak = 0;

        for ( i = 0 ; ( i < n ) && !locbreak ; i++ )
        {
            jj[i]--;

            if ( jj[i] < 0 )
            {
                ii[i] = M-1;
                jj[i] = m[i]; //-1;
            }

            else
            {
                locbreak = 1;
            }

            if ( !jj[i] )
            {
                ii[i] = -1;
            }

            else
            {
                while ( iind[i][ii[i]] >= jj[i] )
                {
                    ii[i]--;

                    NiceAssert( ii[i] >= 0 );
                }
            }

            l[i] = ( ii[i] >= 0  ) ? Xsort[i][ii[i]  ][i] : 0;
            u[i] = ( ii[i] < M-1 ) ? Xsort[i][ii[i]+1][i] : 0;

            isedge |= ( ii[i] == M-1 );
        }

        isedge = 0;

        for ( i = 0 ; i < n ; i++ )
        {
            isedge |= ( ii[i] == M-1 );
        }

        if ( ( i == n ) && !locbreak )
        {
            isdone = 1;
        }
    }

    for ( i = 0 ; i < n ; i++ )
    {
        MEMDELARRAY(Xsort[i]);
        MEMDELARRAY(iind[i]);
    }

    MEMDELARRAY(Xsort);
    MEMDELARRAY(iind);
    MEMDELARRAY(m);
    MEMDELARRAY(ii);
    MEMDELARRAY(jj);
    MEMDELARRAY(l);
    MEMDELARRAY(u);
    MEMDELARRAY(idummy);
    MEMDELARRAY(kk);
    MEMDELARRAY(Xalt);

    return;
}



void ehi_cou(double *res, int Nres, double **X, int Ms, int n, double **mu, double **s)
{
    // The method here is very similar to Hupkens.  The cell enumeration
    // code is in fact exactly the same.  What differs is:
    //
    // - we have to do cell enumeration *twice*
    // - the calculation is a touch more complicated - see pg 582 of
    //
    //  Couckuyt et al 2014, Fast Calculation of multiobjective probability
    //  of improvement and expected improvement criteria for pareto
    //  optimization.  Journal of Global Optimization, 60(3):575-594
    //
    // For brevity as much of this is cut/paste of the Hupkens code most
    // comments have been removed.  Again: the only difference is the
    // calculation of the contribution of each cell.

    int blueswirlything;
    int i,j;
    int iprime,jprime;

    if ( !Nres )
    {
        return;
    }

    for ( blueswirlything = 0 ; blueswirlything < Nres ; blueswirlything++ )
    {
        res[blueswirlything] = 0;
    }

    int M = dom(X,Ms,n);

    double ***Xsort;
    int **iind;
    int *m;

    MEMNEWARRAY(Xsort,double **,n);
    MEMNEWARRAY(iind,int *,n);
    MEMNEWARRAY(m,int,n);

    NiceAssert( Xsort );
    NiceAssert( iind );
    NiceAssert( m );

    int *ii;
    int *jj;
    double *l;
    double *u;

    MEMNEWARRAY(ii,int,n);
    MEMNEWARRAY(jj,int,n);
    MEMNEWARRAY(l,double,n);
    MEMNEWARRAY(u,double,n);

    NiceAssert( ii );
    NiceAssert( jj );
    NiceAssert( l );
    NiceAssert( u );

    int *iiprime;
    int *jjprime;
    double *lprime;
    double *uprime;

    MEMNEWARRAY(iiprime,int,n);
    MEMNEWARRAY(jjprime,int,n);
    MEMNEWARRAY(lprime,double,n);
    MEMNEWARRAY(uprime,double,n);

    NiceAssert( iiprime );
    NiceAssert( jjprime );
    NiceAssert( lprime );
    NiceAssert( uprime );

    for ( i = 0 ; i < n ; i++ )
    {
        MEMNEWARRAY(Xsort[i],double *,M);
        MEMNEWARRAY(iind[i],int,M);

        NiceAssert( Xsort[i] );
        NiceAssert( iind[i] );

        for ( j = 0 ; j < M ; j++ )
        {
            Xsort[i][j] = X[j];
        }

        axissort(Xsort[i],M,i+1);

        m[i] = cluster(Xsort[i],iind[i],M,i+1);

        NiceAssert( m[i] >= 0 );
    }

    int isdone = 0;
    int locbreak = 0;
    int udomX = 1;
    int isedge = 0;
    double ll,uu;

    int udomXprime = 1;
    int isedgeprime = 0;
    int isdoneprime = 0;
    int locbreakprime = 0;
    double G;
    double Gc;
    double Gu;
    double llprime,uuprime;

    double ss,mumu;
    double a,b,e,f,varadj;

    isedge = 0;

    for ( i = 0 ; i < n ; i++ )
    {
        ii[i] = M-1;
        jj[i] = m[i]; //-1;

        if ( !jj[i] )
        {
            ii[i] = -1;
        }

        else
        {
            while ( iind[i][ii[i]] >= jj[i] )
            {
                ii[i]--;

                NiceAssert( ii[i] >= 0 );
            }
        }

        l[i] = ( ii[i] >= 0  ) ? Xsort[i][ii[i]  ][i] : 0;
        u[i] = ( ii[i] < M-1 ) ? Xsort[i][ii[i]+1][i] : 0;

        isedge |= ( ii[i] == M-1 );
    }

    while ( !isdone )
    {
        udomX = 1;

        for ( j = 0 ; j < M ; j++ )
        {
            if ( !isedge )
            {
                if ( xdomy(X[j],u,n) )
                {
                    udomX = 0;
                    break;
                }
            }
        }

        if ( udomX )
        {
            isedgeprime = 0;

            for ( iprime = 0 ; iprime < n ; iprime++ )
            {
                iiprime[iprime] = M-1;
                jjprime[iprime] = m[iprime]; //-1;

                if ( !jjprime[iprime] )
                {
                    iiprime[iprime] = -1;
                }

                else
                {
                    while ( iind[iprime][iiprime[iprime]] >= jjprime[iprime] )
                    {
                        iiprime[iprime]--;

                        NiceAssert( iiprime[iprime] >= 0 );
                    }
                }

                lprime[iprime] = ( iiprime[iprime] >= 0  ) ? Xsort[iprime][iiprime[iprime]  ][iprime] : 0;
                uprime[iprime] = ( iiprime[iprime] < M-1 ) ? Xsort[iprime][iiprime[iprime]+1][iprime] : 0;

                isedgeprime |= ( iiprime[iprime] == M-1 );
            }

            isdoneprime = 0;

            while ( !isdoneprime )
            {
                udomXprime = 1;

                for ( jprime = 0 ; jprime < M ; jprime++ )
                {
                    if ( !isedgeprime )
                    {
                        if ( xdomy(X[jprime],uprime,n) )
                        {
                            udomXprime = 0;
                            break;
                        }
                    }
                }

                if ( udomXprime )
                {
                    for ( blueswirlything = 0 ; blueswirlything < Nres ; blueswirlything++ )
                    {
                        G = 1;

                        for ( jprime = 0 ; jprime < n ; jprime++ )
                        {
                            ll = l[jprime];
                            uu = u[jprime];

                            llprime = lprime[jprime];
                            uuprime = uprime[jprime];

                            ss = s[blueswirlything][jprime];
                            mumu = mu[blueswirlything][jprime];
                            varadj = sqrt(ss);

                            if ( ( ( uu      == 0 ) || ( llprime < uu ) ) &&
                                 ( ( uuprime == 0 ) || ( uuprime > ll ) )     )
                            {
                                Gc = 0;

                                if ( llprime < ll )
                                {
                                    a = NUMBASE_SQRT1ON2*(ll     -mumu)/varadj;
                                    b = NUMBASE_SQRT1ON2*(llprime-mumu)/varadj;

                                    Gc = (ll-llprime)*(erf(a)-erf(b))/2;

                                    llprime = ll;
                                }

                                if ( ( uuprime > uu ) || ( uuprime == 0 ) )
                                {
                                    uuprime = uu;
                                }

                                Gu = 0;

                                if ( uuprime > 0 )
                                {
                                    e = uuprime-mumu;
                                    a = NUMBASE_SQRT1ON2*e/varadj;

                                    f = llprime-mumu;
                                    b = NUMBASE_SQRT1ON2*f/varadj;

                                    Gu = (f*(erf(b)-erf(a))/2) +
                                         (varadj*NUMBASE_1ONSQRT2PI*(exp(-b*b)-exp(-a*a)));
                                }

                                else
                                {
                                    f = llprime-mumu;
                                    b = NUMBASE_SQRT1ON2*f/varadj;

                                    Gu = (f*(erf(b)-1)/2) +
                                         (varadj*NUMBASE_1ONSQRT2PI*exp(-b*b));
                                }

                                G *= Gc+Gu;
                            }

                            else if ( ( uuprime > 0 ) && ( uuprime <= ll ) )
                            {
                                Gc = 0;

                                if ( uu > 0 )
                                {
                                    a = NUMBASE_SQRT1ON2*(uu-mumu)/varadj;
                                    b = NUMBASE_SQRT1ON2*(ll-mumu)/varadj;

                                    Gc = (uuprime-llprime)*(erf(a)-erf(b))/2;
                                }

                                else
                                {
                                    b = NUMBASE_SQRT1ON2*(ll-mumu)/varadj;

                                    Gc = (uuprime-llprime)*(1-erf(b))/2;
                                }

                                G *= Gc;
                            }

                            else
                            {
                                G = 0;
                                break;
                            }
                        }

                        res[blueswirlything] += G;
                    }
                }

                else
                {
                    jjprime[0] = 0;
                    iiprime[0] = -1;
                }

                locbreakprime = 0;

                for ( iprime = 0 ; ( iprime < n ) && !locbreakprime ; iprime++ )
                {
                    jjprime[iprime]--;

                    if ( jjprime[iprime] < 0 )
                    {
                        iiprime[iprime] = M-1;
                        jjprime[iprime] = m[iprime]; //-1;
                    }

                    else
                    {
                        locbreakprime = 1;
                    }

                    if ( !jjprime[iprime] )
                    {
                        iiprime[iprime] = -1;
                    }

                    else
                    {
                        while ( iind[iprime][iiprime[iprime]] >= jjprime[iprime] )
                        {
                            iiprime[iprime]--;

                            NiceAssert( iiprime[iprime] >= 0 );
                        }
                    }

                    lprime[iprime] = ( iiprime[iprime] >= 0  ) ? Xsort[iprime][iiprime[iprime]  ][iprime] : 0;
                    uprime[iprime] = ( iiprime[iprime] < M-1 ) ? Xsort[iprime][iiprime[iprime]+1][iprime] : 0;
                }

                isedgeprime = 0;

                for ( iprime = 0 ; iprime < n ; iprime++ )
                {
                    isedgeprime |= ( iiprime[iprime] == M-1 );
                }

                if ( ( iprime == n ) && !locbreakprime )
                {
                    isdoneprime = 1;
                }
            }
        }

        else
        {
            jj[0] = 0;
            ii[0] = -1;
        }

        locbreak = 0;

        for ( i = 0 ; ( i < n ) && !locbreak ; i++ )
        {
            jj[i]--;

            if ( jj[i] < 0 )
            {
                ii[i] = M-1;
                jj[i] = m[i]; //-1;
            }

            else
            {
                locbreak = 1;
            }

            if ( !jj[i] )
            {
                ii[i] = -1;
            }

            else
            {
                while ( iind[i][ii[i]] >= jj[i] )
                {
                    ii[i]--;

                    NiceAssert( ii[i] >= 0 );
                }
            }

            l[i] = ( ii[i] >= 0  ) ? Xsort[i][ii[i]  ][i] : 0;
            u[i] = ( ii[i] < M-1 ) ? Xsort[i][ii[i]+1][i] : 0;
        }

        isedge = 0;

        for ( i = 0 ; i < n ; i++ )
        {
            isedge |= ( ii[i] == M-1 );
        }

        if ( ( i == n ) && !locbreak )
        {
            isdone = 1;
        }
    }

    for ( i = 0 ; i < n ; i++ )
    {
        MEMDELARRAY(Xsort[i]);
        MEMDELARRAY(iind[i]);
    }

    MEMDELARRAY(Xsort);
    MEMDELARRAY(iind);
    MEMDELARRAY(m);
    MEMDELARRAY(ii);
    MEMDELARRAY(jj);
    MEMDELARRAY(l);
    MEMDELARRAY(u);
    MEMDELARRAY(iiprime);
    MEMDELARRAY(jjprime);
    MEMDELARRAY(lprime);
    MEMDELARRAY(uprime);

    return;
}









