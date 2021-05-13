
//
// Pareto Test Functions as per:
//
// Deb, Thiele, Laumanns, Zitzler (DTLZ)
// Scalable Test Problems for Evolutionary Multiobjective Optimisation
//
// Version: 
// Date: 1/12/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "paretotest.h"
#include "numbase.h"
#include <math.h>

int evalTestFn(int fnnum, int n, int M, Vector<double> &res, const Vector<double> &x, double alpha)
{
    int nonfeas = 0;
    int i,j;

    NiceAssert( M >= 1 );

    res.resize(M);

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retVector<double> tmpvc;

    const Vector<double> &xL = x(0,1,M-2,tmpva);
    const Vector<double> &xM = x(M-1,1,n-1,tmpvb);

    switch ( fnnum )
    {
        case 1:
        {
            NiceAssert( M <= n );

            double g = n-M+1;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += ((xM(i)-0.5)*(xM(i)-0.5)) - cos(20*NUMBASE_PI*(xM(i)-0.5));
            }

            g *= 100;

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;
                res("&",i) *= prod(xL(zeroint(),1,M-2-i,tmpvc));
                res("&",i) *= ( i ? (1-xL(M-1-i)) : 1 );
                res("&",i) *= 1+g;
                res("&",i) /= 2;
            }

            break;
        }

        case 2:
        {
            NiceAssert( M <= n );

            double g = 0;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += ((xM(i)-0.5)*(xM(i)-0.5));
            }

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;

                if ( M-1+i > 0 )
                {
                    for ( j = 0 ; j < M-1-i ; j++ )
                    {
                        res("&",i) *= cos(NUMBASE_PION2*xL(j));
                    }
                }

                res("&",i) *= ( i ? sin(NUMBASE_PION2*xL(M-1-i)) : 1 );
                res("&",i) *= 1+g;
            }

            break;
        }

        case 3:
        {
            NiceAssert( M <= n );

            double g = n-M+1;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += ((xM(i)-0.5)*(xM(i)-0.5)) - cos(20*NUMBASE_PI*(xM(i)-0.5));
            }

            g *= 100;

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;

                if ( M-1+i > 0 )
                {
                    for ( j = 0 ; j < M-1-i ; j++ )
                    {
                        res("&",i) *= cos(NUMBASE_PION2*xL(j));
                    }
                }

                res("&",i) *= ( i ? sin(NUMBASE_PION2*xL(M-1-i)) : 1 );
                res("&",i) *= 1+g;
            }

            break;
        }

        case 4:
        {
            NiceAssert( M <= n );

            double g = 0;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += ((xM(i)-0.5)*(xM(i)-0.5));
            }

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;

                if ( M-1+i > 0 )
                {
                    for ( j = 0 ; j < M-1-i ; j++ )
                    {
                        res("&",i) *= cos(NUMBASE_PION2*pow(xL(j),alpha));
                    }
                }

                res("&",i) *= ( i ? sin(NUMBASE_PION2*pow(xL(M-1-i),alpha)) : 1 );
                res("&",i) *= 1+g;
            }

            break;
        }

        case 5:
        {
            NiceAssert( M <= n );

            double g = 0;
            double theta;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += ((xM(i)-0.5)*(xM(i)-0.5));
            }

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;

                if ( M-1+i > 0 )
                {
                    for ( j = 0 ; j < M-1-i ; j++ )
                    {
                        theta = (NUMBASE_PI/(4*(1+g)))*(1+(2*g*xL(j)));

                        res("&",i) *= cos(NUMBASE_PION2*theta);
                    }
                }

                if ( i )
                {
                    theta = (NUMBASE_PI/(4*(1+g)))*(1+(2*g*xL(M-1-i)));

                    res("&",i) *= sin(NUMBASE_PION2*theta);
                }

                res("&",i) *= 1+g;
            }

            break;
        }

        case 6:
        {
            NiceAssert( M <= n );

            double g = 0;
            double theta;

            for ( i = 0 ; i < n-M+1 ; i++ )
            {
                g += pow(xM(i),0.1);
            }

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) =  1;

                if ( M-1+i > 0 )
                {
                    for ( j = 0 ; j < M-1-i ; j++ )
                    {
                        theta = (NUMBASE_PI/(4*(1+g)))*(1+(2*g*xL(j)));

                        res("&",i) *= cos(NUMBASE_PION2*theta);
                    }
                }

                if ( i )
                {
                    theta = (NUMBASE_PI/(4*(1+g)))*(1+(2*g*xL(M-1-i)));

                    res("&",i) *= sin(NUMBASE_PION2*theta);
                }

                res("&",i) *= 1+g;
            }

            break;
        }

        case 7:
        {
            NiceAssert( M <= n );

            double g = 1 + (9*sum(xM)/((double) (n-M+1)));
            double h = M;

            if ( M > 1 )
            {
                for ( i = 0 ; i < M-1 ; i++ )
                {
                    res("&",i) = xL(i);

                    h -= (res(i)/(1+g))*(1+sin(3*NUMBASE_PI*res(i)));
                }
            }

            res("&",M-1) = (1+g)*h;

            break;
        }

        case 8:
        {
            NiceAssert( M < n );

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) = 0;

                for ( j = (int) ((i*n)/((double) M))-1 ; j < (int) (((i+1)*n)/((double) M))-1 ; j++ )
                {
                    // NB: there appears to be a typo in DTLZ here.  They run
                    //     indices from 1..n for decision space, 1..M for
                    //     target space.  But if you consider the lower
                    //     bound on the sum for fj evaluations in 6.26 (and
                    //     6.27) that is:
                    //
                    //     floor((j-1)*(n/M)) = floor(0) = 0
                    //
                    //     which is outside the range of their indices (in our
                    //     ranging 0..n-1 this translates to index -1).  My
                    //     guess is that we just don't include the part of
                    //     the sum that lies outside the range - hence the
                    //     following if statement.

                    NiceAssert( j >= -1 );
                    NiceAssert( j < n );

                    if ( j >= 0 )
                    {
                        res("&",i) += x(j);
                    }
                }

                res *= 1.0/((double) ((int) (n/((double) M))));
            }

            // Feasibility tests

            double g;

            if ( M > 1 )
            {
                for ( j = 0 ; ( j < M-1 ) && !nonfeas ; j++ )
                {
                    g = res(M-1) + (4*res(j)) - 1;

                    if ( g < 0 )
                    {
                        nonfeas = 1;
                    }
                }
            }

            if ( !nonfeas )
            {
                double temp,minsum = 0;

                if ( M > 2 )
                {
                    for ( i = 0 ; i < M-1 ; i++ )
                    {
                        for ( j = 0 ; j < M-1 ; j++ )
                        {
                            temp = res(i)+res(j);

                            if ( ( i != j ) && ( ( !i && ( j == 1 ) ) || ( temp < minsum ) ) )
                            {
                                minsum = temp;
                            }
                        }
                    }
                }

                g = (2*res(M-1)) + minsum - 1;

                if ( g < 0 )
                {
                    nonfeas = 1;
                }
            }

            break;
        }

        case 9:
        {
            NiceAssert( M < n );

            for ( i = 0 ; i < M ; i++ )
            {
                res("&",i) = 0;

                for ( j = (int) ((i*n)/((double) M))-1 ; j < (int) (((i+1)*n)/((double) M))-1 ; j++ )
                {
                    // NB: there appears to be a typo in DTLZ here.  They run
                    //     indices from 1..n for decision space, 1..M for
                    //     target space.  But if you consider the lower
                    //     bound on the sum for fj evaluations in 6.26 (and
                    //     6.27) that is:
                    //
                    //     floor((j-1)*(n/M)) = floor(0) = 0
                    //
                    //     which is outside the range of their indices (in our
                    //     ranging 0..n-1 this translates to index -1).  My
                    //     guess is that we just don't include the part of
                    //     the sum that lies outside the range - hence the
                    //     following if statement.

                    NiceAssert( j >= -1 );
                    NiceAssert( j < n );

                    if ( j >= 0 )
                    {
                        res("&",i) += pow(x(j),0.1);
                    }
                }
            }

            // Feasibility tests

            double g;

            if ( M > 1 )
            {
                for ( j = 0 ; ( j < M-1 ) && !nonfeas ; j++ )
                {
                    g = (res(M-1)*res(M-1)) + (res(j)*res(j)) - 1;

                    if ( g < 0 )
                    {
                        nonfeas = 1;
                    }
                }
            }

            break;
        }

        case 10:
        {
            // FON

            NiceAssert( M == 2 );

            res("&",0) = 1-exp( -sqsum(x)+(2*sum(x)/sqrt((double) n))-1 );
            res("&",1) = 1-exp( -sqsum(x)-(2*sum(x)/sqrt((double) n))-1 );

            break;
        }

        case 11:
        {
case11:
            // SCH1

            NiceAssert( n == 1 );
            NiceAssert( M == 2 );

            res("&",0) = x(0)*x(0);
            res("&",1) = (x(0)-2.0)*(x(0)-2.0);

            break;
        }

        case 12:
        {
            // SCH2

            NiceAssert( n == 1 );
            NiceAssert( M == 2 );

                 if ( x(0) <= 1 ) { res("&",0) = -x(0);    }
            else if ( x(0) <= 3 ) { res("&",0) = x(0)-2.0; }
            else if ( x(0) <= 4 ) { res("&",0) = 4.0-x(0); }
            else                  { res("&",0) = x(0)-4.0; }

            res("&",1) = (x(0)-5.0)*(x(0)-5.0);

            break;
        }

        default:
        {
            goto case11;

            break;
        }
    }

    return nonfeas;
}

