
//
// Expected improvement (EHI for multi-objective)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "imp_expect.h"
#include "hyper_alt.h"
#include "hyper_base.h"


IMP_Expect::IMP_Expect(int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(NULL);

    xminval = 0.0;
    hc      = NULL;
    X       = NULL;

    return;
}

IMP_Expect::IMP_Expect(const IMP_Expect &src, int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

IMP_Expect::IMP_Expect(const IMP_Expect &src, const ML_Base *xsrc, int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

IMP_Expect::~IMP_Expect()
{
    untrain(); //this will delete hc and X if required.

    return;
}

std::ostream &IMP_Expect::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Expected improvement block\n";

    repPrint(output,'>',dep) << "x minima: " << xminval << "\n";

    return IMP_Generic::printstream(output,dep+1);
}

std::istream &IMP_Expect::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> xminval;

    if ( xspaceDim() > 1 )
    {
        untrain();
    }

    return IMP_Generic::inputstream(input);
}

int IMP_Expect::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    int res = imp(resg,x(i),realgentype());

    resh = resg;

    return res;
}












int IMP_Expect::train(int &res, svmvolatile int &killSwitch)
{
    (void) res;

    int retval = 0;

    incgvernum();

    if ( !isTrained() )
    {
        retval = IMP_Generic::train(res,killSwitch);

        xminval = 0.0;

        if ( N()-NNC(0) )
        {
            if ( xspaceDim() <= 1 )
            {
                NiceAssert( xspaceDim() == 1 );

                int i;
                gentype temp;

                xelm(xminval,0,0);

                if ( N() > 1 )
                {
                    for ( i = 1 ; ( i < N() ) && !killSwitch ; i++ )
                    {
                        if ( isenabled(i) )
                        {
                            if ( xelm(temp,i,0) < xminval )
                            {
                                xminval = temp;
                            }
                        }
                    }
                }
            }

            else
            {
                int M = N()-NNC(0);
                int n = xspaceDim();
                gentype temp;

                MEMNEWARRAY(X,double *,M+1);

                int i,j,k;

                for ( i = 0, j = 0 ; i < N() ; i++ )
                {
                    if ( isenabled(i) )
                    {
                        MEMNEWARRAY(X[j],double,xspaceDim());

                        for ( k = 0 ; k < xspaceDim() ; k++ )
                        {
                            xelm(temp,i,k);
                            X[j][k] = -(((double) temp)-zref());
                        }

                        j++;
                    }
                }

                if ( ehimethod() <= 1 )
                {
                    hc = make_cache(X,M,ehimethod() ? -n : n);
                }
            }
        }
    }

    return retval;
}

void IMP_Expect::untrain(void)
{
    incgvernum();

    if ( hc )
    {
        del_cache(hc);
        hc = NULL;
    }

    if ( X )
    {
        int j;

        for ( j = 0 ; j < N() ; j++ )
        {
            MEMDELARRAY(X[j]);
        }

        MEMDELARRAY(X);
        X = NULL;
    }

    IMP_Generic::untrain();

    return;
}

int IMP_Expect::imp(gentype &resi, const SparseVector<gentype> &xxmean, const gentype &xxvar) const
{
    NiceAssert( isTrained() );

    // NB: following calculation actually for increase, not decrease, so we
    //     have an infestation of negatives.

    double res = 0;

    if ( !N() )
    {
        res = 0.0;

        if ( xxmean.size() )
        {
            res = 1.0;

            int j;

            for ( j = 0 ; j < xxmean.size() ; j++ )
            {
                res *= -(((double) xxmean(j))-zref());
            }
        }
    }

    else if ( xspaceDim() <= 1 )
    {
        const gentype &muy = xxmean(zeroint());
        const gentype &sigmay = xxvar;

        double ymax = -( (double) xminval );

        if ( !(N()-NNC(0)) )
        {
            res = -( (double) muy );
        }

        else if ( (double) sigmay > zerotol() )
        {
            double z    = ( -( (double) muy ) - ymax ) / ( (double) sigmay );
            double Phiz = 0.5 + (0.5*erf(z*NUMBASE_SQRT1ON2));
            double phiz = exp(-z*z/2)/2.506628;

            res = ( ( -( (double) muy ) - ymax ) * Phiz )
                + ( ( (double) sigmay ) * phiz );
        }

        else
        {
            // if muy > ymax then z = +infty, so Phiz = +1, phiz = 0
            // if muy < ymax then z = -infty, so Phiz = 0,  phiz = infty
            // assume lim_{z->-infty} sigmay.phiz = 0

            if ( -( (double) muy > ymax ) )
            {
                res = -( (double) muy ) - ymax;
            }

            else
            {
                res = 0.0;
            }
        }
    }

    else
    {
        if ( !(N()-NNC(0)) )
        {
            res = 1.0;

            int j;

            for ( j = 0 ; j < xspaceDim() ; j++ )
            {
                res *= -( (double) xxmean(j) );
            }
        }

        else
        {
            int j;

            double *mu;
            double *s;

            MEMNEWARRAY(mu,double,xspaceDim());
            MEMNEWARRAY(s ,double,xspaceDim());

            for ( j = 0 ; j < xspaceDim() ; j++ )
            {
                mu[j] = -(((double) xxmean(j))-zref());
                s[j]  =  ((double) xxvar);
            }

            switch ( ehimethod() )
            {
                case 0:
                case 1:
                {
                    NiceAssert( hc );
                    res = ehi(mu,s,hc);
                    break;
                }

                case 2:
                {
                    res = ehi(X,N(),xspaceDim(),mu,s);
                    break;
                }

                case 3:
                {
                    res = ehi_hup(X,N(),xspaceDim(),mu,s);
                    break;
                }

                default:
                {
                    NiceAssert( ehimethod() == 4 );
                    res = ehi_cou(X,N(),xspaceDim(),mu,s);
                    break;
                }
            }

            MEMDELARRAY(mu);
            MEMDELARRAY(s);
        }
    }

    resi.force_double() = -res;

    return ( res == 0 ) ? 0 : ( ( -res > 0 ) ? +1 : -1 );
}
