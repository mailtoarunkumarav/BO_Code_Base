
//
// Probability of improvement.
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
#include "imp_probab.h"


IMP_Probab::IMP_Probab(int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(NULL);

    xminval = 0.0;

    return;
}

IMP_Probab::IMP_Probab(const IMP_Probab &src, int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

IMP_Probab::IMP_Probab(const IMP_Probab &src, const ML_Base *xsrc, int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

IMP_Probab::~IMP_Probab()
{
    return;
}

std::ostream &IMP_Probab::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Probability of improvement block\n";

    repPrint(output,'>',dep) << "x minima: " << xminval << "\n";

    return IMP_Generic::printstream(output,dep+1);
}

std::istream &IMP_Probab::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> xminval;

    return IMP_Generic::inputstream(input);
}

int IMP_Probab::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    int res = imp(resg,x(i),realgentype());

    resh = resg;

    return res;
}











int IMP_Probab::train(int &res, svmvolatile int &killSwitch)
{
    (void) res;

    NiceAssert( xspaceDim() <= 1 );

    int retval = 0;

    if ( !isTrained() )
    {
        retval = IMP_Generic::train(res,killSwitch);

        xminval = 0.0;

        if ( N()-NNC(0) )
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
    }

    return retval;
}

int IMP_Probab::imp(gentype &resi, const SparseVector<gentype> &xxmean, const gentype &xxvar) const
{
    NiceAssert( isTrained() );

    const gentype &muy = xxmean(zeroint());
    const gentype &sigmay = xxvar;

    // NB: following calculation actually for increase, not decrease, so we
    //     have an infestation of negatives.

    double ymax = -( (double) xminval );
    double res = 0;

    if ( !(N()-NNC(0)) )
    {
        res = 1.0;
    }

    else if ( (double) sigmay > zerotol() )
    {
        res = 1.0 - (1.0+erf(( (-((double) muy)-ymax)*NUMBASE_SQRT1ON2/((double) sigmay) ))/2.0);
    }

    else
    {
        // if sigmay = 0 then the result is a simple yes/no question.
        // if muy > ymax then we will improve (res = 1), otherwise
        // we will not (res = 0)

        if ( -( (double) muy ) > ymax )
        {
            res = 1.0;
        }

        else
        {
            res = 0.0;
        }
    }

    resi.force_double() = -res;

    return ( res == 0 ) ? 0 : ( ( -res > 0 ) ? +1 : -1 );
}
