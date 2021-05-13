
//
// Improvement measure base class
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
#include "imp_generic.h"
#include "hyper_base.h"


std::ostream &IMP_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "zref:       " << xzref      << "\n";
    repPrint(output,'>',dep) << "EHI method: " << xehimethod << "\n";
    repPrint(output,'>',dep) << "isTrained:  " << disTrained << "\n";

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &IMP_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> xzref;
    input >> dummy; input >> xehimethod;
    input >> dummy; input >> disTrained;

    ML_Base::inputstream(input);

    return input;
}


IMP_Generic::IMP_Generic(int isIndPrune) : ML_Base(isIndPrune)
{
    setaltx(NULL);

    xzref      = 0;
    xehimethod = 0;
    disTrained = 0;

    return;
}

IMP_Generic::IMP_Generic(const IMP_Generic &src, int isIndPrune) : ML_Base(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

IMP_Generic::IMP_Generic(const IMP_Generic &src, const ML_Base *xsrc, int isIndPrune) : ML_Base(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

IMP_Generic::~IMP_Generic()
{
    return;
}

int IMP_Generic::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    untrain();
    ML_Base::addTrainingVector(i,y,x,Cweigh,epsweigh);

    return 1;
}

int IMP_Generic::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    untrain();
    ML_Base::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    return 1;
}

int IMP_Generic::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    if ( y.size() )
    {
        untrain();
        ML_Base::addTrainingVector(i,y,x,Cweigh,epsweigh);
    }

    return 1;
}

int IMP_Generic::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    if ( y.size() )
    {
        untrain();
        ML_Base::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    }

    return 1;
}

int IMP_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    untrain();
    ML_Base::removeTrainingVector(i,y,x);

    return 1;
}

int IMP_Generic::removeTrainingVector(int i, int num)
{
    untrain();
    ML_Base::removeTrainingVector(i,num);

    return 1;
}

int IMP_Generic::setx(int i, const SparseVector<gentype> &x)
{
    untrain();
    ML_Base::setx(i,x);

    return 1;
}

int IMP_Generic::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x)
{
    untrain();
    ML_Base::setx(i,x);

    return 1;
}

int IMP_Generic::setx(const Vector<SparseVector<gentype> > &x)
{
    untrain();
    ML_Base::setx(x);

    return 1;
}

int IMP_Generic::qswapx(int i, SparseVector<gentype> &x, int dontupdate)
{
    untrain();
    ML_Base::qswapx(i,x,dontupdate);

    return 1;
}

int IMP_Generic::qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate)
{
    untrain();
    ML_Base::qswapx(i,x,dontupdate);

    return 1;
}

int IMP_Generic::qswapx(Vector<SparseVector<gentype> > &x, int dontupdate)
{
    untrain();
    ML_Base::qswapx(x,dontupdate);

    return 1;
}

int IMP_Generic::setd(int i, int nd)
{
    untrain();
    ML_Base::setd(i,nd);

    return 1;
}

int IMP_Generic::setd(const Vector<int> &i, const Vector<int> &nd)
{
    untrain();
    ML_Base::setd(i,nd);

    return 1;
}

int IMP_Generic::setd(const Vector<int> &nd)
{
    untrain();
    ML_Base::setd(nd);

    return 1;
}

double IMP_Generic::hypervol(void) const
{
    double retval = 0;
    gentype xminval;

    if ( N()-NNC(0) )
    {
        if ( xspaceDim() == 1 )
        {
            int i;
            gentype temp;

            xelm(xminval,0,0);

            for ( i = 1 ; i < N() ; i++ )
            {
                if ( isenabled(i) )
                {
                    if ( xelm(temp,i,0) < xminval )
                    {
                        xminval = temp;
                    }
                }
            }

            retval =  (double) xminval;
            retval -= zref();
        }

        else if ( xspaceDim() > 1 )
        {
            int M = N()-NNC(0);
            int n = xspaceDim();
            gentype temp;

            double **X;

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

            retval = h(X,M,n);

            for ( i = 0, j = 0 ; i < N() ; i++ )
            {
                if ( isenabled(i) )
                {
                    MEMDELARRAY(X[j]);

                    j++;
                }
            }

            MEMDELARRAY(X);
        }
    }

    return retval;
}







int IMP_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const
{
    int res = 0;

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    if ( ( ( ind >= 3000 ) && ( ind <= 3099 ) && !ia && !ib ) || ( ( ind >= 3100 ) && ( ind <= 3199 ) && !ib ) || ( ind <= 2999 ) )
    {
        switch ( ind )
        {
            case 3000: { val = zref();      break; }
            case 3001: { val = ehimethod(); break; }
            case 3002: { val = needdg();    break; }
            case 3003: { val = hypervol();  break; }

            case 3100:
            {
                SparseVector<gentype> xx;

                if ( convertSetToSparse(xx,xa,ia) )
                {
                    res = 1;
                }

                else
                {
                    if ( !ib )
                    {
                        imp(val,xx,xb);
                    }

                    else
                    {
                        val.force_null();
                    }
                }

                break;
            }

            default:
            {
                res = ML_Base::getparam(ind,val,xa,ia,xb,ib);

                break;
            }
        }
    }

    else
    {
        val.force_null();
    }

    return res;
}

