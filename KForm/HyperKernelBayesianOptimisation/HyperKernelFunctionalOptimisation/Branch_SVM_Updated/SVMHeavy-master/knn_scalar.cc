
//
// k-nearest-neighbour scalar regression
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
#include "knn_scalar.h"


std::ostream &KNN_Scalar::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Scalar KNN\n\n";

    repPrint(output,'>',dep) << "Class labels:  " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts:  " << classcnt    << "\n";
    repPrint(output,'>',dep) << "Class targets: " << z           << "\n\n";

    KNN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &KNN_Scalar::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;
    input >> dummy; input >> z;

    KNN_Generic::inputstream(input);

    return input;
}

KNN_Scalar::KNN_Scalar() : KNN_Generic()
{
    setaltx(NULL);

    classlabels.resize(3); // include -1 and +1, even though not allowed
    classcnt.resize(4); // includes class 0 (other two don't)

    classlabels("&",0) = -1;
    classlabels("&",1) = +1;
    classlabels("&",2) = +2;

    classcnt = zeroint();

    return;
}

KNN_Scalar::KNN_Scalar(const KNN_Scalar &src) : KNN_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

KNN_Scalar::KNN_Scalar(const KNN_Scalar &src, const ML_Base *xsrc) : KNN_Generic()
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

KNN_Scalar::~KNN_Scalar()
{
    return;
}

double KNN_Scalar::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db == +1 )
    {
        // treat as lower bound constraint ha >= hb

        if ( (double) ha < (double) hb )
        {
            res = ( (double) ha ) - ( (double) hb );
            res *= res;
        }
    }

    else if ( db == -1 )
    {
        // treat as upper bound constraint ha <= hb

        if ( (double) ha > (double) hb )
        {
            res = ( (double) ha ) - ( (double) hb );
            res *= res;
        }
    }

    else if ( db )
    {
        res = ( (double) ha ) - ( (double) hb );
        res *= res;
    }

    return res;
}

int KNN_Scalar::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToRealWithoutLoss() );

    classcnt("&",3)++;

    z.add(i); z("&",i) = (double) y;

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = 2;

    return 1;
}

int KNN_Scalar::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToRealWithoutLoss() );

    classcnt("&",3)++;

    z.add(i); z("&",i) = (double) y;

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = 2;

    return 1;
}

int KNN_Scalar::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            addTrainingVector(i+ii,y(ii),x(ii),Cweigh(ii),epsweigh(ii));
        }
    }

    return 1;
}

int KNN_Scalar::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            qaddTrainingVector(i+ii,y(ii),x("&",ii),Cweigh(ii),epsweigh(ii));
        }
    }

    return 1;
}

int KNN_Scalar::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        classcnt("&",dd(i)+1)--;
    }

    z.remove(i);

    KNN_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

int KNN_Scalar::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isCastableToRealWithoutLoss() );

    z("&",i) = (double) yy;

    KNN_Generic::sety(i,yy);

    return 1;
}

int KNN_Scalar::sety(const Vector<int> &i, const Vector<gentype> &y)
{
    NiceAssert( i.size() == y.size() );

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            sety(i(ii),y(ii));
        }
    }

    return 1;
}

int KNN_Scalar::sety(const Vector<gentype> &y)
{
    NiceAssert( N() == y.size() );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            sety(ii,y(ii));
        }
    }

    return 1;
}

int KNN_Scalar::setd(int i, int dd)
{
    NiceAssert( ( dd == +2 ) || ( dd == 0 ) );

    KNN_Generic::setd(i,dd);

    return 1;
}

int KNN_Scalar::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i.size() == d.size() );

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            setd(i(ii),d(ii));
        }
    }

    return 1;
}

int KNN_Scalar::setd(const Vector<int> &d)
{
    NiceAssert( N() == d.size() );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            setd(ii,d(ii));
        }
    }

    return 1;
}

void KNN_Scalar::hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;
    (void) effkay;

    if ( !res.isValReal() ) { res.force_double(); }
    setzero(res.dir_double());

    if ( yk.size() )
    {
        mean(res,yk,weights);
    }

    return;
}

void KNN_Scalar::hfn(double &res, const Vector<double> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;
    (void) effkay;

    res = 0.0;

    if ( yk.size() )
    {
        mean(res,yk,weights);
    }

    return;
}

int KNN_Scalar::randomise(double sparsity)
{
    (void) sparsity;

    int res = 0;
    int Nnotz = N()-NNC(0);

    if ( Nnotz )
    {
        res = 1;

        retVector<int> tmpva;

        Vector<int> canmod(cntintvec(N(),tmpva));

        int i,j;

        for ( i = N()-1 ; i >= 0 ; i-- )
        {
            if ( !d()(i) )
            {
                canmod.remove(i);
            }
        }

        // Randomise

        double lbloc = -1.0;
        double ubloc = +1.0;

        for ( i = 0 ; i < canmod.size() ; i++ )
        {
            j = canmod(i);

            NiceAssert( d()(j) );

            double &amod = ML_Base::y_unsafe()("&",j).force_double();

            setrand(amod);
            amod = lbloc+((ubloc-lbloc)*amod);

            z("&",j) = amod;
        }
    }

    return res;
}

