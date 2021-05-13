
//
// k-nearest-neighbour density estimation
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
#include "knn_densit.h"


std::ostream &KNN_Densit::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "KNN Density Estimator\n";

    KNN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &KNN_Densit::inputstream(std::istream &input )
{
    KNN_Generic::inputstream(input);

    return input;
}

KNN_Densit::KNN_Densit() : KNN_Generic()
{
    setaltx(NULL);

    return;
}

KNN_Densit::KNN_Densit(const KNN_Densit &src) : KNN_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

KNN_Densit::KNN_Densit(const KNN_Densit &src, const ML_Base *xsrc) : KNN_Generic()
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

KNN_Densit::~KNN_Densit()
{
    return;
}

double KNN_Densit::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( (double) ha ) - ( (double) hb );
        res *= res;
    }

    return res;
}

int KNN_Densit::NNC(int xd) const
{
    int res = 0;

    if ( xd )
    {
        res = sum(d())/2;
    }

    else
    {
        res = N()-(sum(d())/2);
    }

    return res;
}

const Vector<int> &KNN_Densit::ClassLabels(void) const
{
    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    svmvolatile static Vector<int> res(3);
    svmvolatile static int firstrun = 1;

    if ( firstrun )
    {
        firstrun = 0;

        const_cast<Vector<int> &>(res)("&",0) = -1;
        const_cast<Vector<int> &>(res)("&",1) = +1;
        const_cast<Vector<int> &>(res)("&",2) = 2;
    }

    svm_mutex_unlock(eyelock);

    return const_cast<Vector<int> &>(res);
}

int KNN_Densit::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isValNull() );

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);

    return 1;
}

int KNN_Densit::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isValNull() );

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    return 1;
}

int KNN_Densit::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            NiceAssert( y(ii).isValNull() );
        }
    }

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);

    return 1;
}

int KNN_Densit::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            NiceAssert( y(ii).isValNull() );
        }
    }

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    return 1;
}

int KNN_Densit::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isValNull() );

    KNN_Generic::sety(i,yy);

    return 1;
}

int KNN_Densit::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int KNN_Densit::sety(const Vector<gentype> &y)
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



void KNN_Densit::hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) yk;
    (void) weights;

    if ( !res.isValReal() ) { res.force_double(); }
    res = 1.0;

    // p(x) = K/NV
    //
    // K = number of samples in sphere
    // N = total number of non-constrained vectors
    // V = volume of minimal sphere containing points

    if ( Nnz )
    {
        double V = spherevol(kdistsq(effkay-1),xspaceDim());

        if ( V < 1e-12 ) { V = 1e-12; }

        res = effkay/(Nnz*V);
    }

    return;
}





















