
//
// k-nearest-neighbour anionic regressor
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
#include "knn_anions.h"


std::ostream &KNN_Anions::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Anionic KNN\n\n";

    repPrint(output,'>',dep) << "Class labels:  " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts:  " << classcnt    << "\n";
    repPrint(output,'>',dep) << "Order:         " << dorder      << "\n";
    repPrint(output,'>',dep) << "Class targets: " << z           << "\n\n";

    KNN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &KNN_Anions::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;
    input >> dummy; input >> dorder;
    input >> dummy; input >> z;

    KNN_Generic::inputstream(input);

    return input;
}

KNN_Anions::KNN_Anions() : KNN_Generic()
{
    setaltx(NULL);

    classlabels.resize(1);
    classcnt.resize(2); // includes class 0 (other two don't)

    classlabels("&",0) = +2;
    classcnt = zeroint();

    dorder = 0;

    return;
}

KNN_Anions::KNN_Anions(const KNN_Anions &src) : KNN_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

KNN_Anions::KNN_Anions(const KNN_Anions &src, const ML_Base *xsrc) : KNN_Generic()
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

KNN_Anions::~KNN_Anions()
{
    return;
}

double KNN_Anions::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

int KNN_Anions::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToAnionWithoutLoss() );

    classcnt("&",1)++;

    z.add(i); z("&",i) = (const d_anion &) y;

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = 2;

    if ( y_unsafe()("&",i).dir_anion().order() > dorder )
    {
        dorder = y_unsafe()("&",i).dir_anion().order();
    }

    return 1;
}

int KNN_Anions::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToAnionWithoutLoss() );

    classcnt("&",1)++;

    z.add(i); z("&",i) = (const d_anion &) y;

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = 2;

    if ( y_unsafe()("&",i).dir_anion().order() > dorder )
    {
        dorder = y_unsafe()("&",i).dir_anion().order();
    }

    return 1;
}

int KNN_Anions::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

int KNN_Anions::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

int KNN_Anions::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        classcnt("&",dd(i)/2)--;
    }

    z.remove(i);

    KNN_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

int KNN_Anions::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isCastableToAnionWithoutLoss() );

    z("&",i) = (const d_anion &) yy;

    KNN_Generic::sety(i,yy);

    if ( y_unsafe()("&",i).dir_anion().order() > dorder )
    {
        dorder = y_unsafe()("&",i).dir_anion().order();
    }

    return 1;
}

int KNN_Anions::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int KNN_Anions::sety(const Vector<gentype> &y)
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

int KNN_Anions::setd(int i, int dd)
{
    NiceAssert( ( dd == +2 ) || ( dd == 0 ) );

    KNN_Generic::setd(i,dd);

    return 1;
}

int KNN_Anions::setd(const Vector<int> &i, const Vector<int> &d)
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

int KNN_Anions::setd(const Vector<int> &d)
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

void KNN_Anions::hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;
    (void) effkay;

    if ( !res.isValAnion() ) { res.force_anion(); }
    setzero(res.dir_anion());

    if ( yk.size() )
    {
        mean(res,yk,weights);
    }

    return;
}

void KNN_Anions::hfn(d_anion &res, const Vector<d_anion> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;
    (void) effkay;

    setzero(res);

    if ( yk.size() )
    {
        mean(res,yk,weights);
    }

    return;
}

int KNN_Anions::randomise(double sparsity)
{
    (void) sparsity;

    int res = 0;
    int Nnotz = N()-NNC(0);

    if ( Nnotz )
    {
        res = 1;

        retVector<int> tmpva;

        Vector<int> canmod(cntintvec(N(),tmpva));

        int i,j,k;

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

            d_anion &bmod = y_unsafe()("&",j).dir_anion();

            if ( bmod.size() )
            {
                for ( k = 0 ; k < bmod.size() ; k++ )
                {
                    double &amod = bmod("&",k);

                    setrand(amod);
                    amod = lbloc+((ubloc-lbloc)*amod);
                }
            }

            z("&",j) = bmod;
        }
    }

    return res;
}

int KNN_Anions::setorder(int neword)
{
    NiceAssert( neword >= 0 );

    dorder = neword;

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_anion().setorder(dorder);
            z("&",ii).setorder(dorder);
        }
    }

    return 1;
}

