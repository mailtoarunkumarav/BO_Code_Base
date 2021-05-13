
//
// k-nearest-neighbour binary classifier
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
#include "knn_binary.h"


std::ostream &KNN_Binary::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Binary KNN\n\n";

    repPrint(output,'>',dep) << "Class labels: " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts: " << classcnt    << "\n\n";

    KNN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &KNN_Binary::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;

    KNN_Generic::inputstream(input);

    return input;
}

KNN_Binary::KNN_Binary() : KNN_Generic()
{
    setaltx(NULL);

    classlabels.resize(2);
    classcnt.resize(3); // includes class 0 (other two don't)

    classlabels("&",0) = -1;
    classlabels("&",1) = +1;

    classcnt("&",0) = 0;
    classcnt("&",1) = 0;
    classcnt("&",2) = 0;

    return;
}

KNN_Binary::KNN_Binary(const KNN_Binary &src) : KNN_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

KNN_Binary::KNN_Binary(const KNN_Binary &src, const ML_Base *xsrc) : KNN_Generic()
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

KNN_Binary::~KNN_Binary()
{
    return;
}

double KNN_Binary::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int KNN_Binary::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isValInteger() );
    NiceAssert( ( (int) y == +1 ) || ( (int) y == -1 ) );

    classcnt("&",(((int) y)+1))++;

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = (int) y;

    return 1;
}

int KNN_Binary::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isValInteger() );
    NiceAssert( ( (int) y == +1 ) || ( (int) y == -1 ) );

    classcnt("&",(((int) y)+1))++;

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = (int) y;

    return 1;
}

int KNN_Binary::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            NiceAssert( y(ii).isValInteger() );
            NiceAssert( ( (int) y(ii) == +1 ) || ( (int) y(ii) == -1 ) );

            classcnt("&",(((int) y(ii))+1))++;
        }
    }

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            KNN_Generic::dd("&",i+ii) = (int) y(ii);
        }
    }

    return 1;
}

int KNN_Binary::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            NiceAssert( y(ii).isValInteger() );
            NiceAssert( ( (int) y(ii) == +1 ) || ( (int) y(ii) == -1 ) );

            classcnt("&",(((int) y(ii))+1))++;
        }
    }

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            KNN_Generic::dd("&",i+ii) = (int) y(ii);
        }
    }

    return 1;
}

int KNN_Binary::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        classcnt("&",(((int) y()(i))+1))--;
    }

    KNN_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

int KNN_Binary::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isValInteger() );
    NiceAssert( ( (int) yy == +1 ) || ( (int) yy == -1 ) );

    if ( d()(i) )
    {
        classcnt("&",(((int) y()(i))+1))--;
        classcnt("&",(((int) yy)+1))++;
    }

    KNN_Generic::sety(i,yy);

    if ( d()(i) )
    {
        KNN_Generic::dd("&",i) = (int) yy;
    }

    return 1;
}

int KNN_Binary::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int KNN_Binary::sety(const Vector<gentype> &y)
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

int KNN_Binary::setd(int i, int dd)
{
    NiceAssert( ( dd == +1 ) || ( dd == -1 ) || ( dd == 0 ) );

    KNN_Generic::setd(i,dd);

    if ( dd )
    {
        gentype yy(dd);

        KNN_Generic::sety(i,yy);
    }

    return 1;
}

int KNN_Binary::setd(const Vector<int> &i, const Vector<int> &d)
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

int KNN_Binary::setd(const Vector<int> &d)
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

void KNN_Binary::hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;

    if ( !res.isValInteger() ) { res.force_int(); }
    setzero(res.dir_int());

    // Take 1: simple vote
    // Take 2: which is closest?
    // Take 3: arbitrarily label -1
    //
    // Assumption y=+/-1 as required

    if ( effkay )
    {
        gentype temp;

        double tally = (double) sum(temp,yk,weights);

        if ( tally > 0 )
        {
            res = +1;
        }

        else if ( tally < 0 )
        {
            res = -1;
        }

        else
        {
            res = (int) yk(zeroint());
        }
    }

    else
    {
        res = -1; // This is arbitrary
    }

    return;
}


int KNN_Binary::randomise(double sparsity)
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

        for ( i = 0 ; i < canmod.size() ; i++ )
        {
            j = canmod(i);

            NiceAssert( d()(j) );

            int &amod = y_unsafe()("&",j).force_int();

            setrand(amod);
        }
    }

    return res;
}
