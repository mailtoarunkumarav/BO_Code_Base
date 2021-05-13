
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
#include "knn_gentyp.h"


std::ostream &KNN_Gentyp::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Scalar KNN\n\n";

    repPrint(output,'>',dep) << "Class labels:  " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts:  " << classcnt    << "\n";

    KNN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &KNN_Gentyp::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;

    KNN_Generic::inputstream(input);

    return input;
}

KNN_Gentyp::KNN_Gentyp() : KNN_Generic()
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

KNN_Gentyp::KNN_Gentyp(const KNN_Gentyp &src) : KNN_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

KNN_Gentyp::KNN_Gentyp(const KNN_Gentyp &src, const ML_Base *xsrc) : KNN_Generic()
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

KNN_Gentyp::~KNN_Gentyp()
{
    return;
}

double KNN_Gentyp::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( ha.isValNull() || ha.isValInteger() || ha.isValReal() )
    {
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
    }

    else if ( ha.isValAnion() || ha.isValVector() || ha.isValMatrix() )
    {
        NiceAssert( ( db == 0 ) || ( db == 2 ) );

        if ( db )
        {
            res = (double) norm2(ha-hb);
        }
    }

    else
    {
        // Sets, graphs and strings are comparable by binary multiplication

        NiceAssert( ( db == 0 ) || ( db == 2 ) );

        if ( db )
        {
            res = ( ha == hb ) ? 0 : 1;
        }
    }

    return res;
}

int KNN_Gentyp::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    classcnt("&",3)++;

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = 2;

    return 1;
}

int KNN_Gentyp::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    classcnt("&",3)++;

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = 2;

    return 1;
}

int KNN_Gentyp::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

int KNN_Gentyp::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

int KNN_Gentyp::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        classcnt("&",dd(i)+1)--;
    }

    KNN_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

void KNN_Gentyp::hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;
    (void) effkay;

    if ( yk.size() )
    {
        mean(res,yk,weights);
    }

    else
    {
        res.force_null();
    }

    return;
}
