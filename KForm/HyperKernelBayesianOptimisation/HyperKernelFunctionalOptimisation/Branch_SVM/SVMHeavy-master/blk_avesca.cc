
//
// Average result block
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
#include "blk_avesca.h"


std::ostream &BLK_AveSca::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Scalar Average BLK\n\n";

    repPrint(output,'>',dep) << "Class labels: " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts: " << classcnt    << "\n\n";

    BLK_Generic::printstream(output,dep+1);

    return output;
}

std::istream &BLK_AveSca::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;

    BLK_Generic::inputstream(input);

    return input;
}

BLK_AveSca::BLK_AveSca(int isIndPrune) : BLK_Generic(isIndPrune)
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

BLK_AveSca::BLK_AveSca(const BLK_AveSca &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_AveSca::BLK_AveSca(const BLK_AveSca &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_AveSca::~BLK_AveSca()
{
    return;
}




double BLK_AveSca::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
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


int BLK_AveSca::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xx(x);

    return qaddTrainingVector(i,y,xx,Cweigh,epsweigh);
}

int BLK_AveSca::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToRealWithoutLoss() );

    classcnt("&",3)++;

    BLK_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    return 1;
}

int BLK_AveSca::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size()        );
    NiceAssert( y.size() == Cweigh.size()   );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;
    int j;

    if ( y.size() )
    {
        for ( j = 0 ; j < y.size() ; j++ )
        {
            res |= addTrainingVector(i+j,y(j),x(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int BLK_AveSca::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size()        );
    NiceAssert( y.size() == Cweigh.size()   );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;
    int j;

    if ( y.size() )
    {
        for ( j = 0 ; j < y.size() ; j++ )
        {
            res |= qaddTrainingVector(i+j,y(j),x("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int BLK_AveSca::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        classcnt("&",d()(i)+1)--;
    }

    BLK_Generic::removeTrainingVector(i,y,x);

    return 1;
}

int BLK_AveSca::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isCastableToRealWithoutLoss() );

    BLK_Generic::sety(i,yy);

    return 1;
}

int BLK_AveSca::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int BLK_AveSca::sety(const Vector<gentype> &y)
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

int BLK_AveSca::setd(int i, int dd)
{
    NiceAssert( ( dd == +2 ) || ( dd == 0 ) );

    BLK_Generic::setd(i,dd);

    return 1;
}

int BLK_AveSca::setd(const Vector<int> &i, const Vector<int> &d)
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

int BLK_AveSca::setd(const Vector<int> &d)
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





































int BLK_AveSca::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    resg = 0.0;

    if ( xindsize(i) )
    {
        int j;
        gentype dummy;

        for ( j = 0 ; j < xindsize(i) ; j++ )
        {
            resg += xelm(dummy,i,j);
        }

        resg *= (1.0/xindsize(i));
    }

    if ( outfn().isValNull() )
    {
	resh = resg;
    }

    else
    {
        resh = outfn()(resg);
    }

    return 0;
}

void BLK_AveSca::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    res.resize(N());

    gentype zerotemplate(0.0);

    res  = zerotemplate;
    resn = zerotemplate;

    if ( i >= 0 )
    {
        res("&",i) = ( xindsize(i) ? 1.0/xindsize(i) : 1.0 );
    }

    else
    {
        resn = ( xindsize(i) ? 1.0/xindsize(i) : 1.0 );
    }

    return;
}

