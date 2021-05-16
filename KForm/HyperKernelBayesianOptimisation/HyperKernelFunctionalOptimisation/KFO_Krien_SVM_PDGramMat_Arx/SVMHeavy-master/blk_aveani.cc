
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
#include "blk_aveani.h"


std::ostream &BLK_AveAni::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Scalar Average BLK\n\n";

    repPrint(output,'>',dep) << "Class labels: " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts: " << classcnt    << "\n\n";
    repPrint(output,'>',dep) << "Order:        " << dorder      << "\n\n";

    BLK_Generic::printstream(output,dep+1);

    return output;
}

std::istream &BLK_AveAni::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;
    input >> dummy; input >> dorder;

    BLK_Generic::inputstream(input);

    return input;
}

BLK_AveAni::BLK_AveAni(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    classlabels.resize(1);
    classcnt.resize(2); // includes class 0 (other two don't)

    classlabels("&",0) = +2;
    classcnt = zeroint();

    dorder = 0;

    return;
}

BLK_AveAni::BLK_AveAni(const BLK_AveAni &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_AveAni::BLK_AveAni(const BLK_AveAni &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_AveAni::~BLK_AveAni()
{
    return;
}

double BLK_AveAni::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

int BLK_AveAni::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xx(x);

    return qaddTrainingVector(i,y,xx,Cweigh,epsweigh);
}

int BLK_AveAni::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToAnionWithoutLoss() );

    classcnt("&",1)++;

    BLK_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    if ( y_unsafe()("&",i).dir_anion().order() > dorder )
    {
        dorder = y_unsafe()("&",i).dir_anion().order();
    }

    return 1;
}

int BLK_AveAni::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

int BLK_AveAni::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

int BLK_AveAni::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        classcnt("&",d()(i)/2)--;
    }

    BLK_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

int BLK_AveAni::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isCastableToAnionWithoutLoss() );

    BLK_Generic::sety(i,yy);

    if ( y_unsafe()("&",i).dir_anion().order() > dorder )
    {
        dorder = y_unsafe()("&",i).dir_anion().order();
    }

    return 1;
}

int BLK_AveAni::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int BLK_AveAni::sety(const Vector<gentype> &y)
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

int BLK_AveAni::setd(int i, int dd)
{
    NiceAssert( ( dd == +2 ) || ( dd == 0 ) );

    BLK_Generic::setd(i,dd);

    return 1;
}

int BLK_AveAni::setd(const Vector<int> &i, const Vector<int> &d)
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

int BLK_AveAni::setd(const Vector<int> &d)
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

int BLK_AveAni::setorder(int neword)
{
    NiceAssert( neword >= 0 );

    dorder = neword;

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_anion().setorder(dorder);
        }
    }

    return 1;
}





































int BLK_AveAni::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
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

void BLK_AveAni::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
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

