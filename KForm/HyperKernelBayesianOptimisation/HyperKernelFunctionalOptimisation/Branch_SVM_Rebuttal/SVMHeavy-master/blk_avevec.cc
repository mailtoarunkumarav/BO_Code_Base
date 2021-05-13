
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
#include "blk_avevec.h"


std::ostream &BLK_AveVec::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector Average BLK\n\n";

    repPrint(output,'>',dep) << "Class labels: " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts: " << classcnt    << "\n\n";
    repPrint(output,'>',dep) << "Dimension:    " << dim         << "\n\n";

    BLK_Generic::printstream(output,dep+1);

    return output;
}

std::istream &BLK_AveVec::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;
    input >> dummy; input >> dim;

    BLK_Generic::inputstream(input);

    return input;
}

BLK_AveVec::BLK_AveVec(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    classlabels.resize(1);
    classcnt.resize(2); // includes class 0 (other two don't)

    classlabels("&",0) = +2;
    classcnt = zeroint();

    dim = -1;

    return;
}

BLK_AveVec::BLK_AveVec(const BLK_AveVec &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_AveVec::BLK_AveVec(const BLK_AveVec &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_AveVec::~BLK_AveVec()
{
    return;
}

double BLK_AveVec::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

int BLK_AveVec::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xx(x);

    return qaddTrainingVector(i,y,xx,Cweigh,epsweigh);
}

int BLK_AveVec::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToVectorWithoutLoss() );
    NiceAssert( ( dim == -1 ) || ( y.size() == dim ) );

    if ( dim == -1 )
    {
        dim = y.size();
    }

    classcnt("&",1)++;

    BLK_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    y_unsafe()("&",i).morph_vector();

    return 1;
}

int BLK_AveVec::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

int BLK_AveVec::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

int BLK_AveVec::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        classcnt("&",d()(i)/2)--;
    }

    BLK_Generic::removeTrainingVector(i,y,x);

    return 1;
}

int BLK_AveVec::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isCastableToVectorWithoutLoss() );
    NiceAssert( ( dim == -1 ) || ( yy.size() == dim ) );

    if ( dim == -1 )
    {
        dim = yy.size();
    }

    BLK_Generic::sety(i,yy);

    y_unsafe()("&",i).morph_vector();

    return 1;
}

int BLK_AveVec::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int BLK_AveVec::sety(const Vector<gentype> &y)
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

int BLK_AveVec::setd(int i, int dd)
{
    NiceAssert( ( dd == +2 ) || ( dd == 0 ) );

    BLK_Generic::setd(i,dd);

    return 1;
}

int BLK_AveVec::setd(const Vector<int> &i, const Vector<int> &d)
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

int BLK_AveVec::setd(const Vector<int> &d)
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

int BLK_AveVec::settspaceDim(int newdim)
{
    NiceAssert( ( ( N() == 0 ) && ( newdim >= -1 ) ) || ( newdim >= 0 ) );

    dim = newdim;

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_vector().resize(dim);
        }
    }

    return 1;
}

int BLK_AveVec::addtspaceFeat(int i)
{
    NiceAssert( ( ( i >= 0 ) && ( i <= dim ) ) || ( dim == -1 ) );

    dim++;

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_vector().add(i);
        }
    }

    return 1;
}

int BLK_AveVec::removetspaceFeat(int i)
{
    NiceAssert( ( i >= 0 ) && ( i < dim ) );

    dim--;

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_vector().remove(i);
        }
    }

    return 1;
}





































int BLK_AveVec::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
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

void BLK_AveVec::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
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

