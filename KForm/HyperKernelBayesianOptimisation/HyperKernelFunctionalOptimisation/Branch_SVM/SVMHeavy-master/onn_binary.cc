
//
// 1 layer neural network binary classification
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
#include "onn_binary.h"


std::ostream &ONN_Binary::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Binary ONN\n\n";

    repPrint(output,'>',dep) << "Class labels: " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts: " << classcnt    << "\n\n";

    ONN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &ONN_Binary::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;

    ONN_Generic::inputstream(input);

    return input;
}

ONN_Binary::ONN_Binary() : ONN_Generic()
{
    setaltx(NULL);

    getKernel_unsafe().setType(200);

    classlabels.resize(2);
    classcnt.resize(3); // includes class 0 (other two don't)

    classlabels("&",0) = -1;
    classlabels("&",1) = +1;

    classcnt("&",0) = 0;
    classcnt("&",1) = 0;
    classcnt("&",2) = 0;

    return;
}

ONN_Binary::ONN_Binary(const ONN_Binary &src) : ONN_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

ONN_Binary::ONN_Binary(const ONN_Binary &src, const ML_Base *xsrc) : ONN_Generic()
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

ONN_Binary::~ONN_Binary()
{
    return;
}

double ONN_Binary::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}
























int ONN_Binary::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> temp(x);

    return qaddTrainingVector(i,y,temp,Cweigh,epsweigh);
}

int ONN_Binary::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<SparseVector<gentype> > temp(x);

    return addTrainingVector(i,y,temp,Cweigh,epsweigh);
}

int ONN_Binary::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isValInteger() );
    NiceAssert( ( (int) y == +1 ) || ( (int) y == -1 ) );






    classcnt("&",(((int) y)+1))++;
    ONN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    return 1;
}

int ONN_Binary::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int res = 0;

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            res |= qaddTrainingVector(i+ii,y(ii),x("&",ii),Cweigh(ii),epsweigh(ii));
        }
    }

    return res;
}

int ONN_Binary::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        classcnt("&",(((int) y()(i))+1))--;
    }

    ONN_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

int ONN_Binary::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isValInteger() );
    NiceAssert( ( (int) yy == +1 ) || ( (int) yy == -1 ) );

    if ( d()(i) )
    {
        classcnt("&",(((int) y()(i))+1))--;
        classcnt("&",(((int) yy)+1))++;
    }

    ONN_Generic::sety(i,yy);

    if ( d()(i) )
    {
        ONN_Generic::setd(i,(int) yy);
    }

    return 1;
}

int ONN_Binary::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int ONN_Binary::sety(const Vector<gentype> &y)
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

int ONN_Binary::setd(int i, int dd)
{
    NiceAssert( ( dd == +1 ) || ( dd == -1 ) || ( dd == 0 ) );

    ONN_Generic::setd(i,dd);

    if ( dd )
    {
        gentype yy(dd);

        sety(i,yy);
    }

    return 1;
}

int ONN_Binary::setd(const Vector<int> &i, const Vector<int> &d)
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

int ONN_Binary::setd(const Vector<int> &d)
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

int ONN_Binary::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    ONN_Generic::ghTrainingVector(resh,resg,i,retaltg,pxyprodi);

    int res = ((double) resg) > 0 ? 1 : -1;
    resh = res;

    return res;
}
