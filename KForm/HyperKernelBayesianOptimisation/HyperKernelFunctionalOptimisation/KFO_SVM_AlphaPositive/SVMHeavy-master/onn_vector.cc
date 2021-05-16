
//
// 1 layer neural network vector regression
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
#include "onn_vector.h"


std::ostream &ONN_Vector::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector ONN\n\n";

    repPrint(output,'>',dep) << "Class labels: " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts: " << classcnt    << "\n\n";

    ONN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &ONN_Vector::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;

    ONN_Generic::inputstream(input);

    return input;
}

ONN_Vector::ONN_Vector() : ONN_Generic()
{
    setaltx(NULL);

    classlabels.resize(1);
    classcnt.resize(2); // includes class 0 (other two don't)

    classlabels("&",0) = +2;
    classcnt = zeroint();





    return;
}

ONN_Vector::ONN_Vector(const ONN_Vector &src) : ONN_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

ONN_Vector::ONN_Vector(const ONN_Vector &src, const ML_Base *xsrc) : ONN_Generic()
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

ONN_Vector::~ONN_Vector()
{
    return;
}

double ONN_Vector::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}
























int ONN_Vector::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> temp(x);

    return qaddTrainingVector(i,y,temp,Cweigh,epsweigh);
}

int ONN_Vector::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<SparseVector<gentype> > temp(x);

    return qaddTrainingVector(i,y,temp,Cweigh,epsweigh);
}

int ONN_Vector::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToVectorWithoutLoss() );
    NiceAssert( ( !N() ) || ( y.size() == tspaceDim() ) );

    if ( !N() )
    {
        settspaceDim(y.size());
    }

    classcnt("&",1)++;
    ONN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    return 1;
}

int ONN_Vector::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

int ONN_Vector::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        classcnt("&",d()(i)/2)--;
    }

    ONN_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

int ONN_Vector::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isCastableToVectorWithoutLoss() );
    NiceAssert( yy.size() == tspaceDim() );







    ONN_Generic::sety(i,yy);






    return 1;
}

int ONN_Vector::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int ONN_Vector::sety(const Vector<gentype> &y)
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

int ONN_Vector::setd(int i, int dd)
{
    NiceAssert( ( dd == +2 ) || ( dd == 0 ) );

    ONN_Generic::setd(i,dd);








    return 1;
}

int ONN_Vector::setd(const Vector<int> &i, const Vector<int> &d)
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

int ONN_Vector::setd(const Vector<int> &d)
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
