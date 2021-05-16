
//
// Multi-user Binary Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_mulbin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>




SVM_MulBin::SVM_MulBin() : SVM_MvRank()
{
    setaltx(NULL);

    return;
}

SVM_MulBin::SVM_MulBin(const SVM_MulBin &src) : SVM_MvRank()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

SVM_MulBin::SVM_MulBin(const SVM_MulBin &src, const ML_Base *xsrc) : SVM_MvRank()
{
    setaltx(xsrc);

    assign(src,1);

    return;
}

SVM_MulBin::~SVM_MulBin()
{
    return;
}

std::ostream &SVM_MulBin::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Multi-expert Binary SVM\n\n";

    repPrint(output,'>',dep) << "y:  " << locy  << "\n";
    repPrint(output,'>',dep) << "zR: " << loczR << "\n";
    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base SVM_MvRank: ";
    SVM_MvRank::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_MulBin::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> locy;
    input >> dummy; input >> loczR;
    input >> dummy;
    SVM_MvRank::inputstream(input);

    return input;
}

int SVM_MulBin::prealloc(int expectedN)
{
    locy.prealloc(expectedN);
    loczR.prealloc(expectedN);
    SVM_MvRank::prealloc(expectedN);

    return 0;
}

int SVM_MulBin::preallocsize(void) const
{
    return SVM_MvRank::preallocsize();
}

int SVM_MulBin::qaddTrainingVector(int i, int d, SparseVector<gentype> &x, double Cweigh, double epsweigh, double z)
{
    NiceAssert( i <= N() );

    locy.add(i);  locy("&",i)  = d;
    loczR.add(i); loczR("&",i) = z;

    return SVM_MvRank::qaddTrainingVector(i,z+d,x,Cweigh,epsweigh,d);
}

int SVM_MulBin::sety(int i, double z)
{
    NiceAssert( i < N() );

    loczR("&",i) = z;

    gentype zz(((double) loczR(i))+((int) locy(i)));

    return SVM_MvRank::sety(i,zz);
}

int SVM_MulBin::setd(int i, int d)
{
    NiceAssert( i < N() );

    int res = 0;

    locy("&",i) = d;

    gentype zz(((double) loczR(i))+((int) locy(i)));

    res |= SVM_MvRank::setd(i,d);
    res |= SVM_MvRank::sety(i,zz);

    return res;
}

int SVM_MulBin::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i < N() );

    y = locy(i);

    locy.remove(i);
    loczR.remove(i);

    gentype dummy;

    return SVM_MvRank::removeTrainingVector(i,dummy,x);
}






















int SVM_MulBin::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    int d = (int) z;

    return qaddTrainingVector(i,d,x,Cweigh,epsweigh,0.0);
}

int SVM_MulBin::sety(int i, const gentype &z)
{
    int d = (int) z;

    return setd(i,d);
}






















int SVM_MulBin::addTrainingVector(int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xx(x);

    return qaddTrainingVector(i,z,xx,Cweigh,epsweigh);
}

int SVM_MulBin::addTrainingVector(int i, int d, const SparseVector<gentype> &x, double Cweigh, double epsweigh, double z)
{
    SparseVector<gentype> xx(x);

    return qaddTrainingVector(i,d,xx,Cweigh,epsweigh,z);
}

int SVM_MulBin::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= addTrainingVector(i+j,z(j),x(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_MulBin::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= qaddTrainingVector(i+j,z(j),x("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_MulBin::addTrainingVector(int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z)
{
    NiceAssert( d.size() == x.size() );
    NiceAssert( d.size() == Cweigh.size() );
    NiceAssert( d.size() == epsweigh.size() );
    NiceAssert( d.size() == z.size() );

    int res = 0;

    if ( d.size() )
    {
        int j;

        for ( j = 0 ; j < d.size() ; j++ )
        {
            res |= addTrainingVector(i+j,d(j),x(j),Cweigh(j),epsweigh(j),z(j));
        }
    }

    return res;
}

int SVM_MulBin::qaddTrainingVector(int i, const Vector<int> &d, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z)
{
    NiceAssert( d.size() == x.size() );
    NiceAssert( d.size() == Cweigh.size() );
    NiceAssert( d.size() == epsweigh.size() );
    NiceAssert( d.size() == z.size() );

    int res = 0;

    if ( d.size() )
    {
        int j;

        for ( j = 0 ; j < d.size() ; j++ )
        {
            res |= qaddTrainingVector(i+j,d(j),x("&",j),Cweigh(j),epsweigh(j),z(j));
        }
    }

    return res;
}

int SVM_MulBin::sety(const Vector<int> &i, const Vector<double> &z)
{
    NiceAssert( i.size() == z.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= sety(i(j),z(j));
        }
    }

    return res;
}

int SVM_MulBin::sety(const Vector<double> &z)
{
    NiceAssert( N() == z.size() );

    int res = 0;

    if ( N() )
    {
        int j;

        for ( j = 0 ; j < N() ; j++ )
        {
            res |= sety(j,z(j));
        }
    }

    return res;
}

int SVM_MulBin::sety(const Vector<int> &i, const Vector<gentype> &z)
{
    NiceAssert( i.size() == z.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= sety(i(j),z(j));
        }
    }

    return res;
}

int SVM_MulBin::sety(const Vector<gentype> &z)
{
    NiceAssert( N() == z.size() );

    int res = 0;

    if ( N() )
    {
        int j;

        for ( j = 0 ; j < N() ; j++ )
        {
            res |= sety(j,z(j));
        }
    }

    return res;
}

int SVM_MulBin::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i.size() == d.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= setd(i(j),d(j));
        }
    }

    return res;
}

int SVM_MulBin::setd(const Vector<int> &d)
{
    NiceAssert( N() == d.size() );

    int res = 0;

    if ( N() )
    {
        int j;

        for ( j = 0 ; j < N() ; j++ )
        {
            res |= setd(j,d(j));
        }
    }

    return res;
}












