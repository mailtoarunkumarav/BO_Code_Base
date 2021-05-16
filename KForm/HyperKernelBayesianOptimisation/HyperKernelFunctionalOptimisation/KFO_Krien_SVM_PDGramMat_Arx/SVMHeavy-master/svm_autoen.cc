
//
// Auto Encoder SVM
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "svm_autoen.h"
#include <iostream>
#include <sstream>
#include <string>


SVM_AutoEn::SVM_AutoEn() : SVM_Vector()
{
    return;
}

SVM_AutoEn::SVM_AutoEn(const SVM_AutoEn &src) : SVM_Vector(src)
{
    return;
}

SVM_AutoEn::SVM_AutoEn(const SVM_AutoEn &src, const ML_Base *xsrc) : SVM_Vector(src,xsrc)
{
    return;
}

SVM_AutoEn::~SVM_AutoEn()
{
    return;
}

double SVM_AutoEn::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

std::ostream &SVM_AutoEn::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "AutoEncoding SVM\n\n";

    repPrint(output,'>',dep) << "Vector base: \n"; SVM_Vector::printstream(output,dep+1);

    return output;
}

std::istream &SVM_AutoEn::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; SVM_Vector::inputstream(input);

    return input;
}

int SVM_AutoEn::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    (void) z;

    return addTrainingVector(i,x,Cweigh,epsweigh,2);
}

int SVM_AutoEn::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    (void) z;

    return qaddTrainingVector(i,x,Cweigh,epsweigh,2);
}

int SVM_AutoEn::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_AutoEn::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_AutoEn::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_AutoEn::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_AutoEn::addTrainingVector (int i, const SparseVector<gentype> &xx, double Cweigh, double epsweigh, int dregress)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    int res = 0;

    Vector<double> zz;

    // NB: can't xlateFromSparse until after call to addTrainingVector, as
    // it is the addTrainingVector call that updates the index key.

    res |= SVM_Vector::addTrainingVector(i,zz,xx,Cweigh,epsweigh,dregress);
    xlateFromSparseTrainingVector(zz,i);
    res |= sety(i,zz);

    NiceAssert( zz.size() == xspaceDim() );

    return res;
}

int SVM_AutoEn::qaddTrainingVector(int i, SparseVector<gentype> &xx, double Cweigh, double epsweigh, int dregress)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    int res = 0;

    Vector<double> zz;

    // NB: can't xlateFromSparse until after call to addTrainingVector, as
    // it is the addTrainingVector call that updates the index key.

    res |= SVM_Vector::qaddTrainingVector(i,zz,xx,Cweigh,epsweigh,dregress);
    xlateFromSparseTrainingVector(zz,i);
    res |= sety(i,zz);

    NiceAssert( zz.size() == xspaceDim() );

    return res;
}

int SVM_AutoEn::addTrainingVector (int i, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &dregress)
{
    NiceAssert( xx.size() == Cweigh.size() );
    NiceAssert( xx.size() == epsweigh.size() );
    NiceAssert( xx.size() == dregress.size() );

    int res = 0;

    if ( xx.size() )
    {
        int j;

        for ( j = 0 ; j < xx.size() ; j++ )
        {
            res |= SVM_AutoEn::addTrainingVector(i+j,xx(j),Cweigh(j),epsweigh(j),dregress(j));
        }
    }

    return res;
}

int SVM_AutoEn::qaddTrainingVector(int i,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &dregress)
{
    NiceAssert( xx.size() == Cweigh.size() );
    NiceAssert( xx.size() == epsweigh.size() );
    NiceAssert( xx.size() == dregress.size() );

    int res = 0;

    if ( xx.size() )
    {
        int j;

        for ( j = 0 ; j < xx.size() ; j++ )
        {
            res |= SVM_AutoEn::qaddTrainingVector(i+j,xx("&",j),Cweigh(j),epsweigh(j),dregress(j));
        }
    }

    return res;
}

int SVM_AutoEn::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &xx)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    return SVM_Vector::removeTrainingVector(i,y,xx);
}

int SVM_AutoEn::prealloc(int expectedN)
{
    SVM_Generic::prealloc(expectedN);

    return 0;
}

int SVM_AutoEn::preallocsize(void) const
{
    return SVM_Generic::preallocsize();
}
