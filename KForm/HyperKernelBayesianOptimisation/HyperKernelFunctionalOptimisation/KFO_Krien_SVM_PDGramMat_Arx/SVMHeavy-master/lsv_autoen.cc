
//
// Auto Encoder LSV
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "lsv_autoen.h"
#include <iostream>
#include <sstream>
#include <string>


LSV_AutoEn::LSV_AutoEn() : LSV_Vector()
{
    return;
}

LSV_AutoEn::LSV_AutoEn(const LSV_AutoEn &src) : LSV_Vector(src)
{
    return;
}

LSV_AutoEn::LSV_AutoEn(const LSV_AutoEn &src, const ML_Base *xsrc) : LSV_Vector(src,xsrc)
{
    return;
}

LSV_AutoEn::~LSV_AutoEn()
{
    return;
}

double LSV_AutoEn::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

std::ostream &LSV_AutoEn::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "AutoEncoding LSV\n\n";

    repPrint(output,'>',dep) << "Vector base: \n"; LSV_Vector::printstream(output,dep+1);

    return output;
}

std::istream &LSV_AutoEn::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; LSV_Vector::inputstream(input);

    return input;
}

int LSV_AutoEn::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    (void) z;

    return LSV_AutoEn::addTrainingVector(i,x,Cweigh,epsweigh);
}

int LSV_AutoEn::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    (void) z;

    return LSV_AutoEn::qaddTrainingVector(i,x,Cweigh,epsweigh);
}

int LSV_AutoEn::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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
            res |= LSV_AutoEn::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int LSV_AutoEn::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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
            res |= LSV_AutoEn::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int LSV_AutoEn::addTrainingVector (int i, const SparseVector<gentype> &xx, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xxx(xx);

    return LSV_AutoEn::qaddTrainingVector(i,xxx,Cweigh,epsweigh);
}

int LSV_AutoEn::qaddTrainingVector(int i, SparseVector<gentype> &xx, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    int res = 0;

    gentype zz('V');

    // NB: can't xlateFromSparse until after call to addTrainingVector, as
    // it is the addTrainingVector call that updates the index key.

    res |= LSV_Vector::qaddTrainingVector(i,zz,xx,Cweigh,epsweigh);
    xlateFromSparseTrainingVector(zz.force_vector(),i);
    res |= sety(i,zz);

    NiceAssert( zz.size() == xspaceDim() );

    return res;
}

int LSV_AutoEn::addTrainingVector (int i, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( xx.size() == Cweigh.size() );
    NiceAssert( xx.size() == epsweigh.size() );

    int res = 0;

    if ( xx.size() )
    {
        int j;

        for ( j = 0 ; j < xx.size() ; j++ )
        {
            res |= LSV_AutoEn::addTrainingVector(i+j,xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int LSV_AutoEn::qaddTrainingVector(int i,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( xx.size() == Cweigh.size() );
    NiceAssert( xx.size() == epsweigh.size() );

    int res = 0;

    if ( xx.size() )
    {
        int j;

        for ( j = 0 ; j < xx.size() ; j++ )
        {
            res |= LSV_AutoEn::qaddTrainingVector(i+j,xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int LSV_AutoEn::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &xx)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    return LSV_Vector::removeTrainingVector(i,y,xx);
}
