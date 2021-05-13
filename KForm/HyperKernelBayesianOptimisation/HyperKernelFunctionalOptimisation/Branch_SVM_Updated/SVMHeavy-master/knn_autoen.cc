
//
// Auto Encoder KNN
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "knn_autoen.h"
#include <iostream>
#include <sstream>
#include <string>


KNN_AutoEn::KNN_AutoEn() : KNN_Vector()
{
    return;
}

KNN_AutoEn::KNN_AutoEn(const KNN_AutoEn &src) : KNN_Vector(src)
{
    return;
}

KNN_AutoEn::KNN_AutoEn(const KNN_AutoEn &src, const ML_Base *xsrc) : KNN_Vector(src,xsrc)
{
    return;
}

KNN_AutoEn::~KNN_AutoEn()
{
    return;
}

double KNN_AutoEn::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

std::ostream &KNN_AutoEn::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "AutoEncoding KNN\n\n";

    repPrint(output,'>',dep) << "Vector base: \n"; KNN_Vector::printstream(output,dep+1);

    return output;
}

std::istream &KNN_AutoEn::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; KNN_Vector::inputstream(input);

    return input;
}

int KNN_AutoEn::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    (void) z;

    return KNN_AutoEn::addTrainingVector(i,x,Cweigh,epsweigh);
}

int KNN_AutoEn::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    (void) z;

    return KNN_AutoEn::qaddTrainingVector(i,x,Cweigh,epsweigh);
}

int KNN_AutoEn::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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
            res |= KNN_AutoEn::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int KNN_AutoEn::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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
            res |= KNN_AutoEn::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int KNN_AutoEn::addTrainingVector (int i, const SparseVector<gentype> &xx, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xxx(xx);

    return KNN_AutoEn::qaddTrainingVector(i,xxx,Cweigh,epsweigh);
}

int KNN_AutoEn::qaddTrainingVector(int i, SparseVector<gentype> &xx, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    int res = 0;

    gentype zz('V');

    // NB: can't xlateFromSparse until after call to addTrainingVector, as
    // it is the addTrainingVector call that updates the index key.

    res |= KNN_Vector::qaddTrainingVector(i,zz,xx,Cweigh,epsweigh);
    xlateFromSparseTrainingVector(zz.force_vector(),i);
    res |= sety(i,zz);

    NiceAssert( zz.size() == xspaceDim() );

    return res;
}

int KNN_AutoEn::addTrainingVector (int i, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( xx.size() == Cweigh.size() );
    NiceAssert( xx.size() == epsweigh.size() );

    int res = 0;

    if ( xx.size() )
    {
        int j;

        for ( j = 0 ; j < xx.size() ; j++ )
        {
            res |= KNN_AutoEn::addTrainingVector(i+j,xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int KNN_AutoEn::qaddTrainingVector(int i,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( xx.size() == Cweigh.size() );
    NiceAssert( xx.size() == epsweigh.size() );

    int res = 0;

    if ( xx.size() )
    {
        int j;

        for ( j = 0 ; j < xx.size() ; j++ )
        {
            res |= KNN_AutoEn::qaddTrainingVector(i+j,xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int KNN_AutoEn::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &xx)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    return KNN_Vector::removeTrainingVector(i,y,xx);
}
