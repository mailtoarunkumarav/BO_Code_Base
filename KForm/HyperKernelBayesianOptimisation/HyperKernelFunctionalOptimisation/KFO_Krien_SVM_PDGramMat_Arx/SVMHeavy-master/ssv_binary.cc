
//
// Binary Classification SSV
//
// Version: 7
// Date: 01/12/2017
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "ssv_binary.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


class SSV_Single;

SSV_Binary::SSV_Binary() : SSV_Scalar()
{
    setaltx(NULL);

    return;
}

SSV_Binary::SSV_Binary(const SSV_Binary &src) : SSV_Scalar()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

SSV_Binary::SSV_Binary(const SSV_Binary &src, const ML_Base *xsrc) : SSV_Scalar()
{
    setaltx(xsrc);

    assign(src,1);

    return;
}

SSV_Binary::~SSV_Binary()
{
    return;
}

double SSV_Binary::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int SSV_Binary::sety(int i, const gentype &zn)
{
    return setd(i,(int) zn);
}

int SSV_Binary::sety(const Vector<int> &j, const Vector<gentype> &yn)
{
    NiceAssert( j.size() == yn.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
        {
            res |= sety(j(i),yn(i));
        }
    }

    return res;
}

int SSV_Binary::sety(const Vector<gentype> &yn)
{
    NiceAssert( SSV_Binary::N() == yn.size() );

    int res = 0;

    if ( SSV_Binary::N() )
    {
        int i;

        for ( i = 0 ; i < SSV_Binary::N() ; i++ )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int SSV_Binary::setd(int i, int xd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SSV_Binary::N() );
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) );

    int res = 0;

    if ( xd != d()(i) )
    {
        res = 1;

        setdinternal(i,xd);
    }

    return res;
}

int SSV_Binary::setd(const Vector<int> &j, const Vector<int> &d)
{
    NiceAssert( j.size() == d.size() );

    int res = 0;

    if ( j.size() )
    {
        res = 1;

        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setd(j(i),d(i));
	}
    }

    return res;
}

int SSV_Binary::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == SSV_Binary::N() );

    int i;
    int res = 0;

    if ( SSV_Binary::N() )
    {
        for ( i = 0 ; i < SSV_Binary::N() ; i++ )
	{
            res |= setd(i,d(i));
	}
    }

    return res;
}

int SSV_Binary::setdinternal(int i, int xd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SSV_Binary::N() );
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) );

    int res = 0;

    if ( xd != d()(i) )
    {
        res = 1;

        gentype dtmp(d()(i));

        res |= SSV_Scalar::setd(i,xd);
        res |= SSV_Scalar::sety(i,dtmp);
    }

    return res;
}

int SSV_Binary::addTrainingVector(int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xx(x);

    return qaddTrainingVector(i,z,xx,Cweigh,epsweigh);
}

int SSV_Binary::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SSV_Binary::N() );

    int xd = (int) z;

    gentype inz(xd);

    int res = 0;

    res |= SSV_Scalar::qaddTrainingVector(i,inz,x,Cweigh,epsweigh);
    res |= SSV_Scalar::setd(i,xd);

    return res;
}

int SSV_Binary::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            res |= SSV_Binary::addTrainingVector(i+j,y(j),x(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SSV_Binary::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            res |= SSV_Binary::qaddTrainingVector(i+j,y(j),x("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SSV_Binary::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SSV_Binary::N() );

    int res = SSV_Scalar::removeTrainingVector(i,y,x);

    return res;
}

std::ostream &SSV_Binary::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Binary SSV\n\n";

    repPrint(output,'>',dep) << "Base SVR:                        ";
    SSV_Scalar::printstream(output,dep+1);

    return output;
}

std::istream &SSV_Binary::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy;
    SSV_Scalar::inputstream(input);

    return input;
}

int SSV_Binary::prealloc(int expectedN)
{
    SSV_Scalar::prealloc(expectedN);

    return 0;
}

int SSV_Binary::preallocsize(void) const
{
    return SSV_Scalar::preallocsize();
}

int SSV_Binary::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int res = SSV_Scalar::ghTrainingVector(resh,resg,i,retaltg,pxyprodi);

    resh.force_int() = res;

    return res;
}

