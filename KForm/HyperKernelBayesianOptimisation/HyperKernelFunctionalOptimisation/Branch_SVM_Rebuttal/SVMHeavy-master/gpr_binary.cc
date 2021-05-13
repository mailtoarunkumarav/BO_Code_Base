
//
// Binary Classification GPR by EP
//
// Version: 7
// Date: 18/12/2019
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "gpr_binary.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


GPR_Binary::GPR_Binary() : GPR_Scalar()
{
    setaltx(NULL);

    return;
}

GPR_Binary::GPR_Binary(const GPR_Binary &src) : GPR_Scalar()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

GPR_Binary::GPR_Binary(const GPR_Binary &src, const ML_Base *xsrc) : GPR_Scalar()
{
    setaltx(xsrc);

    assign(src,1);

    return;
}

GPR_Binary::~GPR_Binary()
{
    return;
}

double GPR_Binary::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int GPR_Binary::sety(const Vector<int> &j, const Vector<gentype> &yn)
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

int GPR_Binary::sety(const Vector<gentype> &yn)
{
    int res = 0;

    if ( yn.size() )
    {
        int i;

        for ( i = 0 ; i < yn.size() ; i++ )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int GPR_Binary::sety(const Vector<int> &j, const Vector<double> &yn)
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

int GPR_Binary::sety(const Vector<double> &yn)
{
    int res = 0;

    if ( yn.size() )
    {
        int i;

        for ( i = 0 ; i < yn.size() ; i++ )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int GPR_Binary::setd(const Vector<int> &j, const Vector<int> &yn)
{
    NiceAssert( j.size() == yn.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
        {
            res |= setd(j(i),yn(i));
        }
    }

    return res;
}

int GPR_Binary::setd(const Vector<int> &yn)
{
    int res = 0;

    if ( yn.size() )
    {
        int i;

        for ( i = 0 ; i < yn.size() ; i++ )
        {
            res |= setd(i,yn(i));
        }
    }

    return res;
}

int GPR_Binary::setd(int i, int xd)
{
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) );

    int res = 0;

    if ( xd != d()(i) )
    {
        res = 1;

        bintraintarg("&",i) = xd;

        GPR_Scalar::setd(i,xd);
    }

    return res;
}

int GPR_Binary::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToIntegerWithoutLoss() && ( ( (int) y == -1 ) || ( (int) y == 0 ) || ( (int) y == +1 ) ) );

    bintraintarg.add(i);
    bintraintarg("&",i) = y;

    static gentype dummy(0.0);

    int res = 0;

    res |= GPR_Scalar::addTrainingVector(i,dummy,x,Cweigh,epsweigh);
    res |= GPR_Scalar::setd(i,(int) y);

    return res;
}

int GPR_Binary::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isCastableToIntegerWithoutLoss() && ( ( (int) y == -1 ) || ( (int) y == 0 ) || ( (int) y == +1 ) ) );

    bintraintarg.add(i);
    bintraintarg("&",i) = (int) y;

    static gentype dummy(0.0);

    int res = 0;

    res |= GPR_Scalar::qaddTrainingVector(i,dummy,x,Cweigh,epsweigh);
    res |= GPR_Scalar::setd(i,(int) y);

    return res;
}

int GPR_Binary::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    y = bintraintarg(i);
    bintraintarg.remove(i);

    gentype dummy;

    return GPR_Scalar::removeTrainingVector(i,dummy,x);
}

int GPR_Binary::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int tempresh = GPR_Scalar::ghTrainingVector(resh,resg,i,retaltg,pxyprodi);
    double resgd = (double) resg;

    resh = zeroint();

    if ( resgd > 0 ) { resh = 1; }
    if ( resgd < 0 ) { resh = -1; }

    return tempresh;
}

std::ostream &GPR_Binary::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Binary GPR\n\n";

    repPrint(output,'>',dep) << "y: " << bintraintarg << "\n";
    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base GPR: ";
    GPR_Scalar::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &GPR_Binary::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> bintraintarg;
    input >> dummy;
    GPR_Scalar::inputstream(input);

    return input;
}

int GPR_Binary::prealloc(int expectedN)
{
    bintraintarg.prealloc(expectedN);
    GPR_Scalar::prealloc(expectedN);

    return 0;
}

int GPR_Binary::preallocsize(void) const
{
    return GPR_Scalar::preallocsize();
}


