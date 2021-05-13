
//
// Linear optimisation context
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "optlinbasecontext.h"

// Stream operators

std::ostream &operator<<(std::ostream &output, const optLinBaseContext &src)
{
    output << "Gpn (empty):  " << src.Gpn  << "\n";
    output << "Gn (empty):   " << src.Gn   << "\n";
    output << "bn (empty):   " << src.bn   << "\n";
    output << "apos (empty): " << src.apos << "\n";
    output << "bpos (empty): " << src.bpos << "\n";
    output << "Q:            " << src.Q    << "\n";

    return output;
}

std::istream &operator>>(std::istream &input, optLinBaseContext &dest)
{
    wait_dummy dummy;

    input >> dummy; input >> dest.Gpn;
    input >> dummy; input >> dest.Gn;
    input >> dummy; input >> dest.bn;
    input >> dummy; input >> dest.apos;
    input >> dummy; input >> dest.bpos;
    input >> dummy; input >> dest.Q;

    return input;
}

optLinBaseContext::optLinBaseContext(void)
{
    Matrix<double> Htemp;

    Q.refact(Htemp,Gn,Gpn,1);

    return;
}

optLinBaseContext::optLinBaseContext(const optLinBaseContext &src)
{
    *this = src;

    return;
}

optLinBaseContext &optLinBaseContext::operator=(const optLinBaseContext &src)
{
    Q    = src.Q;
    Gpn  = src.Gpn;
    Gn   = src.Gn;
    bn   = src.bn;
    apos = src.apos;
    bpos = src.bpos;

    return *this;
}

int optLinBaseContext::add(int i)
{
    Gpn.addRow(i);

    return Q.addAlpha(i);
}

int optLinBaseContext::remove(int i)
{
    Gpn.removeRow(i);

    return Q.removeAlpha(i);
}
