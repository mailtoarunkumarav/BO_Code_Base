
//
// Scalar function type
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
#include "scalarfn.h"


scalarfn::scalarfn() : gentype(0.0)
{
    varid_xi = DEFAULTVARI;
    varid_xj = DEFAULTVARJ;

    numpts = DEFAULT_INTEGRAL_SLICES;

    return;
}

scalarfn::scalarfn(const scalarfn &src) : gentype(static_cast<const gentype &>(src))
{
    varid_xi = DEFAULTVARI;
    varid_xj = DEFAULTVARJ;

    numpts = DEFAULT_INTEGRAL_SLICES;

    return;
}

scalarfn::scalarfn(const gentype &src) : gentype(src)
{
    varid_xi = DEFAULTVARI;
    varid_xj = DEFAULTVARJ;

    numpts = DEFAULT_INTEGRAL_SLICES;

    return;
}

scalarfn::scalarfn(const double &src) : gentype(src)
{
    varid_xi = DEFAULTVARI;
    varid_xj = DEFAULTVARJ;

    numpts = DEFAULT_INTEGRAL_SLICES;

    return;
}

scalarfn::scalarfn(const Vector<double> &src) : gentype(src)
{
    varid_xi = DEFAULTVARI;
    varid_xj = DEFAULTVARJ;

    numpts = DEFAULT_INTEGRAL_SLICES;

    return;
}

scalarfn::scalarfn(const std::string &src) : gentype(src)
{
    varid_xi = DEFAULTVARI;
    varid_xj = DEFAULTVARJ;

    numpts = DEFAULT_INTEGRAL_SLICES;

    return;
}

scalarfn::scalarfn(const char *src) : gentype(src)
{
    varid_xi = DEFAULTVARI;
    varid_xj = DEFAULTVARJ;

    numpts = DEFAULT_INTEGRAL_SLICES;

    return;
}

scalarfn &scalarfn::operator=(const scalarfn &src)
{
    varid_xi = src.varid_xi;
    varid_xj = src.varid_xj;

    numpts = src.numpts;

    static_cast<gentype &>(*this) = src;

    return *this;
}

scalarfn &scalarfn::operator=(const gentype &src)
{
    static_cast<gentype &>(*this) = src;

    return *this;
}

scalarfn &scalarfn::operator=(const double &src)
{
    static_cast<gentype &>(*this) = src;

    return *this;
}

scalarfn &scalarfn::operator=(const Vector<double> &src)
{
    static_cast<gentype &>(*this) = src;

    return *this;
}

scalarfn &scalarfn::operator=(const std::string &src)
{
    static_cast<gentype &>(*this) = src;

    return *this;
}

scalarfn &scalarfn::operator=(const char *src)
{
    static_cast<gentype &>(*this) = src;

    return *this;
}

scalarfn scalarfn::operator()(const double &xx) const
{
    scalarfn res;
    gentype x(xx);

    if ( !varid_xi && !varid_xj )
    {
        res = static_cast<const gentype &>(*this)(x);
    }

    else if ( !varid_xi && ( varid_xj == 1 ) )
    {
        gentype dummy;

        res = static_cast<const gentype &>(*this)(dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 2 ) )
    {
        gentype dummy;

        res = static_cast<const gentype &>(*this)(dummy,dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 3 ) )
    {
        gentype dummy;

        res = static_cast<const gentype &>(*this)(dummy,dummy,dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 4 ) )
    {
        gentype dummy;

        res = static_cast<const gentype &>(*this)(dummy,dummy,dummy,dummy,x);
    }

    else
    {
        SparseVector<SparseVector<gentype> > temp;

        temp("&",varid_xi)("&",varid_xj) = x;

        res = static_cast<const gentype &>(*this)(x);
    }

    return res;
}

scalarfn scalarfn::operator()(const scalarfn &xx) const
{
    scalarfn res;
    const gentype &x = static_cast<const gentype &>(xx);

    if ( !varid_xi && !varid_xj )
    {
        res = static_cast<const gentype &>(*this)(x);
    }

    else if ( !varid_xi && ( varid_xj == 1 ) )
    {
        gentype dummy;

        res = static_cast<const gentype &>(*this)(dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 2 ) )
    {
        gentype dummy;

        res = static_cast<const gentype &>(*this)(dummy,dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 3 ) )
    {
        gentype dummy;

        res = static_cast<const gentype &>(*this)(dummy,dummy,dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 4 ) )
    {
        gentype dummy;

        res = static_cast<const gentype &>(*this)(dummy,dummy,dummy,dummy,x);
    }

    else
    {
        SparseVector<SparseVector<gentype> > temp;

        temp("&",varid_xi)("&",varid_xj) = x;

        res = static_cast<const gentype &>(*this)(x);
    }

    return res;
}

scalarfn &scalarfn::ident(void)
{
    gentype::ident();

    return *this = 1.0;
}

scalarfn &scalarfn::zero(void)
{
    gentype::zero();

    return *this = 0.0;
}

int scalarfn::substitute(const double &xx)
{
    int res = 0;

    gentype x(xx);

    if ( !varid_xi && !varid_xj )
    {
        res = static_cast<gentype &>(*this).substitute(x);
    }

    else if ( !varid_xi && ( varid_xj == 1 ) )
    {
        gentype dummy;

        res = static_cast<gentype &>(*this).substitute(dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 2 ) )
    {
        gentype dummy;

        res = static_cast<gentype &>(*this).substitute(dummy,dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 3 ) )
    {
        gentype dummy;

        res = static_cast<gentype &>(*this).substitute(dummy,dummy,dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 4 ) )
    {
        gentype dummy;

        res = static_cast<gentype &>(*this).substitute(dummy,dummy,dummy,dummy,x);
    }

    else
    {
        SparseVector<SparseVector<gentype> > temp;

        temp("&",varid_xi)("&",varid_xj) = x;

        res = static_cast<gentype &>(*this).substitute(x);
    }

    return res;
}

int scalarfn::substitute(const scalarfn &xx)
{
    int res = 0;

    const gentype &x = static_cast<const gentype &>(xx);

    if ( !varid_xi && !varid_xj )
    {
        res = static_cast<gentype &>(*this).substitute(x);
    }

    else if ( !varid_xi && ( varid_xj == 1 ) )
    {
        gentype dummy;

        res = static_cast<gentype &>(*this).substitute(dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 2 ) )
    {
        gentype dummy;

        res = static_cast<gentype &>(*this).substitute(dummy,dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 3 ) )
    {
        gentype dummy;

        res = static_cast<gentype &>(*this).substitute(dummy,dummy,dummy,x);
    }

    else if ( !varid_xi && ( varid_xj == 4 ) )
    {
        gentype dummy;

        res = static_cast<gentype &>(*this).substitute(dummy,dummy,dummy,dummy,x);
    }

    else
    {
        SparseVector<SparseVector<gentype> > temp;

        temp("&",varid_xi)("&",varid_xj) = x;

        res = static_cast<gentype &>(*this).substitute(x);
    }

    return res;
}

void scalarfn::setvarid(int i, int j)
{
    varid_xi = i;
    varid_xj = j;

    return;
}

void scalarfn::getvarid(int &i, int &j) const
{
    i = varid_xi;
    j = varid_xj;

    return;
}

void scalarfn::sernumpts(int xnumpts)
{
    numpts = xnumpts;

    return;
}

void scalarfn::getnumpts(int &xnumpts) const
{
    xnumpts = numpts;

    return;
}

scalarfn &setident(scalarfn &a)
{
    return a.ident();
}

scalarfn &setzero(scalarfn &a)
{
    return a.zero();
}

/*
inline void innerProduct(scalarfn &res, const scalarfn &a, const scalarfn &b)
inline void innerProductNoConj(scalarfn &res, const scalarfn &a, const scalarfn &b)
inline void innerProductRevConj(scalarfn &res, const scalarfn &a, const scalarfn &b)

double abs(const scalarfn &a)
double abs1(const scalarfn &a)
double absp(const scalarfn &a, const scalarfn &q)
double absp(const scalarfn &a, const double &q)
double absinf(const scalarfn &a)
double norm(const scalarfn &a)
double norm1(const scalarfn &a)
double normp(const scalarfn &a, const scalarfn &q)
double normp(const scalarfn &a, const double &q)
*/

