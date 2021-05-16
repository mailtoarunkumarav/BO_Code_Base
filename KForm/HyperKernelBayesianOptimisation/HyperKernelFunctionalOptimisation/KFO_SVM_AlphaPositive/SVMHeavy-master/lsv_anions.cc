
//
// LS-SVM anionic class
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
#include "lsv_anions.h"

std::ostream &LSV_Anions::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Base training alphaA: " << dalphaA << "\n";
    repPrint(output,'>',dep) << "Base training biasA:  " << dbiasA  << "\n";

    LSV_Generic::printstream(output,dep+1);

    return output;
}

std::istream &LSV_Anions::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> dalphaA;
    input >> dummy; input >> dbiasA;

    LSV_Generic::inputstream(input);

    return input;
}

LSV_Anions::LSV_Anions() : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    dbias.force_anion().setorder(0);

    return;
}

LSV_Anions::LSV_Anions(const LSV_Anions &src) : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    assign(src,0);
    dbias.force_anion().setorder(0);

    return;
}

LSV_Anions::LSV_Anions(const LSV_Anions &src, const ML_Base *srcx) : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(srcx);

    assign(src,0);
    dbias.force_anion().setorder(0);

    return;
}

int LSV_Anions::prealloc(int expectedN)
{
    LSV_Generic::prealloc(expectedN);
    dalphaA.prealloc(expectedN);
    alltraintargA.prealloc(expectedN);

    return 0;
}

int LSV_Anions::getInternalClass(const gentype &y) const
{
    (void) y;

    return 0;
}

int LSV_Anions::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( y.size() == tspaceDim() );

    dalphaA.add(i);
    dalphaA("&",i).setorder(order());
    setzero(dalphaA("&",i));

    alltraintargA.add(i);
    alltraintargA("&",i) = (const d_anion &) y;

    return LSV_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);
}

int LSV_Anions::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( y.size() == tspaceDim() );

    dalphaA.add(i);
    dalphaA("&",i).setorder(order());
    setzero(dalphaA("&",i));

    alltraintargA.add(i);
    alltraintargA("&",i) = (const d_anion &) y;

    return LSV_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
}

int LSV_Anions::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = LSV_Generic::removeTrainingVector(i,y,x);

    dalphaA.remove(i);
    alltraintargA.remove(i);

    return res;
}

int LSV_Anions::sety(int i, const gentype &y)
{
    int res = LSV_Generic::sety(i,y);

    alltraintargA("&",i) = (const d_anion &) y;

    return res;
}

int LSV_Anions::sety(const Vector<int> &i, const Vector<gentype> &y)
{
    int j;
    int res = LSV_Generic::sety(i,y);

    if ( i.size() )
    {
        for ( j = 0 ; j < i.size() ; j++ )
        {
            alltraintargA("&",i(j)) = (const d_anion &) y(j);
        }
    }

    return res;
}

int LSV_Anions::sety(const Vector<gentype> &y)
{
    int j;
    int res = LSV_Generic::sety(y);

    if ( N() )
    {
        for ( j = 0 ; j < N() ; j++ )
        {
            alltraintargA("&",j) = (const d_anion &) y(j);
        }
    }

    return res;
}

int LSV_Anions::sety(int i, const d_anion &y)
{
    gentype yy(y);

    return sety(i,yy);
}

int LSV_Anions::sety(const Vector<int> &i, const Vector<d_anion> &y)
{
    Vector<gentype> yy(i.size());

    int j;

    if ( i.size() )
    {
        for ( j = 0 ; j < i.size() ; j++ )
        {
            yy("&",j) = y(j);
        }
    }

    return sety(i,yy);
}

int LSV_Anions::sety(const Vector<d_anion> &y)
{
    Vector<gentype> yy(N());

    int j;

    if ( N() )
    {
        for ( j = 0 ; j < N() ; j++ )
        {
            yy("&",j) = y(j);
        }
    }

    return sety(yy);
}

int LSV_Anions::setd(int i, int nd)
{
    int res = LSV_Generic::setd(i,nd);

    if ( !nd )
    {
        dalphaA("&",i) *= 0.0;
    }

    return res;
}

int LSV_Anions::setd(const Vector<int> &i, const Vector<int> &nd)
{
    int res = LSV_Generic::setd(i,nd);

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            if ( !nd(j) )
            {
                dalphaA("&",i(j)) *= 0.0;
            }
        }
    }

    return res;
}

int LSV_Anions::setd(const Vector<int> &nd)
{
    int res = LSV_Generic::setd(nd);

    if ( N() )
    {
        int j;

        for ( j = 0 ; j < N() ; j++ )
        {
            if ( !nd(j) )
            {
                dalphaA("&",j) *= 0.0;
            }
        }
    }

    return res;
}

int LSV_Anions::scale(double a)
{
    LSV_Generic::scale(a);
    dalphaA.scale(a);
    dbiasA *= a;

    return 1;
}

int LSV_Anions::reset(void)
{
    LSV_Generic::reset();

    d_anion zerotemp(0.0);

    dalphaA = zerotemp;
    dbiasA = zerotemp;

    return 1;
}

int LSV_Anions::setgamma(const Vector<gentype> &newW)
{
    int res = LSV_Generic::setgamma(newW);

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            dalphaA("&",i) = (const d_anion &) newW(i);
        }
    }

    return res;
}

int LSV_Anions::setdelta(const gentype &newB)
{
    int res = LSV_Generic::setdelta(newB);

    dbiasA = (const d_anion &) newB;

    return res;
}

int LSV_Anions::train(int &res, svmvolatile int &killSwitch)
{
    if ( tspaceDim() )
    {
        incgvernum();

        int i,j;

        Vector<double> dalphaR(N());
        Vector<double> alltraintargR(N());

        Vector<double> dbetaR(1);
        Vector<double> dybeta(1);

        if ( N() )
        {
            for ( i = 0 ; i < N() ; i++ )
            {
                dalphaA("&",i).setorder(order()) *= 0.0;
            }
        }

        dbiasA.setorder(order()) *= 0.0;

        LSV_Generic::train(res,killSwitch);

        for ( j = 0 ; j < tspaceDim() ; j++ )
        {
            if ( N() )
            {
                for ( i = 0 ; i < N() ; i++ )
                {
                    alltraintargR("&",i) = alltraintargA(i)(j);
                }
            }

            dybeta = 0.0;
            dbetaR = 0.0;

            fact_minverse(dalphaR,dbetaR,alltraintargR,dybeta);

            if ( N() )
            {
                for ( i = 0 ; i < N() ; i++ )
                {
                    dalphaA("&",i)("&",j) = dalphaR(i);
                }
            }

            dbiasA("&",j) = dbetaR(zeroint());

            if ( N() )
            {
                for ( i = 0 ; i < N() ; i++ )
                {
                    dalpha("&",i).dir_anion()("&",j) = dalphaA(i)(j);
                }
            }

            dbias.dir_anion()("&",j) = dbiasA(j);
        }
    }

    return 0;
}

int LSV_Anions::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int dtv = 0;

    (void) retaltg;

    d_anion &res = resg.force_anion().setorder(order());
    setzero(res);

    int isloc = ( ( i >= 0 ) && d()(i) ) ? 1 : 0;

    if ( isloc ) 
    {
        res = (const d_anion &) dalpha(i);
        res *= -((diagoffset())(i));
        res += (const d_anion &) alltraintarg(i);
    }

    else if ( ( dtv = xtang(i) & 7 ) )
    {
        NiceAssert( !( dtv & 4 ) );

        res = 0.0;

        if ( ( dtv > 0 ) && N() )
        {
            int j;
            double Kxj;
            d_anion temp;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( d()(j) )
                {
                    if ( i >= 0 )
                    {
                        // It is *vital* that we go through the kernel cache here!  Otherwise
                        // in for example grid-search we will just end up calculating the same
                        // kernel evaluations over and over again!

                        Kxj = Gp()(i,j);

                        if ( i == j )
                        {
                            Kxj -= diagoffset()(i);
                        }
                    }

                    else
                    {
                        K2(Kxj,i,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                    }

                    temp = (const d_anion &) dalpha(j);
                    temp *= Kxj;
                    res += temp;
                }
            }
        }
    }

    else
    {
        res = (const d_anion &) dbias;

        if ( N() )
        {
            int j;
            double Kxj;
            d_anion temp;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( d()(j) )
                {
                    if ( i >= 0 )
                    {
                        // It is *vital* that we go through the kernel cache here!  Otherwise
                        // in for example grid-search we will just end up calculating the same
                        // kernel evaluations over and over again!

                        Kxj = Gp()(i,j);

                        if ( i == j )
                        {
                            Kxj -= diagoffset()(i);
                        }
                    }

                    else
                    {
                        K2(Kxj,i,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                    }

                    temp = (const d_anion &) dalpha(j);
                    temp *= Kxj;
                    res += temp;
                }
            }
        }
    }

    resh = resg;

    return 0;
}



int LSV_Anions::setorder(int neword)
{
    int res = 0;

    if ( neword != order() )
    {
        res = 1;

        dbias.dir_anion().setorder(neword);

        if ( N() )
        {
            int j;

            for ( j = 0 ; j < N() ; j++ )
            {
                dalpha("&",j).dir_anion().setorder(neword);
                alltraintarg("&",j).dir_anion().setorder(neword);
            }
        }
    }

    return res;
}
