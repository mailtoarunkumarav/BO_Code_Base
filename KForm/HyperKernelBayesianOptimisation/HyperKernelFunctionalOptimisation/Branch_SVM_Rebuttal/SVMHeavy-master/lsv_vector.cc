
//
// LS-SVM vector class
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
#include "lsv_vector.h"

std::ostream &LSV_Vector::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Base training alphaV: " << dalphaV << "\n";
    repPrint(output,'>',dep) << "Base training biasV:  " << dbiasV  << "\n";

    LSV_Generic::printstream(output,dep+1);

    return output;
}

std::istream &LSV_Vector::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> dalphaV;
    input >> dummy; input >> dbiasV;

    LSV_Generic::inputstream(input);

    return input;
}

LSV_Vector::LSV_Vector() : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    dbias.force_vector(0);

    return;
}

LSV_Vector::LSV_Vector(const LSV_Vector &src) : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    dbias.force_vector(0); 
    assign(src,0);

    return;
}

LSV_Vector::LSV_Vector(const LSV_Vector &src, const ML_Base *srcx) : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(srcx);

    dbias.force_vector(0);
    assign(src,0);

    return;
}

int LSV_Vector::prealloc(int expectedN)
{
    LSV_Generic::prealloc(expectedN);
    dalphaV.prealloc(expectedN);
    alltraintargV.prealloc(expectedN);

    return 0;
}

int LSV_Vector::getInternalClass(const gentype &y) const
{
    (void) y;

    return 0;
}

int LSV_Vector::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( ( y.size() == tspaceDim() ) || !tspaceDim() );

    if ( !tspaceDim() && y.size() )
    {
        settspaceDim(y.size());
    }

    dalphaV.add(i);
    dalphaV("&",i).resize(tspaceDim()) = 0.0;

    alltraintargV.add(i);
    alltraintargV("&",i) = (const Vector<double> &) y;

    return LSV_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);
}

int LSV_Vector::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( ( y.size() == tspaceDim() ) || !tspaceDim() );

    if ( !tspaceDim() && y.size() )
    {
        settspaceDim(y.size());
    }

    dalphaV.add(i);
    dalphaV("&",i).resize(tspaceDim()) = 0.0;

    alltraintargV.add(i);
    alltraintargV("&",i) = (const Vector<double> &) y;

    return LSV_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
}

int LSV_Vector::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = LSV_Generic::removeTrainingVector(i,y,x);

    dalphaV.remove(i);
    alltraintargV.remove(i);

    return res;
}

int LSV_Vector::sety(int i, const gentype &y)
{
    int res = LSV_Generic::sety(i,y);

    alltraintargV("&",i) = (const Vector<double> &) y;

    return res;
}

int LSV_Vector::sety(const Vector<int> &i, const Vector<gentype> &y)
{
    int j;
    int res = LSV_Generic::sety(i,y);

    if ( i.size() )
    {
        for ( j = 0 ; j < i.size() ; j++ )
        {
            alltraintargV("&",i(j)) = (const Vector<double> &) y(j);
        }
    }

    return res;
}

int LSV_Vector::sety(const Vector<gentype> &y)
{
    int j;
    int res = LSV_Generic::sety(y);

    if ( N() )
    {
        for ( j = 0 ; j < N() ; j++ )
        {
            alltraintargV("&",j) = (const Vector<double> &) y(j);
        }
    }

    return res;
}

int LSV_Vector::sety(int i, const Vector<double> &y)
{
    gentype yy(y);

    return sety(i,yy);
}

int LSV_Vector::sety(const Vector<int> &i, const Vector<Vector<double> > &y)
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

int LSV_Vector::sety(const Vector<Vector<double> > &y)
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

int LSV_Vector::setd(int i, int nd)
{
    int res = LSV_Generic::setd(i,nd);

    if ( !nd )
    {
        dalphaV("&",i) = 0.0;
    }

    return res;
}

int LSV_Vector::setd(const Vector<int> &i, const Vector<int> &nd)
{
    int res = LSV_Generic::setd(i,nd);

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            if ( !nd(j) )
            {
                dalphaV("&",i(j)) = 0.0;
            }
        }
    }

    return res;
}

int LSV_Vector::setd(const Vector<int> &nd)
{
    int res = LSV_Generic::setd(nd);

    if ( N() )
    {
        int j;

        for ( j = 0 ; j < N() ; j++ )
        {
            if ( !nd(j) )
            {
                dalphaV("&",j) = 0.0;
            }
        }
    }

    return res;
}

int LSV_Vector::scale(double a)
{
    LSV_Generic::scale(a);
    dalphaV.scale(a);
    dbiasV *= a;

    return 1;
}

int LSV_Vector::reset(void)
{
    LSV_Generic::reset();

    Vector<double> zerotemp(tspaceDim());

    zerotemp = 0.0;

    dalphaV = zerotemp;
    dbiasV = zerotemp;

    return 1;
}

int LSV_Vector::setgamma(const Vector<gentype> &newW)
{
    int res = LSV_Generic::setgamma(newW);

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            dalphaV("&",i) = (const Vector<double> &) newW(i);
        }
    }

    return res;
}

int LSV_Vector::setdelta(const gentype &newB)
{
    int res = LSV_Generic::setdelta(newB);

    dbiasV = (const Vector<double> &) newB;

    return res;
}

int LSV_Vector::train(int &res, svmvolatile int &killSwitch)
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
                dalphaV("&",i).resize(tspaceDim()) = 0.0;
            }
        }

        dbiasV.resize(tspaceDim()) = 0.0;

        LSV_Generic::train(res,killSwitch);

        for ( j = 0 ; j < tspaceDim() ; j++ )
        {
            if ( N() )
            {
                for ( i = 0 ; i < N() ; i++ )
                {
                    alltraintargR("&",i) = alltraintargV(i)(j);
                }
            }

            dybeta = 0.0;
            dbetaR = 0.0;

            fact_minverse(dalphaR,dbetaR,alltraintargR,dybeta);

            if ( N() )
            {
                for ( i = 0 ; i < N() ; i++ )
                {
                    dalphaV("&",i)("&",j) = dalphaR(i);
                }
            }

            dbiasV("&",j) = dbetaR(zeroint());

            if ( N() )
            {
                for ( i = 0 ; i < N() ; i++ )
                {
                    ((dalpha("&",i).dir_vector())("&",j)).dir_double() = dalphaV(i)(j);
                }
            }

            ((dbias.dir_vector())("&",j)).dir_double() = dbiasV(j);
        }
    }

    return 0;
}

int LSV_Vector::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int dtv = 0;

    (void) retaltg;

    Vector<gentype> &res = resg.force_vector(tspaceDim()).zero();

    int isloc = ( ( i >= 0 ) && d()(i) ) ? 1 : 0;

    if ( isloc ) 
    {
        res.scaleAdd(-((diagoffset())(i)),(const Vector<gentype> &) dalpha(i));
        res += (const Vector<gentype> &) alltraintarg(i);
    }

    else if ( ( dtv = xtang(i) & 7 ) )
    {
        NiceAssert( !( dtv & 4 ) );

        res.resize(tspaceDim()) = zerogentype();

        if ( ( dtv > 0 ) && N() )
        {
            int j;
            double Kxj;

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

                    res.scaleAdd(Kxj,(const Vector<gentype> &) dalpha(j));
                }
            }
        }
    }

    else
    {
        res.resize(tspaceDim()) = (const Vector<gentype> &) dbias;

        if ( N() )
        {
            int j;
            double Kxj;

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

                    res.scaleAdd(Kxj,(const Vector<gentype> &) dalpha(j));
                }
            }
        }
    }

    resh = resg;

    return 0;
}

int LSV_Vector::settspaceDim(int newdim)
{
    int res = 0;

    if ( newdim != tspaceDim() )
    {
        res = 1;

        int olddim = ( tspaceDim() < newdim ) ?  tspaceDim() : newdim;

        retVector<gentype> tmpva;

        dbias.dir_vector().resize(newdim);
        dbias.dir_vector()("&",olddim,1,newdim-1,tmpva) = zerogentype();

        if ( N() )
        {
            int j;

            for ( j = 0 ; j < N() ; j++ )
            {
                dalpha("&",j).dir_vector().resize(newdim);
                alltraintarg("&",j).dir_vector().resize(newdim);
                dalpha("&",j).dir_vector()("&",olddim,1,newdim-1,tmpva) = zerogentype();
                alltraintarg("&",j).dir_vector()("&",olddim,1,newdim-1,tmpva) = zerogentype();
            }
        }
    }

    return res;
}

int LSV_Vector::addtspaceFeat(int i)
{
    int res = 0;

    {
        res = 1;

        dbias.dir_vector().add(i);
        dbias.dir_vector()("&",i) = zerogentype();

        if ( N() )
        {
            int j;

            for ( j = 0 ; j < N() ; j++ )
            {
                dalpha("&",j).dir_vector().add(i);
                alltraintarg("&",j).dir_vector().add(i);

                dalpha("&",j).dir_vector()("&",i) = zerogentype();
                alltraintarg("&",j).dir_vector()("&",i) = zerogentype();
            }
        }
    }

    return res;
}

int LSV_Vector::removetspaceFeat(int i)
{
    int res = 0;

    {
        res = 1;

        dbias.dir_vector().remove(i);

        if ( N() )
        {
            int j;

            for ( j = 0 ; j < N() ; j++ )
            {
                dalpha("&",j).dir_vector().remove(i);
                alltraintarg("&",j).dir_vector().remove(i);
            }
        }
    }

    return res;
}

