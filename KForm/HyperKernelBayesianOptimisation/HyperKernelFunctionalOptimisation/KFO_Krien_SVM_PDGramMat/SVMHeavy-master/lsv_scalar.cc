//TO DO: for some reason d = +1 (indicated by > 0 in training set) is not getting passed through to here
//to trigger SVM_Scalar training fallback.  Find out why and fix it!

//
// LS-SVM scalar class
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
#include "lsv_scalar.h"

std::ostream &LSV_Scalar::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Base training alphaR: " << dalphaR << "\n";
    repPrint(output,'>',dep) << "Base training biasR:  " << dbiasR  << "\n";

    LSV_Generic::printstream(output,dep+1);

    return output;
}

std::istream &LSV_Scalar::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> dalphaR;
    input >> dummy; input >> dbiasR;

    LSV_Generic::inputstream(input);

    return input;
}

LSV_Scalar::LSV_Scalar() : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    dbias.force_double() = 0.0;

    return;
}

LSV_Scalar::LSV_Scalar(const LSV_Scalar &src) : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    assign(src,0);
    dbias.force_double() = 0.0;

    return;
}

LSV_Scalar::LSV_Scalar(const LSV_Scalar &src, const ML_Base *srcx) : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(srcx);

    assign(src,0);
    dbias.force_double() = 0.0;

    return;
}

int LSV_Scalar::prealloc(int expectedN)
{
    LSV_Generic::prealloc(expectedN);
    dalphaR.prealloc(expectedN);
    alltraintargR.prealloc(expectedN);

    return 0;
}

int LSV_Scalar::getInternalClass(const gentype &y) const
{
    return ( ( (double) y ) < 0 ) ? 0 : 1;
}

int LSV_Scalar::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    dalphaR.add(i);
    dalphaR("&",i) = 0.0;

    alltraintargR.add(i);
    alltraintargR("&",i) = (double) y;

    return LSV_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);
}

int LSV_Scalar::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    dalphaR.add(i);
    dalphaR("&",i) = 0.0;

    alltraintargR.add(i);
    alltraintargR("&",i) = (double) y;

    return LSV_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
}

int LSV_Scalar::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = LSV_Generic::removeTrainingVector(i,y,x);

    dalphaR.remove(i);
    alltraintargR.remove(i);

    return res;
}

int LSV_Scalar::removeTrainingVector(int i, int num)
{
    int res = 0;
    gentype y;
    SparseVector<gentype> x;

    while ( num )
    {
        num--; 
        res |= removeTrainingVector(i+num,y,x);
    }

    return res;
}

int LSV_Scalar::sety(int i, const gentype &y)
{
    int res = LSV_Generic::sety(i,y);

    alltraintargR("&",i) = (double) y;

    return res;
}

int LSV_Scalar::sety(const Vector<int> &i, const Vector<gentype> &y)
{
    int j;
    int res = LSV_Generic::sety(i,y);

    if ( i.size() )
    {
        for ( j = 0 ; j < i.size() ; j++ )
        {
            alltraintargR("&",i(j)) = (double) y(j);
        }
    }

    return res;
}

int LSV_Scalar::sety(const Vector<gentype> &y)
{
    int j;
    int res = LSV_Generic::sety(y);

    if ( N() )
    {
        for ( j = 0 ; j < N() ; j++ )
        {
            alltraintargR("&",j) = (double) y(j);
        }
    }

    return res;
}

int LSV_Scalar::sety(int i, const double &y)
{
    gentype yy(y);

    return sety(i,yy);
}

int LSV_Scalar::sety(const Vector<int> &i, const Vector<double> &y)
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

int LSV_Scalar::sety(const Vector<double> &y)
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

int LSV_Scalar::setd(int i, int nd)
{
    int res = LSV_Generic::setd(i,nd);

    if ( !nd )
    {
        dalphaR("&",i) = 0.0;
    }

    return res;
}

int LSV_Scalar::setd(const Vector<int> &i, const Vector<int> &nd)
{
    int res = LSV_Generic::setd(i,nd);

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            if ( !nd(j) )
            {
                dalphaR("&",i(j)) = 0.0;
            }
        }
    }

    return res;
}

int LSV_Scalar::setd(const Vector<int> &nd)
{
    int res = LSV_Generic::setd(nd);

    if ( N() )
    {
        int j;

        for ( j = 0 ; j < N() ; j++ )
        {
            if ( !nd(j) )
            {
                dalphaR("&",j) = 0.0;
            }
        }
    }

    return res;
}

int LSV_Scalar::scale(double a)
{
    LSV_Generic::scale(a);
    dalphaR.scale(a);
    dbiasR *= a;

    return 1;
}

int LSV_Scalar::reset(void)
{
    LSV_Generic::reset();
    dalphaR = 0.0;
    dbiasR = 0.0;

    return 1;
}

int LSV_Scalar::setgamma(const Vector<gentype> &newW)
{
    int res = LSV_Generic::setgamma(newW);

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            dalphaR("&",i) = (double) newW(i);
        }
    }

    return res;
}

int LSV_Scalar::setdelta(const gentype &newB)
{
    int res = LSV_Generic::setdelta(newB);

    dbiasR = (double) newB;

    return res;
}

int LSV_Scalar::train(int &res, svmvolatile int &killSwitch)
{
    incgvernum();

//errstream() << "phantomx lsvtrain 0: " << NNC(-1) << "," << NNC(+1) << "\n";
    if ( !NNC(-1) && !NNC(+1) )
    {
//errstream() << "phantomx lsvtrain 1\n";
        Vector<double> dybeta(1);
        Vector<double> dbetaR(1);
//errstream() << "phantomx lsvtrain 2\n";

        LSV_Generic::train(res,killSwitch);

        dybeta = 0.0;
        dbetaR = 0.0;

        dalphaR = 0.0;
        dbiasR  = 0.0;

//errstream() << "phantomx lsvtrain 3: " << alltraintargR << "\n";
//errstream() << "phantomx lsvtrain 4: " << dybeta << "\n";
//errstream() << "phantomx lsvtrain 4b: " << Gp() << "\n";
        fact_minverse(dalphaR,dbetaR,alltraintargR,dybeta);
//errstream() << "phantomx lsvtrain 5: " << dalphaR << "\n";
//errstream() << "phantomx lsvtrain 6: " << dbetaR << "\n";

        dbiasR = dbetaR(zeroint());
    }

    else
    {
        SVM_Scalar::train(res,killSwitch);

        dalphaR = SVM_Scalar::alphaR();
        dbiasR  = SVM_Scalar::biasR();
    }

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            dalpha("&",i).dir_double() = dalphaR(i);
        }
    }

    dbias.dir_double() = dbiasR;

    return 0;
}

int LSV_Scalar::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const 
{ 
//errstream() << "phantomx lsv_scalar gh 0\n";
    int dtv = 0;

    (void) retaltg;

    int isloc = ( ( i >= 0 ) && d()(i) ) ? 1 : 0;

    if ( isloc ) 
    {
        double &res = resg.force_double();
        setzero(res);

        res = ( -((double) dalpha(i)) * ((diagoffset())(i)) ) + ((double) alltraintarg(i));
    }

    else if ( ( dtv = xtang(i) & 7 ) )
    {
        double &res = resg.force_double();
        setzero(res);

        // NB: the derivative might not be double!

        int firstbit = 1;

        if ( ( dtv > 0 ) && ( !( dtv & 4 ) ) && N() )
        {
            int j;
            gentype Kxj;

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

                    if ( firstbit )
                    {
                        firstbit = 0;
                        resg = Kxj*dalphaR(j);
                    }

                    else
                    {
                        resg += Kxj*dalphaR(j);
                    }
                }
//errstream() << "phantomx lsv_scalar gh resg = " << resg << "\n";
            }
        }

        else if ( ( dtv > 0 ) && ( dtv & 4 ) && N() )
        {
            int j;
            int isfirst = 1;
            gentype Kxj;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( d()(j) )
                {
                    if ( i >= 0 )
                    {
                        // This probably won't happen, but undirected gradient constraints are 
                        // ok in the training set if xspaceDim() == 1, which is the case here.

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

                    if ( isfirst )
                    {
                        resg = Kxj*dalpha(j);

                        isfirst = 0;
                    }

                    else
                    {
                        resg += Kxj*dalpha(j);
                    }
                }
//errstream() << "phantomx lsv_scalar gh resg = " << resg << "\n";
            }
        }
    }

    else
    {
        double &res = resg.force_double();
        setzero(res);

        res = (double) dbias;

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
//errstream() << "phantomx lsv_scalar gh Kxj = " << Kxj << "\n";
                    }

//errstream() << "phantomx lsv_scalar gh alpha = " << (double) dalpha(j) << "\n";
                    res += Kxj*dalphaR(j);
//errstream() << "phantomx lsv_scalar gh res = " << res << "\n";
                }
            }
        }
    }

    resh = resg;

    gentype tentype(sgn(resh));

//errstream() << "phantomx lsv_scalar gh 1\n";
    return tentype.isCastableToIntegerWithoutLoss() ? (int) tentype : zeroint();
}


int LSV_Scalar::covTrainingVector(gentype &resv, gentype &resmu, int ia, int ib, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
//errstream() << "phantomxmeep 0: " << x(ia) << "," << x(ib) << "\n";
    int tres = 0;

    int dtva = xtang(ia) & 7;
    int dtvb = xtang(ib) & 7;

    NiceAssert( dtva >= 0 );
    NiceAssert( dtvb >= 0 );

    // This is used elsewhere (ie not scalar), so the following is relevant

//FIXME: resmu
    if ( ( dtva & 4 ) || ( dtvb & 4 ) || !isUnderlyingScalar() )
    {
        if ( NNC(-1) || NNC(+1) || NNC(2) )
        {
            int j;

            Vector<gentype> Kia(N());
            Vector<gentype> Kib(N());
            Vector<gentype> itsone(1);//isVarBias() ? 1 : 0);
            gentype Kii;

            itsone("&",zeroint()) = 1.0;

            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                Kii  = Gp()(ia,ib);
                Kii -= ( ia == ib ) ? diagoffset()(ia) : 0.0;
            }

            else
            { 
                K2(Kii,ia,ib,(const gentype **) pxyprodij);
            }

            if ( ia >= 0 )
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        Kia("&",j) = Gp()(ia,j);
                    }

                    else
                    {
                        Kia("&",j) = 0.0;
                    }
                }

                Kia("&",ia) -= diagoffset()(ia);
            }

            else
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        K2(Kia("&",j),ia,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                    }

                    else
                    {
                        Kia("&",j) = 0.0;
                    }
                }
            }

            if ( ia == ib )
            {
                Kib = Kia;
            }

            else if ( ib >= 0 )
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        Kib("&",j) = Gp()(j,ib);
                    }

                    else
                    {
                        Kib("&",j) = 0.0;
                    }
                }

                Kib("&",ib) -= diagoffset()(ib);

            }

            else
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        //K2(Kib("&",j),j,ib,pxyprodj ? (const gentype **) pxyprodj[j] : NULL);
                        K2(Kib("&",j),ib,j,pxyprodj ? (const gentype **) pxyprodj[j] : NULL); // do this to suit the assumptions in mercer Kxfer (that the unknown "x" is first)
                        setconj(Kib("&",j));
                    }

                    else
                    {
                        Kib("&",j) = 0.0;
                    }
                }
            }

            Vector<gentype> btemp(1);//isVarBias() ? 1 : 0);
            Vector<gentype> Kres(N());

            //NB: this will automatically only do part corresponding to pivAlphaF
            fact_minverse(Kres,btemp,Kib,itsone);

            resv = Kii;

            for ( j = 0 ; j < N() ; j++ )
            {
                // This is important or variance will be fracking negative!!!!

                if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                {
                    resv -= outerProd(Kia(j),Kres(j));
                }
            }

            if ( isVarBias() )
            {
                // This is the additional corrective factor

                resv -= btemp(zeroint());
            }

            // mu calculation

            int firstterm = 1;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( firstterm )
                {
                    if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                    {
                        resmu = Kia(j)*dalpha(j);

                        firstterm = 0;
                    }
                }

                else
                {
                    if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                    {
                        resmu += Kia(j)*dalpha(j);
                    }
                }
            }

            if ( !( dtva & 7 ) )
            {
                if ( firstterm )
                {
                    resmu = dbias;
                    firstterm = 0;
                }

                else
                {
                    resmu += dbias;
                }
            }

            else
            {
                if ( firstterm )
                {
                    resmu =  dbiasR;
                    resmu *= 0.0;
                    firstterm = 0;
                }
            }
        }

        else
        {
            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                resv  = Gp()(ia,ib);
                resv -= ( ia == ib ) ? diagoffset()(ia) : 0.0;
            }

            else
            {
                K2(resv,ia,ib,(const gentype **) pxyprodij);
            }

            if ( !( dtva & 7 ) )
            {
                resmu = dbias;
            }

            else
            {
                resmu  = dbias;
                resmu *= 0.0;
            }
        }

        if ( getKernel().isKVarianceNZ() )
        {
            NiceAssert( ia == ib );

            gentype addres(0.0);
            gentype Kxj;
            int j;

            for ( j = 0 ; j < N() ; j++ )
            {
                K2(Kxj,ia,j,NULL,NULL,NULL,NULL,NULL,0x80);
                addres += ((double) dalpha(j))*((double) dalpha(j))*Kxj;
            }

            resv += addres;
        }

        if ( ( ia == ib ) && ( resv <= zerogentype() ) )
        {
            // Sometimes numerical issues can make the variance (ia == ib, so 
            // this is variance) negative.  This causes issues when eg bayesian
            // optimisation attempts to take the square-root and cast the
            // result to real.  Hence this "fix-fudge" to make it zero instead.

            resv = zerogentype();
        }
    }

    else
    {
//errstream() << "phantomxyggg 0\n";
        double &resvv = resv.force_double();
        double &resgg = resmu.force_double();

        if ( dtva & 7 )
        {
            resgg = 0.0;
        }

        else
        {
            resgg = dbiasR;
        }

//errstream() << "phantomxyggg 1\n";
        if ( NNC(-1) || NNC(+1) || NNC(2) )
        {
//errstream() << "phantomxyggg 2\n";
            int j;

            Vector<double> Kia(N());
            Vector<double> Kib(N());
            Vector<double> itsone(1);//isVarBias() ? 1 : 0);
            double Kii;

            itsone("&",zeroint()) = 1.0;

            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                Kii  = Gp()(ia,ib);
                Kii -= ( ia == ib ) ? diagoffset()(ia) : 0.0;
            }

            else
            { 
                K2(Kii,ia,ib,(const gentype **) pxyprodij);
            }

            if ( ia >= 0 )
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        Kia("&",j) = Gp()(ia,j);
                    }

                    else
                    {
                        Kia("&",j) = 0.0;
                    }
                }

                Kia("&",ia) -= diagoffset()(ia);
            }

            else
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        K2(Kia("&",j),ia,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                    }

                    else
                    {
                        Kia("&",j) = 0.0;
                    }
                }
            }

            if ( ib == ia )
            {
                Kib = Kia;
            }

            else if ( ib >= 0 )
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        Kib("&",j) = Gp()(j,ib);
                    }

                    else
                    {
                        Kib("&",j) = 0.0;
                    }
                }

                Kib("&",ib) -= diagoffset()(ib);
            }

            else
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        //K2(Kib("&",j),j,ib,pxyprodj ? (const gentype **) pxyprodj[j] : NULL); - see above
                        K2(Kib("&",j),ib,j,pxyprodj ? (const gentype **) pxyprodj[j] : NULL);
                    }

                    else
                    {
                        Kib("&",j) = 0.0;
                    }
                }
            }

            Vector<double> btemp(1);//isVarBias() ? 1 : 0);
            Vector<double> Kres(N());

            //NB: this will automatically only do part corresponding to pivAlphaF
            fact_minverse(Kres,btemp,Kib,itsone);

            resvv = Kii;

            for ( j = 0 ; j < N() ; j++ )
            {
                // This is important or variance will be fracking negative!!!!

                if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                {
                    resvv -= Kia(j)*Kres(j);
                }
            }

            if ( isVarBias() )
            {
                // This is the additional corrective factor

                resvv -= btemp(zeroint());
            }

            // mu calculation

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                {
                    resgg += Kia(j)*dalphaR(j);
                }
            }
        }

        else
        {
//errstream() << "phantomxyggg 100\n";
            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                resvv  = Gp()(ia,ib);
                resvv -= ( ia == ib ) ? diagoffset()(ia) : 0.0;
            }

            else
            {
//errstream() << "phantomxyggg 101\n";
                K2(resvv,ia,ib,(const gentype **) pxyprodij);
            }

            if ( !( dtva & 7 ) )
            {
                resgg = dbias;
            }

            else
            {
                resgg  = dbias;
                resgg *= 0.0;
            }
//errstream() << "phantomxyggg 103\n";
        }

        if ( getKernel().isKVarianceNZ() )
        {
            NiceAssert( ia == ib );

            double addres = 0.0;
            double Kxj;
            int j;

            for ( j = 0 ; j < N() ; j++ )
            {
                K2(Kxj,ia,j,NULL,NULL,NULL,NULL,NULL,0x80);
                addres += ((double) dalpha(j))*((double) dalpha(j))*Kxj;
            }

            resvv += addres;
        }

        if ( ( ia == ib ) && ( resvv <= 0.0 ) )
        {
            // Sometimes numerical issues can make the variance (ia == ib, so 
            // this is variance) negative.  This causes issues when eg bayesian
            // optimisation attempts to take the square-root and cast the
            // result to real.  Hence this "fix-fudge" to make it zero instead.

            resvv = 0.0;
        }
    }

    return tres;
}










