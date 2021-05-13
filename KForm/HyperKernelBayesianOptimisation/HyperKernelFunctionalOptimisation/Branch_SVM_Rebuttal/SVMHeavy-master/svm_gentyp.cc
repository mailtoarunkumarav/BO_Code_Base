
//
// Generic target SVM
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
#include "svm_gentyp.h"


std::ostream &SVM_Gentyp::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Gentyp SVM\n\n";

    repPrint(output,'>',dep) << "y values:           " << locyval     << "\n";
    repPrint(output,'>',dep) << "local basis:        " << isBasisUser << "\n";
    repPrint(output,'>',dep) << "y basis:            " << locbasis    << "\n";
    repPrint(output,'>',dep) << "default projection: " << defbasis    << "\n";
    repPrint(output,'>',dep) << "M:                  " << M           << "\n";
    repPrint(output,'>',dep) << "L:                  " << L           << "\n";

    SVM_Vector::printstream(output,dep+1);

    return output;
}

std::istream &SVM_Gentyp::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> locyval;
    input >> dummy; input >> isBasisUser;
    input >> dummy; input >> locbasis;
    input >> dummy; input >> defbasis;
    input >> dummy; input >> M;
    input >> dummy; input >> L;

    dummyGpn.resize(NbasisUU(),0);
    dummyGn.resize(0,0);

    SVM_Vector::inputstream(input);

    return input;
}


SVM_Gentyp::SVM_Gentyp() : SVM_Vector()
{
    isBasisUser = 0;
    defbasis = -1;

    SVM_Vector::setQuadraticCost();

    retVector<double> tmpva;

    L.fudgeOn(onedoublevec(NbasisUU(),tmpva));

    return;
}

SVM_Gentyp::SVM_Gentyp(const SVM_Gentyp &src) : SVM_Vector(static_cast<const SVM_Vector &>(src))
{
    setaltx(NULL);

    assign(src,0);

    return;
}

SVM_Gentyp::SVM_Gentyp(const SVM_Gentyp &src, const ML_Base *xsrc) : SVM_Vector(static_cast<const SVM_Vector &>(src),xsrc)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

SVM_Gentyp::~SVM_Gentyp()
{
    return;
}

int SVM_Gentyp::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    Vector<double> yrep;

    if ( isBasisUser )
    {
        isBasisUser = 0;
        addToBasisUU(i,y);
        isBasisUser = 1;
    }

    ConvertYtoVec(y,yrep);

    locyval.add(locyval.size());
    locyval("&",locyval.size()-1) = y;

    return SVM_Vector::addTrainingVector(i,yrep,x,Cweigh,epsweigh);
}

int SVM_Gentyp::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    Vector<double> yrep;

    if ( isBasisUser )
    {
        isBasisUser = 0;
        addToBasisUU(i,y);
        isBasisUser = 1;
    }

    ConvertYtoVec(y,yrep);

    locyval.add(locyval.size());
    locyval("&",locyval.size()-1) = y;

    return SVM_Vector::qaddTrainingVector(i,yrep,x,Cweigh,epsweigh);
}

int SVM_Gentyp::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == xx.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Gentyp::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_Gentyp::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == xx.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Gentyp::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_Gentyp::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    if ( isBasisUser )
    {
        isBasisUser = 0;
        removeFromBasisUU(i);
        isBasisUser = 1;
    }

    y = locyval(i);

    gentype dummy;

    return SVM_Vector::removeTrainingVector(i,dummy,x);
}

int SVM_Gentyp::removeTrainingVector(int i, int num)
{
    int res = 0;
    gentype ytemp;
    SparseVector<gentype> xtemp;

    while ( num )
    {
        res |= SVM_Gentyp::removeTrainingVector(i-(--num),ytemp,xtemp);
    }

    return res;
}

int SVM_Gentyp::sety(int i, const gentype &z)
{
    if ( isBasisUser )
    {
        isBasisUser = 0;
        setBasisUU(i,z); // won't recurse as isBasisUser is now 0
        isBasisUser = 1;
    }

    Vector<double> yrep;

    ConvertYtoVec(z,yrep);

    return SVM_Vector::sety(i,yrep);
}

int SVM_Gentyp::sety(const Vector<int> &i, const Vector<gentype> &z)
{
    NiceAssert( i.size() == z.size() );

    int res = 0;

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            res |= SVM_Gentyp::sety(i(ii),z(ii));
        }
    }

    return res;
}

int SVM_Gentyp::sety(const Vector<gentype> &z)
{
    retVector<int> tmpva;

    return SVM_Gentyp::sety(cntintvec(N(),tmpva),z);
}

int SVM_Gentyp::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    Vector<double> resgvec;
    int res = 0;

    res = SVM_Vector::ggTrainingVector(resgvec,i,retaltg,pxyprodi);

    ConvertYtoOut(resgvec,resg);

    if ( defProjUU() >= 0 )
    {
        resh = VbasisUU()(defProjUU())*resg;
    }

    else
    {
        resh = resg;
    }

    return res;
}

//int SVM_Gentyp::gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg, const vecInfo *xinf, gentype ***pxyprodi) const
//{
//    Vector<double> resgvec;
//    int res = 0;
//
//    res = SVM_Vector::gg(resgvec,x,retaltg,xinf,pxyprodi);
//
//    ConvertYtoOut(resgvec,resg);
//
//    if ( defProjUU() >= 0 )
//    {
//        resh = VbasisUU()(defProjUU())*resg;
//    }
//
//    else
//    {
//        resh = resg;
//    }
//
//    return res;
//}





















// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------

int SVM_Gentyp::setUUOutputKernel(const MercerKernel &xkernel, int modind)
{
    SVM_Generic::setUUOutputKernel(xkernel,modind);

    int locisBasisUser = isBasisUser;

    isBasisUser = 0;
    int res = setBasisUU(locbasis);
    isBasisUser = locisBasisUser;

    return res;
}

int SVM_Gentyp::addToBasisUU(int i, const gentype &o)
{
    NiceAssert( !isBasisUser );
    NiceAssert( i >= 0 );
    NiceAssert( i <= NbasisUU() );

    // Add to basis set

    locbasis.add(i);
    locbasis("&",i) = o;

    // Update M matrix

    int j;

    M.addRowCol(i);
    dummyGpn.addRow(i);    

    for ( j = 0 ; j < NbasisUU() ; j++ )
    {
        calcMij(M("&",i,j),i,j);
        calcMij(M("&",j,i),j,i);
    }                 

    // Update cholesky factorisation

    retVector<double> tmpva;

    L.add(i,M,dummyGn,dummyGpn,onedoublevec(NbasisUU(),tmpva));

    // Call sety to update the model

    SVM_Vector::addtspaceFeat(i);

    return sety(locyval);
}

int SVM_Gentyp::removeFromBasisUU(int i)
{
    NiceAssert( !isBasisUser );
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisUU() );

    // Remove from basis set

    locbasis.remove(i);

    // Update M matrix

    M.removeRowCol(i);
    dummyGpn.removeRow(i);    

    // Update cholesky factorisation

    retVector<double> tmpva;

    L.remove(i,M,dummyGn,dummyGpn,onedoublevec(NbasisUU(),tmpva));

    // Call sety to update the model

    SVM_Vector::removetspaceFeat(i);

    return sety(locyval);
}

int SVM_Gentyp::setBasisUU(int i, const gentype &o)
{
    NiceAssert( !isBasisUser );
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisUU() );

    // Remove from basis set

    locbasis.remove(i);

    // Update M matrix

    M.removeRowCol(i);
    dummyGpn.removeRow(i);    

    // Update cholesky factorisation

    retVector<double> tmpva;

    L.remove(i,M,dummyGn,dummyGpn,onedoublevec(NbasisUU(),tmpva));

    // Add to basis set

    locbasis.add(i);
    locbasis("&",i) = o;

    // Update M matrix

    int j;

    M.addRowCol(i);
    dummyGpn.addRow(i);    

    for ( j = 0 ; j < NbasisUU() ; j++ )
    {
        calcMij(M("&",i,j),i,j);
        calcMij(M("&",j,i),j,i);
    }

    // Update cholesky factorisation

    L.add(i,M,dummyGn,dummyGpn,onedoublevec(NbasisUU(),tmpva));

    // Call sety to update the model

    return sety(locyval);
}

int SVM_Gentyp::setBasisUU(const Vector<gentype> &o)
{
    NiceAssert( !isBasisUser );

    int i,j;

    // Add to basis set

    locbasis = o;

    // Update cholesky factorisation

    retVector<double> tmpva;
    retMatrix<double> tmpma;
    retMatrix<double> tmpmb;

    for ( i = NbasisUU()-1 ; i >= 0 ; i-- )
    {
        L.remove(i,M(0,1,i-1,0,1,i-1,tmpma),dummyGn,dummyGpn(0,1,i-1,0,1,-1,tmpmb),onedoublevec(i,tmpva));
    }

    // Update M matrix

    M.resize(NbasisUU(),NbasisUU());
    dummyGpn.resize(NbasisUU(),0);    

    for ( i = 0 ; i < NbasisUU() ; i++ )
    {
        for ( j = 0 ; j < NbasisUU() ; j++ )
        {
            calcMij(M("&",i,j),i,j);
        }
    }

    // Update cholesky factorisation

    for ( i = 0 ; i <= NbasisUU()-1 ; i++ )
    {
        L.add(i,M(0,1,i,0,1,i,tmpma),dummyGn,dummyGpn(0,1,i,0,1,-1,tmpmb),onedoublevec(i+1,tmpva));
    }

    // Call sety to update the model

    SVM_Vector::settspaceDim(NbasisUU());

    return sety(locyval);
}

int SVM_Gentyp::setBasisYUU(void)
{
    int res = 0;

    if ( !isBasisUser )
    {
        setBasisUU(locyval);
        isBasisUser = 1;
        res = 1;
    }

    return res;
}

int SVM_Gentyp::setBasisUUU(void)
{
    int res = 0;

    if ( isBasisUser )
    {
        isBasisUser = 0;
        res = 1;
    }

    return res;
}






















// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================

void SVM_Gentyp::ConvertYtoVec(const gentype &src, Vector<double> &hattau) const
{
    // Convert \tau_j to \widehat{\tau}_j
    //
    // \widehat{\tau}_j = L^{-1} \tau_j
    //
    // where \tau_j = [ <y^j,o^0> ]
    //                [ <y^j,o^1> ]
    //                [    ...    ]

    Vector<double> tau(NbasisUU());
    hattau.resize(NbasisUU());

    if ( NbasisUU() )
    {
        int i;

        for ( i = 0 ; i < NbasisUU() ; i++ )
        {
            calcMiy(tau("&",i),i,src);
        }

        L.forwardElim(hattau,tau);
    }

    return;
}

void SVM_Gentyp::ConvertYtoOut(const Vector<double> &hattau, gentype &dest) const
{
    // Convert \widehat{\alpha}_j to y
    //
    // \alpha_j = L^{-T}.\widehat{\alpha}_j
    // 
    // res = sum_k (\alpha_j)i o_i

    NiceAssert( hattau.size() == NbasisUU() );
    dest.zero();

    if ( NbasisUU() )
    {
        int i;

        Vector<double> tau(NbasisUU());

        L.backwardSubst(tau,hattau);

        for ( i = 0 ; i < NbasisUU() ; i++ )
        {
            if ( !i )
            {
                dest =  VbasisUU()(i);
                dest *= tau(i);
            }

            else
            {
                dest += (tau(i)*VbasisUU()(i));
            }
        }
    }

    dest.scalarfn_setisscalarfn(0);

    return;
}

void SVM_Gentyp::calcMij(double &res, int i, int j) const
{
    calcMxy(res,VbasisUU()(i),VbasisUU()(j));

    return;
}

void SVM_Gentyp::calcMiy(double &res, int i, const gentype &y) const
{
    calcMxy(res,VbasisUU()(i),y);

    return;
}

void SVM_Gentyp::calcMxy(double &res, const gentype &x, const gentype &y) const
{
    SparseVector<gentype> xx;
    SparseVector<gentype> yy;

    xx("&",0) = x;
    yy("&",0) = y;

    vecInfo xxinfo;
    vecInfo yyinfo;

    getUUOutputKernel().getvecInfo(xxinfo,xx);
    getUUOutputKernel().getvecInfo(yyinfo,yy);

    getUUOutputKernel().K2(res,xx,yy,xxinfo,yyinfo,zerointgentype());

    return;
}

double SVM_Gentyp::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( ha.isValEqnDir() && hb.isValEqnDir() )
    {
        NiceAssert( ( db == 0 ) || ( db == 2 ) );

        gentype hha(ha);
        gentype hhb(hb);

        hha.scalarfn_setisscalarfn(1);
        hhb.scalarfn_setisscalarfn(1);

        if ( db )
        {
            res = (double) norm2(hha-hhb);
        }
    }

    else if ( ha.isValEqnDir() )
    {
        NiceAssert( ( db == 0 ) || ( db == 2 ) );

        gentype hha(ha);

        hha.scalarfn_setisscalarfn(1);

        if ( db )
        {
            res = (double) norm2(hha-hb);
        }
    }

    else if ( hb.isValEqnDir() )
    {
        NiceAssert( ( db == 0 ) || ( db == 2 ) );

        gentype hhb(hb);

        hhb.scalarfn_setisscalarfn(1);

        if ( db )
        {
            res = (double) norm2(ha-hhb);
        }
    }

    else if ( ha.isValNull() || ha.isValInteger() || ha.isValReal() )
    {
        if ( db == +1 )
        {
            // treat as lower bound constraint ha >= hb

            if ( (double) ha < (double) hb )
            {
                res = ( (double) ha ) - ( (double) hb );
                res *= res;
            }
        }

        else if ( db == -1 )
        {
            // treat as upper bound constraint ha <= hb

            if ( (double) ha > (double) hb )
            {
                res = ( (double) ha ) - ( (double) hb );
                res *= res;
            }
        }

        else if ( db )
        {
            res = ( (double) ha ) - ( (double) hb );
            res *= res;
        }
    }

    else if ( ha.isValAnion() || ha.isValVector() || ha.isValMatrix() )
    {
        NiceAssert( ( db == 0 ) || ( db == 2 ) );

        if ( db )
        {
            res = (double) norm2(ha-hb);
        }
    }

    else
    {
        // Sets, graphs and strings are comparable by binary multiplication

        NiceAssert( ( db == 0 ) || ( db == 2 ) );

        if ( db )
        {
            res = ( ha == hb ) ? 0 : 1;
        }
    }

    return res;
}


