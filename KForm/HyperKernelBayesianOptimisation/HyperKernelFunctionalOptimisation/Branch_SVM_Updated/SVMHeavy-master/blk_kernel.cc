
//
// Kernel specialisation block
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
#include "blk_kernel.h"


#define MINUINTSIZE 16

BLK_Kernel::BLK_Kernel(int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis[0] = this;

    setaltx(NULL);

    return;
}

BLK_Kernel::BLK_Kernel(const BLK_Kernel &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis[0] = this;

    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_Kernel::BLK_Kernel(const BLK_Kernel &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis[0] = this;

    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_Kernel::~BLK_Kernel()
{
    return;
}

int BLK_Kernel::isKVarianceNZ(void) const
{
    return ML_Base::isKVarianceNZ();
}

std::ostream &BLK_Kernel::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Kernel specialisation block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Kernel::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}

void BLK_Kernel::K0xfer(gentype &res, int &minmaxind, int typeis,
                       int xdim, int densetype, int resmode, int mlid) const
{
    ML_Base::K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid);

    return;
}

void BLK_Kernel::K0xfer(double &res, int &minmaxind, int typeis,
                       int xdim, int densetype, int resmode, int mlid) const
{
    ML_Base::K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid);

    return;
}

void BLK_Kernel::K1xfer(gentype &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, 
                       const vecInfo &xainfo, 
                       int ia, 
                       int xdim, int densetype, int resmode, int mlid) const
{
    gentype temp;
    gentype dummy;

    setzero(res);

    int i;

    for ( i = 0 ; i < N() ; i++ )
    {
        gentype diffis(norm2(xa-x()(i)));

        ML_Base::K2xfer(dummy,diffis,temp,minmaxind,typeis,xa,x()(i),xainfo,xinfo()(i),ia,i,xdim,densetype,resmode,mlid);

        res += lambdaKB()(i)*temp;
    }

    return;
}

void BLK_Kernel::K1xfer(double &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, 
                       const vecInfo &xainfo, 
                       int ia, 
                       int xdim, int densetype, int resmode, int mlid) const
{
    double temp;
    double dummy;

    setzero(res);

    int i;

    for ( i = 0 ; i < N() ; i++ )
    {
        double diffis = (double) norm2(xa-x()(i));

        ML_Base::K2xfer(dummy,diffis,temp,minmaxind,typeis,xa,x()(i),xainfo,xinfo()(i),ia,i,xdim,densetype,resmode,mlid);

        res += lambdaKB()(i)*temp;
    }

    return;
}

void BLK_Kernel::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
{
    (void) dxyprod;
    (void) ddiffis;

    gentype temp;

    setzero(res);

    int i,j;

    for ( i = 0 ; i < N() ; i++ )
    {
        for ( j = 0 ; j < N() ; j++ )
        {
            ML_Base::K4xfer(temp,minmaxind,typeis,xa,xb,x()(i),x()(j),xainfo,xbinfo,xinfo()(i),xinfo()(j),ia,ib,ia,ib,xdim,densetype,resmode,mlid);

            res += lambdaKB()(i)*temp;
        }
    }

    return;
}

void BLK_Kernel::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
{
    (void) dxyprod;
    (void) ddiffis;

    double temp;

    setzero(res);

    int i,j;

    for ( i = 0 ; i < N() ; i++ )
    {
        for ( j = 0 ; j < N() ; j++ )
        {
            ML_Base::K4xfer(temp,minmaxind,typeis,xa,xb,x()(i),x()(j),xainfo,xbinfo,xinfo()(i),xinfo()(j),ia,ib,ia,ib,xdim,densetype,resmode,mlid);

            res += lambdaKB()(i)*temp;
        }
    }

    return;
}

void BLK_Kernel::K3xfer(gentype &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                       int ia, int ib, int ic, 
                       int xdim, int densetype, int resmode, int mlid) const
{
    int m = 3;

    Vector<const SparseVector<gentype> *> xx(m);
    Vector<const vecInfo *> xxinfo(m);
    Vector<int> ii(m);

    int z = 0;

    xx("&",z) = &xa;
    xx("&",1) = &xb;
    xx("&",2) = &xc;

    xxinfo("&",z) = &xainfo;
    xxinfo("&",1) = &xbinfo;
    xxinfo("&",2) = &xcinfo;

    ii("&",z) = ia;
    ii("&",1) = ib;
    ii("&",2) = ic;

    Kmxfer(res,minmaxind,typeis,xx,xxinfo,ii,xdim,m,densetype,resmode,mlid);

    return;
}

void BLK_Kernel::K3xfer(double &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                       int ia, int ib, int ic, 
                       int xdim, int densetype, int resmode, int mlid) const
{
    int m = 3;

    Vector<const SparseVector<gentype> *> xx(m);
    Vector<const vecInfo *> xxinfo(m);
    Vector<int> ii(m);

    int z = 0;

    xx("&",z) = &xa;
    xx("&",1) = &xb;
    xx("&",2) = &xc;

    xxinfo("&",z) = &xainfo;
    xxinfo("&",1) = &xbinfo;
    xxinfo("&",2) = &xcinfo;

    ii("&",z) = ia;
    ii("&",1) = ib;
    ii("&",2) = ic;

    Kmxfer(res,minmaxind,typeis,xx,xxinfo,ii,xdim,m,densetype,resmode,mlid);

    return;
}

void BLK_Kernel::K4xfer(gentype &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                       int ia, int ib, int ic, int id,
                       int xdim, int densetype, int resmode, int mlid) const
{
    int m = 4;

    Vector<const SparseVector<gentype> *> xx(m);
    Vector<const vecInfo *> xxinfo(m);
    Vector<int> ii(m);

    int z = 0;

    xx("&",z) = &xa;
    xx("&",1) = &xb;
    xx("&",2) = &xc;
    xx("&",3) = &xd;

    xxinfo("&",z) = &xainfo;
    xxinfo("&",1) = &xbinfo;
    xxinfo("&",2) = &xcinfo;
    xxinfo("&",3) = &xdinfo;

    ii("&",z) = ia;
    ii("&",1) = ib;
    ii("&",2) = ic;
    ii("&",3) = id;

    Kmxfer(res,minmaxind,typeis,xx,xxinfo,ii,xdim,m,densetype,resmode,mlid);

    return;
}

void BLK_Kernel::K4xfer(double &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                       int ia, int ib, int ic, int id,
                       int xdim, int densetype, int resmode, int mlid) const
{
    int m = 4;

    Vector<const SparseVector<gentype> *> xx(m);
    Vector<const vecInfo *> xxinfo(m);
    Vector<int> ii(m);

    int z = 0;

    xx("&",z) = &xa;
    xx("&",1) = &xb;
    xx("&",2) = &xc;
    xx("&",3) = &xd;

    xxinfo("&",z) = &xainfo;
    xxinfo("&",1) = &xbinfo;
    xxinfo("&",2) = &xcinfo;
    xxinfo("&",3) = &xdinfo;

    ii("&",z) = ia;
    ii("&",1) = ib;
    ii("&",2) = ic;
    ii("&",3) = id;

    Kmxfer(res,minmaxind,typeis,xx,xxinfo,ii,xdim,m,densetype,resmode,mlid);

    return;
}

void BLK_Kernel::Kmxfer(gentype &res, int &minmaxind, int typeis,
                       Vector<const SparseVector<gentype> *> &xx,
                       Vector<const vecInfo *> &xxinfo,
                       Vector<int> &ii,
                       int xdim, int m, int densetype, int resmode, int mlid) const
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xx;
    (void) xxinfo;
    (void) ii;
    (void) xdim;
    (void) m;
    (void) resmode;
    (void) densetype;
    (void) mlid;

    throw("Haven't implemented kernel specialisation above second order yet");

    return;
}

void BLK_Kernel::Kmxfer(double &res, int &minmaxind, int typeis,
                       Vector<const SparseVector<gentype> *> &xx,
                       Vector<const vecInfo *> &xxinfo,
                       Vector<int> &ii,
                       int xdim, int m, int densetype, int resmode, int mlid) const
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xx;
    (void) xxinfo;
    (void) ii;
    (void) xdim;
    (void) m;
    (void) resmode;
    (void) densetype;
    (void) mlid;

    throw("Haven't implemented kernel specialisation above second order yet");

    return;
}
