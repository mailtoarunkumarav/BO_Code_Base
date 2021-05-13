
//
// Simple kernel block.
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_mercer_h
#define _blk_mercer_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"


// Let's you define a kernel that can be (simply) inheritted by other
// blocks.  Literally equivalent to BLK_Nopnop


class BLK_Mercer;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_Mercer &a, BLK_Mercer &b);
inline void qswap(BLK_Mercer *&a, BLK_Mercer *&b);



class BLK_Mercer : public BLK_Generic
{
public:

    // Constructors, destructors, assignment etc..

    BLK_Mercer(int isIndPrune = 0);
    BLK_Mercer(const BLK_Mercer &src, int isIndPrune = 0);
    BLK_Mercer(const BLK_Mercer &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_Mercer &operator=(const BLK_Mercer &src) { assign(src); return *this; }
    virtual ~BLK_Mercer();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual int type(void)    const { return 211; }
    virtual int subtype(void) const { return 0;   }

    virtual int setmercachesize(int nv);

    // Kernel transfer
    //
    // Cacheing only works in the 2-kernel case.  Integers act as
    // indices for merCache, merHit tells us if it is already calculated
    // or needs to be calculated.

    virtual int isKVarianceNZ(void) const;

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const;
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const;

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const;
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const;

private:

    Matrix<gentype> merCache;
    Matrix<unsigned int> merHit;

    Matrix<gentype> merCacheVar;
    Matrix<unsigned int> merHitVar;

    BLK_Mercer *thisthis[2];
};

inline void qswap(BLK_Mercer &a, BLK_Mercer &b)
{
    a.qswapinternal(b);

    return;
}

inline void qswap(BLK_Mercer *&a, BLK_Mercer *&b)
{
    BLK_Mercer *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}

inline void BLK_Mercer::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Mercer &b = dynamic_cast<BLK_Mercer &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    qswap(merCache,b.merCache);
    qswap(merHit  ,b.merHit  );

    qswap(merCacheVar,b.merCacheVar);
    qswap(merHitVar  ,b.merHitVar  );

    return;
}

inline void BLK_Mercer::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Mercer &b = dynamic_cast<const BLK_Mercer &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    //merCache = b.merCache;
    //merHit   = b.merHit;

    //merCacheVar = b.merCacheVar;
    //merHitVar   = b.merHitVar;

    return;
}

inline void BLK_Mercer::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Mercer &src = dynamic_cast<const BLK_Mercer &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    merCache = src.merCache;
    merHit   = src.merHit;

    merCacheVar = src.merCacheVar;
    merHitVar   = src.merHitVar;

    return;
}

#endif
