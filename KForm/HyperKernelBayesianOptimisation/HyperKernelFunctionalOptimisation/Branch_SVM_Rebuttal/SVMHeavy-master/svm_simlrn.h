
//
// Similarity learning SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_simlrn_h
#define _svm_simlrn_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar.h"



//void evalKSVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);
//void evalxySVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);
//void evalSigmaSVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);

class SVM_SimLrn;


// Swap function

inline void qswap(SVM_SimLrn &a, SVM_SimLrn &b);

// Training helper callback function

class SVM_SimLrn : public SVM_Scalar
{
    friend void evalKSVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalxySVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalSigmaSVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);

public:

    SVM_SimLrn();
    SVM_SimLrn(const SVM_SimLrn &src);
    SVM_SimLrn(const SVM_SimLrn &src, const ML_Base *xsrc);
    SVM_SimLrn &operator=(const SVM_SimLrn &src) { assign(src); return *this; }
    virtual ~SVM_SimLrn();

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int type(void)       const { return 19; }
    virtual int subtype(void)    const { return 0;  }

    // Kernel transfer normalisation

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

    // Train the SVM

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);

    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual double theta(void)   const { return xtheta;   }
    virtual int    simnorm(void) const { return xsimnorm; }

    virtual int settheta(double nv) { xtheta   = nv; return 1; }
    virtual int setsimnorm(int nv)  { xsimnorm = nv; return 1; }

private:

    // psd projection regularisation

    double xtheta;
    int xsimnorm;

    // Internal variables held only during training

    Kcache<double> *xycachesim;
    Kcache<double> *kerncachesim;
    Kcache<double> *sigmacachesim;

    Matrix<double> *GpOuter;
    Matrix<double> *GpSigmaOuter;
    Matrix<double> *xyOuter;
};

inline void qswap(SVM_SimLrn &a, SVM_SimLrn &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_SimLrn::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_SimLrn &b = dynamic_cast<SVM_SimLrn &>(bb.getML());

    SVM_Scalar::qswapinternal(b);

    qswap(xtheta  ,b.xtheta  );
    qswap(xsimnorm,b.xsimnorm);

    return;
}

inline void SVM_SimLrn::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_SimLrn &b = dynamic_cast<const SVM_SimLrn &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    xtheta   = b.xtheta;
    xsimnorm = b.xsimnorm;

    return;
}

inline void SVM_SimLrn::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_SimLrn &src = dynamic_cast<const SVM_SimLrn &>(bb.getMLconst());

    SVM_Scalar::assign(static_cast<const SVM_Scalar &>(src),onlySemiCopy);

    xtheta   = src.xtheta;
    xsimnorm = src.xsimnorm;

    return;
}

#endif
