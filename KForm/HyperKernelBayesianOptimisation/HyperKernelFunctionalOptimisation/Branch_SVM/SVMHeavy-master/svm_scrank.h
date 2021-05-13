
//
// Scalar+Ranking SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_scrank_h
#define _svm_scrank_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar.h"


// Like SVM_BiRank, but with scalar classifier base.  Hence both equalities
// and inequalities are allowed.
//
// This retains classes 0 and 2 as per the base class, but replaces
// classes +1 and -1 with rank-style classes.  Does not enforce fixed
// bias.  Note that is assumed that x values for rank constraints do
// not change.


class SVM_ScRank;
class SVM_ScScor;
template <class T> class SVM_Vector_redbin;


std::ostream &operator<<(std::ostream &output, const SVM_ScRank &src );
std::istream &operator>>(std::istream &input,        SVM_ScRank &dest);

// Swap function

inline void qswap(SVM_ScRank &a, SVM_ScRank &b);


class SVM_ScRank : public SVM_Scalar
{
    friend class SVM_ScScor;

public:

    // Constructors, destructors, assignment etc..

    SVM_ScRank();
    SVM_ScRank(const SVM_ScRank &src);
    SVM_ScRank(const SVM_ScRank &src, const ML_Base *xsrc);
    SVM_ScRank &operator=(const SVM_ScRank &src) { assign(src); return *this; }
    virtual ~SVM_ScRank();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int restart(void) { SVM_ScRank temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information functions (training data):

    virtual int type(void)       const { return 10; }
    virtual int subtype(void)    const { return 0;  }

    // Kernel Modification - this does all the work by changing how K is
    // evaluated.

    virtual double &KReal(double &res, int i, int j) const;

    // Add/remove data:

    virtual int addTrainingVector (int i, double z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2, double Cweighfuzz = 1);
    virtual int qaddTrainingVector(int i, double z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2, double Cweighfuzz = 1);

    virtual int addTrainingVector (int i, const Vector<double> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d, const Vector<double> &Cweighfuzz);
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d, const Vector<double> &Cweighfuzz);

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(const Vector<int> &d);
    virtual int setd(int i, int d);

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg) const;

    virtual void  eTrainingVector(gentype         &res,                int i) const;
    virtual void dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const;
    virtual void dgTrainingVector(SparseVector<gentype> &resx, int i) const;
    virtual void drTrainingVector(Vector<gentype> &res, gentype &resn, int i) const;

protected:

    // By default kernel evaluation falls through to ML_Base after 
    // pre-processing.  Setting the following is an alternative.

    double &(*evalKReal)(double &res, int i, int j, void *evalarg);
    void *evalKRealarg;

    // Protected function passthrough

    int maxFreeAlphaBias(void) { return SVM_Scalar::maxFreeAlphaBias(); }
    int fact_minverse(Vector<double> &dalpha, Vector<double> &dbeta, const Vector<double> &bAlpha, const Vector<double> &bBeta) { return SVM_Scalar::fact_minverse(dalpha,dbeta,bAlpha,bBeta); }


private:

    // Blocked functions

    void setGpnExt(Matrix<double> *GpnExtOld, Matrix<double> *GpnExtNew) { (void) GpnExtOld; (void) GpnExtNew; throw("setGpnExt blocked in svm_scrank"); return; }
    void naivesetGpnExt(Matrix<double> *GpnExtVal) { (void) GpnExtVal; throw("naivesetGpnExt blocked in svm_scrank"); return; }

    void setbiasdim(int xbiasdim, int addpos, double addval, int rempos) { (void) xbiasdim; (void) addpos; (void) addval; (void) rempos; throw("setbiasdim blocked in svm_scrank"); return; }
    void setBiasVMulti(const Vector<double> &nwbias) { (void) nwbias; throw("setBiasVMulti blocked in svm_scrank"); return; }

    void setgn(const Vector<double> &gnnew) { (void) gnnew; throw("setgn blocked in svm_scrank"); return; }
    void setGn(const Matrix<double> &Gnnew) { (void) Gnnew; throw("setGn blocked in svm_scrank"); return; }

private:

    // This sets mode for KReal.
    //
    // locTestMode = 0: K([i0,j0],[i1,k1]) = K(x_i0,x_i1) + K(x_j0,x_j1) - 2K(x_i0,x_j1)
    // locTestMode = 1: K([i0,j0],y) = K(x_i0,y) - K(x_i0,y)
      
    int locTestMode;

    // Gpn vector: bias does not affect the ranking constraint variables,
    //             so we need to set Gpn = 0 for these
    // locd: resetKernel does some strange things with setd at svm_scalar
    //       level, so we need to keep a local copy.

    Matrix<double> inGpn;
    Vector<int> locd;

    SVM_ScRank *thisthis;
    SVM_ScRank **thisthisthis;
};

inline void qswap(SVM_ScRank &a, SVM_ScRank &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_ScRank::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_ScRank &b = dynamic_cast<SVM_ScRank &>(bb.getML());

    SVM_Scalar::qswapinternal(b);
    
    qswap(locTestMode,b.locTestMode);
    qswap(inGpn      ,b.inGpn      );
    qswap(locd       ,b.locd       );

    double &(*temp)(double &res, int i, int j, void *evalarg);
    void *temparg;

    temp    = evalKReal;    evalKReal    = b.evalKReal;    b.evalKReal    = temp;
    temparg = evalKRealarg; evalKRealarg = b.evalKRealarg; b.evalKRealarg = temparg;

    SVM_Scalar::naivesetGpnExt(&inGpn);

    return;
}

inline void SVM_ScRank::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_ScRank &b = dynamic_cast<const SVM_ScRank &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    locTestMode = b.locTestMode;
    inGpn       = b.inGpn;
    locd        = b.locd;

    evalKReal    = b.evalKReal;
    evalKRealarg = b.evalKRealarg;

    SVM_Scalar::naivesetGpnExt(&inGpn);

    return;
}

inline void SVM_ScRank::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_ScRank &src = dynamic_cast<const SVM_ScRank &>(bb.getMLconst());

    SVM_Scalar::assign(static_cast<const SVM_Scalar &>(src),onlySemiCopy);

    locTestMode = src.locTestMode;
    inGpn       = src.inGpn;
    locd        = src.locd;

    evalKReal    = src.evalKReal;
    evalKRealarg = src.evalKRealarg;

    SVM_Scalar::naivesetGpnExt(&inGpn);

    return;
}

#endif
