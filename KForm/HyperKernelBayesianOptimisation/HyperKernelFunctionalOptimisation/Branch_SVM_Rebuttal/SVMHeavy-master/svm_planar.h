//FIXME: change to use cheatscache via kernel 800 callback.  Then you can get rid of K2 stuff here because the cache is *automatically* called direct from ML_Base.  Note that all kernel
// references including getKernel must be diverted appropriately to cheatscache: NOTHING must be allowed to change the kernel in SVM_Planar!

//
// Planar SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_planar_h
#define _svm_planar_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar.h"



class SVM_Planar;

// Swap function

inline void qswap(SVM_Planar &a, SVM_Planar &b);


class SVM_Planar : public SVM_Scalar
{
public:

    // Constructors, destructors, assignment etc..

    SVM_Planar();
    SVM_Planar(const SVM_Planar &src);
    SVM_Planar(const SVM_Planar &src, const ML_Base *xsrc);
    SVM_Planar &operator=(const SVM_Planar &src) { assign(src); return *this; }
    virtual ~SVM_Planar();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int restart(void) { SVM_Planar temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information functions (training data):
    // 
    // calcDist: - i == -1 then this assumes that both ha and hb are vectors
    //           - i >= 0, i < N then assumes ha is a vector and hb a scalar
    //             and will first convert ha -> ha.x(i).fff(7) to get scalar
    //           - i >= N also assumes ha is a vector and hb a scalar but
    //             will convert ha -> ha.u(i-N)

    virtual int type(void)    const { return 16; }
    virtual int subtype(void) const { return 0;  }

    virtual int tspaceDim(void) const { return bdim ; }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return ( defproj == -1 ) ? gOutType() : 'R'; }
    virtual char targType(void) const { return 'R'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int i = -1, int db = 2) const;

    virtual int isUnderlyingScalar(void) const { return 1; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    // Kernel Modification - this does all the work by changing how K is
    // evaluated.

    virtual gentype &K2(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { return ML_Base::K2(res,xa,xb,xainf,xbinf); }

    virtual gentype        &K2(              gentype        &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;
    virtual gentype        &K2(              gentype        &res, int i, int j, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;
    virtual gentype        &K2(              gentype        &res, int i, int j, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;
    virtual double         &K2(              double         &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;
    virtual Matrix<double> &K2(int spaceDim, Matrix<double> &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;
    virtual d_anion        &K2(int order,    d_anion        &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;

    // Add/remove data:

    virtual int addTrainingVector (int i, double z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, double z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<double> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);
    virtual int removeTrainingVector(int i, int num);

    virtual int setx(int                i, const SparseVector<gentype>          &x);
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x);
    virtual int setx(                      const Vector<SparseVector<gentype> > &x);

    virtual int setd(int                i, int                d);
    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(                      const Vector<int> &d);

    // Modification

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void prepareKernel(void) { return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1);
    virtual void setmemsize(int memsize);

    virtual int setVarBias(void);
    virtual int setPosBias(void);
    virtual int setNegBias(void);
    virtual int setFixedBias(double newbias = 0.0);
    virtual int setVarBias(int q);
    virtual int setPosBias(int q);
    virtual int setNegBias(int q);
    virtual int setFixedBias(int q, double newbias = 0.0);

    // Training

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi = NULL) const;

    // Output basis

    virtual int NbasisVV(void)    const { return locbasis.size();       }
    virtual int basisTypeVV(void) const { return 0;                     }
    virtual int defProjVV(void)   const { return defproj;               }

    virtual const Vector<gentype> &VbasisVV(void) const { return locbasisgt; }

    virtual int setBasisYVV(void) { throw("Function setBasisY not available for this ML type"); return 0; }
    virtual int setBasisUVV(void) { return 0; }
    virtual int addToBasisVV(int i, const gentype &o);
    virtual int removeFromBasisVV(int i);
    virtual int setBasisVV(int i, const gentype &o);
    virtual int setBasisVV(const Vector<gentype> &o);
    virtual int setDefaultProjectionVV(int d) { defproj = d; return 1; }
    virtual int setBasisVV(int i, int d) { return ML_Base::setBasisVV(i,d); }

protected:

    // Local basis and factorisation

    Vector<Vector<double> > locbasis;
    Vector<gentype> locbasisgt;
    Matrix<double> VV;

    // Protected function passthrough

    int maxFreeAlphaBias(void) { return SVM_Scalar::maxFreeAlphaBias(); }
    int fact_minverse(Vector<double> &dalpha, Vector<double> &dbeta, const Vector<double> &bAlpha, const Vector<double> &bBeta) { return SVM_Scalar::fact_minverse(dalpha,dbeta,bAlpha,bBeta); }

    void refactorVV(int updateGpn = 1);
    Vector<Vector<double> > &reflocbasis(void) { return locbasis; }
    void reconstructlocbasisgt(void);
    Vector<int> &reflocd(void) { return locd; }
    int getbdim(void) { return bdim; }
    virtual int setBasisVV(int i, const gentype &o, int updateU);

    // Force exhaustive evaluation of gh(i) for i >= 0 (used by svm_cyclic)
    // Assumed set briefly then reset immediately afterward

    int ghEvalFull;

private:

    // Blocked functions

    void setGpnExt(Matrix<double> *GpnExtOld, Matrix<double> *GpnExtNew) { (void) GpnExtOld; (void) GpnExtNew; throw("setGpnExt blocked in svm_Planar"); return; }
    void naivesetGpnExt(Matrix<double> *GpnExtVal) { (void) GpnExtVal; throw("naivesetGpnExt blocked in svm_Planar"); return; }

    void setbiasdim(int xbiasdim, int addpos, double addval, int rempos) { (void) xbiasdim; (void) addpos; (void) addval; (void) rempos; throw("setbiasdim blocked in svm_Planar"); return; }
    void setBiasVMulti(const Vector<double> &nwbias) { (void) nwbias; throw("setBiasVMulti blocked in svm_Planar"); return; }

    void setgn(const Vector<double> &gnnew) { (void) gnnew; throw("setgn blocked in svm_Planar"); return; }
    void setGn(const Matrix<double> &Gnnew) { (void) Gnnew; throw("setGn blocked in svm_Planar"); return; }

private:

    SVM_Scalar cheatscache;
    int midadd;

    // Gpn vector: bias does not affect the ranking constraint variables,
    //             so we need to set Gpn = 0 for these
    // locd: resetKernel does some strange things with setd at svm_scalar
    //       level, so we need to keep a local copy.

    Matrix<double> inGpn;
    Vector<int> locd;

    // Basis dimensions

    int bdim;
    int defproj;

    // Helper functions

    void calcVVij(double &res, int i, int j) const;
    int rankcalcGpn(Vector<double> &res, int d, const SparseVector<gentype> &x, int i);

    // thisthisthisthisthisthisthatthisthisthis

    SVM_Planar *thisthis;
    SVM_Planar **thisthisthis;
};

inline void qswap(SVM_Planar &a, SVM_Planar &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Planar::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Planar &b = dynamic_cast<SVM_Planar &>(bb.getML());

    SVM_Scalar::qswapinternal(b);
    
    qswap(inGpn         ,b.inGpn         );
    qswap(locd          ,b.locd          );
    qswap(locbasis      ,b.locbasis      );
    qswap(VV            ,b.VV            );
    qswap(bdim          ,b.bdim          );
    qswap(cheatscache   ,b.cheatscache   );
    qswap(midadd        ,b.midadd        );
    qswap(defproj       ,b.defproj       );

    SVM_Scalar::naivesetGpnExt(&inGpn);

    return;
}

inline void SVM_Planar::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Planar &b = dynamic_cast<const SVM_Planar &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    inGpn          = b.inGpn;
    locd           = b.locd;
    //locbasis       = b.locbasis;
    //VV             = b.VV;
    //bdim           = b.bdim;
    midadd         = b.midadd;
    defproj        = b.defproj;

    SVM_Scalar::naivesetGpnExt(&inGpn);

    return;
}

inline void SVM_Planar::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Planar &src = dynamic_cast<const SVM_Planar &>(bb.getMLconst());

    SVM_Scalar::assign(static_cast<const SVM_Scalar &>(src),onlySemiCopy);

    inGpn          = src.inGpn;
    locd           = src.locd;
    locbasis       = src.locbasis;
    VV             = src.VV;
    bdim           = src.bdim;
    cheatscache    = src.cheatscache;
    midadd         = src.midadd;
    defproj        = src.defproj;

    SVM_Scalar::naivesetGpnExt(&inGpn);

    return;
}

#endif
