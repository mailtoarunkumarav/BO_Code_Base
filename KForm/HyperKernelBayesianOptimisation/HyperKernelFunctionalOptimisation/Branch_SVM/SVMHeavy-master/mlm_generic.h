
//
// Type-II multi-layer kernel-machine base class
//
// Version: 7
// Date: 06/07/2018
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// This operates as a wrap-around for SVM_Generic, with some indirection,
// renaming, functional hijacking etc.
//
// These functions need to be provided overwritten in variants of MLM_Generic:
//
// virtual       SVM_Generic &getQ(void)            { return QQ; }
// virtual const SVM_Generic &getQconst(void) const { return QQ; }
//
// MLM_Generic();
// MLM_Generic(const MLM_Generic &src);
// MLM_Generic(const MLM_Generic &src, const ML_Base *srcx);
//
// virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
// virtual void semicopy(const ML_Base &src);
// virtual void qswapinternal(ML_Base &b);
// virtual std::ostream &printstream(std::ostream &output) const;
// virtual std::istream &inputstream(std::istream &input );
//
// int type(void)    const { return 800; }
// int subtype(void) const { return 0;   }
//
// virtual int train(int &res, svmvolatile int &killSwitch);
//


#ifndef _mlm_generic_h
#define _mlm_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.h"
#include "svm_generic.h"
#include "svm_scalar.h"





class MLM_Generic;

// Swap and zeroing (restarting) functions

inline void qswap(MLM_Generic &a, MLM_Generic &b);
inline MLM_Generic &setzero(MLM_Generic &a);

class MLM_Generic : public ML_Base
{
public:

    // Constructors, destructors, assignment etc..

    MLM_Generic();
    MLM_Generic(const MLM_Generic &src);
    MLM_Generic(const MLM_Generic &src, const ML_Base *srcx);
    MLM_Generic &operator=(const MLM_Generic &src) { assign(src); return *this; }
    virtual ~MLM_Generic() { return; }

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void setmemsize(int memsize);

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const;

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual       ML_Base &getML(void)            { return static_cast<      ML_Base &>(getMLM());      }
    virtual const ML_Base &getMLconst(void) const { return static_cast<const ML_Base &>(getMLMconst()); }

    // Information functions (training data):

    virtual int N(void)       const { return getQconst().N();    }
    virtual int NNC(int d)    const { return getQconst().NNC(d); }
    virtual int type(void)    const { return -1;                 }
    virtual int subtype(void) const { return -1;                 }

    virtual int tspaceDim(void)    const { return getQconst().tspaceDim();    }
    virtual int xspaceDim(void)    const { return getQconst().xspaceDim();    }
    virtual int fspaceDim(void)    const { return getQconst().fspaceDim();    }
    virtual int tspaceSparse(void) const { return getQconst().tspaceSparse(); }
    virtual int xspaceSparse(void) const { return getQconst().xspaceSparse(); }
    virtual int numClasses(void)   const { return getQconst().numClasses();   }
    virtual int order(void)        const { return getQconst().order();        }

    virtual int isTrained(void) const { return getQconst().isTrained() && xistrained; }
    virtual int isMutable(void) const { return getQconst().isMutable(); }
    virtual int isPool   (void) const { return getQconst().isPool   (); }

    virtual char gOutType(void) const { return getQconst().gOutType(); }
    virtual char hOutType(void) const { return getQconst().hOutType(); }
    virtual char targType(void) const { return getQconst().targType(); }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const { return getQconst().calcDist(ha,hb,ia,db); }

    virtual int isUnderlyingScalar(void) const { return getQconst().isUnderlyingScalar(); }
    virtual int isUnderlyingVector(void) const { return getQconst().isUnderlyingVector(); }
    virtual int isUnderlyingAnions(void) const { return getQconst().isUnderlyingAnions(); }

    virtual const Vector<int> &ClassLabels(void)   const { return getQconst().ClassLabels();        }
    virtual int getInternalClass(const gentype &y) const { return getQconst().getInternalClass(y);  }
    virtual int numInternalClasses(void)           const { return getQconst().numInternalClasses(); }
    virtual int isenabled(int i)                   const { return getQconst().isenabled(i);         }

    virtual double C(void)         const { return getQconst().C();         }
    virtual double sigma(void)     const { return getQconst().sigma();     }
    virtual double eps(void)       const { return getQconst().eps();       }
    virtual double Cclass(int d)   const { return getQconst().Cclass(d);   }
    virtual double epsclass(int d) const { return getQconst().epsclass(d); }

    virtual int    memsize(void)      const { return getQconst().memsize();      }
    virtual double zerotol(void)      const { return getQconst().zerotol();      }
    virtual double Opttol(void)       const { return getQconst().Opttol();       }
    virtual int    maxitcnt(void)     const { return getQconst().maxitcnt();     }
    virtual double maxtraintime(void) const { return getQconst().maxtraintime(); }

    virtual int    maxitermvrank(void) const { return getQconst().maxitermvrank(); }
    virtual double lrmvrank(void)      const { return getQconst().lrmvrank();      }
    virtual double ztmvrank(void)      const { return getQconst().ztmvrank();      }

    virtual double betarank(void) const { return getQconst().betarank(); }

    virtual double sparlvl(void) const { return getQconst().sparlvl(); }

    virtual const Vector<SparseVector<gentype> > &x          (void) const { return getQconst().x();           }
    virtual const Vector<gentype>                &y          (void) const { return getQconst().y();           }
    virtual const Vector<vecInfo>                &xinfo      (void) const { return getQconst().xinfo();       }
    virtual const Vector<int>                    &xtang      (void) const { return getQconst().xtang();       }
    virtual const Vector<int>                    &d          (void) const { return getQconst().d();           }
    virtual const Vector<double>                 &Cweight    (void) const { return getQconst().Cweight();     }
    virtual const Vector<double>                 &Cweightfuzz(void) const { return getQconst().Cweightfuzz(); }
    virtual const Vector<double>                 &sigmaweight(void) const { return getQconst().sigmaweight(); }
    virtual const Vector<double>                 &epsweight  (void) const { return getQconst().epsweight();   }
    virtual const Vector<int>                    &alphaState (void) const { return getQconst().alphaState();  }

    virtual int isClassifier(void) const { return getQconst().isClassifier(); }
    virtual int isRegression(void) const { return getQconst().isRegression(); }

    // Version numbers

    virtual int xvernum(void)        const { return getQconst().xvernum();        }
    virtual int xvernum(int altMLid) const { return getQconst().xvernum(altMLid); }
    virtual int incxvernum(void)           { return getQ().incxvernum();          }
    virtual int gvernum(void)        const { return getQconst().gvernum();        }
    virtual int gvernum(int altMLid) const { return getQconst().gvernum(altMLid); }
    virtual int incgvernum(void)           { return getQ().incgvernum();          }

    virtual int MLid(void) const { return getQconst().MLid(); }
    virtual int setMLid(int nv) { return getQ().setMLid(nv); }
    virtual int getaltML(kernPrecursor *&res, int altMLid) const { return getQconst().getaltML(res,altMLid); }

    // Kernel Modification

    virtual const MercerKernel &getKernel(void) const                                           { return getKnumMLconst().getKernel();                                                          }
    virtual MercerKernel &getKernel_unsafe(void)                                                { return getKnumML().getKernel_unsafe();                                                        }
    virtual void prepareKernel(void)                                                            {        getKnumML().prepareKernel(); return;                                                   }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1)        { int res = getKnumML().resetKernel(modind,onlyChangeRowI,updateInfo); fixMLTree(); return res; }
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) { int res = getKnumML().setKernel(xkernel,modind,onlyChangeRowI);      fixMLTree(); return res; }

    virtual void fillCache(void) { getQ().fillCache(); return; }

    virtual void K2bypass(const Matrix<gentype> &nv) { getQ().K2bypass(nv); return; }

    gentype &Keqn(gentype &res,                           int resmode = 1) const { return getQconst().Keqn(res,     resmode); }
    gentype &Keqn(gentype &res, const MercerKernel &altK, int resmode = 1) const { return getQconst().Keqn(res,altK,resmode); }

    virtual gentype &K1(gentype &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { return getQconst().K1(res,xa,xainf); }
    virtual gentype &K2(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { return getQconst().K2(res,xa,xb,xainf,xbinf); }
    virtual gentype &K3(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL) const { return getQconst().K3(res,xa,xb,xc,xainf,xbinf,xcinf); }
    virtual gentype &K4(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL, const vecInfo *xdinf = NULL) const { return getQconst().K4(res,xa,xb,xc,xd,xainf,xbinf,xcinf,xdinf); }
    virtual gentype &Km(gentype &res, const Vector<SparseVector<gentype> > &xx) const { return getQconst().Km(res,xx); }

    virtual double &K2ip(double &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { return getQconst().K2ip(res,xa,xb,xainf,xbinf); }
    virtual double distK(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { return getQconst().distK(xa,xb,xainf,xbinf); }

    virtual Vector<gentype> &phi2(Vector<gentype> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { return getQconst().phi2(res,xa,xainf); }
    virtual Vector<gentype> &phi2(Vector<gentype> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const { return getQconst().phi2(res,ia,xa,xainf); }

    virtual Vector<double> &phi2(Vector<double> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { return getQconst().phi2(res,xa,xainf); }
    virtual Vector<double> &phi2(Vector<double> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const { return getQconst().phi2(res,ia,xa,xainf); }

    virtual double &K0ip(       double &res, const gentype **pxyprod = NULL) const { return getQconst().K0ip(res,pxyprod); }
    virtual double &K1ip(       double &res, int i, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const vecInfo *xxinfo = NULL) const { return  getQconst().K1ip(res,i,pxyprod,xx,xxinfo); }
    virtual double &K2ip(       double &res, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { return  getQconst().K2ip(res,i,j,pxyprod,xx,yy,xxinfo,yyinfo); }
    virtual double &K3ip(       double &res, int ia, int ib, int ic, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL) const { return getQconst().K3ip(res,ia,ib,ic,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double &K4ip(       double &res, int ia, int ib, int ic, int id, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL) const { return getQconst().K4ip(res,ia,ib,ic,id,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double &Kmip(int m, double &res, Vector<int> &i, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL) const { return getQconst().Kmip(m,res,i,pxyprod,xx,xxinfo); }

    virtual gentype        &K0(              gentype        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(         res     ,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const gentype &bias     , const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(         res,bias,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const MercerKernel &altK, const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(         res,altK,pxyprod,resmode); }
    virtual double         &K0(              double         &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(         res     ,pxyprod,resmode); }
    virtual Matrix<double> &K0(int spaceDim, Matrix<double> &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(spaceDim,res     ,pxyprod,resmode); }
    virtual d_anion        &K0(int order,    d_anion        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(order   ,res     ,pxyprod,resmode); }

    virtual gentype        &K1(              gentype        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(         res,ia,bias,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(         res,ia,altK,pxyprod,xa,xainfo,resmode); }
    virtual double         &K1(              double         &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual Matrix<double> &K1(int spaceDim, Matrix<double> &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(spaceDim,res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual d_anion        &K1(int order,    d_anion        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(order   ,res,ia     ,pxyprod,xa,xainfo,resmode); }

    virtual gentype        &K2(              gentype        &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const { return getQconst().K2(         res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const { return getQconst().K2(         res,ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const { return getQconst().K2(         res,ia,ib,altK,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual double         &K2(              double         &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const { return getQconst().K2(         res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual Matrix<double> &K2(int spaceDim, Matrix<double> &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const { return getQconst().K2(spaceDim,res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual d_anion        &K2(int order,    d_anion        &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const { return getQconst().K2(order,   res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }

    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(         res,ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(         res,ia,ib,ic,altK,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual double         &K3(              double         &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual Matrix<double> &K3(int spaceDim, Matrix<double> &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(spaceDim,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual d_anion        &K3(int order,    d_anion        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(order   ,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }

    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(         res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(         res,ia,ib,ic,id,altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual double         &K4(              double         &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual Matrix<double> &K4(int spaceDim, Matrix<double> &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(spaceDim,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual d_anion        &K4(int order,    d_anion        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(order   ,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }

    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m         ,res,i,pxyprod     ,xx,xxinfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const gentype &bias     , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m         ,res,i,bias,pxyprod,xx,xxinfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const MercerKernel &altK, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m         ,res,i,altK,pxyprod,xx,xxinfo,resmode); }
    virtual double         &Km(int m              , double         &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m         ,res,i,pxyprod     ,xx,xxinfo,resmode); }
    virtual Matrix<double> &Km(int m, int spaceDim, Matrix<double> &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m,spaceDim,res,i,pxyprod     ,xx,xxinfo,resmode); }
    virtual d_anion        &Km(int m, int order   , d_anion        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m,order   ,res,i,pxyprod     ,xx,xxinfo,resmode); }

    virtual void dK(gentype &xygrad, gentype &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const { getQconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv); return; }
    virtual void dK(double  &xygrad, double  &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const { getQconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv); return; }

    virtual void d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void dK2delx(double  &xscaleres, double  &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual double distK(int i, int j) const { return getQconst().distK(i,j); }

    virtual void densedKdx(double &res, int i, int j) const { return getQconst().densedKdx(res,i,j); }
    virtual void denseintK(double &res, int i, int j) const { return getQconst().denseintK(res,i,j); }

    virtual void ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, int i, int j) const { getQconst().ddistKdx(xscaleres,yscaleres,minmaxind,i,j); return; }

    virtual int isKVarianceNZ(void) const { return getQconst().isKVarianceNZ(); }

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, int xdim, int densetype, int resmode, int mlid) const { getQconst().K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis,const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const { getQconst().K1xfer(res,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const { getQconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const { getQconst().K3xfer(res,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const { getQconst().K4xfer(res,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const { getQconst().Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, int xdim, int densetype, int resmode, int mlid) const { getQconst().K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const { getQconst().K1xfer(res,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const { getQconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const { getQconst().K3xfer(res,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const { getQconst().K4xfer(res,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const { getQconst().Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual const gentype &xelm(gentype &res, int i, int j) const { return getQconst().xelm(res,i,j); }
    virtual int xindsize(int i) const { return getQconst().xindsize(i); }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double sigmaweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double sigmaweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &sigmaweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &sigmaweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i)                                       { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);
    virtual int removeTrainingVector(int i, int num)                              { return ML_Base::removeTrainingVector(i,num); }

    virtual int setx(int                i, const SparseVector<gentype>          &x) { int res = getQ().setx(i,x); resetKernelTree(); return res; }
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) { int res = getQ().setx(i,x); resetKernelTree(); return res; }
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) { int res = getQ().setx(  x); resetKernelTree(); return res; }

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) { int res = getQ().qswapx(i,x,dontupdate); resetKernelTree(); return res; }
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) { int res = getQ().qswapx(i,x,dontupdate); resetKernelTree(); return res; }
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) { int res = getQ().qswapx(  x,dontupdate); resetKernelTree(); return res; }

    virtual int sety(int                i, const gentype         &y) { return getQ().sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) { return getQ().sety(i,y); }
    virtual int sety(                      const Vector<gentype> &y) { return getQ().sety(  y); }

    virtual int sety(int                i, double                z) { return getQ().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<double> &z) { return getQ().sety(i,z); }
    virtual int sety(                      const Vector<double> &z) { return getQ().sety(z); }

    virtual int sety(int                i, const Vector<double>          &z) { return getQ().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z) { return getQ().sety(i,z); }
    virtual int sety(                      const Vector<Vector<double> > &z) { return getQ().sety(z); }

    virtual int sety(int                i, const d_anion         &z) { return getQ().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &z) { return getQ().sety(i,z); }
    virtual int sety(                      const Vector<d_anion> &z) { return getQ().sety(z); }

    virtual int setd(int                i, int                nd);
    virtual int setd(const Vector<int> &i, const Vector<int> &nd);
    virtual int setd(                      const Vector<int> &nd);

    virtual int setCweight(int i,                double nv               ) { return getQ().setCweight(i,nv); }
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv) { return getQ().setCweight(i,nv); }
    virtual int setCweight(                      const Vector<double> &nv) { return getQ().setCweight(  nv); }

    virtual int setCweightfuzz(int i,                double nv               ) { return getQ().setCweightfuzz(i,nv); }
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv) { return getQ().setCweightfuzz(i,nv); }
    virtual int setCweightfuzz(                      const Vector<double> &nv) { return getQ().setCweightfuzz(  nv); }

    virtual int setsigmaweight(int i,                double nv               ) { return getQ().setsigmaweight(i,nv); }
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &nv) { return getQ().setsigmaweight(i,nv); }
    virtual int setsigmaweight(                      const Vector<double> &nv) { return getQ().setsigmaweight(  nv); }

    virtual int setepsweight(int i,                double nv               ) { return getQ().setepsweight(i,nv); }
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &nv) { return getQ().setepsweight(i,nv); }
    virtual int setepsweight(                      const Vector<double> &nv) { return getQ().setepsweight(  nv); }

    virtual int scaleCweight    (double s) { return getQ().scaleCweight(s);     }
    virtual int scaleCweightfuzz(double s) { return getQ().scaleCweightfuzz(s); }
    virtual int scalesigmaweight(double s) { return getQ().scalesigmaweight(s); }
    virtual int scaleepsweight  (double s) { return getQ().scaleepsweight(s);   }

    virtual void assumeConsistentX  (void) { getQ().assumeConsistentX();   return; }
    virtual void assumeInconsistentX(void) { getQ().assumeInconsistentX(); return; }

    virtual int isXConsistent(void)        const { return getQconst().isXConsistent();        }
    virtual int isXAssumedConsistent(void) const { return getQconst().isXAssumedConsistent(); }

    virtual void xferx(const ML_Base &xsrc) { getQ().xferx(xsrc); resetKernelTree(); return; }

    virtual const vecInfo &xinfo(int i)                       const { return getQconst().xinfo(i); }
    virtual int xtang(int i)                                  const { return getQconst().xtang(i); }
    virtual const SparseVector<gentype> &x(int i)             const { return getQconst().x(i);}
    virtual int xisrank(int i)                                const { return getQconst().xisrank(i); }
    virtual int xisgrad(int i)                                const { return getQconst().xisgrad(i); }
    virtual int xisrankorgrad(int i)                          const { return getQconst().xisrankorgrad(i); }
    virtual int xisclass(int i, int defaultclass, int q = -1) const { return getQconst().xisclass(i,defaultclass,q); }
    virtual const gentype &y(int i)                           const { return getQconst().y(i); }

    // Basis stuff

    virtual int NbasisUU(void)    const { return getQconst().NbasisUU();    }
    virtual int basisTypeUU(void) const { return getQconst().basisTypeUU(); }
    virtual int defProjUU(void)   const { return getQconst().defProjUU();   }

    virtual const Vector<gentype> &VbasisUU(void) const { return getQconst().VbasisUU(); }

    virtual int setBasisYUU(void)                     { return getQ().setBasisYUU();             }
    virtual int setBasisUUU(void)                     { return getQ().setBasisUUU();             }
    virtual int addToBasisUU(int i, const gentype &o) { return getQ().addToBasisUU(i,o);         }
    virtual int removeFromBasisUU(int i)              { return getQ().removeFromBasisUU(i);      }
    virtual int setBasisUU(int i, const gentype &o)   { return getQ().setBasisUU(i,o);           }
    virtual int setBasisUU(const Vector<gentype> &o)  { return getQ().setBasisUU(o);             }
    virtual int setDefaultProjectionUU(int d)         { return getQ().setDefaultProjectionUU(d); }
    virtual int setBasisUU(int n, int d)              { return getQ().setBasisUU(n,d);           }

    virtual int NbasisVV(void)    const { return getQconst().NbasisVV();    }
    virtual int basisTypeVV(void) const { return getQconst().basisTypeVV(); }
    virtual int defProjVV(void)   const { return getQconst().defProjVV();   }

    virtual const Vector<gentype> &VbasisVV(void) const { return getQconst().VbasisVV(); }

    virtual int setBasisYVV(void)                     { return getQ().setBasisYVV();             }
    virtual int setBasisUVV(void)                     { return getQ().setBasisUVV();             }
    virtual int addToBasisVV(int i, const gentype &o) { return getQ().addToBasisVV(i,o);         }
    virtual int removeFromBasisVV(int i)              { return getQ().removeFromBasisVV(i);      }
    virtual int setBasisVV(int i, const gentype &o)   { return getQ().setBasisVV(i,o);           }
    virtual int setBasisVV(const Vector<gentype> &o)  { return getQ().setBasisVV(o);             }
    virtual int setDefaultProjectionVV(int d)         { return getQ().setDefaultProjectionVV(d); }
    virtual int setBasisVV(int n, int d)              { return getQ().setBasisVV(n,d);           }

    virtual const MercerKernel &getUUOutputKernel(void) const                  { return getQconst().getUUOutputKernel();          }
    virtual MercerKernel &getUUOutputKernel_unsafe(void)                       { return getQ().getUUOutputKernel_unsafe();        }
    virtual int resetUUOutputKernel(int modind = 1)                            { return getQ().resetUUOutputKernel(modind);       }
    virtual int setUUOutputKernel(const MercerKernel &xkernel, int modind = 1) { return getQ().setUUOutputKernel(xkernel,modind); }

    // General modification and autoset functions

    virtual int randomise(double sparsity);
    virtual int autoen(void)      { return getQ().autoen();        }
    virtual int renormalise(void) { return ML_Base::renormalise(); }
    virtual int realign(void);

    virtual int setzerotol(double zt)                 { return getQ().setzerotol(zt);                 }
    virtual int setOpttol(double xopttol)             { return getQ().setOpttol(xopttol);             }
    virtual int setmaxitcnt(int xmaxitcnt)            { return getQ().setmaxitcnt(xmaxitcnt);         }
    virtual int setmaxtraintime(double xmaxtraintime) { return getQ().setmaxtraintime(xmaxtraintime); }

    virtual int setmaxitermvrank(int nv) { return getQ().setmaxitermvrank(nv); }
    virtual int setlrmvrank(double nv)   { return getQ().setlrmvrank(nv);      }
    virtual int setztmvrank(double nv)   { return getQ().setztmvrank(nv);      }

    virtual int setbetarank(double nv) { return getQ().setbetarank(nv); }

    virtual int setC       (       double xC) { return getQ().setC(xC);          }
    virtual int setsigma   (       double xC) { return getQ().setsigma(xC);      }
    virtual int seteps     (       double xC) { return getQ().seteps(xC);        }
    virtual int setCclass  (int d, double xC) { return getQ().setCclass(d,xC);   }
    virtual int setepsclass(int d, double xC) { return getQ().setepsclass(d,xC); }

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void);
    virtual int home(void);

    virtual ML_Base &operator*=(double sf) { scale(sf); return *this; }

    virtual int settspaceDim(int newdim) { return getQ().settspaceDim(newdim); }
    virtual int addtspaceFeat(int i)     { return getQ().addtspaceFeat(i);     }
    virtual int removetspaceFeat(int i)  { return getQ().removetspaceFeat(i);  }
    virtual int addxspaceFeat(int i)     { int res = getQ().addxspaceFeat(i);    resetKernelTree(); return res; }
    virtual int removexspaceFeat(int i)  { int res = getQ().removexspaceFeat(i); resetKernelTree(); return res; }

    virtual int setsubtype(int i) { return getQ().setsubtype(i); }

    virtual int setorder(int neword)                 { return getQ().setorder(neword);        }
    virtual int addclass(int label, int epszero = 0) { return getQ().addclass(label,epszero); }

    // Sampling mode

    virtual int isSampleMode(void) const { return getQconst().isSampleMode(); }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp = DEFAULT_SAMPLES_SAMPLE) { return getQ().setSampleMode(nv,xmin,xmax,Nsamp); }

    // Training functions:

    virtual void fudgeOn(void)  { getQ().fudgeOn();  return; }
    virtual void fudgeOff(void) { getQ().fudgeOff(); return; }

    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) { (void) res; killSwitch = 0; return 0; }

    // Evaluation Functions:

    virtual int ggTrainingVector(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return getQconst().ggTrainingVector(     resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = NULL) const { return getQconst().hhTrainingVector(resh,     i,        pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return getQconst().ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = NULL, gentype ***pxyprodj = NULL, gentype **pxyprodij = NULL) const { return getQconst().covTrainingVector(resv,resmu,i,j,pxyprodi,pxyprodj,pxyprodij); }

    virtual void dgTrainingVector(Vector<gentype> &res, int i) const { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>  &res, int i) const { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const { getQconst().dgTrainingVector(res,resn,i); return; }

    virtual int ggTrainingVector(double &resg,         int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(d_anion &resg,        int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }

    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const { return getQconst().dgTrainingVector(res,resn,i); }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const { return getQconst().dgTrainingVector(res,resn,i); }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const { return getQconst().dgTrainingVector(res,resn,i); }

    virtual void stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const { getQconst().stabProbTrainingVector(res,i,p,pnrm,rot,mu,B); return; }


    virtual int gg(               gentype &resg, const SparseVector<gentype> &x                 , const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().gg(     resg,x        ,xinf,pxyprodx); }
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x                 , const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().hh(resh,     x        ,xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().gh(resh,resg,x,retaltg,xinf,pxyprodx); }

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, gentype ***pxyprodx = NULL, gentype ***pxyprody = NULL, gentype **pxyprodij = NULL) const { return getQconst().cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodx,pxyprody,pxyprodij); }

    virtual void dg(Vector<gentype> &res, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dg(res,x,xinf); return; }
    virtual void dg(Vector<double>  &res, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dg(res,x,xinf); return; }
    virtual void dg(Vector<gentype> &res, gentype &resn, const gentype &y, const SparseVector<gentype> &x) const { getQconst().dg(res,resn,y,x); return; }

    virtual int gg(double &resg,         const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(Vector<double> &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(d_anion &resg,        const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }

    virtual void dg(Vector<double>          &res, double         &resn, const gentype &y, const SparseVector<gentype> &x) const { return getQconst().dg(res,resn,y,x); }
    virtual void dg(Vector<Vector<double> > &res, Vector<double> &resn, const gentype &y, const SparseVector<gentype> &x) const { return getQconst().dg(res,resn,y,x); }
    virtual void dg(Vector<d_anion>         &res, d_anion        &resn, const gentype &y, const SparseVector<gentype> &x) const { return getQconst().dg(res,resn,y,x); }

    virtual void stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const { getQconst().stabProb(res,x,p,pnrm,rot,mu,B); return; }

    // var and covar functions

    virtual int varTrainingVector(gentype &resv, gentype &resmu, int i, gentype ***pxyprodi = NULL, gentype **pxyprodii = NULL) const { return getQconst().varTrainingVector(resv,resmu,i,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL, gentype ***pxyprodx = NULL, gentype **pxyprodxx = NULL) const { return getQconst().var(resv,resmu,xa,xainf,pxyprodx,pxyprodxx); }

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const { return getQconst().covarTrainingVector(resv,i); }
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const { return getQconst().covar(resv,x); }

    // Training data tracking functions:

    virtual const Vector<int>          &indKey(void)          const { return getQconst().indKey();          }
    virtual const Vector<int>          &indKeyCount(void)     const { return getQconst().indKeyCount();     }
    virtual const Vector<int>          &dattypeKey(void)      const { return getQconst().dattypeKey();      }
    virtual const Vector<Vector<int> > &dattypeKeyBreak(void) const { return getQconst().dattypeKeyBreak(); }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) { getQ().setaltx(_altxsrc); resetKernelTree(); return; }

    virtual int disable(int i);
    virtual int disable(const Vector<int> &i);

    // Training data information functions (all assume no far/farfar/farfarfar or multivectors)

    virtual const SparseVector<gentype> &xsum      (SparseVector<gentype> &res) const { return getQconst().xsum(res);       }
    virtual const SparseVector<gentype> &xmean     (SparseVector<gentype> &res) const { return getQconst().xmean(res);      }
    virtual const SparseVector<gentype> &xmeansq   (SparseVector<gentype> &res) const { return getQconst().xmeansq(res);    }
    virtual const SparseVector<gentype> &xsqsum    (SparseVector<gentype> &res) const { return getQconst().xsqsum(res);     }
    virtual const SparseVector<gentype> &xsqmean   (SparseVector<gentype> &res) const { return getQconst().xsqmean(res);    }
    virtual const SparseVector<gentype> &xmedian   (SparseVector<gentype> &res) const { return getQconst().xmedian(res);    }
    virtual const SparseVector<gentype> &xvar      (SparseVector<gentype> &res) const { return getQconst().xvar(res);       }
    virtual const SparseVector<gentype> &xstddev   (SparseVector<gentype> &res) const { return getQconst().xstddev(res);    }
    virtual const SparseVector<gentype> &xmax      (SparseVector<gentype> &res) const { return getQconst().xmax(res);       }
    virtual const SparseVector<gentype> &xmin      (SparseVector<gentype> &res) const { return getQconst().xmin(res);       }

    // Kernel normalisation function

    virtual int normKernelZeroMeanUnitVariance  (int flatnorm = 0, int noshift = 0) { return getQ().normKernelZeroMeanUnitVariance(flatnorm,noshift);   }
    virtual int normKernelZeroMedianUnitVariance(int flatnorm = 0, int noshift = 0) { return getQ().normKernelZeroMedianUnitVariance(flatnorm,noshift); }
    virtual int normKernelUnitRange             (int flatnorm = 0, int noshift = 0) { return getQ().normKernelUnitRange(flatnorm,noshift);              }

    // Helper functions for sparse variables

    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<gentype>      &src) const { return getQconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<double>       &src) const { return getQconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const SparseVector<double> &src) const { return getQconst().xlateToSparse(dest,src); }

    virtual Vector<gentype> &xlateFromSparse(Vector<gentype> &dest, const SparseVector<gentype> &src) const { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<gentype> &src) const { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<double>  &src) const { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<gentype>       &src) const { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<double>        &src) const { return getQconst().xlateFromSparse(dest,src); }

    virtual Vector<double>  &xlateFromSparseTrainingVector(Vector<double>  &dest, int i) const { return getQconst().xlateFromSparseTrainingVector(dest,i); }
    virtual Vector<gentype> &xlateFromSparseTrainingVector(Vector<gentype> &dest, int i) const { return getQconst().xlateFromSparseTrainingVector(dest,i); }

    virtual SparseVector<gentype> &makeFullSparse(SparseVector<gentype> &dest) const { return getQconst().makeFullSparse(dest); }

    // x detangling

    virtual int detangle_x(int i, int usextang = 0) const
    {
        return getQconst().detangle_x(i,usextang);
    }

    virtual int detangle_x(SparseVector<gentype> *&xuntang, vecInfo *&xinfountang,
                   const SparseVector<gentype> *&xnear, const SparseVector<gentype> *&xfar, const SparseVector<gentype> *&xfarfar, 
                   const vecInfo *&xnearinfo, const vecInfo *&xfarinfo, 
                   int &inear, int &ifar, const gentype *&ineartup, const gentype *&ifartup,
                   int &ilr, int &irr, int &igr, 
                   int &iokr, int &iok,
                   int i, int &idiagr, const SparseVector<gentype> *xx, const vecInfo *xxinfo, int &gradOrder, 
                   int &iplanr, int &iplan, int &iset, int usextang = 1, int allocxuntangifneeded = 1) const
    {
        return getQconst().detangle_x(xuntang,xinfountang,xnear,xfar,xfarfar,xnearinfo,xfarinfo,inear,ifar,ineartup,ifartup,ilr,irr,igr,iokr,iok,i,idiagr,xx,xxinfo,gradOrder,iplanr,iplan,iset,usextang,allocxuntangifneeded);
    }







    // ================================================================
    //     Common functions for all MLMs
    // ================================================================

    virtual       MLM_Generic &getMLM     (void)       { return *this; }
    virtual const MLM_Generic &getMLMconst(void) const { return *this; }

    // Back-propogation control
    //
    // tsize: size (depth) of kernel network (not including output layer)
    // knum:  sets which layer does get/set/reset/prepareKernel return (-1 to tsize()-1).
    //        at -1 you should only set type of inheritance
    //
    // regtype:  regularisation type. 1 for 1-norm, 2 for 2-norm
    // regC:     "C" value for layer knum (C for output layer can also be set using setC)
    // mlmlr:    learning rate
    // diffstop: stop when average decrease in total training error for a step is less than this
    // lsparse:  sparsity for initial random layer initialisation (1 means dense, 0 means all zero)

    virtual int tsize(void) const { return mltree.size(); }
    virtual int knum(void)  const { return xknum;         }

    virtual int    regtype(int l) const { NiceAssert( ( l >= 0 ) && ( l < tsize() ) ); return xregtype(l); }
    virtual double regC(int l)    const { NiceAssert( ( l >= 0 ) && ( l < tsize() ) ); return getKnumMLconst(l).C(); }
    virtual double mlmlr(void)    const { return xmlmlr; }
    virtual double diffstop(void) const { return xdiffstop; }
    virtual double lsparse(void)  const { return xlsparse; }

    virtual const Matrix<double> &GGp(int l) const { return (dynamic_cast<const SVM_Generic &>(getKnumMLconst(l))).Gp(); }

    virtual int settsize(int nv) { NiceAssert( ( nv >= 1 ) ); int ov = tsize(); xistrained = 0; mltree.resize(nv); xregtype.resize(nv); if ( ov > nv ) { retVector<int> tmpva; xregtype("&",ov,1,nv-1,tmpva) = DEFAULT_REGTYPE; } fixMLTree(); return 1; }
    virtual int setknum(int nv)  { NiceAssert( ( nv >= -1 ) && ( nv < tsize() ) ); xknum = nv; return 0; }

    virtual int setregtype(int l, int nv) { NiceAssert( ( l >= 0 ) && ( l < tsize() ) ); NiceAssert( ( nv == 1 ) || ( nv == 2 ) ); xistrained = 0; xregtype("&",l) = nv; return 1; }
    virtual int setregC(int l, double nv) { NiceAssert( ( l >= 0 ) && ( l < tsize() ) ); NiceAssert( nv > 0 ); xistrained = 0; return getKnumML(l).setC(nv); }
    virtual int setmlmlr(double nv)       { NiceAssert( nv > 0.0 ); xmlmlr = nv; return 0; }
    virtual int setdiffstop(double nv)    { NiceAssert( nv > 0.0 ); xdiffstop = nv; return 0; }
    virtual int setlsparse(double nv)     { NiceAssert( nv >= 0.0 ); xlsparse = nv; return 0; }

    // Base-level stuff
    //
    // This is overloaded by children to return correct Q type

    virtual       SVM_Generic &getQ(void)            { return QQ; }
    virtual const SVM_Generic &getQconst(void) const { return QQ; }

protected:

    SVM_Generic QQ;

    int xistrained;
    int xknum; // which kernel is referenced by get/set/resetKernel functions (-1 for QQ, otherwise mltree element)
    Vector<int> xregtype;
    double xmlmlr;
    double xdiffstop;
    double xlsparse;

    Vector<SVM_Scalar> mltree;

    void fixMLTree(int modind = 0)
    {
        if ( tsize() )
        {
            int z = 0;
            int i;

            // Set alternative evaluation references at all levels.
            // Turn off cholesky factorisation for kernel tree.

            getQ().setaltx(&(mltree(z)));

            for ( i = 0 ; i < tsize() ; i++ )
            {
                if ( i )
                {
                    mltree("&",i).setaltx(&(mltree(z)));
                }

                mltree("&",i).setQuadraticCost();
                mltree("&",i).setOptSMO();
            }

            // Keep xy cache at base level

            mltree("&",z).getKernel_unsafe().setsuggestXYcache(1);
            mltree("&",z).resetKernel(modind);

            // Ensure tree makes sense

            if ( tsize() > 1 )
            {
                for ( i = 1 ; i < tsize() ; i++ )
                {
                    if ( (mltree(i).getKernel().cType(0)/100) != 8 )
                    {
                        mltree("&",i).getKernel_unsafe().add(0);
                    }

                    mltree("&",i).getKernel_unsafe().setType(802,0);
                    mltree("&",i).getKernel_unsafe().setChained(0);
                    mltree("&",i).getKernel_unsafe().setAltCall(mltree(i-1).MLid(),0);
                    mltree("&",i).resetKernel();
                }
            }

            if ( (getQ().getKernel().cType(0)/100) != 8 )
            {
                getQ().getKernel_unsafe().add(0);
            }

            getQ().getKernel_unsafe().setType(802,0);
            getQ().getKernel_unsafe().setChained(0);
            getQ().getKernel_unsafe().setAltCall(mltree(tsize()-1).MLid(),0);
            getQ().resetKernel();
        }

        else if ( (getQconst().getKernel().cType()/100) == 8 )
        {
            getQ().getKernel_unsafe().remove(0);
            getQ().resetKernel(modind);
        }

        return;
    }

    void resetKernelTree(int modind = 0)
    {
        if ( tsize() )
        {
            int ii;

            for ( ii = 0 ; ii < tsize() ; ii++ )
            {
                mltree("&",ii).resetKernel(modind && !ii);
            }
        }

        getQ().resetKernel(modind && !tsize());

        return;
    }

    ML_Base &getKnumML(int ovr = -2)
    {
        int i = ( ovr <= -2 ) ? xknum : ovr;

        if ( i == -1 )
        {
            return getQ();
        }

        return mltree("&",i);
    }

    const ML_Base &getKnumMLconst(int ovr = -2) const
    {
        int i = ( ovr <= -2 ) ? xknum : ovr;

        if ( i == -1 )
        {
            return getQconst();
        }

        return mltree(i);
    }

    MLM_Generic *thisthis;
    MLM_Generic **thisthisthis;
};

inline void qswap(MLM_Generic &a, MLM_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline MLM_Generic &setzero(MLM_Generic &a)
{
    a.restart();

    return a;
}

inline void MLM_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    MLM_Generic &b = dynamic_cast<MLM_Generic &>(bb.getML());

    ML_Base::qswapinternal(b);

    qswap(xregtype  ,b.xregtype  );
    qswap(xmlmlr    ,b.xmlmlr    );
    qswap(xdiffstop ,b.xdiffstop );
    qswap(xlsparse  ,b.xlsparse  );
    qswap(xknum     ,b.xknum     );
    qswap(xistrained,b.xistrained);
    qswap(mltree    ,b.mltree    );

    fixMLTree();

    return;
}

inline void MLM_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const MLM_Generic &b = dynamic_cast<const MLM_Generic &>(bb.getMLconst());

    ML_Base::semicopy(b);

    xistrained = b.xistrained;

    xregtype   = b.xregtype;
    xmlmlr     = b.xmlmlr;
    xdiffstop  = b.xdiffstop;
    xlsparse   = b.xlsparse;
    xknum      = b.xknum;
    //mltree     = b.mltree;

    return;
}

inline void MLM_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const MLM_Generic &src = dynamic_cast<const MLM_Generic &>(bb.getMLconst());

    ML_Base::assign(src,onlySemiCopy);

    xregtype   = src.xregtype;
    xmlmlr     = src.xmlmlr;
    xdiffstop  = src.xdiffstop;
    xlsparse   = src.xlsparse;
    xknum      = src.xknum;
    xistrained = src.xistrained;

    if ( !onlySemiCopy )
    {
        mltree = src.mltree;

        //fixMLTree() - need to do this (can't do it in constructor)
    }

    else
    {
        // Assume in virtual non-constructor!

        fixMLTree();

        int i;

        for ( i = 0 ; i < tsize() ; i++ )
        {
            mltree("&",i).assign(bb,onlySemiCopy);
        }
    }

    return;
}

#endif
