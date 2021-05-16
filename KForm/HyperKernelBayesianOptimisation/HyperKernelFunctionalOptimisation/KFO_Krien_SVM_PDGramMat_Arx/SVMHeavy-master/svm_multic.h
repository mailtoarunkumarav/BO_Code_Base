
//
// Multiclass classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_multic_h
#define _svm_multic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_generic.h"
#include "svm_multic_redbin.h"
#include "svm_multic_atonce.h"








class SVM_MultiC;


// Swap function

inline void qswap(SVM_MultiC &a, SVM_MultiC &b);


class SVM_MultiC : public SVM_Generic
{
public:

    // Constructors, destructors, assignment etc..

    SVM_MultiC();
    SVM_MultiC(const SVM_MultiC &src);
    SVM_MultiC(const SVM_MultiC &src, const ML_Base *xsrc);
    SVM_MultiC &operator=(const SVM_MultiC  &src) { assign(src); return *this; }
    virtual ~SVM_MultiC();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void setmemsize(int memsize) { Qatonce.setmemsize(memsize); Qredbin.setmemsize(memsize); return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);






    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    virtual       ML_Base &getML     (void)       { return static_cast<      ML_Base &>(getSVM());      }
    virtual const ML_Base &getMLconst(void) const { return static_cast<const ML_Base &>(getSVMconst()); }

    // Information functions (training data):

    virtual int N(void)       const { return locgetSVMconst().N  ();     }
    virtual int NNC(int d)    const { return locgetSVMconst().NNC(d);    }
    virtual int type(void)    const { return locgetSVMconst().type();    }
    virtual int subtype(void) const { return locgetSVMconst().subtype(); }

    virtual int tspaceDim(void)    const { return locgetSVMconst().tspaceDim();    }
    virtual int xspaceDim(void)    const { return locgetSVMconst().xspaceDim();    }
    virtual int fspaceDim(void)    const { return locgetSVMconst().fspaceDim();    }
    virtual int tspaceSparse(void) const { return locgetSVMconst().tspaceSparse(); }
    virtual int xspaceSparse(void) const { return locgetSVMconst().xspaceSparse(); }
    virtual int numClasses(void)   const { return locgetSVMconst().numClasses();   }
    virtual int order(void)        const { return locgetSVMconst().order();        }

    virtual int isTrained(void) const { return locgetSVMconst().isTrained(); }
    virtual int isMutable(void) const { return locgetSVMconst().isMutable(); }
    virtual int isPool   (void) const { return locgetSVMconst().isPool   (); }

    virtual char gOutType(void) const { return locgetSVMconst().gOutType(); }
    virtual char hOutType(void) const { return locgetSVMconst().hOutType(); }
    virtual char targType(void) const { return locgetSVMconst().targType(); }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const { return locgetSVMconst().calcDist(ha,hb,ia,db); }

    virtual int isUnderlyingScalar(void) const { return locgetSVMconst().isUnderlyingScalar(); }
    virtual int isUnderlyingVector(void) const { return locgetSVMconst().isUnderlyingVector(); }
    virtual int isUnderlyingAnions(void) const { return locgetSVMconst().isUnderlyingAnions(); }

    virtual const Vector<int> &ClassLabels(void)   const { return locgetSVMconst().ClassLabels();        }
    virtual int getInternalClass(const gentype &y) const { return locgetSVMconst().getInternalClass(y);  }
    virtual int numInternalClasses(void)           const { return locgetSVMconst().numInternalClasses(); }
    virtual int isenabled(int i)                   const { return locgetSVMconst().isenabled(i);         }

    virtual double C(void)         const { return locgetSVMconst().C();            }
    virtual double sigma(void)     const { return locgetSVMconst().sigma();        }
    virtual double eps(void)       const { return locgetSVMconst().eps();          }
    virtual double Cclass(int d)   const { return locgetSVMconst().Cclass(d);      }
    virtual double epsclass(int d) const { return locgetSVMconst().epsclass(d);    }

    virtual int    memsize(void)      const { return locgetSVMconst().memsize();      }
    virtual double zerotol(void)      const { return locgetSVMconst().zerotol();      }
    virtual double Opttol(void)       const { return locgetSVMconst().Opttol();       }
    virtual int    maxitcnt(void)     const { return locgetSVMconst().maxitcnt();     }
    virtual double maxtraintime(void) const { return locgetSVMconst().maxtraintime(); }

    virtual int    maxitermvrank(void) const { return locgetSVMconst().maxitermvrank(); }
    virtual double lrmvrank(void)      const { return locgetSVMconst().lrmvrank();      }
    virtual double ztmvrank(void)      const { return locgetSVMconst().ztmvrank();      }

    virtual double betarank(void) const { return locgetSVMconst().betarank(); }

    virtual double sparlvl(void) const { return locgetSVMconst().sparlvl(); }

    virtual const Vector<SparseVector<gentype> > &x          (void) const { return locgetSVMconst().x();           }
    virtual const Vector<gentype>                &y          (void) const { return locgetSVMconst().y();           }
    virtual const Vector<vecInfo>                &xinfo      (void) const { return locgetSVMconst().xinfo();       }
    virtual const Vector<int>                    &xtang      (void) const { return locgetSVMconst().xtang();       }
    virtual const Vector<int>                    &d          (void) const { return locgetSVMconst().d();           }
    virtual const Vector<double>                 &Cweight    (void) const { return locgetSVMconst().Cweight();     }
    virtual const Vector<double>                 &Cweightfuzz(void) const { return locgetSVMconst().Cweightfuzz(); }
    virtual const Vector<double>                 &sigmaweight(void) const { return locgetSVMconst().sigmaweight(); }
    virtual const Vector<double>                 &epsweight  (void) const { return locgetSVMconst().epsweight();   }
    virtual const Vector<int>                    &alphaState (void) const { return locgetSVMconst().alphaState();  }

    virtual int isClassifier(void) const { return locgetSVMconst().isClassifier(); }
    virtual int isRegression(void) const { return locgetSVMconst().isRegression(); }

    // Version numbers

    virtual int xvernum(void)        const { return locgetSVMconst().xvernum();        }
    virtual int xvernum(int altMLid) const { return locgetSVMconst().xvernum(altMLid); }
    virtual int incxvernum(void)           { return locgetSVM().incxvernum();          }
    virtual int gvernum(void)        const { return locgetSVMconst().gvernum();        }
    virtual int gvernum(int altMLid) const { return locgetSVMconst().gvernum(altMLid); }
    virtual int incgvernum(void)           { return locgetSVM().incgvernum();          }

    virtual int MLid(void) const { return locgetSVMconst().MLid(); }
    virtual int setMLid(int nv) { return locgetSVM().setMLid(nv); }
    virtual int getaltML(kernPrecursor *&res, int altMLid) const { return locgetSVMconst().getaltML(res,altMLid); }

    // Kernel Modification

    virtual const MercerKernel &getKernel(void) const                                           { return locgetSVMconst().getKernel();        }
    virtual MercerKernel &getKernel_unsafe(void)                                                { return locgetSVM().getKernel_unsafe();      }
    virtual void prepareKernel(void)                                                            {        locgetSVM().prepareKernel(); return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1)        { int res = Qatonce.resetKernel(modind,onlyChangeRowI,updateInfo); res |= Qredbin.resetKernel(modind,-1,updateInfo); return res; }
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) { int res = Qatonce.setKernel(xkernel,modind,onlyChangeRowI);      res |= Qredbin.setKernel(xkernel,modind);         return res; }

    virtual void fillCache(void) { locgetSVM().fillCache(); return; }

    virtual void K2bypass(const Matrix<gentype> &nv) { locgetSVM().K2bypass(nv); return; }

    gentype &Keqn(gentype &res,                           int resmode = 1) const { return locgetSVMconst().Keqn(res,     resmode); }
    gentype &Keqn(gentype &res, const MercerKernel &altK, int resmode = 1) const { return locgetSVMconst().Keqn(res,altK,resmode); }

    virtual gentype &K1(gentype &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { return locgetSVMconst().K1(res,xa,xainf); }
    virtual gentype &K2(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { return locgetSVMconst().K2(res,xa,xb,xainf,xbinf); }
    virtual gentype &K3(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL) const { return locgetSVMconst().K3(res,xa,xb,xc,xainf,xbinf,xcinf); }
    virtual gentype &K4(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL, const vecInfo *xdinf = NULL) const { return locgetSVMconst().K4(res,xa,xb,xc,xd,xainf,xbinf,xcinf,xdinf); }
    virtual gentype &Km(gentype &res, const Vector<SparseVector<gentype> > &xx) const { return locgetSVMconst().Km(res,xx); }

    virtual double &K2ip(double &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { return locgetSVMconst().K2ip(res,xa,xb,xainf,xbinf); }
    virtual double distK(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { return locgetSVMconst().distK(xa,xb,xainf,xbinf); }

    virtual Vector<gentype> &phi2(Vector<gentype> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { return locgetSVMconst().phi2(res,xa,xainf); }
    virtual Vector<gentype> &phi2(Vector<gentype> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const { return locgetSVMconst().phi2(res,ia,xa,xainf); }

    virtual Vector<double> &phi2(Vector<double> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { return locgetSVMconst().phi2(res,xa,xainf); }
    virtual Vector<double> &phi2(Vector<double> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const { return locgetSVMconst().phi2(res,ia,xa,xainf); }

    virtual double &K0ip(       double &res, const gentype **pxyprod = NULL) const { return locgetSVMconst().K0ip(res,pxyprod); }
    virtual double &K1ip(       double &res, int i, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const vecInfo *xxinfo = NULL) const { return  locgetSVMconst().K1ip(res,i,pxyprod,xx,xxinfo); }
    virtual double &K2ip(       double &res, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { return locgetSVMconst().K2ip(res,i,j,pxyprod,xx,yy,xxinfo,yyinfo); }
    virtual double &K3ip(       double &res, int ia, int ib, int ic, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL) const { return locgetSVMconst().K3ip(res,ia,ib,ic,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double &K4ip(       double &res, int ia, int ib, int ic, int id, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL) const { return locgetSVMconst().K4ip(res,ia,ib,ic,id,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double &Kmip(int m, double &res, Vector<int> &i, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL) const { return locgetSVMconst().Kmip(m,res,i,pxyprod,xx,xxinfo); }

    virtual gentype        &K0(              gentype        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return locgetSVMconst().K0(         res     ,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const gentype &bias     , const gentype **pxyprod = NULL, int resmode = 0) const { return locgetSVMconst().K0(         res,bias,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const MercerKernel &altK, const gentype **pxyprod = NULL, int resmode = 0) const { return locgetSVMconst().K0(         res,altK,pxyprod,resmode); }
    virtual double         &K0(              double         &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return locgetSVMconst().K0(         res     ,pxyprod,resmode); }
    virtual Matrix<double> &K0(int spaceDim, Matrix<double> &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return locgetSVMconst().K0(spaceDim,res     ,pxyprod,resmode); }
    virtual d_anion        &K0(int order,    d_anion        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return locgetSVMconst().K0(order   ,res     ,pxyprod,resmode); }

    virtual gentype        &K1(              gentype        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return locgetSVMconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return locgetSVMconst().K1(         res,ia,bias,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return locgetSVMconst().K1(         res,ia,altK,pxyprod,xa,xainfo,resmode); }
    virtual double         &K1(              double         &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return locgetSVMconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual Matrix<double> &K1(int spaceDim, Matrix<double> &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return locgetSVMconst().K1(spaceDim,res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual d_anion        &K1(int order,    d_anion        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return locgetSVMconst().K1(order   ,res,ia     ,pxyprod,xa,xainfo,resmode); }

    virtual gentype        &K2(              gentype        &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return locgetSVMconst().K2(         res,i,j     ,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int i, int j, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return locgetSVMconst().K2(         res,i,j,bias,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int i, int j, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return locgetSVMconst().K2(         res,i,j,altK,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }
    virtual double         &K2(              double         &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return locgetSVMconst().K2(         res,i,j     ,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }
    virtual Matrix<double> &K2(int spaceDim, Matrix<double> &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return locgetSVMconst().K2(spaceDim,res,i,j     ,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }
    virtual d_anion        &K2(int order,    d_anion        &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return locgetSVMconst().K2(order,   res,i,j     ,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }

    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return locgetSVMconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return locgetSVMconst().K3(         res,ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return locgetSVMconst().K3(         res,ia,ib,ic,altK,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual double         &K3(              double         &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return locgetSVMconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual Matrix<double> &K3(int spaceDim, Matrix<double> &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return locgetSVMconst().K3(spaceDim,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual d_anion        &K3(int order,    d_anion        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return locgetSVMconst().K3(order   ,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }

    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return locgetSVMconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return locgetSVMconst().K4(         res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return locgetSVMconst().K4(         res,ia,ib,ic,id,altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual double         &K4(              double         &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return locgetSVMconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual Matrix<double> &K4(int spaceDim, Matrix<double> &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return locgetSVMconst().K4(spaceDim,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual d_anion        &K4(int order,    d_anion        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return locgetSVMconst().K4(order   ,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }

    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return locgetSVMconst().Km(m         ,res,i     ,pxyprod,xx,xxinfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const gentype &bias     , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return locgetSVMconst().Km(m         ,res,i,bias,pxyprod,xx,xxinfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const MercerKernel &altK, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return locgetSVMconst().Km(m         ,res,i,altK,pxyprod,xx,xxinfo,resmode); }
    virtual double         &Km(int m              , double         &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return locgetSVMconst().Km(m         ,res,i     ,pxyprod,xx,xxinfo,resmode); }
    virtual Matrix<double> &Km(int m, int spaceDim, Matrix<double> &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return locgetSVMconst().Km(m,spaceDim,res,i     ,pxyprod,xx,xxinfo,resmode); }
    virtual d_anion        &Km(int m, int order   , d_anion        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return locgetSVMconst().Km(m,order   ,res,i     ,pxyprod,xx,xxinfo,resmode); }

    virtual void dK(gentype &xygrad, gentype &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const { locgetSVMconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv); return; }
    virtual void dK(double  &xygrad, double  &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const { locgetSVMconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv); return; }

    virtual void d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { locgetSVMconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { locgetSVMconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const  vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { locgetSVMconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void dK2delx(double  &xscaleres, double  &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { locgetSVMconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { locgetSVMconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { locgetSVMconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { locgetSVMconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { locgetSVMconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { locgetSVMconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { locgetSVMconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual double distK(int i, int j) const { return locgetSVMconst().distK(i,j); }

    virtual void densedKdx(double &res, int i, int j) const { return locgetSVMconst().densedKdx(res,i,j); }
    virtual void denseintK(double &res, int i, int j) const { return locgetSVMconst().denseintK(res,i,j); }

    virtual void ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, int i, int j) const { locgetSVMconst().ddistKdx(xscaleres,yscaleres,minmaxind,i,j); return; }

    virtual int isKVarianceNZ(void) const { return locgetSVMconst().isKVarianceNZ(); }

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, int xdim, int densetype, int resmode, int mlid) const { locgetSVMconst().K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const { locgetSVMconst().K1xfer(res,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const { locgetSVMconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const { locgetSVMconst().K3xfer(res,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const { locgetSVMconst().K4xfer(res,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const { locgetSVMconst().Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid); return;    }

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, int xdim, int densetype, int resmode, int mlid) const { locgetSVMconst().K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const { locgetSVMconst().K1xfer(res,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const { locgetSVMconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const { locgetSVMconst().K3xfer(res,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const { locgetSVMconst().K4xfer(res,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const { locgetSVMconst().Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual const gentype &xelm(gentype &res, int i, int j) const { return locgetSVMconst().xelm(res,i,j); }
    virtual int xindsize(int i) const { return locgetSVMconst().xindsize(i); }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweight = 1, double epsweight = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweight = 1, double epsweight = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweight, const Vector<double> &epsweight);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweight, const Vector<double> &epsweight);

    virtual int removeTrainingVector(int i)                                              { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) { return locgetSVM().removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, int num)                                     { return locgetSVM().removeTrainingVector(i,num); }

    virtual int setx(int i, const SparseVector<gentype> &x)                         { return locgetSVM().setx(i,x); }
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) { return locgetSVM().setx(i,x); }
    virtual int setx(const Vector<SparseVector<gentype> > &x)                       { return locgetSVM().setx(x);   }

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) { return locgetSVM().qswapx(i,x,dontupdate); }
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) { return locgetSVM().qswapx(i,x,dontupdate); }
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) { return locgetSVM().qswapx(  x,dontupdate); }

    virtual int sety(int i, const gentype &z)                        { return locgetSVM().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) { return locgetSVM().sety(i,z); }
    virtual int sety(const Vector<gentype> &z)                       { return locgetSVM().sety(z);   }

    virtual int sety(int                i, double                z) { return locgetSVM().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<double> &z) { return locgetSVM().sety(i,z); }
    virtual int sety(                      const Vector<double> &z) { return locgetSVM().sety(z); }

    virtual int sety(int                i, const Vector<double>          &z) { return locgetSVM().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z) { return locgetSVM().sety(i,z); }
    virtual int sety(                      const Vector<Vector<double> > &z) { return locgetSVM().sety(z); }

    virtual int sety(int                i, const d_anion         &z) { return locgetSVM().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &z) { return locgetSVM().sety(i,z); }
    virtual int sety(                      const Vector<d_anion> &z) { return locgetSVM().sety(z); }

    virtual int setd(int i, int d)                               { return locgetSVM().setd(i,d); }
    virtual int setd(const Vector<int> &i, const Vector<int> &d) { return locgetSVM().setd(i,d); }
    virtual int setd(const Vector<int> &d)                       { return locgetSVM().setd(d);   }

    virtual int setCweight(int i, double xCweight)                               { return locgetSVM().setCweight(i,xCweight); }
    virtual int setCweight(const Vector<int> &i, const Vector<double> &xCweight) { return locgetSVM().setCweight(i,xCweight); }
    virtual int setCweight(const Vector<double> &xCweight)                       { return locgetSVM().setCweight(xCweight);   }

    virtual int setCweightfuzz(int i, double nw)                               { return locgetSVM().setCweightfuzz(i,nw); }
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nw) { return locgetSVM().setCweightfuzz(i,nw); }
    virtual int setCweightfuzz(const Vector<double> &nw)                       { return locgetSVM().setCweightfuzz(nw);   }

    virtual int setsigmaweight(int i, double xCweight)                               { return locgetSVM().setsigmaweight(i,xCweight); }
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &xCweight) { return locgetSVM().setsigmaweight(i,xCweight); }
    virtual int setsigmaweight(const Vector<double> &xCweight)                       { return locgetSVM().setsigmaweight(xCweight);   }

    virtual int setepsweight(int i, double xepsweight)                               { return locgetSVM().setepsweight(i,xepsweight); }
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &xepsweight) { return locgetSVM().setepsweight(i,xepsweight); }
    virtual int setepsweight(const Vector<double> &xepsweight)                       { return locgetSVM().setepsweight(xepsweight);   }

    virtual int scaleCweight    (double scalefactor) { return locgetSVM().scaleCweight(scalefactor);     }
    virtual int scaleCweightfuzz(double scalefactor) { return locgetSVM().scaleCweightfuzz(scalefactor); }
    virtual int scalesigmaweight(double scalefactor) { return locgetSVM().scalesigmaweight(scalefactor); }
    virtual int scaleepsweight  (double scalefactor) { return locgetSVM().scaleepsweight(scalefactor);   }

    virtual void assumeConsistentX  (void) { locgetSVM().assumeConsistentX  (); return; }
    virtual void assumeInconsistentX(void) { locgetSVM().assumeInconsistentX(); return; }

    virtual int isXConsistent(void)        const { return locgetSVMconst().isXConsistent();        }
    virtual int isXAssumedConsistent(void) const { return locgetSVMconst().isXAssumedConsistent(); }

    virtual void xferx(const ML_Base &xsrc) { locgetSVM().xferx(xsrc); return; }

    virtual const vecInfo &xinfo(int i)            const { return locgetSVMconst().xinfo(i); }
    virtual int xtang(int i)                       const { return locgetSVMconst().xtang(i); }
    virtual const SparseVector<gentype> &x(int i)  const { return locgetSVMconst().x(i); }
    virtual int xisrank(int i)                     const { return locgetSVMconst().xisrank(i);  }
    virtual int xisgrad(int i)                     const { return locgetSVMconst().xisgrad(i);  }
    virtual int xisrankorgrad(int i)               const { return locgetSVMconst().xisrankorgrad(i);  }
    virtual int xisclass(int i, int d, int q = -1) const { return locgetSVMconst().xisclass(i,d,q); }
    virtual const gentype &y(int i)                const { return locgetSVMconst().y(i); }

    // Basis stuff

    virtual int NbasisUU(void)    const { return locgetSVMconst().NbasisUU();    }
    virtual int basisTypeUU(void) const { return locgetSVMconst().basisTypeUU(); }
    virtual int defProjUU(void)   const { return locgetSVMconst().defProjUU();   }

    virtual const Vector<gentype> &VbasisUU(void) const { return locgetSVMconst().VbasisUU(); }

    virtual int setBasisYUU(void)                     { return locgetSVM().setBasisYUU();             }
    virtual int setBasisUUU(void)                     { return locgetSVM().setBasisUUU();             }
    virtual int addToBasisUU(int i, const gentype &o) { return locgetSVM().addToBasisUU(i,o);         }
    virtual int removeFromBasisUU(int i)              { return locgetSVM().removeFromBasisUU(i);      }
    virtual int setBasisUU(int i, const gentype &o)   { return locgetSVM().setBasisUU(i,o);           }
    virtual int setBasisUU(const Vector<gentype> &o)  { return locgetSVM().setBasisUU(o);             }
    virtual int setDefaultProjectionUU(int d)         { return locgetSVM().setDefaultProjectionUU(d); }
    virtual int setBasisUU(int n, int d)              { return locgetSVM().setBasisUU(n,d);           }

    virtual int NbasisVV(void)    const { return locgetSVMconst().NbasisVV();    }
    virtual int basisTypeVV(void) const { return locgetSVMconst().basisTypeVV(); }
    virtual int defProjVV(void)   const { return locgetSVMconst().defProjVV();   }

    virtual const Vector<gentype> &VbasisVV(void) const { return locgetSVMconst().VbasisVV(); }

    virtual int setBasisYVV(void)                     { return locgetSVM().setBasisYVV();             }
    virtual int setBasisUVV(void)                     { return locgetSVM().setBasisUVV();             }
    virtual int addToBasisVV(int i, const gentype &o) { return locgetSVM().addToBasisVV(i,o);         }
    virtual int removeFromBasisVV(int i)              { return locgetSVM().removeFromBasisVV(i);      }
    virtual int setBasisVV(int i, const gentype &o)   { return locgetSVM().setBasisVV(i,o);           }
    virtual int setBasisVV(const Vector<gentype> &o)  { return locgetSVM().setBasisVV(o);             }
    virtual int setDefaultProjectionVV(int d)         { return locgetSVM().setDefaultProjectionVV(d); }
    virtual int setBasisVV(int n, int d)              { return locgetSVM().setBasisVV(n,d);           }

    virtual const MercerKernel &getUUOutputKernel(void) const                  { return locgetSVMconst().getUUOutputKernel();          }
    virtual MercerKernel &getUUOutputKernel_unsafe(void)                       { return locgetSVM().getUUOutputKernel_unsafe();        }
    virtual int resetUUOutputKernel(int modind = 1)                            { return locgetSVM().resetUUOutputKernel(modind);       }
    virtual int setUUOutputKernel(const MercerKernel &xkernel, int modind = 1) { return locgetSVM().setUUOutputKernel(xkernel,modind); }

    // General modification and autoset functions

    virtual int randomise(double sparsity) { return locgetSVM().randomise(sparsity); }
    virtual int autoen(void)               { return locgetSVM().autoen();            }
    virtual int renormalise(void)          { return locgetSVM().renormalise();       }
    virtual int realign(void)              { return locgetSVM().realign();           }

    virtual int setzerotol(double zt)                 { int res = Qatonce.setzerotol(zt);                 res |= Qredbin.setzerotol(zt);                 return res; }
    virtual int setOpttol(double xopttol)             { int res = Qatonce.setOpttol(xopttol);             res |= Qredbin.setOpttol(xopttol);             return res; }
    virtual int setmaxitcnt(int xmaxitcnt)            { int res = Qatonce.setmaxitcnt(xmaxitcnt);         res |= Qredbin.setmaxitcnt(xmaxitcnt);         return res; }
    virtual int setmaxtraintime(double xmaxtraintime) { int res = Qatonce.setmaxtraintime(xmaxtraintime); res |= Qredbin.setmaxtraintime(xmaxtraintime); return res; }

    virtual int setmaxitermvrank(int nv) { return locgetSVM().setmaxitermvrank(nv); }
    virtual int setlrmvrank(double nv)   { return locgetSVM().setlrmvrank(nv);      }
    virtual int setztmvrank(double nv)   { return locgetSVM().setztmvrank(nv);      }

    virtual int setbetarank(double nv) { return locgetSVM().setbetarank(nv); }

    virtual int setC(double xC)                 { int res = Qatonce.setC(xC);            res |= Qredbin.setC(xC);            return res; }
    virtual int setsigma(double xC)             { int res = Qatonce.setsigma(xC);        res |= Qredbin.setsigma(xC);        return res; }
    virtual int seteps(double xeps)             { int res = Qatonce.seteps(xeps);        res |= Qredbin.seteps(xeps);        return res; }
    virtual int setCclass(int d, double xC)     { int res = Qatonce.setCclass(d,xC);     res |= Qredbin.setCclass(d,xC);     return res; }
    virtual int setepsclass(int d, double xeps) { int res = Qatonce.setepsclass(d,xeps); res |= Qredbin.setepsclass(d,xeps); return res; }

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { SVM_MultiC temp; *this = temp; return 1; }
    virtual int home(void)    { return locgetSVM().home(); }

    virtual int settspaceDim(int newdim) { return locgetSVM().settspaceDim(newdim); }
    virtual int addtspaceFeat(int i)     { return locgetSVM().addtspaceFeat(i);     }
    virtual int removetspaceFeat(int i)  { return locgetSVM().removetspaceFeat(i);  }
    virtual int addxspaceFeat(int i)     { return locgetSVM().addxspaceFeat(i);     }
    virtual int removexspaceFeat(int i)  { return locgetSVM().removexspaceFeat(i);  }

    virtual int setsubtype(int i);

    virtual int setorder(int neword)                 { return locgetSVM().setorder(neword); }
    virtual int addclass(int label, int epszero = 0) { NiceAssert( !epszero || isatonce() ); int res = Qatonce.addclass(label,epszero); res |= Qredbin.addclass(label); return res; }

    // Training functions:

    virtual void fudgeOn(void)  { Qatonce.fudgeOn();  Qredbin.fudgeOn();  return; }
    virtual void fudgeOff(void) { Qatonce.fudgeOff(); Qredbin.fudgeOff(); return; }

    virtual int train(int &res) { return locgetSVM().train(res); }
    virtual int train(int &res, svmvolatile int &killSwitch) { return locgetSVM().train(res,killSwitch); }

    // Evaluation Functions:

    virtual int ggTrainingVector(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return locgetSVMconst().ggTrainingVector(     resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = NULL) const { return locgetSVMconst().hhTrainingVector(resh,     i,        pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return locgetSVMconst().ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = NULL, gentype ***pxyprodj = NULL, gentype **pxyprodij = NULL) const { return locgetSVMconst().covTrainingVector(resv,resmu,i,j,pxyprodi,pxyprodj,pxyprodij); }

    virtual void dgTrainingVector(Vector<gentype> &res, int i) const { locgetSVMconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>  &res, int i) const { locgetSVMconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const { locgetSVMconst().dgTrainingVector(res,resn,i); return; }

    virtual int ggTrainingVector(double &resg,         int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return locgetSVMconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return locgetSVMconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(d_anion &resg,        int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return locgetSVMconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }

//    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const { return getQconst().dgTrainingVector(res,resn,i); }
//    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const { return getQconst().dgTrainingVector(res,resn,i); }
//    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const { return getQconst().dgTrainingVector(res,resn,i); }

    virtual void stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const { locgetSVMconst().stabProbTrainingVector(res,i,p,pnrm,rot,mu,B); return; }


    virtual int gg(               gentype &resg, const SparseVector<gentype> &x                 , const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return locgetSVMconst().gg(     resg,x        ,xinf,pxyprodx); }
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x                 , const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return locgetSVMconst().hh(resh,     x        ,xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return locgetSVMconst().gh(resh,resg,x,retaltg,xinf,pxyprodx); }

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, gentype ***pxyprodx = NULL, gentype ***pxyprody = NULL, gentype **pxyprodij = NULL) const { return locgetSVMconst().cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodx,pxyprody,pxyprodij); }

    virtual void dg(Vector<gentype> &res, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { locgetSVMconst().dg(res,x,xinf); return; }
    virtual void dg(Vector<double>  &res, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { locgetSVMconst().dg(res,x,xinf); return; }
    virtual void dg(Vector<gentype> &res, gentype &resn, const gentype &y, const SparseVector<gentype> &x) const { locgetSVMconst().dg(res,resn,y,x); return; }

    virtual int gg(double &resg,         const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return locgetSVMconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(Vector<double> &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return locgetSVMconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(d_anion &resg,        const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return locgetSVMconst().gg(resg,x,retaltg,xinf,pxyprodx); }

//    virtual void dg(Vector<double>          &res, double         &resn, const gentype &y, const SparseVector<gentype> &x) const { return getQconst().dg(res,resn,y,x); }
//    virtual void dg(Vector<Vector<double> > &res, Vector<double> &resn, const gentype &y, const SparseVector<gentype> &x) const { return getQconst().dg(res,resn,y,x); }
//    virtual void dg(Vector<d_anion>         &res, d_anion        &resn, const gentype &y, const SparseVector<gentype> &x) const { return getQconst().dg(res,resn,y,x); }

    virtual void stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const { locgetSVMconst().stabProb(res,x,p,pnrm,rot,mu,B); return; }



    virtual int varTrainingVector(gentype &resv, gentype &resmu, int i, gentype ***pxyprodi = NULL, gentype **pxyprodii = NULL) const { return locgetSVMconst().varTrainingVector(resv,resmu,i,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL, gentype ***pxyprodx = NULL, gentype **pxyprodxx = NULL) const { return locgetSVMconst().var(resv,resmu,xa,xainf,pxyprodx,pxyprodxx); }

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const { return locgetSVMconst().covarTrainingVector(resv,i); }
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const { return locgetSVMconst().covar(resv,x); }

    // Training data tracking functions:

    virtual const Vector<int>          &indKey(void)          const { return locgetSVMconst().indKey();          }
    virtual const Vector<int>          &indKeyCount(void)     const { return locgetSVMconst().indKeyCount();     }
    virtual const Vector<int>          &dattypeKey(void)      const { return locgetSVMconst().dattypeKey();      }
    virtual const Vector<Vector<int> > &dattypeKeyBreak(void) const { return locgetSVMconst().dattypeKeyBreak(); }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) { Qatonce.setaltx(_altxsrc); Qredbin.setaltx(_altxsrc); return; }

    virtual int disable(int i)                { return locgetSVM().disable(i); }
    virtual int disable(const Vector<int> &i) { return locgetSVM().disable(i); }

    // Training data information functions (all assume no far/farfar/farfarfar or multivectors)

    virtual const SparseVector<gentype> &xsum      (SparseVector<gentype> &res) const { return locgetSVMconst().xsum(res);       }
    virtual const SparseVector<gentype> &xmean     (SparseVector<gentype> &res) const { return locgetSVMconst().xmean(res);      }
    virtual const SparseVector<gentype> &xmeansq   (SparseVector<gentype> &res) const { return locgetSVMconst().xmeansq(res);    }
    virtual const SparseVector<gentype> &xsqsum    (SparseVector<gentype> &res) const { return locgetSVMconst().xsqsum(res);     }
    virtual const SparseVector<gentype> &xsqmean   (SparseVector<gentype> &res) const { return locgetSVMconst().xsqmean(res);    }
    virtual const SparseVector<gentype> &xmedian   (SparseVector<gentype> &res) const { return locgetSVMconst().xmedian(res);    }
    virtual const SparseVector<gentype> &xvar      (SparseVector<gentype> &res) const { return locgetSVMconst().xvar(res);       }
    virtual const SparseVector<gentype> &xstddev   (SparseVector<gentype> &res) const { return locgetSVMconst().xstddev(res);    }
    virtual const SparseVector<gentype> &xmax      (SparseVector<gentype> &res) const { return locgetSVMconst().xmax(res);       }
    virtual const SparseVector<gentype> &xmin      (SparseVector<gentype> &res) const { return locgetSVMconst().xmin(res);       }

    // Kernel normalisation function

    virtual int normKernelZeroMeanUnitVariance  (int flatnorm = 0, int noshift = 0) { return locgetSVM().normKernelZeroMeanUnitVariance(flatnorm,noshift);   }
    virtual int normKernelZeroMedianUnitVariance(int flatnorm = 0, int noshift = 0) { return locgetSVM().normKernelZeroMedianUnitVariance(flatnorm,noshift); }
    virtual int normKernelUnitRange             (int flatnorm = 0, int noshift = 0) { return locgetSVM().normKernelUnitRange(flatnorm,noshift);              }

    // Helper functions for sparse variables

    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<gentype>      &src) const { return locgetSVMconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<double>       &src) const { return locgetSVMconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const SparseVector<double> &src) const { return locgetSVMconst().xlateToSparse(dest,src); }

    virtual Vector<gentype> &xlateFromSparse(Vector<gentype> &dest, const SparseVector<gentype> &src) const { return locgetSVMconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<gentype> &src) const { return locgetSVMconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<double>  &src) const { return locgetSVMconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<gentype>       &src) const { return locgetSVMconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<double>        &src) const { return locgetSVMconst().xlateFromSparse(dest,src); }

    virtual Vector<double>  &xlateFromSparseTrainingVector(Vector<double>  &dest, int i) const { return locgetSVMconst().xlateFromSparseTrainingVector(dest,i); }
    virtual Vector<gentype> &xlateFromSparseTrainingVector(Vector<gentype> &dest, int i) const { return locgetSVMconst().xlateFromSparseTrainingVector(dest,i); }

    virtual SparseVector<gentype> &makeFullSparse(SparseVector<gentype> &dest) const { return locgetSVMconst().makeFullSparse(dest); }

    // x detangling

    virtual int detangle_x(int i, int usextang = 0) const
    {
        return locgetSVMconst().detangle_x(i,usextang);
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
        return locgetSVMconst().detangle_x(xuntang,xinfountang,xnear,xfar,xfarfar,xnearinfo,xfarinfo,inear,ifar,ineartup,ifartup,ilr,irr,igr,iokr,iok,i,idiagr,xx,xxinfo,gradOrder,iplanr,iplan,iset,usextang,allocxuntangifneeded);
    }









    // ================================================================
    //     Common functions for all SVMs
    // ================================================================

    virtual void setsingmethod(int nv) { locgetSVM().setsingmethod(nv); return; }
    virtual void setRejectThreshold(double nv) { locgetSVM().setRejectThreshold(nv); return; }

    virtual int singmethod(void) const { return locgetSVMconst().singmethod(); }
    virtual double rejectThreshold(void) const { return locgetSVMconst().rejectThreshold(); }

    virtual       SVM_Generic &getSVM     (void)       { return *this; }
    virtual const SVM_Generic &getSVMconst(void) const { return *this; }

    // Constructors, destructors, assignment etc..

    virtual int setAlpha(const Vector<gentype> &newAlpha) { return locgetSVM().setAlpha(newAlpha); }
    virtual int setBias (const gentype         &newBias ) { return locgetSVM().setBias (newBias);  }

    virtual int setAlphaR(const Vector<double>          &newAlpha) { return locgetSVM().setAlphaR(newAlpha); }
    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha) { return locgetSVM().setAlphaV(newAlpha); }
    virtual int setAlphaA(const Vector<d_anion>         &newAlpha) { return locgetSVM().setAlphaA(newAlpha); }

    virtual int setBiasR(const double         &newBias) { return locgetSVM().setBiasR(newBias); }
    virtual int setBiasV(const Vector<double> &newBias) { return locgetSVM().setBiasV(newBias); }
    virtual int setBiasA(const d_anion        &newBias) { return locgetSVM().setBiasA(newBias); }

    // Information functions (training data):

    virtual int NZ (void)  const { return locgetSVMconst().NZ ();  }
    virtual int NF (void)  const { return locgetSVMconst().NF ();  }
    virtual int NS (void)  const { return locgetSVMconst().NS ();  }
    virtual int NC (void)  const { return locgetSVMconst().NC ();  }
    virtual int NLB(void)  const { return locgetSVMconst().NLB();  }
    virtual int NLF(void)  const { return locgetSVMconst().NLF();  }
    virtual int NUF(void)  const { return locgetSVMconst().NUF();  }
    virtual int NUB(void)  const { return locgetSVMconst().NUB();  }
    virtual int NF (int q) const { return locgetSVMconst().NF (q); }
    virtual int NZ (int q) const { return locgetSVMconst().NZ (q); }
    virtual int NS (int q) const { return locgetSVMconst().NS (q); }
    virtual int NC (int q) const { return locgetSVMconst().NC (q); }
    virtual int NLB(int q) const { return locgetSVMconst().NLB(q); }
    virtual int NLF(int q) const { return locgetSVMconst().NLF(q); }
    virtual int NUF(int q) const { return locgetSVMconst().NUF(q); }
    virtual int NUB(int q) const { return locgetSVMconst().NUB(q); }

    virtual const Vector<Vector<int> > &ClassRep(void)  const { return locgetSVMconst().ClassRep();    }
    virtual int                         findID(int ref) const { return locgetSVMconst().findID(ref);   }

    virtual int isLinearCost(void)    const { return locgetSVMconst().isLinearCost();    }
    virtual int isQuadraticCost(void) const { return locgetSVMconst().isQuadraticCost(); }
    virtual int is1NormCost(void)     const { return locgetSVMconst().is1NormCost();     }
    virtual int isVarBias(void)       const { return locgetSVMconst().isVarBias();       }
    virtual int isPosBias(void)       const { return locgetSVMconst().isPosBias();       }
    virtual int isNegBias(void)       const { return locgetSVMconst().isNegBias();       }
    virtual int isFixedBias(void)     const { return locgetSVMconst().isFixedBias();     }
    virtual int isVarBias(int dq)     const { return locgetSVMconst().isVarBias(dq);     }
    virtual int isPosBias(int dq)     const { return locgetSVMconst().isPosBias(dq);     }
    virtual int isNegBias(int dq)     const { return locgetSVMconst().isNegBias(dq);     }
    virtual int isFixedBias(int dq)   const { return locgetSVMconst().isFixedBias(dq);   }

    virtual int isNoMonotonicConstraints(void)    const { return locgetSVMconst().isNoMonotonicConstraints();    }
    virtual int isForcedMonotonicIncreasing(void) const { return locgetSVMconst().isForcedMonotonicIncreasing(); }
    virtual int isForcedMonotonicDecreasing(void) const { return locgetSVMconst().isForcedMonotonicDecreasing(); }

    virtual int isOptActive(void) const { return locgetSVMconst().isOptActive(); }
    virtual int isOptSMO(void)    const { return locgetSVMconst().isOptSMO();    }
    virtual int isOptD2C(void)    const { return locgetSVMconst().isOptD2C();    }
    virtual int isOptGrad(void)   const { return locgetSVMconst().isOptGrad();   }

    virtual int isFixedTube(void)  const { return locgetSVMconst().isFixedTube(); }
    virtual int isShrinkTube(void) const { return locgetSVMconst().isShrinkTube(); }

    virtual int isRestrictEpsPos(void) const { return locgetSVMconst().isRestrictEpsPos(); }
    virtual int isRestrictEpsNeg(void) const { return locgetSVMconst().isRestrictEpsNeg(); }

    virtual int isClassifyViaSVR(void) const { return locgetSVMconst().isClassifyViaSVR(); }
    virtual int isClassifyViaSVM(void) const { return locgetSVMconst().isClassifyViaSVM(); }

    virtual int is1vsA(void)    const { return locgetSVMconst().is1vsA();    }
    virtual int is1vs1(void)    const { return locgetSVMconst().is1vs1();    }
    virtual int isDAGSVM(void)  const { return locgetSVMconst().isDAGSVM();  }
    virtual int isMOC(void)     const { return locgetSVMconst().isMOC();     }
    virtual int ismaxwins(void) const { return locgetSVMconst().ismaxwins(); }
    virtual int isrecdiv(void)  const { return locgetSVMconst().isrecdiv();  }

    virtual int isatonce(void) const { return locgetSVMconst().isatonce(); }
    virtual int isredbin(void) const { return locgetSVMconst().isredbin(); }

    virtual int isKreal(void)   const { return locgetSVMconst().isKreal();   }
    virtual int isKunreal(void) const { return locgetSVMconst().isKunreal(); }

    virtual int isanomalyOn(void)  const { return locgetSVMconst().isanomalyOn();  }
    virtual int isanomalyOff(void) const { return locgetSVMconst().isanomalyOff(); }

    virtual int isautosetOff(void)          const { return locgetSVMconst().isautosetOff();          }
    virtual int isautosetCscaled(void)      const { return locgetSVMconst().isautosetCscaled();      }  
    virtual int isautosetCKmean(void)       const { return locgetSVMconst().isautosetCKmean();       }
    virtual int isautosetCKmedian(void)     const { return locgetSVMconst().isautosetCKmedian();     }
    virtual int isautosetCNKmean(void)      const { return locgetSVMconst().isautosetCNKmean();      }
    virtual int isautosetCNKmedian(void)    const { return locgetSVMconst().isautosetCNKmedian();    }
    virtual int isautosetLinBiasForce(void) const { return locgetSVMconst().isautosetLinBiasForce(); }

    virtual double outerlr(void)       const { return locgetSVMconst().outerlr();       }
    virtual double outermom(void)      const { return locgetSVMconst().outermom();      }
    virtual int    outermethod(void)   const { return locgetSVMconst().outermethod();   }
    virtual double outertol(void)      const { return locgetSVMconst().outertol();      }
    virtual double outerovsc(void)     const { return locgetSVMconst().outerovsc();     }
    virtual int    outermaxitcnt(void) const { return locgetSVMconst().outermaxitcnt(); }
    virtual int    outermaxcache(void) const { return locgetSVMconst().outermaxcache(); }

    virtual       int      maxiterfuzzt(void) const { return locgetSVMconst().maxiterfuzzt(); }
    virtual       int      usefuzzt(void)     const { return locgetSVMconst().usefuzzt();     }
    virtual       double   lrfuzzt(void)      const { return locgetSVMconst().lrfuzzt();      }
    virtual       double   ztfuzzt(void)      const { return locgetSVMconst().ztfuzzt();      }
    virtual const gentype &costfnfuzzt(void)  const { return locgetSVMconst().costfnfuzzt();  }

    virtual int m(void) const { return locgetSVMconst().m(); }

    virtual double LinBiasForce(void)    const { return locgetSVMconst().LinBiasForce();    }
    virtual double QuadBiasForce(void)   const { return locgetSVMconst().QuadBiasForce();   }
    virtual double LinBiasForce(int dq)  const { return locgetSVMconst().LinBiasForce(dq);  }
    virtual double QuadBiasForce(int dq) const { return locgetSVMconst().QuadBiasForce(dq); }

    virtual double nu(void)     const { return locgetSVMconst().nu();     }
    virtual double nuQuad(void) const { return locgetSVMconst().nuQuad(); }

    virtual double anomalyNu(void)    const { return locgetSVMconst().anomalyNu();    }
    virtual int    anomalyClass(void) const { return locgetSVMconst().anomalyClass(); }

    virtual double autosetCval(void)  const { return locgetSVMconst().autosetCval();  }
    virtual double autosetnuval(void) const { return locgetSVMconst().autosetnuval(); }

    virtual int anomclass(void) const { return locgetSVMconst().anomclass(); }

    virtual const Matrix<double>          &Gp         (void)        const { return locgetSVMconst().Gp();          }
    virtual const Matrix<double>          &XX         (void)        const { return locgetSVMconst().XX();          }
    virtual const Vector<double>          &kerndiag   (void)        const { return locgetSVMconst().kerndiag();    }
    virtual const Vector<Vector<double> > &getu       (void)        const { return locgetSVMconst().getu();        }
    virtual const gentype                 &bias       (void)        const { return locgetSVMconst().bias();        }
    virtual const Vector<gentype>         &alpha      (void)        const { return locgetSVMconst().alpha();       }
    virtual const Vector<double>          &zR         (void)        const { return locgetSVMconst().zR();          }
    virtual const Vector<Vector<double> > &zV         (int raw = 0) const { return locgetSVMconst().zV(raw);       }
    virtual const Vector<d_anion>         &zA         (void)        const { return locgetSVMconst().zA();          }
    virtual const double                  &biasR      (void)        const { return locgetSVMconst().biasR();       }
    virtual const Vector<double>          &biasV      (int raw = 0) const { return locgetSVMconst().biasV(raw);    }
    virtual const d_anion                 &biasA      (void)        const { return locgetSVMconst().biasA();       }
    virtual const Vector<double>          &alphaR     (void)        const { return locgetSVMconst().alphaR();      }
    virtual const Vector<Vector<double> > &alphaV     (int raw = 0) const { return locgetSVMconst().alphaV(raw);   }
    virtual const Vector<d_anion>         &alphaA     (void)        const { return locgetSVMconst().alphaA();      }

    // Training set modification:

    virtual int removeNonSupports(void)      { return locgetSVM().removeNonSupports();      }
    virtual int trimTrainingSet(int maxsize) { return locgetSVM().trimTrainingSet(maxsize); }

    // General modification and autoset functions

    virtual int setLinearCost(void)                        { int res = Qatonce.setLinearCost();    res |= Qredbin.setLinearCost();    return res; }
    virtual int setQuadraticCost(void)                     { int res = Qatonce.setQuadraticCost(); res |= Qredbin.setQuadraticCost(); return res; }
    virtual int set1NormCost(void)                         { int res = Qatonce.set1NormCost();     res |= Qredbin.set1NormCost();     return res; }
    virtual int setVarBias(void)                           { return locgetSVM().setVarBias();             }
    virtual int setPosBias(void)                           { return locgetSVM().setPosBias();             }
    virtual int setNegBias(void)                           { return locgetSVM().setNegBias();             }
    virtual int setFixedBias(double newbias = 0.0)         { return locgetSVM().setFixedBias(newbias);    }
    virtual int setVarBias(int dq)                         { return locgetSVM().setVarBias(dq);           }
    virtual int setPosBias(int dq)                         { return locgetSVM().setPosBias(dq);           }
    virtual int setNegBias(int dq)                         { return locgetSVM().setNegBias(dq);           }
    virtual int setFixedBias(int dq, double newbias = 0.0) { return locgetSVM().setFixedBias(dq,newbias); }
    virtual int setFixedBias(const gentype &newBias)       { return locgetSVM().setFixedBias(newBias);    }

    virtual int setNoMonotonicConstraints(void)    { return locgetSVM().setNoMonotonicConstraints();    }
    virtual int setForcedMonotonicIncreasing(void) { return locgetSVM().setForcedMonotonicIncreasing(); }
    virtual int setForcedMonotonicDecreasing(void) { return locgetSVM().setForcedMonotonicDecreasing(); }

    virtual int setOptActive(void) { int res = Qatonce.setOptActive(); res |= Qredbin.setOptActive(); return res; }
    virtual int setOptSMO(void)    { int res = Qatonce.setOptSMO();    res |= Qredbin.setOptSMO();    return res; }
    virtual int setOptD2C(void)    { int res = Qatonce.setOptD2C();    res |= Qredbin.setOptD2C();    return res; }
    virtual int setOptGrad(void)   { int res = Qatonce.setOptGrad();   res |= Qredbin.setOptGrad();   return res; }

    virtual int setFixedTube(void)  { return locgetSVM().setFixedTube();  }
    virtual int setShrinkTube(void) { return locgetSVM().setShrinkTube(); }

    virtual int setRestrictEpsPos(void) { return locgetSVM().setRestrictEpsPos(); }
    virtual int setRestrictEpsNeg(void) { return locgetSVM().setRestrictEpsNeg(); }

    virtual int setClassifyViaSVR(void) { return locgetSVM().setClassifyViaSVR(); }
    virtual int setClassifyViaSVM(void) { return locgetSVM().setClassifyViaSVM(); }

    virtual int set1vsA(void)    { int res = setredbin(); res |= Qredbin.set1vsA();    return res; }
    virtual int set1vs1(void)    { int res = setredbin(); res |= Qredbin.set1vs1();    return res; }
    virtual int setDAGSVM(void)  { int res = setredbin(); res |= Qredbin.setDAGSVM();  return res; }
    virtual int setMOC(void)     { int res = setredbin(); res |= Qredbin.setMOC();     return res; }
    virtual int setmaxwins(void) { int res = setatonce(); res |= Qatonce.setmaxwins(); return res; }
    virtual int setrecdiv(void)  { int res = setatonce(); res |= Qatonce.setrecdiv();  return res; }

    virtual int setatonce(void);
    virtual int setredbin(void);

    virtual int setKreal(void)   { int res = Qatonce.setKreal();   res |= Qredbin.setKreal();   return res; }
    virtual int setKunreal(void) { int res = Qatonce.setKunreal(); res |= Qredbin.setKunreal(); return res; }

    virtual int anomalyOn(int danomalyClass, double danomalyNu) { int res = Qatonce.anomalyOn(danomalyClass,danomalyNu); res |= Qredbin.anomalyOn(danomalyClass,danomalyNu); return res; }
    virtual int anomalyOff(void)                                { int res = Qatonce.anomalyOff();                        res |= Qredbin.anomalyOff();                        return res; }

    virtual int setouterlr(double xouterlr)           { int res = Qatonce.setouterlr(xouterlr);              res |= Qredbin.setouterlr(xouterlr);              return res; }
    virtual int setoutermom(double xoutermom)         { int res = Qatonce.setoutermom(xoutermom);            res |= Qredbin.setoutermom(xoutermom);            return res; }
    virtual int setoutermethod(int xoutermethod)      { int res = Qatonce.setoutermethod(xoutermethod);      res |= Qredbin.setoutermethod(xoutermethod);      return res; }
    virtual int setoutertol(double xoutertol)         { int res = Qatonce.setoutertol(xoutertol);            res |= Qredbin.setoutertol(xoutertol);            return res; }
    virtual int setouterovsc(double xouterovsc)       { int res = Qatonce.setouterovsc(xouterovsc);          res |= Qredbin.setouterovsc(xouterovsc);          return res; }
    virtual int setoutermaxitcnt(int xoutermaxits)    { int res = Qatonce.setoutermaxitcnt(xoutermaxits);    res |= Qredbin.setoutermaxitcnt(xoutermaxits);    return res; }
    virtual int setoutermaxcache(int xoutermaxcacheN) { int res = Qatonce.setoutermaxcache(xoutermaxcacheN); res |= Qredbin.setoutermaxcache(xoutermaxcacheN); return res; }

    virtual int setmaxiterfuzzt(int xmaxiterfuzzt)              { int res = Qatonce.setmaxiterfuzzt(xmaxiterfuzzt); res |= Qredbin.setmaxiterfuzzt(xmaxiterfuzzt); return res; }
    virtual int setusefuzzt(int xusefuzzt)                      { int res = Qatonce.setusefuzzt(xusefuzzt);         res |= Qredbin.setusefuzzt(xusefuzzt);         return res; }
    virtual int setlrfuzzt(double xlrfuzzt)                     { int res = Qatonce.setlrfuzzt(xlrfuzzt);           res |= Qredbin.setlrfuzzt(xlrfuzzt);           return res; }
    virtual int setztfuzzt(double xztfuzzt)                     { int res = Qatonce.setztfuzzt(xztfuzzt);           res |= Qredbin.setztfuzzt(xztfuzzt);           return res; }
    virtual int setcostfnfuzzt(const gentype &xcostfnfuzzt)     { int res = Qatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= Qredbin.setcostfnfuzzt(xcostfnfuzzt);   return res; }
    virtual int setcostfnfuzzt(const std::string &xcostfnfuzzt) { int res = Qatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= Qredbin.setcostfnfuzzt(xcostfnfuzzt);   return res; }

    virtual int setm(int xm) { return locgetSVM().setm(xm); }

    virtual int setLinBiasForce(double newval)          { int res = Qatonce.setLinBiasForce(newval);         res |= Qredbin.setLinBiasForce(newval);    return res; }
    virtual int setQuadBiasForce(double newval)         { int res = locgetSVM().setQuadBiasForce(newval);                                               return res; }
    virtual int setLinBiasForce(int dq, double newval)  { int res = Qatonce.setLinBiasForce(dq,newval);      res |= Qredbin.setLinBiasForce(dq,newval); return res; }
    virtual int setQuadBiasForce(int dq, double newval) { int res = locgetSVM().setQuadBiasForce(dq,newval);                                            return res; }

    virtual int setnu(double xnu)         { return locgetSVM().setnu(xnu);         }
    virtual int setnuQuad(double xnuQuad) { return locgetSVM().setnuQuad(xnuQuad); }

    virtual int autosetOff(void)                                     { int res = Qatonce.autosetOff();         res |= Qredbin.autosetOff();         return res; }
    virtual int autosetCscaled(double Cval)                          { int res = Qatonce.autosetCscaled(Cval); res |= Qredbin.autosetCscaled(Cval); return res; }
    virtual int autosetCKmean(void)                                  { int res = Qatonce.autosetCKmean();      res |= Qredbin.autosetCKmean();      return res; }
    virtual int autosetCKmedian(void)                                { int res = Qatonce.autosetCKmedian();    res |= Qredbin.autosetCKmedian();    return res; }
    virtual int autosetCNKmean(void)                                 { int res = Qatonce.autosetCNKmean();     res |= Qredbin.autosetCNKmean();     return res; }
    virtual int autosetCNKmedian(void)                               { int res = Qatonce.autosetCNKmedian();   res |= Qredbin.autosetCNKmedian();   return res; }
    virtual int autosetLinBiasForce(double nuval, double Cval = 1.0) { return locgetSVM().autosetLinBiasForce(nuval,Cval); }

    virtual void setanomalyclass(int n) { locgetSVM().setanomalyclass(n); return; }


    virtual double quasiloglikelihood(void) const { return locgetSVMconst().quasiloglikelihood(); }


protected:
    // ================================================================
    //     Base level functions
    // ================================================================

    // SVM specific

    virtual int addTrainingVector (int i, int d, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<int> &d,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);


private:

    virtual       SVM_Generic &locgetSVM     (void)       { if ( isQatonce ) { return static_cast<      SVM_Generic &>(Qatonce); } return static_cast<      SVM_Generic &>(Qredbin); }
    virtual const SVM_Generic &locgetSVMconst(void) const { if ( isQatonce ) { return static_cast<const SVM_Generic &>(Qatonce); } return static_cast<const SVM_Generic &>(Qredbin); }

    int isQatonce;

    SVM_MultiC_redbin Qredbin;
    SVM_MultiC_atonce Qatonce;
};

inline void qswap(SVM_MultiC &a, SVM_MultiC &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_MultiC::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_MultiC &b = dynamic_cast<SVM_MultiC &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(Qredbin  ,b.Qredbin  );
    qswap(Qatonce  ,b.Qatonce  );
    qswap(isQatonce,b.isQatonce);

    return;
}

inline void SVM_MultiC::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_MultiC &b = dynamic_cast<const SVM_MultiC &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

    if ( isQatonce ) { Qatonce.semicopy(b.Qatonce); }
    else             { Qredbin.semicopy(b.Qredbin); }

    return;
}

inline void SVM_MultiC::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_MultiC &src = dynamic_cast<const SVM_MultiC &>(bb.getMLconst());

    SVM_Generic::assign(src,onlySemiCopy);

    isQatonce = src.isQatonce;

    Qredbin.assign(src.Qredbin,onlySemiCopy);
    Qatonce.assign(src.Qatonce,onlySemiCopy);

    return;
}

#endif
