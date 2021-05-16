
//
// ML averaging block
//
// g(x) = mean(gi(x))
// gv(x) = mean(gv(x)) + var(gi(x))
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_conect_h
#define _blk_conect_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"
#include "blk_consen.h"
#include "svm_scalar.h"
#include "idstore.h"


// Defines a very basic set of blocks for use in machine learning.


class BLK_Conect;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_Conect &a, BLK_Conect &b);


class BLK_Conect : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Conect(int isIndPrune = 0);
    BLK_Conect(const BLK_Conect &src, int isIndPrune = 0);
    BLK_Conect(const BLK_Conect &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_Conect &operator=(const BLK_Conect &src) { assign(src); return *this; }
    virtual ~BLK_Conect();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void setmemsize(int memsize);

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions

    virtual int N(void)       const { return getRepConst().N();    }
    virtual int NNC(int d)    const { return getRepConst().NNC(d); }
    virtual int type(void)    const { return 212; }
    virtual int subtype(void) const { return 0;   }

    virtual int tspaceDim(void)    const { return getRepConst().tspaceDim();    }
    virtual int xspaceDim(void)    const { return getRepConst().xspaceDim();    }
    virtual int fspaceDim(void)    const { return getRepConst().fspaceDim();    }
    virtual int tspaceSparse(void) const { return getRepConst().tspaceSparse(); }
    virtual int xspaceSparse(void) const { return getRepConst().xspaceSparse(); }
    virtual int numClasses(void)   const { return getRepConst().numClasses();   }
    virtual int order(void)        const { return getRepConst().order();        }

    virtual int isTrained(void) const { return getRepConst().isTrained(); }
    virtual int isMutable(void) const { return getRepConst().isMutable(); }
    virtual int isPool   (void) const { return getRepConst().isPool();    }

    virtual char gOutType(void) const { return getRepConst().gOutType(); }
    virtual char hOutType(void) const { return getRepConst().hOutType(); }
    virtual char targType(void) const { return getRepConst().targType(); }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const { return getRepConst().calcDist(ha,hb,ia,db); }

    virtual int isUnderlyingScalar(void) const { return getRepConst().isUnderlyingScalar(); }
    virtual int isUnderlyingVector(void) const { return getRepConst().isUnderlyingVector(); }
    virtual int isUnderlyingAnions(void) const { return getRepConst().isUnderlyingAnions(); }

    virtual const Vector<int> &ClassLabels(void)   const { return getRepConst().ClassLabels(); }
    virtual int getInternalClass(const gentype &y) const { return getRepConst().getInternalClass(y); }
    virtual int numInternalClasses(void)           const { return getRepConst().numInternalClasses(); }
    virtual int isenabled(int i)                   const { return getRepConst().isenabled(i); }

    virtual double C(void)         const { return getRepConst().C();         }
    virtual double sigma(void)     const { return getRepConst().sigma();     }
    virtual double eps(void)       const { return getRepConst().eps();       }
    virtual double Cclass(int d)   const { return getRepConst().Cclass(d);   }
    virtual double epsclass(int d) const { return getRepConst().epsclass(d); }

    virtual int    memsize(void)      const { return getRepConst().memsize();      }
    virtual double zerotol(void)      const { return getRepConst().zerotol();      }
    virtual double Opttol(void)       const { return getRepConst().Opttol();       }
    virtual int    maxitcnt(void)     const { return getRepConst().maxitcnt();     }
    virtual double maxtraintime(void) const { return getRepConst().maxtraintime(); }

    virtual int    maxitermvrank(void) const { return getRepConst().maxitermvrank(); }
    virtual double lrmvrank(void)      const { return getRepConst().lrmvrank();      }
    virtual double ztmvrank(void)      const { return getRepConst().ztmvrank();      }

    virtual double betarank(void) const { return getRepConst().betarank(); }

    virtual double sparlvl(void) const;

    virtual const Vector<SparseVector<gentype> > &x          (void) const { return getRepConst().x();           }
    virtual const Vector<gentype>                &y          (void) const { return getRepConst().y();           }
    virtual const Vector<vecInfo>                &xinfo      (void) const { return getRepConst().xinfo();       }
    virtual const Vector<int>                    &d          (void) const;
    virtual const Vector<double>                 &Cweight    (void) const { return getRepConst().Cweight();     }
    virtual const Vector<double>                 &Cweightfuzz(void) const { return getRepConst().Cweightfuzz(); }
    virtual const Vector<double>                 &sigmaweight(void) const { return getRepConst().sigmaweight(); }
    virtual const Vector<double>                 &epsweight  (void) const { return getRepConst().epsweight();   }
    virtual const Vector<int>                    &alphaState (void) const;

    virtual int isClassifier(void) const { return getRepConst().isClassifier(); }
    virtual int isRegression(void) const { return getRepConst().isRegression(); }

    // Kernel Modification
    //
    // (need to be implemented)

    virtual void fillCache(void);

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i);
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);
    virtual int removeTrainingVector(int i, int num);

    virtual int setx(int                i, const SparseVector<gentype>          &x);
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x);
    virtual int setx(                      const Vector<SparseVector<gentype> > &x);

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) { (void) i; (void) x; (void) dontupdate; throw("blk_connect: qswapx not implemented here."); return 1; }
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) { (void) i; (void) x; (void) dontupdate; throw("blk_connect: qswapx not implemented here."); return 1; }
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) {           (void) x; (void) dontupdate; throw("blk_connect: qswapx not implemented here."); return 1; }

    virtual int sety(int                i, const gentype         &y);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y);
    virtual int sety(                      const Vector<gentype> &y);

    virtual int sety(int                i, double                z);
    virtual int sety(const Vector<int> &i, const Vector<double> &z);
    virtual int sety(                      const Vector<double> &z);

    virtual int sety(int                i, const Vector<double>          &z);
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z);
    virtual int sety(                      const Vector<Vector<double> > &z);

    virtual int sety(int                i, const d_anion         &z);
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &z);
    virtual int sety(                      const Vector<d_anion> &z);

    virtual int setd(int                i, int                nd);
    virtual int setd(const Vector<int> &i, const Vector<int> &nd);
    virtual int setd(                      const Vector<int> &nd);

    virtual int setCweight(int i,                double nv               );
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv);
    virtual int setCweight(                      const Vector<double> &nv);

    virtual int setCweightfuzz(int i,                double nv               );
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv);
    virtual int setCweightfuzz(                      const Vector<double> &nv);

    virtual int setsigmaweight(int i,                double nv               );
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &nv);
    virtual int setsigmaweight(                      const Vector<double> &nv);

    virtual int setepsweight(int i,                double nv               );
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &nv);
    virtual int setepsweight(                      const Vector<double> &nv);

    virtual int scaleCweight    (double s);
    virtual int scaleCweightfuzz(double s);
    virtual int scalesigmaweight(double s);
    virtual int scaleepsweight  (double s);

    virtual void assumeConsistentX  (void);
    virtual void assumeInconsistentX(void);

    virtual int isXConsistent(void)        const { return getRepConst().isXConsistent();        }
    virtual int isXAssumedConsistent(void) const { return getRepConst().isXAssumedConsistent(); }

    virtual void xferx(const ML_Base &xsrc) { (void) xsrc; throw("blk_connect: xferx not implemented here."); return; }

    virtual const vecInfo &xinfo(int i)                       const { return getRepConst().xinfo(i); }
    virtual const SparseVector<gentype> &x(int i)             const { return getRepConst().x(i); }
    virtual int xisrank(int i)                                const { return getRepConst().xisrank(i); }
    virtual int xisgrad(int i)                                const { return getRepConst().xisgrad(i); }
    virtual int xisrankorgrad(int i)                          const { return getRepConst().xisrankorgrad(i); }
    virtual int xisclass(int i, int defaultclass, int q = -1) const { return getRepConst().xisclass(i,defaultclass,q); }
    virtual const gentype &y(int i)                           const { return getRepConst().y(i); }

    // Generic target controls:
    //
    // (need to implement these)

    // General modification and autoset functions

    virtual int randomise(double sparsity);
    virtual int autoen(void);
    virtual int renormalise(void);
    virtual int realign(void);

    virtual int setzerotol(double zt);
    virtual int setOpttol(double xopttol);
    virtual int setmaxitcnt(int xmaxitcnt);
    virtual int setmaxtraintime(double xmaxtraintime);

    virtual int setmaxitermvrank(int nv);
    virtual int setlrmvrank(double nv);
    virtual int setztmvrank(double nv);

    virtual int setbetarank(double nv);

    virtual int setC(double xC);
    virtual int setsigma(double xC);
    virtual int seteps(double xC);
    virtual int setCclass(int d, double xC);
    virtual int setepsclass(int d, double xC);

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void);
    virtual int home(void);

    virtual int settspaceDim(int newdim);
    virtual int addtspaceFeat(int i);
    virtual int removetspaceFeat(int i);
    virtual int addxspaceFeat(int i);
    virtual int removexspaceFeat(int i);

    virtual int setsubtype(int i);

    virtual int setorder(int neword);
    virtual int addclass(int label, int epszero = 0);

    // Training functions:

    virtual void fudgeOn(void);
    virtual void fudgeOff(void);

    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch);

    // Evaluation Functions:

    virtual int ggTrainingVector(gentype &resg,                int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { gentype resh; return ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(               gentype &resh, int i                   gentype ***pxyprodi = NULL) const { gentype resg; return ghTrainingVector(resh,resg,i,        pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = NULL, gentype ***pxyprodj = NULL, gentype **pxyprodij = NULL) const;

    virtual void dgTrainingVector(Vector<gentype> &res, int i) const { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>  &res, int i) const { ML_Base::dgTrainingVector(res,i); return; }
//    virtual void dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const;

    virtual int ggTrainingVector(double &resg,         int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { gentype res; int resi = ggTrainingVector(res,i,retaltg,pxyprodi); resg = (double) res;           return resi; }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;
    virtual int ggTrainingVector(d_anion &resg,        int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { gentype res; int resi = ggTrainingVector(res,i,retaltg,pxyprodi); resg = (const d_anion &) res;  return resi; }

//    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const;
//    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const;
//    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const;

//    virtual void stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const;


    virtual int gg(gentype &resg, const SparseVector<gentype> &x, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { gentype resh; return gh(resh,resg,x,0,xinf,pxyprodx); }
    virtual int hh(gentype &resh, const SparseVector<gentype> &x, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { gentype resg; return gh(resh,resg,x,0,xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const;

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, gentype ***pxyprodx = NULL, gentype ***pxyprody = NULL, gentype **pxyprodij = NULL) const;

    virtual void dg(Vector<gentype> &res, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { ML_Base::dg(res,x,xinf); return; }
    virtual void dg(Vector<double>  &res, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { ML_Base::dg(res,x,xinf); return; }
//    virtual void dg(Vector<gentype> &res, gentype &resn, const gentype &y, const SparseVector<gentype> &x) const;

    virtual int gg(double &resg,         const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { (void) retaltg; gentype res; int resi = gg(res,x,xinf,pxyprodx); resg = (double) res;           return resi; }
    virtual int gg(Vector<double> &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const;
    virtual int gg(d_anion &resg,        const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { (void) retaltg; gentype res; int resi = gg(res,x,xinf,pxyprodx); resg = (const d_anion &) res;  return resi; }

//    virtual void dg(Vector<double>          &res, double         &resn, const gentype &y, const SparseVector<gentype> &x) const;
//    virtual void dg(Vector<Vector<double> > &res, Vector<double> &resn, const gentype &y, const SparseVector<gentype> &x) const;
//    virtual void dg(Vector<d_anion>         &res, d_anion        &resn, const gentype &y, const SparseVector<gentype> &x) const;

//    virtual void stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const { setInnerWildp(&x); stabProbTrainingVector(res,-1,p,pnrm,rot,mu,B); resetInnerWildp(); return; }



    virtual int varTrainingVector(gentype &resv, gentype &resmu, int i, gentype ***pxyprodi = NULL, gentype **pxyprodii = NULL) const { return covTrainingVector(resv,resmu,i,i,pxyprodi,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL, gentype ***pxyprodx = NULL, gentype **pxyprodxx = NULL) const { return cov(resv,resmu,xa,xa,xainf,xainf,pxyprodx,pxyprodx,pxyprodxx); }

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const { return ML_Base::covarTrainingVector(resv,i); }
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const { return ML_Base::covar(resv,x); }

    // Training data tracking functions:

    virtual const Vector<int>          &indKey(void)          const { return getRepConst().indKey();          }
    virtual const Vector<int>          &indKeyCount(void)     const { return getRepConst().indKeyCount();     }
    virtual const Vector<int>          &dattypeKey(void)      const { return getRepConst().dattypeKey();      }
    virtual const Vector<Vector<int> > &dattypeKeyBreak(void) const { return getRepConst().dattypeKeyBreak(); }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) { NiceAssert( !_altxsrc ); (void) _altxsrc; return; }

    virtual int disable(int i);
    virtual int disable(const Vector<int> &i);

private:

    ML_Base defbase;
    BLK_Consen combit;

    int numReps(void) const
    {
        return mlqlist().indsize();
    }

    const ML_Base &getRepConst(int i = -1) const
    {
        if ( numReps() )
        { 
            return *(mlqlist().direcref( ( i >= 0 ) ? i : 0 ));
        }

        return defbase;
    }

    ML_Base &getRep(int i = -1)
    {
        if ( numReps() )
        { 
            return *(mlqlist().direcref( ( i >= 0 ) ? i : 0 ));
        }

        return defbase;
    }

    const double &getRepWeight(int i = -1) const
    {
        if ( numReps() )
        { 
            return (mlqweight().direcref( ( i >= 0 ) ? i : 0 ));
        }

        const static double retoneval = 1.0;

        return retoneval;
    }

    Vector<int> dscratch;
    Vector<int> alphaStateScratch;

    BLK_Conect *thisthis;
    BLK_Conect **thisthisthis;
};

inline void qswap(BLK_Conect &a, BLK_Conect &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_Conect::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Conect &b = dynamic_cast<BLK_Conect &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_Conect::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Conect &b = dynamic_cast<const BLK_Conect &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_Conect::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Conect &src = dynamic_cast<const BLK_Conect &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
