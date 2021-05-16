
//
// ML (machine learning) base type
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _ml_base_h
#define _ml_base_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "mercer.h"
#include "vector.h"
#include "sparsevector.h"
#include "matrix.h"
#include "gentype.h"
#include "mlcommon.h"
#include "basefn.h"
#include "numbase.h"


class ML_Base;
class SVM_Planar;
class SVM_Scalar;
class SVM_Generic;
class ONN_Generic;
class KNN_Scalar;
class KNN_Vector;
class KNN_MultiC;
class KNN_Binary;
class KNN_Anions;
class BLK_AveVec;
class BLK_AveAni;

#define NUMXTYPES 12
//#define DEFAULT_TUPLE_INDEX_STEP 100

inline std::ostream &operator<<(std::ostream &output, const ML_Base &src );
inline std::istream &operator>>(std::istream &input,        ML_Base &dest);

// Compatibility functions

int isSemicopyCompat(const ML_Base &a, const ML_Base &b);
int isQswapCompat(const ML_Base &a, const ML_Base &b);
int isAssignCompat(const ML_Base &a, const ML_Base &b);

// Swap and zeroing (restarting) functions

inline void qswap(ML_Base &a, ML_Base &b);
inline void qswap(ML_Base *&a, ML_Base *&b);
inline void qswap(const ML_Base *&a, const ML_Base *&b);

inline ML_Base &setident (ML_Base &a) { throw("something"); return a; }
inline ML_Base &setzero  (ML_Base &a);
inline ML_Base &setposate(ML_Base &a) { return a; }
inline ML_Base &setnegate(ML_Base &a) { throw("something"); return a; }
inline ML_Base &setconj  (ML_Base &a) { throw("something"); return a; }
inline ML_Base &setrand  (ML_Base &a) { throw("something"); return a; }
inline ML_Base &postProInnerProd(ML_Base &a) { return a; }

inline ML_Base *&setident (ML_Base *&a) { throw("something"); return a; }
inline ML_Base *&setzero  (ML_Base *&x);
inline ML_Base *&setposate(ML_Base *&a) { return a; }
inline ML_Base *&setnegate(ML_Base *&a) { throw("something"); return a; }
inline ML_Base *&setconj  (ML_Base *&a) { throw("something"); return a; }
inline ML_Base *&setrand  (ML_Base *&a) { throw("something"); return a; }
inline ML_Base *&postProInnerProd(ML_Base *&a) { return a; }

inline const ML_Base *&setident (const ML_Base *&a) { throw("something"); return a; }
inline const ML_Base *&setzero  (const ML_Base *&x);
inline const ML_Base *&setposate(const ML_Base *&a) { return a; }
inline const ML_Base *&setnegate(const ML_Base *&a) { throw("something"); return a; }
inline const ML_Base *&setconj  (const ML_Base *&a) { throw("something"); return a; }
inline const ML_Base *&setrand  (const ML_Base *&a) { throw("something"); return a; }
inline const ML_Base *&postProInnerProd(const ML_Base *&a) { return a; }


// Training vector conversion (ONLY for use in getparam function)
//
// convertSetToSparse: res = { [ s0 : s1 :: s2 ... ] if src = { s0, s1, s2, ... }
//                           { s0                    if src = s0
// convertSparseToSet: res = { { s0, s1, s2, ... } } if src = [ s0 : s1 :: s2 ... ]
//                           { s0                    if src = [ s0 ]
//
// Key assumption: s0, s1, ... non-sparse
//
// If idiv > 0 then res.fff(6) += idiv (additional idiv derivatives)
//
// Return value: 1 if function but not scalar function 0 otherwise.

int convertSetToSparse(SparseVector<gentype> &res, const gentype &src, int idiv = 0);
int convertSparseToSet(gentype &res, const SparseVector<gentype> &src);

// Similarity callbacks - UU uses output kernel and m can be anything, VV has no kernel and assumed m = 2

gentype &UUcallbacknon(gentype &res, int m, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);
gentype &UUcallbackdef(gentype &res, int m, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);

const gentype &VVcallbacknon(gentype &res, int m, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);
const gentype &VVcallbackdef(gentype &res, int m, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);

class ML_Base : public kernPrecursor
{
    friend class SVM_Planar;
    friend class SVM_Scalar;
    friend class SVM_Generic;
    friend class ONN_Generic;
    friend class KNN_Scalar;
    friend class KNN_Vector;
    friend class KNN_MultiC;
    friend class KNN_Binary;
    friend class KNN_Anions;
    friend class BLK_AveVec;
    friend class BLK_AveAni;

public:

    // Constructors, destructors, assignment etc..
    //
    // prealloc: call this to set the expected size of the training set.
    //     this allows memory to be preallocated as a single block rather
    //     than incrementally on a point-by-point basis.  Using this is both
    //     quicker and more memory efficient than the alternative.
    // preallocsize: returns the current preallocation size (0 if none).
    // setmemsize: set size of kernel etc caches.
    //
    // assign: copy constructor
    // semicopy: this is like an assignment operator, but assumes that dest
    //     has all training data and therefore only copies state information.
    //     It should be used when variables have been constrained zero
    //     temporarily (eg when doing cross-fold validation) and you want to
    //     quickly regain old state.
    // qswapinternal: qswap function
    //
    // getparam: get parameter via indexed callback.  Return 0 on success, 1
    //     is answer cannot be resolved (ie is still a function).
    //
    // printstream: print ML to output stream.
    // inputstream: get ML from input stream.
    //
    // NB: - all of these actually dynamic the objects based on the type()
    //       argument and then call the relevant cast version.
    //     - print and input functions are just placeholders called by the
    //       stream operators << and >> on this class.  Polymorph as needed.

    ML_Base(int _isIndPrune = 0);
    ML_Base &operator=(const ML_Base &src) { assign(src); return *this; }
    virtual ~ML_Base();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const { return xpreallocsize; }
    virtual void setmemsize(int memsize);

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const;

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual       ML_Base &getML     (void)       { return *this; }
    virtual const ML_Base &getMLconst(void) const { return *this; }

    // Information functions (training data):
    //
    // N:    the number of training vectors
    // NNC:  the number of training vectors in a given internal class
    // type: the type of the machine.
    // Nw:   size of w vector
    // NwS:  number of non-zero elements in w
    // NwZ:  number of zero elements in w
    //
    // tspaceDim:  the dimensionality of target space.
    // xspaceDim:  index dimensionality of input space.
    // fspaceDim:  dimensionality of feature space (-1 if infinite)
    // numClasses: the number of classes
    // order:      log2(tspaceDim)
    //
    // isTrained: true if machine is trained and up to date
    // isMutable: true if machine is mutable (type can be changed)
    // isPool:    true if machine is ML_Pool
    //
    // getML:      reference to "actual" ML (differs from this if mutable)
    // getMLconst: reference to "actual" ML (differs from this if mutable)
    //
    // isUnderlyingScalar: true if underlying weight type is double
    // isUnderlyingVector: true if underlying weight type is vector
    // isUnderlyingAnions: true if underlying weight type is anions
    //
    // ClassLabels: returns a vector of all class labels.
    // getInternalClass: for classifiers this returns the internal class
    //     representation number (0 for regressor).  For all classifiers each 
    //     class is assigned a number 0,1,...,m, where m = numInternalClasses, 
    //     which is the number of actual classes plus the anomaly class, if
    //     there is one.
    // numInternalClasses: number of internal classes
    //
    // gOutType: unprocessed g(x) output type of machine (see gentype)
    // hOutType: processed h(x) output type of machine (see gentype)
    // targType: target data y type of machine (see gentype)
    // calcDist: given processed outputs ha, hb calculate, for this ML, the
    //     norm  squared error.  db applies to scalar types and is ignored 
    //     elsewhere.  +1 indicates lower bound only, -1 upper bound only.  
    //     For all types 0 means don't include.
    //
    // sparlvl: sparsity level (1 completely sparse, 0 non-sparse)
    //
    // isSVM.../isONN...: returns true if type matches
    //
    // zerotol:      zero tolerance
    // Opttol:       optimality tolerance
    // maxitcnt:     maximum iteration count for training
    // maxtraintime: maximum training time
    //
    // x,y,Cweight,epsweight: training data
    // b,w,ws,v,vn: trained machine
    // ws_i = 0 if w_i = 0, nz otherwise
    // vn_i = the norm of v_i
    //
    // x is the training vectors
    // y is the targets
    //
    // alphaState: has usual meaning for svm_generic, but also used to
    //             indicate if a particular training pair has any influence
    //             on the trained machine.  By default 1 for all training
    //             variables

    virtual int N(void)       const { return altxsrc ? (*altxsrc).N() : allxdatagent.size(); }
    virtual int NNC(int d)    const { return d ? 0 : xdzero; }
    virtual int type(void)    const { return -1;             }
    virtual int subtype(void) const { return 0;              }

    virtual int tspaceDim(void)    const { return 1;                        }
    virtual int xspaceDim(void)    const { int res = indKey().size(); return ( wildxdim > res ) ? wildxdim : res; }
    virtual int fspaceDim(void)    const { return getKernel().phidim(1,xspaceDim()); }
    virtual int tspaceSparse(void) const { return 0;                        }
    virtual int xspaceSparse(void) const { return 1;                        }
    virtual int numClasses(void)   const { return 0;                        }
    virtual int order(void)        const { return ceilintlog2(tspaceDim()); }

    virtual int isTrained(void) const { return 0; }
    virtual int isMutable(void) const { return 0; }
    virtual int isPool   (void) const { return 0; }

    virtual char gOutType(void) const { return '?'; }
    virtual char hOutType(void) const { return '?'; }
    virtual char targType(void) const { return '?'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const { (void) ha; (void) hb; (void) ia; (void) db; throw("calcDist undefined at this level."); return 0; }

    virtual int isUnderlyingScalar(void) const { return 1; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual const Vector<int> &ClassLabels(void)   const { const static Vector<int> temp; return temp;         }
    virtual int getInternalClass(const gentype &y) const { (void) y;                      return 0;            }
    virtual int numInternalClasses(void)           const {                                return numClasses(); }
    virtual int isenabled(int i)                   const {                                return d()(i);       }

    virtual double C(void)         const { return DEFAULT_C;   }
    virtual double sigma(void)     const { return 1.0/C();     }
    virtual double eps(void)       const { return DEFAULTEPS;  }
    virtual double Cclass(int d)   const { (void) d; return 1; }
    virtual double epsclass(int d) const { (void) d; return 1; }

    virtual int    memsize(void)      const { return DEFAULT_MEMSIZE;      }
    virtual double zerotol(void)      const { return globalzerotol;        }
    virtual double Opttol(void)       const { return DEFAULT_OPTTOL;       }
    virtual int    maxitcnt(void)     const { return DEFAULT_MAXITCNT;     }
    virtual double maxtraintime(void) const { return DEFAULT_MAXTRAINTIME; }

    virtual int    maxitermvrank(void) const { return DEFAULT_MAXITERMVRANK; }
    virtual double lrmvrank(void)      const { return DEFAULT_LRMVRANK;      }
    virtual double ztmvrank(void)      const { return DEFAULT_ZTMVRANK;      }

    virtual double betarank(void) const { return DEFAULT_BETARANK; }

    virtual double sparlvl(void) const { return 0; }

    virtual const Vector<SparseVector<gentype> > &x          (void) const { return altxsrc ? (*altxsrc).x() : allxdatagent; }
    virtual const Vector<gentype>                &y          (void) const { return alltraintarg;                            }
    virtual const Vector<vecInfo>                &xinfo      (void) const { return traininfo;                               }
    virtual const Vector<int>                    &xtang      (void) const { return traintang;                               }
    virtual const Vector<int>                    &d          (void) const { return xd;                                      }
    virtual const Vector<double>                 &Cweight    (void) const { return xCweight;                                }
    virtual const Vector<double>                 &Cweightfuzz(void) const { return xCweightfuzz;                            }
    virtual const Vector<double>                 &sigmaweight(void) const;
    virtual const Vector<double>                 &epsweight  (void) const { return xepsweight;                              }
    virtual const Vector<int>                    &alphaState (void) const { return xalphaState;                             }

    virtual int isClassifier(void) const { return 0; }
    virtual int isRegression(void) const { return !isClassifier(); }

    // Version numbers, ML ids etc
    //
    // xvernum(): An integer that gets incremented whenever x is changed in a
    //            non-simple fashion.  Non-simple changes are anything except
    //            adding new vectors to the end of the dataset.
    //            Starts at 0, handy if you want to cache something x related.
    // xvernum(altmlid): gives x version number for a different ML
    // incxvernum(): increments xvernum() (for this ML)
    // gvernum(): An int that gets incremented whenever gh is changed.
    // gvernum(altmlid): gives g version number for a different ML
    // incgvernum(): increments gvernum() (for this ML)
    // getaltML(): get reference to ML with given ID.  Return 0 on success, 1 if NULL.

    virtual int MLid(void) const { return kernPrecursor::MLid(); }
    virtual int setMLid(int nv);
    virtual int getaltML(kernPrecursor *&res, int altMLid) const { return kernPrecursor::getaltML(res,altMLid); }

    virtual int xvernum(void)        const { svm_mutex_lock((**thisthisthis).mleyelock); int res = (*const_cast<SparseVector<int>*>(xvernumber))(MLid());       svm_mutex_unlock((**thisthisthis).mleyelock); return res; }
    virtual int xvernum(int altMLid) const { svm_mutex_lock((**thisthisthis).mleyelock); int res = (*const_cast<SparseVector<int>*>(xvernumber))(altMLid);      svm_mutex_unlock((**thisthisthis).mleyelock); return res; }
    virtual int incxvernum(void)           { svm_mutex_lock((**thisthisthis).mleyelock); int res = ++(*const_cast<SparseVector<int>*>(xvernumber))("&",MLid()); svm_mutex_unlock((**thisthisthis).mleyelock); return res; }
    virtual int gvernum(void)        const { svm_mutex_lock((**thisthisthis).mleyelock); int res = (*const_cast<SparseVector<int>*>(gvernumber))(MLid());       svm_mutex_unlock((**thisthisthis).mleyelock); return res; }
    virtual int gvernum(int altMLid) const { svm_mutex_lock((**thisthisthis).mleyelock); int res = (*const_cast<SparseVector<int>*>(gvernumber))(altMLid);      svm_mutex_unlock((**thisthisthis).mleyelock); return res; }
    virtual int incgvernum(void)           { svm_mutex_lock((**thisthisthis).mleyelock); int res = ++(*const_cast<SparseVector<int>*>(gvernumber))("&",MLid()); svm_mutex_unlock((**thisthisthis).mleyelock); return res; }

    // RKHS inner-product support: inner product between this and m others 
    // (so e.g. if m = 1 this is a two-product).

    virtual void mProdPt(double &res, int m, int *x) { (void) res; (void) m; (void) x; throw("RKHS inner product not supported for this ML type"); return; }

    // Kernel Modification
    //
    // The safe way to modify the kernel is to use k = getKernel() to get a 
    // copy of the current kernel, modify the copy, and then use the function
    // setKernel(k) to update (set) the kernel used by the ML.  Which can be
    // slow due to all the copying of kernels required.
    //
    // An faster alternative (unsafe) method is to use getKernel_unsafe to
    // obtain a reference to the actual kernel being used by the ML and modify
    // it directly, then call resetKernel() to force make the ML aware that
    // changes have been made, so for example:
    //
    // SVM_Scalar x
    // ...
    // (x.getKernel_unsafe()).setType(4,1);
    // x.resetKernel();
    //
    // is functionally equivalent to the slower alternative:
    //
    // SVM_Scalar x;
    // MercerKernel k;
    // ...
    // k = x.getKernel();
    // k.setType(4,1);
    // x.setKernel(k);
    //
    // The latter requires 3 calls to MercerKernel's copy constructor
    // plus memory to store k.  The former requires no calls to the copy
    // constructor and no additional memory to store the kernel.
    //
    // The modind argument may be set 0 when calling setKernel or resetKernel
    // provided that indexing has not been switched on, switched off, or the
    // indexes themselves changed.
    //
    // The onlyChangeRowI argument is used to indicate that the change
    // only affects that row.  Set -1 if change affects all rows (default).
    //
    // The updateInfo argument may be used to suppress updating traininfo
    // if not required.  By default this is 1 (do update), set 0 if modind
    // or shift/scale are unchanged (these are the only kernel attributes
    // that will have an impact on traininfo).
    //
    // Note that the kernel is present in all kernel methods.  In non-kernel
    // based methods it must remain linear but may still be used for
    // shifting and scaling of data for normalisation purposes.
    //
    // prepareKernel: if using the resetKernel trick and making changes that
    // don't change x or its inner product (indexing, shift/scale, 8xx kernels)
    // then you can call this first to save information and time.
    //
    // Kalt: evaluate K for alternative kernel function.
    // K2xfer: used to borrow "learnt" kernels
    //
    // 800: Trivial:  K(x,y) = Kx(x,y)
    // 801: m-inner:  K(x,y) = sum_ij a_i a_j Kx(x_i,x_j,x,y)
    //                (eg. for SVM a_i = alpha_i)
    // 802: Moment:   K(x,y) = sum_ij a_i a_j Kx(x_i,x_j)
    //                (eg. for SVM a_i = alpha_i)
    // 803: k-learn:  K(x,y) = sum_ij a_i Kx(x_i,(x,y))  - indices not passed through
    //                (eg. for SVM a_i = alpha_i.  Typically x_i = (xa_i,xb_i))
    // 804: k-learn:  K(x,y) = sum_ij a_i Kx(x_i,(x,y))  - indices passed through
    //                (eg. for SVM a_i = alpha_i.  Typically x_i = (xa_i,xb_i))
    // 805: k2-learn: K(x,y) = (sum_i a_i Kx(x_i,(x,y)))^2  - indices not passed through
    //                (eg. for SVM a_i = alpha_i.  Typically x_i = (xa_i,xb_i))
    // 81x: like above but with indice pass-through
    //
    // fillCache: runs through all vectors and calls K(res,i,j).  This is handy
    //            to pre-fill any kernel caches (blk type, not Gp type).
    //
    // K2bypass: if matrix (of non-zero size) is put here then K2 function with i,j>=0
    //           is taken directly from this matrix
    //
    // Notes: - K4 does not calculate gradient constraints
    //        - Km does not calculate gradient constraints or rank constraints
    //        - Km assumes any NULLs in xx (and xxinfo) are at the end, not the start
    //        - K4 if xa == xb == NULL, xc == xd given then assumes xc := xa.*xb, xd := xc.xd
    //        - K2ip calculates the inner product as used by the kernel calculation.
    //
    // Polymorphing K2xfer:
    //
    // - pass kernel 800 back to ML_Base.
    // - always add elements to end of i,xx,xxinfo.
    // - if you change the ordering of these remember to change them back!
    //
    // Gradients: 
    //
    // - dK calculates gradient wrt <x,y> (or K2xfer(x,y)) and ||x||^2 (or K2xfer(x,x))
    //   (set deepDeriv 1 for derivative of <x,y>, ||x|| regardless)
    // - dK2delx calculates gradient wrt x and returns result as:
    //   xscaleres.x + yscaleres.y   (or xscaleres.x(minmaxind) + yscaleres.y(minmaxind) 
    //                                if minmaxind >= 0 in result)
    // - d2K2delxdelx calculates the drivative d/dx d/dx K(x,y) and returns result as:
    //   xxscaleres.x.x' + xyscaleres.x.y' + yxscaleres.y.x' + yyscaleres.y.y' + constres.I
    // - d2K2delxdely calculates the drivative d/dx d/dy K(x,y) and returns result as:
    //   xxscaleres.x.x' + xyscaleres.x.y' + yxscaleres.y.x' + yyscaleres.y.y' + constres.I
    // - dnK2del calculates the derivative:
    //   d/dxq0 d/dxq1 ... K(x0,x1)
    //   and returns a vectorised result of the form:
    //   sum_i sc_i kronprod_j [ x{nn_ij}   if nn_ij == 0,1
    //                         [ kd{nn_ij}  if nn_ij < 0
    //   where kd{a} kd_{a} is the vectorised identity matrix

    virtual const MercerKernel &getKernel(void) const { return kernel; }
    virtual MercerKernel &getKernel_unsafe(void)      { return kernel; }
    virtual void prepareKernel(void)                  { return;        }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1);

    virtual void fillCache(void);

    virtual void K2bypass(const Matrix<gentype> &nv) { K2mat = nv; return; }

    gentype &Keqn(gentype &res,                           int resmode = 1) const;
    gentype &Keqn(gentype &res, const MercerKernel &altK, int resmode = 1) const;

    virtual gentype &K1(gentype &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { setInnerWildpa(&xa,xainf); K1(res,-1); resetInnerWildp(( xainf == NULL )); return res; }
    virtual gentype &K2(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); K2(res,-1,-3); resetInnerWildp(( xainf == NULL ),( xbinf == NULL )); return res; }
    virtual gentype &K3(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); setInnerWildpc(&xc,xcinf); K3(res,-1,-3,-4);  resetInnerWildp(( xainf == NULL ),( xbinf == NULL ),( xcinf == NULL )); return res; }
    virtual gentype &K4(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL, const vecInfo *xdinf = NULL) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); setInnerWildpc(&xc,xcinf); setInnerWildpd(&xd,xdinf); K4(res,-1,-3,-4,-5); resetInnerWildp(( xainf == NULL ),( xbinf == NULL ),(xcinf == NULL),(xdinf == NULL)); return res; }
    virtual gentype &Km(gentype &res, const Vector<SparseVector<gentype> > &xx) const { int m = xx.size(); setInnerWildpx(&xx); retVector<int> tmpva; Vector<int> ii(cntintvec(m,tmpva)); ii += 1; ii *= -100; Km(m,res,ii); resetInnerWildp(); return res; }

    virtual double &K2ip(double &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); K2ip(res,-1,-3,0.0); resetInnerWildp(( xainf == NULL ),( xbinf == NULL )); return res; }
    virtual double distK(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); double res = distK(-1,-3); resetInnerWildp(( xainf == NULL ),( xbinf == NULL )); return res; }

    virtual Vector<gentype> &phi2(Vector<gentype> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { setInnerWildpa(&xa,xainf); phi2(res,-1); resetInnerWildp(( xainf == NULL )); return res; }
    virtual Vector<gentype> &phi2(Vector<gentype> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const;

    virtual Vector<double> &phi2(Vector<double> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { setInnerWildpa(&xa,xainf); phi2(res,-1); resetInnerWildp(( xainf == NULL )); return res; }
    virtual Vector<double> &phi2(Vector<double> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const;

    virtual double &K0ip(       double &res, const gentype **pxyprod = NULL) const { return KK0ip(res,0.0,pxyprod); }
    virtual double &K1ip(       double &res, int ia, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL) const { return KK1ip(res,ia,0.0,pxyprod,xa,xainfo); }
    virtual double &K2ip(       double &res, int ia, int ib, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL) const { return KK2ip(res,ia,ib,0.0,pxyprod,xa,xb,xainfo,xbinfo); }
    virtual double &K3ip(       double &res, int ia, int ib, int ic, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL) const { return KK3ip(res,ia,ib,ic,0.0,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double &K4ip(       double &res, int ia, int ib, int ic, int id, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL) const { return KK4ip(res,ia,ib,ic,id,0.0,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double &Kmip(int m, double &res, Vector<int> &i, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL) const { return KKmip(m,res,i,0.0,pxyprod,xx,xxinfo); }

    virtual double &K0ip(       double &res, const double &bias, const gentype **pxyprod = NULL) const { return KK0ip(res,bias,pxyprod); }
    virtual double &K1ip(       double &res, int ia, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL) const { return KK1ip(res,ia,bias,pxyprod,xa,xainfo); }
    virtual double &K2ip(       double &res, int ia, int ib, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL) const { return KK2ip(res,ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo); }
    virtual double &K3ip(       double &res, int ia, int ib, int ic, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL) const { return KK3ip(res,ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double &K4ip(       double &res, int ia, int ib, int ic, int id, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL) const { return KK4ip(res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double &Kmip(int m, double &res, Vector<int> &i, const double &bias, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL) const { return KKmip(m,res,i,bias,pxyprod,xx,xxinfo); }

    virtual gentype        &K0(              gentype        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const;
    virtual gentype        &K0(              gentype        &res, const gentype &bias     , const gentype **pxyprod = NULL, int resmode = 0) const;
    virtual gentype        &K0(              gentype        &res, const MercerKernel &altK, const gentype **pxyprod = NULL, int resmode = 0) const;
    virtual double         &K0(              double         &res                          , const gentype **pxyprod = NULL, int resmode = 0) const;
    virtual Matrix<double> &K0(int spaceDim, Matrix<double> &res                          , const gentype **pxyprod = NULL, int resmode = 0) const;
    virtual d_anion        &K0(int order,    d_anion        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const;

    virtual gentype        &K1(              gentype        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const;
    virtual gentype        &K1(              gentype        &res, int ia, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const;
    virtual gentype        &K1(              gentype        &res, int ia, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const;
    virtual double         &K1(              double         &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const;
    virtual Matrix<double> &K1(int spaceDim, Matrix<double> &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const;
    virtual d_anion        &K1(int order,    d_anion        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const;

    virtual gentype        &K2(              gentype        &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;
    virtual double         &K2(              double         &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;
    virtual Matrix<double> &K2(int spaceDim, Matrix<double> &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;
    virtual d_anion        &K2(int order,    d_anion        &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const;

    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const;
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const;
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const;
    virtual double         &K3(              double         &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const;
    virtual Matrix<double> &K3(int spaceDim, Matrix<double> &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const;
    virtual d_anion        &K3(int order,    d_anion        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const;

    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const;
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const;
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const;
    virtual double         &K4(              double         &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const;
    virtual Matrix<double> &K4(int spaceDim, Matrix<double> &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const;
    virtual d_anion        &K4(int order,    d_anion        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const;

    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const;
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const gentype &bias     , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const;
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const MercerKernel &altK, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const;
    virtual double         &Km(int m              , double         &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const;
    virtual Matrix<double> &Km(int m, int spaceDim, Matrix<double> &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const;
    virtual d_anion        &Km(int m, int order   , d_anion        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const;

    virtual void dK(gentype &xygrad, gentype &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const;
    virtual void dK(double  &xygrad, double  &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const;

    virtual void d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;
    virtual void d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;

    virtual void dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;
    virtual void dK2delx(double  &xscaleres, double  &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;

    virtual void d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;
    virtual void d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;

    virtual void d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;
    virtual void d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;

    virtual void dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;
    virtual void dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;

    virtual double distK(int i, int j) const;

    virtual void densedKdx(double &res, int i, int j) const { return densedKdx(res,i,j,0.0); }
    virtual void denseintK(double &res, int i, int j) const { return denseintK(res,i,j,0.0); }

    virtual void densedKdx(double &res, int i, int j, const double &bias) const;
    virtual void denseintK(double &res, int i, int j, const double &bias) const;

    virtual void ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, int i, int j) const;

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

    virtual const gentype &xelm(gentype &res, int i, int j) const;
    virtual int xindsize(int i) const;

    // Training set modification:
    //
    // addTrainingVector:  add training vector to training set.
    // qaddTrainingVector: like addTrainingVector, but uses qswap for speed.
    //
    // removeTraingVector: remove training vector from training set.
    //
    // NB: - x is not preserved by qaddTrainingVector
    //     - if x,y are included in removeTrainingVector then these are
    //       qswapped out of data before removal
    //     - all functions that modify the ML return 0 if the trained machine
    //       is unchanged, 1 otherwise
    //
    // qswapx swaps vectors rather than overwriting them
    //  - set dontupdate to prevent any updates being processed
    //  - see also assumeConsistentX if you want fast.
    //
    // x can also be accessed directly with the x_unsafe function.
    // If any changes are made you also need to call update_x.
    //
    // Optimality note: if x vectors have 3ent indices (ie non-sparse,
    // same dimension, or sparse but with the same zeros and non-zeros for
    // all cases) then you can set assumeConsistentX.  This will only set
    // or change indexKey when vector x(0) is modified, and will lead to
    // speedups elsewhere.

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i)                                       { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);
    virtual int removeTrainingVector(int i, int num);

    virtual int setx(int                i, const SparseVector<gentype>          &x);
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x);
    virtual int setx(                      const Vector<SparseVector<gentype> > &x);

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0);
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0);
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0);

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

    virtual int setd(int                i, int                d);
    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(                      const Vector<int> &d);

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
    virtual int scalesigmaweight(double s) { return scaleCweight(1/s); }
    virtual int scaleepsweight  (double s);

    virtual void assumeConsistentX  (void) { xassumedconsist = 1; xconsist = 0;              return; }
    virtual void assumeInconsistentX(void) { xassumedconsist = 0; xconsist = testxconsist(); return; }

    virtual int isXConsistent(void)        const { return xassumedconsist || xconsist; }
    virtual int isXAssumedConsistent(void) const { return xassumedconsist;             }

    virtual void xferx(const ML_Base &xsrc);

    virtual const vecInfo &xinfo(int i)                       const;
    virtual int xtang(int i)                                  const;
    virtual const SparseVector<gentype> &x(int i)             const { return xgetloc(i);  }
    virtual int xisrank(int i)                                const { const SparseVector<gentype> &xres = x(i); return xres.isfaroffindpresent() || xres.isfarfarfarindpresent(1);  }
    virtual int xisgrad(int i)                                const { const SparseVector<gentype> &xres = x(i); return xres.isfarfaroffindpresent(); }
    virtual int xisrankorgrad(int i)                          const { const SparseVector<gentype> &xres = x(i); return xres.isfaroffindpresent() || xres.isfarfarfarindpresent(1) || xres.isfarfaroffindpresent(); }
    virtual int xisclass(int i, int defaultclass, int q = -1) const { const SparseVector<gentype> &xres = x(i); return ( q == -1 ) ? defaultclass : ( xres.isfarfarfarindpresent((100*q)+0) ? ( (int) xres.fff((100*q)+0) ) : defaultclass ); }
    virtual const gentype &y(int i)                           const { if ( i >= 0 ) { return y()(i); } return ytargdata; }

    // Generic target controls: in some generic target classes the output
    // is restricted to lie in the span of a particular basis.  In this case
    // these functions control contents of this basis.  The output kernel 
    // specifies the similarity measure between basis elements.
    //
    // basisType: 0 = gentype basis defined by user
    //            1 = basis same as y() vector
    //
    // setBasis(n,d) sets random 1-norm unit basis of n elements, each a
    // d-dimensional real-valued 1-norm unit vector.

    virtual int NbasisUU(void)    const { return locbasisUU.size(); }
    virtual int basisTypeUU(void) const { return isBasisUserUU;     }
    virtual int defProjUU(void)   const { return defbasisUU;        }

    virtual const Vector<gentype> &VbasisUU(void) const { return locbasisUU; }

    virtual int setBasisYUU(void);
    virtual int setBasisUUU(void);
    virtual int addToBasisUU(int i, const gentype &o);
    virtual int removeFromBasisUU(int i);
    virtual int setBasisUU(int i, const gentype &o);
    virtual int setBasisUU(const Vector<gentype> &o);
    virtual int setDefaultProjectionUU(int d) { int res = defbasisUU; defbasisUU = d; return res; }
    virtual int setBasisUU(int n, int d);

    virtual int NbasisVV(void)    const { return locbasisVV.size(); }
    virtual int basisTypeVV(void) const { return isBasisUserVV;     }
    virtual int defProjVV(void)   const { return defbasisVV;        }

    virtual const Vector<gentype> &VbasisVV(void) const { return locbasisVV; }

    virtual int setBasisYVV(void);
    virtual int setBasisUVV(void);
    virtual int addToBasisVV(int i, const gentype &o);
    virtual int removeFromBasisVV(int i);
    virtual int setBasisVV(int i, const gentype &o);
    virtual int setBasisVV(const Vector<gentype> &o);
    virtual int setDefaultProjectionVV(int d) { int res = defbasisVV; defbasisVV = d; return res; }
    virtual int setBasisVV(int n, int d);

    virtual const MercerKernel &getUUOutputKernel(void) const                  { return UUoutkernel;                                   }
    virtual MercerKernel &getUUOutputKernel_unsafe(void)                       { return UUoutkernel;                                   }
    virtual int resetUUOutputKernel(int modind = 1)                            { return setUUOutputKernel(getUUOutputKernel(),modind); }
    virtual int setUUOutputKernel(const MercerKernel &xkernel, int modind = 1);

    // General modification and autoset functions
    //
    // scale:   scale y and K
    // reset:   set w = 0, b = 0, start as per starting state
    // restart: reset to state immediately after construction
    // home:    for serial/parallel blocks this sets the active element to
    //          -1, which is the parent (overall) view.
    //
    // set...: set various things
    //
    // randomise: randomise "weights" to uniform between 0 and 1, with given
    //            fraction set zero (sparsity)
    // autoen: set targets equal to inputs and train.  If output dimension
    //         does not match input then trim or pad with zeros.  If output
    //         type does not match input type then attempt "closest copy"
    //         (no guarantees that this means anything).
    // renormalise: randomisation can lead to very large training outputs.
    //            this scales so that training outputs range from zero to 1
    // realign:   set targets equal to output of system for training set.
    //
    // addxspaceFeat/removexspaceFeat: These functions are for adding and
    // removing dimensions from input space.  Now input sparse is made up of
    // sparse features, so their dimensionality is actually undefined.
    // However the features used by the training set are well defined, and
    // in some cases it is important to know which of these are being used
    // (for neural nets, for example, need to know this to set up the
    // weights).  By default these functions do nothing.
    //
    // When vectors are added/removed the code checks for changes to input
    // space and calls these functions to reflect changes.  They can be
    // polymorphed by child functions where this knowledge is important.

    virtual int randomise(double sparsity) { (void) sparsity; return 0; }
    virtual int autoen(void);
    virtual int renormalise(void);
    virtual int realign(void);

    virtual int setzerotol(double zt);
    virtual int setOpttol(double xopttol)             { (void) xopttol;       return 0; }
    virtual int setmaxitcnt(int xmaxitcnt)            { (void) xmaxitcnt;     return 0; }
    virtual int setmaxtraintime(double xmaxtraintime) { (void) xmaxtraintime; return 0; }

    virtual int setmaxitermvrank(int nv) { (void) nv; throw("Function setmaxitermvrank not available for this ML type."); return 0; }
    virtual int setlrmvrank(double nv)   { (void) nv; throw("Function setlrmvrank not available for this ML type."); return 0; }
    virtual int setztmvrank(double nv)   { (void) nv; throw("Function setztmvrank not available for this ML type."); return 0; }

    virtual int setbetarank(double nv) { (void) nv; throw("Function setbetarank not available for this ML type."); return 0; }

    virtual int setC    (double xC)             {           (void) xC;   return 0; }
    virtual int setsigma(double xsigma)         { return setC(1/xsigma);           }
    virtual int seteps  (double xeps)           {           (void) xeps; return 0; }
    virtual int setCclass  (int d, double xC)   { (void) d; (void) xC;   return 0; }
    virtual int setepsclass(int d, double xeps) { (void) d; (void) xeps; return 0; }

    virtual int scale(double a) { (void) a; return 0; }
    virtual int reset(void)     {           return 0; }
    virtual int restart(void)   {           return 0; }
    virtual int home(void)      {           return 0; }

    virtual ML_Base &operator*=(double sf) { scale(sf); return *this; }

    virtual int settspaceDim(int newdim) { (void) newdim; throw("Function settspaceDim not available for this ML type.");     return 0; }
    virtual int addtspaceFeat(int i)     { (void) i;      throw("Function addtspaceFeat not available for this ML type.");    return 0; }
    virtual int removetspaceFeat(int i)  { (void) i;      throw("Function removetspaceFeat not available for this ML type."); return 0; }
    virtual int addxspaceFeat(int i)     { (void) i;                                                                          return 0; }
    virtual int removexspaceFeat(int i)  { (void) i;                                                                          return 0; }

    virtual int setsubtype(int i) { (void) i; NiceAssert( i == 0 ); return 0; }

    virtual int setorder(int neword)                 { (void) neword;                throw("Function setorder not available for this ML type"); return 0; }
    virtual int addclass(int label, int epszero = 0) { (void) label; (void) epszero; throw("Function addclass not available for this ML type"); return 0; }

    // Sampling mode
    //
    // For stochastic models, setting sample mode NZ makes
    // the model act like a sample from the distribution (so for example
    // a GP in sample mode, upon evaluating gg, takes a sample from
    // the posterior and adds it to the prior).

    virtual int isSampleMode(void) const { return 0; }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp = DEFAULT_SAMPLES_SAMPLE) { (void) nv; (void) xmin; (void) xmax; (void) Nsamp; return 0; }

    // Training functions:
    //
    // res not changed on success, set nz on fail (so set zero before calling)
    // returns 0 if trained machine unchanged, 1 otherwise
    //
    // NB: - killSwitch is polled periodically, and training will terminate
    //       early if it is set.  This is designed for multi-thread use.

    virtual void fudgeOn(void)  { return; }
    virtual void fudgeOff(void) { return; }

    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) { (void) res; killSwitch = 0; return 0; }

    // Evaluation Functions:
    //
    // - gg(resg ,y): writes unprocessed result to resg
    // - hh(resh ,y): writes processed result to resh
    // - gh(rh,rg,y): writes both processed and unprocessed results
    //
    // - cov(res,mu,x,y): calculate covariance in resg (if available)
    //                Note that this is always scalar (variance is a function
    //                of x, which is shared by all axis in the scalar case).
    //                If y is a vector of vectors the result is a covariance 
    //                matrix.  mu is also returned because this is a negligable-
    //                cost operation in most cases.  mu relates to x.
    //
    // -  e(res,y,x): writes smoothed error to res (sigmode not sgn for
    //                classifiers)
    //
    // - dg(res,y,x): writes dg/dx to res
    //
    // - stabProb: probability of mu_{1:p} stability, using ||.||_pnrm (or rotated inf-norm if rot != 0)
    //
    // The error is q(x)-y, where q(x) = h(x) for the regression types, and
    // some version of the sigmoid function for classification.  Note that
    // the error may be a scalar, vector or anion.  The gradient de is a
    // sparse vector gradient wrt raw output.  Note that this is not currently
    // well defined if raw output includes things like g and x.
    //
    // Gradients: grad = sum_i res_i mod_vj(x_i) + resn x_-1
    //            mod_vj(x_i) = x_i          if vj <  0
    //                          [ 0        ]
    //                        = [ x_{i,vj} ] if vj >= 0
    //                          [ 0        ]
    //
    // pxyprod: if you know the inner products (or diff) of a vector wrt all training vectors you
    //          can put it here to speed up calculations.  This is a vector of pxyprod pointers as
    //          described previously.

    virtual int ggTrainingVector(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { gentype resh; return ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = NULL) const { gentype resg; return ghTrainingVector(resh,resg,i,0,      pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { (void) resh; (void) resg; (void) i; (void) retaltg; (void) pxyprodi; throw("ghTrainingVector not implemented for this ML."); return 0; }

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = NULL, gentype ***pxyprodj = NULL, gentype **pxyprodij = NULL) const { (void) resv; (void) resmu; (void) i; (void) j; (void) pxyprodi; (void) pxyprodj; (void) pxyprodij; throw("covTrainingVector not implemented for this ML."); return 0; }

    virtual void dgTrainingVector(Vector<gentype> &res, int i) const;
    virtual void dgTrainingVector(Vector<double>  &res, int i) const;
    virtual void dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const { (void) res; (void) resn; (void) i; throw("Function dgTrainingVector not available for this ML type."); return; }

    virtual int ggTrainingVector(double         &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { (void) pxyprodi; gentype res; int resi = ggTrainingVector(res,i,retaltg); resg = (double) res;           return resi; }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;
    virtual int ggTrainingVector(d_anion        &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { (void) pxyprodi; gentype res; int resi = ggTrainingVector(res,i,retaltg); resg = (const d_anion &) res;  return resi; }

    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const { (void) res; (void) resn; (void) i; throw("Function dgTrainingVector not available for this ML type."); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const { (void) res; (void) resn; (void) i; throw("Function dgTrainingVector not available for this ML type."); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const { (void) res; (void) resn; (void) i; throw("Function dgTrainingVector not available for this ML type."); return; }

    virtual void stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const;


    virtual int gg(               gentype &resg, const SparseVector<gentype> &x,                  const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { gentype resh; return gh(resh,resg,x,0,xinf,pxyprodx); }
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x,                  const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { gentype resg; return gh(resh,resg,x,0,xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { setInnerWildpa(&x,xinf); int res = ghTrainingVector(resh,resg,-1,retaltg,pxyprodx); resetInnerWildp(xinf == NULL); return res; }

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, gentype ***pxyprodxa = NULL, gentype ***pxyprodxb = NULL, gentype **pxyprodij = NULL) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); int res = covTrainingVector(resv,resmu,-1,-3,pxyprodxa,pxyprodxb,pxyprodij); resetInnerWildp(( xainf == NULL ),( xbinf == NULL )); return res; }

    virtual void dg(Vector<gentype> &res, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { setInnerWildpa(&x,xinf); dgTrainingVector(res,-1); resetInnerWildp(xinf == NULL); return; }
    virtual void dg(Vector<double>  &res, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { setInnerWildpa(&x,xinf); dgTrainingVector(res,-1); resetInnerWildp(xinf == NULL); return; }
    virtual void dg(Vector<gentype> &res, gentype &resn, const gentype &y, const SparseVector<gentype> &x) const { setInnerWildpa(&x); setWildTargpp(y); dgTrainingVector(res,resn,-1); resetInnerWildp(); return; }

    virtual int gg(double &resg,         const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { setInnerWildpa(&x,xinf); int resi = ggTrainingVector(resg,-1,retaltg,pxyprodx); resetInnerWildp(xinf == NULL); return resi; }
    virtual int gg(Vector<double> &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { setInnerWildpa(&x,xinf); int resi = ggTrainingVector(resg,-1,retaltg,pxyprodx); resetInnerWildp(xinf == NULL); return resi; }
    virtual int gg(d_anion &resg,        const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { setInnerWildpa(&x,xinf); int resi = ggTrainingVector(resg,-1,retaltg,pxyprodx); resetInnerWildp(xinf == NULL); return resi; }

    virtual void dg(Vector<double>          &res, double         &resn, const gentype &y, const SparseVector<gentype> &x) const { (void) y; setInnerWildpa(&x); setWildTargpp(defaultgentype()); dgTrainingVector(res,resn,-1); resetInnerWildp(); return; }
    virtual void dg(Vector<Vector<double> > &res, Vector<double> &resn, const gentype &y, const SparseVector<gentype> &x) const { (void) y; setInnerWildpa(&x); setWildTargpp(defaultgentype()); dgTrainingVector(res,resn,-1); resetInnerWildp(); return; }
    virtual void dg(Vector<d_anion>         &res, d_anion        &resn, const gentype &y, const SparseVector<gentype> &x) const { (void) y; setInnerWildpa(&x); setWildTargpp(defaultgentype()); dgTrainingVector(res,resn,-1); resetInnerWildp(); return; }

    virtual void stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const { setInnerWildpa(&x); stabProbTrainingVector(res,-1,p,pnrm,rot,mu,B); resetInnerWildp(); return; }

    // var and covar functions

    virtual int varTrainingVector(gentype &resv, gentype &resmu, int i, gentype ***pxyprodi = NULL, gentype **pxyprodii = NULL) const { return covTrainingVector(resv,resmu,i,i,pxyprodi,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL, gentype ***pxyprodx = NULL, gentype **pxyprodxx = NULL) const { setInnerWildpa(&xa,xainf); int res = covTrainingVector(resv,resmu,-1,-1,pxyprodx,pxyprodx,pxyprodxx); resetInnerWildp(xainf == NULL); return res; }

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const;
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const;

    // Training data tracking functions:
    //
    // The indexing and type functions are described below.  They give
    // information about the contents of the training set - which features
    // are  used, how often each feature is used, and the type of data in any
    // given feature.  Full description is below.
    //
    // Fucntions to translate to/from sparse form are also defined.

    virtual const Vector<int>          &indKey(void)          const { return indexKey;      }
    virtual const Vector<int>          &indKeyCount(void)     const { return indexKeyCount; }
    virtual const Vector<int>          &dattypeKey(void)      const { return typeKey;       }
    virtual const Vector<Vector<int> > &dattypeKeyBreak(void) const { return typeKeyBreak;  }

    // Other functions
    //
    // setaltx: if non-null, set alternative x source (NULL to reset)
    //          Function is naive - it does not update anything to reflect
    //          possible changes in x from the new source.  Use resetKernel
    //          to propogate such changes manually (with updateInfo = 1).
    // disable: removes influence of points without removing them from the
    //          training set.  Note that this additionally goes through
    //          all altx sources (and so on) and disables x in those as
    //          well.

    virtual void setaltx(const ML_Base *_altxsrc) { incxvernum(); altxsrc = _altxsrc; return; }

    virtual int disable(int i);
    virtual int disable(const Vector<int> &i);

    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Training data information functions (all assume no far/farfar/farfarfar or multivectors)

    virtual const SparseVector<gentype> &xsum      (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xmean     (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xmeansq   (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xsqsum    (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xsqmean   (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xmedian   (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xvar      (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xstddev   (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xmax      (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xmin      (SparseVector<gentype> &res) const;

    // Kernel normalisation function
    // =============================
    //
    // Effectively Normalise the training data (zero mean, unit variance) by
    // setting shifting/scaling in the kernel as follows:
    //
    // xmean   = (1/N) sum_i x_i
    // xmeansq = (1/N) sum_i x_i.^2
    // xvar    = (1/N) sum_i (x_i-xmean).^2
    //         = (1/N) sum_i x_i.^2 + (1/N) sum_i xmean.^2 - (2/N) sum_i x_i.*xmean
    //         = xmeansq + xmean.^2 - 2 xmean.*((1/N) sum_i x_i)
    //         = xmeansq + xmean.^2 - 2 xmean.*xmean
    //         = xmeansq - xmean.^2
    //
    // xshift = -xmean
    // xscale = 1./sqrt(xvar)
    //
    // Individual components may be any one of the types supported by gentype.
    // This includes:
    //
    // - real (int or double)
    // - anion (assume only real, complex, quaternion or octonian)
    // - vector (assume only of real, anion, vector or matrix)
    // - matrix (assume only of real, anion, vector or matrix)
    // - set
    // - dgraph
    // - string
    //
    // Now:
    //
    // - sets, dgraphs and strings cannot be normalised (it makes no sense), so
    //   we need to detect any feature having this type of argument and then
    //   place a 0 in the relevant mean, 1 in the relevant variance
    // - reals evaluate trivially
    //
    // Vectors and matrices are more complicated.  For such "scalars":
    //
    // mean = (1/N) sum_i y_i
    // var  = (1/N) sum_i (y_i-mean).(y_i-mean)'
    //
    // which is an outer product, ' means conjugate transpose.
    //
    // (y_i-mean)  -> A.(y_i-mean)
    // (y_i-mean)' -> (y_i-mean)'.A'
    //
    // so: var -> newvar = A.var.A' = I
    //     var = BB', B = chol(var)
    //
    // choose: A = inv(B)
    // => newvar = I
    //
    // normalisation: y -> A.(y-mean)
    //
    // Treatment of missing features:
    //
    // By default, "missing" features (ie indices present in some vectors but
    // not others) are treated as 0s.  An alternative approach may be selected
    // by setting replaceMissingFeatures = 1.  Under this alternative scheme
    // missing features are treated as not present and replaced (in the ML)
    // by the mean value of this feature for those vectors in which it is
    // present.  This is done prior to calculation of shifting and scaling
    // factors.
    //
    // Unit range: applies to reals/integers/nulls only.  Asserts range of input
    // must lie between 0 (minimum value) and 1 (maximum value).  Thus
    //
    // shift = min(x)
    // scale = 1/(max(x)-min(x))
    //
    // Option: flatnorm: rather than work on a per-feature basis, this sets
    //         the scale to the min scale.
    //         noshift: do not apply shifting, only scaling.

    virtual int normKernelNone                  (void);
    virtual int normKernelZeroMeanUnitVariance  (int flatnorm = 0, int noshift = 0);
    virtual int normKernelZeroMedianUnitVariance(int flatnorm = 0, int noshift = 0);
    virtual int normKernelUnitRange             (int flatnorm = 0, int noshift = 0);

    // Helper functions for sparse variables
    //
    // These functions convert to/from sparse vectors.  It is assumed that
    // indexing in the sparse vectors follows the index key defined for
    // this training set.  makeFullSparse ensures that all indices are
    // present in the sparse vector without setting them (so that they
    // default to zero).
    //
    // Assumptions: this assumes no far/farfar/farfarfar elements and no [ ... ~ ... ] style multi-vectors

    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<gentype>      &src) const;
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<double>       &src) const;
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const SparseVector<double> &src) const;

    virtual Vector<gentype> &xlateFromSparse(Vector<gentype> &dest, const SparseVector<gentype> &src) const;
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<gentype> &src) const;
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<double>  &src) const;
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<gentype>       &src) const;
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<double>        &src) const;

    virtual Vector<double>  &xlateFromSparseTrainingVector(Vector<double>  &dest, int i) const { return xlateFromSparse(dest,x(i)); }
    virtual Vector<gentype> &xlateFromSparseTrainingVector(Vector<gentype> &dest, int i) const { return xlateFromSparse(dest,x(i)); }

    virtual SparseVector<gentype> &makeFullSparse(SparseVector<gentype> &dest) const;

    // x detangling
    //
    // x "vectors" can get a little confusing, as they can refer to "normal" vectors, 
    // rank constraints, gradient constraints and other things.  To keep it together
    // the following function disentangles it all.  Given i, xx and xxinfo this function
    // returns the following:
    //
    // 0: xnear{info} points to x vector {info}, inear is index (xnear always set)
    // 1: xfar{info} and ifar refer to "other side" of rank constraint
    // 2: xfarfar refers to direction of gradient constraint, gradOrder > 0
    // 4: gradOrder > 0, but xfarfar not set
    // 8: treat distributions as sample from sets, allowing whole-set constraints
    // 16: idiagr set
    // 3,5,9,10,11,12,13,16: allowable combinations
    // NOT ANY MORE: -1: idiagr set (simple version only)
    //
    // For tuple disambiguation will detect and set ineartup, ifartup non-NULL if found.
    // inear/ifar, xnear/xfar, xnearinfo/xfarinfo not to be trusted if ineartup/ifartup set non-NULL
    //
    // Notes:
    //
    // - If xx (and xxinfo) are non-NULL the result is always 0, xnear{info} = xx{info}
    // - if iokr set then kernel evaluation should be scaled by UU product with index iok.
    // - xnear{info}, xfar{info}, xfarfar are never left NULL.
    // - case 3 cannot be properly dealt with at present
    // - usextang: set if you want to use precalculated xtang, otherwise calculate from scratch

    virtual int detangle_x(int i, int usextang = 0) const
    {
        //xx     = xx     ? xx     : &xgetloc(i);
        //xxinfo = xxinfo ? xxinfo : &locxinfo(i);

        int res = 0;

        if ( !usextang )
        {
            const SparseVector<gentype> *xx     = &xgetloc(i);
            const vecInfo               *xxinfo = &locxinfo(i);

            const SparseVector<gentype> *xnear   = NULL;
            const SparseVector<gentype> *xfar    = NULL;
            const SparseVector<gentype> *xfarfar = NULL;
            const vecInfo *xnearinfo = NULL;
            const vecInfo *xfarinfo  = NULL;
            int inear = 0;
            int ifar  = 0;
            const gentype *ineartup = NULL;
            const gentype *ifartup  = NULL;
            int iokr   = 0;
            int iok    = 0;
            int idiagr = 0;
            int igradOrder = 0;
            int iplanr = 0;
            int iplan  = 0;
            int iset   = 0;
            int ilr;
            int irr;
            int igr;

            SparseVector<gentype> *xuntang = NULL;
            vecInfo *xinfountang = NULL;

            //Final 0 here suppresses allocation of xuntang/xuntanginfo 
            res = detangle_x(xuntang,xinfountang,xnear,xfar,xfarfar,xnearinfo,xfarinfo,inear,ifar,ineartup,ifartup,ilr,irr,igr,iokr,iok,i,idiagr,xx,xxinfo,igradOrder,iplanr,iplan,iset,usextang,0);
        }

        else
        {
            res = locxtang(i);
        }

        return res;
        //return idiagr ? -1 : res;
    }

    // if the vector itself needs to change (redirection of xnear/xfar etc) then xuntang will be allocated and pointed to
    // otherwise it will just be NULL.

    virtual int detangle_x(SparseVector<gentype> *&xuntang, vecInfo *&xinfountang,
                   const SparseVector<gentype> *&xnear, const SparseVector<gentype> *&xfar, const SparseVector<gentype> *&xfarfar, 
                   const vecInfo *&xnearinfo, const vecInfo *&xfarinfo, 
                   int &inear, int &ifar, const gentype *&ineartup, const gentype *&ifartup,
                   int &ilr, int &irr, int &igr, 
                   int &iokr, int &iok,
                   int i, int &idiagr, const SparseVector<gentype> *xx, const vecInfo *xxinfo, int &gradOrder, 
                   int &iplanr, int &iplan, int &iset, int usextang = 1, int allocxuntangifneeded = 1) const
    {
        NiceAssert( xx ); //&& xxinfo );

        int methodkey = 0;

        // Base references (no indirection yet)

        const SparseVector<gentype> &xib = *xx;

        xxinfo = xxinfo ? xxinfo : &xinfo(i);

        xnearinfo = xxinfo;
        xfarinfo  = xxinfo;

        (void) usextang;

        {
            // ilr:    is (leftside  of rank) index redirected?
            // irr:    is (rightside of rank) index redirected?
            // igr:    is gradient index redirected?
            // iokr:   is output kernel invoked?
            // idiagr: is diagonal kernel bypass invoked?
            // itup:   is il or ir a tuple?
            // iset:   are distributions treated as distributions (0) or samples from set(s) (1)

            int z = 0;
            int ind0present = xib.isfarfarfarindpresent(z) && !(xib(z).isValNull());
            int ind1present = xib.isfarfarfarindpresent(1) && !(xib(1).isValNull());
            int ind2present = xib.isfarfarfarindpresent(2) && !(xib(2).isValNull());
            int ind3present = xib.isfarfarfarindpresent(3) && !(xib(3).isValNull());
            int ind4present = xib.isfarfarfarindpresent(4) && !(xib(4).isValNull());
//            int ind5present = xib.isfarfarfarindpresent(5) && !(xib(5).isValNull());
            int ind6present = xib.isfarfarfarindpresent(6) && !(xib(6).isValNull());
            int ind7present = xib.isfarfarfarindpresent(7) && !(xib(7).isValNull());
            int ind8present = xib.isfarfarfarindpresent(8) && !(xib(8).isValNull());

            ilr     = ind0present ? 1 : 0;
            irr     = ind1present ? 1 : 0;
            igr     = ind2present ? 1 : 0;
            iokr    = ind3present ? 1 : 0;
            idiagr  = ind4present ? 1 : 0;
            iplanr  = ind7present ? 1 : 0;
            iset    = ind8present ? 1 : 0;

            ineartup = NULL;
            ifartup  = NULL;

            if ( ilr && (xib.fff(0)).isValVector() )
            {
                ineartup = &xib.fff(0);
            }

            if ( irr && (xib.fff(1)).isValVector() )
            {
                ifartup = &xib.fff(1);
            }

            // il:  (leftside of rank) index
            // ir:  (rightside of rank) index
            // ig:  gradient index
            // iok: basis references for output kernel

            int il    = ilr && !ineartup ? ( (int) xib.fff(0) ) : i;
            int ir    = irr && !ifartup  ? ( (int) xib.fff(1) ) : i;
            int ig    = igr ? ( (int) xib.fff(2) ) : i;
                iok   = ( iokr   && (xib.fff(3)).isValInteger() ) ? ( (int) xib.fff(3) ) : -1;
                iplan = ( iplanr && (xib.fff(7)).isValInteger() ) ? ( (int) xib.fff(7) ) : -1;

            // ilfar:    is (leftside  of rank) a far reference?
            // irfar:    is (rightside of rank) a far reference?
            // igfarfar: is gradient a farfar reference?

            int ilfar    = 0;
            int irfar    = ( !irr && xib.isfaroffindpresent()    ) ? 1 : 0;
            int igfarfar = ( !igr && xib.isfarfaroffindpresent() ) ? 2 : 0;

            // gradient order calculations

            gradOrder = ind6present ? ( (int) xib.fff(6) ) : ( ( igfarfar || igr ) ? 1 : 0 );

            // What are the method keys for indexes?

            methodkey = ( ( irfar || irr ) ? 1 : 0 ) 
                      | ( (  ( igfarfar || igr ) && ( gradOrder > 0 ) ) ? 2 : 0 ) 
                      | ( ( !( igfarfar || igr ) && ( gradOrder > 0 ) ) ? 4 : 0 )
                      | ( iset ? 8 : 0 )
                      | ( idiagr ? 16 : 0 );

            // NB: idiagr over-rides all other options here.

            if ( !idiagr )
            {
                // xnear:   (leftside  of rank) vector references 
                // xfar:    (rightside of rank) vector references
                // xfarfar: gradient vector references
                //
                // Notes: - ternary operator short-circuits, so only relevant branch evaluated

                const SparseVector<gentype> &xxl = ( i == il ) ? *xx : xgetloc(il);
                const SparseVector<gentype> &xxr = ( i == ir ) ? *xx : xgetloc(ir);
                const SparseVector<gentype> &xxg = ( i == ig ) ? *xx : xgetloc(ig);

                xnear   = ( !methodkey ? &(xxl.nearref()) : ( ilfar    ? &(xxl.farref())    : &(xxl.nearref()) ) );
                xfar    = ( !methodkey ? &(xxr.nearref()) : ( irfar    ? &(xxr.farref())    : &(xxr.nearref()) ) );
                xfarfar = ( !methodkey ? &(xxg.nearref()) : ( igfarfar ? &(xxg.farfarref()) : &(xxg.nearref()) ) );

                // xnearinfo: (leftside  of rank) information
                // xfarinfo:  (rightside of rank) information

                xnearinfo = ( i == il ) ? xxinfo : &(xinfo(il));
                xfarinfo  = ( i == ir ) ? xxinfo : &(xinfo(ir));

                xnearinfo = &((*xnearinfo)(0,-1));
                xfarinfo  = &((*xfarinfo )(1,-1));

                // (leftside  of rank) index rename and recalc
                // (rightside of rank) index rename and recalc

                inear = il;
                ifar  = -(((ir+1)*100)+1);
            }
        }

        xuntang     = NULL;
        xinfountang = NULL;

        if ( allocxuntangifneeded && !idiagr && ( ineartup && ifartup ) )
        {
            int q;

            // Process indirections

            xuntang     = new SparseVector<gentype>(xib);
            xinfountang = new vecInfo;

                         (*xuntang).zeronear();
                         (*xuntang).zerofar();
            if ( igr ) { (*xuntang).overwritefarfar(*xfarfar); }

            const Vector<gentype> &iain = (*ineartup).cast_vector();
            const Vector<gentype> &iaif = (*ifartup).cast_vector();

            int iains = iain.size();
            int iaifs = iaif.size();

            for ( q = 0 ; q < iains ; q++ )
            {
                (*xuntang).overwritenear(xgetloc((int) iain(q)),q);
            }

            for ( q = 0 ; q < iaifs ; q++ )
            {
                (*xuntang).overwritefar(xgetloc((int) iaif(q)),q);
            }

            (**thisthisthis).getKernel().getvecInfo((*xinfountang),(*xuntang));
        }

        else if ( allocxuntangifneeded && !idiagr && ( ineartup && irr ) )
        {
            int q;

            // Process indirections

            xuntang     = new SparseVector<gentype>(xib);
            xinfountang = new vecInfo;

                         (*xuntang).zeronear();
            if ( ilr ) { (*xuntang).overwritefar(*xfar);       }
            if ( igr ) { (*xuntang).overwritefarfar(*xfarfar); }

            const Vector<gentype> &iain = (*ineartup).cast_vector();

            int iains = iain.size();

            for ( q = 0 ; q < iains ; q++ )
            {
                (*xuntang).overwritenear(xgetloc((int) iain(q)),q);
            }

            (**thisthisthis).getKernel().getvecInfo((*xinfountang),(*xuntang));
        }

        else if ( allocxuntangifneeded && !idiagr && ( ilr && ifartup ) )
        {
            int q;

            // Process indirections

            xuntang     = new SparseVector<gentype>(xib);
            xinfountang = new vecInfo;

            if ( ilr ) { (*xuntang).overwritenear(*xnear);     }
                         (*xuntang).zerofar();
            if ( igr ) { (*xuntang).overwritefarfar(*xfarfar); }

            const Vector<gentype> &iaif = (*ifartup).cast_vector();

            int iaifs = iaif.size();

            for ( q = 0 ; q < iaifs ; q++ )
            {
                (*xuntang).overwritefar(xgetloc((int) iaif(q)),q);
            }

            (**thisthisthis).getKernel().getvecInfo((*xinfountang),(*xuntang));
        }

        else if ( allocxuntangifneeded && !idiagr && ( ilr || irr || igr ) )
        {
            // Process indirections

            xuntang     = new SparseVector<gentype>(xib);
            xinfountang = new vecInfo;

            if ( ilr ) { (*xuntang).overwritenear(*xnear);     }
            if ( irr ) { (*xuntang).overwritefar(*xfar);       }
            if ( igr ) { (*xuntang).overwritefarfar(*xfarfar); }

            (**thisthisthis).getKernel().getvecInfo((*xinfountang),(*xuntang));
        }

        return methodkey;
    }

 



protected:

    // ONN precursor:
    //
    // We keep this as a placeholder.  The weight is actually required as
    // index -2 when evaluating kernels.

    virtual const SparseVector<gentype> &W(void) const { throw("Can't used index -2 in non-ONN type W.");     const static SparseVector<gentype> dummy; return dummy; }
    virtual const vecInfo &getWinfo(void)        const { throw("Can't used index -2 in non-ONN type Winfo."); const static vecInfo               dummy; return dummy; }
    virtual int getWtang(void)                   const { throw("Can't used index -2 in non-ONN type Wtang.");                                           return -1;    }

    // Kernel cache access
    //
    // If a kernel cache is attached to this then this will set isgood = 0 if
    // the value is not in the cache, otherwise isgood = 1.

    virtual double getvalIfPresent(int numi, int numj, int &isgood) const;

    // Inner-product cache: over-write this with a non-NULL return in classes where
    // a kernel cache is available

    virtual const Matrix<double> *getxymat(void) const { return NULL; }


private:

    virtual int xvecdim(const SparseVector<gentype> &xa) const
    {
        int xdm = 0;

        if ( ( xa.nearupsize() > 1 ) || ( xa.farupsize() > 1 ) || xa.isfaroffindpresent() || xa.isfarfaroffindpresent() || xa.isfarfarfaroffindpresent() )
        {
            int i,s;
            int dim = 0;

            s = xa.nearupsize();

            for ( i = 0 ; i < s ; i++ )
            {
                dim = xa.nearrefupsize(i);
                xdm = ( dim > xdm ) ? dim : xdm;
            }

            s = xa.farupsize();

            for ( i = 0 ; i < s ; i++ )
            {
                dim = xa.farrefupsize(i);
                xdm = ( dim > xdm ) ? dim : xdm;
            }
        }

        else if ( xa.nearindsize() )
        {
            xdm = xa.nearrefupsize(0);
        }

        return xdm;
    }

    // Templated to limit code redundancy

    template <class T> T &K0(T &res, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, int resmode) const;
    template <class T> T &K1(T &res, int ia, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xa, const vecInfo *xainfo, int resmode) const;
    template <class T> T &K2(T &res, int ia, int ib, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const vecInfo *xainfo, const vecInfo *xbinfo, int resmode) const;
    template <class T> T &K3(T &res, int ia, int ib, int ic, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, int resmode) const;
    template <class T> T &K4(T &res, int ia, int ib, int ic, int id, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, int resmode) const;
    template <class T> T &Km(int m, T &res, Vector<int> &ii, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, Vector<const SparseVector<gentype> *> *xx, Vector<const vecInfo *> *xxinfo, int resmode) const;

    virtual double &KK0ip(       double &res, const double &bias, const gentype **pxyprod) const;
    virtual double &KK1ip(       double &res, int ia, const double &bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const vecInfo *xainfo) const;
    virtual double &KK2ip(       double &res, int ia, int ib, const double &bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const vecInfo *xainfo, const vecInfo *xbinfo) const;
    virtual double &KK3ip(       double &res, int ia, int ib, int ic, const double &bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo) const;
    virtual double &KK4ip(       double &res, int ia, int ib, int ic, int id, const double &bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo) const;
    virtual double &KKmip(int m, double &res, Vector<int> &i, const double &bias, const gentype **pxyprod, Vector<const SparseVector<gentype> *> *xx, Vector<const vecInfo *> *xxinfo) const;

    template <class T>
    void dK(T &xygrad, T &xnormgrad, int i, int j, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo, int deepDeriv) const;
    template <class T>
    void dK2delx(T &xscaleres, T &yscaleres, int &minmaxind, int i, int j, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const;

    template <class T>
    void d2K(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, int i, int j, const T &bias, const MercerKernel &altK, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const;
    template <class T>
    void d2K2delxdelx(T &xxscaleres, T &yyscaleres, T &xyscaleres, T & yxscaleres, T &constres, int &minmaxind, int i, int j, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const;
    template <class T>
    void d2K2delxdely(T &xxscaleres, T &yyscaleres, T &xyscaleres, T & yxscaleres, T &constres, int &minmaxind, int i, int j, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const;

    template <class T>
    void dnK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const;

    // Base data

    Vector<SparseVector<gentype> > allxdatagent;
//FIXME    Vector<const SparseVector<gentype> *> allxdatagentp;
    MercerKernel kernel;
    gentype ytargdata;
    Vector<gentype> alltraintarg;
    Vector<vecInfo> traininfo;
    Vector<int> traintang;
//FIXME    Vector<const vecInfo *> traininfop;
    Vector<int> xd;
    Vector<double> xCweight;
    Vector<double> xCweightfuzz;
    Vector<double> xsigmaweightonfly;
    Vector<double> xepsweight;
    Vector<int> xalphaState;
    Matrix<gentype> K2mat;

    int xdzero; // number of elements in xd that are zero

    int xpreallocsize;

    // Output kernel

    MercerKernel UUoutkernel;

    // isBasisUser: controls if basis is user controlled (0, default) or fixed to equal y (ie target set).
    // defbasis: -1 if not defined, otherwise default basis onto which projection is done (assumed unit)
    // locbasis: basis

    int isBasisUserUU;
    int defbasisUU;
    Vector<gentype> locbasisUU;

    int isBasisUserVV;
    int defbasisVV;
    Vector<gentype> locbasisVV;

    gentype &(*UUcallback)(gentype &res, int m, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);
    const gentype &(*VVcallback)(gentype &res, int m, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);

    // Not used in older classes that implemented zerotol locally

    double globalzerotol;

    // Temporary store so that callback knows what data element -1 is

    const SparseVector<gentype> *wildxgenta;
    const SparseVector<gentype> *wildxgentb;
    const SparseVector<gentype> *wildxgentc;
    const SparseVector<gentype> *wildxgentd;

    vecInfo *wildxinfoa;
    vecInfo *wildxinfob;
    vecInfo *wildxinfoc;
    vecInfo *wildxinfod;

    int wildxtanga;
    int wildxtangb;
    int wildxtangc;
    int wildxtangd;

    int wildxdima;
    int wildxdimb;
    int wildxdimc;
    int wildxdimd;

    const Vector<SparseVector<gentype> > *wildxxgent;
    Vector<vecInfo> wildxxinfo;
    Vector<int> wildxxtang;
    int wildxxdim;

    int wildxdim;

    // Training data information:
    //
    // indexKey: which indices (in the sparse training vectors) are in use.
    // indexKeyCount: how often each index is used.
    // typeKey: what data type is in each index
    // typeKeyBreak: a detailed breakdown of the number of vectors in each type.
    //
    // isIndPrune: set 1 to indicate that all vectors should have the same
    //             index (null full as required) and that indices that only
    //             have nulls should be removed.
    //
    // typeKey: - 0:  mixture of the below categories
    //          - 1:  all null
    //          - 2:  all binary integer 0/1 (or null (0))
    //          - 3:  all integers (or null (0))
    //          - 4:  all doubles (or integers or null)
    //          - 5:  all anions (or doubles or integers or null)
    //          - 6:  all vectors of doubles
    //          - 7:  all matrices of doubles
    //          - 8:  all sets
    //          - 9:  all dgraphs
    //          - 10: all strings
    //          - 11: all equations
    //
    // typeKeyBreak: - 0:  sum of the below categories
    //               - 1:  null
    //               - 2:  binary integer 1
    //               - 3:  integer not one
    //               - 4:  double
    //               - 5:  anion
    //               - 6:  vector
    //               - 7:  matrix
    //               - 8:  set
    //               - 9:  dgraph
    //               - 10: string
    //               - 11: equations

    Vector<int> indexKey;
    Vector<int> indexKeyCount;
    Vector<int> typeKey;
    Vector<Vector<int> > typeKeyBreak;

    // Each ML_Base instantiated has a unique ID, and corresponding to
    // that ID are x and g version numbers (see above).  These are 
    // shared.

    svmvolatile static SparseVector<int>* xvernumber;
    svmvolatile static SparseVector<int>* gvernumber;
    svmvolatile static svm_mutex mleyelock;

    // indPrune: 0 by default, 1 to indicate that x should be "filled out"
    //           with nulls so that each training vector has the same
    //           index vector.  This will also cause pruning of indexes
    //           that contain only nulls.
    // xassumedconsist: 0 by default, 1 to indicate that x can safely be assumed to
    //           be consistent throughout (ie all sparse vectors have the
    //           same index vector)

    int isIndPrune;
    int xassumedconsist;
    int xconsist;

    void recalcIndTypFromScratch(void);
    virtual int indPrune(void) const { NiceAssert( ( isIndPrune == 0 ) || ( isIndPrune == 1 ) ); return isIndPrune; }

    // unfillIndex: Call this function to remove an index from all vectors
    //           in the training set.  Note that this does not actually
    //           remove the index itself (that is the job of the caller).
    // fillIndex: Call this function to ensure an index is present in all
    //           training vectors.  Note that index must be present before
    //           calling.

    void unfillIndex(int i);
    void fillIndex(int i);

    // Functions to update index information for addition/removal

    void addToIndexKeyAndUpdate(const SparseVector<gentype> &newz);
    void removeFromIndexKeyAndUpdate(const SparseVector<gentype> &oldz);

    // Returns the appropriate index type corresponding to variable y

    int gettypeind(const gentype &y) const;

    // Alternate x source (NULL if data local)

    const ML_Base *altxsrc;

    ML_Base **that;

    // Direct y access

    virtual Vector<gentype> &y_unsafe(void) { return alltraintarg; }

    // Fixes x pointer vector

    void fixpvects(void)
    {
return;
//FIXME
/*
        if ( allxdatagent.size() )
        {
            int i;

            allxdatagentp.resize(allxdatagent.size());

            for ( i = 0 ; i < allxdatagent.size() ; i++ )
            {
                allxdatagentp("&",i) = &allxdatagent(i);
            }
        }

        if ( traininfo.size() )
        {
            int i;

            traininfop.resize(traininfo.size());

            for ( i = 0 ; i < traininfo.size() ; i++ )
            {
                traininfop("&",i) = &traininfo(i);
            }
        }

        return;
*/
    }

    // Test for x consistency (same indices for all)

    int testxconsist(void)
    {
        if ( indKeyCount().size() )
        {
            return indKeyCount() == indKeyCount()(0);
        }

        return 1;
    }

    // Functions to control wilds

    virtual void setInnerWildpa(const SparseVector<gentype> *xl, const vecInfo *xinf = NULL) const
    {
        const_cast<SparseVector<gentype> &>(*xl).makealtcontent();

        (**thisthisthis).wildxgenta = xl;

        if ( ( xinf == NULL ) && ( type() == 216 ) )
        {
            // For BLK_Batter a *lot* of data could be in x and xinfo is not used, so don't waste time here!

            MEMNEW((**thisthisthis).wildxinfoa,vecInfo);
        }

        else if ( xinf == NULL )
        {
            MEMNEW((**thisthisthis).wildxinfoa,vecInfo);
            (**thisthisthis).getKernel().getvecInfo(*((**thisthisthis).wildxinfoa),*xl);
        }

        else
        {
            MEMNEW((**thisthisthis).wildxinfoa,vecInfo(*xinf));
        }

        (**thisthisthis).wildxdima = xvecdim(*wildxgenta);

        (**thisthisthis).wildxdim = 0;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdima > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdima : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimb > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimb : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimc > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimc : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimd > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimd : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxxdim > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxxdim : (**thisthisthis).wildxdim;

        (**thisthisthis).wildxtanga = detangle_x(-1);
        (**thisthisthis).wildxaReal = isxreal(-1);

        if ( !wildxaReal )
        {
            calcSetAssumeReal(0);
        }

        return;
    }

    virtual void setInnerWildpb(const SparseVector<gentype> *xl, const vecInfo *xinf = NULL) const
    {
        const_cast<SparseVector<gentype> &>(*xl).makealtcontent();

        (**thisthisthis).wildxgentb = xl;

        if ( xinf == NULL )
        {
            MEMNEW((**thisthisthis).wildxinfob,vecInfo);
            (**thisthisthis).getKernel().getvecInfo(*((**thisthisthis)).wildxinfob,*xl);
        }

        else
        {
            MEMNEW((**thisthisthis).wildxinfob,vecInfo(*xinf));
        }

        (**thisthisthis).wildxdimb = xvecdim(*wildxgentb);

        (**thisthisthis).wildxdim = 0;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdima > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdima : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimb > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimb : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimc > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimc : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimd > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimd : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxxdim > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxxdim : (**thisthisthis).wildxdim;

        (**thisthisthis).wildxtangb = detangle_x(-3);
        (**thisthisthis).wildxbReal = isxreal(-3);

        if ( !wildxbReal )
        {
            calcSetAssumeReal(0);
        }

        return;
    }

    virtual void setInnerWildpc(const SparseVector<gentype> *xl, const vecInfo *xinf = NULL) const
    {
        const_cast<SparseVector<gentype> &>(*xl).makealtcontent();

        (**thisthisthis).wildxgentc = xl;

        if ( xinf == NULL )
        {
            MEMNEW((**thisthisthis).wildxinfoc,vecInfo);
            (**thisthisthis).getKernel().getvecInfo(*((**thisthisthis)).wildxinfoc,*xl);
        }

        else
        {
            MEMNEW((**thisthisthis).wildxinfoc,vecInfo(*xinf));
        }

        (**thisthisthis).wildxdimc = xvecdim(*wildxgentc);

        (**thisthisthis).wildxdim = 0;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdima > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdima : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimb > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimb : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimc > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimc : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimd > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimd : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxxdim > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxxdim : (**thisthisthis).wildxdim;

        (**thisthisthis).wildxtangc = detangle_x(-4);
        (**thisthisthis).wildxcReal = isxreal(-4);

        if ( !wildxcReal )
        {
            calcSetAssumeReal(0);
        }

        return;
    }

    virtual void setInnerWildpd(const SparseVector<gentype> *xl, const vecInfo *xinf = NULL) const
    {
        const_cast<SparseVector<gentype> &>(*xl).makealtcontent();

        (**thisthisthis).wildxgentd = xl;

        if ( xinf == NULL )
        {
            MEMNEW((**thisthisthis).wildxinfod,vecInfo);
            (**thisthisthis).getKernel().getvecInfo(*((**thisthisthis)).wildxinfod,*xl);
        }

        else
        {
            MEMNEW((**thisthisthis).wildxinfod,vecInfo(*xinf));
        }

        (**thisthisthis).wildxdimd = xvecdim(*wildxgentd);

        (**thisthisthis).wildxdim = 0;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdima > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdima : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimb > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimb : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimc > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimc : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimd > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimd : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxxdim > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxxdim : (**thisthisthis).wildxdim;

        (**thisthisthis).wildxtangd = detangle_x(-5);
        (**thisthisthis).wildxdReal = isxreal(-5);

        if ( !wildxdReal )
        {
            calcSetAssumeReal(0);
        }

        return;
    }

    virtual void setInnerWildpx(const Vector<SparseVector<gentype> > *xl) const
    {
        int i;

        (**thisthisthis).wildxxgent = xl;
        ((**thisthisthis).wildxxinfo).resize((*xl).size());
        ((**thisthisthis).wildxxtang).resize((*xl).size());

        for ( i = 0 ; i < (*xl).size() ; i++ )
        {
            const_cast<SparseVector<gentype> &>((*xl)(i)).makealtcontent();

            (**thisthisthis).getKernel().getvecInfo(((**thisthisthis).wildxxinfo)("&",i),(*xl)(i));
        }

        (**thisthisthis).wildxxdim = 0;

        if ( (*wildxxgent).size() )
        {
            int dimx;

            for ( i = 0 ; i < (*wildxxgent).size() ; i++ )
            {
                dimx = xvecdim((*wildxxgent)(i));
                (**thisthisthis).wildxxdim = ( (**thisthisthis).wildxxdim > dimx ) ? (**thisthisthis).wildxxdim : dimx;
            }
        }

        (**thisthisthis).wildxdim = 0;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdima > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdima : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimb > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimb : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimc > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimc : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxdimd > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxdimd : (**thisthisthis).wildxdim;
        (**thisthisthis).wildxdim = ( (**thisthisthis).wildxxdim > (**thisthisthis).wildxdim ) ? (**thisthisthis).wildxxdim : (**thisthisthis).wildxdim;

        (**thisthisthis).wildxxReal = 1;

        for ( i = 0 ; i < (*xl).size() ; i++ )
        {
            ((**thisthisthis).wildxxtang)("&",i) = detangle_x(-100*(i+1));
            (**thisthisthis).wildxxReal = ( wildxxReal && isxreal(-100*(i+1)) );
        }

        if ( !wildxxReal )
        {
            calcSetAssumeReal(0);
        }

        return;
    }

    virtual void setWildTargpp(const gentype &yI) const
    {
        (**thisthisthis).ytargdata = yI;

        return;
    }

    virtual void resetInnerWildp(int wasnulla = 0, int wasnullb = 0, int wasnullc = 0, int wasnulld = 0) const
    {
        (**thisthisthis).wildxgenta = NULL;
        (**thisthisthis).wildxgentb = NULL;
        (**thisthisthis).wildxgentc = NULL;
        (**thisthisthis).wildxgentd = NULL;
        (**thisthisthis).wildxxgent = NULL;

        if ( wildxinfoa && wasnulla )
        {
            MEMDEL((**thisthisthis).wildxinfoa);
        }

        if ( wildxinfob && wasnullb )
        {
            MEMDEL((**thisthisthis).wildxinfob);
        }

        if ( wildxinfoc && wasnullc )
        {
            MEMDEL((**thisthisthis).wildxinfoc);
        }

        if ( wildxinfod && wasnulld )
        {
            MEMDEL((**thisthisthis).wildxinfod);
        }

        (**thisthisthis).wildxinfoa = NULL;
        (**thisthisthis).wildxinfob = NULL;
        (**thisthisthis).wildxinfoc = NULL;
        (**thisthisthis).wildxinfod = NULL;

        (**thisthisthis).wildxdim = 0;

        (**thisthisthis).wildxdima = 0;
        (**thisthisthis).wildxdimb = 0;
        (**thisthisthis).wildxdimc = 0;
        (**thisthisthis).wildxdimd = 0;
        (**thisthisthis).wildxxdim = 0;

        (**thisthisthis).wildxaReal = 1;
        (**thisthisthis).wildxbReal = 1;
        (**thisthisthis).wildxcReal = 1;
        (**thisthisthis).wildxdReal = 1;
        (**thisthisthis).wildxxReal = 1;

        calcSetAssumeReal(0);

        //((**thisthisthis).wildxxinfo).resize(0); - slight speedup by not resizing as the same size is often repeated

        return;
    }


    // Local x retrieval function

    virtual const SparseVector<gentype> &xgetloc(int i) const
    {
        if ( i >= 0 )
        {
            return altxsrc ? (*altxsrc).allxdatagent(i) : allxdatagent(i);
        }

        else if ( i == -1 )
        {
            // Testing vector

            return *wildxgenta;
        }

        else if ( i == -2 )
        {
            // ONN weight vector (error-throwing placeholder unless ONN type).

            return W();
        }

        else if ( i == -3 )
        {
            // Testing vector

            return *wildxgentb;
        }

        else if ( i == -4 )
        {
            // Testing vector

            return *wildxgentc;
        }

        else if ( i == -5 )
        {
            // Testing vector

            return *wildxgentd;
        }

        else if ( ( i <= -100 ) && !((-i)%100) )
        {
            // Testing vector
            //
            // -100 -> 0
            // -200 -> 1
            // -300 -> 2
            // -400 -> 3
            //   ...

            return (*wildxxgent)((-(i+100))/100);
        }

        else if ( ( i <= -100 ) && !((-(i+1))%100) )
        {
            // faroff part to be used (but return as usual)
            //
            // -101 -> 0
            // -201 -> 1
            // -301 -> 2
            // -401 -> 3
            //   ...

            return xgetloc((-(i+101))/100);
        }

        throw("Error: xgetloc index not valid");

        const static SparseVector<gentype> temp;

        return temp;
    }

    virtual int locxtang(int i) const
    { 
        if ( i >= 0 )
        {
            return xtang()(i);
        }

        else if ( i == -1 )
        {
            return wildxtanga;
        }

        else if ( i == -2 )
        {
            return getWtang();
        }

        else if ( i == -3 )
        {
            return wildxtangb;
        }

        else if ( i == -4 )
        {
            return wildxtangc;
        }

        else if ( i == -5 )
        {
            return wildxtangd;
        }

        else if ( ( i <= -100 ) && !((-i)%100) )
        {
            // Testing vector
            //
            // -100 -> 0
            // -200 -> 1
            // -300 -> 2
            // -400 -> 3
            //   ...

            return wildxxtang((-(i+100))/100);
        }

        else if ( ( i <= -100 ) && !((-(i+1))%100) )
        {
            // faroff part to be used (but return as usual)
            //
            // -101 -> 0
            // -201 -> 1
            // -301 -> 2
            // -401 -> 3
            //   ...

            return locxtang((-(i+101))/100);
        }

        throw("Error: x info for invalid index requested");

        return -1;
    }

    virtual const vecInfo &locxinfo(int i) const
    {
        if ( i >= 0 )
        {
            return xinfo()(i);
        }

        else if ( i == -1 )
        {
            return *wildxinfoa;
        }

        else if ( i == -2 )
        {
            return getWinfo();
        }

        else if ( i == -3 )
        {
            return *wildxinfob;
        }

        else if ( i == -4 )
        {
            return *wildxinfoc;
        }

        else if ( i == -5 )
        {
            return *wildxinfod;
        }

        else if ( ( i <= -100 ) && !((-i)%100) )
        {
            // Testing vector
            //
            // -100 -> 0
            // -200 -> 1
            // -300 -> 2
            // -400 -> 3
            //   ...

            return wildxxinfo((-(i+100))/100);
        }

        else if ( ( i <= -100 ) && !((-(i+1))%100) )
        {
            // faroff part to be used (but return as usual)
            //
            // -101 -> 0
            // -201 -> 1
            // -301 -> 2
            // -401 -> 3
            //   ...

            return locxinfo((-(i+101))/100);
        }

        throw("Error: x info for invalid index requested");

        const static vecInfo temp;

        return temp;
    }

    int assumeReal;
    int trainingDataReal;
    int wildxaReal;
    int wildxbReal;
    int wildxcReal;
    int wildxdReal;
    int wildxxReal;

    int isxreal(int i) const
    {
        int res = 1;
        int j,k;

        const SparseVector<gentype> &xx = xgetloc(i);

        if ( xx.indsize() )
        {
            for ( j = 0 ; j < xx.indsize() ; j++ )
            {
                k = gettypeind(xx.direcref(j));

                if ( !( ( k >= 2 ) && ( k <= 4 ) ) )
                {
                    res = 0;

                    break;
                }
            }
        }

        return res;
    }

    void calcSetAssumeReal(int fulltest = 1, int assumeUnreal = 0) const
    {
        if ( fulltest )
        {
            if ( dattypeKey().size() == 0 )
            {
                (**thisthisthis).trainingDataReal = 1;
            }

            else if ( ( dattypeKey() <= 4 ) && ( dattypeKey() >= 2 ) )
            {
                (**thisthisthis).trainingDataReal = 1;
            }

            else
            {
                (**thisthisthis).trainingDataReal = 0;
            }
        }

        if ( !assumeReal && ( !assumeUnreal && trainingDataReal && wildxaReal && wildxbReal && wildxcReal && wildxdReal && wildxxReal ) )
        {
            (**thisthisthis).assumeReal = 1;
        }

        else if ( assumeReal && !( !assumeUnreal && trainingDataReal && wildxaReal && wildxbReal && wildxcReal && wildxdReal && wildxxReal ) )
        {
            (**thisthisthis).assumeReal = 0;
        }

        return;
    }

    ML_Base *thisthis;
    ML_Base **thisthisthis;
};

inline void mProdPt(double &res, int m, int *x)
{
    NiceAssert( m > 0 );

    kernPrecursor *altres = NULL;

    static ML_Base locML;

    int ires = locML.getaltML(altres,x[0]);

    (void) ires;
    NiceAssert( !ires );

    ML_Base &tired = dynamic_cast<ML_Base &>(*altres);

    tired.mProdPt(res,m-1,x+1);

    return;
}

inline void qswap(ML_Base &a, ML_Base &b)
{
    a.qswapinternal(b);

    return;
}

inline ML_Base &setzero(ML_Base &a)
{
    a.restart();

    return a;
}

inline std::ostream &operator<<(std::ostream &output, const ML_Base &src)
{
    return src.printstream(output,0);
}

inline std::istream &operator>>(std::istream &input, ML_Base &dest)
{
    return dest.inputstream(input);
}

inline ML_Base *&setzero(ML_Base *&x)
{
    return ( x = NULL );
}

inline const ML_Base *&setzero(const ML_Base *&x)
{
    return ( x = NULL );
}

inline void qswap(ML_Base *&a, ML_Base *&b)
{
    ML_Base *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}

inline int isQswapCompat(const ML_Base &a, const ML_Base &b)
{
    if ( ( a.isPool()    == b.isPool()    ) &&
         ( a.isMutable() == b.isMutable() ) &&
         ( a.isPool()    || a.isMutable() )    )
    {
        return 1;
    }

    return ( a.type()    == b.type()    ) &&
           ( a.subtype() == b.subtype() );
}

inline int isSemicopyCompat(const ML_Base &a, const ML_Base &b)
{
    return ( a.type()    == b.type()    ) &&
           ( a.subtype() == b.subtype() );
}

inline int isAssignCompat(const ML_Base &a, const ML_Base &b)
{
    if ( a.isPool() && b.isPool() )
    {
        return 1;
    }

    if ( a.isMutable() )
    {
        return 1;
    }

    switch ( a.type() + 10000*b.type() )
    {
        case     0:  { return 1; break; }
        case     1:  { return 0; break; }
        case     2:  { return 0; break; }
        case 10000:  { return 1; break; }
        case 10001:  { return 1; break; }
        case 10002:  { return 0; break; }
        case 20000:  { return 1; break; }
        case 20001:  { return 1; break; }
        case 20002:  { return 1; break; }

        default:
        {
            break;
        }
    }

    return ( a.type() == b.type() );
}

inline void qswap(const ML_Base *&a, const ML_Base *&b)
{
    const ML_Base *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}

inline void ML_Base::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ML_Base &b = bb.getML();

    kernPrecursor::qswapinternal(b);

    qswap(kernel         ,b.kernel         );
    qswap(UUoutkernel    ,b.UUoutkernel    );
    qswap(allxdatagent   ,b.allxdatagent   );
//FIXME    qswap(allxdatagentp  ,b.allxdatagentp  );
    qswap(ytargdata      ,b.ytargdata      );
    qswap(alltraintarg   ,b.alltraintarg   );
    qswap(xd             ,b.xd             );
    qswap(xdzero         ,b.xdzero         );
    qswap(xCweight       ,b.xCweight       );
    qswap(xCweightfuzz   ,b.xCweightfuzz   );
    qswap(xepsweight     ,b.xepsweight     );
    qswap(traininfo      ,b.traininfo      );
    qswap(traintang      ,b.traintang      );
//FIXME    qswap(traininfop     ,b.traininfop     );
    qswap(xalphaState    ,b.xalphaState    );
    qswap(indexKey       ,b.indexKey       );
    qswap(indexKeyCount  ,b.indexKeyCount  );
    qswap(typeKey        ,b.typeKey        );
    qswap(typeKeyBreak   ,b.typeKeyBreak   );
    qswap(isIndPrune     ,b.isIndPrune     );
    qswap(xassumedconsist,b.xassumedconsist);
    qswap(xconsist       ,b.xconsist       );
    qswap(globalzerotol  ,b.globalzerotol  );
    qswap(isBasisUserUU  ,b.isBasisUserUU  );
    qswap(defbasisUU     ,b.defbasisUU     );
    qswap(locbasisUU     ,b.locbasisUU     );
    qswap(isBasisUserVV  ,b.isBasisUserVV  );
    qswap(defbasisVV     ,b.defbasisVV     );
    qswap(locbasisVV     ,b.locbasisVV     );
    qswap(xpreallocsize  ,b.xpreallocsize  );
    qswap(K2mat          ,b.K2mat          );

    qswap(assumeReal      ,b.assumeReal      );
    qswap(trainingDataReal,b.trainingDataReal);
    qswap(wildxaReal      ,b.wildxaReal      );
    qswap(wildxbReal      ,b.wildxbReal      );
    qswap(wildxcReal      ,b.wildxcReal      );
    qswap(wildxdReal      ,b.wildxdReal      );

    incxvernum();
    b.incxvernum();

    incgvernum();
    b.incgvernum();

    gentype &(*UUcallbackxx)(gentype &res, int m, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);
    const gentype &(*VVcallbackxx)(gentype &res, int m, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);

    UUcallbackxx = UUcallback;
    UUcallback   = b.UUcallback;
    b.UUcallback = UUcallbackxx;

    VVcallbackxx = VVcallback;
    VVcallback   = b.VVcallback;
    b.VVcallback = VVcallbackxx;

    return;
}

inline void ML_Base::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ML_Base &b = bb.getMLconst();

    kernPrecursor::semicopy(b);

    //kernel
    //UUoutkernel
    //traininfo
    //allxdatagent
    //ytargdata
    //indexKey
    //indexKeyCount
    //typeKey
    //typeKeyBreak
    //isIndPrune
    //xassumedconsist
    //xconsist
    //xCweight
    //xCweightfuzz
    //xepsweight
    //isBasisUser
    //locbasis
    //K2mat

    xd            = b.xd;
    xdzero        = b.xdzero;
    alltraintarg  = b.alltraintarg;
    globalzerotol = b.globalzerotol;
    defbasisUU    = b.defbasisUU;
    defbasisVV    = b.defbasisVV;
    UUcallback    = b.UUcallback;
    VVcallback    = b.VVcallback;

    return;
}

inline void ML_Base::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ML_Base &src = bb.getMLconst();

    kernPrecursor::assign(src);

    kernel        = src.kernel;
    UUoutkernel   = src.UUoutkernel;
    traininfo     = src.traininfo;
    traintang     = src.traintang;
    xalphaState   = src.xalphaState;
    globalzerotol = src.globalzerotol;

    isBasisUserUU = src.isBasisUserUU;
    defbasisUU    = src.defbasisUU;
    locbasisUU    = src.locbasisUU;
    isBasisUserVV = src.isBasisUserVV;
    defbasisVV    = src.defbasisVV;
    locbasisVV    = src.locbasisVV;
    UUcallback    = src.UUcallback;
    VVcallback    = src.VVcallback;

    assumeReal       = src.assumeReal;
    trainingDataReal = src.trainingDataReal;
    wildxaReal       = src.wildxaReal;
    wildxbReal       = src.wildxbReal;
    wildxcReal       = src.wildxcReal;
    wildxdReal       = src.wildxdReal;

    incxvernum();
    incgvernum();

    if ( !onlySemiCopy )
    {
        allxdatagent  = src.allxdatagent;
        ytargdata     = src.ytargdata;

        alltraintarg = src.alltraintarg;

        xd           = src.xd;
        xdzero       = src.xdzero;
        xCweight     = src.xCweight;
        xCweightfuzz = src.xCweightfuzz;
        xepsweight   = src.xepsweight;
        K2mat        = src.K2mat;

        fixpvects();
    }

    else
    {
        allxdatagent .resize((src.allxdatagent ).size());

        alltraintarg = src.alltraintarg;

        xd           = src.xd;
        xdzero       = src.xdzero;
        xCweight     = src.xCweight;
        xCweightfuzz = src.xCweightfuzz;
        xepsweight   = src.xepsweight;

        fixpvects();
    }

    indexKey      = src.indexKey;
    indexKeyCount = src.indexKeyCount;
    typeKey       = src.typeKey;
    typeKeyBreak  = src.typeKeyBreak;

    isIndPrune      = src.isIndPrune;
    xassumedconsist = src.xassumedconsist;
    xconsist        = src.xconsist;

    // xpreallocsize unchanged!

    return;
}

#endif

