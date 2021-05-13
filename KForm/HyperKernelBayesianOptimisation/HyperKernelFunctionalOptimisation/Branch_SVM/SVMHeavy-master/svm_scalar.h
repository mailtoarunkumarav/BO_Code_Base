
//
// Scalar regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_scalar_h
#define _svm_scalar_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include "svm_generic.h"
#include "kcache.h"
#include "optstate.h"
#include "sQgraddesc.h"


void evalKSVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalxySVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalSigmaSVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);

void evalKSVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalxySVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalSigmaSVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);

double emmupfixer(fullOptState<double,double> &x, void *y, const Vector<double> &diagoffAdd, const Vector<double> &alphaGradAdd, double &gpgnhpGpnGnscalefactor);

class SVM_Scalar;
class SVM_Densit;
class SVM_PFront;
template <class T> class SVM_Vector_redbin;
class SVM_SimLrn;
class LSV_Generic;
class LSV_Scalar;
class LSV_Vector;
class LSV_Anions;
class LSV_ScScor;
class LSV_Planar;
class LSV_MvRank;
class SSV_Scalar;
class SVM_MultiC_atonce;
class SVM_MultiC_redbin;

OVERLAYMAKEFNVECTOR(SVM_Scalar)
OVERLAYMAKEFNVECTOR(Vector<SVM_Scalar>)
OVERLAYMAKEFNVECTOR(Matrix<SVM_Scalar>)

// Swap function

inline void qswap(SVM_Scalar &a, SVM_Scalar &b);


class SVM_Scalar : public SVM_Generic
{
    friend void evalKSVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalxySVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalSigmaSVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);

    friend void evalKSVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalxySVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalSigmaSVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner);

    friend double emmupfixer(fullOptState<double,double> &x, void *y, const Vector<double> &diagoffAdd, const Vector<double> &alphaGradAdd, double &gpgnhpGpnGnscalefactor);

    // Density and single class SVMs must be able to access and modify Qn,
    // Qnp and qn.

    friend class SVM_Binary;
    friend class SVM_Densit;
    friend class SVM_PFront;
    template <class T> friend class SVM_Vector_redbin;
    friend class SVM_SimLrn;
    friend class LSV_Generic;
    friend class LSV_Scalar;
    friend class LSV_Vector;
    friend class LSV_Anions;
    friend class LSV_ScScor;
    friend class LSV_Planar;
    friend class LSV_MvRank;
    friend class SSV_Scalar;
    friend class SVM_MultiC_atonce; 
    friend class SVM_MultiC_redbin;

public:

    // Constructors, destructors, assignment operators and similar

    SVM_Scalar();
    SVM_Scalar(const SVM_Scalar &src);
    SVM_Scalar(const SVM_Scalar &src, const ML_Base *xsrc);
    SVM_Scalar &operator=(const SVM_Scalar &src) { assign(src); return *this; }
    virtual ~SVM_Scalar();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void setmemsize(int memsize);

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { SVM_Scalar temp; *this = temp; return 1; }

    virtual int setAlphaR(const Vector<double> &newAlpha);
    virtual int setBiasR(const double &newBias);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int isTrained(void) const { return isStateOpt; }

    virtual int N  (void)  const { return Q.aN();                            }
    virtual int NS (void)  const { return NF()+NLB()+NUB();                  }
    virtual int NZ (void)  const { return Q.aNZ();                           }
    virtual int NF (void)  const { return Q.aNF();                           }
    virtual int NC (void)  const { return Q.aNC();                           }
    virtual int NLB(void)  const { return Q.aNLB();                          }
    virtual int NLF(void)  const { return Q.aNLF();                          }
    virtual int NUF(void)  const { return Q.aNUF();                          }
    virtual int NUB(void)  const { return Q.aNUB();                          }
    virtual int NNC(int d) const { return Nnc(d+1);                          }
    virtual int NS (int q) const { NiceAssert( q == 0 ); (void) q; return NS();  }
    virtual int NZ (int q) const { NiceAssert( q == 0 ); (void) q; return NZ();  }
    virtual int NF (int q) const { NiceAssert( q == 0 ); (void) q; return NF();  }
    virtual int NC (int q) const { NiceAssert( q == 0 ); (void) q; return NC();  }
    virtual int NLB(int q) const { NiceAssert( q == 0 ); (void) q; return NLB(); }
    virtual int NLF(int q) const { NiceAssert( q == 0 ); (void) q; return NLF(); }
    virtual int NUF(int q) const { NiceAssert( q == 0 ); (void) q; return NUF(); }
    virtual int NUB(int q) const { NiceAssert( q == 0 ); (void) q; return NUB(); }

    virtual int tspaceDim(void)  const { return 1; }
    virtual int numClasses(void) const { return 0; }
    virtual int type(void)       const { return 0; }
    virtual int subtype(void)    const { return 0; }
    virtual int order(void)      const { return 0; }

    virtual int numInternalClasses(void) const { return 2; }

    virtual const Vector<int>          &ClassLabels(void)   const { return classLabelsval;                                                                                    }
    virtual const Vector<Vector<int> > &ClassRep(void)      const { return classRepval;                                                                                       }
    virtual int                         findID(int ref)     const { NiceAssert( ref && ( ref >= -1 ) && ( ref <= 2 ) ); return ( ref == -1 ) ? 0 : ( ( ref == +1 ) ? 1 : 2 ); }

    virtual int isLinearCost(void)      const { return costType == 0;                                                }
    virtual int isQuadraticCost(void)   const { return costType == 1;                                                }
    virtual int is1NormCost(void)       const { return costType == 2;                                                }
    virtual int isVarBias(void)         const { return isVarBias(-1);                                                }
    virtual int isPosBias(void)         const { return isPosBias(-1);                                                }
    virtual int isNegBias(void)         const { return isNegBias(-1);                                                }
    virtual int isFixedBias(void)       const { return isFixedBias(-1);                                              }
    virtual int isVarBias(int q)        const { return ( q == -1 ) ? ( biasType == 0 ) : ( Q.betaRestrict(q) == 0 ); }
    virtual int isPosBias(int q)        const { return ( q == -1 ) ? ( biasType == 1 ) : ( Q.betaRestrict(q) == 1 ); }
    virtual int isNegBias(int q)        const { return ( q == -1 ) ? ( biasType == 2 ) : ( Q.betaRestrict(q) == 2 ); }
    virtual int isFixedBias(int q)      const { return ( q == -1 ) ? ( biasType == 3 ) : ( Q.betaRestrict(q) == 3 ); }

    virtual int isNoMonotonicConstraints(void)    const { return ( makeConvex == 0 ); }
    virtual int isForcedMonotonicIncreasing(void) const { return ( makeConvex == 1 ); }
    virtual int isForcedMonotonicDecreasing(void) const { return ( makeConvex == 2 ); }

    virtual int isOptActive(void) const { return optType == 0; }
    virtual int isOptSMO(void)    const { return optType == 1; }
    virtual int isOptD2C(void)    const { return optType == 2; }
    virtual int isOptGrad(void)   const { return optType == 3; }

    virtual int m(void) const { return emm; }

    virtual double C(void)            const { return CNval;          }
    virtual double eps(void)          const { return epsval;         }
    virtual double Cclass(int d)      const { return xCclass(d+1);   }
    virtual double epsclass(int d)    const { return xepsclass(d+1); }

    virtual int    memsize(void)       const { return kerncache.get_memsize(); }
    virtual double zerotol(void)       const { return Q.zerotol();             }
    virtual double Opttol(void)        const { return opttolval;               }
    virtual int    maxitcnt(void)      const { return maxitcntval;             }
    virtual double maxtraintime(void)  const { return maxtraintimeval;         }
    virtual double outerlr(void)       const { return outerlrval;              }
    virtual double outermom(void)      const { return outermomval;             }
    virtual int    outermethod(void)   const { return outermethodval;          }
    virtual double outertol(void)      const { return outertolval;             }
    virtual double outerovsc(void)     const { return outerovscval;            }
    virtual int    outermaxitcnt(void) const { return outermaxits;             }
    virtual int    outermaxcache(void) const { return outermaxcacheN;          }

    virtual       int      maxiterfuzzt(void) const { return maxiterfuzztval; }
    virtual       int      usefuzzt(void)     const { return usefuzztval;     }
    virtual       double   lrfuzzt(void)      const { return lrfuzztval;      }
    virtual       double   ztfuzzt(void)      const { return ztfuzztval;      }
    virtual const gentype &costfnfuzzt(void)  const { return costfnfuzztval;  }

    virtual double LinBiasForce(void)   const { return LinBiasForce(0);  }
    virtual double QuadBiasForce(void)  const { return QuadBiasForce(0); }
    virtual double LinBiasForce(int q)  const { return gn(q);            }
    virtual double QuadBiasForce(int q) const { return -Gn(q,q);         }

    virtual int isFixedTube(void)  const { return tubeshrink == 0; }
    virtual int isShrinkTube(void) const { return tubeshrink == 1; }

    virtual int isRestrictEpsPos(void) const { return epsrestrict == 1; }
    virtual int isRestrictEpsNeg(void) const { return epsrestrict == 2; }

    virtual double nu(void)     const { return nuLin;   }
    virtual double nuQuad(void) const { return nuQuadv; }

    virtual int isClassifyViaSVR(void) const { return 1; }
    virtual int isClassifyViaSVM(void) const { return 0; }

    virtual int is1vsA(void)    const { return 0; }
    virtual int is1vs1(void)    const { return 0; }
    virtual int isDAGSVM(void)  const { return 0; }
    virtual int isMOC(void)     const { return 0; }
    virtual int ismaxwins(void) const { return 0; }
    virtual int isrecdiv(void)  const { return 0; }

    virtual int isatonce(void) const { return 1; }
    virtual int isredbin(void) const { return 1; }

    virtual int isKreal(void)   const { return 0; }
    virtual int isKunreal(void) const { return 0; }

    virtual int isClassifier(void) const { return 0; }

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'R'; }
    virtual char targType(void) const { return 'R'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int isUnderlyingScalar(void) const { return 1; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual int isanomalyOn(void)  const { return 0; }
    virtual int isanomalyOff(void) const { return 1; }

    virtual double anomalyNu(void)    const { return 0; }
    virtual int    anomalyClass(void) const { return 0; }

    virtual int isautosetOff(void)          const { return autosetLevel == 0; }
    virtual int isautosetCscaled(void)      const { return autosetLevel == 1; }
    virtual int isautosetCKmean(void)       const { return autosetLevel == 2; }
    virtual int isautosetCKmedian(void)     const { return autosetLevel == 3; }
    virtual int isautosetCNKmean(void)      const { return autosetLevel == 4; }
    virtual int isautosetCNKmedian(void)    const { return autosetLevel == 5; }
    virtual int isautosetLinBiasForce(void) const { return autosetLevel == 6; }

    virtual double autosetCval(void)  const { return autosetCvalx;  }
    virtual double autosetnuval(void) const { return autosetnuvalx; }

    virtual const Vector<double>               &Cweight    (void)    const { return Cweightval;                                                      }
    virtual const Vector<double>               &Cweightfuzz(void)    const { return Cweightfuzzval;                                                  }
    virtual const Vector<double>               &epsweight  (void)    const { return epsweightval;                                                    }
    virtual const Matrix<double>               &Gp         (void)    const { return *Gpval;                                                          }
    virtual const Matrix<double>               &XX         (void)    const { return *xyval;                                                          }
    virtual const Vector<double>               &kerndiag   (void)    const { return kerndiagval;                                                     }
    virtual const Vector<double>               &diagoffset (void)    const { return diagoff;                                                         }
    virtual const Vector<int>                  &alphaState (void)    const { return Q.alphaState();                                                  }
    virtual const Vector<Vector<double> >      &getu       (void)    const { return u;                                                               }
    virtual const Vector<int>                  &d          (void)    const { return trainclass;                                                      }
    virtual const Vector<double>               &zR         (void)    const { return traintarg;                                                       }
    virtual const double                       &biasR      (void)    const { return ( isFixedBias() || !(Q.beta().size()) ) ? bfixval : Q.beta()(0); }
    virtual const Vector<double>               &alphaR     (void)    const { return Q.alpha();                                                       }

    // Modification and autoset functions

    virtual int setLinearCost(void);
    virtual int setQuadraticCost(void);
    virtual int set1NormCost(void);
    virtual int setVarBias(void);
    virtual int setPosBias(void);
    virtual int setNegBias(void);
    virtual int setFixedBias(double newbias = 0.0);
    virtual int setVarBias(int q);
    virtual int setPosBias(int q);
    virtual int setNegBias(int q);
    virtual int setFixedBias(int q, double newbias = 0.0);

    virtual int setNoMonotonicConstraints(void);
    virtual int setForcedMonotonicIncreasing(void);
    virtual int setForcedMonotonicDecreasing(void);

    virtual int setm(int xm) { NiceAssert( ( xm >= 2 ) && !(xm%2) ); emm = xm; return 0; }

    virtual int setC(double xC);
    virtual int seteps(double xeps);
    virtual int setCclass(int d, double xC);
    virtual int setepsclass(int d, double xeps);

    virtual int setOptActive(void);
    virtual int setOptSMO(void);
    virtual int setOptD2C(void);
    virtual int setOptGrad(void);

    virtual int setzerotol(double zt);
    virtual int setOpttol(double xopttol);
    virtual int setmaxitcnt(int xmaxitcnt)            { NiceAssert( xmaxitcnt       >= 0 ); maxitcntval     = xmaxitcnt;       return 0; }
    virtual int setmaxtraintime(double xmaxtraintime) { NiceAssert( xmaxtraintime   >= 0 ); maxtraintimeval = xmaxtraintime;   return 0; }
    virtual int setouterlr(double xouterlr)           { NiceAssert( xouterlr        >  0 ); outerlrval      = xouterlr;        return 0; }
    virtual int setoutermom(double xoutermom)         { NiceAssert( xoutermom       >= 0 ); outermomval     = xoutermom;       return 0; }
    virtual int setoutermethod(int xoutermethod)      { NiceAssert( xoutermethod    >= 0 ); outermethodval  = xoutermethod;    return 0; }
    virtual int setoutertol(double xoutertol)         { NiceAssert( xoutertol       >  0 ); outertolval     = xoutertol;       return 0; }
    virtual int setouterovsc(double xouterovsc)       { NiceAssert( xouterovsc      >= 0 ); outerovscval    = xouterovsc;      return 0; }
    virtual int setoutermaxitcnt(int xoutermaxits)    { NiceAssert( xoutermaxits    >= 0 ); outermaxits     = xoutermaxits;    return 0; }
    virtual int setoutermaxcache(int xoutermaxcacheN) { NiceAssert( xoutermaxcacheN >= 0 ); outermaxcacheN  = xoutermaxcacheN; return 0; }

    virtual int setmaxiterfuzzt(int xmaxiterfuzzt)               { NiceAssert( xmaxiterfuzzt >= 0 );                     maxiterfuzztval = xmaxiterfuzzt; return 0; }
    virtual int setusefuzzt(int xusefuzzt)                       {                                                   usefuzztval     = xusefuzzt;     return 0; }
    virtual int setlrfuzzt(double xlrfuzzt)                      { NiceAssert( ( xlrfuzzt >= 0 ) && ( xlrfuzzt <= 1 ) ); lrfuzztval      = xlrfuzzt;      return 0; }
    virtual int setztfuzzt(double xztfuzzt)                      { NiceAssert( xztfuzzt >= 0 );                          ztfuzztval      = xztfuzzt;      return 0; }
    virtual int setcostfnfuzzt(const gentype &xcostfnfuzzt)      {                                                   costfnfuzztval  = xcostfnfuzzt;  return 0; }
    virtual int setcostfnfuzzt(const std::string &xcostfnfuzzt)  {                                                   costfnfuzztval  = xcostfnfuzzt;  return 0; }

    virtual int sety(int i, double z);
    virtual int sety(const Vector<int> &i, const Vector<double> &z);
    virtual int sety(const Vector<double> &z);

    virtual int sety(int                i, const Vector<double>          &y) { return SVM_Generic::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &y) { return SVM_Generic::sety(i,y); }
    virtual int sety(                      const Vector<Vector<double> > &y) { return SVM_Generic::sety(  y); }

    virtual int sety(int                i, const d_anion         &y) { return SVM_Generic::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &y) { return SVM_Generic::sety(i,y); }
    virtual int sety(                      const Vector<d_anion> &y) { return SVM_Generic::sety(  y); }

    virtual int setCweightfuzz(int i, double xCweight);
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &xCweight);
    virtual int setCweightfuzz(const Vector<double> &xCweight);

    virtual int setLinBiasForce(double newval);
    virtual int setQuadBiasForce(double newval);
    virtual int setLinBiasForce(int q, double newval);
    virtual int setQuadBiasForce(int q, double newval) { NiceAssert( q == 0 ); (void) q; setQuadBiasForce(newval); return 0; }

    virtual int setFixedTube(void);
    virtual int setShrinkTube(void);

    virtual int setRestrictEpsPos(void); // This is the default for regression
    virtual int setRestrictEpsNeg(void); // This is used for classification

    virtual int setnu(double xnuLin)      { NiceAssert( xnuLin  >= 0 ); isStateOpt = 0; nuLin   = xnuLin;  return 0; }
    virtual int setnuQuad(double xnuQuad) { NiceAssert( xnuQuad >= 0 ); isStateOpt = 0; nuQuadv = xnuQuad; return 0; }

    virtual int setatonce(void) { return 0; }
    virtual int setredbin(void) { return 0; }

    virtual int autosetOff(void) { autosetLevel = 0; return 0; }
    virtual int autosetCscaled(double Cval);
    virtual int autosetCKmean(void);
    virtual int autosetCKmedian(void);
    virtual int autosetCNKmean(void);
    virtual int autosetCNKmedian(void);
    virtual int autosetLinBiasForce(double nuval, double Cval = 1.0);

    // Kernel Modification

    virtual void prepareKernel(void);
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1);

    virtual void fillCache(void);

    // Training set control

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

    virtual int sety(const Vector<int> &i, const Vector<gentype> &y);
    virtual int sety(const Vector<gentype> &y);
    virtual int sety(int i, const gentype &z);

    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(const Vector<int> &d);
    virtual int setd(int i, int d);

    virtual int setCweight(int i, double xCweight);
    virtual int setCweight(const Vector<int> &i, const Vector<double> &xCweight);
    virtual int setCweight(const Vector<double> &xCweight);

    virtual int setepsweight(int i, double xepsweight);
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &xepsweight);
    virtual int setepsweight(const Vector<double> &xepsweight);

    virtual int scaleCweight(double scalefactor);
    virtual int scaleCweightfuzz(double scalefactor);
    virtual int scaleepsweight(double scalefactor);

    virtual int randomise(double sparsity);

    // Train the SVM

    virtual void fudgeOn(void);
    virtual void fudgeOff(void);

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Evaluation:
    //
    // The trained machine is assumed to have the general form:
    //
    // g(y) = sum_i alpha_i K(y,x_i) + b
    //
    // This is the unprocessed output.  The processed output has an
    // additional classing function applied to it (for example binary
    // classification uses sgn).
    //
    // The raw result of is:
    //
    //         [ K(x,x_i0) ]
    //   res = [ K(x,x_i1) ]
    //         [   ...     ]
    //
    //   in sparse format where i0,i1,... are the supports.

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodx = NULL) const;

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodx = NULL, gentype ***pxyprody = NULL, gentype **pxyprodij = NULL) const;

//    virtual void dgTrainingVector(Vector<double> &res, int i) const { ML_Base::dgTrainingVector(res,i); return; }

    virtual double quasiloglikelihood(void) const;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);

    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    // Advanced features - external Hessian
    //
    // Usually the hessian Gp is calculated internally using the kernel
    // function.*  In some circumstances however it may be convenient to
    // have the SVM refer to an external matrix (for example if an
    // array of SVMs is in use that share a common set of feature vectors).
    //
    // NB: - when quadratic cost is used Gp_ij = K_ij + o_i delta_ij, where
    //       o_i is a diagonal offset term that is a function of C.  When
    //       combining quadratic cost with externel Gp note that Gp should
    //       always be updated *before* changing C, Cclass, Cweight, or
    //       changing to/from quadratic cost.
    //     - Gp should still be compatible with the kernel function if
    //       g(x) is to be evaluated correctly.
    //     - the Gp pointer will be lost during assignment (but not for the
    //       semicopy function) and if the stream operator is used to
    //       save and then reload the SVM.
    //     - Gpsigma is used by SMO and D2C optimisation techniques and
    //       is given by Gpsigma_ij = Gp_ii + Gp_jj - 2Gp_ij
    //     - for external Hessian both Gp and Gpsigma pointers must be
    //       given.
    //     - external Gp is incompatible with generalised cost functions /
    //       iterative fuzzy techniques.
    //
    // * Technically values are calculated on demand and cached up to
    //   the memory limit approximately set by the setmemsize function.
    //
    // The setGp function is used to turn the external Hessian on and
    // off.  If extGp and extGpsigma are non-NULL then the externel Hessian
    // is turned on.  If both are NULL then it is turned off.  If refactsol
    // is set NZ then all problem components related to Gp are recalculated,
    // otherwise Gp is assumed to be unchanged (except for being stored
    // externally).
    //
    //
    // On some occasions it is also useful to set Gpn in the same manner.
    // This is achieved using the setGpnExt and naivesetGpnExt functions.
    // See also the "multiple bias" options below.
    //
    // - setGpnExt moves from old Gpn matrix to new Gpn matrix and
    //   refactorises as required.
    // - naivesetGpnExt just sets the pointer, with no checking or
    //   refactorising.  This must be used with great care.
    // - Use GpnExt == NULL to revert to usual behaviour (or throw if
    //   biasdim != 0 - see below)
    // - GpnExt must be extended prior to calling addTrainingVector
    // - GpnExt must will be shrunk by removeTrainingVector (set d = 0 first)
    // - GpnExt is incompatible with tube shrinking
    // - no attempt is made to keep track of GpnExt for assignment, semicopy
    //   or streaming (none of which touch GpnExt).
    //
    // refactGpnElm(i,j): update a single element in Gpn.  Gpn must be
    //   updated *after* this call.

    void setGp(Matrix<double> *extGp = NULL, Matrix<double> *extGpsigma = NULL, Matrix<double> *extxy = NULL, int refactsol = 1);
    void setGpnExt(Matrix<double> *GpnExtOld, Matrix<double> *GpnExtNew);
    void naivesetGpnExt(Matrix<double> *GpnExtVal) { GpnExt = GpnExtVal; }
    void refactGpnElm(int i, int j, double GpnijNew);

    // Related to the above considerations, in some cases (for example if
    // the external code controlling the Gp matrix is applying a quadratic
    // cost function) it may be useful to update SVM_Scalar to reflect
    // changes along the diagonal of Gp only.  The following functions
    // allow this to be done.  In both cases the offset is the amount
    // being added to the diagonal of Gp (at position (i,i) in the second
    // case), and Gp is assumed to be updated prior to calling these
    // functions.

    void recalcdiagoff(const Vector<double> &offset);
    void recalcdiagoff(int i, double offset);

    // Advanced features - using "multiple biases"
    //
    // For convience when coding multiclass SVMs using recursive division
    // it is helpful to have multiple "biases".
    //
    // If biasdim == 0 then bias is as per usual
    // If biasdim == 1 then Gpn has zero width
    // If biasdim >  0 then Gpn has width biasdim-1 and is given by GpnExt
    //
    // - to use biasdim != 0 GpnExt must be set first.
    // - calling setbiasdim will automatically refactorise for changes in
    //   GpnExt.
    // - add all additional columns to GpnExt prior to calling setbiasdim.
    // - remove all redundant columns from GpnExt after calling setbiasdim.
    // - bias(i) returns the ith bias element, where 0 <= i < max(1,biasdim-1)
    // - setting a fixed bias with the scalar function affects all biases
    //   equally.
    // - biasdim != 0 is incompatible with tube shrinking
    // - normally new beta elements are added to the end of beta.  This can
    //   be modified by setting addpos >= 0 (add in this position).  addval
    //   is the value of the beta element added.
    // - normally beta elements are removed from the end of beta.  This can
    //   be modified by setting rempos >= 0 (remove from this position).  Can
    //   only have rempos == -1 or rempos == 0.
    // - setgn and setGn should be self-explanatory.

    int    getbiasdim(void)       const { return biasdim;     }
    double biasVMulti(int i)      const { return Q.beta()(i); }
    void   setbiasdim(int xbiasdim, int addpos = -1, double addval = 0.0, int rempos = -1);
//    void   setLinBiasForce(int i, double newval);
    void   setBiasVMulti(const Vector<double> &nwbias);

    void setgn(const Vector<double> &gnnew);
    void setGn(const Matrix<double> &Gnnew);

    // Other functions

    const Vector<double> &getgn(void)      const { return gn;             }
    const Matrix<double> &getGn(void)      const { return Gn;             }
    const Vector<double> &getgp(void)      const { return gp;             }
    const Matrix<double> &getGpn(void)     const { return Gpn;            }
    const Vector<int>    &pivAlphaZ(void)  const { return Q.pivAlphaZ();  }
    const Vector<int>    &pivAlphaLB(void) const { return Q.pivAlphaLB(); }
    const Vector<int>    &pivAlphaUB(void) const { return Q.pivAlphaUB(); }
    const Vector<int>    &pivAlphaF(void)  const { return Q.pivAlphaF();  }







protected:

    // Helper function for LS-SVM.  This will "maximally free" alpha and
    // beta - that is, it will free all alpha for which d != 0 and all beta
    // that are not fixed (will throw if beta is strictly positive of
    // strictly negative as beta is assumed to be either strictly zero
    // or completely free).
    //
    // Does not fix gradients.

   int maxFreeAlphaBias(void);
   int fact_minverse(Vector<double> &dalpha, Vector<double> &dbeta, const Vector<double> &bAlpha, const Vector<double> &bBeta) const;
   int fact_minverse(Vector<gentype> &dalpha, Vector<gentype> &dbeta, const Vector<gentype> &bAlpha, const Vector<gentype> &bBeta) const;



    // Kernel cache selective access for gradient calculation

    virtual double getvalIfPresent(int numi, int numj, int &isgood) const;

    // Inner-product cache: over-write this with a non-NULL return in classes where
    // a kernel cache is available

    virtual const Matrix<double> *getxymat(void) const { return ( ( ( getKernel().suggestXYcache() || getKernel().wantsXYprod() ) && ( iskip != -2 ) ) ) ? xyval : NULL; }



    // Cached access to K matrix
public:
    virtual double getKval(int i, int j) const { return Gp()(i,j) - ( ( i == j ) ? diagoff(i) : 0.0 ); }

private:

    virtual int gTrainingVector(double &res, int &unusedvar, int i, int raw = 0, gentype ***pxyprodi = NULL) const;

    int costType;           // 0 = linear, 1 = LS, 2 = linear with 1-norm regularisation
    int biasType;           // 0 = unconstrained, 3 = fixed (variable), 1 = positive, 2 = negative
    int optType;            // 0 = active set, 1 = SMO, 2 D2C, 3 grad
    int tubeshrink;         // 0 = normal, 1 = tube shrinking on
    int epsrestrict;        // 1 = regression type eps > 0, 2 = classification type eps < 0 (when tube shrinking is applied)
    int emm;                // 2 = standard 2-norm SVM, 4 = 1 1/3 norm SVM
    int biasdim;
    int makeConvex;         // 0 = no constraints, 1 = monotonic increasing (for finite-dim feature space and strictly non-negative training vectors), 2 = monotonic decreasing

    int maxitcntval;        // maximum number of iterations for training (or 0 if unlimited)
    double maxtraintimeval; // maximum time for training (or 0 if unlimited) - seconds
    double opttolval;       // optimality tolerance
    double outerlrval;      // learning rate for 4-norm SVM
    double outermomval;     // momentum for 4-norm SVM
    int outermethodval;     // method for 4-norm SVM (0 for quadratic inner, grad outer, 1 for straight Salverio grad descent with line-search)
    double outertolval;     // zero tolerance for 4-norm SVM
    double outerovscval;    // lr scale-down factor if step makes error worse for 4-norm svm
    int outermaxits;        // max num its for outer loop 4-norm training (0 is unlimited)
    int outermaxcacheN;     // max num K4 that can be cached for outer loop 4-norm training (0 is unlimited)

    double linbiasforceval; // linear bias forcing (kept for the case where biasdim = 0)
    double quadbiasforceval; // linear bias forcing (kept for the case where biasdim = 0)
    double nuLin;           // linear tube shrinking factor
    double nuQuadv;         // quadratic tube shrinking factor
    double CNval;           // C (tradeoff) value
    double epsval;          // eps (tube) value
    double bfixval;         // fixed bias value (if used)

    Vector<double> xCclass;   // classwise C weights (0 = -1, 1 = zero, 2 = +1, 3 = free)
    Vector<double> xepsclass; // classwise eps weights

    Kcache<double> xycache;            // inner-product cache
    Kcache<double> kerncache;          // kernel cache
    Kcache<double> sigmacache;         // sigma cache
    Vector<double> kerndiagval;        // kernel diagonals
    Vector<double> diagoff;            // diagonal offset for hessian (used by quadratic cost)

    Vector<int>             classLabelsval; // Convenience: [ -1 +1 2 ]
    Vector<Vector<int> >    classRepval;    // Convenience: [ [ -1 ] [ +1 ] [ 0 ] ]
    Vector<Vector<double> > u;              // Convenience: [ [ -1 ] [ +1 ] ]

    int autosetLevel;     // 0 = none, 1 = C/N, 2 = Cmean, 3 = Cmedian, 4 = CNmean, 5 = CNmedian, 6 = LinBiasForce
    double autosetCvalx;  // Cval used if autosetLevel == 1,6
    double autosetnuvalx; // nuval used if autosetLevel == 6

    // Optimisation state

    protected:
    optState<double,double> Q;
    private:
    Vector<int> Nnc; // number of vectors in each class (-1,0,+1,+2)
    int isStateOpt;  // set if SVM is in optimal state

    // Training data

    Vector<double> traintarg;
    Vector<int> trainclass;
    Vector<double> Cweightval;
    Vector<double> Cweightfuzzval;
    Vector<double> epsweightval;

    // Quadratic program definition
    //
    // Gplocal 1 = normal, 0 = Gp, Gpsigma both point elsewhere, so don't delete
    // Gpsigma(i,j) = Gp(i,i)+Gp(j,j)-(2.0*Gp(i,j)) (assumes Gpn = 1, Gn = 0)

    int Gplocal;
    Matrix<double> *Gpval;
    Matrix<double> *xyval;
public:
    virtual const Matrix<double> &xy(void) const { return *xyval; }
private:
    Matrix<double> *Gpsigma;
    Matrix<double> Gn;
    Matrix<double> Gpn;
    Vector<double> gp;
    Vector<double> gn;
    Vector<double> hp;
    Vector<double> hpscale; // like hp, but without epsilon term
    Vector<double> lb;
    Vector<double> ub;
    Matrix<double> *GpnExt;

    int intrain(svmvolatile int &killSwitch);
    int inintrain(svmvolatile int &killSwitch, double (*fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1);

    // Generalised cost functions / iterative fuzzy support

    int maxiterfuzztval;
    int usefuzztval;
    double lrfuzztval;
    double ztfuzztval;
    gentype costfnfuzztval;

    // 1-norm SVM stuff
    //
    // wr is used in the cost (w'.alpha), by default 1
    // cr is the empirical risk applied to negative errors (like lb)
    // ddr is the empirical risk applied to negative errors (like ub)
    // Qnp,Qn,qn are used for equality constraints in linear cost.
    //     (Qnp, Qn, qn are empty by default and can only be set by friend classes)
    //
    // alpharestrictoverride: additional restriction on alpha *in addition to*
    // that defined in optstate.  Uses same convention, namely
    //    alpharestrictoverride = 0: lb[i] <= alpha[i] <= ub[i]
    //    alpharestrictoverride = 1:     0 <= alpha[i] <= ub[i]
    //    alpharestrictoverride = 2: lb[i] <= alpha[i] <= 0
    //    alpharestrictoverride = 3:     0 <= alpha[i] <= 0
    //
    // Qconstype: 0 (default) Q constraints are >=
    //            1 Q constraints are ==

    Vector<double> wr;
    Vector<double> cr;
    Vector<double> ddr;
    Matrix<double> Qnp;
    Matrix<double> Qn;
    Vector<double> qn;
    int alpharestrictoverride;
    int Qconstype;

    // Internal functions
    //
    // recalcdiagoff(i): recalculate and update diagoff(i), or all of diagoff if i == -1
    // recalcuLUB: recalculate lb and ub and adjust state accordingly

    void recalcdiagoff(int ival);
    void recalcLUB(int ival);
    void recalcCRDR(int ival);

    // Elementwise bias control

//    int isVarBias(int q)   const { return Q.betaRestrict(q) == 0; }
//    int isPosBias(int q)   const { return Q.betaRestrict(q) == 1; }
//    int isNegBias(int q)   const { return Q.betaRestrict(q) == 2; }
//    int isFixedBias(int q) const { return Q.betaRestrict(q) == 3; }
//    void setVarBias(int q);
//    void setPosBias(int q);
//    void setNegBias(int q);

    // Miscellaneous

    int fixautosettings(int kernchange, int Nchange);
    int setdinternal(int i, int d); // like setd, but without fixing auto settings
    double autosetkerndiagmean(void);
    double autosetkerndiagmedian(void);

    int qtaddTrainingVector(int i, double z, double Cweigh, double epsweigh, int d);

    int presolveit(double betaGrad);

    void setalleps(double xeps, const Vector<double> &xepsclass);

    // Used by emm = 4 solver

public:
    int inEmm4Solve;
private:
    Vector<double> diagoffsetBase; // temporary
    Vector<double> gpBase; // temporary
    Vector<double> alphaPrev;
    Vector<int> alphaPrevPivNZ;
    int prevNZ;
    double ****emm4K4cache;

    // Quick-and-dirty sety function.  This does not update the gentype y value and assumes the scratch is of size N()

    int setyqnd(const Vector<double> &zn, Vector<double> &gpscratch);

    // Skip setting for kernel evaluation (used when adding training vectors)

    int iskip;

    // Cheat to quickly calculate C - assumes a whole bunch of things, only to be used by SSV

public:
    double calcCvalquick(int i) const { return CNval*xCclass(trainclass(i)+1)*Cweightval(i)*Cweightfuzzval(i); }
private:

    virtual void fastg(double &res) const;
    virtual void fastg(double &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const;

    virtual void fastg(gentype &res) const;
    virtual void fastg(gentype &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const;

    SVM_Scalar 
*thisthis;
    SVM_Scalar **thisthisthis;

public:

    // when adding a single vector it is sometimes handy to be able to 
    // pass the value of K2(x,x) in directly - eg if you've already calculated
    // it and it is computationally expensive.  To do this, set the following
    // pointer to point to it (but don't forget to set it back to NULL when
    // you're done).

    const double *diagkernvalcheat;
};

inline SVM_Scalar &setident (SVM_Scalar &a) { throw("something"); return a; }
inline SVM_Scalar &setzero  (SVM_Scalar &a) { a.restart(); return a; }
inline SVM_Scalar &setposate(SVM_Scalar &a) { return a; }
inline SVM_Scalar &setnegate(SVM_Scalar &a) { throw("something"); return a; }
inline SVM_Scalar &setconj  (SVM_Scalar &a) { throw("something"); return a; }
inline SVM_Scalar &setrand  (SVM_Scalar &a) { throw("something"); return a; }
inline SVM_Scalar &postProInnerProd(SVM_Scalar &a) { return a; }

inline void qswap(SVM_Scalar &a, SVM_Scalar &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Scalar::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Scalar &b = dynamic_cast<SVM_Scalar &>(bb.getML());

    SVM_Generic::qswapinternal(b);

        qswap(costType                   ,b.costType                   );
        qswap(biasType                   ,b.biasType                   );
        qswap(optType                    ,b.optType                    );
        qswap(tubeshrink                 ,b.tubeshrink                 );
        qswap(epsrestrict                ,b.epsrestrict                );
        qswap(emm                        ,b.emm                        );
        qswap(biasdim                    ,b.biasdim                    );
        qswap(makeConvex                 ,b.makeConvex                 );
        qswap(maxitcntval                ,b.maxitcntval                );
        qswap(maxtraintimeval            ,b.maxtraintimeval            );
        qswap(opttolval                  ,b.opttolval                  );
        qswap(outerlrval                 ,b.outerlrval                 );
        qswap(outermomval                ,b.outermomval                );
        qswap(outermethodval             ,b.outermethodval             );
        qswap(outertolval                ,b.outertolval                );
        qswap(outerovscval               ,b.outerovscval               );
        qswap(outermaxits                ,b.outermaxits                );
        qswap(outermaxcacheN             ,b.outermaxcacheN             );
        qswap(linbiasforceval            ,b.linbiasforceval            );
        qswap(quadbiasforceval           ,b.quadbiasforceval           );
        qswap(nuLin                      ,b.nuLin                      );
        qswap(nuQuadv                    ,b.nuQuadv                    );
        qswap(CNval                      ,b.CNval                      );
        qswap(epsval                     ,b.epsval                     );
        qswap(bfixval                    ,b.bfixval                    );
        qswap(xCclass                    ,b.xCclass                    );
        qswap(xepsclass                  ,b.xepsclass                  );
        qswap(xycache                    ,b.xycache                    );
        qswap(kerncache                  ,b.kerncache                  );
        qswap(sigmacache                 ,b.sigmacache                 );
        qswap(kerndiagval                ,b.kerndiagval                );
        qswap(diagoff                    ,b.diagoff                    );
        qswap(classLabelsval             ,b.classLabelsval             );
        qswap(classRepval                ,b.classRepval                );
        qswap(u                          ,b.u                          );
        qswap(autosetLevel               ,b.autosetLevel               );
        qswap(autosetnuvalx              ,b.autosetnuvalx              );
        qswap(autosetCvalx               ,b.autosetCvalx               );
        qswap(Q                          ,b.Q                          );
        qswap(Nnc                        ,b.Nnc                        );
        qswap(isStateOpt                 ,b.isStateOpt                 );
        qswap(traintarg                  ,b.traintarg                  );
        qswap(trainclass                 ,b.trainclass                 );
        qswap(Cweightval                 ,b.Cweightval                 );
        qswap(Cweightfuzzval             ,b.Cweightfuzzval             );
        qswap(epsweightval               ,b.epsweightval               );
        qswap(Gplocal                    ,b.Gplocal                    );
        qswap(Gn                         ,b.Gn                         );
        qswap(Gpn                        ,b.Gpn                        );
        qswap(gp                         ,b.gp                         );
        qswap(gn                         ,b.gn                         );
        qswap(hp                         ,b.hp                         );
        qswap(hpscale                    ,b.hpscale                    );
        qswap(lb                         ,b.lb                         );
        qswap(ub                         ,b.ub                         );
        qswap(cr                         ,b.cr                         );
        qswap(ddr                        ,b.ddr                        );
        qswap(wr                         ,b.wr                         );
        qswap(Qnp                        ,b.Qnp                        );
        qswap(Qn                         ,b.Qn                         );
        qswap(qn                         ,b.qn                         );
        qswap(alpharestrictoverride      ,b.alpharestrictoverride      );
        qswap(Qconstype                  ,b.Qconstype                  );
        qswap(maxiterfuzztval            ,b.maxiterfuzztval            );
        qswap(usefuzztval                ,b.usefuzztval                );
        qswap(lrfuzztval                 ,b.lrfuzztval                 );
        qswap(ztfuzztval                 ,b.ztfuzztval                 );
        qswap(costfnfuzztval             ,b.costfnfuzztval             );
//        qswap(inEmm4Solve                ,b.inEmm4Solve                );
//
//        double ****temp;
//
//        temp = emm4K4cache; emm4K4cache = b.emm4K4cache; b.emm4K4cache = temp;

        Matrix<double> *txy;
        Matrix<double> *tGp;
        Matrix<double> *tGpsigma;
        Matrix<double> *tGpnExt;

        txy      = xyval;   xyval   = b.xyval;   b.xyval   = txy;
        tGp      = Gpval;   Gpval   = b.Gpval;   b.Gpval   = tGp;
        tGpsigma = Gpsigma; Gpsigma = b.Gpsigma; b.Gpsigma = tGpsigma;
        tGpnExt  = GpnExt;  GpnExt  = b.GpnExt;  b.GpnExt  = tGpnExt;

        // The kernel (and sigma) cache, as well as Gp (and Gpsigma) will have
        // been messed around by the above switching.  We need to make sure that
        // their pointers are set to rights before we continue.

        (xycache).cheatSetEvalArg((void *) this);
        (kerncache).cheatSetEvalArg((void *) this);
        (sigmacache).cheatSetEvalArg((void *) this);

        if ( Gplocal )
        {
            (xyval)->cheatsetcdref((void *) &(xycache));
            (Gpval)->cheatsetcdref((void *) &(kerncache));
            (Gpsigma)->cheatsetcdref((void *) &(sigmacache));
        }

        (b.xycache).cheatSetEvalArg((void *) &b);
        (b.kerncache).cheatSetEvalArg((void *) &b);
        (b.sigmacache).cheatSetEvalArg((void *) &b);

        if ( b.Gplocal )
        {
            (b.xyval)->cheatsetcdref((void *) &(b.xycache));
            (b.Gpval)->cheatsetcdref((void *) &(b.kerncache));
            (b.Gpsigma)->cheatsetcdref((void *) &(b.sigmacache));
        }

        // GpnExt may have also been messed up in the next level up, but that is
        // not something we can fix at this level.

        return;
}

inline void SVM_Scalar::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Scalar &b = dynamic_cast<const SVM_Scalar &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

    // NB: GpnExt is assumed to be restored prior to or after calling this

    //Gplocal;
    //xyval;
    //Gpval;
    //Gpsigma;
    //GpnExt;

    //classLabelsval = b.classLabelsval;
    //classRepval    = b.classRepval;
    //u              = b.u;

    //Qnp = b.Qnp;
    //Qn  = b.Qn;
    //qn  = b.qn;
    //alpharestrictoverride = b.alpharestrictoverride

    costType    = b.costType;
    biasType    = b.biasType;
    optType     = b.optType;
    tubeshrink  = b.tubeshrink;
    epsrestrict = b.epsrestrict;
    emm         = b.emm;
    biasdim     = b.biasdim;
    makeConvex  = b.makeConvex;

    maxitcntval     = b.maxitcntval;
    maxtraintimeval = b.maxtraintimeval;
    opttolval       = b.opttolval;
    outerlrval      = b.outerlrval;
    outermomval     = b.outermomval;
    outermethodval  = b.outermethodval;
    outertolval     = b.outertolval;
    outerovscval    = b.outerovscval;
    outermaxits     = b.outermaxits;
    outermaxcacheN  = b.outermaxcacheN;

    quadbiasforceval = b.quadbiasforceval;
    nuLin            = b.nuLin;
    nuQuadv          = b.nuQuadv;
    epsval           = b.epsval;
    bfixval          = b.bfixval;

    xCclass   = b.xCclass;
    xepsclass = b.xepsclass;

    autosetLevel  = b.autosetLevel;
    autosetCvalx  = b.autosetCvalx;
    autosetnuvalx = b.autosetnuvalx;

    Cweightval     = b.Cweightval;
    Cweightfuzzval = b.Cweightfuzzval;
    epsweightval   = b.epsweightval;

    maxiterfuzztval = b.maxiterfuzztval;
    usefuzztval     = b.usefuzztval;
    lrfuzztval      = b.lrfuzztval;
    ztfuzztval      = b.ztfuzztval;
    costfnfuzztval  = b.costfnfuzztval;

//    inEmm4Solve = b.inEmm4Solve;
    iskip       = b.iskip;

    isStateOpt = b.isStateOpt;

    linbiasforceval  = b.linbiasforceval;
    CNval            = b.CNval;

    Gpn     = b.Gpn;
    Gn      = b.Gn;
    gn      = b.gn;
    lb      = b.lb;
    ub      = b.ub;
    cr      = b.cr;
    ddr     = b.ddr;
    wr      = b.wr;
    gp      = b.gp;
    hp      = b.hp;
    hpscale = b.hpscale;

    Q          = b.Q;
    Nnc        = b.Nnc;
    trainclass = b.trainclass;
    traintarg  = b.traintarg;

    if ( isQuadraticCost() )
    {
        kerndiagval = b.kerndiagval;
        diagoff     = b.diagoff;

        xycache.recalcDiag();
        kerncache.recalcDiag();
        sigmacache.recalcDiag();
    }

    return;
}

inline void SVM_Scalar::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Scalar &src = dynamic_cast<const SVM_Scalar &>(bb.getMLconst());

    // NB: GpnExt is assumed to be dealt with elsewhere.  We just naively assume
    //     it won't change here.

    SVM_Generic::assign(src,onlySemiCopy);

    costType    = src.costType;
    biasType    = src.biasType;
    optType     = src.optType;
    tubeshrink  = src.tubeshrink;
    epsrestrict = src.epsrestrict;
    emm         = src.emm;
    biasdim     = src.biasdim;
    makeConvex  = src.makeConvex;

    maxitcntval     = src.maxitcntval;
    maxtraintimeval = src.maxtraintimeval;
    opttolval       = src.opttolval;
    outerlrval      = src.outerlrval;
    outermomval     = src.outermomval;
    outermethodval  = src.outermethodval;
    outertolval     = src.outertolval;
    outerovscval    = src.outerovscval;
    outermaxits     = src.outermaxits;
    outermaxcacheN  = src.outermaxcacheN;

    linbiasforceval  = src.linbiasforceval;
    quadbiasforceval = src.quadbiasforceval;
    nuLin            = src.nuLin;
    nuQuadv          = src.nuQuadv;
    CNval            = src.CNval;
    epsval           = src.epsval;
    bfixval          = src.bfixval;

    xCclass   = src.xCclass;
    xepsclass = src.xepsclass;

    kerndiagval = src.kerndiagval;
    diagoff     = src.diagoff;

    classLabelsval = src.classLabelsval;
    classRepval    = src.classRepval;
    u              = src.u;

    autosetLevel  = src.autosetLevel;
    autosetCvalx  = src.autosetCvalx;
    autosetnuvalx = src.autosetnuvalx;

    Q          = src.Q;
    Nnc        = src.Nnc;
    isStateOpt = src.isStateOpt;

    traintarg      = src.traintarg;
    trainclass     = src.trainclass;
    Cweightval     = src.Cweightval;
    Cweightfuzzval = src.Cweightfuzzval;
    epsweightval   = src.epsweightval;

    Gn      = src.Gn;
    Gpn     = src.Gpn;
    gp      = src.gp;
    gn      = src.gn;
    hp      = src.hp;
    hpscale = src.hpscale;
    lb      = src.lb;
    ub      = src.ub;
    cr      = src.cr;
    ddr     = src.ddr;
    wr      = src.wr;
    Qnp     = src.Qnp;
    Qn      = src.Qn;
    qn      = src.qn;
    alpharestrictoverride = src.alpharestrictoverride;
    Qconstype = src.Qconstype;

    maxiterfuzztval = src.maxiterfuzztval;
    usefuzztval     = src.usefuzztval;
    lrfuzztval      = src.lrfuzztval;
    ztfuzztval      = src.ztfuzztval;
    costfnfuzztval  = src.costfnfuzztval;

//    inEmm4Solve = src.inEmm4Solve;
    iskip       = src.iskip;

    xycache    = src.xycache;
    kerncache  = src.kerncache;
    sigmacache = src.sigmacache;

    xycache.cheatSetEvalArg((void *) this);
    kerncache.cheatSetEvalArg((void *) this);
    sigmacache.cheatSetEvalArg((void *) this);

    if ( Gplocal )
    {
	MEMDEL(xyval);
	xyval = NULL;

	MEMDEL(Gpval);
	Gpval = NULL;

        MEMDEL(Gpsigma);
	Gpsigma = NULL;
    }

    if ( src.Gplocal )
    {
	Gplocal = 1;

        MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &xycache   ,trainclass.size(),trainclass.size()));
        MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,trainclass.size(),trainclass.size()));
        MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,trainclass.size(),trainclass.size()));
    }

    else
    {
	Gplocal = 0;

	xyval   = src.xyval;
	Gpval   = src.Gpval;
	Gpsigma = src.Gpsigma;
    }

    // NB: this needs to be done last as it re-evaluates all the
    // diagonals

//    xycache.setEvalArg((void *) this);
//    kerncache.setEvalArg((void *) this);
//    sigmacache.setEvalArg((void *) this);

    return;
}

#endif
