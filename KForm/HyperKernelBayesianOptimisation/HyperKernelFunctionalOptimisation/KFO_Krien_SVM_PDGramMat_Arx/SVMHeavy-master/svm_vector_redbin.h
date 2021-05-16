
//
// Vector regression SVM (reduction to binary)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_vector_redbin_h
#define _svm_vector_redbin_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_generic.h"
#include "svm_scalar.h"
#include "kcache.h"


void evalKSVM_Vector_redbin_SVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalXYSVM_Vector_redbin_SVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalSigmaSVM_Vector_redbin_SVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);



template <class BaseRegressorClass>
class SVM_Vector_redbin;
class SVM_Vector;


// Swap function

template <class BaseRegressorClass>
inline void qswap(SVM_Vector_redbin<BaseRegressorClass> &a, SVM_Vector_redbin<BaseRegressorClass> &b);


template <class BaseRegressorClass>
class SVM_Vector_redbin : public SVM_Generic
{
    friend void evalKSVM_Vector_redbin_SVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalXYSVM_Vector_redbin_SVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalSigmaSVM_Vector_redbin_SVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner);

    friend class SVM_Vector;

public:

    // Constructors, destructors, assignment operators and similar

    SVM_Vector_redbin();
    SVM_Vector_redbin(const SVM_Vector_redbin<BaseRegressorClass> &src);
    SVM_Vector_redbin(const SVM_Vector_redbin<BaseRegressorClass> &src, const ML_Base *xsrc);
    SVM_Vector_redbin<BaseRegressorClass> &operator=(const SVM_Vector_redbin<BaseRegressorClass> &src) { assign(src); return *this; }
    virtual ~SVM_Vector_redbin();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void setmemsize(int memsize);

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { SVM_Vector_redbin<BaseRegressorClass> temp; *this = temp; return 1; }

    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha);
    virtual int setBiasV(const Vector<double> &newBias);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int isTrained(void) const { return isStateOpt; }

    virtual int N  (void)  const { return trainclass.size();                                }
    virtual int NS (void)  const { return NS(-1);                                           }
    virtual int NZ (void)  const { return NZ(-1);                                           }
    virtual int NF (void)  const { return NF(-1);                                           }
    virtual int NC (void)  const { return NC(-1);                                           }
    virtual int NLB(void)  const { return NLB(-1);                                          }
    virtual int NLF(void)  const { return NLF(-1);                                          }
    virtual int NUF(void)  const { return NUF(-1);                                          }
    virtual int NUB(void)  const { return NUB(-1);                                          }
    virtual int NNC(int d) const { return Nnc(d/2);                                         }
    virtual int NS (int q) const { return ( q == -1 ) ? Ns         : (NF(q)+NLB(q)+NUB(q)); }
    virtual int NZ (int q) const { return ( q == -1 ) ? (N()-NS()) : (Q(q).NZ());           }
    virtual int NF (int q) const { return ( q == -1 ) ? NS()       : Q(q).NF();             }
    virtual int NC (int q) const { return ( q == -1 ) ? NZ()       : Q(q).NC();             }
    virtual int NLB(int q) const { return ( q == -1 ) ? 0          : Q(q).NLB();            }
    virtual int NLF(int q) const { return ( q == -1 ) ? 0          : Q(q).NLF();            }
    virtual int NUF(int q) const { return ( q == -1 ) ? NS()       : Q(q).NUF();            }
    virtual int NUB(int q) const { return ( q == -1 ) ? 0          : Q(q).NUB();            }

    virtual int tspaceDim(void)  const { return db.size(); }
    virtual int numClasses(void) const { return 0;         }
    virtual int type(void)       const { return 4;         }
    virtual int subtype(void)    const { return 1;         }

    virtual int numInternalClasses(void) const { return 1; }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'V'; }
    virtual char targType(void) const { return 'V'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual const Vector<int>          &ClassLabels(void)   const { return classLabelsval;                    }
    virtual const Vector<Vector<int> > &ClassRep(void)      const { return classRepval;                       }
    virtual int                         findID(int ref)     const { NiceAssert( ref == 2 ); (void) ref; return 2; }

    virtual int isLinearCost(void)      const { return costType == 0;      }
    virtual int isQuadraticCost(void)   const { return costType == 1;      }
    virtual int is1NormCost(void)       const { return 0;                  }
    virtual int isVarBias(void)         const { return isVarBias(0);       }
    virtual int isPosBias(void)         const { return isPosBias(0);       }
    virtual int isNegBias(void)         const { return isNegBias(0);       }
    virtual int isFixedBias(void)       const { return isFixedBias(0);     }
    virtual int isVarBias(int q)        const { return Q(q).isVarBias();   }
    virtual int isPosBias(int q)        const { return Q(q).isPosBias();   }
    virtual int isNegBias(int q)        const { return Q(q).isNegBias();   }
    virtual int isFixedBias(int q)      const { return Q(q).isFixedBias(); }

    virtual int isOptActive(void) const { return optType == 0; }
    virtual int isOptSMO(void)    const { return optType == 1; }
    virtual int isOptD2C(void)    const { return optType == 2; }
    virtual int isOptGrad(void)   const { return optType == 3; }

    virtual int m(void) const { return Q(0).m(); }

    virtual double C(void)            const { return CNval;         }
    virtual double eps(void)          const { return Q(0).eps();    }
    virtual double Cclass(int d)      const { (void) d; return 1;   }
    virtual double epsclass(int d)    const { (void) d; return 1;   }

    virtual int    memsize(void)      const { return kerncache.get_memsize(); }
    virtual double zerotol(void)      const { return Q(0).zerotol();          }
    virtual double Opttol(void)       const { return Q(0).Opttol();           }
    virtual int    maxitcnt(void)     const { return Q(0).maxitcnt();         }
    virtual double maxtraintime(void) const { return Q(0).maxtraintime();     }
    virtual double outerlr(void)      const { return Q(0).outerlr();          }
    virtual double outertol(void)     const { return Q(0).outertol();         }

    virtual       int      maxiterfuzzt(void) const { return Q(0).maxiterfuzzt(); }
    virtual       int      usefuzzt(void)     const { return Q(0).usefuzzt();     }
    virtual       double   lrfuzzt(void)      const { return Q(0).lrfuzzt();      }
    virtual       double   ztfuzzt(void)      const { return Q(0).ztfuzzt();      }
    virtual const gentype &costfnfuzzt(void)  const { return Q(0).costfnfuzzt();  }

    virtual double LinBiasForce(void)        const { return LinBiasForce(0);      }
    virtual double QuadBiasForce(void)       const { return QuadBiasForce(0);     }
    virtual double LinBiasForce(int q)       const { return Q(q).LinBiasForce();  }
    virtual double QuadBiasForce(int q)      const { return Q(q).QuadBiasForce(); }

    virtual int isFixedTube(void)  const { return Q(0).isFixedTube();  }
    virtual int isShrinkTube(void) const { return Q(0).isShrinkTube(); }

    virtual int isRestrictEpsPos(void) const { return 1; }
    virtual int isRestrictEpsNeg(void) const { return 0; }

    virtual double nu(void)     const { return Q(0).nu();     }
    virtual double nuQuad(void) const { return Q(0).nuQuad(); }

    virtual int isClassifyViaSVR(void) const { return 1; }
    virtual int isClassifyViaSVM(void) const { return 0; }

    virtual int is1vsA(void)    const { return 0; }
    virtual int is1vs1(void)    const { return 0; }
    virtual int isDAGSVM(void)  const { return 0; }
    virtual int isMOC(void)     const { return 0; }
    virtual int ismaxwins(void) const { return 0; }
    virtual int isrecdiv(void)  const { return 0; }

    virtual int isatonce(void) const { return 0; }
    virtual int isredbin(void) const { return 1; }

    virtual int isKreal(void)   const { return 1; }
    virtual int isKunreal(void) const { return 0; }

    virtual int isClassifier(void) const { return 0; }

    virtual int isUnderlyingScalar(void) const { return 0; }
    virtual int isUnderlyingVector(void) const { return 1; }
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
    virtual int isautosetLinBiasForce(void) const { return 0;                 }

    virtual double autosetCval(void)  const { return autosetCvalx; }
    virtual double autosetnuval(void) const { return 0;            }

    virtual const Vector<int>                  &d          (void)      const { return trainclass;                                  }
    virtual const Vector<double>               &Cweight    (void)      const { return Q(0).Cweight();                              }
    virtual const Vector<double>               &Cweightfuzz(void)      const { return Q(0).Cweightfuzz();                          }
    virtual const Vector<double>               &epsweight  (void)      const { return Q(0).epsweight();                            }
    virtual const Matrix<double>               &Gp         (void)      const { return *Gpval;                                      }
    virtual const Vector<double>               &kerndiag   (void)      const { return kerndiagval;                                 }
    virtual const Vector<int>                  &alphaState (void)      const { return dalphaState;                                 }
    virtual const Vector<Vector<double> >      &zV         (int raw=0) const { (void) raw; return traintarg;                       }
    virtual const Vector<double>               &biasV      (int raw=0) const { (void) raw; return db;                              }
    virtual const Vector<Vector<double> >      &alphaV     (int raw=0) const { (void) raw; return dalpha;                          }
    virtual const Vector<Vector<double> >      &getu       (void)      const { return u;                                           }

    // Modification:

    virtual int setLinearCost(void);
    virtual int setQuadraticCost(void);
    virtual int set1NormCost(void) { return SVM_Generic::set1NormCost(); }

    virtual int setC(double xC);
    virtual int seteps(double xeps);

    virtual int setOptActive(void);
    virtual int setOptSMO(void);
    virtual int setOptD2C(void);
    virtual int setOptGrad(void);

    virtual int setzerotol(double zt);
    virtual int setOpttol(double xopttol);
    virtual int setmaxitcnt(int xmaxitcnt);
    virtual int setmaxtraintime(double xmaxtraintime);
    virtual int setouterlr(double xouterlr)   { (void) xouterlr;  return 0; }
    virtual int setoutertol(double xoutertol) { (void) xoutertol; return 0; }

    virtual int randomise(double sparsity);

    virtual int sety(int i, const Vector<double> &z);
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z);
    virtual int sety(const Vector<Vector<double> > &z);

    virtual int autosetOff(void)            { autosetLevel = 0; return 0; }
    virtual int autosetCscaled(double Cval) { NiceAssert( Cval > 0 ); autosetCvalx = Cval; int res = setC( (N()-NNC(0)) ? (Cval/((N()-NNC(0)))) : 1.0); autosetLevel = 1; return res; }
    virtual int autosetCKmean(void)         { double diagsum = ( (N()-NNC(0)) ? autosetkerndiagmean()                : 1 ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 2; return res; }
    virtual int autosetCKmedian(void)       { double diagsum = ( (N()-NNC(0)) ? autosetkerndiagmedian()              : 1 ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 3; return res; }
    virtual int autosetCNKmean(void)        { double diagsum = ( (N()-NNC(0)) ? (N()-NNC(0))*autosetkerndiagmean()   : 1 ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 4; return res; }
    virtual int autosetCNKmedian(void)      { double diagsum = ( (N()-NNC(0)) ? (N()-NNC(0))*autosetkerndiagmedian() : 1 ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 5; return res; }

    virtual int settspaceDim(int newdim);
    virtual int addtspaceFeat(int i);
    virtual int removetspaceFeat(int i);

    // Kernel Modification

    virtual const MercerKernel &getKernel(void)  const { return SVM_Generic::getKernel(); }
    virtual MercerKernel &getKernel_unsafe(void)       { return SVM_Generic::getKernel_unsafe(); }
    virtual void prepareKernel(void) { return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1);

    virtual void fillCache(void);

    virtual double  &K2ip(double  &res, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;
    virtual double  &K2(  double  &res, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;
    virtual gentype &K2(  gentype &res, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const;

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int addTrainingVector( int i, const Vector<double> &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector( int i, const Vector<gentype> &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector( int i, const Vector<Vector<double> > &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i, const Vector<Vector<double> > &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    virtual int sety(int i, const gentype &z);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z);
    virtual int sety(const Vector<gentype> &z);

    virtual int setd(int i, int d);
    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(const Vector<int> &d);

    virtual int setCweight(int i, double xCweight);
    virtual int setCweight(const Vector<int> &i, const Vector<double> &xCweight);
    virtual int setCweight(const Vector<double> &xCweight);

    virtual int setCweightfuzz(int i, double xCweight);
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &xCweight);
    virtual int setCweightfuzz(const Vector<double> &xCweight);

    virtual int setepsweight(int i, double xepsweight);
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &xepsweight);
    virtual int setepsweight(const Vector<double> &xepsweight);

    virtual int scaleCweight(double scalefactor);
    virtual int scaleCweightfuzz(double scalefactor);
    virtual int scaleepsweight(double scalefactor);

    // Train the SVM

    virtual void fudgeOn(void);
    virtual void fudgeOff(void);

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = NULL, gentype ***pxyprodj = NULL, gentype **pxyprodij = NULL) const;

    virtual void dgTrainingVector(Vector<gentype> &res, int i) const;
    virtual void dgTrainingVector(Vector<double>  &res, int i) const;

    virtual void dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const { SVM_Generic::dgTrainingVector(res,resn,i); return; }

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    // Training set control:

    int setFixedBias(const Vector<double> &newbias);
    virtual int setFixedBias(double newbias);
    virtual int setFixedBias(int q, double newbias);
    virtual int setFixedBias(const gentype &newbias) { return SVM_Generic::setFixedBias(newbias);   }

private:

    virtual int gTrainingVector(Vector<double> &gproject, int &locclassrep, int i, int raw = 0, gentype ***pxyprodi = NULL) const;

    int costType; // 0 = linear, 1 = LS
    int optType;  // 0 = active set, 1 = SMO, 2 D2C, 3 grad
    double CNval; // C (tradeoff) value (must be stored locally for kernel offset)

    Kcache<double> xycache;           // xy cache
    Kcache<double> kerncache;         // kernel cache
    Kcache<double> sigmacache;        // sigma cache
    Vector<double> kerndiagval;       // kernel diagonals
    Vector<double> diagoff;           // diagonal offset for hessian (used by quadratic cost)

    Vector<int>          classLabelsval; // Convenience: [ -1 +1 2 ] (-1,+1 unused)
    Vector<Vector<int> > classRepval;    // Convenience: [ [ -1 ] [ +1 ] [ 0 ] ] 
    Vector<Vector<double> > u; // Convenience, empty

    int autosetLevel;    // 0 = none, 1 = C/N, 2 = Cmean, 3 = Cmedian, 4 = CNmean, 5 = CNmedian, 6 = LinBiasForce
    double autosetCvalx;  // Cval used if autosetLevel == 1,6

    // Optimisation state

    Vector<BaseRegressorClass> Q;
    Vector<Vector<double> > dalpha;
    Vector<int> dalphaState;
    Vector<double> db;
    int Ns;
    Vector<int> Nnc;
    int isStateOpt;

    // Training data

    Vector<int> trainclass;
    Vector<Vector<double> > traintarg;

    // Quadratic program definition
    //
    // Gplocal 1 = normal, 0 = Gp, Gpsigma both point elsewhere, so don't delete
    // Gpsigma(i,j) = Gp(i,i)+Gp(j,j)-(2.0*Gp(i,j)) (assumes Gpn = 1, Gn = 0)

    int Gplocal;
    Matrix<double> *xyval;
    Matrix<double> *Gpval;
    Matrix<double> *Gpsigma;

    // Used to keep track of fudging when tspaceDim changes

    int isFudged;

    // Internal functions
    //
    // locsetGp: call setGp for all scalar SVMs Q
    // recalcdiagoff(i): recalculate and update diagoff(i), or all of diagoff if i == -1
    // setKernel: set the kernel

    void setGp(Matrix<double> *extGp = NULL, Matrix<double> *extGpsigma = NULL, Matrix<double> *extxy = NULL, int refactsol = 1);
    void locsetGp(int refactsol = 1);
    void recalcdiagoff(int ival);

    // Miscellaneous

    int fixautosettings(int kernchange, int Nchange);
    int setdinternal(int i, int d); // like setd, but without fixing auto settings
    double autosetkerndiagmean(void);
    double autosetkerndiagmedian(void);

    SVM_Vector_redbin<BaseRegressorClass> *thisthis;
    SVM_Vector_redbin<BaseRegressorClass> **thisthisthis;




    int dtrans(int i, int q) const;

    int setalldifrank(void);

};

template <class BaseRegressorClass>
inline void qswap(SVM_Vector_redbin<BaseRegressorClass> &a, SVM_Vector_redbin<BaseRegressorClass> &b)
{
    a.qswapinternal(b);

    return;
}

template <class BaseRegressorClass>
inline void SVM_Vector_redbin<BaseRegressorClass>::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Vector_redbin<BaseRegressorClass> &b = dynamic_cast<SVM_Vector_redbin<BaseRegressorClass> &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(costType              ,b.costType              );
    qswap(optType               ,b.optType               );
    qswap(CNval                 ,b.CNval                 );
    qswap(xycache               ,b.xycache               );
    qswap(kerncache             ,b.kerncache             );
    qswap(sigmacache            ,b.sigmacache            );
    qswap(kerndiagval           ,b.kerndiagval           );
    qswap(diagoff               ,b.diagoff               );
    qswap(autosetLevel          ,b.autosetLevel          );
    qswap(autosetCvalx          ,b.autosetCvalx          );
    qswap(classLabelsval        ,b.classLabelsval        );
    qswap(classRepval           ,b.classRepval           );
    qswap(Q                     ,b.Q                     );
    qswap(dalpha                ,b.dalpha                );
    qswap(dalphaState           ,b.dalphaState           );
    qswap(db                    ,b.db                    );
    qswap(Ns                    ,b.Ns                    );
    qswap(Nnc                   ,b.Nnc                   );
    qswap(isStateOpt            ,b.isStateOpt            );
    qswap(trainclass            ,b.trainclass            );
    qswap(traintarg             ,b.traintarg             );
    qswap(isFudged              ,b.isFudged              );

    Matrix<double> *txy;
    Matrix<double> *tGp;
    Matrix<double> *tGpsigma;

    txy      = xyval;   xyval   = b.xyval;   b.xyval   = txy;
    tGp      = Gpval;   Gpval   = b.Gpval;   b.Gpval   = tGp;
    tGpsigma = Gpsigma; Gpsigma = b.Gpsigma; b.Gpsigma = tGpsigma;

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

    return;
}

template <class BaseRegressorClass>
inline void SVM_Vector_redbin<BaseRegressorClass>::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Vector_redbin<BaseRegressorClass> &b = dynamic_cast<const SVM_Vector_redbin<BaseRegressorClass> &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

    //classLabelsval
    //classRepval
    //u

    //Gplocal
    //Gpval
    //Gpsigma

    costType  = b.costType;
    optType   = b.optType;

    autosetLevel = b.autosetLevel;
    autosetCvalx = b.autosetCvalx;

    traintarg = b.traintarg;

    CNval = b.CNval;

    isFudged    = b.isFudged;

    dalpha      = b.dalpha;
    dalphaState = b.dalphaState;
    db          = b.db;
    Ns          = b.Ns;
    Nnc         = b.Nnc;
    isStateOpt  = b.isStateOpt;

    trainclass = b.trainclass;

    if ( isQuadraticCost() )
    {
        kerndiagval = b.kerndiagval;
        diagoff     = b.diagoff;

        xycache.recalcDiag();
        kerncache.recalcDiag();
        sigmacache.recalcDiag();
    }

    Q.resize(b.Q.size());

    int q;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        Q("&",q).semicopy((b.Q)(q));
    }

    return;
}

template <class BaseRegressorClass>
inline void SVM_Vector_redbin<BaseRegressorClass>::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Vector_redbin<BaseRegressorClass> &src = dynamic_cast<const SVM_Vector_redbin<BaseRegressorClass> &>(bb.getMLconst());

    SVM_Generic::assign(src,onlySemiCopy);

    classLabelsval = src.classLabelsval;
    classRepval    = src.classRepval;

    isStateOpt = src.isStateOpt;

    isFudged    = src.isFudged;

    costType  = src.costType;
    optType   = src.optType;

    autosetLevel = src.autosetLevel;
    autosetCvalx = src.autosetCvalx;

    CNval = src.CNval;

    xycache     = src.xycache;
    kerncache   = src.kerncache;
    sigmacache  = src.sigmacache;

    xycache.cheatSetEvalArg((void *) this);
    kerncache.cheatSetEvalArg((void *) this);
    sigmacache.cheatSetEvalArg((void *) this);

    kerndiagval = src.kerndiagval;
    diagoff     = src.diagoff;

    int i;

    Q.resize(src.Q.size());

    for ( i = 0 ; i < Q.size() ; i++ )
    {
        Q("&",i).assign((src.Q)(i),onlySemiCopy);
    }

    dalpha      = src.dalpha;
    dalphaState = src.dalphaState;
    db          = src.db;

    Ns  = src.Ns;
    Nnc = src.Nnc;

    trainclass = src.trainclass;
    traintarg  = src.traintarg;

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

// This bit makes no sense, but I'm leaving the comment here in case it was relevant
//    if ( Gpval != NULL )
//    {
//        MEMDEL(Gpval);
//        MEMDEL(Gpsigma);
//
//	Gpval   = NULL;
//	Gpsigma = NULL;
//    }

    locsetGp();

    return;
}


























//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================
//==========================================================================

// NB: C_CALC actually calculates the upper bound used in the scalar SVM
//     base.  Thus calling Q.setC(C_CALC) will not invoke the recalculation
//     of any diagonals on the Hessian, so we must call recalcdiagoff
//     separately to fix the diagonals in the case of quadratic cost

#define C_CALC                                               ( isLinearCost() ? CNval : (MAXBOUND) )
#define QUADCOSTDIAGOFFSETB(_xclass_,_Cweigh_,_Cweighfuzz_)  ( (_xclass_) ? ( isQuadraticCost() ? (1/(CNval*(_Cweigh_)*(_Cweighfuzz_))) : 0.0 ) : 0.0 )


template <class BaseRegressorClass>
SVM_Vector_redbin<BaseRegressorClass>::SVM_Vector_redbin(const SVM_Vector_redbin<BaseRegressorClass> &src) : SVM_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    isFudged    = 0;

    Gplocal = 0;

    xyval   = NULL;
    Gpval   = NULL;
    Gpsigma = NULL;

    assign(src,0);

    return;
}

template <class BaseRegressorClass>
SVM_Vector_redbin<BaseRegressorClass>::SVM_Vector_redbin(const SVM_Vector_redbin<BaseRegressorClass> &src, const ML_Base *xsrc) : SVM_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(xsrc);

    isFudged    = 0;

    Gplocal = 0;

    xyval   = NULL;
    Gpval   = NULL;
    Gpsigma = NULL;

    assign(src,1);

    return;
}

template <class BaseRegressorClass>
SVM_Vector_redbin<BaseRegressorClass>::~SVM_Vector_redbin()
{
    if ( Gplocal )
    {
	MEMDEL(xyval);
	xyval = NULL;

	MEMDEL(Gpval);
	Gpval = NULL;

        MEMDEL(Gpsigma);
	Gpsigma = NULL;
    }

    return;
}

template <class BaseRegressorClass>
double SVM_Vector_redbin<BaseRegressorClass>::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}


template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::scale(double a)
{
    NiceAssert( a >= 0.0 );
    NiceAssert( a <= 1.0 );

    isStateOpt = 0;

    int i,q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).scale(a);
    }

    if ( tspaceDim() && N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
	    dalpha("&",i) *= a;

            if ( a == 0.0 )
            {
                dalphaState("&",i) = 0;
            }
	}

	db *= a;
    }

    SVM_Generic::basescalealpha(a);
    SVM_Generic::basescalebias(a);

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::reset(void)
{
    isStateOpt = 0;

    int i,q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).reset();
    }

    if ( tspaceDim() && N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            dalpha("&",i) = 0.0;
            dalphaState("&",i) = 0;
	}

        db = 0.0;
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setAlphaV(const Vector<Vector<double> > &newAlpha)
{
    NiceAssert( newAlpha.size() == N() );

    if ( N() && tspaceDim() )
    {
	isStateOpt = 0;

	int i,q;

	Vector<double> localpha(N());

	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                NiceAssert( newAlpha(i).size() == tspaceDim() );

		localpha("&",i) = newAlpha(i)(q);
	    }

            Q("&",q).setAlphaR(localpha);
	}

	dalphaState = zeroint();

	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                dalpha("&",i)("&",q) = (Q(q).alphaR())(i);

		if ( (Q(q).alphaState())(i) )
		{
		    dalphaState("&",i) = 1;
		}
	    }
	}

	Ns = sum(dalphaState);
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setBiasV(const Vector<double> &newBias)
{
    NiceAssert( newBias.size() == tspaceDim() );

    if ( tspaceDim() )
    {
	isStateOpt = 0;

	int q;

	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
            Q("&",q).setBiasR(newBias(q));
	}

        db = newBias;

        SVM_Generic::basesetbias(biasV());
    }

    return 1;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setLinearCost(void)
{
    if ( isQuadraticCost() )
    {
	if ( N() )
	{
	    isStateOpt = 0;
	}

        costType = 0;

	recalcdiagoff(-1);
        setC(CNval);
    }

    return 1;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setQuadraticCost(void)
{
    if ( isLinearCost() )
    {
	if ( N() )
	{
	    isStateOpt = 0;
	}

        costType = 1;

	recalcdiagoff(-1);
        setC(CNval);
    }

    return 1;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    (void) onlyChangeRowI;

    int res = 0;
    int fixxycache = getKernel().isIPdiffered();

    if ( N() )
    {
        res |= 1;
        isStateOpt = 0;
    }

    res |= SVM_Generic::resetKernel(modind,onlyChangeRowI,updateInfo);

    if ( fixxycache )
    {
        xycache.setSymmetry(1);
    }

    kerncache.setSymmetry(getKernel().getSymmetry());
    sigmacache.setSymmetry(1);

    if ( N() )
    {
	int i;

	for ( i = 0 ; i < N() ; i++ )
	{
            K2(kerndiagval("&",i),i,i);
	}
    }

    if ( fixxycache )
    {
        xycache.clear();
    }

    kerncache.clear();
    sigmacache.clear();

    res |= fixautosettings(1,0);
    locsetGp();

    getKernel_unsafe().setIPdiffered(0);

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    (void) onlyChangeRowI;

    int res = 0;

    if ( N() )
    {
        res |= 1;
        isStateOpt = 0;
    }

    res |= SVM_Generic::setKernel(xkernel,modind,onlyChangeRowI);

    xycache.setSymmetry(1);
    kerncache.setSymmetry(getKernel().getSymmetry());
    sigmacache.setSymmetry(1);

    if ( N() )
    {
	int i;

	for ( i = 0 ; i < N() ; i++ )
	{
            K2(kerndiagval("&",i),i,i);
	}
    }


    xycache.clear();
    kerncache.clear();
    sigmacache.clear();

    res |= fixautosettings(1,0);
    locsetGp();

    getKernel_unsafe().setIPdiffered(0);

    return res;
}

template <class BaseRegressorClass>
void SVM_Vector_redbin<BaseRegressorClass>::fillCache(void)
{
    if ( Q.size() )
    {
        int i;

        for ( i = 0 ; i < Q.size() ; i++ )
        {
            Q("&",i).fillCache();
        }
    }

    return;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setCweight(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( xCweight > 0 );

    int q;

    isStateOpt = 0;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(i);
    }

    else
    {
        for ( q = 0 ; q < Q.size() ; q++ )
	{
            Q("&",q).setCweight(i,xCweight);
            
            if ( q < tspaceDim() )
            {
                dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
            }
	}

        SVM_Generic::basesetalpha(i,alphaV()(i));
    }

    return 1;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setCweightfuzz(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( xCweight > 0 );

    int q;

    isStateOpt = 0;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(i);
    }

    else
    {
        for ( q = 0 ; q < Q.size() ; q++ )
	{
            Q("&",q).setCweightfuzz(i,xCweight);
            
            if ( q < tspaceDim() )
            {
                dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
            }
	}

        SVM_Generic::basesetalpha(i,alphaV()(i));
    }

    return 1;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setepsweight(int i, double xepsweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( xepsweight > 0 );

    isStateOpt = 0;

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setepsweight(i,xepsweight);
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setd(const Vector<int> &j, const Vector<int> &d)
{
    NiceAssert( d.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setd(j(i),d(i));
	}

        res |= fixautosettings(0,1);
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setCweight(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setCweight(j(i),xCweight(i));
	}
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setCweightfuzz(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setCweightfuzz(j(i),xCweight(i));
	}
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setepsweight(const Vector<int> &j, const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setepsweight(j(i),xepsweight(i));
	}
    }

    return res;
}


template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= setd(i,d(i));
	}

        res |= fixautosettings(0,1);
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setCweight(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= setCweight(i,xCweight(i));
	}
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setCweightfuzz(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= setCweightfuzz(i,xCweight(i));
	}
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setepsweight(const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= setepsweight(i,xepsweight(i));
	}
    }

    return res;
}








template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setC(double xC)
{
    NiceAssert( xC > 0 );

    int res = 0;

    autosetOff();

    if ( N() )
    {
	isStateOpt = 0;
        res = 1;
    }

    int i,q;

    CNval = xC;

    if ( isQuadraticCost() )
    {
        recalcdiagoff(-1);
    }

    // Need to set C as upper bound on alpha in all cases as this
    // function is also used to move between linear and quadratic
    // cost.

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setC(C_CALC);
    }

    if ( N() && tspaceDim() )
    {
	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
	    }
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::seteps(double xeps)
{
    NiceAssert( xeps >= 0 );

    int q;
    int res = 0;

    if ( N() )
    {
	isStateOpt = 0;
    }

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).seteps(xeps);
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setFixedBias(const Vector<double> &newbias)
{
    NiceAssert( newbias.size() == Q.size() );

    int q;
    int res = 0;

    if ( N() )
    {
	isStateOpt = 0;
    }

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setFixedBias(newbias(q));
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setFixedBias(double newbias)
{
    int q;
    int res = 0;

    if ( N() )
    {
	isStateOpt = 0;
    }

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setFixedBias(newbias);
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setFixedBias(int q, double newbias)
{
    NiceAssert( q < Q.size() );
    NiceAssert( q >= 0 );

    if ( N() )
    {
	isStateOpt = 0;
    }

    return Q("&",q).setFixedBias(newbias);
}


template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::scaleCweight(double scalefactor)
{
    NiceAssert( scalefactor > 0 );

    int res = 0;

    if ( N() )
    {
	isStateOpt = 0;
        res = 1;
    }

    int i,q;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else
    {
        for ( q = 0 ; q < Q.size() ; q++ )
	{
            res |= Q("&",q).scaleCweight(scalefactor);
	}

	if ( N() && tspaceDim() )
	{
	    for ( q = 0 ; q < tspaceDim() ; q++ )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
		}
	    }
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::scaleCweightfuzz(double scalefactor)
{
    NiceAssert( scalefactor > 0 );

    int res = 0;

    if ( N() )
    {
	isStateOpt = 0;
        res = 1;
    }

    int i,q;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else
    {
        for ( q = 0 ; q < Q.size() ; q++ )
	{
            res |= Q("&",q).scaleCweightfuzz(scalefactor);
	}

	if ( N() && tspaceDim() )
	{
	    for ( q = 0 ; q < tspaceDim() ; q++ )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
		}
	    }
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::scaleepsweight(double scalefactor)
{
    int q;
    int res = 0;

    if ( N() )
    {
	isStateOpt = 0;
    }

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).scaleepsweight(scalefactor);
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setOptActive(void)
{
    int q;

    if ( !isOptActive() )
    {
	optType = 0;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setOptActive();
	}
    }

    return 0;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setOptSMO(void)
{
    int q;

    if ( !isOptSMO() )
    {
        optType = 1;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setOptSMO();
	}
    }

    return 0;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setOptD2C(void)
{
    int q;

    if ( !isOptD2C() )
    {
	optType = 2;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setOptD2C();
	}
    }

    return 0;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setOptGrad(void)
{
    int q;

    if ( !isOptGrad() )
    {
	optType = 3;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setOptGrad();
	}
    }

    return 0;
}

template <class BaseRegressorClass>
void SVM_Vector_redbin<BaseRegressorClass>::setmemsize(int memsize)
{
    NiceAssert( memsize > 0 );

    int q;

    xycache.setmemsize(memsize,xycache.get_min_rowdim());
    kerncache.setmemsize(memsize,kerncache.get_min_rowdim());
    sigmacache.setmemsize(memsize,sigmacache.get_min_rowdim());

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        Q("&",q).setmemsize(memsize);
    }

    return;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setzerotol(double zt)
{
    NiceAssert( zt > 0 );

    isStateOpt = 0;

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setzerotol(zt);
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setOpttol(double xopttol)
{
    NiceAssert( xopttol > 0 );

    isStateOpt = 0;

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setOpttol(xopttol);
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setmaxitcnt(int xmaxitcnt)
{
    NiceAssert( xmaxitcnt > 0 );

    isStateOpt = 0;

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setmaxitcnt(xmaxitcnt);
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::setmaxtraintime(double xmaxtraintime)
{
    NiceAssert( xmaxtraintime > 0 );

    isStateOpt = 0;

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setmaxtraintime(xmaxtraintime);
    }

    return res;
}
















































































template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::addTrainingVector( int i, const Vector<gentype> &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    SparseVector<gentype> xx(x);

    return SVM_Vector_redbin<BaseRegressorClass>::qaddTrainingVector(i,z,xx,Cweigh,epsweigh,d);
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xx(x);

    return SVM_Vector_redbin<BaseRegressorClass>::qaddTrainingVector(i,z,xx,Cweigh,epsweigh);
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::addTrainingVector(int i, const Vector<double> &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    SparseVector<gentype> xxx(x);

    return SVM_Vector_redbin<BaseRegressorClass>::qaddTrainingVector(i,z,xxx,Cweigh,epsweigh,d);
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Vector_redbin<BaseRegressorClass>::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::addTrainingVector (int i, const Vector<Vector<double> > &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );
    NiceAssert( z.size() == d.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Vector_redbin<BaseRegressorClass>::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::qaddTrainingVector(int i, const Vector<Vector<double> > &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );
    NiceAssert( z.size() == d.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Vector_redbin<BaseRegressorClass>::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Vector_redbin<BaseRegressorClass>::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}



















template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 1;

    isStateOpt = 0;

    Nnc("&",trainclass(i)/2)--;

    int q;

    setd(i,0);

    res |= SVM_Generic::removeTrainingVector(i,y,x);

    trainclass.remove(i);
    traintarg.remove(i);
    diagoff.remove(i);
    kerndiagval.remove(i);
    dalpha.remove(i);
    dalphaState.remove(i);

    if ( Gplocal )
    {
        xyval->removeRowCol(i);
        Gpval->removeRowCol(i);
        Gpsigma->removeRowCol(i);
    }

    xycache.remove(i);
    kerncache.remove(i);
    sigmacache.remove(i);

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).ML_Base::removeTrainingVector(i);
    }

    // Fix the cache

    if ( ( kerncache.get_min_rowdim() >= (int) (N()*ROWDIMSTEPRATIO) ) && ( N() > MINROWDIM ) )
    {
	xycache.setmemsize(memsize(),N()-1);
	kerncache.setmemsize(memsize(),N()-1);
	sigmacache.setmemsize(memsize(),N()-1);
    }

    res |= fixautosettings(0,1);

    return res;
}











































































































template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::train(int &res, svmvolatile int &killSwitch)
{
    int i,q,result = 0;

    if ( tspaceDim() )
    {
        if ( tspaceDim() )
        {
	    result = 0;

            xycache.padCol(4);
            kerncache.padCol(4);
            sigmacache.padCol(4);

	    for ( q = 0 ; q < tspaceDim() ; q++ )
	    {
                result |= Q("&",q).train(res,killSwitch);
	    }

            xycache.padCol(0);
            kerncache.padCol(0);
            sigmacache.padCol(0);
        }

	db = 0.0;
	dalphaState = zeroint();

	if ( N() )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                dalpha("&",i) = 0.0;
	    }
	}

	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
            db("&",q) = Q(q).biasR();

	    if ( N() )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    dalpha("&",i)("&",q) = (Q(q).alphaR())(i);

		    if ( (Q(q).alphaState())(i) )
		    {
			dalphaState("&",i) = 1;
		    }
		}
	    }
	}
    }

    Ns = sum(dalphaState);

    if ( result )
    {
	isStateOpt = 0;
    }

    else
    {
        isStateOpt = 1;
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return result;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int unusedvar = 0;
    int tempresh = 0;
    Vector<double> tempresg;

    tempresh = gTrainingVector(tempresg,unusedvar,i,retaltg,pxyprodi);
    resh = tempresg; // processed output of regressor is scalar
    resg = tempresg;

    return tempresh;
}














template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::covTrainingVector(gentype &resv, gentype &resmu, int ia, int ib, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
    NiceAssert( !( getKernel().isKVarianceNZ() ) );

    NiceAssert( isFixedBias() );
    NiceAssert( NS() == NF() );
    NiceAssert( ( NF() == 0 ) || ( NF() == N()-NNC(0) ) );

    // We know that, for the fixed bias LS-SVM:
    //
    // var = K(ia,ib) - K(ia)'.inv(Gp).K(ib)
    //
    // where Gp = Kp + diag(sigma) (though it has a different name here), and
    // can assume the existence of the factorisation of Gp.  In this case
    // we simply calculate the second part for the free part of K(i) and
    // and leave it at that (which will work if this is an LS-SVM and has
    // been trained).
    //
    // As per Bull, Convergence rates of efficient global optimization algorithms,
    // in the variable bias case the prediction is precisely that of the LS-SVR,
    // and moreover the variance can be shown to simply be:
    //
    // K(ia,ib) - [ K(ia) ]' inv([ Gp 1 ]) [ K(ib) ]
    //            [  1    ]      [ 1  0 ]  [  1    ]
    //
    // = K(ia,ib) - K(ia)'.inv(Gp).K(bi) + ( 1 - 1'.inv(Gp).K(ia) )/( 1'.inv(Gp).1 )
    //
    // ... _ 1'.inv(Gp).q ( 1 - bias )
    //
    // and we can mostly use the same code for both.
    //
    // If Ns = 0 then the variable bias case is ill-defined

    int dtva = xtang(ia) & 7;
    int dtvb = xtang(ib) & 7;

    NiceAssert( dtva >= 0 );
    NiceAssert( dtvb >= 0 );

    (void) dtvb;

    // This is used elsewhere (ie not scalar), so the following is relevant

//FIXME: resmu
    {
        if ( NS() )
        {
            int j;

            Vector<gentype> Kia(N());
            Vector<gentype> Kib(N());
            Vector<gentype> itsone(1);//isVarBias() ? 1 : 0);
            gentype Kii;

            itsone("&",zeroint()) = 1.0;

            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                Kii  = Gp()(ia,ib);
                Kii -= ( ia == ib ) ? kerndiag()(ia) : 0.0;
            }

            else
            {
                K2(Kii,ia,ib,(const gentype **) pxyprodij);
            }

            if ( ia >= 0 )
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    Kia("&",j) = Gp()(ia,j);
                }

                Kia("&",ia) -= kerndiag()(ia);
            }

            else
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    K2(Kia("&",j),ia,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                }
            }

            if ( ib == ia )
            {
                Kib = Kia;
            }

            else if ( ib >= 0 )
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    Kib("&",j) = Gp()(j,ib);
                }

                Kib("&",ib) -= kerndiag()(ib);
            }

            else
            {
                for ( j = 0 ; j < N() ; j++ )
                {
                    K2(Kib("&",j),j,ib,pxyprodj ? (const gentype **) pxyprodj[j] : NULL);
                }
            }

            Vector<gentype> btemp(1);//isVarBias() ? 1 : 0);
            Vector<gentype> Kres(N());

            //NB: this will automatically only do part corresponding to pivAlphaF
            Q(0).fact_minverse(Kres,btemp,Kib,itsone);

            resv = Kii;

            for ( j = 0 ; j < Q(0).pivAlphaF().size() ; j++ )
            {
                resv -= outerProd(Kia(Q(0).pivAlphaF()(j)),Kres(Q(0).pivAlphaF()(j)));
            }

            if ( isVarBias() )
            {
                // This is the additional corrective factor

                resv -= btemp(zeroint());
            }

            // mu calculation

            int firstterm = 1;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( firstterm )
                {
                    resmu = Kia(j)*alpha()(j);

                    firstterm = 0;
                }

                else
                {
                    resmu += Kia(j)*alpha()(j);
                }
            }

            if ( !( dtva & 7 ) )
            {
                if ( firstterm )
                {
                    resmu = bias();

                    firstterm = 0;
                }

                else
                {
                    resmu += bias();
                }
            }

            else
            {
                if ( firstterm )
                {
                    resmu =  bias();
                    resmu *= 0.0;

                    firstterm = 0;
                }
            }
        }

        else
        {
            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                resv  = Gp()(ia,ib);
                resv -= ( ia == ib ) ? kerndiag()(ia) : 0.0;
            }

            else
            {
                K2(resv,ia,ib,(const gentype **) pxyprodij);
            }

            if ( !( dtva & 7 ) )
            {
                resmu = bias();
            }

            else
            {
                resmu  = bias();
                resmu *= 0.0;
            }
        }
    }

    return 0;
}

template <class BaseRegressorClass>
void SVM_Vector_redbin<BaseRegressorClass>::dgTrainingVector(Vector<gentype> &resx, int i) const
{
    Vector<gentype> res;
    gentype resn;
    gentype zv(0.0);

    dgTrainingVector(res,resn,i);

    resx.resize(xspaceDim()) = zv;

    int j,k;

    if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; j++ )
        {
            if ( x(j).nearindsize() )
            {
                for ( k = 0 ; k < x(j).nearindsize() ; k++ )
                {
                    resx("&",x(j).ind(k)) += ((res(j)*(x(j)).direcref(k)));
                }
            }
        }
    }

//    if ( i <= -1 )
    {
        if ( x(i).nearindsize() )
        {
            for ( k = 0 ; k < x(i).nearindsize() ; k++ )
            {
                resx("&",x(i).ind(k)) += ((resn*(x(i)).direcref(k)));
            }
        }
    }

    return;
}

template <class BaseRegressorClass>
void SVM_Vector_redbin<BaseRegressorClass>::dgTrainingVector(Vector<double> &resx, int i) const
{
    Vector<gentype> res;
    gentype resn;

    dgTrainingVector(res,resn,i);

    resx.resize(xspaceDim()) = 0.0;

    int j,k;

    if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; j++ )
        {
            if ( x(j).nearindsize() )
            {
                for ( k = 0 ; k < x(j).nearindsize() ; k++ )
                {
                    resx("&",x(j).ind(k)) += (double) ((res(j)*(x(j)).direcref(k)));
                }
            }
        }
    }

//    if ( i <= -1 )
    {
        if ( x(i).nearindsize() )
        {
            for ( k = 0 ; k < x(i).nearindsize() ; k++ )
            {
                resx("&",x(i).ind(k)) += (double) ((resn*(x(i)).direcref(k)));
            }
        }
    }

    return;
}






















template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::gTrainingVector(Vector<double> &gproject, int &dummy, int i, int raw, gentype ***pxyprodi) const
{
    int dtv = 0;

    dummy = 0;

    if ( i >= 0 )
    {
        gproject.resize(tspaceDim());

        int q;

        if ( tspaceDim() )
        {
            for ( q = 0 ; q < tspaceDim() ; q++ )
            {
                gentype tempsomeh,tempsomeg;

                //Q(q).gTrainingVector(gproject("&",q),dummy,i,raw);
                Q(q).ghTrainingVector(tempsomeh,tempsomeg,i,raw,pxyprodi);

                gproject("&",q) = (double) tempsomeg;
            }

            // Need to subtract diagonal offset here is either case, as Q(q) has no awareness of it.

            gproject.scaleAdd(-diagoff(i),dalpha(i));
        }
    }

    else if ( ( dtv = xtang(i) & 7 ) )
    {
        gproject.resize(tspaceDim()) = 0.0;

        if ( ( dtv > 0 ) && tspaceDim() && N() )
        {
            int j;
            double Kij;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( alphaState()(j) )
                {
                    K2(Kij,i,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                    gproject.scaleAdd(Kij,dalpha(j));
                }
            }
        }
    }

    else
    {
        gproject.resize(tspaceDim()) = db;

        if ( tspaceDim() )
        {
            if ( N() )
            {
                int j;
                double Kij;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        K2(Kij,i,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                        gproject.scaleAdd(Kij,dalpha(j));
                    }
                }
            }
        }
    }

    return 0;
}




template <class BaseRegressorClass>
void SVM_Vector_redbin<BaseRegressorClass>::recalcdiagoff(int i)
{
    NiceAssert( i >= -1 );
    NiceAssert( i < N() );

    int q;

    // This updates the diagonal offsets.

    if ( N() )
    {
	isStateOpt = 0;

	if ( i == -1 )
	{
	    Vector<double> bp(N());

	    for ( i = 0 ; i < N() ; i++ )
	    {
		bp("&",i) = -diagoff(i);
                diagoff("&",i) = QUADCOSTDIAGOFFSETB(trainclass(i),(Q(zeroint()).Cweight())(i),(Q(zeroint()).Cweightfuzz())(i));
		bp("&",i) += diagoff(i);
	    }

	    kerndiagval += bp;

	    kerncache.recalcDiag();

            sigmacache.clear();

            //int oldmemsize = sigmacache.get_memsize();
            //int oldrowdim  = sigmacache.get_min_rowdim();

            //sigmacache.reset(N(),&evalSigmaSVM_Vector_redbin,(void *) this);
            //sigmacache.setmemsize(oldmemsize,oldrowdim);

	    for ( q = 0 ; q < Q.size() ; q++ )
	    {
		Q("&",q).recalcdiagoff(bp);
	    }
	}

	else
	{
	    double bpoff = 0.0;

	    bpoff = -diagoff(i);
            diagoff("&",i) = QUADCOSTDIAGOFFSETB(trainclass(i),(Q(zeroint()).Cweight())(i),(Q(zeroint()).Cweightfuzz())(i));
	    bpoff += diagoff(i);

	    kerndiagval("&",i) += bpoff;

	    kerncache.recalcDiag(i);

	    sigmacache.remove(i);
	    sigmacache.add(i);

	    for ( q = 0 ; q < Q.size() ; q++ )
	    {
		Q("&",q).recalcdiagoff(i,bpoff);
	    }
	}
    }

    return;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::fixautosettings(int kernchange, int Nchange)
{
    int res = 0;

    if ( kernchange || Nchange )
    {
	switch ( autosetLevel )
	{
        case 1: { if ( Nchange ) { res = 1; autosetCscaled(autosetCvalx);                  } break; }
        case 2: {                  res = 1; autosetCKmean();                                 break; }
        case 3: {                  res = 1; autosetCKmedian();                               break; }
        case 4: {                  res = 1; autosetCNKmean();                                break; }
        case 5: {                  res = 1; autosetCNKmedian();                              break; }
	default: { break; }
	}
    }

    return res;
}

template <class BaseRegressorClass>
double SVM_Vector_redbin<BaseRegressorClass>::autosetkerndiagmean(void)
{
    Vector<int> dnonzero;

    if ( N()-NNC(0) )
    {
	int i,j = 0;

	for ( i = 0 ; i < N() ; i++ )
	{
	    if ( trainclass(i) != 0 )
	    {
		dnonzero.add(j);
		dnonzero("&",j) = i;

                j++;
	    }
	}
    }

    retVector<double> tmpva;

    return mean(kerndiagval(dnonzero,tmpva));
}

template <class BaseRegressorClass>
double SVM_Vector_redbin<BaseRegressorClass>::autosetkerndiagmedian(void)
{
    Vector<int> dnonzero;

    int i,j = 0;

    if ( N()-NNC(0) )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
	    if ( trainclass(i) != 0 )
	    {
		dnonzero.add(j);
		dnonzero("&",j) = i;

                j++;
	    }
	}
    }

    retVector<double> tmpva;

    return median(kerndiagval(dnonzero,tmpva),i);
}

template <class BaseRegressorClass>
void SVM_Vector_redbin<BaseRegressorClass>::setGp(Matrix<double> *extGp, Matrix<double> *extGpsigma, Matrix<double> *extxy, int refactsol)
{
    if ( N() )
    {
	isStateOpt = 0;
    }

    if ( Gplocal )
    {
	if ( extGp != NULL )
	{
            NiceAssert( extGpsigma != NULL );

	    MEMDEL(xyval);
	    xyval = NULL;

	    MEMDEL(Gpval);
	    Gpval = NULL;

	    MEMDEL(Gpsigma);
	    Gpsigma = NULL;

	    Gplocal = 0;

	    xyval   = extxy;
	    Gpval   = extGp;
	    Gpsigma = extGpsigma;
	}
    }

    else
    {
	if ( extGp != NULL )
	{
            NiceAssert( extGpsigma != NULL );

	    Gplocal = 0;

	    xyval   = extxy;
	    Gpval   = extGp;
            Gpsigma = extGpsigma;
	}

	else
	{
            NiceAssert( extGpsigma == NULL );

	    Gplocal = 1;

            MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &xycache   ,N(),N()));
            MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,N(),N()));
            MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,N(),N()));
	}
    }

    locsetGp(refactsol);

    return;
}

template <class BaseRegressorClass>
std::ostream &SVM_Vector_redbin<BaseRegressorClass>::printstream(std::ostream &output, int dep) const
{
    int i;

    repPrint(output,'>',dep) << "Vector redbin SVM\n\n";

    repPrint(output,'>',dep) << "Cost type:                       " << costType          << "\n";
    repPrint(output,'>',dep) << "Opt type (0 act, 1 smo, 2 d2c, 3 grad):  " << optType           << "\n";

    repPrint(output,'>',dep) << "C:                               " << CNval             << "\n";

    repPrint(output,'>',dep) << "Parameter autoset level:         " << autosetLevel      << "\n";
    repPrint(output,'>',dep) << "Parameter autoset C value:       " << autosetCvalx      << "\n";

    repPrint(output,'>',dep) << "XY cache details:                " << xycache           << "\n";
    repPrint(output,'>',dep) << "Kernel cache details:            " << kerncache         << "\n";
    repPrint(output,'>',dep) << "Sigma cache details:             " << sigmacache        << "\n";
    repPrint(output,'>',dep) << "Kernel diagonals:                " << kerndiagval       << "\n";
    repPrint(output,'>',dep) << "Diagonal offsets:                " << diagoff           << "\n";

    repPrint(output,'>',dep) << "Alpha:                           " << dalpha            << "\n";
    repPrint(output,'>',dep) << "Alpha state:                     " << dalphaState       << "\n";
    repPrint(output,'>',dep) << "Bias:                            " << db                << "\n";
    repPrint(output,'>',dep) << "Ns:                              " << Ns                << "\n";
    repPrint(output,'>',dep) << "Nnc:                             " << Nnc               << "\n";
    repPrint(output,'>',dep) << "Is SVM optimal:                  " << isStateOpt        << "\n";

    repPrint(output,'>',dep) << "Is fudgeing on: " << isFudged    << "\n";

    SVM_Generic::printstream(output,dep+1);

    repPrint(output,'>',dep) << "Training classes:                " << trainclass        << "\n";
    repPrint(output,'>',dep) << "Training targets:                " << traintarg         << "\n";

    repPrint(output,'>',dep) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    repPrint(output,'>',dep) << "Optimisation state:              " << (Q).size()        << "\n";

    for ( i = 0 ; i < (Q).size() ; i++ )
    {
        repPrint(output,'>',dep) << "Submachine " << i << ": " << (Q)(i) << "\n";
    }

    repPrint(output,'>',dep) << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";

    return output;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::prealloc(int expectedN)
{
    kerndiagval.prealloc(expectedN);
    diagoff.prealloc(expectedN);
    dalpha.prealloc(expectedN);
    dalphaState.prealloc(expectedN);
    trainclass.prealloc(expectedN);
    traintarg.prealloc(expectedN);
    xycache.prealloc(expectedN);
    kerncache.prealloc(expectedN);
    sigmacache.prealloc(expectedN);
    SVM_Generic::prealloc(expectedN);

    if ( Q.size() )
    {
        int i;

        for ( i = 0 ; i < Q.size() ; i++ )
        {
            Q("&",i).prealloc(expectedN);
        }
    }

    return 0;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::preallocsize(void) const
{
    return SVM_Generic::preallocsize();
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::randomise(double sparsity)
{
    NiceAssert( sparsity >= 0 );
    NiceAssert( sparsity <= 1 );

    int res = 0;
    int i,q;

    if ( tspaceDim() )
    {
        for ( q = 0 ; q < tspaceDim() ; q++ )
        {
            res |= Q("&",q).randomise(sparsity);
        }
    }

    if ( res )
    {
        // NB: FOLLOWING CODE TAKEN FROM TRAINING FUNCTION

        if ( tspaceDim() )
        {
            db = 0.0;
            dalphaState = zeroint();

            if ( N() )
            {
                for ( i = 0 ; i < N() ; i++ )
                {
                    dalpha("&",i) = 0.0;
                }
            }

            for ( q = 0 ; q < tspaceDim() ; q++ )
            {
                db("&",q) = Q(q).biasR();
    
                if ( N() )
                {
                    for ( i = 0 ; i < N() ; i++ )
                    {
                        dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
    
                        if ( (Q(q).alphaState())(i) )
                        {
                            dalphaState("&",i) = 1;
                        }
                    }
                }
            }
        }

        Ns = sum(dalphaState);

        isStateOpt = 0;

        SVM_Generic::basesetAlphaBiasFromAlphaBiasV();
    }

    return res;
}


template <class T>
void SVM_Vector_redbin<T>::fudgeOn(void)
{
    int q;

    isFudged = 1;

    for ( q = 0 ; q < tspaceDim() ; q++ )
    {
        Q("&",q).fudgeOn();
    }

    return;
}

template <class T>
void SVM_Vector_redbin<T>::fudgeOff(void)
{
    int q;

    isFudged = 0;

    for ( q = 0 ; q < tspaceDim() ; q++ )
    {
        Q("&",q).fudgeOff();
    }

    return;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::sety(const Vector<int> &j, const Vector<gentype> &yn)
{
    NiceAssert( j.size() == yn.size() );

    int res =0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
        {
            res |= SVM_Vector_redbin<BaseRegressorClass>::sety(j(i),yn(i));
        }
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::sety(const Vector<gentype> &yn)
{
    NiceAssert( N() == yn.size() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            res |= SVM_Vector_redbin<BaseRegressorClass>::sety(i,yn(i));
        }
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::sety(const Vector<int> &j, const Vector<Vector<double> > &z)
{
    NiceAssert( z.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= SVM_Vector_redbin<BaseRegressorClass>::sety(j(i),z(i));
	}
    }

    return res;
}

template <class BaseRegressorClass>
int SVM_Vector_redbin<BaseRegressorClass>::sety(const Vector<Vector<double> > &z)
{
    NiceAssert( z.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= SVM_Vector_redbin<BaseRegressorClass>::sety(i,z(i));
	}
    }

    return res;
}


#endif
