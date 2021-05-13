
//
// Vector (at once) regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_vector_atonce_temp_h
#define _svm_vector_atonce_temp_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <sstream>
#include "svm_generic.h"
#include "kcache.h"
#include "optstate.h"
#include "sQsLsAsWs.h"
#include "sQsmo.h"
#include "sQd2c.h"
#include "sQsmoVect.h"

template <class S> void evalKSVM_Vector_atonce_temp(S &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalSigmaSVM_Vector_atonce_temp_double(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalSigmaSVM_Vector_atonce_temp_matrix(double &res, int i, int j, const gentype **pxyprod, const void *owner);

template <class T> class SVM_Vector_atonce_temp;

Matrix<double> *alloc_gp(void *kerncache, int nrows, int ncols, const double &dummy);
Matrix<Matrix<double> > *alloc_gp(void *kerncache, int nrows, int ncols, const Matrix<double> &dummy);

// Swap function

template <class S> inline void qswap(SVM_Vector_atonce_temp<S> &a, SVM_Vector_atonce_temp<S> &b);

template <class T>
class SVM_Vector_atonce_temp : public SVM_Generic
{
    template <class S> friend void evalKSVM_Vector_atonce_temp(S &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalSigmaSVM_Vector_atonce_temp_double(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalSigmaSVM_Vector_atonce_temp_matrix(double &res, int i, int j, const gentype **pxyprod, const void *owner);

public:

    // Constructors, destructors, assignment operators and similar

    SVM_Vector_atonce_temp();
    SVM_Vector_atonce_temp(const SVM_Vector_atonce_temp<T> &src);
    SVM_Vector_atonce_temp(const SVM_Vector_atonce_temp<T> &src, const ML_Base *xsrc);
    SVM_Vector_atonce_temp<T> &operator=(const SVM_Vector_atonce_temp<T> &src) { assign(src); return *this; }
    virtual ~SVM_Vector_atonce_temp();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void setmemsize(int memsize);

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { SVM_Vector_atonce_temp<T> temp; *this = temp; return 1; }

    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha);
    virtual int setBiasV(const Vector<double> &newBias);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int isTrained(void) const { return isStateOpt; }

    virtual int N  (void)  const { return Q.aN();          }
    virtual int NS (void)  const { return Q.aNUF();        }
    virtual int NZ (void)  const { return Q.aNZ();         }
    virtual int NF (void)  const { return NS();            }
    virtual int NC (void)  const { return NZ();            }
    virtual int NLB(void)  const { return 0;               }
    virtual int NLF(void)  const { return 0;               }
    virtual int NUF(void)  const { return NF();            }
    virtual int NUB(void)  const { return 0;               }
    virtual int NNC(int d) const { return Nnc(d+1);        }
    virtual int NS (int q) const { (void) q; return NS();  }
    virtual int NZ (int q) const { (void) q; return NZ();  }
    virtual int NF (int q) const { (void) q; return NF();  }
    virtual int NC (int q) const { (void) q; return NC();  }
    virtual int NLB(int q) const { (void) q; return NLB(); }
    virtual int NLF(int q) const { (void) q; return NLF(); }
    virtual int NUF(int q) const { (void) q; return NUF(); }
    virtual int NUB(int q) const { (void) q; return NUB(); }

    virtual int tspaceDim(void)  const { return biasV().size();  }
    virtual int numClasses(void) const { return 0;               }
    virtual int type(void)       const { return 4;               }

    virtual int numInternalClasses(void) const { return 1; }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'V'; }
    virtual char targType(void) const { return 'V'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual const Vector<int>          &ClassLabels(void)   const { return classLabelsval;                    }
    virtual const Vector<Vector<int> > &ClassRep(void)      const { return classRepval;                       }
    virtual int                         findID(int ref)     const { NiceAssert( ref == 2 ); (void) ref; return 2; }

    virtual int isLinearCost(void)      const { return costType == 0;    }
    virtual int isQuadraticCost(void)   const { return costType == 1;    }
    virtual int is1NormCost(void)       const { return 0;                }
    virtual int isVarBias(void)         const { return 1;                }
    virtual int isPosBias(void)         const { return 0;                }
    virtual int isNegBias(void)         const { return 0;                }
    virtual int isFixedBias(void)       const { return 0;                }
    virtual int isVarBias(int d)        const { (void) d; return 1;      }
    virtual int isPosBias(int d)        const { (void) d; return 0;      }
    virtual int isNegBias(int d)        const { (void) d; return 0;      }
    virtual int isFixedBias(int d)      const { (void) d; return 0;      }

    virtual int isOptActive(void) const { return 1; }
    virtual int isOptSMO(void)    const { return 0; }
    virtual int isOptD2C(void)    const { return 0; }
    virtual int isOptGrad(void)   const { return 0; }

    virtual int m(void) const { return 2; }

    virtual double C(void)            const { return CNval;       }
    virtual double eps(void)          const { return epsval;      }
    virtual double Cclass(int d)      const { (void) d; return 1; }
    virtual double epsclass(int d)    const { (void) d; return 1; }

    virtual int    memsize(void)      const { return kerncache.get_memsize(); }
    virtual double zerotol(void)      const { return Q.zerotol();             }
    virtual double Opttol(void)       const { return opttolval;               }
    virtual int    maxitcnt(void)     const { return maxitcntval;             }
    virtual double maxtraintime(void) const { return maxtraintimeval;         }
    virtual double outerlr(void)      const { return MULTINORM_OUTERSTEP;     }
    virtual double outertol(void)     const { return MULTINORM_OUTERACCUR;    }

    virtual       int      maxiterfuzzt(void) const { return DEFAULT_MAXITERFUZZT;                                 }
    virtual       int      usefuzzt(void)     const { return 0;                                                    }
    virtual       double   lrfuzzt(void)      const { return DEFAULT_LRFUZZT;                                      }
    virtual       double   ztfuzzt(void)      const { return DEFAULT_ZTFUZZT;                                      }
    virtual const gentype &costfnfuzzt(void)  const { const static gentype temp(DEFAULT_COSTFNFUZZT); return temp; }

    virtual double LinBiasForce(void)        const { return 0;           }
    virtual double QuadBiasForce(void)       const { return 0;           }
    virtual double LinBiasForce(int q)       const { (void) q; return 0; }
    virtual double QuadBiasForce(int q)      const { (void) q; return 0; }

    virtual int isFixedTube(void)  const { return 1; }
    virtual int isShrinkTube(void) const { return 0; }

    virtual int isRestrictEpsPos(void) const { return 1; }
    virtual int isRestrictEpsNeg(void) const { return 0; }

    virtual double nu(void)     const { return 0; }
    virtual double nuQuad(void) const { return 0; }

    virtual int isClassifyViaSVR(void) const { return 1; }
    virtual int isClassifyViaSVM(void) const { return 0; }

    virtual int is1vsA(void)    const { return 0; }
    virtual int is1vs1(void)    const { return 0; }
    virtual int isDAGSVM(void)  const { return 0; }
    virtual int isMOC(void)     const { return 0; }
    virtual int ismaxwins(void) const { return 0; }
    virtual int isrecdiv(void)  const { return 0; }

    virtual int isatonce(void) const { return 1; }
    virtual int isredbin(void) const { return 0; }

    virtual int isKreal(void)   const;
    virtual int isKunreal(void) const;

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

    virtual double autosetCval(void)  const { return autosetCvalx;  }
    virtual double autosetnuval(void) const { return 0;            }

    virtual const Vector<int>                  &d          (void)      const { return trainclass;                                  }
    virtual const Vector<double>               &Cweight    (void)      const { return Cweightval;                                  }
    virtual const Vector<double>               &Cweightfuzz(void)      const { return Cweightvalfuzz;                              }
    virtual const Vector<double>               &epsweight  (void)      const { return epsweightval;                                }
    virtual const Vector<int>                  &alphaState (void)      const { return Q.alphaState();                              }
    virtual const Vector<Vector<double> >      &zV         (int raw=0) const { (void) raw; return traintarg;                       }
    virtual const Vector<double>               &biasV      (int raw=0) const { (void) raw; return Q.beta()(0);                     }
    virtual const Vector<Vector<double> >      &alphaV     (int raw=0) const { (void) raw; return Q.alpha();                       }
    virtual const Vector<Vector<double> >      &getu       (void)      const { return u;                                           }

    // Modification:

    virtual int setLinearCost(void);
    virtual int setQuadraticCost(void);

    virtual int setC(double xC);
    virtual int seteps(double xeps);

    virtual int setOptActive(void) { return 0; }
    virtual int setOptSMO(void)    { return 0; }
    virtual int setOptD2C(void)    { return 0; }
    virtual int setOptGrad(void)   { return 0; }

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

    virtual void prepareKernel(void) { return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1);

    virtual void fillCache(void);

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int addTrainingVector (int i, const Vector<double> &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<Vector<double> > &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
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

    virtual void fudgeOn(void)  { Q.fact_fudgeOn(*GpGrad,Gn,Gpn);  return; }
    virtual void fudgeOff(void) { Q.fact_fudgeOff(*GpGrad,Gn,Gpn); return; }

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // Other functions

    virtual void assign(const ML_Base &src, int isOnlySemi = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

private:

    virtual int gTrainingVector(Vector<double> &gproject, int &locclassrep, int i, int raw = 0, gentype ***pxyprodi = NULL) const;

    int costType;           // 0 = linear, 1 = LS
    double opttolval;       // optimality tolerance
    double CNval;           // C (tradeoff) value
    int maxitcntval;        // maximum number of iterations for training (or 0 if unlimited)
    double maxtraintimeval; // maximum time for training (or 0 if unlimited) - seconds
    double epsval;          // eps (tube) value
    int autosetLevel;       // 0 = none, 1 = C/N, 2 = Cmean, 3 = Cmedian, 4 = CNmean, 5 = CNmedian
    double autosetCvalx;    // Cval used if autosetLevel == 1,6

    Kcache<T> kerncache;              // kernel cache
    Kcache<double> sigmacache;        // sigma cache
    Vector<T> kerndiag;               // kernel diagonals
    Vector<double> diagoff;           // diagonal offset for hessian (used by quadratic cost)

    Vector<int>          classLabelsval; // Convenience: [ -1 +1 2 ] (-1,+1 unused)
    Vector<Vector<int> > classRepval;    // Convenience: [ [ -1 ] [ +1 ] [ 0 ] ] 
    Vector<Vector<double> > u; // Convenience, empty

    T dummyarg;

    // Optimisation state

    optState<Vector<double>,T> Q;
    Vector<int> Nnc; // number of vectors in each class (0,+2)
    int isStateOpt; // set if SVM is in optimal state

    // Training data

    Vector<Vector<double> > traintarg;
    Vector<int> trainclass;
    Vector<double> Cweightval;
    Vector<double> Cweightvalfuzz;
    Vector<double> epsweightval;

    // Quadratic program definition
    //
    // Gpsigma(i,j) = Gp(i,i)+Gp(j,j)-(2.0*Gp(i,j)) (assumes Gpn = 1, Gn = 0)
    //
    // NB: because this is a vector target regressor with potential matrix
    //     kernel elements, GpGrad is used instead of Gp.  Where Gp is
    //     required by Q as a "dummy" argument (dummy because the Hessian
    //     is never stored), Gpsigma is used as a stand-in.

    Matrix<T> *GpGrad;
    Matrix<double> *Gpsigma;
    Matrix<double> Gn;
    Matrix<double> Gpn;
    Vector<Vector<double> > gp;
    Vector<Vector<double> > gn;
    Vector<double> hp;
    Vector<double> ub;

    // Internal functions
    //
    // setGP: make the SVM access Gp externally.
    // setalleps: set eps and epsclass
    // recalcuLUB: recalculate lb and ub and adjust state accordingly
    // recalcdiagoff(i): recalculate and update diagoff(i), or all of diagoff if i == -1
    // recalcdiagoff(offsetvector): recalculate and update kernel diagonals if as offset has been added to them
    // recalcdiagoff(i,offset): recalculate and update kernel diagonal i as if offset has been added to them

    void setalleps(double xeps, const Vector<double> &xepsclass);
    void recalcLUB(int ival);
    void recalcdiagoff(int ival);
    void recalcdiagoff(const Vector<double> &offset);
    void recalcdiagoff(int ival, double offset);

    // Miscellaneous

    int fixautosettings(int kernchange, int Nchange);
    int setdinternal(int i, int d); // like setd, but without fixing auto settings
    double autosetkerndiagmean(void);
    double autosetkerndiagmedian(void);

    int qtaddTrainingVector(int i, const Vector<double> &z, double Cweigh, double epsweigh, int d);

    int presolveit(const Vector<double> &betaGrad);
};

template <class T>
inline void qswap(SVM_Vector_atonce_temp<T> &a, SVM_Vector_atonce_temp<T> &b)
{
    a.qswapinternal(b);

    return;
}

template <class T>
inline void SVM_Vector_atonce_temp<T>::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Vector_atonce_temp<T> &b = dynamic_cast<SVM_Vector_atonce_temp<T> &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(costType              ,b.costType              );
    qswap(maxitcntval           ,b.maxitcntval           );
    qswap(maxtraintimeval       ,b.maxtraintimeval       );
    qswap(opttolval             ,b.opttolval             );
    qswap(CNval                 ,b.CNval                 );
    qswap(epsval                ,b.epsval                );
    qswap(autosetLevel          ,b.autosetLevel          );
    qswap(autosetCvalx          ,b.autosetCvalx          );
    qswap(kerncache             ,b.kerncache             );
    qswap(sigmacache            ,b.sigmacache            );
    qswap(kerndiag              ,b.kerndiag              );
    qswap(diagoff               ,b.diagoff               );
    qswap(classLabelsval        ,b.classLabelsval        );
    qswap(classRepval           ,b.classRepval           );
    qswap(Q                     ,b.Q                     );
    qswap(Nnc                   ,b.Nnc                   );
    qswap(isStateOpt            ,b.isStateOpt            );
    qswap(traintarg             ,b.traintarg             );
    qswap(trainclass            ,b.trainclass            );
    qswap(Cweightval            ,b.Cweightval            );
    qswap(Cweightvalfuzz        ,b.Cweightvalfuzz        );
    qswap(epsweightval          ,b.epsweightval          );
    qswap(Gn                    ,b.Gn                    );
    qswap(Gpn                   ,b.Gpn                   );
    qswap(gp                    ,b.gp                    );
    qswap(gn                    ,b.gn                    );
    qswap(hp                    ,b.hp                    );
    qswap(ub                    ,b.ub                    );

    Matrix<T> *tGpGrad;
    Matrix<double> *tGpsigma;

    tGpGrad  = GpGrad;  GpGrad  = b.GpGrad;  b.GpGrad  = tGpGrad;
    tGpsigma = Gpsigma; Gpsigma = b.Gpsigma; b.Gpsigma = tGpsigma;

    // The kernel (and sigma) cache, as well as Gp (and Gpsigma) will have
    // been messed around by the above switching.  We need to make sure that
    // their pointers are set to rights before we continue.

    (kerncache).cheatSetEvalArg((void *) this);
    (sigmacache).cheatSetEvalArg((void *) this);

    (GpGrad)->cheatsetcdref((void *) &(kerncache));
    (Gpsigma)->cheatsetcdref((void *) &(sigmacache));

    (b.kerncache).cheatSetEvalArg((void *) &b);
    (b.sigmacache).cheatSetEvalArg((void *) &b);

    (b.GpGrad)->cheatsetcdref((void *) &(b.kerncache));
    (b.Gpsigma)->cheatsetcdref((void *) &(b.sigmacache));

    return;
}

template <class T>
inline void SVM_Vector_atonce_temp<T>::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Vector_atonce_temp<T> &b = dynamic_cast<const SVM_Vector_atonce_temp<T> &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

    //classLabelsval
    //classRepval
    //u

    //GpGrad;
    //Gpsigma;

    costType        = b.costType;
    opttolval       = b.opttolval;
    maxitcntval     = b.maxitcntval;
    maxtraintimeval = b.maxtraintimeval;
    autosetLevel    = b.autosetLevel;
    autosetCvalx    = b.autosetCvalx;

    dummyarg = b.dummyarg;

    traintarg      = b.traintarg;
    Cweightval     = b.Cweightval;
    Cweightvalfuzz = b.Cweightvalfuzz;
    epsweightval   = b.epsweightval;

    Gn  = b.Gn;
    Gpn = b.Gpn;
    gp  = b.gp;
    hp  = b.hp;

    isStateOpt = b.isStateOpt;

    CNval  = b.CNval;
    epsval = b.epsval;

    gn  = b.gn;
    ub  = b.ub;

    Q   = b.Q;
    Nnc = b.Nnc;

    trainclass = b.trainclass;

    if ( isQuadraticCost() )
    {
        kerndiag = b.kerndiag;
        diagoff  = b.diagoff;

        kerncache.recalcDiag();
        sigmacache.recalcDiag();
    }

    return;
}

template <class T>
inline void SVM_Vector_atonce_temp<T>::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Vector_atonce_temp<T> &src = dynamic_cast<const SVM_Vector_atonce_temp<T> &>(bb);

    SVM_Generic::assign(src,onlySemiCopy);

    costType               = src.costType;
    maxitcntval            = src.maxitcntval;
    maxtraintimeval        = src.maxtraintimeval;
    opttolval              = src.opttolval;
    CNval                  = src.CNval;
    epsval                 = src.epsval;
    autosetLevel           = src.autosetLevel;
    autosetCvalx           = src.autosetCvalx;

    kerndiag               = src.kerndiag;
    Q                      = src.Q;
    traintarg              = src.traintarg;
    trainclass             = src.trainclass;
    Cweightval             = src.Cweightval;
    Cweightvalfuzz         = src.Cweightvalfuzz;
    epsweightval           = src.epsweightval;
    diagoff                = src.diagoff;
    Nnc                    = src.Nnc;
    isStateOpt             = src.isStateOpt;

    Gn  = src.Gn;
    Gpn = src.Gpn;
    gp  = src.gp;
    gn  = src.gn;
    hp  = src.hp;
    ub  = src.ub;

    kerncache  = src.kerncache;
    sigmacache = src.sigmacache;

    kerncache.cheatSetEvalArg((void *) this);
    sigmacache.cheatSetEvalArg((void *) this);

    MEMDEL(GpGrad);
    GpGrad = NULL;

    MEMDEL(Gpsigma);
    Gpsigma = NULL;

    const static T dummy = src.dummyarg;

    GpGrad  = alloc_gp((void *) &kerncache,trainclass.size(),trainclass.size(),dummy);
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,trainclass.size(),trainclass.size()));

    classLabelsval = src.classLabelsval;
    classRepval    = src.classRepval;

    return;
}
















































typedef void (*evalCacheFn)(double &, int, int, const gentype **pxyprod, const void *);


// The following are non-template converter functions for kernel result
// conversion to either double or matrix, and the offset of same by
// a double.

void KFinaliser(Matrix<double> &res, gentype &src, int tspaceDim);
void KFinaliser(double &res, gentype &src, int tspaceDim);

void KOffset(Matrix<double> &res, double diagoff, int tspaceDim);
void KOffset(double &res, double diagoff, int tspaceDim);

evalCacheFn getsigmacallback(const double &dummy);
evalCacheFn getsigmacallback(const Matrix<double> &dummy);






#define HCALC(_costType_,_C_,_Cweigh_,_Cweighfuzz_)              ( (_costType_) ?  (MAXBOUND) :  ( (_C_) * (_Cweigh_) * (_Cweighfuzz_) ) )
#define HPCALC(_E_,_Eweigh_)                                     (                               ( (_E_) * (_Eweigh_) ) )
#define QUADCOSTDIAGOFFSET(_costType_,_C_,_Cweigh_,_Cweighfuzz_) ( (_costType_) ? ( 1 / ( (_C_) * (_Cweigh_) * (_Cweighfuzz_) ) ) : 0 )
#define ALPHARESTRICT(_xclass_)                                  ( ( (_xclass_) == +2 ) ? 0 : 3 )

// DEFZCMAXITCNT - zero crossing maximum iteration count
// DEFINMAXITCNT - inner loop of interior point solver max iteration count
// MUFACTOR - mu factor for t scaling in interior point algorithm

//#define DEFZCMAXITCNT 100
//#define DEFINMAXITCNT 100
// moved to mlcommon.h



template <class T>
void evalKSVM_Vector_atonce_temp(T &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    gentype tempres;

    SVM_Vector_atonce_temp<T> *realOwner = (SVM_Vector_atonce_temp<T> *) owner;

    NiceAssert( realOwner );

    if ( i != j )
    {
        realOwner->K2(tempres,i,j,pxyprod);
    }

    else
    {
        tempres = (realOwner->kerndiag)(i);
    }

    KFinaliser(res,tempres,realOwner->tspaceDim());

    if ( i == j )
    {
        KOffset(res,(realOwner->diagoff)(i),realOwner->tspaceDim());
    }

    return;
}

template <class T>
SVM_Vector_atonce_temp<T>::SVM_Vector_atonce_temp() : SVM_Generic()
{
    setaltx(NULL);

    costType        = 0;
    maxitcntval     = DEFAULT_MAXITCNT;
    maxtraintimeval = DEFAULT_MAXTRAINTIME;
    opttolval       = DEFAULT_OPTTOL;
    isStateOpt      = 1;
    CNval           = DEFAULT_C;
    epsval          = DEFAULTEPS;
    autosetLevel    = 0;
    autosetCvalx    = 0.0;

    kerncache.reset(0,&evalKSVM_Vector_atonce_temp,(void *) this);
    kerncache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    sigmacache.reset(0,getsigmacallback(dummyarg),(void *) this);
    sigmacache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    const static T dummy = dummyarg;

    GpGrad  = alloc_gp((void *) &kerncache,0,0,dummy);
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,0,0));

    Q.setkeepfact(*Gpsigma,Gn,Gpn,0);

    gn.add(0);
    (gn("&",0)).zero();

    Gn.addRowCol(0);
    Gn("&",0,0) = 0.0;

    Gpn.addCol(0);

    Vector<double> zeroeg;

    Q.addBeta(0,0,zeroeg);
    Q.setopttolhpzero(*GpGrad,Gn,Gpn,gp,gn,opttolval);

    Nnc.resize(4);
    Nnc = zeroint();

    classLabelsval.resize(3);
    classRepval.resize(3);

    classLabelsval("&",0) = -1;
    classLabelsval("&",1) = +1;
    classLabelsval("&",2) = 2;

    classRepval("&",0).resize(1); classRepval("&",0)("&",0) = -1;
    classRepval("&",1).resize(1); classRepval("&",1)("&",0) = +1;
    classRepval("&",2).resize(1); classRepval("&",2)("&",0) = 2;

    return;
}

template <class T>
SVM_Vector_atonce_temp<T>::SVM_Vector_atonce_temp(const SVM_Vector_atonce_temp<T> &src) : SVM_Generic()
{
    setaltx(NULL);

    kerncache.reset(0,&evalKSVM_Vector_atonce_temp,(void *) this);
    kerncache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    sigmacache.reset(0,getsigmacallback(dummyarg),(void *) this);
    sigmacache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    const static T dummy = dummyarg;

    GpGrad  = alloc_gp((void *) &kerncache,0,0,dummy);
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,0,0));

    assign(src,0);

    return;
}

template <class T>
SVM_Vector_atonce_temp<T>::SVM_Vector_atonce_temp(const SVM_Vector_atonce_temp<T> &src, const ML_Base *xsrc) : SVM_Generic()
{
    setaltx(xsrc);

    kerncache.reset(0,&evalKSVM_Vector_atonce_temp,(void *) this);
    kerncache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    sigmacache.reset(0,getsigmacallback(dummyarg),(void *) this);
    sigmacache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    const static T dummy = dummyarg;

    GpGrad  = alloc_gp((void *) &kerncache,0,0,dummy);
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,0,0));

    assign(src,1);

    return;
}

template <class T>
SVM_Vector_atonce_temp<T>::~SVM_Vector_atonce_temp()
{
    MEMDEL(GpGrad);
    GpGrad = NULL;

    MEMDEL(Gpsigma);
    Gpsigma = NULL;

    return;
}

int isKreal_nontemp(const double &dummy);
int isKunreal_nontemp(const double &dummy);
int isKreal_nontemp(const Matrix<double> &dummy);
int isKunreal_nontemp(const Matrix<double> &dummy);

template <class T>
int SVM_Vector_atonce_temp<T>::isKreal(void) const
{
    const static T temp = dummyarg;

    return isKreal_nontemp(temp);
}

template <class T>
int SVM_Vector_atonce_temp<T>::isKunreal(void) const
{
    const static T temp = dummyarg;

    return isKunreal_nontemp(temp);
}

template <class T>
double SVM_Vector_atonce_temp<T>::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setAlphaV(const Vector<Vector<double> > &newAlpha)
{
    if ( N() )
    {
	isStateOpt = 0;
    }

    Q.setAlphahpzero(newAlpha,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn,ub,ub);

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setBiasV(const Vector<double> &newBias)
{
    isStateOpt = 0;

    Vector<Vector<double> > newBeta(1);

    newBeta("&",0) = newBias;
    Q.setBetahpzero(newBeta,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);

    SVM_Generic::basesetbias(biasV());

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::scale(double a)
{
    NiceAssert( a >= 0.0 );
    NiceAssert( a <= 1.0 );

    int i,d;

    if ( a == 0.0 )
    {
	isStateOpt = 0;

	// Constrain all alphas to zero (use setd to cheat here)

	if ( N() )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
		d = trainclass(i);

		setdinternal(i,0);
                setdinternal(i,d);
	    }
	}

        Q.scalehpzero(a,*GpGrad,Gn,Gpn,gp,gn);
    }

    else if ( a < 1.0 )
    {
	isStateOpt = 0;

        // There are no alphas "at bounds" in the optstate sense
	// scale alpha and b

        Q.scalehpzero(a,*GpGrad,Gn,Gpn,gp,gn);
    }

    SVM_Generic::basescalealpha(a);
    SVM_Generic::basescalebias(a);

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::reset(void)
{
    Q.resethpzero(*Gpsigma,Gn,Gpn,gp,gn);

    isStateOpt = 0;

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setLinearCost(void)
{
    if ( isQuadraticCost() )
    {
	if ( N() )
	{
	    isStateOpt = 0;
	}

	costType = 0;
	recalcdiagoff(-1);
	recalcLUB(-1);
    }

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setQuadraticCost(void)
{
    if ( isLinearCost() )
    {
	if ( N() )
	{
	    isStateOpt = 0;
	}



	costType = 1;

	recalcdiagoff(-1);

        // NLB == NUB == 0

	recalcLUB(-1);
    }

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setOpttol(double xopttol)
{
    NiceAssert( xopttol >= 0 );

    if ( N() )
    {
	isStateOpt = 0;
    }

    opttolval = xopttol;

    Q.setopttolhpzero(*GpGrad,Gn,Gpn,gp,gn,opttolval);

    return 0;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setd(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( ( d == 0 ) || ( d == 2 ) );

    int res = 0;

    if ( d != trainclass(i) )
    {
        res = 1;
        int oldd = trainclass(i);

        res |= setdinternal(i,d);

	if ( !d || !oldd )
	{
            res |= fixautosettings(0,1);
	}
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::sety(int i, const gentype &z)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( z.isCastableToVectorWithoutLoss() );

    Vector<gentype> zz((const Vector<gentype> &) z);
    Vector<double> zzz(zz.size());

    if ( zz.size() )
    {
        int k;

        for ( k = 0 ; k < zz.size() ; k++ )
        {
            zzz("&",k) = (double) zz(k);
        }
    }

    return sety(i,zzz);
}

template <class T>
int SVM_Vector_atonce_temp<T>::sety(const Vector<int> &j, const Vector<gentype> &yn)
{
    NiceAssert( j.size() == yn.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
        {
            res |= sety(j(i),yn(i));
        }
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::sety(const Vector<gentype> &yn)
{
    NiceAssert( N() == yn.size() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    (void) onlyChangeRowI;

    int res = 0;

    if ( N() )
    {
        res |= 1;
        isStateOpt = 0;
    }

    res |= SVM_Generic::resetKernel(modind,onlyChangeRowI,updateInfo);

    kerncache.setSymmetry(getKernel().getSymmetry());
    sigmacache.setSymmetry(getKernel().getSymmetry());

    if ( N() )
    {
	int i;
        gentype tempres;

	for ( i = 0 ; i < N() ; i++ )
	{
            K2(tempres,i,i);
            KFinaliser(kerndiag("&",i),tempres,tspaceDim());
	}
    }

    kerncache.clear();
    sigmacache.clear();

    Q.refacthpzero(*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);

    res |= fixautosettings(1,0);

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    (void) onlyChangeRowI;

    int res = 0;

    if ( N() )
    {
        res |= 1;
        isStateOpt = 0;
    }

    res |= SVM_Generic::setKernel(xkernel,modind,onlyChangeRowI);

    kerncache.setSymmetry(getKernel().getSymmetry());
    sigmacache.setSymmetry(getKernel().getSymmetry());

    if ( N() )
    {
	int i;
        gentype tempres;

	for ( i = 0 ; i < N() ; i++ )
	{
            K2(tempres,i,i);
            KFinaliser(kerndiag("&",i),tempres,tspaceDim());
	}
    }

    kerncache.clear();
    sigmacache.clear();

    Q.refacthpzero(*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);

    res |= fixautosettings(1,0);

    return res;
}

template <class T>
void SVM_Vector_atonce_temp<T>::fillCache(void)
{
    if ( (*GpGrad).numRows() )
    {
        int i;

        retVector<T> tmpma;

        for ( i = 0 ; i < (*GpGrad).numRows() ; i++ )
        {
            (*GpGrad)(i,tmpma);
        }
    }

    return;
}

template <class T>
int SVM_Vector_atonce_temp<T>::sety(int i, const Vector<double> &z)
{
    NiceAssert( i >= -1 );
    NiceAssert( i < N() );

    int res = 0;

    isStateOpt = 0;

    Vector<Vector<double> > gpnew(gp);

    if ( i >= 0 )
    {
	traintarg("&",i) = z;
        gentype yn;
        yn = z;
        res |= SVM_Generic::sety(i,yn);

        gpnew("&",i) = z;
        (gpnew("&",i)).negate();

        Q.refactgphpzero(*GpGrad,Gn,Gpn,gp,gpnew,gn,i);

	gp("&",i) = gpnew(i);
    }

    else
    {
	traintarg = z;
        gentype yn;
        yn = z;
        res |= SVM_Generic::sety(i,yn);

        gpnew = z;
        gpnew.negate();

        Q.refactgphpzero(*GpGrad,Gn,Gpn,gp,gpnew,gn);

	gp = gpnew;
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setCweight(int i, double xC)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( xC > 0 );

    isStateOpt = 0;

    Cweightval("&",i) = xC;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(i);
    }

    else
    {
	recalcLUB(i);
    }

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setCweightfuzz(int i, double xC)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( xC > 0 );

    isStateOpt = 0;

    Cweightvalfuzz("&",i) = xC;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(i);
    }

    else
    {
	recalcLUB(i);
    }

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setepsweight(int i, double xepsweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    isStateOpt = 0;

    epsweightval("&",i) = xepsweight;
    hp("&",i) = HPCALC(epsval,epsweightval(i));

    return 0;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setd(const Vector<int> &j, const Vector<int> &d)
{
    NiceAssert( d.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setdinternal(j(i),d(i));
	}

        res |= fixautosettings(0,1);
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::sety(const Vector<int> &j, const Vector<Vector<double> > &z)
{
    NiceAssert( z.size() == j.size() );

    int res = 0;

    if ( z.size() )
    {
        isStateOpt = 0;

        Vector<Vector<double> > gpnew(gp);

        retVector<Vector<double> > tmpva;

        traintarg("&",j,tmpva) = z;

        Vector<gentype> zng(z.size());
        for ( int k = 0 ; k < z.size() ; k++ )
        {
            zng("&",k) = z(k);
        }
        res |= SVM_Generic::sety(j,zng);

        gpnew = z;
        gpnew.negate();

        Q.refactgphpzero(*GpGrad,Gn,Gpn,gp,gpnew,gn,-1);

        gp = gpnew;
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setCweight(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == j.size() );

    if ( j.size() )
    {
        retVector<double> tmpva;

        Cweightval("&",j,tmpva) = xCweight;

	isStateOpt = 0;

        if ( isQuadraticCost() )
        {
            recalcdiagoff(-1);
        }

        else
        {
            recalcLUB(-1);
        }
    }

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setCweightfuzz(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == j.size() );

    if ( j.size() )
    {
        retVector<double> tmpva;

        Cweightvalfuzz("&",j,tmpva) = xCweight;

	isStateOpt = 0;

        if ( isQuadraticCost() )
        {
            recalcdiagoff(-1);
        }

        else
        {
            recalcLUB(-1);
        }
    }

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setepsweight(const Vector<int> &j, const Vector<double> &xepsweight)
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

template <class T>
int SVM_Vector_atonce_temp<T>::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= setdinternal(i,d(i));
	}

        res |= fixautosettings(0,1);
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::sety(const Vector<Vector<double> > &z)
{
    NiceAssert( z.size() == N() );

    int res = 0;

    if ( N() )
    {
        isStateOpt = 0;

        Vector<Vector<double> > gpnew(gp);

        traintarg = z;

        Vector<gentype> zng(z.size());
        for ( int k = 0 ; k < z.size() ; k++ )
        {
            zng("&",k) = z(k);
        }
        res |= SVM_Generic::sety(zng);

        gpnew = z;
        gpnew.negate();

        Q.refactgphpzero(*GpGrad,Gn,Gpn,gp,gpnew,gn,-1);

        gp = gpnew;
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setCweight(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == N() );

    Cweightval = xCweight;

    if ( N() )
    {
	isStateOpt = 0;

        if ( isQuadraticCost() )
        {
            recalcdiagoff(-1);
        }

        else
        {
            recalcLUB(-1);
        }
    }

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setCweightfuzz(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == N() );

    Cweightvalfuzz = xCweight;

    if ( N() )
    {
	isStateOpt = 0;

        if ( isQuadraticCost() )
        {
            recalcdiagoff(-1);
        }

        else
        {
            recalcLUB(-1);
        }
    }

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setepsweight(const Vector<double> &xepsweight)
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

template <class T>
int SVM_Vector_atonce_temp<T>::setC(double xC)
{
    NiceAssert( xC > 0 );

    int res = 0;

    autosetOff();

    if ( N() )
    {
	isStateOpt = 0;
        res = 1;
    }

    if ( isQuadraticCost() )
    {
        CNval = xC;

	recalcdiagoff(-1);
    }

    else
    {
        Q.scalehpzero(xC/CNval,*GpGrad,Gn,Gpn,gp,gn);
        ub *= xC/CNval;

        CNval = xC;

	//recalcLUB(-1);
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::seteps(double xeps)
{
    int i;

    isStateOpt = 0;

    if ( autosetLevel == 6 )
    {
	autosetOff();
    }

    if ( N() )
    {
	isStateOpt = 0;
    }

    epsval = xeps;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            hp("&",i) = HPCALC(epsval,epsweightval(i));
	}
    }

    return 0;
}

template <class T>
int SVM_Vector_atonce_temp<T>::scaleCweight(double scalefactor)
{
    NiceAssert( scalefactor > 0 );

    if ( N() )
    {
	isStateOpt = 0;
    }

    Cweightval *= scalefactor;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else
    {
	recalcLUB(-1);
    }

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::scaleCweightfuzz(double scalefactor)
{
    NiceAssert( scalefactor > 0 );

    if ( N() )
    {
	isStateOpt = 0;
    }

    Cweightvalfuzz *= scalefactor;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else
    {
	recalcLUB(-1);
    }

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::scaleepsweight(double scalefactor)
{
    if ( N() )
    {
	isStateOpt = 0;
    }

    epsweightval *= scalefactor;
    hp           *= scalefactor;

    return 0;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setdinternal(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( ( d == 0 ) || ( d == 2 ) );

    int res = 0;

    if ( d != trainclass(i) )
    {
        res = 1;
        isStateOpt = 0;

        Nnc("&",trainclass(i)+1)--;
        Nnc("&",d+1)++;

        Q.changeAlphaRestricthpzero(i,3,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn); // this will also zero alpha

	int alphrestrict = ALPHARESTRICT(d);
	trainclass("&",i) = d;

        Q.changeAlphaRestricthpzero(i,alphrestrict,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);

        SVM_Generic::basesetalpha(i,alphaV()(i));
    }

    return res;
}

template <class T>
void SVM_Vector_atonce_temp<T>::setmemsize(int memsize)
{
    kerncache.setmemsize(memsize,kerncache.get_min_rowdim());
    sigmacache.setmemsize(memsize,sigmacache.get_min_rowdim());

    return;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setzerotol(double zt)
{
    isStateOpt = 0;

    Q.setzt(*Gpsigma,Gn,Gpn,zt);

    return 0;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setmaxitcnt(int xmaxitcnt)
{
    NiceAssert( xmaxitcnt >= 0 );

    maxitcntval = xmaxitcnt;

    return 0;
}

template <class T>
int SVM_Vector_atonce_temp<T>::setmaxtraintime(double xmaxtraintime)
{
    NiceAssert( xmaxtraintime >= 0 );

    maxtraintimeval = xmaxtraintime;

    return 0;
}

template <class T>
int SVM_Vector_atonce_temp<T>::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    Vector<double> zd;
    Vector<gentype> zz((const Vector<gentype> &) z);

    zd.resize(zz.size());

    if ( zz.size() )
    {
        int i;

        for ( i = 0 ; i < zz.size() ; i++ )
        {
            zd("&",i) = (double) zz(i);
        }
    }

    return SVM_Vector_atonce_temp<T>::addTrainingVector(i,zd,x,Cweigh,epsweigh,2);
}

template <class T>
int SVM_Vector_atonce_temp<T>::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    Vector<double> zd(z.size());
    Vector<gentype> zz((const Vector<gentype> &) z);

    zd.resize(zz.size());

    if ( zz.size() )
    {
        int i;

        for ( i = 0 ; i < zz.size() ; i++ )
        {
            zd("&",i) = (double) zz(i);
        }
    }

    return SVM_Vector_atonce_temp<T>::qaddTrainingVector(i,zd,x,Cweigh,epsweigh,2);
}

template <class T>
int SVM_Vector_atonce_temp<T>::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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
            res |= SVM_Vector_atonce_temp<T>::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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
            res |= SVM_Vector_atonce_temp<T>::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}


template <class T>
int SVM_Vector_atonce_temp<T>::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    if ( alphaState()(i) )
    {
        res = 1;
        isStateOpt = 0;
    }

    Nnc("&",trainclass(i)+1)--;

    if ( Q.alphaRestrict(i) != 3 )
    {
        Q.changeAlphaRestricthpzero(i,3,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);
    }

    res |= SVM_Generic::removeTrainingVector(i,y,x);

    traintarg.remove(i);
    trainclass.remove(i);
    Cweightval.remove(i);
    Cweightvalfuzz.remove(i);
    epsweightval.remove(i);
    kerndiag.remove(i);

    GpGrad->removeRowCol(i);
    Gpsigma->removeRowCol(i);

    Gpn.removeRow(i);

    gp.remove(i);
    hp.remove(i);
    ub.remove(i);
    diagoff.remove(i);
    kerncache.remove(i);
    sigmacache.remove(i);

    Q.removeAlpha(i);

    // Fix the cache

    if ( ( kerncache.get_min_rowdim() >= (int) (N()*ROWDIMSTEPRATIO) ) && ( N() > MINROWDIM ) )
    {
	kerncache.setmemsize(memsize(),N()-1);
	sigmacache.setmemsize(memsize(),N()-1);
    }

    res |= fixautosettings(0,1);

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::addTrainingVector(int i, const Vector<double> &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    int res = 0;

    if ( tspaceDim() != z.size() )
    {
        res |= settspaceDim(z.size());
    }

    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( ( d == 0 ) || ( d == 2 ) );
    NiceAssert( !N() || ( biasV().size() == z.size() ) );

    isStateOpt = 0;

    Nnc("&",d+1)++;

    if ( kerncache.get_min_rowdim() <= N() )
    {
	kerncache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	sigmacache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
    }

    gentype yn;
    yn = z;
    res |= SVM_Generic::addTrainingVector(i,yn,x);
    res |= qtaddTrainingVector(i,z,Cweigh,epsweigh,d);

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::qaddTrainingVector(int i, const Vector<double> &z, SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    int res = 0;

    if ( tspaceDim() != z.size() )
    {
        res |= settspaceDim(z.size());
    }

    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( ( d == 0 ) || ( d == 2 ) );
    NiceAssert( !N() || ( biasV().size() == z.size() ) );

    isStateOpt = 0;

    Nnc("&",d+1)++;

    if ( kerncache.get_min_rowdim() <= N() )
    {
	kerncache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	sigmacache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
    }

    gentype yn;
    yn = z;
    res |= SVM_Generic::qaddTrainingVector(i,yn,x);
    res |= qtaddTrainingVector(i,z,Cweigh,epsweigh,d);

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::addTrainingVector(int i, const Vector<Vector<double> > &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
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
            res |= SVM_Vector_atonce_temp<T>::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    return res;
}

template <class T>
int SVM_Vector_atonce_temp<T>::qaddTrainingVector(int i, const Vector<Vector<double> > &z,      Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
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
            res |= SVM_Vector_atonce_temp<T>::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    return res;
}








void redimelmsvm(Vector<double> &x, int olddim, int newdim);
void addfeatsvm(Vector<double> &x, int iii, int dummy);
void removefeatsvm(Vector<double> &x, int iii, int dummy);

template <class T>
int SVM_Vector_atonce_temp<T>::settspaceDim(int newdim)
{
    NiceAssert( newdim >= 0 );
    int olddim = tspaceDim();

    if ( newdim != tspaceDim() )
    {
        isStateOpt = 0;

        Q.redimensionalise(&redimelmsvm,olddim,newdim);

        int i;

        if ( N() )
        {
            retVector<double> tmpva;

            for ( i = 0 ; i < N() ; i++ )
            {
                traintarg("&",i).resize(newdim);
                gp("&",i).resize(newdim);

                if ( newdim > olddim )
		{
                    traintarg("&",i)("&",olddim,1,newdim-1,tmpva) = 0.0;
                    gp("&",i)("&",olddim,1,newdim-1,tmpva) = 0.0;
		}

                gentype yn;
                yn = traintarg(i);
                SVM_Generic::sety(i,yn);
            }
        }

        gn("&",zeroint()).resize(newdim);

        if ( newdim > olddim )
        {
            retVector<double> tmpva;

            gn("&",zeroint())("&",olddim,1,newdim-1,tmpva) = 0.0;
        }

        Q.refacthpzero(*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::addtspaceFeat(int iii)
{
    NiceAssert( iii >= 0 );
    NiceAssert( iii <= tspaceDim() );

    {
        isStateOpt = 0;

        Q.redimensionalise(&addfeatsvm,iii,iii);

        int i;

        if ( N() )
        {
            for ( i = 0 ; i < N() ; i++ )
            {
                traintarg("&",i).add(iii);
                gp("&",i).add(iii);
                traintarg("&",i)("&",iii) = 0.0;
                gp("&",i)("&",iii) = 0.0;

                gentype yn;
                yn = traintarg(i);
                SVM_Generic::sety(i,yn);
            }
        }

        gn("&",zeroint()).add(iii);
        gn("&",zeroint())("&",iii) = 0.0;

        Q.refacthpzero(*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::removetspaceFeat(int iii)
{
    NiceAssert( iii >= 0 );
    NiceAssert( iii < tspaceDim() );

    {
        isStateOpt = 0;

        Q.redimensionalise(&removefeatsvm,iii,iii);
        int i;

        if ( N() )
        {
            for ( i = 0 ; i < N() ; i++ )
            {
                traintarg("&",i).remove(iii);
                gp("&",i).remove(iii);

                gentype yn;
                yn = traintarg(i);
                SVM_Generic::sety(i,yn);
            }
        }

        gn("&",zeroint()).remove(iii);

        Q.refacthpzero(*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::qtaddTrainingVector(int i, const Vector<double> &z, double Cweigh, double epsweigh, int d)
{
    gentype tempres;

    int res = 0;

    traintarg.add(i);
    traintarg("&",i) = z;
    trainclass.add(i);
    trainclass("&",i) = d;
    Cweightval.add(i);
    Cweightval("&",i) = Cweigh;
    Cweightvalfuzz.add(i);
    Cweightvalfuzz("&",i) = 1.0;
    epsweightval.add(i);
    epsweightval("&",i) = epsweigh;
    diagoff.add(i);
    diagoff("&",i) = QUADCOSTDIAGOFFSET(costType,CNval,Cweigh,1.0);
    kerndiag.add(i);
    K2(tempres,i,i);
    KFinaliser(kerndiag("&",i),tempres,tspaceDim());

    GpGrad->addRowCol(i);
    Gpsigma->addRowCol(i);

    Gpn.addRow(i);
    Gpn("&",i,0) = 1.0;

    gp.add(i);
    gp("&",i) = z;
    gp("&",i).negate();
    hp.add(i);
    hp("&",i) = HPCALC(epsval,epsweigh);
    ub.add(i);
    ub("&",i) = HCALC(costType,CNval,Cweigh,1.0);
    kerncache.add(i);
    sigmacache.add(i);

    int alphrestrict = ALPHARESTRICT(d);

    Q.addAlpha(i,alphrestrict,biasV());
    Q.fixGradhpzero(*GpGrad,Gn,Gpn,gp,gn);

    res |= fixautosettings(0,1);

    SVM_Generic::basesetalpha(i,alphaV()(i));

    return res;
}


template <class T>
int SVM_Vector_atonce_temp<T>::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int unusedvar = 0;
    int tempresh = 0;
    Vector<double> tempresg;

    tempresh = gTrainingVector(tempresg,unusedvar,i,retaltg,pxyprodi);
    resh = tempresg; // processed output of regressor is scalar
    resg = tempresg;

    return tempresh;
}

template <class T>
int SVM_Vector_atonce_temp<T>::gTrainingVector(Vector<double> &res, int &locclassrep, int i, int raw, gentype ***pxyprodi) const
{
    int dtv = 0;

    if ( i >= 0 )
    {
        Q.unAlphaGradhpzero(res,i,*GpGrad,Gpn,gp);

        res += traintarg(i);
        res.scaleAdd(-diagoff(i),alphaV()(i));

        locclassrep = raw;
    }

    else if ( ( dtv = xtang(i) & 7 ) )
    {
        res.resize(tspaceDim()) = 0.0;

        if ( dtv > 0 )
        {
            int iP;
            T Kix;
            gentype tempres;

            // NLB = NUB = 0

            if ( NF() )
            {
                for ( iP = 0 ; iP < NF() ; iP++ )
                {
                    K2(tempres,i,(Q.pivAlphaF()(iP)),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaF()(iP)] : NULL);
                    KFinaliser(Kix,tempres,tspaceDim());
                    res += (Kix*((alphaV())(((Q.pivAlphaF())(iP)))));
                }
            }
        }

        locclassrep = raw;
    }

    else
    {
        int iP;
        T Kix;
        gentype tempres;

        res.resize(tspaceDim()) = biasV();

        // NLB = NUB = 0

        if ( NF() )
        {
            for ( iP = 0 ; iP < NF() ; iP++ )
            {
                K2(tempres,i,(Q.pivAlphaF()(iP)),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaF()(iP)] : NULL);
                KFinaliser(Kix,tempres,tspaceDim());
                res += (Kix*((alphaV())(((Q.pivAlphaF())(iP)))));
            }
        }

        locclassrep = raw;
    }

    return 0;
}









template <class T>
void SVM_Vector_atonce_temp<T>::recalcLUB(int ival)
{
    // This updates the ub (upper bound on alpha) vector
    //
    // If ival == -1 then all i is scanned.  Otherwise it only does a specific
    // value.

    NiceAssert( ival >= -1 );
    NiceAssert( ival < N() );

    if ( N() )
    {
	isStateOpt = 0;

	int i,imin,imax;

	if ( ival == -1 )
	{
	    imin = 0;
	    imax = N();
	}

	else
	{
	    imin = ival;
	    imax = ival+1;
	}

	if ( imax > imin )
	{
	    for ( i = imin ; i < imax ; i++ )
	    {
                ub("&",i) = HCALC(costType,CNval,Cweightval(i),Cweightvalfuzz(i));

                // alphaState == 0,1

                if ( alphaState()(i) )
		{
                    if ( abs2(alphaV()(i)) > ub("&",i) )
		    {
                        Q.alphaStephpzero(i,-alphaV()(i)+((ub(i))*angle(alphaV()(i))),*GpGrad,Gn,Gpn,gp,gn,1);
		    }
		}
	    }
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return;
}

template <class T>
void SVM_Vector_atonce_temp<T>::recalcdiagoff(int i)
{
    NiceAssert( i >= -1 );
    NiceAssert( i < N() );

    // This updates the diagonal offsets.

    if ( N() )
    {
	isStateOpt = 0;

	if ( i == -1 )
	{
	    Vector<double> bp(N());
            Vector<double> bn(1);

	    bn.zero();

	    for ( i = 0 ; i < N() ; i++ )
	    {
                bp("&",i) = -diagoff(i);
                diagoff("&",i) = QUADCOSTDIAGOFFSET(costType,CNval,Cweightval(i),Cweightvalfuzz(i));
		bp("&",i) += diagoff(i);
	    }

	    kerncache.recalcDiag();

            sigmacache.clear();

            //sigmacache.reset(N(),getsigmacallback(dummyarg),(void *) this);
	    //sigmacache.setmemsize(kerncache.get_memsize(),kerncache.get_min_rowdim());

            Q.diagoffsethpzero(bp,bn,bp,bn,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);
	}

	else
	{
	    double bpoff = 0.0;

	    bpoff = -diagoff(i);
            diagoff("&",i) = QUADCOSTDIAGOFFSET(costType,CNval,Cweightval(i),Cweightvalfuzz(i));
	    bpoff += diagoff(i);

	    kerncache.recalcDiag(i);

	    sigmacache.remove(i);
	    sigmacache.add(i);

            //sigmacache.reset(N(),getsigmacallback(dummyarg),(void *) this);
	    //sigmacache.setmemsize(kerncache.get_memsize(),kerncache.get_min_rowdim());

            Q.diagoffsethpzero(i,bpoff,bpoff,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);
	}
    }

    return;
}

template <class T>
void SVM_Vector_atonce_temp<T>::recalcdiagoff(const Vector<double> &offset)
{
    NiceAssert( offset.size() == N() );

    if ( N() )
    {
	isStateOpt = 0;

        Vector<double> bn(1);

	bn.zero();

	kerndiag += offset;
	kerncache.recalcDiag();

        sigmacache.clear();

        //sigmacache.reset(N(),getsigmacallback(dummyarg),(void *) this);
	//sigmacache.setmemsize(kerncache.get_memsize(),kerncache.get_min_rowdim());

        Q.diagoffsethpzero(offset,bn,offset,bn,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);
    }

    return;
}

template <class T>
void SVM_Vector_atonce_temp<T>::recalcdiagoff(int i, double offset)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    isStateOpt = 0;

    kerndiag("&",i) += offset;
    kerncache.recalcDiag(i);

    sigmacache.clear();

    //sigmacache.reset(N(),getsigmacallback(dummyarg),(void *) this);
    //sigmacache.setmemsize(kerncache.get_memsize(),kerncache.get_min_rowdim());

    Q.diagoffsethpzero(i,offset,offset,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);

    return;
}


template <class T>
int SVM_Vector_atonce_temp<T>::fixautosettings(int kernchange, int Nchange)
{
    int res = 0;

    if ( kernchange || Nchange )
    {
	switch ( autosetLevel )
	{
        case 1: { if ( Nchange ) { res = 1; autosetCscaled(autosetCvalx); } break; }
        case 2: {                  res = 1; autosetCKmean();                break; }
        case 3: {                  res = 1; autosetCKmedian();              break; }
        case 4: {                  res = 1; autosetCNKmean();               break; }
        case 5: {                  res = 1; autosetCNKmedian();             break; }
	default: { break; }
	}
    }

    return res;
}

template <class T>
double SVM_Vector_atonce_temp<T>::autosetkerndiagmean(void)
{
    Vector<int> dnonzero;

    if ( N()-NNC(0) )
    {
	int i,j = 0;

	for ( i = 0 ; i < N() ; i++ )
	{
            if ( trainclass(i) )
	    {
		dnonzero.add(j);
		dnonzero("&",j) = i;

                j++;
	    }
	}
    }

    retVector<T> tmpva;

    return (double) mean(kerndiag(dnonzero,tmpva));
}

template <class T>
double SVM_Vector_atonce_temp<T>::autosetkerndiagmedian(void)
{
    Vector<int> dnonzero;

    int i,j = 0;

    if ( N()-NNC(0) )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            if ( trainclass(i) )
	    {
		dnonzero.add(j);
		dnonzero("&",j) = i;

                j++;
	    }
	}
    }

    retVector<T> tmpva;

    return (double) median(kerndiag(dnonzero,tmpva),i);
}

template <class T>
int SVM_Vector_atonce_temp<T>::train(int &res, svmvolatile int &killSwitch)
{
    Vector<double> betaGrad = sum(alphaV());

//    if ( abs2(betaGrad) > Opttol() )
//    {
//        if ( ( res |= presolveit(betaGrad) ) )
//        {
//            return 1;
//        }
//    }

    {
//        fullOptStateSMOVect<T> xx(Q,*GpGrad,*Gpsigma,Gn,Gpn,gp,gn,hp,ub,maxitcntval,(int) maxtraintimeval,NULL,NULL,1,DEFZCMAXITCNT,DEFINMAXITCNT,Opttol(),zerotol(),zerotol(),biasV().size());
//
//        res = xx.wrapsolve(killSwitch);

        int oldGpSize = (*GpGrad).numRows();
        int oldGpsigmaSize = (*Gpsigma).numRows();

        kerncache.padCol(4*(Gpn.numCols()));
        sigmacache.padCol(4*(Gpn.numCols()));
        (*GpGrad).resize(oldGpSize,oldGpSize+(2*(Gpn.numCols())));
        (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize+(2*(Gpn.numCols())));

        {
            fullOptStateSMOVect<T> xx(Q,*GpGrad,*Gpsigma,Gn,Gpn,gp,gn,hp,ub,NULL,NULL,1,DEFZCMAXITCNT,DEFINMAXITCNT,Opttol(),zerotol(),zerotol(),biasV().size());

            xx.maxitcnt   = maxitcntval;
            xx.maxruntime = maxtraintimeval;

            res = xx.wrapsolve(killSwitch);
        }

        kerncache.padCol(0);
        sigmacache.padCol(0);
        (*GpGrad).resize(oldGpSize,oldGpSize);
        (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize);
    }

//    res |= solve_quadratic_program_smoVect(killSwitch,
//                                           Q,
//                                           *GpGrad,
//                                           *Gpsigma,
//                                           Gn,
//                                           Gpn,
//                                           gp,
//                                           gn,
//                                           hp,
//                                           ub,
//                                           maxitcntval,
//                                           (int) maxtraintimeval,
//                                           DEFZCMAXITCNT,
//                                           DEFINMAXITCNT,
//                                           Opttol(),
//                                           zerotol(),
//                                           zerotol(),
//                                           biasV().size());

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

template <class T>
int SVM_Vector_atonce_temp<T>::presolveit(const Vector<double> &betaGrad)
{
    int i;
    Vector<double> betaGradUnit(betaGrad);
    Vector<double> alphaProject(N());
    Vector<Vector<double> > alphaRest(N());
    Vector<double> alphaRestNorm(N());
    Vector<double> alphaProjectLB(N());
    double absbetaGrad = abs2(betaGrad);
    int isfeasible = 0;
    double iota = zerotol();

    betaGradUnit /= absbetaGrad;

    for ( i = 0 ; i < N() ; i++ )
    {
        twoProductNoConj(alphaProject("&",i),betaGradUnit,alphaV()(i));

        alphaRest("&",i) =  betaGradUnit;
        alphaRest("&",i) *= -alphaProject(i);
        alphaRest("&",i) += alphaV()(i);

        alphaRestNorm("&",i) = norm2(alphaRest(i));
    }

    // Need to change sum of alphaProject without changing sum of alphaRest

    if ( NF() )
    {
        // Find max change in alphaProject that will still leave alpha
        // withing relevant bounds

        for ( i = 0 ; i < NF() ; i++ )
        {
            alphaProjectLB("&",(Q.pivAlphaF())(i)) = -sqrt(((ub((Q.pivAlphaF())(i))-iota)*(ub((Q.pivAlphaF())(i))-iota))-alphaRestNorm((Q.pivAlphaF())(i)));
        }

        // First try: change alphaProject step alone in free alphas, change
        // nothing else.

        double maxchange = 0;

        for ( i = 0 ; i < NF() ; i++ )
        {
            maxchange += (alphaProjectLB((Q.pivAlphaF())(i))-alphaProject((Q.pivAlphaF())(i)));
        }

        if ( absbetaGrad+maxchange <= Opttol() )
        {
            // Can satisfy the inequality by taking a scaled step in the
            // projection alone.
            //
            // x.maxchange + absbetaGrad = 0
            // => x = (0-absbetaGrad)/maxchange

            double modscale = ( ( ( (maxchange+absbetaGrad) <= 0 ) ? 0 : (maxchange+absbetaGrad) ) - absbetaGrad ) / maxchange;

            for ( i = 0 ; i < NF() ; i++ )
            {
                alphaProject("&",(Q.pivAlphaF())(i)) += modscale*(alphaProjectLB((Q.pivAlphaF())(i))-alphaProject((Q.pivAlphaF())(i)));
            }

            isfeasible = 1;
            absbetaGrad = 0;
        }

        else
        {
            // Maximum step in projection is not enough.  Take anyhow,
            // then continue

            for ( i = 0 ; i < NF() ; i++ )
            {
                alphaProject("&",(Q.pivAlphaF())(i)) = alphaProjectLB((Q.pivAlphaF())(i));
            }

            absbetaGrad += maxchange;

            // Second try: change alphaProject with scaling in alphaRest

            maxchange = 0;

            for ( i = 0 ; i < NF() ; i++ )
            {
                alphaProjectLB("&",(Q.pivAlphaF())(i)) = -(ub((Q.pivAlphaF())(i))-iota);
                maxchange += (alphaProjectLB((Q.pivAlphaF())(i))-alphaProject((Q.pivAlphaF())(i)));
            }

            if ( absbetaGrad+maxchange <= Opttol() )
            {
                // Can satisfy the inequality by taking a scaled step in the
                // projection plus a scaling of alphaRest.
                //
                // x.maxchange + absbetaGrad = 0
                // => x = -absbetaGrad/maxchange

                double modscale = ( ( ( (maxchange+absbetaGrad) <= 0 ) ? 0 : (maxchange+absbetaGrad) ) - absbetaGrad ) / maxchange;

                // ap = alphaProject
                // ar = alphaRest
                //
                //    sqrt(norm2(ap(i)+x.(-(ub(i)-iota)-ap(i)))) + norm2(y.ar(i))) <= (ub(i)-iota)
                // => norm2(ap(i)+x.(-(ub(i)-iota)-ap(i)))) + y*y.norm2(ar(i)) <= norm2(ub(i)-iota)
                // => y <= sqrt( ( norm2(ub(i)-iota) - norm2(ap(i)+x.(-(ub(i)-iota)-ap(i)))) ) / norm2(ar(i)) )

                double arscale = 1;
                double arscalebnd;
                
                for ( i = 0 ; i < NF() ; i++ )
                {
                    arscalebnd = sqrt( ( norm2( alphaProjectLB("&",(Q.pivAlphaF())(i)) ) -
                                         norm2( alphaProject((Q.pivAlphaF())(i)) + (modscale*(alphaProjectLB("&",(Q.pivAlphaF())(i))-alphaProject((Q.pivAlphaF())(i)))) )
                                       ) / alphaRestNorm((Q.pivAlphaF())(i)) );

                    arscale = ( arscale < arscalebnd ) ? arscale : arscalebnd;
                }

                for ( i = 0 ; i < NF() ; i++ )
                {
                    alphaProject("&",(Q.pivAlphaF())(i)) += modscale*(alphaProjectLB("&",(Q.pivAlphaF())(i))-alphaProject((Q.pivAlphaF())(i)));
                    alphaRest("&",(Q.pivAlphaF())(i)) *= arscale;
                }

                isfeasible = 1;
                absbetaGrad = 0;
            }

            else
            {
                // Take maximum step and zero alphaRest

                for ( i = 0 ; i < NF() ; i++ )
                {
                    alphaProject("&",(Q.pivAlphaF())(i)) = alphaProjectLB("&",(Q.pivAlphaF())(i));
                    alphaRest("&",(Q.pivAlphaF())(i)).zero();
                }

                absbetaGrad += maxchange;
            }
        }
    }

    while ( !isfeasible && NZ() )
    {
        i = (Q.pivAlphaF())(zeroint());
        Q.modAlphaZtoUFhpzero(0,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn);

        if ( absbetaGrad-(ub(i)-iota) <= Opttol() )
        {
            alphaProject("&",i) = ( absbetaGrad-(ub(i)-iota) <= 0 ) ? 0 : (absbetaGrad-(ub(i)-iota));
            absbetaGrad = 0;
            isfeasible = 1;
        }

        else
        {
            alphaProject("&",i) = -(ub(i)-iota);
            absbetaGrad += -(ub(i)-iota);
        }
    }

    Vector<Vector<double> > newalpha(alphaV());

    for ( i = 0 ; i < N() ; i++ )
    {
        newalpha("&",i) = alphaRest(i);
        newalpha("&",i).scaleAdd(alphaProject(i),betaGradUnit);
    }

    Q.setAlphahpzero(newalpha,*Gpsigma,*GpGrad,Gn,Gpn,gp,gn,ub,ub);

    return !isfeasible;
}



// Stream operators

template <class T>
std::ostream &SVM_Vector_atonce_temp<T>::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector atonce template SVM\n\n";

    repPrint(output,'>',dep) << "Cost type (0 linear, 1 LS):      " << costType               << "\n";
    repPrint(output,'>',dep) << "Maximum training iterations:     " << maxitcntval            << "\n";
    repPrint(output,'>',dep) << "Maximum training time (sec):     " << maxtraintimeval        << "\n";
    repPrint(output,'>',dep) << "Optimal tolerance:               " << opttolval              << "\n";
    repPrint(output,'>',dep) << "C:                               " << CNval                  << "\n";
    repPrint(output,'>',dep) << "eps:                             " << epsval                 << "\n";
    repPrint(output,'>',dep) << "Parameter autoset level:         " << autosetLevel           << "\n";
    repPrint(output,'>',dep) << "Parameter autoset C value:       " << autosetCvalx           << "\n";

    repPrint(output,'>',dep) << "Kernel cache details:            " << kerncache              << "\n";
    repPrint(output,'>',dep) << "Sigma cache details:             " << sigmacache             << "\n";
    repPrint(output,'>',dep) << "Kernel diagonals:                " << kerndiag               << "\n";
    repPrint(output,'>',dep) << "Diagonal offsets:                " << diagoff                << "\n";

    repPrint(output,'>',dep) << "Nnc:                             " << Nnc                    << "\n";
    repPrint(output,'>',dep) << "Is SVM optimal:                  " << isStateOpt             << "\n";

    SVM_Generic::printstream(output,dep+1);

    repPrint(output,'>',dep) << "Training targets:                " << traintarg              << "\n";
    repPrint(output,'>',dep) << "Training classes:                " << trainclass             << "\n";
    repPrint(output,'>',dep) << "Training C weights:              " << Cweightval             << "\n";
    repPrint(output,'>',dep) << "Training C weights (fuzz):       " << Cweightvalfuzz         << "\n";
    repPrint(output,'>',dep) << "Training eps weights:            " << epsweightval           << "\n";

    repPrint(output,'>',dep) << "Gn:                              " << Gn                     << "\n";
    repPrint(output,'>',dep) << "Gpn:                             " << Gpn                    << "\n";
    repPrint(output,'>',dep) << "gp:                              " << gp                     << "\n";
    repPrint(output,'>',dep) << "gn:                              " << gn                     << "\n";
    repPrint(output,'>',dep) << "hp:                              " << hp                     << "\n";
    repPrint(output,'>',dep) << "ub:                              " << ub                     << "\n";

    repPrint(output,'>',dep) << "*********************************************************************\n";
    repPrint(output,'>',dep) << "Optimisation state:              " << Q                      << "\n";
    repPrint(output,'>',dep) << "#####################################################################\n";

    return output;
}

template <class T>
std::istream &SVM_Vector_atonce_temp<T>::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> costType;
    input >> dummy; input >> maxitcntval;
    input >> dummy; input >> maxtraintimeval;
    input >> dummy; input >> opttolval;
    input >> dummy; input >> CNval;
    input >> dummy; input >> epsval;
    input >> dummy; input >> autosetLevel;
    input >> dummy; input >> autosetCvalx;

    input >> dummy; input >> kerncache;
    input >> dummy; input >> sigmacache;
    input >> dummy; input >> kerndiag;
    input >> dummy; input >> diagoff;

    input >> dummy; input >> Nnc;
    input >> dummy; input >> isStateOpt;

    SVM_Generic::inputstream(input);

    input >> dummy; input >> traintarg;
    input >> dummy; input >> trainclass;
    input >> dummy; input >> Cweightval;
    input >> dummy; input >> Cweightvalfuzz;
    input >> dummy; input >> epsweightval;

    input >> dummy; input >> Gn;
    input >> dummy; input >> Gpn;
    input >> dummy; input >> gp;
    input >> dummy; input >> gn;
    input >> dummy; input >> hp;
    input >> dummy; input >> ub;

    input >> dummy; input >> Q;

    MEMDEL(GpGrad);
    GpGrad = NULL;

    MEMDEL(Gpsigma);
    Gpsigma = NULL;

    const static T dummydupe = dummyarg;

    GpGrad  = alloc_gp((void *) &kerncache,N(),N(),dummydupe);
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &(sigmacache),N(),N()));

    int oldmemsize = (kerncache).get_memsize();
    int oldrowdim  = (kerncache).get_min_rowdim();

    (kerncache).reset(N(),&evalKSVM_Vector_atonce_temp,this);
    (kerncache).setmemsize(oldmemsize,oldrowdim);

    (sigmacache).reset(N(),getsigmacallback(dummyarg),this);
    (sigmacache).setmemsize(oldmemsize,oldrowdim);

    return input;
}

template <class T>
int SVM_Vector_atonce_temp<T>::prealloc(int expectedN)
{
    kerndiag.prealloc(expectedN);
    diagoff.prealloc(expectedN);
    traintarg.prealloc(expectedN);
    trainclass.prealloc(expectedN);
    Cweightval.prealloc(expectedN);
    Cweightvalfuzz.prealloc(expectedN);
    epsweightval.prealloc(expectedN);
    gp.prealloc(expectedN);
    gn.prealloc(expectedN);
    hp.prealloc(expectedN);
    ub.prealloc(expectedN);
    kerncache.prealloc(expectedN);
    sigmacache.prealloc(expectedN);
    Gn.prealloc(Gn.numRows(),Gn.numCols());
    Gpn.prealloc(expectedN,Gpn.numCols());
    SVM_Generic::prealloc(expectedN);

    return 0;
}

template <class T>
int SVM_Vector_atonce_temp<T>::preallocsize(void) const
{
    return SVM_Generic::preallocsize();
}

template <class T>
int SVM_Vector_atonce_temp<T>::randomise(double sparsity)
{
    NiceAssert( sparsity >= 0 );
    NiceAssert( sparsity <= 1 );

    int res = 0;
    int Nnotz = (int) (((double) (N()-NNC(0)))*sparsity);

    if ( Nnotz && tspaceDim() )
    {
        res = 1;

        retVector<int> tmpva;

        Vector<int> canmod(cntintvec(N(),tmpva));

        int i,j,k;

        for ( i = N()-1 ; i >= 0 ; i-- )
        {
            if ( !d()(i) )
            {
                canmod.remove(i);
            }
        }

        // Observe sparsity

        while ( canmod.size() > Nnotz )
        {
            canmod.remove(svm_rand()%(canmod.size()));
        }

        // Need to randomise canmod alphas, set rest to zero
        // (need to take care as meaning of zero differs depending on goutType)

        Vector<Vector<double> > newalpha(N());

        // Set zero

        Vector<double> zerotemp(tspaceDim());
        zerotemp = 0.0;
        newalpha = zerotemp;

        // Next randomise

        double lbloc;
        double ubloc;

        for ( i = 0 ; i < canmod.size() ; i++ )
        {
            j = canmod(i);

            lbloc = isLinearCost() ? -ub(j) : -1.0;
            ubloc = isLinearCost() ?  ub(j) : +1.0;

            if ( tspaceDim() )
            {
                for ( k = 0 ; k < tspaceDim() ; k++ )
                {
                    double &amod = newalpha("&",j)("&",k);

                    setrand(amod);
                    amod = lbloc+((ubloc-lbloc)*amod);
                }
            }

            lbloc = abs2(newalpha(j));

            if ( lbloc > ubloc )
            {
                newalpha("&",j) *= ubloc/lbloc;
            }
        }

        // Lastly set alpha

        setAlphaV(newalpha);
    }

    return res;
}




















































#endif
