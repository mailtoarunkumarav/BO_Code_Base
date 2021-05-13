//NB: QA cannot be referenced via common Gp, as Gp is in fact referenced via
//    kernel but rather rescaled through CCmCl (or something) and besides,
//    only exists when there is more than one class present

//
// Multiclass classification SVM (all at once)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_multic_atonce_h
#define _svm_multic_atonce_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_generic.h"
#include "svm_binary.h"
#include "svm_single.h"
#include "kcache.h"
#include "idstore.h"

void evalKSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalXYSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalsigmaSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner);

void evalSubKSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalSubXYSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalSubsigmaSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner);



class SVM_MultiC_atonce;
class SVM_MultiC;

// Swap function

inline void qswap(SVM_MultiC_atonce &a, SVM_MultiC_atonce &b);


class SVM_MultiC_atonce : public SVM_Generic
{
    friend void evalKSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalXYSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalsigmaSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend class SVM_MultiC;

public:

    // Constructors, destructors, assignment operators and similar

    SVM_MultiC_atonce();
    SVM_MultiC_atonce(const SVM_MultiC_atonce &src);
    SVM_MultiC_atonce(const SVM_MultiC_atonce &src, const ML_Base *xsrc);
    SVM_MultiC_atonce &operator=(const SVM_MultiC_atonce &src) { assign(src); return *this; }
    virtual ~SVM_MultiC_atonce();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void setmemsize(int memsize);

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { SVM_MultiC_atonce temp; *this = temp; return 1; }

    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha);
    virtual int setBiasV(const Vector<double> &newBias);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int isTrained(void) const { return isStateOpt; }

    virtual int N  (void)  const { return trainclass.size();                                                  }
    virtual int NS (void)  const { return Ns;                                                                 }
    virtual int NZ (void)  const { return N()-NS();                                                           }
    virtual int NF (void)  const { return (Q.NF() )/(numClasses()-1);                                         }
    virtual int NC (void)  const { return (Q.NC() )/(numClasses()-1);                                         }
    virtual int NLB(void)  const { return (Q.NLB())/(numClasses()-1);                                         }
    virtual int NLF(void)  const { return (Q.NLF())/(numClasses()-1);                                         }
    virtual int NUF(void)  const { return (Q.NUF())/(numClasses()-1);                                         }
    virtual int NUB(void)  const { return (Q.NUB())/(numClasses()-1);                                         }
    virtual int NNC(int d) const { return ( d == anomalyd ) ? QA.NNC(1) : Nnc(label_placeholder.findID(d)+1); }
    virtual int NS (int q) const { return ( q == -3 ) ? QA.NS()  : NS();                                      }
    virtual int NZ (int q) const { return ( q == -3 ) ? QA.NZ()  : NZ();                                      }
    virtual int NF (int q) const { return ( q == -3 ) ? QA.NF()  : NF();                                      }
    virtual int NC (int q) const { return ( q == -3 ) ? QA.NC()  : NC();                                      }
    virtual int NLB(int q) const { return ( q == -3 ) ? QA.NLB() : NLB();                                     }
    virtual int NLF(int q) const { return ( q == -3 ) ? QA.NLF() : NLF();                                     }
    virtual int NUF(int q) const { return ( q == -3 ) ? QA.NUF() : NUF();                                     }
    virtual int NUB(int q) const { return ( q == -3 ) ? QA.NUB() : NUB();                                     }

    virtual int tspaceDim(void)  const { return isrecdiv() ? numClasses() : numClasses()+1; }
    virtual int numClasses(void) const { return label_placeholder.size();                   }
    virtual int type(void)       const { return 3;                                          }
    virtual int subtype(void)    const { return 0;                                          }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'Z'; }
    virtual char targType(void) const { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual const Vector<int>          &ClassLabels(void)   const { return label_placeholder.getreftoID(); }
    virtual const Vector<Vector<int> > &ClassRep(void)      const { return classRepval;                    }
    virtual int                         findID(int ref)     const { return label_placeholder.findID(ref);  }

    virtual int isLinearCost(void)      const { return costType == 0; }
    virtual int isQuadraticCost(void)   const { return costType == 1; }
    virtual int is1NormCost(void)       const { return 0;             }
    virtual int isVarBias(void)         const { return 1;             }
    virtual int isPosBias(void)         const { return 0;             }
    virtual int isNegBias(void)         const { return 0;             }
    virtual int isFixedBias(void)       const { return 0;             }
    virtual int isVarBias(int d)        const { (void) d; return 1;   }
    virtual int isPosBias(int d)        const { (void) d; return 0;   }
    virtual int isNegBias(int d)        const { (void) d; return 0;   }
    virtual int isFixedBias(int d)      const { (void) d; return 0;   }

    virtual int isOptActive(void) const { return optType == 0; }
    virtual int isOptSMO(void)    const { return optType == 1; }
    virtual int isOptD2C(void)    const { return optType == 2; }
    virtual int isOptGrad(void)   const { return optType == 3; }

    virtual int m(void) const { return 2; }

    virtual double C(void)            const { return CNval;                                    }
    virtual double eps(void)          const { return epsval;                                   }
    virtual double Cclass(int d)      const { return mulCclass(label_placeholder.findID(d));   }
    virtual double epsclass(int d)    const { return mulepsclass(label_placeholder.findID(d)); }

    virtual int    memsize(void)      const { return 2*QA.memsize();       }
    virtual double zerotol(void)      const { return Q.zerotol();          }
    virtual double Opttol(void)       const { return Q.Opttol();           }
    virtual int    maxitcnt(void)     const { return Q.maxitcnt();         }
    virtual double maxtraintime(void) const { return Q.maxtraintime();     }
    virtual double outerlr(void)      const { return MULTINORM_OUTERSTEP;  }
    virtual double outertol(void)     const { return MULTINORM_OUTERACCUR; }

    virtual       int      maxiterfuzzt(void) const { return Q.maxiterfuzzt(); }
    virtual       int      usefuzzt(void)     const { return Q.usefuzzt();     }
    virtual       double   lrfuzzt(void)      const { return Q.lrfuzzt();      }
    virtual       double   ztfuzzt(void)      const { return Q.ztfuzzt();      }
    virtual const gentype &costfnfuzzt(void)  const { return Q.costfnfuzzt();  }

    virtual double LinBiasForce(void)        const { return LinBiasForce(0);                                                                            }
    virtual double QuadBiasForce(void)       const { return QuadBiasForce(0);                                                                           }
    virtual double LinBiasForce(int d)       const { if ( ( linbfq >= 0 ) && ( d == linbfd ) ) { return linbiasforceval(linbfq); } else { return 0.0; } }
    virtual double QuadBiasForce(int d)      const { (void) d; return 0.0;                                                                              }

    virtual int isFixedTube(void)  const { return 1; }
    virtual int isShrinkTube(void) const { return 0; }

    virtual int isRestrictEpsPos(void) const { return 0; }
    virtual int isRestrictEpsNeg(void) const { return 0; }

    virtual double nu(void)     const { return 0; }
    virtual double nuQuad(void) const { return 0; }

    virtual int isClassifyViaSVR(void) const { return 0; }
    virtual int isClassifyViaSVM(void) const { return 1; }

    virtual int is1vsA(void)    const { return 0;              }
    virtual int is1vs1(void)    const { return 0;              }
    virtual int isDAGSVM(void)  const { return 0;              }
    virtual int isMOC(void)     const { return 0;              }
    virtual int ismaxwins(void) const { return multitype == 4; }
    virtual int isrecdiv(void)  const { return multitype == 5; }

    virtual int isatonce(void) const { return 1; }
    virtual int isredbin(void) const { return 0; }

    virtual int isKreal(void)   const { return 0; }
    virtual int isKunreal(void) const { return 0; }

    virtual int isUnderlyingScalar(void) const { return 0; }
    virtual int isUnderlyingVector(void) const { return 1; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual int isanomalyOn(void)  const { return anomalyd != -3;                                          }
    virtual int isanomalyOff(void) const { return anomalyd == -3;                                          }

    virtual double anomalyNu(void)    const { return QA.autosetnuval();                                       }
    virtual int    anomalyClass(void) const { return linbfd ? linbfd : ( ( anomalyd == -3 ) ? 0 : anomalyd ); }

    virtual int isautosetOff(void)          const { return autosetLevel == 0; }
    virtual int isautosetCscaled(void)      const { return autosetLevel == 1; }
    virtual int isautosetCKmean(void)       const { return autosetLevel == 2; }
    virtual int isautosetCKmedian(void)     const { return autosetLevel == 3; }
    virtual int isautosetCNKmean(void)      const { return autosetLevel == 4; }
    virtual int isautosetCNKmedian(void)    const { return autosetLevel == 5; }
    virtual int isautosetLinBiasForce(void) const { return autosetLevel == 6; }

    virtual double autosetCval(void)  const { return autosetCvalx;  }
    virtual double autosetnuval(void) const { return autosetnuvalx; }

    virtual const Vector<int>                  &d          (void)      const { return trainclass;                                      }
    virtual const Vector<double>               &Cweight    (void)      const { return Cweightval;                                      }
    virtual const Vector<double>               &Cweightfuzz(void)      const { return onedvec;                                         }
    virtual const Vector<double>               &epsweight  (void)      const { return epsweightval;                                    }
    virtual const Matrix<double>               &Gp         (void)      const { return *Gpval;                                          }
    virtual const Vector<double>               &kerndiag   (void)      const { return QA.kerndiag(); } // kerndiagval;                                     }
    virtual const Vector<int>                  &alphaState (void)      const { return dalphaState;                                     }
    virtual const Vector<double>               &biasV      (int raw=0) const { return ( ismaxwins() || raw ) ? db : dbReduced;         }
    virtual const Vector<Vector<double> >      &alphaV     (int raw=0) const { return ( ismaxwins() || raw ) ? dalpha : dalphaReduced; }
    virtual const Vector<Vector<double> >      &getu       (void)      const { return u;                                               }

    virtual int isClassifier(void) const { return 1; }
    virtual int singmethod(void) const { return QA.singmethod(); }

    // Classify-with-reject (see svm_binary)

    virtual double rejectThreshold(void) const { return dthres; }
    virtual void setRejectThreshold(double nv) { NiceAssert( ( nv >= 0 ) || ( nv <= 0.5 ) ); dthres = nv; return; }

    // Modification:

    virtual int setLinearCost(void);
    virtual int setQuadraticCost(void);

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
    virtual int setmaxitcnt(int xmaxitcnt);
    virtual int setmaxtraintime(double xmaxtraintime);
    virtual int setouterlr(double xouterlr)   { (void) xouterlr;  return 0; }
    virtual int setoutertol(double xoutertol) { (void) xoutertol; return 0; }

    virtual int setmaxiterfuzzt(int xmaxiterfuzzt)               { return Q.setmaxiterfuzzt(xmaxiterfuzzt); }
    virtual int setusefuzzt(int xusefuzzt)                       { return Q.setusefuzzt(xusefuzzt);         }
    virtual int setlrfuzzt(double xlrfuzzt)                      { return Q.setlrfuzzt(xlrfuzzt);           }
    virtual int setztfuzzt(double xztfuzzt)                      { return Q.setztfuzzt(xztfuzzt);           }
    virtual int setcostfnfuzzt(const gentype &xcostfnfuzzt)      { return Q.setcostfnfuzzt(xcostfnfuzzt);   }
    virtual int setcostfnfuzzt(const std::string &xcostfnfuzzt)  { return Q.setcostfnfuzzt(xcostfnfuzzt);   }

    virtual int setLinBiasForce(double newval) { return setLinBiasForce(0,newval); }
    virtual int setLinBiasForce(int d,  double newval);

    virtual int setmaxwins(void);
    virtual int setrecdiv(void);

    virtual int anomalyOn(int danomalyClass, double danomalyNu);
    virtual int anomalyOff(void);

    virtual int randomise(double sparsity);

    virtual int autosetOff(void)                                     { return autosetLevel = 0;                  }
    virtual int autosetCscaled(double Cval)                          { return autosetCscaled(Cval,0);            }
    virtual int autosetCKmean(void)                                  { return autosetCKmean(0);                  }
    virtual int autosetCKmedian(void)                                { return autosetCKmedian(0);                }
    virtual int autosetCNKmean(void)                                 { return autosetCNKmean(0);                 }
    virtual int autosetCNKmedian(void)                               { return autosetCNKmedian(0);               }
    virtual int autosetLinBiasForce(double nuval, double Cval = 1.0) { return autosetLinBiasForce(nuval,Cval,0); }

    virtual int addclass(int label, int epszero = 0);
    virtual void setsingmethod(int nv) { QA.setsingmethod(nv); return; }

    // Kernel Modification

    virtual void prepareKernel(void) { return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1);

    virtual void fillCache(void) { Q.fillCache(); QA.fillCache(); return; }

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int addTrainingVector (int i, int d, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<int> &d,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    virtual int sety(int i, const gentype &y);
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

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual void setaltx(const ML_Base *_altxsrc) { ML_Base::setaltx(_altxsrc); /* NB this will ripple through to QA, but don't modify QA's pointer */ return; }

    // Don't use this!

    virtual int grablinbfq(void) const { return linbfq; }

private:

    virtual int gTrainingVector(Vector<double> &gproject, int &locclassrep, int i, int raw = 0, gentype ***pxyprodi = NULL) const;
    virtual int gTrainingVector(Vector<gentype> &gproject, int &locclassrep, int i, int raw = 0, gentype ***pxyprodi = NULL) const;

    int costType;            // 0 = linear, 1 = LS
    int multitype;           // 4 = max wins, 5 = recursive division
    int optType;             // 0 = active set, 1 = SMO, 2 D2C, 3 grad

    double CNval;               // C (tradeoff) value
    double epsval;              // eps (tube) value
    Vector<double> mulCclass;   // stored classlabel-wise C weights (0 = -1, 1 = zero, 2 = +1, 3 = free)
    Vector<double> mulepsclass; // stored classlabel-wise eps weights

    Kcache<double> xycache;           // kernel cache
    Kcache<double> kerncache;         // kernel cache
    Kcache<double> sigmacache;        // sigma cache
//    Vector<double> kerndiagval;       // kernel diagonals
    Vector<double> diagoff;           // diagonal offset for hessian (used by quadratic cost)

    Vector<double> linbiasforceval; // linear bias forcing vector
    int linbiasforceset;            // set if linbiasforce != 0
    int linbfd;                     // linear bias, epsilon = 0 class (d)
    int linbfq;                     // linear bias, epsilon = 0 class (s)

    int autosetLevel;    // 0 = none, 1 = C/N, 2 = Cmean, 3 = Cmedian, 4 = CNmean, 5 = CNmedian, 6 = LinBiasForce
    double autosetCvalx;  // Cval used if autosetLevel == 1,6
    double autosetnuvalx; // nuval used if autosetLevel == 6

    SVM_Binary Q;

    // Optimisation state
    //
    // label_placeholder: The user defines class in terms of labels.
    //                    They may be any int.  These are translated into
    //                    classlabels, namely 0,1,2,..., by label_placeholders.
    // classRep: Each class has a corresponding ternary representation (eg class 2 might be -1,+1,0,-1, where 0 is "don't care").  These are stored here.
    //           The number of classes is given by classRep.size()
    // Ns: number of support vectors

    Vector<Vector<double> > dalpha; // stored classrep-wise
    Vector<Vector<double> > dalphaReduced; // reduced version of above
    Vector<int> dalphaState;
    Vector<double> db;              // stored classrep-wise
    Vector<double> dbReduced;       // reduced version of above
    IDStore label_placeholder;
    Vector<Vector<int> > classRepval;  // stored classlabel-wise.
    int Ns;                         // number of support vectors
    Vector<int> Nnc;                // number of vector in each class
    Vector<Vector<double> > u;
    int isStateOpt;                 // set if SVM is in optimal state

    // Secondary anomaly detection (if used)

    int anomalyd;
    SVM_Single QA;

    // Training data
    //
    // NB: Gp and Gpn are both dependant on the class vector.  However if
    //     the class is 0 then we can base Gp and Gpn on any class we like
    //     as alpha for such a variable is always zero.  Thus when doing
    //     cross-validation we can set the class to zero without changing
    //     Gp or Gpn, which saves a lot of time and means semicopy doesn't
    //     need to clear the cache.

    Vector<int> trainclass;
    Vector<double> Cweightval;
    Vector<double> Cweightvalfuzz;
    Vector<double> epsweightval;
    Vector<double> onedvec;

    // Quadratic program definition

    Matrix<double> *xyval;
    Matrix<double> *Gpval;
    Matrix<double> *Gpsigma;
    Matrix<double> *GpSubval;
    Matrix<double> *xySubval;
    Matrix<double> *GpSubsigma;
    Matrix<double> Gpn;

    // Matrix of cut-matrices.  CMcl(s+1,t+1) is the cut matrix for
    // crossover between classes s+1 and t+1.  CMcl(0,t+1) is the cut
    // matrix for the Gpn elements.
    //
    // CCMcl is the compound cut matrix calculated as follows:
    //
    // CCMcl(0,0)     = CMcl(0,0) (max-wins) or CMcl(1,1) (recursive division)
    // CCMcl(s+1,0)   = CMcl(s+1,0) (max-wins) or CMcl(s+1,1) (recursive division)
    // CCMcl(0,t+1)   = CMcl(0,t+1) (max-wins) or CMcl(1,t+1) (recursive division)
    // CCMcl(s+1,t+1) = CCMcl(s+1,0)*CCMcl(0,t+1)

    Matrix<Matrix<double> > CMcl;
    Matrix<Matrix<double> > CCMcl;

    // Functions to calculate these

    void calcAlphaReduced(int i = -1);
    void calcBiasReduced(void);

    // label: class numbers given by the user, which are any non-consecutive set of integers
    // classlabel: labels translated to 0,1,2,...
    // classrep: underlying SVMs, which are classes expanded using classRep.

    // Internal functions
    //
    // locsetGp: call setGp for all binary SVMs Q
    // locnaivesetGpnExt: pass to naivesetGpnExt on svm_binary
    // classify: place gproject into appropriate class
    // recalcdiagoff(i): recalculate and update diagoff(i), or all of diagoff if i == -1
    // fixCMcl: set CMcl and CCMcl matrices
    // naivesetdzero: naively set d = 0 for vector x.
    // resetCandeps: completely recalculate C and epsilon (use after changing maxwins <-> recdiv)

    void locsetGp(int refactsol = 1);
    void locnaivesetGpnExt(void);
    int classify(int &locclassrep, const Vector<double> &gproject) const;
    void recalcdiagoff(int ival);
    void fixCMcl(int numclassestarg, int fixCCMclOnly = 0);
    int naivesetdzero(int i);
    void resetCandeps(void);

    // Other useful functions

    Vector<double> &upDim(  const Vector<double> &dTDimArg, Vector<double> &nDimArg ) const;
    Vector<double> &downDim(const Vector<double> &nDimArg , Vector<double> &dTDimArg) const;

    Vector<Vector<double> > &upDim(  const Vector<Vector<double> > &dTDimArg, Vector<Vector<double> > &nDimArg ) const;
    Vector<Vector<double> > &downDim(const Vector<Vector<double> > &nDimArg , Vector<Vector<double> > &dTDimArg) const;

    // Various bits needed by the active set optimisation hack for rec div but 
    // which should probably be removed.
    //
    // idivsplit etc: used when growing the add matrices. -1 means don't use numclasspre/post, otherwise break at this point

    int idivsplit;
    int numclasspre;
    int numclassat;
    int numclasspost;

    int fixautosettings(int kernchange, int Nchange, int ncut = 0); // ncut is subtracted from N
    double autosetkerndiagmean(void);
    double autosetkerndiagmedian(void);

    int autosetCscaled(double Cval, int ncut) { NiceAssert( Cval > 0 ); autosetCvalx = Cval; int res = setC( (N()-NNC(0)-ncut) ? (Cval/((N()-NNC(0)-ncut))) : 1.0);                                              autosetLevel = 1; return res; }
    int autosetCKmean(int ncut)               { double diagsum = ( (N()-NNC(0)-ncut) ? autosetkerndiagmean()                     : 1   ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 2; return res; }
    int autosetCKmedian(int ncut)             { double diagsum = ( (N()-NNC(0)-ncut) ? autosetkerndiagmedian()                   : 1   ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 3; return res; }
    int autosetCNKmean(int ncut)              { double diagsum = ( (N()-NNC(0)-ncut) ? (N()-NNC(0)-ncut)*autosetkerndiagmean()   : 1   ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 4; return res; }
    int autosetCNKmedian(int ncut)            { double diagsum = ( (N()-NNC(0)-ncut) ? (N()-NNC(0)-ncut)*autosetkerndiagmedian() : 1   ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 5; return res; }
    int autosetLinBiasForce(double nuval, double Cval, int ncut);

    int qtaddTrainingVector(int i, int d, double Cweigh, double epsweigh);

    // Path corrector bit.  If 0 then normal kernel, if 1 then path corrector

    int pathcorrect;

    // These were macros

    double C_CALC(void);
    double EPS_CALC(void);
    double CWEIGH_CALC(int _xclass_, double _Cweigh_, double _Cweighfuzz_, int _s_, double _dthres_);
    double EPSWEIGH_CALC(int _xclass_, double _epsweigh_, int _s_);

    // local training and Bartlett (see svm_binary)

    double dthres;

    virtual int loctrain(int &res, svmvolatile int &killSwitch, int realN, int assumeDNZ = 0);

    // A count of "fakes" to make Bartlett work in non-treacle time

    int bartfake; // number of "fakes" in bartlett
    int bartN; // set to N when bartlett in action
    Vector<int> fakeredir;
};

Vector<double> &updim(const Vector<Vector<double> > &u, const Vector<double> &dTDimArg, Vector<double> &nDimArg);
Vector<double> &downdim(const Vector<Vector<double> > &u, const Vector<double> &nDimArg, Vector<double> &dTDimArg);

Vector<Vector<double> > &updim(const Vector<Vector<double> > &u, const Vector<Vector<double> > &dTDimArg, Vector<Vector<double> > &nDimArg);
Vector<Vector<double> > &downdim(const Vector<Vector<double> > &u, const Vector<Vector<double> > &nDimArg, Vector<Vector<double> > &dTDimArg);

// max wins: for each Q, Cclass and epsclass are set to 1 and Cweight, epsweight are set instead.
// rec div:  for each Q, Cclass and epsclass are set to 1 and Cweight, epsweight are set instead.

inline void qswap(SVM_MultiC_atonce &a, SVM_MultiC_atonce &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_MultiC_atonce::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_MultiC_atonce &b = dynamic_cast<SVM_MultiC_atonce &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(Q,b.Q);
    qswap(QA,b.QA);

    qswap(costType         ,b.costType         );
    qswap(multitype        ,b.multitype        );
    qswap(optType          ,b.optType          );
    qswap(CNval            ,b.CNval            );
    qswap(epsval           ,b.epsval           );
    qswap(mulCclass        ,b.mulCclass        );
    qswap(mulepsclass      ,b.mulepsclass      );
    qswap(xycache          ,b.xycache          );
    qswap(kerncache        ,b.kerncache        );
    qswap(sigmacache       ,b.sigmacache       );
//    qswap(kerndiagval      ,b.kerndiagval      );
    qswap(diagoff          ,b.diagoff          );
    qswap(linbiasforceval  ,b.linbiasforceval  );
    qswap(linbfd           ,b.linbfd           );
    qswap(linbfq           ,b.linbfq           );
    qswap(linbiasforceset  ,b.linbiasforceset  );
    qswap(autosetLevel     ,b.autosetLevel     );
    qswap(autosetnuvalx    ,b.autosetnuvalx    );
    qswap(autosetCvalx     ,b.autosetCvalx     );
    qswap(dalpha           ,b.dalpha           );
    qswap(dalphaReduced    ,b.dalphaReduced    );
    qswap(dalphaState      ,b.dalphaState      );
    qswap(db               ,b.db               );
    qswap(dbReduced        ,b.dbReduced        );
    qswap(label_placeholder,b.label_placeholder);
    qswap(classRepval      ,b.classRepval      );
    qswap(Ns               ,b.Ns               );
    qswap(Nnc              ,b.Nnc              );
    qswap(u                ,b.u                );
    qswap(isStateOpt       ,b.isStateOpt       );
    qswap(trainclass       ,b.trainclass       );
    qswap(Cweightval       ,b.Cweightval       );
    qswap(Cweightvalfuzz   ,b.Cweightvalfuzz   );
    qswap(epsweightval     ,b.epsweightval     );
    qswap(onedvec          ,b.onedvec          );
    qswap(Gpn              ,b.Gpn              );
    qswap(CMcl             ,b.CMcl             );
    qswap(CCMcl            ,b.CCMcl            );
    qswap(idivsplit        ,b.idivsplit        );
    qswap(numclasspre      ,b.numclasspre      );
    qswap(numclassat       ,b.numclassat       );
    qswap(numclasspost     ,b.numclasspost     );
    qswap(pathcorrect      ,b.pathcorrect      );
    qswap(dthres           ,b.dthres           );

    Matrix<double> *b_xy;
    Matrix<double> *b_Gp;
    Matrix<double> *b_Gpsigma;
    Matrix<double> *b_xySub;
    Matrix<double> *b_GpSub;
    Matrix<double> *b_GpSubsigma;

    b_xy         = xyval;      xyval      = b.xyval;      b.xyval      = b_xy;
    b_Gp         = Gpval;      Gpval      = b.Gpval;      b.Gpval      = b_Gp;
    b_Gpsigma    = Gpsigma;    Gpsigma    = b.Gpsigma;    b.Gpsigma    = b_Gpsigma;
    b_xySub      = xySubval;   xySubval   = b.xySubval;   b.xySubval   = b_xySub;
    b_GpSub      = GpSubval;   GpSubval   = b.GpSubval;   b.GpSubval   = b_GpSub;
    b_GpSubsigma = GpSubsigma; GpSubsigma = b.GpSubsigma; b.GpSubsigma = b_GpSubsigma;

    // The kernel (and sigma) cache, as well as Gp (and Gpsigma) will have
    // been messed around by the above switching.  We need to make sure that
    // their pointers are set to rights before we continue.

    (xycache).cheatSetEvalArg((void *) this);
    (kerncache).cheatSetEvalArg((void *) this);
    (sigmacache).cheatSetEvalArg((void *) this);

    (xyval)->cheatsetcdref((void *) &(xycache));
    (Gpval)->cheatsetcdref((void *) &(kerncache));
    (Gpsigma)->cheatsetcdref((void *) &(sigmacache));

    (b.xycache).cheatSetEvalArg((void *) &b);
    (b.kerncache).cheatSetEvalArg((void *) &b);
    (b.sigmacache).cheatSetEvalArg((void *) &b);

    (b.xyval)->cheatsetcdref((void *) &(b.xycache));
    (b.Gpval)->cheatsetcdref((void *) &(b.kerncache));
    (b.Gpsigma)->cheatsetcdref((void *) &(b.sigmacache));

    (xySubval)->cheatsetcdref((void *) &(xycache));
    (GpSubval)->cheatsetcdref((void *) &(kerncache));
    (GpSubsigma)->cheatsetcdref((void *) &(sigmacache));

    (b.xySubval)->cheatsetcdref((void *) &(b.xycache));
    (b.GpSubval)->cheatsetcdref((void *) &(b.kerncache));
    (b.GpSubsigma)->cheatsetcdref((void *) &(b.sigmacache));

    // Also need to correct Gpn

    locnaivesetGpnExt();
    b.locnaivesetGpnExt();

    return;
}

inline void SVM_MultiC_atonce::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_MultiC_atonce &b = dynamic_cast<const SVM_MultiC_atonce &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

    //Gpval;
    //Gpsigma;
    //GpSubval;
    //GpSubsigma;

    //CMcl;
    //CCMcl;

    //label_placeholder
    //classRepval
    //u

    costType  = b.costType;
    multitype = b.multitype;
    optType   = b.optType;

    mulCclass   = b.mulCclass;
    mulepsclass = b.mulepsclass;

    autosetLevel  = b.autosetLevel;
    autosetnuvalx = b.autosetnuvalx;
    autosetCvalx  = b.autosetCvalx;

    anomalyd = b.anomalyd;

    Cweightval     = b.Cweightval;
    Cweightvalfuzz = b.Cweightvalfuzz;
    epsweightval   = b.epsweightval;
    onedvec        = b.onedvec;

    idivsplit    = b.idivsplit;
    numclasspre  = b.numclasspre;
    numclassat   = b.numclassat;
    numclasspost = b.numclasspost;

    pathcorrect = b.pathcorrect;

    isStateOpt = b.isStateOpt;

    CNval       = b.CNval;
    epsval      = b.epsval;

    linbiasforceval = b.linbiasforceval;
    linbfd          = b.linbfd;
    linbfq          = b.linbfq;
    linbiasforceset = b.linbiasforceset;

    dalpha        = b.dalpha;
    dalphaReduced = b.dalphaReduced;
    dalphaState   = b.dalphaState;
    db            = b.db;
    dbReduced     = b.dbReduced;
    Ns            = b.Ns;
    Nnc           = b.Nnc;

    trainclass = b.trainclass;

    dthres = b.dthres;

    if ( isQuadraticCost() )
    {
//        kerndiagval = b.kerndiagval;
        diagoff     = b.diagoff;

        kerncache.recalcDiag();
        sigmacache.recalcDiag();
    }

    Gpn = b.Gpn;

    // Setting d zero has absolutely no impact on trainclass member
    // variable, and hence no impact on the kernel or sigma caches.  So
    // no need to worry about them.

    //kerncache.clear();
    //sigmacache.clear();

    QA.semicopy(b.QA);
    Q.semicopy(b.Q);

    return;
}

inline void SVM_MultiC_atonce::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_MultiC_atonce &src = dynamic_cast<const SVM_MultiC_atonce &>(bb.getMLconst());

    SVM_Generic::assign(src,onlySemiCopy);

    anomalyd = src.anomalyd;

    CMcl  = src.CMcl;
    CCMcl = src.CCMcl;

    costType  = src.costType;
    multitype = src.multitype;
    optType   = src.optType;

    CNval       = src.CNval;
    epsval      = src.epsval;
    mulCclass   = src.mulCclass;
    mulepsclass = src.mulepsclass;

    autosetLevel  = src.autosetLevel;
    autosetnuvalx = src.autosetnuvalx;
    autosetCvalx  = src.autosetCvalx;

    linbiasforceval = src.linbiasforceval;
    linbfd          = src.linbfd;
    linbfq          = src.linbfq;
    linbiasforceset = src.linbiasforceset;

    xycache     = src.xycache;
    kerncache   = src.kerncache;
    sigmacache  = src.sigmacache;
//    kerndiagval = src.kerndiagval;
    diagoff     = src.diagoff;

    QA.assign(src.QA,onlySemiCopy);
    Q.assign(src.Q,onlySemiCopy);

    dalpha            = src.dalpha;
    dalphaReduced     = src.dalphaReduced;
    dalphaState       = src.dalphaState;
    db                = src.db;
    dbReduced         = src.dbReduced;
    label_placeholder = src.label_placeholder;
    classRepval       = src.classRepval;
    Ns                = src.Ns;
    Nnc               = src.Nnc;
    u                 = src.u;
    isStateOpt        = src.isStateOpt;

    trainclass     = src.trainclass;
    Cweightval     = src.Cweightval;
    Cweightvalfuzz = src.Cweightvalfuzz;
    epsweightval   = src.epsweightval;
    onedvec        = src.onedvec;

    idivsplit    = src.idivsplit;
    numclasspre  = src.numclasspre;
    numclassat   = src.numclassat;
    numclasspost = src.numclasspost;

    dthres = src.dthres;

    xycache.cheatSetEvalArg((void *) this);
    kerncache.cheatSetEvalArg((void *) this);
    sigmacache.cheatSetEvalArg((void *) this);

    if ( Gpval != NULL )
    {
        MEMDEL(xyval);
        MEMDEL(Gpval);
        MEMDEL(Gpsigma);

        xyval = NULL;
        Gpval = NULL;
        Gpsigma = NULL;
    }

    MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &xycache   ,(trainclass.size())*((numClasses()-1)),(trainclass.size())*((numClasses()-1))));
    MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,(trainclass.size())*((numClasses()-1)),(trainclass.size())*((numClasses()-1))));
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,(trainclass.size())*((numClasses()-1)),(trainclass.size())*((numClasses()-1))));
    Gpn     = src.Gpn;

    // NB: this needs to be done last as it re-evaluates all the
    // diagonals

    //kerncache.setEvalArg((void *) this);
    //sigmacache.setEvalArg((void *) this);

    locnaivesetGpnExt();
    locsetGp();

    return;
}

#endif
