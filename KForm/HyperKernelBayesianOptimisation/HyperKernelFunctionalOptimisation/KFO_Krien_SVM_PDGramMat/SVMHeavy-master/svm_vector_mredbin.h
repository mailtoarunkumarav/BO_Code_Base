
//
// Vector regression SVM (matrix reduction to binary)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_vector_mredbin_h
#define _svm_vector_mredbin_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_generic.h"
#include "svm_scalar.h"



class SVM_Vector_Mredbin;
class scalar_callback : public kernPrecursor
{
public:
    virtual int isKVarianceNZ(void) const;

    virtual void K0xfer(gentype &res, int &minmaxind, int typeis,
                       int xdim, int densetype, int resmode, int mlid) const;

    virtual void K1xfer(gentype &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K3xfer(gentype &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K4xfer(gentype &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void Kmxfer(gentype &res, int &minmaxind, int typeis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xinfo,
                        Vector<int> &ii,
                        int xdim, int m, int densetype, int resmode, int mlid) const;

    virtual void K0xfer(double &res, int &minmaxind, int typeis,
                       int xdim, int densetype, int resmode, int mlid) const;

    virtual void K1xfer(double &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K3xfer(double &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K4xfer(double &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void Kmxfer(double &res, int &minmaxind, int typeis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xinfo,
                        Vector<int> &ii,
                        int xdim, int m, int densetype, int resmode, int mlid) const;

    SVM_Vector_Mredbin *realOwner;
};



// Swap function

inline void qswap(SVM_Vector_Mredbin &a, SVM_Vector_Mredbin &b);



class SVM_Vector_Mredbin : public SVM_Generic
{
    friend class scalar_callback;

public:

    // Constructors, destructors, assignment operators and similar

    SVM_Vector_Mredbin();
    SVM_Vector_Mredbin(const SVM_Vector_Mredbin &src);
    SVM_Vector_Mredbin(const SVM_Vector_Mredbin &src, const ML_Base *xsrc);
    SVM_Vector_Mredbin &operator=(const SVM_Vector_Mredbin &src) { assign(src); return *this; }
    virtual ~SVM_Vector_Mredbin();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void setmemsize(int memsize) { Q.setmemsize(memsize); return; }

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { SVM_Vector_Mredbin temp; *this = temp; return 1; }

    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha);
    virtual int setBiasV(const Vector<double>  &newBias);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int isTrained(void) const { return Q.isTrained(); }

    virtual int N  (void)  const { return aN;                 }
    virtual int NS (void)  const { return aNS;                }
    virtual int NZ (void)  const { return aNZ;                }
    virtual int NF (void)  const { return aNF;                }
    virtual int NC (void)  const { return aNC;                }
    virtual int NLB(void)  const { return 0;                  }
    virtual int NLF(void)  const { return 0;                  }
    virtual int NUF(void)  const { return aNF;                }
    virtual int NUB(void)  const { return aNC-aNZ;            }
    virtual int NNC(int d) const { return (Q.NNC(d))/order(); }
    virtual int NS (int q) const { (void) q; return NS();     }
    virtual int NZ (int q) const { (void) q; return NZ();     }
    virtual int NF (int q) const { (void) q; return NF();     }
    virtual int NC (int q) const { (void) q; return NC();     }
    virtual int NLB(int q) const { (void) q; return NLB();    }
    virtual int NLF(int q) const { (void) q; return NLF();    }
    virtual int NUF(int q) const { (void) q; return NUF();    }
    virtual int NUB(int q) const { (void) q; return NUB();    }

    virtual int tspaceDim(void)  const { return dbiasA.size();  }
    virtual int numClasses(void) const { return 0;              }
    virtual int type(void)       const { return 4;              }
    virtual int subtype(void)    const { return 3;              }

    virtual int numInternalClasses(void) const { return 1; }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'V'; }
    virtual char targType(void) const { return 'V'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual const Vector<int>          &ClassLabels(void)   const { return Q.ClassLabels(); }
    virtual const Vector<Vector<int> > &ClassRep(void)      const { return Q.ClassRep();    }
    virtual int                         findID(int ref)     const { return Q.findID(ref);   }

    virtual int isLinearCost(void)      const { return Q.isLinearCost();    }
    virtual int isQuadraticCost(void)   const { return Q.isQuadraticCost(); }
    virtual int is1NormCost(void)       const { return 0;                   }
    virtual int isVarBias(void)         const { return Q.isVarBias();       }
    virtual int isPosBias(void)         const { return Q.isPosBias();       }
    virtual int isNegBias(void)         const { return Q.isNegBias();       }
    virtual int isFixedBias(void)       const { return Q.isFixedBias();     }
    virtual int isVarBias(int q)        const { return Q.isVarBias(q);      }
    virtual int isPosBias(int q)        const { return Q.isPosBias(q);      }
    virtual int isNegBias(int q)        const { return Q.isNegBias(q);      }
    virtual int isFixedBias(int q)      const { return Q.isFixedBias(q);    }

    virtual int isOptActive(void) const { return Q.isOptActive(); }
    virtual int isOptSMO(void)    const { return Q.isOptSMO();    }
    virtual int isOptD2C(void)    const { return Q.isOptD2C();    }
    virtual int isOptGrad(void)   const { return Q.isOptGrad();   }

    virtual int m(void) const { return Q.m(); }

    virtual double C(void)            const { return Q.C();            }
    virtual double eps(void)          const { return Q.eps();          }
    virtual double Cclass(int d)      const { return Q.Cclass(d);      }
    virtual double epsclass(int d)    const { return Q.epsclass(d);    }

    virtual int    memsize(void)      const { return Q.memsize();      }
    virtual double zerotol(void)      const { return Q.zerotol();      }
    virtual double Opttol(void)       const { return Q.Opttol();       }
    virtual int    maxitcnt(void)     const { return Q.maxitcnt();     }
    virtual double maxtraintime(void) const { return Q.maxtraintime(); }
    virtual double outerlr(void)      const { return Q.outerlr();      }
    virtual double outertol(void)     const { return Q.outertol();     }

    virtual       int      maxiterfuzzt(void) const { return Q.maxiterfuzzt(); }
    virtual       int      usefuzzt(void)     const { return Q.usefuzzt();     }
    virtual       double   lrfuzzt(void)      const { return Q.lrfuzzt();      }
    virtual       double   ztfuzzt(void)      const { return Q.ztfuzzt();      }
    virtual const gentype &costfnfuzzt(void)  const { return Q.costfnfuzzt();  }

    virtual double LinBiasForce(void)        const { return Q.LinBiasForce();   }
    virtual double QuadBiasForce(void)       const { return Q.QuadBiasForce();  }
    virtual double LinBiasForce(int q)       const { return Q.LinBiasForce(q);  }
    virtual double QuadBiasForce(int q)      const { return Q.QuadBiasForce(q); }

    virtual int isFixedTube(void)  const { return Q.isFixedTube();  }
    virtual int isShrinkTube(void) const { return Q.isShrinkTube(); }

    virtual int isRestrictEpsPos(void) const { return Q.isRestrictEpsPos(); }
    virtual int isRestrictEpsNeg(void) const { return Q.isRestrictEpsNeg(); }

    virtual double nu(void)     const { return Q.nu();     }
    virtual double nuQuad(void) const { return Q.nuQuad(); }

    virtual int isClassifyViaSVR(void) const { return Q.isClassifyViaSVR(); }
    virtual int isClassifyViaSVM(void) const { return Q.isClassifyViaSVM(); }

    virtual int is1vsA(void)    const { return 0; }
    virtual int is1vs1(void)    const { return 0; }
    virtual int isDAGSVM(void)  const { return 0; }
    virtual int isMOC(void)     const { return 0; }
    virtual int ismaxwins(void) const { return 0; }
    virtual int isrecdiv(void)  const { return 0; }

    virtual int isatonce(void) const { return 0; }
    virtual int isredbin(void) const { return 1; }

    virtual int isKreal(void)   const { return 0; }
    virtual int isKunreal(void) const { return 1; }

    virtual int isClassifier(void) const { return 0; }

    virtual int isUnderlyingScalar(void) const { return 0; }
    virtual int isUnderlyingVector(void) const { return 1; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual int isanomalyOn(void)  const { return 0; }
    virtual int isanomalyOff(void) const { return 1; }

    virtual double anomalyNu(void)    const { return 0; }
    virtual int    anomalyClass(void) const { return 0; }

    virtual int isautosetOff(void)          const { return Q.isautosetOff();       }
    virtual int isautosetCscaled(void)      const { return Q.isautosetCscaled();   }
    virtual int isautosetCKmean(void)       const { return Q.isautosetCKmean();    }
    virtual int isautosetCKmedian(void)     const { return Q.isautosetCKmedian();  }
    virtual int isautosetCNKmean(void)      const { return Q.isautosetCNKmean();   }
    virtual int isautosetCNKmedian(void)    const { return Q.isautosetCNKmedian(); }
    virtual int isautosetLinBiasForce(void) const { return 0;                      }

    virtual double autosetCval(void)  const { return Q.autosetCval()/tspaceDim(); }
    virtual double autosetnuval(void) const { return 0;                           }

    virtual const Vector<int>                  &d          (void)      const { return (Q.d())(interlace,const_cast<retVector<int> &>(retva));            }
    virtual const Vector<double>               &Cweight    (void)      const { return (Q.Cweight())(interlace,const_cast<retVector<double> &>(retvb));   }
    virtual const Vector<double>               &Cweightfuzz(void)      const { return onedvec;                                                           }
    virtual const Vector<double>               &epsweight  (void)      const { return (Q.epsweight())(interlace,const_cast<retVector<double> &>(retvc)); }
    virtual const Matrix<double>               &Gp         (void)      const { return Q.Gp();                                                            }
    virtual const Vector<double>               &kerndiag   (void)      const { return Q.kerndiag();                                                      }
    virtual const Vector<int>                  &alphaState (void)      const { return xalphaState;                                                       }
    virtual const Vector<Vector<double> >      &zV         (int raw=0) const { (void) raw; return traintarg;                                             }
    virtual const Vector<double>               &biasV      (int raw=0) const { (void) raw; return dbiasA;                                                }
    virtual const Vector<Vector<double> >      &alphaV     (int raw=0) const { (void) raw; return dalphaA;                                               }
    virtual const Vector<Vector<double> >      &getu       (void)      const { return Q.getu();                                                          }

    // Modification:

    virtual int setLinearCost(void)    { return Q.setLinearCost();    }
    virtual int setQuadraticCost(void) { return Q.setQuadraticCost(); }

    virtual int setC(double xC)     { return Q.setC(xC);     }
    virtual int seteps(double xeps) { return Q.seteps(xeps); }

    virtual int setOptActive(void) { return Q.setOptActive(); }
    virtual int setOptSMO(void)    { return Q.setOptSMO();    }
    virtual int setOptD2C(void)    { return Q.setOptD2C();    }
    virtual int setOptGrad(void)   { return Q.setOptGrad();   }

    virtual int setzerotol(double zt)                 { return Q.setzerotol(zt);                 }
    virtual int setOpttol(double xopttol)             { return Q.setOpttol(xopttol);             }
    virtual int setmaxitcnt(int xmaxitcnt)            { return Q.setmaxitcnt(xmaxitcnt);         }
    virtual int setmaxtraintime(double xmaxtraintime) { return Q.setmaxtraintime(xmaxtraintime); }
    virtual int setouterlr(double xouterlr)           { return Q.setouterlr(xouterlr);           }
    virtual int setoutertol(double xoutertol)         { return Q.setoutertol(xoutertol);         }

    virtual int randomise(double sparsity);

    virtual int sety(int i, const Vector<double> &z);
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z);
    virtual int sety(const Vector<Vector<double> > &z);

    virtual int autosetOff(void)            { return Q.autosetOff();                     }
    virtual int autosetCscaled(double Cval) { return Q.autosetCscaled(Cval*tspaceDim()); }
    virtual int autosetCKmean(void)         { return Q.autosetCKmean();                  }
    virtual int autosetCKmedian(void)       { return Q.autosetCKmedian();                }
    virtual int autosetCNKmean(void)        { return Q.autosetCNKmean();                 }
    virtual int autosetCNKmedian(void)      { return Q.autosetCNKmedian();               }

    virtual int settspaceDim(int newdim);
    virtual int addtspaceFeat(int i);
    virtual int removetspaceFeat(int i);

    // Kernel Modification

    virtual void prepareKernel(void) { return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1);

    virtual void fillCache(void) { Q.fillCache(); return; }

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int addTrainingVector (int i, const Vector<double>  &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const Vector<double>  &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<Vector<double> >  &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i, const Vector<Vector<double> >  &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

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

    virtual int scaleCweight(double scalefactor)     { return Q.scaleCweight(scalefactor);     }
    virtual int scaleCweightfuzz(double scalefactor) { return Q.scaleCweightfuzz(scalefactor); }
    virtual int scaleepsweight(double scalefactor)   { return Q.scaleepsweight(scalefactor);   }

    // Train the SVM

    virtual void fudgeOn(void)  { Q.fudgeOn();  return; }
    virtual void fudgeOff(void) { Q.fudgeOff(); return; }

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    // Training set control:

    virtual int setFixedBias(const Vector<double> &newbias);
    virtual int setFixedBias(int q, double newbias)  { return SVM_Generic::setFixedBias(q,newbias); }
    virtual int setFixedBias(const gentype &newbias) { return SVM_Generic::setFixedBias(  newbias); }
    virtual int setFixedBias(double newbias)         { return SVM_Generic::setFixedBias(  newbias); }

private:

    virtual int gTrainingVector(Vector<double> &gproject, int &dummy, int i, int raw = 0, gentype ***pxyprodi = NULL) const;

    int aN;
    int aNS;
    int aNZ;
    int aNF;
    int aNC;

    Vector<int> interlace; // [ 0 m 2m ... ] (m=tspaceDim)

    Vector<Vector<double> > traintarg;
    Vector<int> xalphaState;
    Vector<double> onedvec;
    Vector<Vector<double> > dalphaA;
    Vector<double> dbiasA;
    Matrix<double> Gpn;

    SVM_Scalar Q;

    scalar_callback Kcall;

    int ixsplit;
    int iqsplit;
    int ixskip;
    int ixskipc;

    retVector<int> retva;
    retVector<double> retvb;
    retVector<double> retvc;

    void updateBias(void);
    void updateAlpha(void);
    void fixKcallback(void);
    int qtaddTrainingVector(int i, const Vector<double>  &z, double Cweigh = 1, double epsweigh = 1, int d = 2);
    void locnaivesetGpnExt(void);
    int inintrain(int &res, svmvolatile int &killSwitch);
    int intrain(int &res, svmvolatile int &killSwitch);
};

inline void qswap(SVM_Vector_Mredbin &a, SVM_Vector_Mredbin &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Vector_Mredbin::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Vector_Mredbin &b = dynamic_cast<SVM_Vector_Mredbin &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(aN         ,b.aN         );
    qswap(aNS        ,b.aNS        );
    qswap(aNZ        ,b.aNZ        );
    qswap(aNF        ,b.aNF        );
    qswap(aNC        ,b.aNC        );
    qswap(interlace  ,b.interlace  );
    qswap(traintarg  ,b.traintarg  );
    qswap(xalphaState,b.xalphaState);
    qswap(onedvec    ,b.onedvec    );
    qswap(dalphaA    ,b.dalphaA    );
    qswap(dbiasA     ,b.dbiasA     );
    qswap(Gpn        ,b.Gpn        );
    qswap(Q          ,b.Q          );
    qswap(ixsplit    ,b.ixsplit    );
    qswap(iqsplit    ,b.iqsplit    );
    qswap(ixskip     ,b.ixskip     );
    qswap(ixskipc    ,b.ixskipc    );

    locnaivesetGpnExt();
    b.locnaivesetGpnExt();

    return;
}

inline void SVM_Vector_Mredbin::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Vector_Mredbin &b = dynamic_cast<const SVM_Vector_Mredbin &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

    //interlace
    //Gpn

    traintarg = b.traintarg;

    ixsplit = b.ixsplit;
    iqsplit = b.iqsplit;
    ixskip  = b.ixskip;
    ixskipc = b.ixskipc;

    aN  = b.aN;
    aNS = b.aNS;
    aNZ = b.aNZ;
    aNF = b.aNF;
    aNC = b.aNC;

    xalphaState = b.xalphaState;
    onedvec     = b.onedvec;
    dalphaA     = b.dalphaA;
    dbiasA      = b.dbiasA;

    Q.semicopy(b.Q);

    return;
}

inline void SVM_Vector_Mredbin::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Vector_Mredbin &src = dynamic_cast<const SVM_Vector_Mredbin &>(bb.getMLconst());

    aN  = src.aN;
    aNS = src.aNS;
    aNZ = src.aNZ;
    aNF = src.aNF;
    aNC = src.aNC;

    interlace = src.interlace;

    SVM_Generic::assign(src,onlySemiCopy);

    traintarg   = src.traintarg;
    xalphaState = src.xalphaState;
    onedvec     = src.onedvec;
    dalphaA     = src.dalphaA;
    dbiasA      = src.dbiasA;
    Gpn         = src.Gpn;

    Q.assign(src.Q,onlySemiCopy);

    locnaivesetGpnExt();

    ixsplit = src.ixsplit;
    iqsplit = src.iqsplit;
    ixskip  = src.ixskip;
    ixskipc = src.ixskipc;

    fixKcallback();

    return;
}

#endif
