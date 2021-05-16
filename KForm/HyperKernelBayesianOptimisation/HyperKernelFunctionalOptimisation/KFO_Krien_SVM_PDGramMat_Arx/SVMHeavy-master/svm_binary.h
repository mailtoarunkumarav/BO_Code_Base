
//
// Binary Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_binary_h
#define _svm_binary_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar.h"








class SVM_Binary;


// Swap function

inline void qswap(SVM_Binary &a, SVM_Binary &b);


class SVM_Binary : public SVM_Scalar
{
public:

    SVM_Binary();
    SVM_Binary(const SVM_Binary &src);
    SVM_Binary(const SVM_Binary &src, const ML_Base *xsrc);
    SVM_Binary &operator=(const SVM_Binary &src) { assign(src); return *this; }
    virtual ~SVM_Binary();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;

    virtual int restart(void) { SVM_Binary temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int NNC(int d) const { return binNnc(d+1); }

    virtual int tspaceDim(void)  const { return 1; }
    virtual int numClasses(void) const { return 2; }
    virtual int type(void)       const { return 1; }
    virtual int subtype(void)    const { return 0; }

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'Z'; }
    virtual char targType(void) const { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int numInternalClasses(void) const { return SVM_Generic::numInternalClasses(); }

    virtual double eps(void)          const { return bineps;           }
    virtual double Cclass(int d)      const { return binCclass(d+1);   }
    virtual double epsclass(int d)    const { return binepsclass(d+1); }

    virtual double LinBiasForce(void)  const { return SVM_Scalar::LinBiasForce(); }

    virtual double nuQuad(void) const { return 0; }

    virtual int isClassifyViaSVR(void) const { return isSVMviaSVR;  }
    virtual int isClassifyViaSVM(void) const { return !isSVMviaSVR; }

    virtual int is1vsA(void)    const { return 0; }
    virtual int is1vs1(void)    const { return 1; }
    virtual int isDAGSVM(void)  const { return 1; }
    virtual int isMOC(void)     const { return 1; }
    virtual int ismaxwins(void) const { return 0; }
    virtual int isrecdiv(void)  const { return 1; }

    virtual int isautosetOff(void)          const { return binautosetLevel == 0; }
    virtual int isautosetCscaled(void)      const { return binautosetLevel == 1; }
    virtual int isautosetCKmean(void)       const { return binautosetLevel == 2; }
    virtual int isautosetCKmedian(void)     const { return binautosetLevel == 3; }
    virtual int isautosetCNKmean(void)      const { return binautosetLevel == 4; }
    virtual int isautosetCNKmedian(void)    const { return binautosetLevel == 5; }
    virtual int isautosetLinBiasForce(void) const { return binautosetLevel == 6; }

    virtual double autosetCval(void)  const { return binautosetCval;  }
    virtual double autosetnuval(void) const { return binautosetnuval; }

    virtual const Vector<int>    &d(void)           const { return bintrainclass;  }
    virtual const Vector<double> &Cweight(void)     const { return binCweight;     }
    virtual const Vector<double> &Cweightfuzz(void) const { return binCweightfuzz; }
    virtual const Vector<double> &epsweight(void)   const { return binepsweight;   }
    virtual const Vector<double> &zR(void)          const { return bintraintarg;   }

    virtual int isClassifier(void) const { return 1; }

    // Modification:

    virtual int setC(double xC);
    virtual int seteps(double xeps);
    virtual int setCclass(int d, double xC);
    virtual int setepsclass(int d, double xeps);

    virtual int sety(int i, double z);
    virtual int sety(const Vector<int> &i, const Vector<double> &d);
    virtual int sety(const Vector<double> &z);

    virtual int sety(int                i, const Vector<double>          &y) { return SVM_Scalar::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &y) { return SVM_Scalar::sety(i,y); }
    virtual int sety(                      const Vector<Vector<double> > &y) { return SVM_Scalar::sety(  y); }

    virtual int sety(int                i, const d_anion         &y) { return SVM_Scalar::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &y) { return SVM_Scalar::sety(i,y); }
    virtual int sety(                      const Vector<d_anion> &y) { return SVM_Scalar::sety(  y); }

    virtual int setLinBiasForce(double newval);

    virtual int setFixedTube(void);
    virtual int setShrinkTube(void);

    virtual int setClassifyViaSVR(void);
    virtual int setClassifyViaSVM(void);

    virtual int autosetOff(void);
    virtual int autosetCscaled(double Cval);
    virtual int autosetCKmean(void);
    virtual int autosetCKmedian(void);
    virtual int autosetCNKmean(void);
    virtual int autosetCNKmedian(void);
    virtual int autosetLinBiasForce(double nuval, double Cval = 1.0);

    // Classify-with-reject:
    //
    // Set threshold zero for normal operation, otherwise this is "d" in Bartlett

    virtual double rejectThreshold(void) const { return dthres; }
    virtual void setRejectThreshold(double nv) { NiceAssert( ( nv >= 0 ) || ( nv <= 0.5 ) ); dthres = nv; return; }

    // Kernel Modification

    virtual void prepareKernel(void);
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1);

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int addTrainingVector (int i, int d, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);
    virtual int qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);

    virtual int addTrainingVector (int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);
    virtual int qaddTrainingVector(int i, const Vector<int> &d,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);

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

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);

    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

private:

    // For speed reasons we want to put tube information in gp (via z) rather
    // than hp (via eps in the scalar regression level).  For this reason we
    // need to keep a local version of the following variables and translate
    // them down for the regression level.

    double bineps;
    Vector<int>    bintrainclass;
    Vector<double> bintraintarg;
    Vector<double> binepsclass;
    Vector<double> binepsweight;
    Vector<double> binCclass;
    Vector<double> binCweight;     // rely on ML_Base callback to do sigma stuff
    Vector<double> binCweightfuzz; // rely on ML_Base callback to do sigma stuff

    int binautosetLevel;    // 0 = none, 1 = C/N, 2 = Cmean, 3 = Cmedian, 4 = CNmean, 5 = CNmedian, 6 = LinBiasForce
    double binautosetCval;  // Cval used if autosetLevel == 1,6
    double binautosetnuval; // nuval used if autosetLevel == 6

    // This flag forces the SVM to act as a regressor with targets +-1 and
    // no insensitive zero (zero tube width).  Combining this with quadratic
    // cost implements the binary LS-SVM described by Suykens

    int isSVMviaSVR;

    // Information on class counts (must be kept locally, as the scalar
    // SVM version will be inaccurate if running SVM as SVR)

    Vector<int> binNnc; // number of vectors in each class (-1,0,+1)

    // Bartlett's "classification with reject option" "d" threshold
    // 
    // d = 0: not using reject, so do nothing special
    // d > 0: during training we need to duplicate a bunch of alphas 
    //        (cheat on x) that are y_i g(x_i) >= 0 (no boundary eps), 
    //        train, then recombine alpha

    double dthres;

    // Internal functions

    int fixautosettings(int kernchange, int Nchange);
    int setdinternal(int i, int d);
    double autosetkerndiagmean(void);
    double autosetkerndiagmedian(void);
    void setalleps(double xeps, const Vector<double> &xepsclass);

    // These functions are invalid for SVM_Binary (though they were valid
    // for the base SVM_Scalar).

    int setRestrictEpsPos(void) { return SVM_Generic::setRestrictEpsPos(); }
    int setRestrictEpsNeg(void) { return SVM_Generic::setRestrictEpsNeg(); }

    int addTrainingVector( int i, double z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) { (void) i; (void) z; (void) x; (void) Cweigh; (void) epsweigh; (void) d; throw("Scalar form of function addTrainingVector  not available for SVM_Binary"); return 0; }
    int qaddTrainingVector(int i, double z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) { (void) i; (void) z; (void) x; (void) Cweigh; (void) epsweigh; (void) d; throw("Scalar form of function qaddTrainingVector not available for SVM_Binary"); return 0; }

public:
    // Train, with reject, but assuming extra bits already added

    virtual int loctrain(int &res, svmvolatile int &killSwitch, int realN, int assumeDNZ = 0);
};

inline SVM_Binary &setident (SVM_Binary &a) { throw("something"); return a; }
inline SVM_Binary &setzero  (SVM_Binary &a) { a.restart(); return a; }
inline SVM_Binary &setposate(SVM_Binary &a) { return a; }
inline SVM_Binary &setnegate(SVM_Binary &a) { throw("something"); return a; }
inline SVM_Binary &setconj  (SVM_Binary &a) { throw("something"); return a; }
inline SVM_Binary &setrand  (SVM_Binary &a) { throw("something"); return a; }
inline SVM_Binary &postProInnerProd(SVM_Binary &a) { return a; }


inline void qswap(SVM_Binary &a, SVM_Binary &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Binary::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Binary &b = dynamic_cast<SVM_Binary &>(bb.getML());

    SVM_Scalar::qswapinternal(b);

    qswap(bineps         ,b.bineps         );
    qswap(binautosetLevel,b.binautosetLevel);
    qswap(binautosetnuval,b.binautosetnuval);
    qswap(binautosetCval ,b.binautosetCval );
    qswap(bintrainclass  ,b.bintrainclass  );
    qswap(bintraintarg   ,b.bintraintarg   );
    qswap(binepsclass    ,b.binepsclass    );
    qswap(binepsweight   ,b.binepsweight   );
    qswap(binCclass      ,b.binCclass      );
    qswap(binCweight     ,b.binCweight     );
    qswap(binCweightfuzz ,b.binCweightfuzz );
    qswap(isSVMviaSVR    ,b.isSVMviaSVR    );
    qswap(binNnc         ,b.binNnc         );
    qswap(dthres         ,b.dthres         );

    return;
}

inline void SVM_Binary::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Binary &b = dynamic_cast<const SVM_Binary &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    bineps         = b.bineps;
    binepsclass    = b.binepsclass;
    binepsweight   = b.binepsweight;
    binCclass      = b.binCclass;
    binCweight     = b.binCweight;
    binCweightfuzz = b.binCweightfuzz;

    binautosetLevel = b.binautosetLevel;
    binautosetCval  = b.binautosetCval;
    binautosetnuval = b.binautosetnuval;

    isSVMviaSVR = b.isSVMviaSVR;

    binNnc        = b.binNnc;
    bintraintarg  = b.bintraintarg;
    bintrainclass = b.bintrainclass;

    dthres = b.dthres;

    return;
}

inline void SVM_Binary::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Binary &src = dynamic_cast<const SVM_Binary &>(bb.getMLconst());

    SVM_Scalar::assign(static_cast<const SVM_Scalar &>(src),onlySemiCopy);

    bineps         = src.bineps;
    bintrainclass  = src.bintrainclass;
    bintraintarg   = src.bintraintarg;
    binepsclass    = src.binepsclass;
    binepsweight   = src.binepsweight;
    binCclass      = src.binCclass;
    binCweight     = src.binCweight;
    binCweightfuzz = src.binCweightfuzz;
    isSVMviaSVR    = src.isSVMviaSVR;
    binNnc         = src.binNnc;

    binautosetLevel = src.binautosetLevel;
    binautosetnuval = src.binautosetnuval;
    binautosetCval  = src.binautosetCval;

    dthres = src.dthres;

    return;
}

#endif
