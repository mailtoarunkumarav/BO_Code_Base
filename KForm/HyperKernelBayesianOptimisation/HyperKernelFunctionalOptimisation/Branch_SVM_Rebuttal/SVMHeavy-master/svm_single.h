
//
// 1-class Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_single_h
#define _svm_single_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_binary.h"








class SVM_Single;


// Swap function

inline void qswap(SVM_Single &a, SVM_Single &b);


class SVM_Single : public SVM_Binary
{
public:

    // Constructors, destructors, assignment operators and similar

    SVM_Single();
    SVM_Single(const SVM_Single &src);
    SVM_Single(const SVM_Single &src, const ML_Base *xsrc);
    SVM_Single &operator=(const SVM_Single &src) { assign(src); return *this; }
    virtual ~SVM_Single();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);

    virtual int restart(void)   { SVM_Single temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int type(void)    const { return 2; }
    virtual int subtype(void) const { return 0; }

    virtual SVM_Generic &getSVM(void)                  { return static_cast<      SVM_Generic &>(*this); }
    virtual const SVM_Generic &getSVMconst(void) const { return static_cast<const SVM_Generic &>(*this); }

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'Z'; }
    virtual char targType(void) const { return 'N'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int isClassifier(void) const { return 0; }

    // Kernels

    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1);

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int addTrainingVector (int i, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);
    virtual int qaddTrainingVector(int i,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);

    virtual int addTrainingVector (int i, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);
    virtual int qaddTrainingVector(int i,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);

    // 0 = normal 1-class SVM
    // 1 = Tax and Duin style

    virtual int singmethod(void) const { return xsingmethod; }
    virtual void setsingmethod(int nv) { NiceAssert( ( nv == 0 ) || ( nv == 1 ) ); if ( xsingmethod != nv ) { xsingmethod = nv; fixz(); } return; }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // Other functions

    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    // Information functions (training data):

    virtual double LinBiasForce(void)   const { return dclass*SVM_Binary::LinBiasForce(); }
    virtual int anomclass(void) const { return dclass; }

    // Modification and autoset functions

    virtual int setLinBiasForce(double newval) { return SVM_Binary::setLinBiasForce(dclass*newval); }
    virtual void setanomalyclass(int n);

private:

    // Non-anomaly class (+1 by default)

    int dclass;

    // Method: 0 = normal
    //         1 = Tax and Duin

    int xsingmethod;

    void fixz(void);

    double calcz(const SparseVector<gentype> &xx, double zval) const 
    { 
        double res = zval;

        if ( xsingmethod ) 
        { 
            double K2val = 0;

            K2(K2val,-1,-1,NULL,&xx,&xx);

            res = dclass*K2val/2;
        }

        return res;
        //return xsingmethod ? dclass*K2(res,-1,-1,NULL,&xx,&xx)/2 : zval; 
    }

    double calcz(double zval, double K2val) const 
    { 
        return xsingmethod ? dclass*K2val/2 : zval; 
    }

    double calcz(int i) const 
    { 
        return xsingmethod ? dclass*kerndiag()(i)/2 : zR()(i); 
    }

    // Blocked functions

    int addTrainingVector( int i, int d, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0) { (void) i; (void) d; (void) x; (void) Cweigh; (void) epsweigh; (void) z; throw("Binary form of function addTrainingVector  not available for SVM_Single"); return 0; }
    int qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0) { (void) i; (void) d; (void) x; (void) Cweigh; (void) epsweigh; (void) z; throw("Binary form of function addTrainingVector  not available for SVM_Single"); return 0; }

public:

    // when adding a single vector it is sometimes handy to be able to 
    // pass the value of K2(x,x) in directly - eg if you've already calculated
    // it and it is computationally expensive.  To do this, set the following
    // pointer to point to it (but don't forget to set it back to NULL when
    // you're done).

    const double *diagkernvalcheat;
};

inline void qswap(SVM_Single &a, SVM_Single &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Single::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Single &b = dynamic_cast<SVM_Single &>(bb.getML());

    SVM_Binary::qswapinternal(b);

    qswap(dclass     ,b.dclass     );
    qswap(xsingmethod,b.xsingmethod);

    return;
}

inline void SVM_Single::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Single &b = dynamic_cast<const SVM_Single &>(bb.getMLconst());

    SVM_Binary::semicopy(b);

    dclass      = b.dclass;
    xsingmethod = b.xsingmethod;

    return;
}

inline void SVM_Single::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Single &src = dynamic_cast<const SVM_Single &>(bb.getMLconst());

    SVM_Binary::assign(src,onlySemiCopy);

    dclass      = src.dclass;
    xsingmethod = src.xsingmethod;

    return;
}

#endif
