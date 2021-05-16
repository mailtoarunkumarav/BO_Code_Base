
//
// 1-class Classification SSV
//
// Version: 7
// Date: 06/12/2017
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _ssv_single_h
#define _ssv_single_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ssv_binary.h"







// Really just a modified version of SVM_Single set to inherit from SSV_Binary rather than SVM_Binary.
// Don't know about the theoretical properties but some potential maybe?  Sparse anomaly detection?
// biasForce is multiplied by Nnz during training and represents the average error

class SSV_Single;


// Swap function

inline void qswap(SSV_Single &a, SSV_Single &b);


class SSV_Single : public SSV_Binary
{
public:

    // Constructors, destructors, assignment operators and similar

    SSV_Single();
    SSV_Single(const SSV_Single &src);
    SSV_Single(const SSV_Single &src, const ML_Base *xsrc);
    SSV_Single &operator=(const SSV_Single &src) { assign(src); return *this; }
    virtual ~SSV_Single();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);

    virtual int restart(void) { SSV_Single temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int type(void)    const { return 702; }
    virtual int subtype(void) const { return 0;   }

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'Z'; }
    virtual char targType(void) const { return 'N'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int isClassifier(void) const { return 0; }

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    // Other functions

    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    // Information functions (training data):

    virtual double biasForce(void) const { return dclass*SSV_Binary::biasForce(); }
    virtual int anomalclass(void)  const { return dclass; }

    // Modification and autoset functions

    virtual int setBiasForce(double newval) { return SSV_Binary::setBiasForce(dclass*newval); }
    virtual int setanomalclass(int n);

    // Training functions:

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

private:

    // Non-anomaly class (+1 by default)

    int dclass;
};

inline void qswap(SSV_Single &a, SSV_Single &b)
{
    a.qswapinternal(b);

    return;
}

inline void SSV_Single::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SSV_Single &b = dynamic_cast<SSV_Single &>(bb.getML());

    SSV_Binary::qswapinternal(b);

    qswap(dclass,b.dclass);

    return;
}

inline void SSV_Single::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SSV_Single &b = dynamic_cast<const SSV_Single &>(bb.getMLconst());

    SSV_Binary::semicopy(b);

    dclass = b.dclass;

    return;
}

inline void SSV_Single::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SSV_Single &src = dynamic_cast<const SSV_Single &>(bb.getMLconst());

    SSV_Binary::assign(src,onlySemiCopy);

    dclass = src.dclass;

    return;
}

#endif
