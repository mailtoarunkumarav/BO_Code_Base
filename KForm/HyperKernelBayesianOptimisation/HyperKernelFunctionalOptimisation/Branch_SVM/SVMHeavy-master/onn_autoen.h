
//
// Auto Encoder ONN
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _onn_autoen_h
#define _onn_autoen_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "onn_vector.h"







class ONN_AutoEn;


// Swap function

inline void qswap(ONN_AutoEn &a, ONN_AutoEn &b);


class ONN_AutoEn : public ONN_Vector
{
public:

    // Constructors, destructors, assignment operators and similar

    ONN_AutoEn();
    ONN_AutoEn(const ONN_AutoEn &src);
    ONN_AutoEn(const ONN_AutoEn &src, const ML_Base *xsrc);
    ONN_AutoEn &operator=(const ONN_AutoEn &src) { assign(src); return *this; }
    virtual ~ONN_AutoEn();

    virtual int restart(void) { ONN_AutoEn temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int type(void)    const { return 104; }
    virtual int subtype(void) const { return   0; }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'V'; }
    virtual char targType(void) const { return 'N'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int isClassifier(void) const { return 0; }

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    // Other functions

    virtual void assign(const ML_Base &src, int isOnlySemi = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    // Overloads to make sure target space aligns with input space

    virtual int addxspaceFeat(int i)    { return ONN_Vector::addtspaceFeat(i);    }
    virtual int removexspaceFeat(int i) { return ONN_Vector::removetspaceFeat(i); }

private:

    virtual int addTrainingVector (int i, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
};

inline void qswap(ONN_AutoEn &a, ONN_AutoEn &b)
{
    a.qswapinternal(b);

    return;
}

inline void ONN_AutoEn::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ONN_AutoEn &b = dynamic_cast<ONN_AutoEn &>(bb.getML());

    ONN_Vector::qswapinternal(b);

    return;
}

inline void ONN_AutoEn::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ONN_AutoEn &b = dynamic_cast<const ONN_AutoEn &>(bb.getMLconst());

    ONN_Vector::semicopy(b);

    return;
}

inline void ONN_AutoEn::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ONN_AutoEn &src = dynamic_cast<const ONN_AutoEn &>(bb.getMLconst());

    ONN_Vector::assign(src,onlySemiCopy);

    return;
}

#endif
