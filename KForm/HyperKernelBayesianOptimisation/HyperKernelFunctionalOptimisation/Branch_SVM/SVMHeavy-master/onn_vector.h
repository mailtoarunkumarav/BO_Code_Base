
//
// 1 layer neural network vector regression
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _onn_vector_h
#define _onn_vector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "onn_generic.h"



class ONN_Vector;


// Swap and zeroing (restarting) functions

inline void qswap(ONN_Vector &a, ONN_Vector &b);
inline ONN_Vector &setzero(ONN_Vector &a);

class ONN_Vector : public ONN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    ONN_Vector();
    ONN_Vector(const ONN_Vector &src);
    ONN_Vector(const ONN_Vector &src, const ML_Base *xsrc);
    ONN_Vector &operator=(const ONN_Vector &src) { assign(src); return *this; }
    virtual ~ONN_Vector();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions (training data):

    virtual int NNC(int d)    const { return classcnt(d/2); }
    virtual int type(void)    const { return 101;           }
    virtual int subtype(void) const { return 0;             }

    virtual int tspaceDim(void)  const { return ONN_Generic::tspaceDim(); }
    virtual int numClasses(void) const { return 0;                        }
    virtual int order(void)      const { return 0;                        }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'V'; }
    virtual char targType(void) const { return 'V'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual const Vector<int> &ClassLabels(void)   const { return classlabels; }
    virtual int getInternalClass(const gentype &y) const { (void) y; return 0; }
    virtual int numInternalClasses(void)           const {           return 1; }

    virtual int isUnderlyingScalar(void) const { return 0; }
    virtual int isUnderlyingVector(void) const { return 1; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual int isClassifier(void) const { return 0; }

    // Training set modification - need to overload to maintain counts

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    virtual int sety(int i, const gentype &y);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y);
    virtual int sety(const Vector<gentype> &y);

    virtual int setd(int i, int d);
    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(const Vector<int> &d);



    virtual int setorder(int neword)     { return ML_Base::setorder(neword);     }



private:

    Vector<int> classlabels;
    Vector<int> classcnt;
};

inline void qswap(ONN_Vector &a, ONN_Vector &b)
{
    a.qswapinternal(b);

    return;
}

inline ONN_Vector &setzero(ONN_Vector &a)
{
    a.restart();

    return a;
}

inline void ONN_Vector::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ONN_Vector &b = dynamic_cast<ONN_Vector &>(bb.getML());

    ONN_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );

    return;
}

inline void ONN_Vector::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ONN_Vector &b = dynamic_cast<const ONN_Vector &>(bb.getMLconst());

    ONN_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;

    return;
}

inline void ONN_Vector::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ONN_Vector &src = dynamic_cast<const ONN_Vector &>(bb.getMLconst());

    ONN_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;

    return;
}

#endif
