
//
// 1 layer neural network binary classification
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _onn_binary_h
#define _onn_binary_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "onn_generic.h"



class ONN_Binary;


// Swap and zeroing (restarting) functions

inline void qswap(ONN_Binary &a, ONN_Binary &b);
inline ONN_Binary &setzero(ONN_Binary &a);

class ONN_Binary : public ONN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    ONN_Binary();
    ONN_Binary(const ONN_Binary &src);
    ONN_Binary(const ONN_Binary &src, const ML_Base *xsrc);
    ONN_Binary &operator=(const ONN_Binary &src) { assign(src); return *this; }
    virtual ~ONN_Binary();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions (training data):

    virtual int NNC(int d)    const { return classcnt(d+1); }
    virtual int type(void)    const { return 103;           }
    virtual int subtype(void) const { return 0;             }

    virtual int tspaceDim(void)  const { return 1; }
    virtual int numClasses(void) const { return 2; }
    virtual int order(void)      const { return 0; }

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'Z'; }
    virtual char targType(void) const { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual const Vector<int> &ClassLabels(void)   const { return classlabels;     }
    virtual int getInternalClass(const gentype &y) const { NiceAssert( y.isValInteger() ); NiceAssert( ( (int) y == -1 ) || ( (int) y == +1 ) ); return (((int) y)+1)/2; }
    virtual int numInternalClasses(void)           const { return numClasses(); }

    virtual int isUnderlyingScalar(void) const { return 1; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual int isClassifier(void) const { return 1; }

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

    virtual int settspaceDim(int newdim) { return ML_Base::settspaceDim(newdim); }
    virtual int addtspaceFeat(int i)     { return ML_Base::addtspaceFeat(i);     }
    virtual int removetspaceFeat(int i)  { return ML_Base::removetspaceFeat(i);  }
    virtual int setorder(int neword)     { return ML_Base::setorder(neword);     }

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

private:

    Vector<int> classlabels;
    Vector<int> classcnt;
};

inline void qswap(ONN_Binary &a, ONN_Binary &b)
{
    a.qswapinternal(b);

    return;
}

inline ONN_Binary &setzero(ONN_Binary &a)
{
    a.restart();

    return a;
}

inline void ONN_Binary::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ONN_Binary &b = dynamic_cast<ONN_Binary &>(bb.getML());

    ONN_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );

    return;
}

inline void ONN_Binary::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ONN_Binary &b = dynamic_cast<const ONN_Binary &>(bb.getMLconst());

    ONN_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;

    return;
}

inline void ONN_Binary::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ONN_Binary &src = dynamic_cast<const ONN_Binary &>(bb.getMLconst());

    ONN_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;

    return;
}

#endif
