
//
// 1 layer neural network anionic regression
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _onn_anions_h
#define _onn_anions_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "onn_generic.h"



class ONN_Anions;


// Swap and zeroing (restarting) functions

inline void qswap(ONN_Anions &a, ONN_Anions &b);
inline ONN_Anions &setzero(ONN_Anions &a);

class ONN_Anions : public ONN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    ONN_Anions();
    ONN_Anions(const ONN_Anions &src);
    ONN_Anions(const ONN_Anions &src, const ML_Base *xsrc);
    ONN_Anions &operator=(const ONN_Anions &src) { assign(src); return *this; }
    virtual ~ONN_Anions();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions (training data):

    virtual int NNC(int d)    const { return classcnt(d/2); }
    virtual int type(void)    const { return 102;           }
    virtual int subtype(void) const { return 0;             }

    virtual int tspaceDim(void)  const { return 1<<order();           }
    virtual int numClasses(void) const { return 0;                    }
    virtual int order(void)      const { return ONN_Generic::order(); }

    virtual char gOutType(void) const { return 'A'; }
    virtual char hOutType(void) const { return 'A'; }
    virtual char targType(void) const { return 'A'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual const Vector<int> &ClassLabels(void)   const { return classlabels; }
    virtual int getInternalClass(const gentype &y) const { (void) y; return 0; }
    virtual int numInternalClasses(void)           const {           return 1; }

    virtual int isUnderlyingScalar(void) const { return 0; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 1; }

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

    virtual int settspaceDim(int newdim) { return ML_Base::settspaceDim(newdim); }
    virtual int addtspaceFeat(int i)     { return ML_Base::addtspaceFeat(i);     }
    virtual int removetspaceFeat(int i)  { return ML_Base::removetspaceFeat(i);  }



private:

    Vector<int> classlabels;
    Vector<int> classcnt;
};

inline void qswap(ONN_Anions &a, ONN_Anions &b)
{
    a.qswapinternal(b);

    return;
}

inline ONN_Anions &setzero(ONN_Anions &a)
{
    a.restart();

    return a;
}

inline void ONN_Anions::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ONN_Anions &b = dynamic_cast<ONN_Anions &>(bb.getML());

    ONN_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );

    return;
}

inline void ONN_Anions::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ONN_Anions &b = dynamic_cast<const ONN_Anions &>(bb.getMLconst());

    ONN_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;

    return;
}

inline void ONN_Anions::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ONN_Anions &src = dynamic_cast<const ONN_Anions &>(bb.getMLconst());

    ONN_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;

    return;
}

#endif
