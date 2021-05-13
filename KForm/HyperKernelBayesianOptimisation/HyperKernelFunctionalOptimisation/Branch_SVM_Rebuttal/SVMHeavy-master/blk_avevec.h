
//
// Average result block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_avevec_h
#define _blk_avevec_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"


// Defines a very basic set of blocks for use in machine learning.


class BLK_AveVec;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_AveVec &a, BLK_AveVec &b);


class BLK_AveVec : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_AveVec(int isIndPrune = 0);
    BLK_AveVec(const BLK_AveVec &src, int isIndPrune = 0);
    BLK_AveVec(const BLK_AveVec &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_AveVec &operator=(const BLK_AveVec &src) { assign(src); return *this; }
    virtual ~BLK_AveVec();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions (training data):

    virtual int NNC(int d)    const { return classcnt(d/2); }
    virtual int type(void)    const { return 202;           }
    virtual int subtype(void) const { return 0;             }

    virtual int tspaceDim(void)    const { return dim; }
    virtual int numClasses(void)   const { return 0;   }
      
    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'V'; }
    virtual char targType(void) const { return 'V'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual const Vector<int> &ClassLabels(void)   const { return classlabels; }
    virtual int getInternalClass(const gentype &y) const { (void) y; return 0; }

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

    virtual int sety(int                i, const gentype         &y);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y);
    virtual int sety(                      const Vector<gentype> &y);

    virtual int setd(int                i, int                d);
    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(                      const Vector<int> &d);

    virtual int settspaceDim(int newdim);
    virtual int addtspaceFeat(int i);
    virtual int removetspaceFeat(int i);

    // Evaluation Functions:
    //
    // Output g(x) is average of input vector.
    // Output h(x) is g(x) with outfn applied to it (or g(x) if outfn null).
    // Raw output is sum of vectors (not average)

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    virtual void dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const;

private:

    Vector<int> classlabels;
    Vector<int> classcnt;

    int dim;
};

inline void qswap(BLK_AveVec &a, BLK_AveVec &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_AveVec::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_AveVec &b = dynamic_cast<BLK_AveVec &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );
    qswap(dim        ,b.dim        );

    return;
}

inline void BLK_AveVec::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_AveVec &b = dynamic_cast<const BLK_AveVec &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;
    dim         = b.dim;

    return;
}

inline void BLK_AveVec::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_AveVec &src = dynamic_cast<const BLK_AveVec &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;
    dim         = src.dim;

    return;
}

#endif
