
//
// Consensus result block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_consen_h
#define _blk_consen_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"
#include "idstore.h"


// Defines a very basic set of blocks for use in machine learning.


class BLK_Consen;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_Consen &a, BLK_Consen &b);


class BLK_Consen : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Consen(int isIndPrune = 0);
    BLK_Consen(const BLK_Consen &src, int isIndPrune = 0);
    BLK_Consen(const BLK_Consen &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_Consen &operator=(const BLK_Consen &src) { assign(src); return *this; }
    virtual ~BLK_Consen();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions

    virtual int NNC(int d)    const { return Nnc(label_placeholder.findID(d)+1); }
    virtual int type(void)    const { return 201; }
    virtual int subtype(void) const { return 0;   }

    virtual int tspaceDim(void)  const { return 1;                        }
    virtual int numClasses(void) const { return label_placeholder.size(); }

    virtual char gOutType(void) const { return 'Z'; }
    virtual char hOutType(void) const { return 'Z'; }
    virtual char targType(void) const { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const { (void) ia; return db ? ( ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0 ) : 0; }

    virtual const Vector<int> &ClassLabels(void)   const { return label_placeholder.getreftoID(); }
    virtual int getInternalClass(const gentype &y) const { return label_placeholder.findID((int) y); }
    virtual int numInternalClasses(void)           const { return numClasses(); }

    virtual int isClassifier(void) const { return 1; }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    virtual int sety(int                i, const gentype         &y);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y);
    virtual int sety(                      const Vector<gentype> &y);

    virtual int setd(int                i, int                nd);
    virtual int setd(const Vector<int> &i, const Vector<int> &nd);
    virtual int setd(                      const Vector<int> &nd);

    // General modification and autoset functions

    virtual int addclass(int label, int epszero = 0);

    // Evaluation Functions:
    //
    // Output g(x) is the consensus result
    // Output h(x) is the concensus result
    // Output raw is the concensus result

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // Private

    IDStore label_placeholder;
    Vector<int> Nnc;
};

inline void qswap(BLK_Consen &a, BLK_Consen &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_Consen::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Consen &b = dynamic_cast<BLK_Consen &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    qswap(label_placeholder,b.label_placeholder);
    qswap(Nnc              ,b.Nnc              );

    return;
}

inline void BLK_Consen::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Consen &b = dynamic_cast<const BLK_Consen &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    //label_placeholder

    Nnc = b.Nnc;

    return;
}

inline void BLK_Consen::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Consen &src = dynamic_cast<const BLK_Consen &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    label_placeholder = src.label_placeholder;

    Nnc = src.Nnc;

    return;
}

#endif
