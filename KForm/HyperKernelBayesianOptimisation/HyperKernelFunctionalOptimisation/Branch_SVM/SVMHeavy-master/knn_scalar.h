
//
// k-nearest-neighbour scalar regression
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _knn_scalar_h
#define _knn_scalar_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "knn_generic.h"



class KNN_Scalar;


// Swap and zeroing (restarting) functions

inline void qswap(KNN_Scalar &a, KNN_Scalar &b);
inline KNN_Scalar &setzero(KNN_Scalar &a);

class KNN_Scalar : public KNN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    KNN_Scalar();
    KNN_Scalar(const KNN_Scalar &src);
    KNN_Scalar(const KNN_Scalar &src, const ML_Base *xsrc);
    KNN_Scalar &operator=(const KNN_Scalar &src) { assign(src); return *this; }
    virtual ~KNN_Scalar();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions (training data):

    virtual int NNC(int d)    const { return classcnt(d+1); }
    virtual int type(void)    const { return 303;           }
    virtual int subtype(void) const { return 0;             }

    virtual int tspaceDim(void)    const { return 1; }
    virtual int numClasses(void)   const { return 0; }
      
    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'R'; }
    virtual char targType(void) const { return 'R'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual const Vector<int> &ClassLabels(void)   const { return classlabels; }
    virtual int getInternalClass(const gentype &y) const { return ( ( (double) y ) < 0 ) ? 0 : 1; }

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

    // Fast version of g(x)

    virtual int ggTrainingVector(double &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return ggTrainingVectorInt(resg,i,retaltg,pxyprodi); }

    // Randomisation: for KNN type, randomisation occurs in the target, as
    // this is the only variable available

    virtual int randomise(double sparsity);

private:

    virtual const Vector<double> &yR(void) const { return z; }

    Vector<int> classlabels;
    Vector<int> classcnt;

    Vector<double> z;

    virtual void hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const;
    virtual void hfn(double &res, const Vector<double> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const;
};

inline void qswap(KNN_Scalar &a, KNN_Scalar &b)
{
    a.qswapinternal(b);

    return;
}

inline KNN_Scalar &setzero(KNN_Scalar &a)
{
    a.restart();

    return a;
}

inline void KNN_Scalar::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    KNN_Scalar &b = dynamic_cast<KNN_Scalar &>(bb.getML());

    KNN_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );
    qswap(z          ,b.z          );

    return;
}

inline void KNN_Scalar::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const KNN_Scalar &b = dynamic_cast<const KNN_Scalar &>(bb.getMLconst());

    KNN_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;
    z           = b.z;

    return;
}

inline void KNN_Scalar::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const KNN_Scalar &src = dynamic_cast<const KNN_Scalar &>(bb.getMLconst());

    KNN_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;
    z           = src.z;

    return;
}

#endif
