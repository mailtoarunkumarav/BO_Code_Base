
//
// k-nearest-neighbour vector classifier
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _knn_vector_h
#define _knn_vector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "knn_generic.h"



class KNN_Vector;


// Swap and zeroing (restarting) functions

inline void qswap(KNN_Vector &a, KNN_Vector &b);
inline KNN_Vector &setzero(KNN_Vector &a);

class KNN_Vector : public KNN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    KNN_Vector();
    KNN_Vector(const KNN_Vector &src);
    KNN_Vector(const KNN_Vector &src, const ML_Base *xsrc);
    KNN_Vector &operator=(const KNN_Vector &src) { assign(src); return *this; }
    virtual ~KNN_Vector();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions (training data):

    virtual int NNC(int d)    const { return classcnt(d/2); }
    virtual int type(void)    const { return 304;           }
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

    virtual int sety(int i, const gentype &y);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y);
    virtual int sety(const Vector<gentype> &y);

    virtual int setd(int i, int d);
    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(const Vector<int> &d);

    virtual int settspaceDim(int newdim);
    virtual int addtspaceFeat(int i);
    virtual int removetspaceFeat(int i);

    // Fast version of g(x)

    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return ggTrainingVectorInt(resg,i,retaltg,pxyprodi); }

    // Randomisation: for KNN type, randomisation occurs in the target, as
    // this is the only variable available

    virtual int randomise(double sparsity);

private:

    virtual const Vector<Vector<double> > &yV(void) const { return z; }

    Vector<int> classlabels;
    Vector<int> classcnt;

    Vector<Vector<double> > z;

    int dim;

    virtual void hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const;
    virtual void hfn(Vector<double> &res, const Vector<Vector<double> > &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const;
};

inline void qswap(KNN_Vector &a, KNN_Vector &b)
{
    a.qswapinternal(b);

    return;
}

inline KNN_Vector &setzero(KNN_Vector &a)
{
    a.restart();

    return a;
}

inline void KNN_Vector::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    KNN_Vector &b = dynamic_cast<KNN_Vector &>(bb.getML());

    KNN_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );
    qswap(dim        ,b.dim        );
    qswap(z          ,b.z          );

    return;
}

inline void KNN_Vector::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const KNN_Vector &b = dynamic_cast<const KNN_Vector &>(bb.getMLconst());

    KNN_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;
    dim         = b.dim;
    //z           = b.z;

    return;
}

inline void KNN_Vector::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const KNN_Vector &src = dynamic_cast<const KNN_Vector &>(bb.getMLconst());

    KNN_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;
    dim         = src.dim;
    z           = src.z;

    return;
}

#endif
