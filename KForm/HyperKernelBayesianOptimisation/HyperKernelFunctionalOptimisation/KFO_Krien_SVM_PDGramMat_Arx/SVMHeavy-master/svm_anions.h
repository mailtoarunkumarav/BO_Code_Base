
//
// Anionic regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_anions_h
#define _svm_anions_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_vector.h"











class SVM_Anions;


// Swap function

inline void qswap(SVM_Anions &a, SVM_Anions &b);


class SVM_Anions : public SVM_Vector
{
public:

    // Constructors, destructors, assignment operators and similar

    SVM_Anions();
    SVM_Anions(const SVM_Anions &src);
    SVM_Anions(const SVM_Anions &src, const ML_Base *xsrc);
    SVM_Anions &operator=(const SVM_Anions &src) { assign(src); return *this; }
    virtual ~SVM_Anions();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { SVM_Anions temp; *this = temp; return 1; }

    virtual int setAlphaA(const Vector<d_anion> &newAlpha);
    virtual int setBiasA(const d_anion &newBias);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int type(void)    const { return 5; }
    virtual int subtype(void) const { return 0; }

    virtual char gOutType(void) const { return 'A'; }
    virtual char hOutType(void) const { return 'A'; }
    virtual char targType(void) const { return 'A'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int isUnderlyingScalar(void) const { return 0; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 1; }

    virtual const Vector<d_anion> &zA(void)     const { return traintarg; }
    virtual const d_anion         &biasA(void)  const { return db;        }
    virtual const Vector<d_anion> &alphaA(void) const { return dalpha;    }

    virtual int isClassifier(void) const { return 0; }

    // Modification:

    virtual int setLinearCost(void);
    virtual int setQuadraticCost(void);

    virtual int setC(double xC);

    virtual int sety(int i, const d_anion &z);
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &z);
    virtual int sety(const Vector<d_anion> &z);

    virtual int settspaceDim(int newdim);
    virtual int addtspaceFeat(int i);
    virtual int removetspaceFeat(int i);
    virtual int setorder(int neword);

    // Train the SVM

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Training set control:

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int addTrainingVector( int i, const d_anion &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const d_anion &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector( int i, const Vector<d_anion> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i, const Vector<d_anion> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

    virtual int sety(int i, const gentype &z);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z);
    virtual int sety(const Vector<gentype> &z);

    virtual int setd(int i, int d);

    virtual int setCweight(int i, double xCweight);
    virtual int setCweight(const Vector<int> &i, const Vector<double> &xCweight);
    virtual int setCweight(const Vector<double> &xCweight);

    virtual int setCweightfuzz(int i, double nw);
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nw);
    virtual int setCweightfuzz(const Vector<double> &nw);

    virtual int scaleCweight(double scalefactor);
    virtual int scaleCweightfuzz(double scalefactor);

    virtual int randomise(double sparsity);

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

private:

    Vector<d_anion> dalpha;
    d_anion db;
    Vector<d_anion> traintarg;

    void grabalpha(void);
    void grabdb(void);
    void grabtraintarg(void);

    // Blocked functions

    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha);
    virtual int setBiasV(const Vector<double> &newBias);

    virtual int sety(int i, const Vector<double> &z);
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z);
    virtual int sety(const Vector<Vector<double> > &z);

    virtual int addTrainingVector( int i, const Vector<double> &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
};

inline void qswap(SVM_Anions &a, SVM_Anions &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Anions::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Anions &b = dynamic_cast<SVM_Anions &>(bb.getML());

    SVM_Vector::qswapinternal(b);

    qswap(dalpha,b.dalpha);
    qswap(db,b.db);
    qswap(traintarg,b.traintarg);

    return;
}

inline void SVM_Anions::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Anions &b = dynamic_cast<const SVM_Anions &>(bb.getMLconst());

    SVM_Vector::semicopy(b);

    traintarg = b.traintarg;

    dalpha = b.dalpha;
    db     = b.db;

    return;
}

inline void SVM_Anions::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Anions &src = dynamic_cast<const SVM_Anions &>(bb.getMLconst());

    SVM_Vector::assign(src,onlySemiCopy);
            
    dalpha    = src.dalpha;
    db        = src.db;
    traintarg = src.traintarg;

    return;
}


#endif
