
//
// LS-SVM anionic class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _lsv_anions_h
#define _lsv_anions_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "lsv_generic.h"


class LSV_Anions;

// Swap and zeroing (restarting) functions

inline void qswap(LSV_Anions &a, LSV_Anions &b);
inline LSV_Anions &setzero(LSV_Anions &a);

class LSV_Anions : public LSV_Generic
{
public:

    // Constructors, destructors, assignment etc..

    LSV_Anions();
    LSV_Anions(const LSV_Anions &src);
    LSV_Anions(const LSV_Anions &src, const ML_Base *srcx);
    LSV_Anions &operator=(const LSV_Anions &src) { assign(src); return *this; }
    virtual ~LSV_Anions() { return; }

    virtual int prealloc(int expectedN);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    // Information functions (training data):

    virtual int type(void)      const { return 502; }
    virtual int subtype(void)   const { return 0;   }
    virtual int tspaceDim(void) const { return dbias.size(); }
    virtual int order(void)     const { return dbias.order(); }

    virtual char gOutType(void) const { return 'A'; }
    virtual char hOutType(void) const { return 'A'; }
    virtual char targType(void) const { return 'A'; }

    virtual int isUnderlyingScalar(void) const { return 0; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 1; }

    virtual int setorder(int neword);

    virtual int getInternalClass(const gentype &y) const;

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) { return LSV_Generic::addTrainingVector (i,y,x,Cweigh,epsweigh); }
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) { return LSV_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh); }

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);
    virtual int removeTrainingVector(int i, int num) { return ML_Base::removeTrainingVector(i,num); }

    virtual int sety(int                i, const gentype         &y);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y);
    virtual int sety(                      const Vector<gentype> &y);

    virtual int sety(int                i, const d_anion         &y);
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &y);
    virtual int sety(                      const Vector<d_anion> &y);

    virtual int setd(int                i, int                nd);
    virtual int setd(const Vector<int> &i, const Vector<int> &nd);
    virtual int setd(                      const Vector<int> &nd);

    // General modification and autoset functions

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { LSV_Anions temp; *this = temp; return 1; }

    // Training functions:

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Use functions

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // ================================================================
    //     Common functions for all LS-SVMs
    // ================================================================

    virtual int setgamma(const Vector<gentype> &newgamma);
    virtual int setdelta(const gentype         &newdelta);

    // ================================================================
    //     Required by K2xfer
    // ================================================================

    virtual const d_anion         &biasA (void) const { return dbiasA;  }
    virtual const Vector<d_anion> &alphaA(void) const { return dalphaA; }

private:

    virtual gentype &makezero(gentype &val) 
    { 
        val.force_anion().setorder(order()) *= 0.0;

        return val; 
    }

    Vector<d_anion> dalphaA;
    d_anion dbiasA;

    Vector<d_anion> alltraintargA;

    LSV_Anions *thisthis;
    LSV_Anions **thisthisthis;
};

inline void qswap(LSV_Anions &a, LSV_Anions &b)
{
    a.qswapinternal(b);

    return;
}

inline LSV_Anions &setzero(LSV_Anions &a)
{
    a.restart();

    return a;
}

inline void LSV_Anions::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    LSV_Anions &b = dynamic_cast<LSV_Anions &>(bb.getML());

    LSV_Generic::qswapinternal(b);

    qswap(dalphaA      ,b.dalphaA      );
    qswap(dbiasA       ,b.dbiasA       );
    qswap(alltraintargA,b.alltraintargA);

    return;
}

inline void LSV_Anions::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const LSV_Anions &b = dynamic_cast<const LSV_Anions &>(bb.getMLconst());

    LSV_Generic::semicopy(b);

    dalphaA       = b.dalphaA;
    dbiasA        = b.dbiasA;
//    alltraintargA = b.alltraintargA;

    return;
}

inline void LSV_Anions::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const LSV_Anions &src = dynamic_cast<const LSV_Anions &>(bb.getMLconst());

    LSV_Generic::assign(src,onlySemiCopy);

    dalphaA       = src.dalphaA;
    dbiasA        = src.dbiasA;
    alltraintargA = src.alltraintargA;

    return;
}


#endif
