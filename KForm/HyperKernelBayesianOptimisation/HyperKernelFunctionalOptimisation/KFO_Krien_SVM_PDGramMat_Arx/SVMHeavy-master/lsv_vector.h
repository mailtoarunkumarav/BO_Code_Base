
//
// LS-SVM vector class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _lsv_vector_h
#define _lsv_vector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "lsv_generic.h"


class LSV_Vector;

// Swap and zeroing (restarting) functions

inline void qswap(LSV_Vector &a, LSV_Vector &b);
inline LSV_Vector &setzero(LSV_Vector &a);

class LSV_Vector : public LSV_Generic
{
public:

    // Constructors, destructors, assignment etc..

    LSV_Vector();
    LSV_Vector(const LSV_Vector &src);
    LSV_Vector(const LSV_Vector &src, const ML_Base *srcx);
    LSV_Vector &operator=(const LSV_Vector &src) { assign(src); return *this; }
    virtual ~LSV_Vector() { return; }

    virtual int prealloc(int expectedN);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    // Information functions (training data):

    virtual int type(void)      const { return 501; }
    virtual int subtype(void)   const { return 0;   }
    virtual int tspaceDim(void) const { return dbias.size(); }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'V'; }
    virtual char targType(void) const { return 'V'; }

    virtual int isUnderlyingScalar(void) const { return 0; }
    virtual int isUnderlyingVector(void) const { return 1; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual int settspaceDim(int newdim);
    virtual int addtspaceFeat(int i);
    virtual int removetspaceFeat(int i);

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

    virtual int sety(int                i, const Vector<double>          &y);
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &y);
    virtual int sety(                      const Vector<Vector<double> > &y);

    virtual int setd(int                i, int                nd);
    virtual int setd(const Vector<int> &i, const Vector<int> &nd);
    virtual int setd(                      const Vector<int> &nd);

    // General modification and autoset functions

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { LSV_Vector temp; *this = temp; return 1; }

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

    virtual const Vector<double>          &biasV (void) const { return dbiasV;  }
    virtual const Vector<Vector<double> > &alphaV(void) const { return dalphaV; }

private:

    virtual gentype &makezero(gentype &val) 
    { 
        val.force_vector(tspaceDim());

        if ( tspaceDim() )
        {
            int i;

            for ( i = 0 ; i < tspaceDim() ; i++ )
            {
                val.dir_vector()("&",i).force_double() = 0.0;
            }
        }

        return val; 
    }

    Vector<Vector<double> > dalphaV;
    Vector<double> dbiasV;

    Vector<Vector<double> > alltraintargV;

    LSV_Vector *thisthis;
    LSV_Vector **thisthisthis;
};

inline void qswap(LSV_Vector &a, LSV_Vector &b)
{
    a.qswapinternal(b);

    return;
}

inline LSV_Vector &setzero(LSV_Vector &a)
{
    a.restart();

    return a;
}

inline void LSV_Vector::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    LSV_Vector &b = dynamic_cast<LSV_Vector &>(bb.getML());

    LSV_Generic::qswapinternal(b);

    qswap(dalphaV      ,b.dalphaV      );
    qswap(dbiasV       ,b.dbiasV       );
    qswap(alltraintargV,b.alltraintargV);

    return;
}

inline void LSV_Vector::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const LSV_Vector &b = dynamic_cast<const LSV_Vector &>(bb.getMLconst());

    LSV_Generic::semicopy(b);

    dalphaV       = b.dalphaV;
    dbiasV        = b.dbiasV;
//    alltraintargV = b.alltraintargV;

    return;
}

inline void LSV_Vector::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const LSV_Vector &src = dynamic_cast<const LSV_Vector &>(bb.getMLconst());

    LSV_Generic::assign(src,onlySemiCopy);

    dalphaV       = src.dalphaV;
    dbiasV        = src.dbiasV;
    alltraintargV = src.alltraintargV;

    return;
}

#endif
