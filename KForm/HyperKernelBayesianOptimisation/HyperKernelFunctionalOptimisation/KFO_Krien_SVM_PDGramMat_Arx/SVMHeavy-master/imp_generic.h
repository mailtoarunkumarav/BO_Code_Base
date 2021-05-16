
//
// Improvement measure base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _imp_generic_h
#define _imp_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.h"



// Defines blocks used as mono-surrogates in multitarget bayesian
// optimisation.  It is assumed that we are trying to solve the multi-
// objective minimisation problem:
//
// min_x f(x) = (f0(x), f1(x), ...)
//
// The output vectors f(x0), f(x1), ... are collected and added to this
// block as training data.  The *negated*  improvement from adding a new
// vector f(x) to the training set is given by the imp(...) function.  The
// smallest (most negative) improvement indicates the best candidate.
//
// Note that it is generally a good idea to enforce:
//
// f(x) <= 0
//
// This is only strictly required by imp_expect in the multi-objective
// case, but nevertheless it is a good idea for compatibility with this
// class.
//
// Note also that all x vectors should be in dense form rather than sparse.


class IMP_Generic;


// Swap and zeroing (restarting) functions

inline void qswap(IMP_Generic &a, IMP_Generic &b);
inline IMP_Generic &setzero(IMP_Generic &a);

class IMP_Generic : public ML_Base
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    IMP_Generic(int isIndPrune = 0);
    IMP_Generic(const IMP_Generic &src, int isIndPrune = 0);
    IMP_Generic(const IMP_Generic &src, const ML_Base *xsrc, int isIndPrune = 0);
    IMP_Generic &operator=(const IMP_Generic &src) { assign(src); return *this; }
    virtual ~IMP_Generic();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const;

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual       ML_Base &getML(void)            { return static_cast<      ML_Base &>(getIMP()     ); }
    virtual const ML_Base &getMLconst(void) const { return static_cast<const ML_Base &>(getIMPconst()); }

    // Information functions (training data):

    virtual int subtype(void) const { return 0;   }

    virtual int isTrained(void) const { return disTrained; }

    virtual char targType(void) const { return 'N'; }

    // Data modification

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);
    virtual int removeTrainingVector(int i, int num);

    virtual int setx(int                i, const SparseVector<gentype>          &x);
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x);
    virtual int setx(                      const Vector<SparseVector<gentype> > &x);

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0);
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0);
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0);

    virtual int setd(int                i, int                nd);
    virtual int setd(const Vector<int> &i, const Vector<int> &nd);
    virtual int setd(                      const Vector<int> &nd);

    // Training functions:

    virtual int train(int &res, svmvolatile int &) { (void) res; disTrained = 1; return 0; }
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // General modification and autoset functions

    virtual int reset(void)   { untrain(); return 1;       }
    virtual int restart(void) { return ML_Base::restart(); }




    // ================================================================
    //     IMP Specific functions
    // ================================================================

    virtual       IMP_Generic &getIMP     (void)       { return *this; }
    virtual const IMP_Generic &getIMPconst(void) const { return *this; }

    // Improvement functions: given mean and variance of input x calculate
    // relevant measure of improvement/goodness of result.

    virtual int imp(gentype &resi, const SparseVector<gentype> &xxmean, const gentype &xxvar) const { (void) resi; (void) xxmean; (void) xxvar; throw("Error: imp() not defined at generic level."); return 0; }

    // Information functions
    //
    // zref:      zero reference used by EHI
    // ehimethod: EHI calculation method (for EI with vector input)
    //            0: use optimised EHI calculation with full cache
    //            1: use optimised EHI calculation with partial cache
    //            2: use optimised EHI calculation with no cache
    //            3: use Hupkens method
    //            4: use Couckuyt method
    // needdg:    does this require dg/dx?
    //            0: no
    //            1: yes
    // hypervol:  hypervolume of training set

    virtual double zref     (void) const { return xzref;      }
    virtual int    ehimethod(void) const { return xehimethod; }
    virtual int    needdg   (void) const { return 0;          }
    virtual double hypervol (void) const;

    // Modification function

    virtual int setzref     (double nv) {                                               xzref      = nv; return 0; }
    virtual int setehimethod(int    nv) { NiceAssert( nv >= 0 ); NiceAssert( nv <= 4 ); xehimethod = nv; return 0; }

protected:

    // Overload these in all derived classes

    virtual void untrain(void)
    {
        disTrained = 0;

        return;
    }

private:

    double xzref;
    int xehimethod;
    int disTrained;
};

inline void qswap(IMP_Generic &a, IMP_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline IMP_Generic &setzero(IMP_Generic &a)
{
    a.restart();

    return a;
}

inline void IMP_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    IMP_Generic &b = dynamic_cast<IMP_Generic &>(bb.getML());

    qswap(xzref     ,b.xzref     );
    qswap(xehimethod,b.xehimethod);
    qswap(disTrained,b.disTrained);

    ML_Base::qswapinternal(b);

    return;
}

inline void IMP_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const IMP_Generic &b = dynamic_cast<const IMP_Generic &>(bb.getMLconst());

    xzref      = b.xzref;
    xehimethod = b.xehimethod;
    disTrained = b.disTrained;

    ML_Base::semicopy(b);

    return;
}

inline void IMP_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const IMP_Generic &src = dynamic_cast<const IMP_Generic &>(bb.getMLconst());

    xzref      = src.xzref;
    xehimethod = src.xehimethod;
    disTrained = src.disTrained;

    ML_Base::assign(src,onlySemiCopy);

    return;
}

#endif
