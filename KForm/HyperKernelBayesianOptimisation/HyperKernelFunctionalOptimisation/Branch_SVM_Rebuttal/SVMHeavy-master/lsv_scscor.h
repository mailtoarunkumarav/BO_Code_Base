
//
// Scalar regression with scoring LSV
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// Currently this is basically a wrap-around for a LS-SVR with C mapped to
// 1/sigma for noise regularisation.  This is equivalent to the standard
// GP regressor assuming Gaussian measurement noise.
//

#ifndef _lsv_scscor_h
#define _lsv_scscor_h

#include "lsv_scalar.h"
#include "svm_scscor.h"




class LSV_ScScor;

// Swap and zeroing (restarting) functions

inline void qswap(LSV_ScScor &a, LSV_ScScor &b);
inline LSV_ScScor &setzero(LSV_ScScor &a);

class LSV_ScScor : public LSV_Scalar
{
public:

    // Constructors, destructors, assignment etc..

    LSV_ScScor();
    LSV_ScScor(const LSV_ScScor &src);
    LSV_ScScor(const LSV_ScScor &src, const ML_Base *srcx);
    LSV_ScScor &operator=(const LSV_ScScor &src) { assign(src); return *this; }
    virtual ~LSV_ScScor() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual int type(void)    const { return 505; }
    virtual int subtype(void) const { return 0;   }

    // ...
    // No need to do the rest - they will be covered by the polymorph
    // of locgetspecSVM
    // ...

protected:

    virtual       SVM_Generic &locgetspecSVM(void)            { return static_cast<      SVM_Generic &>(QQ); }
    virtual const SVM_Generic &locgetspecSVMconst(void) const { return static_cast<const SVM_Generic &>(QQ); }

private:

    SVM_ScScor QQ;

    LSV_ScScor *thisthis;
    LSV_ScScor **thisthisthis;
};

inline void qswap(LSV_ScScor &a, LSV_ScScor &b)
{
    a.qswapinternal(b);

    return;
}

inline LSV_ScScor &setzero(LSV_ScScor &a)
{
    a.restart();

    return a;
}

inline void LSV_ScScor::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    LSV_ScScor &b = dynamic_cast<LSV_ScScor &>(bb.getML());

    LSV_Scalar::qswapinternal(b);

    qswap(QQ,b.QQ);

    return;
}

inline void LSV_ScScor::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const LSV_ScScor &b = dynamic_cast<const LSV_ScScor &>(bb.getMLconst());

    LSV_Scalar::semicopy(b);

    QQ.semicopy(b.QQ);

    return;
}

inline void LSV_ScScor::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const LSV_ScScor &src = dynamic_cast<const LSV_ScScor &>(bb.getMLconst());

    LSV_Scalar::assign(src,onlySemiCopy);

    QQ.assign(src.QQ,onlySemiCopy);

    return;
}

#endif
