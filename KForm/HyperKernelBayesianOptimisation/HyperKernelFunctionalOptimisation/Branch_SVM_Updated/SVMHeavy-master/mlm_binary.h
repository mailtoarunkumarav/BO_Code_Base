
//
// Binary classifier Type-II multi-layer kernel-machine
//
// Version: 7
// Date: 07/07/2018
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// Currently this is basically a wrap-around for a LS-SVR with C mapped to
// 1/sigma for noise regularisation.  This is equivalent to the standard
// GP regressor assuming Gaussian measurement noise.
//

#ifndef _mlm_binary_h
#define _mlm_binary_h

#include "mlm_generic.h"
#include "svm_binary.h"




class MLM_Binary;

// Swap and zeroing (restarting) functions

inline void qswap(MLM_Binary &a, MLM_Binary &b);
inline MLM_Binary &setzero(MLM_Binary &a);

class MLM_Binary : public MLM_Generic
{
public:

    // Constructors, destructors, assignment etc..

    MLM_Binary();
    MLM_Binary(const MLM_Binary &src);
    MLM_Binary(const MLM_Binary &src, const ML_Base *srcx);
    MLM_Binary &operator=(const MLM_Binary &src) { assign(src); return *this; }
    virtual ~MLM_Binary() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );


    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Information functions

    virtual int type(void)    const { return 801; }
    virtual int subtype(void) const { return 0;   }

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }



    // ================================================================
    //     Common functions for all GPs
    // ================================================================

    // Information functions (training data):

    virtual       MLM_Generic &getMLM(void)            { return *this; }
    virtual const MLM_Generic &getMLMconst(void) const { return *this; }

    // General modification and autoset functions

    virtual       ML_Base &getML(void)            { return static_cast<      ML_Base &>(getMLM());      }
    virtual const ML_Base &getMLconst(void) const { return static_cast<const ML_Base &>(getMLMconst()); }



    // Base-level stuff

    virtual       SVM_Generic &getQ(void)            { return QQQ; }
    virtual const SVM_Generic &getQconst(void) const { return QQQ; }

private:

    SVM_Binary QQQ;

    MLM_Binary *thisthis;
    MLM_Binary **thisthisthis;
};

inline void qswap(MLM_Binary &a, MLM_Binary &b)
{
    a.qswapinternal(b);

    return;
}

inline MLM_Binary &setzero(MLM_Binary &a)
{
    a.restart();

    return a;
}

inline void MLM_Binary::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    MLM_Binary &b = dynamic_cast<MLM_Binary &>(bb.getML());

    MLM_Generic::qswapinternal(b);

    qswap(getQ(),b.getQ());

    return;
}

inline void MLM_Binary::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const MLM_Binary &b = dynamic_cast<const MLM_Binary &>(bb.getMLconst());

    MLM_Generic::semicopy(b);

    getQ().semicopy(b.getQconst());

    return;
}

inline void MLM_Binary::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const MLM_Binary &src = dynamic_cast<const MLM_Binary &>(bb.getMLconst());

    MLM_Generic::assign(src,onlySemiCopy);

    getQ().assign(src.getQconst(),onlySemiCopy);

    if ( !onlySemiCopy )
    {
        fixMLTree();
    }

    return;
}

#endif
