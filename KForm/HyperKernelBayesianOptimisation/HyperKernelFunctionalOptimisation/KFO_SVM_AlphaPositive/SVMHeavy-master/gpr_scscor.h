
//
// Scalar regression with ranking GP
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

#ifndef _gpr_scscor_h
#define _gpr_scscor_h

#include "gpr_generic.h"
#include "lsv_scscor.h"




class GPR_ScScor;

std::ostream &operator<<(std::ostream &output, const GPR_ScScor &src );
std::istream &operator>>(std::istream &input,        GPR_ScScor &dest);

// Swap and zeroing (restarting) functions

inline void qswap(GPR_ScScor &a, GPR_ScScor &b);
inline GPR_ScScor &setzero(GPR_ScScor &a);

class GPR_ScScor : public GPR_Generic
{
public:

    // Constructors, destructors, assignment etc..

    GPR_ScScor();
    GPR_ScScor(const GPR_ScScor &src);
    GPR_ScScor(const GPR_ScScor &src, const ML_Base *srcx);
    GPR_ScScor &operator=(const GPR_ScScor &src) { assign(src); return *this; }
    virtual ~GPR_ScScor() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output) const;
    virtual std::istream &inputstream(std::istream &input );


    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Information functions

    virtual int type(void)    const { return 405; }
    virtual int subtype(void) const { return 0;   }

    virtual int isClassifier(void) const { return 0; }



    // ================================================================
    //     Common functions for all GPs
    // ================================================================

    // Information functions (training data):

    virtual       GPR_Generic &getGPR(void)            { return *this; }
    virtual const GPR_Generic &getGPRconst(void) const { return *this; }

    // General modification and autoset functions

    virtual       ML_Base &getML(void)            { return static_cast<      ML_Base &>(getGPR());      }
    virtual const ML_Base &getMLconst(void) const { return static_cast<const ML_Base &>(getGPRconst()); }



    // Base-level stuff

    virtual       LSV_Generic &getQ(void)            { return QQ; }
    virtual const LSV_Generic &getQconst(void) const { return QQ; }






private:

    LSV_ScScor QQ;

    GPR_ScScor *thisthis;
    GPR_ScScor **thisthisthis;
};

inline void qswap(GPR_ScScor &a, GPR_ScScor &b)
{
    a.qswapinternal(b);

    return;
}

inline GPR_ScScor &setzero(GPR_ScScor &a)
{
    a.restart();

    return a;
}

inline void GPR_ScScor::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_ScScor &b = dynamic_cast<GPR_ScScor &>(bb.getML());

    GPR_Generic::qswapinternal(b);

    qswap(getQ(),b.getQ());

    return;
}

inline void GPR_ScScor::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_ScScor &b = dynamic_cast<const GPR_ScScor &>(bb.getMLconst());

    GPR_Generic::semicopy(b);

    getQ().semicopy(b.getQconst());

    return;
}

inline void GPR_ScScor::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_ScScor &src = dynamic_cast<const GPR_ScScor &>(bb.getMLconst());

    GPR_Generic::assign(src,onlySemiCopy);

    getQ().assign(src.getQconst(),onlySemiCopy);

    return;
}

#endif
