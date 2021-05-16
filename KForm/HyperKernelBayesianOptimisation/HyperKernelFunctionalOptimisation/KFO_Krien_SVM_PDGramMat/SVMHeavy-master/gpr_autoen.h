
//
// Auto-encoding GP
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

#ifndef _gpr_autoen_h
#define _gpr_autoen_h

#include "gpr_generic.h"
#include "lsv_autoen.h"




class GPR_AutoEn;

std::ostream &operator<<(std::ostream &output, const GPR_AutoEn &src );
std::istream &operator>>(std::istream &input,        GPR_AutoEn &dest);

// Swap and zeroing (restarting) functions

inline void qswap(GPR_AutoEn &a, GPR_AutoEn &b);
inline GPR_AutoEn &setzero(GPR_AutoEn &a);

class GPR_AutoEn : public GPR_Generic
{
public:

    // Constructors, destructors, assignment etc..

    GPR_AutoEn();
    GPR_AutoEn(const GPR_AutoEn &src);
    GPR_AutoEn(const GPR_AutoEn &src, const ML_Base *srcx);
    GPR_AutoEn &operator=(const GPR_AutoEn &src) { assign(src); return *this; }
    virtual ~GPR_AutoEn() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output) const;
    virtual std::istream &inputstream(std::istream &input );


    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Information functions

    virtual int type(void)    const { return 407; }
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

    LSV_AutoEn QQ;

    GPR_AutoEn *thisthis;
    GPR_AutoEn **thisthisthis;
};

inline void qswap(GPR_AutoEn &a, GPR_AutoEn &b)
{
    a.qswapinternal(b);

    return;
}

inline GPR_AutoEn &setzero(GPR_AutoEn &a)
{
    a.restart();

    return a;
}

inline void GPR_AutoEn::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_AutoEn &b = dynamic_cast<GPR_AutoEn &>(bb.getML());

    GPR_Generic::qswapinternal(b);

    qswap(getQ(),b.getQ());

    return;
}

inline void GPR_AutoEn::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_AutoEn &b = dynamic_cast<const GPR_AutoEn &>(bb.getMLconst());

    GPR_Generic::semicopy(b);

    getQ().semicopy(b.getQconst());

    return;
}

inline void GPR_AutoEn::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_AutoEn &src = dynamic_cast<const GPR_AutoEn &>(bb.getMLconst());

    GPR_Generic::assign(src,onlySemiCopy);

    getQ().assign(src.getQconst(),onlySemiCopy);

    return;
}

#endif
