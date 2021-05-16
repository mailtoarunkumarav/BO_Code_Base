
//
// Multi-directional ranking SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_mvrank_h
#define _svm_mvrank_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_planar.h"





class SVM_MvRank;

// Swap function

inline void qswap(SVM_MvRank &a, SVM_MvRank &b);


class SVM_MvRank : public SVM_Planar
{
public:

    // Constructors, destructors, assignment etc..

    SVM_MvRank();
    SVM_MvRank(const SVM_MvRank &src);
    SVM_MvRank(const SVM_MvRank &src, const ML_Base *xsrc);
    SVM_MvRank &operator=(const SVM_MvRank &src) { assign(src); return *this; }
    virtual ~SVM_MvRank();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int restart(void) { SVM_MvRank temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information functions (training data):

    virtual int type(void)    const { return 17; }
    virtual int subtype(void) const { return 0;  }

    virtual int    maxitermvrank(void) const { return xmaxitermvrank; }
    virtual double lrmvrank(void)      const { return xlrmvrank;      }
    virtual double ztmvrank(void)      const { return xztmvrank;      }

    virtual double betarank(void) const { return xbetarank; }

    // Modification

    virtual int setmaxitermvrank(int nv) { xmaxitermvrank = nv; return 1; }
    virtual int setlrmvrank(double nv)   { xlrmvrank      = nv; return 1; }
    virtual int setztmvrank(double nv)   { xztmvrank      = nv; return 1; }

    virtual int setbetarank(double nv) { xbetarank = nv; return 1; }

    // Training

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

protected:

    // Protected function passthrough

    void calcLalpha(Matrix<double> &res);

private:

    int xmaxitermvrank;
    double xlrmvrank;
    double xztmvrank;
    double xbetarank;

    // thisthisthisthisthisthisthatthisthisthis

    SVM_MvRank *thisthis;
    SVM_MvRank **thisthisthis;
};

inline void qswap(SVM_MvRank &a, SVM_MvRank &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_MvRank::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_MvRank &b = dynamic_cast<SVM_MvRank &>(bb.getML());

    SVM_Planar::qswapinternal(b);
    
    qswap(xmaxitermvrank,b.xmaxitermvrank);
    qswap(xlrmvrank     ,b.xlrmvrank     );
    qswap(xztmvrank     ,b.xztmvrank     );
    qswap(xbetarank     ,b.xbetarank     );

    return;
}

inline void SVM_MvRank::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_MvRank &b = dynamic_cast<const SVM_MvRank &>(bb.getMLconst());

    SVM_Planar::semicopy(b);

    xmaxitermvrank = b.xmaxitermvrank;
    xlrmvrank      = b.xlrmvrank;
    xztmvrank      = b.xztmvrank;
    xbetarank      = b.xbetarank;

    return;
}

inline void SVM_MvRank::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_MvRank &src = dynamic_cast<const SVM_MvRank &>(bb.getMLconst());

    SVM_Planar::assign(static_cast<const SVM_Planar &>(src),onlySemiCopy);

    xmaxitermvrank = src.xmaxitermvrank;
    xlrmvrank      = src.xlrmvrank;
    xztmvrank      = src.xztmvrank;
    xbetarank      = src.xbetarank;

    return;
}

#endif
