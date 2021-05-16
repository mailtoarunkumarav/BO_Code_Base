
//
// Bernstein polynomial block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_bernst_h
#define _blk_bernst_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"


// Defines a very basic set of blocks for use in machine learning.


class BLK_Bernst;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_Bernst &a, BLK_Bernst &b);


class BLK_Bernst : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Bernst(int isIndPrune = 0);
    BLK_Bernst(const BLK_Bernst &src, int isIndPrune = 0);
    BLK_Bernst(const BLK_Bernst &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_Bernst &operator=(const BLK_Bernst &src) { assign(src); return *this; }
    virtual ~BLK_Bernst();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions

    virtual int type(void)    const { return 215; }
    virtual int subtype(void) const { return 0;   }

    virtual int tspaceDim(void)  const { return 1; }
    virtual int numClasses(void) const { return 1; }

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'R'; }
    virtual char targType(void) const { return 'R'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const { (void) ia; return db ? ( ( (double) ha ) - ( (double) hb ) )*( ( (double) ha ) - ( (double) hb ) ) : 0; }

    virtual int isClassifier(void) const { return 0; }
    virtual int isRegression(void) const { return 1; }

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // Sampling stuff

    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp = DEFAULT_SAMPLES_SAMPLE)
    {
        localygood    = 0;
        locsampleMode = nv;
        locxmin       = xmin;
        locxmax       = xmax;
        locNsamp      = Nsamp;

        return BLK_Generic::setSampleMode(nv,xmin,xmax,Nsamp); 
    }

    // Trips for y update

    virtual int setBernDegree(const gentype &nv) { localygood = 0; return BLK_Generic::setBernDegree(nv); }
    virtual int setBernIndex(const gentype &nv)  { localygood = 0; return BLK_Generic::setBernIndex(nv);  }

    // This is really only used in one place - see globalopt.h

    virtual const Vector<gentype> &y(void) const;

private:

    // For speed, these emulate blk_conect stuff

    Vector<gentype> localy;
    int localygood; // 0 not good, 1 good, -1 individual components good, sum bad

    // Need these for getting "y" (which is sample data, ymmv) of mixed models

    int locsampleMode;
    Vector<gentype> locxmin;
    Vector<gentype> locxmax;
    int locNsamp;

    BLK_Bernst *thisthis;
    BLK_Bernst **thisthisthis;

};

inline void qswap(BLK_Bernst &a, BLK_Bernst &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_Bernst::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Bernst &b = dynamic_cast<BLK_Bernst &>(bb.getML());

    qswap(localy       ,b.localy       );
    qswap(localygood   ,b.localygood   );
    qswap(locsampleMode,b.locsampleMode);
    qswap(locxmin      ,b.locxmin      );
    qswap(locxmax      ,b.locxmax      );
    qswap(locNsamp     ,b.locNsamp     );

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_Bernst::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Bernst &b = dynamic_cast<const BLK_Bernst &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    localy        = b.localy;
    localygood    = b.localygood;
    locsampleMode = b.locsampleMode;
    locxmin       = b.locxmin;
    locxmax       = b.locxmax;
    locNsamp      = b.locNsamp;

    return;
}

inline void BLK_Bernst::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Bernst &src = dynamic_cast<const BLK_Bernst &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    localy        = src.localy;
    localygood    = src.localygood;
    locsampleMode = src.locsampleMode;
    locxmin       = src.locxmin;
    locxmax       = src.locxmax;
    locNsamp      = src.locNsamp;

    return;
}

#endif
