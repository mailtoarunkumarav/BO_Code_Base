
//
// Battery simulation block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_batter_h
#define _blk_batter_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"


// Defines a very basic set of blocks for use in machine learning.

class BLK_Batter;



// Swap and zeroing (restarting) functions

inline void qswap(BLK_Batter &a, BLK_Batter &b);


class BLK_Batter : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Batter(int isIndPrune = 0);
    BLK_Batter(const BLK_Batter &src, int isIndPrune = 0);
    BLK_Batter(const BLK_Batter &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_Batter &operator=(const BLK_Batter &src) { assign(src); return *this; }
    virtual ~BLK_Batter();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions

    virtual int type(void)    const { return 216; }
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
    //
    // g(3,func,v) - time to reach target voltage given battery charging with current = func(t) > 0
    // g(2,func,v) - time to reach target voltage given battery charging with voltage = func(t) > 0
    // g(1,func,v) - time to reach target voltage given battery charging with power = func(t) > 0
    // g(0,func,v) - time to drop to target voltage given battery discharging with current = func(t) > 0
    // g(-1,t,i,v) - given time (t), current (i) and voltage (v) vectors, assuming current charging, returns how close the simulation is to the given data (in terms of voltage)
    // g(-2,dfile,m,N,s) - given real battery data in datafile, return how colse the simulation is to the given data (in terms of voltage)
    //             - m is the startpoint in file (0 for first line)
    //             - N is the max number of observations to compare (-1 for all)
    //             - s is the scalarisation
    //             - result is s*earlystop + ave_error, where earlystop is the number of observations skipped because of model failure
    //
    // dfile format:
    //
    // - first line ignored, then on each line:
    //          C1_Rec_ (iteration count, ignored
    //          C1_Tst_T_ (time (min) in simulation
    //          C1_Cur_A_ (current in or out, unsigned
    //          C1_Volt_V_ (terminal voltage
    //          C1_DPT_ (timestamp, ignored
    //          C1_Aux1_Tc_ (auxilliary temp 1 (ambient?)
    //          C1_Aux2_Tc_ (auxilliary temp 1 (ambient?)
    //          C1_zStage (state string, ignored
    //          C1_Stage_1 (1 for stage 1 charge - constant power C1_Cur_A_*C1_Volt_V_, 0 otherwise
    //          C1_Stage_2 (1 for stage 2 charge - constant voltageC1_Volt_V_, 0 otherwise
    //          C1_Stage_3 (1 for stage 3 charge - constant current C1_Cur_A_, 0 otherwise
    //
    // return integer: 0 if all good, 1 if result is infinite, -1 if result is NaN

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

private:

    BLK_Batter *thisthis;
    BLK_Batter **thisthisthis;

};

inline void qswap(BLK_Batter &a, BLK_Batter &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_Batter::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Batter &b = dynamic_cast<BLK_Batter &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_Batter::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Batter &b = dynamic_cast<const BLK_Batter &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_Batter::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Batter &src = dynamic_cast<const BLK_Batter &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
