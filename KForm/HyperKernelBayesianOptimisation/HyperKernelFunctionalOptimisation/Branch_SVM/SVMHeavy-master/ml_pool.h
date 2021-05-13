
//
// Mutable ML pool
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


//
// Controls a sparse vector of mutable ML blocks accessible by ID number
//

#ifndef _ml_pool_h
#define _ml_pool_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "mlcommon.h"
#include "vector.h"
#include "sparsevector.h"
#include "gentype.h"
#include "ml_base.h"
#include "ml_mutable.h"

class ML_Pool;

std::ostream &operator<<(std::ostream &output, const ML_Pool &src );
std::istream &operator>>(std::istream &input,        ML_Pool &dest);

// Swap function

inline void qswap(ML_Pool  &a, ML_Pool  &b);
inline void qswap(ML_Pool *&a, ML_Pool *&b);


class ML_Pool : public ML_Mutable
{
public:

    // Pool control functions
    //
    // Basically ML_Pool is a sparse vector of mutable ML blocks.  Each block
    // is associated with a unique non-negative integer ID.  You can add and
    // remove blocks.  Access is via setting the active block ID then using
    // the usual functions inheritted from ML_Mutable.
    //
    // NB: - you don't actually need to use add.  You can simply set the
    //       active MLID and start using it if you prefer.
    //     - when passing to errortest etc better to use activeML() result
    //       so that semicopy etc work as expected.
    //     - upon construction the active ML ID is set to 0 but this ML is
    //       not actually constructed (yet - if you don't add and set active
    //       a new ML ID then on use of the ML the ML ID 0 will be made).

    virtual int add(int type);
    virtual void remove(int MLID);
    virtual void removeall(void);
    virtual void setactiveML(int MLID);

    virtual int activeMLID(void) const { return actMLID; }
    virtual const Vector<int> &allactiveMLID(void) const { return MLstore.ind(); }

    virtual       ML_Mutable &activeML     (void)       { return MLstore("&",actMLID); }
    virtual const ML_Mutable &activeMLconst(void) const { return MLstore(     actMLID); }




    // Mutation functions

    virtual void setMLTypeMorph(int newmlType) { activeML().setMLTypeMorph(newmlType); return; }
    virtual void setMLTypeClean(int newmlType) { activeML().setMLTypeClean(newmlType); return; }

    // Constructors, destructors, assignment etc..

    ML_Pool();
    ML_Pool(const ML_Pool &src);
    ML_Pool(const ML_Pool &src, const ML_Base *srcx);
    ML_Pool &operator=(const ML_Pool &src) { assign(src); return *this; }
    virtual ~ML_Pool();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual       ML_Base &getML     (void)       { return activeML     ().getML();      }
    virtual const ML_Base &getMLconst(void) const { return activeMLconst().getMLconst(); }

    virtual       ML_Mutable &getSERIAL     (void)       { return activeML     ().getSERIAL();      }
    virtual const ML_Mutable &getSERIALconst(void) const { return activeMLconst().getSERIALconst(); }

    virtual       SVM_Generic &getSVM     (void)       { return activeML     ().getSVM();      }
    virtual const SVM_Generic &getSVMconst(void) const { return activeMLconst().getSVMconst(); }

    virtual       ONN_Generic &getONN     (void)       { return activeML     ().getONN();      }
    virtual const ONN_Generic &getONNconst(void) const { return activeMLconst().getONNconst(); }

    virtual       BLK_Generic &getBLK     (void)       { return activeML     ().getBLK();      }
    virtual const BLK_Generic &getBLKconst(void) const { return activeMLconst().getBLKconst(); }

    virtual       KNN_Generic &getKNN     (void)       { return activeML     ().getKNN();      }
    virtual const KNN_Generic &getKNNconst(void) const { return activeMLconst().getKNNconst(); }

    virtual       GPR_Generic &getGPR     (void)       { return activeML     ().getGPR();      }
    virtual const GPR_Generic &getGPRconst(void) const { return activeMLconst().getGPRconst(); }

    // Information functions (training data):

    virtual int isMutable(void) const { return 1; }
    virtual int isPool   (void) const { return 1; }

    virtual char targType(void) const { return activeMLconst().targType(); }





private:
    
    int actMLID;
    SparseVector<ML_Mutable> MLstore;
};



inline void qswap(ML_Pool &a, ML_Pool &b)
{
    a.qswapinternal(b);

    return;
}

inline void qswap(ML_Pool *&a, ML_Pool *&b)
{
    ML_Pool *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}

inline void ML_Pool::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ML_Pool &b = dynamic_cast<ML_Pool &>(bb.getML());

    qswap(actMLID,b.actMLID);
    qswap(MLstore,b.MLstore);

    return;
}

inline void ML_Pool::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ML_Pool &b = dynamic_cast<const ML_Pool &>(bb.getMLconst());

    actMLID = b.actMLID;
    MLstore = b.MLstore;

    return;
}

inline void ML_Pool::assign(const ML_Base &bb, int onlySemiCopy)
{
    (void) onlySemiCopy;

    NiceAssert( isAssignCompat(*this,bb) );

    const ML_Pool &src = dynamic_cast<const ML_Pool &>(bb.getMLconst());

    actMLID = src.actMLID;
    MLstore = src.MLstore;

    return;
}

#endif
