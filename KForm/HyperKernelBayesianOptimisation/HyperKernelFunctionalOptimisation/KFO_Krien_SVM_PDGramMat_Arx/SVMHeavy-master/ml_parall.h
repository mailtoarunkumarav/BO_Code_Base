//FIXME: adding data should add to all blocks
//FIXME: removing data should remove from all blocks
//FIXME: see backup 16/11/2016 for how that should work

//
// Parallel ML module
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _ml_parall_h
#define _ml_parall_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_mutable.h"



class ML_Parall;


class ParallBackcall;

ML_Base *makeMLParall(void);
void assigntoMLParall(ML_Base **dest, const ML_Base *src, int onlySemiCopy);
void xferMLParall(ML_Base &dest, ML_Base &src);

inline void qswap(ParallBackcall &a, ParallBackcall &b);

class ParallBackcall : public BLK_Nopnop
{
public:

    // Constructors etc

    ParallBackcall(ML_Parall *xowner = NULL);
    ParallBackcall(ML_Parall *xowner, const ParallBackcall &src);
    ParallBackcall(ML_Parall *xowner, const ParallBackcall &src, const ML_Base *srcx);
    ParallBackcall &operator=(const ParallBackcall &src) { assign(src); return *this; }
    virtual ~ParallBackcall();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual       ML_Base &getML     (void)       { return *this; }
    virtual const ML_Base &getMLconst(void) const { return *this; }

    // Information functions (training data):

    virtual int type(void)      const { return -3; }
    virtual int subtype(void)   const { return 0;  }

    virtual int isTrained(void) const;

    virtual int tspaceDim(void)    const;
    virtual int xspaceDim(void)    const;
    virtual int tspaceSparse(void) const { return 0; }
    virtual int xspaceSparse(void) const { return 1; }
    virtual int numClasses(void)   const { return ML_Base::order(); }
    virtual int order(void)        const { return ML_Base::order(); }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'V'; }
    virtual char targType(void) const { return 'V'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int isUnderlyingScalar(void) const { return 0; }
    virtual int isUnderlyingVector(void) const { return 1; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual const Vector<int> &ClassLabels(void)   const { return ML_Base::ClassLabels();        }
    virtual int getInternalClass(const gentype &y) const { return ML_Base::getInternalClass(y);  }
    virtual int numInternalClasses(void)           const { return ML_Base::numInternalClasses(); }

    virtual double sparlvl(void) const;

    virtual int isClassifier(void) const { return 0; }
    virtual int isRegression(void) const { return 1; }

    // General modification and autoset functions

    virtual int randomise(double sparsity);
    virtual int autoen(void);
    virtual int renormalise(void);
    virtual int realign(void);

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void);
    virtual int home(void);

    // Training functions:

    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch);

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0) const;
    virtual int gvTrainingVector(gentype &resv,                int i                 ) const;
//    virtual int gvTrainingVector(Matrix<double> &resv,         int Nx, int dummy     ) const;
    virtual void eTrainingVector(gentype &res,                 int i                 ) const;

    virtual int ggTrainingVector(double &resg,         int i, int retaltg = 0) const { (void) resg; (void) i; (void) retaltg; throw("Error: ggTrainingVector only defined for vectors in ml_parall."); return 0; }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0) const;
    virtual int ggTrainingVector(d_anion &resg,        int i, int retaltg = 0) const { (void) resg; (void) i; (void) retaltg; throw("Error: ggTrainingVector only defined for vectors in ml_parall."); return 0; }

    virtual void dgTrainingVector(SparseVector<gentype> &resx, int i) const;

private:

    // This class always requires a reference to its "owner"
    // NB: the ONLY variable must be a pointer to the owner

    ML_Parall *owner;
};





std::ostream &operator<<(std::ostream &output, const ML_Parall &src );
std::istream &operator>>(std::istream &input,        ML_Parall &dest);



// Swap and zeroing (restarting) functions

inline void qswap(ML_Parall &a, ML_Parall &b);

class ML_Parall : public ML_Mutable
{
    friend class ParallBackcall;

public:

    // Mutation functions

    virtual void setMLTypeMorph(int newmlType) { if ( type() >= 0 ) { ML_Mutable::setMLTypeMorph(newmlType); } else { dynamic_cast<ML_Mutable &>(*(theML(mlind))).setMLTypeMorph(newmlType); } return; }
    virtual void setMLTypeClean(int newmlType) { if ( type() >= 0 ) { ML_Mutable::setMLTypeClean(newmlType); } else { dynamic_cast<ML_Mutable &>(*(theML(mlind))).setMLTypeClean(newmlType); } return; }

    // Constructors, destructors, assignment etc..

    ML_Parall();
    ML_Parall(const ML_Parall &src);
    ML_Parall(const ML_Parall &src, const ML_Base *srcx);
    ML_Parall &operator=(const ML_Parall &src) { assign(src); return *this; }
    virtual ~ML_Parall();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output) const { return getMLconst().printstream(output); }
    virtual std::istream &inputstream(std::istream &input )       { return getML     ().inputstream(input ); }

    virtual       ML_Base &getML     (void)       { return ( mlind == -1 ) ? datastore : ML_Mutable::getML();      }
    virtual const ML_Base &getMLconst(void) const { return ( mlind == -1 ) ? datastore : ML_Mutable::getMLconst(); }

    virtual int isMutable(void) const { return 0; }

    // ================================================================
    //     Serial/Parallel placeholder functions
    // ================================================================

    virtual       ML_Mutable &getSERIAL     (void)       { return *this; }
    virtual const ML_Mutable &getSERIALconst(void) const { return *this; }

    // Parallel information:
    //
    // numLayers:   the number of MLs in parallel
    // activeLayer: which ML is visible (-1 for overall parallel block)
                                                                
    virtual int numLayers  (void) const { return theML.size();                         }
    virtual int activeLayer(void) const { return mlind;                                }
    virtual int loffset    (void) const { return ( mlind < 0 ) ? 0 : resoffset(mlind); }

    // Serial modification
    //
    // addLayer:       add ML to position i in parallel
    // removeLayer:    remove ML from position i in parallel
    // setNumLayers:   set number of MLs in parallel
    // setActivelayer: set which ML is visible (-1 for overall parallel block)

    virtual int addLayer(int i);
    virtual int removeLayer(int i);
    virtual int setNumLayers(int n);
    virtual int setActiveLayer(int i);
    virtual int setloffset(int i);





private:

    // resoffset: controls where the output of each ML block is placed in
    // the output.  In general:
    //
    //  tspaceDim() == resoffset(n-1) + theML(n-1)->tspaceDim()
    //
    // where n is the number of blocks in parallel.  This is also used to
    // decode y when doing blockwise operations.

    Vector<int> resoffset;
    ParallBackcall datastore;

    // Inherited from ML_Mutable
    //
    // int mlType;
    // Vector<ML_Base *> theML;
    // int mlind;
};

inline void qswap(ML_Parall &a, ML_Parall &b)
{
    a.qswapinternal(b);

    return;
}

inline ML_Parall &setzero(ML_Parall &a)
{
    a.restart();

    return a;
}

inline void ML_Parall::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ML_Parall &b = dynamic_cast<ML_Parall &>(bb.getML());

    ML_Base::qswapinternal(b);

    qswap(resoffset,b.resoffset);
    //qswap(datastore,b.datastore); - deliberately *DO NOT* swap these.
    // all the datastore actually contains is a pointer back to this
    // parent.  Qswapping would only mess up these pointers!

    return;
}

inline void ML_Parall::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ML_Parall &b = dynamic_cast<const ML_Parall &>(bb.getMLconst());

    NiceAssert( theML.size() == (b.theML).size() );

    if ( theML.size() )
    {
        int i;

        for ( i = 0 ; i < theML.size() ; i++ )
        {
            (*((theML)("&",i))).semicopy((*((b.theML)(i))));
        }
    }

    //mlind
    //mlType
    //resoffset

    ML_Base::semicopy(b);

    return;
}

inline void ML_Parall::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ML_Parall &b = dynamic_cast<const ML_Parall &>(bb.getMLconst());

    resizetheML((b.theML).size());

    if ( theML.size() )
    {
        int i;

        for ( i = 0 ; i < theML.size() ; i++ )
        {
            (*((theML)("&",i))).assign((*((b.theML)(i))),onlySemiCopy);
        }
    }

    mlind     = b.mlind;
    mlType    = b.mlType;
    resoffset = b.resoffset;

    ML_Base::semicopy(b);

    return;
}


inline void qswap(ParallBackcall &a, ParallBackcall &b)
{
    a.qswapinternal(b);

    return;
}

inline void ParallBackcall::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ParallBackcall &b = dynamic_cast<ParallBackcall &>(bb.getML());

    ML_Base::qswapinternal(b);

    ML_Parall *temp;

    temp    = owner;
    owner   = b.owner;
    b.owner = temp;

    return;
}

inline void ParallBackcall::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ParallBackcall &b = dynamic_cast<const ParallBackcall &>(bb.getMLconst());

    ML_Base::semicopy(b);

    owner = b.owner;

    return;
}

inline void ParallBackcall::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ParallBackcall &src = dynamic_cast<const ParallBackcall &>(bb.getMLconst());

    ML_Base::assign(src,onlySemiCopy);

    owner = src.owner;

    return;
}

#endif


