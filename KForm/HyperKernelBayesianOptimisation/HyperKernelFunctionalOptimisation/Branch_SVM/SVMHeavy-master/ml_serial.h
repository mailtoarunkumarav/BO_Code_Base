//FIXME: adding data should add to all blocks
//FIXME: removing data should remove from all blocks
//FIXME: see ml_serblk for how that should work (ie calculation of intermediate vectors and targets

//
// Serial ML module
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _ml_serial_h
#define _ml_serial_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_mutable.h"



class ML_Serial;


class SerialBackcall;

ML_Base *makeMLSerial(void);
void assigntoMLSerial(ML_Base **dest, const ML_Base *src, int onlySemiCopy);
void xferMLSerial(ML_Base &dest, ML_Base &src);

inline void qswap(SerialBackcall &a, SerialBackcall &b);

class SerialBackcall : public BLK_Nopnop
{
public:

    // Constructors etc

    SerialBackcall(ML_Serial *xowner = NULL);
    SerialBackcall(ML_Serial *xowner, const SerialBackcall &src);
    SerialBackcall(ML_Serial *xowner, const SerialBackcall &src, const ML_Base *srcx);
    SerialBackcall &operator=(const SerialBackcall &src) { assign(src); return *this; }
    virtual ~SerialBackcall();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual       ML_Base &getML     (void)       { return *this; }
    virtual const ML_Base &getMLconst(void) const { return *this; }

    // Information functions (training data):

    virtual int type(void)    const { return -2; }
    virtual int subtype(void) const { return 0;  }

    virtual int isTrained(void) const;

    virtual int tspaceDim(void)    const;
    virtual int xspaceDim(void)    const;
    virtual int tspaceSparse(void) const;
    virtual int xspaceSparse(void) const;
    virtual int numClasses(void)   const;
    virtual int order(void)        const;

    virtual char gOutType(void) const;
    virtual char hOutType(void) const;
    virtual char targType(void) const;
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int isUnderlyingScalar(void) const { return 0; }
    virtual int isUnderlyingVector(void) const { return 1; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual const Vector<int> &ClassLabels(void)   const;
    virtual int getInternalClass(const gentype &y) const;
    virtual int numInternalClasses(void)           const;

    virtual double sparlvl(void) const;

    virtual int isClassifier(void) const;
    virtual int isRegression(void) const;

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

    virtual int ggTrainingVector(double &resg,         int i, int retaltg = 0) const;
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0) const;
    virtual int ggTrainingVector(d_anion &resg,        int i, int retaltg = 0) const;

    virtual void dgTrainingVector(SparseVector<gentype> &resx, int i) const;

private:

    // This class always requires a reference to its "owner".
    // NB: the ONLY variable must be a pointer to the owner

    ML_Serial *owner;
};





std::ostream &operator<<(std::ostream &output, const ML_Serial &src );
std::istream &operator>>(std::istream &input,        ML_Serial &dest);



// Swap and zeroing (restarting) functions

inline void qswap(ML_Serial &a, ML_Serial &b);

class ML_Serial : public ML_Mutable
{
    friend class SerialBackcall;

public:

    // Mutation functions

    virtual void setMLTypeMorph(int newmlType) { if ( type() >= 0 ) { ML_Mutable::setMLTypeMorph(newmlType); } else { dynamic_cast<ML_Mutable &>(*(theML(mlind))).setMLTypeMorph(newmlType); } return; }
    virtual void setMLTypeClean(int newmlType) { if ( type() >= 0 ) { ML_Mutable::setMLTypeClean(newmlType); } else { dynamic_cast<ML_Mutable &>(*(theML(mlind))).setMLTypeClean(newmlType); } return; }

    // Constructors, destructors, assignment etc..

    ML_Serial();
    ML_Serial(const ML_Serial &src);
    ML_Serial(const ML_Serial &src, const ML_Base *srcx);
    ML_Serial &operator=(const ML_Serial &src) { assign(src); return *this; }
    virtual ~ML_Serial();

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

    // Serial information:
    //
    // numLayers:   the number of MLs in series
    // activeLayer: which ML is visible (-1 for overall serial block)
                                                                
    virtual int numLayers  (void) const { return theML.size(); }
    virtual int activeLayer(void) const { return mlind;        }

    // Serial modification
    //
    // addLayer:       add ML to position i in serial
    // removeLayer:    remove ML from position i in serial
    // setNumLayers:   set number of MLs in serial
    // setActivelayer: set which ML is visible (-1 for overall serial block)

    virtual int addLayer(int i);
    virtual int removeLayer(int i);
    virtual int setNumLayers(int n);
    virtual int setActiveLayer(int i);





private:

    SerialBackcall datastore;

    // Inherited from ML_Mutable
    //
    // int mlType;
    // Vector<ML_Base *> theML;
    // int mlind;

    // These functions return either the ML i or the block if i == -1

    const ML_Mutable &firstMLconst(void) const { return dynamic_cast<const ML_Mutable &>((*(theML(0            )))); }
    const ML_Mutable &lastMLconst (void) const { return dynamic_cast<const ML_Mutable &>((*(theML(numLayers()-1)))); }

    ML_Mutable &firstML(void) { return dynamic_cast<ML_Mutable &>((*(theML("&",0            )))); }
    ML_Mutable &lastML (void) { return dynamic_cast<ML_Mutable &>((*(theML("&",numLayers()-1)))); }
};

inline void qswap(ML_Serial &a, ML_Serial &b)
{
    a.qswapinternal(b);

    return;
}

inline ML_Serial &setzero(ML_Serial &a)
{
    a.restart();

    return a;
}

inline void ML_Serial::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ML_Serial &b = dynamic_cast<ML_Serial &>(bb.getML());

    ML_Base::qswapinternal(b);


    //qswap(datastore,b.datastore); - deliberately *DO NOT* swap these.
    // all the datastore actually contains is a pointer back to this
    // parent.  Qswapping would only mess up these pointers!

    return;
}

inline void ML_Serial::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ML_Serial &b = dynamic_cast<const ML_Serial &>(bb.getMLconst());

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


    ML_Base::semicopy(b);

    return;
}

inline void ML_Serial::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ML_Serial &b = dynamic_cast<const ML_Serial &>(bb.getMLconst());

    resizetheML((b.theML).size());

    if ( theML.size() )
    {
        int i;

        for ( i = 0 ; i < theML.size() ; i++ )
        {
            (*((theML)("&",i))).assign((*((b.theML)(i))),onlySemiCopy);
        }
    }

    mlind  = b.mlind;
    mlType = b.mlType;


    ML_Base::semicopy(b);

    return;
}


inline void qswap(SerialBackcall &a, SerialBackcall &b)
{
    a.qswapinternal(b);

    return;
}

inline void SerialBackcall::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SerialBackcall &b = dynamic_cast<SerialBackcall &>(bb.getML());

    ML_Base::qswapinternal(b);

    ML_Serial *temp;

    temp    = owner;
    owner   = b.owner;
    b.owner = temp;

    return;
}

inline void SerialBackcall::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SerialBackcall &b = dynamic_cast<const SerialBackcall &>(bb.getMLconst());

    ML_Base::semicopy(b);

    owner = b.owner;

    return;
}

inline void SerialBackcall::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SerialBackcall &src = dynamic_cast<const SerialBackcall &>(bb.getMLconst());

    ML_Base::assign(src,onlySemiCopy);

    owner = src.owner;

    return;
}

#endif


