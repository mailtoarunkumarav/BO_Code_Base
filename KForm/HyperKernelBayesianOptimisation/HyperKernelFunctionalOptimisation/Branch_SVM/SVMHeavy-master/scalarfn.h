
//
// Scalar function type
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _scalarfn_h
#define _scalarfn_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "gentype.h"

#define DEFAULTVARI 0
#define DEFAULTVARJ 0

#define DEFAULT_INTEGRAL_SLICES 100


//
// This is basically a limited version of gentype that assumes everything is
// a scalar function from the real interval [0,1] to reals.  Also has inner
// products etc defined.
//


class scalarfn;


// Swap function

inline void qswap(scalarfn &a, scalarfn &b);

class scalarfn : public gentype
{
    friend inline void qswap(scalarfn &a, scalarfn &b);

public:

    // Constructors, destructor and assignment
    //
    // - Default is real valued 0.0.
    // - Same rules for equation parsing apply here as do in gentype.
    // - Many functions are inheritted unchanged.
    // - By default the "evaluation variable" is x = var(0,0).  This can
    //   be changed.  It is this variable that is used for inner product
    //   and norm evaluation.
    // - The default number of points used to approximate inner product
    //   and norm can also be changed.

    scalarfn();
    scalarfn(const scalarfn       &src);
    scalarfn(const gentype        &src);
    scalarfn(const double         &src);
    scalarfn(const Vector<double> &src);
    scalarfn(const std::string    &src);
    scalarfn(const char           *src);

    scalarfn &operator=(const scalarfn       &src);
    scalarfn &operator=(const gentype        &src);
    scalarfn &operator=(const double         &src);
    scalarfn &operator=(const Vector<double> &src);
    scalarfn &operator=(const std::string    &src);
    scalarfn &operator=(const char           *src);

    // Evaluate equation
    //
    // The single-variable form is the only "well defined" one (though you
    // can technically use the others).  The variable x by default maps to
    // var(0,0) as per gentype, but this behaviour can be changed.
    //
    // - Vectors are interpretted a little differently.  It is assumed that
    //   they represent an even distribution of evaluations of the function
    //   over the interval [0,1], and are interpretted thus.  So for example
    //   if scalarfn f is a vector then f(0.5) will evaluate the element in
    //   the vector closest to the "centre" of the vector.

    scalarfn operator()(const double   &x) const;
    scalarfn operator()(const scalarfn &x) const;

    // Other stuff
    //
    // - ident sets this = 1.0
    // - zero sets this = 0.0
    // - setvarid: sets what variable "x" is (default var(0,0))
    // - setpoints: sets number of points used to approximate inner product
    //   and norms etc.

    scalarfn &ident(void);
    scalarfn &zero(void);

    int substitute(const double   &x);
    int substitute(const scalarfn &x);

    void setvarid(int i, int j);
    void getvarid(int &i, int &j) const;

    void sernumpts(int numpts);
    void getnumpts(int &numpts) const;

private:

    // Identify which variable "x" is

    int varid_xi;
    int varid_xj;

    // Number of points taken for norm and inner products

    int numpts;
};

inline void qswap(scalarfn &a, scalarfn &b)
{
    qswap(a.varid_xi,b.varid_xi);
    qswap(a.varid_xj,b.varid_xj);
    qswap(a.numpts  ,b.numpts  );

    return;
}

inline void innerProduct       (scalarfn &res, const scalarfn &a, const scalarfn &b);
inline void innerProductNoConj (scalarfn &res, const scalarfn &a, const scalarfn &b);
inline void innerProductRevConj(scalarfn &res, const scalarfn &a, const scalarfn &b);

scalarfn &setident(scalarfn &a);
scalarfn &setzero (scalarfn &a);

scalarfn abs   (const scalarfn &a);
scalarfn abs1  (const scalarfn &a);
scalarfn absp  (const scalarfn &a, const scalarfn &q);
scalarfn absp  (const scalarfn &a, const double   &q);
scalarfn absinf(const scalarfn &a);
scalarfn norm  (const scalarfn &a);
scalarfn norm1 (const scalarfn &a);
scalarfn normp (const scalarfn &a, const scalarfn &q);
scalarfn normp (const scalarfn &a, const double   &q);



#endif

