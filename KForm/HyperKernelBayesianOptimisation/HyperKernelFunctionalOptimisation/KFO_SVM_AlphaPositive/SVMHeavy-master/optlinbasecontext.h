
//
// Linear optimisation context
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _optlinbasecontext_h
#define _optlinbasecontext_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "optcontext.h"

// Background
// ==========
//
// Given a matrix H, retain the maximally large factorisation
//
// chol(H(pivF,pivF))
//
// where row/columns can be set either free (F, included in factorisation)
// or zero(Z, not included in factorisation).
//
// Loosely follows optcontext


// Stream operators

class optLinBaseContext;

std::ostream &operator<<(std::ostream &output, const optLinBaseContext &src );
std::istream &operator>>(std::istream &input,        optLinBaseContext &dest);

// Swap function

inline void qswap(optLinBaseContext &a, optLinBaseContext &b);

class optLinBaseContext
{
    friend std::ostream &operator<<(std::ostream &output, const optLinBaseContext &src );
    friend std::istream &operator>>(std::istream &input,        optLinBaseContext &dest);

    friend inline void qswap(optLinBaseContext &a, optLinBaseContext &b);

public:

    // Constructors and assignment operators

    optLinBaseContext(void);
    optLinBaseContext(const optLinBaseContext &src);
    optLinBaseContext &operator=(const optLinBaseContext &src);

    // Reconstructors:

    void refact(const Matrix<double> &H, double xzt = -1) { Q.refact(H,Gn,Gpn,1,xzt); return; }
    void reset (const Matrix<double> &H)                  { Q.reset(H,Gn,Gpn);        return; }

    // Control functions using unpivotted index:

    int add(int i);
    int remove(int i);

    // Find position in pivotted variables

    int findInZ(int i) const { return Q.findInAlphaZ(i); }
    int findInF(int i) const { return Q.findInAlphaF(i); }

    // Variable state control using index to relevant pivot vector

    int modZtoF(int iP, const Matrix<double> &H) { return Q.modAlphaZtoUF(iP,H,Gn,Gpn,apos,bpos);  }
    int modFtoZ(int iP, const Matrix<double> &H) { return Q.modAlphaUFtoZ(iP,H,Gn,Gpn,apos,bpos);  }

    // Pivotting and constraint data

    const Vector<int> &pivZ  (void) const { return Q.pivAlphaZ();  }
    const Vector<int> &pivF  (void) const { return Q.pivAlphaF();  }
    const Vector<int> &hState(void) const { return Q.alphaState(); }

    // Information functions

    int NZ(void) const { return Q.aNZ();  }
    int NF(void) const { return Q.aNF();  }
    int N (void) const { return Q.aN();   }

    double zt(void) const { return Q.zt(); }

    // Factorisation functions
    //
    // rankone:    H := H + c.bp.bp'
    // diagmult:   diag(H) := bp.*diag(H)
    // diagoffset: diag(H) := diag(H) + bp
    //
    // minverse: solve H.ap = bp for ap
    //
    // pfact:  number of elements in pivF that are actually in the factorisation
    // nofact: returns 1 if factorised part of H is empty
    //
    // fudgeOn:  adds small diagonals when needed to ensure H has full size
    // fudgeOff: maximise size of H, but don't cheat by adding diagonals
    //
    // NB: fudgeOn is numerically dangerous - advise caution!

    void rankone   (const Vector<double> &bp, const double &c, const Matrix<double> &H) { Q.fact_rankone(bp,bn,c,H,Gn,Gpn,apos,bpos);  return; }
    void diagmult  (const Vector<double> &bp                                          ) { Q.fact_diagmult(bp,bn,apos,bpos);             return; }
    void diagoffset(const Vector<double> &bp,                  const Matrix<double> &H) { Q.fact_diagoffset(bp,bn,H,Gn,Gpn,apos,bpos); return; }

    int minverse(Vector<double> &ap, const Vector<double> &bp) const { Vector<double> an; return Q.fact_minverse(ap,an,bp,bn,Gn,Gpn); }

    int pfact (void) const { return Q.fact_pfact(Gn,Gpn);  }
    int nofact(void) const { return Q.fact_nofact(Gn,Gpn); }

    void fudgeOn (const Matrix<double> &H) { Q.fact_fudgeOn(H,Gn,Gpn,apos,bpos);  return; }
    void fudgeOff(const Matrix<double> &H) { Q.fact_fudgeOff(H,Gn,Gpn,apos,bpos); return; }

private:

    optContext Q;

    // Empty stuff we only want to have to allocate once

    Matrix<double> Gpn; // note that size is x*0
    Matrix<double> Gn;
    Vector<double> bn;
    int apos;
    int bpos;
};

inline void qswap(optLinBaseContext &a, optLinBaseContext &b)
{
    qswap(a.Q   ,b.Q   );
    qswap(a.Gpn ,b.Gpn );
    qswap(a.Gn  ,b.Gn  );
    qswap(a.bn  ,b.bn  );
    qswap(a.apos,b.apos);
    qswap(a.bpos,b.bpos);

    return;
}

#endif
