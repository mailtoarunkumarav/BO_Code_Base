
//
// Linear optimisation context
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _optlincontext_h
#define _optlincontext_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "optlinbasecontext.h"

// Background
// ==========
//
// Given a matrix G, retain the maximally large factorisation
//
// chol(G(pivR,pivAlpha).G(pivR,pivAlpha)')
//
// Rows and columns can be separately added to and removed from G.
//
// rowState: 0 = row not in factorisation (constraint inactive)
//           1 = row in factorisation (constraint active)
//
// alphaState: 0 = alpha actively constrained at zero
//             1 = alpha free to be >= 0


// Stream operators

class optLinContext;

std::ostream &operator<<(std::ostream &output, const optLinContext &src );
std::istream &operator>>(std::istream &input,        optLinContext &dest);

// Swap function

inline void qswap(optLinContext &a, optLinContext &b);

class optLinContext
{
    friend std::ostream &operator<<(std::ostream &output, const optLinContext &src );
    friend std::istream &operator>>(std::istream &input,        optLinContext &dest);

    friend inline void qswap(optLinContext &a, optLinContext &b);

public:

    // Constructors and assignment operators

    optLinContext(void);
    optLinContext(const optLinContext &src);
    optLinContext &operator=(const optLinContext &src);

    // Reconstructors:
    //
    // refact: does not change pivoting
    // reset:  constrains all to zero

    void refact(const Matrix<double> &G, double xzt = -1);
    void reset (const Matrix<double> &G);

    // Control functions using unpivotted index:
    //
    // add row: G = [ Ga ] -> G = [ Ga  ]
    //              [ Gb ]        [ ... ]
    //                            [ Gb  ]
    // add alpha (col): G = [ Ga Ga ] -> G = [ Ga ... Gb ]
    // remove ...: reverse of above
    //
    // where it is assumed that the row/column is constrained zero

    int addRow   (int i);
    int removeRow(int i);

    int addAlpha   (int i);
    int removeAlpha(int i);

    // Find position in pivotted variables

    int findInRowZ(int i) const;
    int findInRowF(int i) const;

    int findInAlphaZ(int i) const;
    int findInAlphaF(int i) const;

    // Variable state control using index to relevant pivot vector

    int modRowZtoF(int iP, const Matrix<double> &G);
    int modRowFtoZ(int iP, const Matrix<double> &G);

    int modAlphaZtoF(int iP, const Matrix<double> &G);
    int modAlphaFtoZ(int iP, const Matrix<double> &G);

    // Pivotting and constraint data

    const Vector<int> &pivRowZ(void) const { return Q.pivZ(); }
    const Vector<int> &pivRowF(void) const { return Q.pivF(); }

    const Vector<int> &rowState(void) const { return Q.hState();  }

    const Vector<int> &pivAlphaZ(void) const { return xpivAlphaZ; }
    const Vector<int> &pivAlphaF(void) const { return xpivAlphaF; }

    const Vector<int> &alphaState(void) const { return xalphaState; }

    // Information functions

    int rowNZ(void) const { return Q.NZ(); }
    int rowNF(void) const { return Q.NF(); }
    int rowN (void) const { return Q.N (); }

    int aNZ(void) const { return xpivAlphaZ.size(); }
    int aNF(void) const { return xpivAlphaF.size(); }
    int aN (void) const { return aNZ()+aNF();       }

    double zt(void) const { return Q.zt(); }

    // Factorisation functions
    //
    // minverse: ap(pivRowF) = inv(G(pivRowF,pivAlphaF).G(pivRowF,pivAlphaF)').bp(pivRowF)
    // project: ap(pivAlphaF) = G(pivRowF,pivAlphaF)'.inv(G(pivRowF,pivAlphaF).G(pivRowF,pivAlphaF)').G(pivRowF,pivAlphaF).bp
    //          where pivRowF will be trimmed to size of factorisation and
    //          ap(pivAlphaZ) = not set.  Returns size of factorisation

    int minverse(Vector<double> &ap, const Vector<double> &bp) const;
    int project (Vector<double> &ap, const Vector<double> &bp, const Matrix<double> &G);

    int rowpfact (void) const { return Q.pfact();  }
    int rownofact(void) const { return Q.nofact(); }

    void fudgeOn (void) { Q.fudgeOn(H);  return; }
    void fudgeOff(void) { Q.fudgeOff(H); return; }

private:

    optLinBaseContext Q;
    Matrix<double> H;
    Vector<int> xpivAlphaF;
    Vector<int> xpivAlphaZ;
    Vector<int> xalphaState;
    Vector<double> rtempa;
    Vector<double> rtempb;

    // When you call addRow and removeRow G cannot be passed, so there will
    // be blank left in H matrix.  These are recorded and the following fn
    // can be called later to fill them in.

    Vector<int> Hblanks;

    void fixH(const Matrix<double> &G);
};

inline void qswap(optLinContext &a, optLinContext &b)
{
    qswap(a.Q          ,b.Q          );
    qswap(a.H          ,b.H          );
    qswap(a.xpivAlphaF ,b.xpivAlphaF );
    qswap(a.xpivAlphaZ ,b.xpivAlphaZ );
    qswap(a.xalphaState,b.xalphaState);
    qswap(a.Hblanks    ,b.Hblanks    );
    qswap(a.rtempa     ,b.rtempa     );
    qswap(a.rtempb     ,b.rtempb     );

    return;
}

#endif
