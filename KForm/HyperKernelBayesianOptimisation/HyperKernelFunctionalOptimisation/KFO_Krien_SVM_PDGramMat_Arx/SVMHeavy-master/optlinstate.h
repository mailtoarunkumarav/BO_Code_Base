
//
// Linear optimisation state
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _optlinstate_h
#define _optlinstate_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "optlincontext.h"
#include "optlinstate.h"

// Consider the linear programming problem:
//
// min_alpha c'.alpha
//
// Such that:
//
// e = G.alpha + g subject to eRestrict constraints
//
// - alphaRestrict is a vector controlling the range of alpha:
//
//    alphaRestrict[i] = 1: alpha[i] >= 0
//    alphaRestrict[i] = 3: alpha[i] == 0
//
// - eRestrict is a vector controlling the range of beta:
//
//    eRestrict[i] = 0: e[i] == 0
//    eRestrict[i] = 1: e[i] >= 0
//    eRestrict[i] = 2: e[i] <= 0
//    eRestrict[i] = 3: e[i] unrestricted
//
// NB: estate is 0/1, sign of e not taken into account
//
// Step: step direction is by projected gradient descent
//
// step_alpha_pivRowF = - ( ca - Gra'.inv(Gra.Gra').Gra.ca )
//                    = -( I - Gra'.inv(Gra.Gra').Gra ).ca
//                    = -( ca - Gra'.br )
//
// where:
//
//  - a = pivAlphaF
//  - r = pivRowF
//  - ca = c_pivAlphaF
//  - Gra = G_{pivRowF,pivAlphaF}
//  - Gra'.inv(Gra.Gra').Gra.ca = projection of ca perp to Gra.ca = 0
//  - ( ca - Gra'.inv(Gra.Gra').Gra.ca ) = "       parallel to Gra.ca = 0
//
// and:
//
//  - b = Lagrange multipliers for G.alpha + g constraints
//  - br = b_pivRowF
//  - br = inv(Gra.Gra').Gra.ca
//
// NB: each element br_i of br will tend to lead to a step in direction:
//
//       partstep_alpha_pivRowF = br_i*G_{pivRowF_i,pivAlphaF}'
//
//     which, if taken, will lead to a change in e_i of:
//
//       partstep_e_i = G_{pivRowF_i,pivAlphaF}.partstep_alpha_pivRowF
//                    = br_i.||G_{pivRowF_i,pivAlphaF}||^2
//
//     This step will take us into the feasible region if br_i > 0.  So
//     we see that an active constraint e_i = 0 is optimal if:
//
//       - br_i <= 0 and eRestrictr_i == 1,3
//       - br_i >= 0 and eRestrictr_i == 2,3
//
//     but should be relaxed if:
//
//       - br_i > 0 and eRestrictr_i == 1,3
//       - br_i < 0 and eRestrictr_i == 2,3
//
//     Active constraints e_i == 0 are in pivRowF
//     Inactive constraints are in pivRowZ
//
// Stored here are:
//
// - alpha: main variable
// - alphaGrad: see above alphaGrad' = ( c' - br'.Gr: )
// - e: G.alpha + g
// - b: br = inv(Gra.Gra').Gra.ca  (r = pivRowF, a = pivAlphaF)
//      bz = 0                     (z = pivRowZ)
// - alphaRestrict: see above
// - eRestrict: see above
// - optlincontext: see optcontext.h
// - opttol: tolerance for optimality conditions


// Stream operators

class optLinState;

std::ostream &operator<<(std::ostream &output, const optLinState &src );
std::istream &operator>>(std::istream &input,        optLinState &dest);

// Swap function

inline void qswap(optLinState &a, optLinState &b);


class optLinState
{
    friend std::ostream &operator<<(std::ostream &output, const optLinState &src );
    friend std::istream &operator>>(std::istream &input,        optLinState &dest);

    friend void qswap(optLinState &a, optLinState &b);

public:

    // Constructors and assignment operators
    //
    // Note that constructors assume that all alphas are constrained to zero.

    optLinState(void);
    optLinState(const optLinState &src);
    optLinState &operator=(const optLinState &src);

    // Set alpha

    void setAlpha(const Vector<double> &newAlpha, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    // Refactorisation / reset

    void refact   (const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
    void refactlin(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    void setopttol(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g, double opttol);
    void setzt    (const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g, double zt    );

    void reset(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    // Find position in pivotted variables

    int findInAlphaZ(int i) const { return Q.findInAlphaZ(i); }
    int findInAlphaF(int i) const { return Q.findInAlphaF(i); }

    int findInRowZ(int i) const { return Q.findInRowZ(i); }
    int findInRowF(int i) const { return Q.findInRowF(i); }

    // Scaling
    //
    // scale: update alpha *= a

    void scale(double a, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    // Control functions using unpivotted index:
    //
    // NB: remove alpha/row assumes constrained Z

    int addAlpha(int i, int alpharestrict);
    int removeAlpha(int i);

    int addRow(int i, int erestrict);
    int removeRow(int i);

    // Change alpha and beta restrictions, taking steps and constraining first if required

    void changeAlphaRestrict(int i, int alphrestrict, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
    void changeeRestrict    (int i, int betrestrict , const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    // Variable state control using index to relevant pivot vector

    int modAlphaZtoF(int iP, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
    int modAlphaFtoZ(int iP, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    int modRowZtoF(int iP, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
    int modRowFtoZ(int iP, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    // Pivotting

    const Vector<int> &pivAlphaZ(void) const { return Q.pivAlphaZ(); }
    const Vector<int> &pivAlphaF(void) const { return Q.pivAlphaF(); }

    const Vector<int> &alphaState(void) const { return Q.alphaState(); }

    const Vector<int> &pivRowZ(void) const { return Q.pivRowZ(); }
    const Vector<int> &pivRowF(void) const { return Q.pivRowF(); }

    const Vector<int> &rowState(void) const { return Q.rowState(); }

    // Information

    int aN (void) const { return Q.aN();  }
    int aNF(void) const { return Q.aNF(); }
    int aNZ(void) const { return Q.aNZ(); }

    int eN (void) const { return Q.rowN();  }
    int eNF(void) const { return Q.rowNF(); }
    int eNZ(void) const { return Q.rowNZ(); }

    double zerotol(void) const { return Q.zt(); }
    double opttol (void) const { return dopttol; }

    // Access to contents

    const Vector<double> &alpha(void) const { return dalpha; }
    const Vector<double> &e    (void) const { return de;     }

    const Vector<double> &alphaGrad(void) const { return dalphaGrad; }
    const Vector<double> &b        (void) const { return db;         }

    const Vector<int> &alphaRestrict(void) const { return dalphaRestrict; }
    const Vector<int> &rowRestrict  (void) const { return deRestrict;     }

    // Scale step
    //
    // scaleFStep: scale step to fit bounds and restrictions on alpha and e.
    //             returns 0 if step is infeasible or 1 if it does.
    //                     (usually 1, 0 if step has magnitude zero)
    //             scale is set to the scale used, and the step is scaled.
    //             alphaFIndex is set to the free alpha index that hits bounds (-1 if none)
    //             eFIndex is set to the free e index that hits bounds (-1 if none)
    //             alphaFStep should be of size aNF() (ie corresponse to alpha_pivAlphaF)

    int scaleFStep(double &scale, int &alphaFIndex, int &eFIndex, const Vector<double> &alphaFStep, const Vector<double> &stepde, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    // Take step
    //
    // stepalpha: single step in free alpha(i)
    // stepFGeneral: step in all free variables alpha(pivAlphaF()(0,1,asize-1))

    void stepalpha(int i, double alphaStep, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
    void stepFGeneral(const Vector<double> &alphaFStep, const Vector<double> &stepde, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    // Calculate step direction
    //
    // This is just the negative projected gradient as described previously.
    // You will need to call scaleFStep to finish calculating the step.

    void calcStep(Vector<double> &stepAlpha, Vector<double> &stepde, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    // Calculate least optimal active constraint
    //
    // alphaZIndex: >= 0 if this alpha index is most non-optimal
    // eFIndex: >= 0 if this row constraint is most non-optimal
    //
    // Return 0 if non-optimal gradient found or 1 if solution optimal to
    // within given tolerance.

    int maxGradNonOpt(int &alphaZIndex, int &eFIndex, double &gradmag, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    // Factorisation fudging

    int rowpfact (void) const { return Q.rowpfact();  }
    int rownofact(void) const { return Q.rownofact(); }

    void fudgeOn (const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
    void fudgeOff(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

private:

    Vector<double> dalpha;
    Vector<double> dalphaGrad;
    Vector<double> de;
    Vector<double> db;
    Vector<double> btemp;

    Vector<int> dalphaRestrict;
    Vector<int> deRestrict;

    double dopttol;
    optLinContext Q;

    // alphaGradBad is used to record alphaGrads that haven't been filled in yet.
    // eBad is used to record de that haven't been filled in yet.

    Vector<int> alphaGradBad;
    Vector<int> eBad;

    void fixalphaGradBad(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
    void recalcAlphaGrad(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
    void recalcAlphaGrad(int i, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    void fixeBad(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
    void recalce(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
    void recalce(int i, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);

    void recalcb(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g);
};

inline void qswap(optLinState &a, optLinState &b)
{
    qswap(a.dalpha        ,b.dalpha        );
    qswap(a.dalphaGrad    ,b.dalphaGrad    );
    qswap(a.de            ,b.de            );
    qswap(a.db            ,b.db            );
    qswap(a.btemp         ,b.btemp         );
    qswap(a.dalphaRestrict,b.dalphaRestrict);
    qswap(a.deRestrict    ,b.deRestrict    );
    qswap(a.dopttol       ,b.dopttol       );
    qswap(a.Q             ,b.Q             );
    qswap(a.alphaGradBad  ,b.alphaGradBad  );
    qswap(a.eBad          ,b.eBad          );

    return;
}

#endif
