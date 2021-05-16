
//
// Quadratic optimisation context
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _optcontext_h
#define _optcontext_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "vector.h"
#include "matrix.h"
#include "chol.h"
#include "numbase.h"

// Background
// ==========
//
// Consider the quadratic programming problem:
//
// [ alpha ]' [ Gp   Gpn ] [ alpha ] + [ alpha ]' [ gp ] + | alpha' |' [ hp ]
// [ beta  ]  [ Gpn' Gn  ] [ beta  ]   [ beta  ]  [ gn ]   | beta   |  [ 0  ]
//
// where alpha \in \Re^{aN} and beta \in \Re^{bN}.  It is assumed that:
//
// - Gp is positive semi-definite hermitian
// - Gn is negative semi-definite hermitian
//
// where each variable is in one of the following states:
//
// - alpha[i] = lb[i]         => sgn(alpha[i]) = -1 (constrained)
// - lb[i] <= alpha[i] <= 0   => sgn(alpha[i]) = -1 (free)
// - alpha[i] = 0             => sgn(alpha[i]) = 0  (constrained)
// - 0 <= alpha[i] <= ub[i]   => sgn(alpha[i]) = +1 (free)
// - alpha[i] = ub[i]         => sgn(alpha[i]) = +1 (constrained)
//
// - beta[i] = 0  (constrained)
// - beta[i] free (free)
//
// The positive and negative states of alpha are differentiated to allow for
// the hp term, which presents a step discontinuity in the gradient at
// alpha[i] = 0.  Pivotting identifies these states: specifically, we have the
// following pivot vectors (integer vectors):
//
// - pAlphaLB  s.t. alpha[pAlphaLB] == lb[pAlphaLB]      (constrained)
// - pAlphaZ   s.t. alpha[pAlphaZ] == 0                  (constrained)
// - pAlphaUB  s.t. alpha[pAlphaUB] == ub[pAlphaUB]      (constrained)
// - pAlphaF   s.t. lb[pAlphaLF] <= alpha[pAlphaLF] <= 0 (free)
//               or 0 <= alpha[pAlphaUF] <= ub[pAlphaUF] (free)
//
// - pBetaC s.t. beta[pBetaC] = 0  (constrained)
// - pBetaF s.t. beta[pBetaF] free (free)
//
// where every 0 <= i <= aN-1 is located in precisely one of the pAlpha* pivot
// vectors, and likewise every 0 <= j <= bN-1 is located in precisely one of
// the pBeta* pivot vectors.  We also define
//
// - dalphaState is a vector recording the state of alpha:
//
//    dalphaState[i] = -2: alpha[i] == lb[i]         (dalphaState[pAlphaLB] == -2)
//    dalphaState[i] = -1: lb[i] <= alpha[i] <= 0    (dalphaState[pAlphaF]  == +-1)
//    dalphaState[i] =  0: alpha[i] == 0             (dalphaState[pAlphaZ]  == 0)
//    dalphaState[i] = +1: 0 <= alpha[i] <= ub[i]    (dalphaState[pAlphaF]  == +-1)
//    dalphaState[i] = +2: alpha[i] == ub[i]         (dalphaState[pAlphaLB] == +2)
//
// - dbetaState is a vector recording the state of beta:
//
//    dbetaState[i] = 0: beta[i] == 0           (dbetaState[pBetaC] = 0)
//    dbetaState[i] = 1: beta[i] unconstrained  (dbetaState[pBetaF] = 1)
//
// The task of this class is to keep track of all the relevant pivot vectors
// and also maintain a (part) cholesky factorisation of the active part of
// the hessian if required, where the active part of the hessian is:
//
// [ Gp[pAlphaF,pAlphaF]   Gpn[pAlphaF,pBetaF] ]
// [ Gpn[pAlphaF,pBetaF]'  Gn[pBetaF,pBetaF]   ]
//
//
// Cholesky Factorisation
// ======================
//
// If keepfact is set then the class also maintains the cholesky factorisation,
// as stated above.  This is done using the chol.h class.  All of the pAlphaF
// components are placed into this class as well as the first nfact pBetaF
// elements.  The precise order in which the elements are placed into the
// factorisation (the interleaving) is controlled by the D vector:
//
// - D is the same size as the factorisation, namely nfact+size(pAlphaF).
// - D[i] == +1 implies that a row/column from Gp[pAlphaF,pAlphaf] should
//   be inserted at row/column i of the factorisation
// - D[i] == -1 implies that a row/column from Gn[pBetaF,pBetaF] should
//   be inserted at row/column i of the factorisation
//
// We also maintain the factor vectors fAlphaF and fBetaF as follows:
//
// - fAlphaF[i] = the position of the Gp[pAlphaF,pAlphaF][i,i] in the
//   factorisation
// - fBetaF[i] = the position of the Gn[pBetaF,pBetaF][i,i] in the
//   factorisation, or -1 if it is not included in the factorisation.
//
// Where we note that:
//
// - D[fAlphaF[i]] = +1
// - D[fBetaF[i]]  = -1
// - fAlphaF[i] < fAlphaF[i+1]
// - fBetaF[i]  < fBetaF[i+1] for all i : fBetaF[i] != -1 and fBetaF[i+1] != -1
//
// General rules for the factorisation:
//
// - the first nfact elements of pBetaF are included in the non-singular part
//   of the factorisation and hence have fBetaF[i] != -1 (i < nfact).
// - the remainder of elements of pBetaF are not included in the factorisation
//   and hence have fBetaF[i] == -1 (i >= nfact).
// - the first pfact elements of pAlphaF are included in the non-singular part
//   of the factorisation.
// - the remainder of elements of pAlphaF are in the singular part of the
//   factorisation.
//
// The ordering of the elements in pBetaF is set to maximise the number of
// elements (nfact) there are in the factorisation.  Ditto the elements in
// pAlphaF.
//
// When elements have been added to or removed from the factorisation or
// elements have been added to pBetaF but not the factorisation then the
// algorithm to update the factorisation is:
//
// 1. If there are elements i such that fBetaF[i] != -1 is in the singular
//    part of the factorisation then remove them.
// 2. If the non-singular part of the factorisation is empty or consists
//    entirely of elements from Gn then:
//    a. add as many elements of Gp to non-singular part of factorisation as
//       possible.
//    b. add as many elements of Gn to non-singular part of factorisation as
//       possible.
//    c. goto a until no more progress can be made.
//
// Finally, betaFix is defined as follows:
//
// - betaFix[i] == 0 means that beta[i] may be included in the factorisation
//   if unconstrained
// - betaFix[i] == 1 means that beta[i] cannot be included in the factorisation
//   even if it is not constrained.
// - betaFix[i] == -1 means that the state is unknown.
//
// This variable is intended to speed things up for those betas corresponding
// to all zero columns in the Gpn matrix which therefore cannot be successfully
// included in the factorisation, so there is no need to waste time trying.
// betaFix[i] == 0 unless otherwise set.


// Stream operators

class optContext;

std::ostream &operator<<(std::ostream &output, const optContext &src );
std::istream &operator>>(std::istream &input,        optContext &dest);

// Swap function

inline void qswap(optContext &a, optContext &b);

class optContext
{
    friend std::ostream &operator<<(std::ostream &output, const optContext &src );
    friend std::istream &operator>>(std::istream &input,        optContext &dest);

    friend inline void qswap(optContext &a, optContext &b);

public:

    // Constructors and assignment operators
    //
    // Note that constructors assume that all alphas are constrained to zero
    // and that all betas are constrained.

    optContext(void);
    optContext(const optContext &src);
    optContext &operator=(const optContext &src);

    // Reconstructors:
    //
    // refact: like the constructor, but keeps the existing alpha LB/LF/Z/UF/UB
    //         and beta F/C information.
    // reset:  using modAlphaxxtoZ and modBetaFtoC functions to put all alpha
    //         in Z and all beta constrained.
    //
    // keepfact = 0: do not maintain the cholesky factorisation.
    //            1: do maintain the cholesky factorisation.
    // zt = zero tolerance for cholesky factorisation.

    void refact(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int xkeepfact = -1, double xzt = -1);
    void reset (const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn);

    // Control functions using unpivotted index:
    //
    // addAlpha: add alpha[i], presumed constrained to zero
    // addBeta:  add beta[i],  presumed constrained to zero
    //
    // removeAlpha: remove alpha[i], presumed constrained to zero
    // removeBeta:  remove beta[i],  presumed constrained to zero
    //
    // Notes:
    //
    // - addAlpha and addBeta return the new position of the variable in the
    //   relevant pivot vector (either pAlphaZ or pBetaC)
    // - removeAlpha and removeBeta return the old position of the variable in
    //   the relevant pivot vector (either pAlphaZ or pBetaC)

    int addAlpha(int i);
    int addBeta (int i);

    int removeAlpha(int i);
    int removeBeta (int i);

    // Find position in pivotted variables

    int findInAlphaLB(int i) const;
    int findInAlphaZ (int i) const;
    int findInAlphaUB(int i) const;
    int findInAlphaF (int i) const;

    int findInBetaC(int i) const;
    int findInBetaF(int i) const;

    // Variable state control using index to relevant pivot vector
    //
    // modAlpha*to#: remove variable pAlpha*[iP] and put it in pAlpha#
    // modBeta*to#:  remove variable pBeta*[iP]  and put it in pBeta#
    //
    // Notes:
    //
    // - Each function returns the new position in the relevant pivot vector
    //   (either pAlphaLB, pAlphaZ, pAlphaUB, pAlphaF, pBetaC, or pBetaF).
    // - pAlphaF may change if all elements of Gp and Gn are zero before or
    //   after the operation.
    // - Because the ordering of variables in pAlphaF and pBetaF may change
    //   we include arguments apos and bpos.  Basically if point apos in
    //   pAlphaF moves then the function will change apos to the position
    //   to which it has moved.  Ditto bpos and pBetaF.

    int modAlphaLBtoLF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaLBtoZ (int iP);
    int modAlphaLBtoUF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaLBtoUB(int iP);
    int modAlphaLFtoLB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaLFtoZ (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaLFtoUF(int iP);
    int modAlphaLFtoUB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaZtoLB (int iP);
    int modAlphaZtoLF (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaZtoUF (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaZtoUB (int iP);
    int modAlphaUFtoLB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaUFtoLF(int iP);
    int modAlphaUFtoZ (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaUFtoUB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaUBtoLB(int iP);
    int modAlphaUBtoLF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modAlphaUBtoZ (int iP);
    int modAlphaUBtoUF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);

    int modBetaCtoF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int modBetaFtoC(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);

    // Batch variable state control
    //
    // modAlphaFAlltoLowerBound: remove all variables from pAlphaLF and put in pAlphaLB
    //                           remove all variables from pAlphaUF and put in pAlphaZ
    // modAlphaFAlltoUpperBound: remove all variables from pAlphaLF and put in pAlphaZ
    //                           remove all variables from pAlphaUF and put in pAlphaUB

    void modAlphaFAlltoLowerBound(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    void modAlphaFAlltoUpperBound(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);

    // Pivotting and constraint data

    const Vector<int> &pivAlphaLB(void) const { return pAlphaLB; }
    const Vector<int> &pivAlphaZ (void) const { return pAlphaZ;  }
    const Vector<int> &pivAlphaUB(void) const { return pAlphaUB; }
    const Vector<int> &pivAlphaF (void) const { return pAlphaF;  }

    const Vector<int> &pivBetaC(void) const { return pBetaC; }
    const Vector<int> &pivBetaF(void) const { return pBetaF; }

    const Vector<int> &alphaState(void) const { return dalphaState; }
    const Vector<int> &betaState(void)  const { return dbetaState;  }

    // Information functions

    int aNLB(void) const { return pAlphaLB.size();     }
    int aNLF(void) const { return daNLF;               }
    int aNZ (void) const { return pAlphaZ.size();      }
    int aNUF(void) const { return daNUF;               }
    int aNUB(void) const { return pAlphaUB.size();     }
    int aNF (void) const { return pAlphaF.size();      }
    int aNC (void) const { return aNLB()+aNZ()+aNUB(); }
    int aN  (void) const { return aNF()+aNC();         }

    int bNF(void) const { return pBetaF.size(); }
    int bNC(void) const { return pBetaC.size(); }
    int bN (void) const { return bNF()+bNC();   }

    int keepfact(void) const { return dkeepfact; }
    double zt(void)    const { return dzt;       }

    // Factorisation functions
    //
    // These basically reflect the functions of chol.h, but take care of all the
    // requisit matrix pivotting.  That is:
    //
    // - bp is replaced by bp(pAlphaF)
    // - bn is replaced by bn(pBetaF)(0,1,nfact-1)
    // - Gp is replaced by Gp(pAlphaF,pAlphaF)
    // - Gn is replaced by Gn(pBetaF,pBetaF)(0,1,nfact-1,0,1,nfact-1)
    // - Gpn is replaced by Gpn(pAlphaF,pBetaF)(0,1,pAlphaF.size()-1,0,1,nfact-1)
    //
    // When calling fact_minverse and near_invert the following pivots are
    // also used:
    //
    // - ap is replaced by ap("&",pAlphaF)("&",0,1,fact.npos()-fact.nbadpos()-1)
    // - an is replaced by an("&",pBetaF)("&",0,1,nfact-1)
    // - bp is replaced by bp("&",pAlphaF)("&",0,1,fact.npos()-fact.nbadpos()-1)
    // - bn is replaced by bn("&",pBetaF)("&",0,1,nfact-1)
    //
    // and moreover when returning from these:
    //
    // - ap("&",pAlphaF)("&",fact.npos()-fact.nbadpos(),1,pAlphaF.size()-1) = 0
    // - an("&",pBetaF)("&",nfact,1,pBetaF.size()-1) = 0
    //
    // Notes:
    //
    // - When calling rank-one, bn is assumed to be zero.
    // - Because the ordering of variables in pAlphaF and pBetaF may change
    //   we include arguments apos and bpos.  Basically if point apos in
    //   pAlphaF moves then the function will change apos to the position
    //   to which it has moved.  Ditto bpos and pBetaF.
    // - bnZero = 1 tells the code to assume that bn is zero, bnZero = 0
    //              tells it to assume that bn is nonzero.
    // - bpNZ == -2 tells the code to assume that all of bp is nonzero.
    //   bpNZ == -1 tells the code to assume that all of bp is zero
    //   bpNZ >= 0  tells the code to assume that bp(pAlphaF)(0,1,fact.npos()-fact.nbadpos()-1)
    //              is zero except for one element, namely bp(pAlphaF)(bpNZ).
    // - bnNZ == -2 tells the code to assume that all of bn is nonzero.
    //   bnNZ == -1 tells the code to assume that all of bn is zero
    //   bnNZ >= 0  tells the code to assume that bn(pBetaF)(0,1,nfact-1)
    //              is zero except for one element, namely bn(pBetaF)(bnNZ).
    // - rankone and diagmult can both be called even if there is no factorisation.
    // - fact_minverse returns pfact+nfact (ie the size of the inverted hessian).
    // - There is a special case here.  If Gp == Gn == 0 then no factorisation is
    //   possible, but nevertheless if Gpn is nonzero then it may be possible to
    //   invert part of all of the active hessian in the form:
    //
    //    inv([  0      Gpnx  ]) = [  0          inv(Gpnx)'  ]
    //        [  Gpnx'  0     ]    [  inv(Gpnx)  0           ]
    //
    //    where Gpnx here represents the largest part of Gpn that is (a) square and
    //    (b) invertible.  In this case pfact() and nfact() will both return the
    //    size of Gpnx and fact_nofact returns true.

    void fact_rankone   (const Vector<double> &bp, const Vector<double> &bn, const double &c, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    void fact_diagmult  (const Vector<double> &bp, const Vector<double> &bn,                                                                                                 int &apos, int &bpos);
    void fact_diagoffset(const Vector<double> &bp, const Vector<double> &bn,                  const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);

    template <class S> int fact_minverse   (Vector<S> &ap, Vector<S> &an, const Vector<S> &bp, const Vector<S> &bn,                           const Matrix<double> &Gn, const Matrix<double> &Gpn, int bpNZ = -2, int bnNZ = -2) const;
    template <class S> int fact_near_invert(Vector<S> &ap, Vector<S> &an,                                           const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn                              ) const;

    double fact_det(void) const;

    // Factorisation information functions
    //
    // fact_nfact:  returns the number of Gn row/columns in the factorised (or
    //              invertible if Gp == Gn == 0) part
    // fact_pfact:  returns the number of Gp row/columns in the factorised (or
    //              invertible if Gp == Gn == 0) part
    // fact_nofact: returns true if Gp == Gn == 0 (nonzero size).

    int fact_nfact (const Matrix<double> &Gn, const Matrix<double> &Gpn) const;
    int fact_pfact (const Matrix<double> &Gn, const Matrix<double> &Gpn) const;
    int fact_nofact(const Matrix<double> &Gn, const Matrix<double> &Gpn) const;

    // Factorisation control functions
    //
    // fudgeOn:  turns on  fudging (diagonal offsetting) to ensure full factorisation
    // fudgeOff: turns off fudging (diagonal offsetting) to ensure full factorisation
    //
    // NB: fudging doesn't work very well - appears to be numerically unstable

    void fact_fudgeOn (const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    void fact_fudgeOff(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);

    // Test factoriation
    //
    // fact_testFact reconstructs the factorisation in dest matrices and returns the
    // maximum absolute difference between an element of Gp, Gn and Gpn and the
    // reconstructed version.

    double fact_testFact(Matrix<double> &Gpdest, Matrix<double> &Gndest, Matrix<double> &Gpndest, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const;

private:

    // Pivotting variables

    Vector<int> pAlphaLB;
    Vector<int> pAlphaZ;
    Vector<int> pAlphaUB;
    Vector<int> pAlphaF;

    Vector<int> pBetaC;
    Vector<int> pBetaF;

    Vector<int> dalphaState;
    Vector<int> dbetaState;

    // Miscellaneous variables
    //
    // dzt       = zero tolerance
    // dkeepfact = true if factorisation kept
    //
    // daNLF = number of alphas in LF
    // daNUF = number of alphas in UF

    double dzt;
    int dkeepfact;

    int daNLF;
    int daNUF;

    // Factorisation related stuff
    // ===========================
    //
    // GpnFColNorm(i) = sum_jP Gpn(pivAlphaF(jP),i)*Gpn(pivAlphaF(jP),i) + sum_jP Gn(pBetaF(jP),i)*Gn(pBetaF(jP),i)
    //
    // which is updated incrementally.  If the norm is below the threshold
    // GPNCOLNORMMIN then the column is considered to be effectively zero
    // and hence we can set betaFix(i) in the context.  Otherwise betaFix(i)
    // is zero and the column may be included in the factorisation.
    //
    // betaFixUpdate = 0: betaFix and GpnFColNorm are both up-to-date
    //               = 1: those betaFix elements set to -1 need to be updated.

    Vector<double> GpnFColNorm;
    Vector<int> betaFix;
    int betaFixUpdate;

    // State variables
    //
    // nfact: the total number of elements in pBetaF included in the
    //        factorisation.  Note that elements of pBetaF should not be
    //        included in the singular part of the factorisation.
    // pfact: the total number of elements in pAlphaF included in the
    //        non-singular part of the factorisation.

    int dnfact;
    int dpfact;
    int factsize;

    // Diagonal +-1 vector for LDL' cholesky style factorisation

    Vector<double> D;

    // The Cholesky Factorisation, suitably modified.

    Chol<double> freeVarChol;

    // Position in factorisation
    //
    // fAlphaF(i) is the position of alpha(pAlphaF(i)) in the factorisation
    // fBetaF(i)  is the position of beta(pBetaF(i))   in the non-singular part of the factorisation (-1 otherwise)

    Vector<int> fAlphaF;
    Vector<int> fBetaF;

    // Hacketty hack hack, see fact_nofact.

    optContext *thisredirect[1];

    // Factorisation upkeep functions
    //
    // extendFactAlpha: a new alpha has been added, namely the final element in pAlphaF.
    //                  Extend the factorisation if possible.  By default leave at end of
    //                  factorisation, unless Gp(pAlphaF,pAlphaF) = 0 and Gn(pBetaF,pBetaF) = 0
    //                  prior to the extension, in which case we add it to the start.  The
    //                  returned value is where the alpha ends up in pAlphaF, so either
    //                  pAlphaF.size()-1 or zero.
    // extendFactBeta:  a new beta has been added, namely the final element in pBetaF.
    //                  Extend the factorisation if possible.  The returned value is where
    //                  the beta ends up in pBetaF.
    // shrinkFactAlpha: alpha element i in position fAlphaF(ipos) in the factorisation has been
    //                  removed.  Fix the factorisation where possible.
    // shrinkFactBeta:  beta element i in position fBetaF(ipos) in the factorisation has been
    //                  removed.  Fix the factorisation where possible.
    //
    // fixfact:    attempt to maximise the size of the factorisation.  This is achieved by changing
    //             the interleaving of alpha and beta components and also by changing the pivots
    //             pAlphaF and pBetaF.  pAlphaF will only be changed if none freeVarChol.ngood() == 0
    //             and either numNZGpDiag > 0 or numNZGnDiag > 0, and then only by squareswapping
    //             to elements in pAlphaF.  pBetaF may be changed arbitrarily.
    //             The arguments apos and bpos are used as follows: if the element pAlphaF(apos) is
    //             swapped then apos will be changed to the new position, and likewise if pBetaF(bpos)
    //             is moved then bpos will be changed to the new position.
    // fixbetaFix: fix betaFix and GpnFColNorm

    int extendFactAlpha(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    int extendFactBeta(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    void shrinkFactAlpha(int i, int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    void shrinkFactBeta(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos);
    void fixfact(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos, int &aposalt, int &bposalt);
    void fixbetaFix(const Matrix<double> &Gn, const Matrix<double> &Gpn);
};

inline void qswap(optContext &a, optContext &b)
{
    qswap(a.pAlphaLB     ,b.pAlphaLB     );
    qswap(a.pAlphaZ      ,b.pAlphaZ      );
    qswap(a.pAlphaUB     ,b.pAlphaUB     );
    qswap(a.pAlphaF      ,b.pAlphaF      );
    qswap(a.pBetaC       ,b.pBetaC       );
    qswap(a.pBetaF       ,b.pBetaF       );
    qswap(a.dalphaState  ,b.dalphaState  );
    qswap(a.dbetaState   ,b.dbetaState   );
    qswap(a.dzt          ,b.dzt          );
    qswap(a.dkeepfact    ,b.dkeepfact    );
    qswap(a.daNLF        ,b.daNLF        );
    qswap(a.daNUF        ,b.daNUF        );
    qswap(a.GpnFColNorm  ,b.GpnFColNorm  );
    qswap(a.betaFix      ,b.betaFix      );
    qswap(a.betaFixUpdate,b.betaFixUpdate);
    qswap(a.dnfact       ,b.dnfact       );
    qswap(a.dpfact       ,b.dpfact       );
    qswap(a.factsize     ,b.factsize     );
    qswap(a.D            ,b.D            );
    qswap(a.freeVarChol  ,b.freeVarChol  );
    qswap(a.fAlphaF      ,b.fAlphaF      );
    qswap(a.fBetaF       ,b.fBetaF       );

    return;
}

template <class S>
int optContext::fact_minverse(Vector<S> &ap, Vector<S> &an, const Vector<S> &bp, const Vector<S> &bn, const Matrix<double> &Gn, const Matrix<double> &Gpn, int bpNZ, int bnNZ) const
{
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( ap.size() == aN() );
    NiceAssert( an.size() == bN() );
    NiceAssert( bp.size() == aN() );
    NiceAssert( bn.size() == bN() );
    NiceAssert( dkeepfact );

    int retval = 0;

    int zp_start = 0;
    int zn_start = 0;
    int zp_end = 0;
    int zn_end = 0;

    if ( bpNZ == -2 )
    {
	zp_start = 0;
        zp_end   = 0;
    }

    else if ( bpNZ == -1 )
    {
        zp_start = 0;
        zp_end   = dpfact;
    }

    else if ( bpNZ < dpfact )
    {
        zp_start = bpNZ;
	zp_end   = dpfact-bpNZ-1;
    }

    else
    {
	zp_start = 0;
        zp_end   = dpfact;
    }

    if ( bnNZ == -2 )
    {
	zn_start = 0;
        zn_end   = 0;
    }

    else if ( bnNZ == -1 )
    {
        zn_start = 0;
        zn_end   = dnfact;
    }

    else if ( bnNZ < dnfact )
    {
        zn_start = bnNZ;
	zn_end   = dnfact-bnNZ-1;
    }

    else
    {
	zn_start = 0;
        zn_end   = dnfact;
    }

    if ( freeVarChol.ngood() )
    {
        retVector<S> tmpva;
        retVector<S> tmpvb;
        retVector<S> tmpvc;
        retVector<S> tmpvd;
        retVector<S> tmpve;
        retVector<S> tmpvf;
        retVector<S> tmpvg;
        retVector<S> tmpvh;

	freeVarChol.minverse(ap("&",pAlphaF,tmpva)("&",0,1,dpfact-1,tmpvb),an("&",pBetaF,tmpvc)("&",0,1,dnfact-1,tmpvd),bp(pAlphaF,tmpve)(0,1,dpfact-1,tmpvf),bn(pBetaF,tmpvg)(0,1,dnfact-1,tmpvh),zp_start,zp_end,zn_start,zn_end);

        ap("&",pAlphaF,tmpva)("&",dpfact,1,(pAlphaF.size())-1,tmpvb).zero();
        an("&",pBetaF,tmpva)("&",dnfact,1,(pBetaF.size())-1,tmpvb).zero();

        retval = dpfact+dnfact;
    }

    else if ( pAlphaF.size() && pBetaF.size() )
    {
	int nonsingsize = fact_pfact(Gn,Gpn);

	if ( nonsingsize )
	{
            retVector<S>      tmpva;
            retVector<S>      tmpvb;
            retVector<S>      tmpvc;
            retVector<S>      tmpvd;
            retMatrix<double> tmpma;
            retMatrix<double> tmpmb;

	    Matrix<double> Gpninv((Gpn(pAlphaF,pBetaF,tmpma)(0,1,nonsingsize-1,0,1,nonsingsize-1,tmpmb)).inve());

            ap("&",pAlphaF,tmpva)("&",0,1,nonsingsize-1,tmpvb) = bn(pBetaF,tmpvc)(0,1,nonsingsize-1,tmpvd);
	    an("&",pBetaF,tmpva)("&",0,1,nonsingsize-1,tmpvb)  = bp(pAlphaF,tmpvc)(0,1,nonsingsize-1,tmpvd);

            ap("&",pAlphaF,tmpva)("&",0,1,nonsingsize-1,tmpvb) *= Gpninv;
	    rightmult(Gpninv,an("&",pBetaF,tmpva)("&",0,1,nonsingsize-1,tmpvb));
	}

        retVector<S> tmpva;
        retVector<S> tmpvb;

        ap("&",pAlphaF,tmpva)("&",nonsingsize,1,(pAlphaF.size())-1,tmpvb).zero();
        an("&",pBetaF,tmpva)("&",nonsingsize,1,(pBetaF.size())-1,tmpvb).zero();

	retval = 2*nonsingsize;
    }

    return retval;
}

template <class S> int optContext::fact_near_invert(Vector<S> &ap, Vector<S> &an, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( ap.size() == aN() );
    NiceAssert( an.size() == bN() );
    NiceAssert( dkeepfact );

    int retval = 0;

    if ( freeVarChol.ngood() )
    {
        retVector<S> tmpva;
        retVector<S> tmpvb;
        retVector<S> tmpvc;
        retVector<S> tmpvd;

	freeVarChol.near_invert(ap("&",pAlphaF,tmpva)("&",0,1,dpfact-1,tmpvb),an("&",pBetaF,tmpvc)("&",0,1,dnfact-1,tmpvd));

	ap("&",pAlphaF,tmpva)("&",dpfact,1,(pAlphaF.size())-1,tmpvb) = 0.0;
	an("&",pBetaF,tmpva)("&",dnfact,1,(pBetaF.size())-1,tmpvb)   = 0.0;

        retval = dpfact+dnfact;
    }

    else if ( pAlphaF.size() && pBetaF.size() )
    {
	int nonsingsize = fact_pfact(Gn,Gpn);

        NiceAssert( ( nonsingsize < aNF() ) || ( nonsingsize < bNF() ) );

	if ( nonsingsize )
	{
	    if ( nonsingsize < aNF() )
	    {
		int i;

		if ( nonsingsize )
		{
		    for ( i = 0 ; i < nonsingsize ; i++ )
		    {
			ap("&",pAlphaF(i)) = Gpn(pAlphaF(nonsingsize),pBetaF(i));
			an("&",pBetaF(i))  = Gp(pAlphaF(nonsingsize),pAlphaF(i));
		    }
		}
	    }

	    else
	    {
		Vector<S> bp(nonsingsize);

		int iP;

		for ( iP = 0 ; iP < nonsingsize ; iP++ )
		{
		    bp("&",iP) = Gpn(pAlphaF(pAlphaF(iP)),pBetaF(nonsingsize));
		}

                retVector<S>      tmpva;
                retVector<S>      tmpvb;
                retVector<S>      tmpvc;
                retMatrix<double> tmpma;

		ap("&",pAlphaF,tmpva)("&",zeroint(),1,nonsingsize-1,tmpvb) = Gn(pBetaF,pBetaF,tmpma)(nonsingsize,zeroint(),1,nonsingsize-1,tmpvc);
		an("&",pBetaF,tmpva)("&",zeroint(),1,nonsingsize-1,tmpvb)  = bp;
	    }

            retVector<S>      tmpva;
            retVector<S>      tmpvb;
            retMatrix<double> tmpma;
            retMatrix<double> tmpmb;

	    Matrix<double> Gpninv((Gpn(pAlphaF,pBetaF,tmpma)(0,1,nonsingsize-1,0,1,nonsingsize-1,tmpmb)).inve());

            ap("&",pAlphaF,tmpva)("&",0,1,nonsingsize-1,tmpvb) *= Gpninv;
	    rightmult(Gpninv,an("&",pBetaF,tmpva)("&",0,1,nonsingsize-1,tmpvb));
	}

        retVector<S> tmpva;
        retVector<S> tmpvb;

        ap("&",pAlphaF,tmpva)("&",nonsingsize,1,(pAlphaF.size())-1,tmpvb) = 0.0;
        an("&",pBetaF,tmpva)("&",nonsingsize,1,(pBetaF.size())-1,tmpvb)   = 0.0;

	retval = 2*nonsingsize;
    }

    return retval;
}

#endif
