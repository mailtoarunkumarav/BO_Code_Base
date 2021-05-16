
//
// Quadratic optimisation state
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _optstate_h
#define _optstate_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "optcontext.h"
#include "mlcommon.h"

// Consider the quadratic programming problem:
//
// [ alpha ]' [ Gp   Gpn ] [ alpha ] + [ alpha ]' [ gp ] + | alpha' |' [ hp ]
// [ beta  ]  [ Gpn' Gn  ] [ beta  ]   [ beta  ]  [ gn ]   | beta   |  [ 0  ]
//
// where it is assumed that:
//
// - Gp is positive semi-definite hermitian
// - Gn is negative semi-definite hermitian
//
// and other details can be found in optcontext.h.  In addition to what is
// contained there this header adds the following information:
//
// - alphaRestrict is a vector controlling the range of alpha:
//
//    alphaRestrict[i] = 0: lb[i] <= alpha[i] <= ub[i]
//    alphaRestrict[i] = 1:     0 <= alpha[i] <= ub[i]
//    alphaRestrict[i] = 2: lb[i] <= alpha[i] <= 0
//    alphaRestrict[i] = 3:     0 <= alpha[i] <= 0
//
// - betaRestrict is a vector controlling the range of beta:
//
//    betaRestrict[i] = 0: -inf <= beta[i] <= inf
//    betaRestrict[i] = 1:    0 <= beta[i] <= inf
//    betaRestrict[i] = 2: -inf <= beta[i] <= 0
//    betaRestrict[i] = 3:    0 <= beta[i] <= 0
//
// Stored here are:
//
// - alpha: the alpha vector
// - beta: the beta vector
// - alphagrad: the gradient wrt to the alpha vector
// - betagrad: the gradient wrt to the beta vector
// - optcontext: see optcontext.h
// - alphaRestrict: see above
// - betaRestrict: see above
// - opttol: tolerance for optimality conditions
//
// Optimality
// ----------
//
// - beta:
//
//    betaRestrict[i] = 0: -inf <= beta[i] <= inf   betaGrad == 0
//    betaRestrict[i] = 1:    0 <= beta[i] <= inf   betaGrad <= 0
//    betaRestrict[i] = 2: -inf <= beta[i] <= 0     betaGrad >= 0
//    betaRestrict[i] = 3:    0 <= beta[i] <= 0
//
// Gradient/Factorisation split
// ----------------------------
//
// If GpGrad != Gp then GpGrad is used for all
// gradient caculations and Gp for any factorisation /
// step calculation operations.

//#define DEFAULT_OPTTOL 0.001


// Stream operators

template <class T, class S> class optState;

template <class T, class S> std::ostream &operator<<(std::ostream &output, const optState<T,S> &src );
template <class T, class S> std::istream &operator>>(std::istream &input,        optState<T,S> &dest);

// Swap function

template <class T, class S> void qswap(optState<T,S> &a, optState<T,S> &b);

// STEPTRIGGER:    when variables are constrained there may be a small implicit
//                 step when variable goes from its old value to its new one.
//                 This leads to an error in the gradient equal to the implicit
//                 step times the relevant hessian element when the value is
//                 moved to the bound.  This is bounded above by the relevant
//                 diagonal of the hessian times the implicit step.  Hence to
//                 combat this if the diagonal times the magnitude of the
//                 implicit step exceeds STEPTRIGGER*opttol then we make an
//                 *explicit* step, thereby correcting the gradients.
// CUMGRADTRIGGER: cumulative error bound trigger point

#define STEPTRIGGER 1e-5
#define CUMGRADTRIGGER 1e-4

template <class T, class S>
class optState
{
    template <class U, class V> friend std::ostream &operator<<(std::ostream &output, const optState<U,V> &src );
    template <class U, class V> friend std::istream &operator>>(std::istream &input,        optState<U,V> &dest);

    template <class U, class V> friend void qswap(optState<U,V> &a, optState<U,V> &b);

public:

    // Constructors and assignment operators
    //
    // Note that constructors assume that all alphas are constrained to zero
    // and that all betas are constrained, but that alpha and beta are not
    // restrained.

    optState(void);
    optState(const optState<T,S> &src);
    optState<T,S> &operator=(const optState<T,S> &src);

    // Set alpha and beta

    void setAlpha(const Vector<T> &newAlpha, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &lb, const Vector<double> &ub, double ztoloverride = -1);
    void setBeta (const Vector<T> &newBeta,  const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, double ztoloverride = -1);

    void setAlphahpzero(const Vector<T> &newAlpha, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &lb, const Vector<double> &ub, double ztoloverride = -1);
    void setBetahpzero (const Vector<T> &newBeta,  const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, double ztoloverride = -1);

    // Reconstructor:
    //
    // Like the constructor, but keeps the existing alpha LB/LF/Z/UF/UB and
    // beta F/C information.
    //
    // refact:       fixes for changes in Gp, Gn, Gpn, gp, gn, and hp
    // refactlin:    fixes for changes in gp, gn and hp, but recalculates gradients from scratch
    // refactgp:     fixes for changes in gp (or just gp(i) if i given)
    // refactgn:     fixes for changes in gn (or just gn(i) if i given)
    // refacthp:     fixes for changes in hp (or just hp(i) if i given)
    // refactGpn:    fixes for changes in Gpn
    // refactGpnElm: fixes for changes in a single element of Gpn, assuming that the element is not in the factorisation (if any).
    // setopttol:    fixes for changes in opttol
    // setzt:        fixes for changes in zero tolerance
    // setkeepfact:  set keepfact, which controls if the factorisation is kept
    // factstepgp:   fixes for the change gp(i) -> gp(i)+gpistep
    // factstepgn:   fixes for the change gn(i) -> gn(i)+gnistep

    void refact      (const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gp,                            const Vector<T> &gn,                            const Vector<T> &hp                                          );
    void refactlin   (                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gp,                            const Vector<T> &gn,                            const Vector<T> &hp                                          );
    void refactgp    (                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gpOld, const Vector<T> &gpNew, const Vector<T> &gn,                            const Vector<T> &hp,                            int iv = -1  );
    void refactgn    (                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gp,                            const Vector<T> &gnOld, const Vector<T> &gnNew, const Vector<T> &hp,                            int iv = -1  );
    void refacthp    (                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gp,                            const Vector<T> &gn,                            const Vector<T> &hpOld, const Vector<T> &hpNew, int iv = -1  );
    void refactGpn   (const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &GpnOld, const Matrix<double> &GpnNew, const Vector<T> &gp,                            const Vector<T> &gn,                            const Vector<T> &hp                                          );
    void refactGpnElm(                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &GpnOld, double GpnijNew,              const Vector<T> &gp,                            const Vector<T> &gn,                            const Vector<T> &hp,                            int i, int j );
    void setopttol   (                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gp,                            const Vector<T> &gn,                            const Vector<T> &hp,                            double opttol);
    void setzt       (const Matrix<double> &Gp,                          const Matrix<double> &Gn, const Matrix<double> &Gpn,                                                                                                                                                                                  double zt    );
    void setkeepfact (const Matrix<double> &Gp,                          const Matrix<double> &Gn, const Matrix<double> &Gpn,                                                                                                                                                                                  int xkeepfact);
    void factstepgp  (                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gpOld, const T &gpistep,       const Vector<T> &gn,                            const Vector<T> &hp,                            int iv       );
    void factstepgn  (                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gp,                            const Vector<T> &gnOld, const T &gnistep,       const Vector<T> &hp,                            int iv       );

    void refacthpzero   (const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gp,                            const Vector<T> &gn               );
    void refactlinhpzero(                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gp,                            const Vector<T> &gn               );
    void refactgphpzero (                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gpOld, const Vector<T> &gpNew, const Vector<T> &gn, int iv = -1  );
    void refactgnhpzero (                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gp,                            const Vector<T> &gnOld, const Vector<T> &gnNew, int iv = -1  );
    void refactGpnhpzero(const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &GpnOld, const Matrix<double> &GpnNew, const Vector<T> &gp,                            const Vector<T> &gn               );
    void setopttolhpzero(                          const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn,                                  const Vector<T> &gp,                            const Vector<T> &gn, double opttol);

    // Reset state:
    //
    // Sets all alpha = 0 constrained to zero, beta = 0 (state unchanged)

    void reset(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    void resethpzero(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);

    // Find position in pivotted variables

    int findInAlphaLB(int i) const { return probContext.findInAlphaLB(i); }
    int findInAlphaZ (int i) const { return probContext.findInAlphaZ(i);  }
    int findInAlphaUB(int i) const { return probContext.findInAlphaUB(i); }
    int findInAlphaF (int i) const { return probContext.findInAlphaF(i);  }

    int findInBetaC(int i) const { return probContext.findInBetaC(i); }
    int findInBetaF(int i) const { return probContext.findInBetaF(i); }

    // Gradient refreshing: based on a cumulative gradient error bound
    // recalculate the gradients from scratch if required.
    //
    // ..._anyhow variants recalculate gradients regardless.  i = -1 for all,
    // otherwise just do the one.

    void refreshGrad      (const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    void refreshGradhpzero(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     );

    void refreshGrad_anyhow      (const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int i = -1);
    void refreshGradhpzero_anyhow(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     , int i = -1);

    // Scaling
    //
    // scale: update alpha,beta *= a
    // scaleAlpha: update alpha *= a

    void scale     (double a, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    void scaleAlpha(double a, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);

    void scalehpzero     (double a, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);
    void scaleAlphahpzero(double a, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);

    // Gp modifications.  These update the Gp matrix, updating the gradients and
    // the factorisation (if kept).  The matrices are assumed to be the matrices
    // after the change has been made to them.

    void rankone   (const Vector<double> &bp, const Vector<double> &bn, const double &c, const Vector<double> &bpGrad, const Vector<double> &bnGrad, const double &cGrad, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    void diagmult  (const Vector<double> &bp, const Vector<double> &bn,                                                                                                   const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    void diagoffset(const Vector<double> &bp, const Vector<double> &bn,                  const Vector<double> &bpGrad, const Vector<double> &bnGrad,                      const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    void diagoffset(int i, double bpoff, double bpoffGrad,                                                                                                                const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);

    void diagoffsethpzero(const Vector<double> &bp, const Vector<double> &bn,                  const Vector<double> &bpGrad, const Vector<double> &bnGrad,                      const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);
    void diagoffsethpzero(int i, double bpoff, double bpoffGrad,                                                                                                                const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);

    // Control functions using unpivotted index:
    //
    // addAlpha: add alpha[i], presumed constrained to zero
    // addBeta:  add beta[i],  presumed constrained
    //
    // removeAlpha: remove alpha[i], assuming it is restricted to zero
    // removeBeta:  remove beta[i],  assuming it is restricted to zero
    //
    // addAlpha and addBeta return the new position in the relevant pivot
    // vector (either pAlphaZ or pBetaC)
    //
    // removeAlpha and removeBeta return the old position in the relevant pivot
    // vector (either pAlphaZ or pBetaC)

    int addAlpha(int i, int alphrestrict, const T &zeroeg);
    int addBeta (int i, int betrestrict , const T &zeroeg);

    int removeAlpha(int i);
    int removeBeta (int i);

    // Change alpha and beta restrictions, taking steps and constraining first if required

    void changeAlphaRestrict(int i, int alphrestrict, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    void changeBetaRestrict (int i, int betrestrict , const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);

    void changeAlphaRestricthpzero(int i, int alphrestrict, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);
    void changeBetaRestricthpzero (int i, int betrestrict , const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);

    // Variable state control using index to relevant pivot vector
    //
    // modAlpha*to#: remove variable pAlpha*[iP] and put it in pAlpha#
    // modBeta*to#:  remove variable pBeta*[iP]  and put it in pBeta#
    //
    // Each function returns the new position in the relevant pivot vector
    // (either pAlphaLB, pAlphaZ, pAlphaUB, pAlphaF, pBetaC, or pBetaF.
    // NB: positions are *NOT* returned in pAlphaLF or pAlphaUF!).
    //
    // NB2: note that a step will be required for some changes, specifically:
    //      - LB to Z,UF,UB
    //      - LF to UF,UB
    //      - Z to LB,UB
    //      - UF to LF,LB
    //      - UB to Z,LF,LB

    int modAlphaLBtoLF(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaLBtoZ (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaLBtoUF(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaLBtoUB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaLFtoLB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &lb);
    int modAlphaLFtoZ (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaLFtoUF(int iP,                           const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaLFtoUB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &ub);
    int modAlphaZtoLB (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaZtoLF (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaZtoUF (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaZtoUB (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaUFtoLB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &lb);
    int modAlphaUFtoLF(int iP,                           const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaUFtoZ (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaUFtoUB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &ub);
    int modAlphaUBtoLB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaUBtoLF(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaUBtoZ (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );
    int modAlphaUBtoUF(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                          );

    int modBetaCtoF(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    int modBetaFtoC(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);

    int modAlphaLBtoLFhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaLBtoZhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaLBtoUFhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaLBtoUBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaLFtoLBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &lb);
    int modAlphaLFtoZhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaLFtoUFhpzero(int iP,                           const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaLFtoUBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &ub);
    int modAlphaZtoLBhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaZtoLFhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaZtoUFhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaZtoUBhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaUFtoLBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &lb);
    int modAlphaUFtoLFhpzero(int iP,                           const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaUFtoZhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaUFtoUBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &ub);
    int modAlphaUBtoLBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaUBtoLFhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaUBtoZhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );
    int modAlphaUBtoUFhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                          );

    int modBetaCtoFhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);
    int modBetaFtoChpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);

    // Sign-ignoring batch modifiers
    //
    // Free/constrain all alphas and betas based on alphaRestrict and
    // betaRestrict (free if restrict == 0, constrained if restrict == 3,
    // throw otherwise).  Will also change (or set) alpha UF or LF depending
    // on the sign of alpha.

    int modAllToDesthpzero(const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);

    // Pivotting

    const Vector<int> &pivAlphaLB(void) const { return probContext.pivAlphaLB(); }
    const Vector<int> &pivAlphaZ (void) const { return probContext.pivAlphaZ();  }
    const Vector<int> &pivAlphaUB(void) const { return probContext.pivAlphaUB(); }
    const Vector<int> &pivAlphaF (void) const { return probContext.pivAlphaF();  }

    const Vector<int> &pivBetaC(void) const { return probContext.pivBetaC(); }
    const Vector<int> &pivBetaF(void) const { return probContext.pivBetaF(); }

    // Information

    int aN  (void) const { return dalpha.size();       }
    int aNF (void) const { return probContext.aNF();   }
    int aNC (void) const { return aN()-aNF();          }
    int aNNZ(void) const { return aNF()+aNLB()+aNUB(); }
    int aNLB(void) const { return probContext.aNLB();  }
    int aNLF(void) const { return probContext.aNLF();  }
    int aNZ (void) const { return probContext.aNZ();   }
    int aNUF(void) const { return probContext.aNUF();  }
    int aNUB(void) const { return probContext.aNUB();  }

    int bN  (void) const { return dbeta.size();        }
    int bNF (void) const { return (pivBetaF()).size(); }
    int bNC (void) const { return (pivBetaC()).size(); }

    double zerotol(void) const { return probContext.zt(); }
    double opttol (void) const { return dopttol;          }

    int keepfact(void) const { return probContext.keepfact(); }

    // Returns true if factorisation available but not complete in Gp part.
    // Note short-circuit.

    int factbad(const Matrix<double> &Gn, const Matrix<double> &Gpn) const { return !( !probContext.keepfact() || ( probContext.fact_pfact(Gn,Gpn) == aNF() ) ); }

    // Access to contents

    const Vector<T> &alpha(void) const { return dalpha; }
    const Vector<T> &beta (void) const { return dbeta;  }

    const Vector<T> &alphaGrad(void) const { NiceAssert( !gradFixAlphaInd ); return dalphaGrad; }
    const Vector<T> &betaGrad (void) const { NiceAssert( !gradFixBetaInd  ); return dbetaGrad;  }

    const Vector<int> &alphaRestrict(void) const { return dalphaRestrict; }
    const Vector<int> &betaRestrict (void) const { return dbetaRestrict;  }

    const Vector<int> &alphaState(void) const { return probContext.alphaState(); }
    const Vector<int> &betaState (void) const { return probContext.betaState();  }

    const T &alpha(int i) const { return dalpha(i); }
    const T &beta (int i) const { return dbeta(i);  }

    const T &alphaGrad(int i) const { NiceAssert( !gradFixAlpha(i) ); return dalphaGrad(i); }
    const T &betaGrad (int i) const { NiceAssert( !gradFixBeta(i) );  return dbetaGrad(i);  }

    int alphaRestrict(int i) const { return dalphaRestrict(i); }
    int betaRestrict (int i) const { return dbetaRestrict(i);  }

    int alphaState(int i) const { return alphaState()(i); }
    int betaState (int i) const { return betaState()(i);  }

    // Uncorrected gradients
    //
    // unAlphaGrad: returns alphaGrad()(i) with hp terms removed
    // reAlphaGrad: returns alphaGrad()(i) if the sign was reversed (UF<->LF, UB<->LB)
    // unBetaGrad: returns betaGrad()(i) (useful as it calculates the gradient first)
    // reBetaGrad: returns betaGrad()(i) (useful as it calculates the gradient first)
    // posAlphaGrad: returns alphaGrad()(i) with alpha state positive (ie fixing hp terms)
    // negAlphaGrad: returns alphaGrad()(i) with alpha state negative (ie fixing hp terms)
    // spAlphaGrad:  returns alphaGrad()(i), pos/neg if alphaState(i) == 0 and alphaRestrict == 1/2
    //
    // unAlphaGradIfPresent: calc alphaGrad()(i) with hp terms removed if it is in the cache.
    //                       return 0 on success, 1 if not in cache

    int unAlphaGradIfPresent(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const;

    T &unAlphaGrad (T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const;
    T &reAlphaGrad (T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const;
    T &posAlphaGrad(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const;
    T &negAlphaGrad(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const;
    T &spAlphaGrad (T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const;

    T &unBetaGrad(T &result, int i, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gn) const;
    T &reBetaGrad(T &result, int i, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gn) const;

    T &unAlphaGradhpzero (T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp) const;
    T &reAlphaGradhpzero (T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp) const;
    T &posAlphaGradhpzero(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp) const;
    T &negAlphaGradhpzero(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp) const;
    T &spAlphaGradhpzero (T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp) const;

    // Scale step
    //
    // scaleFStep: scale step to fit bounds and restrictions on alpha and beta.
    //             returns 0 if step doesn't need to be scaled or 1 if it does.
    //             scale is set to the scale used, and the step is scaled.
    //             alphaFIndex is set to the free alpha index that hits bounds (-1 if none)
    //             betaFIndex is set to the free beta index that hits bounds (-1 if none)
    //
    // NB: this uses the sign of the step and is therefore unsuitable for use
    //     with classes that do not have well defined order operators >,>=,<,<=
    //
    // ***********************************************************************************
    // ***********************************************************************************
    // ***                                                                             ***
    // *** NB: this function makes one BIG ASSUMPTION, namely that unrestricted betas  ***
    // ***     are never constrained.  This is readily ensured by calling the function ***
    // ***     initGradBeta(...) before starting any optimisation.                     ***
    // ***                                                                             ***
    // ***********************************************************************************
    // ***********************************************************************************
    //
    // Only the first asize element of alphaFstep and the first bsize elements
    // of betaFStep are considered.
    //
    // stateChange tells which bound has been hit: -2 == lower bound
    //                                             -1 == zero from below (or free to negative)
    //                                             +1 == zero from above (or free to positive)
    //                                             +2 == upper bound
    //
    // NaN can occur if:
    //
    // alpha is NaN
    // alpha step is NaN
    // (alpha gap ~= 0)/(alpha step ~= 0) occurs (approximately)
    //
    // and likewise with beta.  The first two cases trigger bailout,
    // the last involves infinitessimal creap that can be safely
    // ignored.
    //
    // bailout codes:
    //
    // 0 - no problem
    // 1 - alphaF has NaN elements
    // 2 - alphaFstep has NaN elements
    // 3 - betaF has NaN elements
    // 4 - betaFstep has NaN elements
    // 5 - betaCgrad has NaN elements
    // 6 - betaCgradstep has NaN elements

    int scaleFStep      (double &scale, int &alphaFIndex, int &betaFIndex, int &betaCIndex, int &stateChange, int asize, int bsize, int &bailout, Vector<T> &alphaFStep, Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &lb, const Vector<double> &ub);
    int scaleFStephpzero(double &scale, int &alphaFIndex, int &betaFIndex, int &betaCIndex, int &stateChange, int asize, int bsize, int &bailout, Vector<T> &alphaFStep, Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn,                      const Vector<double> &lb, const Vector<double> &ub);

    // Take step
    //
    // alphaStep:       Step of a single free alpha variable alpha(i) (unpivotted)
    // betaStep:        Step of a single beta variable beta(i) (unpivotted)
    //
    // stepFNewton:     Scaled Newton step in free variables alpha(pivAlphaF()),
    //                  beta(pivBetaF()).  Gradients decrease linearly (to zero
    //                  if scale == 1).  Step is assumed to have been scaled
    //                  prior to calling, so "scale" is only used when updating
    //                  the gradients of the free variables.
    // stepFNewtonFull: Calls stepFNewton with scale == 1 and notes the fact that
    //                  the active gradients will now be zero
    // stepFLinear:     Linear step in free variables alpha(pivAlphaF()(0,1,asize-1)),
    //                  beta(pivBetaF()(0,1,bsize-1)), assuming no change in gradients
    //                  dalphaGrad(pivAlphaF()(0,1,asize-1)), dbetaGrad(pivBetaF()(0,1,bsize-1))
    //                  It is assumed that the step has been calculated with calcStep and
    //                  subsequently scaled linearly if required.
    // stepFGeneral:    General step in all free variables alpha(pivAlphaF()(0,1,asize-1)),
    //                  beta(pivBetaF()(0,1,bsize-1))
    //
    // OPTIMISATIONS:
    // ==============
    //
    // The doCupdate argument is used to prevent the function updating the non-free alpha gradients.
    // In this way you can take a number of (stepFNewton{Full}) steps in a row (presumably
    // with constrains between them so that the non-free gradients are not required), then
    // update the non-free gradients on the combined step using the updateGradOpt function.
    // Set doCupdate = 0 for this type of operation.
    //
    // The doFupdate is similar and can be used to inhibit updating of the free alpha gradients.
    // These gradients are *not*, however, fixed by updateGradOpt.
    //
    // NB: when using a sequence of doCupdates alphas will presumably move from F to nF.
    // You need to store the gradients of F at the start of the sequence so that the complete
    // step can update from the start of the sequence, not midway, for the gradients that started
    // in F and ended in nF.  Note that updateGradOpt takes care of hp factors if alphas have
    // gone from F to Z.  Add the relevant index to FnF to indicate that the alpha in question
    // has moved from F to nF.
    //
    // NB: the set of free betas must not change in such a sequence.

    void alphaStep(int i, const T &alphaStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int dontCheckState = 0);
    void betaStep (int i, const T &betaStep,  const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int dontCheckState = 0);

    void alphaStephpzero(int i, const T &alphaStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int dontCheckState = 0);
    void betaStephpzero (int i, const T &betaStep,  const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int dontCheckState = 0);

    void stepFNewton    (double scale,            int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int doCupdate = 1, int doFupdate = 1);
    void stepFNewtonFull(                         int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int doCupdate = 1, int doFupdate = 1);
    void stepFLinear    (              int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int doCupdate = 1, int doFupdate = 1);
    void stepFGeneral   (              int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int doCupdate = 1, int doFupdate = 1);

    void stepFNewtonhpzero    (double scale,            int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int doCupdate = 1, int doFupdate = 1);
    void stepFNewtonFullhpzero(                         int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int doCupdate = 1, int doFupdate = 1);
    void stepFLinearhpzero    (              int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int doCupdate = 1, int doFupdate = 1);
    void stepFGeneralhpzero   (              int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int doCupdate = 1, int doFupdate = 1);

    void updateGradOpt(const Vector<T> &combAlphaFStep, const Vector<T> &combBetaFStep, const Vector<T> &startAlphaGrad, const Vector<int> &FnF, const Vector<int> &startPivAlphaF, const Matrix<S> &GpGrad, const Matrix<double> &Gpn);

    // Test gradients
    //
    // Re-calculates gradients from scratch, stores then in alphaGradTest and
    // betaGradTest, and returns biggest error between these and the stored
    // gradients
    //
    // Int version just returns the max error and not the gradients.

    double testGrad   (int &aerr, int &berr, Vector<T> &alphaGradTest, Vector<T> &betaGradTest, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    double testGradInt(int &aerr, int &berr,                                                    const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);

    double testGradhpzero   (int &aerr, int &berr, Vector<T> &alphaGradTest, Vector<T> &betaGradTest, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);
    double testGradInthpzero(int &aerr, int &berr,                                                    const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn);

    // Find maximum non-optimal constrained gradient.  Returns 0 on success or
    // 1 if gradients are all optimal to within opttol precision.  If there are
    // free betas not in the factorisation then opttol is replaced by a large
    // negative number.  This assumes that the optimality of the free variables
    // has been ensured by a full newton step that does not necessarily contain
    // all free betas.
    //
    // alphaCIndex: index in relevant pivot vector if alpha, -1 otherwise
    // betaCIndex:  index in relevant pivot vector if beta, -1 otherwise (not possible if ignorebeta != 0)
    // stateChange: -2 if lower bound alpha should be freed to negative
    //              -1 if zero alpha/beta should be freed to negative
    //              0  if current solution optimal to withing opttol precision
    //                 (or checkfree set and free variable most non-optimal)
    //              +1 if zero alpha/beta should be freed to positive
    //              +2 if upper bound alpha should be freed to positive
    // gradmag: magnitude of most non-optimal gradient.
    // toloverride: if set (>= 0) then this tolerance is used rather than
    //              the opttol set here.
    // checkfree: if set then will also check optimality of free alphas.
    //
    // NB: this function is not suitable for anything other than doubles as it
    //     bases its optimality conditions on the doubles only.
    //
    // Beta versions find the most non-optimal beta, constrained or otherwise

    int maxGradNonOpt      (int &alphaCIndex, int &betaCIndex, int &stateChange, double &gradmag, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, double toloverride = -1, int checkfree = 0, int ignorebeta = 0);
    int maxGradNonOpthpzero(int &alphaCIndex, int &betaCIndex, int &stateChange, double &gradmag, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     , double toloverride = -1, int checkfree = 0, int ignorebeta = 0);

    int maxBetaGradNonOpt      (int &betaCIndex, int &stateChange, double &gradmag, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    int maxBetaGradNonOpthpzero(int &betaCIndex, int &stateChange, double &gradmag, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     );

    // Pre-optimisation function: call this function before proceeding with
    // optimisation.  It does two things:
    //
    // - free all unconstrained betas
    // - free all constrained betas with non-optimal gradients
    // - return -2 if there is more than one non-optimal free beta, -1 if all
    //   free betas are zero, and >= 0 if just one free beta is non-optimal.

    int initGradBeta      (const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    int initGradBetahpzero(const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     );

    // Factorisation functions: the following functions are only available if
    // (or make sense if) keepfact == 1.
    //
    // ========
    //
    // Calculate step
    //
    // Calculate step that is either an unscaled newton step or a direction of
    // linear ascent/descent.  The step is calculated so that:
    //
    // dalpha(pivAlphaLB()) = 0
    // dalpha(pivAlphaZ ()) = 0
    // dalpha(pivAlphaUB()) = 0
    // dbeta(pivBetaC()) = 0
    //
    // The function returns 0 if the step is a Newton step or 1 if it is a
    // linear step.  asize and bsize are set to the number of elements of
    // dalpha(pivAlphaF()) and dbeta(pivBetaF()) that are nonzero in the
    // step.
    //
    // The step is stored in stepAlpha(pivAlphaF()) and stepBeta(pivBetaF())
    //
    // NB: the function fact_calcStep is not suitable for anything other than
    //     doubles as in the linear case it bases its scaling laws on the
    //     operators >,>=,<,<=

    int fact_calcStep      (Vector<T> &stepAlpha, Vector<T> &stepBeta, int &asize, int &bsize, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &lb, const Vector<double> &ub);
    int fact_calcStephpzero(Vector<T> &stepAlpha, Vector<T> &stepBeta, int &asize, int &bsize, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn,                      const Vector<double> &lb, const Vector<double> &ub);

    // Direct use of factorisation
    //
    // Calculate a, where Gu.a = b, given a and the usual notational convensions

    template <class U>
    void fact_minverse(Vector<U> &aAlpha, Vector<U> &aBeta, const Vector<U> &bAlpha, const Vector<U> &bBeta, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const;
    double fact_det(void) const { return probContext.fact_det(); }

    // Test factorisation

    double fact_testFact   (Matrix<double> &Gpdest, Matrix<double> &Gndest, Matrix<double> &Gpndest, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const;
    double fact_testFactInt(                                                                         const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const;

    // Factorisation fudging

    void fact_fudgeOn (const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn);
    void fact_fudgeOff(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn);

    // Snookering test
    //
    // The factorisation is said to be "snookered" if beta is non-optimal
    // and there are non-optimal beta components that are not stored in
    // the factorisation.  Taking an unscaled Newton step in this situation
    // can lead to the incorrect assumption that beta will become optimal
    // when in fact it does not.

    int fact_snookered(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn);

    // Dimensionality function:
    //
    // On occasion it may be necessary to apply a nuetral re-dimensioning
    // operation to all T elements stored here (for example if T is a vector).
    // The following function allows this to occur.  Essentially for every
    // element elm it will call:
    //
    // redimelm(elm,olddim,newdim)
    //
    // and redimelm can do what needs to be done.  Note that you might need to
    // refact the state after calling this function (if relevant parts of gp,
    // gn, hp are nonzero).  Assumes "new" elements are set zero.

    void redimensionalise(void (*redimelm)(T &, int, int), int olddim, int newdim);

    void fixGrad      (const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);
    void fixGradhpzero(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     );

    // Gradient calculations

    void recalcAlphaGrad(T &res, const Matrix<S> &GpGrad,  const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp, int i) const;
    void recalcBetaGrad (T &res, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gn,                      int i) const;

    void recalcAlphaGradhpzero(T &res, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, int i) const;

    // Objective evaluation

    T &calcObj(T &res, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp);

private:

    Vector<T> dalpha;
    Vector<T> dbeta;
    Vector<T> dalphaGrad;
    Vector<T> dbetaGrad;

    Vector<int> dalphaRestrict;
    Vector<int> dbetaRestrict;

    double dopttol;
    optContext probContext;

    // gradstate: -2  means all free (or more than one) gradients are nonzero
    //            -1  means all free gradients are zero
    //            n>0 means precisely one free gradient is nonzero, namely pivAlphaF(alphagradstate) (or pivBetaF(betagradstate)).
    //
    // NOTE: these variables are designed for speed.  As such, certain
    //       assumptions are made.  In particular, when converting a variable
    //       from zero constrained to free we simple *assume* the associated
    //       gradient is nonzero.  They are only used if keepfact is set.

    int alphagradstate;
    int betagradstate;

    // gradient fixing: when alpha/beta are added it is best not to require
    // Gp, Gn and Gpn be included in the call.  For this reason it is not
    // possible to update the gradients at this point.  Instead
    // gradFixAlphaInd or gradFixBetaInd is set and the relevant element of
    // gradFixAlpha/gradFixBeta element is set.
    //
    // The function fixGrad() may be called later to fix the gradients.

    int gradFixAlphaInd;
    int gradFixBetaInd;
    Vector<int> gradFixAlpha;
    Vector<int> gradFixBeta;

    // cumgraderr: this is a bound on the cumulative gradient errors
    //
    // These are worked out as follows: when a free alpha (beta) is constrained
    // there is a small implicit step from a value close to the bound.  For example
    // if alpha_i is constrained to zero then for all j the gradient change is:
    //
    // \delta_j = -G_ij \alpha_j
    //
    // As G_ij is bounded above by G_ii (ie Gp is positive definite, Gn is
    // negative definite) then:
    //
    // | \delta_j | = G_ii | \alpha_i |
    //
    // Hence all gradients errors have the same bound!  So, every time a variable
    // is constrained we add this (small) value to cumgraderr.  When refreshGrad()
    // is called then if cumgraderr > CUMGRADERRTRIGGER then the gradients are
    // re-calculated from scratch and alphagradstate, betagradstate are reset
    // appropriately.

    double cumgraderr;

    // base functions (common parts for functions with hpzero and standard versions)

    int scaleFStepbase(double &scale, int &alphaFIndex, int &betaFIndex, int &betaCIndex, int &stateChange, int asize, int bsize, int &bailout, Vector<T> &alphaFStep, Vector<T> &betaFStep, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &lb, const Vector<double> &ub, int hpzero);
    void alphaStepbase(int i, const T &alphaStep, const Matrix<S> &GpGrad,      const Matrix<double> &Gpn, int dontCheckState);
    void betaStepbase (int i, const T &betaStep , const Matrix<double> &Gn,     const Matrix<double> &Gpn, int dontCheckState);
    void stepFNewtonbase    (double scale,            int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, int doCupdate, int doFupdate);
    void stepFNewtonFullbase(                         int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, int doCupdate, int doFupdate);
    void stepFLinearbase    (              int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, int doCupdate, int doFupdate);
    void stepFGeneralbase   (              int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, int doCupdate, int doFupdate);
    int fact_calcStepbase(Vector<T> &stepAlpha, Vector<T> &stepBeta, int &asize, int &bsize, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &lb, const Vector<double> &ub);

    // NB: this make threaded use (for a single object over multiple threads) impossible

    Vector<double> betaGradStepC_scaleFStepbase;
};

template <class T, class S> void qswap(optState<T,S> &a, optState<T,S> &b)
{
    qswap(a.dalpha         ,b.dalpha         );
    qswap(a.dbeta          ,b.dbeta          );
    qswap(a.dalphaGrad     ,b.dalphaGrad     );
    qswap(a.dbetaGrad      ,b.dbetaGrad      );
    qswap(a.dalphaRestrict ,b.dalphaRestrict );
    qswap(a.dbetaRestrict  ,b.dbetaRestrict  );
    qswap(a.dopttol        ,b.dopttol        );
    qswap(a.probContext    ,b.probContext    );
    qswap(a.alphagradstate ,b.alphagradstate );
    qswap(a.betagradstate  ,b.betagradstate  );
    qswap(a.gradFixAlphaInd,b.gradFixAlphaInd);
    qswap(a.gradFixBetaInd ,b.gradFixBetaInd );
    qswap(a.gradFixAlpha   ,b.gradFixAlpha   );
    qswap(a.gradFixBeta    ,b.gradFixBeta    );
    qswap(a.cumgraderr     ,b.cumgraderr     );

    return;
}


template <class T, class S>
optState<T,S>::optState(void)
{
    dopttol = DEFAULT_OPTTOL;

    alphagradstate = -2;
    betagradstate  = -2;

    gradFixAlphaInd = 0;
    gradFixBetaInd  = 0;

    cumgraderr = 0;

    return;
}

template <class T, class S>
optState<T,S>::optState(const optState<T,S> &src)
{
    *this = src;

    return;
}

template <class T, class S>
optState<T,S> &optState<T,S>::operator=(const optState<T,S> &src)
{
    dalpha     = src.dalpha;
    dbeta      = src.dbeta;
    dalphaGrad = src.dalphaGrad;
    dbetaGrad  = src.dbetaGrad;

    dalphaRestrict = src.dalphaRestrict;
    dbetaRestrict  = src.dbetaRestrict;

    dopttol = src.dopttol;

    probContext = src.probContext;

    alphagradstate = src.alphagradstate;
    betagradstate  = src.betagradstate;

    gradFixAlphaInd = src.gradFixAlphaInd;
    gradFixBetaInd  = src.gradFixBetaInd;
    gradFixAlpha    = src.gradFixAlpha;
    gradFixBeta     = src.gradFixBeta;

    cumgraderr = src.cumgraderr;

    return *this;
}

template <class T, class S>
void optState<T,S>::setAlpha(const Vector<T> &newAlpha, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &lb, const Vector<double> &ub, double ztoloverride)
{
    NiceAssert( newAlpha.size() == aN() );

    ztoloverride = ( ztoloverride >= 0 ) ? ztoloverride : zerotol();

    if ( aN() )
    {
	int i,iP;

	for ( i = 0 ; i < aN() ; i++ )
	{
	    alphaStep(i,newAlpha(i)-dalpha(i),GpGrad,Gn,Gpn,gp,gn,hp,1);

	    if ( alphaState(i) == -2 )
	    {
		iP = findInAlphaLB(i);

		if ( newAlpha(i) > ub(i)-ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaLBtoUB(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( ( newAlpha(i) >= ztoloverride ) && ( newAlpha(i) > 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaLBtoUF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( ( newAlpha(i) > -ztoloverride ) || ( newAlpha(i) == 0.0 ) )
		{
		    iP = modAlphaLBtoZ(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( ( newAlpha(i) >= lb(i)+ztoloverride ) && ( newAlpha(i) > lb(i) ) )
		{
		    iP = modAlphaLBtoLF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}
	    }

	    else if ( alphaState(i) == +2 )
	    {
		iP = findInAlphaUB(i);

		if ( newAlpha(i) < lb(i)+ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaUBtoLB(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( ( newAlpha(i) <= -ztoloverride ) && ( newAlpha(i) < 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaUBtoLF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( ( newAlpha(i) < ztoloverride ) || ( newAlpha(i) == 0.0 ) )
		{
		    iP = modAlphaUBtoZ(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( ( newAlpha(i) <= ub(i)-ztoloverride ) && ( newAlpha(i) < ub(i) ) )
		{
		    iP = modAlphaUBtoUF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}
	    }

	    else if ( alphaState(i) == -1 )
	    {
		iP = findInAlphaF(i);

		if ( newAlpha(i) > ub(i)-ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaLFtoUB(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp,ub);
		}

		else if ( ( newAlpha(i) >= ztoloverride ) && ( newAlpha(i) > 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaLFtoUF(iP,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( ( newAlpha(i) > -ztoloverride ) || ( newAlpha(i) == 0.0 ) )
		{
		    iP = modAlphaLFtoZ(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( newAlpha(i) < lb(i)+ztoloverride )
		{
		    iP = modAlphaLFtoLB(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp,lb);
		}
	    }

	    else if ( alphaState(i) == +1 )
	    {
		iP = findInAlphaF(i);

		if ( newAlpha(i) < lb(i)+ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaUFtoLB(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp,lb);
		}

		else if ( ( newAlpha(i) <= -ztoloverride ) && ( newAlpha(i) < 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaUFtoLF(iP,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( ( newAlpha(i) < ztoloverride ) || ( newAlpha(i) == 0.0 ) )
		{
		    iP = modAlphaUFtoZ(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( newAlpha(i) > ub(i)-ztoloverride )
		{
		    iP = modAlphaUFtoUB(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp,ub);
		}
	    }

	    else if ( alphaState(i) == 0 )
	    {
		iP = findInAlphaZ(i);

		if ( newAlpha(i) < lb(i)+ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaZtoLB(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( newAlpha(i) > ub(i)-ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaZtoUB(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( ( newAlpha(i) <= -ztoloverride ) && ( newAlpha(i) < 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaZtoLF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}

		else if ( ( newAlpha(i) >= ztoloverride ) && ( newAlpha(i) > 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

		    iP = modAlphaZtoUF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::setBeta(const Vector<T> &newBeta, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, double ztoloverride)
{
    NiceAssert( newBeta.size() == bN() );

    ztoloverride = ( ztoloverride >= 0 ) ? ztoloverride : zerotol();

    if ( bN() )
    {
	int i,iP;

	for ( i = 0 ; i < bN() ; i++ )
	{
	    if ( ( betaState(i) == 0 ) && ( ( newBeta(i) >= ztoloverride ) || ( newBeta(i) <= -ztoloverride ) ) )
	    {
		iP = findInBetaC(i);
		iP = modBetaCtoF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
	    }

	    if ( newBeta(i) >= ztoloverride )
	    {
                NiceAssert( ( dbetaRestrict(i) != 2 ) && ( dbetaRestrict(i) != 3 ) );

		betaStep(i,newBeta(i)-dbeta(i),GpGrad,Gn,Gpn,gp,gn,hp);
	    }

	    else if ( newBeta(i) <= -ztoloverride )
	    {
                NiceAssert( ( dbetaRestrict(i) != 1 ) && ( dbetaRestrict(i) != 3 ) );

		betaStep(i,newBeta(i)-dbeta(i),GpGrad,Gn,Gpn,gp,gn,hp);
	    }

	    else
	    {
		betaStep(i,newBeta(i)-dbeta(i),GpGrad,Gn,Gpn,gp,gn,hp,1);
	    }
	}
    }

    return;
}

inline int isTVector(const Vector<double> &temp);
inline int isTVector(const Vector<double> &temp)
{
    (void) temp;

    return 1;
}

inline int isTVector(const double &temp);
inline int isTVector(const double &temp)
{
    (void) temp;

    return 0;
}

template <class T, class S>
void optState<T,S>::setAlphahpzero(const Vector<T> &newAlpha, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &lb, const Vector<double> &ub, double ztoloverride)
{
    NiceAssert( newAlpha.size() == aN() );

    ztoloverride = ( ztoloverride >= 0 ) ? ztoloverride : zerotol();

    T dummy;

    // if isTVector(dummy) set then T is vectorial type and should not be
    // compared to scalars.  Everything is either Z or UF in this case

    if ( aN() )
    {
	int i,iP;

	for ( i = 0 ; i < aN() ; i++ )
	{
            alphaStephpzero(i,newAlpha(i)-dalpha(i),GpGrad,Gn,Gpn,gp,gn,1);

	    if ( alphaState(i) == -2 )
	    {
                NiceAssert( !isTVector(dummy) );

		iP = findInAlphaLB(i);

		if ( newAlpha(i) > ub(i)-ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaLBtoUBhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( ( newAlpha(i) >= ztoloverride ) && ( newAlpha(i) > 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaLBtoUFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( ( newAlpha(i) > -ztoloverride ) || ( newAlpha(i) == 0.0 ) )
		{
                    iP = modAlphaLBtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( ( newAlpha(i) >= lb(i)+ztoloverride ) && ( newAlpha(i) > lb(i) ) )
		{
                    iP = modAlphaLBtoLFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}
	    }

	    else if ( alphaState(i) == +2 )
	    {
                NiceAssert( !isTVector(dummy) );

		iP = findInAlphaUB(i);

		if ( newAlpha(i) < lb(i)+ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaUBtoLBhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( ( newAlpha(i) <= -ztoloverride ) && ( newAlpha(i) < 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaUBtoLFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( ( newAlpha(i) < ztoloverride ) || ( newAlpha(i) == 0.0 ) )
		{
                    iP = modAlphaUBtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( ( newAlpha(i) <= ub(i)-ztoloverride ) && ( newAlpha(i) < ub(i) ) )
		{
                    iP = modAlphaUBtoUFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}
	    }

	    else if ( alphaState(i) == -1 )
	    {
                NiceAssert( !isTVector(dummy) );

		iP = findInAlphaF(i);

		if ( newAlpha(i) > ub(i)-ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaLFtoUBhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn,ub);
		}

		else if ( ( newAlpha(i) >= ztoloverride ) && ( newAlpha(i) > 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaLFtoUFhpzero(iP,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( ( newAlpha(i) > -ztoloverride ) || ( newAlpha(i) == 0.0 ) )
		{
                    iP = modAlphaLFtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( newAlpha(i) < lb(i)+ztoloverride )
		{
                    iP = modAlphaLFtoLBhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn,lb);
		}
	    }

	    else if ( alphaState(i) == +1 )
	    {
		iP = findInAlphaF(i);

                if ( isTVector(dummy) )
                {
                    if ( abs2(newAlpha(i)) <= ztoloverride )
                    {
                        iP = modAlphaUFtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
                    }
                }

                else if ( newAlpha(i) < lb(i)+ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaUFtoLBhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn,lb);
		}

		else if ( ( newAlpha(i) <= -ztoloverride ) && ( newAlpha(i) < 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaUFtoLFhpzero(iP,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( ( newAlpha(i) < ztoloverride ) || ( newAlpha(i) == 0.0 ) )
		{
                    iP = modAlphaUFtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( newAlpha(i) > ub(i)-ztoloverride )
		{
                    iP = modAlphaUFtoUBhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn,ub);
		}
	    }

	    else if ( alphaState(i) == 0 )
	    {
		iP = findInAlphaZ(i);

                if ( isTVector(dummy) )
                {
                    if ( abs2(newAlpha(i)) > ztoloverride )
                    {
                        iP = modAlphaZtoUFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
                    }
                }

                else if ( newAlpha(i) < lb(i)+ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaZtoLBhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( newAlpha(i) > ub(i)-ztoloverride )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaZtoUBhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( ( newAlpha(i) <= -ztoloverride ) && ( newAlpha(i) < 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 1 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaZtoLFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}

		else if ( ( newAlpha(i) >= ztoloverride ) && ( newAlpha(i) > 0.0 ) )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaZtoUFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::setBetahpzero(const Vector<T> &newBeta, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, double ztoloverride)
{
    NiceAssert( newBeta.size() == bN() );

    ztoloverride = ( ztoloverride >= 0 ) ? ztoloverride : zerotol();

    if ( bN() )
    {
	int i,iP;

	for ( i = 0 ; i < bN() ; i++ )
	{
	    if ( ( betaState(i) == 0 ) && ( ( newBeta(i) >= ztoloverride ) || ( newBeta(i) <= -ztoloverride ) ) )
	    {
		iP = findInBetaC(i);
                iP = modBetaCtoFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
	    }

	    if ( newBeta(i) >= ztoloverride )
	    {
                NiceAssert( ( dbetaRestrict(i) != 2 ) && ( dbetaRestrict(i) != 3 ) );

                betaStephpzero(i,newBeta(i)-dbeta(i),GpGrad,Gn,Gpn,gp,gn);
	    }

	    else if ( newBeta(i) <= -ztoloverride )
	    {
                NiceAssert( ( dbetaRestrict(i) != 1 ) && ( dbetaRestrict(i) != 3 ) );

                betaStephpzero(i,newBeta(i)-dbeta(i),GpGrad,Gn,Gpn,gp,gn);
	    }

	    else
	    {
                betaStephpzero(i,newBeta(i)-dbeta(i),GpGrad,Gn,Gpn,gp,gn,1);
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::refact(const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );

    probContext.refact(Gp,Gn,Gpn,keepfact(),zerotol());
    refactlin(GpGrad,Gn,Gpn,gp,gn,hp);

    return;
}

template <class T, class S>
void optState<T,S>::refacthpzero(const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    probContext.refact(Gp,Gn,Gpn,keepfact(),zerotol());
    refactlinhpzero(GpGrad,Gn,Gpn,gp,gn);

    return;
}

template <class T, class S>
void optState<T,S>::refactGpn(const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &GpnOld, const Matrix<double> &GpnNew, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == GpnOld.numRows() );
    NiceAssert( GpGrad.numRows() == GpnOld.numRows() );
    NiceAssert( Gn.numCols() == GpnOld.numCols() );
    NiceAssert( Gp.numRows() == GpnNew.numRows() );
    NiceAssert( Gn.numCols() == GpnNew.numCols() );
    NiceAssert( GpnOld.numRows() == aN() );
    NiceAssert( GpnOld.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );

    probContext.refact(Gp,Gn,GpnOld,keepfact(),zerotol());

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,GpnOld,gp,gn,hp);
    }

    int i,j,iP;

    if ( bN() && aN() )
    {
        T temp;

	for ( j = 0 ; j < aN() ; j++ )
	{
            retVector<T> tmpva;

            dalphaGrad("&",j) -= twoProductNoConj(temp,GpnOld(j,tmpva),dbeta);
            dalphaGrad("&",j) += twoProductNoConj(temp,GpnNew(j,tmpva),dbeta);

	    for ( i = 0 ; i < bN() ; i++ )
	    {
                dbetaGrad("&",i) -= GpnOld(j,i)*dalpha(j);
                dbetaGrad("&",i) -= GpnNew(j,i)*dalpha(j);
	    }
	}
    }

    if ( keepfact() )
    {
	// Fix gradient state

	alphagradstate = -2;
	betagradstate  = -1;

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::refactGpnhpzero(const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &GpnOld, const Matrix<double> &GpnNew, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == GpnOld.numRows() );
    NiceAssert( GpGrad.numRows() == GpnOld.numRows() );
    NiceAssert( Gn.numCols() == GpnOld.numCols() );
    NiceAssert( Gp.numRows() == GpnNew.numRows() );
    NiceAssert( Gn.numCols() == GpnNew.numCols() );
    NiceAssert( GpnOld.numRows() == aN() );
    NiceAssert( GpnOld.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    probContext.refact(Gp,Gn,GpnOld,keepfact(),zerotol());

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
        fixGradhpzero(GpGrad,Gn,GpnOld,gp,gn);
    }

    int i,j,iP;

    if ( bN() && aN() )
    {
        T temp;

        for ( j = 0 ; j < aN() ; j++ )
	{
            retVector<T> tmpva;

            dalphaGrad("&",j) -= twoProductNoConj(temp,GpnOld(j,tmpva),dbeta);
            dalphaGrad("&",j) += twoProductNoConj(temp,GpnNew(j,tmpva),dbeta);

	    for ( i = 0 ; i < bN() ; i++ )
	    {
                dbetaGrad("&",i) -= GpnOld(j,i)*dalpha(j);
                dbetaGrad("&",i) -= GpnNew(j,i)*dalpha(j);
	    }
	}
    }

    if ( keepfact() )
    {
	// Fix gradient state

	alphagradstate = -2;
	betagradstate  = -1;

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::refactlin(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );

    int i,iP;

    gradFixAlphaInd = 0;
    gradFixBetaInd  = 0;
    gradFixAlpha.zero();
    gradFixBeta.zero();

    // Fix gradients

    if ( aN() )
    {
	for ( i = 0 ; i < aN() ; i++ )
	{
            recalcAlphaGrad(dalphaGrad("&",i),GpGrad,Gpn,gp,hp,i);
	}
    }

    if ( bN() )
    {
	for ( i = 0 ; i < bN() ; i++ )
	{
            recalcBetaGrad(dbetaGrad("&",i),Gn,Gpn,gn,i);
	}
    }

    if ( keepfact() )
    {
	// Fix gradient state

	alphagradstate = -2;
	betagradstate  = -1;

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    cumgraderr = 0;

    return;
}

template <class T, class S>
void optState<T,S>::refactlinhpzero(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    int i,iP;

    gradFixAlphaInd = 0;
    gradFixBetaInd  = 0;
    gradFixAlpha.zero();
    gradFixBeta.zero();

    // Fix gradients

    if ( aN() )
    {
	for ( i = 0 ; i < aN() ; i++ )
	{
            recalcAlphaGradhpzero(dalphaGrad("&",i),GpGrad,Gpn,gp,i);
	}
    }

    if ( bN() )
    {
	for ( i = 0 ; i < bN() ; i++ )
	{
            recalcBetaGrad(dbetaGrad("&",i),Gn,Gpn,gn,i);
	}
    }

    if ( keepfact() )
    {
	// Fix gradient state

	alphagradstate = -2;
	betagradstate  = -1;

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    cumgraderr = 0;

    return;
}

template <class T, class S>
void optState<T,S>::refactgp(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gpOld, const Vector<T> &gpNew, const Vector<T> &gn, const Vector<T> &hp, int iv)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gpOld.size() == aN() );
    NiceAssert( gpNew.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( ( ( iv >= 0 ) && ( iv < aN() ) ) || ( iv == -1 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gpOld,gn,hp);
    }

    if ( iv == -1 )
    {
	dalphaGrad -= gpOld;
	dalphaGrad += gpNew;

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    else
    {
	dalphaGrad("&",iv) -= gpOld(iv);
	dalphaGrad("&",iv) += gpNew(iv);

	if ( keepfact() )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = iv;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::refactgphpzero(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gpOld, const Vector<T> &gpNew, const Vector<T> &gn, int iv)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gpOld.size() == aN() );
    NiceAssert( gpNew.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( ( ( iv >= 0 ) && ( iv < aN() ) ) || ( iv == -1 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
        fixGradhpzero(GpGrad,Gn,Gpn,gpOld,gn);
    }

    if ( iv == -1 )
    {
	dalphaGrad -= gpOld;
	dalphaGrad += gpNew;

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    else
    {
	dalphaGrad("&",iv) -= gpOld(iv);
	dalphaGrad("&",iv) += gpNew(iv);

	if ( keepfact() )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = iv;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::refactGpnElm(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &GpnOld, double GpnijNew, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int i, int j)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == GpnOld.numRows() );
    NiceAssert( Gn.numCols() == GpnOld.numCols() );
    NiceAssert( GpnOld.numRows() == aN() );
    NiceAssert( GpnOld.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( ( i >= 0 ) && ( i < aN() ) );
    NiceAssert( ( j >= 0 ) && ( j < bN() ) );
    NiceAssert( alphaState(i) == 0 );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,GpnOld,gp,gn,hp);
    }

    // Gpn(i,j) is not in the factorisation, so no need to worry about
    // optcontext.  Moreover alpha(i) == 0, so no need to worry about
    // beta gradients.

    dalphaGrad("&",i) -= (GpnOld(i,j)*dbeta(j));
    dalphaGrad("&",i) += (GpnijNew*dbeta(j));

    if ( keepfact() )
    {
	if ( alphagradstate == -1 )
	{
	    alphagradstate = i;
	}

	else if ( alphagradstate >= 0 )
	{
	    alphagradstate = -2;
	}
    }

    return;
}

template <class T, class S>
int optState<T,S>::fact_snookered(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn)
{
    int res = 0;

    if ( ( bN() != probContext.fact_nfact(Gn,Gpn) ) && bN() )
    {
	int i;

	for ( i = 0 ; i < bN() ; i++ )
	{
            i++; //FIXME: is this correct???
	}
    }

    return res;
}


template <class T, class S>
void optState<T,S>::factstepgp(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gpOld, const T &gpistep, const Vector<T> &gn, const Vector<T> &hp, int iv)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gpOld.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( ( iv >= 0 ) && ( iv < aN() ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gpOld,gn,hp);
    }

    dalphaGrad("&",iv) += gpistep;

    if ( keepfact() )
    {
	if ( alphagradstate == -1 )
	{
	    alphagradstate = iv;
	}

	else if ( alphagradstate >= 0 )
	{
	    alphagradstate = -2;
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::refactgn(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gnOld, const Vector<T> &gnNew, const Vector<T> &hp, int iv)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gnOld.size() == bN() );
    NiceAssert( gnNew.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( ( ( iv >= 0 ) && ( iv < bN() ) ) || ( iv == -1 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gnOld,hp);
    }

    if ( iv == -1 )
    {
	dbetaGrad -= gnOld;
	dbetaGrad += gnNew;
    }

    else
    {
	dbetaGrad("&",iv) -= gnOld(iv);
	dbetaGrad("&",iv) += gnNew(iv);
    }

    if ( keepfact() )
    {
	betagradstate  = -1;

	if ( bNF() )
	{
	    int iP;

	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::refactgnhpzero(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gnOld, const Vector<T> &gnNew, int iv)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gnOld.size() == bN() );
    NiceAssert( gnNew.size() == bN() );
    NiceAssert( ( ( iv >= 0 ) && ( iv < bN() ) ) || ( iv == -1 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gnOld);
    }

    if ( iv == -1 )
    {
	dbetaGrad -= gnOld;
	dbetaGrad += gnNew;
    }

    else
    {
	dbetaGrad("&",iv) -= gnOld(iv);
	dbetaGrad("&",iv) += gnNew(iv);
    }

    if ( keepfact() )
    {
	betagradstate  = -1;

	if ( bNF() )
	{
	    int iP;

	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::factstepgn(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gnOld, const T &gnistep, const Vector<T> &hp, int iv)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gnOld.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( ( iv >= 0 ) && ( iv < bN() ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gnOld,hp);
    }

    dbetaGrad("&",iv) += gnistep;

    if ( keepfact() )
    {
	betagradstate  = -1;

	if ( bNF() )
	{
	    int iP;

	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::refacthp(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hpOld, const Vector<T> &hpNew, int iv)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hpOld.size() == aN() );
    NiceAssert( hpNew.size() == aN() );
    NiceAssert( ( ( iv >= 0 ) && ( iv < aN() ) ) || ( iv == -1 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hpOld);
    }

    if ( iv == -1 )
    {
	if ( aN() )
	{
	    int i;

	    for ( i = 0 ; i < aN() ; i++ )
	    {
		if ( alphaState(i) > 0 )
		{
		    dalphaGrad("&",i) -= hpOld(i);
		    dalphaGrad("&",i) += hpNew(i);
		}

		else if ( alphaState(i) < 0 )
		{
		    dalphaGrad("&",i) += hpOld(i);
		    dalphaGrad("&",i) -= hpNew(i);
		}
	    }
	}

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    else
    {
	if ( alphaState(iv) > 0 )
	{
	    dalphaGrad("&",iv) -= hpOld(iv);
	    dalphaGrad("&",iv) += hpNew(iv);

	    if ( keepfact() )
	    {
		if ( alphagradstate == -1 )
		{
		    alphagradstate = iv;
		}

		else if ( alphagradstate >= 0 )
		{
		    alphagradstate = -2;
		}
	    }
	}

	else if ( alphaState(iv) < 0 )
	{
	    dalphaGrad("&",iv) += hpOld(iv);
	    dalphaGrad("&",iv) -= hpNew(iv);

	    if ( keepfact() )
	    {
		if ( alphagradstate == -1 )
		{
		    alphagradstate = iv;
		}

		else if ( alphagradstate >= 0 )
		{
		    alphagradstate = -2;
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::reset(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    probContext.reset(Gp,Gn,Gpn);

    dalpha.zero();
    dbeta.zero();

    dalphaGrad = gp;
    dbetaGrad  = gn;

    int iP;

    retVector<T> tmpva;
    retVector<T> tmpvb;

    dalphaGrad("&",pivAlphaUB(),tmpva) += hp(pivAlphaUB(),tmpvb);
    dalphaGrad("&",pivAlphaLB(),tmpva) -= hp(pivAlphaLB(),tmpvb);

    if ( aNF() )
    {
	for ( iP = 0 ; iP < aNF() ; iP++ )
	{
	    if ( alphaState(pivAlphaF()(iP)) > 0 )
	    {
		dalphaGrad("&",pivAlphaF()(iP)) += hp(pivAlphaF()(iP));
	    }

            else
	    {
		dalphaGrad("&",pivAlphaF()(iP)) -= hp(pivAlphaF()(iP));
	    }
	}
    }

    cumgraderr = 0.0;

    gradFixAlphaInd = 0;
    gradFixBetaInd  = 0;

    gradFixAlpha.zero();
    gradFixBeta.zero();

    if ( keepfact() )
    {
	// Fix gradient state

	alphagradstate = -2;
	betagradstate  = -1;

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::resethpzero(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    probContext.reset(Gp,Gn,Gpn);

    dalpha.zero();
    dbeta.zero();

    dalphaGrad = gp;
    dbetaGrad  = gn;

    int iP;

    cumgraderr = 0.0;

    gradFixAlphaInd = 0;
    gradFixBetaInd  = 0;

    gradFixAlpha.zero();
    gradFixBeta.zero();

    if ( keepfact() )
    {
	// Fix gradient state

	alphagradstate = -2;
	betagradstate  = -1;

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::setopttol(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, double xopttol)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( xopttol > 0 );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int iP;

    dopttol = xopttol;

    if ( keepfact() )
    {
	alphagradstate = -1;
	betagradstate  = -1;

	if ( aNF() )
	{
	    for ( iP = 0 ; iP < aNF() ; iP++ )
	    {
                if ( abs2(dalphaGrad(pivAlphaF()(iP))) > dopttol )
		{
		    if ( alphagradstate == -1 )
		    {
			alphagradstate = iP;
		    }

		    else if ( alphagradstate >= 0 )
		    {
			alphagradstate = -2;

			break;
		    }
		}
	    }
	}

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::setopttolhpzero(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, double xopttol)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( xopttol > 0 );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
        fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int iP;

    dopttol = xopttol;

    if ( keepfact() )
    {
	alphagradstate = -1;
	betagradstate  = -1;

	if ( aNF() )
	{
	    for ( iP = 0 ; iP < aNF() ; iP++ )
	    {
                if ( abs2(dalphaGrad(pivAlphaF()(iP))) > dopttol )
		{
		    if ( alphagradstate == -1 )
		    {
			alphagradstate = iP;
		    }

		    else if ( alphagradstate >= 0 )
		    {
			alphagradstate = -2;

			break;
		    }
		}
	    }
	}

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::setzt(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, double xzt)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( xzt > 0 );

    probContext.refact(Gp,Gn,Gpn,keepfact(),xzt);

    return;
}

template <class T, class S>
void optState<T,S>::setkeepfact(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int xkeepfact)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( ( xkeepfact == 0 ) || ( xkeepfact == 1 ) );

    if ( !keepfact() && xkeepfact )
    {
	// Fix gradient state

	int iP;

	alphagradstate = -2;
	betagradstate  = -1;

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    if ( keepfact() != xkeepfact )
    {
	probContext.refact(Gp,Gn,Gpn,xkeepfact,zerotol());
    }

    return;
}

template <class T, class S>
void optState<T,S>::refreshGrad_anyhow(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int i)
{
    if ( i == -1 )
    {
        cumgraderr = 10 +  CUMGRADTRIGGER*dopttol;

        refreshGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    else
    {
        int iP;

        recalcAlphaGrad(dalphaGrad("&",i),GpGrad,Gpn,gp,hp,i);

	if ( keepfact() )
	{
	    // Fix gradient state

	    alphagradstate = -2;
	    betagradstate  = -1;

	    if ( bNF() )
	    {
		for ( iP = 0 ; iP < bNF() ; iP++ )
		{
                    if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		    {
			if ( betagradstate == -1 )
			{
			    betagradstate = iP;
			}

			else if ( betagradstate >= 0 )
			{
			    betagradstate = -2;

			    break;
			}
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::refreshGrad(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );

    int i,iP;

    if ( cumgraderr > CUMGRADTRIGGER*dopttol )
    {
	// Fix gradients

	if ( aN() )
	{
	    for ( i = 0 ; i < aN() ; i++ )
	    {
		recalcAlphaGrad(dalphaGrad("&",i),GpGrad,Gpn,gp,hp,i);
	    }
	}

	if ( bN() )
	{
	    for ( i = 0 ; i < bN() ; i++ )
	    {
		recalcBetaGrad(dbetaGrad("&",i),Gn,Gpn,gn,i);
	    }
	}

	if ( keepfact() )
	{
	    // Fix gradient state

	    alphagradstate = -2;
	    betagradstate  = -1;

	    if ( bNF() )
	    {
		for ( iP = 0 ; iP < bNF() ; iP++ )
		{
                    if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		    {
			if ( betagradstate == -1 )
			{
			    betagradstate = iP;
			}

			else if ( betagradstate >= 0 )
			{
			    betagradstate = -2;

			    break;
			}
		    }
		}
	    }
	}

	cumgraderr = 0;
    }

    else if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    return;
}

template <class T, class S>
void optState<T,S>::refreshGradhpzero_anyhow(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int i)
{
    if ( i == -1 )
    {
        cumgraderr = 10 +  CUMGRADTRIGGER*dopttol;

        refreshGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    else
    {
        int iP;

        recalcAlphaGradhpzero(dalphaGrad("&",i),GpGrad,Gpn,gp,i);

	if ( keepfact() )
	{
	    // Fix gradient state

	    alphagradstate = -2;
	    betagradstate  = -1;

	    if ( bNF() )
	    {
		for ( iP = 0 ; iP < bNF() ; iP++ )
		{
                    if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		    {
			if ( betagradstate == -1 )
			{
			    betagradstate = iP;
			}

			else if ( betagradstate >= 0 )
			{
			    betagradstate = -2;

			    break;
			}
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::refreshGradhpzero(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    int i,iP;

    if ( cumgraderr > CUMGRADTRIGGER*dopttol )
    {
	// Fix gradients

	if ( aN() )
	{
	    for ( i = 0 ; i < aN() ; i++ )
	    {
		recalcAlphaGradhpzero(dalphaGrad("&",i),GpGrad,Gpn,gp,i);
	    }
	}

	if ( bN() )
	{
	    for ( i = 0 ; i < bN() ; i++ )
	    {
		recalcBetaGrad(dbetaGrad("&",i),Gn,Gpn,gn,i);
	    }
	}

	if ( keepfact() )
	{
	    // Fix gradient state

	    alphagradstate = -2;
	    betagradstate  = -1;

	    if ( bNF() )
	    {
		for ( iP = 0 ; iP < bNF() ; iP++ )
		{
                    if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		    {
			if ( betagradstate == -1 )
			{
			    betagradstate = iP;
			}

			else if ( betagradstate >= 0 )
			{
			    betagradstate = -2;

			    break;
			}
		    }
		}
	    }
	}

	cumgraderr = 0;
    }

    else if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    return;
}


template <class T, class S>
void optState<T,S>::scalehpzero(double a, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( a >= 0 );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    dalpha.scale(a);
    dbeta.scale(a);



    if ( aN() )
    {
	dalphaGrad -= gp;

	dalphaGrad.scale(a);

        dalphaGrad += gp;

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    if ( bN() )
    {
	dbetaGrad -= gn;
	dbetaGrad.scale(a);
	dbetaGrad += gn;

	if ( keepfact() )
	{
	    betagradstate  = -1;

	    if ( bNF() )
	    {
		int iP;

		for ( iP = 0 ; iP < bNF() ; iP++ )
		{
                    if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		    {
			if ( betagradstate == -1 )
			{
			    betagradstate = iP;
			}

			else if ( betagradstate >= 0 )
			{
			    betagradstate = -2;

			    break;
			}
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::scale(double a, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( a >= 0 );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalpha.scale(a);
    dbeta.scale(a);

    int i;

    if ( aN() )
    {
	dalphaGrad -= gp;

	for ( i = 0 ; i < aN() ; i++ )
	{
	    if ( alphaState(i) > 0 )
	    {
		dalphaGrad("&",i) -= hp(i);
	    }

	    else if ( alphaState(i) < 0 )
	    {
		dalphaGrad("&",i) += hp(i);
	    }
	}

	dalphaGrad.scale(a);

        dalphaGrad += gp;

	for ( i = 0 ; i < aN() ; i++ )
	{
	    if ( alphaState(i) > 0 )
	    {
		dalphaGrad("&",i) += hp(i);
	    }

	    else if ( alphaState(i) < 0 )
	    {
		dalphaGrad("&",i) -= hp(i);
	    }
	}

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    if ( bN() )
    {
	dbetaGrad -= gn;
	dbetaGrad.scale(a);
	dbetaGrad += gn;

	if ( keepfact() )
	{
	    betagradstate  = -1;

	    if ( bNF() )
	    {
		int iP;

		for ( iP = 0 ; iP < bNF() ; iP++ )
		{
                    if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		    {
			if ( betagradstate == -1 )
			{
			    betagradstate = iP;
			}

			else if ( betagradstate >= 0 )
			{
			    betagradstate = -2;

			    break;
			}
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::scaleAlphahpzero(double a, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( a >= 0 );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    dalpha.scale(a);

    int i;

    if ( aN() )
    {
	dalphaGrad -= gp;

        T temp;

	for ( i = 0 ; i < aN() ; i++ )
	{
            dalphaGrad("&",i) -= twoProductNoConj(temp,Gpn(i,pivBetaF()),dbeta(pivBetaF()));
	}

	dalphaGrad.scale(a);

        dalphaGrad += gp;

	for ( i = 0 ; i < aN() ; i++ )
	{
            dalphaGrad("&",i) += twoProductNoConj(temp,Gpn(i,pivBetaF()),dbeta(pivBetaF()));
	}

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::scaleAlpha(double a, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( a >= 0 );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalpha.scale(a);

    int i;

    if ( aN() )
    {
	dalphaGrad -= gp;

        T temp;

	for ( i = 0 ; i < aN() ; i++ )
	{
	    if ( alphaState(i) > 0 )
	    {
		dalphaGrad("&",i) -= hp(i);
	    }

	    else if ( alphaState(i) < 0 )
	    {
		dalphaGrad("&",i) += hp(i);
	    }

            dalphaGrad("&",i) -= twoProductNoConj(temp,Gpn(i,pivBetaF()),dbeta(pivBetaF()));
	}

	dalphaGrad.scale(a);

        dalphaGrad += gp;

	for ( i = 0 ; i < aN() ; i++ )
	{
	    if ( alphaState(i) > 0 )
	    {
		dalphaGrad("&",i) += hp(i);
	    }

	    else if ( alphaState(i) < 0 )
	    {
		dalphaGrad("&",i) -= hp(i);
	    }

            dalphaGrad("&",i) += twoProductNoConj(temp,Gpn(i,pivBetaF()),dbeta(pivBetaF()));
	}

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::rankone(const Vector<double> &bp, const Vector<double> &bn, const double &c, const Vector<double> &bpGrad, const Vector<double> &bnGrad, const double &cGrad, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( bp.size() == aN() );
    NiceAssert( bn.size() == bN() );
    NiceAssert( bpGrad.size() == aN() );
    NiceAssert( bnGrad.size() == bN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    if ( ( aNZ() < aN() ) || ( bNC() < bN() ) )
    {
	T inner;
        T temp;

        inner  = twoProductNoConj(temp,bpGrad(pivAlphaLB()),dalpha(pivAlphaLB()));
        inner += twoProductNoConj(temp,bpGrad(pivAlphaF ()),dalpha(pivAlphaF ()));
        inner += twoProductNoConj(temp,bpGrad(pivAlphaUB()),dalpha(pivAlphaUB()));
        inner += twoProductNoConj(temp,bnGrad(pivBetaF  ()),dbeta (pivBetaF  ()));
        inner *= cGrad;

	dalphaGrad.scaleAddB(inner,bpGrad);
	dbetaGrad.scaleAddB(inner,bnGrad);

	if ( keepfact() )
	{
	    alphagradstate = -2;
	    betagradstate  = -1;

	    int iP;

	    if ( bNF() )
	    {
		for ( iP = 0 ; iP < bNF() ; iP++ )
		{
                    if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		    {
			if ( betagradstate == -1 )
			{
			    betagradstate = iP;
			}

			else if ( betagradstate >= 0 )
			{
			    betagradstate = -2;

			    break;
			}
		    }
		}
	    }
	}
    }

    if ( keepfact() )
    {
	probContext.fact_rankone(bp,bn,c,Gp,Gn,Gpn,alphagradstate,betagradstate);
    }

    return;
}

template <class T, class S>
void optState<T,S>::diagmult(const Vector<double> &bp, const Vector<double> &bn, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( bp.size() == aN() );
    NiceAssert( bn.size() == bN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int i,iP;

    if ( aNZ() < aN() )
    {
	for ( i = 0 ; i < aN() ; i++ )
	{
            recalcAlphaGrad(dalphaGrad("&",i),GpGrad,Gpn,gp,hp,i);
	}
    }

    if ( bNC() < bN() )
    {
	for ( i = 0 ; i < bN() ; i++ )
	{
            recalcBetaGrad(dbetaGrad("&",i),Gn,Gpn,gn,i);
	}
    }

    if ( keepfact() )
    {
	probContext.fact_diagmult(bp,bn,alphagradstate,betagradstate);
    }

    if ( keepfact() )
    {
	alphagradstate = -2;
	betagradstate  = -1;

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    cumgraderr = 0;

    return;
}

template <class T, class S>
void optState<T,S>::diagoffset(const Vector<double> &bp, const Vector<double> &bn, const Vector<double> &bpGrad, const Vector<double> &bnGrad, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( bp.size() == aN() );
    NiceAssert( bn.size() == bN() );
    NiceAssert( bpGrad.size() == aN() );
    NiceAssert( bnGrad.size() == bN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    if ( aNZ() < aN() )
    {
	Vector<T> upvectp(dalpha);

	if ( upvectp.size() )
	{
	    int i;

	    for ( i = 0 ; i < upvectp.size() ; i++ )
	    {
                upvectp("&",i) *= bpGrad(i);
	    }
	}

	dalphaGrad += upvectp;

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    if ( bNC() < bN() )
    {
	Vector<T> upvectn(dbeta);

	if ( upvectn.size() )
	{
	    int i;

	    for ( i = 0 ; i < upvectn.size() ; i++ )
	    {
                upvectn("&",i) *= bnGrad(i);
	    }
	}

	dbetaGrad += upvectn;

	if ( keepfact() )
	{
	    betagradstate  = -1;

	    int iP;

	    if ( bNF() )
	    {
		for ( iP = 0 ; iP < bNF() ; iP++ )
		{
                    if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		    {
			if ( betagradstate == -1 )
			{
			    betagradstate = iP;
			}

			else if ( betagradstate >= 0 )
			{
			    betagradstate = -2;

			    break;
			}
		    }
		}
	    }
	}
    }

    if ( keepfact() )
    {
	probContext.fact_diagoffset(bp,bn,Gp,Gn,Gpn,alphagradstate,betagradstate);
    }

    return;
}

template <class T, class S>
void optState<T,S>::diagoffset(int i, double bpoff, double bpoffGrad, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i <= aN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    if ( alphaState(i) != 0 )
    {
	dalphaGrad("&",i) += bpoffGrad*dalpha(i);

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    if ( keepfact() )
    {
	Vector<double> bp(aN());
	Vector<double> bn(bN());

	bp.zero();
	bn.zero();

        bp("&",i) = bpoff;

	probContext.fact_diagoffset(bp,bn,Gp,Gn,Gpn,alphagradstate,betagradstate);
    }

    return;
}

template <class T, class S>
void optState<T,S>::diagoffsethpzero(const Vector<double> &bp, const Vector<double> &bn, const Vector<double> &bpGrad, const Vector<double> &bnGrad, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( bp.size() == aN() );
    NiceAssert( bn.size() == bN() );
    NiceAssert( bpGrad.size() == aN() );
    NiceAssert( bnGrad.size() == bN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
        fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    if ( aNZ() < aN() )
    {
	Vector<T> upvectp(dalpha);

	if ( upvectp.size() )
	{
	    int i;

	    for ( i = 0 ; i < upvectp.size() ; i++ )
	    {
                upvectp("&",i) *= bpGrad(i);
	    }
	}

	dalphaGrad += upvectp;

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    if ( bNC() < bN() )
    {
	Vector<T> upvectn(dbeta);

	if ( upvectn.size() )
	{
	    int i;

	    for ( i = 0 ; i < upvectn.size() ; i++ )
	    {
                upvectn("&",i) *= bnGrad(i);
	    }
	}

	dbetaGrad += upvectn;

	if ( keepfact() )
	{
	    betagradstate  = -1;

	    int iP;

	    if ( bNF() )
	    {
		for ( iP = 0 ; iP < bNF() ; iP++ )
		{
                    if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		    {
			if ( betagradstate == -1 )
			{
			    betagradstate = iP;
			}

			else if ( betagradstate >= 0 )
			{
			    betagradstate = -2;

			    break;
			}
		    }
		}
	    }
	}
    }

    if ( keepfact() )
    {
	probContext.fact_diagoffset(bp,bn,Gp,Gn,Gpn,alphagradstate,betagradstate);
    }

    return;
}

template <class T, class S>
void optState<T,S>::diagoffsethpzero(int i, double bpoff, double bpoffGrad, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( i >= 0 );
    NiceAssert( i <= aN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
        fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    if ( alphaState(i) != 0 )
    {
	dalphaGrad("&",i) += bpoffGrad*dalpha(i);

	if ( keepfact() )
	{
	    alphagradstate = -2;
	}
    }

    if ( keepfact() )
    {
	Vector<double> bp(aN());
	Vector<double> bn(bN());

	bp.zero();
	bn.zero();

        bp("&",i) = bpoff;

	probContext.fact_diagoffset(bp,bn,Gp,Gn,Gpn,alphagradstate,betagradstate);
    }

    return;
}

template <class T, class S>
int optState<T,S>::addAlpha(int i, int alphrestrict, const T &zeroeg)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= aN() );

    int res;

    // add to context

    res = probContext.addAlpha(i);

    // Add to vectors

    dalpha.add(i);
    dalpha("&",i) = zeroeg;
    setzero(dalpha("&",i));
    dalphaGrad.add(i);
    dalphaGrad("&",i) = zeroeg;
    setzero(dalphaGrad("&",i));
    dalphaRestrict.add(i);
    dalphaRestrict("&",i) = alphrestrict;
    gradFixAlpha.add(i);
    gradFixAlpha("&",i) = 1;
    gradFixAlphaInd = 1;

    return res;
}

template <class T, class S>
int optState<T,S>::addBeta(int i, int betrestrict, const T &zeroeg)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= bN() );

    int res;

    // add to context

    res = probContext.addBeta(i);

    // Add to vectors

    dbeta.add(i);
    dbeta("&",i) = zeroeg;
    setzero(dbeta("&",i));
    dbetaGrad.add(i);
    dbetaGrad("&",i) = zeroeg;
    setzero(dbetaGrad("&",i));
    dbetaRestrict.add(i);
    dbetaRestrict("&",i) = betrestrict;
    gradFixBeta.add(i);
    gradFixBeta("&",i) = 1;
    gradFixBetaInd = 1;

    return res;
}

template <class T, class S>
int optState<T,S>::removeAlpha(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );
    NiceAssert( dalphaRestrict(i) == 3 );

    int res;

    // Remove from Vectors

    dalpha.remove(i);
    dalphaGrad.remove(i);
    dalphaRestrict.remove(i);
    gradFixAlpha.remove(i);

    // Remove from context

    res = probContext.removeAlpha(i);

    return res;
}

template <class T, class S>
int optState<T,S>::removeBeta(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < bN() );
    NiceAssert( dbetaRestrict(i) == 3 );

    int res;

    // Remove from Vectors

    dbeta.remove(i);
    dbetaGrad.remove(i);
    dbetaRestrict.remove(i);
    gradFixBeta.remove(i);

    // Remove from context

    res = probContext.removeBeta(i);

    return res;
}

template <class T, class S>
void optState<T,S>::changeAlphaRestrict(int i, int alphrestrict, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );
    NiceAssert( ( alphrestrict == 0 ) || ( alphrestrict == 1 ) || ( alphrestrict == 2 ) || ( alphrestrict == 3 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int iP;

    switch ( alphaState(i) )
    {
    case -2:
	{
	    if ( ( alphrestrict == 1 ) || ( alphrestrict == 3 ) )
	    {
		for ( iP = 0 ; iP < aNLB() ; iP++ )
		{
		    if ( pivAlphaLB()(iP) == i )
		    {
			break;
		    }
		}

                NiceAssert( pivAlphaLB()(iP) == i );

//FIXME: this can be done without disturbing the active set
		iP = modAlphaLBtoLF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		alphaStep(i,-dalpha(i),GpGrad,Gn,Gpn,gp,gn,hp);
		iP = modAlphaLFtoZ(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
	    }

            break;
	}

    case -1:
	{
	    if ( ( alphrestrict == 1 ) || ( alphrestrict == 3 ) )
	    {
		for ( iP = 0 ; iP < aNF() ; iP++ )
		{
		    if ( pivAlphaF()(iP) == i )
		    {
			if ( keepfact() && ( iP == alphagradstate ) )
			{
			    alphagradstate = -1;
			}

			break;
		    }
		}

                NiceAssert( pivAlphaF()(iP) == i );

		alphaStep(i,-dalpha(i),GpGrad,Gn,Gpn,gp,gn,hp);
		iP = modAlphaLFtoZ(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
	    }

            break;
	}

    case +1:
	{
	    if ( ( alphrestrict == 2 ) || ( alphrestrict == 3 ) )
	    {
		int iP;

		for ( iP = 0 ; iP < aNF() ; iP++ )
		{
		    if ( pivAlphaF()(iP) == i )
		    {
			if ( keepfact() && ( iP == alphagradstate ) )
			{
			    alphagradstate = -1;
			}

			break;
		    }
		}

                NiceAssert( pivAlphaF()(iP) == i );

		alphaStep(i,-dalpha(i),GpGrad,Gn,Gpn,gp,gn,hp);
		iP = modAlphaUFtoZ(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
	    }

            break;
	}

    case +2:
	{
	    if ( ( alphrestrict == 2 ) || ( alphrestrict == 3 ) )
	    {
		int iP;

		for ( iP = 0 ; iP < aNUB() ; iP++ )
		{
		    if ( pivAlphaUB()(iP) == i )
		    {
			break;
		    }
		}

                NiceAssert( pivAlphaUB()(iP) == i );

//FIXME: this can be done without disturbing the active set
		iP = modAlphaUBtoUF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
		alphaStep(i,-dalpha(i),GpGrad,Gn,Gpn,gp,gn,hp);
		iP = modAlphaUFtoZ(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
	    }

            break;
	}

    default:
	{
	    break;
	}
    }

    dalphaRestrict("&",i) = alphrestrict;

    return;
}

template <class T, class S>
void optState<T,S>::changeBetaRestrict (int i, int betrestrict, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < bN() );
    NiceAssert( ( betrestrict == 0 ) || ( betrestrict == 1 ) || ( betrestrict == 2 ) || ( betrestrict == 3 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int iP;

    if ( betaState(i) )
    {
	for ( iP = 0 ; iP < bNF() ; iP++ )
	{
	    if ( pivBetaF()(iP) == i )
	    {
		if ( keepfact() && ( iP == betagradstate ) )
		{
		    betagradstate = -1;
		}

		break;
	    }
	}

        NiceAssert( pivBetaF()(iP) == i );

	if ( ( ( betaState(i) < 0 ) && ( ( betrestrict == 1 ) || ( betrestrict == 3 ) ) ) || ( ( betaState(i) > 0 ) && ( ( betrestrict == 2 ) || ( betrestrict == 3 ) ) ) )
	{
	    betaStep(i,-dbeta(i),GpGrad,Gn,Gpn,gp,gn,hp);
	}

	iP = modBetaFtoC(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dbetaRestrict("&",i) = betrestrict;

    return;
}

template <class T, class S>
void optState<T,S>::changeAlphaRestricthpzero(int i, int alphrestrict, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );
    NiceAssert( ( alphrestrict == 0 ) || ( alphrestrict == 1 ) || ( alphrestrict == 2 ) || ( alphrestrict == 3 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
        fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int iP;

    switch ( alphaState(i) )
    {
    case -2:
	{
	    if ( ( alphrestrict == 1 ) || ( alphrestrict == 3 ) )
	    {
		for ( iP = 0 ; iP < aNLB() ; iP++ )
		{
		    if ( pivAlphaLB()(iP) == i )
		    {
			break;
		    }
		}

                NiceAssert( pivAlphaLB()(iP) == i );

//FIXME: this can be done without disturbing the active set
                iP = modAlphaLBtoLFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
                alphaStephpzero(i,-dalpha(i),GpGrad,Gn,Gpn,gp,gn);
                iP = modAlphaLFtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
	    }

            break;
	}

    case -1:
	{
	    if ( ( alphrestrict == 1 ) || ( alphrestrict == 3 ) )
	    {
		for ( iP = 0 ; iP < aNF() ; iP++ )
		{
		    if ( pivAlphaF()(iP) == i )
		    {
			if ( keepfact() && ( iP == alphagradstate ) )
			{
			    alphagradstate = -1;
			}

			break;
		    }
		}

                NiceAssert( pivAlphaF()(iP) == i );

                alphaStephpzero(i,-dalpha(i),GpGrad,Gn,Gpn,gp,gn);
                iP = modAlphaLFtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
	    }

            break;
	}

    case +1:
	{
	    if ( ( alphrestrict == 2 ) || ( alphrestrict == 3 ) )
	    {
		int iP;

		for ( iP = 0 ; iP < aNF() ; iP++ )
		{
		    if ( pivAlphaF()(iP) == i )
		    {
			if ( keepfact() && ( iP == alphagradstate ) )
			{
			    alphagradstate = -1;
			}

			break;
		    }
		}

                NiceAssert( pivAlphaF()(iP) == i );

                alphaStephpzero(i,-dalpha(i),GpGrad,Gn,Gpn,gp,gn);
                iP = modAlphaUFtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
	    }

            break;
	}

    case +2:
	{
	    if ( ( alphrestrict == 2 ) || ( alphrestrict == 3 ) )
	    {
		int iP;

		for ( iP = 0 ; iP < aNUB() ; iP++ )
		{
		    if ( pivAlphaUB()(iP) == i )
		    {
			break;
		    }
		}

                NiceAssert( pivAlphaUB()(iP) == i );

//FIXME: this can be done without disturbing the active set
                iP = modAlphaUBtoUFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
                alphaStephpzero(i,-dalpha(i),GpGrad,Gn,Gpn,gp,gn);
                iP = modAlphaUFtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
	    }

            break;
	}

    default:
	{
	    break;
	}
    }

    dalphaRestrict("&",i) = alphrestrict;

    return;
}

template <class T, class S>
void optState<T,S>::changeBetaRestricthpzero (int i, int betrestrict, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < bN() );
    NiceAssert( ( betrestrict == 0 ) || ( betrestrict == 1 ) || ( betrestrict == 2 ) || ( betrestrict == 3 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
        fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int iP;

    if ( betaState(i) )
    {
	for ( iP = 0 ; iP < bNF() ; iP++ )
	{
	    if ( pivBetaF()(iP) == i )
	    {
		if ( keepfact() && ( iP == betagradstate ) )
		{
		    betagradstate = -1;
		}

		break;
	    }
	}

        NiceAssert( pivBetaF()(iP) == i );

	if ( ( ( betaState(i) < 0 ) && ( ( betrestrict == 1 ) || ( betrestrict == 3 ) ) ) || ( ( betaState(i) > 0 ) && ( ( betrestrict == 2 ) || ( betrestrict == 3 ) ) ) )
	{
            betaStephpzero(i,-dbeta(i),GpGrad,Gn,Gpn,gp,gn);
	}

        iP = modBetaFtoChpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
    }

    dbetaRestrict("&",i) = betrestrict;

    return;
}

template <class T, class S>
int optState<T,S>::modAlphaLBtoZhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     )
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaLBtoZ(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLBtoUBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     )
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaLBtoUB(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaZtoLBhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     )
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaZtoLB(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaZtoUBhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     )
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaZtoUB(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUBtoLBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     )
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaUBtoLB(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUBtoZhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     )
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaUBtoZ(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLBtoLFhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaLBtoLF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLBtoUFhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaLBtoUF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUBtoUFhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaUBtoUF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUBtoLFhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn                     )
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaUBtoLF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

inline int isneg(double a);
inline int isneg(double a)
{
    return ( a < 0 );
}


inline int ispos(double a);
inline int ispos(double a)
{
    return ( a > 0 );
}


inline int isneg(double a, double x);
inline int isneg(double a, double x)
{
    return ( a < x );
}


inline int ispos(double a, double x);
inline int ispos(double a, double x)
{
    return ( a > x );
}


inline int isneg(const Vector<double> &a);
inline int isneg(const Vector<double> &a)
{
    (void) a;

    return 1;
}


inline int ispos(const Vector<double> &a);
inline int ispos(const Vector<double> &a)
{
    (void) a;

    return 1;
}


inline int isnegcond(const Vector<double> &a, double x);
inline int isnegcond(const Vector<double> &a, double x)
{
    (void) a;
    (void) x;

    return 1;
}


inline int isposcond(const Vector<double> &a, double x);
inline int isposcond(const Vector<double> &a, double x)
{
    (void) a;
    (void) x;

    return 1;
}


template <class T, class S>
int optState<T,S>::modAlphaLFtoUFhpzero(int iP, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaLFtoUF(iP);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

//    NiceAssert( isposcond(dalpha(pivAlphaF()(res)),-dopttol) );

    if ( ( abs2(GpGrad(pivAlphaF()(res),pivAlphaF()(res))*dalpha(pivAlphaF()(res))) > STEPTRIGGER*dopttol ) && ( isneg(dalpha(pivAlphaF()(res))) ) )
    {
	alphaStephpzero(pivAlphaF()(res),-dalpha(pivAlphaF()(res)),GpGrad,Gn,Gpn,gp,gn,1);
    }

    if ( isneg(dalpha(pivAlphaF()(res))) )
    {
        cumgraderr += abs2((GpGrad(pivAlphaF()(res),pivAlphaF()(res))*dalpha(pivAlphaF()(res))));

	dalpha("&",pivAlphaF()(res)) = 0;
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUFtoLFhpzero(int iP, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaUFtoLF(iP);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

//    NiceAssert( isnegcond(dalpha(pivAlphaF()(res)),dopttol) );

    if ( ( abs2(GpGrad(pivAlphaF()(res),pivAlphaF()(res))*dalpha(pivAlphaF()(res))) > STEPTRIGGER*dopttol ) && ( ispos(dalpha(pivAlphaF()(res))) ) )
    {
	alphaStephpzero(pivAlphaF()(res),-dalpha(pivAlphaF()(res)),GpGrad,Gn,Gpn,gp,gn,1);
    }

    if ( ispos(dalpha(pivAlphaF()(res))) )
    {
        cumgraderr += abs2((GpGrad(pivAlphaF()(res),pivAlphaF()(res))*dalpha(pivAlphaF()(res))));

	dalpha("&",pivAlphaF()(res)) = 0;
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLFtoLBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &lb)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaLFtoLB(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(dalpha(pivAlphaLB()(res))-lb(pivAlphaLB()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaLB()(res),pivAlphaLB()(res))*(lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)))) > STEPTRIGGER*dopttol )
    {
	alphaStephpzero(pivAlphaLB()(res),lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)),GpGrad,Gn,Gpn,gp,gn,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaLB()(res),pivAlphaLB()(res))*(lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)))));

    dalpha("&",pivAlphaLB()(res)) = lb(pivAlphaLB()(res));

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLFtoZhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaLFtoZ(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(dalpha(pivAlphaZ()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaZ()(res),pivAlphaZ()(res))*dalpha(pivAlphaZ()(res))) > STEPTRIGGER*dopttol )
    {
	alphaStephpzero(pivAlphaZ()(res),-dalpha(pivAlphaZ()(res)),GpGrad,Gn,Gpn,gp,gn,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaZ()(res),pivAlphaZ()(res))*dalpha(pivAlphaZ()(res))));

    dalpha("&",pivAlphaZ()(res)) = 0;

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLFtoUBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &ub)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( ub.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaLFtoUB(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaUB()(res),pivAlphaUB()(res))*(ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)))) > STEPTRIGGER*dopttol )
    {
	alphaStephpzero(pivAlphaUB()(res),ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)),GpGrad,Gn,Gpn,gp,gn,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaUB()(res),pivAlphaUB()(res))*(ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)))));

    dalpha("&",pivAlphaUB()(res)) = ub(pivAlphaUB()(res));

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaZtoLFhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );
    NiceAssert( ( dalphaRestrict(pivAlphaZ()(iP)) == 0 ) || ( dalphaRestrict(pivAlphaZ()(iP)) == 2 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaZtoLF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaZtoUFhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );
    NiceAssert( ( dalphaRestrict(pivAlphaZ()(iP)) == 0 ) || ( dalphaRestrict(pivAlphaZ()(iP)) == 1 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaZtoUF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUFtoLBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &lb)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaUFtoLB(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaLB()(res),pivAlphaLB()(res))*(lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)))) > STEPTRIGGER*dopttol )
    {
	alphaStephpzero(pivAlphaLB()(res),lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)),GpGrad,Gn,Gpn,gp,gn,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaLB()(res),pivAlphaLB()(res))*(lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)))));

    dalpha("&",pivAlphaLB()(res)) = lb(pivAlphaLB()(res));

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUFtoZhpzero (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaUFtoZ(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(dalpha(pivAlphaZ()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaZ()(res),pivAlphaZ()(res))*dalpha(pivAlphaZ()(res))) > STEPTRIGGER*dopttol )
    {
	alphaStephpzero(pivAlphaZ()(res),-dalpha(pivAlphaZ()(res)),GpGrad,Gn,Gpn,gp,gn,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaZ()(res),pivAlphaZ()(res))*dalpha(pivAlphaZ()(res))));

    dalpha("&",pivAlphaZ()(res)) = 0;

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUFtoUBhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &ub)
{ 
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( ub.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modAlphaUFtoUB(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(dalpha(pivAlphaUB()(res))-ub(pivAlphaUB()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaUB()(res),pivAlphaUB()(res))*(ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)))) > STEPTRIGGER*dopttol )
    {
	alphaStephpzero(pivAlphaUB()(res),ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)),GpGrad,Gn,Gpn,gp,gn,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaUB()(res),pivAlphaUB()(res))*(ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)))));

    dalpha("&",pivAlphaUB()(res)) = ub(pivAlphaUB()(res));

    return res;
}

template <class T, class S>
int optState<T,S>::modAllToDesthpzero(const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    // Free/constrain all alphas and betas based on alphaRestrict and
    // betaRestrict (free if restrict == 0, constrained if restrict == 3,
    // throw otherwise).  Will also change (or set) alpha UF or LF depending
    // on the sign of alpha.

    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    int i,iP;
    int res = 0;

    // Step 1: convert all LB to LF, UB to UF

    if ( aNLB() )
    {
        for ( iP = aNLB()-1 ; iP >= 0 ; iP-- )
        {
            res |= modAlphaLBtoLFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
        }
    }

    if ( aNUB() )
    {
        for ( iP = aNUB()-1 ; iP >= 0 ; iP-- )
        {
            res |= modAlphaUBtoUFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
        }
    }

    // Step 2: sign-correct LF and UF

    if ( aNF() )
    {
        for ( iP = aNF()-1 ; iP >= 0 ; iP-- )
        {                     
            i = pivAlphaF()(iP);

            if ( ispos(alpha()(i)) && ( alphaState(i) == -1 ) )
            {
                res |= modAlphaUFtoLFhpzero(iP,GpGrad,Gn,Gpn,gp,gn);
            }

            else if ( isneg(alpha()(i)) && ( alphaState(i) == +1 ) )
            {
                res |= modAlphaLFtoUFhpzero(iP,GpGrad,Gn,Gpn,gp,gn);
            }
        }
    }

    // Step 3: mod Z to {L,U}F for all Z constrained variabled with astat

    if ( aNZ() )
    {
        for ( iP = aNZ()-1 ; iP >= 0 ; iP-- )
        {                     
            i = pivAlphaZ()(iP);

            NiceAssert( ( alphaRestrict()(i) == 3 ) || !(alphaRestrict()(i)) );

            if ( !(alphaRestrict()(i)) )
            {
                res |= modAlphaZtoUFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
            }
        }
    }

    // Step 4: mod {L,U}F to Z for all Z constrained variabled with !astat

    if ( aNF() )
    {
        // Note that modding {L,U}F to Z may change ordering in pivAlphaF,
        // so to be sure that we don't skip any points we need to start over
        // everytime a potential change occurs to this ordering.  Hence the
        // clearrun flag.

        int clearrun = 0;

        while ( !clearrun )
        {
            clearrun = 1;

            for ( iP = aNF()-1 ; iP >= 0 ; iP-- )
            {
                i = pivAlphaF()(iP);

                NiceAssert( ( alphaRestrict()(i) == 3 ) || !(alphaRestrict()(i)) );

                if ( ( alphaRestrict()(i) == 3 ) && ( alphaState()(i) == +1 ) )
                {
                    alphaStephpzero(i,-dalpha(i),GpGrad,Gn,Gpn,gp,gn);
                    res |= modAlphaUFtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
                    clearrun = 0;
                    break;
                }

                else if ( ( alphaRestrict()(i) == 3 ) && ( alphaState()(i) == -1 ) )
                {
                    alphaStephpzero(i,-dalpha(i),GpGrad,Gn,Gpn,gp,gn);
                    res |= modAlphaLFtoZhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
                    clearrun = 0;
                    break;
                }
            }
        }
    }

    // Step 5: mod C to F for all C constrained beta with bstat

    if ( bNC() )
    {
        for ( iP = bNC()-1 ; iP >= 0 ; iP-- )
        {                     
            i = pivBetaC()(iP);

            NiceAssert( ( betaRestrict()(i) == 3 ) || !(betaRestrict()(i)) );

            if ( !(betaRestrict()(i)) )
            {
                res |= modBetaCtoFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
            }
        }
    }

    // Step 6: mod F to C for all C constrained beta with !bstat

    if ( bNF() )
    {
        // Note that modding F to C may change ordering in pivBetaF,
        // so to be sure that we don't skip any points we need to start over
        // everytime a potential change occurs to this ordering.  Hence the
        // clearrun flag.

        int clearrun = 0;

        while ( !clearrun )
        {
            clearrun = 1;

            for ( iP = bNF()-1 ; iP >= 0 ; iP-- )
            {
                i = pivBetaF()(iP);

                NiceAssert( ( betaRestrict()(i) == 3 ) || !(betaRestrict()(i)) );

                if ( betaRestrict()(i) == 3 )
                {
                    betaStephpzero(i,-dbeta(i),GpGrad,Gn,Gpn,gp,gn);
                    res |= modBetaFtoChpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
                    clearrun = 0;
                    break;
                }
            }
        }
    }

    return res;
}



template <class T, class S>
int optState<T,S>::modAlphaLBtoZ (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                     )
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaLB()(iP)) += hp(pivAlphaLB()(iP));

    int res = probContext.modAlphaLBtoZ(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLBtoUB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                     )
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaLB()(iP)) += 2*hp(pivAlphaLB()(iP));

    int res = probContext.modAlphaLBtoUB(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaZtoLB (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                     )
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaZ()(iP)) -= hp(pivAlphaZ()(iP));

    int res = probContext.modAlphaZtoLB(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaZtoUB (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                     )
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaZ()(iP)) += hp(pivAlphaZ()(iP));

    int res = probContext.modAlphaZtoUB(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUBtoLB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                     )
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaUB()(iP)) -= 2*hp(pivAlphaUB()(iP));

    int res = probContext.modAlphaUBtoLB(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUBtoZ (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                     )
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );

    (void) Gp;

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaUB()(iP)) -= hp(pivAlphaUB()(iP));

    int res = probContext.modAlphaUBtoZ(iP);

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLBtoLF(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int res = probContext.modAlphaLBtoLF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLBtoUF(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaLB()(iP)) += 2*hp(pivAlphaLB()(iP));

    int res = probContext.modAlphaLBtoUF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUBtoUF(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int res = probContext.modAlphaUBtoUF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUBtoLF(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp                     )
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaUB()(iP)) -= 2*hp(pivAlphaUB()(iP));

    int res = probContext.modAlphaUBtoLF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLFtoUF(int iP, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaF()(iP)) += 2*hp(pivAlphaF()(iP));

    int res = probContext.modAlphaLFtoUF(iP);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( dalpha(pivAlphaF()(res)) > -dopttol );

    if ( ( abs2(GpGrad(pivAlphaF()(res),pivAlphaF()(res))*dalpha(pivAlphaF()(res))) > STEPTRIGGER*dopttol ) && ( dalpha(pivAlphaF()(res)) < 0 ) )
    {
	alphaStephpzero(pivAlphaF()(res),-dalpha(pivAlphaF()(res)),GpGrad,Gn,Gpn,gp,gn,1);
    }

    if ( dalpha(pivAlphaF()(res)) < 0 )
    {
        cumgraderr += abs2((GpGrad(pivAlphaF()(res),pivAlphaF()(res))*dalpha(pivAlphaF()(res))));

	dalpha("&",pivAlphaF()(res)) = 0;
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUFtoLF(int iP, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaF()(iP)) -= 2*hp(pivAlphaF()(iP));

    int res = probContext.modAlphaUFtoLF(iP);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( dalpha(pivAlphaF()(res)) < dopttol );

    if ( ( abs2(GpGrad(pivAlphaF()(res),pivAlphaF()(res))*dalpha(pivAlphaF()(res))) > STEPTRIGGER*dopttol ) && ( dalpha(pivAlphaF()(res)) > 0 ) )
    {
	alphaStep(pivAlphaF()(res),-dalpha(pivAlphaF()(res)),GpGrad,Gn,Gpn,gp,gn,hp,1);
    }

    if ( dalpha(pivAlphaF()(res)) > 0 )
    {
        cumgraderr += abs2((GpGrad(pivAlphaF()(res),pivAlphaF()(res))*dalpha(pivAlphaF()(res))));

	dalpha("&",pivAlphaF()(res)) = 0;
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLFtoLB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &lb)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int res = probContext.modAlphaLFtoLB(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(dalpha(pivAlphaLB()(res))-lb(pivAlphaLB()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaLB()(res),pivAlphaLB()(res))*(lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)))) > STEPTRIGGER*dopttol )
    {
	alphaStep(pivAlphaLB()(res),lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)),GpGrad,Gn,Gpn,gp,gn,hp,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaLB()(res),pivAlphaLB()(res))*(lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)))));

    dalpha("&",pivAlphaLB()(res)) = lb(pivAlphaLB()(res));

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLFtoZ (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaF()(iP)) += hp(pivAlphaF()(iP));

    int res = probContext.modAlphaLFtoZ(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(dalpha(pivAlphaZ()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaZ()(res),pivAlphaZ()(res))*dalpha(pivAlphaZ()(res))) > STEPTRIGGER*dopttol )
    {
	alphaStep(pivAlphaZ()(res),-dalpha(pivAlphaZ()(res)),GpGrad,Gn,Gpn,gp,gn,hp,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaZ()(res),pivAlphaZ()(res))*dalpha(pivAlphaZ()(res))));

    dalpha("&",pivAlphaZ()(res)) = 0;

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaLFtoUB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &ub)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( ub.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaF()(iP)) += 2*hp(pivAlphaF()(iP));

    int res = probContext.modAlphaLFtoUB(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaUB()(res),pivAlphaUB()(res))*(ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)))) > STEPTRIGGER*dopttol )
    {
	alphaStep(pivAlphaUB()(res),ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)),GpGrad,Gn,Gpn,gp,gn,hp,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaUB()(res),pivAlphaUB()(res))*(ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)))));

    dalpha("&",pivAlphaUB()(res)) = ub(pivAlphaUB()(res));

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaZtoLF (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );
    NiceAssert( ( dalphaRestrict(pivAlphaZ()(iP)) == 0 ) || ( dalphaRestrict(pivAlphaZ()(iP)) == 2 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaZ()(iP)) -= hp(pivAlphaZ()(iP));

    int res = probContext.modAlphaZtoLF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaZtoUF (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );
    NiceAssert( ( dalphaRestrict(pivAlphaZ()(iP)) == 0 ) || ( dalphaRestrict(pivAlphaZ()(iP)) == 1 ) );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaZ()(iP)) += hp(pivAlphaZ()(iP));

    int res = probContext.modAlphaZtoUF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dalphaGrad("&",pivAlphaF()(res))) > dopttol )
	{
	    if ( alphagradstate == -1 )
	    {
		alphagradstate = res;
	    }

	    else if ( alphagradstate >= 0 )
	    {
		alphagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUFtoLB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &lb)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaF()(iP)) -= hp(pivAlphaF()(iP));
    dalphaGrad("&",pivAlphaF()(iP)) -= hp(pivAlphaF()(iP));

    int res = probContext.modAlphaUFtoLB(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaLB()(res),pivAlphaLB()(res))*(lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)))) > STEPTRIGGER*dopttol )
    {
	alphaStep(pivAlphaLB()(res),lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)),GpGrad,Gn,Gpn,gp,gn,hp,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaLB()(res),pivAlphaLB()(res))*(lb(pivAlphaLB()(res))-dalpha(pivAlphaLB()(res)))));

    dalpha("&",pivAlphaLB()(res)) = lb(pivAlphaLB()(res));

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUFtoZ (int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    dalphaGrad("&",pivAlphaF()(iP)) -= hp(pivAlphaF()(iP));

    int res = probContext.modAlphaUFtoZ(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(dalpha(pivAlphaZ()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaZ()(res),pivAlphaZ()(res))*dalpha(pivAlphaZ()(res))) > STEPTRIGGER*dopttol )
    {
	alphaStep(pivAlphaZ()(res),-dalpha(pivAlphaZ()(res)),GpGrad,Gn,Gpn,gp,gn,hp,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaZ()(res),pivAlphaZ()(res))*dalpha(pivAlphaZ()(res))));

    dalpha("&",pivAlphaZ()(res)) = 0;

    return res;
}

template <class T, class S>
int optState<T,S>::modAlphaUFtoUB(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &ub)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( ub.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int res = probContext.modAlphaUFtoUB(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( alphagradstate == iP )
	{
	    alphagradstate = -1;
	}

	else if ( alphagradstate > iP )
	{
	    alphagradstate--;
	}
    }

    NiceAssert( abs2(dalpha(pivAlphaUB()(res))-ub(pivAlphaUB()(res))) < dopttol );

    if ( abs2(GpGrad(pivAlphaUB()(res),pivAlphaUB()(res))*(ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)))) > STEPTRIGGER*dopttol )
    {
	alphaStep(pivAlphaUB()(res),ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)),GpGrad,Gn,Gpn,gp,gn,hp,1);
    }

    cumgraderr += abs2((GpGrad(pivAlphaUB()(res),pivAlphaUB()(res))*(ub(pivAlphaUB()(res))-dalpha(pivAlphaUB()(res)))));

    dalpha("&",pivAlphaUB()(res)) = ub(pivAlphaUB()(res));

    return res;
}

template <class T, class S>
int optState<T,S>::modBetaCtoF(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < bNC() );
    NiceAssert( dbetaRestrict(pivBetaC()(iP)) != 3 );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int res = probContext.modBetaCtoF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dbetaGrad("&",pivBetaF()(res))) > dopttol )
	{
	    if ( betagradstate == -1 )
	    {
		betagradstate = res;
	    }

	    else if ( betagradstate >= 0 )
	    {
		betagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modBetaFtoC(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < bNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int res = probContext.modBetaFtoC(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( betagradstate == iP )
	{
	    betagradstate = -1;
	}

	else if ( betagradstate > iP )
	{
	    betagradstate--;
	}
    }

    NiceAssert( abs2(dbeta(pivBetaC()(res))) < dopttol );

    if ( Gn(pivBetaC()(res),pivBetaC()(res))*abs2(dbeta(pivBetaC()(res))) > STEPTRIGGER*dopttol )
    {
	betaStep(pivBetaC()(res),-dbeta(pivBetaC()(res)),GpGrad,Gn,Gpn,gp,gn,hp,1);
    }

    cumgraderr += (Gn(pivBetaC()(res),pivBetaC()(res))*abs2(dbeta(pivBetaC()(res))));

    dbeta("&",pivBetaC()(res)) = 0;

    return res;
}

template <class T, class S>
int optState<T,S>::modBetaCtoFhpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < bNC() );
    NiceAssert( dbetaRestrict(pivBetaC()(iP)) != 3 );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modBetaCtoF(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
        if ( abs2(dbetaGrad("&",pivBetaF()(res))) > dopttol )
	{
	    if ( betagradstate == -1 )
	    {
		betagradstate = res;
	    }

	    else if ( betagradstate >= 0 )
	    {
		betagradstate = -2;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::modBetaFtoChpzero(int iP, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < bNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int res = probContext.modBetaFtoC(iP,Gp,Gn,Gpn,alphagradstate,betagradstate);

    if ( keepfact() )
    {
	if ( betagradstate == iP )
	{
	    betagradstate = -1;
	}

	else if ( betagradstate > iP )
	{
	    betagradstate--;
	}
    }

    NiceAssert( abs2(dbeta(pivBetaC()(res))) < dopttol );

    if ( Gn(pivBetaC()(res),pivBetaC()(res))*abs2(dbeta(pivBetaC()(res))) > STEPTRIGGER*dopttol )
    {
	betaStephpzero(pivBetaC()(res),-dbeta(pivBetaC()(res)),GpGrad,Gn,Gpn,gp,gn,1);
    }

    cumgraderr += (Gn(pivBetaC()(res),pivBetaC()(res))*abs2(dbeta(pivBetaC()(res))));

    dbeta("&",pivBetaC()(res)) = 0;

    return res;
}

template <class T, class S>
T &optState<T,S>::posAlphaGradhpzero(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp) const
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );
    NiceAssert( alphaState(i) >= 0 );

    if ( gradFixAlphaInd && gradFixAlpha(i) )
    {
	recalcAlphaGradhpzero(result,GpGrad,Gpn,gp,i);
    }

    else
    {
	result = dalphaGrad(i);
    }

    return result;
}

template <class T, class S>
T &optState<T,S>::negAlphaGradhpzero(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp) const
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );
    NiceAssert( alphaState(i) <= 0 );

    if ( gradFixAlphaInd && gradFixAlpha(i) )
    {
	recalcAlphaGradhpzero(result,GpGrad,Gpn,gp,i);
    }

    else
    {
	result = dalphaGrad(i);
    }

    return result;
}

template <class T, class S>
T &optState<T,S>::posAlphaGrad(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );
    NiceAssert( alphaState(i) >= 0 );

    if ( gradFixAlphaInd && gradFixAlpha(i) )
    {
	recalcAlphaGrad(result,GpGrad,Gpn,gp,hp,i);
    }

    else
    {
	result = dalphaGrad(i);
    }

    if ( alphaState()(i) == 0 )
    {
        result += hp(i);
    }

    return result;
}

template <class T, class S>
T &optState<T,S>::negAlphaGrad(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );
    NiceAssert( alphaState(i) <= 0 );

    if ( gradFixAlphaInd && gradFixAlpha(i) )
    {
	recalcAlphaGrad(result,GpGrad,Gpn,gp,hp,i);
    }

    else
    {
	result = dalphaGrad(i);
    }

    if ( alphaState()(i) == 0 )
    {
        result -= hp(i);
    }

    return result;
}

template <class T, class S>
T &optState<T,S>::spAlphaGradhpzero(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp) const
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );
    NiceAssert( alphaState(i) <= 0 );

    if ( gradFixAlphaInd && gradFixAlpha(i) )
    {
	recalcAlphaGradhpzero(result,GpGrad,Gpn,gp,i);
    }

    else
    {
	result = dalphaGrad(i);
    }

    return result;
}

template <class T, class S>
T &optState<T,S>::spAlphaGrad(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );
    NiceAssert( alphaState(i) <= 0 );

    if ( gradFixAlphaInd && gradFixAlpha(i) )
    {
	recalcAlphaGrad(result,GpGrad,Gpn,gp,hp,i);
    }

    else
    {
	result = dalphaGrad(i);
    }

    if ( ( alphaState()(i) == 0 ) && ( alphaRestrict()(i) == 1 ) )
    {
        result += hp(i);
    }

    else if ( ( alphaState()(i) == 0 ) && ( alphaRestrict()(i) == 2 ) )
    {
        result -= hp(i);
    }

    return result;
}

template <class T, class S>
int optState<T,S>::unAlphaGradIfPresent(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );

    (void) GpGrad;
    (void) Gpn;
    (void) gp;

    if ( gradFixAlphaInd && gradFixAlpha(i) )
    {
	return 1;
    }

    else
    {
	result = dalphaGrad(i);
    }

    if ( alphaState()(i) < 0 )
    {
        result += hp(i);
    }

    else if ( alphaState()(i) > 0 )
    {
        result -= hp(i);
    }

    return 0;
}

template <class T, class S>
T &optState<T,S>::unAlphaGradhpzero(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp) const
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );

    if ( gradFixAlphaInd && gradFixAlpha(i) )
    {
	recalcAlphaGradhpzero(result,GpGrad,Gpn,gp,i);
    }

    else
    {
	result = dalphaGrad(i);
    }

    return result;
}

template <class T, class S>
T &optState<T,S>::unAlphaGrad(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );

    if ( gradFixAlphaInd && gradFixAlpha(i) )
    {
	recalcAlphaGrad(result,GpGrad,Gpn,gp,hp,i);
    }

    else
    {
	result = dalphaGrad(i);
    }

    if ( alphaState()(i) < 0 )
    {
        result += hp(i);
    }

    else if ( alphaState()(i) > 0 )
    {
        result -= hp(i);
    }

    return result;
}

template <class T, class S>
T &optState<T,S>::unBetaGrad(T &result, int i, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gn) const
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gn.numRows() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dbetaGrad.size() );

    if ( gradFixAlphaInd && gradFixBeta(i) )
    {
        recalcBetaGrad(result,Gn,Gpn,gn,i);
    }

    else
    {
	result = dbetaGrad(i);
    }

    return result;
}

template <class T, class S>
T &optState<T,S>::reAlphaGradhpzero(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp) const
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );

    unAlphaGradhpzero(result,i,GpGrad,Gpn,gp);

    return result;
}

template <class T, class S>
T &optState<T,S>::reAlphaGrad(T &result, int i, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp) const
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dalphaGrad.size() );

    unAlphaGrad(result,i,GpGrad,Gpn,gp,hp);

    if ( alphaState()(i) < 0 )
    {
        result += hp(i);
    }

    else if ( alphaState()(i) > 0 )
    {
        result -= hp(i);
    }

    return result;
}

template <class T, class S>
T &optState<T,S>::reBetaGrad(T &result, int i, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gn) const
{
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gn.numRows() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < dbetaGrad.size() );

    unBetaGrad(result,i,Gn,Gpn,gn);

    return result;
}

template <class T, class S>
int optState<T,S>::scaleFStep(double &scale, int &alphaFIndex, int &betaFIndex, int &betaCIndex, int &stateChange, int asize, int bsize, int &bailout, Vector<T> &alphaFStep, Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &lb, const Vector<double> &ub)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( asize >= 0 );
    NiceAssert( asize <= aNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    return scaleFStepbase(scale,alphaFIndex,betaFIndex,betaCIndex,stateChange,asize,bsize,bailout,alphaFStep,betaFStep,Gn,Gpn,lb,ub,0);
}

template <class T, class S>
int optState<T,S>::scaleFStephpzero(double &scale, int &alphaFIndex, int &betaFIndex, int &betaCIndex, int &stateChange, int asize, int bsize, int &bailout, Vector<T> &alphaFStep, Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &lb, const Vector<double> &ub)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( asize >= 0 );
    NiceAssert( asize <= aNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    return scaleFStepbase(scale,alphaFIndex,betaFIndex,betaCIndex,stateChange,asize,bsize,bailout, alphaFStep,betaFStep,Gn,Gpn,lb,ub,1);
}

template <class T, class S>
int optState<T,S>::scaleFStepbase(double &scale, int &alphaFIndex, int &betaFIndex, int &betaCIndex, int &stateChange, int asize, int bsize, int &bailout, Vector<T> &alphaFStep, Vector<T> &betaFStep, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &lb, const Vector<double> &ub, int hpzero)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( ub.size() == aN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( asize >= 0 );
    NiceAssert( asize <= aNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );

    int res = 0;
    int i,iP;
    double potscale;

    // NaN can occur if:
    //
    // alpha is NaN
    // alpha step is NaN
    // (alpha gap ~= 0)/(alpha step ~= 0) occurs (approximately)
    //
    // and likewise with beta.  The first two cases trigger bailout,
    // the last involves infinitessimal creap that can be safely
    // ignored.
    //
    // bailout codes:
    //
    // 0 - no problem
    // 1 - alphaF has NaN elements
    // 2 - alphaFstep has NaN elements
    // 3 - betaF has NaN elements
    // 4 - betaFstep has NaN elements
    // 5 - betaCgrad has NaN elements
    // 6 - betaCgradstep has NaN elements

    // Near-zero step avoidance.  If scale is zero and step is
    // very small then we let that pass!

    bailout = 0;

    scale = 1;
    alphaFIndex = -1;
    betaFIndex  = -1;
    betaCIndex  = -1;

    double zero_overrun = zerotol();

    if ( asize )
    {
	for ( i = 0 ; ( i < asize ) && !bailout ; i++ )
	{
            if ( testisvnan(dalpha(pivAlphaF()(i))) )
            {
                bailout = 1;
            }

            else if ( testisvnan(alphaFStep(i)) )
            {
                bailout = 2;
            }

            else if ( abs2(alphaFStep(i)) > 0 )
	    {
		if ( ( alphaState(pivAlphaF()(i)) == -1 ) && ( scale*alphaFStep(i) > 0 ) && ( alphaRestrict(pivAlphaF()(i)) || !hpzero ) )
		{
		    if ( dalpha(pivAlphaF()(i))+(scale*alphaFStep(i)) >= zero_overrun )
		    {
			potscale = -dalpha(pivAlphaF()(i))/alphaFStep(i);

                        //NiceAssert( potscale >= -zerotol() );

                        if ( testisvnan(potscale) )
                        {
errstream() << "&>";
                            potscale = 1.0;
                        }

			else if ( potscale < 0 )
			{
errstream() << ">";
//errstream() << "WARNING: scaling outside bounds (type 1) " << potscale << "\n";
			    potscale = 0;
			}

			if ( potscale < scale )
			{
			    scale = potscale;
			    alphaFIndex = i;
			    betaFIndex  = -1;
			    betaCIndex  = -1;
			    stateChange = -1;
			    res = 1;
			}
		    }
		}

		else if ( ( alphaState(pivAlphaF()(i)) == -1 ) && ( scale*alphaFStep(i) > 0 ) && ( !alphaRestrict(pivAlphaF()(i)) && hpzero ) )
		{
		    if ( dalpha(pivAlphaF()(i))+(scale*alphaFStep(i)) >= ub(pivAlphaF()(i))+zero_overrun )
		    {
			potscale = (ub(pivAlphaF()(i))-dalpha(pivAlphaF()(i)))/alphaFStep(i);

                        //NiceAssert( potscale >= -zerotol() );

                        if ( testisvnan(potscale) )
                        {
errstream() << "&]";
                            potscale = 1.0;
                        }

			else if ( potscale < 0 )
			{
errstream() << "]";
//errstream() << "WARNING: scaling outside bounds (type 2) " << potscale << "\n";
			    potscale = 0;
			}

			if ( potscale < scale )
			{
			    scale = potscale;
			    alphaFIndex = i;
			    betaFIndex  = -1;
			    betaCIndex  = -1;
			    stateChange = -3;
			    res = 1;
			}
		    }
		}

		else if ( ( alphaState(pivAlphaF()(i)) == -1 ) && ( scale*alphaFStep(i) < 0 ) )
		{
		    if ( dalpha(pivAlphaF()(i))+(scale*alphaFStep(i)) <= lb(pivAlphaF()(i))-zero_overrun )
		    {
			potscale = (lb(pivAlphaF()(i))-dalpha(pivAlphaF()(i)))/alphaFStep(i);

                        //NiceAssert( potscale >= -zerotol() );

                        if ( testisvnan(potscale) )
                        {
errstream() << "&<";
                            potscale = 1.0;
                        }

			else if ( potscale < 0 )
			{
errstream() << "<"; // << "," << alphaRestrict(pivAlphaF()(i)) << "," << dalpha(i) << "," << alphaGrad()(i) << "," << alphaFStep(i);
//errstream() << "WARNING: scaling outside bounds (type 3) " << potscale << "\n";
			    potscale = 0;
			}

			if ( potscale < scale )
			{
			    scale = potscale;
			    alphaFIndex = i;
			    betaFIndex  = -1;
			    betaCIndex  = -1;
			    stateChange = -2;
			    res = 1;
			}
		    }
		}

		else if ( ( alphaState(pivAlphaF()(i)) == +1 ) && ( scale*alphaFStep(i) < 0 ) && ( alphaRestrict(pivAlphaF()(i)) || !hpzero ) )
		{
		    if ( dalpha(pivAlphaF()(i))+(scale*alphaFStep(i)) <= -zero_overrun )
		    {
			potscale = -dalpha(pivAlphaF()(i))/alphaFStep(i);

                        //NiceAssert( potscale >= -zerotol() );

                        if ( testisvnan(potscale) )
                        {
errstream() << "&)";
                            potscale = 1.0;
                        }

			else if ( potscale < 0 )
			{
errstream() << ")";
//errstream() << "WARNING: scaling outside bounds (type 4) " << potscale << "\n";
			    potscale = 0;
			}

			if ( potscale < scale )
			{
			    scale = potscale;
			    alphaFIndex = i;
			    betaFIndex  = -1;
			    betaCIndex  = -1;
			    stateChange = +1;
			    res = 1;
			}
		    }
		}

		else if ( ( alphaState(pivAlphaF()(i)) == +1 ) && ( scale*alphaFStep(i) < 0 ) && ( !alphaRestrict(pivAlphaF()(i)) && hpzero ) )
		{
		    if ( dalpha(pivAlphaF()(i))+(scale*alphaFStep(i)) <= lb(pivAlphaF()(i))-zero_overrun )
		    {
			potscale = (lb(pivAlphaF()(i))-dalpha(pivAlphaF()(i)))/alphaFStep(i);

                        //NiceAssert( potscale >= -zerotol() );

                        if ( testisvnan(potscale) )
                        {
errstream() << "&[";
                            potscale = 1.0;
                        }

			else if ( potscale < 0 )
			{
errstream() << "[";
//errstream() << "WARNING: scaling outside bounds (type 5) " << potscale << "\n";
			    potscale = 0;
			}

			if ( potscale < scale )
			{
			    scale = potscale;
			    alphaFIndex = i;
			    betaFIndex  = -1;
			    betaCIndex  = -1;
			    stateChange = +3;
			    res = 1;
			}
		    }
		}

		else if ( ( alphaState(pivAlphaF()(i)) == +1 ) && ( scale*alphaFStep(i) > 0 ) )
		{
		    if ( dalpha(pivAlphaF()(i))+(scale*alphaFStep(i)) >= ub(pivAlphaF()(i))+zero_overrun )
		    {
			potscale = (ub(pivAlphaF()(i))-dalpha(pivAlphaF()(i)))/alphaFStep(i);

                        //NiceAssert( potscale >= -zerotol() );

                        if ( testisvnan(potscale) )
                        {
errstream() << "&(";
                            potscale = 1.0;
                        }

			else if ( potscale < 0 )
			{
errstream() << "("; // << "," << alphaRestrict(pivAlphaF()(i)) << "," << dalpha(i) << "," << alphaGrad()(i) << "," << alphaFStep(i) << "," << ub(pivAlphaF()(i)) << "\n";
//errstream() << "WARNING: scaling outside bounds (type 6) " << potscale << "\n";
			    potscale = 0;
			}

			if ( potscale < scale )
			{
			    scale = potscale;
			    alphaFIndex = i;
			    betaFIndex  = -1;
			    betaCIndex  = -1;
			    stateChange = +2;
			    res = 1;
			}
		    }
		}
	    }
	}
    }

    if ( bsize && !bailout )
    {
	for ( i = 0 ; ( i < bsize ) && !bailout ; i++ )
	{
            if ( testisvnan(dbeta(pivBetaF()(i))) )
            {
                bailout = 3;
            }

            else if ( testisvnan(betaFStep(i)) )
            {
                bailout = 4;
            }

	    else if ( ( dbetaRestrict(pivBetaF()(i)) == 2 ) && ( scale*betaFStep(i) > 0 ) )
	    {
		if ( dbeta(pivBetaF()(i))+(scale*betaFStep(i)) >= zero_overrun )
		{
		    potscale = -dbeta(pivBetaF()(i))/betaFStep(i);

                    //NiceAssert( potscale >= -zerotol() );

                    if ( testisvnan(potscale) )
                    {
errstream() << "&}";
                        potscale = 1.0;
                    }

		    else if ( potscale < 0 )
		    {
errstream() << "}";
//errstream() << "WARNING: scaling outside bounds (type 7) " << potscale << "\n";
			potscale = 0;
		    }

		    if ( potscale < scale )
		    {
			scale = potscale;
			alphaFIndex = -1;
			betaFIndex  = i;
			betaCIndex  = -1;
			stateChange = -1;
			res = 1;
		    }
		}
	    }

	    else if ( ( dbetaRestrict(pivBetaF()(i)) == 1 ) && ( scale*betaFStep(i) < 0 ) )
	    {
		if ( dbeta(pivBetaF()(i))+(scale*betaFStep(i)) <= -zero_overrun )
		{
		    potscale = -dbeta(pivBetaF()(i))/betaFStep(i);

                    //NiceAssert( potscale >= -zerotol() );

                    if ( testisvnan(potscale) )
                    {
errstream() << "&{";
                        potscale = 1.0;
                    }

		    else if ( potscale < 0 )
		    {
errstream() << "{";
//errstream() << "WARNING: scaling outside bounds (type 8) " << potscale << "\n";
			potscale = 0;
		    }

		    if ( potscale < scale )
		    {
			scale = potscale;
			alphaFIndex = -1;
			betaFIndex  = i;
			betaCIndex  = -1;
			stateChange = +1;
			res = 1;
		    }
		}
	    }
	}
    }

    if ( bNC() && !bailout )
    {
	// Note: it is possible for the algorithm to start with an
	// infeasible solution.  Hence even if asize == bsize == 0
	// and hence betaGradStepC == 0 we may still find betas that
	// need to be freed before we can sensibly continue.
        //
	// NB: this function makes one BIG ASSUMPTION, namely that
	//     unrestricted betas are never constrained.

	Vector<double> &betaGradStepC = betaGradStepC_scaleFStepbase; //Vector<double> betaGradStepC(bNC());
        betaGradStepC.resize(bNC());

	betaGradStepC.zero();

        retVector<T> tmpva;
        retVector<T> tmpvb;

	if ( asize )
	{
	    for ( iP = 0 ; iP < asize ; iP++ )
	    {
                betaGradStepC.scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaC(),tmpva,tmpvb),alphaFStep(iP));
	    }
	}

	if ( bsize )
	{
	    for ( iP = 0 ; iP < bsize ; iP++ )
	    {
                betaGradStepC.scaleAddBR(Gn(pivBetaF()(iP),pivBetaC(),tmpva,tmpvb),betaFStep(iP));
	    }
	}

	for ( iP = 0 ; ( iP < bNC() ) && !bailout ; iP++ )
	{
            if ( testisvnan(dbetaGrad(pivBetaC()(iP))) )
            {
                bailout = 5;
            }

            else if ( testisvnan(betaGradStepC(iP)) )
            {
                bailout = 6;
            }

	    else if ( ( dbetaRestrict(pivBetaC()(iP)) == 1 ) && ( dbetaGrad(pivBetaC()(iP))+(scale*betaGradStepC(iP)) > 0 ) )
	    {
		potscale = -1/zerotol(); // very big number

                if ( abs2(betaGradStepC(iP)) >= zerotol() )
		{
		    potscale = -dbetaGrad(pivBetaC()(iP))/betaGradStepC(iP);

                    if ( testisvnan(potscale) )
                    {
errstream() << "&&";
                        potscale = 1.0;
                    }
		}

		// NB: it is possible for potscale to be negative.  This
		// indicates that our current solution is infeasible if
		// we assume each beta is a lagrange multiplier for an
		// inequality constraint.

		if ( potscale < scale )
		{
		    scale = potscale;
		    alphaFIndex = -1;
		    betaFIndex  = -1;
		    betaCIndex  = iP;
		    stateChange = +1;
		    res = 1;
		}
	    }

	    else if ( ( dbetaRestrict(pivBetaC()(iP)) == 2 ) && ( dbetaGrad(pivBetaC()(iP))+betaGradStepC(iP) < 0 ) )
	    {
		potscale = -1/zerotol(); // very big number

                if ( abs2(betaGradStepC(iP)) >= zerotol() )
		{
		    potscale = -dbetaGrad(pivBetaC()(iP))/betaGradStepC(iP);

                    if ( testisvnan(potscale) )
                    {
errstream() << "&#";
                        potscale = 1.0;
                    }
		}

		// NB: it is possible for potscale to be negative.  This
		// indicates that our current solution is infeasible if
		// we assume each beta is a lagrange multiplier for an
		// inequality constraint.

		if ( potscale < scale )
		{
		    scale = potscale;
		    alphaFIndex = -1;
		    betaFIndex  = -1;
		    betaCIndex  = iP;
		    stateChange = -1;
		    res = 1;
		}
	    }
	}
    }

    if ( scale <= 0 )
    {
	// This can only come about if we started non-feasibly, and the step would
	// need to go backwards to *achieve* feasibility.  This is fine so long as
        // the code responds by modBetaCtoF(betaCIndex).

	scale = 0;
    }

    if ( res && ( scale != 1 ) )
    {
	alphaFStep.scale(scale);
	betaFStep.scale(scale);
    }

    return res;
}

template <class T, class S>
void optState<T,S>::alphaStep(int i, const T &alphaStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int dontCheckState)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );
    NiceAssert( ( alphaState(i) == -1 ) || ( alphaState(i) == +1 ) || dontCheckState );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    alphaStepbase(i,alphaStep,GpGrad,Gpn,dontCheckState);

    return;
}

template <class T, class S>
void optState<T,S>::alphaStephpzero(int i, const T &alphaStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int dontCheckState)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );
    NiceAssert( ( alphaState(i) == -1 ) || ( alphaState(i) == +1 ) || dontCheckState );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    alphaStepbase(i,alphaStep,GpGrad,Gpn,dontCheckState);

    return;
}

template <class T, class S>
void optState<T,S>::alphaStepbase(int i, const T &alphaStep, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, int dontCheckState)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );
    NiceAssert( ( alphaState(i) == -1 ) || ( alphaState(i) == +1 ) || dontCheckState );

    dalpha("&",i) += alphaStep;

    retVector<S>      tmpva;
    retVector<double> tmpvb;

    dalphaGrad.scaleAddBR(GpGrad(i,tmpva),alphaStep);
    dbetaGrad.scaleAddBR(Gpn(i,tmpvb),alphaStep);

    if ( keepfact() )
    {
	int iP = dontCheckState; // this is just done to remove a warning.  It means nothing.

	alphagradstate = -2;
	betagradstate  = -1;

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::betaStep(int i, const T &betaStep , const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int dontCheckState)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < bN() );
    NiceAssert( betaState(i) || dontCheckState );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    betaStepbase(i,betaStep,Gn,Gpn,dontCheckState);

    return;
}

template <class T, class S>
void optState<T,S>::betaStephpzero(int i, const T &betaStep , const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int dontCheckState)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < bN() );
    NiceAssert( betaState(i) || dontCheckState );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    betaStepbase(i,betaStep,Gn,Gpn,dontCheckState);

    return;
}

template <class T, class S>
void optState<T,S>::betaStepbase(int i, const T &betaStep , const Matrix<double> &Gn, const Matrix<double> &Gpn, int dontCheckState)
{
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < bN() );
    NiceAssert( betaState(i) || dontCheckState );

    int j,iP = dontCheckState; // this is just done to remove a warning.  It means nothing.

    dbeta("&",i) += betaStep;

    if ( aN() )
    {
	for ( j = 0 ; j < aN() ; j++ )
	{
	    dalphaGrad("&",j) += (betaStep*Gpn(j,i));
	}
    }

    retVector<double> tmpva;

    dbetaGrad.scaleAddBR(Gn(i,tmpva),betaStep);

    if ( keepfact() )
    {
	alphagradstate = -2;
	betagradstate  = -1;

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::stepFNewton(double scale, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int doCupdate, int doFupdate)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    stepFNewtonbase(scale,bsize,alphaFStep,betaFStep,GpGrad,Gn,Gpn,doCupdate,doFupdate);

    return;
}

template <class T, class S>
void optState<T,S>::stepFNewtonhpzero(double scale, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int doCupdate, int doFupdate)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    stepFNewtonbase(scale,bsize,alphaFStep,betaFStep,GpGrad,Gn,Gpn,doCupdate,doFupdate);

    return;
}

template <class T, class S>
void optState<T,S>::stepFNewtonbase(double scale, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, int doCupdate, int doFupdate)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );

    int iP,jP;

    retVector<T> tmpva;
    retVector<T> tmpvb;
    retVector<T> tmpvc;
    retVector<T> tmpvd;
    retVector<T> tmpve;

    dalpha("&",pivAlphaF(),tmpva)                        += alphaFStep;
    dbeta ("&",pivBetaF (),tmpva)("&",0,1,bsize-1,tmpvb) += betaFStep(0,1,bsize-1,tmpvc);

    if ( doFupdate )
    {
	(dalphaGrad("&",pivAlphaF(),tmpva)).scale(1-scale);
	(dbetaGrad ("&",pivBetaF (),tmpva)("&",0,1,bsize-1,tmpvb)).scale(1-scale);
    }

    else
    {
	(dbetaGrad("&",pivBetaF(),tmpva)("&",0,1,bsize-1,tmpvb)).scale(1-scale);
    }

    if ( aNF() )
    {
	if ( doCupdate )
	{
	    for ( iP = 0 ; iP < aNF() ; iP++ )
	    {
                dalphaGrad("&",pivAlphaLB(),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaLB(),tmpvb,tmpvc),alphaFStep(iP));
                dalphaGrad("&",pivAlphaZ (),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaZ (),tmpvb,tmpvc),alphaFStep(iP));
                dalphaGrad("&",pivAlphaUB(),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaUB(),tmpvb,tmpvc),alphaFStep(iP));

                dbetaGrad("&",pivBetaC(),tmpva).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaC(),tmpvb,tmpvc),alphaFStep(iP));
                dbetaGrad("&",pivBetaF(),tmpva)("&",bsize,1,bNF()-1,tmpvb).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaF(),tmpvc,tmpvd)(bsize,1,bNF()-1,tmpve),alphaFStep(iP));
	    }
	}

	else
	{
	    for ( iP = 0 ; iP < aNF() ; iP++ )
	    {
                dbetaGrad("&",pivBetaC(),tmpva).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaC(),tmpvb,tmpvc),alphaFStep(iP));
                dbetaGrad("&",pivBetaF(),tmpva)("&",bsize,1,bNF()-1,tmpvb).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaF(),tmpvc,tmpvd)(bsize,1,bNF()-1,tmpve),alphaFStep(iP));
	    }
	}
    }

    if ( bsize )
    {
	for ( iP = 0 ; iP < bsize ; iP++ )
	{
	    if ( aNLB() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNLB() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaLB()(jP)) += (betaFStep(iP)*Gpn(pivAlphaLB()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNZ() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNZ() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaZ()(jP)) += (betaFStep(iP)*Gpn(pivAlphaZ()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNUB() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNUB() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaUB()(jP)) += (betaFStep(iP)*Gpn(pivAlphaUB()(jP),pivBetaF()(iP)));
		}
	    }

            dbetaGrad("&",pivBetaC(),tmpva).scaleAddBR(Gn(pivBetaF()(iP),pivBetaC(),tmpvb,tmpvc),betaFStep(iP));
            dbetaGrad("&",pivBetaF(),tmpva)("&",bsize,1,bNF()-1,tmpvb).scaleAddBR(Gn(pivBetaF()(iP),pivBetaF(),tmpvc,tmpvd)(bsize,1,bNF()-1,tmpve),betaFStep(iP));
	}
    }

    if ( keepfact() )
    {
	if ( bsize == bNF()-1 )
	{
	    if ( ( betagradstate == -1 ) || ( betagradstate == bNF()-1 ) )
	    {
                betagradstate = bNF()-1;
	    }

	    else
	    {
                betagradstate = -2;
	    }
	}

	else if ( bsize < bNF()-1 )
	{
            betagradstate = -2;
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::stepFNewtonFull(int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int doCupdate, int doFupdate)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    stepFNewtonFullbase(bsize,alphaFStep,betaFStep,GpGrad,Gn,Gpn,doCupdate,doFupdate);

    return;
}

template <class T, class S>
void optState<T,S>::stepFNewtonFullhpzero(int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int doCupdate, int doFupdate)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    stepFNewtonFullbase(bsize,alphaFStep,betaFStep,GpGrad,Gn,Gpn,doCupdate,doFupdate);

    return;
}

template <class T, class S>
void optState<T,S>::stepFNewtonFullbase(int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, int doCupdate, int doFupdate)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );

    int iP,jP;

    retVector<T> tmpva;
    retVector<T> tmpvb;
    retVector<T> tmpvc;
    retVector<T> tmpvd;
    retVector<T> tmpve;

    dalpha("&",pivAlphaF(),tmpva)                        += alphaFStep;
    dbeta ("&",pivBetaF (),tmpva)("&",0,1,bsize-1,tmpvb) += betaFStep(0,1,bsize-1,tmpvc);

    if ( doFupdate )
    {
	(dalphaGrad("&",pivAlphaF(),tmpva)).zero();
	(dbetaGrad ("&",pivBetaF (),tmpva)("&",0,1,bsize-1,tmpvb)).zero();
    }

    else
    {
	(dbetaGrad ("&",pivBetaF (),tmpva)("&",0,1,bsize-1,tmpvb)).zero();
    }

    if ( aNF() )
    {
	if ( doCupdate )
	{
	    for ( iP = 0 ; iP < aNF() ; iP++ )
	    {
                dalphaGrad("&",pivAlphaLB(),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaLB(),tmpvb,tmpve),alphaFStep(iP));
                dalphaGrad("&",pivAlphaZ (),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaZ (),tmpvb,tmpve),alphaFStep(iP));
                dalphaGrad("&",pivAlphaUB(),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaUB(),tmpvb,tmpve),alphaFStep(iP));

                dbetaGrad("&",pivBetaC(),tmpva).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaC(),tmpvb,tmpve),alphaFStep(iP));
                dbetaGrad("&",pivBetaF(),tmpva)("&",bsize,1,bNF()-1,tmpvb).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaF(),tmpvc,tmpve)(bsize,1,bNF()-1,tmpvd),alphaFStep(iP));
	    }
	}

	else
	{
	    for ( iP = 0 ; iP < aNF() ; iP++ )
	    {
                dbetaGrad("&",pivBetaC(),tmpva).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaC(),tmpvb,tmpvc),alphaFStep(iP));
                dbetaGrad("&",pivBetaF(),tmpva)("&",bsize,1,bNF()-1,tmpvb).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaF(),tmpvc,tmpvd)(bsize,1,bNF()-1,tmpve),alphaFStep(iP));
	    }
	}
    }

    if ( bsize )
    {
	for ( iP = 0 ; iP < bsize ; iP++ )
	{
	    if ( aNLB() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNLB() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaLB()(jP)) += (betaFStep(iP)*Gpn(pivAlphaLB()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNZ() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNZ() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaZ()(jP)) += (betaFStep(iP)*Gpn(pivAlphaZ()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNUB() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNUB() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaUB()(jP)) += (betaFStep(iP)*Gpn(pivAlphaUB()(jP),pivBetaF()(iP)));
		}
	    }

            dbetaGrad("&",pivBetaC(),tmpva).scaleAddBR(Gn(pivBetaF()(iP),pivBetaC(),tmpvb,tmpvc),betaFStep(iP));
            dbetaGrad("&",pivBetaF(),tmpva)("&",bsize,1,bNF()-1,tmpvb).scaleAddBR(Gn(pivBetaF()(iP),pivBetaF(),tmpvc,tmpvd)(bsize,1,bNF()-1,tmpve),betaFStep(iP));
	}
    }

    if ( keepfact() )
    {
	alphagradstate = -1;

	if ( bsize == bNF() )
	{
	    betagradstate = -1;
	}

	else if ( bsize == bNF()-1 )
	{
	    betagradstate = bNF()-1;
	}

	else
	{
	    betagradstate = -2;
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::stepFLinear(int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int doCupdate, int doFupdate)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );
    NiceAssert( asize >= 0 );
    NiceAssert( asize <= aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    stepFLinearbase(asize,bsize,alphaFStep,betaFStep,GpGrad,Gn,Gpn,doCupdate,doFupdate);

    return;
}

template <class T, class S>
void optState<T,S>::stepFLinearhpzero(int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int doCupdate, int doFupdate)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );
    NiceAssert( asize >= 0 );
    NiceAssert( asize <= aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    stepFLinearbase(asize,bsize,alphaFStep,betaFStep,GpGrad,Gn,Gpn,doCupdate,doFupdate);

    return;
}

template <class T, class S>
void optState<T,S>::stepFLinearbase(int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, int doCupdate, int doFupdate)
{ 
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );
    NiceAssert( asize >= 0 );
    NiceAssert( asize <= aNF() );

    int iP,jP;

    retVector<T> tmpva;
    retVector<T> tmpvb;
    retVector<T> tmpvc;
    retVector<T> tmpvd;
    retVector<T> tmpve;

    (dalpha("&",pivAlphaF(),tmpva)("&",0,1,asize-1,tmpvb)) += alphaFStep(0,1,asize-1,tmpvc);
    (dbeta ("&",pivBetaF (),tmpva)("&",0,1,bsize-1,tmpvb)) += betaFStep(0,1,bsize-1,tmpvc);

    if ( asize )
    {
	if ( doCupdate )
	{
	    for ( iP = 0 ; iP < asize ; iP++ )
	    {
                dalphaGrad("&",pivAlphaLB(),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaLB(),tmpvb,tmpvc),alphaFStep(iP));
                dalphaGrad("&",pivAlphaZ (),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaZ (),tmpvb,tmpvc),alphaFStep(iP));
                dalphaGrad("&",pivAlphaUB(),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaUB(),tmpvb,tmpvc),alphaFStep(iP));

                dbetaGrad("&",pivBetaC(),tmpva).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaC(),tmpvb,tmpvc),alphaFStep(iP));

		//NB: in theory asize = pfact+nfact+1.  As optcontext extends the factorisation as
		//far as is possible, and as Gp is positive semi-definite and Gn negative semi-def
		//it follows that the rows of the hessian in the singular part must be linear
		//combinations of those in the non-singular part.  For a linear step the change in
		//the gradients of the nonsingular part are zero, and the changes in the gradients
		//of the singular part are just linear combinations of these, so there *should not*
		//be any change in the gradients of the singular part!  Hence the following lines
		//are commented out and there is no change in alphagradstate or betagradstate.

                (dalphaGrad("&",pivAlphaF(),tmpva)("&",asize,1,aNF()-1,tmpvb)).scaleAddBR((GpGrad(pivAlphaF()(iP),pivAlphaF(),tmpvc,tmpvd)(asize,1,aNF()-1,tmpve)),alphaFStep(iP));
                (dbetaGrad ("&",pivBetaF (),tmpva)("&",bsize,1,bNF()-1,tmpvb)).scaleAddBR((Gpn   (pivAlphaF()(iP),pivBetaF (),tmpvc,tmpvd)(bsize,1,bNF()-1,tmpve)),alphaFStep(iP));

		//PRACTICE: in practice, there must be a mistake somewhere in the above reasoning.

                //dbetaGrad("&",pivBetaF())("&",bsize,1,bNF()-1).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaF())(bsize,1,bNF()-1),alphaFStep(iP));
	    }
	}

	else
	{
	    for ( iP = 0 ; iP < asize ; iP++ )
	    {
                dbetaGrad("&",pivBetaC(),tmpva).scaleAddBR(Gpn(pivAlphaF()(iP),pivBetaC(),tmpvb,tmpvc),alphaFStep(iP));

                (dalphaGrad("&",pivAlphaF(),tmpva)("&",asize,1,aNF()-1,tmpvb)).scaleAddBR((GpGrad(pivAlphaF()(iP),pivAlphaF(),tmpvc,tmpvd)(asize,1,aNF()-1,tmpve)),alphaFStep(iP));
                (dbetaGrad ("&",pivBetaF (),tmpva)("&",bsize,1,bNF()-1,tmpvb)).scaleAddBR((Gpn   (pivAlphaF()(iP),pivBetaF (),tmpvc,tmpvd)(bsize,1,bNF()-1,tmpve)),alphaFStep(iP));
	    }
	}
    }

    if ( bsize )
    {
	for ( iP = 0 ; iP < bsize ; iP++ )
	{
	    if ( aNLB() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNLB() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaLB()(jP)) += (betaFStep(iP)*Gpn(pivAlphaLB()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNZ() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNZ() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaZ()(jP)) += (betaFStep(iP)*Gpn(pivAlphaZ()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNUB() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNUB() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaUB()(jP)) += (betaFStep(iP)*Gpn(pivAlphaUB()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( ( asize < aNF() ) && doFupdate )
	    {
		for ( jP = asize ; jP < aNF() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaF()(jP)) += (betaFStep(iP)*Gpn(pivAlphaF()(jP),pivBetaF()(iP)));
		}
	    }

            dbetaGrad("&",pivBetaC(),tmpva).scaleAddBR(Gn(pivBetaF()(iP),pivBetaC(),tmpvb,tmpvc),betaFStep(iP));
            dbetaGrad("&",pivBetaF(),tmpva)("&",bsize,1,bNF()-1,tmpvb).scaleAddBR(Gn(pivBetaF()(iP),pivBetaF(),tmpvc,tmpvd)(bsize,1,bNF()-1,tmpve),betaFStep(iP));
	}
    }

    if ( keepfact() )
    {
	if ( bsize == bNF()-1 )
	{
	    if ( ( betagradstate == -1 ) || ( betagradstate == bNF()-1 ) )
	    {
                betagradstate = bNF()-1;
	    }

	    else
	    {
                betagradstate = -2;
	    }
	}

	else if ( bsize < bNF()-1 )
	{
            betagradstate = -2;
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::stepFGeneral(int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, int doCupdate, int doFupdate)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );
    NiceAssert( asize >= 0 );
    NiceAssert( asize <= aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    stepFGeneralbase(asize,bsize,alphaFStep,betaFStep,GpGrad,Gn,Gpn,doCupdate,doFupdate);

    return;
}

template <class T, class S>
void optState<T,S>::stepFGeneralhpzero(int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, int doCupdate, int doFupdate)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );
    NiceAssert( asize >= 0 );
    NiceAssert( asize <= aNF() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    stepFGeneralbase(asize,bsize,alphaFStep,betaFStep,GpGrad,Gn,Gpn,doCupdate,doFupdate);

    return;
}

template <class T, class S>
void optState<T,S>::stepFGeneralbase(int asize, int bsize, const Vector<T> &alphaFStep, const Vector<T> &betaFStep, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, int doCupdate, int doFupdate)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( betaFStep.size()  == bNF() );
    NiceAssert( bsize >= 0 );
    NiceAssert( bsize <= bNF() );
    NiceAssert( asize >= 0 );
    NiceAssert( asize <= aNF() );

    int iP,jP;

    retVector<T> tmpva;
    retVector<T> tmpvb;
    retVector<T> tmpvc;

    (dalpha("&",pivAlphaF(),tmpva)("&",0,1,asize-1,tmpvb)) += alphaFStep(0,1,asize-1,tmpvc);
    (dbeta ("&",pivBetaF (),tmpva)("&",0,1,bsize-1,tmpvb)) += betaFStep(0,1,bsize-1,tmpvc);

    if ( asize )
    {
	if ( doCupdate && doFupdate )
	{
	    for ( iP = 0 ; iP < asize ; iP++ )
	    {
                dalphaGrad.scaleAddBR(GpGrad(pivAlphaF()(iP),tmpva),alphaFStep(iP));
                dbetaGrad.scaleAddBR(Gpn(pivAlphaF()(iP),tmpva),alphaFStep(iP));
	    }
	}

	else if ( !doCupdate && doFupdate )
	{
	    for ( iP = 0 ; iP < asize ; iP++ )
	    {
                dalphaGrad("&",pivAlphaF(),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaF(),tmpvb,tmpvc),alphaFStep(iP));
                dbetaGrad.scaleAddBR(Gpn(pivAlphaF()(iP),tmpva),alphaFStep(iP));
	    }
	}

	else if ( doCupdate && !doFupdate )
	{
	    for ( iP = 0 ; iP < asize ; iP++ )
	    {
                dalphaGrad("&",pivAlphaLB(),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaLB(),tmpvb,tmpvc),alphaFStep(iP));
                dalphaGrad("&",pivAlphaZ() ,tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaZ() ,tmpvb,tmpvc),alphaFStep(iP));
                dalphaGrad("&",pivAlphaUB(),tmpva).scaleAddBR(GpGrad(pivAlphaF()(iP),pivAlphaUB(),tmpvb,tmpvc),alphaFStep(iP));
                dbetaGrad.scaleAddBR(Gpn(pivAlphaF()(iP),tmpva),alphaFStep(iP));
	    }
	}
    }

    if ( bsize )
    {
	for ( iP = 0 ; iP < bsize ; iP++ )
	{
	    if ( aNLB() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNLB() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaLB()(jP)) += (betaFStep(iP)*Gpn(pivAlphaLB()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNZ() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNZ() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaZ()(jP)) += (betaFStep(iP)*Gpn(pivAlphaZ()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNUB() && doCupdate )
	    {
		for ( jP = 0 ; jP < aNUB() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaUB()(jP)) += (betaFStep(iP)*Gpn(pivAlphaUB()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNF() && doFupdate )
	    {
		for ( jP = 0 ; jP < aNF() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaF()(jP)) += (betaFStep(iP)*Gpn(pivAlphaF()(jP),pivBetaF()(iP)));
		}
	    }

            dbetaGrad.scaleAddBR(Gn(pivBetaF()(iP),tmpva),betaFStep(iP));
	}
    }

    if ( keepfact() )
    {
	alphagradstate = -2;
	betagradstate  = -2;
    }

    return;
}

template <class T, class S>
void optState<T,S>::updateGradOpt(const Vector<T> &combAlphaFStep, const Vector<T> &combBetaFStep, const Vector<T> &startAlphaGrad, const Vector<int> &FnF, const Vector<int> &startPivAlphaF, const Matrix<S> &GpGrad, const Matrix<double> &Gpn)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );

    int iP,jP;

    retVector<T> tmpva;
    retVector<T> tmpvb;
    retVector<S> tmpvc;

    dalphaGrad("&",FnF,tmpva) = startAlphaGrad(FnF,tmpvb);

    if ( startPivAlphaF.size() )
    {
	for ( iP = 0 ; iP < startPivAlphaF.size() ; iP++ )
	{
            dalphaGrad("&",pivAlphaLB(),tmpva).scaleAddBR(GpGrad(startPivAlphaF(iP),pivAlphaLB(),tmpvb,tmpvc),combAlphaFStep(iP));
            dalphaGrad("&",pivAlphaZ (),tmpva).scaleAddBR(GpGrad(startPivAlphaF(iP),pivAlphaZ (),tmpvb,tmpvc),combAlphaFStep(iP));
            dalphaGrad("&",pivAlphaUB(),tmpva).scaleAddBR(GpGrad(startPivAlphaF(iP),pivAlphaUB(),tmpvb,tmpvc),combAlphaFStep(iP));
	}
    }

    if ( bNF() )
    {
	for ( iP = 0 ; iP < bNF() ; iP++ )
	{
	    if ( aNLB() )
	    {
		for ( jP = 0 ; jP < aNLB() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaLB()(jP)) += (combBetaFStep(iP)*Gpn(pivAlphaLB()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNZ() )
	    {
		for ( jP = 0 ; jP < aNZ() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaZ()(jP)) += (combBetaFStep(iP)*Gpn(pivAlphaZ()(jP),pivBetaF()(iP)));
		}
	    }

	    if ( aNUB() )
	    {
		for ( jP = 0 ; jP < aNUB() ; jP++ )
		{
		    dalphaGrad("&",pivAlphaUB()(jP)) += (combBetaFStep(iP)*Gpn(pivAlphaUB()(jP),pivBetaF()(iP)));
		}
	    }
	}
    }

    return;
}

template <class T, class S>
double optState<T,S>::testGrad(int &aerr, int &berr, Vector<T> &alphaGradTest, Vector<T> &betaGradTest, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int i;

    alphaGradTest = gp;
    betaGradTest  = gn;

    aerr = -1;
    berr = -1;

    if ( aN() )
    {
	for ( i = 0 ; i < aN() ; i++ )
	{
            recalcAlphaGrad(alphaGradTest("&",i),GpGrad,Gpn,gp,hp,i);
	}
    }

    if ( bN() )
    {
	for ( i = 0 ; i < bN() ; i++ )
	{
            recalcBetaGrad(betaGradTest("&",i),Gn,Gpn,gn,i);
	}
    }

    double res = 0;

    if ( alphaGradTest.size() )
    {
	for ( i = 0 ; i < alphaGradTest.size() ; i++ )
	{
            if ( abs2(dalphaGrad(i)-alphaGradTest(i)) > res )
	    {
                res = abs2(dalphaGrad(i)-alphaGradTest(i));
		aerr = i;
                berr = -1;
	    }
	}
    }

    if ( betaGradTest.size() )
    {
	for ( i = 0 ; i < betaGradTest.size() ; i++ )
	{
            if ( abs2(dbetaGrad(i)-betaGradTest(i)) > res )
	    {
                res = abs2(dbetaGrad(i)-betaGradTest(i));
		aerr = -1;
                berr = i;
	    }
	}
    }

    return res;
}

template <class T, class S>
double optState<T,S>::testGradhpzero(int &aerr, int &berr, Vector<T> &alphaGradTest, Vector<T> &betaGradTest, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int i;

    alphaGradTest = gp;
    betaGradTest  = gn;

    aerr = -1;
    berr = -1;

    if ( aN() )
    {
	for ( i = 0 ; i < aN() ; i++ )
	{
            recalcAlphaGradhpzero(alphaGradTest("&",i),GpGrad,Gpn,gp,i);
	}
    }

    if ( bN() )
    {
	for ( i = 0 ; i < bN() ; i++ )
	{
            recalcBetaGrad(betaGradTest("&",i),Gn,Gpn,gn,i);
	}
    }

    double res = 0;

    if ( alphaGradTest.size() )
    {
	for ( i = 0 ; i < alphaGradTest.size() ; i++ )
	{
            if ( abs2(dalphaGrad(i)-alphaGradTest(i)) > res )
	    {
                res = abs2(dalphaGrad(i)-alphaGradTest(i));
		aerr = i;
                berr = -1;
	    }
	}
    }

    if ( betaGradTest.size() )
    {
	for ( i = 0 ; i < betaGradTest.size() ; i++ )
	{
            if ( abs2(dbetaGrad(i)-betaGradTest(i)) > res )
	    {
                res = abs2(dbetaGrad(i)-betaGradTest(i));
		aerr = -1;
                berr = i;
	    }
	}
    }

    return res;
}

template <class T, class S>
double optState<T,S>::testGradInt(int &aerr, int &berr, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );

    Vector<T> alphaGradTest(gp);
    Vector<T> betaGradTest(gn);

    return testGrad(aerr,berr,alphaGradTest,betaGradTest,GpGrad,Gn,Gpn,gp,gn,hp);
}

template <class T, class S>
double optState<T,S>::testGradInthpzero(int &aerr, int &berr, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    Vector<T> alphaGradTest(gp);
    Vector<T> betaGradTest(gn);

    return testGradhpzero(aerr,berr,alphaGradTest,betaGradTest,GpGrad,Gn,Gpn,gp,gn);
}

template <class T, class S>
int optState<T,S>::maxGradNonOpt(int &alphaCIndex, int &betaCIndex, int &stateChange, double &gradmag, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, double toloverride, int checkfree, int ignorebeta)
{
    if ( toloverride < 0 )
    {
        toloverride = dopttol;
    }

//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int i;
    int res = 1;

    gradmag     = toloverride;
    alphaCIndex = -1;
    betaCIndex  = -1;
    stateChange = 0;

    //if ( aNF() > probContext.fact_pfact(Gn,Gpn) )
    //{
    //    gradmag = -1/zerotol();
    //}

    if ( pivAlphaLB().size() )
    {
	for ( i = 0 ; i < pivAlphaLB().size() ; i++ )
	{
	    if ( -dalphaGrad(pivAlphaLB()(i)) > gradmag )
	    {
		gradmag     = -dalphaGrad(pivAlphaLB()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = -2;

                res = 0;
	    }
	}
    }

    if ( pivAlphaZ().size() )
    {
	for ( i = 0 ; i < pivAlphaZ().size() ; i++ )
	{
	    if ( ( dalphaGrad(pivAlphaZ()(i))-hp(pivAlphaZ()(i)) > gradmag ) && ( ( dalphaRestrict(pivAlphaZ()(i)) == 0 ) || ( dalphaRestrict(pivAlphaZ()(i)) == 2 ) ) )
	    {
                gradmag     = dalphaGrad(pivAlphaZ()(i))-hp(pivAlphaZ()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = -1;

                res = 0;
	    }

	    if ( ( -dalphaGrad(pivAlphaZ()(i))-hp(pivAlphaZ()(i)) > gradmag ) && ( ( dalphaRestrict(pivAlphaZ()(i)) == 0 ) || ( dalphaRestrict(pivAlphaZ()(i)) == 1 ) ) )
	    {
                gradmag     = -dalphaGrad(pivAlphaZ()(i))-hp(pivAlphaZ()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = +1;

                res = 0;
	    }
	}
    }

    if ( pivAlphaF().size() && checkfree )
    {
	for ( i = 0 ; i < pivAlphaF().size() ; i++ )
	{
	    if ( dalphaGrad(pivAlphaF()(i))-hp(pivAlphaF()(i)) > gradmag )
	    {
                gradmag     = dalphaGrad(pivAlphaF()(i))-hp(pivAlphaF()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = 0;

                res = 0;
	    }

	    if ( -dalphaGrad(pivAlphaF()(i))-hp(pivAlphaF()(i)) > gradmag )
	    {
                gradmag     = -dalphaGrad(pivAlphaF()(i))-hp(pivAlphaF()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = 0;

                res = 0;
	    }
	}
    }

    if ( pivAlphaUB().size() )
    {
	for ( i = 0 ; i < pivAlphaUB().size() ; i++ )
	{
	    if ( dalphaGrad(pivAlphaUB()(i)) > gradmag )
	    {
                gradmag     = dalphaGrad(pivAlphaUB()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = +2;

                res = 0;
	    }
	}
    }

    if ( pivBetaC().size() && !ignorebeta )
    {
	for ( i = 0 ; i < pivBetaC().size() ; i++ )
	{
	    if ( ( -dbetaGrad(pivBetaC()(i)) > gradmag ) && ( ( dbetaRestrict(pivBetaC()(i)) == 0 ) || ( dbetaRestrict(pivBetaC()(i)) == 2 ) ) )
	    {
                gradmag     = -dbetaGrad(pivBetaC()(i));
		alphaCIndex = -1;
		betaCIndex  = i;
                stateChange = -1;

                res = 0;
	    }

	    if ( ( dbetaGrad(pivBetaC()(i)) > gradmag ) && ( ( dbetaRestrict(pivBetaC()(i)) == 0 ) || ( dbetaRestrict(pivBetaC()(i)) == 1 ) ) )
	    {
                gradmag     = dbetaGrad(pivBetaC()(i));
		alphaCIndex = -1;
		betaCIndex  = i;
                stateChange = +1;

                res = 0;
	    }
	}
    }

    if ( pivBetaF().size() && ( betagradstate != -1 ) && !ignorebeta )
    {
	double locgradmag = toloverride;

	for ( i = 0 ; i < pivBetaF().size() ; i++ )
	{
            if ( abs2(dbetaGrad(pivBetaF()(i))) > locgradmag )
	    {
                locgradmag = abs2(dbetaGrad(pivBetaF()(i)));

                gradmag     = locgradmag;
		alphaCIndex = -1;
		betaCIndex  = -1;
		stateChange = 0;

                res = 0;
	    }
	}

	// At this point, if res == 0 and alphaCIndex == betaCIndex == -1 then
	// we have a problem.  We have taken a Newton step (by assumption), but
	// not all of beta gradient has been zeroed due to not all of betaF being
	// incorporated in the factorisation when, before the step, the solution
	// was not feasible.
        //
	// This condition leads to an impasse.  The optimiser must respond to this
	// be triggering some form of presolver to ensure feasibility before
        // proceeding.
    }

    return res;
}

template <class T, class S>
int optState<T,S>::maxGradNonOpthpzero(int &alphaCIndex, int &betaCIndex, int &stateChange, double &gradmag, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, double toloverride, int checkfree, int ignorebeta)
{
    if ( toloverride < 0 )
    {
        toloverride = dopttol;
    }

//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int i;
    int res = 1;

    gradmag     = toloverride;
    alphaCIndex = -1;
    betaCIndex  = -1;
    stateChange = 0;

    //if ( aNF() > probContext.fact_pfact(Gn,Gpn) )
    //{
    //    gradmag = -1/zerotol();
    //}

    if ( pivAlphaLB().size() )
    {
	for ( i = 0 ; i < pivAlphaLB().size() ; i++ )
	{
	    if ( -dalphaGrad(pivAlphaLB()(i)) > gradmag )
	    {
		gradmag     = -dalphaGrad(pivAlphaLB()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = -2;

                res = 0;
	    }
	}
    }

    if ( pivAlphaZ().size() )
    {
	for ( i = 0 ; i < pivAlphaZ().size() ; i++ )
	{
	    if ( ( dalphaGrad(pivAlphaZ()(i)) > gradmag ) && ( ( dalphaRestrict(pivAlphaZ()(i)) == 0 ) || ( dalphaRestrict(pivAlphaZ()(i)) == 2 ) ) )
	    {
                gradmag     = dalphaGrad(pivAlphaZ()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = -1;

                res = 0;
	    }

	    if ( ( -dalphaGrad(pivAlphaZ()(i)) > gradmag ) && ( ( dalphaRestrict(pivAlphaZ()(i)) == 0 ) || ( dalphaRestrict(pivAlphaZ()(i)) == 1 ) ) )
	    {
                gradmag     = -dalphaGrad(pivAlphaZ()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = +1;

                res = 0;
	    }
	}
    }

    if ( pivAlphaF().size() && checkfree )
    {
	for ( i = 0 ; i < pivAlphaF().size() ; i++ )
	{
	    if ( dalphaGrad(pivAlphaF()(i)) > gradmag )
	    {
                gradmag     = dalphaGrad(pivAlphaF()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = 0;

                res = 0;
	    }

	    if ( -dalphaGrad(pivAlphaF()(i)) > gradmag )
	    {
                gradmag     = -dalphaGrad(pivAlphaF()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = 0;

                res = 0;
	    }
	}
    }

    if ( pivAlphaUB().size() )
    {
	for ( i = 0 ; i < pivAlphaUB().size() ; i++ )
	{
	    if ( dalphaGrad(pivAlphaUB()(i)) > gradmag )
	    {
                gradmag     = dalphaGrad(pivAlphaUB()(i));
		alphaCIndex = i;
		betaCIndex  = -1;
                stateChange = +2;

                res = 0;
	    }
	}
    }

    if ( pivBetaC().size() && !ignorebeta )
    {
	for ( i = 0 ; i < pivBetaC().size() ; i++ )
	{
	    if ( ( -dbetaGrad(pivBetaC()(i)) > gradmag ) && ( ( dbetaRestrict(pivBetaC()(i)) == 0 ) || ( dbetaRestrict(pivBetaC()(i)) == 2 ) ) )
	    {
                gradmag     = -dbetaGrad(pivBetaC()(i));
		alphaCIndex = -1;
		betaCIndex  = i;
                stateChange = -1;

                res = 0;
	    }

	    if ( ( dbetaGrad(pivBetaC()(i)) > gradmag ) && ( ( dbetaRestrict(pivBetaC()(i)) == 0 ) || ( dbetaRestrict(pivBetaC()(i)) == 1 ) ) )
	    {
                gradmag     = dbetaGrad(pivBetaC()(i));
		alphaCIndex = -1;
		betaCIndex  = i;
                stateChange = +1;

                res = 0;
	    }
	}
    }

    if ( pivBetaF().size() && ( betagradstate != -1 ) && !ignorebeta )
    {
	double locgradmag = toloverride;

	for ( i = 0 ; i < pivBetaF().size() ; i++ )
	{
            if ( abs2(dbetaGrad(pivBetaF()(i))) > locgradmag )
	    {
                locgradmag = abs2(dbetaGrad(pivBetaF()(i)));

                gradmag     = locgradmag;
		alphaCIndex = -1;
		betaCIndex  = -1;
		stateChange = 0;

                res = 0;
	    }
	}

	// At this point, if res == 0 and alphaCIndex == betaCIndex == -1 then
	// we have a problem.  We have taken a Newton step (by assumption), but
	// not all of beta gradient has been zeroed due to not all of betaF being
	// incorporated in the factorisation when, before the step, the solution
	// was not feasible.
        //
	// This condition leads to an impasse.  The optimiser must respond to this
	// be triggering some form of presolver to ensure feasibility before
        // proceeding.
    }

    return res;
}

template <class T, class S>
int optState<T,S>::maxBetaGradNonOpt(int &betaCIndex, int &stateChange, double &gradmag, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int i;
    int res = 1;

    gradmag     = dopttol;
    betaCIndex  = -1;
    stateChange = 0;

    if ( bN() )
    {
	for ( i = 0 ; i < bN() ; i++ )
	{
	    if ( ( -dbetaGrad(i) > gradmag ) && ( ( dbetaRestrict(i) == 0 ) || ( dbetaRestrict(i) == 2 ) ) )
	    {
                gradmag     = -dbetaGrad(i);
		betaCIndex  = i;
                stateChange = -1;

                res = 0;
	    }

	    if ( ( dbetaGrad(i) > gradmag ) && ( ( dbetaRestrict(i) == 0 ) || ( dbetaRestrict(i) == 1 ) ) )
	    {
                gradmag     = dbetaGrad(i);
		betaCIndex  = i;
                stateChange = +1;

                res = 0;
	    }
	}
    }

    return res;
}

template <class T, class S>
int optState<T,S>::maxBetaGradNonOpthpzero(int &betaCIndex, int &stateChange, double &gradmag, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int i;
    int res = 1;

    gradmag     = -1/zerotol();;
    betaCIndex  = -1;
    stateChange = 0;

    if ( bN() )
    {
	for ( i = 0 ; i < bN() ; i++ )
	{
	    if ( ( -dbetaGrad(i) > gradmag ) && ( ( dbetaRestrict(i) == 0 ) || ( dbetaRestrict(i) == 2 ) ) )
	    {
                gradmag     = -dbetaGrad(i);
		betaCIndex  = i;
                stateChange = -1;

                res = 0;
	    }

	    if ( ( dbetaGrad(i) > gradmag ) && ( ( dbetaRestrict(i) == 0 ) || ( dbetaRestrict(i) == 1 ) ) )
	    {
                gradmag     = dbetaGrad(i);
		betaCIndex  = i;
                stateChange = +1;

                res = 0;
	    }
	}
    }

    return res;
}


template <class T, class S>
int optState<T,S>::initGradBeta(const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    int iP;

    if ( bNC() )
    {
	for ( iP = bNC()-1 ; iP >= 0 ; iP-- )
	{
	    if ( dbetaRestrict(pivBetaC()(iP)) == 0 )
	    {
		modBetaCtoF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
	    }

	    else if ( ( dbetaRestrict(iP) == 1 ) && ( dbetaGrad(pivBetaC()(iP)) > dopttol ) )
	    {
		modBetaCtoF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
	    }

	    else if ( ( dbetaRestrict(iP) == 2 ) && ( dbetaGrad(pivBetaC()(iP)) < -dopttol ) )
	    {
		modBetaCtoF(iP,Gp,GpGrad,Gn,Gpn,gp,gn,hp);
	    }
	}
    }

    if ( keepfact() )
    {
	alphagradstate = -1;
	betagradstate  = -1;

	if ( aNF() )
	{
	    for ( iP = 0 ; iP < aNF() ; iP++ )
	    {
                if ( abs2(dalphaGrad(pivAlphaF()(iP))) > dopttol )
		{
		    if ( alphagradstate == -1 )
		    {
			alphagradstate = iP;
		    }

		    else if ( alphagradstate >= 0 )
		    {
			alphagradstate = -2;

			break;
		    }
		}
	    }
	}

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return betagradstate;
}

template <class T, class S>
int optState<T,S>::initGradBetahpzero(const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    int iP;

    if ( bNC() )
    {
	for ( iP = bNC()-1 ; iP >= 0 ; iP-- )
	{
	    if ( dbetaRestrict(pivBetaC()(iP)) == 0 )
	    {
		modBetaCtoFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
	    }

	    else if ( ( dbetaRestrict(iP) == 1 ) && ( dbetaGrad(pivBetaC()(iP)) > dopttol ) )
	    {
		modBetaCtoFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
	    }

	    else if ( ( dbetaRestrict(iP) == 2 ) && ( dbetaGrad(pivBetaC()(iP)) < -dopttol ) )
	    {
		modBetaCtoFhpzero(iP,Gp,GpGrad,Gn,Gpn,gp,gn);
	    }
	}
    }

    if ( keepfact() )
    {
	alphagradstate = -1;
	betagradstate  = -1;

	if ( aNF() )
	{
	    for ( iP = 0 ; iP < aNF() ; iP++ )
	    {
                if ( abs2(dalphaGrad(pivAlphaF()(iP))) > dopttol )
		{
		    if ( alphagradstate == -1 )
		    {
			alphagradstate = iP;
		    }

		    else if ( alphagradstate >= 0 )
		    {
			alphagradstate = -2;

			break;
		    }
		}
	    }
	}

	if ( bNF() )
	{
	    for ( iP = 0 ; iP < bNF() ; iP++ )
	    {
                if ( abs2(dbetaGrad(pivBetaF()(iP))) > dopttol )
		{
		    if ( betagradstate == -1 )
		    {
			betagradstate = iP;
		    }

		    else if ( betagradstate >= 0 )
		    {
			betagradstate = -2;

			break;
		    }
		}
	    }
	}
    }

    return betagradstate;
}

template <class T, class S>
int optState<T,S>::fact_calcStep(Vector<T> &stepAlpha, Vector<T> &stepBeta, int &asize, int &bsize, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp, const Vector<double> &lb, const Vector<double> &ub)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( ub.size() == aN() );
    NiceAssert( stepAlpha.size() == aN() );
    NiceAssert( stepBeta.size()  == bN() );
    NiceAssert( keepfact() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    return fact_calcStepbase(stepAlpha,stepBeta,asize,bsize,Gp,Gn,Gpn,lb,ub);
}

template <class T, class S>
int optState<T,S>::fact_calcStephpzero(Vector<T> &stepAlpha, Vector<T> &stepBeta, int &asize, int &bsize, const Matrix<double> &Gp, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<double> &lb, const Vector<double> &ub)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( ub.size() == aN() );
    NiceAssert( stepAlpha.size() == aN() );
    NiceAssert( stepBeta.size()  == bN() );
    NiceAssert( keepfact() );

    if ( gradFixAlphaInd || gradFixBetaInd )
    {
	fixGradhpzero(GpGrad,Gn,Gpn,gp,gn);
    }

    return fact_calcStepbase(stepAlpha,stepBeta,asize,bsize,Gp,Gn,Gpn,lb,ub);
}

#define SCALEMAX 1000

template <class T, class S>
int optState<T,S>::fact_calcStepbase(Vector<T> &stepAlpha, Vector<T> &stepBeta, int &asize, int &bsize, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &lb, const Vector<double> &ub)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( lb.size() == aN() );
    NiceAssert( ub.size() == aN() );
    NiceAssert( stepAlpha.size() == aN() );
    NiceAssert( stepBeta.size()  == bN() );
    NiceAssert( keepfact() );

    int res;

    asize = probContext.fact_pfact(Gn,Gpn);
    bsize = probContext.fact_nfact(Gn,Gpn);

    retVector<T> tmpva;
    retVector<T> tmpvb;
    retVector<T> tmpvc;
    retVector<T> tmpvd;

    if ( asize == aNF() )
    {
	res = 0;

	probContext.fact_minverse(stepAlpha,stepBeta,dalphaGrad,dbetaGrad,Gn,Gpn,alphagradstate,betagradstate);

	stepAlpha("&",pivAlphaF(),tmpva).negate();
	stepBeta("&",pivBetaF(),tmpva)("&",0,1,bsize-1,tmpvb).negate();
	stepBeta("&",pivBetaF(),tmpva)("&",bsize,1,bNF()-1,tmpvb).zero();
    }

    else if ( ( bsize == bNF() ) || ( asize == aNF() ) )
    {
	res = 1;

        asize++;

	// Want to solve:
	//
	// [ stepAlpha(pivAlphaF())(0,1,pfact()-1) ] = inv(H) [ h ]     (ie. call near invert)
	// [ stepBeta (pivBetaF ())(0,1,nfact()-1) ]          [ 1 ]
	//   stepAlpha(pivAlphaF())(pfact()) = -1
	//
	// where H is the usual mix of Gp etc and everything is appropriately
	// pivotted and mixed around.

	probContext.fact_near_invert(stepAlpha,stepBeta,Gp,Gn,Gpn);

        stepAlpha("&",pivAlphaF(),tmpva)("&",asize-1) = -1;

	// Need to ensure that this is a direction of linear descent in
	// alpha and not a direction of linear ascent.  Check and negate if
	// required.  The exception is if there are beta components not in
	// the linear part of the step, in which case we want to minimise
        // these beta gradients after the step.

	double chng = 0;
        T temp;

        chng += real(twoProductNoConj(temp,stepAlpha(pivAlphaF(),tmpva)(0,1,asize-1,tmpvb),dalphaGrad(pivAlphaF(),tmpvc)(0,1,asize-1,tmpvd)));
        chng -= real(twoProductNoConj(temp,stepBeta (pivBetaF (),tmpva)(0,1,bsize-1,tmpvb),dbetaGrad (pivBetaF (),tmpvc)(0,1,bsize-1,tmpvd)));

	if ( chng > 0.0 )
	{
	    (stepAlpha("&",pivAlphaF(),tmpva)("&",0,1,probContext.fact_pfact(Gn,Gpn)  ,tmpvb)).negate();
            (stepBeta ("&",pivBetaF (),tmpva)("&",0,1,probContext.fact_nfact(Gn,Gpn)-1,tmpvb)).negate();
	}

        int iP,jP;
	double scale = 1;
        double pscale;
        double distancetoedge = 0;

	jP = -1;

	for ( iP = 0 ; iP < asize ; iP++ )
	{
	    if ( ( alphaState(pivAlphaF()(iP)) > 0 ) && ( stepAlpha(pivAlphaF()(iP)) >= zerotol() ) )
	    {
                distancetoedge = ub(pivAlphaF()(iP)) - dalpha(pivAlphaF()(iP));
	    }

	    else if ( ( alphaState(pivAlphaF()(iP)) > 0 ) && ( stepAlpha(pivAlphaF()(iP)) <= -zerotol() ) && ( alphaRestrict(pivAlphaF()(iP)) ) )
	    {
                distancetoedge = dalpha(pivAlphaF()(iP));
	    }

	    else if ( ( alphaState(pivAlphaF()(iP)) > 0 ) && ( stepAlpha(pivAlphaF()(iP)) <= -zerotol() ) && ( !alphaRestrict(pivAlphaF()(iP)) ) )
	    {
                distancetoedge = dalpha(pivAlphaF()(iP)) - lb(pivAlphaF()(iP));
	    }

	    else if ( ( alphaState(pivAlphaF()(iP)) < 0 ) && ( stepAlpha(pivAlphaF()(iP)) >= zerotol() ) && ( !alphaRestrict(pivAlphaF()(iP)) ) )
	    {
                distancetoedge = ub(pivAlphaF()(iP)) - dalpha(pivAlphaF()(iP));
	    }

	    else if ( ( alphaState(pivAlphaF()(iP)) < 0 ) && ( stepAlpha(pivAlphaF()(iP)) >= zerotol() ) && ( alphaRestrict(pivAlphaF()(iP)) ) )
	    {
                distancetoedge = -dalpha(pivAlphaF()(iP));
	    }

	    else if ( ( alphaState(pivAlphaF()(iP)) < 0 ) && ( stepAlpha(pivAlphaF()(iP)) <= -zerotol() ) )
	    {
                distancetoedge = dalpha(pivAlphaF()(iP)) - lb(pivAlphaF()(iP));
	    }

	    else
	    {
		distancetoedge = -1;
	    }

	    if ( ( distancetoedge >= zerotol() ) && ( stepAlpha(pivAlphaF()(iP)) >= zerotol() ) )
	    {
                pscale = distancetoedge / stepAlpha(pivAlphaF()(iP));
	    }

	    else if ( ( distancetoedge >= zerotol() ) && ( stepAlpha(pivAlphaF()(iP)) <= -zerotol() ) )
	    {
                pscale = distancetoedge / -stepAlpha(pivAlphaF()(iP));
	    }

	    else
	    {
		pscale = -1;
	    }

	    if ( ( pscale >= zerotol() ) && ( ( jP == -1 ) || ( pscale < scale ) ) )
	    {
		scale = pscale;
                jP = iP;
	    }
	}

	if ( scale < 1 )
	{
            scale = 1;
	}

        scale *= 1.1;

	stepAlpha("&",pivAlphaF(),tmpva)("&",0,1,probContext.fact_pfact(Gn,Gpn)  ,tmpvb) *= scale;
	stepBeta ("&",pivBetaF (),tmpva)("&",0,1,probContext.fact_nfact(Gn,Gpn)-1,tmpvb) *= scale;

        #ifdef DEBUGOPT
        errstream() << "+----------------------------------\n";
        errstream() << "| \n";
        errstream() << "| Step calculation method 1:\n\n";
        errstream() << "| \n";
        errstream() << "| asize = " << asize << "\n";
        errstream() << "| bsize = " << bsize << "\n";
        errstream() << "| aN = " << aNF() << "\n";
        errstream() << "| bN = " << bNF() << "\n";
        errstream() << "| \n";
        errstream() << "| alpha step = " << stepAlpha(pivAlphaF()) << "\n";
        errstream() << "| beta step = " << stepBeta(pivBetaF()) << "\n";
        errstream() << "| \n";
        errstream() << "+----------------------------------\n";
        #endif
    }

    else
    {
        res = 2;

        // At this point aNF() > 0 and bNF() > 0; and asize < aNF() and bsize < bNF()

        // It is likely that the factorisation is non-existent despite the Hessian 
        // inverse existing.  Fallback is projected gradient descent assuming optimality
        // in free betas (which is true for the quadratic solver).
        //
        // -inv( [  I   Gpn ] ) [ e ] = -[ I - Gpn.inv(Gpn'.Gpn).Gpn'  Gpn.inv(Gpn'.Gpn) ] [ e ]
        //       [ Gpn'  0  ]   [ f ]    [         inv(Gpn'.Gpn).Gpn'     -inv(Gpn'.Gpn) ] [ f ]
        //
        //                            = -[ e - Gpn.g ]
        //                               [         g ]
        //
        //
        // p = Gpn'.e
        // h = p-f
        // g = inv(Gpn'.Gpn).h
        // q = Gpn.g
        //
        // So we set asize = aNF() then build up Gpn until Gpn'.Gpn is no longer singular.
        // We actually use Cholesky on Gpn'.Gpn to simplify this.

        asize = aNF();
        bsize = 0; // will grow this

        int i,j,k;

        // First we construct the complete Gpn'.Gpn matrix

        Matrix<double> GpnGpn(bNF(),bNF());

        for ( i = 0 ; i < bNF() ; i++ )
        {
            for ( j = 0 ; j <= i ; j++ )
            {
                GpnGpn("&",i,j) = 0.0;

                for ( k = 0 ; k < aNF() ; k++ )
                {
                    GpnGpn("&",i,j) += Gpn(pivAlphaF()(k),pivBetaF()(i))*Gpn(pivAlphaF()(k),pivBetaF()(j));
                }

                GpnGpn("&",j,i) = GpnGpn(i,j);
            }
        }

        // Next construct the largest index subset of GpnGpn that is non-singular

        retVector<int>    tmpva;
        retVector<int>    tmpvb;
        retVector<int>    tmpvc;
        retVector<int>    tmpvd;
        retVector<double> tmpve;
        retVector<T>      tmpvf;
        retVector<T>      tmpvg;
        retVector<int>    tmpvh;
        retVector<int>    tmpvi;
        retVector<double> tmpvj;
        retVector<int>    tmpvk;
        retMatrix<T>      tmpma;
        retMatrix<double> tmpmb;

        Matrix<double> GnFake(0,0);
        Matrix<double> GpnFake(bsize,0);
        Chol<double> GpnGpnFact(GpnGpn(zeroint(),1,bsize-1,zeroint(),1,bsize-1,tmpmb),GnFake,GpnFake,onedoublevec(bsize,tmpvj),zerotol(),0);
        Vector<int> GpnInd(cntintvec(bNF(),tmpvk));

        int isok = 1;

        while ( ( bsize < bNF() ) && isok )
        {
            isok = 0;

            for ( i = bsize ; ( i < bNF() ) && !isok ; i++ )
            {
                GpnInd.squareswap(bsize,i);
                GpnFake.addRow(bsize);

                if ( !(GpnGpnFact.add(bsize,GpnGpn(GpnInd(zeroint(),1,bsize,tmpvh),GpnInd(zeroint(),1,bsize,tmpvi),tmpmb),GnFake,GpnFake,onedoublevec(bsize+1,tmpvj))) )
                {
                    bsize++;
                    isok = 1;
                }

                else
                {
                    GpnFake.removeRow(bsize);
                    GpnGpnFact.remove(bsize,GpnGpn(GpnInd(zeroint(),1,bsize-1,tmpvh),GpnInd(zeroint(),1,bsize-1,tmpvi),tmpmb),GnFake,GpnFake,onedoublevec(bsize,tmpvj));
                }
            }
        }

        // Calculate g

        Vector<double> p(bNF());
        Vector<double> g(bNF());

        p = 0.0;
        g = 0.0;

        p("&",GpnInd(zeroint(),1,bsize-1,tmpva),tmpve)  = dalphaGrad(pivAlphaF(),tmpvf)*Gpn(pivAlphaF(),pivBetaF()(GpnInd(zeroint(),1,bsize-1,tmpvc),tmpvd),tmpma);
        p("&",GpnInd(zeroint(),1,bsize-1,tmpva),tmpve) -= dbetaGrad(pivBetaF()(GpnInd(zeroint(),1,bsize-1,tmpvc),tmpvd),tmpvf);

        Vector<double> bndummy;
        Vector<double> andummy;

        GpnGpnFact.minverse(g("&",GpnInd(zeroint(),1,bsize-1,tmpva),tmpvf),andummy,p(GpnInd(zeroint(),1,bsize-1,tmpvc),tmpve),bndummy);

        // Calculate step

        stepBeta("&",pivBetaF()(GpnInd,tmpva),tmpvf) = g(GpnInd,tmpve);
        stepBeta("&",pivBetaF()(GpnInd,tmpva),tmpvf).negate();

        stepAlpha("&",pivAlphaF(),tmpvf)  = dalphaGrad(pivAlphaF(),tmpvg);
        stepAlpha("&",pivAlphaF(),tmpvf) -= Gpn(pivAlphaF(),pivBetaF()(GpnInd(zeroint(),1,bsize-1,tmpva),tmpvc),tmpma)*g(GpnInd(zeroint(),1,bsize-1,tmpvb),tmpve);
        stepAlpha("&",pivAlphaF(),tmpvf).negate();

        // Line-search down to ensure max decrease in objective
        //
        // dR = scale^2 1/2 [ stepalpha ]' [ Gp   Gpn ] [ stepalpha ] + scale [ stepalpha ]' [ alphagrad ]
        //                  [ stepbeta  ]  [ Gpn' Gn  ] [ stepbeta  ]         [ stepbeta  ]  [ betaGrad  ]
        //
        // dR/dscale = scale [ stepalpha ]' [ Gp   Gpn ] [ stepalpha ] + [ stepalpha ]' [ alphagrad ]
        //                   [ stepbeta  ]  [ Gpn' Gn  ] [ stepbeta  ]   [ stepbeta  ]  [ betaGrad  ]
        //
        // dR/dscale = 0

        double dRleftaa = 0.0;
        double dRleftab = 0.0;
        double dRleftbb = 0.0;
        double dRrighta = 0.0;
        double dRrightb = 0.0;

        twoProductNoConj(dRleftaa,dalpha(pivAlphaF(),tmpvf),Gp(pivAlphaF(),pivAlphaF(),tmpma)*dalpha(pivAlphaF(),tmpvg));
        twoProductNoConj(dRleftab,dalpha(pivAlphaF(),tmpvf)*Gpn(pivAlphaF(),pivBetaF(),tmpma),dbeta(pivBetaF(),tmpvg));
        twoProductNoConj(dRleftaa,dbeta(pivBetaF(),tmpvf),Gn(pivBetaF(),pivBetaF(),tmpma)*dbeta(pivBetaF(),tmpvg));
        twoProductNoConj(dRrighta,dalpha(pivAlphaF(),tmpvf),stepAlpha(pivAlphaF(),tmpvg));
        twoProductNoConj(dRrightb,dbeta(pivBetaF(),tmpvf),stepBeta(pivBetaF(),tmpvg));

        double scale = ( (dRleftaa+dRleftab+dRleftab+dRleftbb) > zerotol() ) ? -((dRrighta+dRrightb)/(dRleftaa+dRleftab+dRleftab+dRleftbb)) : 1.0;

        NiceAssert( scale > 0 );

        scale = ( scale > SCALEMAX ) ? scale : SCALEMAX;

	stepAlpha("&",pivAlphaF(),tmpvf) *= scale;
	stepBeta ("&",pivBetaF (),tmpvf) *= scale;

        // Pretend

        bsize = bNF();

        #ifdef DEBUGOPT
        errstream() << "+----------------------------------\n";
        errstream() << "| \n";
        errstream() << "| Step calculation method 1:\n\n";
        errstream() << "| \n";
        errstream() << "| asize = " << asize << "\n";
        errstream() << "| bsize = " << bsize << "\n";
        errstream() << "| aN = " << aNF() << "\n";
        errstream() << "| bN = " << bNF() << "\n";
        errstream() << "| \n";
        errstream() << "| alpha step = " << stepAlpha(pivAlphaF()) << "\n";
        errstream() << "| beta step = " << stepBeta(pivBetaF()) << "\n";
        errstream() << "| \n";
        errstream() << "+----------------------------------\n";
        #endif
    }

    return res;
}

template <class T, class S>
template <class U>
void optState<T,S>::fact_minverse(Vector<U> &aAlpha, Vector<U> &aBeta, const Vector<U> &bAlpha, const Vector<U> &bBeta, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    (void) Gp;

    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( aAlpha.size() == aN() );
    NiceAssert( aBeta.size()  == bN() );
    NiceAssert( bAlpha.size() == aN() );
    NiceAssert( bBeta.size()  == bN() );
    NiceAssert( keepfact() );

    if ( probContext.fact_pfact(Gn,Gpn) == aNF() )
    {
        probContext.fact_minverse(aAlpha,aBeta,bAlpha,bBeta,Gn,Gpn);
    }

    else
    {
        throw("Unable to disambiguate required inverse for singular case in direct inverse call.");
    }

    return;
}

template <class T, class S>
double optState<T,S>::fact_testFact(Matrix<double> &Gpdest, Matrix<double> &Gndest, Matrix<double> &Gpndest, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( keepfact() );

    return probContext.fact_testFact(Gpdest,Gndest,Gpndest,Gp,Gn,Gpn);
}

template <class T, class S>
double optState<T,S>::fact_testFactInt(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( keepfact() );

    Matrix<double> Gpdest(Gp);
    Matrix<double> Gndest(Gn);
    Matrix<double> Gpndest(Gpn);

    return fact_testFact(Gpdest,Gndest,Gpndest,Gp,Gn,Gpn);
}

template <class T, class S>
void optState<T,S>::fact_fudgeOn (const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( keepfact() );

    return probContext.fact_fudgeOn(Gp,Gn,Gpn,alphagradstate,betagradstate);
}

template <class T, class S>
void optState<T,S>::fact_fudgeOff(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( keepfact() );

    return probContext.fact_fudgeOff(Gp,Gn,Gpn,alphagradstate,betagradstate);
}

extern int istrig;

template <class T, class S>
void optState<T,S>::fixGrad(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( hp.size() == aN() );

    if ( gradFixAlphaInd )
    {
	gradFixAlphaInd = 0;

	int i;

	if ( aN() )
	{
	    for ( i = 0 ; i < aN() ; i++ )
	    {
		if ( gradFixAlpha(i) )
		{
		    gradFixAlpha("&",i) = 0;
		    recalcAlphaGrad(dalphaGrad("&",i),GpGrad,Gpn,gp,hp,i);
		}
	    }
	}
    }

    if ( gradFixBetaInd )
    {
	gradFixBetaInd = 0;

	int i;

	if ( bN() )
	{
	    for ( i = 0 ; i < bN() ; i++ )
	    {
		if ( gradFixBeta(i) )
		{
		    gradFixBeta("&",i) = 0;
		    recalcBetaGrad(dbetaGrad("&",i),Gn,Gpn,gn,i);
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::fixGradhpzero(const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn)
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( gn.size() == bN() );

    if ( gradFixAlphaInd )
    {
	gradFixAlphaInd = 0;

	int i;

	if ( aN() )
	{
	    for ( i = 0 ; i < aN() ; i++ )
	    {
		if ( gradFixAlpha(i) )
		{
		    gradFixAlpha("&",i) = 0;
		    recalcAlphaGradhpzero(dalphaGrad("&",i),GpGrad,Gpn,gp,i);
		}
	    }
	}
    }

    if ( gradFixBetaInd )
    {
	gradFixBetaInd = 0;

	int i;

	if ( bN() )
	{
	    for ( i = 0 ; i < bN() ; i++ )
	    {
		if ( gradFixBeta(i) )
		{
		    gradFixBeta("&",i) = 0;
		    recalcBetaGrad(dbetaGrad("&",i),Gn,Gpn,gn,i);
		}
	    }
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::recalcAlphaGradhpzero(T &res, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, int i) const
{
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( ( i >= 0 ) && ( i < aN() ) );

    res = gp(i);

    T temp;

    retVector<S>      tmpva;
    retVector<S>      tmpvb;
    retVector<T>      tmpvc;
    retVector<double> tmpvd;
    retVector<double> tmpve;

    if ( aNLB() ) { res += sumb(temp,GpGrad(i,pivAlphaLB(),tmpva,tmpvb),dalpha(pivAlphaLB(),tmpvc)); }
    if ( aNF()  ) { res += sumb(temp,GpGrad(i,pivAlphaF (),tmpva,tmpvb),dalpha(pivAlphaF (),tmpvc)); }
    if ( aNUB() ) { res += sumb(temp,GpGrad(i,pivAlphaUB(),tmpva,tmpvb),dalpha(pivAlphaUB(),tmpvc)); }

    res += sumb(temp,Gpn(i,pivBetaF(),tmpvd,tmpve),dbeta(pivBetaF(),tmpvc));

    return;
}

template <class T, class S>
T &optState<T,S>::calcObj(T &res, const Matrix<S> &GpGrad, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &gn, const Vector<T> &hp)
{
    if ( gradFixAlphaInd || gradFixBetaInd )
    {
        fixGrad(GpGrad,Gn,Gpn,gp,gn,hp);
    }

    res = 0;

    if ( aNNZ() )
    {
        int i;
        int j = 0;

        T tempa,gbi,hpi;

        // O = 1/2 alpha'.GpGrad.alpha + gp'.alpha + hp'.|alpha|
        //   = ( 1/2 GpGrad.alpha + gp + hp sgn(alpha) )'.alpha
        //   = ( 1/2 ( alphaGrad - Gpn.beta - gp - hp sgn(alpha) ) + gp + hp sgn(alpha) )'.alpha

        for ( i = 0 ; ( ( i < aN() ) && ( j < aNNZ() ) ) ; i++ )
        {
            if ( alphaState()(i) )
            {
                unAlphaGrad(tempa,i,GpGrad,Gpn,gp,hp);

                hpi  = hp(i);
                hpi *= ( alphaState()(i) > 0 ) ? -1.0 : 1.0;

                twoProductNoConj(gbi,Gpn(i,pivBetaF()),beta()(pivBetaF()));

                tempa -= gbi;
                tempa -= gp(i);
                tempa -= hpi;
                tempa /= 2.0;
                tempa += gp(i);
                tempa += hpi;
                tempa *= alpha()(i);

                res += tempa;
            }
        }
    }

    return res;
}


template <class T, class S>
void optState<T,S>::recalcAlphaGrad(T &res, const Matrix<S> &GpGrad, const Matrix<double> &Gpn, const Vector<T> &gp, const Vector<T> &hp, int i) const
{
//FIXME: if you want to include vector angles for hp then fix here
    NiceAssert( GpGrad.isSquare() );
    NiceAssert( GpGrad.numRows() == Gpn.numRows() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gp.size() == aN() );
    NiceAssert( hp.size() == aN() );
    NiceAssert( ( i >= 0 ) && ( i < aN() ) );

    recalcAlphaGradhpzero(res,GpGrad,Gpn,gp,i);

    if ( alphaState(i) > 0 )
    {
	res += hp(i);
    }

    else if ( alphaState(i) < 0 )
    {
	res -= hp(i);
    }

    return;
}

template <class T, class S>
void optState<T,S>::recalcBetaGrad(T &res, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<T> &gn, int i) const
{
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( gn.size() == bN() );
    NiceAssert( ( i >= 0 ) && ( i < bN() ) );

    int iP;
    T temp;

    res = gn(i);

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retVector<T>      tmpvc;

    res += sumb(temp,Gn(i,pivBetaF(),tmpva,tmpvb),dbeta(pivBetaF(),tmpvc));

    if ( aNLB() )
    {
	for ( iP = 0 ; iP < aNLB() ; iP++ )
	{
	    res += (dalpha(pivAlphaLB()(iP))*Gpn(pivAlphaLB()(iP),i));
	}
    }

    if ( aNF() )
    {
	int iP;

	for ( iP = 0 ; iP < aNF() ; iP++ )
	{
	    res += (dalpha(pivAlphaF()(iP))*Gpn(pivAlphaF()(iP),i));
	}
    }

    if ( aNUB() )
    {
	int iP;

	for ( iP = 0 ; iP < aNUB() ; iP++ )
	{
	    res += (dalpha(pivAlphaUB()(iP))*Gpn(pivAlphaUB()(iP),i));
	}
    }

    return;
}

template <class T, class S>
void optState<T,S>::redimensionalise(void (*redimelm)(T &, int, int), int olddim, int newdim)
{
    int i;

    if ( aN() )
    {
	for ( i = 0 ; i < aN() ; i++ )
	{
            redimelm(dalpha("&",i),olddim,newdim);
            redimelm(dalphaGrad("&",i),olddim,newdim);
	}
    }

    if ( bN() )
    {
	for ( i = 0 ; i < bN() ; i++ )
	{
            redimelm(dbeta("&",i),olddim,newdim);
            redimelm(dbetaGrad("&",i),olddim,newdim);
	}
    }

    return;
}

template <class T, class S> std::ostream &operator<<(std::ostream &output, const optState<T,S> &src )
{
    output << "Alpha:                " << src.dalpha          << "\n";
    output << "Beta:                 " << src.dbeta           << "\n";
    output << "Alpha gradient:       " << src.dalphaGrad      << "\n";
    output << "Beta gradient:        " << src.dbetaGrad       << "\n";
    output << "Gradient error bound: " << src.cumgraderr      << "\n";
    output << "Alpha restriction:    " << src.dalphaRestrict  << "\n";
    output << "Beta restriction:     " << src.dbetaRestrict   << "\n";
    output << "Optimality tolerance: " << src.dopttol         << "\n";
    output << "Alpha gradient state: " << src.alphagradstate  << "\n";
    output << "Beta gradient state:  " << src.betagradstate   << "\n";
    output << "Alpha Gradient State: " << src.gradFixAlphaInd << "\n";
    output << "Beta Gradient State:  " << src.gradFixBetaInd  << "\n";
    output << "Alpha Gradient fine:  " << src.gradFixAlpha    << "\n";
    output << "Beta Gradient fine:   " << src.gradFixBeta     << "\n";
    output << "Context:              " << src.probContext     << "\n";

    return output;
}


template <class T, class S> std::istream &operator>>(std::istream &input,        optState<T,S> &dest)
{
    wait_dummy dummy;

    input >> dummy; input >> dest.dalpha;
    input >> dummy; input >> dest.dbeta;
    input >> dummy; input >> dest.dalphaGrad;
    input >> dummy; input >> dest.dbetaGrad;
    input >> dummy; input >> dest.cumgraderr;
    input >> dummy; input >> dest.dalphaRestrict;
    input >> dummy; input >> dest.dbetaRestrict;
    input >> dummy; input >> dest.dopttol;
    input >> dummy; input >> dest.alphagradstate;
    input >> dummy; input >> dest.betagradstate;
    input >> dummy; input >> dest.gradFixAlphaInd;
    input >> dummy; input >> dest.gradFixBetaInd;
    input >> dummy; input >> dest.gradFixAlpha;
    input >> dummy; input >> dest.gradFixBeta;
    input >> dummy; input >> dest.probContext;

    return input;
}


#endif
