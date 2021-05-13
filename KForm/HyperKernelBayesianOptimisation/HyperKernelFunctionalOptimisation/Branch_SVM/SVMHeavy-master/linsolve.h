
//
// Linear programming solver - quick and dirty hopdm front-end
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _linsolve_h
#define _linsolve_h

#include <fstream>
#include "vector.h"
#include "matrix.h"
#include "optstate.h"

#define SOLVE_COMMAND "hopdm"
#define PROBFILE      "mps"
#define SOLFILE       "sol"
#define ZERO_PT       1e-6

//
// This is basically a frontend to a linear solver (typically hopdm, but
// by changing the SOLVE_COMMAND you could use pretty much any solver that
// can handle simple MPS format files).  Does not support incremental/
// warm-start at present.
//
// Will solve:
//
// sum_i w_i |alpha_i| + sum_i v_i beta_i + sum_i { c_i if xi_i < 0, d_i if xi_i > 0 } |xi_i|
//
// such that:
//
// [ Gp  Gpn ] [ alpha ] + [ gp ] + | alpha' |' [ hp ] - [ xi ] pc [ 0 ]
// [ Qnp Qn  ] [ beta  ]   [ qn ]   | beta   |  [ 0  ]   [ 0  ] nc [ 0 ]
//
// xi pc 0
//
// alpha,beta constrained as defined by the optimisation state and lb/ub and
// where pc/nc are the relevant constraint:
//
// pc_i is >= if alpha_i >= 0
// pc_i is <= if alpha_i <= 0
// pc_i is == if alpha_i unconstrained
// nc_i is >= if beta_i <= 0
// nc_i is <= if beta_i >= 0
// nc_i is == if beta_i unconstrained
//
// alpharestrictoverride: additional restriction on alpha gradient *in
// addition to* that defined in optstate.  Uses same convention, namely
//    alpharestrictoverride = 0: lb[i] <= alpha[i] <= ub[i]
//    alpharestrictoverride = 1:     0 <= alpha[i] <= ub[i]
//    alpharestrictoverride = 2: lb[i] <= alpha[i] <= 0
//    alpharestrictoverride = 3:     0 <= alpha[i] <= 0
//
// Qconstype = 0: nc >=
//             1: nc ==
//
// No assumptions are made regarding the matrices involved.  The optimisation
// is done by:
//
// - negating alphas/betas that are constrained negative
// - negating betas that are constrained negative
// - splitting unconstrained alphas/betas into positive/negative parts
//   (a = apos-aneg)
// - writing the reconstructed problem in MPS format
// - calling SOLVE_COMMAND
// - reconstructing the result and saving it.
//
// Note that due to the method used killSwitch cannot be implemented.
//
// NB: - lb and ub are assumed to be bignum and are present only because they
//       are required when modifying optState x.
//     - Gn and gn also only present because they are needed when modding x

int solve_linear_program(optState<double,double> &x,
       const Vector<double> &w, const Vector<double> &v,
       const Vector<double> &c, const Vector<double> &d,
       const Matrix<double> &Gp, const Matrix<double> &Gpn,
       const Matrix<double> &Qnp, const Matrix<double> &Qn,
       const Vector<double> &gp, const Vector<double> &hp,
       const Vector<double> &qn,
       const Vector<double> &lb, const Vector<double> &ub,
       const Matrix<double> &Gn, const Vector<double> &gn,
       int alpharestrictoverride, int Qconstype,
       svmvolatile int &killSwitch, int maxitcntint, double maxtraintime);
#endif
