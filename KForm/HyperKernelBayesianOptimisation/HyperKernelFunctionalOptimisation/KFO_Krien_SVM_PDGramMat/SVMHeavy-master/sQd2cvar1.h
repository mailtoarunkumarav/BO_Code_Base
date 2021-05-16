
//
// Sparse quadratic solver - large scale, d2c variant based, warm start
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _sQd2cvar1_h
#define _sQd2cvar1_h

#include "vector.h"
#include "matrix.h"
#include "optstate.h"

//
// Pretty self explanatory: you give is a state and relevant matrices and it
// will solve the coupled optimisation problems:
//
// [ alpha_s ]' [ Gp   Gpn ] [ alpha_s ] + [ alpha_s ]' [ gp_s ] + | alpha_s' |' [ hp_s ]
// [ beta_s  ]  [ Gpn' Gn  ] [ beta_s  ]   [ beta_s  ]  [ gn_s ]   | beta_s   |  [ 0    ]
//
// where s = 0,1,...,n and:
//
// \sum_s \alpha_{si} = 0 for all i - associated Lagrange multipliers \mu_i
// \sum_s \beta = 0                 - associated Lagrange mulitplier \xi
//
// to within precision optsol.  It is assumed that:
//
// - Gp is positive semi-definite hermitian
// - Gn is a 1*1 zero matrix
// - The sign of alpha is fixed (either positive or negative)
// - sigma(i,j) = Gp(i,i) + Gp(j,j) - 2.Gp(i,j)
// - GpnRowTwoSigned = 0
// - fixHigherOrderTerms = NULL
//
// Will return 0 on success or an error code otherwise
//

int solve_quadratic_program_d2cvar1(Vector<optState<double,double> *> &x, Vector<double> &mu, double xi, 
                                    const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
                                    const Vector<Vector<double> > &gpbase, const Vector<Vector<double> > &gnbase, const Vector<Vector<double> > &hp, 
                                    const Vector<Vector<double> > &lbbase, const Vector<Vector<double> > &ubbase, 
                                    int maxitcnt, double maxtraintime);

#endif
