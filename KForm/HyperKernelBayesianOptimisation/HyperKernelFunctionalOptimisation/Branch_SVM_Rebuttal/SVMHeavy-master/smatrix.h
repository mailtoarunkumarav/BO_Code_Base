
//
// Special matrix class
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _smatrix_h
#define _smatrix_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "matrix.h"


// smIdent: m*n identity matrix
// smConst: m*n diagonal matrix with const on the diagonal
// smOnes:  m*n matrix of ones
// smZeros: m*n matrix of zeros
// smVals:  m*n matrix of val
// smDiag:  diag(b)
// smBlockDiag: block diagonal matrix
// smStack: stack matrix
// smOuter: a*b*c'
// smRefl:  dI + a*b*c'
// smRotat: [ sin(a)      b'.cos(a)   ]
//          [ -b.cos(a)   b.b'.sin(a) ]
// smCutM:  m*m cut matrix.  Suppose a'.b = c'.Q.d, where:
//              [  I   0  ]
//          a = [ -1' -1' ].c - row s (indexed from 0)
//              [  0   I  ]
//              [  I   0  ]
//          b = [ -1' -1' ].d - row t (indexed from 0)
//              [  0   I  ]
//          Q is this matrix                 (size (m-1)*(m-1)).
//          If t == -1 then: Q = [ I -1 0  ] (size m*(m-1))
//                               [ 0 -1 I  ]
//          If s == -1 then: Q = [  I   0  ] (size (m-1)*m)
//                               [ -1' -1' ]
//                               [  0   I  ]
//          If s == t == -1: Q = I           (size m*m)
//
// NB: - setting n != m for an identity matrix puts 1s on i == j, 0 elsewhere
//     - if a Vector<double> * is given then this pointer will be used directly
//       in calculations, so care should be taken not to change or delete the
//       vector that is pointed to (unless it is actually desired that the
//       change to the vector be reflected immediately in the matrix, in which
//       case this could be a handy feature).
//     - be careful when using multiple instances of one of these matrices in a
//       single expression.  One variable is used as a return reference for
//       each constructed instance, so if two returns are active then one of
//       them will be overwritten by the other.
//     - all matrices are nominally constant.  You can to a certain extent get
//       around this using the pointer forms by adjusting the pointed to value
//       directly.
//     - Block diagonal uses the extZero trick so that it does not actually
//       have to copy of allocate additional storage.  It is thus compatible
//       with cached matrices.

Matrix<double> *smIdent(int m, int n);
Matrix<double> *smConst(int m, int n, double val, double offdiagval = 0.0);
Matrix<double> *smOnes(int m, int n);
Matrix<double> *smZeros(int m, int n);
Matrix<double> *smVals(int m, int n, double val);
Matrix<double> *smDiag(const Vector<double> &b);
Matrix<double> *smOuter(double a, const Vector<double> &b, const Vector<double> &c);
Matrix<double> *smRefl(double d, double a, const Vector<double> &b, const Vector<double> &c);
Matrix<double> *smRotat(double sina, double cosa, const Vector<double> &b);
Matrix<double> *smCutM(int m, int s, int t);
Matrix<double> *smBlockDiag(const Vector<const Matrix<double> *> &src);
Matrix<double> *smStack(const Matrix<double> *A, const Matrix<double> *B);
Matrix<Matrix<double> > *smStack(const Matrix<Matrix<double> > *A, const Matrix<Matrix<double> > *B);

Matrix<double> *smConst(int m, int n, double *val);
Matrix<double> *smDiag(Vector<double> *b);
Matrix<double> *smOuter(double *a, Vector<double> *b, Vector<double> *c);
Matrix<double> *smRefl(double *d, double *a, Vector<double> *b, Vector<double> *c);
Matrix<double> *smRotat(double *sina, double *cosa, Vector<double> *b);
Matrix<double> *smCutM(int m, int *s, int *t);

#endif
