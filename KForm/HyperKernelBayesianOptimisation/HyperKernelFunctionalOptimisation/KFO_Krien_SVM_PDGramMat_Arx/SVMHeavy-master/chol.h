
//
// Modified Cholesky factorisation class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


//FIXME - really should use std::string instead!
#define BADBUFFER 2048


//
// Addendum:
//
// Consider the following situation:
//
// Gp = [ Gp1  Gp2 ]
//      [ Gp2' Gp3 ]
//
// Gpn = [ Gpn1 Gpn3 ]
//       [ Gpn2 Gpn4 ]
//
// Gn = [ Gn1  Gn2 ]
//      [ Gn2' Gn3 ]
//
// where (after pivotting) we are factorising:
//
// [ Gp1   Gpn1 Gp2   Gpn3 ]
// [ Gpn1' Gn1  Gpn2' Gn2  ]
// [ Gp2   Gpn2 Gp3   Gpn4 ]
// [ Gpn3' Gn2' Gpn4' Gn3  ]
//
// where without loss of generality we ignore any requisit
// pivotting on:
//
// [ Gp1   Gpn1 Gp2   Gpn3 ]
// [ Gpn1' Gn1  Gpn2' Gn2  ]
// [ Gp2   Gpn2 ...   ...  ]
// [ Gpn3' Gn2' ...   ...  ]
//
// Our (partial) factorisation is assumed to be:
//
// [ Gp1   Gpn1 Gp2   Gpn3 ]   [ L1            ] [ I         ] [ L1' L2' L3' L4' ]
// [ Gpn1' Gn1  Gpn2' Gn2  ] = [ L2 L5         ] [   -I      ] [     L5' L6' L7' ]
// [ Gp2   Gpn2 ...   ...  ]   [ L3 L6 ...     ] [      I    ] [         ... ... ]
// [ Gpn3' Gn2' ...   ...  ]   [ L4 L7 ... ... ] [        -I ] [             ... ]
//
//                             [ L1.L1'  L1.L2'         L1.L3'         L1.L4'        ]
//                           = [ L2.L1'  L2.L2'-L5.L5'  L2.L3'-L5.L6'  L2.L4'-L6.L7' ]
//                             [ L3.L1'  L3.L2'-L6.L5'  ...            ...           ]
//                             [ L4.L1'  L4.L2'-L7.L6'  ...            ...           ]
//
// We further assume that this factorisation cannot be extended - that is,
// for any row i in:
//
// [ Gp2 Gpn2 Gp3 Gpn4 ]
//
// we have:
//
// l3i'.l3i - l6i'.l6i = Gp3_ii
//
// where:
//
// [ L3 L6 ... ]_i = [ l3i' l6i' ... ]
//
// and likewise for any row j in:
//
// [ Gpn3 Gn2 Gpn4 Gn3 ]
//
// we have:
//
// l4j'.l4j - l7j'.l7j = Gn3_jj
//
// where:
//
// [ L4 L7 ... ]_j = [ l4j' l4j' ... ]
//
// so that no pivot that brings any of these rows to the top of the 
// unfactored portion can result in an extension without introducing a
// zero onto the diagonal of our partial cholesky factorisation and
// thereby making it singular.  Note that this cannot occur unless
// Gpn has at least two columns, so this is not an issue for the "standard"
// SVM formulations.
//
// Now, the code will stop when it encounters this situation on the
// assumption that the there is no way to increase the size of the partial
// factorisation without making it singular.  However *this is not true in
// general*.  For example, assume that Gp2 and Gpn3 have a single row each.
// Then, letting Gp3 = [ gp ], Gpn4 = [ gpn ] and Gn3 = [ gn ], consider:
//
//      [ Gp1   Gpn1   Gp2   Gpn3 ]
//      [ Gpn1' Gn1    Gpn2' Gn2  ]
// inv( [                         ] )
//      [ Gp2   Gpn2   Gp3   Gpn4 ]
//      [ Gpn3' Gn2'   Gpn4' Gn3  ]
//
// Using the matrix inversion lemma the existence of this inverse depends on
// the matrix:
//
// [ Gp3   Gpn4 ] - [ Gp2   Gpn2 ] inv( [ Gp1   Gpn1 ] ) [ Gp2   Gpn3 ]
// [ Gpn4' Gn3  ]   [ Gpn3' Gn2' ]      [ Gpn1' Gn1  ]   [ Gpn2' Gn2  ]
//
//  = [ Gp3   Gpn4 ] - [ Gp2   Gpn2 ] inv( [ L1    ] [ I    ] [ L1' L2' ] ) [ Gp2   Gpn3 ]
//    [ Gpn4' Gn3  ]   [ Gpn3' Gn2' ]      [ L2 L5 ] [   -I ] [     L5' ]   [ Gpn2' Gn2  ]
//
//  = [ Gp3   Gpn4 ] - ( inv( [ L1    ] ) [ Gp2   Gpn3 ] )' [ I    ] ( inv( [ L1    ] ) [ Gp2   Gpn3 ] )
//    [ Gpn4' Gn3  ]          [ L2 L5 ]   [ Gpn2' Gn2  ]    [   -I ]        [ L2 L5 ]   [ Gpn2' Gn2  ]
//
//  = [ Gp3   Gpn4 ] - [ L3' L6' ]' [ I    ] [ L3 L4 ]
//    [ Gpn4' Gn3  ]   [ L4' L7' ]  [   -I ] [ L6 L7 ]
//
//  = [ gp  gpn ] - [ l3i' l6i' ] [  l3i  l4j ]
//    [ gpn gn  ]   [ l4j' l7j' ] [ -l6i -l7j ]
//
//  = [ gp  gpn ] - [ gp                 l3i'.l4j-l6i'.l7j ]
//    [ gpn gn  ]   [ l4j'.l3i-l7j'.l6i  gn                ]
//
//  = [ 0 q ]
//    [ q 0 ]
//
// which is invertible if q != 0.  Hence the complete hessian is in fact
// non-singular in this case but the factorisation does not exist!
//
// The net result is this: if, during optimisation, you find that the factorisation
// is such that both pbadpos and nbadpos are non-zero then the following steps
// must be taken:
//
// 1. Attempt refactorisation - clean-slate it.
// 2. If 1 fails then find an alternative means of calculating the step!
//
// Otherwise you get the situation where the constraints corresponding to parts
// of Gpn that are not in the factorisation may be violated silently, which 
// causes all manner of difficulties!
//

#ifndef _chol_h
#define _chol_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include "vector.h"
#include "matrix.h"
#include "mlcommon.h"

template <class T> class Chol;

// Stream operator overloading

template <class T> std::ostream &operator<<(std::ostream &output, const Chol<T> &src );
template <class T> std::istream &operator>>(std::istream &input,        Chol<T> &dest);

// Swap function

template <class T> void qswap(Chol<T> &a, Chol<T> &b);

// The class itself

template <class T>
class Chol
{
    template <class S> friend std::istream &operator>>(std::istream &input, Chol<S> &dest);
    template <class S> friend void qswap(Chol<S> &a, Chol<S> &b);

public:

    // These deal with a specific problem.  Suppose we have a symmetric square
    // matrix G and want to be able to efficiently do the operations:
    //
    //     a = inv(Gu).b
    // and a = inv(Gu).x (if G is singular)
    //
    // where:
    //
    // G  = [ Gu  Go' ]
    //      [ Go  Gl  ]
    // Go = [ x'  ] (x' is a row vector)
    //      [ Gol ]
    //
    // and Gu is as large as possible without making R singular.
    //
    // This class approaches the problem by assuming that G may be expressed as a
    // modified cholesky factorisation:
    //
    // G = L.D.L'
    //
    // where L is a lower triangular matrix and D is a (pre-defined) diagonal
    // matrix of the form:
    //
    // D = diag(d0,d1,...)
    //
    // where di = +/-1
    //
    // The implementation is deliberately naive.  L is formed starting from the
    // top left corner with no pivotting and expanded as far as possible, so in
    // general:
    //
    // Gu = L.Du.L'
    //
    // The structure of G
    //
    // G is presumed to be some symmetrically pivotted version of:
    //
    // G = [ Gp    Gpn ]
    //     [ Gpn'  Gn  ]
    //
    // corresponding to:
    //
    // D = [ Dp  0  ]
    //     [ 0   Dn ]
    //
    // the pivotting is controlled by D, which is diagonal with elements +-1

    // Constuctors:
    //
    // G:  the matrix to be factorised
    // D:  the factorisatin diagonal
    // zt: zero tolerance level to be used

    svm_explicit Chol(double zt = DEFAULT_ZTOL, int fudgeit = 0);
    svm_explicit Chol(const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D, double zt = DEFAULT_ZTOL, int fudgeit = 0);
                 Chol(const Chol<T> &src);

    ~Chol();

    // Overwrite assignment operator

    Chol<T> &operator=(const Chol<T> &src);

    // Inversion:
    //
    //     Compute a where Gu.a = b, where a has the form:
    //
    //         [ 0    ] z_start
    //     b = [ bz   ] size(b)-z_start-z_end-{1} = size(bx)-z_start-z_end
    //         [ 0    ] z_end
    //
    //     thus enabling certain optimisations to occur in the calculation.
    //     Both z_start and z_end arguments are optional.
    //
    // Optimal inversion (near_invert):
    //
    //     This calculates a, where Gu.a = x
    //
    // Return value: all return dnbad

    template <class S> int minverse(Vector<S> &ap, Vector<S> &an, const Vector<S> &bp, const Vector<S> &bn, int zp_start = 0, int zp_end = 0, int zn_start = 0, int zn_end = 0) const;
    template <class S> int near_invert(Vector<S> &ap, Vector<S> &an) const;
    template <class S> Vector<S> &forwardElim(Vector<S> &y, const Vector<S> &b)   const { return L.forwardElim(y,b);   }
    template <class S> Vector<S> &backwardSubst(Vector<S> &y, const Vector<S> &b) const { return L.backwardSubst(y,b); }

    // Rank-1 updates:
    //
    // Perform a rank-one update on G and hence fix the factorisation.
    //
    // G := G + b.c.b'
    //
    //     [ 0  ] z_start
    // b = [ bx ] size()-z_start-z_end
    //     [ 0  ] z_end
    //
    //     [ G_11 G_21^ G_31^ ] z_start
    // G = [ G_21 X     G_32^ ] size(G)-z_end-z_start
    //     [ G_31 G_32  G_33  ] z_end
    //
    // NOTE: 1. for c<0, dnbad may increase, decrease or stay constant
    //       2. for c>0, dnbad may only decrease or stay constant
    //
    // G is assumed to be the non-factorised G with the rank-one added prior to
    // calling this function.  Likewise for D.
    //
    // Return value: all return dnbad

    int rankone(const Vector<T> &bp, const Vector<T> &bn, const double &c, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D, int zp_start = 0, int zp_end = 0, int zn_start = 0, int zn_end = 0);

    // Diagonal addition
    //
    // diagoffset: G := G + diag(nd)
    //
    // G is assumed to be the non-factorised G with the diagonal update
    // completed prior to calling this function.  Likewise for D.
    //
    // Return value: all return dnbad

    int diagoffset(const Vector<double> &ndp, const Vector<double> &ndn, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D, int zp_start = 0, int zp_end = 0, int zn_start = 0, int zn_end = 0);

    // Diagonal multiplicative update
    //
    // Perform a diagonal multiplicative update on G: that is:
    //
    // G := J.G.J'
    //
    // where J is a diagonal matrix (stored as a vector) that is assumed to
    // be unit: ie. J.J' = I
    //
    // LL' := J.L.L'.J'
    //     := J.L.I.L'.J'
    //     := J.L.J'.J.L'.J'
    //     := (J.L.J').(J.L'.J')
    //     := (J.L.J').(J.L.J')'
    //
    // G is assumed to be the non-factorised G with the diagonal update
    // completed prior to calling this function.  Likewise for D.
    //
    // In this case ip_start, ip_end, in_start and in_end refer to the number
    // of 1s in at, respectively, the start and end of, respectively, the
    // diagonal matrices Jp and Jn.
    //
    // Return value: all return dnbad

    int diagmult(const Vector<T> &JJp, const Vector<T> &JJn, int ip_start = 0, int ip_end = 0, int in_start = 0, int in_end = 0);

    // Scale update
    //
    // G := a.G
    //
    // Assumes a > 0.  G is assumed to be the non-factorised G with the
    // scaling operation completed prior to calling this function.  Likewise
    // for D.
    //
    // NOTE: does not check for zero tolerance changes!
    //
    // Return value: dnbad

    int scale(double a, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D);

    // Matrix manipulations:
    //
    // add: add row/column i to G (and d).
    // remove: remove row/column i from G (and d).
    //
    // G is assumed to be the non-factorised G with the relevant row/col
    // added/removed prior to calling this function.  Likewise for D.
    //
    // Return value: all return dnbad

    int add(int i, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D);
    int remove(int i, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D);

    // Information functions
    //
    // size()    - size of G
    // npos()    - size of Gp
    // nneg()    - size of Gn
    // nbad()    - 0 if non-singular, >= 1 if singular
    // nbadpos() - 0 if non-singular or if all parts of Gp are in non-singular corner of G
    // nbadneg() - 0 if non-singular or if all parts of Gn are in non-singular corner of G
    // fudge()   - return 0 if fudging is off, 1 if fudging is on
    // posind(i) - index in Gp of element G(i,i), or index in Gp of element G(j,j) where j = argmin_(j>i) G(j,j) : d(j) == +1 otherwise (j = size()-1 if undefined)
    // negind(i) - index in Gn of element G(i,i), or index in Gn of element G(j,j) where j = argmin_(j>i) G(j,j) : d(j) == -1 otherwise (j = size()-1 if undefined)

    int size(void)    const { return dsize;       }
    int npos(void)    const { return dnpos;       }
    int nneg(void)    const { return dnneg;       }
    int nbad(void)    const { return dnbad;       }
    int ngood(void)   const { return dsize-dnbad; }
    int nbadpos(void) const { return dnbadpos;    }
    int nbadneg(void) const { return dnbadneg;    }
    int fudge(void)   const { return dfudge;      }
    int posind(int i) const;
    int negind(int i) const;

    // Element access
    //
    // (i,j) - constant reference to element i,j in L
    // (i)   - constant reference to element i in d
    // ()    - constant reference to the zero tolerance
    //
    // det(): determinant of L (good part)

    const T &operator()(int i, int j) const { return L(i,j); }
    const double &operator()(int i)   const { return d(i);   }
    const double &operator()(void)    const { return zt;     }

    const T det(void) const
    {
        double res = 1;

        if ( dsize-dnbad )
        {
            int i;

            for ( i = 0 ; i < dsize-dnbad ; i++ )
            {
                res *= L(i,i);
            }
        }

        return res;
    }

    // Factorisation testing: reconstructs G in dest using L

    void testFact(Matrix<T> &Gpdest, Matrix<T> &Gndest, Matrix<T> &Gpndest, Vector<double> &Dtest) const;

    // Fudge factor
    //
    // If fudging is on the when a singular matrix is detected rather than 
    // use a partial factorisation add an offset to the diagonal and continue.
    // Note that this is not numerically stable and generally not a good idea:
    // use only as a last resort.

    void fudgeOn(const Vector<double> &D);
    void fudgeOff(void);

    // Positive index shuffling

    const Vector<int> &getdposind(void) const { return dposind; }

private:

    // L - the matrix, stored so that lower triangular part is the
    //     factorisation and upper triangular part the transpose of same.
    // d - diagonals in factorisation (+/- 1)
    //
    // dpnset  - dpnset(i) == 0/1 if d(i) == -1/+1
    // dposind - vector of indices for d(i) == +1 part of factorisation
    // dnegind - vector of indices for d(i) == -1 part of factorisation
    //
    // dsize - size of L
    // dnpos - number of elements d(i) == +1
    // dnneg - number of elements d(i) == -1
    // dnbad - size(Gl)
    //
    // zt - zero tolerance.  Positive is taken to mean >= zt, negative
    //      means <= -zt and zero means < zt and > -zt.
    // dfudge - add fudge factors if true

    Matrix<T> L;
    Vector<double> d;

    Vector<int> ddpnset;
    Vector<int> dposind;
    Vector<int> dnegind;

    int dsize;
    int dnpos;
    int dnneg;
    int dnbad;
    int dnbadpos;
    int dnbadneg;

    double zt;
    double ztbackup;
    int dfudge;

    // Internal functions:
    //
    // xfact:    attempt to extend the factorisation as far as possible.
    //
    // xrankone: internal version of the rank-1 update function.  This
    //           works much the same as the previous version, except that
    //           it has an additional argument, namely hold_off.  If this is
    //           0 then normal operation, if 1 - only update the factorised
    //           portion of G
    //
    //           form 1:     [ 0  ] z_start
    //                   a = [ ax ] size(G)-z_start-z_end
    //                       [ 0  ] z_end
    //
    // xsetfact: initialise L and d.

    int xfact(const Vector<double> &D);
    int xrankone(const Vector<T> &a, const double &b, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D, int hold_off, int z_start, int z_end);
    int xsetfact(const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D);

    void xgetG(T &res, int i, int j, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn) const;
    void xsetG(const T &src, int i, int j, Matrix<T> &Gp, Matrix<T> &Gn, Matrix<T> &Gpn) const;

    template <class S> void xgetvect(S &res, int i, const Vector<S> &vp, const Vector<S> &vn) const;
    template <class S> void xsetvect(const S &src, int i, Vector<S> &vp, Vector<S> &vn) const;

    void calc_zstart_zend(int &z_start, int &z_end, int zp_start, int zp_end, int zn_start, int zn_end, int end_back) const;

    // Design decision: want to avoid calling allocators etc too often, so keep data between
    // calls.  This means you can't use the same chol in multiple threads at once

    Vector<T> b_rankone;
    Vector<T> g_add;
    Vector<T> q_remove;
    Vector<T> b_xfact;
    Vector<T> bd_xfact;
    Vector<T> a_xrankone;
    Vector<T> aaz_xrankone;
    Vector<double> a_minverse;
    Vector<double> b_minverse;
    Vector<double> ce_minverse;
    Vector<double> a_near_invert;

    Chol<T> *thisindirect[1];
};

template <class S> void qswap(Chol<S> &a, Chol<S> &b)
{
    qswap(a.L,b.L);
    qswap(a.d,b.d);

    qswap(a.ddpnset,b.ddpnset);
    qswap(a.dposind,b.dposind);
    qswap(a.dnegind,b.dnegind);

    qswap(a.dsize   ,b.dsize   );
    qswap(a.dnpos   ,b.dnpos   );
    qswap(a.dnneg   ,b.dnneg   );
    qswap(a.dnbad   ,b.dnbad   );
    qswap(a.dnbadpos,b.dnbadpos);
    qswap(a.dnbadneg,b.dnbadneg);

    qswap(a.zt      ,b.zt      );
    qswap(a.ztbackup,b.ztbackup);
    qswap(a.dfudge  ,b.dfudge  );

    return;
}


// Conversion from strings

template <class T> Chol<T> &atoMatrix(Chol<T> &dest, const std::string &src);
template <class T> Chol<T> &atoMatrix(Chol<T> &dest, const std::string &src)
{
    std::istringstream srcmod(src);

    srcmod >> dest;

    return dest;
}


// Stream operator overloading

template <class T>
std::ostream &operator<<(std::ostream &output, const Chol<T> &src)
{
    int dsize = src.size();
    int dnbad = src.nbad();

    int i,j;

    if ( src.fudge() )
    {
	output << "Cholesky(zero " << src() << " fudged)[ ";
    }

    else
    {
	output << "Cholesky(zero " << src() << " strict)[ ";
    }

    if ( dsize )
    {
	for ( i = 0 ; i < dsize ; i++ )
	{
	    for ( j = 0 ; j < dsize ; j++ )
	    {
		if ( i == j )
		{
		    if ( i >= dsize-dnbad )
		    {
			output << src(i) << " b " << src(i,j) << "\t";
		    }

		    else
		    {
			output << src(i) << " g " << src(i,j) << "\t";
		    }
		}

		else
		{
		    output << src(i,j) << "\t";
		}
	    }

	    if ( i < dsize-1 )
	    {
		output << ";\n  ";
	    }

	    else
	    {
		output << "  ";
	    }
	}
    }

    output << "]";

    return output;
}


template <class T>
std::istream &operator>>(std::istream &input, Chol<T> &dest)
{
    char buffer[BADBUFFER];
    char tt;

    input >> buffer;

    NiceAssert( !strcmp(buffer,"Cholesky(zero") );

    input >> dest.zt;
    input >> buffer;

    if ( !strcmp(buffer,"fudged)[") )
    {
        dest.dfudge = 1;
        dest.ztbackup = (dest.zt)/100;
    }

    else
    {
        NiceAssert( !strcmp(buffer,"strict)[") );

        dest.dfudge = 0;
        dest.ztbackup = dest.zt;
    }

    dest.dnbad = 0;
    dest.dnbadpos = 0;
    dest.dnbadneg = 0;

    int numRows = 0;
    int numCols = 0;
    int colcnt = 0;


    while ( isspace(input.peek()) )
    {
        input.get(tt);
    }

    if ( input.peek() != ']' )
    {
        while ( input.peek() != ']' )
	{
            if ( input.peek() == ';' )
	    {
                input >> buffer;

		if ( numRows == 0 )
		{
		    numCols = colcnt;
		}

                NiceAssert( colcnt == numCols );
                (void) numCols;

		numRows++;

		colcnt = 0;
	    }

	    else
	    {
		if ( (dest.L).numRows() == numRows )
		{
		    (dest.L).addRow(numRows);
		    (dest.d).add(numRows);
		}

		if ( ( (dest.L).numCols() == colcnt ) && !numRows )
		{
		    (dest.L).addCol(colcnt);
		}

		if ( colcnt == numRows )
		{
		    input >> (dest.d)("&",numRows);
                    input >> buffer;

		    if ( !strcmp(buffer,"b") )
		    {
			(dest.dnbad)++;

			if ( (dest.d)(numRows) > 0 )
			{
			    (dest.dnbadpos)++;
			}

			else
			{
			    (dest.dnbadneg)++;
			}
		    }
		}

		input >> (dest.L)("&",numRows,colcnt);

		colcnt++;
	    }

            while ( isspace(input.peek()) )
            {
                input.get(tt);
            }
	}

        if ( !numRows )
	{
	    numCols = colcnt;
	}

        NiceAssert( colcnt == numCols );

	numRows++;
    }

    input >> buffer;

    NiceAssert( numRows == numCols );

    dest.dsize = numRows;

    (dest.L).resize(dest.dsize,dest.dsize);
    (dest.d).resize(dest.dsize);

    (dest.ddpnset).resize(dest.dsize);
    (dest.dposind).resize(dest.dsize);
    (dest.dnegind).resize(dest.dsize);

    dest.dnpos = 0;
    dest.dnneg = 0;

    if ( dest.dsize )
    {
	int i;

	for ( i = 0 ; i < dest.dsize ; i++ )
	{
	    (dest.dposind)("&",i) = dest.dnpos;
	    (dest.dnegind)("&",i) = dest.dnneg;

	    if ( (dest.d)(i) > 0 )
	    {
		(dest.ddpnset)("&",i) = 1;
                (dest.dnpos)++;
	    }

	    else
	    {
		(dest.ddpnset)("&",i) = 0;
                (dest.dnneg)++;
	    }
	}
    }

    return input;
}





// Constuctors:

template <class T>
Chol<T>::Chol(double xzt, int fudgeit)
{
    *thisindirect = this;

    dsize = 0;
    dnpos = 0;
    dnneg = 0;
    dnbad = 0;
    dnbadpos = 0;
    dnbadneg = 0;

    zt       = fudgeit ? xzt*100 : xzt;
    ztbackup = xzt;
    dfudge   = fudgeit;

    return;
}

template <class T>
Chol<T>::Chol(const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D, double xzt, int fudgeit)
{
    NiceAssert( Gp.numRows() == Gp.numCols() );
    NiceAssert( Gn.numRows() == Gn.numCols() );
    NiceAssert( Gpn.numRows() == Gp.numRows() );
    NiceAssert( Gpn.numCols() == Gn.numCols() );
    NiceAssert( D.size() == Gp.numRows() + Gn.numRows() );

    *thisindirect = this;

    dsize = Gp.numRows() + Gn.numRows();
    dnpos = Gp.numRows();
    dnneg = Gn.numRows();
    dnbad = dsize;
    dnbadpos = dnpos;
    dnbadneg = dnneg;

    zt       = fudgeit ? xzt*100 : xzt;
    ztbackup = xzt;
    dfudge   = fudgeit;

    xsetfact(Gp,Gn,Gpn,D);
    xfact(D);

    return;
}

template <class T>
Chol<T>::Chol(const Chol<T> &src)
{
    *thisindirect = this;

    *this = src;

    return;
}

template <class T>
Chol<T>::~Chol()
{
    return;
}




// Overwrite assignment operator

template <class T>
Chol<T> &Chol<T>::operator=(const Chol<T> &src)
{
    L = src.L;
    d = src.d;

    ddpnset = src.ddpnset;
    dposind = src.dposind;
    dnegind = src.dnegind;

    dsize = src.dsize;
    dnpos = src.dnpos;
    dnneg = src.dnneg;
    dnbad = src.dnbad;
    dnbadpos = src.dnbadpos;
    dnbadneg = src.dnbadneg;

    zt       = src.zt;
    ztbackup = src.ztbackup;
    dfudge   = src.dfudge;

    return (*this);
}




// Inversion:

template <class T>
template <class S> int Chol<T>::minverse(Vector<S> &ap, Vector<S> &an, const Vector<S> &bp, const Vector<S> &bn, int zp_start, int zp_end, int zn_start, int zn_end) const
{
    NiceAssert( zp_start+zp_end <= dnpos-dnbadpos );
    NiceAssert( zn_start+zn_end <= dnneg-dnbadneg );
    NiceAssert( bp.size() == dnpos-dnbadpos );
    NiceAssert( bn.size() == dnneg-dnbadneg );

    int i;

    ap = bp;
    an = bn;
    ap.zero();
    an.zero();

    if ( zp_start+zp_end+zn_start+zn_end < dsize-dnbad )
    {
	Vector<S> a(dsize-dnbad);
	Vector<S> b(dsize-dnbad);

	for ( i = 0 ; i < dsize-dnbad ; i++ )
	{
	    xgetvect(b("&",i),i,bp,bn);
	}

	int z_start = 0;
        int z_end = 0;

        calc_zstart_zend(z_start,z_end,zp_start,zp_end,zn_start,zn_end,dnbad);

	// L.c = b
        // D.e = c;
	// L'.a = e
	//
	// [ La 0  0  ] [ ca ] = [ La.ca                 ]   [ 0  ]
	// [ Lb Ld 0  ] [ cb ] = [ Lb.ca + Ld.cb         ] = [ bb ]
	// [ Lc Le Lf ] [ cc ] = [ Lc.ca + Le.cb + Lf.cc ]   [ 0  ]
	//
	// ca = 0
	// Ld.cb = bb
	// Lf.cc = -Le.cb
	//
	// ea = 0;
        // ebc = dbc.*cbc;
	//
	// [ Lxa' Lxbc'  ] [ aa  ] = [ ea  ] = [ 0   ]
	// [ 0    Lxdef' ] [ abc ]   [ ebc ]   [ ebc ]
	//
	// Lxdef'.abc = ebc
	// Lxa'.aa = -Lxbc'.abc

	Vector<S> ce(dsize-dnbad);

        a.zero();
	ce.zero();

        retVector<S>      tmpva;
        retVector<S>      tmpvb;
        retVector<double> tmpvc;
        retMatrix<T>      tmpma;
        retMatrix<T>      tmpmb;

	L(z_start,1,dsize-dnbad-z_end-1,z_start,1,dsize-dnbad-z_end-1,tmpma).forwardElim(ce("&",z_start,1,dsize-dnbad-z_end-1,tmpva),b(z_start,1,dsize-dnbad-z_end-1,tmpvb));
	(L(dsize-dnbad-z_end,1,dsize-dnbad-1,dsize-dnbad-z_end,1,dsize-dnbad-1,tmpma).forwardElim(ce("&",dsize-dnbad-z_end,1,dsize-dnbad-1,tmpva),L(dsize-dnbad-z_end,1,dsize-dnbad-1,z_start,1,dsize-dnbad-z_end-1,tmpmb)*ce("&",z_start,1,dsize-dnbad-z_end-1,tmpvb))).negate();

        ce("&",z_start,1,dsize-dnbad-1,tmpva) *= d(z_start,1,dsize-dnbad-1,tmpvc);

	L(z_start,1,dsize-dnbad-1,z_start,1,dsize-dnbad-1,tmpma).backwardSubst(a("&",z_start,1,dsize-dnbad-1,tmpva),ce("&",z_start,1,dsize-dnbad-1,tmpvb));
	(L(0,1,z_start-1,0,1,z_start-1,tmpma).backwardSubst(a("&",0,1,z_start-1,tmpva),L(0,1,z_start-1,z_start,1,dsize-dnbad-1,tmpmb)*a(z_start,1,dsize-dnbad-1,tmpvb))).negate();

	for ( i = 0 ; i < dsize-dnbad ; i++ )
	{
	    xsetvect(a(i),i,ap,an);
	}
    }

    return dnbad;
}

template <> template <> inline int Chol<double>::minverse(Vector<double> &ap, Vector<double> &an, const Vector<double> &bp, const Vector<double> &bn, int zp_start, int zp_end, int zn_start, int zn_end) const;
template <> template <> inline int Chol<double>::minverse(Vector<double> &ap, Vector<double> &an, const Vector<double> &bp, const Vector<double> &bn, int zp_start, int zp_end, int zn_start, int zn_end) const
{
    NiceAssert( zp_start+zp_end <= dnpos-dnbadpos );
    NiceAssert( zn_start+zn_end <= dnneg-dnbadneg );
    NiceAssert( bp.size() == dnpos-dnbadpos );
    NiceAssert( bn.size() == dnneg-dnbadneg );

    int i;

    ap = bp;
    an = bn;
    ap.zero();
    an.zero();

    if ( zp_start+zp_end+zn_start+zn_end < dsize-dnbad )
    {
	Vector<double> &a = (**thisindirect).a_minverse; //Vector<double> a(dsize-dnbad);
	Vector<double> &b = (**thisindirect).b_minverse; //Vector<double> b(dsize-dnbad);

        a.resize(dsize-dnbad);
        b.resize(dsize-dnbad);

	for ( i = 0 ; i < dsize-dnbad ; i++ )
	{
	    xgetvect(b("&",i),i,bp,bn);
	}

	int z_start = 0;
        int z_end = 0;

        calc_zstart_zend(z_start,z_end,zp_start,zp_end,zn_start,zn_end,dnbad);

	// L.c = b
        // D.e = c;
	// L'.a = e
	//
	// [ La 0  0  ] [ ca ] = [ La.ca                 ]   [ 0  ]
	// [ Lb Ld 0  ] [ cb ] = [ Lb.ca + Ld.cb         ] = [ bb ]
	// [ Lc Le Lf ] [ cc ] = [ Lc.ca + Le.cb + Lf.cc ]   [ 0  ]
	//
	// ca = 0
	// Ld.cb = bb
	// Lf.cc = -Le.cb
	//
	// ea = 0;
        // ebc = dbc.*cbc;
	//
	// [ Lxa' Lxbc'  ] [ aa  ] = [ ea  ] = [ 0   ]
	// [ 0    Lxdef' ] [ abc ]   [ ebc ]   [ ebc ]
	//
	// Lxdef'.abc = ebc
	// Lxa'.aa = -Lxbc'.abc

	Vector<double> &ce = (**thisindirect).ce_minverse; //Vector<double> ce(dsize-dnbad);

        ce.resize(dsize-dnbad);

        a.zero();
	ce.zero();

        retVector<double> tmpva;
        retVector<double> tmpvb;
        retMatrix<double> tmpma;
        retMatrix<double> tmpmb;

	L(z_start,1,dsize-dnbad-z_end-1,z_start,1,dsize-dnbad-z_end-1,tmpma).forwardElim(ce("&",z_start,1,dsize-dnbad-z_end-1,tmpva),b(z_start,1,dsize-dnbad-z_end-1,tmpvb));
	(L(dsize-dnbad-z_end,1,dsize-dnbad-1,dsize-dnbad-z_end,1,dsize-dnbad-1,tmpma).forwardElim(ce("&",dsize-dnbad-z_end,1,dsize-dnbad-1,tmpva),L(dsize-dnbad-z_end,1,dsize-dnbad-1,z_start,1,dsize-dnbad-z_end-1,tmpmb)*ce("&",z_start,1,dsize-dnbad-z_end-1,tmpvb))).negate();

        ce("&",z_start,1,dsize-dnbad-1,tmpva) *= d(z_start,1,dsize-dnbad-1,tmpvb);

	L(z_start,1,dsize-dnbad-1,z_start,1,dsize-dnbad-1,tmpma).backwardSubst(a("&",z_start,1,dsize-dnbad-1,tmpva),ce("&",z_start,1,dsize-dnbad-1,tmpvb));
	(L(0,1,z_start-1,0,1,z_start-1,tmpma).backwardSubst(a("&",0,1,z_start-1,tmpva),L(0,1,z_start-1,z_start,1,dsize-dnbad-1,tmpmb)*a(z_start,1,dsize-dnbad-1,tmpvb))).negate();

	for ( i = 0 ; i < dsize-dnbad ; i++ )
	{
	    xsetvect(a(i),i,ap,an);
	}
    }

    return dnbad;
}

template <class T>
template <class S> int Chol<T>::near_invert(Vector<S> &ap, Vector<S> &an) const
{
    NiceAssert( dnbad );

    // [ Lu 0 ... ] [ Du 0 0 ] [ Lu' m ... ]   [ Gu x ... ]
    // [ m' . ... ] [ 0  : : ] [ 0   : ... ] = [ x' : ... ]
    // [ :  . ... ] [ 0  : : ] [ ... : ... ]   [ :  : ... ]
    //
    // so: Lu.Du.m = x
    //
    // Gu.a = x
    // => Lu.Du.Lu'.a = x
    // => Lu.Du.Lu'.a = Lu.Du.m
    // => Lu'.a = m

    ap.resize(dnpos-dnbadpos);
    an.resize(dnneg-dnbadneg);

    if ( dsize-dnbad )
    {
	Vector<S> a(dsize-dnbad);

	if ( dsize-dnbad )
	{
	    int i;

	    for ( i = 0 ; i < dsize-dnbad ; i++ )
	    {
                a("&",i) = conj(L(i,dsize-dnbad));
	    }
	}

	L(0,1,dsize-dnbad-1,0,1,dsize-dnbad-1).backwardSubst(a,a);

	int i;

	for ( i = 0 ; i < dsize-dnbad ; i++ )
	{
	    xsetvect(a(i),i,ap,an);
	}
    }

    return dnbad;
}

template <> template <> inline int Chol<double>::near_invert(Vector<double> &ap, Vector<double> &an) const;
template <> template <> inline int Chol<double>::near_invert(Vector<double> &ap, Vector<double> &an) const
{
    NiceAssert( dnbad );

    // [ Lu 0 ... ] [ Du 0 0 ] [ Lu' m ... ]   [ Gu x ... ]
    // [ m' . ... ] [ 0  : : ] [ 0   : ... ] = [ x' : ... ]
    // [ :  . ... ] [ 0  : : ] [ ... : ... ]   [ :  : ... ]
    //
    // so: Lu.Du.m = x
    //
    // Gu.a = x
    // => Lu.Du.Lu'.a = x
    // => Lu.Du.Lu'.a = Lu.Du.m
    // => Lu'.a = m

    ap.resize(dnpos-dnbadpos);
    an.resize(dnneg-dnbadneg);

    if ( dsize-dnbad )
    {
	Vector<double> &a = (**thisindirect).a_near_invert; //Vector<double> a(dsize-dnbad);

        a.resize(dsize-dnbad);

	if ( dsize-dnbad )
	{
	    int i;

	    for ( i = 0 ; i < dsize-dnbad ; i++ )
	    {
                a("&",i) = conj(L(i,dsize-dnbad));
	    }
	}

        retMatrix<double> tmpma;

	L(0,1,dsize-dnbad-1,0,1,dsize-dnbad-1,tmpma).backwardSubst(a,a);

	int i;

	for ( i = 0 ; i < dsize-dnbad ; i++ )
	{
	    xsetvect(a(i),i,ap,an);
	}
    }

    return dnbad;
}




// Rank-1 updates:

template <class T>
int Chol<T>::rankone(const Vector<T> &bp, const Vector<T> &bn, const double &c, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D, int zp_start, int zp_end, int zn_start, int zn_end)
{
    NiceAssert( Gp.numRows() == Gp.numCols() );
    NiceAssert( Gn.numRows() == Gn.numCols() );
    NiceAssert( Gpn.numRows() == Gp.numRows() );
    NiceAssert( Gpn.numCols() == Gn.numCols() );
    NiceAssert( Gp.numRows() == dnpos );
    NiceAssert( Gn.numRows() == dnneg );
    NiceAssert( D.size() == dsize );
    NiceAssert( zp_start+zp_end <= dnpos );
    NiceAssert( zn_start+zn_end <= dnneg );
    NiceAssert( bp.size() == dnpos );
    NiceAssert( bn.size() == dnneg );

    if ( ( c != 0.0 ) && ( zp_start+zp_end+zn_start+zn_end < dsize ) )
    {
        Vector<T> &b = b_rankone; //Vector<T> b(dsize);
        b.resize(dsize);

	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
	    xgetvect(b("&",i),i,bp,bn);
	}

	int z_start = 0;
        int z_end = 0;

        calc_zstart_zend(z_start,z_end,zp_start,zp_end,zn_start,zn_end,0);

	xrankone(b,c,Gp,Gn,Gpn,D,0,z_start,z_end);

        return xfact(D);
    }

    return dnbad;
}

template <class T>
int Chol<T>::diagoffset(const Vector<double> &ndp, const Vector<double> &ndn, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D, int zp_start, int zp_end, int zn_start, int zn_end)
{
    NiceAssert( Gp.numRows() == Gp.numCols() );
    NiceAssert( Gn.numRows() == Gn.numCols() );
    NiceAssert( Gpn.numRows() == Gp.numRows() );
    NiceAssert( Gpn.numCols() == Gn.numCols() );
    NiceAssert( Gp.numRows() == dnpos );
    NiceAssert( Gn.numRows() == dnneg );
    NiceAssert( D.size() == dsize );
    NiceAssert( zp_start+zp_end <= dnpos );
    NiceAssert( zn_start+zn_end <= dnneg );
    NiceAssert( ndp.size() == dnpos );
    NiceAssert( ndn.size() == dnneg );

    if ( zp_start+zp_end+zn_start+zn_end < dsize )
    {
	int ii,i;

	Vector<T> bp(ndp.size());
	Vector<T> bn(ndn.size());

	bp.zero();
	bn.zero();

	// We need to be careful here.  If we do rank-ones from the
	// top left and dnbad changes then the updated contents of
	// the diagonal will be incorporated *before* the relevant
	// rank-one update is completed, thus potentially added the
	// offset twice.  To get around this we need to work from
        // Alistair Shilton 2014 (c) wrote this code
	// the bottom right, as changes can then only propagate over
	// the part of the matrix which has *already* been updated,
        // preventing doubling-up.

	if ( ( zp_start+zp_end+zn_start+zn_end < dsize ) )
	{
            retVector<T> tmpva;

            for ( ii = dsize-1 ; ii >= 0 ; ii-- )
	    {
		if ( ddpnset(ii) )
		{
		    i = dposind(ii);

		    if ( ( i >= zp_start ) && ( i <= dnpos-zp_end-1 ) )
		    {
			bp("&",i) = +1.0;
			rankone(bp,bn,ndp(i),Gp,Gn,Gpn,D,i,dnpos-i-1,0,dnneg);
			bp("&",i,1,i,tmpva).zero();
		    }
		}

		else
		{
                    i = dnegind(ii);

		    if ( ( i >= zn_start ) && ( i <= dnneg-zn_end-1 ) )
		    {
			bn("&",i) = +1.0;
			rankone(bp,bn,ndn(i),Gp,Gn,Gpn,D,0,dnpos,i,dnneg-i-1);
			bn("&",i,1,i,tmpva).zero();
		    }
		}
	    }
	}
    }

    return dnbad;
}




// Diagonal multiplicative update

template <class T>
int Chol<T>::diagmult(const Vector<T> &Jp, const Vector<T> &Jn, int zp_start, int zp_end, int zn_start, int zn_end)
{
    NiceAssert( Jp.size() == dnpos );
    NiceAssert( Jn.size() == dnneg );

    if ( dsize && ( zp_start+zp_end+zn_start+zn_end < dsize ) )
    {
	Vector<T> J(dsize);
	int i,j;

	for ( i = 0 ; i < dsize ; i++ )
	{
	    xgetvect(J("&",i),i,Jp,Jn);
	}

	int z_start = 0;
        int z_end = 0;

        calc_zstart_zend(z_start,z_end,zp_start,zp_end,zn_start,zn_end,0);

	for ( i = z_start ; i < dsize-z_end ; i++ )
	{
	    for ( j = z_start ; j < dsize-z_end ; j++ )
	    {
		L("&",i,j) = (J(i)*L(i,j))*conj(J(j));
	    }
	}

	if ( z_start )
	{
	    for ( i = z_start ; i < dsize-z_end ; i++ )
	    {
		for ( j = 0 ; j < z_start ; j++ )
		{
		    L("&",i,j) = J(i)*L(i,j);
		    L("&",j,i) = L(j,i)*conj(J(i));
		}
	    }
	}

	if ( z_end )
	{
	    for ( i = z_start ; i < dsize-z_end ; i++ )
	    {
		for ( j = dsize-z_end ; j < dsize ; j++ )
		{
		    L("&",i,j) = J(i)*L(i,j);
		    L("&",j,i) = L(j,i)*conj(J(i));
		}
	    }
	}
    }

    return dnbad;
}



// Scale update

template <class T>
int Chol<T>::scale(double a, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D)
{
    NiceAssert( Gp.numRows() == Gp.numCols() );
    NiceAssert( Gn.numRows() == Gn.numCols() );
    NiceAssert( Gpn.numRows() == Gp.numRows() );
    NiceAssert( Gpn.numCols() == Gn.numCols() );
    NiceAssert( D.size() == dsize );
    NiceAssert( a > 0 );

    if ( dsize )
    {
        double aSqrt = sqrt(a);
	int i,j,k;

	for ( i = 0 ; i < dsize ; i++ )
	{
	    for ( j = 0 ; j < dsize ; j++ )
	    {
		if ( ( i < dsize-dnbad ) || ( j < dsize-dnbad ) )
		{
		    L("&",i,j) *= aSqrt;
		}

		else
		{
		    L("&",i,j) *= a;
		}
	    }

            if ( ( L(i,i) <= zt ) && ( i < dsize-dnbad ) )
	    {
                if ( !dfudge )
                {
                    int dnbadnew = dnbad;

                    for ( j = i ; j < dsize-dnbad ; j++ )
                    {
                        for ( k = i ; k < dsize-dnbad ; k++ )
                        {
                            xgetG(L("&",j,k),j,k,Gp,Gn,Gpn);
                        }

                        if ( D(j) > 0 )
                        {
                            dnbadnew++;
                            dnbadpos++;
                        }

                        else
                        {
                            dnbadnew++;
                            dnbadneg++;
                        }

                        if ( dnbad )
                        {
                            for ( k = dsize-dnbad ; k < dsize ; k++ )
                            {
                                xgetG(L("&",j,k),j,k,Gp,Gn,Gpn);
                                xgetG(L("&",k,j),k,j,Gp,Gn,Gpn);
                            }
                        }
                    }

                    dnbad = dnbadnew;
                }

                else
                {
                    L("&",i,i) = zt;
                }
	    }
	}
    }

    if ( dnbad )
    {
        xfact(D);
    }

    return dnbad;
}




// Matrix manipulations:


template <class T>
int Chol<T>::add(int ix, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D)
{
    NiceAssert( Gp.numRows() == Gp.numCols() );
    NiceAssert( Gn.numRows() == Gn.numCols() );
    NiceAssert( Gpn.numRows() == Gp.numRows() );
    NiceAssert( Gpn.numCols() == Gn.numCols() );
    NiceAssert( D.size() == dsize+1 );
    NiceAssert( ix >= 0 );
    NiceAssert( ix < dsize+1 );

    int i,j,k;

    int pospos = 0;
    int negpos = 0;

    if ( ix == dsize )
    {
	if ( ix )
	{
	    pospos = dposind(ix-1);
	    negpos = dnegind(ix-1);

	    if ( ddpnset(ix-1) )
	    {
                pospos++;
	    }

	    else
	    {
		negpos++;
	    }
	}
    }

    else
    {
	pospos = dposind(ix);
	negpos = dnegind(ix);
    }

    ddpnset.add(ix);
    dposind.add(ix);
    dnegind.add(ix);

    dposind("&",ix) = pospos;
    dnegind("&",ix) = negpos;

    if ( D(ix) > 0 )
    {
	ddpnset("&",ix) = 1;

	if ( ix >= dsize-dnbad )
	{
	    dnbad++;
	    dnbadpos++;
	}

        dnpos++;
	dsize++;

	if ( ix+1 < dsize )
	{
	    for ( i = ix+1 ; i < dsize ; i++ )
	    {
		(dposind("&",i))++;
	    }
	}
    }

    else
    {
	ddpnset("&",ix) = 0;

	if ( ix >= dsize-dnbad )
	{
	    dnbad++;
	    dnbadneg++;
	}

        dnneg++;
	dsize++;

	if ( ix+1 < dsize )
	{
	    for ( i = ix+1 ; i < dsize ; i++ )
	    {
		(dnegind("&",i))++;
	    }
	}
    }

    NiceAssert( Gp.numRows() == dnpos );
    NiceAssert( Gn.numRows() == dnneg );

    Vector<T> &g = g_add; //Vector<T> g(dsize);
    g.resize(dsize);

    for ( i = 0 ; i < dsize ; i++ )
    {
	xgetG(g("&",i),i,ix,Gp,Gn,Gpn);
    }

    d.add(ix);
    d("&",ix) = D(ix);

    L.addRowCol(ix);

    if ( ix >= dsize-dnbad )
    {
	// [ Lu 0 ... ] [ Du 0 0 ] [ Lu' b ... ]   [ Gu x ... ]
	// [ b' . ... ] [ 0  : : ] [ 0   : ... ] = [ x' : ... ]
	// [ :  . ... ] [ 0  : : ] [ ... : ... ]   [ :  : ... ]
	//
	// so: Lu.Du.b = x
	//
	// Lu.mu = x
	// Du.bu = mu

        retVector<T> tmpva; // only use this in defining gpartof!!!
        retVector<T> tmpvb;
        retMatrix<T> tmpma;

	if ( dsize-dnbad )
	{
	    int zer = 0;

            Vector<T> &gpartof = g("&",zer,1,dsize-dnbad-1,tmpva);

	    L(zeroint(),1,dsize-dnbad-1,zeroint(),1,dsize-dnbad-1,tmpma).forwardElim(gpartof,gpartof);

	    for ( i = 0 ; i < dsize-dnbad ; i++ )
	    {
		g("&",i) *= d(i);
	    }
	}

        L("&",0,1,L.numRows()-1,ix,tmpma,"&") = g;
//	L.setCol(ix,g);
	//g.applyon(conj);
	g.conj();
	L("&",ix,0,1,dsize-1,tmpvb) = g;
    }

    else
    {
	// Current:
        //
	// [ Ga  Gc' ] = [ Ma  0  ] [ Da  0  ] [ Ma' Mc' ]
	// [ Gc  Gf  ]   [ Mc  Mf ] [ 0   Dc ] [ 0   Mf' ]
        //
	//             = [ Ma  0  ] [ Da.Ma'  Da.Mc' ]
	//               [ Mc  Mf ] [ 0       Dc.Mf' ]
        //
	//             = [ Ma.Da.Ma'   Ma.Da.Mc'             ]
	//               [ Mc.Da.Ma'   Mc.Da.Mc' + Mf.Dc.Mf' ]
	//
	// Target:
        //
	// [ Ga  gb Gc' ]   [ La  0  0  ] [ Da 0  0  ] [ La' lb Lc' ]
	// [ gb' gd ge' ] = [ lb' ld 0' ] [ 0  db 0  ] [ 0'  ld le' ]
	// [ Gc  ge Gf  ]   [ Lc  le Lf ] [ 0  0  Dc ] [ 0   0  Lf' ]
        //
	//                  [ La   0   0  ] [ Da.La'  Da.lb  Da.Lc' ]
	//                = [ lb'  ld  0' ] [ 0'      db.ld  db.le' ]
	//                  [ Lc   le  Lf ] [ 0       0      Dc.Lf' ]
        //
	//                  [ La.Da.La'   La.Da.lb              La.Da.Lc'                         ]
	//                = [ lb'.Da.La'  lb'.Da.lb + ld.db.ld  lb'.Da.Lc' + ld.db.le'            ]
	//                  [ Lc.Da.La'   Lc.Da.lb + le.db.ld   Lc.Da.Lc' + le.db.le' + Lf.Dc.Lf' ]
	//
	// So: (La): La = Ma
	//     (lb): Da.lb = inv(La).gb
	//     (Lc): Lc = Mc
	//     (ld): ld = sqrt( db.( gd - (Da.lb)'.Da.(Da.lb) ) )    (may need to truncate factorisation at this step)
	//     (le): le = ( ge - Lc.(Da.lb) ) / (db.ld)              (and continue this to the end of g = [ gb' gd ge' ]')
	//     (Lf): Lf.Dc.Lf' = Mf.Dc.Mf + le.(-db).le'             (let xrankone take care of this)

	// Calculate lb:
        //
	//     (lb): Da.lb = inv(La).gb
	//
	// Trick: for now, it's actually more convenient to keep Da.lb

        retVector<T> tmpva;
        retVector<T> tmpvb;
        retMatrix<T> tmpma;

	if ( ix )
	{
	    int zer = 0;

            retVector<T> tmpvc; // only use for gpartof

            Vector<T> &gpartof = g("&",zer,1,ix-1,tmpvc);

	    L(zeroint(),1,ix-1,zeroint(),1,ix-1,tmpma).forwardElim(gpartof,gpartof);
	}

	// complete ld calculation and truncate factorisation is required:
        //
	//     (ld): ld = sqrt( db.( gd - (Da.lb)'.Da.(Da.lb) ) )

        double ld = real(g(ix));

	if ( ix )
	{
	    for ( i = 0 ; i < ix ; i++ )
	    {
                ld -= (d(i)*norm2(g(i)));
	    }
	}

        ld *= d(ix);

	if ( ld <= zt*zt )
	{
	    if ( dfudge )
	    {
		ld = zt*zt;
	    }

	    else
	    {
                int dnbadnew = dnbad;

		if ( ix )
		{
		    // Finish calculate of lb: lb := Da.lb

		    for ( i = 0 ; i < ix ; i++ )
		    {
			g("&",i) *= d(i);
		    }

                    // Write lb to L matrix

		    int zer = 0;

		    L("&",0,1,ix-1,ix,tmpma,"&") = g(zer,1,ix-1,tmpva);
//		    L("&",0,1,ix-1,0,1,ix,tmpma).setCol(ix,g(zer,1,ix-1,tmpva));
                    //(g("&",0,1,ix-1,tmpva)).applyon(conj);
                    (g("&",0,1,ix-1,tmpva)).conj();
		    L("&",ix,0,1,ix-1,tmpva) = g(zer,1,ix-1,tmpvb);
		}

		for ( j = ix ; j < dsize-dnbad ; j++ )
		{
		    for ( k = ix ; k < dsize-dnbad ; k++ )
		    {
			xgetG(L("&",j,k),j,k,Gp,Gn,Gpn);
		    }

		    if ( D(j) > 0 )
		    {
			dnbadnew++;
			dnbadpos++;
		    }

		    else
		    {
			dnbadnew++;
			dnbadneg++;
		    }
		}

		if ( dnbad )
		{
		    for ( j = ix ; j < dsize-dnbad ; j++ )
		    {
			for ( k = dsize-dnbad ; k < dsize ; k++ )
			{
			    xgetG(L("&",j,k),j,k,Gp,Gn,Gpn);
			    xgetG(L("&",k,j),k,j,Gp,Gn,Gpn);
			}
		    }
		}

		dnbad = dnbadnew;

		goto keep_going;
	    }
	}

	ld = sqrt(ld);
	g("&",ix) = ld;

        NiceAssert( !testisvnan(ld) );

	// Calculate le:
        //
	//     (le): le = ( ge - Lc.(Da.lb) ) / (db.ld) (to end of l)

	if ( ix )
	{
	    for ( i = ix+1 ; i < dsize ; i++ )
	    {
		for ( j = 0 ; j < ix ; j++ )
		{
                    g("&",i) -= (L(i,j)*g(j));
		}
	    }
	}

	for ( i = ix+1 ; i < dsize ; i++ )
	{
	    g("&",i) *= (d(ix)/ld);
	}

	// Finish calculate of lb: lb := Da.lb

	if ( ix )
	{
	    for ( i = 0 ; i < ix ; i++ )
	    {
		g("&",i) *= d(i);
	    }
	}

	// Set row/column lb,ld,le in L:

        L("&",0,1,L.numRows()-1,ix,tmpma,"&") = g;
//	L.setCol(ix,g);
        //g.applyon(conj);
        g.conj();
	L("&",ix,tmpva) = g;
        //g.applyon(conj);
        g.conj();

	// Do rank-one update on Lf:
        //
	//     (Lf): Lf.Dc.Lf' = Mf.Dc.Mf + le.(-db).le'             (let xrankone take care of this)

        g("&",0,1,ix,tmpva).zero();

	xrankone(g,-d(ix),Gp,Gn,Gpn,D,1,ix+1,0);
    }

keep_going:

    return xfact(D);
}

template <class T>
int Chol<T>::remove(int ix, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D)
{
    NiceAssert( Gp.numRows() == Gp.numCols() );
    NiceAssert( Gn.numRows() == Gn.numCols() );
    NiceAssert( Gpn.numRows() == Gp.numRows() );
    NiceAssert( Gpn.numCols() == Gn.numCols() );
    NiceAssert( D.size() == dsize-1 );
    NiceAssert( ix >= 0 );
    NiceAssert( ix < dsize );

    int i;

    if ( ddpnset(ix) )
    {
	if ( ix+1 < dsize )
	{
	    for ( i = ix+1 ; i < dsize ; i++ )
	    {
		(dposind("&",i))--;
	    }
	}

        dnpos--;
    }

    else
    {
	if ( ix+1 < dsize )
	{
	    for ( i = ix+1 ; i < dsize ; i++ )
	    {
		(dnegind("&",i))--;
	    }
	}

        dnneg--;
    }

    ddpnset.remove(ix);
    dposind.remove(ix);
    dnegind.remove(ix);

    NiceAssert( Gp.numRows() == dnpos );
    NiceAssert( Gn.numRows() == dnneg );

    if ( ix > dsize-dnbad-1 )
    {
	if ( d(ix) > 0 )
	{
	    dnbad--;
	    dnbadpos--;
	    dsize--;
	}

	else
	{
	    dnbad--;
	    dnbadneg--;
	    dsize--;
	}

	L.removeRowCol(ix);
        d.remove(ix);
    }

    else if ( ix == dsize-dnbad-1 )
    {
	L.removeRowCol(ix);
        d.remove(ix);

	dsize--;
    }

    else
    {
	// Current:
        //
	// [ Ga  gb Gc' ]   [ La  0  0  ] [ Da 0  0  ] [ La' lb Lc' ]
	// [ gb' gd ge' ] = [ lb' ld 0' ] [ 0  Db 0  ] [ 0'  ld le' ]
	// [ Gc  ge Gf  ]   [ Lc  le Lf ] [ 0  0  Dc ] [ 0   0  Lf' ]
        //
	//                  [ La   0   0  ] [ Da.La'  Da.lb  Da.Lc' ]
	//                = [ lb'  ld  0' ] [ 0'      db.ld  db.le' ]
	//                  [ Lc   le  Lf ] [ 0       0      Dc.Lf' ]
        //
	//                  [ La   0   0  ] [ La.Da.La'   La.Da.lb              La.Da.Lc'                         ]
	//                = [ lb'  ld  0' ] [ lb'.Da.La'  lb'.Da.lb + ld.db.ld  lb'.Da.Lc' + ld.db.le'            ]
	//                  [ Lc   le  Lf ] [ Lc.Da.La'   Lc.Da.lb + le.db.ld   Lc.Da.Lc' + le.db.le' + Lf.Dc.Lf' ]
	//
	// Target:
        //
	// [ Ga  Gc' ] = [ Ma  0  ] [ Da  0  ] [ Ma' Mc' ]
	// [ Gc  Gf  ]   [ Mc  Mf ] [ 0   Dc ] [ 0   Mf' ]
        //
	//             = [ Ma  0  ] [ Da.Ma'  Da.Mc' ]
	//               [ Mc  Mf ] [ 0       Dc.Mf' ]
        //
	//             = [ Ma  0  ] [ Ma.Da.Ma'   Ma.Da.Mc'             ]
	//               [ Mc  Mf ] [ Mc.Da.Ma'   Mc.Da.Mc' + Mf.Dc.Mf' ]
	//
	// So: Ma = La
	//     Mc = Lc
	//     Mf.Dc.Mf = Lf.Dc.Lf' + le.db.le'
	//
	// That is: we need to do a rank-one upate on the factorised part that
	//          comes after the removed row/column.

	Vector<T> &q = q_remove; //Vector<T> q(dsize);
        q.resize(dsize);
        double db;

        retVector<T> tmpva;
        retMatrix<T> tmpma;

	//L.getCol(q,ix);
        q = L(zeroint(),1,L.numRows()-1,ix,tmpma,"&");
        db = d(ix);
	q("&",0,1,ix,tmpva).zero();

	L.removeRowCol(ix);
	q.remove(ix);
	d.remove(ix);

        dsize--;

	xrankone(q,db,Gp,Gn,Gpn,D,1,ix,0);
    }

    return xfact(D);
}



// Fudge factor

template <class T>
void Chol<T>::fudgeOn(const Vector<double> &D)
{
    NiceAssert( D.size() == dsize );

    if ( !dfudge )
    {
        ztbackup = zt;
        zt *= 100;
    }

    dfudge = 1;

    xfact(D);

    return;
}

template <class T>
void Chol<T>::fudgeOff(void)
{
    zt = ztbackup;

    dfudge = 0;

    return;
}



// Information functions

template <class T>
int Chol<T>::posind(int i) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= dsize );

    if ( i == dsize )
    {
        return dnpos;
    }

    return dposind(i);
}

template <class T>
int Chol<T>::negind(int i) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= dsize );

    if ( i == dsize )
    {
        return dnneg;
    }

    return dnegind(i);
}


template <class T>
void Chol<T>::testFact(Matrix<T> &Gpdest, Matrix<T> &Gndest, Matrix<T> &Gpndest, Vector<double> &Ddest) const
{
    int i,j,k;
    T g;
    T res;

    //Gpdest.resize(dnpos,dnpos);
    //Gndest.resize(dnneg,dnneg);
    //Gpndest.resize(dnpos,dnneg);

    Ddest = d;

    // L = [ Lu  Lo' ]
    //     [ Lo  Gl  ]
    //
    // G = [ Gu  Go' ] = [ Lu.Du.Lu' Lu.Du.Lo' ] dsize-dnbad
    //     [ Go  Gl  ]   [ Lo.Du.Lu' Gl        ] dnbad

    if ( dsize-dnbad )
    {
	for ( i = 0 ; i < dsize ; i++ )
	{
	    for ( j = 0 ; ( ( j <= i ) && ( j < dsize-dnbad ) ) ; j++ )
	    {
                setzero(res);

		for ( k = 0 ; ( ( k <= i ) && ( k <= j ) ) ; k++ )
		{
		    g = L(i,k);
		    g *= d(k);
		    g *= L(k,j);

                    res += g;
                }            

                xsetG(res,i,j,Gpdest,Gndest,Gpndest);
                xsetG(conj(res),j,i,Gpdest,Gndest,Gpndest);
	    }
	}
    }

    if ( dnbad )
    {
	for ( i = dsize-dnbad ; i < dsize ; i++ )
	{
	    for ( j = dsize-dnbad ; j < dsize ; j++ )
	    {
                xsetG(L(i,j),i,j,Gpdest,Gndest,Gpndest);
	    }
	}
    }

    return;
}




// Internal functions:

template <class T>
int Chol<T>::xfact(const Vector<double> &D)
{
    int i,j;

    if ( dnbad == 0 )
    {
        return 0;
    }

    // [ Lu  0   0 ] [ Du     0 ] [ Lu' lm  Lb' ]   [ Lu  0   0 ] [ Du.Lu' Du.lm  Du.Lb' ]
    // [ lm' lf  0 ] [ 0  df  0 ] [ 0   lf  lp' ] = [ lm' lf  0 ] [ 0      df.lf  df.lp' ]
    // [ Lb  lp    ] [ 0  0     ] [ 0   0       ]   [ Lb  lp    ] [ 0      0             ]
    //
    //                                              [ Lu.Du.Lu'   Lu.Du.lm               Lu.Du.Lb'               ]
    //                                            = [ lm'.Du.Lu'  lm'.Du.lm + lf.df.lf   lm'.Du.Lb' + lf.df.lp'  ]
    //                                              [ Lb.Du.Lu'   Lb.Du.lm + lp.df.lf                            ]
    //
    //                                              [ Gu   gm  Gb' ]
    //                                            = [ gm'  gf  gp' ]
    //                                              [ Gb   gp      ]
    //
    // lf = sqrt( df.gf - df.lm'.Du.lm )
    // lp = ( gp - Lb.Du.lm ) / ( df.lf )

    Vector<T> &b = b_xfact; //Vector<T> b(dsize-dnbad);
    Vector<T> &bd = bd_xfact; //Vector<T> bd(dsize-dnbad);

    b.resize(dsize-dnbad);
    bd.resize(dsize-dnbad);

    double f;
    T g;
    T h;

    //L(zeroint(),1,dsize-dnbad-1,dsize-dnbad,1,dsize-dnbad).getCol(b,0);

    retMatrix<double> tmpma;

    b = L(zeroint(),1,dsize-dnbad-1,dsize-dnbad,tmpma,"&");
    bd = b;

    if ( dsize-dnbad )
    {
	int i;

	for ( i = 0 ; i < dsize-dnbad ; i++ )
	{
            bd("&",i) *= d(i);
	}
    }

    T temp;

    f = d(dsize-dnbad) * ( real(L(dsize-dnbad,dsize-dnbad)) - real(twoProduct(temp,b,bd)) );

if ( testisvnan(f) )
{
errstream() << "phantomx 0: " << d(dsize-dnbad) << "\n";
errstream() << "phantomx 1: " << real(L(dsize-dnbad,dsize-dnbad)) << "\n";
errstream() << "phantomx 2: " << real(twoProduct(temp,b,bd)) << "\n";
errstream() << "phantomx 3: " << L(dsize-dnbad,dsize-dnbad) << "\n";
errstream() << "phantomx 4: " << twoProduct(temp,b,bd) << "\n";
errstream() << "phantomx 5: " << b << "\n";
errstream() << "phantomx 6: " << bd << "\n";
errstream() << "phantomx 7: " << L << "\n";
}
    NiceAssert( !testisvnan(f) );

    if ( f <= zt*zt )
    {
	if ( dfudge )
	{
            //L("&",dsize-dnbad,dsize-dnbad) += (d(dsize-dnbad)*((zt*zt)-f));
	    f = zt*zt; // f += ((zt*zt)-f);
	}

	else
	{
	    return dnbad;
	}
    }

    // No else, this is deliberate

    f = sqrt(f);

    NiceAssert( !testisvnan(f) );

    L("&",dsize-dnbad,dsize-dnbad) = f;

    if ( dnbad > 1 )
    {
	// and now the flow down.

	f = d(dsize-dnbad)/f;

        NiceAssert( !testisvnan(f) );

	for ( i = dsize-dnbad+1 ; i < dsize ; i++ )
	{
	    g = L(i,dsize-dnbad);

	    if ( dsize-dnbad )
	    {
		for ( j = 0 ; j < dsize-dnbad ; j++ )
		{
		    h = L(i,j);
		    h *= d(j);
		    h *= b(j);

		    g -= h;
		}
	    }

	    g *= f;

	    L("&",i,dsize-dnbad) = g;
	    L("&",dsize-dnbad,i) = conj(g);
	}
    }

    if ( D(dsize-dnbad) > 0 )
    {
	dnbad--;
	dnbadpos--;
    }

    else
    {
	dnbad--;
	dnbadneg--;
    }

    return xfact(D);
}

template <class T>
int Chol<T>::xrankone(const Vector<T> &ax, const double &bx, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D, int hold_off, int z_start, int z_end)
{
    NiceAssert( Gp.numRows() == Gp.numCols() );
    NiceAssert( Gn.numRows() == Gn.numCols() );
    NiceAssert( Gpn.numRows() == Gp.numRows() );
    NiceAssert( Gpn.numCols() == Gn.numCols() );
    NiceAssert( Gp.numRows() == dnpos );
    NiceAssert( Gn.numRows() == dnneg );
    NiceAssert( D.size() == dsize );
    NiceAssert( z_start+z_end <= dsize );
    NiceAssert( ax.size() == dsize );

    // Gnew = Gold + s.a.a'
    //
    // Gold_ij = sum_{k=0,min(i,j)} Lold_ik.d_k.Lold_jk
    // Gnew_ij = sum_{k=0,min(i,j)} Lnew_ik.d_k.Lnew_jk
    //         = Gold_ij + s.a_i.a_j
    //         = sum_{k=0,min(i,j)} Lold_ik.d_k.Lold_jk + s.a_i.a_j
    //
    // Hence:
    //
    // Gnew_00 = d_0.Lnew_00^2 = d_0.Lold_00^2 + s.a_0^2
    // => Lnew_00 = sqrt( Lold_00^2 + d_0.s.a_0^2 )
    //
    // Gnew_i0 = d_0.Lnew_i0.Lnew_00 = d_0.Lold_i0.Lold_00 + s.a_i.a_0
    // => Lnew_i0 = ( Lold_i0.Lold_00 + d_0.s.a_i.a_0 ) / Lnew_00
    //
    // For all i,j > 0:
    //
    // sum_{k=1,min(i,j)} Lnew_ik.d_k.Lnew_jk = sum_{k=1,min(i,j)} Lold_ik.d_k.Lold_jk + Lold_i0.d_0.Lold_j0 - Lnew_i0.d_0.Lnew_j0 + s.a_i.a_j
    //                                        = sum_{k=1,min(i,j)} Lold_ik.d_k.Lold_jk + s.amod_i.amod_j
    //
    // where some calculation will show that:
    //
    // amod_i = ( Lold_00.a_i - Lold_io.a_0 ) / Lnew_00
    //
    // Method: do top left corner and column below it, calculate modified
    //         update and recurse in an iterative manner.

    if ( ( bx != 0.0 ) && ( dsize-z_start-z_end > 0 ) )
    {
	int i,j,k;

        Vector<T> &a = a_xrankone;
	double b;

	a = ax;
	b = bx;

	T x;
	double xabs;
	double yabs;
	T alpha;
	T beta;
	T gamma;
	T epsilon;
	double sgnis;
	int dnbadnew = dnbad;
        int dnbadold = dnbad;

	// Step 1: update the factorised corner
	//
        // We first normalise so that b = +-1.

        double babs = abs2(b);
        double bsqabs = sqrt(babs);

	b /= babs; // b = +-1 after this

	for ( i = z_start ; i < dsize-z_end ; i++ )
	{
            a("&",i) *= bsqabs; // Can't do a single step a("&",z_start,1,dsize-z_end-1) *= bsqabs as bsqabs is type double and a of type Vector<T>, and T != double in general
	}

	if ( dsize-dnbad > z_start )
	{
	    Vector<T> &aaz = aaz_xrankone;

	    aaz = a;

	    for ( i = z_start ; i < dsize-dnbad ; i++ )
	    {
		sgnis = (b*d(i));

		xabs =  abs2(L(i,i));
		xabs *= xabs;

		// xabs = L_{ii}^{{\rm old} 2}

		yabs =  abs2(aaz(i));
		yabs *= yabs;
		yabs *= sgnis;

                // yabs = s.D.|a_i|^2

		xabs += yabs;

		// xabs = L_{ii}^{{\rm old} 2} + s.D.|a_i|^2
		//      = L_{ii}^{{\rm new} 2}

		if ( xabs <= zt*zt )
		{
		    if ( dfudge )
		    {
			xabs = zt*zt;
		    }

		    else
		    {
			for ( j = i ; j < dsize-dnbad ; j++ )
			{
			    for ( k = i ; k < dsize-dnbad ; k++ )
			    {
				xgetG(L("&",j,k),j,k,Gp,Gn,Gpn);
			    }

			    if ( D(j) > 0 )
			    {
                                dnbadnew++;
                                dnbadpos++;
			    }

			    else
			    {
                                dnbadnew++;
                                dnbadneg++;
			    }
			}

			if ( dnbad )
			{
			    for ( j = i ; j < dsize-dnbad ; j++ )
			    {
				for ( k = dsize-dnbad ; k < dsize ; k++ )
				{
				    xgetG(L("&",j,k),j,k,Gp,Gn,Gpn);
				    xgetG(L("&",k,j),k,j,Gp,Gn,Gpn);
				}
			    }
			}

                        dnbad = dnbadnew;

			goto keep_going;
		    }
		}

		// NB: we don't want an "else" statement here as this is also
                // the fall-back case if fudging has been used.

		xabs = sqrt(xabs);

                NiceAssert( !testisvnan(xabs) );

		// xabs = L_{ii}^{\rm new}

		x = xabs;

		// x = L_{ii}^{\rm new} but type T, not type double

		alpha =  aaz(i);
		alpha *= (1/xabs);

                NiceAssert( !testisvnan(alpha) );

		// alpha = a_i / L_{ii}^{\rm new}

		beta =  L(i,i);
		beta *= (1/xabs);

                NiceAssert( !testisvnan(beta) );

		// beta = L_{ii}^{\rm old} / L_{ii}^{\rm new}

		if ( i+1 < dsize )
		{
		    for ( j = i+1 ; j < dsize ; j++ )
		    {
			gamma   = aaz(j);
			epsilon = L(j,i);

			// gamma = a_j
			// epsilon = L_{ji}^{\rm old}

			aaz("&",j) =  ( beta * gamma );
			aaz("&",j) -= ( epsilon * alpha );

			// a_j^{\rm mod} = ( L_{ii}^{\rm old} / L_{ii}^{\rm new} ) a_j    -    L_{ji}^{\rm old} ( a_i / L_{ii}^{\rm new} )

			L("&",j,i) =  ( gamma * conj(alpha) );
			L("&",j,i) *= sgnis;
			L("&",j,i) += ( beta * epsilon );

			// L_{ji}^{\rm new} := ( D_0 s_i a_j {\bar a}_i ) / L_{ii}^{\rm new}    +    ( L_{ii}^{\rm old} L_{ji}^{\rm old} ) / L_{ii}^{\rm new}

			L("&",i,j) = conj(L(j,i));

                        // Fill upper right with the transpose
		    }
		}

		L("&",i,i) = x;

		// L_{ii}^{\rm new} := L_{ii}^{\rm new}
	    }
	}

	// Step 2. update unfactorised corner

    keep_going:

	if ( ( dnbadold > z_end ) && !hold_off )
	{
	    for ( i = dsize-dnbadold ; i < dsize-z_end ; i++ )
	    {
		for ( j = dsize-dnbadold ; j < dsize-z_end ; j++ )
		{
		    xgetG(L("&",i,j),i,j,Gp,Gn,Gpn);
		}
	    }
	}
    }

    return dnbad;
}

template <class T>
int Chol<T>::xsetfact(const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn, const Vector<double> &D)
{
    NiceAssert( Gp.numRows() == Gp.numCols() );
    NiceAssert( Gn.numRows() == Gn.numCols() );
    NiceAssert( Gpn.numRows() == Gp.numRows() );
    NiceAssert( Gpn.numCols() == Gn.numCols() );
    NiceAssert( Gp.numRows() == dnpos );
    NiceAssert( Gn.numRows() == dnneg );
    NiceAssert( D.size() == dsize );

    d = D;
    L.resize(dsize,dsize);

    ddpnset.resize(dsize);
    dposind.resize(dsize);
    dnegind.resize(dsize);

    int i,j;

    int ip,in;

    ip = 0;
    in = 0;

    if ( dsize )
    {
	for ( i = 0 ; i < dsize ; i++ )
	{
	    dposind("&",i) = ip;
	    dnegind("&",i) = in;

	    if ( d(i) > 0 )
	    {
		ddpnset("&",i) = 1;
		ip++;
	    }

	    else
	    {
		ddpnset("&",i) = 0;
		in++;
	    }
	}
    }

    NiceAssert( ip == dnpos );
    NiceAssert( in == dnneg );

    if ( dsize )
    {
	for ( i = 0 ; i < dsize ; i++ )
	{
	    for ( j = 0 ; j < dsize ; j++ )
	    {
		xgetG(L("&",i,j),i,j,Gp,Gn,Gpn);
	    }
	}
    }

    return dnbad;
}

template <class T>
void Chol<T>::xgetG(T &res, int i, int j, const Matrix<T> &Gp, const Matrix<T> &Gn, const Matrix<T> &Gpn) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dsize );
    NiceAssert( j >= 0 );
    NiceAssert( j < dsize );

    if ( ddpnset(i) )
    {
	if ( ddpnset(j) )
	{
            res = Gp(dposind(i),dposind(j));
	}

	else
	{
            res = Gpn(dposind(i),dnegind(j));
	}
    }

    else
    {
	if ( ddpnset(j) )
	{
            res = conj(Gpn(dposind(j),dnegind(i)));
	}

	else
	{
            res = Gn(dnegind(i),dnegind(j));
	}
    }

    return;
}


template <class T>
void Chol<T>::xsetG(const T &src, int i, int j, Matrix<T> &Gp, Matrix<T> &Gn, Matrix<T> &Gpn) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dsize );
    NiceAssert( j >= 0 );
    NiceAssert( j < dsize );

    if ( ddpnset(i) )
    {
	if ( ddpnset(j) )
	{
            Gp("&",dposind(i),dposind(j)) = src;
	}

	else
	{
            Gpn("&",dposind(i),dnegind(j)) = src;
	}
    }

    else
    {
	if ( ddpnset(j) )
	{
            Gpn("&",dposind(j),dnegind(i)) = conj(src);
	}

	else
	{
            Gn("&",dnegind(i),dnegind(j)) = src;
	}
    }

    return;
}

template <class T>
template <class S> void Chol<T>::xgetvect(S &res, int i, const Vector<S> &vp, const Vector<S> &vn) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dsize );

    if ( ddpnset(i) )
    {
	res = vp(dposind(i));
    }

    else
    {
	res = vn(dnegind(i));
    }

    return;
}

template <class T>
template <class S> void Chol<T>::xsetvect(const S &src, int i, Vector<S> &vp, Vector<S> &vn) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dsize );

    if ( ddpnset(i) )
    {
	vp("&",dposind(i)) = src;
    }

    else
    {
	vn("&",dnegind(i)) = src;
    }

    return;
}

template <class T>
void Chol<T>::calc_zstart_zend(int &z_start, int &z_end, int zp_start, int zp_end, int zn_start, int zn_end, int end_back) const
{
    int i;

    z_start = 0;
    z_end = 0;

    for ( i = 0 ; i < dsize-end_back ; i++ )
    {
	if ( !ddpnset(i) && !zn_start )
	{
	    break;
	}

	if ( ddpnset(i) && !zp_start )
	{
	    break;
	}

	z_start++;

	if ( !ddpnset(i) )
	{
	    zn_start--;
	}

	else
	{
	    zp_start--;
	}
    }

    for ( i = dsize-end_back-1 ; i >= 0 ; i-- )
    {
	if ( !ddpnset(i) && !zn_end )
	{
	    break;
	}

	if ( ddpnset(i) && !zp_end )
	{
	    break;
	}

	z_end++;

	if ( !ddpnset(i) )
	{
	    zn_end--;
	}

	else
	{
	    zp_end--;
	}
    }

    return;
}



#endif
