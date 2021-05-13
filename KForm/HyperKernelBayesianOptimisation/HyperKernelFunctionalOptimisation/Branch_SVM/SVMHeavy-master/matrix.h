
//
// Matrix class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _matrix_h
#define _matrix_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <sstream>
#include "vector.h"
#include "sparsevector.h"

template <class T> class Matrix;

#define MATRIX_ZTOL 1e-10
// max recursion depth when calculating determinant before invoking LUP decomposition alternative method
#define MATRIX_DETRECURSEMAX 6

OVERLAYMAKEFNVECTOR(Matrix<int>)
OVERLAYMAKEFNVECTOR(Matrix<double>)
OVERLAYMAKEFNVECTOR(Vector<Matrix<int> >)
OVERLAYMAKEFNVECTOR(Vector<Matrix<double> >)

// Swap function

template <class T> void qswap(Matrix<T> &a, Matrix<T> &b);
template <class T> void qswap(Matrix<T> *&a, Matrix<T> *&b);
template <class T> void qswap(const Matrix<T> *&a, const Matrix<T> *&b);

// Matrix return handle.

template <class T> class retMatrix;

template <class T>
class retMatrix : public Matrix<T>
{
public:
    svm_explicit retMatrix() : Matrix<T>("&") { return; }

    // This function resets the return matrix to clean-slate.  No need to
    // call this as it gets called when required by operator().

    retMatrix<T> &reset(void);
};

// The class itself

template <class T>
class Matrix
{
    friend class retMatrix<T>;

    template <class S> friend void qswap(Matrix<S> &a, Matrix<S> &b);

public:

    // Constructors and Destructors
    //
    // - (numRows,numCols): matrix is size numRows * numCols (default 0*0)
    // - (elm,row,dref,celm,crow,cdref,numRows,numCols): like above, but
    //   rather than being stored locally the contents are stored elsewhere
    //   and can be retrieved by calling elm(i,j,dref), row(i,dref)
    //   (writable) or celm(i,j,cdref), crow(i,cdref) (fixed).
    // - (src): copy constructor

    svm_explicit Matrix(int numRows = 0, int numCols = 0);
    svm_explicit Matrix(const T &(*celm)(int, int, const void *), const Vector<T> &(*crow)(int, const void *), const void *cdref, int numRows = 0, int numCols = 0, void (*xcdelfn)(const void *, void *) = NULL, void *dref = NULL);
    svm_explicit Matrix(T &(*elm)(int, int, void *), Vector<T> &(*row)(int, void *), void *dref, const T &(*celm)(int, int, const void *), const Vector<T> &(*crow)(int, const void *), const void *cdref, int numRows = 0, int numCols = 0, void (*xdelfn)(void *) = NULL, void (*xcdelfn)(const void *, void *) = NULL);
                 Matrix(const Matrix<T> &src);

    ~Matrix();

    // Assignment
    //
    // - matrix assignment: unless this matrix is a temporary matrix created
    //   to refer to parts of another matrix then we do not require that sizes
    //   align but rather completely overwrite *this, resetting the size to
    //   that of the source.
    // - vector assignment: if numRows == 1 and numCols = the size of the src
    //   vector then this acts like assignment to the matrix from a row
    //   vector.  Otherwise acts as assignment from a column vector, resizing
    //   if required (if possible - ie. this isn't a temporary matrix created
    //   to refer to parts of another matrix).
    // - scalar assignment: in this case the size of the matrix remains
    //   unchanged, but all elements will be overwritten.
    // - behaviour is undefined if scalar is an element of this.

    Matrix<T> &operator=(const Matrix<T> &src);
    Matrix<T> &operator=(const Vector<T> &src);
    Matrix<T> &operator=(const T &src);

    // simple matrix manipulations
    //
    // ident:      apply ident to diagonal elements of matrix and zero to off-
    //             diagonal elements
    // zero:       apply zero to all elements of the matrix (vectorially)
    // posate:     apply posate to all elements of the matrix (vectorially)
    // negate:     apply negate to all elements of the matrix (vectorially)
    // conj:       apply conj to all elements of the matrix (vectorially)
    // transpose:  transpose matrix
    // symmetrise: set this = 1/2 ( this + transpose(this) )
    //
    // all return *this

    Matrix<T> &ident(void);
    Matrix<T> &zero(void);
    Matrix<T> &posate(void);
    Matrix<T> &negate(void);
    Matrix<T> &conj(void);
    Matrix<T> &rand(void);
    Matrix<T> &transpose(void);
    Matrix<T> &symmetrise(void);

    // Access:
    //
    // - ("&",i,j) - access a reference to scalar element i,j
    // - (i,j)      - access a const reference to scalar element i,j
    //
    // Variants:
    //
    // - if i/j is of type Vector<int> then the reference returned is to the
    //   elements specified in i/j.
    // - if ib,is,im is given then this is the same as a vector i being used
    //   specified by: ( ib ib+is ib+is+is ... max_n(i=ib+(n*s)|i<im) )
    //   (and if im < ib then an empty reference is returned).  Same for j.
    // - if i is a vector or in ib/is/im form and j an int then for technical
    //   reasons (because the matrix is stored as row vectors) we must return
    //   a column matrix, not a vector.
    // - for disambiguation if i is in ib/is/im form and j is an int then an
    //   additional dummy ("&") argument is required as the last argument.
    // - the argument j is optional.  If j is not present then it is assumed
    //   that all columns are included.
    //
    // Scope of result:
    //
    // - The scope of the returned reference is the minimum of the scope of
    //   retVector/retMatrix &tmp or *this (or tmpa if present).
    // - retVector/retMatrix &tmp may or may not be used depending on, so
    //   never *assume* that it will be!
    // - The returned reference is actually *this through a layer of indirection, 
    //   so any changes to it will be reflected in *this (and vice-versa).

    Matrix<T> &operator()(const char *dummy                                                                                       ) { (void) dummy;  return (*this);                                                   }
    Vector<T> &operator()(const char *dummy, int i,                                          retVector<T> &tmp                    ) {                return (*this)(dummy,i,              0,1,numCols()-1,tmp       ); }
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,                           retMatrix<T> &tmp                    ) {                return (*this)(dummy,i,              0,1,numCols()-1,tmp       ); }
    Matrix<T> &operator()(const char *dummy, int ib, int is, int im,                         retMatrix<T> &tmp                    ) {                return (*this)(dummy,ib,is,im,       0,1,numCols()-1,tmp       ); }
//  Matrix<T> &operator()(const char *dummy,                         int j,                  retMatrix<T> &tmp, const char *dummyb) {                return (*this)(dummy,0,1,numRows()-1,j,              tmp,dummyb); }
    T         &operator()(const char *dummy, int i,                  int j                                                        );
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   int j,                  retMatrix<T> &tmp                    ) {                return (*this)(dummy,i,              j,1,j,          tmp       ); }
    Matrix<T> &operator()(const char *dummy, int ib, int is, int im, int j,                  retMatrix<T> &tmp, const char *dummyb) { (void) dummyb; return (*this)(dummy,ib,is,im,       j,1,j,          tmp       ); }
    Matrix<T> &operator()(const char *dummy,                         const Vector<int> &j,   retMatrix<T> &tmp, const char *dummyb) { (void) dummyb; return (*this)(dummy,0,1,numRows()-1,j,              tmp       ); }
    Vector<T> &operator()(const char *dummy, int i,                  const Vector<int> &j,   retVector<T> &tmp, retVector<T> &tmpb);
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   const Vector<int> &j,   retMatrix<T> &tmp                    );
    Matrix<T> &operator()(const char *dummy, int ib, int is, int im, const Vector<int> &j,   retMatrix<T> &tmp                    );
    Matrix<T> &operator()(const char *dummy,                         int jb, int js, int jm, retMatrix<T> &tmp, const char *dummyb) { (void) dummyb; return (*this)(dummy,0,1,numRows()-1,jb,js,jm,       tmp       ); }
    Vector<T> &operator()(const char *dummy, int i,                  int jb, int js, int jm, retVector<T> &tmp                    );
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   int jb, int js, int jm, retMatrix<T> &tmp                    );
    Matrix<T> &operator()(const char *dummy, int ib, int is, int im, int jb, int js, int jm, retMatrix<T> &tmp                    );

    const Matrix<T> &operator()(void                                                                                 ) const {                return (*this);                                             }
    const Vector<T> &operator()(int i,                                          retVector<T> &tmp                    ) const {                return (*this)(i,              0,1,numCols()-1,tmp       ); }
    const Matrix<T> &operator()(const Vector<int> &i,                           retMatrix<T> &tmp                    ) const {                return (*this)(i,              0,1,numCols()-1,tmp       ); }
    const Matrix<T> &operator()(int ib, int is, int im,                         retMatrix<T> &tmp                    ) const {                return (*this)(ib,is,im,       0,1,numCols()-1,tmp       ); }
    const Matrix<T> &operator()(                        int j,                  retMatrix<T> &tmp, const char *dummyb) const {                return (*this)(0,1,numRows()-1,j,              tmp,dummyb); }
    const T         &operator()(int i,                  int j                                                        ) const;
    const Matrix<T> &operator()(const Vector<int> &i,   int j,                  retMatrix<T> &tmp                    ) const {                return (*this)(i,              j,1,j,          tmp       ); }
    const Matrix<T> &operator()(int ib, int is, int im, int j,                  retMatrix<T> &tmp, const char *dummyb) const { (void) dummyb; return (*this)(ib,is,im,       j,1,j,          tmp       ); }
    const Matrix<T> &operator()(                        const Vector<int> &j,   retMatrix<T> &tmp, const char *dummyb) const { (void) dummyb; return (*this)(0,1,numRows()-1,j,              tmp       ); }
    const Vector<T> &operator()(int i,                  const Vector<int> &j,   retVector<T> &tmp, retVector<T> &tmpb) const;
    const Matrix<T> &operator()(const Vector<int> &i,   const Vector<int> &j,   retMatrix<T> &tmp                    ) const;
    const Matrix<T> &operator()(int ib, int is, int im, const Vector<int> &j,   retMatrix<T> &tmp                    ) const;
    const Matrix<T> &operator()(                        int jb, int js, int jm, retMatrix<T> &tmp, const char *dummyb) const { (void) dummyb; return (*this)(0,1,numRows()-1,jb,js,jm,       tmp       ); }
    const Vector<T> &operator()(int i,                  int jb, int js, int jm, retVector<T> &tmp                    ) const;
    const Matrix<T> &operator()(const Vector<int> &i,   int jb, int js, int jm, retMatrix<T> &tmp                    ) const;
    const Matrix<T> &operator()(int ib, int is, int im, int jb, int js, int jm, retMatrix<T> &tmp                    ) const;

    // Row and column norms

    double getColNorm(int i) const;
    double getRowNorm(int i) const;
    double getRowColNorm(void) const;

    double getColAbs(int i) const;
    double getRowAbs(int i) const;
    double getRowColAbs(void) const;

    // Add and remove element functions.
    //
    // Note that these may not be applied to temporary matrices.  Each
    // returns a reference to *this
    //
    // PadCol: adds n columns to right and zeros them
    // PadRow: adds n rows to bottom and zeros them
    // PadRowCol: adds n columns to right and bottom and zeros them

    Matrix<T> &addRow(int i);
    Matrix<T> &removeRow(int i);

    Matrix<T> &addCol(int i);
    Matrix<T> &removeCol(int i);

    Matrix<T> &addRowCol(int i);
    Matrix<T> &removeRowCol(int i);

    Matrix<T> &resize(int targNumRows, int targNumCols);
    template <class S> Matrix<T> &resize(const Matrix<S> &sizeTemplateUsed) { return resize(sizeTemplateUsed.numRows(),sizeTemplateUsed.numCols()); }

    Matrix<T> &appendRow(int rowStart, const Matrix<T> &src);
    Matrix<T> &appendCol(int colStart, const Matrix<T> &src);

    Matrix<T> &padCol(int n);
    Matrix<T> &padRow(int n);
    Matrix<T> &padRowCol(int n);

    // Function application - apply function fn to each element of matrix

    Matrix<T> &applyon(T (*fn)(T));
    Matrix<T> &applyon(T (*fn)(const T &));
    Matrix<T> &applyon(T (*fn)(T, const void *), const void *a);
    Matrix<T> &applyon(T (*fn)(const T &, const void *), const void *a);
    Matrix<T> &applyon(T &(*fn)(T &));
    Matrix<T> &applyon(T &(*fn)(T &, const void *), const void *a);

    // Other functions
    //
    // - rankone adds c*outerproduct(a,b) to the matrix without wasting memory
    //   by calling outerproduct(a,b), creating a temporary matrix and then
    //   adding that.
    // - diagmult G := J.G.K' where J and K are diagonal matrices (stored as
    //   vectors).
    // - diagoffset G += diag(d) (if d is a matrix then diagonal elements used)
    // - rankoneNoConj adds c*outerproductNoConj(a,b)
    // - diagmultNoConj doesn't conjugate K
    // - naiveChol: naive cholesky factorisation.  The matrix is assumed to
    //   be symmetric positive definite and no isnan checking is done.  The 
    //   result is written into the lower triangular part of dest.  The upper
    //   triangular is only zeroed if zeroupper is true, otherwise it not 
    //   touched.  If diagoffset is given then the cholesky factorisation of
    //   G+diag(diagoffset) is calculated instead.
    // - naiveCholNoConj: assumes T is commutative and has a division
    //   operation.
    // - naivepartChol: Like naiveChol, but does maximal partial factorisation
    //   a pivotted version of G.  It sets the pivot vector p and the factorisation
    //   size n, so G(p,p) = L(p,p(0:1:n-1))*L(p,p(0:1:n-1))'.  Assume that G
    //   is positive semi-definite.
    // - SVD: computes SVD, assuming real, to find UDV, where U is 
    //   left-orthogonal, V right-orthogonal and D diagonal.
    // - forwardElim: Solve L*y = b for y, where L is the lower triangular
    //   part of the matrix (must by square).  Returns reference to y.  If 
    //   implicitTranspose is set then use L = transpose of upper triangular
    //   part of matrix
    // - backwardSubst: Solve U*y = b for y, where U is the upper triangular
    //   part of the matrix (must by square).  Returns reference to y.  If 
    //   implicitTranspose is set then use L = transpose of lower triangular
    //   part of matrix
    // - naiveCholInveNoConj: constructs the cholesky factorisation using
    //   naiveCholNoConj, assuming symmetric positive definite (and no
    //   checking) and then uses forwardElim and backwardSubst to find y,
    //   where y = G.b.  If yinterm is given then the intermediate result
    //   yinterm L*yinterm = b is stored in yinterm (which is faster, as no
    //   constructores are required).  If cholres argument is given then
    //   then cholesky factorisation is stored in cholres (which is faster,
    //   as no constructors are required).  The zeroupper argument works as
    //   described in naiveChol in this case.  The diagoffset argument is as
    //   described in naivechol.

    Matrix<T> &rankone(const T &c, const Vector<T> &a, const Vector<T> &b);
    Matrix<T> &diagmult(const Vector<T> &J, const Vector<T> &K);
    Matrix<T> &diagoffset(const T &d);
    Matrix<T> &diagoffset(const Vector<T> &d);
    Matrix<T> &diagoffset(const Matrix<T> &d);
    Matrix<T> &naiveChol(Matrix<T> &dest, int zeroupper = 1) const;
    Matrix<T> &naiveChol(Matrix<T> &dest, const Vector<T> &diagoffset, int zeroupper = 1) const;
    Matrix<T> &naivepartChol(Matrix<T> &dest, Vector<int> &p, int &n, int zeroupper = 1, double ztol = MATRIX_ZTOL) const;
    Matrix<T> &SVD(Matrix<T> &u, Vector<T> &d, Matrix<T> &v) const;
    Matrix<T> &rankoneNoConj(const T &c, const Vector<T> &a, const Vector<T> &b);
    Matrix<T> &diagmultNoConj(const Vector<T> &J, const Vector<T> &K);
    Matrix<T> &diagoffsetNoConj(const Vector<T> &d);
    Matrix<T> &naiveCholNoConj(Matrix<T> &dest, int zeroupper = 1) const;
    Matrix<T> &naiveCholNoConj(Matrix<T> &dest, const Vector<T> &diagoffset, int zeroupper = 1) const;
    template <class S> Vector<S> &forwardElim(Vector<S> &y, const Vector<S> &b, int implicitTranspose = 0) const;
    template <class S> Vector<S> &backwardSubst(Vector<S> &y, const Vector<S> &b, int implicitTranspose = 0) const;
    template <class S> Vector<S> &naiveCholInveNoConj( Vector<S> &y, const Vector<S> &b                                                                                         ) const;
    template <class S> Vector<S> &naiveCholInveNoConj( Vector<S> &y, const Vector<S> &b,                              Vector<S> &yinterm                                        ) const;
    template <class S> Vector<S> &naiveCholInveNoConj( Vector<S> &y, const Vector<S> &b,                                                  Matrix<T> & cholres, int zeroupper = 1) const;
    template <class S> Vector<S> &naiveCholInveNoConj( Vector<S> &y, const Vector<S> &b,                              Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper = 1) const;
    template <class S> Vector<S> &naiveCholInveNoConjx(Vector<S> &y, const Vector<S> &b, const Vector<T> &diagoffset                                                            ) const;
    template <class S> Vector<S> &naiveCholInveNoConjx(Vector<S> &y, const Vector<S> &b, const Vector<T> &diagoffset, Vector<S> &yinterm                                        ) const;
    template <class S> Vector<S> &naiveCholInveNoConjx(Vector<S> &y, const Vector<S> &b, const Vector<T> &diagoffset,                     Matrix<T> & cholres, int zeroupper = 1) const;
    template <class S> Vector<S> &naiveCholInveNoConjx(Vector<S> &y, const Vector<S> &b, const Vector<T> &diagoffset, Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper = 1) const;

    // Determinants, inverses, pseudoinverses etc.
    //
    // minor appears to be taken by some sort of macro (in gcc at least),
    // so using miner in its place.  Sorry 'bout that
    //
    // invtrace calculated 1/trace(inv(A)) = det()/sum_i(miner(i,i))
    //
    // LUPDecompose: construct LU decomposition.  A(P) = LU
    //                                         L lower triangular
    //                                         L has ones on diag
    //                                         U upper triangular
    //                                         P is the pivot vector
    //            res = ((L-I)+U)(P)
    //            where P is a permutation matrix returned as a vector
    //            returns S, where S is the number of permutations
    //            (if res missing then does it on this matrix)
    //            returns -1 on failure
    //
    // tridiag: calculate tridiagonal form, assuming real symmetric
    //
    //        d contains the diagonal elements of the tridiagonal matrix.
    //
    //        e contains the subdiagonal elements of the tridiagonal
    //          matrix in its last n-1 positions.  e(1) is set to zero.
    //
    //        z contains the orthogonal transformation matrix
    //          produced in the reduction.
    //
    //        e2 contains the squares of the corresponding elements of e.
    //          e2 may coincide with e if the squares are not needed.
    //
    // eig: calculates eigenvalues/vectors
    //
    //        w  contains the eigenvalues in ascending order.
    //
    //        z  contains the eigenvectors (eigenvectors are in columns).
    //
    //        return:  is an integer output variable set equal to an error
    //           completion code described in the documentation for tqlrat
    //           and tql2.  the normal completion code is zero.
    //
    //        fv1  and  fv2  are temporary storage arrays.
    //
    // projpsd: project onto nearest positive semidefinite matrix.  fv... are temporary
    // projnsd: project onto nearest negative semidefinite matrix.  fv... are temporary
    //
    // NB: - I make no guarantee that these will work for non-doubles.
    //     - Inverting matrices is a bad idea (tm) in general.
    //     - For non-square matrices, inverse will return conjugate transpose
    //       of the pseudoinverse

    T det(void) const;
    T trace(void) const;
    T invtrace(void) const;
    T miner(int i, int j) const;
    T cofactor(int i, int j) const;
    Matrix<T> &adj( Matrix<T> &res) const;
    Matrix<T> &inve(Matrix<T> &res) const;
    Matrix<T> &inveSymmNoConj(Matrix<T> &res) const;
    int LUPDecompose(Matrix<T> &res, Vector<int> &p, double ztol = MATRIX_ZTOL) const;
    int LUPDecompose(double ztol = MATRIX_ZTOL);
    void tridiag(Vector<T> &d, Vector<T> &e, Vector<T> &e2) const;
    void tridiag(Vector<T> &d, Vector<T> &e, Matrix<T> &z ) const;
    int eig(Vector<T> &w, Vector<T> &fv1, Vector<T> &fv2) const;
    int eig(Vector<T> &w, Matrix<T> &z,   Vector<T> &fv1) const;
    int projpsd(Matrix<T> &res, Vector<T> &fv1, Matrix<T> &fv2, Vector<T> &fv3) const;
    int projnsd(Matrix<T> &res, Vector<T> &fv1, Matrix<T> &fv2, Vector<T> &fv3) const;

    Matrix<T> adj( void) const;
    Matrix<T> inve(void) const;
    Matrix<T> inveSymmNoConj(void) const;

    // Other stuff:
    //
    // rowsum: sum of rows in matrix
    // colsum: sum of columns in matrix
    // vertsum: sum of elements in column j
    // horizsum: sum of elements in row i
    // scale: scale matrix by amount (*this *= a)

    const Vector<T> &rowsum(Vector<T> &res) const;
    const Vector<T> &colsum(Vector<T> &res) const;
    const T &vertsum(int j, T &res) const;
    const T &horizsum(int i, T &res) const;
    template <class S> Matrix<T> &scale(const S &a);
    template <class S> Matrix<T> &scaleAdd(const S &a, const Matrix<T> &b);

    // Information
    //
    // numRows()  = number of rows in matrix
    // numCols()  = number of columns in matrix
    // size()     = max(numRows(),numCols())
    // isSquare() = numRows() == numCols()
    // isEmpty()  = !numRows() && !numCols()

    int numRows(void)  const { return dnumRows;                                      }
    int numCols(void)  const { return dnumCols;                                      }
    int size(void)     const { return ( dnumCols > dnumRows ) ? dnumCols : dnumRows; }
    int isSquare(void) const { return ( dnumRows == dnumCols );                      }
    int isEmpty(void)  const { return ( !dnumCols && !dnumRows );                    }

    // pre-allocation function (see vector.h for more detail)

    void prealloc(int newallocrows, int newalloccols);

    // Slight complication from vector.h

    int shareBase(const Matrix<T> *that) const;

    // Casting operator used by vector regression template
    //
    // This is actually needed when the elements of the kernel matrix in an 
    // SVM are themselves matrices (MSVR, division-algebraic SVR) but we
    // still need some measure of mean/median diagonals for automatic
    // parameter tuning.

    operator T() const { return (*this)(0,0); }

    // "Cheat" ways of setting the external evaluation arguments.  These are
    // needed when we start messing with swap functions at higher levels.

    void cheatsetdref(void *newdref)         { dref  = newdref;  }
    void cheatsetcdref(const void *newcdref) { cdref = newcdref; }

private:

    // dnumRows: the height of the matrix
    // dnumCols: the width of the matrix
    //
    // nbase: 0 if content is local, 1 if it points elsewhere
    //        (NB: if nbase == 0 then pivotRow = pivotCol = ( 0 1 2 ... ))
    // pbaseRow: 0 if pivotRow is local, 1 if it points elsewhere
    //           (NB: if nbase == 0 then pbaseRow == 0 by definition)
    // pbaseCol: 0 if pivotCol is local, 1 if it points elsewhere
    //           (NB: if nbase == 0 then pbaseCol == 0 by definition)
    //
    // iibRow: constant added to row indices
    // iisRow: step for row indices
    //
    // iibCol: constant added to column indices
    // iisCol: step for column indices
    //
    // bkref: if nbase, this is the matrix derived from (and pointed to)
    // content: contents of matrix
    // ccontent: constant pointer to content
    // pivotRow: row pivotting used to access contents
    // pivotCol: column pivotting used to access contents
    //
    // iscover: 0 if matrix normal, 1 if it gets data from elsewhere
    // elmfn: if iscover then this function is called to access data
    // rowfn: if iscover then this function is called to access rows of data
    // dref: argument passed to elmfn and rowfn, presumably giving details
    // celmfn: if iscover then this fn is called to access (const) data
    // crowfn: if iscover then this fn is called to access rows of const data
    // cdref: argument passed to celmfn and crowfn, presumably giving details

    int dnumRows;
    int dnumCols;

    int nbase;
    int pbaseRow;
    int pbaseCol;

    int iibRow;
    int iisRow;

    int iibCol;
    int iisCol;

    const Matrix<T> *bkref;
    Vector<Vector<T> > *content;
    const Vector<Vector<T> > *ccontent;
    const DynArray<int> *pivotRow;
    const DynArray<int> *pivotCol;

    int iscover;
    T &(*elmfn)(int, int, void *);
    Vector<T> &(*rowfn)(int, void *);
    void (*delfn)(void *);
    void *dref;
    const T &(*celmfn)(int, int, const void *);
    const Vector<T> &(*crowfn)(int, const void *);
    void (*cdelfn)(const void *, void *);
    const void *cdref;

    // Blind constructor: does no allocation, just sets bkref and defaults

    svm_explicit Matrix(const char *dummy, const Matrix<T> &src);
    svm_explicit Matrix(const char *dummy);

    // Fix bkref

    void fixbkreftree(const Matrix<T> *newbkref);

    // Fast (but could cause problem X) internal version

    Vector<T> &operator()(const char *dummy, int i, const char *dummyb, retVector<T> &tmp);

    // Buffers for speed

    Matrix<T> *matbuff;
    Vector<int> pbuff;

    // I HAVE NO IDEA WHY THIS IS REQUIRED!  For some reason, member
    // functions can't "see" the overloads of conj contained in numbase.h.
    // It's almost as if the void member overloads of conj in this class
    // blocks the compiler from seeing the other options.  Hopefully this is
    // a temporary bug to be fixed in future releases of gcc.  Until now, we
    // need to do this.

    double conj(const double &a) const { return a; }

    // const removal cheat

    Matrix<T> *thisindirect[1];
};

template <class T> void qswap(Matrix<T> &a, Matrix<T> &b)
{
    NiceAssert( a.nbase == 0 );
    NiceAssert( b.nbase == 0 );

    qswap(a.dnumRows,b.dnumRows);
    qswap(a.dnumCols,b.dnumCols);
    qswap(a.nbase   ,b.nbase   );
    qswap(a.pbaseRow,b.pbaseRow);
    qswap(a.pbaseCol,b.pbaseCol);
    qswap(a.iibRow  ,b.iibRow  );
    qswap(a.iisRow  ,b.iisRow  );
    qswap(a.iibCol  ,b.iibCol  );
    qswap(a.iisCol  ,b.iisCol  );
    qswap(a.iscover ,b.iscover );

    const Matrix<T> *bkref;
    Vector<Vector<T> > *content;
    const Vector<Vector<T> > *ccontent;
    const DynArray<int> *pivotRow;
    const DynArray<int> *pivotCol;

    bkref    = a.bkref;    a.bkref    = b.bkref;    b.bkref    = bkref;
    content  = a.content;  a.content  = b.content;  b.content  = content;
    ccontent = a.ccontent; a.ccontent = b.ccontent; b.ccontent = ccontent;
    pivotRow = a.pivotRow; a.pivotRow = b.pivotRow; b.pivotRow = pivotRow;
    pivotCol = a.pivotCol; a.pivotCol = b.pivotCol; b.pivotCol = pivotCol;

    T &(*elmfn)(int, int, void *);
    Vector<T> &(*rowfn)(int, void *);
    void (*delfn)(void *);
    void *dref;
    const T &(*celmfn)(int, int, const void *);
    const Vector<T> &(*crowfn)(int, const void *);
    void (*cdelfn)(const void *, void *);
    const void *cdref;

    elmfn  = a.elmfn;  a.elmfn  = b.elmfn;  b.elmfn  = elmfn;
    rowfn  = a.rowfn;  a.rowfn  = b.rowfn;  b.rowfn  = rowfn;
    delfn  = a.delfn;  a.delfn  = b.delfn;  b.delfn  = delfn;
    dref   = a.dref;   a.dref   = b.dref;   b.dref   = dref;
    celmfn = a.celmfn; a.celmfn = b.celmfn; b.celmfn = celmfn;
    crowfn = a.crowfn; a.crowfn = b.crowfn; b.crowfn = crowfn;
    cdelfn = a.cdelfn; a.cdelfn = b.cdelfn; b.cdelfn = cdelfn;
    cdref  = a.cdref;  a.cdref  = b.cdref;  b.cdref  = cdref;

    // The above will have messed up one important thing, namely bkref and
    // bkref in any child vectors.  We must now repair the child trees if
    // they exist

    a.fixbkreftree(&a);
    b.fixbkreftree(&b);

    return;
}

template <class T>
void qswap(Matrix<T> *&a, Matrix<T> *&b)
{
    Matrix<T> *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}

template <class T>
void qswap(const Matrix<T> *&a, const Matrix<T> *&b)
{
    const Matrix<T> *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}

template <class T>
void Matrix<T>::fixbkreftree(const Matrix<T> *newbkref)
{
    bkref = newbkref;

    return;
}





// Various functions
//
// max: find max element, put index in i,j.
// min: find min element, put index in i,j.
// maxdiag: find max diagonal element, put index in i,j.
// mindiag: find min diagonal element, put index in i,j.
// maxabs: find the |max| element, put index in i,j.
// minabs: find the |min| element, put index in i,j.
// maxabsdiag: find the |max| diagonal element, put index in i,j.
// minabsdiag: find the |min| diagonal element, put index in i,j.
// outerProduct: calculate the outer product of two vectors
// outerProductNoConj: calculate the outer product of two vectors without conjugation
//
// sum: find the sum of elements in a matrix.
// mean: calculate the mean of elements in a matrix.
// median: calculate the median of elements in a matrix.
//
// setident: apply ident to diagonal elements of matrix and zero to off-diagonal elements
// setzero: apply zero t0 all elements of matrix (vectorially)
// setposate: apply posate to all element of matrix (vectorially)
// setnegate: apply negate to all element of matrix (vectorially)
// setconj: apply conj to all element of matrix (vectorially)
// settranspose: apply transpose to all element of matrix (vectorially)
//
// inv: find the inverse of the matrix
// abs2: returns sum of abs2 of elements
// distF: returns the Frobenius distance

template <class T> const T &max(const Matrix<T> &right_op, int &i, int &j);
template <class T> const T &min(const Matrix<T> &right_op, int &i, int &j);
template <class T> const T &maxdiag(const Matrix<T> &right_op, int &i, int &j);
template <class T> const T &mindiag(const Matrix<T> &right_op, int &i, int &j);
template <class T> T maxabs(const Matrix<T> &right_op, int &i, int &j);
template <class T> T minabs(const Matrix<T> &right_op, int &i, int &j);
template <class T> T maxabsdiag(const Matrix<T> &right_op, int &i, int &j);
template <class T> T minabsdiag(const Matrix<T> &right_op, int &i, int &j);
template <class T> Matrix<T> outerProduct(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Matrix<T> outerProductNoConj(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> const Matrix<T> &takeProduct(Matrix<T> &res, const Matrix<T> &a, const Matrix<T> &b);

template <class T> T sum(const Matrix<T> &right_op);
template <class T> T mean(const Matrix<T> &right_op);
template <class T> const T &median(const Matrix<T> &right_op, int &i, int &j);

template <class T> Matrix<T> &setident(Matrix<T> &a);
template <class T> Matrix<T> &setzero(Matrix<T> &a);
template <class T> Matrix<T> &setposate(Matrix<T> &a);
template <class T> Matrix<T> &setnegate(Matrix<T> &a);
template <class T> Matrix<T> &setconj(Matrix<T> &a);
template <class T> Matrix<T> &setrand(Matrix<T> &a);
template <class T> Matrix<T> &settranspose(Matrix<T> &a);
template <class T> Matrix<T> &postProInnerProd(Matrix<T> &a) { return a; }

template <class T> Matrix<T> inv(const Matrix<T> &src);
//template <class T> double abs2(const Matrix<T> &src);
template <class T> double absF(const Matrix<T> &src);
template <class T> double normF(const Matrix<T> &src);
template <class T> double absd(const Matrix<T> &src);
template <class T> double normd(const Matrix<T> &src);
template <class T> double distF(const Matrix<T> &a, const Matrix<T> &b);



// NaN and inf tests

template <class T> int testisvnan(const Matrix<T> &x);
template <class T> int testisinf (const Matrix<T> &x);
template <class T> int testispinf(const Matrix<T> &x);
template <class T> int testisninf(const Matrix<T> &x);


// Conversion from strings

template <class T> Matrix<T> &atoMatrix(Matrix<T> &dest, const std::string &src);
template <class T> Matrix<T> &atoMatrix(Matrix<T> &dest, const std::string &src)
{
    std::istringstream srcmod(src);

    srcmod >> dest;

    return dest;
}

// Random permutation function and random fill
//
// ltfill: lhsres(i,j) = 1 if lhsres(i,j) <  rhs(i,j), 0 otherwise
// gtfill: lhsres(i,j) = 1 if lhsres(i,j) >  rhs(i,j), 0 otherwise
// lefill: lhsres(i,j) = 1 if lhsres(i,j) <= rhs(i,j), 0 otherwise
// gefill: lhsres(i,j) = 1 if lhsres(i,j) >= rhs(i,j), 0 otherwise

template <class T> Matrix<T> &randfill (Matrix<T> &res);
template <class T> Matrix<T> &randnfill(Matrix<T> &res);

inline Matrix<double> &ltfill(Matrix<double> &lhsres, const Matrix<double> &rhs);
inline Matrix<double> &gtfill(Matrix<double> &lhsres, const Matrix<double> &rhs);
inline Matrix<double> &lefill(Matrix<double> &lhsres, const Matrix<double> &rhs);
inline Matrix<double> &gefill(Matrix<double> &lhsres, const Matrix<double> &rhs);



// Mathematical operator overloading
//
// NB: in general it is wise to avoid use of non-assignment operators (ie.
//     those which do not return a reference) as there may be a
//     computational hit when constructors (and possibly copy constructors)
//     are called.
//
// + posation - unary, return rvalue
// - negation - unary, return rvalue

template <class T> Matrix<T>  operator+(const Matrix<T> &left_op);
template <class T> Matrix<T>  operator-(const Matrix<T> &left_op);

// + addition    - binary, return rvalue
// - subtraction - binary, return rvalue
//
// NB: adding a scalar to a matrix adds the scalar to all elements of the
//     matrix.

template <class T> Matrix<T>  operator+ (const Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> Matrix<T>  operator+ (const Matrix<T> &left_op, const T         &right_op);
template <class T> Matrix<T>  operator+ (const T         &left_op, const Matrix<T> &right_op);
template <class T> Matrix<T>  operator- (const Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> Matrix<T>  operator- (const Matrix<T> &left_op, const T         &right_op);
template <class T> Matrix<T>  operator- (const T         &left_op, const Matrix<T> &right_op);

// += additive    assignment - binary, return lvalue
// -= subtractive assignment - binary, return lvalue

template <class T> Matrix<T> &operator+=(      Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> Matrix<T> &operator+=(      Matrix<T> &left_op, const T         &right_op);
template <class T> Matrix<T> &operator-=(      Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> Matrix<T> &operator-=(      Matrix<T> &left_op, const T         &right_op);

// * multiplication - binary, return rvalue
//
// NB: if A,B are matrices, b a vector and c a scalar
//     A*B = A.B  is standard matrix-matrix multiplication
//     A*b = A.b  is standard matrix-vector multiplication
//     A*c = A.c  is standard matrix-scalar multiplication
//     b*A = b'.A is standard matrix-vector multiplication, where b' is the conjugate transpose
//     c*A = c.A  is standard matrix-scalar multiplication

template <         class T> Matrix<T>  operator* (const Matrix<T> &left_op, const Matrix<T> &right_op);
template <class S, class T> Vector<S>  operator* (const Vector<S> &left_op, const Matrix<T> &right_op);
template <         class T> Matrix<T>  operator* (const Matrix<T> &left_op, const T         &right_op);
template <class S, class T> Vector<S>  operator* (const Matrix<T> &left_op, const Vector<S> &right_op);
template <         class T> Matrix<T>  operator* (const T         &left_op, const Matrix<T> &right_op);

// *= multiplicative assignment - binary, return lvalue
//
// NB: if A,B are matrices, b a vector and c a scalar
//     A *= B sets A  := A*B  and returns a reference to A
//     A *= c sets A  := A*c  and returns a reference to A
//     b *= A sets b' := b'*A and returns a reference to b
//
// leftmult:  sets A = A*B and returns a reference to A
// rightmult: sets B = A*B and returns a reference to B
//
// mult: A = B*C

template <         class T> Matrix<T> &operator*=(      Matrix<T> &left_op, const Matrix<T> &right_op);
template <class S, class T> Vector<S> &operator*=(      Vector<S> &left_op, const Matrix<T> &right_op);
template <         class T> Matrix<T> &operator*=(      Matrix<T> &left_op, const T         &right_op);

template <         class T> Matrix<T> &leftmult (      Matrix<T> &left_op, const Matrix<T> &right_op);
template <class S, class T> Vector<S> &leftmult (      Vector<S> &left_op, const Matrix<T> &right_op);
template <         class T> Matrix<T> &leftmult (      Matrix<T> &left_op, const T         &right_op);
template <         class T> Matrix<T> &rightmult(const Matrix<T> &left_op,       Matrix<T> &right_op);
template <class S, class T> Vector<S> &rightmult(const Matrix<T> &left_op,       Vector<S> &right_op);
template <         class T> Matrix<T> &rightmult(const T         &left_op,       Matrix<T> &right_op);

template <class T> Matrix<T> &mult(Matrix<T> &A, const Matrix<T> &B, const Matrix<T> &C);
template <class T> Vector<T> &mult(Vector<T> &a, const Vector<T> &b, const Matrix<T> &C);
template <class T> Vector<T> &mult(Vector<T> &a, const Matrix<T> &B, const Vector<T> &c);

template <class T> Vector<T> &multtrans(Vector<T> &a, const Vector<T> &b, const Matrix<T> &C);
template <class T> Vector<T> &multtrans(Vector<T> &a, const Matrix<T> &B, const Vector<T> &c);

// Relational operator overloading

template <class T> int operator==(const Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> int operator==(const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator==(const T         &left_op, const Matrix<T> &right_op);

template <class T> int operator!=(const Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> int operator!=(const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator!=(const T         &left_op, const Matrix<T> &right_op);

template <class T> int operator< (const Matrix<T> &left_op, const T &right_op);
template <class T> int operator< (const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator< (const T         &left_op, const Matrix<T> &right_op);

template <class T> int operator<=(const Matrix<T> &left_op, const T &right_op);
template <class T> int operator<=(const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator<=(const T         &left_op, const Matrix<T> &right_op);

template <class T> int operator> (const Matrix<T> &left_op, const T &right_op);
template <class T> int operator> (const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator> (const T         &left_op, const Matrix<T> &right_op);

template <class T> int operator>=(const Matrix<T> &left_op, const T &right_op);
template <class T> int operator>=(const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator>=(const T         &left_op, const Matrix<T> &right_op);


// Stream operators

template <class T> std::ostream &operator<<(std::ostream &output, const Matrix<T> &src );
template <class T> std::istream &operator>>(std::istream &input,        Matrix<T> &dest);

template <class T> inline std::istream &streamItIn(std::istream &input, Matrix<T>& dest, int processxyzvw = 1);
template <class T> inline std::ostream &streamItOut(std::ostream &output, const Matrix<T>& src, int retainTypeMarker = 0);






/*
      subroutine tql1(n,d,e,ierr)
c
      integer i,j,l,m,n,ii,l1,l2,mml,ierr
      double precision d(n),e(n)
      double precision c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2,pythag
c
c     this subroutine is a translation of the algol procedure tql1,
c     num. math. 11, 293-306(1968) by bowdler, martin, reinsch, and
c     wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 227-240(1971).
c
c     this subroutine finds the eigenvalues of a symmetric
c     tridiagonal matrix by the ql method.
c
c     on input
c
c        n is the order of the matrix.
c
c        d contains the diagonal elements of the input matrix.
c
c        e contains the subdiagonal elements of the input matrix
c          in its last n-1 positions.  e(1) is arbitrary.
c
c      on output
c
c        d contains the eigenvalues in ascending order.  if an
c          error exit is made, the eigenvalues are correct and
c          ordered for indices 1,2,...ierr-1, but may not be
c          the smallest eigenvalues.
c
c        e has been destroyed.
c
c        ierr is set to
c          zero       for normal return,
c          j          if the j-th eigenvalue has not been
c                     determined after 30 iterations.
c
c     calls pythag for  dsqrt(a*a + b*b) .
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
c
*/

template <class T>
int tql1(Vector<T> &d, Vector<T> &e);

/*
      subroutine tql2(nm,n,d,e,z,ierr)
c
      integer i,j,k,l,m,n,ii,l1,l2,nm,mml,ierr
      double precision d(n),e(n),z(nm,n)
      double precision c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2,pythag
c
c     this subroutine is a translation of the algol procedure tql2,
c     num. math. 11, 293-306(1968) by bowdler, martin, reinsch, and
c     wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 227-240(1971).
c
c     this subroutine finds the eigenvalues and eigenvectors
c     of a symmetric tridiagonal matrix by the ql method.
c     the eigenvectors of a full symmetric matrix can also
c     be found if  tred2  has been used to reduce this
c     full matrix to tridiagonal form.
c
c     on input
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement.
c
c        n is the order of the matrix.
c
c        d contains the diagonal elements of the input matrix.
c
c        e contains the subdiagonal elements of the input matrix
c          in its last n-1 positions.  e(1) is arbitrary.
c
c        z contains the transformation matrix produced in the
c          reduction by  tred2, if performed.  if the eigenvectors
c          of the tridiagonal matrix are desired, z must contain
c          the identity matrix.
c
c      on output
c
c        d contains the eigenvalues in ascending order.  if an
c          error exit is made, the eigenvalues are correct but
c          unordered for indices 1,2,...,ierr-1.
c
c        e has been destroyed.
c
c        z contains orthonormal eigenvectors of the symmetric
c          tridiagonal (or full) matrix.  if an error exit is made,
c          z contains the eigenvectors associated with the stored
c          eigenvalues.
c
c        ierr is set to
c          zero       for normal return,
c          j          if the j-th eigenvalue has not been
c                     determined after 30 iterations.
c
c     calls pythag for  dsqrt(a*a + b*b) .
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
c
*/

//int tql2(int nm, int n, Vector<double> &d, Vector<double> &e, Matrix<double> &z);
template <class T>
inline int tql2(Vector<T> &d, Vector<T> &e, Matrix<T> &z);








































template <class T>
retMatrix<T> &retMatrix<T>::reset(void)
{
    Matrix<T> &gimme = *this;

    if ( !(gimme.pbaseRow) && gimme.pivotRow )
    {
        MEMDEL(gimme.pivotRow);
        gimme.pivotRow = NULL;
    }

    if ( !(gimme.pbaseCol) && gimme.pivotCol )
    {
        MEMDEL(gimme.pivotCol);
        gimme.pivotCol = NULL;
    }

    gimme.matbuff = NULL;

    gimme.dnumRows = 0;
    gimme.dnumCols = 0;

    gimme.nbase    = 0;
    gimme.pbaseRow = 1;
    gimme.pbaseCol = 1;

    gimme.iibRow = 0;
    gimme.iisRow = 0;

    gimme.iibCol = 0;
    gimme.iisCol = 0;

    gimme.bkref    = NULL;
    gimme.content  = NULL;
    gimme.ccontent = NULL;
    gimme.pivotRow = NULL;
    gimme.pivotCol = NULL;

    gimme.iscover = 0;
    gimme.elmfn   = NULL;
    gimme.rowfn   = NULL;
    gimme.delfn   = NULL;
    gimme.dref    = NULL;
    gimme.celmfn  = NULL;
    gimme.crowfn  = NULL;
    gimme.cdelfn  = NULL;
    gimme.cdref   = NULL;

    return *this;
}


// Constructors and Destructors

template <class T>
Matrix<T>::Matrix(int numRows, int numCols)
{
    thisindirect[0] = this;

    matbuff = NULL;

    int i;

    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );

    dnumRows = numRows;
    dnumCols = numCols;

    nbase    = 0;
    pbaseRow = 1;
    pbaseCol = 1;

    iibRow = 0;
    iisRow = 1;

    iibCol = 0;
    iisCol = 1;

    bkref    = this;
    MEMNEW(content,Vector<Vector<T> >(dnumRows));
    ccontent = content;
    pivotRow = cntintarray(dnumRows);
    pivotCol = cntintarray(dnumCols);

    NiceAssert( content );

    if ( dnumRows )
    {
	for ( i = 0 ; i < dnumRows ; i++ )
	{
	    (*content).add(i);
            (*content)("&",i).resize(dnumCols);
	}
    }

    iscover = 0;
    elmfn   = NULL;
    rowfn   = NULL;
    delfn   = NULL;
    dref    = NULL;
    celmfn  = NULL;
    crowfn  = NULL;
    cdelfn  = NULL;
    cdref   = NULL;

    return;
}

template <class T>
Matrix<T>::Matrix(const T &(*celm)(int, int, const void *), const Vector<T> &(*crow)(int, const void *), const void *cxdref, int numRows, int numCols, void (*xcdelfn)(const void *, void *), void *xdref)
{
    thisindirect[0] = this;

    matbuff = NULL;

    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );

    dnumRows = numRows;
    dnumCols = numCols;

    nbase    = 0;
    pbaseRow = 1;
    pbaseCol = 1;

    iibRow = 0;
    iisRow = 1;

    iibCol = 0;
    iisCol = 1;

    bkref    = this;
    content  = NULL;
    ccontent = NULL;
    pivotRow = cntintarray(dnumRows);
    pivotCol = cntintarray(dnumCols);

    iscover = 1;
    elmfn   = NULL;
    rowfn   = NULL;
    delfn   = NULL;
    dref    = xdref;
    celmfn  = celm;
    crowfn  = crow;
    cdelfn  = xcdelfn;
    cdref   = cxdref;

    return;
}

template <class T>
Matrix<T>::Matrix(T &(*elm)(int, int, void *), Vector<T> &(*row)(int, void *), void *xdref, const T &(*celm)(int, int, const void *), const Vector<T> &(*crow)(int, const void *), const void *cxdref, int numRows, int numCols, void (*xdelfn)(void *), void (*xcdelfn)(const void *, void *))
{
    thisindirect[0] = this;

    matbuff = NULL;

    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );

    dnumRows = numRows;
    dnumCols = numCols;

    nbase    = 0;
    pbaseRow = 1;
    pbaseCol = 1;

    iibRow = 0;
    iisRow = 1;

    iibCol = 0;
    iisCol = 1;

    bkref    = this;
    content  = NULL;
    ccontent = NULL;
    pivotRow = cntintarray(dnumRows);
    pivotCol = cntintarray(dnumCols);

    iscover = 1;
    elmfn   = elm;
    rowfn   = row;
    delfn   = xdelfn;
    dref    = xdref;
    celmfn  = celm;
    crowfn  = crow;
    cdelfn  = xcdelfn;
    cdref   = cxdref;

    return;
}

template <class T>
Matrix<T>::Matrix(const Matrix<T> &src)
{
    thisindirect[0] = this;

    matbuff = NULL;

    dnumRows = 0;
    dnumCols = 0;

    nbase    = 0;
    pbaseRow = 1;
    pbaseCol = 1;

    iibRow = 0;
    iisRow = 1;

    iibCol = 0;
    iisCol = 1;

    bkref    = this;
    MEMNEW(content,Vector<Vector<T> >);
    ccontent = content;
    pivotRow = cntintarray(0);
    pivotCol = cntintarray(0);

    NiceAssert( content );

    iscover = 0;
    elmfn   = NULL;
    rowfn   = NULL;
    delfn   = NULL;
    dref    = NULL;
    celmfn  = NULL;
    crowfn  = NULL;
    cdelfn  = NULL;
    cdref   = NULL;

    *this = src;

    return;
}

template <class T>
Matrix<T>::Matrix(const char *dummy, const Matrix<T> &src)
{
    thisindirect[0] = this;

    matbuff = NULL;

    (void) dummy;

    dnumRows = 0;
    dnumCols = 0;

    nbase    = 0;
    pbaseRow = 1;
    pbaseCol = 1;

    iibRow = 0;
    iisRow = 0;

    iibCol = 0;
    iisCol = 0;

    bkref    = src.bkref;
    content  = NULL;
    ccontent = NULL;
    pivotRow = NULL;
    pivotCol = NULL;

    iscover = 0;
    elmfn   = NULL;
    rowfn   = NULL;
    delfn   = NULL;
    dref    = NULL;
    celmfn  = NULL;
    crowfn  = NULL;
    cdelfn  = NULL;
    cdref   = NULL;

    return;
}

template <class T>
Matrix<T>::Matrix(const char *dummy)
{
    thisindirect[0] = this;

    matbuff = NULL;

    (void) dummy;

    dnumRows = 0;
    dnumCols = 0;

    nbase    = 0;
    pbaseRow = 1;
    pbaseCol = 1;

    iibRow = 0;
    iisRow = 0;

    iibCol = 0;
    iisCol = 0;

    bkref    = NULL; // NEED TO FILL THIS IN src.bkref;
    content  = NULL;
    ccontent = NULL;
    pivotRow = NULL;
    pivotCol = NULL;

    iscover = 0;
    elmfn   = NULL;
    rowfn   = NULL;
    delfn   = NULL;
    dref    = NULL;
    celmfn  = NULL;
    crowfn  = NULL;
    cdelfn  = NULL;
    cdref   = NULL;

    return;
}

template <class T>
Matrix<T>::~Matrix()
{
    if ( matbuff )
    {
        MEMDEL(matbuff);
        matbuff = NULL;
    }

    if ( !nbase && !iscover && content )
    {
	MEMDEL(content);
        content = NULL;
    }

    if ( !nbase && iscover )
    {
	if ( delfn )
	{
	    delfn(dref);
	}

	if ( cdelfn )
	{
	    cdelfn(cdref,dref);
	}
    }

    if ( !pbaseRow && pivotRow )
    {
        MEMDEL(pivotRow);
        pivotRow = NULL;
    }

    if ( !pbaseCol && pivotCol )
    {
        MEMDEL(pivotCol);
        pivotCol = NULL;
    }

    return;
}




// Assignment

template <class T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &src)
{
    if ( shareBase(&src) )
    {
	Matrix<T> temp(src);

	*this = temp;
    }

    else
    {
	int srcnumRows = src.numRows();
	int srcnumCols = src.numCols();
	int i;

	// Fix size if this is not a reference

	if ( !nbase )
	{
	    resize(srcnumRows,srcnumCols);
	}

        NiceAssert( dnumRows == srcnumRows );
        NiceAssert( dnumCols == srcnumCols );

        retVector<T> tmpva;
        retVector<T> tmpvb;

	if ( dnumRows && dnumCols )
	{
	    for ( i = 0 ; i < dnumRows ; i++ )
	    {
		(*this)("&",i,"&",tmpva) = src(i,tmpvb);
	    }
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const Vector<T> &src)
{
    if ( ( dnumRows == 1 ) && ( dnumCols == src.size() ) )
    {
	int i;

	if ( src.size() )
	{
	    for ( i = 0 ; i < src.size() ; i++ )
	    {
		(*this)("&",zeroint(),i) = src(i);
	    }
	}
    }

    else if ( ( dnumRows == src.size() ) && ( dnumCols == 1 ) )
    {
	int i;

	if ( src.size() )
	{
	    for ( i = 0 ; i < src.size() ; i++ )
	    {
		(*this)("&",i,zeroint()) = src(i);
	    }
	}
    }

    else
    {
        NiceAssert( !nbase );

	resize(src.size(),1);

        *this = src;
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::operator=(const Matrix<T> &src)
{
    NiceAssert( !infsize() );
    NiceAssert( ( src.numRows() == 1 ) || ( src.numCols() == 1 ) );

    if ( src.numRows() == 1 )
    {
	// Row vector

	int srcsize = src.numCols();
	int i;

	if ( !nbase )
	{
	    resize(srcsize);
	}

        NiceAssert( dsize == srcsize );

	if ( dsize )
	{
	    for ( i = 0 ; i < dsize ; i++ )
	    {
		(*this)("&",i) = src(0,i);
	    }
	}
    }

    else
    {
	// Column vector

	int srcsize = src.numRows();
	int i;

	if ( !nbase )
	{
	    resize(srcsize);
	}

        NiceAssert( dsize == srcsize );

	if ( dsize )
	{
	    for ( i = 0 ; i < dsize ; i++ )
	    {
		(*this)("&",i) = src(i,0);
	    }
	}
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::operator=(const Matrix<T> &src)
{
    NiceAssert( ( src.numRows() == 1 ) || ( src.numCols() == 1 ) );

    zero();

    if ( src.numRows() == 1 )
    {
	// Row vector

	int i;

        if ( size() )
	{
            retVector<T> tmpva;

            for ( i = 0 ; i < size() ; i++ )
	    {
		(*this)("&",i,tmpva) = src(0,i);
	    }
	}
    }

    else
    {
	// Column vector

	int i;

        if ( size() )
	{
            retVector<T> tmpva;

            for ( i = 0 ; i < size() ; i++ )
	    {
                (*this)("&",i,tmpva) = src(i,0);
	    }
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const T &src)
{
    int i,j;

    // Copy over

    if ( dnumRows && dnumCols )
    {
	for ( i = 0 ; i < dnumRows ; i++ )
	{
	    for ( j = 0 ; j < dnumCols ; j++ )
	    {
		(*this)("&",i,j) = src;
	    }
	}
    }

    return *this;
}


// Access:

template <class T>
T &Matrix<T>::operator()(const char *dummy, int i, int j)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( j >= 0 );
    NiceAssert( j < dnumCols );
    NiceAssert( dummy[0] );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    if ( iscover )
    {
        NiceAssert( elmfn );

        return (*elmfn)((*pivotRow)(iibRow+(i*iisRow)),(*pivotCol)(iibCol+(j*iisCol)),dref);
    }

    NiceAssert( content );

    return ((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow))))(dummy,(*pivotCol)(iibCol+(j*iisCol)));
}

template <class T>
const T &Matrix<T>::operator()(int i, int j) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( j >= 0 );
    NiceAssert( j < dnumCols );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    if ( iscover )
    {
        NiceAssert( celmfn );

        return (*celmfn)((*pivotRow)(iibRow+(i*iisRow)),(*pivotCol)(iibCol+(j*iisCol)),cdref);
    }

    NiceAssert( ccontent );

    return ((*ccontent)((*pivotRow)(iibRow+(i*iisRow))))((*pivotCol)(iibCol+(j*iisCol)));
}

template <class T>
Vector<T> &Matrix<T>::operator()(const char *dummy, int i, int jb, int js, int jm, retVector<T> &res)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( jb >= 0 );
    NiceAssert( js > 0 );
    NiceAssert( jm < dnumCols );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    res.reset();

    if ( iscover )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
	    return ((*rowfn)((*((**thisindirect).pivotRow))(iibRow+(i*iisRow)),dref))(dummy,jb,js,jm,res);
	}

	//return (((*rowfn)((*((**thisindirect).pivotRow))(iibRow+(i*iisRow)),dref))(dummy,*((**thisindirect).pivotCol),iibCol,iisCol,iibCol+((dnumCols-1)*iisCol)))(jb,js,jm,res);

        int ressize = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

        int iib0 = iibCol;
        int iis0 = iisCol;
        //int iim0 = iibCol+((dnumCols-1)*iisCol);

        int iib1 = jb;
        int iis1 = js;
        //int iim1 = jm;

        int iib = iib0+(iis0*iib1);
        int iis = iis0*iis1;
        int iim = iib+((ressize-1)*iis);

	return ((*rowfn)((*((**thisindirect).pivotRow))(iibRow+(i*iisRow)),dref))(dummy,*((**thisindirect).pivotCol),iib,iis,iim,res);
    }

    NiceAssert( content );

    if ( !nbase )
    {
	return ((*content)(dummy,(*((**thisindirect).pivotRow))(iibRow+(i*iisRow))))(dummy,jb,js,jm,res);
    }

    //return (((*ccontent)((*pivotRow)(iibRow+(i*iisRow))))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol)))(jb,js,jm,res);

    int ressize = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

    int iib0 = iibCol;
    int iis0 = iisCol;
    //int iim0 = iibCol+((dnumCols-1)*iisCol);

    int iib1 = jb;
    int iis1 = js;
    //int iim1 = jm;

    int iib = iib0+(iis0*iib1);
    int iis = iis0*iis1;
    int iim = iib+((ressize-1)*iis);

    return ((*content)(dummy,(*((**thisindirect).pivotRow))(iibRow+(i*iisRow))))(dummy,*((**thisindirect).pivotCol),iib,iis,iim,res);
}

template <class T>
const Vector<T> &Matrix<T>::operator()(int i, int jb, int js, int jm, retVector<T> &res) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( jb >= 0 );
    NiceAssert( js > 0 );
    NiceAssert( jm < dnumCols );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    res.reset();

    if ( iscover )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
	    return ((*crowfn)((*pivotRow)(iibRow+(i*iisRow)),cdref))(jb,js,jm,res);
	}

	//return (((*crowfn)((*pivotRow)(iibRow+(i*iisRow)),cdref))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol)))(jb,js,jm,res);

        int ressize = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

        int iib0 = iibCol;
        int iis0 = iisCol;
        //int iim0 = iibCol+((dnumCols-1)*iisCol);

        int iib1 = jb;
        int iis1 = js;
        //int iim1 = jm;

        int iib = iib0+(iis0*iib1);
        int iis = iis0*iis1;
        int iim = iib+((ressize-1)*iis);

	return ((*crowfn)((*pivotRow)(iibRow+(i*iisRow)),cdref))(*pivotCol,iib,iis,iim,res);
    }

    NiceAssert( ccontent );

    if ( !nbase )
    {
	return ((*ccontent)((*pivotRow)(iibRow+(i*iisRow))))(jb,js,jm,res);
    }

    //return (((*ccontent)((*pivotRow)(iibRow+(i*iisRow))))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol)))(jb,js,jm,res);

    int ressize = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

    int iib0 = iibCol;
    int iis0 = iisCol;
    //int iim0 = iibCol+((dnumCols-1)*iisCol);

    int iib1 = jb;
    int iis1 = js;
    //int iim1 = jm;

    int iib = iib0+(iis0*iib1);
    int iis = iis0*iis1;
    int iim = iib+((ressize-1)*iis);

    return ((*ccontent)((*pivotRow)(iibRow+(i*iisRow))))(*pivotCol,iib,iis,iim,res);
}

template <class T>
Vector<T> &Matrix<T>::operator()(const char *dummy, int i, const Vector<int> &j, retVector<T> &tmp, retVector<T> &res)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( checkRange(0,dnumCols-1,j) );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    res.reset();
    tmp.reset();

    if ( iscover )
    {
        NiceAssert( rowfn );

	if ( !nbase )
	{
	    return ((*rowfn)((*pivotRow)(iibRow+(i*iisRow)),dref))(dummy,j,res);
	}

	return (((*rowfn)((*pivotRow)(iibRow+(i*iisRow)),dref))(dummy,*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(dummy,j,res);
    }

    NiceAssert( content );

    if ( !nbase )
    {
	return ((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow))))(dummy,j,res);
    }

    return (((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(dummy,j,res);
}

template <class T>
const Vector<T> &Matrix<T>::operator()(int i, const Vector<int> &j, retVector<T> &tmp, retVector<T> &res) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( checkRange(0,dnumCols-1,j) );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    res.reset();
    tmp.reset();

    if ( iscover )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
	    return ((*crowfn)((*pivotRow)(iibRow+(i*iisRow)),cdref))(j,res);
	}

	return (((*crowfn)((*pivotRow)(iibRow+(i*iisRow)),cdref))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(j,res);
    }

    NiceAssert( ccontent );

    if ( !nbase )
    {
	return ((*ccontent)((*pivotRow)(iibRow+(i*iisRow))))(j,res);
    }

    return (((*ccontent)((*pivotRow)(iibRow+(i*iisRow))))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(j,res);
}

//template <class T>
//Vector<T> &Matrix<T>::operator()(const char *dummy, int i, retVector<T> &res)
//{
//    return (*this)(dummy,i,0,1,dnumCols-1,res);
//}
//    NiceAssert( i >= 0 );
//    NiceAssert( i < dnumRows );
//    NiceAssert( pivotRow );
//    NiceAssert( pivotCol );
//
//    if ( iscover )
//    {
//      NiceAssert( rowfn );
//
//	if ( !nbase )
//	{
//	    return (*rowfn)((*pivotRow)(iibRow+(i*iisRow)),dref);
//	}
//
//	return (((*rowfn)((*pivotRow)(iibRow+(i*iisRow)),dref))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol));
//    }
//
//    NiceAssert( content );
//
//    if ( !nbase )
//    {
//// problem X at end	((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow)))).fixsize = 1;
//
//	return (*content)(dummy,(*pivotRow)(iibRow+(i*iisRow)));
//    }
//
////problem X at end    ((((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol))).fixsize = 1;
//
//    return (((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol));
//
//
// Problem X: when you refer to the row of a matrix, ie
//
// G(i) or G("&",i)
//
// then we should set fixsize on that row to protect it from being
// resized, thereby messing up the matrix (one row is too long or too
// short and doesn't match dnumCols).  However if you do set fixsize
// then *it stays set*.  So when you come to add or remove a row, or
// Alistair Shilton 2014 (c) wrote this code
// shuffle rows, the required resize call within that will assert
// that !fixsize and then fail via a throw.
//
// This can come up with the following code:
//
// G("&",i) = whatever;
// G.addRow(0);
//
// The problem is that if we try to record if/what fixsize in contents
// have been set and reset them when required then we end up stuck in a
// constant thrash of resetting them, which is a waste of computational
// time.  The alternative is to return a child of the vector, but this
// is computationally even worse.
//
// Current workaround: don't set fixsize, assume that the user doesn't
// do anything suicidal.
//
// Possible solution: have a "fixsize set" bit and a vector of integers
// for which vectors fixsize is set for.  Whenever a call to add, remove,
// resize, shuffle etc is made if this bit is set then all of the
// relevant fixsize's are reset and the bit cleared.
//
// SOLUTION: rather than use fixsize, return a child.  It's a bit slower,
// but will prevent add/remove being called successfully without resorting
// to fixsize.  Hence the commenting out and replacement of the above code.
//}

template <class T>
Vector<T> &Matrix<T>::operator()(const char *dummy, int i, const char *dummyb, retVector<T> &tempdonotassume)
{
    // This version is like the above, but doesn't care about problem X as it
    // is for internal use only "&"), so we can use it for speed where we
    // know that it will be safe.

    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    tempdonotassume.reset();

    if ( iscover )
    {
        NiceAssert( rowfn );

	if ( !nbase )
	{
	    return (*rowfn)((*pivotRow)(iibRow+(i*iisRow)),dref);
	}

        int iib = iibCol;
        int iis = iisCol;
        int iim = iibCol+((dnumCols-1)*iisCol);

	return  ((*rowfn)((*pivotRow)(iibRow+(i*iisRow)),dref))(dummy,*pivotCol,iib,iis,iim,tempdonotassume);

	//return (((*rowfn)((*pivotRow)(iibRow+(i*iisRow)),dref))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tempdonotassume);
    }

    NiceAssert( content );

    if ( !nbase )
    {
// problem X at end	((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow)))).fixsize = 1;
	return (*content)(dummy,(*pivotRow)(iibRow+(i*iisRow)));
    }

    int iib = iibCol;
    int iis = iisCol;
    int iim = iibCol+((dnumCols-1)*iisCol);

    return ((*content)(dummyb,(*pivotRow)(iibRow+(i*iisRow))))(dummy,*pivotCol,iib,iis,iim,tempdonotassume);

////problem X at end    ((((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol))).fixsize = 1;
//    return (((*content)(dummyb,(*pivotRow)(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tempdonotassume);
}

template <class T>
Matrix<T> &Matrix<T>::operator()(const char *dummy, int ib, int is, int im, int jb, int js, int jm, retMatrix<T> &res)
{
    NiceAssert( ib >= 0 );
    NiceAssert( is > 0 );
    NiceAssert( im < dnumRows );
    NiceAssert( jb >= 0 );
    NiceAssert( js > 0 );
    NiceAssert( jm < dnumCols );

    res.reset();

    (void) dummy;

    res.bkref = bkref;

    res.dnumRows = ( ib > im ) ? 0 : ( ((im-ib)/is)+1 );

    res.nbase    = 1;
    res.pbaseRow = 1;

    res.iibRow = iibRow+(iisRow*ib);
    res.iisRow = iisRow*is;

    res.content  = content;
    res.ccontent = ccontent;
    res.pivotRow = pivotRow;

    res.iscover = iscover;
    res.elmfn   = elmfn;
    res.rowfn   = rowfn;
    res.delfn   = delfn;
    res.dref    = dref;
    res.celmfn  = celmfn;
    res.crowfn  = crowfn;
    res.cdelfn  = cdelfn;
    res.cdref   = cdref;

    res.dnumCols = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

    res.nbase    = 1;
    res.pbaseCol = 1;

    res.iibCol = iibCol+(iisCol*jb);
    res.iisCol = iisCol*js;

    res.content  = content;
    res.ccontent = ccontent;
    res.pivotCol = pivotCol;

    res.iscover = iscover;
    res.elmfn   = elmfn;
    res.rowfn   = rowfn;
    res.delfn   = delfn;
    res.dref    = dref;
    res.celmfn  = celmfn;
    res.crowfn  = crowfn;
    res.cdelfn  = cdelfn;
    res.cdref   = cdref;

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(int ib, int is, int im, int jb, int js, int jm, retMatrix<T> &res) const
{
    NiceAssert( ib >= 0 );
    NiceAssert( is > 0 );
    NiceAssert( im < dnumRows );
    NiceAssert( jb >= 0 );
    NiceAssert( js > 0 );
    NiceAssert( jm < dnumCols );

    res.reset();

    res.bkref = bkref;

    res.dnumRows = ( ib > im ) ? 0 : ( ((im-ib)/is)+1 );

    res.nbase    = 1;
    res.pbaseRow = 1;

    res.iibRow = iibRow+(iisRow*ib);
    res.iisRow = iisRow*is;

    res.content  = NULL;
    res.ccontent = ccontent;
    res.pivotRow = pivotRow;

    res.iscover = iscover;
    res.elmfn   = NULL;
    res.rowfn   = NULL;
    res.delfn   = NULL;
    res.dref    = NULL;
    res.celmfn  = celmfn;
    res.crowfn  = crowfn;
    res.cdelfn  = cdelfn;
    res.cdref   = cdref;

    res.dnumCols = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

    res.nbase    = 1;
    res.pbaseCol = 1;

    res.iibCol = iibCol+(iisCol*jb);
    res.iisCol = iisCol*js;

    res.content  = NULL;
    res.ccontent = ccontent;
    res.pivotCol = pivotCol;

    res.iscover = iscover;
    res.elmfn   = NULL;
    res.rowfn   = NULL;
    res.delfn   = NULL;
    res.dref    = NULL;
    res.celmfn  = celmfn;
    res.crowfn  = crowfn;
    res.cdelfn  = cdelfn;
    res.cdref   = cdref;

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::operator()(const char *dummy, int ib, int is, int im, const Vector<int> &j, retMatrix<T> &res)
{
    NiceAssert( ib >= 0 );
    NiceAssert( is > 0 );
    NiceAssert( im < dnumRows );
    NiceAssert( checkRange(0,dnumCols-1,j) );

    res.reset();

    (void) dummy;

    res.bkref = bkref;

    res.dnumRows = ( ib > im ) ? 0 : ( ((im-ib)/is)+1 );

    res.nbase    = 1;
    res.pbaseRow = 1;

    res.iibRow = iibRow+(iisRow*ib);
    res.iisRow = iisRow*is;

    res.content  = content;
    res.ccontent = ccontent;
    res.pivotRow = pivotRow;

    res.iscover = iscover;
    res.elmfn   = elmfn;
    res.rowfn   = rowfn;
    res.delfn   = delfn;
    res.dref    = dref;
    res.celmfn  = celmfn;
    res.crowfn  = crowfn;
    res.cdelfn  = cdelfn;
    res.cdref   = cdref;

    if ( !nbase && !(j.base()) )
    {
	res.dnumCols = j.size();

	res.nbase    = 1;
	res.pbaseCol = 1;

	res.iibCol = 0;
	res.iisCol = 1;

	res.content  = content;
	res.ccontent = ccontent;

	res.iscover = iscover;
        res.elmfn   = elmfn;
	res.rowfn   = rowfn;
	res.delfn   = delfn;
	res.dref    = dref;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;

	if ( res.dnumCols )
	{
            res.pivotCol = &(j.grabcontentdirect());
	}

	else
	{
            res.pivotCol = cntintarray(0);
	}
    }

    else
    {
	res.dnumCols = j.size();

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dnumCols));

        NiceAssert( temppivot );

	if ( res.dnumCols )
	{
	    int jj;

	    for ( jj = 0 ; jj < res.dnumCols ; jj++ )
	    {
		(*temppivot)("&",jj) = (*pivotCol)(iibCol+(iisCol*(j(jj))));
	    }
	}

	res.nbase    = 1;
	res.pbaseCol = 0;

	res.iibCol   = 0;
	res.iisCol   = 1;

	res.content  = content;
	res.ccontent = ccontent;
        res.pivotCol = temppivot;

	res.iscover = iscover;
        res.elmfn   = elmfn;
	res.rowfn   = rowfn;
	res.delfn   = delfn;
	res.dref    = dref;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;
    }

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(int ib, int is, int im, const Vector<int> &j, retMatrix<T> &res) const
{
    NiceAssert( ib >= 0 );
    NiceAssert( is > 0 );
    NiceAssert( im < dnumRows );
    NiceAssert( checkRange(0,dnumCols-1,j) );

    res.reset();

    res.bkref = bkref;

    res.dnumRows = ( ib > im ) ? 0 : ( ((im-ib)/is)+1 );

    res.nbase    = 1;
    res.pbaseRow = 1;

    res.iibRow = iibRow+(iisRow*ib);
    res.iisRow = iisRow*is;

    res.content  = NULL;
    res.ccontent = ccontent;
    res.pivotRow = pivotRow;

    res.iscover = iscover;
    res.elmfn   = NULL;
    res.rowfn   = NULL;
    res.delfn   = NULL;
    res.dref    = NULL;
    res.celmfn  = celmfn;
    res.crowfn  = crowfn;
    res.cdelfn  = cdelfn;
    res.cdref   = cdref;

    if ( !nbase && !(j.base()) )
    {
	res.dnumCols = j.size();

	res.nbase    = 1;
	res.pbaseCol = 1;

	res.iibCol = 0;
	res.iisCol = 1;

	res.content  = NULL;
	res.ccontent = ccontent;

	res.iscover = iscover;
        res.elmfn   = NULL;
	res.rowfn   = NULL;
	res.delfn   = NULL;
	res.dref    = NULL;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;

	if ( res.dnumCols )
	{
            res.pivotCol = &(j.grabcontentdirect());
	}

	else
	{
            res.pivotCol = cntintarray(0);
	}
    }

    else
    {
	res.dnumCols = j.size();

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dnumCols));

        NiceAssert( temppivot );

	if ( res.dnumCols )
	{
	    int jj;

	    for ( jj = 0 ; jj < res.dnumCols ; jj++ )
	    {
		(*temppivot)("&",jj) = (*pivotCol)(iibCol+(iisCol*(j(jj))));
	    }
	}

	res.nbase    = 1;
	res.pbaseCol = 0;

	res.iibCol   = 0;
	res.iisCol   = 1;

	res.content  = NULL;
	res.ccontent = ccontent;
        res.pivotCol = temppivot;

	res.iscover = iscover;
        res.elmfn   = NULL;
	res.rowfn   = NULL;
	res.delfn   = NULL;
	res.dref    = NULL;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;
    }

    return res;
}




template <class T>
Matrix<T> &Matrix<T>::operator()(const char *dummy, const Vector<int> &i, int jb, int js, int jm, retMatrix<T> &res)
{
    NiceAssert( checkRange(0,dnumRows-1,i) );
    NiceAssert( jb >= 0 );
    NiceAssert( js > 0 );
    NiceAssert( jm < dnumCols );

    res.reset();

    (void) dummy;

    res.bkref = bkref;

    if ( !nbase && !(i.base()) )
    {
	res.dnumRows = i.size();

	res.nbase    = 1;
	res.pbaseRow = 1;

	res.iibRow = 0;
	res.iisRow = 1;

	res.content  = content;
	res.ccontent = ccontent;

	res.iscover = iscover;
        res.elmfn   = elmfn;
	res.rowfn   = rowfn;
	res.delfn   = delfn;
	res.dref    = dref;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;

	if ( res.dnumRows )
	{
            res.pivotRow = &(i.grabcontentdirect());
	}

	else
	{
            res.pivotRow = cntintarray(0);
	}
    }

    else
    {
	res.dnumRows = i.size();

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dnumRows));

        NiceAssert( temppivot );

	if ( res.dnumRows )
	{
	    int ii;

	    for ( ii = 0 ; ii < res.dnumRows ; ii++ )
	    {
		(*temppivot)("&",ii) = (*pivotRow)(iibRow+(iisRow*(i(ii))));
	    }
	}

	res.nbase    = 1;
	res.pbaseRow = 0;

	res.iibRow   = 0;
	res.iisRow   = 1;

	res.content  = content;
	res.ccontent = ccontent;
        res.pivotRow = temppivot;

	res.iscover = iscover;
        res.elmfn   = elmfn;
	res.rowfn   = rowfn;
	res.delfn   = delfn;
	res.dref    = dref;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;
    }

    res.dnumCols = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

    res.nbase    = 1;
    res.pbaseCol = 1;

    res.iibCol = iibCol+(iisCol*jb);
    res.iisCol = iisCol*js;

    res.content  = content;
    res.ccontent = ccontent;
    res.pivotCol = pivotCol;

    res.iscover = iscover;
    res.elmfn   = elmfn;
    res.rowfn   = rowfn;
    res.delfn   = delfn;
    res.dref    = dref;
    res.celmfn  = celmfn;
    res.crowfn  = crowfn;
    res.cdelfn  = cdelfn;
    res.cdref   = cdref;

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(const Vector<int> &i, int jb, int js, int jm, retMatrix<T> &res) const
{
    NiceAssert( checkRange(0,dnumRows-1,i) );
    NiceAssert( jb >= 0 );
    NiceAssert( js > 0 );
    NiceAssert( jm < dnumCols );

    res.reset();

    res.bkref = bkref;

    if ( !nbase && !(i.base()) )
    {
	res.dnumRows = i.size();

	res.nbase    = 1;
	res.pbaseRow = 1;

	res.iibRow = 0;
	res.iisRow = 1;

	res.content  = NULL;
	res.ccontent = ccontent;

	res.iscover = iscover;
        res.elmfn   = NULL;
	res.rowfn   = NULL;
	res.delfn   = NULL;
	res.dref    = NULL;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;

	if ( res.dnumRows )
	{
            res.pivotRow = &(i.grabcontentdirect());
	}

	else
	{
            res.pivotRow = cntintarray(0);
	}
    }

    else
    {
	res.dnumRows = i.size();

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dnumRows));

        NiceAssert( temppivot );

	if ( res.dnumRows )
	{
	    int ii;

	    for ( ii = 0 ; ii < res.dnumRows ; ii++ )
	    {
		(*temppivot)("&",ii) = (*pivotRow)(iibRow+(iisRow*(i(ii))));
	    }
	}

	res.nbase    = 1;
	res.pbaseRow = 0;

	res.iibRow   = 0;
	res.iisRow   = 1;

	res.content  = NULL;
	res.ccontent = ccontent;
        res.pivotRow = temppivot;

	res.iscover = iscover;
        res.elmfn   = NULL;
	res.rowfn   = NULL;
	res.delfn   = NULL;
	res.dref    = NULL;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;
    }

    res.dnumCols = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

    res.nbase    = 1;
    res.pbaseCol = 1;

    res.iibCol = iibCol+(iisCol*jb);
    res.iisCol = iisCol*js;

    res.content  = NULL;
    res.ccontent = ccontent;
    res.pivotCol = pivotCol;

    res.iscover = iscover;
    res.elmfn   = NULL;
    res.rowfn   = NULL;
    res.delfn   = NULL;
    res.dref    = NULL;
    res.celmfn  = celmfn;
    res.crowfn  = crowfn;
    res.cdelfn  = cdelfn;
    res.cdref   = cdref;

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::operator()(const char *dummy, const Vector<int> &i, const Vector<int> &j, retMatrix<T> &res)
{
    NiceAssert( checkRange(0,dnumRows-1,i) );
    NiceAssert( checkRange(0,dnumCols-1,j) );

    res.reset();

    (void) dummy;

    res.bkref = bkref;

    if ( !nbase && !(i.base()) )
    {
	res.dnumRows = i.size();

	res.nbase    = 1;
	res.pbaseRow = 1;

	res.iibRow = 0;
	res.iisRow = 1;

	res.content  = content;
	res.ccontent = ccontent;

	res.iscover = iscover;
        res.elmfn   = elmfn;
	res.rowfn   = rowfn;
	res.delfn   = delfn;
	res.dref    = dref;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;

	if ( res.dnumRows )
	{
            res.pivotRow = &(i.grabcontentdirect());
	}

	else
	{
            res.pivotRow = cntintarray(0);
	}
    }

    else
    {
	res.dnumRows = i.size();

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dnumRows));

        NiceAssert( temppivot );

	if ( res.dnumRows )
	{
	    int ii;

	    for ( ii = 0 ; ii < res.dnumRows ; ii++ )
	    {
		(*temppivot)("&",ii) = (*pivotRow)(iibRow+(iisRow*(i(ii))));
	    }
	}

	res.nbase    = 1;
	res.pbaseRow = 0;

	res.iibRow   = 0;
	res.iisRow   = 1;

	res.content  = content;
	res.ccontent = ccontent;
        res.pivotRow = temppivot;

	res.iscover = iscover;
        res.elmfn   = elmfn;
	res.rowfn   = rowfn;
	res.delfn   = delfn;
	res.dref    = dref;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;
    }

    if ( !nbase && !(j.base()) )
    {
	res.dnumCols = j.size();

	res.nbase    = 1;
	res.pbaseCol = 1;

	res.iibCol = 0;
	res.iisCol = 1;

	res.content  = content;
	res.ccontent = ccontent;

	res.iscover = iscover;
        res.elmfn   = elmfn;
	res.rowfn   = rowfn;
	res.delfn   = delfn;
	res.dref    = dref;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;

	if ( res.dnumCols )
	{
            res.pivotCol = &(j.grabcontentdirect());
	}

	else
	{
            res.pivotCol = cntintarray(0);
	}
    }

    else
    {
	res.dnumCols = j.size();

        DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dnumCols));

        NiceAssert( temppivot );

	if ( res.dnumCols )
	{
	    int jj;

	    for ( jj = 0 ; jj < res.dnumCols ; jj++ )
	    {
		(*temppivot)("&",jj) = (*pivotCol)(iibCol+(iisCol*(j(jj))));
	    }
	}

	res.nbase    = 1;
	res.pbaseCol = 0;

	res.iibCol   = 0;
	res.iisCol   = 1;

	res.content  = content;
	res.ccontent = ccontent;
        res.pivotCol = temppivot;

	res.iscover = iscover;
        res.elmfn   = elmfn;
	res.rowfn   = rowfn;
	res.delfn   = delfn;
	res.dref    = dref;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;
    }

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(const Vector<int> &i, const Vector<int> &j, retMatrix<T> &res) const
{
    NiceAssert( checkRange(0,dnumRows-1,i) );
    NiceAssert( checkRange(0,dnumCols-1,j) );

    res.reset();

    res.bkref = bkref;

    if ( !nbase && !(i.base()) )
    {
	res.dnumRows = i.size();

	res.nbase    = 1;
	res.pbaseRow = 1;

	res.iibRow = 0;
	res.iisRow = 1;

	res.content  = NULL;
	res.ccontent = ccontent;

	res.iscover = iscover;
        res.elmfn   = NULL;
	res.rowfn   = NULL;
	res.delfn   = NULL;
	res.dref    = NULL;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;

	if ( res.dnumRows )
	{
            res.pivotRow = &(i.grabcontentdirect());
	}

	else
	{
            res.pivotRow = cntintarray(0);
	}
    }

    else
    {
	res.dnumRows = i.size();

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dnumRows));

        NiceAssert( temppivot );

	if ( res.dnumRows )
	{
	    int ii;

	    for ( ii = 0 ; ii < res.dnumRows ; ii++ )
	    {
		(*temppivot)("&",ii) = (*pivotRow)(iibRow+(iisRow*(i(ii))));
	    }
	}

	res.nbase    = 1;
	res.pbaseRow = 0;

	res.iibRow   = 0;
	res.iisRow   = 1;

	res.content  = NULL;
	res.ccontent = ccontent;
        res.pivotRow = temppivot;

	res.iscover = iscover;
        res.elmfn   = NULL;
	res.rowfn   = NULL;
	res.delfn   = NULL;
	res.dref    = NULL;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;
    }

    if ( !nbase && !(j.base()) )
    {
	res.dnumCols = j.size();

	res.nbase    = 1;
	res.pbaseCol = 1;

	res.iibCol = 0;
	res.iisCol = 1;

	res.content  = NULL;
	res.ccontent = ccontent;

	res.iscover = iscover;
        res.elmfn   = NULL;
	res.rowfn   = NULL;
	res.delfn   = NULL;
	res.dref    = NULL;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
        res.cdref   = cdref;

	if ( res.dnumCols )
	{
            res.pivotCol = &(j.grabcontentdirect());
	}

	else
	{
            res.pivotCol = cntintarray(0);
	}
    }

    else
    {
	res.dnumCols = j.size();

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dnumCols));

        NiceAssert( temppivot );

	if ( res.dnumCols )
	{
	    int jj;

	    for ( jj = 0 ; jj < res.dnumCols ; jj++ )
	    {
		(*temppivot)("&",jj) = (*pivotCol)(iibCol+(iisCol*(j(jj))));
	    }
	}

	res.nbase    = 1;
	res.pbaseCol = 0;

	res.iibCol   = 0;
	res.iisCol   = 1;

	res.content  = NULL;
	res.ccontent = ccontent;
        res.pivotCol = temppivot;

	res.iscover = iscover;
        res.elmfn   = NULL;
	res.rowfn   = NULL;
	res.delfn   = NULL;
	res.dref    = NULL;
	res.celmfn  = celmfn;
	res.crowfn  = crowfn;
	res.cdelfn  = cdelfn;
	res.cdref   = cdref;
    }

    return res;
}






// Column access - NO LONGER NEEDED
//
// The matrix is stored in rows.  To access a single row and treat it like
// a vector is therefore easy (see functions above) - you just return a
// reference to the relevant vector.  To access a single column and treat
// it like a vector is harder: the operator() functions cheat a little here
// and return a matrix with a single column.  You can overwrite with a vector
// easily enough, but its still a matrix.  So (for a matrix A, vector x):
//
// A("&",0,1,A.numRows()-1,2,"&") = x   - overwrites column 2 with vector x
// x = A("&",0,1,A.numRows()-1,2,"&")   - fails to compile.
//
// To get around this you could:
//
// - construct a vector and return it.  This would take time and memory
//   and may or may not incur a significant CPU time penalty due to calls
//   to constructors and copy operators when it is returned depending on
//   optimisation.  Also you couldn't just edit the vector returned and
//   see the results in the matrix, so const returns only.
// - make vectors that can use a function to access elements.  This would
//   be elegant, but bad for speed.  Currently vectors are fast because
//   they just dereference memory: if you add in a "is this a function"
//   check then things like inner products and multiplications would slow
//   down significantly.
// - other?
//
// or you could use these functions.
//
// getCol(dest,i) - overwrite dest with column i (return reference to dest)
// setCol(i,src)  - overwrite column i with src (return reference to src)
//
//Vector<T> &getCol(Vector<T> &dest, int i) const;
//Alistair Shilton 2014 (c) wrote this code
//const Vector<T> &setCol(int i, const Vector<T> &src);
//
//template <class T>
//Vector<T> &Matrix<T>::getCol(Vector<T> &dest, int j) const
//{
//    NiceAssert( j >= 0 );
//    NiceAssert( j < dnumCols );
//
//    dest.resize(dnumRows);
//
//    if ( dnumRows )
//    {
//	int i;
//
//	for ( i = 0 ; i < dnumRows ; i++ )
//	{
//            dest("&",i) = (*this)(i,j);
//	}
//    }
//
//    return dest;
//}
//
//template <class T>
//const Vector<T> &Matrix<T>::setCol(int j, const Vector<T> &src)
//{
//    NiceAssert( j >= 0 );
//    NiceAssert( j < dnumCols );
//    NiceAssert( src.size() == dnumRows );
//
//    if ( dnumRows )
//    {
//	int i;
//
//	for ( i = 0 ; i < dnumRows ; i++ )
//	{
//	    (*this)("&",i,j) = src(i);
//	}
//    }
//
//    return src;
//}








// Add and remove element functions

template <class T>
Matrix<T> &Matrix<T>::addRow(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dnumRows );

    dnumRows++;

    if ( !iscover )
    {
        (*content).add(i);
//	(*content)("&",i).fixsize = 0;
        (*content)("&",i).resize(dnumCols);
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::addCol(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dnumCols );

    dnumCols++;

    if ( !iscover )
    {
	if ( dnumRows )
	{
	    int j;

	    for ( j = 0 ; j < dnumRows ; j++ )
	    {
//		((*content)("&",j)).fixsize = 0;
		((*content)("&",j)).add(i);
	    }
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::padCol(int n)
{
    NiceAssert( !nbase );

    if ( !iscover )
    {
	if ( dnumRows )
	{
	    int j;

	    for ( j = 0 ; j < dnumRows ; j++ )
	    {
                ((*content)("&",j)).pad(n);
	    }
	}
    }

    dnumCols += n;

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::padRow(int n)
{
    NiceAssert( !nbase );

    int i,j;

    while ( n )
    {
        i = numRows();

        addRow(i);

        if ( numCols() )
        {
            for ( j = 0 ; j < numCols() ; j++ )
            {
                setzero((*this)("&",i,j));
            }
        }

        n--;
    }

    return *this;
}

template <class T> 
Matrix<T> &Matrix<T>::padRowCol(int n) 
{ 
    padRow(n);
    padCol(n);

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::addRowCol(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dnumRows );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dnumCols );

    // Speed choice: addCol takes o(numRows) operations to complete, addRow
    // takes o(1) operations (both leverage qswap for speed).  So by calling
    // addCol first things are marginally faster.

    addCol(i);
    addRow(i);

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::removeRow(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );

    dnumRows--;

    if ( !iscover )
    {
//	((*content)("&",i)).fixsize = 0;
	//FIXME: not sure if this is a good idea or not (speed vs memory)
        //       ((*content)("&",i)).resize(0);
	//       speed: leave the vector as it is, even though it may be big
        //              and may remain in memory for a little bit
	//       memory: it may remain in memory after removal, so downsize
        //               *right* *now*
	//DECISION: speed >> memory
	(*content).remove(i);
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::removeCol(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumCols );

    dnumCols--;

    if ( !iscover )
    {
	if ( dnumRows )
	{
	    int j;

	    for ( j = 0 ; j < dnumRows ; j++ )
	    {
//		((*content)("&",j)).fixsize = 0;
		((*content)("&",j)).remove(i);
	    }
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::removeRowCol(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumCols );

    // Speed choice: run the addRowCol argument in reverse.

    removeRow(i);
    removeCol(i);

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::appendRow(int rowStart, const Matrix<T> &src)
{
    NiceAssert( !nbase );
    NiceAssert( rowStart >= 0 );
    NiceAssert( rowStart <= dnumRows );
    NiceAssert( src.numCols() == numCols() );

    dnumRows += src.numRows();

    if ( !iscover && src.numRows() )
    {
        int i;

        retVector<T> tmpva;

        for ( i = rowStart ; i < rowStart+src.numRows() ; i++ )
        {
            (*content).add(i);

//          (*content)("&",i).fixsize = 0;
            (*content)("&",i) = src(i-rowStart,tmpva);
        }
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::appendCol(int colStart, const Matrix<T> &src)
{
    NiceAssert( !nbase );
    NiceAssert( colStart >= 0 );
    NiceAssert( colStart <= dnumCols );
    NiceAssert( src.numRows() == numRows() );

    dnumCols += src.numCols();

    if ( !iscover && src.numCols() && dnumRows )
    {
        int i,j;

        for ( i = 0 ; i < dnumRows ; i++ )
        {
            for ( j = colStart ; j < colStart+src.numCols() ; j++ )
            {
//              ((*content)("&",i)).fixsize = 0;
                ((*content)("&",i)).add(j);
                ((*content)("&",i))("&",j) = src(i,j-colStart);
	    }
	}
    }

    return *this;
}


template <class T>
Matrix<T> &Matrix<T>::resize(int targNumRows, int targNumCols)
{
    int i;

    if ( !iscover )
    {
        (*content).resize(targNumRows);

        if ( ( dnumRows < targNumRows ) && ( dnumCols == targNumCols ) )
        {
            for ( i = dnumRows ; i < targNumRows ; i++ )
            {
                (*content)("&",i).resize(targNumCols);
            }
        }

        dnumRows = targNumRows;

        if ( dnumRows && ( targNumCols != dnumCols ) )
        {
            for ( i = 0 ; i < dnumRows ; i++ )
            {
                (*content)("&",i).resize(targNumCols);
            }
        }
    }

    dnumRows = targNumRows;
    dnumCols = targNumCols;

    return *this;
}


// Function application

template <class T>
Matrix<T> &Matrix<T>::applyon(T (*fn)(T))
{
    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            (*this)("&",i,"&",tmpva).applyon(fn);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::applyon(T (*fn)(const T &))
{
    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            (*this)("&",i,"&",tmpva).applyon(fn);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::applyon(T &(*fn)(T &))
{
    if ( dnumRows && dnumCols )
    {
        int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            (*this)("&",i,"&",tmpva).applyon(fn);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::applyon(T (*fn)(T, const void *), const void *a)
{
    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            (*this)("&",i,"&",tmpva).applyon(fn,a);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::applyon(T (*fn)(const T &, const void *), const void *a)
{
    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            (*this)("&",i,"&",tmpva).applyon(fn,a);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::applyon(T &(*fn)(T &, const void *), const void *a)
{
    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            (*this)("&",i,"&",tmpva).applyon(fn,a);
	}
    }

    return *this;
}


// Other functions

template <class T>
Matrix<T> &Matrix<T>::rankone(const T &c, const Vector<T> &a, const Vector<T> &b)
{
    NiceAssert( dnumRows == a.size() );
    NiceAssert( dnumCols == b.size() );

    Vector<T> bb(b);
    bb.conj();
    rankoneNoConj(c,a,bb);

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::diagmult(const Vector<T> &J, const Vector<T> &K)
{
    NiceAssert( dnumRows == J.size() );
    NiceAssert( dnumCols == K.size() );

    Vector<T> KK(K);
    KK.conj();
    diagmultNoConj(J,KK);

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::diagoffset(const Vector<T> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;
    int i;

    NiceAssert( minsize == d.size() );

    if ( minsize )
    {
	for ( i = 0 ; i < minsize ; i++ )
	{
            (*this)("&",i,i) += d(i);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::diagoffset(const Matrix<T> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;
    int i;

    NiceAssert( minsize == ( ( d.numRows() < d.numCols() ) ? d.numRows() : d.numCols() ) );

    if ( minsize )
    {
	for ( i = 0 ; i < minsize ; i++ )
	{
            (*this)("&",i,i) += d(i,i);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::diagoffset(const T &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;
    int i;

    if ( minsize )
    {
	for ( i = 0 ; i < minsize ; i++ )
	{
            (*this)("&",i,i) += d;
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::diagoffsetNoConj(const Vector<T> &d)
{
    return diagoffset(d);
}

template <class T>
Matrix<T> &Matrix<T>::naiveChol(Matrix<T> &dest, int zeroupper) const
{
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( isSquare() );

    if ( dnumRows )
    {
	int i,j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( i = 0 ; i < dnumRows ; i++ )
	{
	    dest("&",i,i) = (*this)(i,i);

	    if ( i > 0 )
	    {
		for ( k = 0 ; k < i ; k++ )
		{
		    dest("&",i,i) -= dest(i,k)*conj(dest(i,k));
		}
	    }

            if ( abs2((double) dest(i,i)) < MATRIX_ZTOL )
            {
                dest("&",i,i) = MATRIX_ZTOL;
            }

            dest("&",i,i) = sqrt(dest(i,i));

	    if ( i < dnumRows-1 )
	    {
		for ( j = i+1 ; j < dnumRows ; j++ )
		{
		    dest("&",j,i) = (*this)(j,i);

		    if ( i > 0 )
		    {
			for ( k = 0 ; k < i ; k++ )
			{
			    dest("&",j,i) -= dest(j,k)*conj(dest(i,k));
			}
		    }

		    dest("&",j,i) /= real(dest(i,i));
		}
	    }
	}
    }

    return dest;
}

template <class T>
Matrix<T> &Matrix<T>::naiveChol(Matrix<T> &dest, const Vector<T> &diagoffset, int zeroupper) const
{
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == diagoffset.size() );

    if ( dnumRows )
    {
	int i,j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( i = 0 ; i < dnumRows ; i++ )
	{
	    dest("&",i,i) = (*this)(i,i);
	    dest("&",i,i) += diagoffset(i);

	    if ( i > 0 )
	    {
		for ( k = 0 ; k < i ; k++ )
		{
		    dest("&",i,i) -= dest(i,k)*conj(dest(i,k));
		}
	    }

            if ( abs2(dest(i,i)) < MATRIX_ZTOL )
            {
                dest("&",i,i) = MATRIX_ZTOL;
            }

	    dest("&",i,i) = sqrt(dest(i,i));

	    if ( i < dnumRows-1 )
	    {
		for ( j = i+1 ; j < dnumRows ; j++ )
		{
		    dest("&",j,i) = (*this)(j,i);

		    if ( i > 0 )
		    {
			for ( k = 0 ; k < i ; k++ )
			{
			    dest("&",j,i) -= dest(j,k)*conj(dest(i,k));
			}
		    }

		    dest("&",j,i) /= real(dest(i,i));
		}
	    }
	}
    }

    return dest;
}

template <class T>
Matrix<T> &Matrix<T>::naivepartChol(Matrix<T> &dest, Vector<int> &p, int &n, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == p.size() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );

    retVector<int> tmpva;

    n = dnumCols;
    p = cntintvec(n,tmpva);

    if ( dnumRows )
    {
        int i,j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

        for ( i = 0 ; i < n ; i++ )
        {
            dest("&",p(i),p(i)) = (*this)(p(i),p(i));

            if ( i > 0 )
            {
                for ( k = 0 ; k < i ; k++ )
                {
                    dest("&",p(i),p(i)) -= dest(p(i),p(k))*conj(dest(p(i),p(k)));
                }
            }

            if ( abs2((double) dest(i,i)) < ztol )
            {
                for ( j = i ; j < dnumRows ; j++ )
                {
                    dest("&",p(j),p(i)) = 0.0;
                }

                p.blockswap(i,n-1);

                n--;
                i--;
            }

            else
            {
                dest("&",p(i),p(i)) = sqrt(dest(p(i),p(i)));

                if ( i < dnumRows-1 )
                {
                    for ( j = i+1 ; j < dnumRows ; j++ )
                    {
                        dest("&",p(j),p(i)) = (*this)(p(j),p(i));

                        if ( i > 0 )
                        {
                            for ( k = 0 ; k < i ; k++ )
                            {
                                dest("&",p(j),p(i)) -= dest(p(j),p(k))*conj(dest(p(i),p(k)));
                            }
                        }

                        dest("&",p(j),p(i)) /= real(dest(p(i),p(i)));
                    }
                }
            }
        }
    }

    return dest;
}

// Based on something on github by phillip Burckhardt, which was adapted from
// code by Luke Tierney and David Betz.

#define SVDSIGN(a, b) ((b) >= 0.0 ? abs2(a) : -abs2(a))
#define SVDMAX(x,y) ((x)>(y)?(x):(y))

template <class T> 
double SVDPYTHAG(const T &a, const T &b)
{
    double at = abs2(a);
    double bt = abs2(b);
    double ct,result;

    if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
    else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
    else               { result = 0.0; }

    return result;
}

template <class T>
Matrix<T> &Matrix<T>::SVD(Matrix<T> &a, Vector<T> &w, Matrix<T> &v) const
{
    int m = numRows();
    int n = numCols();

    NiceAssert( m >= n );

    a = *this;
    a.resize(m,m);

    w.resize(n);
    v.resize(n,n);

    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;

    Vector<T> rv1(n);
  
/* Householder reduction to bidiagonal form */
    for (i = 0; i < n; i++) 
    {
        /* left-hand reduction */
        l = i + 1;
        rv1("&",i) = scale * g;
        g = s = scale = 0.0;
        if (i < m) 
        {
            for (k = i; k < m; k++) 
                scale += abs2((double)a(k,i));
            if (scale) 
            {
                for (k = i; k < m; k++) 
                {
                    a("&",k,i) = (double)((double)a(k,i)/scale);
                    s += ((double)a(k,i) * (double)a(k,i));
                }
                f = (double)a(i,i);
                g = -SVDSIGN(sqrt(s), f);
                h = f * g - s;
                a("&",i,i) = (double)(f - g);
                if (i != n - 1) 
                {
                    for (j = l; j < n; j++) 
                    {
                        for (s = 0.0, k = i; k < m; k++) 
                            s += ((double)a(k,i) * (double)a(k,j));
                        f = s / h;
                        for (k = i; k < m; k++) 
                            a("&",k,j) += (double)(f * (double)a(k,i));
                    }
                }
                for (k = i; k < m; k++) 
                    a("&",k,i) = (double)((double)a(k,i)*scale);
            }
        }
        w("&",i) = (double)(scale * g);
    
        /* right-hand reduction */
        g = s = scale = 0.0;
        if (i < m && i != n - 1) 
        {
            for (k = l; k < n; k++) 
                scale += abs2((double)a(i,k));
            if (scale) 
            {
                for (k = l; k < n; k++) 
                {
                    a("&",i,k) = (double)((double)a(i,k)/scale);
                    s += ((double)a(i,k) * (double)a(i,k));
                }
                f = (double)a(i,l);
                g = -SVDSIGN(sqrt(s), f);
                h = f * g - s;
                a("&",i,l) = (double)(f - g);
                for (k = l; k < n; k++) 
                    rv1("&",k) = (double)a(i,k) / h;
                if (i != m - 1) 
                {
                    for (j = l; j < m; j++) 
                    {
                        for (s = 0.0, k = l; k < n; k++) 
                            s += ((double)a(j,k) * (double)a(i,k));
                        for (k = l; k < n; k++) 
                            a("&",j,k) += (double)(s * rv1(k));
                    }
                }
                for (k = l; k < n; k++) 
                    a("&",i,k) = (double)((double)a(i,k)*scale);
            }
        }
        anorm = SVDMAX(anorm, (abs2((double)w(i)) + abs2(rv1(i))));
    }
  
    /* accumulate the right-hand transformation */
    for (i = n - 1; i >= 0; i--) 
    {
        if (i < n - 1) 
        {
            if (g) 
            {
                for (j = l; j < n; j++)
                    v("&",j,i) = (double)(((double)a(i,j) / (double)a(i,l)) / g);
                    /* double division to avoid underflow */
                for (j = l; j < n; j++) 
                {
                    for (s = 0.0, k = l; k < n; k++) 
                        s += ((double)a(i,k) * (double)v(k,j));
                    for (k = l; k < n; k++) 
                        v("&",k,j) += (double)(s * (double)v(k,i));
                }
            }
            for (j = l; j < n; j++) 
                v("&",i,j) = v("&",j,i) = 0.0;
        }
        v("&",i,i) = 1.0;
        g = rv1(i);
        l = i;
    }
  
    /* accumulate the left-hand transformation */
    for (i = n - 1; i >= 0; i--) 
    {
        l = i + 1;
        g = (double)w(i);
        if (i < n - 1) 
            for (j = l; j < n; j++) 
                a("&",i,j) = 0.0;
        if (g) 
        {
            g = 1.0 / g;
            if (i != n - 1) 
            {
                for (j = l; j < n; j++) 
                {
                    for (s = 0.0, k = l; k < m; k++) 
                        s += ((double)a(k,i) * (double)a(k,j));
                    f = (s / (double)a(i,i)) * g;
                    for (k = i; k < m; k++) 
                        a("&",k,j) += (double)(f * (double)a(k,i));
                }
            }
            for (j = i; j < m; j++) 
                a("&",j,i) = (double)((double)a(j,i)*g);
        }
        else 
        {
            for (j = i; j < m; j++) 
                a("&",j,i) = 0.0;
        }
        ++a("&",i,i);
    }

    /* diagonalize the bidiagonal form */
    for (k = n - 1; k >= 0; k--) 
    {                             /* loop over singular values */
        for (its = 0; its < 30; its++) 
        {                         /* loop over allowed iterations */
            flag = 1;
            for (l = k; l >= 0; l--) 
            {                     /* test for splitting */
                nm = l - 1;
                if (abs2(rv1(l)) + anorm == anorm) 
                {
                    flag = 0;
                    break;
                }
                if (abs2((double)w(nm)) + anorm == anorm) 
                    break;
            }
            if (flag) 
            {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++) 
                {
                    f = s * rv1(i);
                    if (abs2(f) + anorm != anorm) 
                    {
                        g = (double)w(i);
                        h = SVDPYTHAG(f, g);
                        w("&",i) = (double)h; 
                        h = 1.0 / h;
                        c = g * h;
                        s = (- f * h);
                        for (j = 0; j < m; j++) 
                        {
                            y = (double)a(j,nm);
                            z = (double)a(j,i);
                            a("&",j,nm) = (double)(y * c + z * s);
                            a("&",j,i) = (double)(z * c - y * s);
                        }
                    }
                }
            }
            z = (double)w(k);
            if (l == k) 
            {                  /* convergence */
                if (z < 0.0) 
                {              /* make singular value nonnegative */
                    w("&",k) = (double)(-z);
                    for (j = 0; j < n; j++) 
                        v("&",j,k) = (-v(j,k));
                }
                break;
            }
            if (its >= 30) {
                throw("No convergence after 30,000! iterations");
            }
    
            /* shift from bottom 2 x 2 minor */
            x = (double)w(l);
            nm = k - 1;
            y = (double)w(nm);
            g = rv1(nm);
            h = rv1(k);
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = SVDPYTHAG(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + SVDSIGN(g, f))) - h)) / x;
          
            /* next QR transformation */
            c = s = 1.0;
            for (j = l; j <= nm; j++) 
            {
                i = j + 1;
                g = rv1(i);
                y = (double)w(i);
                h = s * g;
                g = c * g;
                z = SVDPYTHAG(f, h);
                rv1("&",j) = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++) 
                {
                    x = (double)v(jj,j);
                    z = (double)v(jj,i);
                    v("&",jj,j) = (double)(x * c + z * s);
                    v("&",jj,i) = (double)(z * c - x * s);
                }
                z = SVDPYTHAG(f, h);
                w("&",j) = (double)z;
                if (z) 
                {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++) 
                {
                    y = (double)a(jj,j);
                    z = (double)a(jj,i);
                    a("&",jj,j) = (double)(y * c + z * s);
                    a("&",jj,i) = (double)(z * c - y * s);
                }
            }
            rv1("&",l) = 0.0;
            rv1("&",k) = f;
            w("&",k) = (double)x;
        }
    }

    return v;
}


template <class T>
Matrix<T> &Matrix<T>::rankoneNoConj(const T &c, const Vector<T> &a, const Vector<T> &b)
{
    NiceAssert( dnumRows == a.size() );
    NiceAssert( dnumCols == b.size() );

    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            (*this)("&",i,"&",tmpva) += (((a(i))*c)*b);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::diagmultNoConj(const Vector<T> &J, const Vector<T> &K)
{
    NiceAssert( dnumRows == J.size() );
    NiceAssert( dnumCols == K.size() );

    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            (*this)("&",i,"&",tmpva) = J(i)*(((*this)(i))*K);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::naiveCholNoConj(Matrix<T> &dest, int zeroupper) const
{
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( isSquare() );

    if ( dnumRows )
    {
	int i,j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( i = 0 ; i < dnumRows ; i++ )
	{
	    dest("&",i,i) = (*this)(i,i);

	    if ( i > 0 )
	    {
		for ( k = 0 ; k < i ; k++ )
		{
		    dest("&",i,i) -= dest(i,k)*dest(i,k);
		}
	    }

            if ( abs2(dest(i,i)) < MATRIX_ZTOL )
            {
                dest("&",i,i) = MATRIX_ZTOL;
            }

	    dest("&",i,i) = sqrt(dest(i,i));

	    if ( i < dnumRows-1 )
	    {
		for ( j = i+1 ; j < dnumRows ; j++ )
		{
		    dest("&",j,i) = (*this)(j,i);

		    if ( i > 0 )
		    {
			for ( k = 0 ; k < i ; k++ )
			{
			    dest("&",j,i) -= dest(j,k)*dest(i,k);
			}
		    }

		    dest("&",j,i) /= dest(i,i);
		}
	    }
	}
    }

    return dest;
}

template <class T>
Matrix<T> &Matrix<T>::naiveCholNoConj(Matrix<T> &dest, const Vector<T> &diagoffset, int zeroupper) const
{
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == diagoffset.size() );

    if ( dnumRows )
    {
	int i,j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( i = 0 ; i < dnumRows ; i++ )
	{
	    dest("&",i,i) = (*this)(i,i);
	    dest("&",i,i) += diagoffset(i);

	    if ( i > 0 )
	    {
		for ( k = 0 ; k < i ; k++ )
		{
		    dest("&",i,i) -= dest(i,k)*dest(i,k);
		}
	    }

            if ( abs2(dest(i,i)) < MATRIX_ZTOL )
            {
                dest("&",i,i) = MATRIX_ZTOL;
            }

	    dest("&",i,i) = sqrt(dest(i,i));

	    if ( i < dnumRows-1 )
	    {
		for ( j = i+1 ; j < dnumRows ; j++ )
		{
		    dest("&",j,i) = (*this)(j,i);

		    if ( i > 0 )
		    {
			for ( k = 0 ; k < i ; k++ )
			{
			    dest("&",j,i) -= dest(j,k)*dest(i,k);
			}
		    }

		    dest("&",j,i) /= dest(i,i);
		}
	    }
	}
    }

    return dest;
}

template <class T>
template <class S> Vector<S> &Matrix<T>::forwardElim(Vector<S> &y, const Vector<S> &b, int implicitTranspose) const
{
//SEE ALSO SPECIALISATION BELOW
    // G.y = b  know G and b, calculate y

    NiceAssert( isSquare() );
    NiceAssert( ( dnumRows == b.size() ) || isEmpty() );

    if ( y.shareBase(b) )
    {
	Vector<S> bb(b);

        forwardElim(y,bb,implicitTranspose);
    }

    else if ( b.size() )
    {
        if ( isEmpty() )
        {
            y = b;
        }

        else
        {
            if ( !implicitTranspose )
            {
                int zer = 0;
                int i;
                S temp;

                y = b;

                for ( i = 0 ; i < dnumRows ; i++ )
                {
                    y("&",i) -= twoProductNoConj(temp,(*this)(i,zer,1,i-1),y(zer,1,i-1));
                    y("&",i) = (inv((*this)(i,i))*y(i));
                }
            }

            else
            {
                int i,j;

                y = b;

                for ( i = 0 ; i < dnumRows ; i++ )
                {
                    if ( i )
                    {
                        for ( j = 0 ; j < i ; j++ )
                        {
                            y("&",i) -= (*this)(j,i)*y(j);
                        }
                    }

                    y("&",i) = (inv((*this)(i,i))*y(i));
                }
            }
        }
    }

    return y;
}

#define MINDIAGLOC 1e-6

template <> template <> inline Vector<double> &Matrix<double>::forwardElim(Vector<double> &y, const Vector<double> &b, int implicitTranspose) const;
template <> template <> inline Vector<double> &Matrix<double>::forwardElim(Vector<double> &y, const Vector<double> &b, int implicitTranspose) const
{
    // G.y = b  know G and b, calculate y

    NiceAssert( isSquare() );
    NiceAssert( ( dnumRows == b.size() ) || isEmpty() );

    if ( y.shareBase(b) && ( &y != &b ) )
    {
	Vector<double> bb(b);

        forwardElim(y,bb,implicitTranspose);
    }

    else if ( b.size() )
    {
        if ( isEmpty() )
        {
            if ( &y != &b )
            {
                y = b;
            }
        }

        else
        {
            if ( !implicitTranspose )
            {
                int zer = 0;
                int i;
                double temp;

                if ( &y != &b )
                {
                    y = b;
                }

                retVector<double> tmpva;
                retVector<double> tmpvb;

                for ( i = 0 ; i < dnumRows ; i++ )
                {
                    y("&",i) -= twoProductNoConj(temp,(*this)(i,zer,1,i-1,tmpva),y(zer,1,i-1,tmpvb));

                    double thisii = ( (*this)(i,i) < MINDIAGLOC ) ? MINDIAGLOC : (*this)(i,i);
                    double ywas = y(i);

tryagaina:
                    y("&",i) /= thisii;

                    if ( ( thisii < 1 ) && ( testisvnan(y(i)) || testisinf(y(i)) ) )
                    {
                        thisii *= 2;
                        y("&",i) = ywas;

                        goto tryagaina;
                    }
                }
            }

            else
            {
                int i,j;

                y = b;

                for ( i = 0 ; i < dnumRows ; i++ )
                {
                    if ( i )
                    {
                        for ( j = 0 ; j < i ; j++ )
                        {
                            y("&",i) -= (*this)(j,i)*y(j);
                        }
                    }

                    double thisii = ( (*this)(i,i) < MINDIAGLOC ) ? MINDIAGLOC : (*this)(i,i);
                    double ywas = y(i);

tryagainb:
                    y("&",i) /= thisii;

                    if ( ( thisii < 1 ) && ( testisvnan(y(i)) || testisinf(y(i)) ) )
                    {
                        thisii *= 2;
                        y("&",i) = ywas;

                        goto tryagainb;
                    }
                }
            }
        }
    }

    return y;
}

template <class T>
template <class S> Vector<S> &Matrix<T>::backwardSubst(Vector<S> &y, const Vector<S> &b, int implicitTranspose) const
{
//SEE ALSO SPECIALISATION BELOW
    // G.y = b  know G and b, calculate y

    NiceAssert( isSquare() );
    NiceAssert( ( dnumRows == b.size() ) || isEmpty() );

    if ( y.shareBase(b) )
    {
	Vector<S> bb(b);

        backwardSubst(y,bb,implicitTranspose);
    }

    else if ( b.size() )
    {
        if ( isEmpty() )
        {
            y = b;
        }

        else
        {
            if ( !implicitTranspose )
            {
                int i;
                S temp;

                y = b;

                for ( i = dnumRows-1 ; i >= 0 ; i-- )
                {
                    y("&",i) -= twoProductNoConj(temp,(*this)(i,i+1,1,dnumRows-1),y(i+1,1,dnumRows-1));
                    y("&",i) = (inv((*this)(i,i))*y(i));
                }
            }

            else
            {
                int i,j;

                y = b;

                for ( i = dnumRows-1 ; i >= 0 ; i-- )
                {
                    if ( i+1 < dnumRows )
                    {
                        for ( j = i+1 ; j < dnumRows ; j++ )
                        {
                            y("&",i) -= (*this)(j,i),y(j);
                        }
                    }

                    y("&",i) = (inv((*this)(i,i))*y(i));
                }
            }
        }
    }

    return y;
}

template <> template <> inline Vector<double> &Matrix<double>::backwardSubst(Vector<double> &y, const Vector<double> &b, int implicitTranspose) const;
template <> template <> inline Vector<double> &Matrix<double>::backwardSubst(Vector<double> &y, const Vector<double> &b, int implicitTranspose) const
{
    // G.y = b  know G and b, calculate y

    NiceAssert( isSquare() );
    NiceAssert( ( dnumRows == b.size() ) || isEmpty() );

    if ( y.shareBase(b) && ( &y != &b ) )
    {
	Vector<double> bb(b);

        backwardSubst(y,bb,implicitTranspose);
    }

    else if ( b.size() )
    {
        if ( isEmpty() )
        {
            if ( &y != &b )
            {
                y = b;
            }
        }

        else
        {
            if ( !implicitTranspose )
            {
                int i;
                double temp;

                if ( &y != &b )
                {
                    y = b;
                }

                retVector<double> tmpva;
                retVector<double> tmpvb;

                for ( i = dnumRows-1 ; i >= 0 ; i-- )
                {
                    y("&",i) -= twoProductNoConj(temp,(*this)(i,i+1,1,dnumRows-1,tmpva),y(i+1,1,dnumRows-1,tmpvb));

                    double thisii = ( (*this)(i,i) < MINDIAGLOC ) ? MINDIAGLOC : (*this)(i,i);
                    double ywas = y(i);

tryagaina:
                    y("&",i) /= thisii;

                    if ( ( thisii < 1 ) && ( testisvnan(y(i)) || testisinf(y(i)) ) )
                    {
                        thisii *= 2;
                        y("&",i) = ywas;

                        goto tryagaina;
                    }
                }
            }

            else
            {
                int i,j;

                y = b;

                for ( i = dnumRows-1 ; i >= 0 ; i-- )
                {
                    if ( i+1 < dnumRows )
                    {
                        for ( j = i+1 ; j < dnumRows ; j++ )
                        {
                            y("&",i) -= (*this)(j,i),y(j);
                        }
                    }

                    double thisii = ( (*this)(i,i) < MINDIAGLOC ) ? MINDIAGLOC : (*this)(i,i);
                    double ywas = y(i);

tryagainb:
                    y("&",i) /= thisii;

                    if ( ( thisii < 1 ) && ( testisvnan(y(i)) || testisinf(y(i)) ) )
                    {
                        thisii *= 2;
                        y("&",i) = ywas;

                        goto tryagainb;
                    }
                }
            }
        }
    }

    return y;
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInveNoConj(Vector<S> &y, const Vector<S> &b) const
{
    Vector<S> yinterm(y);
    Matrix<T> cholres(*this);

    return naiveCholInveNoConj(y,b,yinterm,cholres,0);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInveNoConj(Vector<S> &y, const Vector<S> &b, Vector<S> &yinterm) const
{
    Matrix<T> cholres(*this);

    return naiveCholInveNoConj(y,b,yinterm,cholres,0);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInveNoConj(Vector<S> &y, const Vector<S> &b, Matrix<T> & cholres, int zeroupper) const
{
    Vector<S> yinterm(y);

    return naiveCholInveNoConj(y,b,yinterm,cholres,zeroupper);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInveNoConj(Vector<S> &y, const Vector<S> &b, Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper) const
{
    naiveCholNoConj(cholres,zeroupper);

    cholres.forwardElim(yinterm,b);
    cholres.backwardSubst(y,yinterm,1);

    return y;
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInveNoConjx(Vector<S> &y, const Vector<S> &b, const Vector<T> &diagoffset) const
{
    Vector<S> yinterm(y);
    Matrix<T> cholres(*this);

    return naiveCholInveNoConjx(y,b,diagoffset,yinterm,cholres,0);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInveNoConjx(Vector<S> &y, const Vector<S> &b, const Vector<T> &diagoffset, Vector<S> &yinterm) const
{
    Matrix<T> cholres(*this);

    return naiveCholInveNoConjx(y,b,diagoffset,yinterm,cholres,0);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInveNoConjx(Vector<S> &y, const Vector<S> &b, const Vector<T> &diagoffset, Matrix<T> & cholres, int zeroupper) const
{
    Vector<S> yinterm(y);

    return naiveCholInveNoConjx(y,b,diagoffset,yinterm,cholres,zeroupper);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInveNoConjx(Vector<S> &y, const Vector<S> &b, const Vector<T> &diagoffset, Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper) const
{
    naiveCholNoConj(cholres,diagoffset,zeroupper);

    cholres.forwardElim(yinterm,b);
    cholres.backwardSubst(y,yinterm,1);

    return y;
}




// Determinants, inverses etc

template <class T>
T Matrix<T>::det(void) const
{
    NiceAssert( isSquare() );

    T res;

    if ( dnumRows == 0 )
    {
	setident(res); // this will be 1 if matrix of scalars, empty matrix if matrix of matrices, error if matrix of vectors
    }

    else if ( dnumRows == 1 )
    {
        // Trivial result

	res = (*this)(0,0);
    }

    else if ( dnumRows <= MATRIX_DETRECURSEMAX )
    {
        // For small matrices use method of cofactors

	int i;

	res = ( (*this)(0,0) * cofactor(0,0) );

	for ( i = 1 ; i < dnumCols ; i++ )
	{
	    res += ( (*this)(0,i) * cofactor(0,i) );
	}
    }

    else
    {
        // For large matrices use LUP decomposition method

        if ( !matbuff )
        {
            MEMNEW((**thisindirect).matbuff,Matrix<T>);
        }

        (*((**thisindirect).matbuff)) = *this;

        int s = (*(((**thisindirect).matbuff))).LUPDecompose();
        int i;

        const Vector<int> &p = (*matbuff).pbuff;

        if ( s >= 0 )
        {
            res = (*matbuff)(p(zeroint()),zeroint());

            for ( i = 1 ; i < dnumRows ; i++ )
            {
                res *= (*matbuff)(p(i),i);
            }

            if ( s % 2 )
            {
                setnegate(res);
            }
        }

        else
        {
            setzero(res);
        }
    }

    return res;
}

template <class T>
T Matrix<T>::trace(void) const
{
    NiceAssert( isSquare() );

    T res;

    if ( !dnumRows )
    {
        setzero(res);
    }

    else
    {
	res = (*this)(0,0);

	if ( dnumRows > 1 )
	{
	    int i;

	    for ( i = 1 ; i < dnumRows ; i++ )
	    {
		res += (*this)(i,i);
	    }
	}
    }

    return res;
}

template <class T>
T Matrix<T>::invtrace(void) const
{
    NiceAssert( isSquare() );

    T res;

    // Calculate 1/tr(inv(A)) = det(A)/

    if ( !dnumRows )
    {
        // We treat this case as an analogy of dnumRows == 1

	setzero(res);
    }

    else if ( dnumRows == 1 )
    {
        // In this case 1/tr(inv(A)) = 1/(1/A00) = A00

	res = (*this)(0,0);
    }

    else
    {
        // In this case 1/tr(inv()) = 1/(tr(adj()/det()))
        //                          = 1/(sum_i(miner(i,i)/det()))
        //                          = det()/(sum_i(miner(i,i)))

        int i;
        T ressum;

        Vector<int> rowcolsel(dnumRows-1);

        // The following is the miner function specialised for diagonals.
        // rowcolsel is the index vector with current diagonal removed.
        // It is set once at start and then can be updated incrementally.

        for ( i = 1 ; i < dnumRows ; i++ )
        {
            rowcolsel("&",i-1) = i;
        }

        retMatrix<T> tmpma;

        for ( i = 0 ; i < dnumCols ; i++ )
        {
            if ( i )
            {
                rowcolsel("&",i-1)--;
            }

            if ( !i )
            {
                ressum = ((*this)(rowcolsel,rowcolsel,tmpma)).det();
            }

            else
            {
                ressum += ((*this)(rowcolsel,rowcolsel,tmpma)).det();
            }
        }

        res =  det();
        res /= ressum;
    }

    return res;
}

template <class T>
T Matrix<T>::miner(int i, int j) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows );

    T res;

    if ( dnumRows == 1 )
    {
	// Need a special case here.  If the matrix
	// is a 1x1 matrix then the miner of the only
	// element should be the identity matrix.  However to
	// ensure the size is compatible with other elements we
	// want the identity the size of that element.

	res = (*this)(0,0);
        setident(res);
    }

    else
    {
	Vector<int> rowsel(dnumRows-1);
	Vector<int> colsel(dnumCols-1);

	int ii,jj,kk;

	ii = 0;
	jj = 0;

	for ( kk = 0 ; kk < dnumRows ; kk++ )
	{
	    if ( kk != i )
	    {
		rowsel("&",ii) = kk;

		ii++;
	    }

	    if ( kk != j )
	    {
		colsel("&",jj) = kk;

		jj++;
	    }
	}

        retMatrix<T> tmpma;

	res = ((*this)(rowsel,colsel,tmpma)).det();
    }

    return res;
}

template <class T>
T Matrix<T>::cofactor(int i, int j) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows );

    T res = miner(i,j);

    if ( (i+j)%2 )
    {
        setnegate(res);
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::adj(Matrix<T> &res) const
{
    NiceAssert( isSquare() );

    res.resize(dnumRows,dnumCols);

    if ( dnumRows )
    {
	int i,j;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
	    for ( j = 0 ; j < dnumCols ; j++ )
	    {
		res("&",i,j) = cofactor(j,i);
	    }
	}
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::inve(Matrix<T> &res) const
{
    res.resize(dnumRows,dnumCols);

    if ( dnumRows && dnumCols )
    {
        if ( isSquare() )
	{
	    T matdet;
	    matdet = det();

	    if ( dnumRows == 1 )
	    {
		res("&",0,0) = inv(matdet);
	    }

            else if ( dnumRows == 2 )
            {
                res("&",0,0) = (*this)(1,1)*inv(matdet);
                res("&",0,1) = -(*this)(0,1)*inv(matdet);
                res("&",1,0) = -(*this)(1,0)*inv(matdet);
                res("&",1,1) = (*this)(0,0)*inv(matdet);
            }

	    else
	    {
		int i,j;

		for ( i = 0 ; i < dnumRows ; i++ )
		{
		    for ( j = 0 ; j < dnumCols ; j++ )
		    {
			res("&",i,j) = (cofactor(j,i)*inv(matdet));
		    }
		}
	    }
	}

	else if ( dnumRows > dnumCols )
	{
	    // Tall matrix
	    //
	    // A+ = ((A*.A)+).(A*)
	    // A+* = A.((A*.A)+*)
	    //     = A.(((A*.A)*)+) (return this)

	    Matrix<T> submatr(dnumCols,dnumCols);
	    Matrix<T> subinv(dnumCols,dnumCols);
	    T leftargx;
            T rightargx;

	    int i,j,k;

	    for ( i = 0 ; i < dnumCols ; i++ )
	    {
		for ( j = 0 ; j < dnumCols ; j++ )
		{
		    for ( k = 0 ; k < dnumRows ; k++ )
		    {
			////submatr("&",i,j) += conj((*this)(k,i))*((*this)(k,j));
			//submatr("&",j,i) += conj((*this)(k,j))*((*this)(k,i));

			leftargx  = conj((*this)(k,j));
			rightargx = ((*this)(k,i));

			submatr("&",j,i) += (leftargx*rightargx);
		    }
		}
	    }

	    //submatr.conj();
	    //submatr.transpose();

            res = *this;
            leftmult(res,submatr.inve(subinv));
	}

	else
	{
	    // Wide matrix
	    //
	    // A+ = (A*).((A.A*)+)
	    // A+* = ((A.A*)+*).A
	    //     = (((A.A*)*)+).A (return this)

	    Matrix<T> submatr(dnumRows,dnumRows);
	    Matrix<T> subinv(dnumRows,dnumRows);
	    T leftargx;
            T rightargx;

	    int i,j,k;

	    for ( i = 0 ; i < dnumRows ; i++ )
	    {
		for ( j = 0 ; j < dnumRows ; j++ )
		{
		    for ( k = 0 ; k < dnumCols ; k++ )
		    {
			////submatr("&",i,j) += ((*this)(i,k))*conj((*this)(j,k));
			//submatr("&",j,i) += ((*this)(j,k))*conj((*this)(i,k));

			leftargx  = ((*this)(j,k));
			rightargx = conj((*this)(i,k));

			submatr("&",j,i) += (leftargx*rightargx);
		    }
		}
	    }

	    //submatr.conj();
            //submatr.transpose();

            res = *this;
            rightmult(submatr.inve(subinv),res);
	}
    }

    else
    {
        res = *this;
        res.transpose();
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::inveSymmNoConj(Matrix<T> &res) const
{
    NiceAssert( isSquare() );

    res.resize(dnumRows,dnumCols);

    if ( dnumRows && dnumCols )
    {
        T matdet;
        matdet = det();

        if ( dnumRows == 1 )
        {
            res("&",0,0) = inv(matdet);
        }

        else
        {
            int i,j;

            for ( i = 0 ; i < dnumRows ; i++ )
            {
                for ( j = 0 ; j <= i ; j++ )
                {
                    res("&",i,j) = (cofactor(j,i)*inv(matdet));
                    res("&",j,i) = res("&",i,j);
                }
            }
        }
    }

    else
    {
        res = *this;
    }

    return res;
}

template <class T>
int Matrix<T>::LUPDecompose(Matrix<T> &res, Vector<int> &p, double ztol) const
{
    res = *this;

    int s = res.LUPDecompose(ztol);

    p = res.pbuff;

    return s;
}

template <class T>
int Matrix<T>::LUPDecompose(double ztol)
{
    NiceAssert( isSquare() );

    int s = 0;

    // Based on wikipedia

    int N = dnumRows;
    int i,j,k,imax; 
    double maxA,absA;

    retVector<int> tmpva;

    pbuff = cntintvec(N,tmpva);

    for ( i = 0 ; i < N ; i++ )
    {
        maxA = 0.0;
        imax = i;

        for ( k = i ; k < N ; k++ )
        {
            absA = abs2((double) (*this)(pbuff(k),i));

            if ( absA > maxA ) 
            {
                maxA = absA;
                imax = k;
            }
        }

        if ( maxA < ztol )
        { 
            // Degenerate case, computation failed

            return -1;
        }

        if ( imax != i ) 
        {
            pbuff.squareswap(i,imax);

            s++;
        }

        for ( j = i+1 ; j < N ; j++ )
        {
            (*this)("&",pbuff(j),i) /= (*this)(pbuff(i),i);

            for ( k = i+1 ; k < N ; k++ )
            {
                (*this)("&",pbuff(j),k) -= (*this)(pbuff(j),i)*(*this)(pbuff(i),k);
            }
        }
    }

    return s;
}

template <class T>
void Matrix<T>::tridiag(Vector<T> &d, Vector<T> &e, Vector<T> &e2) const
{
    Matrix<T> &a = **thisindirect;

/*
      subroutine tred1(nm,n,a,d,e,e2)
c
      integer i,j,k,l,n,ii,nm,jp1
      double precision a(nm,n),d(n),e(n),e2(n)
      double precision f,g,h,scale
c
c     this subroutine is a translation of the algol procedure tred1,
c     num. math. 11, 181-195(1968) by martin, reinsch, and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 212-226(1971).
c
c     this subroutine reduces a real symmetric matrix
c     to a symmetric tridiagonal matrix using
c     orthogonal similarity transformations.
c
c     on input
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement.
c
c        n is the order of the matrix.
c
c        a contains the real symmetric input matrix.  only the
c          lower triangle of the matrix need be supplied.
c
c     on output
c
c        a contains information about the orthogonal trans-
c          formations used in the reduction in its strict lower
c          triangle.  the full upper triangle of a is unaltered.
c
c        d contains the diagonal elements of the tridiagonal matrix.
c
c        e contains the subdiagonal elements of the tridiagonal
c          matrix in its last n-1 positions.  e(1) is set to zero.
c
c        e2 contains the squares of the corresponding elements of e.
c          e2 may coincide with e if the squares are not needed.
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
c
*/

////void tred1(int nm, int n, Matrix<double> &a, Vector<double> &d, Vector<double> &e, Vector<double> &e2);
//void tred1(Matrix<double> &a, Vector<double> &d, Vector<double> &e, Vector<double> &e2);


    NiceAssert( a.isSquare() );

    int n = a.size();

    // Save diagonal

    int ix,jx;

    Vector<T> adiag(n);

    for ( ix = 0 ; ix < n ; ix++ )
    {
        adiag("&",ix) = a(ix,ix);
    }

    d.resize(n);
    e.resize(n);
    e2.resize(n);

    int i,j,k,l,ii,jp1;
    double f,g,h,scale;

    for ( i = 1 ; i <= n ; i++ )
    {
        d("&",i-1) = a(n-1,i-1);
        a("&",n-1,i-1) = a(i-1,i-1);
    }

    //c     .......... for i=n step -1 until 1 do -- ..........

    for ( ii = 1 ; ii <= n ; ii++ )
    {
        i = n + 1 - ii;
        l = i - 1;
        h = 0.0;
        scale = 0.0;

        if ( l < 1 )
        {
            goto l130;
        }

        //c     .......... scale row (algol tol then not needed) ..........

        for ( k = 1 ; k <= l ; k++ )
        {
            scale = scale + abs2(d(k-1));
        }

        if ( scale != 0.0 )
        {
            goto l140;
        }

        for ( j = 1 ; j <= l ; j++ )
        {
            d("&",j-1) = a(l-1,j-1);
            a("&",l-1,j-1) = a(i-1,j-1);
            a("&",i-1,j-1) = 0.0;
        }

l130:
        e("&",i-1) = 0.0;
        e2("&",i-1) = 0.0;

        goto l300;

l140:
        for ( k = 1 ; k <= l ; k++ )
        {
            d("&",k-1) = d(k-1) / scale;
            h = h + d(k-1) * d(k-1);
        }

        e2("&",i-1) = scale * scale * h;
        f = d(l-1);
        g = -dsign(sqrt(h),f);
        e("&",i-1) = scale * g;
        h = h - f * g;
        d("&",l-1) = f - g;

        if ( l == 1 )
        {
            goto l285;
        }

        //c     .......... form a*u ..........

        for ( j = 1 ; j <= l ; j++ )
        {
            e("&",j-1) = 0.0;
        }

        for ( j = 1 ; j <= l ; j++ )
        {
            f = d(j-1);

            g = e(j-1) + a(j-1,j-1) * f;
            jp1 = j + 1;

            if ( l < jp1 )
            {
                goto l220;
            }

            for ( k = jp1 ; k <= l ; k++ )
            {
                g = g + a(k-1,j-1) * d(k-1);
                e("&",k-1) = e(k-1) + a(k-1,j-1) * f;
            }

l220:
            e("&",j-1) = g;
        }


        //c     .......... form p ..........

        f = 0.0;

        for ( j = 1 ; j <= l ; j++ )
        {
            e("&",j-1) = e(j-1) / h;
            f = f + e(j-1) * d(j-1);
        }

        h = f / (h + h);

        //c     .......... form q ..........

        for ( j = 1 ; j <= l ; j++ )
        {
            e("&",j-1) = e(j-1) - h * d(j-1);
        }

        //c     .......... form reduced a ..........

        for ( j = 1 ; j <= l ; j++ )
        {
            f = d(j-1);
            g = e(j-1);

            for ( k = j ; k <= l ; k++ )
            {
                a("&",k-1,j-1) = a(k-1,j-1) - f * e(k-1) - g * d(k-1);
            }
        }

l285:
        for ( j = 1 ; j <= l ; j++ )
        {
            f = d(j-1);
            d("&",j-1) = a(l-1,j-1);
            a("&",l-1,j-1) = a(i-1,j-1);
            a("&",i-1,j-1) = f * scale;
        }
    }

    // Reconstruct lower-triangular part

l300:
    for ( ix = 0 ; ix < n ; ix++ )
    {
        a("&",ix,ix) = adiag(ix);

        for ( jx = ix+1 ; jx < n ; jx++ )
        {
            a("&",jx,ix) = a(ix,jx);
        }
    }
    
    return;
}

template <class T>
void Matrix<T>::tridiag(Vector<T> &d, Vector<T> &e, Matrix<T> &z) const
{
    const Matrix<T> &a = *this;

/*
      subroutine tred2(nm,n,a,d,e,z)
c
      integer i,j,k,l,n,ii,nm,jp1
      double precision a(nm,n),d(n),e(n),z(nm,n)
      double precision f,g,h,hh,scale
c
c     this subroutine is a translation of the algol procedure tred2,
c     num. math. 11, 181-195(1968) by martin, reinsch, and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 212-226(1971).
c
c     this subroutine reduces a real symmetric matrix to a
c     symmetric tridiagonal matrix using and accumulating
c     orthogonal similarity transformations.
c
c     on input
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement.
c
c        n is the order of the matrix.
c
c        a contains the real symmetric input matrix.  only the
c          lower triangle of the matrix need be supplied.
c
c     on output
c
c        d contains the diagonal elements of the tridiagonal matrix.
c
c        e contains the subdiagonal elements of the tridiagonal
c          matrix in its last n-1 positions.  e(1) is set to zero.
c
c        z contains the orthogonal transformation matrix
c          produced in the reduction.
c
c        a and z may coincide.  if distinct, a is unaltered.
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
c
*/

////void tred2(int nm, int n, Matrix<double> &a, Vector<double> &d, Vector<double> &e, Matrix<double> &z);
//void tred2(const Matrix<double> &a, Vector<double> &d, Vector<double> &e, Matrix<double> &z);

    NiceAssert( a.isSquare() );

    int n = a.numRows();

    d.resize(n);
    e.resize(n);
    z.resize(n,n);

    int i,j,k,l,ii,jp1;
    double f,g,h,hh,scale;

    for ( i = 1 ; i <= n ; i++ )
    {
        for ( j = i ; j <= n ; j++ )
        {
            z("&",j-1,i-1) = a(j-1,i-1);
        }

//a("&",n-1,i-1) = a(i-1,i-1);
        d("&",i-1) = a(n-1,i-1);
    }

    if ( n == 1 )
    {
        goto l510;
    }

    //c     .......... for i=n step -1 until 2 do -- ..........

    for ( ii = 2 ; ii <= n ; ii++ )
    {
        i = n + 2 - ii;
        l = i - 1;
        h = 0.0;
        scale = 0.0;

        if ( l < 2 )
        {
            goto l130;
        }

        //c     .......... scale row (algol tol then not needed) ..........

        for ( k = 1 ; k <= l ; k++ )
        {
            scale = scale + abs2(d(k-1));
        }

        if ( scale != 0.0 )
        {
            goto l140;
        }

l130:
        e("&",i-1) = d(l-1);

        for ( j = 1 ; j <= l ; j++ )
        {
            d("&",j-1) = z(l-1,j-1);
            z("&",i-1,j-1) = 0.0;
            z("&",j-1,i-1) = 0.0;
        }

        goto l290;

l140:
        for ( k = 1 ; k <= l ; k++ )
        {
            d("&",k-1) = d(k-1) / scale;
            h = h + d(k-1) * d(k-1);
        }

        f = d(l-1);
        g = -dsign(sqrt(h),f);
        e("&",i-1) = scale * g;
        h = h - f * g;
        d("&",l-1) = f - g;

        //c     .......... form a*u ..........

        for ( j = 1 ; j <= l ; j++ )
        {
            e("&",j-1) = 0.0;
        }

        for ( j = 1 ; j <= l ; j++ )
        {
            f = d(j-1);
            z("&",j-1,i-1) = f;
            g = e(j-1) + z(j-1,j-1) * f;
            jp1 = j + 1;

            if ( l < jp1 )
            {
                goto l220;
            }

            for ( k = jp1 ; k <= l ; k++ )
            {
                g = g + z(k-1,j-1) * d(k-1);
                e("&",k-1) = e(k-1) + z(k-1,j-1) * f;
            }

l220:
            e("&",j-1) = g;
        }

        //c     .......... form p ..........

        f = 0.0;

        for ( j = 1 ; j <= l ; j++ )
        {
            e("&",j-1) = e(j-1) / h;
            f = f + e(j-1) * d(j-1);
        }

        hh = f / ( h + h );

        //c     .......... form q ..........

        for ( j = 1 ; j <= l ; j++ )
        {
            e("&",j-1) = e(j-1) - hh * d(j-1);
        }

        //c     .......... form reduced a ..........
        for ( j = 1 ; j <= l ; j++ )
        {
            f = d(j-1);
            g = e(j-1);

            for ( k = j ; k <= l ; k++ )
            {
                z("&",k-1,j-1) = z(k-1,j-1) - f * e(k-1) - g * d(k-1);
            }

            d("&",j-1) = z(l-1,j-1);
            z("&",i-1,j-1) = 0.0;
        }

l290:
        d("&",i-1) = h;
    }

    //c     .......... accumulation of transformation matrices ..........

    for ( i = 2 ; i <= n ; i++ )
    {
        l = i - 1;
        z("&",n-1,l-1) = z(l-1,l-1);
        z("&",l-1,l-1) = 1.0;
        h = d(i-1);

        if ( h == 0.0 )
        {
            goto l380;
        }

        for ( k = 1 ; k <= l ; k++ )
        {
            d("&",k-1) = z(k-1,i-1) / h;
        }

        for ( j = 1 ; j <= l ; j++ )
        {
            g = 0.0;

            for ( k = 1 ; k <= l ; k++ )
            {
                g = g + z(k-1,i-1) * z(k-1,j-1);
            }


            for ( k = 1 ; k <= l ; k++ )
            {
                z("&",k-1,j-1) = z(k-1,j-1) - g * d(k-1);
            }
        }

l380:
        for ( k = 1 ; k <= l ; k++ )
        {
            z("&",k-1,i-1) = 0.0;
        }
    }

l510:
    for ( i = 1 ; i <= n ; i++ )
    {
        d("&",i-1) = z(n-1,i-1);
        z("&",n-1,i-1) = 0.0;
    }

    z("&",n-1,n-1) = 1.0;
    e("&",1-1) = 0.0;

    return;
}

template <class T>
int tql1(Vector<T> &d, Vector<T> &e)
{
    NiceAssert( d.size() == e.size() );

    int n = d.size();

    int ierr = 0;

    int i,j,l,m,ii,l1,l2,mml;
    T c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2;

    s2 = 0.0; // Not strictly necessary (s2 is set in loop before being used as mml >= 1 is guaranteed), but stops gcc from complaining
    c3 = 0.0;

    if ( n == 1 )
    {
        goto l1001;
    }

    for ( i = 2 ; i <= n ; i++ )
    {
        e("&",i-1-1) = e(i-1);
    }

    f = 0.0;
    tst1 = 0.0;
    e("&",n-1) = 0.0;

    for ( l = 1 ; l <= n ; l++ )
    {
        j = 0;
        h = abs2(d(l-1)) + abs2(e(l-1));

        if ( tst1 < h )
        {
            tst1 = h;
        }

        //c     .......... look for small sub-diagonal element ..........

        for ( m = l ; m <= n ; m++ )
        {
            tst2 = tst1 + abs2(e(m-1));

            if ( tst2 == tst1 )
            {
                break;
            }

            //c     .......... e(n) is always zero, so there is no exit
            //c                through the bottom of the loop ..........
        }

        if ( m == l )
        {
            goto l210;
        }

l130:
        if ( j == 30 )
        {
            goto l1000;
        }

        j = j + 1;

        //c     .......... form shift ..........

        l1 = l + 1;
        l2 = l1 + 1;
        g = d(l-1);
        p = ( d(l1-1) - g ) / ( 2.0 * e(l-1) );
        r = sppythag(p,1.0);
        d("&",l-1) = e(l-1) / ( p + dsign(r,p) );
        d("&",l1-1) = e(l-1) * ( p + dsign(r,p) );
        dl1 = d(l1-1);
        h = g - d(l-1);

        if ( l2 > n )
        {
            goto l145;
        }

        for ( i = l2 ; i <= n ; i++ )
        {
            d("&",i-1) -= h;
        }

l145:
        f += h;

        //c     .......... ql transformation ..........

        p = d(m-1);
        c = 1.0;
        c2 = c;
        el1 = e(l1-1);
        s = 0.0;
        mml = m - l;

        //c     .......... for i=m-1 step -1 until l do -- ..........

        for ( ii = 1 ; ii <= mml ; ii++ )
        {
            c3 = c2;
            c2 = c;
            s2 = s;
            i = m - ii;
            g = c * e(i-1);
            h = c * p;
            r = sppythag(p,e(i-1));
            e("&",i+1-1) = s * r;
            s = e(i-1) / r;
            c = p / r;
            p = c * d(i-1) - s * g;
            d("&",i+1-1) = h + s * ( c * g + s * d(i-1) );
        }

        p = -s * s2 * c3 * el1 * e(l-1) / dl1;
        e("&",l-1) = s * p;
        d("&",l-1) = c * p;
        tst2 = tst1 + abs2(e(l-1));

        if ( tst2 > tst1 )
        {
            goto l130;
        }

l210:
        p = d(l-1) + f;

        //c     .......... order eigenvalues ..........

        if ( l == 1 )
        {
            goto l250;
        }

        //c     .......... for i=l step -1 until 2 do -- ..........

        for ( ii = 2 ; ii <= l ; ii++ )
        {
            i = l + 2 - ii;

            if ( p >= d(i-1-1) )
            {
                goto l270;
            }

            d("&",i-1) = d(i-1-1);
        }

l250:
        i = 1;
l270:
        d("&",i-1) = p;

    }

    goto l1001;

    //c     .......... set error -- no convergence to an
    //c                eigenvalue after 30 iterations ..........

l1000:
    ierr = l;
l1001:
    return ierr;
}

template <class T>
int tql2(Vector<T> &d, Vector<T> &e, Matrix<T> &z)
{
    NiceAssert( z.isSquare() );
    NiceAssert( d.size() == e.size() );
    NiceAssert( d.size() == z.size() );

    int n = d.size();

    int i,j,k,l,m,ii,l1,l2,mml;
    T c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2;

    s2 = 0.0; // See comment in tql1
    c3 = 0.0;

    int ierr = 0;

    if ( n == 1 )
    {
        goto l1001;
    }

    for ( i = 2 ; i <= n ; i++ )
    {
        e("&",i-1-1) = e(i-1);
    }

    f = 0.0;
    tst1 = 0.0;
    e("&",n-1) = 0.0;

    for ( l = 1 ; l <= n ; l++ )
    {
        j = 0;
        h = abs2(d(l-1)) + abs2(e(l-1));

        if ( tst1 < h )
        {
            tst1 = h;
        }

        //c     .......... look for small sub-diagonal element ..........

        for ( m = l ; m <= n ; m++ )
        {
            tst2 = tst1 + abs2(e(m-1));

            if ( tst2 == tst1 )
            {
                break;
            }

            //c     .......... e(n) is always zero, so there is no exit
            //c                through the bottom of the loop ..........
        }

        if ( m == l )
        {
            goto l220;
        }

l130:
        if ( j == 30 )
        {
            goto l1000;
        }

        j++;

        //c     .......... form shift ..........

        l1 = l + 1;
        l2 = l1 + 1;
        g = d(l-1);
        p = ( d(l1-1) - g ) / ( 2.0 * e(l-1) );
        r = sppythag(p,1.0);
        d("&",l-1) = e(l-1) / ( p + dsign(r,p) );
        d("&",l1-1) = e(l-1) * ( p + dsign(r,p) );
        dl1 = d(l1-1);
        h = g - d(l-1);

        if (l2 > n)
        {
            goto l145;
        }

        for ( i = l2 ; i <= n ; i++ )
        {
            d("&",i-1) -= h;
        }

l145:
        f = f + h;

        //c     .......... ql transformation ..........

        p = d(m-1);
        c = 1.0;
        c2 = c;
        el1 = e(l1-1);
        s = 0.0;
        mml = m - l;

        //c     .......... for i=m-1 step -1 until l do -- ..........

        for ( ii = 1 ; ii <= mml ; ii++ )
        {
            c3 = c2;
            c2 = c;
            s2 = s;
            i = m - ii;
            g = c * e(i-1);
            h = c * p;
            r = sppythag(p,e(i-1));
            e("&",i+1-1) = s * r;
            s = e(i-1) / r;
            c = p / r;
            p = c * d(i-1) - s * g;
            d("&",i+1-1) = h + s * ( c * g + s * d(i-1) );

            //c     .......... form vector ..........

            for ( k = 1 ; k <= n ; k++ )
            {
                h = z(k-1,i+1-1);
                z("&",k-1,i+1-1) = s * z(k-1,i-1) + c * h;
                z("&",k-1,i-1) = c * z(k-1,i-1) - s * h;
            }
        }

         p = -s * s2 * c3 * el1 * e(l-1) / dl1;
         e("&",l-1) = s * p;
         d("&",l-1) = c * p;
         tst2 = tst1 + abs2(e(l-1));

         if ( tst2 > tst1 )
         {
             goto l130;
         }

l220:
        d("&",l-1) += f;
    }

    //c     .......... order eigenvalues and eigenvectors ..........

    for ( ii = 2 ; ii <= n ; ii++ )
    {
        i = ii - 1;
        k = i;
        p = d(i-1);

        for ( j = ii ; j <= n ; j++ )
        {
            if ( d(j-1) >= p )
            {
                break;
            }

            k = j;
            p = d(j-1);
        }

        if ( k == i )
        {
            break;
        }

        d("&",k-1) = d(i-1);
        d("&",i-1) = p;

        for ( j = 1 ; j <= n ; j++ )
        {
            p = z(j-1,i-1);
            z("&",j-1,i-1) = z(j-1,k-1);
            z("&",j-1,k-1) = p;
        }
    }

    goto l1001;

    //c     .......... set error -- no convergence to an
    //c                eigenvalue after 30 iterations ..........

l1000:
    ierr = l;
l1001:
    return ierr;
}

template <class T>
int Matrix<T>::eig(Vector<T> &w, Vector<T> &fv1, Vector<T> &fv2) const
{
//    const Matrix<T> &a = *this;

/*
      subroutine rs(nm,n,a,w,matz,z,fv1,fv2,ierr)
c
      integer n,nm,ierr,matz
      double precision a(nm,n),w(n),z(nm,n),fv1(n),fv2(n)
c
c     this subroutine calls the recommended sequence of
c     subroutines from the eigensystem subroutine package (eispack)
c     to find the eigenvalues and eigenvectors (if desired)
c     of a real symmetric matrix.
c
c     on input
c
c        nm  must be set to the row dimension of the two-dimensional
c        array parameters as declared in the calling program
c        dimension statement.
c
c        n  is the order of the matrix  a.
c
c        a  contains the real symmetric matrix.
c
c        matz  is an integer variable set equal to zero if
c        only eigenvalues are desired.  otherwise it is set to
c        any non-zero integer for both eigenvalues and eigenvectors.
c
c     on output
c
c        w  contains the eigenvalues in ascending order.
c
c        z  contains the eigenvectors if matz is not zero (eigenvectors are in columns).
c
c        ierr  is an integer output variable set equal to an error
c           completion code described in the documentation for tqlrat
c           and tql2.  the normal completion code is zero.
c
c        fv1  and  fv2  are temporary storage arrays.
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
*/

    //c     .......... find eigenvalues only ..........

    tridiag(w,fv1,fv2);
//NOT WORKING?Matrix<double> z;
//NOT WORKING?(void) fv2;
//NOT WORKING?     tridiag(w,fv1,z);

    //*  tqlrat encounters catastrophic underflow on the Vax
    //*     call  tqlrat(n,w,fv2,ierr)

    return tql1(w,fv1);
}

template <class T>
int Matrix<T>::eig(Vector<T> &w, Matrix<T> &z, Vector<T> &fv1) const
{
//    const Matrix<T> &a = *this;

/*
      subroutine rs(nm,n,a,w,matz,z,fv1,fv2,ierr)
c
      integer n,nm,ierr,matz
      double precision a(nm,n),w(n),z(nm,n),fv1(n),fv2(n)
c
c     this subroutine calls the recommended sequence of
c     subroutines from the eigensystem subroutine package (eispack)
c     to find the eigenvalues and eigenvectors (if desired)
c     of a real symmetric matrix.
c
c     on input
c
c        nm  must be set to the row dimension of the two-dimensional
c        array parameters as declared in the calling program
c        dimension statement.
c
c        n  is the order of the matrix  a.
c
c        a  contains the real symmetric matrix.
c
c        matz  is an integer variable set equal to zero if
c        only eigenvalues are desired.  otherwise it is set to
c        any non-zero integer for both eigenvalues and eigenvectors.
c
c     on output
c
c        w  contains the eigenvalues in ascending order.
c
c        z  contains the eigenvectors if matz is not zero (eigenvectors are in columns).
c
c        ierr  is an integer output variable set equal to an error
c           completion code described in the documentation for tqlrat
c           and tql2.  the normal completion code is zero.
c
c        fv1  and  fv2  are temporary storage arrays.
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
*/

    //c     .......... find both eigenvalues and eigenvectors ..........

     tridiag(w,fv1,z);

     return tql2(w,fv1,z);
}

template <class T>
int Matrix<T>::projpsd(Matrix<T> &res, Vector<T> &w, Matrix<T> &z, Vector<T> &fv1) const
{
    const Matrix<T> &a = *this;

    int n = a.numRows();

    w.resize(n);
    z.resize(n,n);
    fv1.resize(n);

    int ierr = eig(w,z,fv1);

    if ( !ierr )
    {
        res.resize(n,n);
        res = 0.0;

        int i,j,k;

        for ( k = 0 ; k < n ; k++ )
        {
            if ( w(k) > 0.0 )
            {
                for ( i = 0 ; i < n ; i++ )
                {
                    for ( j = 0 ; j < n ; j++ )
                    {
                        res("&",i,j) += w(k)*z(i,k)*z(j,k);
                    }
                }
            }
        }
    }

    return ierr;
}

template <class T>
int Matrix<T>::projnsd(Matrix<T> &res, Vector<T> &w, Matrix<T> &z, Vector<T> &fv1) const
{
    const Matrix<T> &a = *this;

    NiceAssert( a.isSquare() );

    int n = a.numRows();

    w.resize(n);
    z.resize(n,n);
    fv1.resize(n);

    int ierr = eig(w,z,fv1);

    if ( !ierr )
    {
        res.resize(n,n);
        res = 0.0;

        int i,j,k;

        for ( k = 0 ; k < n ; k++ )
        {
            if ( w(k) < 0.0 )
            {
                for ( i = 0 ; i < n ; i++ )
                {
                    for ( j = 0 ; j < n ; j++ )
                    {
                        res("&",i,j) += w(k)*z(i,k)*z(j,k);
                    }
                }
            }
        }
    }

    return ierr;
}

template <class T>
Matrix<T> Matrix<T>::adj(void) const
{
    Matrix<T> res;

    adj(res);

    return res;
}

template <class T>
Matrix<T> Matrix<T>::inve(void) const
{
    Matrix<T> res;

    inve(res);

    return res;
}

template <class T>
Matrix<T> Matrix<T>::inveSymmNoConj(void) const
{
    Matrix<T> res;

    inveSymmNoConj(res);

    return res;
}

template <class T>
double Matrix<T>::getColNorm(int j) const
{
    int i;
    double result = 0;

    if ( numRows() )
    {
	result = norm2((*this)(0,j));

	if ( numRows() > 1 )
	{
	    for ( i = 1 ; i < numRows() ; i++ )
	    {
		result += norm2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
double Matrix<T>::getRowNorm(int i) const
{
    int j;
    double result = 0;

    if ( numCols() )
    {
	result = norm2((*this)(i,0));

	if ( numCols() > 1 )
	{
	    for ( j = 1 ; j < numCols() ; j++ )
	    {
		result += norm2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
double Matrix<T>::getRowColNorm(void) const
{
    int i,j;
    double result = 0;

    if ( numCols() && numRows() )
    {
        for ( i = 0 ; i < numCols() ; i++ )
        {
            for ( j = 0 ; j < numCols() ; j++ )
	    {
		result += norm2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
double Matrix<T>::getColAbs(int j) const
{
    int i;
    double result = 0;

    if ( numRows() )
    {
        result = abs2((*this)(0,j));

	if ( numRows() > 1 )
	{
	    for ( i = 1 ; i < numRows() ; i++ )
	    {
                result += abs2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
double Matrix<T>::getRowAbs(int i) const
{
    int j;
    double result = 0;

    if ( numCols() )
    {
        result = abs2((*this)(i,0));

	if ( numCols() > 1 )
	{
	    for ( j = 1 ; j < numCols() ; j++ )
	    {
                result += abs2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
double Matrix<T>::getRowColAbs(void) const
{
    int i,j;
    double result = 0;

    if ( numCols() && numRows() )
    {
        for ( i = 0 ; i < numCols() ; i++ )
        {
            for ( j = 0 ; j < numCols() ; j++ )
	    {
                result += abs2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
const Vector<T> &Matrix<T>::rowsum(Vector<T> &res) const
{
    res.resize(numCols());
    res.zero();

    if ( numRows() )
    {
        int i;

        for ( i = 0 ; i < numRows() ; i++ )
        {
            res += (*this)(i);
        }
    }

    return res;
}

template <class T>
const Vector<T> &Matrix<T>::colsum(Vector<T> &res) const
{
    res.resize(numRows());
    res.zero();

    if ( numCols() && numRows() )
    {
        int i,j;

        for ( i = 0 ; i < numCols() ; i++ )
        {
            for ( j = 0 ; j < numRows() ; j++ )
            {
                res("&",j) += (*this)(j,i);
            }
        }
    }

    return res;
}

template <class T>
const T &Matrix<T>::vertsum(int j, T &res) const
{
    setzero(res);

    if ( numRows() && numCols() )
    {
        int i;

        for ( i = 0 ; i < numRows() ; i++ )
        {
            res += (*this)(i,j);
        }
    }

    return res;
}

template <class T>
const T &Matrix<T>::horizsum(int i, T &res) const
{
    setzero(res);

    if ( numRows() && numCols() )
    {
        int j;

        for ( j = 0 ; j < numCols() ; j++ )
        {
            res += (*this)(i,j);
        }
    }

    return res;
}

template <class T>
template <class S> Matrix<T> &Matrix<T>::scale(const S &a)
{
    if ( numCols() && numRows() )
    {
        int i,j;

        for ( i = 0 ; i < numRows() ; i++ )
        {
            for ( j = 0 ; j < numCols() ; j++ )
            {
                (*this)("&",i,j) *= a;
            }
        }
    }

    return *this;
}

template <class T>
template <class S> Matrix<T> &Matrix<T>::scaleAdd(const S &a, const Matrix<T> &b)
{
    if ( shareBase(&b) )
    {
        Matrix<T> temp(b);

        scaleAdd(a,temp);
    }

    else
    {
        if ( !size() )
	{
            resize(b.numRows(),b.numCols());
            zero();
	}

        NiceAssert( ( numRows() == b.numRows() ) );
        NiceAssert( ( numCols() == b.numCols() ) );

        if ( numRows() && numCols() )
	{
            int i,j;

            for ( i = 0 ; i < numRows() ; i++ )
	    {
                for ( j = 0 ; j < numCols() ; j++ )
                {
                    (*this)("&",i,j) += (a*(b(i,j)));
                }
            }
	}
    }

    return *this;
}



































// Complexity

template <class T>
int Matrix<T>::shareBase(const Matrix<T> *that) const
{
    return ( bkref == that->bkref );
}




// Various functions

template <class T>
const T &max(const Matrix<T> &right_op, int &ii, int &jj)
{
    NiceAssert( right_op.numRows() && right_op.numCols() );

    int i,j;

    ii = 0;
    jj = 0;

    retVector<T> tmpva;

    for ( i = 0 ; i < right_op.numRows() ; i++ )
    {
        j = 0;

        if ( max(right_op(i,tmpva),j) > right_op(ii,jj) )
	{
            ii = i;
            jj = j;
	}
    }

    return right_op(ii,jj);
}

template <class T>
const T &min(const Matrix<T> &right_op, int &ii, int &jj)
{
    NiceAssert( right_op.numRows() && right_op.numCols() );

    int i,j;

    ii = 0;
    jj = 0;

    retVector<T> tmpva;

    for ( i = 0 ; i < right_op.numRows() ; i++ )
    {
        j = 0;

        if ( max(right_op(i,tmpva),j) < right_op(ii,jj) )
	{
            ii = i;
            jj = j;
	}
    }

    return right_op(ii,jj);
}

template <class T>
const T &maxdiag(const Matrix<T> &right_op, int &ii, int &jj)
{
    NiceAssert( right_op.numRows() && right_op.numCols() );
    NiceAssert( right_op.numRows() == right_op.numCols() );

    int i;

    ii = 0;
    jj = 0;

    for ( i = 0 ; i < right_op.numRows() ; i++ )
    {
        if ( right_op(i,i) > right_op(ii,jj) )
	{
            ii = i;
            jj = i;
	}
    }

    return right_op(ii,jj);
}

template <class T>
const T &mindiag(const Matrix<T> &right_op, int &ii, int &jj)
{
    NiceAssert( right_op.numRows() && right_op.numCols() );
    NiceAssert( right_op.numRows() == right_op.numCols() );

    int i;

    ii = 0;
    jj = 0;

    for ( i = 0 ; i < right_op.numRows() ; i++ )
    {
        if ( right_op(i,i) < right_op(ii,jj) )
	{
            ii = i;
            jj = i;
	}
    }

    return right_op(ii,jj);
}

template <class T>
T maxabs(const Matrix<T> &right_op, int &ii, int &jj)
{
    int dnumRows = right_op.numRows();
    int dnumCols = right_op.numCols();

    NiceAssert( dnumRows && dnumCols );

    (void) dnumCols;

    T maxrow;
    T maxval;
    int maxargi = 0;
    int maxargj = 0;

    maxval = abs2(right_op(0,0));

    int i,j;

    retVector<T> tmpva;

    for ( i = 0 ; i < dnumRows ; i++ )
    {
	maxrow = maxabs(right_op(i,tmpva),j);

	if ( maxrow > maxval )
	{
	    maxval  = maxrow;
	    maxargi = i;
            maxargj = j;
	}
    }

    ii = maxargi;
    jj = maxargj;

    return maxval;
}

template <class T>
T minabs(const Matrix<T> &right_op, int &ii, int &jj)
{
    int dnumRows = right_op.numRows();
    int dnumCols = right_op.numCols();

    NiceAssert( dnumRows && dnumCols );

    (void) dnumCols;

    T minrow;
    T minval;
    int minargi = 0;
    int minargj = 0;

    minval = abs2(right_op(0,0));

    int i,j;

    retVector<T> tmpva;

    for ( i = 0 ; i < dnumRows ; i++ )
    {
	minrow = minabs(right_op(i,tmpva),j);

	if ( minrow < minval )
	{
	    minval  = minrow;
	    minargi = i;
            minargj = j;
	}
    }

    ii = minargi;
    jj = minargj;

    return minval;
}

template <class T>
T maxabsdiag(const Matrix<T> &right_op, int &ii, int &jj)
{
    int dnumRows = right_op.numRows();
    int dnumCols = right_op.numCols();

    NiceAssert( dnumRows && dnumCols );
    NiceAssert( dnumRows == dnumCols );

    (void) dnumCols;

    T maxval;
    int maxargi = 0;

    maxval = abs2(right_op(0,0));

    int i;

    for ( i = 0 ; i < dnumRows ; i++ )
    {
	if ( abs2(right_op(i,i)) > maxval )
	{
	    maxval  = abs2(right_op(i,i));
	    maxargi = i;
	}
    }

    ii = maxargi;
    jj = maxargi;

    return maxval;
}

template <class T>
T minabsdiag(const Matrix<T> &right_op, int &ii, int &jj)
{
    int dnumRows = right_op.numRows();
    int dnumCols = right_op.numCols();

    NiceAssert( dnumRows && dnumCols );
    NiceAssert( dnumRows == dnumCols );

    (void) dnumCols;

    T minval;
    int minargi = 0;

    minval = abs2(right_op(0,0));

    int i;

    for ( i = 0 ; i < dnumRows ; i++ )
    {
	if ( abs2(right_op(i,i)) < minval )
	{
	    minval  = abs2(right_op(i,i));
	    minargi = i;
	}
    }

    ii = minargi;
    jj = minargi;

    return minval;
}



template <class T>
T sum(const Matrix<T> &right_op)
{
    int i;

    int size = right_op.numRows();
    T res;

    if ( size )
    {
	res = sum(right_op(0));

	if ( size > 1 )
	{
            retVector<T> tmpva;

	    for ( i = 1 ; i < size ; i++ )
	    {
		res += sum(right_op(i,tmpva));
	    }
	}
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <class T>
T mean(const Matrix<T> &right_op)
{
    T res;

    if ( ((right_op.numRows()) && (right_op.numCols())) )
    {
	res  = sum(right_op);
        res /= (double) ((right_op.numRows())*(right_op.numCols()));
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
const T &median(const Matrix<T> &right_op, int &miniv, int &minjv)
{
    if ( ( right_op.numRows() == 1 ) && ( right_op.numCols() == 1 ) )
    {
        miniv = 0;
        minjv = 0;

        return right_op(miniv,minjv);
    }

    else if ( right_op.numRows() && right_op.numCols() )
    {
        // Aim: right_op(outdexi,outdexj) should be arranged from largest to smallest

        Vector<int> outdexi;
        Vector<int> outdexj;

        int i,j,k;

	for ( i = 0 ; i < right_op.numRows() ; i++ )
	{
	    for ( j = 0 ; j < right_op.numCols() ; j++ )
	    {
                k = 0;

                if ( outdexi.size() )
                {
                    for ( k = 0 ; k < outdexi.size() ; k++ )
                    {
                        if ( right_op(outdexi(k),outdexj(k)) <= right_op(i,j) )
                        {
                            break;
                        }
                    }
                }

                outdexi.add(k);
                outdexi("&",k) = i;

                outdexj.add(k);
                outdexj("&",k) = j;
            }
        }

        miniv = outdexi(outdexi.size()/2);
        minjv = outdexj(outdexj.size()/2);

        return right_op(miniv,minjv);
    }

    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    svmvolatile static int frun = 1;
    svmvolatile static T defres;

    miniv = 0;
    minjv = 0;

    if ( frun )
    {
        setzero(const_cast<T &>(defres));
        frun = 0;
    }

    svm_mutex_unlock(eyelock);

    return const_cast<T &>(defres);
}

template <class T>
Matrix<T> outerProduct(const Vector<T> &left_op, const Vector<T> &right_op)
{
    Matrix<T> res(left_op.size(),right_op.size());

    T oneval; oneval = 1;

    res.zero();
    res.rankone(oneval,left_op,right_op);

    return res;
}

template <class T>
Matrix<T> outerProductNoConj(const Vector<T> &left_op, const Vector<T> &right_op)
{
    Matrix<T> res(left_op.size(),right_op.size());

    res.zero();
    res.rankoneNoConj(1,left_op,right_op);

    return res;
}

template <class T>
const Matrix<T> &takeProduct(Matrix<T> &res, const Matrix<T> &a, const Matrix<T> &b)
{
    NiceAssert( a.numCols() == b.numRows() );

    int numRows = a.numRows();
    int innerdim = a.numCols();
    int numCols = b.numCols();

    res.resize(numRows,numCols);
    res.zero();

    if ( numRows && numCols )
    {
        int i,j,k;

        for ( i = 0 ; i < numRows ; i++ )
        {
            for ( j = 0 ; j < numCols ; j++ )
            {
                for ( k = 0 ; k < innerdim ; k++ )
                {
                    res("&",i,j) += (a(i,k)*b(k,j));
                }
            }
        }
    }

    return res;
}


//template <class T> double abs2(const Matrix<T> &a) 
//{ 
//    return a.getRowColAbs(); 
//}

template <class T> double absF(const Matrix<T> &a) 
{ 
    return sqrt(normF(a)); 
}

template <class T> double normF(const Matrix<T> &a)
{
    int i,j;
    double result = 0;
    //T temp;

    if ( a.numCols() && a.numRows() )
    {
        for ( i = 0 ; i < a.numCols() ; i++ )
        {
            for ( j = 0 ; j < a.numCols() ; j++ )
	    {
		result = norm2(a(i,j));
	    }
	}
    }

    return result;
}

template <class T> double absd(const Matrix<T> &a) 
{ 
    return absF(a);
}

template <class T> double normd(const Matrix<T> &a)
{
    return normF(a);
}

template <class T> double distF(const Matrix<T> &a, const Matrix<T> &b)
{
    NiceAssert( a.numRows() == b.numRows() );
    NiceAssert( a.numCols() == b.numCols() );

    int i,j;
    double result = 0;
    T temp;

    if ( a.numCols() && a.numRows() )
    {
        for ( i = 0 ; i < a.numCols() ; i++ )
        {
            for ( j = 0 ; j < a.numCols() ; j++ )
	    {
                twoProduct(temp,a(i,j),b(i,j));

		result += norm2(a(i,j));
		result += norm2(b(i,j));
		result -= abs2(temp);
	    }
	}
    }

    return result;
}



template <class T> Matrix<T> &setident(Matrix<T> &a)
{
    return a.ident();
}

template <class T> Matrix<T> &setzero(Matrix<T> &a)
{
    return a.zero();
}

template <class T> Matrix<T> &setposate(Matrix<T> &a)
{
    return a.posate();
}

template <class T> Matrix<T> &setnegate(Matrix<T> &a)
{
    return a.negate();
}

template <class T> Matrix<T> &setconj(Matrix<T> &a)
{
    return a.conj();
}

template <class T> Matrix<T> &setrand(Matrix<T> &a)
{
    return a.rand();
}

template <class T> Matrix<T> &settranspose(Matrix<T> &a)
{
    return a.transpose();
}

template <class T> Matrix<T> inv(const Matrix<T> &src)
{
    return src.inve();
}




template <class T>
Matrix<T> &Matrix<T>::ident(void)
{
    if ( dnumRows && dnumCols )
    {
	int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
	    setzero((*this)("&",i,"&",tmpva));

	    if ( i < dnumCols )
	    {
		setident((*this)("&",i,i));
	    }
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::zero(void)
{
    if ( dnumRows && dnumCols )
    {
	int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            setzero((*this)("&",i,"&",tmpva));
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::posate(void)
{
    if ( dnumRows && dnumCols )
    {
	int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            setposate((*this)("&",i,"&",tmpva));
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::negate(void)
{
    if ( dnumRows && dnumCols )
    {
	int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            setnegate((*this)("&",i,"&",tmpva));
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::conj(void)
{
    static T dummy;

    if ( !isitadouble(dummy) && dnumRows && dnumCols )
    {
	int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            setconj((*this)("&",i,"&",tmpva));
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::rand(void)
{
    if ( dnumRows && dnumCols )
    {
	int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; i++ )
	{
            setrand((*this)("&",i,"&",tmpva));
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::transpose(void)
{
    if ( dnumRows || dnumCols )
    {
        int maxwide = ( dnumRows > dnumCols ) ? dnumRows : dnumCols;
	int newnumRows = dnumCols;
	int newnumCols = dnumRows;

	resize(maxwide,maxwide);

	//T temp;

	int i,j;

	for ( i = 0 ; i < maxwide ; i++ )
	{
	    for ( j = i ; j < maxwide ; j++ )
	    {
		if ( i != j )
		{
                    qswap((*this)("&",i,j),(*this)("&",j,i));
		    //temp              = (*this)(i,j);
		    //(*this)("&",i,j) = (*this)(j,i);
		    //(*this)("&",j,i) = temp;
		}
	    }
	}

        resize(newnumRows,newnumCols);
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::symmetrise(void)
{
    if ( dnumRows || dnumCols )
    {
        int maxwide = ( dnumRows > dnumCols ) ? dnumRows : dnumCols;
	int oldnumRows = dnumRows;
	int oldnumCols = dnumCols;

	resize(maxwide,maxwide);

	//T temp;

	int i,j;

        if ( oldnumRows != oldnumCols )
        {
            for ( i = 0 ; i < maxwide ; i++ )
            {
                for ( j = 0 ; j < maxwide ; j++ )
                {
                    if ( ( i >= oldnumRows ) || ( j >= oldnumCols ) )
                    {
                        setzero((*this)("&",i,j));
                    } 
                }
            }
	}

	for ( i = 0 ; i < maxwide ; i++ )
	{
	    for ( j = i+1 ; j < maxwide ; j++ )
	    {
                (*this)("&",i,j) += (*this)(j,i);
                (*this)("&",i,j) /= 2;
                (*this)("&",j,i)  = (*this)(i,j);
	    }
	}
    }

    return *this;
}




// Mathematical operator overloading

template <class T> Matrix<T>  operator+ (const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(left_op);

    return ( res += right_op );
}

template <class T> Matrix<T>  operator+ (const Matrix<T> &left_op, const T         &right_op)
{
    Matrix<T> res(left_op);

    return ( res += right_op );
}

template <class T> Matrix<T>  operator+ (const T         &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(right_op);

    return ( res += left_op );
}

template <class T> Matrix<T>  operator- (const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(left_op);

    return ( res -= right_op );
}

template <class T> Matrix<T>  operator- (const Matrix<T> &left_op, const T         &right_op)
{
    Matrix<T> res(left_op);

    return ( res -= right_op );
}

template <class T> Matrix<T>  operator- (const T         &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(right_op);
    setnegate(res);

    return ( res += left_op );
}

template <         class T> Matrix<T>  operator* (const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(left_op);

    return leftmult(res,right_op);
}

template <class S, class T> Vector<S>  operator* (const Vector<S> &left_op, const Matrix<T> &right_op)
{
    Vector<S> res(left_op);

    return leftmult(res,right_op);
}

template <         class T> Matrix<T>  operator* (const Matrix<T> &left_op, const T         &right_op)
{
    Matrix<T> res(left_op);

    return leftmult(res,right_op);
}

template <class S, class T> Vector<S>  operator* (const Matrix<T> &left_op, const Vector<S> &right_op)
{
    Vector<S> res(right_op);

    return rightmult(left_op,res);
}

template <         class T> Matrix<T>  operator* (const T         &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(right_op);

    return rightmult(left_op,res);
}

template <class T> Matrix<T> &operator*=(      Matrix<T> &left_op, const Matrix<T> &right_op)
{
    return leftmult(left_op,right_op);
}

template <class S, class T> Vector<S> &operator*=(      Vector<S> &left_op, const Matrix<T> &right_op)
{
    return leftmult(left_op,right_op);
}

template <class T> Matrix<T> &operator*=(      Matrix<T> &left_op, const T         &right_op)
{
    return leftmult(left_op,right_op);
}






template <class T> Matrix<T>  operator+(const Matrix<T> &left_op)
{
    int i;
    Matrix<T> res(left_op);

    if ( left_op.numRows() && left_op.numCols() )
    {
        retVector<T> tmpva;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
            setposate(res("&",i,tmpva));
	}
    }

    return res;
}

template <class T> Matrix<T>  operator-(const Matrix<T> &left_op)
{
    int i;
    Matrix<T> res(left_op);

    if ( left_op.numRows() && left_op.numCols() )
    {
        retVector<T> tmpva;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
            setnegate(res("&",i,tmpva));
	}
    }

    return res;
}

template <class T> Matrix<T> &operator+=(      Matrix<T> &left_op, const Matrix<T> &right_op)
{
    NiceAssert( ( left_op.numRows() == right_op.numRows() ) || left_op.isEmpty() || right_op.isEmpty() );
    NiceAssert( ( left_op.numCols() == right_op.numCols() ) || left_op.isEmpty() || right_op.isEmpty() );

    if ( left_op.numRows() && left_op.numCols() && !(left_op.isEmpty()) && !(right_op.isEmpty()) )
    {
        if ( left_op.shareBase(&right_op) )
        {
	    Matrix<T> temp(right_op);

            left_op += temp;
	}

	else
	{
	    int i;

            retVector<T> tmpva;
            retVector<T> tmpvb;

	    for ( i = 0 ; i < left_op.numRows() ; i++ )
	    {
		left_op("&",i,tmpva) += right_op(i,tmpvb);
	    }
        }
    }

    else if ( left_op.isEmpty() )
    {
        left_op = right_op;
    }

    return left_op;
}

template <class T> Matrix<T> &operator+=(      Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
	int i;
        int mindim = ( left_op.numRows() <= left_op.numCols() ) ? left_op.numRows() : left_op.numCols();

        for ( i = 0 ; i < mindim ; i++ )
	{
            left_op("&",i,i) += right_op;
	}
    }

    return left_op;
}

template <class T> Matrix<T> &operator-=(      Matrix<T> &left_op, const Matrix<T> &right_op)
{
    NiceAssert( ( left_op.numRows() == right_op.numRows() ) || left_op.isEmpty() || right_op.isEmpty() );
    NiceAssert( ( left_op.numCols() == right_op.numCols() ) || left_op.isEmpty() || right_op.isEmpty() );

    if ( left_op.numRows() && left_op.numCols() && !(left_op.isEmpty()) && !(right_op.isEmpty()) )
    {
        if ( left_op.shareBase(&right_op) )
        {
	    Matrix<T> temp(right_op);

            left_op += temp;
	}

	else
	{
	    int i;

            retVector<T> tmpva;
            retVector<T> tmpvb;

	    for ( i = 0 ; i < left_op.numRows() ; i++ )
	    {
                left_op("&",i,tmpva) -= right_op(i,tmpvb);
	    }
        }
    }

    else if ( left_op.isEmpty() )
    {
        left_op = -right_op;
    }

    return left_op;
}

template <class T> Matrix<T> &operator-=(      Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
	int i;
        int mindim = ( left_op.numRows() <= left_op.numCols() ) ? left_op.numRows() : left_op.numCols();

        for ( i = 0 ; i < mindim ; i++ )
	{
            left_op("&",i,i) -= right_op;
	}
    }

    return left_op;
}

template <class T> Matrix<T> &leftmult(      Matrix<T> &left_op, const Matrix<T> &right_op)
{
    NiceAssert( ( left_op.numCols() == right_op.numRows() ) || left_op.isEmpty() || right_op.isEmpty() );

    if ( left_op.isEmpty() )
    {
        left_op = right_op;
    }

    else if ( right_op.isEmpty() )
    {
        ;
    }

    else
    {
        int i,j;
        int innerdim = left_op.numCols();
        int resnumRows = left_op.numRows();
        int resnumCols = right_op.numCols();

        if ( resnumRows && resnumCols && innerdim )
        {
            if ( resnumCols > left_op.numCols() )
            {
                left_op.resize(left_op.numRows(),resnumCols);
            }

            Vector<T> rightcol(innerdim);
            Vector<T> leftrow(resnumCols);

            retVector<T> tmpva;
            retMatrix<T> tmpma;

            for ( i = 0 ; i < resnumRows ; i++ )
            {
                for ( j = 0 ; j < resnumCols ; j++ )
                {
                    //right_op.getCol(rightcol,j);
                    rightcol = right_op(0,1,right_op.numRows()-1,j,tmpma,"&");

                    twoProductNoConj(leftrow("&",j),left_op(i,0,1,innerdim-1,tmpva),rightcol);
                }

                left_op("&",i,zeroint(),1,resnumCols-1,tmpva) = leftrow;
            }

            //if ( resnumCols < left_op.numCols() )
            //{
            //    left_op.resize(resnumRows,left_op.numCols());
            //}
            left_op.resize(resnumRows,resnumCols);
        }

        else
        {
            left_op.resize(resnumRows,resnumCols);
            left_op.zero();
        }
    }

    return left_op;
}

template <class S, class T> Vector<S> &leftmult (      Vector<S> &left_op, const Matrix<T> &right_op)
{
    NiceAssert( ( left_op.size() == right_op.numRows() ) || right_op.isEmpty() );

    if ( right_op.isEmpty() )
    {
        ;
    }

    else
    {
        int i,j;
        int innerdim = right_op.numRows();
        int resnumCols = right_op.numCols();
        Vector<S> res(resnumCols);

        if ( resnumCols )
        {
//            Vector<T> rightcol(innerdim);

            retVector<T> tmpva;

            for ( i = 0 ; i < resnumCols ; i++ )
            {
/*
                //right_op.getCol(rightcol,i);
                rightcol = right_op(0,1,right_op.numRows()-1,i,tmpva,"&");

                twoProduct(res("&",i),left_op,rightcol); // this will conjugate left_op
*/
                setzero(res("&",i));

                for ( j = 0 ; j < innerdim ; j++ )
                {
                    res("&",i) += left_op(j)*right_op(j,i);
                }
            }
        }

        left_op = res;
    }

    return left_op;
}

template <         class T> Matrix<T> &leftmult (      Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
	int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    left_op("&",i,tmpva) *= right_op;
        }
    }

    return left_op;
}

template <class T> Matrix<T> &rightmult(const Matrix<T> &left_op, Matrix<T> &right_op)
{
    NiceAssert( ( left_op.numCols() == right_op.numRows() ) || left_op.isEmpty() || right_op.isEmpty() );

    if ( left_op.isEmpty() )
    {
        ;
    }

    else if ( right_op.isEmpty() )
    {
        right_op = left_op;
    }

    else
    {
        int i,j;
        int innerdim = left_op.numCols();
        int resnumRows = left_op.numRows();
        int resnumCols = right_op.numCols();

        if ( resnumRows && resnumCols && innerdim )
        {
            if ( resnumRows > right_op.numCols() )
            {
                right_op.resize(resnumRows,resnumCols);
            }

            Vector<T> rightcol(innerdim); // actual dimension doesn't matter here

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retMatrix<T> tmpma;

            for ( j = 0 ; j < resnumCols ; j++ )
            {
                //right_op.getCol(rightcol,j);
                rightcol = right_op(zeroint(),1,right_op.numRows()-1,j,tmpma,"&");

                for ( i = 0 ; i < resnumRows ; i++ )
                {
                    twoProductNoConj(right_op("&",i,j),left_op(i,tmpva),rightcol(zeroint(),1,innerdim-1,tmpvb));
                }
            }

            //if ( resnumRows < right_op.numCols() )
            //{
                right_op.resize(resnumRows,resnumCols);
            //}
        }

        else
        {
            right_op.resize(resnumRows,resnumCols);
            right_op.zero();
        }
    }

    return right_op;
}

template <class S, class T> Vector<S> &rightmult(const Matrix<T> &left_op,       Vector<S> &right_op)
{
    NiceAssert( ( left_op.numCols() == right_op.size() ) || left_op.isEmpty() );

    if ( left_op.isEmpty() )
    {
        ;
    }

    else
    {
        int i;
        //int innerdim = left_op.numCols();
        int resnumRows = left_op.numRows();
        Vector<S> res(resnumRows);

        if ( resnumRows )
        {
            retVector<T> tmpva;

            for ( i = 0 ; i < resnumRows ; i++ )
            {
                // Do this even if innerdim == 0 to ensure zeroing

                sumb(res("&",i),left_op(i,tmpva),right_op);
            }
        }

        right_op = res;
    }

    return right_op;
}

template <         class T> Matrix<T> &rightmult(const T         &left_op,       Matrix<T> &right_op)
{
    if ( right_op.numRows() && right_op.numCols() )
    {
	int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < right_op.numRows() ; i++ )
	{
	    right_op("&",i,tmpva) = ( left_op * right_op(i,tmpvb) );
        }
    }

    return right_op;
}

template <class T> Matrix<T> &mult(Matrix<T> &A, const Matrix<T> &B, const Matrix<T> &C)
{
    NiceAssert( ( B.numCols() == C.numRows() ) || B.isEmpty() || C.isEmpty() );

    if ( B.isEmpty() )
    {
        A = C;
    }

    else if ( C.isEmpty() )
    {
        A = B;
    }

    else
    {
        int i,j,k;
        int resnumRows = B.numRows();
        int innerdim   = B.numCols();
        int resnumCols = C.numCols();

        A.resize(resnumRows,resnumCols);

        if ( resnumRows && resnumCols && innerdim )
        {
            for ( i = 0 ; i < resnumRows ; i++ )
            {
                for ( j = 0 ; j < resnumCols ; j++ )
                {
                    setzero(A("&",i,j));

                    for ( k = 0 ; k < innerdim ; k++ )
                    {
                        A("&",i,j) += B(i,k)*C(k,j);
                    }
                }
            }
        }

        else
        {
            A.zero();
        }
    }

    return A;
}

template <class T> Vector<T> &mult(Vector<T> &a, const Vector<T> &b, const Matrix<T> &C)
{
    NiceAssert( ( b.size() == C.numRows() ) || C.isEmpty() );

    if ( C.isEmpty() )
    {
        a = b;
    }

    else
    {
        int i,k;
        int innerdim = b.size();
        int ressize  = C.numCols();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            for ( i = 0 ; i < ressize ; i++ )
            {
                setzero(a("&",i));

                for ( k = 0 ; k < innerdim ; k++ )
                {
                    a("&",i) += b(k)*C(k,i);
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}

template <class T> Vector<T> &mult(Vector<T> &a, const Matrix<T> &B, const Vector<T> &c)
{
    NiceAssert( ( B.numCols() == c.size() ) || B.isEmpty() );

    if ( B.isEmpty() )
    {
        a = c;
    }

    else
    {
        int i,k;
        int ressize  = B.numRows();
        int innerdim = c.size();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            for ( i = 0 ; i < ressize ; i++ )
            {
                setzero(a("&",i));

                for ( k = 0 ; k < innerdim ; k++ )
                {
                    a("&",i) += B(i,k)*c(k);
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}

template <class T> Vector<T> &multtrans(Vector<T> &a, const Vector<T> &b, const Matrix<T> &C)
{
    NiceAssert( ( b.size() == C.numCols() ) || C.isEmpty() );

    if ( C.isEmpty() )
    {
        a = b;
    }

    else
    {
        int i,k;
        int innerdim = b.size();
        int ressize  = C.numRows();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            for ( i = 0 ; i < ressize ; i++ )
            {
                setzero(a("&",i));

                for ( k = 0 ; k < innerdim ; k++ )
                {
                    a("&",i) += b(k)*C(i,k);
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}

template <class T> Vector<T> &multtrans(Vector<T> &a, const Matrix<T> &B, const Vector<T> &c)
{
    NiceAssert( ( B.numRows() == c.size() ) || B.isEmpty() );

    if ( B.isEmpty() )
    {
        a = c;
    }

    else
    {
        int i,k;
        int ressize  = B.numCols();
        int innerdim = c.size();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            for ( i = 0 ; i < ressize ; i++ )
            {
                setzero(a("&",i));

                for ( k = 0 ; k < innerdim ; k++ )
                {
                    a("&",i) += B(k,i)*c(k);
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}



template <class T> int operator==(const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    if ( ( left_op.numRows() != right_op.numRows() ) || ( left_op.numCols() != right_op.numCols() ) )
    {
        return 0;
    }

    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    if ( left_op(i,tmpva) != right_op(i,tmpvb) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator==(const Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    if ( left_op(i,tmpva) != right_op )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator==(const T         &left_op, const Matrix<T> &right_op)
{
    return ( right_op == left_op );
}

template <class T> int operator!=(const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator!=(const Matrix<T> &left_op, const T         &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator!=(const T         &left_op, const Matrix<T> &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator< (const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    if ( ( left_op.numRows() != right_op.numRows() ) || ( left_op.numCols() != right_op.numCols() ) )
    {
        return 0;
    }

    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    if ( !( left_op(i,tmpva) <  right_op(i,tmpvb) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator< (const Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    if ( !( left_op(i,tmpva) <  right_op ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator< (const T         &left_op, const Matrix<T> &right_op)
{
    if ( right_op.numRows() && right_op.numCols() )
    {
        int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < right_op.numRows() ; i++ )
	{
	    if ( !( left_op <  right_op(i,tmpva) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator<=(const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    if ( ( left_op.numRows() != right_op.numRows() ) || ( left_op.numCols() != right_op.numCols() ) )
    {
        return 0;
    }

    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    if ( !( left_op(i,tmpva) <= right_op(i,tmpvb) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator<=(const Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    if ( !( left_op(i,tmpva) <= right_op ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator<=(const T         &left_op, const Matrix<T> &right_op)
{
    if ( right_op.numRows() && right_op.numCols() )
    {
        int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < right_op.numRows() ; i++ )
	{
	    if ( !( left_op <= right_op(i,tmpva) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator> (const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    if ( ( left_op.numRows() != right_op.numRows() ) || ( left_op.numCols() != right_op.numCols() ) )
    {
        return 0;
    }

    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    if ( !( left_op(i,tmpva) >  right_op(i,tmpvb) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator> (const Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    if ( !( left_op(i,tmpva) >  right_op ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator> (const T         &left_op, const Matrix<T> &right_op)
{
    if ( right_op.numRows() && right_op.numCols() )
    {
        int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < right_op.numRows() ; i++ )
	{
	    if ( !( left_op >  right_op(i,tmpva) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator>=(const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    if ( ( left_op.numRows() != right_op.numRows() ) || ( left_op.numCols() != right_op.numCols() ) )
    {
        return 0;
    }

    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    if ( !( left_op(i,tmpva) >= right_op(i,tmpvb) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator>=(const Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < left_op.numRows() ; i++ )
	{
	    if ( !( left_op(i,tmpva) >= right_op ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator>=(const T         &left_op, const Matrix<T> &right_op)
{
    if ( right_op.numRows() && right_op.numCols() )
    {
        int i;

        retVector<T> tmpva;

	for ( i = 0 ; i < right_op.numRows() ; i++ )
	{
	    if ( !( left_op >= right_op(i,tmpva) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> Matrix<T> &randfill (Matrix<T> &res)
{
    return res.applyon(randfill);
}

template <class T> Matrix<T> &randnfill(Matrix<T> &res)
{
    return res.applyon(randnfill);
}

inline Matrix<double> &ltfill(Matrix<double> &lhsres, const Matrix<double> &rhs)
{
    NiceAssert( lhsres.numRows() == rhs.numRows() );
    NiceAssert( lhsres.numCols() == rhs.numCols() );

    if ( lhsres.numRows() && lhsres.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < lhsres.numRows() ; i++ )
        {
            for ( j = 0 ; j < lhsres.numCols() ; j++ )
            {
                lhsres("&",i,j) = ( lhsres(i,j) < rhs(i,j) ) ? 1 : 0;
            }
        }
    }

    return lhsres;
}

inline Matrix<double> &gtfill(Matrix<double> &lhsres, const Matrix<double> &rhs)
{
    NiceAssert( lhsres.numRows() == rhs.numRows() );
    NiceAssert( lhsres.numCols() == rhs.numCols() );

    if ( lhsres.numRows() && lhsres.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < lhsres.numRows() ; i++ )
        {
            for ( j = 0 ; j < lhsres.numCols() ; j++ )
            {
                lhsres("&",i,j) = ( lhsres(i,j) > rhs(i,j) ) ? 1 : 0;
            }
        }
    }

    return lhsres;
}

inline Matrix<double> &lefill(Matrix<double> &lhsres, const Matrix<double> &rhs)
{
    NiceAssert( lhsres.numRows() == rhs.numRows() );
    NiceAssert( lhsres.numCols() == rhs.numCols() );

    if ( lhsres.numRows() && lhsres.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < lhsres.numRows() ; i++ )
        {
            for ( j = 0 ; j < lhsres.numCols() ; j++ )
            {
                lhsres("&",i,j) = ( lhsres(i,j) <= rhs(i,j) ) ? 1 : 0;
            }
        }
    }

    return lhsres;
}

inline Matrix<double> &gefill(Matrix<double> &lhsres, const Matrix<double> &rhs)
{
    NiceAssert( lhsres.numRows() == rhs.numRows() );
    NiceAssert( lhsres.numCols() == rhs.numCols() );

    if ( lhsres.numRows() && lhsres.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < lhsres.numRows() ; i++ )
        {
            for ( j = 0 ; j < lhsres.numCols() ; j++ )
            {
                lhsres("&",i,j) = ( lhsres(i,j) >= rhs(i,j) ) ? 1 : 0;
            }
        }
    }

    return lhsres;
}


template <class T>
void preallocsubfn(Vector<T> &x, int newalloccols);

template <class T>
void preallocsubfn(Vector<T> &x, int newalloccols)
{
    x.prealloc(newalloccols);

    return;
}

template <class T>
void Matrix<T>::prealloc(int newallocrows, int newalloccols)
{
    NiceAssert( !nbase );
    NiceAssert( ( newallocrows >= 0 ) || ( newallocrows == -1 ) );
    NiceAssert( ( newalloccols >= 0 ) || ( newalloccols == -1 ) );

    if ( !iscover && content )
    {
        (*content).prealloc(newallocrows);
        (*content).applyOnAll(preallocsubfn,newalloccols);
    }

    return;
}













template <class T> int testisvnan(const Matrix<T> &x)
{
    int res = 0;

    if ( x.numRows() && x.numCols() )
    {
        int i,j;

        for ( i = 0 ; !res && ( i < x.numRows() ) ; i++ )
        {
            for ( j = 0 ; !res && ( j < x.numCols() ) ; j++ )
            {
                if ( testisvnan(x(i,j)) )
                {
                    res = 1;
                }
            }
        }
    }

    return res;
}

template <class T> int testisinf (const Matrix<T> &x)
{
    int res = 0;

    if ( x.numRows() && x.numCols() && !testisvnan(x) )
    {
        int i,j;

        for ( i = 0 ; !res && ( i < x.numRows() ) ; i++ )
        {
            for ( j = 0 ; !res && ( j < x.numCols() ) ; j++ )
            {
                if ( testisinf(x(i,j)) )
                {
                    res = 1;
                }
            }
        }
    }

    return res;
}

template <class T> int testispinf(const Matrix<T> &x)
{
    int pinfcnt = 0;

    if ( x.numRows() && x.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < x.numRows() ; i++ )
        {
            for ( j = 0 ; j < x.numCols() ; j++ )
            {
                if ( testispinf(x(i,j)) )
                {
                    pinfcnt++;
                }
            }
        }
    }

    return ( ( pinfcnt == ((x.numRows())*(x.numCols())) ) && ( ((x.numRows())*(x.numCols())) > 0 ) ) ? 1 : 0;
}

template <class T> int testisninf(const Matrix<T> &x)
{
    int ninfcnt = 0;

    if ( x.numRows() && x.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < x.numRows() ; i++ )
        {
            for ( j = 0 ; j < x.numRows() ; j++ )
            {
                if ( testisninf(x(i,j)) )
                {
                    ninfcnt++;
                }
            }
        }
    }

    return ( ( ninfcnt == ((x.numRows())*(x.numCols())) ) && ( ((x.numRows())*(x.numCols())) > 0 ) ) ? 1 : 0;
}










template <class T>
std::ostream &operator<<(std::ostream &output, const Matrix<T> &src)
{
    int numRows = src.numRows();
    int numCols = src.numCols();

    int i,j;

    output << "[ ";

    if ( numRows )
    {
	for ( i = 0 ; i < numRows ; i++ )
	{
	    if ( numCols )
	    {
		for ( j = 0 ; j < numCols ; j++ )
		{
		    if ( j < numCols-1 )
		    {
			output << src(i,j) << " \t";
		    }

		    else
		    {
			output << src(i,j) << " \t";
		    }
		}

		if ( i < numRows-1 )
		{
		    output << ";\n  ";
		}

		else
		{
		    output << "  ";
		}
	    }

	    else
	    {
                output << ";  ";
	    }
	}
    }

    else
    {
	if ( numCols )
	{
	    for ( j = 0 ; j < numCols ; j++ )
	    {
                output << ",  ";
	    }
	}
    }

    output << "]";

    return output;
}

template <class T>
std::istream &operator>>(std::istream &input, Matrix<T> &dest)
{
    char ss,tt;

    //OLD VERSION input >> buffer;
    //OLD VERSION 
    //OLD VERSION NiceAssert( !strcmp(buffer,"[") );

    while ( isspace(input.peek()) )
    {
	input.get(ss);
    }

    input.get(tt);

    NiceAssert( tt == '[' );

    int numRows = 0;
    int numCols = 0;
    int colcnt  = 0;
    int elmread = 0;

    while ( isspace(input.peek()) )
    {
        input.get(ss);
    }

    if ( input.peek() != ']' )
    {
        while ( input.peek() != ']' )
	{
            if ( input.peek() == ';' )
	    {
                input.get(tt);

		if ( !numRows && !colcnt )
		{
		    goto semicoloncount;
		}

                NiceAssert( elmread );
                (void) elmread;

		if ( !numRows )
		{
		    numCols = colcnt;
		}

                NiceAssert( colcnt == numCols );

		numRows++;

                elmread = 0;
		colcnt  = 0;
	    }

            else if ( input.peek() == ',' )
	    {
                input.get(tt);

		while ( isspace(input.peek()) )
		{
		    input.get(ss);
		}

		if ( !numRows && !colcnt )
		{
		    goto commacount;
		}

                NiceAssert( elmread );

                elmread = 0;
	    }

	    else
	    {
		if ( dest.numRows() == numRows )
		{
		    dest.addRow(numRows);
		}

		if ( ( dest.numCols() == colcnt ) && !numRows )
		{
		    dest.addCol(colcnt);
		}

		input >> dest("&",numRows,colcnt);

                elmread = 1;
		colcnt++;
	    }

            while ( isspace(input.peek()) )
            {
                input.get(ss);
            }
	}

	if ( !numRows )
	{
	    numCols = colcnt;
	}

        NiceAssert( colcnt == numCols );

	numRows++;
    }

    input.get(tt);

    NiceAssert( tt == ']' );

    dest.resize(numRows,numCols);

    return input;

semicoloncount:

    numCols = 0;
    numRows = 0;

    while ( tt == ';' )
    {
	numRows++;

	while ( isspace(input.peek()) )
	{
	    input.get(ss);
	}

        input.get(tt);
    }

    NiceAssert( tt == ']' );

    dest.resize(numRows,numCols);

    return input;

commacount:

    numCols = 0;
    numRows = 0;

    while ( tt == ',' )
    {
	numCols++;

	while ( isspace(input.peek()) )
	{
	    input.get(ss);
	}

        input.get(tt);
    }

    NiceAssert( tt == ']' );

    dest.resize(numRows,numCols);

    return input;
}


template <class T> 
std::istream &streamItIn(std::istream &input, Matrix<T>& dest, int processxyzvw)
{
    (void) processxyzvw;

    input >> dest;

    return input;
}

template <class T> 
std::ostream &streamItOut(std::ostream &output, const Matrix<T>& src, int retainTypeMarker)
{
    (void) retainTypeMarker;

    output << src;

    return output;
}

#endif

