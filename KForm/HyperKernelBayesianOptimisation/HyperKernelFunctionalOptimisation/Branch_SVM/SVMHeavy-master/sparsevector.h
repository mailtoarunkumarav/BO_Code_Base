
/*
 *  Sparse Vector class
 *  Copyright (C) 2010  Alistair Shilton
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


//
// Sparse vector class
//
// Version: 7
// Date: 18/09/2019
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _sparsevector_h
#define _sparsevector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <sstream>
#include "vector.h"
#include "strfns.h"

// Sparsevector UUID generator

inline unsigned int getnewID(void);

// offset for "far off" indices (used to store gradient constraint information)

#define INDNEAROFFSTART      0
#define INDFAROFFSTART       ((1*(INT_MAX/4)))
#define INDFAROFFEND         ((2*(INT_MAX/4))-1)
#define INDFARFAROFFSTART    ((2*(INT_MAX/4)))
#define INDFARFAROFFEND      ((3*(INT_MAX/4))-1)
#define INDFARFARFAROFFSTART ((3*(INT_MAX/4)))
#define INDFARFARFAROFFEND   ((4*(INT_MAX/4))-1)

// Separation of x vectors in multivector interpretation

#define DEFAULT_NUM_TUPLES         4096
#define DEFAULT_TUPLE_INDEX_STEP   (((1*(INT_MAX/4)))/DEFAULT_NUM_TUPLES)


template <class T> class Matrix;
template <class T> class SparseVector;

// Stream operators

template <class T> std::ostream &operator<<(std::ostream &output, const SparseVector<T> &src );
template <class T> std::ostream &printoneline(std::ostream &output, const SparseVector<T> &src );
template <class T> std::ostream &printnoparen(std::ostream &output, const SparseVector<T> &src );
template <class T> std::istream &operator>>(std::istream &input, SparseVector<T> &dest);
//template <class T> std::istream &streamItIn(std::istream &input, SparseVector<T> &dest, int processxyzvw = 1);
template <class T> std::istream &streamItInAlt(std::istream &input, SparseVector<T> &dest, int processxyzvw = 1, int removeZeros = 0);

// Tops

OVERLAYMAKEFNVECTOR(SparseVector<int>)
OVERLAYMAKEFNVECTOR(SparseVector<double>)

// Swap function

template <class T> void qswap(SparseVector<T> &a, SparseVector<T> &b);
template <class T> void qswap(const SparseVector<T> *&a, const SparseVector<T> *&b);
template <class T> void qswap(SparseVector<T> *&a, SparseVector<T> *&b);

// Lossy qswap functions - these lose indexing when applied
//NB: commented as not used, changes since may break this function
//
//template <class T> void lossyqswap(Vector<T> &a, SparseVector<T> &b);
//template <class T> void lossyqswap(SparseVector<T> &a, Vector<T> &b);

template <class T> SparseVector<T> &operator+=(SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T> &operator-=(SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T> &operator*=(SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T> &operator/=(SparseVector<T> &left_op, const SparseVector<T> &right_op);

// The class itself

template <class T>
class SparseVector
{
    template <class S> friend class SparseVector;
    friend class Matrix<T>;
    template <class S> friend void qswap(SparseVector<S> &a, SparseVector<S> &b);
//    template <class S> friend void lossyqswap(Vector<S> &a, SparseVector<S> &b);
//    template <class S> friend void lossyqswap(SparseVector<S> &a, Vector<S> &b);

    template <class S> friend SparseVector<S> &operator+=(SparseVector<S> &left_op, const SparseVector<S> &right_op);
    template <class S> friend SparseVector<S> &operator-=(SparseVector<S> &left_op, const SparseVector<S> &right_op);
    template <class S> friend SparseVector<S> &operator*=(SparseVector<S> &left_op, const SparseVector<S> &right_op);
    template <class S> friend SparseVector<S> &operator/=(SparseVector<S> &left_op, const SparseVector<S> &right_op);

    template <class S> friend std::ostream &operator<<(std::ostream &output, const SparseVector<S> &src );
    template <class S> friend std::ostream &printoneline(std::ostream &output, const SparseVector<S> &src );
    template <class S> friend std::ostream &printnoparen(std::ostream &output, const SparseVector<S> &src );

public:

    // Constructors and Destructors

                 SparseVector(const SparseVector<T> &src);
    svm_explicit SparseVector(const Vector<T> &src);
    svm_explicit SparseVector(int size = 0, const T *src = NULL);

    ~SparseVector();

    // Assignment
    //
    // - vector assignment: unless this vector is a temporary vector created
    //   to refer to parts of another vector then we do not require that sizes
    //   align but rather completely overwrite the destination, resetting the
    //   size to that of the source.
    // - scalar assignment: overwrites all *non-zero* elements.
    // - behaviour is undefined if scalar is an element of this.
    // - assignment from a matrix only possible for 1*d or d*1 matrix.
    //
    // overwrite: use elements in src to overwrite corresponding elements
    //   here but otherwise do not touch this vector.

    SparseVector<T> &operator=(const SparseVector<T> &src) { return assign(src); }
    SparseVector<T> &operator=(const Vector<T> &src);
    SparseVector<T> &operator=(const T &src);
    SparseVector<T> &operator=(const Matrix<T> &src);

    template <class S> SparseVector<T> &assign    (const SparseVector<S> &src);
    template <class S> SparseVector<T> &castassign(const SparseVector<S> &src);
    template <class S> SparseVector<T> &nearassign(const SparseVector<S> &src);

    SparseVector<T> &overwrite(const SparseVector<T> &src);
    SparseVector<T> &nearadd(const SparseVector<T> &src);

    // Simple vector manipulations
    //
    // ident:  apply setident() to all elements of the vector
    // zero:   apply setzero() to all elements of the vector
    //        (apply setzero() to element i if argument given).
    //         zeroing is achieved by removing elements from the sparsevector
    // zeromost: zero all elements except that specified.  Specifying -1
    //         indicates do not zero any elements
    // softzero: sets all elements zero without removing indices
    // posate: apply setposate() to all elements of the vector
    // negate: apply setnegate() to all elements of the vector
    // conj:   apply setconj() to all elements of the vector
    // rand:   apply .rand() to all elements of the vector
    // offset: amoff > 0: insert amoff elements at the start of the vector
    //         amoff < 0: remove amoff elements from the start of the vector
    //         offset simply adds amoff to the indices vector.  No bound
    //         checking is done to ensure non-negativity.
    // iprune: remove (prune) all elements not in index vector given
    // indalign: modify size/indices to match src
    //
    // each returns a reference to *this

    SparseVector<T> &ident(void);
    SparseVector<T> &zero(void);
    SparseVector<T> &zero(int i);
    SparseVector<T> &zero(const Vector<int> &i);
    SparseVector<T> &zeromost(int i);
    SparseVector<T> &softzero(void);
    SparseVector<T> &zeropassive(void);
    SparseVector<T> &posate(void);
    SparseVector<T> &negate(void);
    SparseVector<T> &conj(void);
    SparseVector<T> &rand(void);
    SparseVector<T> &offset(int amoff);
    SparseVector<T> &iprune(const Vector<int> &refind);
    SparseVector<T> &iprune(int refind);
    template <class S> SparseVector<T> &indalign(const SparseVector<S> &src);
    SparseVector<T> &indalign(const Vector<int> &srcind);

    // Zero/Overwrite relevant part of *this with *near* part of src
    //
    // if u present then relates to relevant ~ part

    SparseVector<T> &zeronear(void);
    SparseVector<T> &zerofar(void);
    SparseVector<T> &zerofarfar(void);
    SparseVector<T> &zerofarfarfar(void);

    SparseVector<T> &overwritenear(const SparseVector<T> &src);
    SparseVector<T> &overwritefar(const SparseVector<T> &src);
    SparseVector<T> &overwritefarfar(const SparseVector<T> &src);
    SparseVector<T> &overwritefarfarfar(const SparseVector<T> &src);

    SparseVector<T> &zeronear(int u);
    SparseVector<T> &zerofar(int u);
    SparseVector<T> &zerofarfar(int u);
    SparseVector<T> &zerofarfarfar(int u);

    SparseVector<T> &overwritenear(const SparseVector<T> &src, int u);
    SparseVector<T> &overwritefar(const SparseVector<T> &src, int u);
    SparseVector<T> &overwritefarfar(const SparseVector<T> &src, int u);
    SparseVector<T> &overwritefarfarfar(const SparseVector<T> &src, int u);

    // Access:
    //
    // - (...) operators as per vector.h.
    // - vector forms (i), ("&",i) have short-cut form.  (ind()) forms are
    //   very fast (they return direct reference to contents).
    // - direcref(i) is functionally equivalent to (ind(i)), but faster.
    // - direref(i) is functionally equivalent to ("&",ind(i)), but faster.
    //
    // Sparse vectors: - there is no bound-checking on i.  Instead the
    //                   element will be added if required during the
    //                   call ("&",i).
    //                 - direcref(i) and direref(i) are faster than (ind(i))
    //                   and ("&",ind(i)).
    //                 - if index vector has same address as index vector
    //                   then return is very fast

    Vector<T> &operator()(const char *dummy,                         retVector<T> &tmp);
    T         &operator()(const char *dummy, int i                                    );
    Vector<T> &operator()(const char *dummy, const Vector<int> &i,   retVector<T> &tmp);
    Vector<T> &operator()(const char *dummy, int ib, int is, int im, retVector<T> &tmp);

    const Vector<T> &operator()(                        retVector<T> &tmp) const;
    const T         &operator()(int i                                    ) const;
    const Vector<T> &operator()(const Vector<int> &i,   retVector<T> &tmp) const;
    const Vector<T> &operator()(int ib, int is, int im, retVector<T> &tmp) const;

    T         &direref(int i                                  ) { NiceAssert( ( i >= zeroint() ) && ( i < indsize() ) ); resetvecID(); killnearfar(); killaltcontent(); return (*content)("&",i);     }
    Vector<T> &direref(const Vector<int> &i, retVector<T> &tmp) { NiceAssert( ( i >= zeroint() ) && ( i < indsize() ) ); resetvecID(); killnearfar(); killaltcontent(); return (*content)("&",i,tmp); }

    const T         &direcref(int i                                  ) const { NiceAssert( ( i >= zeroint() ) && ( i < indsize() ) ); return (*content)(i);     }
    const Vector<T> &direcref(const Vector<int> &i, retVector<T> &tmp) const { NiceAssert( ( i >= zeroint() ) && ( i < indsize() ) ); return (*content)(i,tmp); }

    // Equivalents of () operators but for near, far, farfar and farfarfar

    T &n  (const char *dummy, int i);
    T &f  (const char *dummy, int i);
    T &ff (const char *dummy, int i);
    T &fff(const char *dummy, int i);

    const T &n  (int i) const;
    const T &f  (int i) const;
    const T &ff (int i) const;
    const T &fff(int i) const;

    // Information functions
    //
    // Note: indices relate to sparse vectors.  The index vector contains
    //       the positions of all non-zero elements in the vector, arranged
    //       from first to last.  For non-sparse vectors the index vector
    //       is simply ( 0 1 ... size()-1 ).
    //
    // ind(i):          returns the ith element of this vector
    // indsize():       returns the size of the index vector
    // findind(i):      looks for element in index vector (if present) and
    //                  return min(j): indices(j) >= i
    // isindpresent(i): return 1 if element is in index vector, 0 otherwise.
    // size():          returns indices(indsize()-1)
    //
    // order():         returns angry geckos
    // issparse():      returns true
    // nearnonsparse(): returns 1 if the index vector is ( 0 1 ... dim-1 )

    int ind(int pos)             const { return ((*indices)(pos)); }
    const Vector<int> &ind(void) const { return (*indices); }
    int indsize(void)            const { return (*indices).size(); }
    int findind(int i)           const;
    int isindpresent(int i)      const { int ii = findind(i); if ( ii >= indsize() ) { return 0; } return ind(ii) == i; }
    int size(void)               const { if ( indsize() ) { return ind(indsize()-1)+1; } return 0; }
    int order(void)              const { return ceilintlog2(size()); }
    int issparse(void)           const { return 1; }
    int nearnonsparse(void)      const { if ( indsize() ) { return ( size() == indsize() ) && ( (*indices)(zeroint()) == 0 ); } return 1; }

    // : on its own marks the start of a "far offset" vector part - that
    // is, part of the vector located roughly halfway through all the
    // possible integers (2^30 for 32 bit integers for example).  This is
    // helpful if you want to store "two vectors in one".
    //
    // This returns 0 if far-off not present, 1 if it is.
    //
    // nearindsize: number of elements < INDFAROFFSTART
    // farindsize:  number of elements >= INDFAROFFSTART

    int isnearindpresent(int i)      const { return isindpresent(i); }
    int isfarindpresent(int i)       const { return isindpresent(i+INDFAROFFSTART); }
    int isfarfarindpresent(int i)    const { return isindpresent(i+INDFARFAROFFSTART); }
    int isfarfarfarindpresent(int i) const { return isindpresent(i+INDFARFARFAROFFSTART); }

    int isnearindpresent(void)         const { return ( 0                             < findind(INDFAROFFSTART)       ) ? 1 : 0; }
    int isfaroffindpresent(void)       const { return ( findind(INDFAROFFSTART)       < findind(INDFARFAROFFSTART)    ) ? 1 : 0; }
    int isfarfaroffindpresent(void)    const { return ( findind(INDFARFAROFFSTART)    < findind(INDFARFARFAROFFSTART) ) ? 1 : 0; }
    int isfarfarfaroffindpresent(void) const { return ( findind(INDFARFARFAROFFSTART) < indsize()                     ) ? 1 : 0; }

    int nearindsize(void)          const { return findind(INDFAROFFSTART);       }
    int nearfarindsize(void)       const { return findind(INDFARFAROFFSTART);    }
    int nearfarfarindsize(void)    const { return findind(INDFARFARFAROFFSTART); }
    int nearfarfarfarindsize(void) const { return indsize();                     }

    int farindsize(void)       const { return findind(INDFARFAROFFSTART)   -findind(INDFAROFFSTART);       }
    int farfarindsize(void)    const { return findind(INDFARFARFAROFFSTART)-findind(INDFARFAROFFSTART);    }
    int farfarfarindsize(void) const { return indsize()                    -findind(INDFARFARFAROFFSTART); }

    int isnofaroffindpresent(void)  const { return ( findind(INDFAROFFSTART) == indsize() ) ? 1 : 0; }

    const SparseVector<T> &nearref     (void) const;
    const SparseVector<T> &farref      (void) const;
    const SparseVector<T> &farfarref   (void) const;
    const SparseVector<T> &farfarfarref(void) const;

    // Minors: return near/far part u (see ~)

    int nearupsize     (void) const;
    int farupsize      (void) const;
    int farfarupsize   (void) const;
    int farfarfarupsize(void) const;

    const SparseVector<T> &nearrefup     (int u) const;
    const SparseVector<T> &farrefup      (int u) const;
    const SparseVector<T> &farfarrefup   (int u) const;
    const SparseVector<T> &farfarfarrefup(int u) const;

    int nearrefupindsize     (int u) const;
    int farrefupindsize      (int u) const;
    int farfarrefupindsize   (int u) const;
    int farfarfarrefupindsize(int u) const;

    int nearrefupsize     (int u) const;
    int farrefupsize      (int u) const;
    int farfarrefupsize   (int u) const;
    int farfarfarrefupsize(int u) const;

    // Vector scaling:
    //
    // Apply (*this)("&",i) *= a for all i, or (*this)("&",i) *= a(i) for the
    // vectorial version.  This is useful for scaling vectors of vectors.

    template <class S> SparseVector<T> &scale(const S &a);
    template <class S> SparseVector<T> &scale(const SparseVector<S> &a);
    SparseVector<T> &lscale(const T &a);
    SparseVector<T> &lscale(const SparseVector<T> &a);

    // Scaled addition:
    //
    // The following is functionally equivalent to *this += (a*b).  However
    // it is quicker and uses less memory as no temporary variables are
    // constructed.

    template <class S> SparseVector<T> &scaleAdd  (const S &a, const SparseVector<T> &b);
    template <class S> SparseVector<T> &scaleAddR (const SparseVector<T> &a, const S &b);
    template <class S> SparseVector<T> &scaleAddB (const T &a, const SparseVector<S> &b);
    template <class S> SparseVector<T> &scaleAddBR(const SparseVector<S> &b, const T &a);

    // Add and remove element functions
    //
    // add:    ( c ) (i)          ( c ) (i)
    //         ( d ) (...)  ->    ( 0 ) (1)
    //                            ( d ) (...)
    // remove: ( c ) (i)          ( c ) (i)
    //         ( d ) (1)    ->    ( d ) (...)
    //         ( e ) (...)
    // resize: either add to end or remove from end until desired size is
    //         obtained.
    // append: add a to end of vector at position i >= size()
    //
    // Optional second argument: gives position in ind where element should
    //         go.  This can significantly speed up the operation.

    SparseVector<T> &add(int i)      { if ( !isindpresent(i) ) { (*this)("&",i) = zerelm(); } return *this; }
    SparseVector<T> &add(int i, int ipos);
    SparseVector<T> &remove(int i)   { zero(i);  return *this; }
    SparseVector<T> &resize(int i);
    template <class S> SparseVector<T> &resize(const SparseVector<S> &indexTemplateUsed) { return resize(indexTemplateUsed.ind()); }
    SparseVector<T> &resize(const Vector<int> &indexTemplateUsed);
    SparseVector<T> &setorder(int i);
    SparseVector<T> &append(int i) { add(i,indsize()); return *this; }
    SparseVector<T> &append(int i, const T &a);
    SparseVector<T> &append(int i, const SparseVector<T> &a);

    // Function application - apply function fn to each element of vector.

    SparseVector<T> &applyon(T (*fn)(T));
    SparseVector<T> &applyon(T (*fn)(const T &));
    SparseVector<T> &applyon(T (*fn)(T, const void *), const void *a);
    SparseVector<T> &applyon(T (*fn)(const T &, const void *), const void *a);
    SparseVector<T> &applyon(T &(*fn)(T &));
    SparseVector<T> &applyon(T &(*fn)(T &, const void *), const void *a);

    // Various swap functions
    //
    // blockswap  ( i < j ): ( c ) (i)          ( c ) (i)
    //                       ( e ) (1)    ->    ( d ) (j-i)
    //                       ( d ) (j-i)        ( e ) (1)
    //                       ( f ) (...)        ( f ) (...)
    //
    // blockswap  ( i > j ): ( c ) (j)          ( c ) (j)
    //                       ( d ) (i-j)  ->    ( e ) (1)
    //                       ( e ) (1)          ( d ) (i-j)
    //                       ( f ) (...)        ( f ) (...)
    //
    // squareswap ( i > j ): ( c ) (i)          ( c ) (i)
    //                       ( d ) (1)          ( f ) (1)
    //                       ( e ) (j-i)  ->    ( e ) (j-i)
    //                       ( f ) (1)          ( d ) (1)
    //                       ( g ) (...)        ( g ) (...)
    //
    // squareswap ( i < j ): ( c ) (j)          ( c ) (j)
    //                       ( d ) (1)          ( f ) (1)
    //                       ( e ) (i-j)  ->    ( e ) (i-j)
    //                       ( f ) (1)          ( d ) (1)
    //                       ( g ) (...)        ( g ) (...)

    SparseVector<T> &blockswap (int i, int j);
    SparseVector<T> &squareswap(int i, int j);

    // Vector pre-allocation and allocation strategy control

    void prealloc(int newallocsize);
    void useStandardAllocation(void);
    void useTightAllocation(void);
    void useSlackAllocation(void);
    void applyOnAll(void (*fn)(T &, int), int argx);
    int array_norm (void) const { return (*content).array_norm(); }
    int array_tight(void) const { return (*content).array_tight(); }
    int array_slack(void) const { return (*content).array_slack(); }

private:

    // Elements which index is in the indices vector will have the corresponding
    // value in the content vector.  All other elements in the vector have value
    // zerelm, which is set to zero and can only be accessed as a const.
    //
    // NOTE: the indices vector is ordered increasing.
    //       deliberate choice to not have child sparse vectors for simplicity.
    //       zerelm is stored at end of content vector

    Vector<int> *indices;
    Vector<T> *content;
    //T zerelm; - NB: this is not element indsize in content

    const T &zerelm(void) const { return (*content)((*content).size()-1); }

    SparseVector<T> *nearpointer;      // first element is complete near vector, rest are upsize elements
    SparseVector<T> *farpointer;       // first element is complete far vector, rest are upsize elements
    SparseVector<T> *farfarpointer;    // first element is complete farfar vector, rest are upsize elements
    SparseVector<T> *farfarfarpointer; // first element is complete farfarfar vector, rest are upsize elements

    SparseVector<T> *thisthis;
    SparseVector<T> **thisthisthis;

    // Private constructor that initialises exactly nothing

    svm_explicit SparseVector(const char *dummy);

    // Unique ID vector

    unsigned int xvecID;

public:
    unsigned int vecID(void) const { return xvecID; }
    void resetvecID(void)          { xvecID = getnewID(); return; }
    void setvecID(unsigned int nv) { xvecID = nv;         return; }

public:

    // Suppose for example T = Vector<double>.  In that case it might be
    // important to make sure the size of the "zero element" is appropriately
    // set.  Hence the following function that you really shouldn't use
    // except as a last resort.

    T &zerelmunsafedontuse(void) { resetvecID(); killnearfar(); killaltcontent(); return (*content)("&",(*content).size()-1); }

    // DON'T USE DIRECTLY.  This is used for speeding up a specialisation to 
    // SparseVector<gentype>.  If the vector is non-sparse and simple then this
    // is allocated and used in things like inner products etc for speed.

    double *altcontent;

    void killaltcontent(void)
    {
        if ( altcontent )
        {
            MEMDELARRAY(altcontent);
            altcontent = NULL;
        }

        return;
    }

    void killnearfar(void)
    {
        killnear();
        killfar();
        killfarfar();
        killfarfarfar();

        return;
    }

    void killnear(void)
    {
        if ( nearpointer )
        {
            MEMDELARRAY(nearpointer);
            nearpointer = NULL;
        }

        return;
    }

    void killfar(void)
    {
        if ( farpointer )
        {
            MEMDELARRAY(farpointer);
            farpointer = NULL;
        }

        return;
    }

    void killfarfar(void)
    {
        if ( farfarpointer )
        {
            MEMDELARRAY(farfarpointer);
            farfarpointer = NULL;
        }

        return;
    }

    void killfarfarfar(void)
    {
        if ( farfarfarpointer )
        {
            MEMDELARRAY(farfarfarpointer);
            farfarfarpointer = NULL;
        }

        return;
    }

    void makealtcontent(void); // construct altcontent *if possible*
};

template <class T>
void SparseVector<T>::makealtcontent(void)
{
    //killaltcontent();

    // In the default case there is nothing to be done

    return;
}

template <class T> void qswap(SparseVector<T> &a, SparseVector<T> &b)
{
    Vector<int> *indices;
    Vector<T> *content;

    indices = a.indices; a.indices = b.indices; b.indices = indices;
    content = a.content; a.content = b.content; b.content = content;

    SparseVector<T> *nearpointer;
    SparseVector<T> *farpointer;
    SparseVector<T> *farfarpointer;
    SparseVector<T> *farfarfarpointer;
    double *altcontent;
    int altxvecID;

    nearpointer      = a.nearpointer;      a.nearpointer      = b.nearpointer;      b.nearpointer      = nearpointer;
    farpointer       = a.farpointer;       a.farpointer       = b.farpointer;       b.farpointer       = farpointer;
    farfarpointer    = a.farfarpointer;    a.farfarpointer    = b.farfarpointer;    b.farfarpointer    = farfarpointer;
    farfarfarpointer = a.farfarfarpointer; a.farfarfarpointer = b.farfarfarpointer; b.farfarfarpointer = farfarfarpointer;
    altcontent       = a.altcontent;       a.altcontent       = b.altcontent;       b.altcontent       = altcontent;
    altxvecID        = a.xvecID;           a.xvecID           = b.xvecID;           b.xvecID           = altxvecID;

    return;
}


//template <class T> void lossyqswap(Vector<T> &a, SparseVector<T> &b)
//{
//    int asize = a.size();
//
//    qswap(a,(*(b.content)));
//    (*(b.indices)) = cntintvec(asize);
//
//    // Bit of extra stuff to deal with zerelm in content
//
//    (*(b.content)).add((*(b.content)).size());
//    qswap((*(b.content))("&",(*(b.content)).size()-1),a("&",a.size()-1));
//    a.remove(a.size()-1);
//
//    return;
//}
//
//template <class T> void lossyqswap(SparseVector<T> &a, Vector<T> &b)
//{
//    lossyqswap(b,a);
//
//    return;
//}
















template <class T> void qswap(const SparseVector<T> *&a, const SparseVector<T> *&b)
{
    const SparseVector<T> *x;

    x = a; a = b; b = x;

    return;
}

template <class T> void qswap(SparseVector<T> *&a, SparseVector<T> *&b)
{
    SparseVector<T> *x;

    x = a; a = b; b = x;

    return;
}





// Various functions
//
// max: find max element, put index in i.  If two vectors are given then finds max a-b
// min: find min element, put index in i.  If two vectors are given then finds min a-b
// maxabs: find the |max| element, put index in i.
// minabs: find the |min| element, put index in i.
// sqabsmax: find the |max|*|max| element, put index in i.
// sqabsmin: find the |min|*|min| element, put index in i.
// sum: find the sum of elements in a vector
// prod: find the product of elements in a vector, top to bottom
// Prod: find the product of elements in a vector, bottom to top
// mean: calculate the mean of.  Ill-defined if vector empty.
// median: calculate the median.  Put the index into i.
//
// twoProduct: calculate the inner product of two vectors a'.b
// twoProductNoConj: calculate the inner product of two vectors but without conjugating a
// twoProductRevConj: calculate the inner product of two vectors conj(a)'.conj(b)
// fourProduct: calculate the four-product sum_i (a_i.b_i).(c_i.d_i)  (note order of operation)
// mProduct: calculate the m-product sum_i prod_j (a_{2j,i}.a_{2j,i+1})  (note order of operation)
// Indexed versions: a -> a(n), b -> b(n), c -> c(n), ....  It is assumed that n is sorted from smallest to largest.
// Scaled versions: a -> a./scale, b -> b./scale, c -> c./scale ...
//
// faroff: 0 = use indices 0 to INDNEAREND
//         1 = use indices INDFAROFFSTART to INDFAROFFEND
//         2 = use indices INDFARFAROFFSTART to INDFARFAROFFEND
//         3 = use indices INDFARFARFAROFFSTART to INDFARFARFAROFFEND
//
// setident: call a.ident()
// setzero: call a.zero()
// setposate: call a.posate()
// setnegate: call a.negate()
// setconj: call a.conj()
//
// angle:    calculate a/abs2(a) (0 if abs2(0) = 0)
// vangle:   calculate a/abs2(a) (defsign if abs2(0) = 0
// seteabs2: calculate the elemetwise absolute of a (ie a_i = ||a_i||)
//
// abs2:   return the square root of the 2-norm of the vector
// abs1:   return the 1-norm of the vector
// absp:   return the p-root of the p-norm of the vector
// absinf: return the inf-norm of the vector
// abs0:   return the 0-norm of the vector
// norm2:  return the 2-norm of the vector ||a||_2^2
// norm1:  return the 1-norm of the vector ||a||_1
// normp:  return the p-norm of the vector ||a||_p^p

template <class T> T max     (const SparseVector<T> &a, const SparseVector<T> &b, int &i);
template <class T> T min     (const SparseVector<T> &a, const SparseVector<T> &b, int &i);
template <class T> T maxabs  (const SparseVector<T> &a, int &i);
template <class T> T minabs  (const SparseVector<T> &a, int &i);
template <class T> T sqabsmax(const SparseVector<T> &a);
template <class T> T sqabsmin(const SparseVector<T> &a);
template <class T> T sum     (const SparseVector<T> &a);
template <class T> T sqsum   (const SparseVector<T> &a);
template <class T> T prod    (const SparseVector<T> &a);
template <class T> T Prod    (const SparseVector<T> &a);
template <class T> T mean    (const SparseVector<T> &a);
template <class T> T sqmean  (const SparseVector<T> &a);
template <class T> T vari    (const SparseVector<T> &a);
template <class T> T stdev   (const SparseVector<T> &a);

template <class T> const T &max   (const SparseVector<T> &a, int &i);
template <class T> const T &min   (const SparseVector<T> &a, int &i);
template <class T> const T &median(const SparseVector<T> &a, int &i);

template <class T> const T &max     (T &res, const SparseVector<T> &a, const SparseVector<T> &b, int &i);
template <class T> const T &min     (T &res, const SparseVector<T> &a, const SparseVector<T> &b, int &i);
template <class T> const T &maxabs  (T &res, const SparseVector<T> &a, int &i);
template <class T> const T &minabs  (T &res, const SparseVector<T> &a, int &i);
template <class T> const T &sqabsmax(T &res, const SparseVector<T> &a);
template <class T> const T &sqabsmin(T &res, const SparseVector<T> &a);
template <class T> const T &sum     (T &res, const SparseVector<T> &a);
template <class T> const T &sqsum   (T &res, const SparseVector<T> &a);
template <class T> const T &prod    (T &res, const SparseVector<T> &a);
template <class T> const T &Prod    (T &res, const SparseVector<T> &a);
template <class T> const T &mean    (T &res, const SparseVector<T> &a);
template <class T> const T &sqmean  (T &res, const SparseVector<T> &a);
template <class T> const T &vari    (T &res, const SparseVector<T> &a);
template <class T> const T &stdev   (T &res, const SparseVector<T> &a);

template <class T> const T &indexedsum   (T &res, const Vector<int> &n, const SparseVector<T> &a);
template <class T> const T &indexedsqsum (T &res, const Vector<int> &n, const SparseVector<T> &a);
template <class T> const T &indexedmean  (T &res, const Vector<int> &n, const SparseVector<T> &a);
template <class T> const T &indexedmedian(        const Vector<int> &n, const SparseVector<T> &a, int &i);
template <class T> const T &indexedsqmean(T &res, const Vector<int> &n, const SparseVector<T> &a);
template <class T> const T &indexedvari  (T &res, const Vector<int> &n, const SparseVector<T> &a);
template <class T> const T &indexedstdev (T &res, const Vector<int> &n, const SparseVector<T> &a);
template <class T> const T &indexedmax   (        const Vector<int> &n, const SparseVector<T> &a, int &i);
template <class T> const T &indexedmin   (        const Vector<int> &n, const SparseVector<T> &a, int &i);
template <class T> const T &indexedmaxabs(T &res, const Vector<int> &n, const SparseVector<T> &a, int &i);

template <class T> T &twoProductNoConj                   (T &res,                       const SparseVector<T> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductRevConj                  (T &res,                       const SparseVector<T> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductScaledNoConj             (T &res,                       const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductScaledRevConj            (T &res,                       const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductLeftScaled               (T &res,                       const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductLeftScaledNoConj         (T &res,                       const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductLeftScaledRevConj        (T &res,                       const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductRightScaled              (T &res,                       const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductRightScaledNoConj        (T &res,                       const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductRightScaledRevConj       (T &res,                       const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedtwoProductNoConj            (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedtwoProductRevConj           (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedtwoProductScaledNoConj      (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedtwoProductScaledRevConj     (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedtwoProductLeftScaled        (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedtwoProductLeftScaledNoConj  (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedtwoProductLeftScaledRevConj (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedtwoProductRightScaled       (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedtwoProductRightScaledNoConj (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedtwoProductRightScaledRevConj(T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);

template <class T> T &oneProduct  (T &res, const SparseVector<T> &a);
template <class T> T &twoProduct  (T &res, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff = 0, int bfaroff = 0);
template <class T> T &threeProduct(T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c);
template <class T> T &fourProduct (T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d);
template <class T> T &mProduct    (T &res, const Vector<const SparseVector <T> *> &a);

template <class T> double &oneProductAssumeReal  (double &res, const SparseVector<T> &a);
template <class T> double &twoProductAssumeReal(double &res, const SparseVector<T> &a, const SparseVector<T> &b);
template <class T> double &threeProductAssumeReal(double &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c);
template <class T> double &fourProductAssumeReal (double &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d);
template <class T> double &mProductAssumeReal    (double &res, const Vector<const SparseVector <T> *> &a);

template <class T> T &oneProductScaled  (T &res, const SparseVector<T> &a, const SparseVector<T> &scale);
template <class T> T &twoProductScaled  (T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &threeProductScaled(T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &scale);
template <class T> T &fourProductScaled (T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d, const SparseVector<T> &scale);
template <class T> T &mProductScaled    (T &res, const Vector<const SparseVector <T> *> &a, const SparseVector<T> &scale);


template <class T> T &oneProductPrelude  (T &res, const SparseVector<T> &a);
template <class T> T &twoProductPrelude  (T &res, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff = 0, int bfaroff = 0);
template <class T> T &threeProductPrelude(T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c);
template <class T> T &fourProductPrelude (T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d);
template <class T> T &mProductPrelude    (T &res, const Vector<const SparseVector <T> *> &a);

template <class T> double &oneProductAssumeRealPrelude  (double &res, const SparseVector<T> &a);
template <class T> double &twoProductAssumeRealPrelude  (double &res, const SparseVector<T> &a, const SparseVector<T> &b);
template <class T> double &threeProductAssumeRealPrelude(double &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c);
template <class T> double &fourProductAssumeRealPrelude (double &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d);
template <class T> double &mProductAssumeRealPrelude    (double &res, const Vector<const SparseVector <T> *> &a);

template <class T> T &oneProductScaledPrelude  (T &res, const SparseVector<T> &a, const SparseVector<T> &scale);
template <class T> T &twoProductScaledPrelude  (T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &threeProductScaledPrelude(T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &scale);
template <class T> T &fourProductScaledPrelude (T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d, const SparseVector<T> &scale);
template <class T> T &mProductScaledPrelude    (T &res, const Vector<const SparseVector <T> *> &a, const SparseVector<T> &scale);

template <class T> T &twoProductNoConjPrelude (T &res, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductRevConjPrelude(T &res, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff = 0, int bfaroff = 0);

template <class T> T &twoProductScaledNoConjPrelude (T &res, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff = 0, int bfaroff = 0);
template <class T> T &twoProductScaledRevConjPrelude(T &res, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff = 0, int bfaroff = 0);


template <class T> T &indexedoneProduct  (T &res, const Vector<int> &n, const SparseVector<T> &a);
template <class T> T &indexedtwoProduct  (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedthreeProduct(T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c);
template <class T> T &indexedfourProduct (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d);
template <class T> T &indexedmProduct    (T &res, const Vector<int> &n, const Vector<const SparseVector <T> *> &a);

template <class T> T &indexedoneProductScaled  (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &scale);
template <class T> T &indexedtwoProductScaled  (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff = 0, int bfaroff = 0);
template <class T> T &indexedthreeProductScaled(T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &scale);
template <class T> T &indexedfourProductScaled (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d, const SparseVector<T> &scale);
template <class T> T &indexedmProductScaled    (T &res, const Vector<int> &n, const Vector<const SparseVector <T> *> &a, const SparseVector<T> &scale);

// <a^ia,b^ib,c^ic,d^id>

template <class T> T &fourProductPow(T &res, const SparseVector<T> &a, int ia, const SparseVector<T> &b, int ib, const SparseVector<T> &c, int ic, const SparseVector<T> &d, int id);
template <class T> T &threeProductPow(T &res, const SparseVector<T> &a, int ia, const SparseVector<T> &b, int ib, const SparseVector<T> &c, int ic);

template <class T> SparseVector<T> &setident(SparseVector<T> &a);
template <class T> SparseVector<T> &setzero(SparseVector<T> &a);
template <class T> SparseVector<T> &setposate(SparseVector<T> &a);
template <class T> SparseVector<T> &setnegate(SparseVector<T> &a);
template <class T> SparseVector<T> &setconj(SparseVector<T> &a);
template <class T> SparseVector<T> &setrand(SparseVector<T> &a);
template <class T> SparseVector<T> &postProInnerProd(SparseVector<T> &a) { return a; }

template <class T> SparseVector<T> *&setident (SparseVector<T> *&a) { throw("sdk;jlawdf"); return a; }
template <class T> SparseVector<T> *&setzero  (SparseVector<T> *&a) { return a = NULL; }
template <class T> SparseVector<T> *&setposate(SparseVector<T> *&a) { return a; }
template <class T> SparseVector<T> *&setnegate(SparseVector<T> *&a) { throw("sdk;jlawdf"); return a; }
template <class T> SparseVector<T> *&setconj  (SparseVector<T> *&a) { throw("sdk;jlawdf"); return a; }
template <class T> SparseVector<T> *&setrand  (SparseVector<T> *&a) { throw("sdk;jlawdf"); return a; }
template <class T> SparseVector<T> *&postProInnerProd(SparseVector<T> *&a) { return a; }

template <class T> const SparseVector<T> *&setident (const SparseVector<T> *&a) { throw("sdk;jlawdf"); return a; }
template <class T> const SparseVector<T> *&setzero  (const SparseVector<T> *&a) { return a = NULL; }
template <class T> const SparseVector<T> *&setposate(const SparseVector<T> *&a) { return a; }
template <class T> const SparseVector<T> *&setnegate(const SparseVector<T> *&a) { throw("sdk;jlawdf"); return a; }
template <class T> const SparseVector<T> *&setconj  (const SparseVector<T> *&a) { throw("sdk;jlawdf"); return a; }
template <class T> const SparseVector<T> *&setrand  (const SparseVector<T> *&a) { throw("sdk;jlawdf"); return a; }
template <class T> const SparseVector<T> *&postProInnerProd(const SparseVector<T> *&a) { return a; }

template <class S> SparseVector<S> angle (const SparseVector<S> &a);
template <class S> SparseVector<S> vangle(const SparseVector<S> &a, const SparseVector<S> &defsign);
template <class S> SparseVector<double> &seteabs2(SparseVector<S> &a);

template <class S> double abs1  (const SparseVector<S> &a);
template <class S> double abs2  (const SparseVector<S> &a);
template <class S> double absp  (const SparseVector<S> &a, double p);
template <class S> double absinf(const SparseVector<S> &a);
template <class S> double abs0  (const SparseVector<S> &a);
template <class S> double abs2  (const SparseVector<S> &a);
template <class S> double norm1 (const SparseVector<S> &a);
template <class S> double norm2 (const SparseVector<S> &a);
template <class S> double normp (const SparseVector<S> &a, double p);
template <class S> double normd (const SparseVector<S> &a);

template <class S> double absinfPrelude(const SparseVector<S> &a);
template <class S> double abs0Prelude  (const SparseVector<S> &a);
template <class S> double norm1Prelude (const SparseVector<S> &a);
template <class S> double norm2Prelude (const SparseVector<S> &a);
template <class S> double normpPrelude (const SparseVector<S> &a, double p);


// Random permutation function and random fill

template <class T> SparseVector<T> &randfill (SparseVector<T> &res);
template <class T> SparseVector<T> &randnfill(SparseVector<T> &res);


// Conversion from strings

template <class T> SparseVector<T> &atoSparseVector(SparseVector<T> &dest, const std::string &src);

// Mathematical operator overloading
//
// NB: in general it is wise to avoid use of non-assignment operators (ie.
//     those which do not return a reference) as there may be a
//     computational hit when constructors (and possibly copy constructors)
//     are called.
//
// +  posation          - unary, return rvalue
// -  negation          - unary, return rvalue
//
// NB: - all unary operators are elementwise

template <class T> SparseVector<T>  operator+ (const SparseVector<T> &left_op);
template <class T> SparseVector<T>  operator- (const SparseVector<T> &left_op);

// +  addition           - binary, return rvalue
// -  subtraction        - binary, return rvalue
// *  multiplication     - binary, return rvalue
// /  division           - binary, return rvalue
// %  modulo             - binary, return rvalue
//
// NB: - adding a scalar to a vector adds the scalar to all elements of the
//       vector.
//     - we don't assume commutativity over T, so division is not well defined
//     - multiplying two vectors performs element-wise multiplication.
//     - division: vector/vector will do elementwise division                (and return reference to left_op)
//                 vector/scalar will do right division (vector*inv(scalar)) (and return reference to left_op)
//                 scalar/vector will do left division (inv(scalar)*vector)  (and return reference to right_op)
//     - for sparse vectors, division and modulus operators assume 0/... = 0
//     - for sparse vectors, adding/subtracting a scalar only affects non-zero elements

template <class T> SparseVector<T>  operator+ (const SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T>  operator- (const SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T>  operator* (const SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T>  operator/ (const SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T>  operator% (const SparseVector<T> &left_op, const SparseVector<T> &right_op);

template <class T> SparseVector<T>  operator+ (const SparseVector<T> &left_op, const T               &right_op);
template <class T> SparseVector<T>  operator- (const SparseVector<T> &left_op, const T               &right_op);
template <class T> SparseVector<T>  operator* (const SparseVector<T> &left_op, const T               &right_op);
template <class T> SparseVector<T>  operator/ (const SparseVector<T> &left_op, const T               &right_op);
template <class T> SparseVector<T>  operator% (const SparseVector<T> &left_op, const T               &right_op);

template <class T> SparseVector<T>  operator+ (const T               &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T>  operator- (const T               &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T>  operator* (const T               &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T>  operator/ (const T               &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T>  operator% (const T               &left_op, const SparseVector<T> &right_op);

// +=  additive       assignment - binary, return lvalue
// -=  subtractive    assignment - binary, return lvalue
// *=  multiplicative assignment - binary, return lvalue
// /=  divisive       assignment - binary, return lvalue
// %=  modulo         assignment - binary, return lvalue
//
// NB: - adding a scalar to a vector adds the scalar to all elements of the
//       vector.
//     - left-shift and right-shift operate elementwise.
//     - when left_op is not a vector, the result is stored in right_op and returned as a reference
//     - it is assumed that addition and subtraction are commutative
//     - scalar /= vector does left division (that is, vector = inv(scalar)*vector).
//     - for sparse vectors, division and modulus operators assume 0/... = 0
//     - for sparse vectors, adding/subtracting a scalar only affects non-zero elements

//template <class T> SparseVector<T> &operator+=(      SparseVector<T> &left_op, const SparseVector<T> &right_op);
//template <class T> SparseVector<T> &operator-=(      SparseVector<T> &left_op, const SparseVector<T> &right_op);
//template <class T> SparseVector<T> &operator*=(      SparseVector<T> &left_op, const SparseVector<T> &right_op);
//template <class T> SparseVector<T> &operator/=(      SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T> &operator%=(      SparseVector<T> &left_op, const SparseVector<T> &right_op);

template <class T> SparseVector<T> &operator+=(      SparseVector<T> &left_op, const T               &right_op);
template <class T> SparseVector<T> &operator-=(      SparseVector<T> &left_op, const T               &right_op);
template <class T> SparseVector<T> &operator*=(      SparseVector<T> &left_op, const T               &right_op);
template <class T> SparseVector<T> &operator/=(      SparseVector<T> &left_op, const T               &right_op);
template <class T> SparseVector<T> &operator%=(      SparseVector<T> &left_op, const T               &right_op);

template <class T> SparseVector<T> &operator+=(const T               &left_op,       SparseVector<T> &right_op);
template <class T> SparseVector<T> &operator-=(const T               &left_op,       SparseVector<T> &right_op);
template <class T> SparseVector<T> &operator*=(const T               &left_op,       SparseVector<T> &right_op);
template <class T> SparseVector<T> &operator/=(const T               &left_op,       SparseVector<T> &right_op);
template <class T> SparseVector<T> &operator%=(const T               &left_op,       SparseVector<T> &right_op);

// Related non-commutative operations
//
// leftmult:  equivalent to *=
// rightmult: like *=, but result is stored in right_op and ref to right_op is returned

template <class T> SparseVector<T> &leftmult (      SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> SparseVector<T> &leftmult (      SparseVector<T> &left_op, const T               &right_op);
template <class T> SparseVector<T> &rightmult(const SparseVector<T> &left_op,       SparseVector<T> &right_op);
template <class T> SparseVector<T> &rightmult(const T               &left_op,       SparseVector<T> &right_op);

// Relational operator overloading

template <class T> int operator==(const SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> int operator==(const SparseVector<T> &left_op, const T               &right_op);
template <class T> int operator==(const T               &left_op, const SparseVector<T> &right_op);

template <class T> int operator!=(const SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> int operator!=(const SparseVector<T> &left_op, const T               &right_op);
template <class T> int operator!=(const T               &left_op, const SparseVector<T> &right_op);

template <class T> int operator< (const SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> int operator< (const Vector<T> &left_op, const T         &right_op);
template <class T> int operator< (const T         &left_op, const Vector<T> &right_op);

template <class T> int operator<=(const SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> int operator<=(const Vector<T> &left_op, const T         &right_op);
template <class T> int operator<=(const T         &left_op, const Vector<T> &right_op);

template <class T> int operator> (const SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> int operator> (const Vector<T> &left_op, const T         &right_op);
template <class T> int operator> (const T         &left_op, const Vector<T> &right_op);

template <class T> int operator>=(const SparseVector<T> &left_op, const SparseVector<T> &right_op);
template <class T> int operator>=(const Vector<T> &left_op, const T         &right_op);
template <class T> int operator>=(const T         &left_op, const Vector<T> &right_op);





// Kronecker product

template <class T> SparseVector<T> &kronprod(SparseVector<T> &res, int &dimres, const SparseVector<T> &a, const SparseVector<T> &b, int dima, int dimb);
template <class T> Vector<T> &kronprod(Vector<T> &res, const SparseVector<T> &a, const SparseVector<T> &b, int dima, int dimb);

template <class T> SparseVector<T> &kronpow(SparseVector<T> &res, int &dimres, const SparseVector<T> &a, int dima, int n);
template <class T> Vector<T> &kronpow(Vector<T> &res, const SparseVector<T> &a, int dima, int n);






















// Now for the actual code (no *.o files with templates, which is annoying as hell)


template <class T>
int SparseVector<T>::nearupsize(void) const
{
    int res = 1;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ind(i) < INDFAROFFSTART )
        {
            break;
        }
    }

    if ( ( i >= 0 ) && ( i < indsize() ) )
    {
        if ( ( ind(i) >= 0 ) && ( ind(i) < INDFAROFFSTART ) )
        {
            res = (ind(i)/DEFAULT_TUPLE_INDEX_STEP)+1;
        }
    }

    return res ? res : 1;
}

template <class T>
int SparseVector<T>::farupsize(void) const
{
    int res = 1;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ind(i) < INDFARFAROFFSTART )
        {
            break;
        }
    }

    if ( ( i >= 0 ) && ( i < indsize() ) )
    {
        if ( ( ind(i) >= INDFAROFFSTART ) && ( ind(i) < INDFARFAROFFSTART ) )
        {
            res = ((ind(i)-INDFAROFFSTART)/DEFAULT_TUPLE_INDEX_STEP)+1;
        }
    }

    return res ? res : 1;
}

template <class T>
int SparseVector<T>::farfarupsize(void) const
{
    int res = 1;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ind(i) < INDFARFARFAROFFSTART )
        {
            break;
        }
    }

    if ( ( i >= 0 ) && ( i < indsize() ) )
    {
        if ( ( ind(i) >= INDFARFAROFFSTART ) && ( ind(i) < INDFARFARFAROFFSTART ) )
        {
            res = ((ind(i)-INDFARFAROFFSTART)/DEFAULT_TUPLE_INDEX_STEP)+1;
        }
    }

    return res ? res : 1;
}

template <class T>
int SparseVector<T>::farfarfarupsize(void) const
{
    int res = 1;
    int i = indsize()-1;

    if ( ( i >= 0 ) && ( i < indsize() ) )
    {
        if ( ind(i) >= INDFARFARFAROFFSTART )
        {
            res = ((ind(i)-INDFARFARFAROFFSTART)/DEFAULT_TUPLE_INDEX_STEP)+1;
        }
    }

    return res ? res : 1;
}

template <class T>
int SparseVector<T>::nearrefupindsize(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    int res = 0;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u) ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1)) ) )
        {
            res++;
        }

        else if ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*u) )
        {
            break;
        }
    }

    return res;
}

template <class T>
int SparseVector<T>::farrefupindsize(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    int res = 0;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u)+INDFAROFFSTART ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1))+INDFAROFFSTART ) )
        {
            res++;
        }

        else if ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*u)+INDFAROFFSTART )
        {
            break;
        }
    }

    return res;
}

template <class T>
int SparseVector<T>::farfarrefupindsize(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    int res = 0;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFAROFFSTART ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1))+INDFARFAROFFSTART ) )
        {
            res++;
        }

        else if ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFAROFFSTART )
        {
            break;
        }
    }

    return res;
}

template <class T>
int SparseVector<T>::farfarfarrefupindsize(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    int res = 0;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFARFAROFFSTART ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1))+INDFARFARFAROFFSTART ) )
        {
            res++;
        }

        else if ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFARFAROFFSTART )
        {
            break;
        }
    }

    return res;
}

template <class T>
int SparseVector<T>::nearrefupsize(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    int res = 0;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u) ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1)) ) )
        {
            if ( (ind(i)-(DEFAULT_TUPLE_INDEX_STEP*u))+1 > res )
            {
                res = (ind(i)-(DEFAULT_TUPLE_INDEX_STEP*u))+1;
            }
        }

        else if ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*u) )
        {
            break;
        }
    }

    return res;
}

template <class T>
int SparseVector<T>::farrefupsize(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    int res = 0;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u)+INDFAROFFSTART ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1))+INDFAROFFSTART ) )
        {
            if ( (ind(i)-((DEFAULT_TUPLE_INDEX_STEP*u)+INDFAROFFSTART))+1 > res )
            {
                res = (ind(i)-((DEFAULT_TUPLE_INDEX_STEP*u)+INDFAROFFSTART))+1;
            }
        }

        else if ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*u)+INDFAROFFSTART )
        {
            break;
        }
    }

    return res;
}

template <class T>
int SparseVector<T>::farfarrefupsize(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    int res = 0;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFAROFFSTART ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1))+INDFARFAROFFSTART ) )
        {
            if ( (ind(i)-((DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFAROFFSTART))+1 > res )
            {
                res = (ind(i)-((DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFAROFFSTART))+1;
            }
        }

        else if ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFAROFFSTART )
        {
            break;
        }
    }

    return res;
}

template <class T>
int SparseVector<T>::farfarfarrefupsize(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    int res = 0;
    int i;

    for ( i = indsize()-1 ; i >= 0 ; i-- )
    {
        if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFARFAROFFSTART ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1))+INDFARFARFAROFFSTART ) )
        {
            if ( (ind(i)-((DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFARFAROFFSTART))+1 > res )
            {
                res = (ind(i)-((DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFARFAROFFSTART))+1;
            }
        }

        else if ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFARFAROFFSTART )
        {
            break;
        }
    }

    return res;
}

template <class T>
const SparseVector<T> &SparseVector<T>::nearrefup(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < nearupsize() );

    // Don't risk destroying altcontent if we don't have to

    if ( ( nearupsize() == 1 ) && isnofaroffindpresent() )
    {
        return *this;
    }

    nearref();

    return nearpointer[u+1];
}

template <class T>
const SparseVector<T> &SparseVector<T>::farrefup(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < farupsize() );

    farref();

    return farpointer[u+1];
}

template <class T>
const SparseVector<T> &SparseVector<T>::farfarrefup(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < farfarupsize() );

    farfarref();

    return farfarpointer[u+1];
}

template <class T>
const SparseVector<T> &SparseVector<T>::farfarfarrefup(int u) const
{
    NiceAssert( u >= 0 );
    NiceAssert( u < farfarfarupsize() );

    farfarfarref();

    return farfarfarpointer[u+1];
}

template <class T>
const SparseVector<T> &SparseVector<T>::nearref(void) const
{
    int i,j,s,p,qs,rs;

    p  = 0;
    qs = nearupsize();
    rs = 0;

    // Don't risk destroying altcontent if we don't have to
    // (also don't need to allocate nearpointer in this case as nearrefup(0) will just return *this)

    if ( ( nearupsize() == 1 ) && isnofaroffindpresent() )
    {
        return *this;
    }

    if ( !nearpointer )
    {
        MEMNEWARRAY((**thisthisthis).nearpointer,SparseVector<T>,qs+1);

        {
            SparseVector<T> &nearvector = ((**thisthisthis).nearpointer)[0];

            Vector<int> &nearind = *(nearvector.indices);
            Vector<T>   &nearval = *(nearvector.content);

            s = nearindsize();

            nearind.resize(s);
            nearval.resize(s+1);
        
            nearval("&",s) = (*content)((*content).size()-1); // zero sample

            for ( j = 0 ; j < s ; j++ )
            {
                nearind("&",j) = (*indices)(j+p)-rs;
                nearval("&",j) = (*content)(j+p);
            }

            nearvector.makealtcontent();
        }

        for ( i = 0 ; i < qs ; i++ )
        {
            SparseVector<T> &nearvector = ((**thisthisthis).nearpointer)[i+1];

            Vector<int> &nearind = *(nearvector.indices);
            Vector<T>   &nearval = *(nearvector.content);

            s = nearrefupindsize(i);

            nearind.resize(s);
            nearval.resize(s+1);
        
            nearval("&",s) = (*content)((*content).size()-1); // zero sample

            for ( j = 0 ; j < s ; j++ )
            {
                nearind("&",j) = (*indices)(j+p)-rs;
                nearval("&",j) = (*content)(j+p);
            }

            nearvector.makealtcontent();

            p  += s;
            rs += DEFAULT_TUPLE_INDEX_STEP;
        }
    }

    return *nearpointer;
}

template <class T>
const SparseVector<T> &SparseVector<T>::farref(void) const
{
    int i,j,s,p,qs,rs;

    p  = nearindsize();
    qs = farupsize();
    rs = INDFAROFFSTART;

    if ( !farpointer )
    {
        MEMNEWARRAY((**thisthisthis).farpointer,SparseVector<T>,qs+1);

        {
            SparseVector<T> &farvector = ((**thisthisthis).farpointer)[0];

            Vector<int> &farind = *(farvector.indices);
            Vector<T>   &farval = *(farvector.content);

            s = farindsize();

            farind.resize(s);
            farval.resize(s+1);
        
            farval("&",s) = (*content)((*content).size()-1); // zero sample

            for ( j = 0 ; j < s ; j++ )
            {
                farind("&",j) = (*indices)(j+p)-rs;
                farval("&",j) = (*content)(j+p);
            }

            farvector.makealtcontent();
        }

        for ( i = 0 ; i < qs ; i++ )
        {
            SparseVector<T> &farvector = ((**thisthisthis).farpointer)[i+1];

            Vector<int> &farind = *(farvector.indices);
            Vector<T>   &farval = *(farvector.content);

            s = farrefupindsize(i);

            farind.resize(s);
            farval.resize(s+1);
        
            farval("&",s) = (*content)((*content).size()-1); // zero sample

            for ( j = 0 ; j < s ; j++ )
            {
                farind("&",j) = (*indices)(j+p)-rs;
                farval("&",j) = (*content)(j+p);
            }

            farvector.makealtcontent();

            p  += s;
            rs += DEFAULT_TUPLE_INDEX_STEP;
        }
    }

    return *farpointer;
}

template <class T>
const SparseVector<T> &SparseVector<T>::farfarref(void) const
{
    int i,j,s,p,qs,rs;

    p  = nearfarindsize();
    qs = farfarupsize();
    rs = INDFARFAROFFSTART;

    if ( !farfarpointer )
    {
        MEMNEWARRAY((**thisthisthis).farfarpointer,SparseVector<T>,qs+1);

        {
            SparseVector<T> &farfarvector = ((**thisthisthis).farfarpointer)[0];

            Vector<int> &farfarind = *(farfarvector.indices);
            Vector<T>   &farfarval = *(farfarvector.content);

            s = farfarindsize();

            farfarind.resize(s);
            farfarval.resize(s+1);
        
            farfarval("&",s) = (*content)((*content).size()-1); // zero sample

            for ( j = 0 ; j < s ; j++ )
            {
                farfarind("&",j) = (*indices)(j+p)-rs;
                farfarval("&",j) = (*content)(j+p);
            }

            farfarvector.makealtcontent();
        }

        for ( i = 0 ; i < qs ; i++ )
        {
            SparseVector<T> &farfarvector = ((**thisthisthis).farfarpointer)[i+1];

            Vector<int> &farfarind = *(farfarvector.indices);
            Vector<T>   &farfarval = *(farfarvector.content);

            s = farfarrefupindsize(i);

            farfarind.resize(s);
            farfarval.resize(s+1);
        
            farfarval("&",s) = (*content)((*content).size()-1); // zero sample

            for ( j = 0 ; j < s ; j++ )
            {
                farfarind("&",j) = (*indices)(j+p)-rs;
                farfarval("&",j) = (*content)(j+p);
            }

            farfarvector.makealtcontent();

            p  += s;
            rs += DEFAULT_TUPLE_INDEX_STEP;
        }
    }

    return *farfarpointer;

/*
    int j,s,p,rs;

    p  = nearfarindsize();
    rs = INDFARFAROFFSTART;

    if ( !farfarpointer )
    {
        MEMNEWARRAY((**thisthisthis).farfarpointer,SparseVector<T>,1);

        {
            SparseVector<T> &farfarvector = ((**thisthisthis).farfarpointer)[0];

            Vector<int> &farfarind = *(farfarvector.indices);
            Vector<T>   &farfarval = *(farfarvector.content);

            s = farfarindsize();

            farfarind.resize(s);
            farfarval.resize(s+1);
        
            farfarval("&",s) = (*content)((*content).size()-1); // zero sample

            for ( j = 0 ; j < s ; j++ )
            {
                farfarind("&",j) = (*indices)(j+p)-rs;
                farfarval("&",j) = (*content)(j+p);
            }

            farfarvector.makealtcontent();
        }
    }

    return *farfarpointer;
*/
}

template <class T>
const SparseVector<T> &SparseVector<T>::farfarfarref(void) const
{
    int i,j,s,p,qs,rs;

    p  = nearfarfarindsize();
    qs = farfarfarupsize();
    rs = INDFARFARFAROFFSTART;

    if ( !farfarfarpointer )
    {
        MEMNEWARRAY((**thisthisthis).farfarfarpointer,SparseVector<T>,qs+1);

        {
            SparseVector<T> &farfarfarvector = ((**thisthisthis).farfarfarpointer)[0];

            Vector<int> &farfarfarind = *(farfarfarvector.indices);
            Vector<T>   &farfarfarval = *(farfarfarvector.content);

            s = farfarfarindsize();

            farfarfarind.resize(s);
            farfarfarval.resize(s+1);
        
            farfarfarval("&",s) = (*content)((*content).size()-1); // zero sample

            for ( j = 0 ; j < s ; j++ )
            {
                farfarfarind("&",j) = (*indices)(j+p)-rs;
                farfarfarval("&",j) = (*content)(j+p);
            }

            farfarfarvector.makealtcontent();
        }

        for ( i = 0 ; i < qs ; i++ )
        {
            SparseVector<T> &farfarfarvector = ((**thisthisthis).farfarfarpointer)[i+1];

            Vector<int> &farfarfarind = *(farfarfarvector.indices);
            Vector<T>   &farfarfarval = *(farfarfarvector.content);

            s = farfarfarrefupindsize(i);

            farfarfarind.resize(s);
            farfarfarval.resize(s+1);
        
            farfarfarval("&",s) = (*content)((*content).size()-1); // zero sample

            for ( j = 0 ; j < s ; j++ )
            {
                farfarfarind("&",j) = (*indices)(j+p)-rs;
                farfarfarval("&",j) = (*content)(j+p);
            }

            farfarfarvector.makealtcontent();

            p  += s;
            rs += DEFAULT_TUPLE_INDEX_STEP;
        }
    }

    return *farfarfarpointer;

/*
    int j,s,p,rs;

    p  = nearfarfarindsize();
    rs = INDFARFARFAROFFSTART;

    if ( !farfarfarpointer )
    {
        MEMNEWARRAY((**thisthisthis).farfarfarpointer,SparseVector<T>,1);

        {
            SparseVector<T> &farfarfarvector = ((**thisthisthis).farfarfarpointer)[0];

            Vector<int> &farfarfarind = *(farfarfarvector.indices);
            Vector<T>   &farfarfarval = *(farfarfarvector.content);

            s = farfarfarindsize();

            farfarfarind.resize(s);
            farfarfarval.resize(s+1);
        
            farfarfarval("&",s) = (*content)((*content).size()-1); // zero sample

            for ( j = 0 ; j < s ; j++ )
            {
                farfarfarind("&",j) = (*indices)(j+p)-rs;
                farfarfarval("&",j) = (*content)(j+p);
            }

            farfarfarvector.makealtcontent();
        }
    }

    return *farfarfarpointer;
*/
}


// Constructors and Destructors

template <class T>
SparseVector<T>::SparseVector(int size, const T *src)
{
    NiceAssert( size >= 0 );

    resetvecID();

    altcontent = NULL;

    MEMNEW(indices,Vector<int>);
    MEMNEW(content,Vector<T>(1));

    NiceAssert(indices);
    NiceAssert(content);

    nearpointer      = NULL;
    farpointer       = NULL;
    farfarpointer    = NULL;
    farfarfarpointer = NULL;

    setzero((*content)("&",zeroint()));

    if ( size )
    {
        NiceAssert(src);

        int i;

        for ( i = 0 ; i < size ; i++ )
        {
            (*this)("&",i) = *src;
        }
    }

    thisthis = this;
    thisthisthis = &thisthis;

    return;
}

template <class T>
SparseVector<T>::SparseVector(const SparseVector<T> &src)
{
    MEMNEW(indices,Vector<int>);
    MEMNEW(content,Vector<T>(1));

    resetvecID();

    altcontent = NULL;

    NiceAssert(indices);
    NiceAssert(content);

    setzero((*content)("&",zeroint()));

    nearpointer      = NULL;
    farpointer       = NULL;
    farfarpointer    = NULL;
    farfarfarpointer = NULL;

    *this = src;

    thisthis = this;
    thisthisthis = &thisthis;

    return;
}

template <class T>
SparseVector<T>::SparseVector(const Vector<T> &src)
{
    MEMNEW(indices,Vector<int>);
    MEMNEW(content,Vector<T>(1));

    resetvecID();

    altcontent = NULL;

    NiceAssert(indices);
    NiceAssert(content);

    setzero((*content)("&",zeroint()));

    nearpointer      = NULL;
    farpointer       = NULL;
    farfarpointer    = NULL;
    farfarfarpointer = NULL;

    *this = src;

    thisthis = this;
    thisthisthis = &thisthis;

    return;
}

template <class T>
SparseVector<T>::SparseVector(const char *dummy)
{
    (void) dummy;

    resetvecID();

    altcontent = NULL;

    content = NULL;
    indices = NULL;

    nearpointer      = NULL;
    farpointer       = NULL;
    farfarpointer    = NULL;
    farfarfarpointer = NULL;

    thisthis = this;
    thisthisthis = &thisthis;

    return;
}

template <class T>
SparseVector<T>::~SparseVector()
{
    killaltcontent();
    killnearfar();

    if ( indices )
    {
        MEMDEL(indices);
        indices = NULL;
    }

    if ( content )
    {
        MEMDEL(content);
        content = NULL;
    }

    return;
}




// Assignment

template <class T>
template <class S>
SparseVector<T> &SparseVector<T>::assign(const SparseVector<S> &src)
{
    // No need to worry about shared base here

    killnearfar();

    (*indices).assign(*(src.indices));
    (*content).assign(*(src.content));

    if ( src.altcontent )
    {
        if ( altcontent && ( src.size() != size() ) )
        {
            killaltcontent();
        }

        if ( !altcontent )
        {
            MEMNEWARRAY(altcontent,double,src.size());
        }

        int i;

        for ( i = 0 ; i < src.size() ; i++ )
        {
            altcontent[i] = (src.altcontent)[i];
        }
    }

    else
    {
        killaltcontent();
    }

    setvecID(src.vecID());

    return *this;
}

template <class T>
template <class S>
SparseVector<T> &SparseVector<T>::nearassign(const SparseVector<S> &src)
{
    // No need to worry about shared base here

    resetvecID();
    killaltcontent();
    killnear();

    int lstart = 0;
    int lend   = src.nearindsize()-1;

    retVector<int> tmpva;
    retVector<S>   tmpvb;

    *indices = (*(src.indices))(lstart,1,lend,tmpva);
    *content = (*(src.content))(lstart,1,lend,tmpvb);

    int cs = (*content).size();

    (*content).add(cs);
    setzero((*content)("&",cs));

    return *this;
}

template <class T>
template <class S>
SparseVector<T> &SparseVector<T>::castassign(const SparseVector<S> &src)
{
    // No need to worry about shared base here

    resetvecID();
    killaltcontent();
    killnearfar();

    (*indices).assign(*(src.indices));
    (*content).castassign(*(src.content));

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::operator=(const Vector<T> &src)
{
    // No need to worry about shared base here

    resetvecID();
    killaltcontent();
    killnearfar();

    retVector<int> tmpva;

    (*indices) = cntintvec(src.size(),tmpva);
    (*content) = src;

    int cs = (*content).size();

    (*content).add(cs);
    setzero((*content)("&",cs));

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::operator=(const T &src)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    retVector<T> tmpva;

    (*content)("&",zeroint(),1,indsize()-1,tmpva) = src;

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::overwrite(const SparseVector<T> &src)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( src.indsize() )
    {
        int i;
        
        for ( i = 0 ; i < src.indsize() ; i++ )
        {
            (*this)("&",src.ind(i)) = src.direcref(i);
        }
    }

    return *this;
}



// Basic operations.

template <class T>
SparseVector<T> &SparseVector<T>::ident(void)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    retVector<T> tmpva;

    (*content)("&",zeroint(),1,indsize()-1,tmpva).ident();

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::conj(void)
{
    resetvecID();
    killnearfar();

    retVector<T> tmpva;

    (*content)("&",zeroint(),1,indsize()-1,tmpva).conj();

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::rand(void)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    retVector<T> tmpva;

    (*content)("&",zeroint(),1,indsize()-1,tmpva).rand();

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zero(void)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    while ( indsize() )
    {
        (*content).remove(indsize()-1);
        (*indices).remove(indsize()-1);
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zeronear(int u)
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    if ( u < nearupsize() )
    {
        resetvecID();
        killaltcontent();
        killnear();

        int i = indsize()-1;
        int zs = nearrefupindsize(u);

        while ( zs )
        {
            if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u) ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1)) ) )
            {
                (*content).remove(i);
                (*indices).remove(i);

                zs--;
           }

            i--;
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zerofar(int u)
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    if ( u < farupsize() )
    {
        resetvecID();
        killaltcontent();
        killnear();

        int i = indsize()-1;
        int zs = farrefupindsize(u);

        while ( zs )
        {
            if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u)+INDFAROFFSTART ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1))+INDFAROFFSTART ) )
            {
                (*content).remove(i);
                (*indices).remove(i);

                zs--;
           }

            i--;
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zerofarfar(int u)
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    if ( u < farfarupsize() )
    {
        resetvecID();
        killaltcontent();
        killnear();

        int i = indsize()-1;
        int zs = farfarrefupindsize(u);

        while ( zs )
        {
            if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFAROFFSTART ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1))+INDFARFAROFFSTART ) )
            {
                (*content).remove(i);
                (*indices).remove(i);

                zs--;
           }

            i--;
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zerofarfarfar(int u)
{
    NiceAssert( u >= 0 );
    NiceAssert( u < DEFAULT_NUM_TUPLES );

    if ( u < farfarfarupsize() )
    {
        resetvecID();
        killaltcontent();
        killnear();

        int i = indsize()-1;
        int zs = farfarfarrefupindsize(u);

        while ( zs )
        {
            if ( ( ind(i) >= (DEFAULT_TUPLE_INDEX_STEP*u)+INDFARFARFAROFFSTART ) && ( ind(i) < (DEFAULT_TUPLE_INDEX_STEP*(u+1))+INDFARFARFAROFFSTART ) )
            {
                (*content).remove(i);
                (*indices).remove(i);

                zs--;
           }

            i--;
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zeronear(void)
{
    resetvecID();
    killaltcontent();
    killnear();

    int i = indsize()-1;
    int zs = nearindsize();

    while ( zs )
    {
        if ( ind(i) < INDFAROFFSTART )
        {
            (*content).remove(i);
            (*indices).remove(i);

            zs--;
        }

        i--;
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zerofar(void)
{
    resetvecID();
    killaltcontent();
    killfar();

    int i = indsize()-1;
    int zs = farindsize();

    while ( zs )
    {
        if ( ( ind(i) >= INDFAROFFSTART ) && ( ind(i) < INDFARFAROFFSTART ) )
        {
            (*content).remove(i);
            (*indices).remove(i);

            zs--;
        }

        i--;
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zerofarfar(void)
{
    resetvecID();
    killaltcontent();
    killfarfar();

    int i = indsize()-1;
    int zs = farfarindsize();

    while ( zs )
    {
        if ( ( ind(i) >= INDFARFAROFFSTART ) && ( ind(i) < INDFARFARFAROFFSTART ) )
        {
            (*content).remove(i);
            (*indices).remove(i);

            zs--;
        }

        i--;
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zerofarfarfar(void)
{
    resetvecID();
    killaltcontent();
    killfarfarfar();

    int i = indsize()-1;
    int zs = farfarfarindsize();

    while ( zs )
    {
        if ( ind(i) >= INDFARFARFAROFFSTART )
        {
            (*content).remove(i);
            (*indices).remove(i);

            zs--;
        }

        i--;
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::overwritenear(const SparseVector<T> &src, int u)
{
    zeronear(u);

    int srcsize = src.nearindsize();

    if ( srcsize )
    {
        int i;

        for ( i = 0 ; i < srcsize ; i++ )
        {
            n("&",(DEFAULT_TUPLE_INDEX_STEP*u)+src.ind(i)) = src.direcref(i);
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::overwritefar(const SparseVector<T> &src, int u)
{
    zerofar(u);

    int srcsize = src.nearindsize();

    if ( srcsize )
    {
        int i;

        for ( i = 0 ; i < srcsize ; i++ )
        {
            f("&",(DEFAULT_TUPLE_INDEX_STEP*u)+src.ind(i)) = src.direcref(i);
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::overwritefarfar(const SparseVector<T> &src, int u)
{
    zerofarfar(u);

    int srcsize = src.nearindsize();

    if ( srcsize )
    {
        int i;

        for ( i = 0 ; i < srcsize ; i++ )
        {
            ff("&",(DEFAULT_TUPLE_INDEX_STEP*u)+src.ind(i)) = src.direcref(i);
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::overwritefarfarfar(const SparseVector<T> &src, int u)
{
    zerofarfarfar(u);

    int srcsize = src.nearindsize();

    if ( srcsize )
    {
        int i;

        for ( i = 0 ; i < srcsize ; i++ )
        {
            fff("&",(DEFAULT_TUPLE_INDEX_STEP*u)+src.ind(i)) = src.direcref(i);
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::overwritenear(const SparseVector<T> &src)
{
    zeronear();

    int srcsize = src.nearindsize();

    if ( srcsize )
    {
        int i;

        for ( i = 0 ; i < srcsize ; i++ )
        {
            n("&",src.ind(i)) = src.direcref(i);
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::overwritefar(const SparseVector<T> &src)
{
    zerofar();

    int srcsize = src.farindsize();

    if ( srcsize )
    {
        int i;

        for ( i = 0 ; i < srcsize ; i++ )
        {
            f("&",src.ind(i)) = src.direcref(i);
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::overwritefarfar(const SparseVector<T> &src)
{
    zerofarfar();

    int srcsize = src.farfarindsize();

    if ( srcsize )
    {
        int i;

        for ( i = 0 ; i < srcsize ; i++ )
        {
            ff("&",src.ind(i)) = src.direcref(i);
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::overwritefarfarfar(const SparseVector<T> &src)
{
    zerofarfarfar();

    int srcsize = src.farfarfarindsize();

    if ( srcsize )
    {
        int i;

        for ( i = 0 ; i < srcsize ; i++ )
        {
            fff("&",src.ind(i)) = src.direcref(i);
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zero(int i)
{
    // FIXME: really should only delete the array containing the relevant element

    resetvecID();
    killaltcontent();
    killnearfar();

    int pos = findind(i);

    if ( pos < indsize() )
    {
	if ( ind(pos) == i )
	{
            (*indices).remove(pos);
            (*content).remove(pos);
	}
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zero(const Vector<int> &i)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            zero(i(j));
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zeromost(int i)
{
    if ( i == -1 )
    {
        return *this;
    }

    resetvecID();
    killaltcontent();
    killnearfar();

    int pos = findind(i);

    if ( indsize() )
    {
        int j;

        for ( j = 0 ; j < indsize() ; j++ )
        {
            if ( pos != j )
            {
                setzero((*content)("&",j));
            }
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::softzero(void)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    retVector<T> tmpva;

    (*content)("&",zeroint(),1,indsize()-1,tmpva).zero();

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::zeropassive(void)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    retVector<T> tmpva;

    (*content)("&",zeroint(),1,indsize()-1,tmpva).zeropassive();

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::posate(void)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    retVector<T> tmpva;

    (*content)("&",zeroint(),1,indsize()-1,tmpva).posate();

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::negate(void)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    retVector<T> tmpva;

    (*content)("&",zeroint(),1,indsize()-1,tmpva).negate();

    return *this;
}


// Access:

template <class T>
Vector<T> &SparseVector<T>::operator()(const char *dummy, retVector<T> &tmp)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( nearnonsparse() )
    {
        return (*content)(dummy,zeroint(),1,size()-1,tmp);
    }

    return (*this)(dummy,zeroint(),1,size()-1,tmp);
}

template <class T>
T &SparseVector<T>::operator()(const char *dummy, int i)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    (void) dummy;

    NiceAssert( i >= 0 );

    int pos = findind(i);

    NiceAssert( pos <= indsize() );

    if ( pos == indsize() )
    {
	(*indices).add(pos);
	(*indices)("&",pos) = i;
        (*content).add(pos);
        (*content)("&",pos) = zerelm();
    }

    else if ( ind(pos) != i )
    {
	(*indices).add(pos);
	(*indices)("&",pos) = i;
        (*content).add(pos);
        (*content)("&",pos) = zerelm();
    }

    return (*content)("&",pos);
}

template <class T>
T &SparseVector<T>::n(const char *dummy, int i)
{
    i += INDNEAROFFSTART;

    resetvecID();
    killaltcontent();
    killnear();

    (void) dummy;

    NiceAssert( i >= INDNEAROFFSTART );
    NiceAssert( i < INDFAROFFSTART );

    int pos = findind(i);

    NiceAssert( pos <= indsize() );

    if ( pos == indsize() )
    {
	(*indices).add(pos);
	(*indices)("&",pos) = i;
        (*content).add(pos);
        (*content)("&",pos) = zerelm();
    }

    else if ( ind(pos) != i )
    {
	(*indices).add(pos);
	(*indices)("&",pos) = i;
        (*content).add(pos);
        (*content)("&",pos) = zerelm();
    }

    return (*content)("&",pos);
}

template <class T>
T &SparseVector<T>::f(const char *dummy, int i)
{
    i += INDFAROFFSTART;

    resetvecID();
    killaltcontent();
    killfar();

    (void) dummy;

    NiceAssert( i >= INDFAROFFSTART );
    NiceAssert( i < INDFARFAROFFSTART );

    int pos = findind(i);

    NiceAssert( pos <= indsize() );

    if ( pos == indsize() )
    {
	(*indices).add(pos);
	(*indices)("&",pos) = i;
        (*content).add(pos);
        (*content)("&",pos) = zerelm();
    }

    else if ( ind(pos) != i )
    {
	(*indices).add(pos);
	(*indices)("&",pos) = i;
        (*content).add(pos);
        (*content)("&",pos) = zerelm();
    }

    return (*content)("&",pos);
}

template <class T>
T &SparseVector<T>::ff(const char *dummy, int i)
{
    i += INDFARFAROFFSTART;

    resetvecID();
    killaltcontent();
    killfarfar();

    (void) dummy;

    NiceAssert( i >= INDFARFAROFFSTART );
    NiceAssert( i < INDFARFARFAROFFSTART );

    int pos = findind(i);

    NiceAssert( pos <= indsize() );

    if ( pos == indsize() )
    {
	(*indices).add(pos);
	(*indices)("&",pos) = i;
        (*content).add(pos);
        (*content)("&",pos) = zerelm();
    }

    else if ( ind(pos) != i )
    {
	(*indices).add(pos);
	(*indices)("&",pos) = i;
        (*content).add(pos);
        (*content)("&",pos) = zerelm();
    }

    return (*content)("&",pos);
}

template <class T>
T &SparseVector<T>::fff(const char *dummy, int i)
{
    i += INDFARFARFAROFFSTART;

    resetvecID();
    killaltcontent();
    killfarfarfar();

    (void) dummy;

    NiceAssert( i >= INDFARFARFAROFFSTART );

    int pos = findind(i);

    NiceAssert( pos <= indsize() );

    if ( pos == indsize() )
    {
	(*indices).add(pos);
	(*indices)("&",pos) = i;
        (*content).add(pos);
        (*content)("&",pos) = zerelm();
    }

    else if ( ind(pos) != i )
    {
	(*indices).add(pos);
	(*indices)("&",pos) = i;
        (*content).add(pos);
        (*content)("&",pos) = zerelm();
    }

    return (*content)("&",pos);
}

template <class T>
Vector<T> &SparseVector<T>::operator()(const char *dummy, const Vector<int> &i, retVector<T> &tmp)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( indices == &i )
    {
        return (*content)(dummy,zeroint(),1,indsize()-1,tmp);
    }

    int j;
    Vector<int> iii(i);

    if ( i.size() )
    {
        // Make sure all indices are present

        for ( j = 0 ; j < i.size() ; j++ )
        {
            if ( !isindpresent(i(j)) )
            {
                (*this)("&",i(j)) = zerelm();
            }
        }

        // Set up dereferenced index vector

        for ( j = 0 ; j < i.size() ; j++ )
        {
            iii("&",j) = findind(i(j));
        }
    }

    // This gets ugly as we don't want to return a reference to a local hidden inside the result
    // Note that the assignment here will remove the local reference

    {
        retVector<T> tmpva;

        static_cast<Vector<T> &>(tmp) = (*content)(dummy,iii,tmpva);
    }

    return tmp;
}

template <class T>
Vector<T> &SparseVector<T>::operator()(const char *dummy, int ib, int is, int im, retVector<T> &tmp)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    int j;
    Vector<int> iii;

    if ( is >= ib )
    {
        for ( j = ib ; j <= im ; j += is )
        {
            iii.add(iii.size());
            iii("&",iii.size()-1) = j;
        }
    }

    // This gets ugly as we don't want to return a reference to a local hidden inside the result
    // Note that the assignment here will remove the local reference

    {
        retVector<T> tmpva;

        static_cast<Vector<T> &>(tmp) = (*content)(dummy,iii,tmpva);
    }

    return tmp;
}

template <class T>
const Vector<T> &SparseVector<T>::operator()(retVector<T> &tmp) const
{
    if ( nearnonsparse() )
    {
        return (*content)(zeroint(),1,size()-1,tmp);
    }

    return (*this)(zeroint(),1,size()-1,tmp);
}

template <class T>
const T &SparseVector<T>::operator()(int i) const
{
    NiceAssert( i >= 0 );

    int pos = findind(i);

    if ( ( pos < indsize() ) && ( ind(pos) != i ) )
    {
        pos = indsize();
    }

    return (*content)(pos);
}

template <class T>
const T &SparseVector<T>::n(int i) const
{
    i += INDNEAROFFSTART;

    NiceAssert( i >= INDNEAROFFSTART );
    NiceAssert( i < INDFAROFFSTART );

    int pos = findind(i);

    if ( ( pos < indsize() ) && ( ind(pos) != i ) )
    {
        pos = indsize();
    }

    return (*content)(pos);
}

template <class T>
const T &SparseVector<T>::f(int i) const
{
    i += INDFAROFFSTART;

    NiceAssert( i >= INDFAROFFSTART );
    NiceAssert( i < INDFARFAROFFSTART );

    int pos = findind(i);

    if ( ( pos < indsize() ) && ( ind(pos) != i ) )
    {
        pos = indsize();
    }

    return (*content)(pos);
}

template <class T>
const T &SparseVector<T>::ff(int i) const
{
    i += INDFARFAROFFSTART;

    NiceAssert( i >= INDFARFAROFFSTART );
    NiceAssert( i < INDFARFARFAROFFSTART );

    int pos = findind(i);

    if ( ( pos < indsize() ) && ( ind(pos) != i ) )
    {
        pos = indsize();
    }

    return (*content)(pos);
}

template <class T>
const T &SparseVector<T>::fff(int i) const
{
    i += INDFARFARFAROFFSTART;

    NiceAssert( i >= INDFARFARFAROFFSTART );

    int pos = findind(i);

    if ( ( pos < indsize() ) && ( ind(pos) != i ) )
    {
        pos = indsize();
    }

    return (*content)(pos);
}

template <class T>
const Vector<T> &SparseVector<T>::operator()(const Vector<int> &i, retVector<T> &tmp) const
{
    if ( indices == &i )
    {
        return (*content)(zeroint(),1,indsize()-1,tmp);
    }

    int j,pos;
    Vector<int> iii(i);

    if ( i.size() )
    {
        // Set up dereferenced index vector

        for ( j = 0 ; j < i.size() ; j++ )
        {
            pos = findind(i(j));

            if ( ( pos < indsize() ) && ( ind(pos) != i(j) ) )
            {
                pos = indsize();
            }

            iii("&",j) = pos;
        }
    }

    // This gets ugly as we don't want to return a reference to a local hidden inside the result
    // Note that the assignment here will remove the local reference

    {
        retVector<T> tmpva;

        static_cast<Vector<T> &>(tmp) = (*content)(iii,tmpva);
    }

    return tmp;
}

template <class T>
const Vector<T> &SparseVector<T>::operator()(int ib, int is, int im, retVector<T> &tmp) const
{
    int j;
    Vector<int> iii;

    if ( is >= ib )
    {
        for ( j = ib ; j <= im ; j += is )
        {
            iii.add(iii.size());
            iii("&",iii.size()-1) = j;
        }
    }

    // This gets ugly as we don't want to return a reference to a local hidden inside the result
    // Note that the assignment here will remove the local reference

    {
        retVector<T> tmpva;

        static_cast<Vector<T> &>(tmp) = (*content)(iii,tmpva);
    }

    return tmp;
}





// find element in index vector: returns min(j) such that indices(j) >= i

template <class T>
int SparseVector<T>::findind(int i) const
{
    int pos = 0;

    if ( indsize() )
    {
        if ( nearnonsparse() )
        {
            pos = ( i < indsize() ) ? i : indsize();
        }

        else
        {
            while ( ind(pos) < i )
            {
                pos++;

                if ( pos == indsize() )
                {
                    break;
                }
            }
        }
    }

    return pos;
}

template <class T>
SparseVector<T> &SparseVector<T>::offset(int amoff)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( amoff )
    {
        *indices += amoff;
    }

    return *this;
}

template <class T>
template <class S>
SparseVector<T> &SparseVector<T>::indalign(const SparseVector<S> &src)
{
    NiceAssert( (*content).size() );

    resetvecID();
    killaltcontent();
    killnearfar();

    *indices = src.ind();

    while ( (*content).size() > indsize()+1 )
    {
        (*content).remove((*content).size()-2);
    }

    while ( (*content).size() < indsize()+1 )
    {
        (*content).add((*content).size()-1);
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::indalign(const Vector<int> &srcind)
{
    NiceAssert( (*content).size() );

    resetvecID();
    killaltcontent();
    killnearfar();

    *indices = srcind;

    while ( (*content).size() > indsize()+1 )
    {
        (*content).remove((*content).size()-2);
    }

    while ( (*content).size() < indsize()+1 )
    {
        (*content).add((*content).size()-1);
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::iprune(const Vector<int> &refind)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    int i,j;

    i = 0;
    j = 0;

    while ( ( i < (*indices).size() ) && ( j < refind.size() ) )
    {
        if ( (*indices)(i) == refind(j) )
        {
            i++;
            j++;
        }

        else if ( (*indices)(i) > refind(j) )
        {
            j++;
        }

        else
        {
            NiceAssert( (*indices)(i) < refind(j) );

            // index i not present in reference

            (*indices).remove(i);
            (*content).remove(i);
        }
    }

    while ( i < (*indices).size() )
    {
        (*indices).remove(i);
        (*content).remove(i);
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::iprune(int refind)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    int i,j;

    i = 0;
    j = 0;

    while ( ( i < (*indices).size() ) && ( j < 1 ) )
    {
        if ( (*indices)(i) == refind )
        {
            i++;
            j++;
        }

        else if ( (*indices)(i) > refind )
        {
            j++;
        }

        else
        {
            NiceAssert( (*indices)(i) < refind );

            // index i not present in reference

            (*indices).remove(i);
            (*content).remove(i);
        }
    }

    while ( i < (*indices).size() )
    {
        (*indices).remove(i);
        (*content).remove(i);
    }

    return *this;
}

template <class T>
void SparseVector<T>::prealloc(int newallocsize)
{
    NiceAssert( ( newallocsize >= 0 ) || ( newallocsize == -1 ) );

    (*indices).prealloc(newallocsize);
    (*content).prealloc(newallocsize+1);

    return;
}

template <class T>
void SparseVector<T>::useStandardAllocation(void)
{
    (*indices).useStandardAllocation();
    (*content).useStandardAllocation();

    return;
}

template <class T>
void SparseVector<T>::useTightAllocation(void)
{
    (*indices).useTightAllocation();
    (*content).useTightAllocation();

    return;
}

template <class T>
void SparseVector<T>::useSlackAllocation(void)
{
    (*indices).useSlackAllocation();
    (*content).useSlackAllocation();

    return;
}

template <class T>
void SparseVector<T>::applyOnAll(void (*fn)(T &, int), int argx)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    (*content)("&",zeroint(),1,indsize()-1).applyOnAll(fn,argx);

    return;
}

template <class T>
SparseVector<T> &SparseVector<T>::blockswap(int i, int j)
{
    NiceAssert( i >= 0 );
    NiceAssert( j >= 0 );

    resetvecID();
    killaltcontent();
    killnearfar();

    // blockswap  ( i < j ): ( c ) (i)          ( c ) (i)
    //                       ( e ) (1)    ->    ( d ) (j-i)
    //                       ( d ) (j-i)        ( e ) (1)
    //                       ( f ) (...)        ( f ) (...)
    //
    // blockswap  ( i > j ): ( c ) (j)          ( c ) (j)
    //                       ( d ) (i-j)  ->    ( e ) (1)
    //                       ( e ) (1)          ( d ) (i-j)
    //                       ( f ) (...)        ( f ) (...)

    if ( nearnonsparse() && ( i < size() ) && ( j < size() ) )
    {
        (*content).blockswap(i,j);
    }

    else
    {
        if ( ( i > j ) && ( j < size() ) )
        {
            T temp;
            int ipres = isindpresent(i);

            if ( ipres )
            {
                qswap((*this)("&",i),temp);
                zero(i);
            }

            // Need this test just in case zeroing above means j now outside bounds

            if ( j < size() )
            {
                int ii = findind(i);
                int jj = findind(j);

                (*indices)("&",jj,1,((ii>indsize())?indsize():ii))++;
            }

            if ( ipres )
            {
                qswap((*this)("&",j),temp);
            }
        }

        else if ( ( i < j ) && ( i < size() ) )
        {
            T temp;
            int ipres = isindpresent(i);

            if ( ipres )
            {
                qswap((*this)("&",i),temp);
                zero(i);
            }

            // Need this test just in case zeroing above means i now outside bounds

            if ( i < size() )
            {
                int ii = findind(i);
                int jj = findind(j);

                (*indices)("&",ii,1,((jj>indsize())?indsize():jj))--;
            }

            if ( ipres )
            {
                qswap((*this)("&",j),temp);
            }
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::squareswap(int i, int j)
{
    NiceAssert( i >= 0 );
    NiceAssert( j >= 0 );

    resetvecID();
    killaltcontent();
    killnearfar();

    // squareswap ( i > j ): ( c ) (i)          ( c ) (i)
    //                       ( d ) (1)          ( f ) (1)
    //                       ( e ) (j-i)  ->    ( e ) (j-i)
    //                       ( f ) (1)          ( d ) (1)
    //                       ( g ) (...)        ( g ) (...)
    //
    // squareswap ( i < j ): ( c ) (j)          ( c ) (j)
    //                       ( d ) (1)          ( f ) (1)
    //                       ( e ) (i-j)  ->    ( e ) (i-j)
    //                       ( f ) (1)          ( d ) (1)
    //                       ( g ) (...)        ( g ) (...)

    if ( nearnonsparse() && ( i < size() ) && ( j < size() ) )
    {
        (*content).squareswap(i,j);
    }

    else
    {
        if ( i != j )
        {
            if ( isindpresent(i) && isindpresent(j) )
            {
                qswap((*this)("&",i),(*this)("&",j));
            }

            else if ( isindpresent(i) )
            {
                qswap((*this)("&",i),(*this)("&",j));

                zero(i);
            }

            else if ( isindpresent(j) )
            {
                qswap((*this)("&",i),(*this)("&",j));

                zero(j);
            }

            // Nothing to do in other case (swapping two zeros that refer back
            // to the same variable in any case)
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::nearadd(const SparseVector<T> &right_op)
{
    resetvecID();
    killaltcontent();
    killnear();

    if ( nearnonsparse() && right_op.nearnonsparse() && ( size() == right_op.size() ) )
    {
        retVector<T> tmpva;
        retVector<T> tmpvb;

        (*this)("&",ind(),tmpva) += right_op(right_op.ind(),tmpvb);

        return *this;
    }

    if ( right_op.indsize() == 0 )
    {
        return *this;
    }

    int i = 0;
    int j = 0;

    for ( j = 0 ; j < right_op.nearindsize() ; j++ )
    {
        if ( i >= nearindsize() )
        {
            NiceAssert( i == nearindsize() );

            append(right_op.ind(j));
        }

        if ( ind(i) < right_op.ind(j) )
        {
            while ( ind(i) < right_op.ind(j) )
            {
                i++;

                if ( i >= nearindsize() )
                {
                    NiceAssert( i == nearindsize() );

                    append(right_op.ind(j));
                }
            }
        }

        if ( ind(i) > right_op.ind(j) )
        {
            add(right_op.ind(j),i);
        }

        NiceAssert( i < nearindsize() );
        NiceAssert( ind(i) == right_op.ind(j) );

        direref(i) += right_op.direcref(j);

        i++;
    }

    return *this;
}


template <class T>
template <class S> SparseVector<T> &SparseVector<T>::scaleAdd(const S &a, const SparseVector<T> &right_op)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( nearnonsparse() && right_op.nearnonsparse() && ( size() == right_op.size() ) )
    {
        retVector<T> tmpva;
        retVector<S> tmpvb;

        ((*this)("&",(*this).ind(),tmpva)).scaleAdd(a,right_op(right_op.ind(),tmpvb));

        return *this;
    }

    SparseVector<T> &left_op = *this;

    if ( right_op.indsize() == 0 )
    {
        return left_op;
    }

    int i = 0;
    int j = 0;

    for ( j = 0 ; j < right_op.indsize() ; j++ )
    {
        if ( i >= left_op.indsize() )
        {
            NiceAssert( i == left_op.indsize() );

            left_op.append(right_op.ind(j));
        }

        if ( left_op.ind(i) < right_op.ind(j) )
        {
            while ( left_op.ind(i) < right_op.ind(j) )
            {
                i++;

                if ( i >= left_op.indsize() )
                {
                    NiceAssert( i == left_op.indsize() );

                    left_op.append(right_op.ind(j));
                }
            }
        }

        if ( left_op.ind(i) > right_op.ind(j) )
        {
            left_op.add(right_op.ind(j),i);
        }

        NiceAssert( i < left_op.indsize() );
        NiceAssert( left_op.ind(i) == right_op.ind(j) );

        left_op.direref(i) += (a*(right_op.direcref(j)));

        i++;
    }

    return left_op;
}

template <class T>
template <class S> SparseVector<T> &SparseVector<T>::scaleAddR(const SparseVector<T> &right_op, const S &b)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( nearnonsparse() && right_op.nearnonsparse() && ( size() == right_op.size() ) )
    {
        ((*this)("&",(*this).ind())).scaleAddR(right_op(right_op.ind()),6);

        return *this;
    }

    SparseVector<T> &left_op = *this;

    if ( right_op.indsize() == 0 )
    {
        return left_op;
    }

    int i = 0;
    int j = 0;

    for ( j = 0 ; j < right_op.indsize() ; j++ )
    {
        if ( i >= left_op.indsize() )
        {
            NiceAssert( i == left_op.indsize() );

            left_op.append(right_op.ind(j));
        }

        if ( left_op.ind(i) < right_op.ind(j) )
        {
            while ( left_op.ind(i) < right_op.ind(j) )
            {
                i++;

                if ( i >= left_op.indsize() )
                {
                    NiceAssert( i == left_op.indsize() );

                    left_op.append(right_op.ind(j));
                }
            }
        }

        if ( left_op.ind(i) > right_op.ind(j) )
        {
            left_op.add(right_op.ind(j),i);
        }

        NiceAssert( i < left_op.indsize() );
        NiceAssert( left_op.ind(i) == right_op.ind(j) );

        left_op.direref(i) += (right_op.direcref(j)*b);

        i++;
    }

    return left_op;
}

template <class T>
template <class S> SparseVector<T> &SparseVector<T>::scaleAddB(const T &a, const SparseVector<S> &right_op)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( nearnonsparse() && right_op.nearnonsparse() && ( size() == right_op.size() ) )
    {
        ((*this)("&",(*this).ind())).scaleAddB(a,right_op(right_op.ind()));

        return *this;
    }

    SparseVector<T> &left_op = *this;

    if ( right_op.indsize() == 0 )
    {
        return left_op;
    }

    int i = 0;
    int j = 0;

    for ( j = 0 ; j < right_op.indsize() ; j++ )
    {
        if ( i >= left_op.indsize() )
        {
            NiceAssert( i == left_op.indsize() );

            left_op.append(right_op.ind(j));
        }

        if ( left_op.ind(i) < right_op.ind(j) )
        {
            while ( left_op.ind(i) < right_op.ind(j) )
            {
                i++;

                if ( i >= left_op.indsize() )
                {
                    NiceAssert( i == left_op.indsize() );

                    left_op.append(right_op.ind(j));
                }
            }
        }

        if ( left_op.ind(i) > right_op.ind(j) )
        {
            left_op.add(right_op.ind(j),i);
        }

        NiceAssert( i < left_op.indsize() );
        NiceAssert( left_op.ind(i) == right_op.ind(j) );

        left_op.direref(i) += (a*right_op.direcref(j));

        i++;
    }

    return left_op;
}

template <class T>
template <class S> SparseVector<T> &SparseVector<T>::scaleAddBR(const SparseVector<S> &right_op, const T &b)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( nearnonsparse() && right_op.nearnonsparse() && ( size() == right_op.size() ) )
    {
        ((*this)("&",(*this).ind())).scaleAddBR(right_op(right_op.ind()),b);

        return *this;
    }

    SparseVector<T> &left_op = *this;

    if ( right_op.indsize() == 0 )
    {
        return left_op;
    }

    int i = 0;
    int j = 0;

    for ( j = 0 ; j < right_op.indsize() ; j++ )
    {
        if ( i >= left_op.indsize() )
        {
            NiceAssert( i == left_op.indsize() );

            left_op.append(right_op.ind(j));
        }

        if ( left_op.ind(i) < right_op.ind(j) )
        {
            while ( left_op.ind(i) < right_op.ind(j) )
            {
                i++;

                if ( i >= left_op.indsize() )
                {
                    NiceAssert( i == left_op.indsize() );

                    left_op.append(right_op.ind(j));
                }
            }
        }

        if ( left_op.ind(i) > right_op.ind(j) )
        {
            left_op.add(right_op.ind(j),i);
        }

        NiceAssert( i < left_op.indsize() );
        NiceAssert( left_op.ind(i) == right_op.ind(j) );

        left_op.direref(i) += (right_op.direcref(j)*b);

        i++;
    }

    return left_op;
}


template <class T>
template <class S> SparseVector<T> &SparseVector<T>::scale(const S &a)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    // Don't zero if a = 0.0 - we want to keep indices (and speed)

    if ( indsize() )
    {
        int i;

        for ( i = 0 ; i < indsize() ; i++ )
        {
            direref(i) *= a;
        }
    }

    return *this;
}

template <class T>
template <class S> SparseVector<T> &SparseVector<T>::scale(const SparseVector<S> &a)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( nearnonsparse() && a.nearnonsparse() && ( size() == a.size() ) )
    {
        ((*this)("&",(*this).ind())).scale(a(a.ind()));

        return *this;
    }

    int lsize = indsize();
    int rsize = a.indsize();

    if ( lsize && rsize )
    {
	int lpos = 0;
	int rpos = 0;
	int lelm;
        int relm;

	while ( ( lpos < lsize ) && ( rpos < rsize ) )
	{
            lelm = ind(lpos);
            relm = a.ind(rpos);

	    if ( lelm == relm )
	    {
                direref(lpos) *= a.direcref(rpos);

		lpos++;
                rpos++;
	    }

	    else if ( lelm < relm )
	    {
                zero(lelm);
                lsize--;
	    }

	    else
	    {
                rpos++;
	    }
	}

	while ( lpos < lsize )
	{
            zero(ind(lpos));
            lsize--;
	}
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::lscale(const T &a)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( indsize() )
    {
        int i;

        for ( i = 0 ; i < indsize() ; i++ )
        {
            rightmult(a,direref(i));
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::lscale(const SparseVector<T> &a)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( nearnonsparse() && a.nearnonsparse() && ( size() == a.size() ) )
    {
        retVector<T> tmpva;
        retVector<T> tmpvb;

        ((*this)("&",(*this).ind(),tmpva)).lscale(a(a.ind(),tmpvb));

        return *this;
    }

    int lsize = indsize();
    int rsize = a.indsize();

    if ( lsize && rsize )
    {
	int lpos = 0;
	int rpos = 0;
	int lelm;
        int relm;

	while ( ( lpos < lsize ) && ( rpos < rsize ) )
	{
            lelm = ind(lpos);
            relm = a.ind(rpos);

	    if ( lelm == relm )
	    {
                rightmult(a.direcref(rpos),direref(lpos));

		lpos++;
                rpos++;
	    }

	    else if ( lelm < relm )
	    {
                zero(lelm);
                lsize--;
	    }

	    else
	    {
                rpos++;
	    }
	}

	while ( lpos < lsize )
	{
            zero(ind(lpos));
            lsize--;
	}
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::resize(int i)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    NiceAssert( i >= 0 );

    while ( size() > i )
    {
        remove(ind(indsize()-1));
    }

    while ( size() < i )
    {
        (*this)("&",size()) = zerelm();
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::resize(const Vector<int> &itu)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( !(itu.size()) )
    {
        zero();
    }

    else
    {
        int i,j,ispresent;

        // add missing indices

        for ( i = 0 ; i < itu.size() ; i++ )
        {
            NiceAssert( itu(i) >= 0 );

            if ( !isindpresent(itu(i)) )
            {
                (*this)("&",itu(i)) = zerelm();
            }
        }

        // remove superfluous indices

        for ( i = indsize()-1 ; ( i >= 0 ) && ( indsize() > itu.size() ) ; i-- )
        {
            ispresent = 0;

            for ( j = 0 ; j < itu.size() ; j++ )
            {
                if ( ind(i) == itu(j) )
                {
                    ispresent = 1;
                    break;
                }
            }

            if ( !ispresent )
            {
                zero(ind(i));
            }
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::setorder(int i)
{
    return resize(1<<i);
}

template <class T>
SparseVector<T> &SparseVector<T>::append(int i, const T &a)
{
    NiceAssert( i >= size() );

    resetvecID();
    killaltcontent();
    killnearfar();

    int pos = indsize();

    (*indices).add(pos);
    (*indices)("&",pos) = i;
    (*content).add(pos);
    (*content)("&",pos) = a;

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::append(int i, const SparseVector<T> &a)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    NiceAssert( i >= size() );

    if ( a.indsize() )
    {
        int j;

        for ( j = 0 ; j < a.indsize() ; j++ )
        {
            (*this)("&",i+(a.ind(j))) = a.direcref(j);
        }
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::add(int i, int ipos)
{
    NiceAssert( ipos <= indsize() );
    NiceAssert( ipos >= 0 );
    NiceAssert( ( ipos == indsize() ) || ( i < ind(ipos) ) );
    NiceAssert( !ipos || ( i > ind(ipos-1) ) );

    resetvecID();
    killaltcontent();
    killnearfar();

    (*indices).add(ipos);
    (*content).add(ipos);
    (*indices)("&",ipos) = i;
    (*content)("&",ipos) = zerelm();

    return *this;
}


template <class T>
SparseVector<T> &SparseVector<T>::applyon(T (*fn)(T))
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( indsize() )
    {
	int i;

        for ( i = 0 ; i < indsize() ; i++ )
	{
            (*this).direref(i) = fn((*this).direcref(i));
	}
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::applyon(T &(*fn)(T &))
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( indsize() )
    {
	int i;

        for ( i = 0 ; i < indsize() ; i++ )
	{
            fn((*this).direref(i));
	}
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::applyon(T &(*fn)(T &, const void *), const void *a)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( indsize() )
    {
	int i;

        for ( i = 0 ; i < indsize() ; i++ )
	{
            fn((*this).direref(i),a);
	}
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::applyon(T (*fn)(const T &))
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( indsize() )
    {
	int i;

        for ( i = 0 ; i < indsize() ; i++ )
	{
            (*this).direref(i) = fn((*this).direcref(i));
	}
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::applyon(T (*fn)(T, const void *), const void *a)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( indsize() )
    {
	int i;

        for ( i = 0 ; i < indsize() ; i++ )
	{
            (*this).direref(i) = fn((*this).direcref(i),a);
	}
    }

    return *this;
}


template <class T>
SparseVector<T> &SparseVector<T>::applyon(T (*fn)(const T &, const void *), const void *a)
{
    resetvecID();
    killaltcontent();
    killnearfar();

    if ( indsize() )
    {
	int i;

        for ( i = 0 ; i < indsize() ; i++ )
	{
            (*this).direref(i) = fn((*this).direcref(i),a);
	}
    }

    return *this;
}



























template <class T>
const T &max(const SparseVector<T> &a, int &ii)
{
    // NB: indsize zero means all elements the same.

    ii = 0;

    if ( a.indsize() )
    {
	int i;

        for ( i = 0 ; i < a.indsize() ; i++ )
	{
            if ( !i || ( a.direcref(i) > a.direcref(ii) ) )
	    {
                ii = i;
	    }
	}

        ii = a.ind(ii);
    }

    return a(ii);
}

template <class T>
const T &min(const SparseVector<T> &a, int &ii)
{
    // NB: indsize zero means all elements the same.

    ii = 0;

    if ( a.indsize() )
    {
        int i;

        for ( i = 0 ; i < a.indsize() ; i++ )
	{
            if ( !i || ( a.direcref(i) < a.direcref(ii) ) )
	    {
                ii = i;
	    }
	}

        ii = a.ind(ii);
    }

    return a(ii);
}

template <class T>
const T &max(T &maxval, const SparseVector<T> &a, const SparseVector<T> &b, int &ii)
{
    T locval;
    int isab = 0;

    ii = 0;

    maxval  = a(0);
    maxval -= b(0);

    if ( a.indsize() )
    {
	int i;

        for ( i = 0 ; i < a.indsize() ; i++ )
	{
            locval  = a.direcref(i);
            locval -= b(a.ind(i));

            if ( ( !isab && !i ) || ( locval > maxval ) )
	    {
		maxval = locval;
                ii = i;
                isab = 1;
	    }
        }
    }

    if ( b.indsize() )
    {
	int i;

        for ( i = 0 ; i < b.indsize() ; i++ )
	{
            locval  = a(b.ind(i));
            locval -= b.direcref(i);

            if ( ( !isab && !i ) || ( locval > maxval ) )
	    {
		maxval = locval;
                ii = i;
                isab = 2;
	    }
	}
    }

    if ( isab == 1 )
    {
        ii = a.ind(ii);
    }

    else if ( isab == 2 )
    {
        ii = b.ind(ii);
    }

    return maxval;
}

template <class T>
T max(const SparseVector<T> &a, const SparseVector<T> &b, int &ii)
{
    T res;

    return max(res,a,b,ii);
}

template <class T>
const T &min(T &maxval, const SparseVector<T> &a, const SparseVector<T> &b, int &ii)
{
    T locval;
    int isab = 0;

    ii = 0;

    maxval  = a(0);
    maxval -= b(0);

    if ( a.indsize() )
    {
	int i;

        for ( i = 0 ; i < a.indsize() ; i++ )
	{
            locval  = a.direcref(i);
            locval -= b(a.ind(i));

            if ( ( !isab && !i ) || ( locval < maxval ) )
	    {
		maxval = locval;
                ii = i;
                isab = 1;
	    }
        }
    }

    if ( b.indsize() )
    {
	int i;

        for ( i = 0 ; i < b.indsize() ; i++ )
	{
            locval  = a(b.ind(i));
            locval -= b.direcref(i);

            if ( ( !isab && !i ) || ( locval < maxval ) )
	    {
		maxval = locval;
                ii = i;
                isab = 2;
	    }
	}
    }

    if ( isab == 1 )
    {
        ii = a.ind(ii);
    }

    else if ( isab == 2 )
    {
        ii = b.ind(ii);
    }

    return maxval;
}

template <class T>
T min(const SparseVector<T> &a, const SparseVector<T> &b, int &ii)
{
    T res;

    return max(res,a,b,ii);
}

template <class T>
const T &maxabs(T &res, const SparseVector<T> &a, int &ii)
{
    // NB: indsize zero means all elements the same.

    ii = 0;

    if ( a.indsize() )
    {
	int i;

        for ( i = 0 ; i < a.indsize() ; i++ )
	{
            if ( !i || ( abs2(a.direcref(i)) > abs2(a.direcref(ii)) ) )
	    {
                ii = i;
	    }
	}

        ii = a.ind(ii);
    }

    res = abs2(a(ii));

    return res;
}

template <class T>
T maxabs(const SparseVector<T> &a, int &ii)
{
    T res;

    return maxabs(res,a,ii);
}

template <class T>
const T &minabs(T &res, const SparseVector<T> &a, int &ii)
{
    // NB: indsize zero means all elements the same.

    ii = 0;

    if ( a.indsize() )
    {
	int i;

        for ( i = 0 ; i < a.indsize() ; i++ )
	{
            if ( !i || ( abs2(a.direcref(i)) < abs2(a.direcref(ii)) ) )
	    {
                ii = i;
	    }
	}

        ii = a.ind(ii);
    }

    res = abs2(a(ii));

    return res;
}

template <class T>
T minabs(const SparseVector<T> &a, int &ii)
{
    T res;

    return minabs(res,a,ii);
}

template <class T>
const T &sqabsmax(T &res, const SparseVector<T> &a)
{
    // NB: indsize zero means all elements the same.

    int ii = 0;

    if ( a.indsize() )
    {
	int i;

        for ( i = 0 ; i < a.indsize() ; i++ )
	{
            if ( !i || ( norm2(a.direcref(i)) > norm2(a.direcref(ii)) ) )
	    {
                ii = i;
	    }
	}
    }

    res = norm2(a.direcref(ii));

    return res;
}

template <class T>
T sqabsmax(const SparseVector<T> &a)
{
    T res;

    return sqabsmax(res,a);
}

template <class T>
const T &sqabsmin(T &res, const SparseVector<T> &a)
{
    // NB: indsize zero means all elements the same.

    int ii = 0;

    if ( a.indsize() )
    {
	int i;

        for ( i = 0 ; i < a.indsize() ; i++ )
	{
            if ( !i || ( norm2(a.direcref(i)) < norm2(a.direcref(ii)) ) )
	    {
                ii = i;
	    }
	}
    }

    res = norm2(a.direcref(ii));

    return res;
}

template <class T>
T sqabsmin(const SparseVector<T> &a)
{
    T res;

    return sqabsmin(res,a);
}

template <class T>
T sum(const SparseVector<T> &a)
{
    T res;

    return sum(res,a);
}

template <class T>
T sqsum(const SparseVector<T> &a)
{
    T res;

    return sqsum(res,a);
}

template <class T>
const T &sum(T &res, const SparseVector<T> &a)
{
    if ( a.indsize() )
    {
        res = a.direcref(0);

        if ( a.indsize() > 1 )
        {
            int i;

            for ( i = 1 ; i < a.indsize() ; i++ )
            {
                res += a.direcref(i);
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
const T &sqsum(T &res, const SparseVector<T> &a)
{
    T temp;

    if ( a.indsize() )
    {
        res =  a.direcref(0);
        res *= a.direcref(0);

        if ( a.indsize() > 1 )
        {
            int i;

            for ( i = 1 ; i < a.indsize() ; i++ )
            {
                temp =  a.direcref(i);
                temp *= a.direcref(i);

                res += temp;
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
const T &prod(T &res, const SparseVector<T> &a)
{
    if ( a.indsize() )
    {
        res = a.direcref(0);

        if ( a.indsize() > 1 )
        {
            int i;

            for ( i = 1 ; i < a.indsize() ; i++ )
            {
                res *= a.direcref(i);
            }
	}
    }

    else
    {
        setident(res);
    }

    return res;
}

template <class T>
T prod(const SparseVector<T> &a)
{
    T res;

    return prod(res,a);
}

template <class T>
const T &Prod(T &res, const SparseVector<T> &a)
{
    if ( a.indsize() )
    {
        res = a.direcref(a.indsize()-1);

        if ( a.indsize() > 1 )
        {
            int i;

            for ( i = a.indsize()-2 ; i >= 0 ; i-- )
            {
                res *= a.direcref(i);
            }
	}
    }

    else
    {
        setident(res);
    }

    return res;
}

template <class T>
T Prod(const SparseVector<T> &a)
{
    T res;

    return Prod(res,a);
}







































template <class T>
const T &indexedsum(T &res, const Vector<int> &n, const SparseVector<T> &a)
{
    if ( n.size() )
    {
        res = a(n(0));

        if ( n.size() > 1 )
        {
            int i;

            for ( i = 1 ; i < n.size() ; i++ )
            {
                res += a(n(i));
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
const T &indexedsqsum(T &res, const Vector<int> &n, const SparseVector<T> &a)
{
    T temp;

    if ( n.size() )
    {
        res =  a(n(0));
        res *= a(n(0));

        if ( n.size() > 1 )
        {
            int i;

            for ( i = 1 ; i < n.size() ; i++ )
            {
                temp =  a(n(i));
                temp *= a(n(i));

                res += temp;
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
const T &indexedmean(T &res, const Vector<int> &n, const SparseVector<T> &a)
{
    if ( n.size() )
    {
        indexedsum(res,n,a);
        res *= 1/((double) n.size());
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
const T &indexedmedian(const Vector<int> &n, const SparseVector<T> &a, int &ii)
{
    ii = 0;

    if ( n.size() == 1 )
    {
        ii = n(0);
    }

    else if ( n.size() > 1 )
    {
        // Aim: a.direcref(outdex) should be arranged from largest to smallest

        Vector<int> outdex;

        int i,j;

        for ( i = n.size()-1 ; i >= 0 ; i-- )
        {
            j = 0;

            if ( outdex.size() )
            {
                for ( j = 0 ; j < outdex.size() ; j++ )
                {
                    if ( a(n(outdex(j))) <= a(n(i)) )
                    {
                        break;
                    }
                }
            }

            outdex.add(j);
            outdex("&",j) = i;
        }

        ii = n(outdex(outdex.size()/2));
    }

    return a(ii);
}

template <class T>
const T &indexedsqmean(T &res, const Vector<int> &n, const SparseVector<T> &a)
{
    if ( n.size() )
    {
        indexedsqsum(res,n,a);
        res *= 1/((double) n.size());
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
const T &indexedvari(T &res, const Vector<int> &n, const SparseVector<T> &a)
{
    // mean(a) = 1/N sum_i a_i
    // sqmean(a) = 1/N sum_i a_i^2
    //
    // vari(a) = 1/N sum_i ( a_i - mean(a) )^2
    //         = 1/N sum_i a_i^2 - 2 mean(a) 1/N sum_i a_i + mean(a)^2 1/N sum_i 1
    //         = sqmean(a) - 2 mean(a)^2 + mean(a)^2
    //         = sqmean(a) - mean(a)^2

    T ameansq;

    indexedsqmean(res,n,a);
    indexedmean(ameansq,n,a);
    ameansq *= ameansq;
    res -= ameansq;

    return res;
}

template <class T>
const T &indexedstdev(T &res, const Vector<int> &n, const SparseVector<T> &a)
{
    res = sqrt(indexedvari(res,n,a));

    return res;
}
template <class T>
const T &indexedmax(const Vector<int> &n, const SparseVector<T> &a, int &ii)
{
    // NB: indsize zero means all elements the same.

    ii = 0;

    if ( n.size() )
    {
	int i;

        for ( i = 0 ; i < n.size() ; i++ )
	{
            if ( !i || ( a(n(i)) > a(ii) ) )
	    {
                ii = n(i);
	    }
	}
    }

    return a(ii);
}

template <class T>
const T &indexedmin(const Vector<int> &n, const SparseVector<T> &a, int &ii)
{
    // NB: indsize zero means all elements the same.

    ii = 0;

    if ( n.size() )
    {
	int i;

        for ( i = 0 ; i < n.size() ; i++ )
	{
            if ( !i || ( a(n(i)) < a(ii) ) )
	    {
                ii = n(i);
	    }
	}
    }

    return a(ii);
}

template <class T>
const T &indexedmaxabs(T &res, const Vector<int> &n, const SparseVector<T> &a, int &ii)
{
    // NB: indsize zero means all elements the same.

    ii = 0;

    if ( n.size() )
    {
	int i;

        for ( i = 0 ; i < n.size() ; i++ )
	{
            if ( !i || ( abs2(a(n(i))) > abs2(a(ii)) ) )
	    {
                ii = n(i);
	    }
	}
    }

    res = abs2(a(ii));

    return res;
}















































template <class T>
const T &mean(T &res, const SparseVector<T> &a)
{
    if ( a.indsize() )
    {
        sum(res,a);
        res *= 1/((double) a.indsize());
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
T mean(const SparseVector<T> &a)
{
    T res;

    return mean(res,a);
}

template <class T>
const T &sqmean(T &res, const SparseVector<T> &a)
{
    if ( a.indsize() )
    {
        sqsum(res,a);
        res *= 1/((double) a.indsize());
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
T sqmean(const SparseVector<T> &a)
{
    T res;

    return sqmean(res,a);
}

template <class T>
const T &vari(T &res, const SparseVector<T> &a)
{
    // mean(a) = 1/N sum_i a_i
    // sqmean(a) = 1/N sum_i a_i^2
    //
    // vari(a) = 1/N sum_i ( a_i - mean(a) )^2
    //         = 1/N sum_i a_i^2 - 2 mean(a) 1/N sum_i a_i + mean(a)^2 1/N sum_i 1
    //         = sqmean(a) - 2 mean(a)^2 + mean(a)^2
    //         = sqmean(a) - mean(a)^2

    T ameansq;

    sqmean(res,a);
    mean(ameansq,a);
    ameansq *= ameansq;
    res -= ameansq;

    return res;
}

template <class T>
T vari(const SparseVector<T> &a)
{
    T res;

    return vari(res,a);
}

template <class T>
const T &stdev(T &res, const SparseVector<T> &a)
{
    res = sqrt(vari(a));

    return res;
}

template <class T>
T stdev(const SparseVector<T> &a)
{
    T res;

    return stdev(res,a);
}

template <class T>
const T &median(const SparseVector<T> &a, int &ii)
{
    ii = 0;

    if ( a.indsize() == 1 )
    {
        ii = a.ind(0);
    }

    else if ( a.indsize() > 1 )
    {
        // Aim: a.direcref(outdex) should be arranged from largest to smallest

        Vector<int> outdex;

        int i,j;

        for ( i = a.indsize()-1 ; i >= 0 ; i-- )
        {
            j = 0;

            if ( outdex.size() )
            {
                for ( j = 0 ; j < outdex.size() ; j++ )
                {
                    if ( a.direcref(outdex(j)) <= a.direcref(i) )
                    {
                        break;
                    }
                }
            }

            outdex.add(j);
            outdex("&",j) = i;
        }

        ii = outdex(a.indsize()/2);
        ii = a.ind(ii);
    }

    return a(ii); // will be reference to zero if a has no elements
}
















template <class S> SparseVector<S> angle(const SparseVector<S> &a)
{
    SparseVector<S> temp(a);

    double tempabs = abs2(temp);

    if ( tempabs != 0.0 )
    {
        temp.scale(1/tempabs);
    }

    return temp;
}

template <class S> SparseVector<S> vangle(const SparseVector<S> &a, const SparseVector<S> &defsign)
{
    SparseVector<S> temp(a);

    double tempabs = abs2(temp);

    if ( tempabs != 0.0 )
    {
        temp.scale(1/tempabs);
    }

    else
    {
        temp = defsign;
    }

    return temp;
}

template <class S> double abs1(const SparseVector<S> &a)
{
    return norm1(a);
}

template <class S> double abs2(const SparseVector<S> &a)
{
    return sqrt(norm2(a));
}

template <class S> double absd(const SparseVector<S> &a)
{
    return abs2(a);
}

template <class S> double normd(const SparseVector<S> &a)
{
    return norm2(a);
}

template <class S> double absp(const SparseVector<S> &a, double p)
{
    return pow(normp(a,p),1/p);
}

template <class S> double absinf(const SparseVector<S> &a)
{
    return absinfPrelude(a);
}

template <class S> double absinfPrelude(const SparseVector<S> &a)
{
    double maxval = 0;
    double locval;

    if ( a.indsize() )
    {
        maxval = absinf(a.direcref(0));

        if ( a.indsize() > 1 )
	{
	    int i;

            for ( i = 1 ; i < a.indsize() ; i++ )
	    {
                locval = absinf(a.direcref(i));

		if ( locval > maxval )
		{
		    maxval = locval;
		}
	    }
	}
    }

    return maxval;
}

template <class S> double abs0(const SparseVector<S> &a)
{
    return abs0Prelude(a);
}

template <class S> double abs0Prelude(const SparseVector<S> &a)
{
    double minval = 0;
    double locval;

    if ( a.indsize() )
    {
        minval = absinf(a.direcref(0));

        if ( a.indsize() > 1 )
	{
	    int i;

            for ( i = 1 ; i < a.indsize() ; i++ )
	    {
                locval = absinf(a.direcref(i));

                if ( locval < minval )
		{
                    minval = locval;
		}
	    }
	}
    }

    return minval;
}

template <class S> double norm2(const SparseVector<S> &a)
{
    return norm2Prelude(a);
}

template <class S> double norm2Prelude(const SparseVector<S> &a)
{
    int i;
    double result = 0;

    if ( a.indsize() )
    {
        result = norm2(a.direcref(0));

        if ( a.indsize() > 1 )
	{
            for ( i = 1 ; i < a.indsize() ; i++ )
	    {
                result += (double) norm2(a.direcref(i));
	    }
	}
    }

    return result;
}

template <class S> double norm1(const SparseVector<S> &a)
{
    return norm1Prelude(a);
}

template <class S> double norm1Prelude(const SparseVector<S> &a)
{
    int i;
    double result = 0;

    if ( a.indsize() )
    {
        result = norm1(a.direcref(0));

        if ( a.indsize() > 1 )
	{
            for ( i = 1 ; i < a.indsize() ; i++ )
	    {
                result += (double) norm1(a.direcref(i));
	    }
	}
    }

    return result;
}

template <class S> double normp(const SparseVector<S> &a, double p)
{
    return normpPrelude(a,p);
}

template <class S> double normpPrelude(const SparseVector<S> &a, double p)
{
    int i;
    double result = 0;

    if ( a.indsize() )
    {
        result = normp(a.direcref(0),p);

        if ( a.indsize() > 1 )
	{
            for ( i = 1 ; i < a.indsize() ; i++ )
	    {
                result += (double) normp(a.direcref(i),p);
	    }
	}
    }

    return result;
}

template <class T> SparseVector<T> &setident(SparseVector<T> &a)
{
    return a.ident();
}

template <class S> SparseVector<double> &seteabs2(SparseVector<S> &a)
{
    if ( a.indsize() )
    {
        int i;

        for ( i = 0 ; i < a.indsize() ; i++ )
        {
            a.direref(i) = abs2(a.direcref(i));
        }
    }

    return a;
}

template <class T> SparseVector<T> &setzero(SparseVector<T> &a)
{
    return a.zero();
}

template <class T> SparseVector<T> &setposate(SparseVector<T> &a)
{
    return a.posate();
}

template <class T> SparseVector<T> &setnegate(SparseVector<T> &a)
{
    return a.negate();
}

template <class T> SparseVector<T> &setconj(SparseVector<T> &a)
{
    return a.conj();
}

template <class T> SparseVector<T> &setrand(SparseVector<T> &a)
{
    return a.rand();
}


#define BOUND_DEF \
\
    int asize = ( afaroff == 3 ) ? a.farfarfarindsize() : ( ( afaroff == 2 ) ? a.farfarindsize() : ( ( afaroff == 1 ) ? a.farindsize() : a.nearindsize() ) ); \
    int bsize = ( bfaroff == 3 ) ? b.farfarfarindsize() : ( ( bfaroff == 2 ) ? b.farfarindsize() : ( ( bfaroff == 1 ) ? b.farindsize() : b.nearindsize() ) ); \
\
    int astart = ( afaroff == 3 ) ? a.nearfarfarindsize() : ( ( afaroff == 2 ) ? a.nearfarindsize() : ( ( afaroff == 1 ) ? a.nearindsize() : 0 ) ); \
    int bstart = ( bfaroff == 3 ) ? b.nearfarfarindsize() : ( ( bfaroff == 2 ) ? b.nearfarindsize() : ( ( bfaroff == 1 ) ? b.nearindsize() : 0 ) ); \
\
    int abaseind = ( afaroff == 3 ) ? INDFARFARFAROFFSTART : ( ( afaroff == 2 ) ? INDFARFAROFFSTART : ( ( afaroff == 1 ) ? INDFAROFFSTART : 0 ) ); \
    int bbaseind = ( bfaroff == 3 ) ? INDFARFARFAROFFSTART : ( ( bfaroff == 2 ) ? INDFARFAROFFSTART : ( ( bfaroff == 1 ) ? INDFAROFFSTART : 0 ) );





































// ------------------------------------
//phantomxyz

template <class T>
T &twoProduct(T &result, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff, int bfaroff)
{
    return twoProductPrelude(result,a,b,afaroff,bfaroff);
}

template <class T>
T &twoProductPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( a.size() == b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;

            twoProduct(result,a(a.ind(),tmpva),b(b.ind(),tmpvb));
        }

        else if ( a.size() < b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProduct(result,a(a.ind(),tmpva),b(b.ind(),tmpvb)(0,1,a.size()-1,tmpvc));
        }

        else if ( a.size() > b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProduct(result,a(a.ind(),tmpva)(0,1,b.size()-1,tmpvc),b(b.ind(),tmpvb));
        }

        return result;
    }

    BOUND_DEF;

    T temp;

    setzero(temp);
    setzero(result);

    if ( asize && bsize )
    {
        int apos = astart;
        int bpos = bstart;
	int aelm;
        int belm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;

	    if ( aelm == belm )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                temp = b.direcref(bpos);
                setconj(temp);
                temp *= a.direcref(apos);
                setconj(temp);

                result += temp;

		apos++;
                bpos++;
	    }

	    else if ( aelm < belm )
	    {
		apos++;
	    }

	    else
	    {
                bpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductNoConj(T &result, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff, int bfaroff)
{
    return twoProductNoConjPrelude(result,a,b,afaroff,bfaroff);
}

template <class T>
T &twoProductNoConjPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( a.size() == b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;

            twoProductNoConj(result,a(a.ind(),tmpva),b(b.ind(),tmpvb));
        }

        else if ( a.size() < b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductNoConj(result,a(a.ind(),tmpva),b(b.ind(),tmpvb)(0,1,a.size()-1,tmpvc));
        }

        else if ( a.size() > b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductNoConj(result,a(a.ind(),tmpva)(0,1,b.size()-1,tmpvc),b(b.ind(),tmpvb));
        }

        return result;
    }

    BOUND_DEF;

    T temp;

    setzero(result);
    setzero(temp);

    if ( asize && bsize )
    {
        int apos = astart;
        int bpos = bstart;
	int aelm;
        int belm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;

	    if ( aelm == belm )
	    {
                temp = b.direcref(bpos);
                rightmult(a.direcref(apos),temp);

                result += temp;

		apos++;
                bpos++;
	    }

	    else if ( aelm < belm )
	    {
		apos++;
	    }

	    else
	    {
                bpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductRevConj(T &result, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff, int bfaroff)
{
    return twoProductRevConjPrelude(result,a,b,afaroff,bfaroff);
}

template <class T>
T &twoProductRevConjPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( a.size() == b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;

            twoProductRevConj(result,a(a.ind(),tmpva),b(b.ind(),tmpvb));
        }

        else if ( a.size() < b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductRevConj(result,a(a.ind(),tmpva),b(b.ind(),tmpvb)(0,1,a.size()-1,tmpvc));
        }

        else if ( a.size() > b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductRevConj(result,a(a.ind(),tmpva)(0,1,b.size()-1,tmpvc),b(b.ind(),tmpvb));
        }

        return result;
    }

    BOUND_DEF;

    T temp;

    setzero(result);
    setzero(temp);

    if ( asize && bsize )
    {
        int apos = astart;
        int bpos = bstart;
	int aelm;
        int belm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;

	    if ( aelm == belm )
	    {
                temp = b.direcref(bpos);
                setconj(temp);
                rightmult(a.direcref(apos),temp);

                result += temp;

		apos++;
                bpos++;
	    }

	    else if ( aelm < belm )
	    {
		apos++;
	    }

	    else
	    {
                bpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductScaled(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    return twoProductScaledPrelude(result,a,b,scale,afaroff,bfaroff);
}

template <class T>
T &twoProductScaledPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && scale.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == scale.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductScaled(result,a(a.ind(),tmpva),b(b.ind(),tmpvb),scale(scale.ind(),tmpvc));
        }

        else
        {
            int dim = ( a.size() < b.size() ) ? a.size() : b.size();
                dim = ( scale.size() < dim ) ? scale.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;

            twoProductScaled(result,a(a.ind(),tmpva)(0,1,dim-1,tmpvd),b(b.ind(),tmpvb)(0,1,dim-1,tmpve),scale(scale.ind(),tmpvc)(0,1,dim-1,tmpvf));
        }

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // conj(s.a).(s.b)

                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductScaledNoConj(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    return twoProductScaledNoConjPrelude(result,a,b,scale,afaroff,bfaroff);
}

template <class T>
T &twoProductScaledNoConjPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && scale.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == scale.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductScaledNoConj(result,a(a.ind(),tmpva),b(b.ind(),tmpvb),scale(scale.ind(),tmpvc));
        }

        else
        {
            int dim = ( a.size() < b.size() ) ? a.size() : b.size();
                dim = ( scale.size() < dim ) ? scale.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;

            twoProductScaledNoConj(result,a(a.ind(),tmpva)(0,1,dim-1,tmpvd),b(b.ind(),tmpvb)(0,1,dim-1,tmpve),scale(scale.ind(),tmpvc)(0,1,dim-1,tmpvf));
        }

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // (s.a).(s.b)

                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductScaledRevConj(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    return twoProductScaledRevConjPrelude(result,a,b,scale,afaroff,bfaroff);
}

template <class T>
T &twoProductScaledRevConjPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && scale.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == scale.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductScaledRevConj(result,a(a.ind(),tmpva),b(b.ind(),tmpvb),scale(scale.ind(),tmpvc));
        }

        else
        {
            int dim = ( a.size() < b.size() ) ? a.size() : b.size();
                dim = ( scale.size() < dim ) ? scale.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;

            twoProductScaledRevConj(result,a(a.ind(),tmpva)(0,1,dim-1,tmpvd),b(b.ind(),tmpvb)(0,1,dim-1,tmpve),scale(scale.ind(),tmpvc)(0,1,dim-1,tmpvf));
        }

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // (s.a).conj(s.b)

                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductLeftScaled(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && scale.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == scale.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductLeftScaled(result,a(a.ind(),tmpva),b(b.ind(),tmpvb),scale(scale.ind(),tmpvc));
        }

        else
        {
            int dim = ( a.size() < b.size() ) ? a.size() : b.size();
                dim = ( scale.size() < dim ) ? scale.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;

            twoProductLeftScaled(result,a(a.ind(),tmpva)(0,1,dim-1,tmpvd),b(b.ind(),tmpvb)(0,1,dim-1,tmpve),scale(scale.ind(),tmpvc)(0,1,dim-1,tmpvf));
        }

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // conj(s.a).(s.b)

                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                //tempb /= scale.direcref(dpos); - only half scaling

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductLeftScaledNoConj(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && scale.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == scale.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductLeftScaledNoConj(result,a(a.ind(),tmpva),b(b.ind(),tmpvb),scale(scale.ind(),tmpvc));
        }

        else
        {
            int dim = ( a.size() < b.size() ) ? a.size() : b.size();
                dim = ( scale.size() < dim ) ? scale.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;

            twoProductLeftScaledNoConj(result,a(a.ind(),tmpva)(0,1,dim-1,tmpvd),b(b.ind(),tmpvb)(0,1,dim-1,tmpve),scale(scale.ind(),tmpvc)(0,1,dim-1,tmpvf));
        }

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // (s.a).(s.b)

                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                //tempb /= scale.direcref(dpos); - only half scaling

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductLeftScaledRevConj(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && scale.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == scale.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductLeftScaledRevConj(result,a(a.ind(),tmpva),b(b.ind(),tmpvb),scale(scale.ind(),tmpvc));
        }

        else
        {
            int dim = ( a.size() < b.size() ) ? a.size() : b.size();
                dim = ( scale.size() < dim ) ? scale.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;

            twoProductLeftScaledRevConj(result,a(a.ind(),tmpva)(0,1,dim-1,tmpvd),b(b.ind(),tmpvb)(0,1,dim-1,tmpve),scale(scale.ind(),tmpvc)(0,1,dim-1,tmpvf));
        }

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // (s.a).conj(s.b)

                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                //tempb /= scale.direcref(dpos); - only half scaling

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductRightScaled(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && scale.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == scale.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductRightScaled(result,a(a.ind(),tmpva),b(b.ind(),tmpvb),scale(scale.ind(),tmpvc));
        }

        else
        {
            int dim = ( a.size() < b.size() ) ? a.size() : b.size();
                dim = ( scale.size() < dim ) ? scale.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;

            twoProductRightScaled(result,a(a.ind(),tmpva)(0,1,dim-1,tmpvd),b(b.ind(),tmpvb)(0,1,dim-1,tmpve),scale(scale.ind(),tmpvc)(0,1,dim-1,tmpvf));
        }

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // conj(s.a).(s.b)

                tempa  = a.direcref(apos);
                //tempa /= scale.direcref(cpos); - only half scaling

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductRightScaledNoConj(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && scale.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == scale.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductRightScaledNoConj(result,a(a.ind(),tmpva),b(b.ind(),tmpvb),scale(scale.ind(),tmpvc));
        }

        else
        {
            int dim = ( a.size() < b.size() ) ? a.size() : b.size();
                dim = ( scale.size() < dim ) ? scale.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;

            twoProductRightScaledNoConj(result,a(a.ind(),tmpva)(0,1,dim-1,tmpvd),b(b.ind(),tmpvb)(0,1,dim-1,tmpve),scale(scale.ind(),tmpvc)(0,1,dim-1,tmpvf));
        }

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // (s.a).(s.b)

                tempa  = a.direcref(apos);
                //tempa /= scale.direcref(cpos); - only half scaling

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductRightScaledRevConj(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && scale.nearnonsparse() && !afaroff && !bfaroff )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == scale.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductRightScaledRevConj(result,a(a.ind(),tmpva),b(b.ind(),tmpvb),scale(scale.ind(),tmpvc));
        }

        else
        {
            int dim = ( a.size() < b.size() ) ? a.size() : b.size();
                dim = ( scale.size() < dim ) ? scale.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;

            twoProductRightScaledRevConj(result,a(a.ind(),tmpva)(0,1,dim-1,tmpvd),b(b.ind(),tmpvb)(0,1,dim-1,tmpve),scale(scale.ind(),tmpvc)(0,1,dim-1,tmpvf));
        }

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // (s.a).conj(s.b)

                tempa  = a.direcref(apos);
                //tempa /= scale.direcref(cpos); - only half scaling

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProduct(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff, int bfaroff)
{
    BOUND_DEF;

    int nsize = n.size();

    T temp;

    setzero(result);
    setzero(temp);

    if ( nsize && asize && bsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;

        int nelm;
	int aelm;
	int belm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
            nelm = n(npos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                temp = b.direcref(bpos);
                setconj(temp);
                temp *= a.direcref(apos);
                setconj(temp);

                result += temp;

                npos++;
		apos++;
		bpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm )  )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm )  )
	    {
		apos++;
	    }

	    else
	    {
                bpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductNoConj(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff, int bfaroff)
{
    BOUND_DEF;

    int nsize = n.size();

    T temp;

    setzero(result);
    setzero(temp);

    if ( nsize && asize && bsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;

        int nelm;
	int aelm;
	int belm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
            nelm = n(npos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) )
	    {
                temp = b.direcref(bpos);
                rightmult(a.direcref(apos),temp);

                result += temp;

                npos++;
		apos++;
		bpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm )  )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm )  )
	    {
		apos++;
	    }

	    else
	    {
                bpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductRevConj(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, int afaroff, int bfaroff)
{
    BOUND_DEF;

    int nsize = n.size();

    T temp;

    setzero(result);
    setzero(temp);

    if ( nsize && asize && bsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;

        int nelm;
	int aelm;
	int belm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
            nelm = n(npos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) )
	    {
                temp = b.direcref(bpos);
                setconj(temp);
                rightmult(a.direcref(apos),temp);

                result += temp;

                npos++;
		apos++;
		bpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm )  )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm )  )
	    {
		apos++;
	    }

	    else
	    {
                bpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductScaled(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int nsize = n.size();
    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductScaledNoConj(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int nsize = n.size();
    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductScaledRevConj(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int nsize = n.size();
    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductLeftScaled(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int nsize = n.size();
    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                //tempb /= scale.direcref(dpos); - only half scaling

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductLeftScaledNoConj(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int nsize = n.size();
    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                //tempb /= scale.direcref(dpos); - only half scaling

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductLeftScaledRevConj(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int nsize = n.size();
    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a.direcref(apos);
                tempa /= scale.direcref(cpos);

                tempb  = b.direcref(bpos);
                //tempb /= scale.direcref(dpos); - only half scaling

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductRightScaled(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int nsize = n.size();
    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a.direcref(apos);
                //tempa /= scale.direcref(cpos); - only half scaling

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductRightScaledNoConj(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int nsize = n.size();
    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a.direcref(apos);
                //tempa /= scale.direcref(cpos); - only half scaling

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductRightScaledRevConj(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &scale, int afaroff, int bfaroff)
{
    // Basically just a four product with conjugation on a and scale included twice

    BOUND_DEF;

    int nsize = n.size();
    int csize = scale.indsize();
    int dsize = scale.indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int apos = astart;
        int bpos = bstart;
        int npos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

        while ( ( npos < nsize ) && ( apos-astart < asize ) && ( bpos-bstart < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;
	    celm = scale.ind(cpos);
	    delm = scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a.direcref(apos);
                //tempa /= scale.direcref(cpos); - only half scaling

                tempb  = b.direcref(bpos);
                tempb /= scale.direcref(dpos);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}


template <class T> 
double &twoProductAssumeReal(double &res, const SparseVector<T> &a, const SparseVector<T> &b)
{
    return twoProductAssumeRealPrelude(res,a,b);
}

template <class T> 
double &twoProductAssumeRealPrelude(double &res, const SparseVector<T> &a, const SparseVector<T> &b)
{
    if ( a.nearnonsparse() && b.nearnonsparse() )
    {
        if ( a.size() == b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;

            twoProductAssumeReal(res,a(a.ind(),tmpva),b(b.ind(),tmpvb));
        }

        else if ( a.size() < b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductAssumeReal(res,a(a.ind(),tmpva),b(b.ind(),tmpvb)(0,1,a.size()-1,tmpvc));
        }

        else if ( a.size() > b.size() )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            twoProductAssumeReal(res,a(a.ind(),tmpva)(0,1,b.size()-1,tmpvc),b(b.ind(),tmpvb));
        }

        return res;
    }

    int asize = a.nearindsize();
    int bsize = b.nearindsize();

    int astart = 0;
    int bstart = 0;

    int abaseind = 0;
    int bbaseind = 0;

    res  = 0;

    if ( asize && bsize )
    {
        int apos = astart;
        int bpos = bstart;
	int aelm;
        int belm;

        while ( ( apos-astart < asize ) && ( bpos-bstart < bsize ) )
	{
            aelm = a.ind(apos) - abaseind;
            belm = b.ind(bpos) - bbaseind;

	    if ( aelm == belm )
	    {
                scaladd(res,a.direcref(apos),b.direcref(bpos));

		apos++;
                bpos++;
	    }

	    else if ( aelm < belm )
	    {
		apos++;
	    }

	    else
	    {
                bpos++;
	    }
	}
    }

    return res;
}

template <class T> 
double &oneProductAssumeReal(double &res, const SparseVector<T> &a) 
{
    return oneProductAssumeRealPrelude(res,a);
}

template <class T> 
double &oneProductAssumeRealPrelude(double &res, const SparseVector<T> &a) 
{
    if ( a.nearnonsparse() )
    {
        retVector<T> tmpva;

        return oneProductAssumeReal(res,a(a.ind(),tmpva));
    }

    int asize = a.indsize();

    res = 0;

    if ( asize )
    {
	int apos = 0;

	while ( apos < asize )
	{
            res += (double) a.direcref(apos);

            apos++;
	}
    }

    return res;
}

template <class T> 
double &threeProductAssumeReal(double &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c) 
{
    return threeProductAssumeRealPrelude(res,a,b,c);
}

template <class T> 
double &threeProductAssumeRealPrelude(double &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c) 
{
    if ( a.nearnonsparse() && b.nearnonsparse() && c.nearnonsparse() )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == c.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;

            return threeProductAssumeReal(res,a(a.ind(),tmpva),b(b.ind(),tmpvb),c(c.ind(),tmpvc));
        }

        else
        {
            int dim = a.size();

            dim = ( b.size() < dim ) ? b.size() : dim;
            dim = ( c.size() < dim ) ? c.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;

            return threeProductAssumeReal(res,a(a.ind(),tmpva)(0,1,dim-1,tmpvd),b(b.ind(),tmpvb)(0,1,dim-1,tmpve),c(c.ind(),tmpvc)(0,1,dim-1,tmpvf));
        }
    }

    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();

    res = 0;

    if ( asize && bsize && csize )
    {
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
	int aelm;
	int belm;
	int celm;

	while ( ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) )
	{
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) )
	    {
                res += (double) (a.direcref(apos)*b.direcref(bpos)*c.direcref(cpos));

		apos++;
		bpos++;
		cpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) )
	    {
		bpos++;
	    }

            else
	    {
		cpos++;
	    }
	}
    }

    return res;
}


template <class T> 
double &fourProductAssumeReal(double &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d)
{
    return fourProductAssumeRealPrelude(res,a,b,c,d);
}

template <class T> 
double &fourProductAssumeRealPrelude(double &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d)
{
    if ( a.nearnonsparse() && b.nearnonsparse() && c.nearnonsparse() && d.nearnonsparse() )
    {
        if ( ( a.size() == b.size() ) && ( a.size() == c.size() ) && ( a.size() == d.size() ) )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;

            return fourProductAssumeReal(res,a(a.ind(),tmpva),b(b.ind(),tmpvb),c(c.ind(),tmpvc),d(d.ind(),tmpvd));
        }

        else
        {
            int dim = a.size();

            dim = ( b.size() < dim ) ? b.size() : dim;
            dim = ( c.size() < dim ) ? c.size() : dim;
            dim = ( d.size() < dim ) ? d.size() : dim;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;
            retVector<T> tmpve;
            retVector<T> tmpvf;
            retVector<T> tmpvg;
            retVector<T> tmpvh;

            return fourProductAssumeReal(res,a(a.ind(),tmpva)(0,1,dim-1,tmpve),b(b.ind(),tmpvb)(0,1,dim-1,tmpvf),c(c.ind(),tmpvc)(0,1,dim-1,tmpvg),d(d.ind(),tmpvd)(0,1,dim-1,tmpvh));
        }
    }

    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();
    int dsize = d.indsize();

    res = 0;

    if ( asize && bsize && csize && dsize )
    {
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);
	    delm = d.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                scaladd(res,a.direcref(apos),b.direcref(bpos),c.direcref(cpos),d.direcref(dpos));

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return res;
}

template <class T> 
double &mProductAssumeReal(double &res, const Vector<const SparseVector <T> *> &a)
{
    return mProductAssumeRealPrelude(res,a);
}

template <class T> 
double &mProductAssumeRealPrelude(double &res, const Vector<const SparseVector <T> *> &a)
{
    int m = a.size();

    res = 0;

    if ( m )
    {
        int i;
        int minsize = 0;

        Vector<int> asize(m);
        Vector<int> aelm(m);
        Vector<int> apos(m);

        for ( i = 0 ; i < m ; i++ )
        {
            asize("&",i) = (*(a(i))).indsize();
            aelm ("&",i) = 0;
            apos ("&",i) = 0;

            if ( !i || ( asize(i) < minsize ) )
            {
                minsize = asize(i);
            }
        }

        if ( minsize )
        {
            int done = 0;
            int samepos = 0;
            int maxelm = 0;

            double temp = 1;

            while ( !done )
            {
                samepos = 1;
                maxelm  = 0;

                for ( i = 0 ; i < m ; i++ )
                {
                    if ( apos(i) >= asize(i) )
                    {
                        done = 1;
                        break;
                    }

                    aelm("&",i) = (*(a(i))).ind(apos(i));

                    if ( !i || ( aelm(i) > maxelm ) )
                    {
                        maxelm = aelm(i);
                    }

                    if ( aelm(zeroint()) != aelm(i) )
                    {
                        samepos = 0;
                    }
                }

                if ( !done )
                {
                    if ( !samepos )
                    {
                        for ( i = 0 ; i < m ; i++ )
                        {
                            if ( aelm(i) < maxelm )
                            {
                                apos("&",i)++;
                            }
                        }
                    }

                    else
                    {
                        temp = 1;

                        for ( i = 0 ; i < m ; i++ )
                        {
                            scalmul(temp,(*(a(i))).direcref(apos(i)));
                            apos("&",i)++;
                        }

                        res += temp;
                    }
                }
            }
        }
    }

    return res;
}

template <class T>
T &oneProduct(T &result, const SparseVector<T> &a)
{
    return oneProductPrelude(result,a);
}

template <class T>
T &oneProductPrelude(T &result, const SparseVector<T> &a)
{
    int asize = a.indsize();

    setzero(result);

    if ( asize )
    {
	int apos = 0;

	while ( apos < asize )
	{
            result += a.direcref(apos);

            apos++;
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &oneProductScaled(T &result, const SparseVector<T> &a, const SparseVector<T> &scale)
{
    return oneProductScaledPrelude(result,a,scale);
}

template <class T>
T &oneProductScaledPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &scale)
{
    int asize = a.indsize();

    setzero(result);

    if ( asize )
    {
	int apos = 0;

	while ( apos < asize )
	{
            result += a.direcref(apos)/scale(a.ind(apos));

            apos++;
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &threeProduct(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c)
{
    return threeProductPrelude(result,a,b,c);
}

template <class T>
T &threeProductPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c)
{
    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();

    setzero(result);

    if ( asize && bsize && csize )
    {
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
	int aelm;
	int belm;
	int celm;

	while ( ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) )
	{
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += (a.direcref(apos)*b.direcref(bpos))*c.direcref(cpos);

		apos++;
		bpos++;
		cpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) )
	    {
		bpos++;
	    }

            else
	    {
		cpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &threeProductScaled(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &scale)
{
    return threeProductScaledPrelude(result,a,b,c,scale);
}

template <class T>
T &threeProductScaledPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &scale)
{
    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();

    setzero(result);

    if ( asize && bsize && csize )
    {
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
	int aelm;
	int belm;
	int celm;

	while ( ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) )
	{
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += ((a.direcref(apos)/scale(aelm))*(b.direcref(bpos)/scale(belm)))*(c.direcref(cpos)/scale(celm));

		apos++;
		bpos++;
		cpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) )
	    {
		bpos++;
	    }

            else
	    {
		cpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &fourProduct(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d)
{
    return fourProductPrelude(result,a,b,c,d);
}

template <class T> 
T &fourProductPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d)
{
    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();
    int dsize = d.indsize();

    setzero(result);

    if ( asize && bsize && csize && dsize )
    {
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);
	    delm = d.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += (a.direcref(apos)*b.direcref(bpos))*(c.direcref(cpos)*d.direcref(dpos));

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T> 
T &fourProductScaled(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d, const SparseVector<T> &scale)
{
    return fourProductScaledPrelude(result,a,b,c,d,scale);
}

template <class T> 
T &fourProductScaledPrelude(T &result, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d, const SparseVector<T> &scale)
{
    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();
    int dsize = d.indsize();

    setzero(result);

    if ( asize && bsize && csize && dsize )
    {
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);
	    delm = d.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += ((a.direcref(apos)/scale(aelm))*(b.direcref(bpos)/scale(belm)))*((c.direcref(cpos)/scale(celm))*(d.direcref(dpos)/scale(delm)));

		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &mProduct(T &result, const Vector<const SparseVector<T> *> &x)
{
    return mProductPrelude(result,x);
}

template <class T>
T &mProductPrelude(T &result, const Vector<const SparseVector<T> *> &x)
{
    setzero(result);

    if ( x.size() )
    {
	int i;
        SparseVector<T> temp(*(x(zeroint())));

	if ( x.size() > 1 )
	{
            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            temp *= *(x(1));

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    if ( i == x.size()-1 )
                    {
                        temp *= *(x(i));
                    }

                    else
                    {
                        temp *= ((*(x(i)))*(*(x(i+1))));
                    }
                }
	    }
	}

        result = sum(temp);
    }

    return postProInnerProd(result);
}

template <class T>
T &mProductScaled(T &result, const Vector<const SparseVector<T> *> &x, const SparseVector<T> &scale)
{
    return mProductScaledPrelude(result,x,scale);
}

template <class T>
T &mProductScaledPrelude(T &result, const Vector<const SparseVector<T> *> &x, const SparseVector<T> &scale)
{
    setzero(result);

    if ( x.size() )
    {
	int i;
        SparseVector<T> temp(*(x(zeroint())));
        temp /= scale;

	if ( x.size() > 1 )
	{
            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            temp *= *(x(1));
            temp /= scale;

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    if ( i == x.size()-1 )
                    {
                        temp *= *(x(i));
                        temp /= scale;
                    }

                    else
                    {
                        temp *= (((*(x(i)))/scale)*((*(x(i+1)))/scale));
                    }
                }
	    }
	}

        result = sum(temp);
    }

    return postProInnerProd(result);
}

template <class T> 
T &threeProductPow(T &res, const SparseVector<T> &a, int ia, const SparseVector<T> &b, int ib, const SparseVector<T> &c, int ic)
{
    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();

    int i;

    res = 0;

    if ( asize && bsize && csize )
    {
	int apos = 0;
	int bpos = 0;
	int cpos = 0;

	int aelm;
	int belm;
	int celm;

        T tmpa;
        T tmpb;
        T tmpc;

	while ( ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) )
	{
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) )
	    {
                tmpa = 1.0;
                tmpb = 1.0;
                tmpc = 1.0;

                if ( ia )
                {
                    for ( i = 0 ; i < ia ; i++ )
                    {
                        tmpa *= a.direcref(apos);
                    }
                }

                if ( ib )
                {
                    for ( i = 0 ; i < ib ; i++ )
                    {
                        tmpb *= b.direcref(bpos);
                    }
                }

                if ( ic )
                {
                    for ( i = 0 ; i < ic ; i++ )
                    {
                        tmpc *= c.direcref(cpos);
                    }
                }

                scaladd(res,tmpa,tmpb,tmpc);

		apos++;
		bpos++;
		cpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) )
	    {
		bpos++;
	    }

	    else
	    {
                cpos++;
	    }
	}
    }

    return res;
}

template <class T> 
T &fourProductPow(T &res, const SparseVector<T> &a, int ia, const SparseVector<T> &b, int ib, const SparseVector<T> &c, int ic, const SparseVector<T> &d, int id)
{
    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();
    int dsize = d.indsize();

    int i;

    res = 0;

    if ( asize && bsize && csize && dsize )
    {
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;

	int aelm;
	int belm;
	int celm;
        int delm;

        T tmpa;
        T tmpb;
        T tmpc;
        T tmpd;

	while ( ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);
	    delm = d.ind(dpos);

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                tmpa = 1.0;
                tmpb = 1.0;
                tmpc = 1.0;
                tmpd = 1.0;

                if ( ia )
                {
                    for ( i = 0 ; i < ia ; i++ )
                    {
                        tmpa *= a.direcref(apos);
                    }
                }

                if ( ib )
                {
                    for ( i = 0 ; i < ib ; i++ )
                    {
                        tmpb *= b.direcref(bpos);
                    }
                }

                if ( ic )
                {
                    for ( i = 0 ; i < ic ; i++ )
                    {
                        tmpc *= c.direcref(cpos);
                    }
                }

                if ( id )
                {
                    for ( i = 0 ; i < id ; i++ )
                    {
                        tmpd *= d.direcref(dpos);
                    }
                }

                scaladd(res,tmpa,tmpb,tmpc,tmpd);

		apos++;
		bpos++;
		cpos++;
		dpos++;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return res;
}

template <class T>
T &indexedoneProduct(T &result, const Vector<int> &n, const SparseVector<T> &a)
{
    int nsize = n.size();
    int asize = a.indsize();

    setzero(result);

    if ( nsize && asize )
    {
        int npos = 0;
	int apos = 0;

        int nelm;
	int aelm;

	while ( ( npos < nsize ) && ( apos < asize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos);

	    if ( ( aelm == nelm ) )
	    {
                result += a.direcref(apos);

                npos++;
		apos++;
	    }

            else if ( ( nelm <= aelm ) )
	    {
		npos++;
	    }

            else
	    {
		apos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedoneProductScaled(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &scale)
{
    int nsize = n.size();
    int asize = a.indsize();

    setzero(result);

    if ( nsize && asize )
    {
        int npos = 0;
	int apos = 0;

        int nelm;
	int aelm;

	while ( ( npos < nsize ) && ( apos < asize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos);

	    if ( ( aelm == nelm ) )
	    {
                result += a.direcref(apos)/scale(aelm);

                npos++;
		apos++;
	    }

            else if ( ( nelm <= aelm ) )
	    {
		npos++;
	    }

            else
	    {
		apos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedthreeProduct(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c)
{
    int nsize = n.size();
    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();

    setzero(result);

    if ( nsize && asize && bsize && csize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;

        int nelm;
	int aelm;
	int belm;
	int celm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += (a.direcref(apos)*b.direcref(bpos))*c.direcref(cpos);

                npos++;
		apos++;
		bpos++;
		cpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) )
	    {
		bpos++;
	    }

            else
	    {
		cpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedthreeProductScaled(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &scale)
{
    int nsize = n.size();
    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();

    setzero(result);

    if ( nsize && asize && bsize && csize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;

        int nelm;
	int aelm;
	int belm;
	int celm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += ((a.direcref(apos)/scale(aelm))*(b.direcref(bpos)/scale(belm)))*(c.direcref(cpos)/scale(celm));

                npos++;
		apos++;
		bpos++;
		cpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) )
	    {
		bpos++;
	    }

            else
	    {
		cpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedfourProduct(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d)
{
    int nsize = n.size();
    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();
    int dsize = d.indsize();

    setzero(result);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
	int dpos = 0;

        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);
	    delm = d.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += (a.direcref(apos)*b.direcref(bpos))*(c.direcref(cpos)*d.direcref(dpos));

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedfourProductScaled(T &result, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d, const SparseVector<T> &scale)
{
    int nsize = n.size();
    int asize = a.indsize();
    int bsize = b.indsize();
    int csize = c.indsize();
    int dsize = d.indsize();

    setzero(result);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
	int dpos = 0;

        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos);
	    belm = b.ind(bpos);
	    celm = c.ind(cpos);
	    delm = d.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += ((a.direcref(apos)/scale(aelm))*(b.direcref(bpos)/scale(belm)))*((c.direcref(cpos)/scale(celm))*(d.direcref(dpos)/scale(celm)));

                npos++;
		apos++;
		bpos++;
		cpos++;
                dpos++;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		npos++;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		apos++;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		bpos++;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		cpos++;
	    }

	    else
	    {
                dpos++;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedmProduct(T &result, const Vector<int> &n, const Vector<const SparseVector <T> *> &x)
{
    setzero(result);

    if ( x.size() )
    {
	int i;
        SparseVector<T> a(*(x(zeroint())));

	int nsize = n.size();
	int asize = a.indsize();

        int npos = 0;
	int apos = 0;

        int nelm;
	int aelm;

	while ( ( npos < nsize ) && ( apos < asize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos);

	    if ( aelm == nelm )
	    {
                npos++;
		apos++;
	    }

	    else if ( nelm < aelm )
	    {
		npos++;
	    }

            else
	    {
                a.zero(a.ind(apos));
		apos++;
	    }
	}

	while ( apos < asize )
	{
	    a.zero(a.ind(apos));
	    apos++;
	}

	if ( x.size() > 1 )
	{
            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            a *= *(x(1));

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    if ( i == x.size()-1 )
                    {
                        a *= *(x(i));
                    }

                    else
                    {
                        a *= ((*(x(i)))*(*(x(i+1))));
                    }
                }
	    }
	}

        result = sum(a);
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedmProductScaled(T &result, const Vector<int> &n, const Vector<const SparseVector <T> *> &x, const SparseVector<T> &scale)
{
    setzero(result);

    if ( x.size() )
    {
	int i;
        SparseVector<T> a(*(x(zeroint())));
        a /= scale;

	int nsize = n.size();
	int asize = a.indsize();

        int npos = 0;
	int apos = 0;

        int nelm;
	int aelm;

	while ( ( npos < nsize ) && ( apos < asize ) )
	{
            nelm = n(npos);
            aelm = a.ind(apos);

	    if ( aelm == nelm )
	    {
                npos++;
		apos++;
	    }

	    else if ( nelm < aelm )
	    {
		npos++;
	    }

            else
	    {
                a.zero(a.ind(apos));
		apos++;
	    }
	}

	while ( apos < asize )
	{
	    a.zero(a.ind(apos));
	    apos++;
	}

	if ( x.size() > 1 )
	{
            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            a *= *(x(1));
            a /= scale;

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    if ( i == x.size()-1 )
                    {
                        a *= *(x(i));
                        a /= scale;
                    }

                    else
                    {
                        a *= (((*(x(i)))/scale)*((*(x(i+1)))/scale));
                    }
                }
	    }
	}

        result = sum(a);
    }

    return postProInnerProd(result);
}

// ------------------------------------
//phantomxyz


























// Basic operator overloading

template <class T> SparseVector<T>  operator+ (const SparseVector<T> &left_op)
{
    SparseVector<T> res(left_op);

    if ( left_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            setposate(res.direref(i));
	}
    }

    return res;
}

template <class T> SparseVector<T>  operator- (const SparseVector<T> &left_op)
{
    SparseVector<T> res(left_op);

    if ( left_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            setnegate(res.direref(i));
	}
    }

    return res;
}

template <class T> SparseVector<T>  operator+ (const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    SparseVector<T> res(left_op);

    return ( res += right_op );
}

template <class T> SparseVector<T>  operator- (const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    SparseVector<T> res(left_op);

    return ( res -= right_op );
}

template <class T> SparseVector<T>  operator* (const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    SparseVector<T> res(left_op);

    return ( res *= right_op );
}

template <class T> SparseVector<T>  operator/ (const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    SparseVector<T> res(left_op);

    return ( res /= right_op );
}

template <class T> SparseVector<T>  operator% (const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    SparseVector<T> res(left_op);

    return ( res %= right_op );
}

template <class T> SparseVector<T>  operator+ (const SparseVector<T> &left_op, const T               &right_op)
{
    SparseVector<T> res(left_op);

    return ( res += right_op );
}

template <class T> SparseVector<T>  operator* (const SparseVector<T> &left_op, const T               &right_op)
{
    SparseVector<T> res(left_op);

    return ( res *= right_op );
}

template <class T> SparseVector<T>  operator/ (const SparseVector<T> &left_op, const T               &right_op)
{
    SparseVector<T> res(left_op);

    return ( res /= right_op );
}

template <class T> SparseVector<T>  operator% (const SparseVector<T> &left_op, const T               &right_op)
{
    SparseVector<T> res(left_op);

    return ( res %= right_op );
}

template <class T> SparseVector<T>  operator+ (const T               &left_op, const SparseVector<T> &right_op)
{
    SparseVector<T> res(right_op);

    return ( left_op += res );
}

template <class T> SparseVector<T>  operator- (const T               &left_op, const SparseVector<T> &right_op)
{
    SparseVector<T> res(right_op);

    return ( left_op -= res );
}

template <class T> SparseVector<T>  operator* (const T               &left_op, const SparseVector<T> &right_op)
{
    SparseVector<T> res(right_op);

    return ( left_op *= res );
}

template <class T> SparseVector<T>  operator/ (const T               &left_op, const SparseVector<T> &right_op)
{
    SparseVector<T> res(right_op);

    return ( left_op /= res );
}

template <class T> SparseVector<T>  operator% (const T               &left_op, const SparseVector<T> &right_op)
{
    SparseVector<T> res(right_op);

    return ( left_op %= res );
}







template <class T> SparseVector<T> &operator+=(SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.nearnonsparse() && right_op.nearnonsparse() && ( left_op.size() == right_op.size() ) )
    {
        retVector<T> tmpva;
        retVector<T> tmpvb;

        left_op("&",left_op.ind(),tmpva) += right_op(right_op.ind(),tmpvb);

        return left_op;
    }

    if ( right_op.indsize() == 0 )
    {
        return left_op;
    }

    int i = 0;
    int j = 0;

    for ( j = 0 ; j < right_op.indsize() ; j++ )
    {
        if ( i >= left_op.indsize() )
        {
            NiceAssert( i == left_op.indsize() );

            left_op.append(right_op.ind(j));
        }

        if ( left_op.ind(i) < right_op.ind(j) )
        {
            while ( left_op.ind(i) < right_op.ind(j) )
            {
                i++;

                if ( i >= left_op.indsize() )
                {
                    NiceAssert( i == left_op.indsize() );

                    left_op.append(right_op.ind(j));
                }
            }
        }

        if ( left_op.ind(i) > right_op.ind(j) )
        {
            left_op.add(right_op.ind(j),i);
        }

        NiceAssert( i < left_op.indsize() );
        NiceAssert( left_op.ind(i) == right_op.ind(j) );

        left_op.direref(i) += right_op.direcref(j);

        i++;
    }

    return left_op;
}


template <class T> SparseVector<T> &operator-=(      SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.nearnonsparse() && right_op.nearnonsparse() && ( left_op.size() == right_op.size() ) )
    {
        retVector<T> tmpva;
        retVector<T> tmpvb;

        left_op("&",left_op.ind(),tmpva) -= right_op(right_op.ind(),tmpvb);

        return left_op;
    }

    if ( right_op.indsize() == 0 )
    {
        return left_op;
    }

    int i = 0;
    int j = 0;

    for ( j = 0 ; j < right_op.indsize() ; j++ )
    {
        if ( i >= left_op.indsize() )
        {
            NiceAssert( i == left_op.indsize() );

            left_op.append(right_op.ind(j));
        }

        if ( left_op.ind(i) < right_op.ind(j) )
        {
            while ( left_op.ind(i) < right_op.ind(j) )
            {
                i++;

                if ( i >= left_op.indsize() )
                {
                    NiceAssert( i == left_op.indsize() );

                    left_op.append(right_op.ind(j));
                }
            }
        }

        if ( left_op.ind(i) > right_op.ind(j) )
        {
            left_op.add(right_op.ind(j),i);
        }

        NiceAssert( i < left_op.indsize() );
        NiceAssert( left_op.ind(i) == right_op.ind(j) );

        left_op.direref(i) -= right_op.direcref(j);

        i++;
    }

    return left_op;
}



template <class T> SparseVector<T> &multass(SparseVector<T> &left_op, const SparseVector<T> &right_op);

template <class T> SparseVector<T> &operator*=(      SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    // SPECIALISED IN GENTYPE

    return multass(left_op,right_op);
}

template <class T> SparseVector<T> &multass(SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.nearnonsparse() && right_op.nearnonsparse() && ( left_op.size() == right_op.size() ) )
    {
        retVector<T> tmpva;
        retVector<T> tmpvb;

        left_op("&",left_op.ind(),tmpva) *= right_op(right_op.ind(),tmpvb);

        return left_op;
    }

    int lsize = left_op.indsize();
    int rsize = right_op.indsize();

    if ( lsize && rsize )
    {
	int lpos = 0;
	int rpos = 0;
	int lelm;
        int relm;

	while ( ( lpos < lsize ) && ( rpos < rsize ) )
	{
            lelm = left_op.ind(lpos);
	    relm = right_op.ind(rpos);

	    if ( lelm == relm )
	    {
		left_op.direref(lpos) *= right_op.direcref(rpos);

		lpos++;
                rpos++;
	    }

	    else if ( lelm < relm )
	    {
                left_op.zero(lelm);
                lsize--;
	    }

	    else
	    {
                rpos++;
	    }
	}

	while ( lpos < lsize )
	{
	    left_op.zero(left_op.ind(lpos));
            lsize--;
	}
    }

    return left_op;
}

template <class T> SparseVector<T> &operator/=(      SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.nearnonsparse() && right_op.nearnonsparse() && ( left_op.size() == right_op.size() ) )
    {
        retVector<T> tmpva;
        retVector<T> tmpvb;

        left_op("&",left_op.ind(),tmpva) /= right_op(right_op.ind(),tmpvb);

        return left_op;
    }

    int lsize = left_op.indsize();
    int rsize = right_op.indsize();

    if ( lsize && rsize )
    {
	int lpos = 0;
	int rpos = 0;
	int lelm;
        int relm;

	while ( ( lpos < lsize ) && ( rpos < rsize ) )
	{
            lelm = left_op.ind(lpos);
	    relm = right_op.ind(rpos);

	    if ( lelm == relm )
	    {
                left_op.direref(lpos) /= right_op.direcref(rpos);

		lpos++;
                rpos++;
	    }

	    else if ( lelm < relm )
	    {
                left_op.zero(lelm);
                lsize--;
	    }

	    else
	    {
                rpos++;
	    }
	}

	while ( lpos < lsize )
	{
	    left_op.zero(left_op.ind(lpos));
            lsize--;
	}
    }

    return left_op;
}

template <class T> SparseVector<T> &operator%=(      SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.nearnonsparse() && right_op.nearnonsparse() && ( left_op.size() == right_op.size() ) )
    {
        left_op("&",left_op.ind()) %= right_op(right_op.ind());

        return left_op;
    }

    int lsize = left_op.indsize();
    int rsize = right_op.indsize();

    if ( lsize && rsize )
    {
	int lpos = 0;
	int rpos = 0;
	int lelm;
        int relm;

	while ( ( lpos < lsize ) && ( rpos < rsize ) )
	{
            lelm = left_op.ind(lpos);
	    relm = right_op.ind(rpos);

	    if ( lelm == relm )
	    {
                left_op.direref(lpos) %= right_op.direcref(rpos);

		lpos++;
                rpos++;
	    }

	    else if ( lelm < relm )
	    {
                left_op.zero(lelm);
                lsize--;
	    }

	    else
	    {
                rpos++;
	    }
	}

	while ( lpos < lsize )
	{
	    left_op.zero(left_op.ind(lpos));
            lsize--;
	}
    }

    return left_op;
}

template <class T> SparseVector<T> &operator+=(      SparseVector<T> &left_op, const T               &right_op)
{
    int i;

    if ( left_op.indsize() )
    {
	for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            left_op.direref(i) += right_op;
	}
    }

    return left_op;
}

template <class T> SparseVector<T> &operator-=(      SparseVector<T> &left_op, const T               &right_op)
{
    int i;

    if ( left_op.indsize() )
    {
	for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            left_op.direref(i) -= right_op;
	}
    }

    return left_op;
}

template <class T> SparseVector<T> &operator*=(      SparseVector<T> &left_op, const T               &right_op)
{
    int i;

    if ( left_op.indsize() )
    {
	for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            left_op.direref(i) *= right_op;
	}
    }

    return left_op;
}

template <class T> SparseVector<T> &operator/=(      SparseVector<T> &left_op, const T               &right_op)
{
    int i;

    if ( left_op.indsize() )
    {
	for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            left_op.direref(i) /= right_op;
	}
    }

    return left_op;
}

template <class T> SparseVector<T> &operator%=(      SparseVector<T> &left_op, const T               &right_op)
{
    int i;

    if ( left_op.indsize() )
    {
	for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            left_op.direref(i) %= right_op;
	}
    }

    return left_op;
}

template <class T> SparseVector<T> &operator+=(const T               &left_op,       SparseVector<T> &right_op)
{
    int i;

    if ( right_op.indsize() )
    {
        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            right_op.direref(i) += left_op;
	}
    }

    return right_op;
}

template <class T> SparseVector<T> &operator-=(const T               &left_op,       SparseVector<T> &right_op)
{
    int i;

    if ( right_op.indsize() )
    {
        right_op.negate();

        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            right_op.direref(i) += left_op;
	}
    }

    return right_op;
}

template <class T> SparseVector<T> &operator*=(const T               &left_op,       SparseVector<T> &right_op)
{
    int i;

    if ( right_op.indsize() )
    {
        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            rightmult(left_op,right_op.direref(i));
	}
    }

    return right_op;
}

template <class T> SparseVector<T> &operator/=(const T               &left_op,       SparseVector<T> &right_op)
{
    int i;

    if ( right_op.indsize() )
    {
        right_op.inv();

        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            rightmult(left_op,right_op.direref(i));
	}
    }

    return right_op;
}

template <class T> SparseVector<T> &operator%=(const T               &left_op,       SparseVector<T> &right_op)
{
    int i;

    if ( right_op.indsize() )
    {
        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            right_op.direref(i) = left_op%right_op.direcref(i);
	}
    }

    return right_op;
}

template <class T> SparseVector<T> &leftmult (      SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    return ( left_op *= right_op );
}

template <class T> SparseVector<T> &leftmult (      SparseVector<T> &left_op, const T               &right_op)
{
    return ( left_op *= right_op );
}

template <class T> SparseVector<T> &rightmult(const SparseVector<T> &left_op,       SparseVector<T> &right_op)
{
    if ( left_op.nearnonsparse() && right_op.nearnonsparse() && ( left_op.size() == right_op.size() ) )
    {
        rightmult(left_op(left_op.ind()),right_op("&",right_op.ind()));

        return right_op;
    }

    int lsize = left_op.indsize();
    int rsize = right_op.indsize();

    if ( lsize && rsize )
    {
	int lpos = 0;
	int rpos = 0;
	int lelm;
        int relm;

	while ( ( lpos < lsize ) && ( rpos < rsize ) )
	{
            lelm = left_op.ind(lpos);
	    relm = right_op.ind(rpos);

	    if ( lelm == relm )
	    {
                rightmult(left_op.direcref(lpos),right_op.direref(rpos));

		lpos++;
                rpos++;
	    }

            else if ( lelm < relm )
	    {
                lpos++;
	    }

	    else
	    {
                right_op.zero(lelm);
                //rpos++;
	    }
	}

        while ( rpos < rsize )
	{
            right_op.zero(right_op.ind(rpos));
	}
    }

    return right_op;
}

template <class T> SparseVector<T> &rightmult(const T               &left_op,       SparseVector<T> &right_op)
{
    int i;

    if ( right_op.indsize() )
    {
        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            rightmult(left_op,right_op.direref(i));
	}
    }

    return right_op;
}








template <class T>
SparseVector<T> &kronprod(SparseVector<T> &res, int &dimres, const SparseVector<T> &a, const SparseVector<T> &b, int dima, int dimb)
{
    NiceAssert( dima >= 0 );
    NiceAssert( dimb >= 0 );

    int ii,jj,i,j;

    dimres = dima*dimb;

    res.zero();

    for ( ii = 0 ; ii < a.indsize() ; ii++ )
    {
        for ( jj = 0 ; jj < b.indsize() ; jj++ )
        {
            i = a.ind(ii);
            j = b.ind(jj);

            res("&",(i*dimb)+j) = a.direcref(ii)*b.direcref(jj);
        }
    }

    return res;
}

template <class T>
Vector<T> &kronprod(Vector<T> &res, const SparseVector<T> &a, const SparseVector<T> &b, int dima, int dimb)
{
    NiceAssert( dima >= 0 );
    NiceAssert( dimb >= 0 );

    int ii,jj,i,j;

    res.resize(dima*dimb).zero();

    (void) dima;

    for ( ii = 0 ; ii < a.indsize() ; ii++ )
    {
        for ( jj = 0 ; jj < b.indsize() ; jj++ )
        {
            i = a.ind(ii);
            j = b.ind(jj);

            res("&",(i*dimb)+j) = a.direcref(ii)*b.direcref(jj);
        }
    }

    return res;
}

template <class T> 
SparseVector<T> &kronpow(SparseVector<T> &res, int &dimres, const SparseVector<T> &a, int dima, int n)
{
    NiceAssert( n >= 0 );

    if ( n == 0 )
    {
        dimres = 1;

        res.zero();

        setident(res("&",0));
    }

    else if ( n == 1 )
    {
        dimres = dima;

        res = a;
    }

    else
    {
        SparseVector<T> b;
        int dimb;

        kronpow(b,dimb,a,dima,n-1);
        kronprod(res,dimres,a,b,dima,dimb);
    }

    return res;
}

template <class T> 
Vector<T> &kronpow(Vector<T> &res, const SparseVector<T> &a, int dima, int n)
{
    NiceAssert( n >= 0 );

    if ( n == 0 )
    {
        setident((res.resize(1))("&",0));
    }

    else if ( n == 1 )
    {
        res.resize(dima).zero();

        int i,ii;

        for ( ii = 0 ; ii < a.indsize() ; ii++ )
        {
            i = a.ind(ii);

            res("&",i) = a.direcref(ii);
        }
    }

    else
    {
        SparseVector<T> b;
        int dimb;

        kronpow(b,dimb,a,dima,n-1);
        kronprod(res,a,b,dima,dimb);
    }

    return res;
}






// Conversion from strings

template <class T> SparseVector<T> &atoSparseVector(SparseVector<T> &dest, const std::string &src)
{
    std::istringstream srcmod(src);

    srcmod >> dest;

    return dest;
}







// Stream operators

template <class T>
std::ostream &operator<<(std::ostream &output, const SparseVector<T> &src)
{
    int size = src.indsize();
    int prevind = -1;
    int baseind = 0;

    output << "[ ";

    if ( size )
    {
	int i;

	for ( i = 0 ; i < size ; i++ )
	{
	    if ( i > 0 )
	    {
                output << "  ";
	    }

            if ( src.ind(i) == prevind+1 )
            {
                output << src.direcref(i) << "\t";
            }

            else if ( ( baseind < INDFAROFFSTART ) && ( src.ind(i) >= INDFAROFFSTART ) && ( src.ind(i) <= INDFAROFFEND ) )
            {
                baseind = INDFAROFFSTART;

                if ( src.ind(i) == baseind )
                {
                    output << ":\t;\n  " << src.direcref(i) << "\t";
                }

                else
                {
                    output << ":\t;\n  " << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else if ( ( baseind < INDFARFAROFFSTART ) && ( src.ind(i) >= INDFARFAROFFSTART ) && ( src.ind(i) <= INDFARFAROFFEND ) )
            {
                baseind = INDFARFAROFFSTART;

                if ( src.ind(i) == baseind )
                {
                    output << "::\t;\n  " << src.direcref(i) << "\t";
                }

                else
                {
                    output << "::\t;\n  " << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else if ( ( baseind < INDFARFARFAROFFSTART ) && ( src.ind(i) >= INDFARFARFAROFFSTART ) && ( src.ind(i) <= INDFARFARFAROFFEND ) )
            {
                baseind = INDFARFARFAROFFSTART;

                if ( src.ind(i) == baseind )
                {
                    output << ":::\t;\n  " << src.direcref(i) << "\t";
                }

                else
                {
                    output << ":::\t;\n  " << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else if ( (src.ind(i)-baseind) >= DEFAULT_TUPLE_INDEX_STEP )
            {
                goover:

                baseind += DEFAULT_TUPLE_INDEX_STEP;

                output << "~\t;\n";

                if ( (src.ind(i)-baseind) >= DEFAULT_TUPLE_INDEX_STEP )
                {
                    goto goover;
                }

                if ( src.ind(i) == baseind )
                {
                    output << "  " << src.direcref(i) << "\t";
                }

                else
                {
                    output << "  " << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else
            {
                output << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
            }

            if ( i < size-1 )
            {
                output << ";\n ";
            }

            prevind = src.ind(i);
	}
    }

    output << "  ]";

    return output;
}

template <class T>
std::ostream &printoneline(std::ostream &output, const SparseVector<T> &src)
{
    int size = src.indsize();
    int prevind = -1;
    int baseind = 0;

    output << "[ ";

    if ( size )
    {
	int i;

	for ( i = 0 ; i < size ; i++ )
	{
	    if ( i > 0 )
	    {
                output << "  ";
	    }

            if ( src.ind(i) == prevind+1 )
            {
                output << src.direcref(i) << "\t";
            }

            else if ( ( baseind < INDFAROFFSTART ) && ( src.ind(i) >= INDFAROFFSTART ) && ( src.ind(i) <= INDFAROFFEND ) )
            {
                baseind = INDFAROFFSTART;

                if ( src.ind(i) == baseind )
                {
                    output << ":\t;  " << src.direcref(i) << "\t";
                }

                else
                {
                    output << ":\t;  " << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else if ( ( baseind < INDFARFAROFFSTART ) && ( src.ind(i) >= INDFARFAROFFSTART ) && ( src.ind(i) <= INDFARFAROFFEND ) )
            {
                baseind = INDFARFAROFFSTART;

                if ( src.ind(i) == baseind )
                {
                    output << "::\t;  " << src.direcref(i) << "\t";
                }

                else
                {
                    output << "::\t;  " << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else if ( ( baseind < INDFARFARFAROFFSTART ) && ( src.ind(i) >= INDFARFARFAROFFSTART ) && ( src.ind(i) <= INDFARFARFAROFFEND ) )
            {
                baseind = INDFARFARFAROFFSTART;

                if ( src.ind(i) == baseind )
                {
                    output << ":::\t;  " << src.direcref(i) << "\t";
                }

                else
                {
                    output << ":::\t;  " << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else if ( (src.ind(i)-baseind) >= DEFAULT_TUPLE_INDEX_STEP )
            {
                goover:

                baseind += DEFAULT_TUPLE_INDEX_STEP;

                output << "~\t;";

                if ( (src.ind(i)-baseind) >= DEFAULT_TUPLE_INDEX_STEP )
                {
                    goto goover;
                }

                if ( src.ind(i) == baseind )
                {
                    output << "  " << src.direcref(i) << "\t";
                }

                else
                {
                    output << "  " << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else
            {
                output << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
            }

            if ( i < size-1 )
            {
                output << "; ";
            }

            prevind = src.ind(i);
	}
    }

    output << "  ]";

    return output;
}

template <class T>
std::ostream &printnoparen(std::ostream &output, const SparseVector<T> &src)
{
    int size = src.indsize();
    int prevind = -1;
    int baseind = 0;

    if ( size )
    {
	int i;

	for ( i = 0 ; i < size ; i++ )
	{
	    if ( i > 0 )
	    {
                output << "  ";
	    }

            if ( src.ind(i) == prevind+1 )
            {
                output << src.direcref(i) << "\t";
            }

            else if ( ( baseind < INDFAROFFSTART ) && ( src.ind(i) >= INDFAROFFSTART ) && ( src.ind(i) <= INDFAROFFEND ) )
            {
                baseind = INDFAROFFSTART;

                if ( src.ind(i) == baseind )
                {
                    output << ":\t" << src.direcref(i) << "\t";
                }

                else
                {
                    output << ":\t" << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else if ( ( baseind < INDFARFAROFFSTART ) && ( src.ind(i) >= INDFARFAROFFSTART ) && ( src.ind(i) <= INDFARFAROFFEND ) )
            {
                baseind = INDFARFAROFFSTART;

                if ( src.ind(i) == baseind )
                {
                    output << "::\t" << src.direcref(i) << "\t";
                }

                else
                {
                    output << "::\t" << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else if ( ( baseind < INDFARFARFAROFFSTART ) && ( src.ind(i) >= INDFARFARFAROFFSTART ) && ( src.ind(i) <= INDFARFARFAROFFEND ) )
            {
                baseind = INDFARFARFAROFFSTART;

                if ( src.ind(i) == baseind )
                {
                    output << ":::\t" << src.direcref(i) << "\t";
                }

                else
                {
                    output << ":::\t" << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else if ( (src.ind(i)-baseind) >= DEFAULT_TUPLE_INDEX_STEP )
            {
                goover:

                baseind += DEFAULT_TUPLE_INDEX_STEP;

                output << "~\t";

                if ( (src.ind(i)-baseind) >= DEFAULT_TUPLE_INDEX_STEP )
                {
                    goto goover;
                }

                if ( src.ind(i) == baseind )
                {
                    output << "  " << src.direcref(i) << "\t";
                }

                else
                {
                    output << "  " << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
                }
            }

            else
            {
                output << src.ind(i)-baseind << ":" << src.direcref(i) << "\t";
            }

            if ( i < size-1 )
            {
                output << "";
            }

            prevind = src.ind(i);
	}
    }

    return output;
}

template <class T>
std::istream &operator>>(std::istream &input, SparseVector<T> &dest)
{
    dest.killaltcontent();
    dest.killnearfar();
    dest.zero();

    int errcode;
    int j;
    int pos;
    int lastpos = -1;
    int maxlen;
    char xxx;
    std::string newstuff;
    int baseind = 0;

    while ( isspace(input.peek()) )
    {
        input.get(xxx);
    }

    NiceAssert( input.peek() == '[' );

    input >> newstuff;

    while ( isspace(input.peek()) )
    {
        input.get(xxx);
    }

    while ( 1 )
    {
        while ( ( isspace(input.peek()) ) || ( input.peek() == ';' ) )
        {
            input.get(xxx);
        }

        if ( input.peek() == ']' )
        {
            input >> newstuff;

            break;
        }

        //input >> newstuff;
        errcode = readParenString(input,newstuff);

        maxlen = (int) newstuff.length();

        NiceAssert( !errcode );
        NiceAssert( maxlen );

        (void) errcode;

        pos = lastpos+1;

        for ( j = 0 ; j < maxlen ; j++ )
        {
            if ( newstuff[j] == ':' )
            {
                pos = -1;
            }

            else if ( newstuff[j] == '~' )
            {
                pos = -1;
            }

            else if ( !( ( newstuff[j] == '1' ) || ( newstuff[j] == '2' ) || ( newstuff[j] == '3' ) ||
                         ( newstuff[j] == '4' ) || ( newstuff[j] == '5' ) || ( newstuff[j] == '6' ) ||
                         ( newstuff[j] == '7' ) || ( newstuff[j] == '8' ) || ( newstuff[j] == '9' ) ||
                         ( newstuff[j] == '0' ) ) )
            {
                break;
            }
        }

        j = 0;

        if ( pos == -1 )
        {
            //NiceAssert( maxlen >= 3 );
            // ...: is allowed - it just sets the position

            pos = 0;
            xxx = newstuff[j];

            while ( ( xxx != ':' ) && ( xxx != '~' ) )
            {
                pos *= 10;
                pos += ( xxx-'0' );

                j++;
                xxx = newstuff[j];

                NiceAssert( j < maxlen );
            }

            j++;

            if ( j == 1 )
            {
                // Number has form :stuff (ie no index), so use pos 2^30

                if ( ( j == maxlen ) && ( newstuff[0] == '~' ) )
                {
                    // newstuff[0]  = '~'

                    pos = 0;
                    baseind += DEFAULT_TUPLE_INDEX_STEP;
                }

                else if ( ( j == maxlen ) || ( newstuff[1] != ':' ) )
                {
                    // newstuff[0]  = ':'
                    // newstuff[1] != ':'

                    pos = 0;
                    baseind = INDFAROFFSTART;
                }

                else if ( ( j == maxlen-1 ) || ( newstuff[j+1] != ':' ) )
                {
                    // newstuff[0]  = ':'
                    // newstuff[1]  = ':'
                    // newstuff[2] != ':'

                    j++;
                    pos = 0;
                    baseind = INDFARFAROFFSTART;
                }

                else
                { 
                    // newstuff[0] = ':'
                    // newstuff[1] = ':'
                    // newstuff[2] = ':'

                    j++;
                    j++;
                    pos = 0;
                    baseind = INDFARFARFAROFFSTART;
                }
            }

            if ( maxlen == j )
            {
                // ind: on its own applies to following argument

                pos--;
            }
        }

        if ( j < maxlen )
        {
            newstuff = newstuff.substr(j,maxlen);

            {
                std::istringstream newbuffer;
                newbuffer.str(newstuff);
                newbuffer >> dest("&",baseind+pos);
            }
        }

        lastpos = pos;
    }

    dest.makealtcontent();

    return input;
}

template <class T>
std::istream &streamItIn(std::istream &input, SparseVector<T> &dest, int processxyzvw)
{
    return streamItInAlt(input,dest,processxyzvw);
}

template <class T> 
std::istream &streamItInAlt(std::istream &input, SparseVector<T> &dest, int processxyzvw, int removeZeros)
{
    dest.killaltcontent();
    dest.killnearfar();
    dest.zero();

    T zerovalis;
    T tempvar;
    setzero(zerovalis);
    setzero(tempvar);

    int errcode;
    int j;
    int pos;
    int lastpos = -1;
    int maxlen;
    int finalzero = 0;
    int isfirstzero = 1;
    char xxx;
    std::string newstuff;
    int baseind = 0;

    while ( isspace(input.peek()) )
    {
        input.get(xxx);
    }

    NiceAssert( input.peek() == '[' );

    input >> newstuff;

    while ( isspace(input.peek()) )
    {
        input.get(xxx);
    }

    while ( 1 )
    {
        while ( ( isspace(input.peek()) ) || ( input.peek() == ';' ) )
        {
            input.get(xxx);
        }

        if ( input.peek() == ']' )
        {
            input >> newstuff;

            break;
        }

        //input >> newstuff;
        errcode = readParenString(input,newstuff);
        (void) errcode;

        maxlen = (int) newstuff.length();

        NiceAssert( !errcode );
        NiceAssert( maxlen );

        pos = lastpos+1;

        for ( j = 0 ; j < maxlen ; j++ )
        {
            if ( newstuff[j] == ':' )
            {
                pos = -1;
            }

            else if ( newstuff[j] == '~' )
            {
                pos = -1;
            }

            else if ( !( ( newstuff[j] == '1' ) || ( newstuff[j] == '2' ) || ( newstuff[j] == '3' ) ||
                         ( newstuff[j] == '4' ) || ( newstuff[j] == '5' ) || ( newstuff[j] == '6' ) ||
                         ( newstuff[j] == '7' ) || ( newstuff[j] == '8' ) || ( newstuff[j] == '9' ) ||
                         ( newstuff[j] == '0' ) ) )
            {
                break;
            }
        }

        j = 0;

        if ( pos == -1 )
        {
            pos = 0;
            xxx = newstuff[j];

            while ( ( xxx != ':' ) && ( xxx != '~' ) )
            {
                pos *= 10;
                pos += ( xxx-'0' );

                j++;
                xxx = newstuff[j];

                NiceAssert( j < maxlen );
            }

            j++;

            if ( j == 1 )
            {
                // Number has form :stuff (ie no index), so use pos 2^30

                if ( ( j == maxlen ) && ( newstuff[0] == '~' ) )
                {
                    // newstuff[0]  = '~'

                    pos = 0;
                    baseind += DEFAULT_TUPLE_INDEX_STEP;
                }

                else if ( ( j == maxlen ) || ( newstuff[1] != ':' ) )
                {
                    // newstuff[0]  = ':'
                    // newstuff[1] != ':'

                    pos = 0;
                    baseind = INDFAROFFSTART;
                }

                else if ( ( j == maxlen-1 ) || ( newstuff[j+1] != ':' ) )
                { 
                    // newstuff[0]  = ':'
                    // newstuff[1]  = ':'
                    // newstuff[2] != ':'

                    j++;
                    pos = 0;
                    baseind = INDFARFAROFFSTART;
                }

                else
                { 
                    // newstuff[0] = ':'
                    // newstuff[1] = ':'
                    // newstuff[2] = ':'

                    j++;
                    j++;
                    pos = 0;
                    baseind = INDFARFARFAROFFSTART;
                }
            }

            if ( maxlen == j )
            {
                // : or ind: on its own applies to following argument

                pos--;
            }
        }

        if ( j < maxlen )
        {
            newstuff = newstuff.substr(j,maxlen);

            if ( !removeZeros )
            {
                // In this case everything is being kept, so just stream it in

                std::istringstream newbuffer;
                newbuffer.str(newstuff);
                streamItIn(newbuffer,dest("&",baseind+pos),processxyzvw);
            }

            else
            {
                // otherwise need to check if zero, only store if nonzero

                std::istringstream newbuffer;
                newbuffer.str(newstuff);
                streamItIn(newbuffer,tempvar,processxyzvw);

                finalzero = 0;

                if ( !( tempvar == zerovalis ) )
                {
                    dest("&",baseind+pos) = tempvar;
                }

                else if ( isfirstzero )
                {
                    // Keep at least one zero representative so max/min work

                    dest("&",baseind+pos) = tempvar;
                    isfirstzero = 0;
                }

                else
                {
                    finalzero = 1;
                }
            }
        }

        lastpos = pos;
    }

    if ( finalzero )
    {
        // This is important for per-vector normalsiation
        // in mercer kernels, where the cim
        // of the vector must be representative, so
        // for example mean(( 0 0 1 0 )) = 0.25, rather
        // then becoming mean((2:1)) = 1, which would
        // cause problems.

        dest("&",lastpos) = tempvar;
    }

    dest.makealtcontent();

    return input;
}

template <class T> int operator==(const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.vecID() == right_op.vecID() )
    {
        return 1;
    }

    if ( left_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            if ( !( left_op.direcref(i) == right_op(left_op.ind(i)) ) )
	    {
                return 0;
	    }
	}
    }

    if ( right_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            if ( !( left_op(right_op.ind(i)) == right_op.direcref(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator==(const SparseVector<T> &left_op, const T               &right_op)
{
    T temp;

    setzero(temp);

    if ( temp != right_op )
    {
        return 0;
    }

    else
    {
        if ( left_op.indsize() )
        {
            int i;

            for ( i = 0 ; i < left_op.indsize() ; i++ )
            {
                if ( !( left_op.direcref(i) == right_op ) )
                {
                    return 0;
                }
            }
        }
    }

    return 0;
}

template <class T> int operator==(const T               &left_op, const SparseVector<T> &right_op)
{
    T temp;

    setzero(temp);

    if ( left_op != temp )
    {
        return 0;
    }

    else
    {
        if ( right_op.indsize() )
        {
            int i;

            for ( i = 0 ; i < right_op.indsize() ; i++ )
            {
                if ( !( left_op == right_op.direcref(i) ) )
                {
                    return 0;
                }
            }
        }
    }

    return 0;
}

template <class T> int operator!=(const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.vecID() == right_op.vecID() )
    {
        return 0;
    }

    return !(left_op == right_op);
}

template <class T> int operator!=(const SparseVector<T> &left_op, const T               &right_op)
{
    return !(left_op == right_op);
}

template <class T> int operator!=(const T               &left_op, const SparseVector<T> &right_op)
{
    return !(left_op == right_op);
}

template <class T> int operator> (const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            if ( !( left_op.direcref(i) > right_op(left_op.ind(i)) ) )
	    {
                return 0;
	    }
	}
    }

    if ( right_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            if ( !( left_op(right_op.ind(i)) > right_op.direcref(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator> (const SparseVector<T> &left_op, const T               &right_op)
{
    T temp;

    setzero(temp);

    if ( temp > right_op )
    {
        return 0;
    }

    else
    {
        if ( left_op.indsize() )
        {
            int i;

            for ( i = 0 ; i < left_op.indsize() ; i++ )
            {
                if ( !( left_op.direcref(i) > right_op ) )
                {
                    return 0;
                }
            }
        }
    }

    return 0;
}

template <class T> int operator> (const T               &left_op, const SparseVector<T> &right_op)
{
    T temp;

    setzero(temp);

    if ( left_op > temp )
    {
        return 0;
    }

    else
    {
        if ( right_op.indsize() )
        {
            int i;

            for ( i = 0 ; i < right_op.indsize() ; i++ )
            {
                if ( !( left_op > right_op.direcref(i) ) )
                {
                    return 0;
                }
            }
        }
    }

    return 0;
}

template <class T> int operator>=(const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.vecID() == right_op.vecID() )
    {
        return 1;
    }

    if ( left_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            if ( !( left_op.direcref(i) >= right_op(left_op.ind(i)) ) )
	    {
                return 0;
	    }
	}
    }

    if ( right_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            if ( !( left_op(right_op.ind(i)) >= right_op.direcref(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator>=(const SparseVector<T> &left_op, const T               &right_op)
{
    T temp;

    setzero(temp);

    if ( temp >= right_op )
    {
        return 0;
    }

    else
    {
        if ( left_op.indsize() )
        {
            int i;

            for ( i = 0 ; i < left_op.indsize() ; i++ )
            {
                if ( !( left_op.direcref(i) >= right_op ) )
                {
                    return 0;
                }
            }
        }
    }

    return 0;
}

template <class T> int operator>=(const T               &left_op, const SparseVector<T> &right_op)
{
    T temp;

    setzero(temp);

    if ( left_op >= temp )
    {
        return 0;
    }

    else
    {
        if ( right_op.indsize() )
        {
            int i;

            for ( i = 0 ; i < right_op.indsize() ; i++ )
            {
                if ( !( left_op >= right_op.direcref(i) ) )
                {
                    return 0;
                }
            }
        }
    }

    return 0;
}

template <class T> int operator< (const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            if ( !( left_op.direcref(i) < right_op(left_op.ind(i)) ) )
	    {
                return 0;
	    }
	}
    }

    if ( right_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            if ( !( left_op(right_op.ind(i)) < right_op.direcref(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator< (const SparseVector<T> &left_op, const T               &right_op)
{
    T temp;

    setzero(temp);

    if ( temp < right_op )
    {
        return 0;
    }

    else
    {
        if ( left_op.indsize() )
        {
            int i;

            for ( i = 0 ; i < left_op.indsize() ; i++ )
            {
                if ( !( left_op.direcref(i) < right_op ) )
                {
                    return 0;
                }
            }
        }
    }

    return 0;
}

template <class T> int operator< (const T               &left_op, const SparseVector<T> &right_op)
{
    T temp;

    setzero(temp);

    if ( left_op < temp )
    {
        return 0;
    }

    else
    {
        if ( right_op.indsize() )
        {
            int i;

            for ( i = 0 ; i < right_op.indsize() ; i++ )
            {
                if ( !( left_op < right_op.direcref(i) ) )
                {
                    return 0;
                }
            }
        }
    }

    return 0;
}

template <class T> int operator<=(const SparseVector<T> &left_op, const SparseVector<T> &right_op)
{
    if ( left_op.vecID() == right_op.vecID() )
    {
        return 1;
    }

    if ( left_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < left_op.indsize() ; i++ )
	{
            if ( !( left_op.direcref(i) <= right_op(left_op.ind(i)) ) )
	    {
                return 0;
	    }
	}
    }

    if ( right_op.indsize() )
    {
	int i;

        for ( i = 0 ; i < right_op.indsize() ; i++ )
	{
            if ( !( left_op(right_op.ind(i)) <= right_op.direcref(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator<=(const SparseVector<T> &left_op, const T               &right_op)
{
    T temp;

    setzero(temp);

    if ( temp <= right_op )
    {
        return 0;
    }

    else
    {
        if ( left_op.indsize() )
        {
            int i;

            for ( i = 0 ; i < left_op.indsize() ; i++ )
            {
                if ( !( left_op.direcref(i) <= right_op ) )
                {
                    return 0;
                }
            }
        }
    }

    return 0;
}

template <class T> int operator<=(const T               &left_op, const SparseVector<T> &right_op)
{
    T temp;

    setzero(temp);

    if ( left_op <= temp )
    {
        return 0;
    }

    else
    {
        if ( right_op.indsize() )
        {
            int i;

            for ( i = 0 ; i < right_op.indsize() ; i++ )
            {
                if ( !( left_op <= right_op.direcref(i) ) )
                {
                    return 0;
                }
            }
        }
    }

    return 0;
}


template <class T> SparseVector<T> &randfill (SparseVector<T> &res)
{
    T &(*disamapply)(T &) = randfill;

    return res.applyon(disamapply);
}

template <class T> SparseVector<T> &randnfill(SparseVector<T> &res)
{
    T &(*disamapply)(T &) = randnfill;

    return res.applyon(disamapply);
}











inline unsigned int getnewID(void)
{
    int res = 0;

    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    static int currid = 0;

    res = currid++;

    svm_mutex_unlock(eyelock);

    return res;
}






#endif
