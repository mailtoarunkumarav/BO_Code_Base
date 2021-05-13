
//
// Set class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _set_h
#define _set_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <sstream>
#include "vector.h"

template <class T> class Set;

// Stream operators

template <class T> std::ostream &operator<<(std::ostream &output, const Set<T> &src );
template <class T> std::istream &operator>>(std::istream &input,        Set<T> &dest);
template <class T> std::istream &streamItIn(std::istream &input,        Set<T> &dest, int processxyzvw = 1);

// Swap function

template <class T> void qswap(const Set<T> *a, const Set<T> *b);
template <class T> void qswap(Set<T> &a, Set<T> &b);

// The class itself

template <class T>
class Set
{
    template <class S> friend std::istream &operator>>(std::istream &input, Set<S> &dest);
    template <class S> friend void qswap(Set<S> &a, Set<S> &b);

public:

    // Constructor:

    svm_explicit Set();
                 Set(const Set<T> &src);
    svm_explicit Set(const Vector<T> &src);

    // Assignment:

    Set<T> &operator=(const Set<T> &src);

    // simple set manipulations
    //
    // ident: set size of set to zero
    // zero: set size of set to zero
    // posate: apply setposate() to all elements of the set
    // negate: apply setnegate() to all elements of the set
    // conj: apply setconj() to all elements of the set
    //
    // each returns a reference to *this

    Set<T> &ident(void);
    Set<T> &zero(void);
    Set<T> &posate(void);
    Set<T> &negate(void);
    Set<T> &conj(void);
    Set<T> &rand(void);

    // Access:
    //
    // - all: returns a vector containing all elements in set
    // - contains(x): returns 1 if x is in set, 0 otherwise

    const Vector<T> &all(void) const;
    int contains(const T &x) const;

    // Add (if not present) and remove (if present) elements:

    Set<T> &add(const T &x);
    Set<T> &remove(const T &x);

    // Information:

    int size(void) const { return contents.size(); }

    // Function application - apply function fn to each element of set.

    Set<T> &applyon(T (*fn)(T));
    Set<T> &applyon(T (*fn)(const T &));
    Set<T> &applyon(T (*fn)(T, const void *), const void *a);
    Set<T> &applyon(T (*fn)(const T &, const void *), const void *a);
    Set<T> &applyon(T &(*fn)(T &));
    Set<T> &applyon(T &(*fn)(T &, const void *), const void *a);

    // Don't use this

    Vector<T> &ncall(void) { return contents; }

private:

    Vector<T> contents;

    void removeDuplicates(void);
};

template <class T> void qswap(Set<T> &a, Set<T> &b)
{
    qswap(a.contents,b.contents);

    return;
}

template <class T> void qswap(const Set<T> *a, const Set<T> *b)
{
    const Set<T> *c;

    c = a;
    a = b;
    b = c;

    return;
}

// Various functions
//
// max: find max element, put index in i.  If two sets are given then finds max element difference
// min: find min element, put index in i.  If two sets are given then finds min element difference
// maxabs: find the |max| element.
// minabs: find the |min| element.
// sqabsmax: find the |max|*|max| element, put index in i.
// sqabsmin: find the |min|*|min| element, put index in i.
// sum: find the sum of elements in a set
// prod: find the product of elements in a set (arbitrary order, not good for non-commutative elements)
// Prod: find the product of elements in a set (arbitrary order, not good for non-commutative elements)
// mean: mean of elements
// median: median of elements
//
// setident: call a.ident()
// setzero: call a.zero()
// setposate: call a.posate()
// setnegate: call a.negate()
// setconj: call a.conj()
//
// norms, inner products, and metrics
// ==================================
//
// These are constructed by identifying the sets with sparse vectors.  To
// be precise, suppose every possible element of any finite set is represented
// by a single non-negative integer.  Then any given finite set can be
// identified with a unique binary-valued sparse vector:
//
// { a,b,c } <-> [ i_a:1 i_b:1 i_c:1 ]
//
// where i_a = the unique integer label associated with set element a.
//
// Let the inner product of two sets equal the inner product of the sparse
// vectors associated with the two sets under the above mapping.  Then:
//
// <A,B> = #(I(A,B)) = number of elements that A and B have in common
//
// (where # is the element count function and I is the intersection).  From
// this we can readily identify:
//
// ||A||_2^2 = <A,A> = #(A)
// ||A|| = sqrt(#(A))
//
// For the non-euclidean norms, we simply associate the set norms with the
// sparse vector representation norms, so:
//
// norm1(A)   = #(A)
// norm2(A)   = #(A)
// normp(A,p) = #(A)
// abs1(A)    = #(A)
// abs2(A)    = sqrt(#(A))
// absp(A,p)  = #(A)^{1/p}
//
// and, finally:
//
// absinf(A) = 0 if A is empty, 1 otherwise.
//
//
// With regard to the SVM code, note that the Mercer kernel calculation
// requires only inner products - where required, the metric ||a-b||^2 is
// calculated using the equation:
//
// ||a-b||^2 = <a,a> + <b,b> - 2<a,b>
//
// and thus no sums or products of sets are needed, and the resultant metric
// thus calculated is:^^
//
// metricDist(A,B)^2 = ||A-B||^2 = #(U(A,B)) - #(I(A,B))
//
// (where U is the union), which is the number of elements in A and B that
// fall outside of the intersection of A and B.
//
// It is important to note that by defining our inner product this way we
// guarantee that Mercer kernels constructed by the code will still be
// Mercer when applied to set-valued data.  The implicit feature map defined
// by the set valued Mercer kernel is simply the composition of the implicit
// feature map defined by the non set valued kernel and the map from sets to
// sparse binary vectors defined above.
//
// ^^technically we actually calculate the right-hand side of:
//
//  (a*-b,a*-b) = ( (a,a) - (a,b*) )* + ( (b,b) - (a*,b) )
//
//  where (a,b) = sum_i a_i.b_i (the non-conjugated inner product) and *
//  means conjugate, which in this case evaluates to the same thing.

template <class T> const T &max(const Set<T> &a);
template <class T> const T &min(const Set<T> &a);
template <class T> T max(const Set<T> &a, const Set<T> &b);
template <class T> T min(const Set<T> &a, const Set<T> &b);
template <class T> T maxabs(const Set<T> &a);
template <class T> T minabs(const Set<T> &a);
template <class T> T sqabsmax(const Set<T> &right_op);
template <class T> T sqabsmin(const Set<T> &right_op);
template <class T> T sum(const Set<T> &right_op);
template <class T> T prod(const Set<T> &right_op);
template <class T> T Prod(const Set<T> &right_op);
template <class T> T mean(const Set<T> &right_op);
template <class T> const T &median(const Set<T> &right_op);

template <class T> Set<T> &setident(Set<T> &a);
template <class T> Set<T> &setzero(Set<T> &a);
template <class T> Set<T> &setposate(Set<T> &a);
template <class T> Set<T> &setnegate(Set<T> &a);
template <class T> Set<T> &setconj(Set<T> &a);
template <class T> Set<T> &setrand(Set<T> &a);
template <class T> Set<T> &postProInnerProd(Set<T> &a) { return a; }

template <class T> double abs1(const Set<T> &a);
template <class T> double abs2(const Set<T> &a);
template <class T> double absp(const Set<T> &a, double p);
template <class T> double absinf(const Set<T> &a);
template <class T> double absd(const Set<T> &a);
template <class T> double norm1(const Set<T> &a);
template <class T> double norm2(const Set<T> &a);
template <class T> double normp(const Set<T> &a, double p);
template <class T> double normd(const Set<T> &a);
template <class T> double metricDist(const Set<T> &a, const Set<T> &b);
template <class T> double metricDist1(const Set<T> &a, const Set<T> &b);
template <class T> double metricDist2(const Set<T> &a, const Set<T> &b);
template <class T> double metricDistp(const Set<T> &a, const Set<T> &b, double p);
template <class T> double metricDistinf(const Set<T> &a, const Set<T> &b);

template <class T> double &oneProduct  (double &res, const Set<T> &a);
template <class T> double &twoProduct  (double &res, const Set<T> &a, const Set<T> &b);
template <class T> double &threeProduct(double &res, const Set<T> &a, const Set<T> &b, const Set<T> &c);
template <class T> double &fourProduct (double &res, const Set<T> &a, const Set<T> &b, const Set<T> &c, const Set<T> &d);
template <class T> double &mProduct    (double &res, int m, const Set<T> *a);

template <class T> double &twoProductNoConj (double &res, const Set<T> &a, const Set<T> &b);
template <class T> double &twoProductRevConj(double &res, const Set<T> &a, const Set<T> &b);

// Conversion from strings

template <class T> Set<T> &atoSet(Set<T> &dest, const std::string &src);

// Set arithmetic
// ==============
//
// The arithmetic operations are tricky to define for sets.  They are not
// in fact used in the SVM code, but we still define them as follows:
//
// 1. Operations on individual set elements (sets A,B,..., scalars a,b,...):
//
//   +   posation                  - elementwise, unary,  return rvalue
//   -   negation                  - elementwise, unary,  return rvalue
//   ++  prefix  increment         - elementwise, unary,  return lvalue
//   --  prefix  decrement         - elementwise, unary,  return lvalue
//   ++  postfix increment         - elementwise, unary,  return rvalue
//   ++  postfix decrement         - elementwise, unary,  return rvalue
//   ~   bitwise not               - elementwise, unary,  return rvalue
//     (A+a,a+A cases, operates on each element of A)
//   +   scalar addition           - elementwise, binary, return rvalue
//   -   scalar subtraction        - elementwise, binary, return rvalue
//   +   scalar multiplication     - elementwise, binary, return rvalue
//   /   scalar division           - elementwise, binary, return rvalue
//   %   scalar modulus            - elementwise, binary, return rvalue
//   &   scalar bitwise and        - elementwise, binary, return rvalue
//   |   scalar bitwise or         - elementwise, binary, return rvalue
//   ^   scalar bitwise xor        - elementwise, binary, return rvalue
//     (A+=a -> A = A+a, a+=A -> A = a+A)
//   +=  additive       assignment - elementwise, binary, return lvalue
//   -=  subtractive    assignment - elementwise, binary, return lvalue
//   +=  multiplicative assignment - elementwise, binary, return lvalue
//   /=  divisive       assignment - elementwise, binary, return lvalue
//   %=  modulo         assignment - elementwise, binary, return lvalue
//   &=  bitwise and    assignment - elementwise, binary, return lvalue
//   |=  bitwise or     assignment - elementwise, binary, return lvalue
//   ^=  bitwise xor    assignment - elementwise, binary, return lvalue
//   >>= left-shift     assignment - elementwise, binary, return lvalue
//   <<= right-shift    assignment - elementwise, binary, return lvalue
//
//    note that a/A  = inv(a)*A (left-division, exception to rule)
//    note that a/=A = a/A = inv(a)*A (left-division, exception to rule)
//
// 2. "True" set-wise operations
//
//   A+B = union of sets A and B
//   A-B = intersection of sets A and B
//   A*B = { [ai;bj] : A = { a0,a1,...}, B = { b0,b1,... } }
//   A+=B and A-=B are also defined, but A*=B is not
//
//   Note however that A*B will not necessarily operate precisely as
//   expected.  For example A*B*C = (A*B)*C is not well defined, and
//   moreover even A*B is not defined unless A and B have the same type
//   of scalar elements.  To fix this, we need to define a dissimilar n-tuple
//   type, and I don't know how to do that.
//
// FIXME: Implement some sort of dissimilar n-tuple type and fix A*B.
//        In a limited way you could define A*B for:
//        Set<int>*Set<int> = Set<ntuple<2>>
//        Set<int>*Set<ntuple<n>> = Set<ntuple<n+1>>
//        Set<ntuple<n>>*Set<int> = Set<ntuple<n+1>>
//        Set<ntuple<n>>*Set<ntupe<m>> = Set<ntuple<n+m>>

template <class T> Set<T>  operator+ (const Set<T> &left_op);
template <class T> Set<T>  operator- (const Set<T> &left_op);
template <class T> Set<T> &operator++(      Set<T> &left_op);
template <class T> Set<T> &operator--(      Set<T> &left_op);
template <class T> Set<T>  operator++(      Set<T> &left_op, int);
template <class T> Set<T>  operator--(      Set<T> &left_op, int);
template <class T> Set<T>  operator~ (const Set<T> &left_op);

template <class T>          Set<T>          operator+ (const Set<T> &left_op, const Set<T> &right_op);
template <class T>          Set<T>          operator+ (const Set<T> &left_op, const T      &right_op);
template <class T>          Set<T>          operator+ (const T      &left_op, const Set<T> &right_op);
template <class T>          Set<T>          operator- (const Set<T> &left_op, const Set<T> &right_op);
template <class T>          Set<T>          operator- (const Set<T> &left_op, const T      &right_op);
template <class T>          Set<T>          operator- (const T      &left_op, const Set<T> &right_op);
template <class T>          Set<Vector<T> > operator* (const Set<T> &left_op, const Set<T> &right_op);
template <class T>          Set<T>          operator* (const Set<T> &left_op, const T      &right_op);
template <class T>          Set<T>          operator* (const T      &left_op, const Set<T> &right_op);
template <class T>          Set<T>          operator/ (const Set<T> &left_op, const T      &right_op);
template <class T>          Set<T>          operator/ (const T      &left_op, const Set<T> &right_op);
template <class T>          Set<T>          operator% (const Set<T> &left_op, const T      &right_op);
template <class T>          Set<T>          operator% (const T      &left_op, const Set<T> &right_op);
template <class T>          Set<T>          operator& (const Set<T> &left_op, const T      &right_op);
template <class T>          Set<T>          operator& (const T      &left_op, const Set<T> &right_op);
template <class T>          Set<T>          operator| (const Set<T> &left_op, const T      &right_op);
template <class T>          Set<T>          operator| (const T      &left_op, const Set<T> &right_op);
template <class T>          Set<T>          operator^ (const Set<T> &left_op, const T      &right_op);
template <class T>          Set<T>          operator^ (const T      &left_op, const Set<T> &right_op);
template <class T, class S> Set<T>          operator<<(const Set<T> &left_op, const S      &right_op);
template <class T, class S> Set<T>          operator>>(const Set<T> &left_op, const S      &right_op);

template <class T>          Set<T> &operator+= (      Set<T> &left_op, const Set<T> &right_op);
template <class T>          Set<T> &operator+= (      Set<T> &left_op, const T      &right_op);
template <class T>          Set<T> &operator+= (const T      &left_op,       Set<T> &right_op);
template <class T>          Set<T> &operator-= (      Set<T> &left_op, const Set<T> &right_op);
template <class T>          Set<T> &operator-= (      Set<T> &left_op, const T      &right_op);
template <class T>          Set<T> &operator-= (const T      &left_op,       Set<T> &right_op);
template <class T>          Set<T> &operator*= (      Set<T> &left_op, const T      &right_op);
template <class T>          Set<T> &operator*= (const T      &left_op,       Set<T> &right_op);
template <class T>          Set<T> &operator/= (      Set<T> &left_op, const T      &right_op);
template <class T>          Set<T> &operator/= (const T      &left_op,       Set<T> &right_op);
template <class T>          Set<T> &operator%= (      Set<T> &left_op, const T      &right_op);
template <class T>          Set<T> &operator%= (      T      &left_op, const Set<T> &right_op);
template <class T>          Set<T> &operator&= (      Set<T> &left_op, const T      &right_op);
template <class T>          Set<T> &operator&= (      T      &left_op, const Set<T> &right_op);
template <class T>          Set<T> &operator|= (      Set<T> &left_op, const T      &right_op);
template <class T>          Set<T> &operator|= (      T      &left_op, const Set<T> &right_op);
template <class T>          Set<T> &operator^= (      Set<T> &left_op, const T      &right_op);
template <class T>          Set<T> &operator^= (      T      &left_op, const Set<T> &right_op);
template <class T, class S> Set<T> &operator<<=(      Set<T> &left_op, const S      &right_op);
template <class T, class S> Set<T> &operator>>=(      Set<T> &left_op, const S      &right_op);

// Related non-commutative operations
//
// leftmult:  equivalent to *=
// rightmult: like *=, but result is stored in right_op and ref to right_op is returned

template <class T> Set<T> &leftmult (Set<T>  &left_op, const T &right_op);
template <class T> Set<T> &rightmult(const T &left_op, Set<T>  &right_op);

// Relational operator overloading
// ===============================
//
// a == b: sets contain same elements
// a != b: logical negation of a == b
// a <  b: max(a) <  min(b)
// a <= b: max(a) <= min(b)
// a >  b: max(a) >  min(b)
// a >= b: max(a) >= min(b)
//
// single elements are evaluated as a set of one

template <class T> int operator==(const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator==(const Set<T> &left_op, const T      &right_op);
template <class T> int operator==(const T      &left_op, const Set<T> &right_op);

template <class T> int operator!=(const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator!=(const Set<T> &left_op, const T      &right_op);
template <class T> int operator!=(const T      &left_op, const Set<T> &right_op);

template <class T> int operator< (const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator< (const Set<T> &left_op, const T      &right_op);
template <class T> int operator< (const T      &left_op, const Set<T> &right_op);

template <class T> int operator<=(const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator<=(const Set<T> &left_op, const T      &right_op);
template <class T> int operator<=(const T      &left_op, const Set<T> &right_op);

template <class T> int operator> (const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator> (const Set<T> &left_op, const T      &right_op);
template <class T> int operator> (const T      &left_op, const Set<T> &right_op);

template <class T> int operator>=(const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator>=(const Set<T> &left_op, const T      &right_op);
template <class T> int operator>=(const T      &left_op, const Set<T> &right_op);




// NaN and inf tests

template <class T> int testisvnan(const Set<T> &x) { (void) x; return 0; }
template <class T> int testisinf (const Set<T> &x) { (void) x; return 0; }
template <class T> int testispinf(const Set<T> &x) { (void) x; return 0; }
template <class T> int testisninf(const Set<T> &x) { (void) x; return 0; }


// Constructors and Destructors

template <class T>
Set<T>::Set()
{
    return;
}

template <class T>
Set<T>::Set(const Set<T> &src)
{
    contents = src.contents;

    return;
}

template <class T>
Set<T>::Set(const Vector<T> &src)
{
    contents = src;
    removeDuplicates();

    return;
}

// Assignment

template <class T>
Set<T> &Set<T>::operator=(const Set<T> &src)
{
    contents = src.contents;

    return *this;
}

// Basic operations.

template <class T>
Set<T> &Set<T>::ident(void)
{
    contents.resize(0);

    return *this;
}

template <class T>
Set<T> &Set<T>::zero(void)
{
    contents.resize(0);

    return *this;
}

template <class T>
Set<T> &Set<T>::posate(void)
{
    contents.posate();

    return *this;
}

template <class T>
Set<T> &Set<T>::negate(void)
{
    contents.negate();

    return *this;
}

template <class T>
Set<T> &Set<T>::conj(void)
{
    contents.conj();

    return *this;
}

template <class T>
Set<T> &Set<T>::rand(void)
{
    contents.rand();

    return *this;
}

// Access:

template <class T>
const Vector<T> &Set<T>::all(void) const
{
    return contents;
}

template <class T>
int Set<T>::contains(const T &x) const
{
    if ( size() )
    {
        int i;

        for ( i = 0 ; i < size() ; i++ )
        {
            if ( contents(i) == x )
            {
                return 1;
            }
        }
    }

    return 0;
}

// Add and remove element functions

template <class T>
Set<T> &Set<T>::add(const T &x)
{
    if ( !contains(x) )
    {
        contents.add(size());
        contents("&",size()-1) = x;
    }

    return *this;
}

template <class T>
Set<T> &Set<T>::remove(const T &x)
{
    if ( contains(x) )
    {
        int i;

        for ( i = 0 ; i < size() ; i++ )
        {
            if ( contents(i) == x )
            {
                contents.remove(i);
                break;
            }
        }
    }

    return *this;
}

// Function application

template <class T>
Set<T> &Set<T>::applyon(T (*fn)(T))
{
    contents.applyon(fn);
    removeDuplicates();

    return *this;
}

template <class T>
Set<T> &Set<T>::applyon(T (*fn)(const T &))
{
    contents.applyon(fn);
    removeDuplicates();

    return *this;
}

template <class T>
Set<T> &Set<T>::applyon(T &(*fn)(T &))
{
    contents.applyon(fn);
    removeDuplicates();

    return *this;
}

template <class T>
Set<T> &Set<T>::applyon(T (*fn)(T, const void *), const void *a)
{
    contents.applyon(fn,a);
    removeDuplicates();

    return *this;
}

template <class T>
Set<T> &Set<T>::applyon(T (*fn)(const T &, const void *), const void *a)
{
    contents.applyon(fn,a);
    removeDuplicates();

    return *this;
}

template <class T>
Set<T> &Set<T>::applyon(T &(*fn)(T &, const void *), const void *a)
{
    contents.applyon(fn,a);
    removeDuplicates();

    return *this;
}

template <class T>
void Set<T>::removeDuplicates(void)
{
    int i,j;

    if ( size() > 1 )
    {
        for ( i = 0 ; i < size()-1 ; i++ )
        {
            for ( j = i+1 ; j < size() ; j++ )
            {
                if ( contents(i) == contents(j) )
                {
                    contents.remove(j);
                    j--;
                }
            }
        }
    }

    return;
}















































template <class T>
const T &max(const Set<T> &a)
{
    int dummy = 0;

    return max(a.all(),dummy);
}

template <class T>
const T &min(const Set<T> &a)
{
    int dummy = 0;

    return min(a.all(),dummy);
}

template <class T>
T max(const Set<T> &a, const Set<T> &b)
{
    return max(a.all(),b.all());
}

template <class T>
T min(const Set<T> &a, const Set<T> &b)
{
    return min(a.all(),b.all());
}

template <class T>
T maxabs(const Set<T> &a)
{
    int i;
    return maxabs(a.all(),i);
}

template <class T>
T minabs(const Set<T> &a)
{
    int i;
    return minabs(a.all(),i);
}

template <class T>
T sqmaxabs(const Set<T> &a)
{
    return sqmaxabs(a.all());
}

template <class T>
T sqminabs(const Set<T> &a)
{
    return sqminabs(a.all());
}

template <class T>
T sum(const Set<T> &a)
{
    return sum(a.all());
}

template <class T>
T prod(const Set<T> &a)
{
    return prod(a.all());
}

template <class T>
T Prod(const Set<T> &a)
{
    return Prod(a.all());
}

template <class T>
T mean(const Set<T> &a)
{
    return mean(a.all());
}

template <class T>
const T &median(const Set<T> &a)
{
    int dummy = 0;

    return median(a.all(),dummy);
}


template <class T>
Set<T> &setident(Set<T> &a)
{
    return a.ident();
}

template <class T>
Set<T> &setzero(Set<T> &a)
{
    return a.zero();
}

template <class T>
Set<T> &setposate(Set<T> &a)
{
    return a.posate();
}

template <class T>
Set<T> &setnegate(Set<T> &a)
{
    return a.negate();
}

template <class T>
Set<T> &setconj(Set<T> &a)
{
    return a.conj();
}

template <class T>
Set<T> &setrand(Set<T> &a)
{
    return a.rand();
}


template <class S> double abs2(const Set<S> &a)
{
    return sqrt(norm2(a));
}

template <class S> double absd(const Set<S> &a)
{
    return abs1(a);
}

template <class S> double abs1(const Set<S> &a)
{
    return norm1(a);
}

template <class S> double absp(const Set<S> &a, double p)
{
    return pow(normp(a,p),1/p);
}

template <class S> double absinf(const Set<S> &a)
{
    return a.size() ? 1 : 0;
}

template <class S> double norm2(const Set<S> &a)
{
    return a.size();
}

template <class S> double normd(const Set<S> &a)
{
    return norm1(a);
}

template <class S> double norm1(const Set<S> &a)
{
    return a.size();
}

template <class S> double normp(const Set<S> &a, double p)
{
    (void) p;

    return a.size();
}

template <class T> double &oneProduct  (double &res, const Set<T> &a)
{
    return res = a;
}

template <class T> double &threeProduct(double &res, const Set<T> &a, const Set<T> &b, const Set<T> &c)
{
    (void) res;
    (void) a;
    (void) b;
    (void) c;

    throw("3-product not defined for set");

    return res;
}

template <class T> double &fourProduct (double &res, const Set<T> &a, const Set<T> &b, const Set<T> &c, const Set<T> &d)
{
    (void) res;
    (void) a;
    (void) b;
    (void) c;
    (void) d;

    throw("4-product not defined for set");

    return res;
}

template <class T> double &mProduct(double &res, int m, const Set<T> *a)
{
    (void) res;
    (void) m;
    (void) a;

    throw("m-product not defined for set");

    return res;
}

template <class T> double &twoProduct(double &res, const Set<T> &a, const Set<T> &b)
{
    res = 0;

    if ( a.size() && b.size() )
    {
        int i;

        for ( i = 0 ; i < a.size() ; i++ )
        {
            res += b.contains((a.all())(i));
        }
    }

    return res;
}

template <class T> double &twoProductNoConj(double &res, const Set<T> &a, const Set<T> &b)
{
    return twoProduct(res,a,b);
}

template <class T> double &twoProductRevConj(double &res, const Set<T> &a, const Set<T> &b)
{
    return twoProduct(res,a,b);
}

template <class T> double metricDist(const Set<T> &a, const Set<T> &b)
{
    double temp;

    return sqrt(norm2(a)+norm2(b)-(2*twoProduct(temp,a,b)));
}

template <class T> double metricDist1(const Set<T> &a, const Set<T> &b)
{
    double temp;

    return norm2(a)+norm2(b)-(2*twoProduct(temp,a,b));
}

template <class T> double metricDistp(const Set<T> &a, const Set<T> &b, double p)
{
    double temp;

    return pow(norm2(a)+norm2(b)-(2*twoProduct(temp,a,b)),1/p);
}

template <class T> double metricDistinf(const Set<T> &a, const Set<T> &b)
{
    double temp;

    return ((int) (norm2(a)+norm2(b)-(2*twoProduct(temp,a,b)))) ? 1 : 0;
}


// Mathematical operator overloading

template <class T> T setfninc(T &x);
template <class T> T setfndec(T &x);
template <class T> T setfnnot(T &x);

template <class T> T setfladd(const T &x, const void *y);
template <class T> T setflsub(const T &x, const void *y);
template <class T> T setflmul(const T &x, const void *y);
template <class T> T setflmod(const T &x, const void *y);
template <class T> T setfland(const T &x, const void *y);
template <class T> T setflior(const T &x, const void *y);
template <class T> T setflxor(const T &x, const void *y);
template <class T> T setfnadd(const T &x, const void *y);
template <class T> T setfnsub(const T &x, const void *y);
template <class T> T setfnmul(const T &x, const void *y);
template <class T> T setfndiv(const T &x, const void *y);
template <class T> T setfnmod(const T &x, const void *y);
template <class T> T setfnand(const T &x, const void *y);
template <class T> T setfnior(const T &x, const void *y);
template <class T> T setfnxor(const T &x, const void *y);
template <class T> T setfnlsh(const T &x, const void *y);
template <class T> T setfnrsh(const T &x, const void *y);

template <class T> T setfninc(T &x) { return ++x; }
template <class T> T setfndec(T &x) { return --x; }
template <class T> T setfnnot(T &x) { return ~x;  }

template <class T> T setfladd(const T &x, const void *y) { return ( *((const T   *) y) ) +  x; }
template <class T> T setflsub(const T &x, const void *y) { return ( *((const T   *) y) ) -  x; }
template <class T> T setflmul(const T &x, const void *y) { return ( *((const T   *) y) ) *  x; }
template <class T> T setflmod(const T &x, const void *y) { return ( *((const T   *) y) ) %  x; }
template <class T> T setfland(const T &x, const void *y) { return ( *((const T   *) y) ) &  x; }
template <class T> T setflior(const T &x, const void *y) { return ( *((const T   *) y) ) |  x; }
template <class T> T setflxor(const T &x, const void *y) { return ( *((const T   *) y) ) ^  x; }
template <class T> T setfnadd(const T &x, const void *y) { return ( x +  *((const T   *) y) ); }
template <class T> T setfnsub(const T &x, const void *y) { return ( x -  *((const T   *) y) ); }
template <class T> T setfnmul(const T &x, const void *y) { return ( x *  *((const T   *) y) ); }
template <class T> T setfndiv(const T &x, const void *y) { return ( x /  *((const T   *) y) ); }
template <class T> T setfnmod(const T &x, const void *y) { return ( x %  *((const T   *) y) ); }
template <class T> T setfnand(const T &x, const void *y) { return ( x &  *((const T   *) y) ); }
template <class T> T setfnior(const T &x, const void *y) { return ( x |  *((const T   *) y) ); }
template <class T> T setfnxor(const T &x, const void *y) { return ( x ^  *((const T   *) y) ); }
template <class T> T setfnlsh(const T &x, const void *y) { return ( x << *((const int *) y) ); }
template <class T> T setfnrsh(const T &x, const void *y) { return ( x >> *((const int *) y) ); }

template <class T> Set<T>  operator+ (const Set<T> &left_op)
{
    Set<T> res(left_op);

    return res.posate();
}

template <class T> Set<T>  operator- (const Set<T> &left_op)
{
    Set<T> res(left_op);

    return res.negate();
}

template <class T> Set<T> &operator++(      Set<T> &left_op)
{
    left_op.applyon(setfninc);

    return left_op;
}

template <class T> Set<T> &operator--(      Set<T> &left_op)
{
    left_op.applyon(setfndec);

    return left_op;
}

template <class T> Set<T>  operator++(      Set<T> &left_op, int)
{
    Set<T> res(left_op);

    left_op.applyon(setfninc);

    return res;
}

template <class T> Set<T>  operator--(      Set<T> &left_op, int)
{
    Set<T> res(left_op);

    left_op.applyon(setfndec);

    return res;
}

template <class T> Set<T>  operator~ (const Set<T> &left_op)
{
    Set<T> res(left_op);

    res.applyon(setfnnot);

    return res.posate();
}

template <class T> Set<Vector<T> > operator*(const Set<T> &left_op, const Set<T> &right_op)
{
    Set<Vector<T> > res;

    if ( left_op.size() && right_op.size() )
    {
        int i,j;

        for ( i = 0 ; i < left_op.size() ; i++ )
        {
            for ( j = 0 ; j < right_op.size() ; j++ )
            {
                Vector<T> temp(2);

                temp("&",0) = (left_op.all())(i);
                temp("&",1) = (right_op.all())(j);

                res.add(temp);
            }
        }
    }

    return res;
}

template <class T> Set<T> operator+(const Set<T> &left_op, const Set<T> &right_op)
{
    Set<T> res(left_op);

    return res += right_op;
}

template <class T> Set<T> operator-(const Set<T> &left_op, const Set<T> &right_op)
{
    Set<T> res(left_op);

    return res -= right_op;
}

template <class T> Set<T> &operator+=(Set<T> &left_op, const Set<T> &right_op)
{
    if ( right_op.size() )
    {
        int i;

        for ( i = 0 ; i < right_op.size() ; i++ )
        {
            // NB: add will only add if !contains

            left_op.add((right_op.all())(i));
        }
    }

    return left_op;
}

template <class T> Set<T> &operator-=(Set<T> &left_op, const Set<T> &right_op)
{
    if ( left_op.size() )
    {
        int i;

        for ( i = (left_op.size())-1 ; i >= 0 ; i-- )
        {
            if ( !(right_op.contains((left_op.all())(i))) )
            {
                left_op.remove((left_op.all())(i));
            }
        }
    }

    return left_op;
}

template <class T>          Set<T>  operator+ (const T      &left_op, const Set<T> &right_op)
{
    Set<T> res(right_op);

    return left_op += res;
}

template <class T>          Set<T>  operator- (const T      &left_op, const Set<T> &right_op)
{
    Set<T> res(right_op);

    return left_op -= res;
}

template <class T>          Set<T>  operator* (const T      &left_op, const Set<T> &right_op)
{
    Set<T> res(right_op);

    return left_op *= res;
}

template <class T>          Set<T>  operator/ (const T      &left_op, const Set<T> &right_op)
{
    Set<T> res(right_op);

    return left_op /= res;
}

template <class T>          Set<T>  operator% (const T      &left_op, const Set<T> &right_op)
{
    Set<T> res(right_op);

    return left_op %= res;
}

template <class T>          Set<T>  operator& (const T      &left_op, const Set<T> &right_op)
{
    Set<T> res(right_op);

    return left_op &= res;
}

template <class T>          Set<T>  operator| (const T      &left_op, const Set<T> &right_op)
{
    Set<T> res(right_op);

    return left_op |= res;
}

template <class T>          Set<T>  operator^ (const T      &left_op, const Set<T> &right_op)
{
    Set<T> res(right_op);

    return left_op ^= res;
}

template <class T>          Set<T>  operator+ (const Set<T> &left_op, const T      &right_op)
{
    Set<T> res(left_op);

    return res += right_op;
}

template <class T>          Set<T>  operator- (const Set<T> &left_op, const T      &right_op)
{
    Set<T> res(left_op);

    return res -= right_op;
}

template <class T>          Set<T>  operator* (const Set<T> &left_op, const T      &right_op)
{
    Set<T> res(left_op);

    return res *= right_op;
}

template <class T>          Set<T>  operator/ (const Set<T> &left_op, const T      &right_op)
{
    Set<T> res(left_op);

    return res /= right_op;
}

template <class T>          Set<T>  operator% (const Set<T> &left_op, const T      &right_op)
{
    Set<T> res(left_op);

    return res %= right_op;
}

template <class T>          Set<T>  operator& (const Set<T> &left_op, const T      &right_op)
{
    Set<T> res(left_op);

    return res &= right_op;
}

template <class T>          Set<T>  operator| (const Set<T> &left_op, const T      &right_op)
{
    Set<T> res(left_op);

    return res |= right_op;
}

template <class T>          Set<T>  operator^ (const Set<T> &left_op, const T      &right_op)
{
    Set<T> res(left_op);

    return res ^= right_op;
}

template <class T, class S> Set<T>  operator<<(const Set<T> &left_op, const S      &right_op)
{
    Set<T> res(left_op);

    return res <<= right_op;
}

template <class T, class S> Set<T>  operator>>(const Set<T> &left_op, const S      &right_op)
{
    Set<T> res(left_op);

    return res >>= right_op;
}

template <class T>          Set<T> &operator+= (const T      &left_op,       Set<T> &right_op)
{
    right_op.applyon(setfladd,(const void *) &left_op);

    return right_op;
}

template <class T>          Set<T> &operator-= (const T      &left_op,       Set<T> &right_op)
{
    right_op.applyon(setflsub,(const void *) &left_op);

    return right_op;
}

template <class T>          Set<T> &operator*= (const T      &left_op,       Set<T> &right_op)
{
    right_op.applyon(setflmul,(const void *) &left_op);

    return right_op;
}

template <class T>          Set<T> &operator/= (const T      &left_op,       Set<T> &right_op)
{
    return inv(left_op)*right_op;
}

template <class T>          Set<T> &operator%= (const T      &left_op,       Set<T> &right_op)
{
    right_op.applyon(setflmod,(const void *) &left_op);

    return right_op;
}

template <class T>          Set<T> &operator&= (const T      &left_op,       Set<T> &right_op)
{
    right_op.applyon(setfland,(const void *) &left_op);

    return right_op;
}

template <class T>          Set<T> &operator|= (const T      &left_op,       Set<T> &right_op)
{
    right_op.applyon(setflior,(const void *) &left_op);

    return right_op;
}

template <class T>          Set<T> &operator^= (const T      &left_op,       Set<T> &right_op)
{
    right_op.applyon(setflxor,(const void *) &left_op);

    return right_op;
}

template <class T>          Set<T> &operator+= (      Set<T> &left_op, const T      &right_op)
{
    left_op.applyon(setfnadd,(const void *) &right_op);

    return left_op;
}

template <class T>          Set<T> &operator-= (      Set<T> &left_op, const T      &right_op)
{
    left_op.applyon(setfnsub,(const void *) &right_op);

    return left_op;
}

template <class T>          Set<T> &operator*= (      Set<T> &left_op, const T      &right_op)
{
    left_op.applyon(setfnmul,(const void *) &right_op);

    return left_op;
}

template <class T>          Set<T> &operator/= (      Set<T> &left_op, const T      &right_op)
{
    left_op.applyon(setfndiv,(const void *) &right_op);

    return left_op;
}

template <class T>          Set<T> &operator%= (      Set<T> &left_op, const T      &right_op)
{
    left_op.applyon(setfnmod,(const void *) &right_op);

    return left_op;
}

template <class T>          Set<T> &operator&= (      Set<T> &left_op, const T      &right_op)
{
    left_op.applyon(setfnand,(const void *) &right_op);

    return left_op;
}

template <class T>          Set<T> &operator|= (      Set<T> &left_op, const T      &right_op)
{
    left_op.applyon(setfnior,(const void *) &right_op);

    return left_op;
}

template <class T>          Set<T> &operator^= (      Set<T> &left_op, const T      &right_op)
{
    left_op.applyon(setfnxor,(const void *) &right_op);

    return left_op;
}

template <class T, class S> Set<T> &operator<<=(      Set<T> &left_op, const S      &right_op)
{
    left_op.applyon(setfnlsh,(const void *) &right_op);

    return left_op;
}

template <class T, class S> Set<T> &operator>>=(      Set<T> &left_op, const S      &right_op)
{
    left_op.applyon(setfnrsh,(const void *) &right_op);

    return left_op;
}

template <class T> Set<T> &leftmult (Set<T>  &left_op, const T &right_op)
{
    return left_op *= right_op;
}

template <class T> Set<T> &rightmult(const T &left_op, Set<T>  &right_op)
{
    return left_op *= right_op;
}

// Logical operator overloading

template <class T> int operator==(const Set<T> &left_op, const Set<T> &right_op)
{
    if ( left_op.size() != right_op.size() )
    {
        return 0;
    }

    if ( left_op.size() )
    {
        int i;

        for ( i = 0 ; i < left_op.size() ; i++ )
        {
            if ( !(right_op.contains((left_op.all())(i))) )
            {
                return 0;
            }
        }
    }

    return 1;
}

template <class T> int operator==(const Set<T> &left_op, const T      &right_op)
{
    return ( ( left_op.size() == 1 ) && left_op.contains(right_op) );
}

template <class T> int operator==(const T      &left_op, const Set<T> &right_op)
{
    return ( ( right_op.size() == 1 ) && right_op.contains(left_op) );
}

template <class T> int operator!=(const Set<T> &left_op, const Set<T> &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator!=(const Set<T> &left_op, const T      &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator!=(const T      &left_op, const Set<T> &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator< (const Set<T> &left_op, const Set<T> &right_op)
{
    if ( !(left_op.size()) || !(right_op.size()) )
    {
        return ( left_op.size() == right_op.size() );
    }

    return max(left_op) < min(right_op);
}

template <class T> int operator< (const Set<T> &left_op, const T      &right_op)
{
    if ( !(left_op.size()) )
    {
        return 0;
    }

    return max(left_op) < right_op;
}

template <class T> int operator< (const T      &left_op, const Set<T> &right_op)
{
    if ( !(right_op.size()) )
    {
        return 0;
    }

    return left_op < min(right_op);
}

template <class T> int operator<=(const Set<T> &left_op, const Set<T> &right_op)
{
    if ( !(left_op.size()) || !(right_op.size()) )
    {
        return ( left_op.size() == right_op.size() );
    }

    return max(left_op) <= min(right_op);
}

template <class T> int operator<=(const Set<T> &left_op, const T      &right_op)
{
    if ( !(left_op.size()) )
    {
        return 0;
    }

    return max(left_op) <= right_op;
}

template <class T> int operator<=(const T      &left_op, const Set<T> &right_op)
{
    if ( !(right_op.size()) )
    {
        return 0;
    }

    return left_op <= min(right_op);
}

template <class T> int operator> (const Set<T> &left_op, const Set<T> &right_op)
{
    return right_op < left_op;
}

template <class T> int operator> (const Set<T> &left_op, const T      &right_op)
{
    return right_op < left_op;
}

template <class T> int operator> (const T      &left_op, const Set<T> &right_op)
{
    return right_op < left_op;
}

template <class T> int operator>=(const Set<T> &left_op, const Set<T> &right_op)
{
    return right_op <= left_op;
}

template <class T> int operator>=(const Set<T> &left_op, const T      &right_op)
{
    return right_op <= left_op;
}

template <class T> int operator>=(const T      &left_op, const Set<T> &right_op)
{
    return right_op <= left_op;
}

// Conversion from strings

template <class T> Set<T> &atoSet(Set<T> &dest, const std::string &src)
{
    std::istringstream srcmod(src);

    srcmod >> dest;

    return dest;
}

// Stream operators

template <class T>
std::ostream &operator<<(std::ostream &output, const Set<T> &src)
{
    // This is just a copy of the vector streamer with some mods

    int size = src.size();

    output << "{ ";

    if ( size )
    {
	int i;

	for ( i = 0 ; i < size ; i++ )
	{
	    if ( i < size-1 )
	    {
                output << (src.all())(i) << " ; ";
	    }

	    else
	    {
                output << (src.all())(i);
	    }
	}
    }

    output << "  }";

    return output;
}

template <class T>
std::istream &operator>>(std::istream &input, Set<T> &dest)
{
    // This is just a copy of the vector streamer with some mods

    (dest.contents).resize(0);

    char tt;

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    NiceAssert( tt == '{' );

    int size = 0;

    while ( 1 )
    {
        while ( ( isspace(input.peek()) ) || ( input.peek() == ';' ) )
        {
            input.get(tt);
        }

        if ( input.peek() == '}' )
        {
            input.get(tt);

            break;
        }

        (dest.contents).add(size);
        input >> (dest.contents)("&",size);

        size++;
    }

    dest.removeDuplicates();

    return input;
}

template <class T>
std::istream &streamItIn(std::istream &input, Set<T> &dest, int processxyzvw)
{
    // This is just a copy of the vector streamItIn with some mods

    (dest.contents).resize(0);

    int i;
    char tt;

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    NiceAssert( tt == '{' );

    int size = 0;

    while ( 1 )
    {
        while ( ( isspace(input.peek()) ) || ( input.peek() == ';' ) )
        {
            input.get(tt);
        }

        if ( input.peek() == '}' )
        {
            input.get(tt);

            break;
        }

        (dest.contents).add(size);
        streamItIn(input,(dest.contents)("&",size),processxyzvw);

        size++;
    }

    dest.removeDuplicates();

    return input;
}

#endif

