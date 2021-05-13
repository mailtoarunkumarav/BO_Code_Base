
//
// Directed graph class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _dgraph_h
#define _dgraph_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <sstream>
#include "matrix.h"

template <class T, class S> class Dgraph;

// Stream operators

template <class T, class S> std::ostream &operator<<(std::ostream &output, const Dgraph<T,S> &src );
template <class T, class S> std::istream &operator>>(std::istream &input,        Dgraph<T,S> &dest);
template <class T, class S> std::istream &streamItIn(std::istream &input,        Dgraph<T,S> &dest, int processxyzvw = 1);

// Swap function

template <class T, class S> void qswap(const Dgraph<T,S> *a, const Dgraph<T,S> *b);
template <class T, class S> void qswap(Dgraph<T,S> &a, Dgraph<T,S> &b);

// The class itself

template <class T, class S>
class Dgraph
{
    template <class U, class V> friend std::istream &operator>>(std::istream &input, Dgraph<U,V> &dest);
    template <class U, class V> friend void qswap(Dgraph<U,V> &a, Dgraph<U,V> &b);

public:

    // Constructor:

    svm_explicit Dgraph();
                 Dgraph(const Dgraph<T,S> &src);
    svm_explicit Dgraph(const Vector<T> &xgnodes, const Matrix<S> &xedgeWeights);

    // Assignment:

    Dgraph<T,S> &operator=(const Dgraph<T,S> &src);

    // simple graph manipulations
    //
    // ident: remove all nodes
    // zero: remove all nodes
    // posate: does nothing
    // negate: reverses direction of all nodes (transposes W)
    // conj: does nothing
    // rand: throws error
    //
    // each returns a reference to *this

    Dgraph<T,S> &ident(void);
    Dgraph<T,S> &zero(void);
    Dgraph<T,S> &posate(void);
    Dgraph<T,S> &negate(void);
    Dgraph<T,S> &conj(void);
    Dgraph<T,S> &rand(void);

    // Access:
    //
    // - nodes: returns the set of nodes
    // - edgeWeights: returns the edge weights (matrix or x to y)
    // - contains(x): returns 1 if x is in nodes or edges, 0 otherwise

    const Vector<T> &all(void) const;
    Vector<T> &allunsafe(void);
    int contains(const T &x) const;
    const Matrix<S> &edgeWeights(void) const;
    const S &edgeWeights(const T &x, const T &y) const;

    // Add (if not present) and remove (if present) elements:

    Dgraph<T,S> &add(const T &x);
    Dgraph<T,S> &add(int n, const T &x);
    Dgraph<T,S> &remove(const T &x);
    Dgraph<T,S> &setWeight(const T &x, const T &y, const S &val);
    Dgraph<T,S> &setWeightxy(int i, int j, const S &val);
    S &accessWeightxy(int i, int j);

    // Information:

    int size(void)    const { return gnodes.size();    }
    int numRows(void) const { return gedges.numRows(); }
    int numCols(void) const { return gedges.numCols(); }

    // Don't use this

    Vector<T> &ncall(void) { return gnodes; }

private:

    Vector<T> gnodes;
    Matrix<S> gedges;
};

template  <class T, class S> void qswap(Dgraph<T,S> &a, Dgraph<T,S> &b)
{
    qswap(a.gnodes,b.gnodes);
    qswap(a.gedges,b.gedges);

    return;
}

template <class T, class S> void qswap(const Dgraph<T,S> *a, const Dgraph<T,S> *b)
{
    const Dgraph<T,S> *c;

    c = a;
    a = b;
    b = c;

    return;
}

// Various functions
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
// In practice, only the metric distance is really well defined for the
// directed graph.  However for the sake of the SVM code, which calculates
// the metric distance from the inner product we need to define the inner
// product, which by inference defines the norms (at least gives a function -
// although the functions may not define inner products or norms in the
// mathematical sense).
//
// So, to that end, and refering to the equations in set.h:
//
// <A,B> = (abs2(A)+abs2(B)-metricDist(A,B))/2
// metricDist(A,B) = abs2(A)+abs2(B)-2<A,B>
//
// abs2:  abs2(A) = sqrt(norm2(A))
// norm2: norm2(A) = <A,A>
//
// metricDist:   metricDist(A,B) = abs2(A)+abs2(B)-2<A,B>
// twoProduct: 

template <class T, class S> Dgraph<T,S> &setident(Dgraph<T,S> &a);
template <class T, class S> Dgraph<T,S> &setzero(Dgraph<T,S> &a);
template <class T, class S> Dgraph<T,S> &setposate(Dgraph<T,S> &a);
template <class T, class S> Dgraph<T,S> &setnegate(Dgraph<T,S> &a);
template <class T, class S> Dgraph<T,S> &setconj(Dgraph<T,S> &a);
template <class T, class S> Dgraph<T,S> &setrand(Dgraph<T,S> &a);
template <class T, class S> Dgraph<T,S> &postProInnerProd(Dgraph<T,S> &a) { return a; }

template <class T, class S> double abs1  (const Dgraph<T,S> &a);
template <class T, class S> double abs2  (const Dgraph<T,S> &a);
template <class T, class S> double absp  (const Dgraph<T,S> &a, double p);
template <class T, class S> double absinf(const Dgraph<T,S> &a);
template <class T, class S> double absd  (const Dgraph<T,S> &a);
template <class T, class S> double norm1  (const Dgraph<T,S> &a);
template <class T, class S> double norm2  (const Dgraph<T,S> &a);
template <class T, class S> double normp  (const Dgraph<T,S> &a);
template <class T, class S> double norminf(const Dgraph<T,S> &a);
template <class T, class S> double normd  (const Dgraph<T,S> &a);

template <class T, class S> double metricDist(const Dgraph<T,S> &a, const Dgraph<T,S> &b);

template <class T, class S> double &oneProduct  (double &res, const Dgraph<T,S> &a);
template <class T, class S> double &twoProduct  (double &res, const Dgraph<T,S> &a, const Dgraph<T,S> &b);
template <class T, class S> double &threeProduct(double &res, const Dgraph<T,S> &a, const Dgraph<T,S> &b, const Dgraph<T,S> &c);
template <class T, class S> double &fourProduct (double &res, const Dgraph<T,S> &a, const Dgraph<T,S> &b, const Dgraph<T,S> &c, const Dgraph<T,S> &d);
template <class T, class S> double &mProduct    (double &res, int m, const Dgraph<T,S> *a);

template <class T, class S> double &twoProductNoConj (double &res, const Dgraph<T,S> &a, const Dgraph<T,S> &b);
template <class T, class S> double &twoProductRevConj(double &res, const Dgraph<T,S> &a, const Dgraph<T,S> &b);

// Directed graph product
//
// The product of two directed graphs:
//
// A = { An ; Aw }
// B = { Bn ; Bw }
//
// is:
//
// A*B = { Cn ; Cw }
//
// Cn = [ ( [An0,Bn0], [An0,Bn1], ..., [An1,Bn0], [An1,Bn1], ... }
//
// Cw = [ Aw00*Bw00  Aw00*Bw01  ...  Aw01*Bw00  Aw01*Bw01  ... ]
//      [ Aw00*Bw10  Aw00*Bw11  ...  Aw01*Bw10  Aw01*Bw11  ... ]
//      [  ...        ...             ...        ...           ]
//      [ Aw10*Bw00  Aw10*Bw01  ...  Aw11*Bw00  Aw11*Bw01  ... ]
//      [ Aw10*Bw10  Aw10*Bw11  ...  Aw11*Bw10  Aw11*Bw11  ... ]
//      [  ...        ...             ...        ...           ]

template <class T, class S> Dgraph<Vector<T>,S> operator* (const Dgraph<T,S> &left_op, const Dgraph<T,S> &right_op);

// Relational operator overloading

template <class T, class S> int operator==(const Dgraph<T,S> &left_op, const Dgraph<T,S> &right_op);
template <class T, class S> int operator!=(const Dgraph<T,S> &left_op, const Dgraph<T,S> &right_op);

// Conversion from strings

template <class T, class S> Dgraph<T,S> &atoSet(Dgraph<T,S> &dest, const std::string &src);




// NaN and inf tests

template <class S, class T> int testisvnan(const Dgraph<S,T> &x) { (void) x; return 0; }
template <class S, class T> int testisinf (const Dgraph<S,T> &x) { (void) x; return 0; }
template <class S, class T> int testispinf(const Dgraph<S,T> &x) { (void) x; return 0; }
template <class S, class T> int testisninf(const Dgraph<S,T> &x) { (void) x; return 0; }












// Constructors and Destructors

template <class T, class S>
Dgraph<T,S>::Dgraph()
{
    return;
}

template <class T, class S>
Dgraph<T,S>::Dgraph(const Dgraph<T,S> &src)
{
    gnodes = src.gnodes;
    gedges = src.gedges;

    return;
}

template <class T, class S>
Dgraph<T,S>::Dgraph(const Vector<T> &xgnodes, const Matrix<S> &xedgeWeights)
{
    gnodes = xgnodes;
    gedges = xedgeWeights;

    NiceAssert( xgnodes.size() == xedgeWeights.numRows() );
    NiceAssert( xgnodes.size() == xedgeWeights.numCols() );

    return;
}

// Assignment

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::operator=(const Dgraph<T,S> &src)
{
    gnodes = src.gnodes;
    gedges = src.gedges;

    return *this;
}

// Basic operations.

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::ident(void)
{
    gnodes.resize(0);
    gedges.resize(0,0);

    return *this;
}

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::zero(void)
{
    gnodes.resize(0);
    gedges.resize(0,0);

    return *this;
}

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::posate(void)
{
    return *this;
}

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::negate(void)
{
    gedges.transpose();

    return *this;
}

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::conj(void)
{
    return *this;
}

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::rand(void)
{
    throw("Random graph generation not supported");

    return *this;
}

// Access:

template <class T, class S>
const Vector<T> &Dgraph<T,S>::all(void) const
{
    return gnodes;
}

template <class T, class S>
Vector<T> &Dgraph<T,S>::allunsafe(void)
{
    return gnodes;
}

template <class T, class S>
int Dgraph<T,S>::contains(const T &x) const
{
    if ( size() )
    {
        int i;

        for ( i = 0 ; i < size() ; i++ )
        {
            if ( gnodes(i) == x )
            {
                return 1;
            }
        }
    }

    return 0;
}

template <class T, class S>
const Matrix<S> &Dgraph<T,S>::edgeWeights(void) const
{
    return gedges;
}

template <class T, class S>
const S &Dgraph<T,S>::edgeWeights(const T &x, const T &y) const
{
    NiceAssert( size() );

    int i,j;

    for ( i = 0 ; i < size() ; i++ )
    {
        if ( gnodes(i) == x )
        {
            break;
        }
    }

    for ( j = 0 ; j < size() ; j++ )
    {
        if ( gnodes(j) == y )
        {
            break;
        }
    }

    return gedges(i,j);
}

// Add and remove element functions

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::add(const T &x)
{
    if ( !contains(x) )
    {
        add(gnodes.size(),x);
    }

    return *this;
}

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::add(int i, const T &x)
{
    gnodes.add(i);
    gedges.addRowCol(i);

    gnodes("&",i) = x;
    gedges("&",i,0,1,size()-1    ) = 0;
    gedges("&",0,1,size()-1,i,"x") = 0;

    return *this;
}

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::remove(const T &x)
{
    if ( contains(x) )
    {
        int i;

        for ( i = 0 ; i < size() ; i++ )
        {
            if ( gnodes(i) == x )
            {
                gnodes.remove(i);
                gedges.removeRowCol(i);
                break;
            }
        }
    }

    return *this;
}

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::setWeight(const T &x, const T &y, const S &val)
{
    NiceAssert( size() );

    int i,j;

    for ( i = 0 ; i < size() ; i++ )
    {
        if ( gnodes(i) == x )
        {
            break;
        }
    }

    for ( j = 0 ; j < size() ; j++ )
    {
        if ( gnodes(j) == y )
        {
            break;
        }
    }

    return setWeightxy(i,j,val);
}

template <class T, class S>
Dgraph<T,S> &Dgraph<T,S>::setWeightxy(int i, int j, const S &val)
{
    NiceAssert( i < size() );
    NiceAssert( j < size() );
    NiceAssert( i >= 0 );
    NiceAssert( j >= 0 );

    gedges("&",i,j) = val;

    return *this;
}

template <class T, class S>
S &Dgraph<T,S>::accessWeightxy(int i, int j)
{
    NiceAssert( i < size() );
    NiceAssert( j < size() );
    NiceAssert( i >= 0 );
    NiceAssert( j >= 0 );

    return gedges("&",i,j);
}














































template <class T, class S>
Dgraph<T,S> &setident(Dgraph<T,S> &a)
{
    return a.ident();
}

template <class T, class S>
Dgraph<T,S> &setzero(Dgraph<T,S> &a)
{
    return a.zero();
}

template <class T, class S>
Dgraph<T,S> &setposate(Dgraph<T,S> &a)
{
    return a.posate();
}

template <class T, class S>
Dgraph<T,S> &setnegate(Dgraph<T,S> &a)
{
    return a.negate();
}

template <class T, class S>
Dgraph<T,S> &setconj(Dgraph<T,S> &a)
{
    return a.conj();
}

template <class T, class S>
Dgraph<T,S> &setrand(Dgraph<T,S> &a)
{
    return a.rand();
}







template <class T, class S> double abs1(const Dgraph<T,S> &a)
{
    return norm1(a);
}

template <class T, class S> double abs2(const Dgraph<T,S> &a)
{
    return sqrt(norm2(a));
}

template <class T, class S> double absd(const Dgraph<T,S> &a)
{
    return abs2(a);
}

template <class T, class S> double absp(const Dgraph<T,S> &a, double p)
{                                                
    return pow(normp(a,p),1/p);
}

template <class T, class S> double absinf(const Dgraph<T,S> &a)
{
    return abs2(a);
}

template <class T, class S> double norm1(const Dgraph<T,S> &a)
{
    return norm2(a);
}

template <class T, class S> double norm2(const Dgraph<T,S> &a)
{
    double temp;

    return twoProduct(temp,a,a);
}

template <class T, class S> double normp(const Dgraph<T,S> &a, double p)
{
    (void) p;

    return norm2(a);
}

template <class T, class S> double normd(const Dgraph<T,S> &a)
{
    return norm2(a);
}

template <class T, class S> double &oneProduct(double &res, const Dgraph<T,S> &a)
{
    return res = a;
}

template <class T, class S> double &twoProduct(double &res, const Dgraph<T,S> &a, const Dgraph<T,S> &b)
{
    throw("2-product undefined for graphs");

    (void) a;
    (void) b;

    res = 0;
    //phantomx
    return res;
}

template <class T, class S> double &threeProduct(double &res, const Dgraph<T,S> &a, const Dgraph<T,S> &b, const Dgraph<T,S> &c)
{
    throw("3-product undefined for graphs");

    (void) a;
    (void) b;
    (void) c;

    res = 0;
    //phantomx
    return res;
}

template <class T, class S> double &fourProduct (double &res, const Dgraph<T,S> &a, const Dgraph<T,S> &b, const Dgraph<T,S> &c, const Dgraph<T,S> &d)
{
    throw("4-product undefined for graphs");

    (void) a;
    (void) b;
    (void) c;
    (void) d;

    res = 0;
    //phantomx
    return res;
}

template <class T, class S> double &mProduct(double &res, int m, const Dgraph<T,S> *a)
{
    throw("m-product undefined for graphs");

    (void) m;
    (void) a;

    res = 0;
    //phantomx
    return res;
}

template <class T, class S> double &twoProductNoConj(double &res, const Dgraph<T,S> &a, const Dgraph<T,S> &b)
{
    throw("2-product undefined for graphs");

    (void) a;
    (void) b;

    res = 0;
    //phantomx
    return res;
}

template <class T, class S> double &twoProductRevConj(double &res, const Dgraph<T,S> &a, const Dgraph<T,S> &b)
{
    throw("2-product undefined for graphs");

    (void) a;
    (void) b;

    res = 0;
    //phantomx
    return res;
}

template <class T, class S> double metricDist(const Dgraph<T,S> &a, const Dgraph<T,S> &b)
{
    return abs2(a)+abs2(b)-(2*metricDist(a,b));
}







// Directed graph product
//
// The product of two directed graphs:
//
// A = { An ; Aw }
// B = { Bn ; Bw }
//
// is:
//
// A*B = { Cn ; Cw }
//
// Cn = [ ( [An0,Bn0], [An0,Bn1], ..., [An1,Bn0], [An1,Bn1], ... }
//
// Cw = [ Aw00*Bw00  Aw00*Bw01  ...  Aw01*Bw00  Aw01*Bw01  ... ]
//      [ Aw00*Bw10  Aw00*Bw11  ...  Aw01*Bw10  Aw01*Bw11  ... ]
//      [  ...        ...             ...        ...           ]
//      [ Aw10*Bw00  Aw10*Bw01  ...  Aw11*Bw00  Aw11*Bw01  ... ]
//      [ Aw10*Bw10  Aw10*Bw11  ...  Aw11*Bw10  Aw11*Bw11  ... ]
//      [  ...        ...             ...        ...           ]

template <class T, class S> Dgraph<Vector<T>,S> operator* (const Dgraph<T,S> &a, const Dgraph<T,S> &b)
{
    Dgraph<Vector<T>,S> res;

    if ( a.size() && b.size() )
    {
        int i,j,k,l,m,n;

        for ( i = 0 ; i < a.size() ; i++ )
        {
            for ( j = 0 ; j < b.size() ; j++ )
            {
                Vector<T> temp(2);

                temp("&",0) = (a.all())(i);
                temp("&",1) = (b.all())(j);

                res.add(temp);
            }
        }

        // up-down on outer loop, left-right on inner loop

        m = 0;

        for ( i = 0 ; i < a.size() ; i++ )
        {
            for ( j = 0 ; j < b.size() ; j++ )
            {
                n = 0;

                for ( k = 0 ; k < a.size() ; k++ )
                {
                    for ( l = 0 ; l < b.size() ; l++ )
                    {
                        res.setWeight(m,n,((a.edgeWeights())(i,k))*((b.edgeWeights())(j,l)));

                        n++;
                    }
                }

                m++;
            }
        }
    }

    return res;
}


// Logical operator overloading

template <class T, class S> int operator==(const Dgraph<T,S> &left_op, const Dgraph<T,S> &right_op)
{
    return ( left_op.all() == right_op.all() ) && ( left_op.edgeWeights() == right_op.edgeWeights() );
}

template <class T, class S> int operator!=(const Dgraph<T,S> &left_op, const Dgraph<T,S> &right_op)
{
    return !( left_op == right_op );
}

// Conversion from strings

template <class T, class S> Dgraph<T,S> &atoSet(Dgraph<T,S> &dest, const std::string &src)
{
    std::istringstream srcmod(src);

    srcmod >> dest;

    return dest;
}

// Stream operators

template <class T, class S>
std::ostream &operator<<(std::ostream &output, const Dgraph<T,S> &src)
{
    output << "{ " << src.all() << " ; " << src.edgeWeights() << " }";

    return output;
}

template <class T, class S>
std::istream &operator>>(std::istream &input, Dgraph<T,S> &dest)
{
    (dest.gnodes).resize(0);
    (dest.gedges).resize(0,0);

    char tt;

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    NiceAssert( tt == '{' );

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input >> dest.gnodes;

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    NiceAssert( tt == ';' );

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input >> dest.gedges;

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    NiceAssert( tt == '}' );

    return input;
}

template <class T, class S>
std::istream &streamItIn(std::istream &input, Dgraph<T,S> &dest, int processxyzvw)
{
    (dest.gnodes).resize(0);
    (dest.gedges).resize(0);

    char tt;

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    NiceAssert( tt == '{' );

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    streamItIn(input,dest.gnodes,processxyzvw);

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    NiceAssert( tt == ';' );

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    streamItIn(input,dest.gedges,processxyzvw);

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    NiceAssert( tt == '}' );

    NiceAssert( dest.sanitycheck() );

    return input;
}

#endif

