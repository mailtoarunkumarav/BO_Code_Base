
//
// RKHS Vector class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _RKHSVector_h
#define _RKHSVector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <sstream>
#include "vector.h"
#include "sparsevector.h"
#include "gentype.h"
#include "mercer.h"



template <class T> class RKHSVector;

// Stream operators

template <class T> std::ostream &operator<<(std::ostream &output, const RKHSVector<T> &src );
template <class T> std::istream &operator>>(std::istream &input,        RKHSVector<T> &dest);
template <class T> std::istream &streamItIn(std::istream &input,        RKHSVector<T> &dest, int processxyzvw = 1);

// Swap function

template <class T> void qswap(const RKHSVector<T> *&a, const RKHSVector<T> *&b);
template <class T> void qswap(RKHSVector<T>       *&a, RKHSVector<T>       *&b);
template <class T> void qswap(RKHSVector<T>        &a, RKHSVector<T>        &b);

// Creation operators

template <> inline void makeRKHSVector(Vector<double> *&res, std::istream &src);
template <> inline void makeRKHSVector(Vector<double> *&res, std::istream &src, int processxyzvw);

template <> inline void makeRKHSVector(Vector<gentype> *&res, std::istream &src);
template <> inline void makeRKHSVector(Vector<gentype> *&res, std::istream &src, int processxyzvw);

// Calculate L2 distance squared from RKHSVector to function of given dimension,
// assuming a function of var(0,0), var(0,1), ..., var(0,dim-1)
//
// It is assume functions are over [0,1]^dim with gran steps per dimension
//
// scaleit 1 means L2 norm, scaleit2 means L2 norm * granularity

//Important: this number needs to be the same as xnSamp in globalopt.h
#define DEFAULT_GRANULARITY 100

inline double calcL2distsq(const Vector<gentype> &f, gentype &g, int dim, int scaleit = 1, int gran = DEFAULT_GRANULARITY);


// The class itself
//
// This represents sum_i a_i a_i K(x,x_i)
//
// T can only be double or gentype, nothing else

template <>
inline int aresame(gentype *, gentype *)
{
    return 1;
}



template <class T>
class RKHSVector : public Vector<T>
{
    template <class S> friend void qswap(RKHSVector<S> &a, RKHSVector<S> &b);

public:

    // Constructors and Destructors

    RKHSVector() : Vector<T>()  { ; } 
    RKHSVector(const RKHSVector<T> &src) : Vector<T>() { assign(src); } 
    virtual ~RKHSVector() { return; }

    // Print and make duplicate

    virtual std::ostream &outstream(std::ostream &output) const;
    virtual std::istream &instream (std::istream &input );

    virtual std::istream &streamItIn(std::istream &input, int processxyzvw = 1);

    virtual Vector<T> *makeDup(void) const
    {
        RKHSVector<T> *dup;

        MEMNEW(dup,RKHSVector<T>(*this));

        return static_cast<Vector<T> *>(dup);
    }

    // Assignment

    RKHSVector<T> &operator=(const RKHSVector<T> &src) { return assign(src); }

    virtual RKHSVector<T> &assign(const RKHSVector<T> &src) 
    { 
        spKern = src.spKern; 
        alpha  = src.alpha; 
        xx     = src.xx; 

        return *this; 
    }

    // Simple vector manipulations

    virtual Vector<T> &softzero(void) { alpha.softzero(); return *this; }
    virtual Vector<T> &zero(void)     { alpha.zero();     return *this; }
    virtual Vector<T> &posate(void)   { alpha.posate();   return *this; }
    virtual Vector<T> &negate(void)   { alpha.negate();   return *this; }
    virtual Vector<T> &conj(void)     { alpha.conj();     return *this; }
    virtual Vector<T> &rand(void)     { alpha.rand();     return *this; }

    // Access:
    //
    // - vector has the functional form f(x) = sum_{i=0}^{N-1} alpha_i K(x_i,x)
    // - to evaluate f(x) use operator()
    // - to access alpha_i use f.a(...)
    // - to access x_i use f.x(...)
    // - to access kernel use f.kern(...)

    virtual T &operator()(T &res, const              gentype  &i) const { SparseVector<gentype> ii; ii("&",zeroint()) = i; return (*this)(res,ii); }
    virtual T &operator()(T &res, const       Vector<gentype> &i) const { SparseVector<gentype> ii(i);                     return (*this)(res,ii); }
    virtual T &operator()(T &res, const SparseVector<gentype> &i) const;

    Vector<T> &a(const char *dummy,                         retVector<T> &tmp) { return alpha(dummy,tmp);          }
    T         &a(const char *dummy, int i                                    ) { return alpha(dummy,i);            }
    Vector<T> &a(const char *dummy, const Vector<int> &i,   retVector<T> &tmp) { return alpha(dummy,i,tmp);        }
    Vector<T> &a(const char *dummy, int ib, int is, int im, retVector<T> &tmp) { return alpha(dummy,ib,is,im,tmp); }

    const Vector<T> &a(                        retVector<T> &tmp) const { return alpha(tmp);          }
    const T         &a(int i                                    ) const { return alpha(i);            }
    const Vector<T> &a(const Vector<int> &i,   retVector<T> &tmp) const { return alpha(i,tmp);        }
    const Vector<T> &a(int ib, int is, int im, retVector<T> &tmp) const { return alpha(ib,is,im,tmp); }

    Vector<SparseVector<gentype> > &x(const char *dummy,                         retVector<SparseVector<gentype> > &tmp) { return xx(dummy,tmp);          }
    SparseVector<gentype>          &x(const char *dummy, int i                                                         ) { return xx(dummy,i);            }
    Vector<SparseVector<gentype> > &x(const char *dummy, const Vector<int> &i,   retVector<SparseVector<gentype> > &tmp) { return xx(dummy,i,tmp);        }
    Vector<SparseVector<gentype> > &x(const char *dummy, int ib, int is, int im, retVector<SparseVector<gentype> > &tmp) { return xx(dummy,ib,is,im,tmp); }

    const Vector<SparseVector<gentype> > &x(                        retVector<SparseVector<gentype> > &tmp) const { return xx(tmp);          }
    const SparseVector<gentype>          &x(int i                                                         ) const { return xx(i);            }
    const Vector<SparseVector<gentype> > &x(const Vector<int> &i,   retVector<SparseVector<gentype> > &tmp) const { return xx(i,tmp);        }
    const Vector<SparseVector<gentype> > &x(int ib, int is, int im, retVector<SparseVector<gentype> > &tmp) const { return xx(ib,is,im,tmp); }

    MercerKernel &kern(const char *dummy) { (void) dummy; return spKern; }
    const MercerKernel &kern(void) const { return spKern; }

    // Information functions

    virtual int infsize(void) const { return 1; }
    virtual int inrkhs(void)  const { return 1; }

    // Pre-allocation control.

    virtual void prealloc(int newallocsize)  { alpha.prealloc(newallocsize);  xx.prealloc(newallocsize);  return; }
    virtual void useStandardAllocation(void) { alpha.useStandardAllocation(); xx.useStandardAllocation(); return; }
    virtual void useTightAllocation(void)    { alpha.useTightAllocation();    xx.useTightAllocation();    return; }
    virtual void useSlackAllocation(void)    { alpha.useSlackAllocation();    xx.useSlackAllocation();    return; }

    virtual int array_norm (void) const { return alpha.array_norm();  }
    virtual int array_tight(void) const { return alpha.array_tight(); }
    virtual int array_slack(void) const { return alpha.array_slack(); }

    // RKHS specific stuff
    //
    // N is the basis size
    // resizeN changes this.  If decreasing size it will sparsify by removing the smallest alpha's (by 2-norm) first.

    virtual int N(void) const { return alpha.size(); }
    virtual void resizeN(int N);

    // Inner-product functions for RKHS
    //
    // conj = 0: noConj
    //        1: normal
    //        2: revConj

    inline virtual T &inner1(T &res                                                            ) const;
    inline virtual T &inner2(T &res, const Vector<T> &b, int conj = 1                          ) const;
    inline virtual T &inner3(T &res, const Vector<T> &b, const Vector<T> &c                    ) const;
    inline virtual T &inner4(T &res, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d) const;

    inline virtual double &inner1Real(double &res                                                            ) const;
    inline virtual double &inner2Real(double &res, const Vector<T> &b, int conj = 1                          ) const;
    inline virtual double &inner3Real(double &res, const Vector<T> &b, const Vector<T> &c                    ) const;
    inline virtual double &inner4Real(double &res, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d) const;

    inline virtual double norm1(void)     const;
    inline virtual double norm2(void)     const;
    inline virtual double normp(double p) const;

private:

    MercerKernel spKern;
    Vector<SparseVector<gentype> > xx;
    Vector<T> alpha;
};

template <class T>
void qswap(RKHSVector<T> &a, RKHSVector<T> &b)
{
    qswap(a.spKern,b.spKern);
    qswap(a.xx    ,b.xx    );
    qswap(a.alpha ,b.alpha );

    return;
}

template <class T>
void qswap(const RKHSVector<T> *&a, const RKHSVector<T> *&b)
{
    const RKHSVector<T> *c;

    c = a;
    a = b;
    b = c;

    return;
}

template <class T>
void qswap(RKHSVector<T> *&a, RKHSVector<T> *&b)
{
    RKHSVector<T> *c;

    c = a;
    a = b;
    b = c;

    return;
}

// Get RKHS part of vector

template <class T>
const RKHSVector<T> &getRKHSpart(const Vector<T> &src);
template <class T>
const RKHSVector<T> &getRKHSpart(const Vector<T> &src)
{
    NiceAssert( src.infsize() );

    if ( src.imoverhere )
    {
        return dynamic_cast<const RKHSVector<T> &>(*(src.imoverhere));
    }

    return dynamic_cast<const RKHSVector<T> &>(src);
}

// resizeN changes this.  If decreasing size it will sparsify by removing the smallest alpha's (by 2-norm) first.

template <class T>
void RKHSVector<T>::resizeN(int N)
{
    xx.resize(N);
    alpha.resize(N);

    return;
}

// Inner-product functions for RKHS
//
// conj = 0: noConj
//        1: normal
//        2: revConj

template <class T>
T &RKHSVector<T>::inner1(T &res) const
{
    int i;

    setzero(res);

    T tmpa;
    T tmpb;
    T tmpc;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        tmpa = a(i);
        tmpa *= spKern.K1(tmpb,x(i),iinfo,tmpc);

        res += tmpa;
    }

    return res;
}

template <class T>
T &RKHSVector<T>::inner2(T &res, const Vector<T> &bb, int conj) const
{
    int i,j;

    const RKHSVector<T> &b = getRKHSpart(bb);

    NiceAssert( kern() == b.kern() );

    setzero(res);

    T tmpa; 
    T tmpb; 
    T tmpc;
    T tmpd;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            tmpa = a(i);
            tmpd = b.a(j);

            if ( conj & 1 )
            {
                setconj(tmpa);
            }

            if ( conj & 2 )
            {
                setconj(tmpd);
            }

            tmpa *= spKern.K2(tmpb,x(i),b.x(j),iinfo,jinfo,tmpc);
            tmpa *= tmpd;

            res += tmpa;
        }
    }

    return res;
}

template <class T>
T &RKHSVector<T>::inner3(T &res, const Vector<T> &bb, const Vector<T> &cc) const
{
    int i,j,k;

    const RKHSVector<T> &b = getRKHSpart(bb);
    const RKHSVector<T> &c = getRKHSpart(cc);

    NiceAssert( kern() == b.kern() );
    NiceAssert( kern() == c.kern() );

    setzero(res);

    T tmpa; 
    T tmpb; 
    T tmpc;
    T tmpd;
    T tmpe;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);
    setzero(tmpe);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            for ( k = 0 ; k < c.N() ; k++ )
            {
                vecInfo kinfo; spKern.getvecInfo(kinfo,c.x(k));

                tmpa = a(i);
                tmpd = b.a(j);
                tmpe = c.a(k);

                tmpa *= spKern.K3(tmpb,x(i),b.x(j),c.x(k),iinfo,jinfo,kinfo,tmpc);
                tmpa *= tmpd;
                tmpa *= tmpe;

                res += tmpa;
            }
        }
    }

    return res;
}

template <class T>
T &RKHSVector<T>::inner4(T &res, const Vector<T> &bb, const Vector<T> &cc, const Vector<T> &dd) const
{
    int i,j,k,l;

    const RKHSVector<T> &b = getRKHSpart(bb);
    const RKHSVector<T> &c = getRKHSpart(cc);
    const RKHSVector<T> &d = getRKHSpart(dd);

    NiceAssert( kern() == b.kern() );
    NiceAssert( kern() == c.kern() );
    NiceAssert( kern() == d.kern() );

    setzero(res);

    T tmpa; 
    T tmpb; 
    T tmpc;
    T tmpd;
    T tmpe;
    T tmpf;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);
    setzero(tmpe);
    setzero(tmpf);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            for ( k = 0 ; k < c.N() ; k++ )
            {
                vecInfo kinfo; spKern.getvecInfo(kinfo,c.x(k));

                for ( l = 0 ; l < d.N() ; l++ )
                {
                    vecInfo linfo; spKern.getvecInfo(linfo,d.x(l));

                    tmpa = a(i);
                    tmpd = b.a(j);
                    tmpe = c.a(k);
                    tmpf = d.a(k);

                    tmpa *= spKern.K4(tmpb,x(i),b.x(j),c.x(k),d.x(l),iinfo,jinfo,kinfo,linfo,tmpc);
                    tmpa *= tmpd;
                    tmpa *= tmpe;
                    tmpa *= tmpf;

                    res += tmpa;
                }
            }
        }
    }

    return res;
}

template <class T>
double &RKHSVector<T>::inner1Real(double &res) const
{
    int i;

    setzero(res);

    double tmpa;
    double tmpb;
    double tmpc;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        tmpa = (double) a(i);
        tmpa *= spKern.K1(tmpb,x(i),iinfo,tmpc);

        res += tmpa;
    }

    return res;
}

template <class T>
double &RKHSVector<T>::inner2Real(double &res, const Vector<T> &bb, int conj) const
{
    (void) conj;

    int i,j;

    const RKHSVector<T> &b = getRKHSpart(bb);

    NiceAssert( kern() == b.kern() );

    setzero(res);

    double tmpa; 
    double tmpb; 
    double tmpc;
    double tmpd;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            tmpa = (double) a(i);
            tmpd = (double) b.a(j);

            tmpa *= spKern.K2(tmpb,x(i),b.x(j),iinfo,jinfo,tmpc);
            tmpa *= tmpd;

            res += tmpa;
        }
    }

    return res;
}

template <class T>
double &RKHSVector<T>::inner3Real(double &res, const Vector<T> &bb, const Vector<T> &cc) const
{
    int i,j,k;

    const RKHSVector<T> &b = getRKHSpart(bb);
    const RKHSVector<T> &c = getRKHSpart(cc);

    NiceAssert( kern() == b.kern() );
    NiceAssert( kern() == c.kern() );

    setzero(res);

    double tmpa; 
    double tmpb; 
    double tmpc;
    double tmpd;
    double tmpe;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);
    setzero(tmpe);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            for ( k = 0 ; k < c.N() ; k++ )
            {
                vecInfo kinfo; spKern.getvecInfo(kinfo,c.x(k));

                tmpa = (double) a(i);
                tmpd = (double) b.a(j);
                tmpe = (double) c.a(k);

                tmpa *= spKern.K3(tmpb,x(i),b.x(j),c.x(k),iinfo,jinfo,kinfo,tmpc);
                tmpa *= tmpd;
                tmpa *= tmpe;

                res += tmpa;
            }
        }
    }

    return res;
}

template <class T>
double &RKHSVector<T>::inner4Real(double &res, const Vector<T> &bb, const Vector<T> &cc, const Vector<T> &dd) const
{
    int i,j,k,l;

    const RKHSVector<T> &b = getRKHSpart(bb);
    const RKHSVector<T> &c = getRKHSpart(cc);
    const RKHSVector<T> &d = getRKHSpart(dd);

    NiceAssert( kern() == b.kern() );
    NiceAssert( kern() == c.kern() );
    NiceAssert( kern() == d.kern() );

    setzero(res);

    double tmpa; 
    double tmpb; 
    double tmpc;
    double tmpd;
    double tmpe;
    double tmpf;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);
    setzero(tmpe);
    setzero(tmpf);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            for ( k = 0 ; k < c.N() ; k++ )
            {
                vecInfo kinfo; spKern.getvecInfo(kinfo,c.x(k));

                for ( l = 0 ; l < d.N() ; l++ )
                {
                    vecInfo linfo; spKern.getvecInfo(linfo,d.x(l));

                    tmpa = (double) a(i);
                    tmpd = (double) b.a(j);
                    tmpe = (double) c.a(k);
                    tmpf = (double) d.a(k);

                    tmpa *= spKern.K4(tmpb,x(i),b.x(j),c.x(k),d.x(l),iinfo,jinfo,kinfo,linfo,tmpc);
                    tmpa *= tmpd;
                    tmpa *= tmpe;
                    tmpa *= tmpf;

                    res += tmpa;
                }
            }
        }
    }

    return res;
}

template <class T>
double RKHSVector<T>::norm1(void) const
{
    double res;

    return inner1Real(res);
}

template <class T>
double RKHSVector<T>::norm2(void) const
{
    double res;
    return inner2Real(res,*this);
}

template <class T>
double RKHSVector<T>::normp(double p) const
{
    int q = (int) p;

    NiceAssert( p == q );
    NiceAssert( q > 0 );

    double res;

         if ( q == 1 ) { inner1Real(res); }
    else if ( q == 2 ) { inner2Real(res,*this); }
    else if ( q == 3 ) { inner3Real(res,*this,*this); }
    else if ( q == 4 ) { inner4Real(res,*this,*this,*this); }
    else { throw("I am not a turnip!"); }

    return res;
}

template <class T>
T &RKHSVector<T>::operator()(T &res, const SparseVector<gentype> &xx) const
{
    setzero(res);

    if ( N() )
    {
        int i;
        T tmp,zerobias;
        vecInfo xinfo;
        vecInfo xxinfo;

        spKern.getvecInfo(xxinfo,xx);

        setzero(tmp);
        setzero(zerobias);

        for ( i = 0 ; i < N() ; i++ )
        {
            spKern.getvecInfo(xinfo,x(i));
            spKern.K2(tmp,x(i),xx,xinfo,xxinfo,zerobias);

            rightmult(a(i),tmp);

            res += tmp;
        }
    }

    return res;
}

template <class T> 
std::ostream &operator<<(std::ostream &output, const RKHSVector<T> &src)
{
    return src.outstream(output);
}

template <class T> 
std::istream &operator>>(std::istream &input, RKHSVector<T> &dest)
{
    return dest.instream(input);
}

template <class T> 
std::istream &streamItIn(std::istream &input, RKHSVector<T> &dest, int processxyzvw)
{
    return dest.streamItIn(input,processxyzvw);
}


template <class T>
std::ostream &RKHSVector<T>::outstream(std::ostream &output) const
{
    output << "[[ RKHS kernel: " << spKern << "\n";
    output << "   RKHS x:      " << xx     << "\n";
    output << "   RKHS a:      " << alpha  << " ]]\n";

    return output;
}

template <class T>
std::istream &RKHSVector<T>::instream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> spKern;
    input >> dummy; input >> xx;
    input >> dummy; input >> alpha;

    return input;
}

template <class T>
std::istream &RKHSVector<T>::streamItIn(std::istream &input, int processxyzvw)
{
    (void) processxyzvw;

    return instream(input);
}


template <> inline void makeRKHSVector(Vector<double> *&res, std::istream &src)
{
    RKHSVector<double> *altres;

    MEMNEW(altres,RKHSVector<double>());

    src >> *altres;

    res = altres;

    return;
}

template <> inline void makeRKHSVector(Vector<double> *&res, std::istream &src, int processxyzvw)
{
    RKHSVector<double> *altres;

    MEMNEW(altres,RKHSVector<double>());

    streamItIn(src,*altres,processxyzvw);

    res = altres;

    return;
}

template <> inline void makeRKHSVector(Vector<gentype> *&res, std::istream &src)
{
    RKHSVector<gentype> *altres;

    MEMNEW(altres,RKHSVector<gentype>());

    src >> *altres;

    res = altres;

    return;
}

template <> inline void makeRKHSVector(Vector<gentype> *&res, std::istream &src, int processxyzvw)
{
    RKHSVector<gentype> *altres;

    MEMNEW(altres,RKHSVector<gentype>());

    streamItIn(src,*altres,processxyzvw);

    res = altres;

    return;
}

// Calculate L2 distance from RKHSVector to function of given dimension,
// assuming a function of var(0,0), var(0,1), ..., var(0,dim-1)
//
// It is assume functions are over [0,1]^dim with gran steps per dimension

inline double calcL2distsq(const Vector<gentype> &ff, gentype &g, int dim, int scaleit, int gran)
{
errstream() << "dim = " << dim << "\n";
    NiceAssert( dim  >= 0 );
    NiceAssert( gran >= 1 );
    NiceAssert( ff.infsize() );

    retVector<gentype> tmp;

    const RKHSVector<gentype> &f = dynamic_cast<const RKHSVector<gentype> &>(ff(tmp));

    Vector<int> i(dim);
    SparseVector<SparseVector<gentype> > xx;
    SparseVector<gentype> &x = xx("&",0);
    gentype fv,gv;
    double res = 0.0;
    double unitsize = sqrt( scaleit ? pow(1.0/((double) gran),dim) : 1.0 );
    int done = dim ? 0 : 1;
    int j;

    i = zeroint();

    while ( !done )
    {
        for ( j = 0 ; j < dim ; j++ )
        {
            x("&",j) = ((double) i(j))/((double) gran);
        }

        f(fv,x).finalise();
        gv = g(xx); gv.finalise();

        fv -= gv;
        fv *= unitsize;

        res += (double) norm2(fv);

        done = 1;

        for ( j = 0 ; done && ( j < dim ) ; j++ )
        {
            i("&",j)++;

            done     = i(j)/gran;
            i("&",j) = i(j)%gran;
        }
    }

    return res;
}



#endif


