//FOR [ x ~ xb ~ ... ] in FNVector, need to move the "up" part of the expansion K2 in ml_base.cc to mercer.h top-level K2 evaluation, including UPNTVI code (remember to add iupm, jupm == 1 test to 
//xymatrix shortcut generator code)



//
// Functional and RKHS Vector class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _FNVector_h
#define _FNVector_h

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
#include "mlcommon.h"


class FuncVector;
class RKHSVector;
class BernVector;


// These classes represent L2 functions on [0,1]^d.  
//
// FuncVector: - the function is defined directly as a function, and the 
//               dimension given.
//             - Format is [[ FN f: fn : d ]], where fn is the function and d the
//               dimension.
//             - Inner products etc are defined by approximating on a grid 
//               over this set.  
//             - If the function evaluates as NULL at some point then this 
//               point is not included in the sum (etc), and the average is 
//               adjusted not to include this gridpoint.
//             - Lp inner products and norms also defined.
//
// RKHSVector: - the function is defined as sum_i alpha_i K(x,x_i)
//             - K, alpha_i and x_i therefore define the function.
//             - inner products etc are RKHS inner products
//               (sum_ij alpha_i alpha_j' K(x_i,x_j')
//             - Kernels must match for inner products etc!
//             - Lp inner products etc defined via m-kernels
//
//             - if m != 2 then this is an m-RKHS, so:
//               f(x) = sum_{i2,i3,...,im} alpha_i2 alpha_i3 ... alpha_im Km(x_i2,x_i3,...,x_im,x)
//               or, more generally, using the ~ feature in sparsevectors
//               f(x1 ~ x2 ~ ... ~ xn) = sum_{i2,i3,...,im} alpha_i2 alpha_i3 ... alpha_im K{m+n-1}(x_i2,x_i3,...,x_im,x1,x1,...,xn)
//               (unless the kernel is defined to use this feature for something else)
//             - The p-inner product is:
//               <v1,v1,...,vp>_p = sum_{i12,i13,...,i1m1,i22,i23,...,i2m2, ..., ip2,ip3,...,ipmp} alpha1_i12 alpha1_i23 ... alpha1_i1m1 alpha2_i22 alpha2_i23 ... alpha2_i2m2 ...
//                                                  ... alphap_ip2 alphap_ip3 ... alphap_ipmp Km(x1_i12,x1_i13,...,x1_i1m1,x2_i22,x2_i23,...,x2_i2m2,...,xp_ip2,xp_ip3,...,xp_ipmp)
//               where m = m1+m2+...+mp
//             - and the induced p-norm becomes:
//               ||v||_p^p = <v,v,...,v>_p
//
//             - use the alpha_as_vector feature for different weights:
//               f(x) = sum_{i2,i3,...,im} alpha_i2[0] alpha_i3[1] ... alpha_im[m-2] Km(x_i2,x_i3,...,x_im,x)
//
// BernVector: - the function is defined by weights wrt Bernstein basis
//             - the weight is w
//
// Notes:
//
// - The m-product between FuncVector and RKHSVector reverts to FuncVector
// - sums of vectors are defined, but...
//   - the sum of RKHSVectors are only defined if kernels match
//   - the sum of RKHSVector and FuncVector reverts to FuncVector



// Stream operators

std::ostream &operator<<(std::ostream &output, const FuncVector &src );
std::istream &operator>>(std::istream &input,        FuncVector &dest);
std::istream &streamItIn(std::istream &input,        FuncVector &dest, int processxyzvw = 1);

std::ostream &operator<<(std::ostream &output, const RKHSVector &src );
std::istream &operator>>(std::istream &input,        RKHSVector &dest);
std::istream &streamItIn(std::istream &input,        RKHSVector &dest, int processxyzvw = 1);

std::ostream &operator<<(std::ostream &output, const BernVector &src );
std::istream &operator>>(std::istream &input,        BernVector &dest);
std::istream &streamItIn(std::istream &input,        BernVector &dest, int processxyzvw = 1);

// Swap function

inline void qswap(const FuncVector *&a, const FuncVector *&b);
inline void qswap(FuncVector       *&a, FuncVector       *&b);
inline void qswap(FuncVector        &a, FuncVector        &b);

inline void qswap(const RKHSVector *&a, const RKHSVector *&b);
inline void qswap(RKHSVector       *&a, RKHSVector       *&b);
inline void qswap(RKHSVector        &a, RKHSVector        &b);

inline void qswap(const BernVector *&a, const BernVector *&b);
inline void qswap(BernVector       *&a, BernVector       *&b);
inline void qswap(BernVector        &a, BernVector        &b);

// Creation operators

void makeFuncVector(const std::string &typestring, Vector<gentype> *&res, std::istream &src);
void makeFuncVector(const std::string &typestring, Vector<gentype> *&res, std::istream &src, int processxyzvw);

// Calculate L2 distance squared from RKHSVector to function of given dimension,
// assuming a function of var(0,0), var(0,1), ..., var(0,dim-1)
//
// It is assume functions are over [0,1]^dim with gran steps per dimension
//
// scaleit 1 means L2 norm, scaleit2 means L2 norm * granularity
//
// dim = -1 means use dimension (for FuncVector)

double calcL2distsq(const Vector<gentype> &f, gentype &g, int dim, int scaleit = 1, int gran = DEFAULT_SAMPLES_SAMPLE);
double calcL2distsq(const gentype &f, gentype &g, int dim, int scaleit = 1, int gran = DEFAULT_SAMPLES_SAMPLE);


// This represents sum_i a_i a_i K(x,x_i)
//
// T can only be double or gentype, nothing else

template <> inline int aresame<gentype,gentype>(gentype *, gentype *);
template <> inline int aresame<gentype,gentype>(gentype *, gentype *) { return 1; }





// The class itself

class FuncVector : public Vector<gentype>
{
    friend class RKHSVector;
    friend class BernVector;
    friend void qswap(FuncVector &a, FuncVector &b);

public:

    // Constructors and Destructors

    FuncVector() : Vector<gentype>()  { thisthis = this; thisthisthis = &thisthis; fdim = 1; valfn = 0; }
    FuncVector(const FuncVector &src) : Vector<gentype>() { thisthis = this; thisthisthis = &thisthis; fdim = 1; valfn = 0; assign(src); } 
    virtual ~FuncVector() { return; }

    // Print and make duplicate

    virtual std::ostream &outstream(std::ostream &output) const;
    virtual std::istream &instream (std::istream &input );

    virtual std::istream &streamItIn(std::istream &input, int processxyzvw = 1);

    virtual Vector<gentype> *makeDup(void) const
    {
        FuncVector *dup;

        MEMNEW(dup,FuncVector(*this));

        return static_cast<Vector<gentype> *>(dup);
    }

    // Assignment

    FuncVector &operator=(const FuncVector &src) { return assign(src); }
    FuncVector &operator=(const gentype &src)    { return assign(src); }

    virtual FuncVector &assign(const FuncVector &src);
    virtual FuncVector &assign(const gentype &src);

    // Simple vector manipulations

    virtual Vector<gentype> &softzero(void) { unsample(); if ( extrapart.size() ) { int i; for ( i = 0 ; i < NE() ; i++ ) { MEMDEL(extrapart("&",i)); extrapart("&",i) = NULL; } extrapart.resize(0); } valfn.zero();   return *this; }
    virtual Vector<gentype> &zero(void)     { unsample(); if ( extrapart.size() ) { int i; for ( i = 0 ; i < NE() ; i++ ) { MEMDEL(extrapart("&",i)); extrapart("&",i) = NULL; } extrapart.resize(0); } valfn.zero();   return *this; }
    virtual Vector<gentype> &posate(void)   { unsample(); if ( extrapart.size() ) { int i; for ( i = 0 ; i < NE() ; i++ ) { (*extrapart("&",i)).posate(); } }                                           valfn.posate(); return *this; }
    virtual Vector<gentype> &negate(void)   { unsample(); if ( extrapart.size() ) { int i; for ( i = 0 ; i < NE() ; i++ ) { (*extrapart("&",i)).negate(); } }                                           valfn.negate(); return *this; }
    virtual Vector<gentype> &conj(void)     { unsample(); if ( extrapart.size() ) { int i; for ( i = 0 ; i < NE() ; i++ ) { (*extrapart("&",i)).conj();   } }                                           valfn.conj();   return *this; }
    virtual Vector<gentype> &rand(void)     { throw("Random functional vectors not implemented"); return *this; }

    // Access:
    //
    // - vector has the functional form f(x) = sum_{i=0}^{N-1} alpha_i K(x_i,x)
    // - to evaluate f(x) use operator()
    // - to access alpha_i use f.a(...)
    // - to access x_i use f.x(...)
    // - to access kernel use f.kern(...)

    virtual gentype &operator()(gentype &res, const              gentype  &i) const { SparseVector<gentype> ii; ii("&",zeroint()) = i; return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const       Vector<gentype> &i) const { SparseVector<gentype> ii(i);                     return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const SparseVector<gentype> &i) const;

    const gentype &f(void) const        { NiceAssert( !ismixed() );                           return valfn; }
          gentype &f(const char *dummy) { NiceAssert( !ismixed() ); unsample(); (void) dummy; return valfn; }

    // Short-cut access:
    //
    // By calling sample, you can pre-generate a 1-d grid-evaluated version of the
    // vector for fast access.  This can then be accessed by the following:

    const Vector<gentype> &operator()(                        retVector<gentype> &tmp) const { if ( !samplesize() ) { unsafesample(); } return precalcVec(tmp);          }
    const gentype         &operator()(int i                                          ) const { if ( !samplesize() ) { unsafesample(); } return precalcVec(i);            }
    const Vector<gentype> &operator()(const Vector<int> &i,   retVector<gentype> &tmp) const { if ( !samplesize() ) { unsafesample(); } return precalcVec(i,tmp);        }
    const Vector<gentype> &operator()(int ib, int is, int im, retVector<gentype> &tmp) const { if ( !samplesize() ) { unsafesample(); } return precalcVec(ib,is,im,tmp); }

    // Information functions

    virtual int type(void)    const { return 1;    }
    virtual int infsize(void) const { return 1;    }
    virtual int ismixed(void) const { return NE(); }

    virtual int testsametype(std::string &typestring) { return typestring == "FN"; }

    // Function application - apply function fn to each element of vector.

    virtual Vector<gentype> &applyon(gentype (*fn)(gentype))                                      { NiceAssert( !ismixed() ); unsample(); valfn = (*fn)(valfn);   return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &))                              { NiceAssert( !ismixed() ); unsample(); valfn = (*fn)(valfn);   return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(gentype, const void *), const void *a)         { NiceAssert( !ismixed() ); unsample(); valfn = (*fn)(valfn,a); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &, const void *), const void *a) { NiceAssert( !ismixed() ); unsample(); valfn = (*fn)(valfn,a); return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &))                                   { NiceAssert( !ismixed() ); unsample();         (*fn)(valfn);   return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &, const void *), const void *a)      { NiceAssert( !ismixed() ); unsample();         (*fn)(valfn,a); return *this; }

    // Pre-allocation control.

    virtual void prealloc(int newallocsize)  { (void) newallocsize; return; }
    virtual void useStandardAllocation(void) {                      return; }
    virtual void useTightAllocation(void)    {                      return; }
    virtual void useSlackAllocation(void)    {                      return; }

    virtual int array_norm (void) const { return 1; }
    virtual int array_tight(void) const { return 0; }
    virtual int array_slack(void) const { return 0; }





    // New stuff specific stuff

    virtual void sample(int Nsamp = DEFAULT_SAMPLES_SAMPLE);
    virtual void unsample(void) { precalcVec.resize(0); return; }
    virtual int samplesize(void) const { return precalcVec.size(); }

    int dim(void) const { return fdim; }
    void setdim(int nv) { fdim = nv; if ( NE() ) { int i; for ( i = 0 ; i < NE() ; i++ ) { (*extrapart("&",i)).setdim(nv); } } return; }

    int NE(void) const { return extrapart.size(); }





    // Inner-product functions for infsize vectors
    //
    // conj = 0: noConj
    //        1: normal
    //        2: revConj

    virtual gentype &inner1(gentype &res                                                                              ) const;
    virtual gentype &inner2(gentype &res, const Vector<gentype> &b, int conj = 1                                      ) const;
    virtual gentype &inner3(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c                          ) const;
    virtual gentype &inner4(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d) const;
    virtual gentype &innerp(gentype &res, const Vector<const Vector<gentype> *> &b                                    ) const;

    virtual double &inner1Real(double &res                                                                              ) const;
    virtual double &inner2Real(double &res, const Vector<gentype> &b, int conj = 1                                      ) const;
    virtual double &inner3Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c                          ) const;
    virtual double &inner4Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d) const;
    virtual double &innerpReal(double &res, const Vector<const Vector<gentype> *> &b                                    ) const;

    virtual double norm1(void)     const { double res; return inner1Real(res);       }
    virtual double norm2(void)     const { double res; return inner2Real(res,*this); }
    virtual double normp(double p) const { NiceAssert( ( (int) p ) == p ); NiceAssert( p > 0 ); double res; Vector<const Vector<gentype> *> b(((int) p)-1); b = this; return innerpReal(res,b); }

    virtual double absinf(void) const;

    // Only the rudinemtary operators are defined: +=, -=, *=, /=, == (and by inference +,-,*,/)

    virtual Vector<gentype> &subit (const Vector<gentype> &b);
    virtual Vector<gentype> &addit (const Vector<gentype> &b);
    virtual Vector<gentype> &subit (const gentype         &b);
    virtual Vector<gentype> &addit (const gentype         &b);
    virtual Vector<gentype> &mulit (const Vector<gentype> &b);
    virtual Vector<gentype> &rmulit(const Vector<gentype> &b);
    virtual Vector<gentype> &divit (const Vector<gentype> &b);
    virtual Vector<gentype> &rdivit(const Vector<gentype> &b);
    virtual Vector<gentype> &mulit (const gentype         &b); // this*b
    virtual Vector<gentype> &rmulit(const gentype         &b); // b*this
    virtual Vector<gentype> &divit (const gentype         &b); // this/b
    virtual Vector<gentype> &rdivit(const gentype         &b); // b\this

    virtual int iseq(const Vector<gentype> &b) { (void) b; throw("I don't know"); return 0; }
    virtual int iseq(const gentype         &b) { (void) b; throw("I really don't know"); return 0; }

    // A lazy cheat for sampling constant vectors - try not to use this if possible.

    virtual void unsafesample(int Nsamp = DEFAULT_SAMPLES_SAMPLE) const
    {
        (**thisthisthis).sample(Nsamp);

        return;
    }




private:

    Vector<FuncVector *> extrapart; // NULL except when you use FuncVector-RKHSVector, which results in a FuncVector with this non-null
    gentype valfn;
    int fdim;
    Vector<gentype> precalcVec;

    FuncVector *thisthis;
    FuncVector **thisthisthis;
};

inline void qswap(FuncVector &a, FuncVector &b)
{
    // DON"T WANT THIS! qswap(static_cast<Vector<gentype> &>(a),static_cast<Vector<gentype> &>(b));

    qswap(a.valfn     ,b.valfn     );
    qswap(a.fdim      ,b.fdim      );
    qswap(a.precalcVec,b.precalcVec);
    qswap(a.extrapart ,b.extrapart );

    return;
}

inline void qswap(const FuncVector *&a, const FuncVector *&b)
{
    const FuncVector *c;

    c = a;
    a = b;
    b = c;

    return;
}

inline void qswap(FuncVector *&a, FuncVector *&b)
{
    FuncVector *c;

    c = a;
    a = b;
    b = c;

    return;
}





















Vector<gentype> &makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const SparseVector<gentype> &x, const gentype &a);
Vector<gentype> &makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const SparseVector<gentype> &x, const gentype &a, int m);

class RKHSVector : public FuncVector
{
    friend void qswap(RKHSVector &a, RKHSVector &b);

public:

    // Constructors and Destructors

    RKHSVector() : FuncVector()  { thisthis = this; thisthisthis = &thisthis; mm = 2; alphaasvector = 0; revertToFunc = 0; } 
    RKHSVector(const RKHSVector &src) : FuncVector(src) { thisthis = this; thisthisthis = &thisthis; mm = 2; alphaasvector = 0; revertToFunc = 0; assign(src); } 
    virtual ~RKHSVector() { return; }

    // Print and make duplicate

    virtual std::ostream &outstream(std::ostream &output) const;
    virtual std::istream &instream (std::istream &input );

    virtual std::istream &streamItIn(std::istream &input, int processxyzvw = 1);

    virtual Vector<gentype> *makeDup(void) const
    {
        RKHSVector *dup;

        MEMNEW(dup,RKHSVector(*this));

        return static_cast<Vector<gentype> *>(dup);
    }

    // Assignment

    RKHSVector &operator=(const RKHSVector &src) { return assign(src); }
    RKHSVector &operator=(const gentype &src)    { return assign(src); }

    virtual RKHSVector &assign(const RKHSVector &src) 
    { 
        FuncVector::assign(static_cast<const FuncVector &>(src));

        spKern        = src.spKern; 
        alpha         = src.alpha; 
        xx            = src.xx; 
        xxinfo        = src.xxinfo; 
        xxinfook      = src.xxinfook; 
        mm            = src.mm; 
        alphaasvector = src.alphaasvector; 

        revertToFunc = src.revertToFunc;

        return *this; 
    }

    virtual RKHSVector &assign(const gentype &src) 
    { 
        (void) src;

        throw("No");

        return *this; 
    }

    // Simple vector manipulations

    virtual Vector<gentype> &softzero(void) { unsample(); if ( revertToFunc ) { FuncVector::softzero(); } else { alpha.softzero(); } return *this; }
    virtual Vector<gentype> &zero(void)     { unsample(); if ( revertToFunc ) { FuncVector::zero();     } else { alpha.zero();     } return *this; }
    virtual Vector<gentype> &posate(void)   { unsample(); if ( revertToFunc ) { FuncVector::posate();   } else { alpha.posate();   } return *this; }
    virtual Vector<gentype> &negate(void)   { unsample(); if ( revertToFunc ) { FuncVector::negate();   } else { alpha.negate();   } return *this; }
    virtual Vector<gentype> &conj(void)     { unsample(); if ( revertToFunc ) { FuncVector::conj();     } else { alpha.conj();     } return *this; }
    virtual Vector<gentype> &rand(void)     { unsample(); if ( revertToFunc ) { FuncVector::rand();     } else { alpha.rand();     } return *this; }

    // Access:
    //
    // - vector has the functional form f(x) = sum_{i=0}^{N-1} alpha_i K(x_i,x)
    // - to evaluate f(x) use operator()
    // - to access alpha_i use f.a(...)
    // - to access x_i use f.x(...)
    // - to access kernel use f.kern(...)
    // - to evaluate f(x1,x2,...) use [ x1 ~ x2 ~ ... ] format sparsevectors
    //
    // NB: don't change N by resizing these references!

    virtual gentype &operator()(gentype &res, const              gentype  &i) const { if ( revertToFunc ) { return FuncVector::operator()(res,i); } SparseVector<gentype> ii; ii("&",zeroint()) = i; return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const       Vector<gentype> &i) const { if ( revertToFunc ) { return FuncVector::operator()(res,i); } SparseVector<gentype> ii(i);                     return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const SparseVector<gentype> &i) const;

    const gentype &f(void) const        { if ( revertToFunc ) { return FuncVector::f();      } throw("Can't use f() on RKHSVector"); static gentype rdummy; return rdummy; }
          gentype &f(const char *dummy) { if ( revertToFunc ) { return FuncVector::f(dummy); } throw("Can't use f() on RKHSVector"); static gentype rdummy; return rdummy; }

    Vector<gentype> &a(const char *dummy,                         retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return alpha(dummy,tmp);          }
    gentype         &a(const char *dummy, int i                                          ) { unsample(); NiceAssert( !revertToFunc ); return alpha(dummy,i);            }
    Vector<gentype> &a(const char *dummy, const Vector<int> &i,   retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return alpha(dummy,i,tmp);        }
    Vector<gentype> &a(const char *dummy, int ib, int is, int im, retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return alpha(dummy,ib,is,im,tmp); }

    const Vector<gentype> &a(                        retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return alpha(tmp);          }
    const gentype         &a(int i                                          ) const { NiceAssert( !revertToFunc ); return alpha(i);            }
    const Vector<gentype> &a(const Vector<int> &i,   retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return alpha(i,tmp);        }
    const Vector<gentype> &a(int ib, int is, int im, retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return alpha(ib,is,im,tmp); }

    Vector<SparseVector<gentype> > &x(const char *dummy,                         retVector<SparseVector<gentype> > &tmp) { unsample(); NiceAssert( !revertToFunc ); retVector<int> tmpvb; xxinfook(dummy,tmpvb)          = zeroint(); return xx(dummy,tmp);          }
    SparseVector<gentype>          &x(const char *dummy, int i                                                         ) { unsample(); NiceAssert( !revertToFunc );                       xxinfook(dummy,i)              = zeroint(); return xx(dummy,i);            }
    Vector<SparseVector<gentype> > &x(const char *dummy, const Vector<int> &i,   retVector<SparseVector<gentype> > &tmp) { unsample(); NiceAssert( !revertToFunc ); retVector<int> tmpvb; xxinfook(dummy,i,tmpvb)        = zeroint(); return xx(dummy,i,tmp);        }
    Vector<SparseVector<gentype> > &x(const char *dummy, int ib, int is, int im, retVector<SparseVector<gentype> > &tmp) { unsample(); NiceAssert( !revertToFunc ); retVector<int> tmpvb; xxinfook(dummy,ib,is,im,tmpvb) = zeroint(); return xx(dummy,ib,is,im,tmp); }

    const Vector<SparseVector<gentype> > &x(                        retVector<SparseVector<gentype> > &tmp) const { NiceAssert( !revertToFunc ); return xx(tmp);          }
    const SparseVector<gentype>          &x(int i                                                         ) const { NiceAssert( !revertToFunc ); return xx(i);            }
    const Vector<SparseVector<gentype> > &x(const Vector<int> &i,   retVector<SparseVector<gentype> > &tmp) const { NiceAssert( !revertToFunc ); return xx(i,tmp);        }
    const Vector<SparseVector<gentype> > &x(int ib, int is, int im, retVector<SparseVector<gentype> > &tmp) const { NiceAssert( !revertToFunc ); return xx(ib,is,im,tmp); }

    const MercerKernel &kern(void) const        {             NiceAssert( !revertToFunc );               return spKern; }
          MercerKernel &kern(const char *dummy) { unsample(); NiceAssert( !revertToFunc ); (void) dummy; return spKern; }

    // Information functions

    virtual int type(void)    const { return 2;                    }
    virtual int infsize(void) const { return 1;                    }
    virtual int ismixed(void) const { return revertToFunc ? 1 : 0; }

    virtual int testsametype(std::string &typestring) { return typestring == "RKHS"; }

    // Function application - apply function fn to each element of vector.

    virtual Vector<gentype> &applyon(gentype (*fn)(gentype))                                      { if ( revertToFunc ) { return FuncVector::applyon(fn);   } throw("Can't apply function to RKHSVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &))                              { if ( revertToFunc ) { return FuncVector::applyon(fn);   } throw("Can't apply function to RKHSVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(gentype, const void *), const void *a)         { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } throw("Can't apply function to RKHSVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &, const void *), const void *a) { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } throw("Can't apply function to RKHSVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &))                                   { if ( revertToFunc ) { return FuncVector::applyon(fn);   } throw("Can't apply function to RKHSVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &, const void *), const void *a)      { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } throw("Can't apply function to RKHSVector"); return *this; }





    // RKHS specific stuff
    //
    // N is the basis size
    // resizeN changes this.  If decreasing size it will sparsify by removing the smallest alpha's (by 2-norm) first.

    virtual int N(void) const { return alpha.size(); }
    virtual void resizeN(int N) { int oldN = N; xx.resize(N); xxinfo.resize(N); xxinfook.resize(N); alpha.resize(N); if ( N > oldN ) { retVector<int> tmpva; retVector<gentype> tmpvb; static gentype zv(0.0); xxinfook("&",oldN,1,N-1,tmpva) = zeroint(); alpha("&",oldN,1,N-1,tmpvb) = zv; } return; }

    virtual int m(void) const { return mm; }
    virtual void setm(int nm) { NiceAssert( nm >= 1 ); mm = nm; return; }

    virtual int treatalphaasvector(void) const { return alphaasvector; }
    virtual void settreatalphaasvector(int nv) { alphaasvector = nv; return; }

    const Vector<vecInfo> &xinfo(                        retVector<vecInfo> &tmp) const { NiceAssert( !revertToFunc ); makeinfo();          return xxinfo(tmp);          }
    const vecInfo         &xinfo(int i                                          ) const { NiceAssert( !revertToFunc ); makeinfo(i);         return xxinfo(i);            }
    const Vector<vecInfo> &xinfo(const Vector<int> &i,   retVector<vecInfo> &tmp) const { NiceAssert( !revertToFunc ); makeinfo(i);         return xxinfo(i,tmp);        }
    const Vector<vecInfo> &xinfo(int ib, int is, int im, retVector<vecInfo> &tmp) const { NiceAssert( !revertToFunc ); makeinfo(ib,is,im);  return xxinfo(ib,is,im,tmp); }






    // Inner-product functions for RKHS
    //
    // conj = 0: noConj
    //        1: normal
    //        2: revConj

    virtual gentype &inner1(gentype &res                                                                              ) const;
    virtual gentype &inner2(gentype &res, const Vector<gentype> &b, int conj = 1                                      ) const;
    virtual gentype &inner3(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c                          ) const;
    virtual gentype &inner4(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d) const;
    virtual gentype &innerp(gentype &res, const Vector<const Vector<gentype> *> &b                                    ) const;

    virtual double &inner1Real(double &res                                                                              ) const;
    virtual double &inner2Real(double &res, const Vector<gentype> &b, int conj = 1                                      ) const;
    virtual double &inner3Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c                          ) const;
    virtual double &inner4Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d) const;
    virtual double &innerpReal(double &res, const Vector<const Vector<gentype> *> &b                                    ) const;

    virtual double norm1(void)     const { double res; return inner1Real(res); }
    virtual double norm2(void)     const { double res; return inner2Real(res,*this); }
    virtual double normp(double p) const { NiceAssert( ( (int) p ) == p ); NiceAssert( p > 0 ); double res; Vector<const Vector<gentype> *> b(((int) p)-1); b = this; return innerpReal(res,b); }

    virtual double absinf(void) const 
    {
        throw("I don't know how to do that");

        return 0.0;
    }

    //subit and addit are not efficient as they just append and fix sign
    virtual Vector<gentype> &subit (const Vector<gentype> &b);
    virtual Vector<gentype> &addit (const Vector<gentype> &b);
    virtual Vector<gentype> &subit (const gentype         &b);
    virtual Vector<gentype> &addit (const gentype         &b);
    virtual Vector<gentype> &mulit (const Vector<gentype> &b) { (void) b; throw("I'm sorry Dave, I don't know how to do that");   return *this; }
    virtual Vector<gentype> &rmulit(const Vector<gentype> &b) { (void) b; throw("I'm sorry Darren, I don't know how to do that"); return *this; }
    virtual Vector<gentype> &divit (const Vector<gentype> &b) { (void) b; throw("I'm sorry Garian, I don't know how to do that"); return *this; }
    virtual Vector<gentype> &rdivit(const Vector<gentype> &b) { (void) b; throw("I'm sorry Fred, I don't know how to do that");   return *this; }
    virtual Vector<gentype> &mulit (const gentype         &b);
    virtual Vector<gentype> &rmulit(const gentype         &b);
    virtual Vector<gentype> &divit (const gentype         &b);
    virtual Vector<gentype> &rdivit(const gentype         &b);

    virtual int iseq(const Vector<gentype> &b) { (void) b; throw("I still don't know"); return 0; }
    virtual int iseq(const gentype         &b) { (void) b; throw("No seriously, I just don't know"); return 0; }







private:

    MercerKernel spKern;
    Vector<SparseVector<gentype> > xx;
    Vector<vecInfo> xxinfo;
    Vector<int> xxinfook;
    Vector<gentype> alpha;

    int alphaasvector;
    int mm; // order of m-kernel RKHS vector
    int revertToFunc; // if 1 then call back to FuncVector (required for -=,+=)

    const gentype &al(int i, int j) const
    {
        return alphaasvector ? alpha(i)(j) : alpha(i);
    }

    void makeinfo(int i) const
    {
        if ( !((**thisthisthis).xxinfook(i)) )
        {
            (**thisthisthis).xxinfook("&",i) = 1;

            spKern.getvecInfo((**thisthisthis).xxinfo("&",i),xx(i));
        }

        return;
    }

    void makeinfo(const Vector<int> &i) const
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            makeinfo(i(ii));
        }

        return;
    }

    void makeinfo(int ib, int is, int im) const
    {
        int ii;

        if ( ib <= im )
        {
            for ( ii = ib ; ii <= im ; ii += is )
            {
                makeinfo(ii);
            }
        }

        return;
    }

    void makeinfo(void) const
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            makeinfo(ii);
        }

        return;
    }


    // Base versions with m factors already taken into account

    virtual gentype &baseinner1(gentype &res                                                                              , int aind                              ) const;
    virtual gentype &baseinner2(gentype &res, const Vector<gentype> &b, int conj                                          , int aind, int bind                    ) const;
    virtual gentype &baseinner3(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c                          , int aind, int bind, int cind          ) const;
    virtual gentype &baseinner4(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d, int aind, int bind, int cind, int dind) const;
    virtual gentype &baseinnerp(gentype &res, const Vector<const Vector<gentype> *> &b                                    , int aind, const Vector<int> &bind     ) const;

    virtual double &baseinner1Real(double &res                                                                              , int aind                              ) const;
    virtual double &baseinner2Real(double &res, const Vector<gentype> &b, int conj                                          , int aind, int bind                    ) const;
    virtual double &baseinner3Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c                          , int aind, int bind, int cind          ) const;
    virtual double &baseinner4Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d, int aind, int bind, int cind, int dind) const;
    virtual double &baseinnerpReal(double &res, const Vector<const Vector<gentype> *> &b                                    , int aind, const Vector<int> &bind     ) const;

    RKHSVector *thisthis;
    RKHSVector **thisthisthis;
};

inline void qswap(RKHSVector &a, RKHSVector &b)
{
    qswap(static_cast<FuncVector &>(a),static_cast<FuncVector &>(b));

    qswap(a.spKern       ,b.spKern       );
    qswap(a.xx           ,b.xx           );
    qswap(a.alpha        ,b.alpha        );
    qswap(a.mm           ,b.mm           );
    qswap(a.alphaasvector,b.alphaasvector);

    qswap(a.revertToFunc,b.revertToFunc);

    return;
}

inline void qswap(const RKHSVector *&a, const RKHSVector *&b)
{
    const RKHSVector *c;

    c = a;
    a = b;
    b = c;

    return;
}

inline void qswap(RKHSVector *&a, RKHSVector *&b)
{
    RKHSVector *c;

    c = a;
    a = b;
    b = c;

    return;
}

















class BernVector : public FuncVector
{
    friend void qswap(BernVector &a, BernVector &b);

public:

    // Constructors and Destructors

    BernVector() : FuncVector()  { revertToFunc = 0; } 
    BernVector(const BernVector &src) : FuncVector(src) { revertToFunc = 0; assign(src); } 
    virtual ~BernVector() { return; }

    // Print and make duplicate

    virtual std::ostream &outstream(std::ostream &output) const;
    virtual std::istream &instream (std::istream &input );

    virtual std::istream &streamItIn(std::istream &input, int processxyzvw = 1);

    virtual Vector<gentype> *makeDup(void) const
    {
        BernVector *dup;

        MEMNEW(dup,BernVector(*this));

        return static_cast<Vector<gentype> *>(dup);
    }

    // Assignment

    BernVector &operator=(const BernVector &src) { return assign(src); }
    BernVector &operator=(const gentype &src)    { return assign(src); }

    virtual BernVector &assign(const BernVector &src) 
    { 
        FuncVector::assign(static_cast<const FuncVector &>(src));

        ww = src.ww;

        revertToFunc = src.revertToFunc;

        return *this; 
    }

    virtual BernVector &assign(const gentype &src) 
    { 
        (void) src;

        throw("Really, no");

        return *this; 
    }

    // Simple vector manipulations

    virtual Vector<gentype> &softzero(void) { unsample(); if ( revertToFunc ) { FuncVector::softzero(); } else { ww.softzero(); } return *this; }
    virtual Vector<gentype> &zero(void)     { unsample(); if ( revertToFunc ) { FuncVector::zero();     } else { ww.zero();     } return *this; }
    virtual Vector<gentype> &posate(void)   { unsample(); if ( revertToFunc ) { FuncVector::posate();   } else { ww.posate();   } return *this; }
    virtual Vector<gentype> &negate(void)   { unsample(); if ( revertToFunc ) { FuncVector::negate();   } else { ww.negate();   } return *this; }
    virtual Vector<gentype> &conj(void)     { unsample(); if ( revertToFunc ) { FuncVector::conj();     } else { ww.conj();     } return *this; }
    virtual Vector<gentype> &rand(void)     { unsample(); if ( revertToFunc ) { FuncVector::rand();     } else { ww.rand();     } return *this; }

    // Access:
    //
    // - vector has the functional form f(x) = sum_{i=0}^{N-1} alpha_i K(x_i,x)
    // - to evaluate f(x) use operator()
    // - to access alpha_i use f.a(...)
    // - to access x_i use f.x(...)
    // - to access kernel use f.kern(...)

    virtual gentype &operator()(gentype &res, const              gentype  &i) const { if ( revertToFunc ) { return FuncVector::operator()(res,i); } SparseVector<gentype> ii; ii("&",zeroint()) = i; return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const       Vector<gentype> &i) const { if ( revertToFunc ) { return FuncVector::operator()(res,i); } SparseVector<gentype> ii(i);                     return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const SparseVector<gentype> &i) const;

    const gentype &f(void) const        { if ( revertToFunc ) { return FuncVector::f();      } throw("Can't use f() on BernVector"); static gentype rdummy; return rdummy; }
          gentype &f(const char *dummy) { if ( revertToFunc ) { return FuncVector::f(dummy); } throw("Can't use f() on BernVector"); static gentype rdummy; return rdummy; }

    Vector<gentype> &w(const char *dummy,                         retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return ww(dummy,tmp);          }
    gentype         &w(const char *dummy, int i                                          ) { unsample(); NiceAssert( !revertToFunc ); return ww(dummy,i);            }
    Vector<gentype> &w(const char *dummy, const Vector<int> &i,   retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return ww(dummy,i,tmp);        }
    Vector<gentype> &w(const char *dummy, int ib, int is, int im, retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return ww(dummy,ib,is,im,tmp); }

    const Vector<gentype> &w(                        retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return ww(tmp);          }
    const gentype         &w(int i                                          ) const { NiceAssert( !revertToFunc ); return ww(i);            }
    const Vector<gentype> &w(const Vector<int> &i,   retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return ww(i,tmp);        }
    const Vector<gentype> &w(int ib, int is, int im, retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return ww(ib,is,im,tmp); }

    // Information functions

    virtual int type(void)    const { return 3;                    }
    virtual int infsize(void) const { return 1;                    }
    virtual int ismixed(void) const { return revertToFunc ? 1 : 0; }

    virtual int testsametype(std::string &typestring) { return typestring == "Bern"; }

    // Function application - apply function fn to each element of vector.

    virtual Vector<gentype> &applyon(gentype (*fn)(gentype))                                      { if ( revertToFunc ) { return FuncVector::applyon(fn);   } throw("Can't apply function to BernVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &))                              { if ( revertToFunc ) { return FuncVector::applyon(fn);   } throw("Can't apply function to BernVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(gentype, const void *), const void *a)         { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } throw("Can't apply function to BernVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &, const void *), const void *a) { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } throw("Can't apply function to BernVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &))                                   { if ( revertToFunc ) { return FuncVector::applyon(fn);   } throw("Can't apply function to BernVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &, const void *), const void *a)      { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } throw("Can't apply function to BernVector"); return *this; }






    // Bernstein specific stuff
    //
    // Nw is the w size

    virtual int Nw(void) const { return ww.size()-1; }





    //subit and addit are not efficient as they just append and fix sign
    virtual Vector<gentype> &subit (const Vector<gentype> &b);
    virtual Vector<gentype> &addit (const Vector<gentype> &b);
    virtual Vector<gentype> &subit (const gentype         &b);
    virtual Vector<gentype> &addit (const gentype         &b);
    virtual Vector<gentype> &mulit (const Vector<gentype> &b) { (void) b; throw("I'm sorry Liam, I don't know how to do that");  return *this; }
    virtual Vector<gentype> &rmulit(const Vector<gentype> &b) { (void) b; throw("I'm sorry Ivy, I don't know how to do that");   return *this; }
    virtual Vector<gentype> &divit (const Vector<gentype> &b) { (void) b; throw("I'm sorry Cindy, I don't know how to do that"); return *this; }
    virtual Vector<gentype> &rdivit(const Vector<gentype> &b) { (void) b; throw("I'm sorry you, I don't know how to do that");   return *this; }
    virtual Vector<gentype> &mulit (const gentype         &b);
    virtual Vector<gentype> &rmulit(const gentype         &b);
    virtual Vector<gentype> &divit (const gentype         &b);
    virtual Vector<gentype> &rdivit(const gentype         &b);

    virtual int iseq(const Vector<gentype> &b) { (void) b; throw("Umm."); return 0; }
    virtual int iseq(const gentype         &b) { (void) b; throw("Nah."); return 0; }







private:

    Vector<gentype> ww;

    int revertToFunc; // if 1 then call back to FuncVector (required for -=,+=)
};

inline void qswap(BernVector &a, BernVector &b)
{
    qswap(static_cast<FuncVector &>(a),static_cast<FuncVector &>(b));

    qswap(a.ww,b.ww);

    qswap(a.revertToFunc,b.revertToFunc);

    return;
}

inline void qswap(const BernVector *&a, const BernVector *&b)
{
    const BernVector *c;

    c = a;
    a = b;
    b = c;

    return;
}

inline void qswap(BernVector *&a, BernVector *&b)
{
    BernVector *c;

    c = a;
    a = b;
    b = c;

    return;
}

















inline const FuncVector *&setident      (const FuncVector *&a);
inline const FuncVector *&setzero       (const FuncVector *&a);
inline const FuncVector *&setzeropassive(const FuncVector *&a);
inline const FuncVector *&setposate     (const FuncVector *&a);
inline const FuncVector *&setnegate     (const FuncVector *&a);
inline const FuncVector *&setconj       (const FuncVector *&a);
inline const FuncVector *&setrand       (const FuncVector *&a);

inline const FuncVector *&setident      (const FuncVector *&a) { return a = NULL; }
inline const FuncVector *&setzero       (const FuncVector *&a) { return a = NULL; }
inline const FuncVector *&setzeropassive(const FuncVector *&a) { return a = NULL; }
inline const FuncVector *&setposate     (const FuncVector *&a) { return a = NULL; }
inline const FuncVector *&setnegate     (const FuncVector *&a) { return a = NULL; }
inline const FuncVector *&setconj       (const FuncVector *&a) { return a = NULL; }
inline const FuncVector *&setrand       (const FuncVector *&a) { return a = NULL; }

inline const RKHSVector *&setident      (const RKHSVector *&a);
inline const RKHSVector *&setzero       (const RKHSVector *&a);
inline const RKHSVector *&setzeropassive(const RKHSVector *&a);
inline const RKHSVector *&setposate     (const RKHSVector *&a);
inline const RKHSVector *&setnegate     (const RKHSVector *&a);
inline const RKHSVector *&setconj       (const RKHSVector *&a);
inline const RKHSVector *&setrand       (const RKHSVector *&a);

inline const RKHSVector *&setident      (const RKHSVector *&a) { return a = NULL; }
inline const RKHSVector *&setzero       (const RKHSVector *&a) { return a = NULL; }
inline const RKHSVector *&setzeropassive(const RKHSVector *&a) { return a = NULL; }
inline const RKHSVector *&setposate     (const RKHSVector *&a) { return a = NULL; }
inline const RKHSVector *&setnegate     (const RKHSVector *&a) { return a = NULL; }
inline const RKHSVector *&setconj       (const RKHSVector *&a) { return a = NULL; }
inline const RKHSVector *&setrand       (const RKHSVector *&a) { return a = NULL; }

inline const BernVector *&setident      (const BernVector *&a);
inline const BernVector *&setzero       (const BernVector *&a);
inline const BernVector *&setzeropassive(const BernVector *&a);
inline const BernVector *&setposate     (const BernVector *&a);
inline const BernVector *&setnegate     (const BernVector *&a);
inline const BernVector *&setconj       (const BernVector *&a);
inline const BernVector *&setrand       (const BernVector *&a);

inline const BernVector *&setident      (const BernVector *&a) { return a = NULL; }
inline const BernVector *&setzero       (const BernVector *&a) { return a = NULL; }
inline const BernVector *&setzeropassive(const BernVector *&a) { return a = NULL; }
inline const BernVector *&setposate     (const BernVector *&a) { return a = NULL; }
inline const BernVector *&setnegate     (const BernVector *&a) { return a = NULL; }
inline const BernVector *&setconj       (const BernVector *&a) { return a = NULL; }
inline const BernVector *&setrand       (const BernVector *&a) { return a = NULL; }



inline FuncVector *&setident      (FuncVector *&a);
inline FuncVector *&setzero       (FuncVector *&a);
inline FuncVector *&setzeropassive(FuncVector *&a);
inline FuncVector *&setposate     (FuncVector *&a);
inline FuncVector *&setnegate     (FuncVector *&a);
inline FuncVector *&setconj       (FuncVector *&a);
inline FuncVector *&setrand       (FuncVector *&a);

inline FuncVector *&setident      (FuncVector *&a) { return a = NULL; }
inline FuncVector *&setzero       (FuncVector *&a) { return a = NULL; }
inline FuncVector *&setzeropassive(FuncVector *&a) { return a = NULL; }
inline FuncVector *&setposate     (FuncVector *&a) { return a = NULL; }
inline FuncVector *&setnegate     (FuncVector *&a) { return a = NULL; }
inline FuncVector *&setconj       (FuncVector *&a) { return a = NULL; }
inline FuncVector *&setrand       (FuncVector *&a) { return a = NULL; }

inline RKHSVector *&setident      (RKHSVector *&a);
inline RKHSVector *&setzero       (RKHSVector *&a);
inline RKHSVector *&setzeropassive(RKHSVector *&a);
inline RKHSVector *&setposate     (RKHSVector *&a);
inline RKHSVector *&setnegate     (RKHSVector *&a);
inline RKHSVector *&setconj       (RKHSVector *&a);
inline RKHSVector *&setrand       (RKHSVector *&a);

inline RKHSVector *&setident      (RKHSVector *&a) { return a = NULL; }
inline RKHSVector *&setzero       (RKHSVector *&a) { return a = NULL; }
inline RKHSVector *&setzeropassive(RKHSVector *&a) { return a = NULL; }
inline RKHSVector *&setposate     (RKHSVector *&a) { return a = NULL; }
inline RKHSVector *&setnegate     (RKHSVector *&a) { return a = NULL; }
inline RKHSVector *&setconj       (RKHSVector *&a) { return a = NULL; }
inline RKHSVector *&setrand       (RKHSVector *&a) { return a = NULL; }

inline BernVector *&setident      (BernVector *&a);
inline BernVector *&setzero       (BernVector *&a);
inline BernVector *&setzeropassive(BernVector *&a);
inline BernVector *&setposate     (BernVector *&a);
inline BernVector *&setnegate     (BernVector *&a);
inline BernVector *&setconj       (BernVector *&a);
inline BernVector *&setrand       (BernVector *&a);

inline BernVector *&setident      (BernVector *&a) { return a = NULL; }
inline BernVector *&setzero       (BernVector *&a) { return a = NULL; }
inline BernVector *&setzeropassive(BernVector *&a) { return a = NULL; }
inline BernVector *&setposate     (BernVector *&a) { return a = NULL; }
inline BernVector *&setnegate     (BernVector *&a) { return a = NULL; }
inline BernVector *&setconj       (BernVector *&a) { return a = NULL; }
inline BernVector *&setrand       (BernVector *&a) { return a = NULL; }



#endif


