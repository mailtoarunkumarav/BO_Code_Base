
//
// Vector class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _Vector_h
#define _Vector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <sstream>
#include "dynarray.h"
#include "basefn.h"
#include "numbase.h"




#ifndef DEFAULT_SAMPLES_SAMPLE
#define DEFAULT_SAMPLES_SAMPLE     100
#endif


template <class T> class Vector;
template <class T> class retVector;

//Spoilers
template <class T> class Matrix;
template <class T> class SparseVector;
class gentype;

// Stream operators

template <class T> std::ostream &operator<<(std::ostream &output, const Vector<T> &src );
template <class T> std::istream &operator>>(std::istream &input,        Vector<T> &dest);
template <class T> std::istream &streamItIn(std::istream &input,        Vector<T> &dest, int processxyzvw = 1);

//Spoilers
template <class T> std::istream &streamItIn(std::istream &input, SparseVector<T> &dest, int processxyzvw = 1);
std::istream &streamItIn(std::istream &input, gentype &dest, int processxyzvw = 1);

template <class T> inline int isitadouble(const T &dummy);
template <class T> inline int isitadouble(const T &dummy) { (void) dummy; return 0; }

template <> inline int isitadouble<double>(const double &dummy);
template <> inline int isitadouble<double>(const double &dummy) { (void) dummy; return 1; }


// Swap function

template <class T> void qswap(const Vector<T> *&a, const Vector<T> *&b);
template <class T> void qswap(Vector<T> *&a, Vector<T> *&b);
template <class T> void qswap(Vector<T> &a, Vector<T> &b);

// For retVector

template <class T> void qswap(retVector<T> &a, retVector<T> &b);

// Base share function

template <class S, class U> int shareBase(const Vector<S> &thus, const Vector<U> &that);

// Vector return handle.

template <class T> class retVector;

// Need to be defined first

template <class S> double norm1 (const Vector<S> &a);
template <class S> double norm2 (const Vector<S> &a);
template <class S> double normp (const Vector<S> &a, double p);
template <class S> double normd (const Vector<S> &a);
template <class S> double absinf(const Vector<S> &a);


template <class T>
class retVector : public Vector<T>
{
public:
    svm_explicit retVector() : Vector<T>("&") { return; }

    // This function resets the return vector to clean-slate.  No need to
    // call this as it gets called when required by operator()

    retVector<T> &reset(void);
};

// Handy fixed vectors
//
// zeroxvec: ( 0 0 0 0 ... )
// onexvec:  ( 1 1 1 1 ... )
// cntxvec:  ( 0 1 2 3 ... )

inline const Vector<int> &zerointvec(int size, retVector<int> &tmpv);
inline const Vector<int> &oneintvec (int size, retVector<int> &tmpv);
inline const Vector<int> &cntintvec (int size, retVector<int> &tmpv);

inline const Vector<double> &zerodoublevec(int size, retVector<double> &tmpv);
inline const Vector<double> &onedoublevec (int size, retVector<double> &tmpv);
inline const Vector<double> &cntdoublevec (int size, retVector<double> &tmpv);

// The class itself

template <class T>
class Vector
{
    friend class Matrix<T>;
    friend class retVector<T>;

    template <class S> friend void qswap(Vector<S> &a, Vector<S> &b);
    template <class S> friend void qswap(retVector<S> &a, retVector<S> &b);

    friend inline const Vector<int> &zerointvec(int size, retVector<int> &tmpv);
    friend inline const Vector<int> &oneintvec (int size, retVector<int> &tmpv);
    friend inline const Vector<int> &cntintvec (int size, retVector<int> &tmpv);

    friend inline const Vector<double> &zerodoublevec(int size, retVector<double> &tmpv);
    friend inline const Vector<double> &onedoublevec (int size, retVector<double> &tmpv);
    friend inline const Vector<double> &cntdoublevec (int size, retVector<double> &tmpv);

    template <class S, class U> friend int shareBase(const Vector<S> &thus, const Vector<U> &that);

public:

    // Constructors and Destructors

                 Vector(const Vector<T> &src);
    svm_explicit Vector(int size = 0, const T *src = NULL);

    virtual ~Vector();

    // Print and make duplicate

    virtual Vector<T> *makeDup(void) const
    {
        Vector<T> *dup;

        MEMNEW(dup,Vector<T>(*this));

        return dup;
    }

    // Assignment
    //
    // - vector assignment: unless this vector is a temporary vector created
    //   to refer to parts of another vector then we do not require that sizes
    //   align but rather completely overwrite the destination, resetting the
    //   size to that of the source.
    // - scalar assignment: in this case the size of the vector remains
    //   unchanged, but all elements will be overwritten.
    // - behaviour is undefined if scalar is an element of this.
    // - assignment from a matrix only possible for 1*d or d*1 matrix.
    // - the assign function allows you to infer assignment between different
    //   template classes.  This is not defined as operator= to ensure that
    //   the second and third forms of operator= work without ambiguity

    Vector<T> &operator=(const Vector<T> &src) { return assign(src); }
    Vector<T> &operator=(const T &src)         { return assign(src); }

    virtual Vector<T> &assign(const Vector<T> &src);
    virtual Vector<T> &assign(const T &src);

    Vector<T> &operator=(const Matrix<T> &src);

    template <class S> Vector<T> &castassign(const Vector<S> &src);

    // Simple vector manipulations
    //
    // ident:  apply setident() to all elements of the vector
    // zero:   apply setzero() to all elements of the vector
    //        (apply setzero() to element i if argument given).
    //
    // zeromost: zero all elements except that specified.  Specifying -1
    //         indicates do not zero any elements
    // softzero: equivalent to zero (for consistency with sparsevector)
    // posate: apply setposate() to all elements of the vector
    // negate: apply setnegate() to all elements of the vector
    // conj:   apply setconj() to all elements of the vector
    // rand:   apply .rand() to all elements of the vector
    // offset: amoff > 0: insert amoff elements at the start of the vector
    //         amoff < 0: remove amoff elements from the start of the vector
    //
    //
    //
    //
    //
    // each returns a reference to *this

    Vector<T> &ident(void);
    Vector<T> &zero(int i);
    Vector<T> &zeromost(int i);
    Vector<T> &zeropassive(void);
    Vector<T> &offset(int amoff);

    virtual Vector<T> &softzero(void);
    virtual Vector<T> &zero(void);
    virtual Vector<T> &posate(void);
    virtual Vector<T> &negate(void);
    virtual Vector<T> &conj(void);
    virtual Vector<T> &rand(void);

    // Access:
    //
    // - ("&",i) access a reference to element i.
    // - (i) access a const reference to element i.
    // - direcref(i) is functionally equivalent to (ind(i)).
    // - direref(i) is functionally equivalent to ("&",ind(i)).
    //
    // - ("&") returns are non-const reference to this
    // - () returns a const reference to this.
    //
    // - zeroExtDeref(i): like (i), where i is a vector, but in this case
    //   there is an additional index -1 that may be used.  Element "-1" is
    //   set zero upon calling.  This is handy for "padding" a vector with
    //   zeros (for example in sparse vectors or extended caches).
    //
    // Variants:
    //
    // - if i is of type Vector<int> then the reference returned is to the
    //   elements specified in i.
    // - if ib,is,im is given then this is the same as a vector i being used
    //   specified by: ( ib ib+is ib+is+is ... max_n(i=ib+(n*s)|i<=im) )
    //   (and if im < ib then an empty reference is returned)
    //
    // Notes:
    //
    // - direcref and direref variants are included for consistency with sparse
    //   vector type.
    //
    // Scope of result:
    //
    // - The scope of the returned reference is the minimum of the scope of
    //   retVector &tmp or *this.
    // - retVector &tmp may or may not be used depending on, so
    //   never *assume* that it will be!
    // - The returned reference is actually *this through a layer of indirection, 
    //   so any changes to it will be reflected in *this (and vice-versa).

    Vector<T> &operator()(const char *dummy,                         retVector<T> &tmp) { (void) dummy; (void) tmp; if ( imoverhere ) { return overhere(); } return *this; }
    T         &operator()(const char *dummy, int i                                    ) { NiceAssert( !infsize() ); NiceAssert( i >= 0 ); NiceAssert( i < dsize ); NiceAssert( content ); NiceAssert( pivot ); return (*content)(dummy,(*pivot)(iib+(i*iis))); }
    Vector<T> &operator()(const char *dummy, const Vector<int> &i,   retVector<T> &tmp);
    Vector<T> &operator()(const char *dummy, int ib, int is, int im, retVector<T> &tmp);

    const Vector<T> &operator()(                        retVector<T> &tmp) const { (void) tmp; if ( imoverhere ) { return overhere(); } return *this; }
    const T         &operator()(int i                                    ) const { NiceAssert( !infsize() ); NiceAssert( i >= 0 ); NiceAssert( i < dsize ); NiceAssert( ccontent ); NiceAssert( pivot ); return (*ccontent)((*pivot)(iib+(i*iis))); }
    const Vector<T> &operator()(const Vector<int> &i,   retVector<T> &tmp) const;
    const Vector<T> &operator()(int ib, int is, int im, retVector<T> &tmp) const;

    // For FuncVector

    virtual T &operator()(T &res, const        T  &i) const { if ( imoverhere ) { return overhere()(res,i); } throw("Continuous dereference not allowed for finite dimensional vector."); return res; }
    virtual T &operator()(T &res, const Vector<T> &i) const { if ( imoverhere ) { return overhere()(res,i); } throw("Continuous dereference not allowed for finite dimensional vector."); return res; }

    // Information functions
    //
    // type():    returns 0 for standard, 1 for FuncVector, 2 for RKHSVector etc 
    // size():    returns the size of the vector (if finite) 
    // order():   returns angry geckos 
    // infsize(): returns 1 for infinite size, 0 otherwise 
    // ismixed(): return 1 if "mixed type" vector (sum of different types, no defined inner products etc)

    virtual int type(void)    const { return imoverhere ? overhere().type()    : 0; }
            int size(void)    const { return dsize;                                 }
            int order(void)   const { return ceilintlog2(size());                   }
    virtual int infsize(void) const { return imoverhere ? overhere().infsize() : 0; }
    virtual int ismixed(void) const { return 0;                                     }

    virtual int testsametype(std::string &typestring)
    {
        return typestring == "";
    }

    // Vector scaling:
    //
    // Apply (*this)("&",i) *= a for all i, or (*this)("&",i) *= a(i) for the
    // vectorial version.  This is useful for scaling vectors of vectors.

    template <class S> Vector<T>  &scale(const S &a);
    template <class S> Vector<T>  &scale(const Vector<S> &a);
    template <class S> Vector<T> &lscale(const S &a);
    template <class S> Vector<T> &lscale(const Vector<S> &a);

    // Scaled addition:
    //
    // The following is functionally equivalent to *this += (a*b).  However
    // it is quicker and uses less memory as no temporary variables are
    // constructed.

    template <class S> Vector<T> &scaleAdd  (const S &a, const Vector<T> &b);
    template <class S> Vector<T> &scaleAddR (const Vector<T> &a, const S &b);
    template <class S> Vector<T> &scaleAddB (const T &a, const Vector<S> &b);
    template <class S> Vector<T> &scaleAddBR(const Vector<S> &b, const T &a);

    // Add and remove element functions
    //
    // add:    ( c ) (i)          ( c ) (i)
    //         ( d ) (...)  ->    ( 0 ) (1)
    //                            ( d ) (...)
    // addpad: ( c ) (i)          ( c ) (i)
    //         ( d ) (...)  ->    ( 0 ) (num)
    //                            ( d ) (...)
    // remove: ( c ) (i)          ( c ) (i)
    //         ( d ) (1)    ->    ( d ) (...)
    //         ( e ) (...)
    // resize: either add to end or remove from end until desired size is
    //         obtained.
    // append: add a to end of vector at position i >= size()
    // pad: add n elements to end of vector
    //
    // Note that these may not be applied to temporary vectors.

    Vector<T> &add(int i);
    Vector<T> &add(int i, int ipos) { NiceAssert( i == ipos ); (void) ipos; return add(i); }
    Vector<T> &addpad(int i, int num);
    Vector<T> &remove(int i);
    Vector<T> &remove(const Vector<int> &i);
    Vector<T> &resize(int i);
    Vector<T> &pad(int n);
    template <class S> Vector<T> &resize(const Vector<S> &sizeTemplateUsed) { return resize(sizeTemplateUsed.size()); }
    Vector<T> &setorder(int i) { return resize(1<<i); }
    Vector<T> &append(int i) { add(i,size()); return *this; } // { add(i,indsize()); return *this; }
    Vector<T> &append(int i, const T &a);
    Vector<T> &append(int i, const Vector<T> &a);

    // Function application - apply function fn to each element of vector.

    virtual Vector<T> &applyon(T (*fn)(T));
    virtual Vector<T> &applyon(T (*fn)(const T &));
    virtual Vector<T> &applyon(T (*fn)(T, const void *), const void *a);
    virtual Vector<T> &applyon(T (*fn)(const T &, const void *), const void *a);
    virtual Vector<T> &applyon(T &(*fn)(T &));
    virtual Vector<T> &applyon(T &(*fn)(T &, const void *), const void *a);

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
    //
    // It should be noted that these function actually move the contents of
    // the vector.  Hence if the contents are nontrivial (say you're dealing
    // with Vector<Vector<double>> or similar) then qswap operators will
    // be called repeatedly and lots of memory shuffling will occur.  Hence it
    // is usually better to keep a pivot vector of type Vector<int> on which
    // these operations are carried out.  For example given:
    //
    // Vector<Vector<double>> x;
    // Vector<int> pivot;
    //
    // it is better to make pivot = ( 0 1 2 3 ... ) and then do blockswap,
    // squareswap type operations on pivot and then use x(pivot) in
    // calculations.  This will have the same effect as doing blockswap etc.
    // operations directly on x but without the subsequent time penalty.

    Vector<T> &blockswap (int i, int j);
    Vector<T> &squareswap(int i, int j);

    // Pre-allocation control.  Vectors are pre-assigned with a given buffer
    // size to "grow" into, which can stretch and shrink dynamically as the
    // vector size evolves over time.  The downside of this is that memory
    // can be wasted when the class over-estimates the amount of memory
    // required, and speed can suffer if it under-estimates repeatedly (say
    // a vector grows from very small to very large) and the buffer must
    // be repeatedly reallocated and the contents copied over.
    //
    // To overcome this the following function lets you preset the size of
    // the allocate-ahead buffer if you know (or have a bound on) the final
    // size of the vector.  Note that this *only* works on base vector, not
    // children.
    //
    // To restore standard behaviour set newallocsize == -1
    //
    // The function applyonall applies an operation to all allocated members,
    // which may include pre-allocated members.
    //
    // Other functions here provide direct access to dynamic array base

    virtual void prealloc(int newallocsize);
    virtual void useStandardAllocation(void);
    virtual void useTightAllocation(void);
    virtual void useSlackAllocation(void);

    void applyOnAll(void (*fn)(T &, int), int argx);

    virtual int array_norm (void) const { return imoverhere ? overhere().array_norm()  : ( content ? (*content).array_norm()  : 1 ); }
    virtual int array_tight(void) const { return imoverhere ? overhere().array_tight() : ( content ? (*content).array_tight() : 0 ); }
    virtual int array_slack(void) const { return imoverhere ? overhere().array_slack() : ( content ? (*content).array_slack() : 0 ); }

    // Slight complication: whenever an assignment is called we need to deal
    // with possibilities like a *= a(pivot) or b = b(pivot), which will
    // royally screw things up if for example pivot = ( 3 2 1 ) and we start
    // naively doing the assignment element by element top to bottom.  To
    // deal with this we need to check if the source and destination share
    // the same root, and if they do make a temporary copy of the source and
    // then call the assignment operator for this temporary copy.  The
    // following function facillitates this by testing if this instance sharesFuncVector
    // a common root with another.

    template <class S> 
    int shareBase(const Vector<S> &that) const { return ::shareBase(*this,that);                        }
    int base(void)                       const { return nbase;                                          }
    int contentalloced(void)             const { return ( content ? 1 : 0 );                            }
    int contentarray_hold(void)          const { NiceAssert( content ); return content->array_hold();   }
    int contentarray_alloc(void)         const { NiceAssert( content ); return content->array_alloc();  }

    // Sorry about this: I would rather use a friend function, but apparently
    // "partial specialisation `Vector<S>' declared `friend'" is a bad thing.
    // This just returns a (completely unpivotted) reference to ccontent.
    // Please don't try to use this, it's just an ugly hack in place of a
    // friend declaration of "template <class S> friend class Vector<S>".
    // And if I can work out a more elegant way of doing things I will be
    // removing this function.

    const DynArray<T> &grabcontentdirect(void) const { NiceAssert( !infsize() ); return *ccontent; }

    // Control how the output looks

    char getnewln(void) const { return newln; }
    char setnewln(char srcvl) { NiceAssert(isspace(srcvl)); return ( newln = srcvl ); }




    // Operators filled-in by functional (****don't use in code, just placeholders for now****)
    //
    // conj = 0: noConj
    //        1: normal
    //        2: revConj

    virtual std::ostream &outstream(std::ostream &output) const { (void) output; throw("no"); return output; } // DO NOT USE THIS!
    virtual std::istream &instream (std::istream &input )       { (void) input;  throw("no"); return input;  } // DO NOT USE THIS!

    virtual std::istream &streamItIn(std::istream &input, int processxyzvw = 1) { (void) processxyzvw; throw("no"); return input; } // DON'T USE THIS!

    virtual T &inner1(T &res                                                            ) const {                                           if ( imoverhere ) { overhere().inner1(res);       } else { throw("Not an FuncVector 1"); } return res; }
    virtual T &inner2(T &res, const Vector<T> &b, int conj = 1                          ) const { (void) b; (void) conj;                    if ( imoverhere ) { overhere().inner2(res,b);     } else if ( b.imoverhere ) { b.inner2(res,*this,( conj == 1 ) ? 2 : ( ( conj == 2 ) ? 1 : 0 )); } else { throw("Not an FuncVector 2"); } return res; }
    virtual T &inner3(T &res, const Vector<T> &b, const Vector<T> &c                    ) const { (void) b; (void) c; (void) res;           if ( imoverhere ) { overhere().inner3(res,b,c);   } else if ( b.imoverhere ) { b.inner3(res,*this,c); } else if ( c.imoverhere ) { b.inner3(res,b,*this); } else { throw("Not an FuncVector 3"); } return res; }
    virtual T &inner4(T &res, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d) const { (void) b; (void) c; (void) d; (void) res; if ( imoverhere ) { overhere().inner4(res,b,c,d); } else if ( b.imoverhere ) { b.inner4(res,*this,c,d); } else if ( c.imoverhere ) { c.inner4(res,b,*this,d); } else if ( d.imoverhere ) { d.inner4(res,b,c,*this); } else { throw("Not an FuncVector 4"); } return res; }
    virtual T &innerp(T &res, const Vector<const Vector<T> *> &b                        ) const 
    {
//        (void) b; - really weird MS bug!  Uncomment this and microsoft visual c-c++ compiler, compile strfns.cc and the compiler basically chews up all available memory and crashes the entire system!  I have no fucking *idea* why!
        (void) res;

        if ( imoverhere )
        {
            overhere().innerp(res,b);
        }

        else
        {
             throw("Not an FuncVector 5");
        }

        return res;
    }

    virtual double &inner1Real(double &res                                                            ) const {              if ( imoverhere ) { overhere().inner1Real(res);       } else { throw("Not an FuncVector 6");  } return res; }
    virtual double &inner2Real(double &res, const Vector<T> &b, int conj = 1                          ) const { (void) conj; if ( imoverhere ) { overhere().inner2Real(res,b);     } else { throw("Not an FuncVector 7");  } return res; }
    virtual double &inner3Real(double &res, const Vector<T> &b, const Vector<T> &c                    ) const {              if ( imoverhere ) { overhere().inner3Real(res,b,c);   } else { throw("Not an FuncVector 8");  } return res; }
    virtual double &inner4Real(double &res, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d) const {              if ( imoverhere ) { overhere().inner4Real(res,b,c,d); } else { throw("Not an FuncVector 9");  } return res; }
    virtual double &innerpReal(double &res, const Vector<const Vector<T> *> &b                        ) const {              if ( imoverhere ) { overhere().innerpReal(res,b);     } else { throw("Not an FuncVector 10"); } return res; } 

    virtual double norm1(void)     const { double res = 0.0;           if ( imoverhere ) { res = overhere().norm1();  } else { throw("Not an FuncVector 11"); } return res; }
    virtual double norm2(void)     const { double res = 0.0;           if ( imoverhere ) { res = overhere().norm2();  } else { throw("Not an FuncVector 12"); } return res; }
    virtual double normp(double p) const { double res = 0.0; (void) p; if ( imoverhere ) { res = overhere().normp(p); } else { throw("Not an FuncVector 13"); } return res; }
    virtual double normd(void)     const { return norm2(); }

    virtual double abs1(void)     const { return norm1(); }
    virtual double abs2(void)     const { return sqrt(norm2()); }
    virtual double absp(double p) const { return pow(normp(p),1/p); }
    virtual double absinf(void)   const { double res = 0.0; if ( imoverhere ) { res = overhere().absinf(); } else { throw("Not an FuncVector 14"); } return res; }
    virtual double absd(void)     const { return abs2(); }

    virtual Vector<T> &subit (const Vector<T> &b) { if ( imoverhere ) { overhere().subit(b);  } else { throw("Not in FuncVector 15"); } return *this; }
    virtual Vector<T> &addit (const Vector<T> &b) { if ( imoverhere ) { overhere().addit(b);  } else { throw("Not in FuncVector 16"); } return *this; }
    virtual Vector<T> &mulit (const T         &b) { if ( imoverhere ) { overhere().mulit(b);  } else { throw("Not in FuncVector 17"); } return *this; } // this*b
    virtual Vector<T> &rmulit(const T         &b) { if ( imoverhere ) { overhere().rmulit(b); } else { throw("Not in FuncVector 18"); } return *this; } // b*this
    virtual Vector<T> &divit (const T         &b) { if ( imoverhere ) { overhere().divit(b);  } else { throw("Not in FuncVector 19"); } return *this; } // this/b
    virtual Vector<T> &rdivit(const T         &b) { if ( imoverhere ) { overhere().rdivit(b); } else { throw("Not in FuncVector 20"); } return *this; } // b\this

    virtual int iseq(const Vector<T> &b) { if ( imoverhere ) { return overhere().iseq(b); } else { throw("Not in FuncVector 21"); } return 0; }

private:

    // dsize: size of vector
    // fixsize: if set then the size cannot be changed
    //
    // nbase: 0 if content is local, 1 if it points elsewhere
    //        (NB: if nbase == 0 then pivot = ( 0 1 2 ... ))
    // pbase: 0 if pivot is local, 1 if it points elsewhere
    //        (NB: if nbase == 0 then pbase == 0 by definition)
    //
    // iib: constant added to indexes
    // iis: step for indexes
    //
    // bkref: if nbase, this is the vector derived from (and pointed to).  This
    //        is used by the shareBase function.
    // content: contents of vector
    // ccontent: constant pointer to content
    // pivot: pivotting used to access contents
    // 
    // newln: usually vectors are printed:
    //                   ( x0 ;
    //                   x1 ;
    //                   ... ;
    //                   xn-1 )
    //        where the newline is given by newln.  By setting for example
    //        newln = ' ' you would get the alternative format
    //                   ( x0 ; x1 ; ... ; xn-1 )

    int dsize;
//    int fixsize;

    int nbase;
    int pbase;

    int iib;
    int iis;

    char newln;

    // The vector class was written a long time ago, *before* I knew about polymorphism.  
    // It got built in to a lot of things.  Much later I wanted to add support for RKHS, 
    // which is an infinite-dimensional vector with an inner product that isn't just an
    // integral.  The easiest was to do this was polymorphism... except I couldn't,
    // because in *many* places (eg gentype) the vectors are alloced *inside* old code,
    // so are by default of type Vector<T>.  To get around this I built the polymorphed 
    // class (RKHSVector) and then added this pointer.  If it's NULL then everything
    // acts like you would expect.  If it's not then virtual functions will detect this
    // and redirect to this pointer.  Thus if this points to RKHSVector then, *even if
    // this is of type Vector<T>*, virtual functions will be appropriately redirected, so
    // it will *act like* it is of type RKHSVector.

public: // because fuck it
    Vector<T> *imoverhere;

    const Vector<T> &overhere(void) const
    {
        NiceAssert(imoverhere);

        const Vector<T> *overthere = imoverhere;

        while ( (*overthere).imoverhere )
        {
            overthere = (*overthere).imoverhere;
        }

        return *overthere;
    }

    Vector<T> &overhere(void)
    {
        NiceAssert(imoverhere);

        Vector<T> *overthere = imoverhere;

        while ( (*overthere).imoverhere )
        {
            overthere = (*overthere).imoverhere;
        }

        return *overthere;
    }
private:

    // This may be returned by ind() call.  Might cause issues but unlikely

    retVector<int> *indtmp;

    const Vector<T> *bkref;
    DynArray<T> *content;
    const DynArray<T> *ccontent;
    const DynArray<int> *pivot;

    // Internal dereferencing operators

    Vector<T> &operator()(const char *dummy, const DynArray<int> &i, int isize, retVector<T> &tmp);
    const Vector<T> &operator()(const DynArray<int> &i, int isize, retVector<T> &tmp) const;

    // These internal versions do two steps in one to prevent the need for two retVector arguments.
    // The steps are: res = ((*this)(i))(ib,is,im)

    Vector<T> &operator()(const char *dummy, const DynArray<int> &i, int ib, int is, int im, retVector<T> &tmp);
    const Vector<T> &operator()(const DynArray<int> &i, int ib, int is, int im, retVector<T> &tmp) const;

    // Blind constructor: does no allocation, just sets bkref and defaults

    svm_explicit Vector(const char *dummy, const Vector<T> &src);
    svm_explicit Vector(const char *dummy);

    // Dynarray constructor - constructs a (const) vector refering to an
    // external dynamic array.  Result is nominally constant: use with care

    svm_explicit Vector(const DynArray<T> *ccontentsrc);

    // Fix bkref

    void fixbkreftree(const Vector<T> *newbkref);

    // const removal cheat

    Vector<T> *thisindirect[1];
};

template <class T> void qswap(Vector<T> &a, Vector<T> &b)
{
    NiceAssert( a.nbase == 0 );
    NiceAssert( b.nbase == 0 );

    qswap(a.dsize   ,b.dsize   );
//    qswap(a.fixsize,b.fixsize);
    qswap(a.nbase   ,b.nbase   );
    qswap(a.pbase   ,b.pbase   );
    qswap(a.iib     ,b.iib     );
    qswap(a.iis     ,b.iis     );

    qswap(a.newln,     b.newln     );
    qswap(a.imoverhere,b.imoverhere);

    const Vector<T> *bkref;
    DynArray<T> *content;
    const DynArray<T> *ccontent;
    const DynArray<int> *pivot;

    bkref    = a.bkref;    a.bkref    = b.bkref;    b.bkref    = bkref;
    content  = a.content;  a.content  = b.content;  b.content  = content;
    ccontent = a.ccontent; a.ccontent = b.ccontent; b.ccontent = ccontent;
    pivot    = a.pivot;    a.pivot    = b.pivot;    b.pivot    = pivot;

    // The above will have messed up one important thing, namely bkref and
    // bkref in any child vectors.  We must now repair the child trees if
    // they exist

    a.fixbkreftree(&a);
    b.fixbkreftree(&b);

    return;
}

template <class T> void qswap(retVector<T> &a, retVector<T> &b)
{
    // Don't want to assert nbase == 0, because it may not be
    // Just reset: this should only be used when a,b are *not* in active use!
    // (eg if you have Vector<retVector<T> >, like in kcache)

    a.reset();
    b.reset();

/*
// Don't check these!
//    NiceAssert( a.nbase == 0 );
//    NiceAssert( b.nbase == 0 );

    qswap(a.nbase   ,b.nbase   );
    qswap(a.pbase   ,b.pbase   );
    qswap(a.dsize   ,b.dsize   );
//    qswap(a.fixsize,b.fixsize);
    qswap(a.nbase   ,b.nbase   );
    qswap(a.pbase   ,b.pbase   );
    qswap(a.iib     ,b.iib     );
    qswap(a.iis     ,b.iis     );

    qswap(a.newln,     b.newln     );
    qswap(a.imoverhere,b.imoverhere);

    const Vector<T> *bkref;
    DynArray<T> *content;
    const DynArray<T> *ccontent;
    const DynArray<int> *pivot;

    bkref    = a.bkref;    a.bkref    = b.bkref;    b.bkref    = bkref;
    content  = a.content;  a.content  = b.content;  b.content  = content;
    ccontent = a.ccontent; a.ccontent = b.ccontent; b.ccontent = ccontent;
    pivot    = a.pivot;    a.pivot    = b.pivot;    b.pivot    = pivot;

    // The above will have messed up one important thing, namely bkref and
    // bkref in any child vectors.  We must now repair the child trees if
    // they exist

    a.fixbkreftree(&a);
    b.fixbkreftree(&b);
*/
    return;
}

template <class T> void qswap(const Vector<T> *&a, const Vector<T> *&b)
{
    const Vector<T> *c;

    c = a;
    a = b;
    b = c;

    return;
}

template <class T> void qswap(Vector<T> *&a, Vector<T> *&b)
{
    Vector<T> *c;

    c = a;
    a = b;
    b = c;

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
// orall: find logical OR of all elements in vector
// andall: find logical OR of all elements in vector
// prod: find the product of elements in a vector, top to bottom
// Prod: find the product of elements in a vector, bottom to top
// mean: calculate the mean of.  Ill-defined if vector empty.
// median: calculate the median.  Put the index into i.
//
// twoProduct: calculate the inner product of two vectors conj(a)'.b
// twoProductNoConj: calculate the inner product of two vectors but without conjugating a
// twoProductRevConj: calculate the inner product of two vectors a'.conj(b)
// fourProduct: calculate the four-product sum_i (a_i.b_i).(c_i.d_i)  (note order of operation)
// mProduct: calculate the m-product sum_i prod_j (a_{2j,i}.a_{2j,i+1})  (note order of operation)
// Indexed versions: a -> a(n), b -> b(n), c -> c(n), ....  It is assumed that n is sorted from smallest to largest.
// Scaled versions: a -> a./scale, b -> b./scale, c -> c./scale ...
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
// abs1:   return the 1-norm of the vector
// abs2:   return the square root of the 2-norm of the vector
// absp:   return the p-root of the p-norm of the vector
// absinf: return the inf-norm of the vector
// norm1:  return the 1-norm of the vector ||a||_1
// norm2:  return the 2-norm of the vector ||a||_2^2
// normp:  return the p-norm of the vector ||a||_p^p
//
// sum and mean can be weighted using second argument

template <class T> const T &sum (T &res, const Vector<T> &a, const Vector<double> &weights);
template <class T> const T &mean(T &res, const Vector<T> &a, const Vector<double> &weights);

template <class S, class T> const T &sumb(T &result, const Vector<S> &left_op, const Vector<T> &right_op);

template <class T> T max     (const Vector<T> &a, const Vector<T> &b, int &i);
template <class T> T min     (const Vector<T> &a, const Vector<T> &b, int &i);
template <class T> T maxabs  (const Vector<T> &a, const Vector<T> &b, int &i);
template <class T> T maxabs  (const Vector<T> &a, int &i);
template <class T> T minabs  (const Vector<T> &a, int &i);
template <class T> T sqabsmax(const Vector<T> &a);
template <class T> T sqabsmin(const Vector<T> &a);
template <class T> T sum     (const Vector<T> &a);
template <class T> T sqsum   (const Vector<T> &a);
template <class T> T orall   (const Vector<T> &a);
template <class T> T andall  (const Vector<T> &a);
template <class T> T prod    (const Vector<T> &a);
template <class T> T Prod    (const Vector<T> &a);
template <class T> T mean    (const Vector<T> &a);
template <class T> T sqmean  (const Vector<T> &a);
template <class T> T vari    (const Vector<T> &a);
template <class T> T stdev   (const Vector<T> &a);

template <class T> const T &max   (const Vector<T> &a, int &i);
template <class T> const T &min   (const Vector<T> &a, int &i);
template <class T> const T &median(const Vector<T> &a, int &i);

template <class T> const T &max     (T &res, const Vector<T> &a, const Vector<T> &b, int &i);
template <class T> const T &min     (T &res, const Vector<T> &a, const Vector<T> &b, int &i);
template <class T> const T &maxabs  (T &res, const Vector<T> &a, int &i);
template <class T> const T &minabs  (T &res, const Vector<T> &a, int &i);
template <class T> const T &sqabsmax(T &res, const Vector<T> &a);
template <class T> const T &sqabsmin(T &res, const Vector<T> &a);
template <class T> const T &sum     (T &res, const Vector<T> &a);
template <class T> const T &sqsum   (T &res, const Vector<T> &a);
template <class T> const T &orall   (T &res, const Vector<T> &a);
template <class T> const T &andall  (T &res, const Vector<T> &a);
template <class T> const T &prod    (T &res, const Vector<T> &a);
template <class T> const T &Prod    (T &res, const Vector<T> &a);
template <class T> const T &mean    (T &res, const Vector<T> &a);
template <class T> const T &sqmean  (T &res, const Vector<T> &a);
template <class T> const T &vari    (T &res, const Vector<T> &a);
template <class T> const T &stdev   (T &res, const Vector<T> &a);


template <class T> T &twoProductNoConj                   (T &res,                       const Vector<T> &a, const Vector<T> &b                        );
template <class T> T &twoProductRevConj                  (T &res,                       const Vector<T> &a, const Vector<T> &b                        );
template <class T> T &twoProductScaledNoConj             (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &twoProductScaledRevConj            (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &twoProductRightScaled              (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &twoProductRightScaledNoConj        (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &twoProductRightScaledRevConj       (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &twoProductLeftScaled               (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &twoProductLeftScaledNoConj         (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &twoProductLeftScaledRevConj        (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedtwoProductNoConj            (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b                        );
template <class T> T &indexedtwoProductRevConj           (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b                        );
template <class T> T &indexedtwoProductScaledNoConj      (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedtwoProductScaledRevConj     (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedtwoProductRightScaled       (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedtwoProductRightScaledNoConj (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedtwoProductRightScaledRevConj(T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedtwoProductLeftScaled        (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedtwoProductLeftScaledNoConj  (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedtwoProductLeftScaledRevConj (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);

template <class T> T &oneProduct  (T &res, const Vector<T> &a);
template <class T> T &twoProduct  (T &res, const Vector<T> &a, const Vector<T> &b);
template <class T> T &threeProduct(T &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c);
template <class T> T &fourProduct (T &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d);
template <class T> T &mProduct    (T &res, const Vector<const Vector <T> *> &a);

template <class T> T &oneProductScaled  (T &res, const Vector<T> &a, const Vector<T> &scale);
template <class T> T &twoProductScaled  (T &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &threeProductScaled(T &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &scale);
template <class T> T &fourProductScaled (T &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d, const Vector<T> &scale);
template <class T> T &mProductScaled    (T &res, const Vector<const Vector <T> *> &a, const Vector<T> &scale);

template <class T> double &oneProductAssumeReal  (double &res, const Vector<T> &a);
template <class T> double &twoProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b);
template <class T> double &threeProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c);
template <class T> double &fourProductAssumeReal (double &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d);
template <class T> double &mProductAssumeReal    (double &res, const Vector<const Vector <T> *> &a);

template <class T> T &indexedoneProduct  (T &res, const Vector<int> &n, const Vector<T> &a);
template <class T> T &indexedtwoProduct  (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b);
template <class T> T &indexedthreeProduct(T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c);
template <class T> T &indexedfourProduct (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d);
template <class T> T &indexedmProduct    (T &res, const Vector<int> &n, const Vector<const Vector <T> *> &a);

template <class T> T &indexedoneProductScaled  (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &scale);
template <class T> T &indexedtwoProductScaled  (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedthreeProductScaled(T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &scale);
template <class T> T &indexedfourProductScaled (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d, const Vector<T> &scale);
template <class T> T &indexedmProductScaled    (T &res, const Vector<int> &n, const Vector<const Vector <T> *> &a, const Vector<T> &scale);

template <class T> Vector<T> &setident(Vector<T> &a);
template <class T> Vector<T> &setzero(Vector<T> &a);
template <class T> Vector<T> &setzeropassive(Vector<T> &a);
template <class T> Vector<T> &setposate(Vector<T> &a);
template <class T> Vector<T> &setnegate(Vector<T> &a);
template <class T> Vector<T> &setconj(Vector<T> &a);
template <class T> Vector<T> &setrand(Vector<T> &a);
template <class T> Vector<T> &postProInnerProd(Vector<T> &a) { return a; }

template <class T> Vector<T> *&setident (Vector<T> *&a) { throw("Whatever"); return a; }
template <class T> Vector<T> *&setzero  (Vector<T> *&a) { return a = NULL; }
template <class T> Vector<T> *&setposate(Vector<T> *&a) { return a; }
template <class T> Vector<T> *&setnegate(Vector<T> *&a) { throw("I reject your reality and substitute my own"); return a; }
template <class T> Vector<T> *&setconj  (Vector<T> *&a) { throw("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
template <class T> Vector<T> *&setrand  (Vector<T> *&a) { throw("Blippity Blappity Blue"); return a; }
template <class T> Vector<T> *&postProInnerProd(Vector<T> *&a) { return a; }

template <class T> const Vector<T> *&setident (const Vector<T> *&a) { throw("Whatever"); return a; }
template <class T> const Vector<T> *&setzero  (const Vector<T> *&a) { return a = NULL; }
template <class T> const Vector<T> *&setposate(const Vector<T> *&a) { return a; }
template <class T> const Vector<T> *&setnegate(const Vector<T> *&a) { throw("I reject your reality and substitute my own"); return a; }
template <class T> const Vector<T> *&setconj  (const Vector<T> *&a) { throw("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
template <class T> const Vector<T> *&setrand  (const Vector<T> *&a) { throw("Blippity Blappity Blue"); return a; }
template <class T> const Vector<T> *&postProInnerProd(const Vector<T> *&a) { return a; }

template <class S> Vector<S> angle(const Vector<S> &a);
template <class S> Vector<double> eabs2(const Vector<S> &a);
template <class S> Vector<S> vangle(const Vector<S> &a, const Vector<S> &defsign);
template <class S> Vector<double> &seteabs2(Vector<S> &a);

template <class S> double abs1  (const Vector<S> &a);
template <class S> double abs2  (const Vector<S> &a);
template <class S> double absp  (const Vector<S> &a, double p);
//template <class S> double absinf(const Vector<S> &a);
template <class S> double absd  (const Vector<S> &a);

//template <class S> double norm1 (const Vector<S> &a);
//template <class S> double norm2 (const Vector<S> &a);
//template <class S> double normp (const Vector<S> &a, double p);
//template <class S> double normd (const Vector<S> &a);


// Kronecker products

template <class T> Vector<T> &kronprod(Vector<T> &res, const Vector<T> &a, const Vector<T> &b);
template <class T> Vector<T> &kronpow(Vector<T> &res, const Vector<T> &a, int n);



// NaN and inf tests

template <class T> int testisvnan(const Vector<T> &x);
template <class T> int testisinf (const Vector<T> &x);
template <class T> int testispinf(const Vector<T> &x);
template <class T> int testisninf(const Vector<T> &x);




// Random permutation function and random fill

inline Vector<int> &randPerm(Vector<int> &res);
template <class T> Vector<T> &randfill (Vector<T> &res);
template <class T> Vector<T> &randnfill(Vector<T> &res);



// Conversion from strings

template <class T> Vector<T> &atoVector(Vector<T> &dest, const std::string &src);

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

template <class T> Vector<T>  operator+ (const Vector<T> &left_op);
template <class T> Vector<T>  operator- (const Vector<T> &left_op);

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
//
//

template <class T> Vector<T>  operator+ (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator- (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator* (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator/ (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator% (const Vector<T> &left_op, const Vector<T> &right_op);

template <class T> Vector<T>  operator+ (const Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T>  operator- (const Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T>  operator* (const Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T>  operator/ (const Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T>  operator% (const Vector<T> &left_op, const T         &right_op);

template <class T> Vector<T>  operator+ (const T         &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator- (const T         &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator* (const T         &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator/ (const T         &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator% (const T         &left_op, const Vector<T> &right_op);

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
//
//

template <class T> Vector<T> &operator+=(      Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T> &operator-=(      Vector<T> &left_op, const Vector<T> &right_op);
//template <class T> Vector<T> &operator*=(      Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T> &operator/=(      Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T> &operator%=(      Vector<T> &left_op, const Vector<T> &right_op);

template <class S, class T> Vector<S> &operator*=(Vector<S> &left_op, const Vector<T> &right_op);

template <class T> Vector<T> &operator+=(      Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T> &operator-=(      Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T> &operator*=(      Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T> &operator/=(      Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T> &operator%=(      Vector<T> &left_op, const T         &right_op);

template <class T> Vector<T> &operator+=(const T         &left_op,       Vector<T> &right_op);
template <class T> Vector<T> &operator-=(const T         &left_op,       Vector<T> &right_op);
template <class T> Vector<T> &operator*=(const T         &left_op,       Vector<T> &right_op);
template <class T> Vector<T> &operator/=(const T         &left_op,       Vector<T> &right_op);
template <class T> Vector<T> &operator%=(const T         &left_op,       Vector<T> &right_op);

// Related non-commutative operations
//
// leftmult:  equivalent to *=
// rightmult: like *=, but result is stored in right_op and ref to right_op is returned

template <class T> Vector<T> &leftmult (      Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T> &leftmult (      Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T> &rightmult(const Vector<T> &left_op,       Vector<T> &right_op);
template <class T> Vector<T> &rightmult(const T         &left_op,       Vector<T> &right_op);

// Relational operator overloading

template <class T> int operator==(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator==(const Vector<T> &left_op, const T         &right_op);
template <class T> int operator==(const T         &left_op, const Vector<T> &right_op);

template <class T> int operator!=(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator!=(const Vector<T> &left_op, const T         &right_op);
template <class T> int operator!=(const T         &left_op, const Vector<T> &right_op);

template <class T> int operator< (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator< (const Vector<T> &left_op, const T         &right_op);
template <class T> int operator< (const T         &left_op, const Vector<T> &right_op);

template <class T> int operator<=(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator<=(const Vector<T> &left_op, const T         &right_op);
template <class T> int operator<=(const T         &left_op, const Vector<T> &right_op);

template <class T> int operator> (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator> (const Vector<T> &left_op, const T         &right_op);
template <class T> int operator> (const T         &left_op, const Vector<T> &right_op);

template <class T> int operator>=(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator>=(const Vector<T> &left_op, const T         &right_op);
template <class T> int operator>=(const T         &left_op, const Vector<T> &right_op);





// return 1 if all elements in vector fit within range

inline int checkRange(int lb, int ub, const Vector<int> &x);
inline int checkRange(int lb, int ub, const Vector<int> &x)
{
    if ( x.size() )
    {
        int i;

        for ( i = 0 ; i < x.size() ; i++ )
        {
            if ( ( x(i) < lb ) || ( x(i) > ub ) )
            {
                return 0;
            }
        }
    }

    return 1;
}


template <class S, class U> 
int shareBase(const Vector<S> &thus, const Vector<U> &that)
{ 
    return ( (void *) thus.bkref == (void *) that.bkref ); 
}



























// Now for the actual code (no *.o files with templates, which is annoying as hell)


template <class T>
retVector<T> &retVector<T>::reset(void)
{
    Vector<T> &gimme = *this;

    if ( !(gimme.nbase) && gimme.content )
    {
	MEMDEL(gimme.content);
        gimme.content = NULL;
    }

    if ( !(gimme.pbase) && gimme.pivot )
    {
        MEMDEL(gimme.pivot);
        gimme.pivot = NULL;
    }

    gimme.newln      = '\n';
    gimme.imoverhere = NULL;

    gimme.dsize = 0;
    gimme.nbase = 0;
    gimme.pbase = 1;

    gimme.iib = 0;
    gimme.iis = 0;

    gimme.bkref    = NULL;
    gimme.content  = NULL;
    gimme.ccontent = NULL;
    gimme.pivot    = NULL;

    gimme.bkref = NULL; // THIS WILL NEED TO BE FIXED src.bkref

    return *this;
}




template <class T>
void Vector<T>::fixbkreftree(const Vector<T> *newbkref)
{
    bkref = newbkref;

    return;
}


// Constructors and Destructors

template <class T>
Vector<T>::Vector(int size, const T *src)
{
    *thisindirect = this;

    indtmp = NULL;

    NiceAssert( size >= 0 );

    newln      = '\n';
    imoverhere = NULL;

    dsize = size;
//    fixsize = 0;

    nbase    = 0;
    pbase    = 1;

    iib = 0;
    iis = 1;

    bkref    = this;
    MEMNEW(content,DynArray<T>(dsize));
    ccontent = content;
    pivot    = cntintarray(dsize);

    NiceAssert( content );

    if ( src && size )
    {
	int i;

	for ( i = 0 ; i < size ; i++ )
	{
            (*this)("&",i) = src[i];
	}
    }

    return;
}

template <class T>
Vector<T>::Vector(const Vector<T> &src)
{
    *thisindirect = this;

    indtmp = NULL;

    newln      = src.newln;
    imoverhere = NULL;

    if ( src.imoverhere )
    {
        imoverhere = (src.overhere()).makeDup();
    }

    else if ( src.infsize() )
    {
        imoverhere = src.makeDup();
    }

    dsize = src.size();
//    fixsize = 0;

    nbase    = 0;
    pbase    = 1;

    iib = 0;
    iis = 1;

    bkref    = this;
    MEMNEW(content,DynArray<T>(dsize));
    ccontent = content;
    pivot    = cntintarray(dsize);

    NiceAssert( content );

    *this = src;

    return;
}

template <class T>
Vector<T>::Vector(const char *dummy, const Vector<T> &src)
{
    *thisindirect = this;

    indtmp = NULL;

    newln      = src.newln;
    imoverhere = NULL;

    (void) dummy;

    dsize = 0;
//    fixsize = 0;

    nbase    = 0;
    pbase    = 1;

    iib = 0;
    iis = 0;

    bkref    = this;
    content  = NULL;
    ccontent = NULL;
    pivot    = NULL;

    bkref = src.bkref;

    return;
}

template <class T>
Vector<T>::Vector(const char *dummy)
{
    *thisindirect = this;

    indtmp = NULL;

    newln      = '\n'; // THIS WILL NEED TO BE FIXED src.newln;
    imoverhere = NULL;

    (void) dummy;

    dsize = 0;
//    fixsize = 0;

    nbase    = 0;
    pbase    = 1;

    iib = 0;
    iis = 0;

    bkref    = this;
    content  = NULL;
    ccontent = NULL;
    pivot    = NULL;

    bkref = NULL; // THIS WILL NEED TO BE FIXED src.bkref

    return;
}

template <class T>
Vector<T>::Vector(const DynArray<T> *ccontentsrc)
{
    *thisindirect = this;

    indtmp = NULL;

    newln      = '\n';
    imoverhere = NULL;

    dsize = ccontentsrc->array_size();
//    fixsize = 0;

    nbase    = 1; // don't want ccontentsrc to be deleted!
    pbase    = 1;

    iib = 0;
    iis = 1;

    bkref    = this;
    content  = NULL;
    ccontent = ccontentsrc;
    pivot    = cntintarray(dsize);

    // NB: we actually want a *lot* of children here so that the circular
    // buffer is less likely to overrun.  Hence we do this directly.

    return;
}

template <class T>
Vector<T>::~Vector()
{
    if ( imoverhere )
    {
        MEMDEL(imoverhere);
        imoverhere = NULL;
    }

    if ( !nbase && content )
    {
	MEMDEL(content);
        content = NULL;
    }

    if ( !pbase && pivot )
    {
        MEMDEL(pivot);
        pivot = NULL;
    }

    if ( indtmp )
    {
        MEMDEL(indtmp);
        indtmp = NULL;
    }

    return;
}



// Assignment

extern int godebug;

template <class T>
Vector<T> &Vector<T>::assign(const Vector<T> &src)
{
    if ( shareBase(src) )
    {
        Vector<T> temp;

	temp  = src;
        assign(temp);
    }

    else if ( imoverhere && src.imoverhere )
    {
              Vector<T> &thisover = overhere();
        const Vector<T> &srcover  = src.overhere();

        if ( thisover.type() == srcover.type() )
        {
            thisover.assign(srcover);
        }

        else
        {
            MEMDEL(imoverhere);
            imoverhere = srcover.makeDup();
        }
    }

    else if ( imoverhere && src.infsize() )
    {
        Vector<T> &thisover = overhere();

        if ( thisover.type() == src.type() )
        {
            thisover.assign(src);
        }

        else
        {
            MEMDEL(imoverhere);
            imoverhere = src.makeDup();
        }
    }

    else if ( !imoverhere && src.imoverhere )
    {
        const Vector<T> &srcover  = src.overhere();

        resize(0);
        imoverhere = srcover.makeDup();
    }

    else if ( !imoverhere && src.infsize() )
    {
        resize(0);
        imoverhere = src.makeDup();
    }

    else
    {
        if ( imoverhere )
        {
            MEMDEL(imoverhere);
            imoverhere = NULL;
        }

	int srcsize = src.size();
	int i;

	if ( !nbase )
	{
	    resize(srcsize);

            if ( !(src.base()) && content && src.contentalloced() )
            {
                if ( src.contentarray_hold() )
                {
                    // Design decision: preallocation is duplicated

                    content->prealloc(src.contentarray_alloc());
                }
            }
	}

        NiceAssert( dsize == srcsize );

	if ( dsize )
	{
	    for ( i = 0 ; i < dsize ; i++ )
	    {
		(*this)("&",i) = src(i);
	    }
	}
    }

    return *this;
}

template <class S, class T>
inline int aresame(T *, S *);

template <class S, class T>
inline int aresame(T *, S *)
{
    return 0;
}

/*  This can't work.  The cast (const T &) will fail (eg if S == double), and it will fail silently!  Instead each case of this is specialised as required (see gentype.h, end of, for use)
template <class T>
template <class S>
Vector<T> &Vector<T>::castassign(const Vector<S> &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        MEMDEL(imoverhere);
        imoverhere = NULL;
    }

    if ( src.imoverhere )
    {
        T *a = NULL;
        S *b = NULL;

        if ( !aresame(a,b) )
        {
            throw("Can't do that");
        }

        else
        {
            Vector<S> *tmp;

            tmp = (*(src.imoverhere)).makeDup();

            imoverhere = (Vector<T> *) ((void *) tmp);
        }
    }

    else if ( src.infsize() )
    {
        T *a = NULL;
        S *b = NULL;

        if ( !aresame(a,b) )
        {
            throw("Can't do that");
        }

        else
        {
            Vector<S> *tmp;

            tmp = src.makeDup();

            imoverhere = (Vector<T> *) ((void *) tmp);
        }
    }

    else
    {
        if ( shareBase(src) )
        {
            Vector<S> temp;

            temp  = src;
            castassign(temp);
        }

        else
        {
	    int srcsize = src.size();
	    int i;

            if ( !nbase )
	    {
	        resize(srcsize);

                if ( !(src.base()) && content && src.contentalloced() )
                {
                    if ( src.contentarray_hold() )
                    {
                        // Design decision: preallocation is duplicated

                        content->prealloc(src.contentarray_alloc());
                    }
                }
	    }

            NiceAssert( dsize == srcsize );

    	    if ( dsize )
	    {
	        for ( i = 0 ; i < dsize ; i++ )
	        {
                    (*this)("&",i) = (const T &) src(i);
                }
	    }
	}
    }

    return *this;
}
*/

template <class T>
Vector<T> &Vector<T>::assign(const T &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        overhere().assign(src);
    }

    else
    {
        int i;

        if ( dsize )
        {
	    for ( i = 0 ; i < dsize ; i++ )
	    {
	        (*this)("&",i) = src;
	    }
        }
    }

    return *this;
}


// Basic operations.

template <class T>
Vector<T> &Vector<T>::ident(void)
{
    NiceAssert( !infsize() );

    if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            setident((*this)("&",i));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::zero(void)
{
    if ( imoverhere )
    {
        overhere().zero();
    }

    else if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            setzero((*this)("&",i));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::zeropassive(void)
{
    NiceAssert( !infsize() );

    if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            setzeropassive((*this)("&",i));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::zero(int i)
{
    NiceAssert( !infsize() );
    NiceAssert( i >= 0 );
    NiceAssert( i < size() );

    setzero((*this)("&",i));

    return *this;
}

template <class T>
Vector<T> &Vector<T>::zeromost(int j)
{
    NiceAssert( !infsize() );

    if ( j == -1 )
    {
        return *this;
    }

    if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            if ( i != j )
            {
                setzero((*this)("&",i));
            }
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::softzero(void)
{
    NiceAssert( !infsize() );

    return zero();
}

template <class T>
Vector<T> &Vector<T>::offset(int amoff)
{
    NiceAssert( !infsize() );
    NiceAssert( amoff >= -size() );

    if ( amoff )
    {
        int i;

        NiceAssert( !nbase );
        NiceAssert( content );
//        NiceAssert( !fixsize );

        if ( dsize+amoff && ( amoff < 0 ) )
        {
            for ( i = 0 ; i < dsize+amoff ; i++ )
            {
                qswap((*this)("&",i-amoff),(*this)("&",i));
            }
        }

        dsize = dsize+amoff;

        (*content).resize(dsize);
        pivot = cntintarray(dsize);

        if ( dsize && ( amoff > 0 ) )
        {
            for ( i = dsize-1 ; i >= amoff ; i-- )
            {
                qswap((*this)("&",i-amoff),(*this)("&",i));
            }

            for ( i = amoff-1 ; i >= 0 ; i-- )
            {
                zero(i);
            }
        }
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::posate(void)
{
    if ( imoverhere )
    {
        overhere().posate();
    }

    else if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            setposate((*this)("&",i));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::negate(void)
{
    if ( imoverhere )
    {
        overhere().negate();
    }

    else if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            setnegate((*this)("&",i));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::conj(void)
{
    static T dummy;

    if ( !isitadouble(dummy) )
    {
        if ( imoverhere )
        {
            overhere().conj();
        }

        else if ( dsize )
        {
	    int i;

            for ( i = 0 ; i < dsize ; i++ )
	    {
                setconj((*this)("&",i));
            }
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::rand(void)
{
    if ( imoverhere )
    {
        overhere().rand();
    }

    else if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            setrand((*this)("&",i));
	}
    }

    return *this;
}



// Access:

template <class T>
Vector<T> &Vector<T>::operator()(const char *dummy, const Vector<int> &i, retVector<T> &res)
{
    NiceAssert( !infsize() );
    NiceAssert( checkRange(0,dsize-1,i) );

    res.reset();

    (void) dummy;

    res.newln = newln;
    res.bkref = bkref;

    if ( ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) ) && !(i.base()) )
    {
	res.dsize   = i.size();
//	res.fixsize = 0;

        res.nbase = 1;
        res.pbase = 1;

	res.iib = 0;
	res.iis = 1;

	res.content  = content;
	res.ccontent = ccontent;

	if ( res.dsize )
	{
            res.pivot = &(i.grabcontentdirect());
	}

	else
	{
            res.pivot = cntintarray(0);
	}
    }

    else
    {
	res.dsize   = i.size();
//	res.fixsize = 0;

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dsize));

        NiceAssert( temppivot );

	if ( res.dsize )
	{
	    int ii;

	    for ( ii = 0 ; ii < res.dsize ; ii++ )
	    {
		(*temppivot)("&",ii) = (*pivot)(iib+(iis*(i(ii))));
	    }
	}

	res.nbase = 1;
	res.pbase = 0;

	res.iib = 0;
	res.iis = 1;

	res.content  = content;
	res.ccontent = ccontent;
        res.pivot    = temppivot;
    }

    return res;
}

template <class T>
Vector<T> &Vector<T>::operator()(const char *dummy, const DynArray<int> &i, int isize, retVector<T> &res)
{
    NiceAssert( !infsize() );
    NiceAssert( isize >= 0 );

#ifndef NDEBUG
    if ( isize )
    {
	int ij;

	for ( ij = 0 ; ij < isize ; ij++ )
	{
            NiceAssert( i(ij) >= 0 );
            NiceAssert( i(ij) < dsize );
	}
    }
#endif

    res.reset();

    (void) dummy;

    res.newln = newln;
    res.bkref = bkref;

    if ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) )
    {
	res.dsize   = isize;
//	res.fixsize = 0;

	res.nbase = 1;
	res.pbase = 1;

	res.iib = 0;
	res.iis = 1;

	res.content  = content;
	res.ccontent = ccontent;

	if ( res.dsize )
	{
	    res.pivot = &i;
	}

	else
	{
            res.pivot = cntintarray(0);
	}
    }

    else
    {
	res.dsize   = isize;
//	res.fixsize = 0;

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dsize));

        NiceAssert( temppivot );

	if ( res.dsize )
	{
	    int ii;

	    for ( ii = 0 ; ii < res.dsize ; ii++ )
	    {
		(*temppivot)("&",ii) = (*pivot)(iib+(iis*(i(ii))));
	    }
	}

	res.nbase = 1;
	res.pbase = 0;

	res.iib = 0;
	res.iis = 1;

	res.content  = content;
	res.ccontent = ccontent;
        res.pivot    = temppivot;
    }

    return res;
}

template <class T>
Vector<T> &Vector<T>::operator()(const char *dummy, const DynArray<int> &i, int ib, int is, int im, retVector<T> &res)
{
    NiceAssert( !infsize() );

    int isize = ( ib > im ) ? 0 : ( ((im-ib)/is)+1 );

    NiceAssert( isize >= 0 );

#ifndef NDEBUG
    if ( isize )
    {
	int ij;

	for ( ij = 0 ; ij < isize ; ij++ )
	{
            NiceAssert( i(ij) >= 0 );
            NiceAssert( i(ij) < dsize );
	}
    }
#endif

    res.reset();

    (void) dummy;

    res.newln = newln;
    res.bkref = bkref;

    if ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) )
    {
	res.dsize   = isize;
//	res.fixsize = 0;

	res.nbase = 1;
	res.pbase = 1;

        // iib + iis.( ib + is.i )
        // = iib + iis.ib + iis.is.i
        // = ( iib + iis.ib ) + ( iis.is ).i

        res.iib = iib+(iis*ib);
        res.iis = iis*is;

	res.content  = content;
	res.ccontent = ccontent;

	if ( res.dsize )
	{
	    res.pivot = &i;
	}

	else
	{
            res.pivot = cntintarray(0);
	}
    }

    else
    {
	res.dsize   = isize;
//	res.fixsize = 0;

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dsize));

        NiceAssert( temppivot );

	if ( res.dsize )
	{
	    int ii,iii;

	    for ( ii = 0 ; ii < res.dsize ; ii++ )
	    {
                iii = ib+(is*ii);

		(*temppivot)("&",ii) = (*pivot)(iib+(iis*(i(iii))));
	    }
	}

	res.nbase = 1;
	res.pbase = 0;

	res.iib = 0;
	res.iis = 1;

	res.content  = content;
	res.ccontent = ccontent;
        res.pivot    = temppivot;
    }

    return res;
}

template <class T>
Vector<T> &Vector<T>::operator()(const char *dummy, int ib, int is, int im, retVector<T> &res)
{
    NiceAssert( !infsize() );
    NiceAssert( ib >= 0 );
    NiceAssert( is > 0 );
    NiceAssert( im < dsize );

    res.reset();

    (void) dummy;

    res.newln = newln;
    res.bkref = bkref;

    res.dsize   = ( ib > im ) ? 0 : ( ((im-ib)/is)+1 );
//  res.fixsize = 0;

    res.nbase = 1;
    res.pbase = 1;

    // iib + iis.( ib + is.i )
    // = iib + iis.ib + iis.is.i
    // = ( iib + iis.ib ) + ( iis.is ).i

    res.iib = iib+(iis*ib);
    res.iis = iis*is;

    res.content  = content;
    res.ccontent = ccontent;
    res.pivot    = pivot;

    return res;
}

template <class T>
const Vector<T> &Vector<T>::operator()(const Vector<int> &i, retVector<T> &res) const
{
    NiceAssert( !infsize() );
    NiceAssert( checkRange(0,dsize-1,i) );

    res.reset();

    res.newln = newln;
    res.bkref = bkref;

    if ( ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) ) && !(i.base()) )
    {
	res.dsize   = i.size();
//	res.fixsize = 0;

	res.nbase = 1;
	res.pbase = 1;

	res.iib = 0;
	res.iis = 1;

	res.content  = NULL;
	res.ccontent = ccontent;

	if ( res.dsize )
	{
            res.pivot = &(i.grabcontentdirect());
	}

	else
	{
            res.pivot = cntintarray(0);
	}
    }

    else
    {
	res.dsize   = i.size();
//	res.fixsize = 0;

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dsize));

        NiceAssert( temppivot );

	if ( res.dsize )
	{
	    int ii;

	    for ( ii = 0 ; ii < res.dsize ; ii++ )
	    {
		(*temppivot)("&",ii) = (*pivot)(iib+(iis*(i(ii))));
	    }
	}

	res.nbase = 1;
	res.pbase = 0;

	res.iib = 0;
	res.iis = 1;

	res.content  = NULL;
	res.ccontent = ccontent;
        res.pivot    = temppivot;
    }

    return res;
}

template <class T>
const Vector<T> &Vector<T>::operator()(const DynArray<int> &i, int isize, retVector<T> &res) const
{
    NiceAssert( !infsize() );
    NiceAssert( isize >= 0 );

#ifndef NDEBUG
    if ( isize )
    {
	int ij;

	for ( ij = 0 ; ij < isize ; ij++ )
	{
            NiceAssert( i(ij) >= 0 );
            NiceAssert( i(ij) < dsize );
	}
    }
#endif

    res.reset();

    res.newln = newln;
    res.bkref = bkref;

    if ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) )
    {
	res.dsize   = isize;
//	res.fixsize = 0;

	res.nbase = 1;
	res.pbase = 1;

	res.iib = 0;
	res.iis = 1;

	res.content  = NULL;
	res.ccontent = ccontent;

	if ( res.dsize )
	{
	    res.pivot = &i;
	}

	else
	{
            res.pivot = cntintarray(0);
	}
    }

    else
    {
	res.dsize   = isize;
//	res.fixsize = 0;

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dsize));

        NiceAssert( temppivot );

	if ( res.dsize )
	{
	    int ii;

	    for ( ii = 0 ; ii < res.dsize ; ii++ )
	    {
		(*temppivot)("&",ii) = (*pivot)(iib+(iis*(i(ii))));
	    }
	}

	res.nbase = 1;
	res.pbase = 0;

	res.iib = 0;
	res.iis = 1;

	res.content  = NULL;
	res.ccontent = ccontent;
        res.pivot    = temppivot;
    }

    return res;
}

template <class T>
const Vector<T> &Vector<T>::operator()(const DynArray<int> &i, int ib, int is, int im, retVector<T> &res) const
{
    NiceAssert( !infsize() );

    int isize = ( ib > im ) ? 0 : ( ((im-ib)/is)+1 );

    NiceAssert( isize >= 0 );

#ifndef NDEBUG
    if ( isize )
    {
	int ij;

	for ( ij = 0 ; ij < isize ; ij++ )
	{
            NiceAssert( i(ij) >= 0 );
            NiceAssert( i(ij) < dsize );
	}
    }
#endif

    res.reset();

    res.newln = newln;
    res.bkref = bkref;

    if ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) )
    {
	res.dsize   = isize;
//	res.fixsize = 0;

	res.nbase = 1;
	res.pbase = 1;

        // iib + iis.( ib + is.i )
        // = iib + iis.ib + iis.is.i
        // = ( iib + iis.ib ) + ( iis.is ).i

        res.iib = iib+(iis*ib);
        res.iis = iis*is;

	res.content  = content;
	res.ccontent = ccontent;

	if ( res.dsize )
	{
	    res.pivot = &i;
	}

	else
	{
            res.pivot = cntintarray(0);
	}
    }

    else
    {
	res.dsize   = isize;
//	res.fixsize = 0;

	DynArray<int> *temppivot;

	MEMNEW(temppivot,DynArray<int>(res.dsize));

        NiceAssert( temppivot );

	if ( res.dsize )
	{
	    int ii,iii;

	    for ( ii = 0 ; ii < res.dsize ; ii++ )
	    {
                iii = ib+(is*ii);

		(*temppivot)("&",ii) = (*pivot)(iib+(iis*(i(iii))));
	    }
	}

	res.nbase = 1;
	res.pbase = 0;

	res.iib = 0;
	res.iis = 1;

	res.content  = content;
	res.ccontent = ccontent;
        res.pivot    = temppivot;
    }

    return res;
}

template <class T>
const Vector<T> &Vector<T>::operator()(int ib, int is, int im, retVector<T> &res) const
{
    NiceAssert( !infsize() );
    NiceAssert( ib >= 0 );
    NiceAssert( is > 0 );
    NiceAssert( im < dsize );

    res.reset();

    res.newln = newln;
    res.bkref = bkref;

    res.dsize   = ( ib > im ) ? 0 : ( ((im-ib)/is)+1 );
//    res.fixsize = 0;

    res.nbase = 1;
    res.pbase = 1;

    // iib + iis.( ib + is.i )
    // = iib + iis.ib + iis.is.i
    // = ( iib + iis.ib ) + ( iis.is ).i

    res.iib = iib+(iis*ib);
    res.iis = iis*is;

    res.content  = NULL;
    res.ccontent = ccontent;
    res.pivot    = pivot;

    return res;
}


























// Scaled addition:

template <class T>
template <class S> Vector<T> &Vector<T>::scaleAdd(const S &a, const Vector<T> &b)
{
    NiceAssert( !infsize() );

    if ( shareBase(b) )
    {
        Vector<T> temp(b);

        scaleAdd(a,temp);
    }

    else
    {
        NiceAssert( ( size() == b.size() ) || !size() || !(b.size()) );

	if ( !size() && b.size() )
	{
	    resize(b.size());
            zero();
	}

	if ( size() && b.size() )
	{
	    int i;

	    for ( i = 0 ; i < size() ; i++ )
	    {
		(*this)("&",i) += (a*(b(i)));
	    }
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scaleAddR(const Vector<T> &a, const S &b)
{
    NiceAssert( !infsize() );

    if ( shareBase(a) )
    {
        Vector<T> temp(a);

        scaleAddR(temp,b);
    }

    else
    {
        NiceAssert( ( size() == a.size() ) || !size() || !(a.size()) );

	if ( !size() && a.size() )
	{
	    resize(a.size());
            zero();
	}

	if ( size() && a.size() )
	{
	    int i;

	    for ( i = 0 ; i < size() ; i++ )
	    {
		(*this)("&",i) += ((a(i))*b);
	    }
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scaleAddB(const T &a, const Vector<S> &b)
{
    NiceAssert( !infsize() );
    NiceAssert( ( size() == b.size() ) || !size() || !(b.size()) );

    if ( b.size() && !size() )
    {
	resize(b.size());
        zero();
    }

    if ( size() )
    {
	int i;

	for ( i = 0 ; i < size() ; i++ )
	{
	    (*this)("&",i) += (a*(b(i)));
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scaleAddBR(const Vector<S> &a, const T &b)
{
    NiceAssert( !infsize() );
    NiceAssert( ( size() == a.size() ) || !size() || !(a.size()) );

    if ( a.size() && !size() )
    {
	resize(a.size());
        zero();
    }

    if ( size() )
    {
	int i;

	for ( i = 0 ; i < size() ; i++ )
	{
	    (*this)("&",i) += ((a(i))*b);
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scale(const S &a)
{
    NiceAssert( !infsize() );

    if ( size() )
    {
	int i;

	for ( i = 0 ; i < size() ; i++ )
	{
	    (*this)("&",i) *= a;
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scale(const Vector<S> &a)
{
    NiceAssert( !infsize() );
    NiceAssert( ( size() == a.size() ) || !(size()) || !(a.size()) );

    if ( !size() && a.size() )
    {
	resize(a.size());
        zero();
    }

    if ( size() && a.size() )
    {
	int i;

	for ( i = 0 ; i < size() ; i++ )
	{
	    (*this)("&",i) *= a(i);
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::lscale(const S &a)
{
    NiceAssert( !infsize() );

    if ( size() )
    {
	int i;

	for ( i = 0 ; i < size() ; i++ )
	{
            rightmult(a,(*this)("&",i));
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::lscale(const Vector<S> &a)
{
    NiceAssert( !infsize() );
    NiceAssert( ( size() == a.size() ) || !(size()) || !(a.size()) );

    if ( !size() && a.size() )
    {
	resize(a.size());
        zero();
    }

    if ( size() && a.size() )
    {
	int i;

	for ( i = 0 ; i < size() ; i++ )
	{
            rightmult(a(i),(*this)("&",i));
	}
    }

    return *this;
}




// Various swap functions

template <class T>
Vector<T> &Vector<T>::blockswap(int i, int j)
{
    NiceAssert( !infsize() );
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < size() );
    NiceAssert( j >= 0 );
    NiceAssert( j < size() );

    // blockswap  ( i < j ): ( c ) (i)          ( c ) (i)
    //                       ( e ) (1)    ->    ( d ) (j-i)
    //                       ( d ) (j-i)        ( e ) (1)
    //                       ( f ) (...)        ( f ) (...)
    //
    // blockswap  ( i > j ): ( c ) (j)          ( c ) (j)
    //                       ( d ) (i-j)  ->    ( e ) (1)
    //                       ( e ) (1)          ( d ) (i-j)
    //                       ( f ) (...)        ( f ) (...)

    if ( i > j )
    {
	int k;

	for ( k = i ; k > j ; k-- )
	{
	    qswap((*this)("&",k),(*this)("&",k-1));
	}
    }

    else if ( i < j )
    {
	int k;

	for ( k = i ; k < j ; k++ )
	{
            qswap((*this)("&",k),(*this)("&",k+1));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::squareswap(int i, int j)
{
    NiceAssert( !infsize() );
    NiceAssert( i >= 0 );
    NiceAssert( i < size() );
    NiceAssert( j >= 0 );
    NiceAssert( j < size() );

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

    if ( i != j )
    {
	qswap((*this)("&",i),(*this)("&",j));
    }

    return *this;
}

/*
template <class T>
Vector<T> &Vector<T>::blockswap(int i, int j)
{
    NiceAssert( !infsize() );
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < dsize );
    NiceAssert( j >= 0 );
    NiceAssert( j < dsize );

    if ( i > j )
    {
        NiceAssert( content );

	T temp;
	int k;

        temp = (*this)(i);

	for ( k = i ; k > j ; k-- )
	{
            (*this)("&",k) = (*this)(k-1);
	}

        (*this)(j) = temp;
    }

    else if ( i < j )
    {
        NiceAssert( content );

	T temp;;
	int k;

	temp = (*this)(i);

	for ( k = i ; k < j ; k++ )
	{
            (*this)("&",k) = (*this)(k+1);
	}

        (*this)(j) = temp;
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::squareswap(int i, int j)
{
    NiceAssert( !infsize() );
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < dsize );
    NiceAssert( j >= 0 );
    NiceAssert( j < dsize );

    if ( i != j )
    {
        NiceAssert( content );

	T temp;

	temp           = (*this)(i);
	(*this)("&",i) = (*this)(j);
        (*this)("&",j) = temp;
    }

    return *this;
}
*/



// Add and remove element functions

template <class T>
Vector<T> &Vector<T>::add(int i)
{
    NiceAssert( !infsize() );
    NiceAssert( !nbase );
//    NiceAssert( !fixsize );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dsize );
    NiceAssert( content );

    dsize++;

    NiceAssert( content );

    (*content).resize(dsize);
    pivot = cntintarray(dsize);

    blockswap(dsize-1,i);

    return *this;
}

template <class T>
Vector<T> &Vector<T>::addpad(int i, int num)
{
    NiceAssert( !infsize() );
    NiceAssert( num >= 0 );

    while ( num > 0 )
    {
        add(i);
        i++;
        num--;
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::remove(int i)
{
    NiceAssert( !infsize() );
    NiceAssert( !nbase );
//    NiceAssert( !fixsize );
    NiceAssert( i >= 0 );
    NiceAssert( i < dsize );
    NiceAssert( content );

    blockswap(i,dsize-1);

    dsize--;
    pivot = cntintarray(dsize);

    (*content).resize(dsize);

    return *this;
}

template <class T>
Vector<T> &Vector<T>::remove(const Vector<int> &i)
{
    NiceAssert( !infsize() );

    Vector<int> ii(i);
    int j;

    while ( ii.size() )
    {
        remove(max(ii,j));
        ii.remove(j);
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::resize(int tsize)
{
    NiceAssert( !infsize() || !tsize );
    NiceAssert( tsize >= 0 );

    if ( dsize != tsize )
    {
        NiceAssert( !nbase );
//        NiceAssert( !fixsize );
        NiceAssert( content );

        dsize = tsize;

        (*content).resize(dsize);
        pivot = cntintarray(dsize);
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::pad(int n)
{
    NiceAssert( !infsize() );

    if ( n > 0 )
    {
        NiceAssert( content );

        int i;

        resize(dsize+n);

        for ( i = dsize-n ; i < dsize ; i++ )
        {
            setzero((*this)("&",i));
        }
    }

    else if ( n < 0 )
    {
        resize(dsize+n);
    }

    return *this;
}

template <class T>
void Vector<T>::prealloc(int newallocsize)
{
    if ( imoverhere )
    {
        overhere().prealloc(newallocsize);
    }

    else 
    {
        NiceAssert( !nbase );
        NiceAssert( ( newallocsize >= 0 ) || ( newallocsize == -1 ) );
        NiceAssert( content );

        (*content).prealloc(newallocsize);
    }

    return;
}

template <class T>
void Vector<T>::useStandardAllocation(void)
{
    if ( imoverhere )
    {
        overhere().useStandardAllocation();
    }

    else 
    {
        NiceAssert( content );

        (*content).useStandardAllocation();
    }

    return;
}

template <class T>
void Vector<T>::useTightAllocation(void)
{
    if ( imoverhere )
    {
        overhere().useTightAllocation();
    }

    else 
    {
        NiceAssert( content );

        (*content).useTightAllocation();
    }

    return;
}

template <class T>
void Vector<T>::useSlackAllocation(void)
{
    if ( imoverhere )
    {
        overhere().useSlackAllocation();
    }

    else 
    {
        NiceAssert( content );

        (*content).useSlackAllocation();
    }

    return;
}

template <class T>
void Vector<T>::applyOnAll(void (*fn)(T &, int), int argx)
{
    NiceAssert( !infsize() );
    NiceAssert( !nbase );
    NiceAssert( content );

    (*content).applyOnAll(fn,argx);

    return;
}

template <class T>
Vector<T> &Vector<T>::append(int i, const T &a)
{
    NiceAssert( !infsize() );
    NiceAssert( i >= size() );

    resize(i+1);

    (*this)("&",i) = a;

    return *this;
}

template <class T>
Vector<T> &Vector<T>::append(int i, const Vector<T> &a)
{
    NiceAssert( !infsize() );
    NiceAssert( i >= size() );

    if ( a.size() )
    {
        resize(i+(a.size()));

        int j;

        for ( j = 0 ; j < a.size() ; j++ )
        {
            (*this)("&",i+j) = a(j);
        }
    }

    return *this;
}


// Function application

template <class T>
Vector<T> &Vector<T>::applyon(T (*fn)(T))
{
    if ( imoverhere )
    {
        overhere().applyon(fn);
    }

    else if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            (*this)("&",i) = fn((*this)(i));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::applyon(T (*fn)(const T &))
{
    if ( imoverhere )
    {
        overhere().applyon(fn);
    }

    else if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            (*this)("&",i) = fn((*this)(i));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::applyon(T &(*fn)(T &))
{
    if ( imoverhere )
    {
        overhere().applyon(fn);
    }

    else if ( dsize )
    {
        int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            fn((*this)("&",i));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::applyon(T (*fn)(T, const void *), const void *a)
{
    if ( imoverhere )
    {
        overhere().applyon(fn,a);
    }

    else if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            (*this)("&",i) = fn((*this)(i),a);
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::applyon(T (*fn)(const T &, const void *), const void *a)
{
    if ( imoverhere )
    {
        overhere().applyon(fn,a);
    }

    else if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            (*this)("&",i) = fn((*this)(i),a);
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::applyon(T &(*fn)(T &, const void *), const void *a)
{
    if ( imoverhere )
    {
        overhere().applyon(fn,a);
    }

    else if ( dsize )
    {
	int i;

	for ( i = 0 ; i < dsize ; i++ )
	{
            fn((*this)("&",i),a);
	}
    }

    return *this;
}


















































































































template <class T>
const T &max(T &res, const Vector<T> &a, const Vector<T> &b, int &ii)
{
    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() );

    T locval;

    ii = 0;

    res  = a(0);
    res -= b(0);

    if ( a.size() > 1 )
    {
	int i;

        for ( i = 1 ; i < a.size() ; i++ )
	{
	    locval  = a(i);
	    locval -= b(i);

            if ( locval > res )
	    {
                res = locval;
                ii = i;
	    }
	}
    }

    return res;
}


template <class T>
const T &max(const Vector<T> &a, int &ii)
{
    NiceAssert( a.size() );

    ii = 0;

    if ( a.size() > 1 )
    {
	int i;

        for ( i = 0 ; i < a.size() ; i++ )
	{
            if ( !i || ( a(i) > a(ii) ) )
	    {
                ii = i;
	    }
	}


    }

    return a(ii);
}

template <class T>
T max(const Vector<T> &a, const Vector<T> &b, int &ii)
{
    T res;

    return max(res,a,b,ii);
}



























template <class T>
const T &min(T &res, const Vector<T> &a, const Vector<T> &b, int &ii)
{
    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() );

    T locval;

    ii = 0;

    res  = a(0);
    res -= b(0);

    if ( a.size() > 1 )
    {
	int i;

        for ( i = 1 ; i < a.size() ; i++ )
	{
	    locval  = a(i);
	    locval -= b(i);

            if ( locval < res )
	    {
                res = locval;
                ii = i;
	    }
	}
    }

    return res;
}

template <class T>
const T &min(const Vector<T> &a, int &ii)
{
    NiceAssert( a.size() );

    ii = 0;

    if ( a.size() > 1 )
    {
        int i;

        for ( i = 0 ; i < a.size() ; i++ )
	{
            if ( !i || ( a(i) < a(ii) ) )
	    {
                ii = i;
	    }
	}


    }

    return a(ii);
}

template <class T>
T min(const Vector<T> &a, const Vector<T> &b, int &ii)
{
    T res;

    return min(res,a,b,ii);
}


























template <class T>
const T &maxabs(T &res, const Vector<T> &a, int &ii)
{
    NiceAssert( a.size() );

    ii = 0;

    if ( a.size() > 1 )
    {
	int i;

        for ( i = 0 ; i < a.size() ; i++ )
	{
            if ( !i || ( abs2(a(i)) > abs2(a(ii)) ) )
	    {
                ii = i;
	    }
	}


    }

    res = abs2(a(ii));

    return res;
}

template <class T>
T maxabs(const Vector<T> &a, int &ii)
{
    T res;

    return maxabs(res,a,ii);
}

template <class T>
const T &minabs(T &res, const Vector<T> &a, int &ii)
{
    NiceAssert( a.size() );

    ii = 0;

    if ( a.size() > 1 )
    {
	int i;

        for ( i = 0 ; i < a.size() ; i++ )
	{
            if ( !i || ( abs2(a(i)) < abs2(a(ii)) ) )
	    {
                ii = i;
	    }
	}


    }

    res = abs2(a(ii));

    return res;
}

template <class T>
T minabs(const Vector<T> &a, int &ii)
{
    T res;

    return minabs(res,a,ii);
}

template <class T>
const T &sqabsmax(T &res, const Vector<T> &a)
{
    NiceAssert( a.size() );

    int ii = 0;

    if ( a.size() > 1 )
    {
	int i;

        for ( i = 0 ; i < a.size() ; i++ )
	{
            if ( !i || ( norm2(a(i)) > norm2(a(ii)) ) )
	    {
                ii = i;
	    }
	}


    }

    res = norm2(a(ii));

    return res;
}

template <class T>
T sqabsmax(const Vector<T> &a)
{
    T res;

    return sqabsmax(res,a);
}

template <class T>
const T &sqabsmin(T &res, const Vector<T> &a)
{
    NiceAssert( a.size() );

    int ii = 0;

    if ( a.size() > 1 )
    {
	int i;

        for ( i = 0 ; i < a.size() ; i++ )
	{
            if ( !i || ( norm2(a(i)) < norm2(a(ii)) ) )
	    {
                ii = i;
	    }
	}


    }

    res = norm2(a(ii));

    return res;
}

template <class T>
T sqabsmin(const Vector<T> &a)
{
    T res;

    return sqabsmin(res,a);
}

template <class T>
T sum(const Vector<T> &a)
{
    T res;

    return sum(res,a);
}

template <class T>
T sqsum(const Vector<T> &a)
{
    T res;

    return sqsum(res,a);
}

template <class T>
T orall(const Vector<T> &a)
{
    T res;

    return orall(res,a);
}

template <class T>
T andall(const Vector<T> &a)
{
    T res;

    return andall(res,a);
}

template <class T>
const T &sum(T &res, const Vector<T> &a)
{
    if ( a.size() == 1 )
    {
        res = a(0);
    }

    else if ( a.size() > 1 )
    {
        res = a(0);

        if ( a.size() > 1 )
        {
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
            {
                res += a(i);
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
const T &sum(T &res, const Vector<T> &a, const Vector<double> &weights)
{
    if ( a.size() )
    {
        res  = a(0);
        res *= weights(0);

        if ( a.size() > 1 )
        {
            int i;
            T temp;

            for ( i = 1 ; i < a.size() ; i++ )
            {
                temp  = a(i);
                temp *= weights(i);

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

template <class S, class T>
const T &sumb(T &result, const Vector<S> &left_op, const Vector<T> &right_op)
{
    if ( left_op.infsize() || right_op.infsize() )
    {
        throw("Sorry, this aint defined 'ere");

        return result;
    }

    NiceAssert( left_op.size() == right_op.size() );

    T temp;

    setzero(result);
    setzero(temp);

    if ( left_op.size() )
    {
        result = right_op(0);
        rightmult(left_op(0),result);

        if ( left_op.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < left_op.size() ; i++ )
	    {
                temp = right_op(i);
                rightmult(left_op(i),temp);

                result += temp;
	    }
	}
    }

    return postProInnerProd(result);
}



template <class T>
const T &sqsum(T &res, const Vector<T> &a)
{
    T temp;

    if ( a.size() )
    {
        res =  a(0);
        res *= a(0);

        if ( a.size() > 1 )
        {
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
            {
                temp =  a(i);
                temp *= a(i);

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
const T &orall(T &res, const Vector<T> &a)
{
    T temp;

    if ( a.size() )
    {
        res =  a(0);

        if ( a.size() > 1 )
        {
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
            {
                res |= a(i);
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
const T &andall(T &res, const Vector<T> &a)
{
    T temp;

    if ( a.size() )
    {
        res =  a(0);

        if ( a.size() > 1 )
        {
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
            {
                res &= a(i);
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
const T &prod(T &res, const Vector<T> &a)
{
    if ( a.size() )
    {
        res = a(0);

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                res *= a(i);
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
T prod(const Vector<T> &a)
{
    T res;

    return prod(res,a);
}

template <class T>
const T &Prod(T &res, const Vector<T> &a)
{
    if ( a.size() )
    {
        res = a(a.size()-1);

        if ( a.size() > 1 )
	{
            int i;

            for ( i = a.size()-2 ; i >= 0 ; i-- )
	    {
                res *= a(i);
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
T Prod(const Vector<T> &a)
{
    T res;

    return Prod(res,a);
}

template <class T>
const T &mean(T &res, const Vector<T> &a)
{
    if ( a.size() == 1 )
    {
        res = a(zeroint());
    }

    else if ( a.size() > 0 )
    {
        sum(res,a);
        res *= 1/((double) a.size());
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
const T &mean(T &res, const Vector<T> &a, const Vector<double> &weights)
{
    NiceAssert( a.size() == weights.size() );

    if ( a.size() )
    {
        sum(res,a,weights);
        res *= 1/((double) a.size());
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
T mean(const Vector<T> &a)
{
    T res;

    return mean(res,a);
}

template <class T>
const T &sqmean(T &res, const Vector<T> &a)
{
    if ( a.size() )
    {
        sqsum(res,a);
        res *= 1/((double) a.size());
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
T sqmean(const Vector<T> &a)
{
    T res;

    return sqmean(res,a);
}

template <class T>
const T &vari(T &res, const Vector<T> &a)
{
    // mean(a) = 1/N sum_i a_i
    // sqmean(a) = 1/N sum_i a_i^2
    //
    // vari(a) = 1/N sum_i ( a_i - mean(a) )^2
    //         = 1/N sum_i a_i^2 - 2 mean(a) 1/N sum_i a_i + mean(a)^2 1/N sum_i 1
    //         = sqmean(a) - 2 mean(a)^2 + mean(a)^2
    //         = sqmean(a) - mean(a)^2

    T ameansq;

    res =  sqmean(a);
    ameansq =  mean(a);
    ameansq *= ameansq;
    res -= ameansq;

    return res;
}

template <class T>
T vari(const Vector<T> &a)
{
    T res;

    return vari(res,a);
}

template <class T>
const T &stdev(T &res, const Vector<T> &a)
{
    res = sqrt(vari(a));

    return res;
}

template <class T>
T stdev(const Vector<T> &a)
{
    T res;

    return stdev(res,a);
}

template <class T>
const T &median(const Vector<T> &a, int &ii)
{
    ii = 0;

    if ( a.size() == 1 )
    {


        return a(0);
    }

    else if ( a.size() > 1 )
    {
        // Aim: a(outdex) should be arranged from largest to smallest

        Vector<int> outdex;

        int i,j;

        for ( i = a.size()-1 ; i >= 0 ; i-- )
        {
            j = 0;

            if ( outdex.size() )
            {
                for ( j = 0 ; j < outdex.size() ; j++ )
                {
                    if ( a(outdex(j)) <= a(i) )
                    {
                        break;
                    }
                }
            }

            outdex.add(j);
            outdex("&",j) = i;
        }

        j = ii;

        ii = outdex(a.size()/2);


        return a(j);
    }

    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    svmvolatile static int frun = 1;
    svmvolatile static T defres;

    ii = 0;


    if ( frun )
    {
        setzero(const_cast<T &>(defres));
        frun = 0;
    }

    svm_mutex_unlock(eyelock);

    return const_cast<T &>(defres);
}

template <class S> Vector<S> angle(const Vector<S> &a)
{
    Vector<S> temp(a);

    double tempabs = abs2(temp);

    if ( tempabs != 0.0 )
    {
        temp.scale(1/tempabs);
    }

    return temp;
}

template <class S> Vector<double> eabs2(const Vector<S> &a)
{
    Vector<double> temp(a.size());

    if ( temp.size() )
    {
        int i;

        for ( i = 0 ; i < temp.size() ; i++ )
        {
            temp("&",i) = abs2(temp(i));
        }
    }

    return temp;
}

template <class S> Vector<S> vangle(const Vector<S> &a, const Vector<S> &defsign)
{
    Vector<S> temp(a);

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

template <class S> double abs2(const Vector<S> &a)
{
    return sqrt(norm2(a));
}

template <class S> double absd(const Vector<S> &a)
{
    return abs2(a);
}

template <class S> double normd(const Vector<S> &a)
{
    return norm2(a);
}

template <class S> double abs1(const Vector<S> &a)
{
    return norm1(a);
}

template <class S> double absp(const Vector<S> &a, double p)
{
    return pow(normp(a,p),1/p);
}

template <class S> double absinf(const Vector<S> &a)
{
    double maxval = 0;
    double locval;

    if ( a.infsize() )
    {
        maxval = a.absinf();
    }

    else if ( a.size() )
    {
        maxval = absinf(a(0));

        if ( a.size() > 1 )
	{
	    int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
		locval = absinf(a(i));

		if ( locval > maxval )
		{
		    maxval = locval;
		}
	    }
	}
    }

    return maxval;
}

template <class S> double norm1(const Vector<S> &a)
{
    int i;
    double result = 0;

    if ( a.infsize() )
    {
        result = a.norm1();
    }

    else if ( a.size() )
    {
	result = norm1(a(0));

	if ( a.size() > 1 )
	{
	    for ( i = 1 ; i < a.size() ; i++ )
	    {
		result += (double) norm1(a(i));
	    }
	}
    }

    return result;
}

template <class S> double norm2(const Vector<S> &a)
{
    int i;
    double result = 0;

    if ( a.infsize() )
    {
        result = a.norm2();
    }

    else if ( a.size() )
    {
	result = norm2(a(0));

	if ( a.size() > 1 )
	{
	    for ( i = 1 ; i < a.size() ; i++ )
	    {
		result += (double) norm2(a(i));
	    }
	}
    }

    return result;
}

template <class S> double normp(const Vector<S> &a, double p)
{
    int i;
    double result = 0;

    if ( a.infsize() )
    {
        result = a.normp(p);
    }

    else if ( a.size() )
    {
	result = normp(a(0),p);

	if ( a.size() > 1 )
	{
	    for ( i = 1 ; i < a.size() ; i++ )
	    {
		result += (double) normp(a(i),p);
	    }
	}
    }

    return result;
}

template <class T> Vector<T> &setident(Vector<T> &a)
{
    return a.ident();
}

template <class S> Vector<double> &seteabs2(Vector<S> &a)
{
    if ( a.size() )
    {
        int i;

        for ( i = 0 ; i < a.size() ; i++ )
        {
            a("&",i) = abs2(a(i));
        }
    }

    return a;
}

template <class T> Vector<T> &setzero(Vector<T> &a)
{
    return a.zero();
}

template <class T> Vector<T> &setzeropassive(Vector<T> &a)
{
    return a.zeropassive();
}

template <class T> Vector<T> &setposate(Vector<T> &a)
{
    return a.posate();
}

template <class T> Vector<T> &setnegate(Vector<T> &a)
{
    return a.negate();
}

template <class T> Vector<T> &setconj(Vector<T> &a)
{
    return a.conj();
}

template <class T> Vector<T> &setrand(Vector<T> &a)
{
    return a.rand();
}




















template <class T> int testisvnan(const Vector<T> &x)
{
    int res = 0;

    if ( x.size() )
    {
        int i = 0;

        for ( i = 0 ; !res && ( i < x.size() ) ; i++ )
        {
            if ( testisvnan(x(i)) )
            {
                res = 1;
            }
        }
    }

    return res;
}

template <class T> int testisinf (const Vector<T> &x)
{
    int res = 0;

    if ( x.size() && !testisvnan(x) )
    {
        int i = 0;

        for ( i = 0 ; !res && ( i < x.size() ) ; i++ )
        {
            if ( testisinf(x(i)) )
            {
                res = 1;
            }
        }
    }

    return res;
}

template <class T> int testispinf(const Vector<T> &x)
{
    int pinfcnt = 0;

    if ( x.size() )
    {
        int i = 0;

        for ( i = 0 ; i < x.size() ; i++ )
        {
            if ( testispinf(x(i)) )
            {
                pinfcnt++;
            }
        }
    }

    return ( ( pinfcnt == x.size() ) && x.size() ) ? 1 : 0;
}

template <class T> int testisninf(const Vector<T> &x)
{
    int ninfcnt = 0;

    if ( x.size() )
    {
        int i = 0;

        for ( i = 0 ; i < x.size() ; i++ )
        {
            if ( testisninf(x(i)) )
            {
                ninfcnt++;
            }
        }
    }

    return ( ( ninfcnt == x.size() ) && x.size() ) ? 1 : 0;
}



























//phantomxyz
//----------------------------------------------

template <class T>
double &oneProductAssumeReal(double &res, const Vector<T> &a)
{
    if ( a.infsize() )
    {
        return a.inner1Real(res);
    }

    int dim = a.size();

    res = 0;

    if ( dim )
    {
        int i;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += (double) a(i);
	}
    }

    return res;
}

template <class T>
T &oneProduct(T &result, const Vector<T> &a)
{
    if ( a.infsize() )
    {
        return a.inner1(result);
    }

    setzero(result);

    if ( a.size() )
    {
        result = a(0);

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                result += a(i);
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &oneProductScaled(T &result, const Vector<T> &a, const Vector<T> &scale)
{
    if ( a.infsize() )
    {
        throw("Scaled FuncVector inner product not supported!");

        return result;
    }

    setzero(result);

    if ( a.size() )
    {
        result  = a(0);
        result /= scale(0);

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                result += a(i)/scale(i);
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedoneProduct(T &result, const Vector<int> &n, const Vector<T> &a)
{
    if ( a.infsize() )
    {
        throw("Index FuncVector inner product not supported!");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();

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
            aelm = apos; //a.ind(apos);

	    if ( aelm == nelm )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += a(apos);

                npos++;
		apos++;
	    }

            else if ( nelm <= aelm )
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
T &indexedoneProductScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &scale)
{
    if ( a.infsize() )
    {
        throw("Index FuncVector inner product not supported!");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();

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
            aelm = apos; //a.ind(apos);

	    if ( aelm == nelm )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += a(apos)/scale(apos);

                npos++;
		apos++;
	    }

            else if ( nelm <= aelm )
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

//----------------------------------------------

template <class T>
T &twoProduct(T &result, const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.infsize() || right_op.infsize() )
    {
        NiceAssert( right_op.infsize() && left_op.infsize() );

        return left_op.inner2(result,right_op,1);
    }

    NiceAssert( left_op.size() == right_op.size() );

        // standard

        T temp;

        setzero(temp);
        setzero(result);

        if ( left_op.size() )
        {
            result = right_op(0);
            setconj(result);
            result *= left_op(0);
            setconj(result);

            if ( left_op.size() > 1 )
            {
                int i;

                for ( i = 1 ; i < left_op.size() ; i++ )
	        {
                    // conj(l).r = conj(conj(r).l) (staying type T at all times)

                    temp = right_op(i);
                    setconj(temp);
                    temp *= left_op(i);
                    setconj(temp);

                    result += temp;
	        }
	    }
        }

    return postProInnerProd(result);
}

template <class T>
T &twoProductNoConj(T &result, const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.infsize() || right_op.infsize() )
    {
        NiceAssert( right_op.infsize() && left_op.infsize() );

        return left_op.inner2(result,right_op,1);
    }

    NiceAssert( left_op.size() == right_op.size() );

        T temp;

        setzero(result);
        setzero(temp);

        if ( left_op.size() )
        {
            result = right_op(0);
            rightmult(left_op(0),result);

            if ( left_op.size() > 1 )
	    {
                int i;

                for ( i = 1 ; i < left_op.size() ; i++ )
	        {
                    temp = right_op(i);
                    rightmult(left_op(i),temp);

                    result += temp;
	        }
            }
        }

    return postProInnerProd(result);
}

template <class T>
double &twoProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceAssert( b.infsize() && a.infsize() );

        return a.inner2Real(res,b,0);
    }

    int dim = a.size();

    NiceAssert( b.size() == dim );

    res = 0;

    if ( dim )
    {
        int i;

        for ( i = 0 ; i < dim ; i++ )
        {
            scaladd(res,a(i),b(i));
	}
    }

    return res;
}

template <class T>
T &twoProductRevConj(T &result, const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.infsize() || right_op.infsize() )
    {
        NiceAssert( right_op.infsize() && left_op.infsize() );

        return left_op.inner2(result,right_op,2);
    }

    NiceAssert( left_op.size() == right_op.size() );

    T temp;

    setzero(result);
    setzero(temp);

    if ( left_op.size() )
    {
        result = right_op(0);
        setconj(result);
        rightmult(left_op(0),result);

        if ( left_op.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < left_op.size() ; i++ )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                temp = right_op(i);
                setconj(temp);
                rightmult(left_op(i),temp);

                result += temp;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( a.size() )
    {
        tempa  = a(0);
        tempa /= scale(0);

        tempb  = b(0);
        tempb /= scale(0);

        setconj(tempa);
        //setconj(tempb);
        rightmult(tempa,tempb);

        result = tempb;

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                tempb /= scale(i);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductScaledNoConj(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( a.size() )
    {
        tempa  = a(0);
        tempa /= scale(0);

        tempb  = b(0);
        tempb /= scale(0);

        //setconj(tempa);
        //setconj(tempb);
        rightmult(tempa,tempb);

        result = tempb;

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                tempb /= scale(i);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductScaledRevConj(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( a.size() )
    {
        tempa  = a(0);
        tempa /= scale(0);

        tempb  = b(0);
        tempb /= scale(0);

        //setconj(tempa);
        setconj(tempb);
        rightmult(tempa,tempb);

        result = tempb;

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                tempb /= scale(i);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductRightScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( a.size() )
    {
        tempa  = a(0);
        //tempa /= scale(0);

        tempb  = b(0);
        tempb /= scale(0);

        setconj(tempa);
        //setconj(tempb);
        rightmult(tempa,tempb);

        result = tempb;

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                //tempa /= scale(i);

                tempb  = b(i);
                tempb /= scale(i);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductRightScaledNoConj(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( a.size() )
    {
        tempa  = a(0);
        //tempa /= scale(0);

        tempb  = b(0);
        tempb /= scale(0);

        //setconj(tempa);
        //setconj(tempb);
        rightmult(tempa,tempb);

        result = tempb;

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                //tempa /= scale(i);

                tempb  = b(i);
                tempa /= scale(i);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductRightScaledRevConj(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( a.size() )
    {
        tempa  = a(0);
        //tempa /= scale(0);

        tempb  = b(0);
        tempb /= scale(0);

        //setconj(tempa);
        setconj(tempb);
        rightmult(tempa,tempb);

        result = tempb;

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                //tempa /= scale(i);

                tempb  = b(i);
                tempb /= scale(i);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductLeftScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( a.size() )
    {
        tempa  = a(0);
        tempa /= scale(0);

        tempb  = b(0);
        //tempb /= scale(0);

        setconj(tempa);
        //setconj(tempb);
        rightmult(tempa,tempb);

        result = tempb;

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                //tempb /= scale(i);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductLeftScaledNoConj(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( a.size() )
    {
        tempa  = a(0);
        tempa /= scale(0);

        tempb  = b(0);
        //tempb /= scale(0);

        //setconj(tempa);
        //setconj(tempb);
        rightmult(tempa,tempb);

        result = tempb;

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                //tempb /= scale(i);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductLeftScaledRevConj(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( a.size() )
    {
        tempa  = a(0);
        tempa /= scale(0);

        tempb  = b(0);
        //tempb /= scale(0);

        //setconj(tempa);
        setconj(tempb);
        rightmult(tempa,tempb);

        result = tempb;

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                //tempb /= scale(i);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProduct(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();

    T temp;

    setzero(result);
    setzero(temp);

    if ( nsize && asize && bsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;

        int nelm;
	int aelm;
	int belm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) )
	{
            nelm = n(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                temp = b(bpos);
                setconj(temp);
                temp *= a(apos);
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
T &indexedtwoProductNoConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();

    T temp;

    setzero(result);
    setzero(temp);

    if ( nsize && asize && bsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;

        int nelm;
	int aelm;
	int belm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) )
	{
            nelm = n(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) )
	    {
                temp = b(bpos);
                rightmult(a(apos),temp);

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
T &indexedtwoProductRevConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();

    T temp;

    setzero(result);
    setzero(temp);

    if ( nsize && asize && bsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;

        int nelm;
	int aelm;
	int belm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) )
	{
            nelm = n(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) )
	    {
                temp = b(bpos);
                setconj(temp);
                rightmult(a(apos),temp);

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
T &indexedtwoProductScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb  = b(bpos);
                tempb /= scale(dpos);

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
T &indexedtwoProductScaledNoConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb  = b(bpos);
                tempb /= scale(dpos);

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
T &indexedtwoProductScaledRevConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb  = b(bpos);
                tempb /= scale(dpos);

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
T &indexedtwoProductLeftScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb  = b(bpos);
                //tempb /= scale(dpos); - only half scaling

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
T &indexedtwoProductLeftScaledNoConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb  = b(bpos);
                //tempb /= scale(dpos); - only half scaling

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
T &indexedtwoProductLeftScaledRevConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb = b(bpos);
                //tempb /= scale(dpos); - only half scaling

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
T &indexedtwoProductRightScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                //tempa /= scale(cpos); - only half scaling
                                             
                tempb  = b(bpos);
                tempb /= scale(dpos);

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
T &indexedtwoProductRightScaledNoConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                //tempa /= scale(cpos); - only half scaling

                tempb  = b(bpos);
                tempb /= scale(dpos);

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
T &indexedtwoProductRightScaledRevConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        throw("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                //tempa /= scale(cpos); - only half scaling

                tempb  = b(bpos);
                tempb /= scale(dpos);

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

//----------------------------------------------

template <class T>
double &threeProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c)
{
    if ( a.infsize() || b.infsize() || c.infsize() )
    {
        NiceAssert( a.infsize() && b.infsize() && c.infsize() );

        return a.inner3Real(res,b,c);
    }

    int dim = a.size();

    NiceAssert( b.size() == dim );
    NiceAssert( c.size() == dim );

    res = 0;

    if ( dim )
    {
        int i;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += (double) (a(i)*b(i)*c(i));
	}
    }

    return res;
}

template <class T>
T &threeProduct(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c)
{
    if ( a.infsize() || b.infsize() || c.infsize() )
    {
        NiceAssert( a.infsize() && b.infsize() && c.infsize() );

        return a.inner3(result,b,c);
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == c.size() );

    setzero(result);

    if ( a.size() )
    {
        result = (a(0)*b(0))*c(0);

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                result += (a(i)*b(i))*c(i);
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &threeProductScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() || c.infsize() )
    {
        throw("Scaled 3-FuncVector-product not supported");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == c.size() );

    setzero(result);

    if ( a.size() )
    {
        result = ((a(0)/scale(0))*(b(0)/scale(0)))*(c(0)/scale(0));

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                result += ((a(i)/scale(i))*(b(i)/scale(i)))*(c(i)/scale(i));
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedthreeProduct(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c)
{
    if ( a.infsize() || b.infsize() || c.infsize() )
    {
        throw("Indexed FuncVector three-product not supported");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = c.size(); //indsize();

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //c.ind(cpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += (a(apos)*b(bpos))*c(cpos);

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
T &indexedthreeProductScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() || c.infsize() )
    {
        throw("Indexed FuncVector three-product not supported");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = c.size(); //indsize();

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //c.ind(cpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += ((a(apos)/scale(apos))*(b(bpos)/scale(apos)))*(c(cpos)/scale(cpos));

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

//----------------------------------------------

template <class T>
double &fourProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d)
{
    if ( a.infsize() || b.infsize() || c.infsize() || d.infsize() )
    {
        NiceAssert( a.infsize() && b.infsize() && c.infsize() && d.infsize() );

        return a.inner4Real(res,b,c,d);
    }

    int dim = a.size();

    NiceAssert( b.size() == dim );
    NiceAssert( c.size() == dim );
    NiceAssert( d.size() == dim );

    res = 0;

    if ( dim )
    {
        int i;

        for ( i = 0 ; i < dim ; i++ )
        {
            scaladd(res,a(i),b(i),c(i),d(i));
	}
    }

    return res;
}

template <class T>
T &fourProduct(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d)
{
    if ( a.infsize() || b.infsize() || c.infsize() || d.infsize() )
    {
        NiceAssert( a.infsize() && b.infsize() && c.infsize() && d.infsize() );

        return a.inner4(result,b,c,d);
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == c.size() );
    NiceAssert( a.size() == d.size() );

    setzero(result);

    if ( a.size() )
    {
        result = (a(0)*b(0))*(c(0)*d(0));

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                result += (a(i)*b(i))*(c(i)*d(i));
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &fourProductScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() || c.infsize() || d.infsize() )
    {
        throw("Scaled 4-FuncVector-product not supported");

        return a.inner4(result,b,c,d);
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == c.size() );
    NiceAssert( a.size() == d.size() );

    setzero(result);

    if ( a.size() )
    {
        result = ((a(0)/scale(0))*(b(0)/scale(0)))*((c(0)/scale(0))*(d(0)/scale(0)));

        if ( a.size() > 1 )
	{
            int i;

            for ( i = 1 ; i < a.size() ; i++ )
	    {
                result += ((a(i)/scale(i))*(b(i)/scale(i)))*((c(i)/scale(i))*(d(i)/scale(i)));
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedfourProduct(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d)
{
    if ( a.infsize() || b.infsize() || c.infsize() || d.infsize() )
    {
        throw("Indexed FuncVector four-product not supported.");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = c.size(); //indsize();
    int dsize = d.size(); //indsize();

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //c.ind(cpos);
	    delm = dpos; //d.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += (a(apos)*b(bpos))*(c(cpos)*d(dpos));

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
T &indexedfourProductScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() || c.infsize() || d.infsize() )
    {
        throw("Indexed FuncVector four-product not supported.");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = c.size(); //indsize();
    int dsize = d.size(); //indsize();

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
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //c.ind(cpos);
	    delm = dpos; //d.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += ((a(apos)/scale(apos))*(b(bpos)/scale(bpos)))*((c(cpos)/scale(cpos))*(d(dpos)/scale(dpos)));

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

//----------------------------------------------

template <class T>
double &mProductAssumeReal(double &res, const Vector<const Vector <T> *> &x)
{
    res = 0;

    int m = x.size();

    if ( m )
    {
        int i,j;
        int isinf = 0;

        for ( j = 0 ; j < m ; j++ )
        {
            isinf += ( (*(x(j))).infsize() ? 1 : 0 );
        }

        if ( isinf )
        {
            NiceAssert( isinf == m );

            if ( m == 1 )
            {
                return (*(x(0))).inner1Real(res);
            }

            else
            {
                retVector<const Vector <T> *> tmpva;

                return (*(x(0))).innerpReal(res,x(1,1,m-1,tmpva));
            }
        }

        int dim = (*(x(zeroint()))).size();

        if ( dim )
        {
            double temp;

            for ( i = 0 ; i < dim ; i++ )
            {
                temp = 1;

                for ( j = 0 ; j < m ; j++ )
                {
                    NiceAssert( (*(x(j))).size() == dim );

                    scalmul(temp,(*(x(j)))(i));
                }

                res += temp;
            }
        }
    }

    return res;
}

template <class T>
T &mProduct(T &result, const Vector<const Vector <T> *> &x)
{
    setzero(result);

    if ( x.size() )
    {
        int i,j;
        int isinf = 0;

        for ( j = 0 ; j < x.size() ; j++ )
        {
            isinf += ( (*(x(j))).infsize() ? 1 : 0 );
        }

        if ( isinf )
        {
            NiceAssert( isinf == x.size() );

            if ( x.size() == 1 )
            {
                return (*(x(0))).inner1(result);
            }

            else
            {
                retVector<const Vector <T> *> tmpva;

                return (*(x(0))).innerp(result,x(1,1,x.size()-1,tmpva));
            }
        }

        Vector<T> temp(*(x(zeroint())));

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
T &mProductScaled(T &result, const Vector<const Vector <T> *> &x, const Vector<T> &scale)
{
    setzero(result);

    if ( x.size() )
    {
        int i,j;
        int isinf = 0;

        for ( j = 0 ; j < x.size() ; j++ )
        {
            isinf += ( (*(x(j))).infsize() ? 1 : 0 );
        }

        if ( isinf )
        {
            throw("Scaled m-FuncVector-product not supported");

            return result;
        }

        Vector<T> temp(*(x(zeroint())));
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
T &indexedmProduct(T &result, const Vector<int> &n, const Vector<const Vector <T> *> &x)
{
    setzero(result);

    if ( x.size() )
    {
        NiceAssert( !((*(x(0))).infsize()) );

	int i;
        Vector<T> a(*(x(zeroint())));

	int nsize = n.size();
	int asize = a.size(); //indsize();

        int npos = 0;
	int apos = 0;

        int nelm;
	int aelm;

	while ( ( npos < nsize ) && ( apos < asize ) )
	{
            nelm = n(npos);
            aelm = apos; //a.ind(apos);

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
                a.zero(apos); //a.ind(apos));
		apos++;
	    }
	}

	while ( apos < asize )
	{
	    a.zero(apos); //a.ind(apos));
	    apos++;
	}

	if ( x.size() > 1 )
	{
            NiceAssert( !((*(x(1))).infsize()) );

            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            a *= *(x(1));

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    NiceAssert( !((*(x(i))).infsize()) );

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
T &indexedmProductScaled(T &result, const Vector<int> &n, const Vector<const Vector <T> *> &x, const Vector<T> &scale)
{
    setzero(result);

    if ( x.size() )
    {
        NiceAssert( !((*(x(0))).infsize()) );

	int i;
        Vector<T> a(*(x(zeroint())));
        a /= scale;

	int nsize = n.size();
	int asize = a.size(); //indsize();

        int npos = 0;
	int apos = 0;

        int nelm;
	int aelm;

	while ( ( npos < nsize ) && ( apos < asize ) )
	{
            nelm = n(npos);
            aelm = apos; //a.ind(apos);

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
                a.zero(apos); //a.ind(apos));
		apos++;
	    }
	}

	while ( apos < asize )
	{
	    a.zero(apos); //a.ind(apos));
	    apos++;
	}

	if ( x.size() > 1 )
	{
            NiceAssert( !((*(x(1))).infsize()) );

            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            a *= *(x(1));
            a /= scale;

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    NiceAssert( !((*(x(i))).infsize()) );

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





























// Mathematical operator overloading

template <class T> Vector<T>  operator+ (const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( !left_op.infsize() && right_op.infsize() )
    {
        return right_op+left_op;
    }

    Vector<T> res(left_op);

    return ( res += right_op );
}

template <class T> Vector<T>  operator+ (const Vector<T> &left_op, const T         &right_op)
{
    Vector<T> res(left_op);

    return ( res += right_op );
}

template <class T> Vector<T>  operator+ (const T         &left_op, const Vector<T> &right_op)
{
    Vector<T> res(right_op);

    return ( res += left_op );
}

template <class T> Vector<T>  operator- (const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( !left_op.infsize() && right_op.infsize() )
    {
        Vector<T> right_alt(right_op);

        right_alt.negate();

        return right_alt+left_op;
    }

    Vector<T> res(left_op);

    return ( res -= right_op );
}

template <class T> Vector<T>  operator- (const Vector<T> &left_op, const T         &right_op)
{
    Vector<T> res(left_op);

    return ( res -= right_op );
}

template <class T> Vector<T>  operator- (const T         &left_op, const Vector<T> &right_op)
{
    Vector<T> res(right_op);

    res.negate();

    return ( res += left_op );
}

template <class T> Vector<T>  operator* (const Vector<T> &left_op, const Vector<T> &right_op)
{
    Vector<T> res(left_op);

    return ( res *= right_op );
}

template <class T> Vector<T>  operator* (const Vector<T> &left_op, const T         &right_op)
{
    Vector<T> res(left_op);

    return ( res *= right_op );
}

template <class T> Vector<T>  operator* (const T         &left_op, const Vector<T> &right_op)
{
    Vector<T> res(right_op);

    return ( left_op *= res );
}

//template <class T> Vector<T>  operator& (const Vector<T> &left_op, const Vector<T> &right_op)
//{
//    Vector<T> res(left_op);
//
//    return ( res &= right_op );
//}

//template <class T> Vector<T>  operator& (const Vector<T> &left_op, const T         &right_op)
//{
//    Vector<T> res(left_op);
//
//    return ( res &= right_op );
//}

//template <class T> Vector<T>  operator& (const T         &left_op, const Vector<T> &right_op)
//{
//    Vector<T> res(right_op);
//
//    return ( res &= left_op );
//}

//template <class T> Vector<T>  operator| (const Vector<T> &left_op, const Vector<T> &right_op)
//{
//    Vector<T> res(left_op);
//
//    return ( res |= right_op );
//}

//template <class T> Vector<T>  operator| (const Vector<T> &left_op, const T         &right_op)
//{
//    Vector<T> res(left_op);
//
//    return ( res |= right_op );
//}

//template <class T> Vector<T>  operator| (const T         &left_op, const Vector<T> &right_op)
//{
//    Vector<T> res(right_op);
//
//    return ( res |= left_op );
//}

//template <class T> Vector<T>  operator^ (const Vector<T> &left_op, const Vector<T> &right_op)
//{
//    Vector<T> res(left_op);
//
//    return ( res ^= right_op );
//}

//template <class T> Vector<T>  operator^ (const Vector<T> &left_op, const T         &right_op)
//{
//    Vector<T> res(left_op);
//
//    return ( res ^= right_op );
//}

//template <class T> Vector<T>  operator^ (const T         &left_op, const Vector<T> &right_op)
//{
//    Vector<T> res(right_op);
//
//    return ( res ^= left_op );
//}

//template <class T, class S> Vector<T>  operator<<(const Vector<T> &left_op, const S         &right_op)
//{
//    Vector<T> res(left_op);
//
//    return ( res <<= right_op );
//}

//template <class T, class S> Vector<T>  operator>>(const Vector<T> &left_op, const S         &right_op)
//{
//    Vector<T> res(left_op);
//
//    return ( res >>= right_op );
//}

template <class T> Vector<T>  operator/ (const Vector<T> &left_op, const Vector<T> &right_op)
{
    Vector<T> res(left_op);

    return ( res /= right_op );
}

template <class T> Vector<T>  operator/ (const Vector<T> &left_op, const T         &right_op)
{
    Vector<T> res(left_op);

    return ( res /= right_op );
}

template <class T> Vector<T>  operator/ (const T         &left_op, const Vector<T> &right_op)
{
    Vector<T> res(right_op);

    return ( left_op /= res );
}

template <class T> Vector<T>  operator% (const Vector<T> &left_op, const Vector<T> &right_op)
{
    Vector<T> res(left_op);

    return ( res %= right_op );
}

template <class T> Vector<T>  operator% (const Vector<T> &left_op, const T         &right_op)
{
    Vector<T> res(left_op);

    return ( res %= right_op );
}

template <class T> Vector<T>  operator% (const T         &left_op, const Vector<T> &right_op)
{
    Vector<T> res(right_op);

    return ( left_op %= res );
}

template <class S, class T> Vector<S> &operator*=(Vector<S> &left_op, const Vector<T> &right_op)
{
    if ( left_op.shareBase(right_op) )
    {
        Vector<T> temp(right_op);

        left_op *= temp;
    }

    else
    {
	// We treat empty vectors as additive identities (ie. zero)

        NiceAssert( !left_op.infsize() && !right_op.infsize() );
        NiceAssert( ( left_op.size() == right_op.size() ) || !(left_op.size()) || !(right_op.size()) );

	if ( !(left_op.size()) && right_op.size() )
	{
	    left_op.resize(right_op.size());
            left_op.zero();
	}

	else if ( !(right_op.size()) )
	{
	    left_op.zero();
	}

	else if ( left_op.size() )
	{
	    int i;

	    for ( i = 0 ; i < left_op.size() ; i++ )
	    {
		left_op("&",i) *= right_op(i);
	    }
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator*=(      Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.infsize() )
    {
        left_op.mulit(right_op);
    }

    else if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
            left_op("&",i) *= right_op;
	}
    }

    return left_op;
}

template <class T>          Vector<T> &operator*= (const T         &left_op,       Vector<T> &right_op)
{
    if ( right_op.infsize() )
    {
        right_op.rmulit(left_op);
    }

    else if ( right_op.size() )
    {
	int i;

	for ( i = 0 ; i < right_op.size() ; i++ )
	{
            rightmult(left_op,right_op("&",i));
	}
    }

    return right_op;
}

template <class T> Vector<T> &operator/= (      Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.shareBase(right_op) )
    {
        Vector<T> temp(right_op);

        left_op /= temp;
    }

    else if ( left_op.infsize() || right_op.infsize() )
    {
        left_op.divit(right_op);
    }

    else
    {
	// We treat empty vectors as additive identities (ie. zero)

        NiceAssert( ( left_op.size() == right_op.size() ) || !(left_op.size()) || !(right_op.size()) );

	if ( !(left_op.size()) && right_op.size() )
	{
	    left_op.resize(right_op.size());
            left_op.zero();
	}

	else if ( !(right_op.size()) )
	{
	    left_op.zero();
	}

	else if ( left_op.size() )
	{
	    int i;

	    for ( i = 0 ; i < left_op.size() ; i++ )
	    {
		left_op("&",i) /= right_op(i);
	    }
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator/= (      Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.infsize() )
    {
        left_op.divit(right_op);
    }

    else if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
            left_op("&",i) /= right_op;
	}
    }

    return left_op;
}

template <class T>          Vector<T> &operator/= (const T         &left_op,       Vector<T> &right_op)
{
    T leftinv(inv(left_op));

    if ( right_op.infsize() )
    {
        right_op.rdivit(left_op);
    }

    else if ( right_op.size() )
    {
	int i;

	for ( i = 0 ; i < right_op.size() ; i++ )
	{
            rightmult(leftinv,right_op("&",i));
	}
    }

    return right_op;
}

template <class T> Vector<T> &operator%= (      Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.shareBase(right_op) )
    {
        Vector<T> temp(right_op);

        left_op %= temp;
    }

    else
    {
	// We treat empty vectors as additive identities (ie. zero)

        NiceAssert( !left_op.infsize() && !right_op.infsize() );
        NiceAssert( ( left_op.size() == right_op.size() ) || !(left_op.size()) || !(right_op.size()) );

	if ( !(left_op.size()) && right_op.size() )
	{
	    left_op.resize(right_op.size());
            left_op.zero();
	}

	else if ( !(right_op.size()) )
	{
	    left_op.zero();
	}

	else if ( left_op.size() )
	{
	    int i;

	    for ( i = 0 ; i < left_op.size() ; i++ )
	    {
		left_op("&",i) %= right_op(i);
	    }
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator%= (      Vector<T> &left_op, const T         &right_op)
{
    NiceAssert( !left_op.infsize() );

    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
            left_op("&",i) %= right_op;
	}
    }

    return left_op;
}

template <class T>          Vector<T> &operator%= (const T         &left_op,       Vector<T> &right_op)
{
    NiceAssert( !left_op.infsize() );

    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
            right_op("&",i) = left_op%right_op(i);
	}
    }

    return right_op;
}

template <class T> Vector<T>  operator+ (const Vector<T> &left_op)
{
    Vector<T> res(left_op);

    if ( res.infsize() )
    {
        res.posate();
    }

    else if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
            setposate(res("&",i));
	}
    }

    return res;
}

template <class T> Vector<T>  operator- (const Vector<T> &left_op)
{
    Vector<T> res(left_op);

    if ( res.infsize() )
    {
        res.negate();
    }

    else if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
            setnegate(res("&",i));
	}
    }

    return res;
}

/*template <class T> Vector<T> &operator++(      Vector<T> &left_op)
{
    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
            ++(left_op("&",i));
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator--(      Vector<T> &left_op)
{
    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
            --(left_op("&",i));
	}
    }

    return left_op;
}

template <class T> Vector<T>  operator++(      Vector<T> &left_op, int)
{
    Vector<T> oldval(left_op);

    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
            (left_op("&",i))++;
	}
    }

    return oldval;
}

template <class T> Vector<T>  operator--(      Vector<T> &left_op, int)
{
    Vector<T> oldval(left_op);

    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
            (left_op("&",i))--;
	}
    }

    return oldval;
}

template <class T> Vector<T>  operator~ (const Vector<T> &left_op)
{
    Vector<T> res(left_op);

    if ( res.size() )
    {
	int i;

	for ( i = 0 ; i < res.size() ; i++ )
	{
            res("&",i) = ~res(i);
	}
    }

    return res;
}*/

template <class T> Vector<T> &operator+=(      Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.shareBase(right_op) )
    {
        Vector<T> temp(right_op);

        left_op += temp;
    }

    else if ( left_op.infsize() )
    {
        left_op.addit(right_op);
    }

    else if ( right_op.infsize() )
    {
        Vector<T> right_alt(left_op);

        left_op =  right_op;
        left_op += right_alt;
    }

    else
    {
	// We treat empty vectors as additive identities (ie. zero)

        NiceAssert( ( left_op.size() == right_op.size() ) || !(left_op.size()) || !(right_op.size()) );

	if ( left_op.size() && right_op.size() )
	{
	    int i;

	    for ( i = 0 ; i < left_op.size() ; i++ )
	    {
		left_op("&",i) += right_op(i);
	    }
	}

	else if ( right_op.size() )
	{
            left_op = right_op;
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator+=(      Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
	    left_op("&",i) += right_op;
	}
    }

    return left_op;
}

template <class T>          Vector<T> &operator+= (const T         &left_op,       Vector<T> &right_op)
{
    return ( right_op += left_op );
}

template <class T> Vector<T> &operator-=(      Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.shareBase(right_op) )
    {
        Vector<T> temp(right_op);

        left_op -= temp;
    }

    else if ( left_op.infsize() )
    {
        left_op.subit(right_op);
    }

    else if ( right_op.infsize() )
    {
        Vector<T> right_alt(left_op);

        left_op =  right_op;
        left_op.negate();
        left_op += right_alt;
    }

    else
    {
	// We treat empty vectors as additive identities (ie. zero)

        NiceAssert( ( left_op.size() == right_op.size() ) || !(left_op.size()) || !(right_op.size()) );

	if ( left_op.size() && right_op.size() )
	{
	    int i;

	    for ( i = 0 ; i < left_op.size() ; i++ )
	    {
		left_op("&",i) -= right_op(i);
	    }
	}

	else if ( right_op.size() )
	{
            left_op = -right_op;
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator-=(      Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
	    left_op("&",i) -= right_op;
	}
    }

    return left_op;
}

template <class T>          Vector<T> &operator-= (const T         &left_op,       Vector<T> &right_op)
{
    right_op.negate();

    return ( right_op += left_op );
}

//template <class T> Vector<T> &operator&=(      Vector<T> &left_op, const Vector<T> &right_op)
//{
//    if ( left_op.shareBase(right_op) )
//    {
//        Vector<T> temp(right_op);
//
//        left_op &= temp;
//    }
//
//    else
//    {
//        NiceAssert( left_op.size() == right_op.size() );
//
//        int i;
//
//        for ( i = 0 ; i < left_op.size() ; i++ )
//        {
//            left_op("&",i) &= right_op(i);
//        }
//    }
//
//    return left_op;
//}

//template <class T> Vector<T> &operator&=(      Vector<T> &left_op, const T         &right_op)
//{
//    if ( left_op.size() )
//    {
//        int i;
//
//        for ( i = 0 ; i < left_op.size() ; i++ )
//        {
//            left_op("&",i) &= right_op;
//        }
//    }
//
//    return left_op;
//}

//template <class T> Vector<T> &operator|=(      Vector<T> &left_op, const Vector<T> &right_op)
//{
//    if ( left_op.shareBase(right_op) )
//    {
//        Vector<T> temp(right_op);
//
//        left_op |= temp;
//    }
//
//    else
//    {
//        NiceAssert( left_op.size() == right_op.size() );
//
//        int i;
//
//        for ( i = 0 ; i < left_op.size() ; i++ )
//        {
//            left_op("&",i) |= right_op(i);
//        }
//    }
//
//    return left_op;
//}

//template <class T> Vector<T> &operator|=(      Vector<T> &left_op, const T         &right_op)
//{
//    if ( left_op.size() )
//    {
//        int i;
//
//        for ( i = 0 ; i < left_op.size() ; i++ )
//        {
//            left_op("&",i) |= right_op;
//        }
//    }
//
//    return left_op;
//}

//template <class T> Vector<T> &operator^=(      Vector<T> &left_op, const Vector<T> &right_op)
//{
//    if ( left_op.shareBase(right_op) )
//    {
//        Vector<T> temp(right_op);
//
//        left_op ^= temp;
//    }
//
//    else
//    {
//        NiceAssert( left_op.size() == right_op.size() );
//
//        int i;
//
//        for ( i = 0 ; i < left_op.size() ; i++ )
//        {
//            left_op("&",i) ^= right_op(i);
//        }
//    }
//
//    return left_op;
//}

//template <class T> Vector<T> &operator^=(      Vector<T> &left_op, const T         &right_op)
//{
//    if ( left_op.size() )
//    {
//        int i;
//
//        for ( i = 0 ; i < left_op.size() ; i++ )
//        {
//            left_op("&",i) ^= right_op;
//        }
//    }
//
//    return left_op;
//}

//template <class T, class S> Vector<T> &operator<<=(      Vector<T> &left_op, const S         &right_op)
//{
//    if ( left_op.size() )
//    {
//        int i;
//
//        for ( i = 0 ; i < left_op.size() ; i++ )
//        {
//            left_op("&",i) <<= right_op;
//        }
//    }
//
//    return left_op;
//}

//template <class T, class S> Vector<T> &operator>>=(      Vector<T> &left_op, const S         &right_op)
//{
//    if ( left_op.size() )
//    {
//        int i;
//
//        for ( i = 0 ; i < left_op.size() ; i++ )
//        {
//            left_op("&",i) >>= right_op;
//        }
//    }
//
//    return left_op;
//}

template <class T> Vector<T> &leftmult (      Vector<T> &left_op, const Vector<T> &right_op)
{
    return ( left_op *= right_op );
}

template <class T> Vector<T> &leftmult (      Vector<T> &left_op, const T         &right_op)
{
    return ( left_op *= right_op );
}

template <class T> Vector<T> &rightmult(const Vector<T> &left_op,       Vector<T> &right_op)
{
    if ( right_op.shareBase(left_op) )
    {
        Vector<T> temp(left_op);

        rightmult(temp,right_op);
    }

    else if ( right_op.infsize() )
    {
        right_op.rmulit(left_op);
    }

    else
    {
	// We treat empty vectors as additive identities (ie. zero)

        NiceAssert( !left_op.infsize() );
        NiceAssert( ( left_op.size() == right_op.size() ) || !(left_op.size()) || !(right_op.size()) );

	if ( !(right_op.size()) && left_op.size() )
	{
	    right_op.resize(left_op.size());
            right_op.zero();
	}

	if ( !(left_op.size()) )
	{
	    right_op.zero();
	}

	else if ( right_op.size() )
	{
	    int i;

	    for ( i = 0 ; i < right_op.size() ; i++ )
	    {
		rightmult(left_op(i),right_op("&",i));
	    }
	}
    }

    return right_op;
}

template <class T> Vector<T> &rightmult(const T         &left_op,       Vector<T> &right_op)
{
    return ( left_op *= right_op );
}


template <class T>
Vector<T> &kronprod(Vector<T> &res, const Vector<T> &a, const Vector<T> &b)
{
    NiceAssert( !a.infsize() );
    NiceAssert( !b.infsize() );

    int i,j;

    res.resize((a.size())*(b.size())).zero();

    for ( i = 0 ; i < a.size() ; i++ )
    {
        for ( j = 0 ; j < b.size() ; j++ )
        {
            res("&",(i*(b.size()))+j) = a(i)*b(j);
        }
    }

    return res;
}

template <class T>
Vector<T> &kronpow(Vector<T> &res, const Vector<T> &a, int n)
{
    NiceAssert( n >= 0 );

    if ( n == 0 )
    {
        setident((res.resize(1))("&",0));
    }

    else if ( n == 1 )
    {
        res = a;
    }

    else
    {
        Vector<T> b;

        kronprod(res,a,kronpow(b,a,n-1));
    }

    return res;
}


template <class T> int operator==(const Vector<T> &left_op, const Vector<T> &right_op)
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
	    if ( !( left_op(i) == right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator==(const Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
	    if ( !( left_op(i) == right_op ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator==(const T         &left_op, const Vector<T> &right_op)
{
    if ( right_op.size() )
    {
	int i;

	for ( i = 0 ; i < right_op.size() ; i++ )
	{
	    if ( !( left_op == right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator!=(const Vector<T> &left_op, const Vector<T> &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator!=(const Vector<T> &left_op, const T         &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator!=(const T         &left_op, const Vector<T> &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator< (const Vector<T> &left_op, const Vector<T> &right_op)
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
            if ( !( left_op(i) < right_op(i) ) )
            {
                return 0;
            }
        }
    }

    return 1;
}

template <class T> int operator< (const Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
	    if ( !( left_op(i) < right_op ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator< (const T         &left_op, const Vector<T> &right_op)
{
    if ( right_op.size() )
    {
	int i;

	for ( i = 0 ; i < right_op.size() ; i++ )
	{
	    if ( !( left_op < right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator<=(const Vector<T> &left_op, const Vector<T> &right_op)
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
	    if ( !( left_op(i) <= right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator<=(const Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
	    if ( !( left_op(i) <= right_op ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator<=(const T         &left_op, const Vector<T> &right_op)
{
    if ( right_op.size() )
    {
	int i;

	for ( i = 0 ; i < right_op.size() ; i++ )
	{
	    if ( !( left_op <= right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator> (const Vector<T> &left_op, const Vector<T> &right_op)
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
	    if ( !( left_op(i) > right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator> (const Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
	    if ( !( left_op(i) > right_op ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator> (const T         &left_op, const Vector<T> &right_op)
{
    if ( right_op.size() )
    {
	int i;
	for ( i = 0 ; i < right_op.size() ; i++ )
	{
	    if ( !( left_op > right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator>=(const Vector<T> &left_op, const Vector<T> &right_op)
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
	    if ( !( left_op(i) >= right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator>=(const Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.size() )
    {
	int i;

	for ( i = 0 ; i < left_op.size() ; i++ )
	{
	    if ( !( left_op(i) >= right_op ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator>=(const T         &left_op, const Vector<T> &right_op)
{
    if ( right_op.size() )
    {
	int i;

	for ( i = 0 ; i < right_op.size() ; i++ )
	{
	    if ( !( left_op >= right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}




// Conversion from strings

template <class T> Vector<T> &atoVector(Vector<T> &dest, const std::string &src)
{
    std::istringstream srcmod(src);

    srcmod >> dest;

    return dest;
}





inline Vector<int> &randPerm(Vector<int> &res)
{
    if ( res.size() )
    {
        Vector<int> temp(res.size());

        int i,j;

        for ( i = 0 ; i < res.size() ; i++ )
        {
            temp("&",i) = i;
        }

        for ( i = 0 ; i < res.size() ; i++ )
        {
            j = svm_rand()%(temp.size());

            res("&",i) = temp(j);
            temp.remove(j);
        }
    }

    return res;
}


template <class T> Vector<T> &randfill (Vector<T> &res)
{
    return res.applyon(randfill);
}

template <class T> Vector<T> &randnfill(Vector<T> &res)
{
    return res.applyon(randnfill);
}

#define MINZEROSIZE        1000
#define ZEROALLOCAHEADFRAC 1.2

inline const Vector<int> &zerointvec(int size, retVector<int> &tmpv)
{
    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    // We use a special constructor here.  It will set the vector to point
    // directly to the underlying dynamic array returned by
    // zerointarray(size);

    svmvolatile static Vector<int> zerores(zerointarray(size));

    NiceAssert( size >= 0 );

    if ( size > zerores.dsize )
    {
        // zerores points directly to underlying static array.  All references
        // to zerores are via child vectors.  So all we need to do is
        // make sure the underlying array has sufficient size and set the
        // size of zerores.  The call zerointarray(size) is sufficient to
        // resize the underlying dynamic array.

        zerores.dsize = zerointarray(size)->array_size();
    }

    // Note that the scope of the following will be min(zerores,tmpv), which
    // is just scope(tmpv) as zerores is a static

    const Vector<int> &res = (const_cast<Vector<int> &>(zerores))(zeroint(),1,size-1,tmpv);

    svm_mutex_unlock(eyelock);

    return res;
}

inline const Vector<int> &oneintvec(int size, retVector<int> &tmpv)
{
    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    svmvolatile static Vector<int> oneres(oneintarray(size));

    NiceAssert( size >= 0 );

    if ( size > oneres.dsize )
    {
        oneres.dsize = oneintarray(size)->array_size();
    }

    const Vector<int> &res = (const_cast<Vector<int> &>(oneres))(zeroint(),1,size-1,tmpv);

    svm_mutex_unlock(eyelock);

    return res;
}

inline const Vector<int> &cntintvec(int size, retVector<int> &tmpv)
{
    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    svmvolatile static Vector<int> cntres(cntintarray(size));

    NiceAssert( size >= 0 );

    if ( size > cntres.dsize )
    {
        cntres.dsize = cntintarray(size)->array_size();
    }

    const Vector<int> &res = (const_cast<Vector<int> &>(cntres))(zeroint(),1,size-1,tmpv);

    svm_mutex_unlock(eyelock);

    return res;
}

inline const Vector<double> &zerodoublevec(int size, retVector<double> &tmpv)
{
    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    svmvolatile static Vector<double> zerores(zerodoublearray(size));

    NiceAssert( size >= 0 );

    if ( size > zerores.dsize )
    {
        zerores.dsize = zerodoublearray(size)->array_size();
    }

    const Vector<double> &res = (const_cast<Vector<double> &>(zerores))(zeroint(),1,size-1,tmpv);

    svm_mutex_unlock(eyelock);

    return res;
}

inline const Vector<double> &onedoublevec(int size, retVector<double> &tmpv)
{
    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    svmvolatile static Vector<double> oneres(onedoublearray(size));

    NiceAssert( size >= 0 );

    if ( size > oneres.dsize )
    {
        oneres.dsize = onedoublearray(size)->array_size();
    }

    const Vector<double> &res = (const_cast<Vector<double> &>(oneres))(zeroint(),1,size-1,tmpv);

    svm_mutex_unlock(eyelock);

    return res;
}

inline const Vector<double> &cntdoublevec(int size, retVector<double> &tmpv)
{
    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    svmvolatile static Vector<double> cntres(cntdoublearray(size));

    NiceAssert( size >= 0 );

    if ( size > cntres.dsize )
    {
        cntres.dsize = cntdoublearray(size)->array_size();
    }

    const Vector<double> &res = (const_cast<Vector<double> &>(cntres))(zeroint(),1,size-1,tmpv);

    svm_mutex_unlock(eyelock);

    return res;
}


































//template <class T> inline void makeFuncVector(const std::string &typestring, Vector<T> *&res, std::istream &src); // operator>> analog
//template <class T> inline void makeFuncVector(const std::string &typestring, Vector<T> *&res, std::istream &src, int processxyzvw); // streamItIn analog

//temp placeholder while FuncVector gets written
//template <class T> inline void makeFuncVector(const std::string &typestring, Vector<T> *&res, std::istream &src) { throw("makeFuncVector stub called"); (void) typestring; (void) src; res = NULL; return; }
//template <class T> inline void makeFuncVector(const std::string &typestring, Vector<T> *&res, std::istream &src, int processxyzvw) { throw("makeFuncVector stub b called"); (void) typestring; (void) src; (void) processxyzvw; res = NULL; return; }

// NOTE: the templated version above "misses" specialisations (sometimes, not always) for some reason I don't understand, so I do the following tedious method instead

#define OVERLAYMAKEFNVECTOR(type) \
\
inline void makeFuncVector(const std::string &typestring, Vector<type > *&res, std::istream &src); \
inline void makeFuncVector(const std::string &typestring, Vector<type > *&res, std::istream &src, int processxyzvw); \
\
inline void makeFuncVector(const std::string &typestring, Vector<type > *&res, std::istream &src) { throw("makeFuncVector stub called"); (void) typestring; (void) src; res = NULL; return; } \
inline void makeFuncVector(const std::string &typestring, Vector<type > *&res, std::istream &src, int processxyzvw) { throw("makeFuncVector stub b called"); (void) typestring; (void) src; (void) processxyzvw; res = NULL; return; } 

OVERLAYMAKEFNVECTOR(int)
OVERLAYMAKEFNVECTOR(double)
OVERLAYMAKEFNVECTOR(Vector<int>)
OVERLAYMAKEFNVECTOR(Vector<double>)



// Stream operators

template <class T>
std::ostream &operator<<(std::ostream &output, const Vector<T> &src)
{
    if ( src.imoverhere )
    {
        (src.overhere()).outstream(output);
    }

    else if ( src.infsize() )
    {
        src.outstream(output);
    }

    else
    {
        int xsize = src.size();

        output << "[ ";

        if ( xsize )
        {
	    int i;

            for ( i = 0 ; i < xsize ; i++ )
	    {
	        if ( i > 0 )
	        {
                    output << "  ";
	        }

	        if ( i < xsize-1 )
	        {
                    output << src(i) << "\t;" << src.getnewln();
	        }

	        else
	        {
		    output << src(i) << "\t";
                }
	    }
	}

        output << "  ]";
    }

    return output;
}

template <class T>
std::istream &operator>>(std::istream &input, Vector<T> &dest)
{
    // Note: we don't actually insist on ;'s being included in the input
    //       stream, so ( 1 5 2.2 ) is perfectly acceptable (as indeed is
    //       ( 2 2 ; 5 ; 1 2 3 ), though I would strongly advise against
    //       the latter... it probably won't be supported later).

    int i;
    char tt;

    // old version: pipe to buffer, NiceAssert(!strcmp(buffer,"["))

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    if ( ( tt == '[' ) && ( input.peek() == '[' ) )
    {
        input.get(tt);
        NiceAssert( tt == '[' );

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        std::string typestring;

        input >> typestring;

        if ( dest.imoverhere )
        {
            // dest is a front for an FuncVector

            if ( (dest.overhere()).testsametype(typestring) )
            {
                (dest.overhere()).instream(input);
            }

            else
            {
                MEMDEL(dest.imoverhere);
                makeFuncVector(typestring,dest.imoverhere,input);
            }
        }

        else if ( dest.infsize() && dest.testsametype(typestring) )
        {
            dest.instream(input);
        }

        else if ( dest.infsize() )
        {
            throw("Vector type mismatch in stream attempt");
        }

        else
        {
            // make dest a front for an FuncVector

            makeFuncVector(typestring,dest.imoverhere,input);
        }
    }

    else if ( tt == '[' )
    {
        if ( dest.imoverhere )
        {
            MEMDEL(dest.imoverhere);
            dest.imoverhere = NULL;
        }

        int xsize = 0;

        while ( 1 )
        {
            while ( ( isspace(input.peek()) ) || ( input.peek() == ';' ) )
            {
                input.get(tt);
            }

            if ( input.peek() == ']' )
            {
                input.get(tt);

                break;
            }

            if ( dest.size() == xsize )
            {
                dest.add(xsize);
            }

            input >> dest("&",xsize);

            xsize++;
        }

        if ( dest.size() > xsize )
        {
	    for ( i = dest.size()-1 ; i >= xsize ; i-- )
	    {
                dest.remove(xsize);
            }
	}
    }

    else
    {
        throw("Attempting to stream non-vector data to vector\n");
    }

    return input;
}

template <class T>
std::istream &streamItIn(std::istream &input, Vector<T> &dest, int processxyzvw)
{
    // Note: we don't actually insist on ;'s being included in the input
    //       stream, so ( 1 5 2.2 ) is perfectly acceptable (as indeed is
    //       ( 2 2 ; 5 ; 1 2 3 ), though I would strongly advise against
    //       the latter... it probably won't be supported later).

    int i;
    char tt;

    // old version: pipe to buffer, NiceAssert(!strcmp(buffer,"["))

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    if ( ( tt == '[' ) && ( input.peek() == '[' ) )
    {
        input.get(tt);
        NiceAssert( tt == '[' );

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        std::string typestring;

        input >> typestring;

        if ( dest.imoverhere )
        {
            // dest is a front for an FuncVector

            if ( (*(dest.imoverhere)).testsametype(typestring) )
            {
                (dest.overhere()).streamItIn(input,processxyzvw);
            }

            else
            {
                MEMDEL(dest.imoverhere);
                makeFuncVector(typestring,dest.imoverhere,input,processxyzvw);
            }
        }

        else if ( dest.infsize() && dest.testsametype(typestring) )
        {
            dest.streamItIn(input,processxyzvw);
        }

        else if ( dest.infsize() )
        {
            throw("Vector type mismatch in stream attempt");
        }

        else
        {
            // make dest a front for an FuncVector

            makeFuncVector(typestring,dest.imoverhere,input,processxyzvw);
        }
    }

    else if ( tt == '[' )
    {
        if ( dest.imoverhere )
        {
            MEMDEL(dest.imoverhere);
            dest.imoverhere = NULL;
        }

        int xsize = 0;

        while ( 1 )
        {
            while ( ( isspace(input.peek()) ) || ( input.peek() == ';' ) )
            {
                input.get(tt);
            }

            if ( input.peek() == ']' )
            {
                input.get(tt);

                break;
            }

            if ( dest.size() == xsize )
            {
                dest.add(xsize);
            }

            ::streamItIn(input,dest("&",xsize),processxyzvw);

            xsize++;
        }

        if ( dest.size() > xsize )
        {
	    for ( i = dest.size()-1 ; i >= xsize ; i-- )
	    {
                dest.remove(xsize);
            }
	}
    }

    else
    {
        throw("Attempting to streamItIn non-vector data to vector\n");
    }

    return input;
}




#endif


