
//
// Generic type
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _gentype_h
#define _gentype_h

class gentype;

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "basefn.h"
#include "numbase.h"
#include "anion.h"
#include "vector.h"
#include "sparsevector.h"
#include "matrix.h"
#include "set.h"
#include "dgraph.h"

#define DEFAULTVARI 0
#define DEFAULTVARJ 0

#define DEFAULT_INTEGRAL_SLICES 100


class gentype;


// Notes: doubleval mimics intval as far as possible to enable certain
// quick-and-dirty optimisations in the codebase.  However this is not
// guaranteed.  Specifically if you modify references directly (dirint, 
// forceint) then this will not be mimiced, so take care.
//
// Optimisation is as follows: scaladd assumes that all values
// are integer or double and does a += b*c.  scalmul performs a *= b.

inline double &scaladd(double &a, const gentype &b);
inline double &scaladd(double &a, const gentype &b, const gentype &c);
inline double &scaladd(double &a, const gentype &b, const gentype &c, const gentype &d);
inline double &scaladd(double &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e);
inline double &scalsub(double &a, const gentype &b);
inline double &scalmul(double &a, const gentype &b);
inline double &scaldiv(double &a, const gentype &b);

// The following make no such assumptions, needed for compatability

inline gentype &scaladd(gentype &a, const gentype &b);
inline gentype &scaladd(gentype &a, const gentype &b, const gentype &c);
inline gentype &scaladd(gentype &a, const gentype &b, const gentype &c, const gentype &d);
inline gentype &scaladd(gentype &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e);
inline gentype &scalsub(gentype &a, const gentype &b);
inline gentype &scalmul(gentype &a, const gentype &b);
inline gentype &scaldiv(gentype &a, const gentype &b);

inline void OP_exp (double &a) { a = exp(a);  }
inline void OP_sqrt(double &a) { a = sqrt(a); }
inline void OP_cos (double &a) { a = cos(a);  }
inline void OP_sin (double &a) { a = sin(a);  }
inline void OP_tan (double &a) { a = tan(a);  }
inline void OP_sec (double &a) { a = sec(a);  }
inline void OP_cosh(double &a) { a = cosh(a); }
inline void OP_sinh(double &a) { a = sinh(a); }
inline void OP_tanh(double &a) { a = tanh(a); }
inline void OP_sech(double &a) { a = sech(a); }
inline void OP_log (double &a) { a = log(a);  }
inline void OP_sinc(double &a) { a = sinc(a); }
inline void OP_agd (double &a) { a = agd(a);  }
inline void OP_einv(double &a) { a = 1/a;     }



// Multithreaded initialisation function:
//
// Initialises all derivatives in one block.  This is not required for single
// threaded operation, but for multithreaded use call this function first
// before starting any new threads.

void initgentype(void);

std::ostream &operator<<(std::ostream &output, const gentype &src );
std::istream &operator>>(std::istream &input,        gentype &dest);

// see makeEqn.  This is operator>> with that option.

//std::istream &streamItIn(std::istream &input, gentype &dest, int processxyzvw = 1);


// Vector assignment simplifications

Vector<gentype> &assign(Vector<gentype> &dest, const Vector<double > &src);
Vector<double > &assign(Vector<double > &dest, const Vector<gentype> &src);

// Simple interactive calculator

void intercalc(std::ostream &output, std::istream &input);

// Global variables.  As a rule var(i,j) is a local variable, but using
// these functions you can turn it into a global variable.  Global 
// variables are detected and evaluated immediately, taking precedence.
//
// setGlobalVar:   turns var(i,j) into global variable with value val
// getGloalVar:    returns value of global variable var(i,j) (NULL if not present)
// testGlobalVar:  return 1 if var(i,j) is global
// unsetGlobalVar: make var(i,j) non-global

void setGlobalVar(int i, int j, const gentype &val);
void getGlobalVar(int i, int j, gentype &res);
int testGlobalVar(int i, int j);
void unsetGlobalVar(int i, int j);

// Global functions.  This function table is access by fnB(i,arg) in
// gentype and allows gentype to build on top of pretty much anything.  
// These functions are assumed non-deterministic, so you need to finalise
// them when evaluating, either with cast_...(3), (double) or (int).
//
// fn(xa,ia) = d^ia/dx^ia f(xa,0)

typedef void (*GenFunc)(int i, int j, gentype &res, const gentype &xa, int ia, const gentype &xb, int ib);
typedef void (*eGenFunc)(int i, int j, Vector<gentype> &eres, const Vector<gentype> &exa, int ia, const Vector<gentype> &exb, int ib);

void setGenFunc(const GenFunc fnaddr);
void seteGenFunc(const eGenFunc efnaddr);

// vector helpers

void makeFuncVector(const std::string &typestring, Vector<gentype> *&res, std::istream &src); // operator>> analog
void makeFuncVector(const std::string &typestring, Vector<gentype> *&res, std::istream &src, int processxyzvw); // streamItIn analog

OVERLAYMAKEFNVECTOR(Vector<gentype>)
OVERLAYMAKEFNVECTOR(SparseVector<gentype>)

OVERLAYMAKEFNVECTOR(d_anion)
OVERLAYMAKEFNVECTOR(Vector<d_anion>)
OVERLAYMAKEFNVECTOR(Matrix<d_anion>)


// Swap function

inline void qswap(gentype &a, gentype &b);
inline void qswap(const gentype *&a, const gentype *&b);
inline void qswap(const gentype **&a, const gentype **&b);
inline void qswap(gentype *&a, gentype *&b);

// See code

class fninfoblock
{
    public:

    const char *fnname;
    int numargs;
    int dirchkargs;
    int widechkargs;
    int preEvalArgs;
    int derivDeffed;
    int isInDetermin;

    gentype (*fn0arg)();
    gentype (*fn1arg)(const gentype &);
    gentype (*fn2arg)(const gentype &, const gentype &);
    gentype (*fn3arg)(const gentype &, const gentype &, const gentype &);
    gentype (*fn4arg)(const gentype &, const gentype &, const gentype &, const gentype &);
    gentype (*fn5arg)(const gentype &, const gentype &, const gentype &, const gentype &, const gentype &);
    gentype (*fn6arg)(const gentype &, const gentype &, const gentype &, const gentype &, const gentype &, const gentype &);

    gentype &(*OP_fn0arg)();
    gentype &(*OP_fn1arg)(gentype &);
    gentype &(*OP_fn2arg)(gentype &, const gentype &);
    gentype &(*OP_fn3arg)(gentype &, const gentype &, const gentype &);
    gentype &(*OP_fn4arg)(gentype &, const gentype &, const gentype &, const gentype &);
    gentype &(*OP_fn5arg)(gentype &, const gentype &, const gentype &, const gentype &, const gentype &);
    gentype &(*OP_fn6arg)(gentype &, const gentype &, const gentype &, const gentype &, const gentype &, const gentype &);

    int conjargmod;
    const char *conjfnname;
    int fnconjind; // index of conjugate function

    int realargcopy;
    int realdrvcopy;
    gentype *realderiv;
    const char *realderivfn;

    ~fninfoblock()
    {
        return;
    }
};

const char *getfnname(int fnnameind);
int getfnind(const std::string &fnname);
int getfnindConj(int fnInd);
const fninfoblock *getfninfo(int fnIndex);

// Error constructors: combine errors from a,b,c,d and add string errstr to create error res
// Note that res = a,b,... is perfectly OK

void constructError(                                                                                                            gentype &res, const char *errstr);
void constructError(const gentype &a,                                                                                           gentype &res, const char *errstr);
void constructError(const gentype &a, const gentype &b,                                                                         gentype &res, const char *errstr);
void constructError(const gentype &a, const gentype &b, const gentype &c,                                                       gentype &res, const char *errstr);
void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d,                                     gentype &res, const char *errstr);
void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e,                   gentype &res, const char *errstr);
void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e, const gentype &f, gentype &res, const char *errstr);

void constructError(                                                                                                            gentype &res, const std::string &errstr);
void constructError(const gentype &a,                                                                                           gentype &res, const std::string &errstr);
void constructError(const gentype &a, const gentype &b,                                                                         gentype &res, const std::string &errstr);
void constructError(const gentype &a, const gentype &b, const gentype &c,                                                       gentype &res, const std::string &errstr);
void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d,                                     gentype &res, const std::string &errstr);
void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e,                   gentype &res, const std::string &errstr);
void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e, const gentype &f, gentype &res, const std::string &errstr);


// Needed because you can't have commas in macro arguments

typedef Dgraph<gentype,double> xDgraph;

// Defined early for reasons

gentype trans(const gentype &a);













class gentype
{
    friend inline void qswap(gentype &a, gentype &b);

    friend inline double &scaladd(double &a, const gentype &b);
    friend inline double &scaladd(double &a, const gentype &b, const gentype &c);
    friend inline double &scaladd(double &a, const gentype &b, const gentype &c, const gentype &d);
    friend inline double &scaladd(double &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e);
    friend inline double &scalsub(double &a, const gentype &b);
    friend inline double &scalmul(double &a, const gentype &b);
    friend inline double &scaldiv(double &a, const gentype &b);

public:

    // Constructors, destructor and assignment
    //
    // Notes: - default type is integer, 0
    //        - constructing from std::string or char * makes an equation,
    //          not a string (use makeString if you want a string).
    //        - likewise, assignment from std::String or char * makes equation

    svm_explicit gentype()
    {
        initBase();
        return;
    }

    gentype(const gentype &src)
    {
        initBase();
        fastcopy(src,1);
        return;
    }

    svm_explicit gentype(char typechar)
    {
        initBase();

        switch ( typechar )
        {
            case 'N': { typeis = 'N';                        break; }
            case 'Z': { typeis = 'Z';                        break; }
            case 'R': { typeis = 'R';                        break; }
            case 'A': { d_anion x;                *this = x; break; }
            case 'V': { Vector<gentype> x;        *this = x; break; }
            case 'M': { Matrix<gentype> x;        *this = x; break; }
            case 'X': { Set<gentype> x;           *this = x; break; }
            case 'G': { Dgraph<gentype,double> x; *this = x; break; }
            case 'S': { std::string x;            *this = x; break; }

            case 'E':
            case 'F':
            {
                throw("Illegal type in gentype constructor");
                break;
            }

            default:
            {
                throw("Unknown type in gentype constructor");
                break;
            }
        }

        return;
    }

//FIXME: string constructor should be explicit, but the alt usage is just too common to bother
    svm_explicit gentype(const int &src)                    { initBase(); typeis = 'Z'; intval    = src; doubleval = intval; return; }
    svm_explicit gentype(const double &src)                 { initBase(); typeis = 'R'; doubleval = src;                     return; }
    svm_explicit gentype(const d_anion &src)                { initBase(); *this = src;                                       return; }
    svm_explicit gentype(const Vector<gentype> &src)        { initBase(); *this = src;                                       return; }
    svm_explicit gentype(const Matrix<gentype> &src)        { initBase(); *this = src;                                       return; }
    svm_explicit gentype(const Set<gentype> &src)           { initBase(); *this = src;                                       return; }
    svm_explicit gentype(const Dgraph<gentype,double> &src) { initBase(); *this = src;                                       return; }
    svm_explicit gentype(const std::string &src)            { initBase(); *this = src;                                       return; }
    svm_explicit gentype(const char *src)                   { initBase(); *this = src;                                       return; }

    template <class T> svm_explicit gentype(const Vector<T> &src)        { initBase(); *this = src; return; }
    template <class T> svm_explicit gentype(const Matrix<T> &src)        { initBase(); *this = src; return; }
    template <class T> svm_explicit gentype(const Set<T> &src)           { initBase(); *this = src; return; }
    template <class T> svm_explicit gentype(const Dgraph<T,double> &src) { initBase(); *this = src; return; }

    ~gentype() { deleteVectMatMem(); return; }

    gentype &operator=(const gentype &src) { return fastcopy(src,0); }

    gentype &operator=(const int &src)                    { deleteVectMatMem('Z'                            ); typeis = 'Z'; intval     = src; doubleval = intval; return *this; }
    gentype &operator=(const double &src)                 { deleteVectMatMem('R'                            ); typeis = 'R'; doubleval  = src;                     return *this; }
    gentype &operator=(const d_anion &src)                { deleteVectMatMem('A'                            ); typeis = 'A'; *anionval  = src;                     return *this; }
    gentype &operator=(const Vector<gentype> &src)        { deleteVectMatMem('V',src.size()                 ); typeis = 'V'; *vectorval = src;                     return *this; }
    gentype &operator=(const Matrix<gentype> &src)        { deleteVectMatMem('M',src.numRows(),src.numCols()); typeis = 'M'; *matrixval = src;                     return *this; }
    gentype &operator=(const Set<gentype> &src)           { deleteVectMatMem('X'                            ); typeis = 'X'; *setval    = src;                     return *this; }
    gentype &operator=(const Dgraph<gentype,double> &src) { deleteVectMatMem('G'                            ); typeis = 'G'; *dgraphval = src;                     return *this; }

    gentype &operator=(const std::string &src) { deleteVectMatMem(); makeEqn(src); return *this; }
    gentype &operator=(const char *src)        { deleteVectMatMem(); makeEqn(src); return *this; }

    template <class T> gentype &operator=(const Vector<T> &src)
    {
        deleteVectMatMem('V',src.size());
        typeis = 'V';
        //(*vectorval).resize(src.size());
        (*vectorval).castassign(src);

//        (*vectorval).resize(src.size());
//
//        if ( src.size() )
//        {
//            int i;
//
//            for ( i = 0 ; i < src.size() ; i++ )
//            {
//                (*vectorval)("&",i) = src(i);
//            }
//        }
    
        return *this;
    }

    template <class T> gentype &operator=(const Matrix<T> &src)
    {
        deleteVectMatMem('M',src.numRows(),src.numCols());
        typeis = 'M';
        (*matrixval).resize(src.numRows(),src.numCols());

        if ( src.numRows() && src.numCols() )
        {
            int i,j;

            for ( i = 0 ; i < src.numRows() ; i++ )
            {
                for ( j = 0 ; j < src.numCols() ; j++ )
                {
                    (*matrixval)("&",i,j) = src(i,j);
                }
            }
        }
    
        return *this;
    }

    template <class T> gentype &operator=(const Set<T> &src)
    {
        deleteVectMatMem('X');
        typeis = 'X';
        (*setval).zero();

        if ( src.size() )
        {
            int i;
            gentype x;

            for ( i = 0 ; i < src.size() ; i++ )
            {
                (*setval).add(x = (src.all())(i));
            }
        }
    
        return *this;
    }

    template <class T> gentype &operator=(const Dgraph<T,double> &src)
    {
        deleteVectMatMem('G');
        typeis = 'G';
        (*dgraphval).zero();

        if ( src.size() )
        {
            int i,j;
            gentype x;

            for ( i = 0 ; i < src.size() ; i++ )
            {
                (*dgraphval).add(x = (src.all())(i));
            }

            for ( i = 0 ; i < src.size() ; i++ )
            {
                for ( j = 0 ; j < src.size() ; j++ )
                {
                    (*dgraphval).setWeight(i,j,(src.edgeWeights())(i,j));
                }
            }
        }
    
        return *this;
    }

    // Make gentype a string

    void makeString(const std::string &src) { deleteVectMatMem('S'); typeis = 'S'; *stringval = src; return; }
    void makeString(const char *src)        { deleteVectMatMem('S'); typeis = 'S'; *stringval = src; return; }

    // Make gentype an error

    void makeError(const std::string &src) { deleteVectMatMem('E'); typeis = 'E'; *stringval = src; return; }
    void makeError(const char *src)        { deleteVectMatMem('E'); typeis = 'E'; *stringval = src; return; }

    // Make gentype a null

    gentype &makeNull(void) { deleteVectMatMem(); typeis = 'N'; return *this; }

    // Write equation string (note that the term *this = (*this)()
    // automatically simplifies the equation)
    //
    // By default processxyzvw == 1, which means that x -> var(0,0), 
    // y -> var(0,1) etc.  So "sin(x)" is processed as sin(var(0,0)).  This
    // is a helpful shortcut for readable equations.  If we set
    // processxyzvw == 0 then this changes to x -> "x" (ie string),
    // y -> "y" etc.
    //
    // x,y,z,v,w,g -> var(0,0), var(0,1), ..., var(0,5)
    // h -> var(2,0)

    int makeEqn(const std::string &src, int processxyzvw = 1);
    int makeEqn(const char *src,        int processxyzvw = 1);

    // Type testing
    //
    // isVal... returns true if type is ...
    // isCastableTo...WithoutLoss returns true if the data can be cast to
    //          type ... without loss of information - eg an integer can be
    //          cast as a double without loss of information, but not
    //          necessarily vice-versa.
    // isCastableTo... returns true if the data can be cast to type ... with
    //          or without loss of information - eg a real can be cast as
    //          an int via rounding off.  Note that this only applies to
    //          rounding type operations, not operations that lose whole
    //          components, so eg an anion can be cast to real if it has
    //          no imaginary component but not otherwise.
    // realDerivDefinedDir returns true if this is not an equation or it is
    //          an equation and the base node in that equation has a well
    //          defined derivative (which is true for most equations except
    //          for things like prod).  Note that this does not imply that
    //          the complete structure has a well defined derivative, only
    //          that this (base) node in the structure does.
    // isfasttype: returns true for integers, reals and anions.
    //
    // Notes: - isValEqnDir return true if this is an equation but not if
    //          this is for example a vector, matrix or set whose components
    //          include equations.
    //        - isValEqn returns true if this is an equation or this is a
    //          vector, matrix or set containing at least one element for
    //          which isValEqn returns true (ie the element is an equation
    //          or a vector/matrix/set containing an equation at any depth
    //          down the evaluation tree).
    //        - isValEqn actually returns bitfield:
    //          1 contains indeterminant random or global parts
    //          2 contains deterministic function
    //          4 contains indeterminant random parts
    //          8 contains indeterminant global parts
    //          16 contains variables (isValEqn() only)
    //        - it is not possible to test the castability of a non-deterministic 
    //          function (randoms, call to global function) without actually
    //          casting.

    int isValNull     (void) const { return typeis == 'N'; }
    int isValInteger  (void) const { return typeis == 'Z'; }
    int isValReal     (void) const { return typeis == 'R'; }
    int isValAnion    (void) const { return typeis == 'A'; }
    int isValVector   (void) const { return typeis == 'V'; }
    int isValMatrix   (void) const { return typeis == 'M'; }
    int isValSet      (void) const { return typeis == 'X'; }
    int isValDgraph   (void) const { return typeis == 'G'; }
    int isValString   (void) const { return typeis == 'S'; }
    int isValError    (void) const { return typeis == 'E'; }
    int isValEqnDir   (void) const { return ( typeis == 'F' ) ? ( ( (*thisfninfo).isInDetermin ? 1 : 2 ) | ( ( (*thisfninfo).isInDetermin & 2 ) ? 4 : 0 ) | ( ( (*thisfninfo).isInDetermin & 1 ) ? 8 : 0 ) ) : 0; }

    int isValStrErr   (void) const { return isValString() |  isValError();         }
    int isValEqn      (void) const { return isValEqnFull(); } //isValEqnDir() |  isValEqnFull();       }
    int isValAnionReal(void) const { return isValAnion()  && (*anionval).isreal(); }

    int isCastableToNullWithoutLoss   (void) const { return isValNull(); }
    int isCastableToIntegerWithoutLoss(void) const { return ( isValNull() || isValInteger() || ( isCastableToRealWithoutLoss() && ( cast_double(0) == (double) cast_int(0) ) ) ); }
    int isCastableToRealWithoutLoss   (void) const { return ( isValNull() || isValInteger() || isValReal() || ( isValAnion() && ( size() == 1 ) ) || ( isValVector() && ( size() == 1 ) && (*vectorval)(zeroint()).isCastableToRealWithoutLoss() ) || ( isValMatrix() && ( numRows() == 1 ) && ( numCols() == 1 ) && (*matrixval)(zeroint(),zeroint()).isCastableToRealWithoutLoss() ) ); }
    int isCastableToAnionWithoutLoss  (void) const { return ( isValNull() || isValInteger() || isValReal() || isValAnion() || ( isValVector() && ( size() == 1 ) && (*vectorval)(zeroint()).isCastableToAnionWithoutLoss() ) || ( isValMatrix() && ( numRows() == 1 ) && ( numCols() == 1 ) && (*matrixval)(zeroint(),zeroint()).isCastableToAnionWithoutLoss() ) ); }
    int isCastableToVectorWithoutLoss (void) const { return ( isValNull() || isValInteger() || isValReal() || isValAnion() || isValVector() || ( isValMatrix() && ( ( (*matrixval).numRows() == 1 ) || ( (*matrixval).numCols() == 1 ) || (*matrixval).isEmpty() ) ) ); }
    int isCastableToMatrixWithoutLoss (void) const { return ( isValNull() || isValInteger() || isValReal() || isValAnion() || ( isValVector() && !infsize() ) || isValMatrix() ); }
    int isCastableToSetWithoutLoss    (void) const { return ( isValNull() || isValSet()    );    }
    int isCastableToDgraphWithoutLoss (void) const { return ( isValNull() || isValDgraph() ); }
    int isCastableToStringWithoutLoss (void) const { return !isValError();  }

    int isCastableToNull   (void) const { return isValNull(); }
    int isCastableToInteger(void) const { return isCastableToReal(); }
    int isCastableToReal   (void) const { return ( isValNull() || isValInteger() || isValReal() || ( isValAnion() && ( size() == 1 ) ) || ( isValVector() && ( size() == 1 ) && (*vectorval)(zeroint()).isCastableToReal() ) || ( isValMatrix() && ( numRows() == 1 ) && ( numCols() == 1 ) && (*matrixval)(zeroint(),zeroint()).isCastableToReal() ) ); }
    int isCastableToAnion  (void) const { return ( isValNull() || isValInteger() || isValReal() || isValAnion() || ( isValVector() && ( size() == 1 ) && (*vectorval)(zeroint()).isCastableToAnion() ) || ( isValMatrix() && ( numRows() == 1 ) && ( numCols() == 1 ) && (*matrixval)(zeroint(),zeroint()).isCastableToAnion() ) ); }
    int isCastableToVector (void) const { return ( isValNull() || isValInteger() || isValReal() || isValAnion() || isValVector() || ( isValMatrix() && ( ( (*matrixval).numRows() == 1 ) || ( (*matrixval).numCols() == 1 ) || (*matrixval).isEmpty() ) ) ); }
    int isCastableToMatrix (void) const { return ( isValNull() || isValInteger() || isValReal() || isValAnion() || isValVector() || isValMatrix() ); }
    int isCastableToSet    (void) const { return ( isValNull() || isValSet()    );  }
    int isCastableToDgraph (void) const { return ( isValNull() || isValDgraph() ); }
    int isCastableToString (void) const { return !isValError(); }

    int realDerivDefinedDir(void) const { return isValEqnDir() ? (*thisfninfo).derivDeffed : 1;  }
    int isfasttype(void) const { return isValNull() || isValInteger() || isValReal(); }
    int iseq(const gentype &b) const;

    int isNotImaginary(void) const { return isCastableToRealWithoutLoss(); }
    int isCommutative (void) const { return isNotImaginary() || ( isValAnion() && ( (*anionval).order() <= 1 ) ); }
    int isAssociative (void) const { return isNotImaginary() || ( isValAnion() && ( (*anionval).order() <= 2 ) ); }

    // Size information:
    //
    // size, numRows and numCols return 1 with the following exceptions:
    //
    // anions:  size is the number of elements (2^n, where n is the order)
    // vectors: size and numRows return the size of the vector
    // matrix:  numRows/numCols reflect matrix dimensions, size=numRows*numCols
    // set:     size and numRows return the number of elements in the set
    // dgraph:  size returns number of nodes, numRows/numCols return size
    //          of the edge matrix.
    // strings: size returns the length of the string.
    // errors:  size returns the length of the error string.

    int size(void) const 
    { 
        switch ( typeis )
        {
            case 'N': { return 0;                           break; }
            case 'Z': { return 1;                           break; }
            case 'R': { return 1;                           break; }
            case 'A': { return (*anionval).size();          break; }
            case 'V': { return (*vectorval).size();         break; }
            case 'M': { return (*matrixval).size();         break; }
            case 'X': { return (*setval).size();            break; }
            case 'G': { return (*dgraphval).size();         break; }
            case 'S': { return (int) (*stringval).length(); break; }
            case 'E': { return (int) (*stringval).length(); break; }
            default:
            {
                int res = 0;
                int i;

                if ( (*eqnargs).size() )
                {
                    for ( i = 0 ; i < (*eqnargs).size() ; i++ )
                    {
                        res += ((*eqnargs)(i)).size();
                    }
                }

                return res;
            }
        }

        return 0;
    }

    int infsize(void) const 
    { 
        return ( typeis == 'V' ) ? (*vectorval).infsize() : 0;
    }

    int order(void) const
    { 
        switch ( typeis )
        {
            case 'N': { return 0;                   break; }
            case 'Z': { return 0;                   break; }
            case 'R': { return 0;                   break; }
            case 'A': { return (*anionval).order(); break; }
            case 'V': { return ceilintlog2(size()); break; }
            case 'M': { return ceilintlog2(size()); break; }
            case 'X': { return ceilintlog2(size()); break; }
            case 'G': { return ceilintlog2(size()); break; }
            case 'S': { return ceilintlog2(size()); break; }
            case 'E': { return ceilintlog2(size()); break; }
            default:
            {
                int res = 0;
                int i;

                if ( (*eqnargs).size() )
                {
                    for ( i = 0 ; i < (*eqnargs).size() ; i++ )
                    {
                        res += ((*eqnargs)(i)).order();
                    }
                }

                return res;
            }
        }

        return 0;
    }

    int numRows(void) const
    { 
        switch ( typeis )
        {
            case 'N': { return 0;                      break; }
            case 'V': { return (*vectorval).size();    break; }
            case 'M': { return (*matrixval).numRows(); break; }
            case 'X': { return (*setval).size();       break; }
            case 'G': { return (*dgraphval).size();    break; }
            default:  { return 1;                      break; }
        }

        return 1;
    }

    int numCols(void) const
    { 
        switch ( typeis )
        {
            case 'N': { return 0;                      break; }
            case 'M': { return (*matrixval).numCols(); break; }
            case 'G': { return (*dgraphval).numCols(); break; }
            default:  { return 1;                      break; }
        }

        return 1;
    }

    // Casting operators (do not change object)
    //
    // casting  returns actual value and throws an exception otherwise
    // cast_... cast object to type ..., return ref, exception on failure.
    // to...    convert res to ... type, overwrite with cast version of
    //          object, make res an error string if cast fails.
    // force... force casting to destination, no errors, data will be lost,
    //          default to zero sized vector, 0x0 matrix
    //
    // Morphing and non-const access (do change object)
    //
    // morph_...changes type of object, converts to error on fail
    // dir_...  changes type of object, returns reference to content
    //          (content may be used or changed as required - ie the
    //          returned references gives direct access to the value).
    //
    // Other:
    //
    // quickrealpart: equivalent to (double) real(*this), but faster.
    //          This function will finalise before returning, so 
    //          non-deterministic parts (globals and randoms) will be evaluated.
    // finalise: evaluate non-deterministic parts as far as possible
    //          0 means don't finalise anything
    //          1 means finalise randoms (indetermin 2)
    //          2 means finalise globals (indetermin 1)
    //          3 means both
    //          NB: finalisation is *bottom up*, which may effect results
    //          eg finalise(3) fnA(...,urand(0,1)) will first evaluate
    //             urand, then fnA.  This may be bad eg for evaluating
    //             a kernel on a distribution, so I recommend instead
    //             using finalise(2), finalise(1)

    int finalise(int finalise = 3) { const static SparseVector<SparseVector<gentype> > temp; return fastevaluate(temp,finalise); }

    operator int()    const { return cast_int(3);    }
    operator double() const { return cast_double(3); }

    operator const d_anion &()                const { return cast_anion(3);       }
    operator const Vector<gentype> &()        const { return cast_vector(3);      }
    operator const Vector<double> &()         const { return cast_vector_real(3); }
    operator const Matrix<gentype> &()        const { return cast_matrix(3);      }
    operator const Matrix<double> &()         const { return cast_matrix_real(3); }
    operator const Set<gentype> &()           const { return cast_set(3);         }
    operator const Dgraph<gentype,double> &() const { return cast_dgraph(3);      }
    operator const std::string &()            const { return cast_string(3);      }

    void                          cast_null       (int finalise = 0) const;
    const int                    &cast_int        (int finalise = 0) const;
    const double                 &cast_double     (int finalise = 0) const;
    const d_anion                &cast_anion      (int finalise = 0) const;
    const Vector<gentype>        &cast_vector     (int finalise = 0) const;
    const Vector<double>         &cast_vector_real(int finalise = 0) const;
    const Matrix<gentype>        &cast_matrix     (int finalise = 0) const;
    const Matrix<double>         &cast_matrix_real(int finalise = 0) const;
    const Set<gentype>           &cast_set        (int finalise = 0) const;
    const Dgraph<gentype,double> &cast_dgraph     (int finalise = 0) const;
    const std::string            &cast_string     (int finalise = 0) const;

    gentype &toNull   (gentype &res) const;
    gentype &toInteger(gentype &res) const;
    gentype &toReal   (gentype &res) const;
    gentype &toAnion  (gentype &res) const;
    gentype &toVector (gentype &res) const;
    gentype &toMatrix (gentype &res) const;
    gentype &toSet    (gentype &res) const;
    gentype &toDgraph (gentype &res) const;
    gentype &toString (gentype &res) const;

    gentype &morph_null  (void);
    gentype &morph_int   (void);
    gentype &morph_double(void);
    gentype &morph_anion (void);
    gentype &morph_vector(void);
    gentype &morph_matrix(void);
    gentype &morph_set   (void);
    gentype &morph_dgraph(void);
    gentype &morph_string(void);

    void                    dir_null  (void) { if ( !isValNull()    ) { morph_null();   } NiceAssert(isValNull());    return;            }
    int                    &dir_int   (void) { if ( !isValInteger() ) { morph_int();    } NiceAssert(isValInteger()); return intval;     }
    double                 &dir_double(void) { if ( !isValReal()    ) { morph_double(); } NiceAssert(isValReal());    return doubleval;  }
    d_anion                &dir_anion (void) { if ( !isValAnion()   ) { morph_anion();  } NiceAssert(isValAnion());   return *anionval;  }
    Vector<gentype>        &dir_vector(void) { if ( !isValVector()  ) { morph_vector(); } NiceAssert(isValVector());  return *vectorval; }
    Matrix<gentype>        &dir_matrix(void) { if ( !isValMatrix()  ) { morph_matrix(); } NiceAssert(isValMatrix());  return *matrixval; }
    Set<gentype>           &dir_set   (void) { if ( !isValSet()     ) { morph_set();    } NiceAssert(isValSet());     return *setval;    }
    Dgraph<gentype,double> &dir_dgraph(void) { if ( !isValDgraph()  ) { morph_dgraph(); } NiceAssert(isValDgraph());  return *dgraphval; }
    std::string            &dir_string(void) { if ( !isValString()  ) { morph_string(); } NiceAssert(isValString());  return *stringval; }

    void                    force_null  (void)                 { makeNull();                                                                                                                    return;            }
    int                    &force_int   (void)                 { const static int                    defaultval = 0;   if ( !isValInteger() ) { *this = defaultval; }                           return intval;     }
    double                 &force_double(void)                 { const static double                 defaultval = 0.0; if ( !isValReal()    ) { *this = defaultval; }                           return doubleval;  }
    d_anion                &force_anion (void)                 {                                                       if ( !isValAnion()   ) { *this = zeroanion(); }                          return *anionval;  }
    Vector<gentype>        &force_vector(int i = 0)            { const static Vector<gentype>        defaultval;       if ( !isValVector()  ) { *this = defaultval; } (*vectorval).resize(i);   return *vectorval; }
    Matrix<gentype>        &force_matrix(int i = 0, int j = 0) { const static Matrix<gentype>        defaultval;       if ( !isValMatrix()  ) { *this = defaultval; } (*matrixval).resize(i,j); return *matrixval; }
    Set<gentype>           &force_set   (void)                 { const static Set<gentype>           defaultval;       if ( !isValSet()     ) { *this = defaultval; }                           return *setval;    }
    Dgraph<gentype,double> &force_dgraph(void)                 { const static Dgraph<gentype,double> defaultval;       if ( !isValDgraph()  ) { *this = defaultval; }                           return *dgraphval; }
    std::string            &force_string(void)                 { makeString("");                                                                                                                return *stringval; }

    double quickrealpart(void) const
    {
        switch ( typeis )
        {
            case 'N': { return 0.0;                                               break; }
            case 'Z': { return intval;                                            break; }
            case 'R': { return doubleval;                                         break; }
            case 'A': { return anionval->realpart();                              break; }
            case 'V': { return (*vectorval)(zeroint()).quickrealpart();           break; }
            case 'M': { return (*matrixval)(zeroint(),zeroint()).quickrealpart(); break; }
            case 'F': { return cast_double(3);                                    break; }
            default:  { throw("Can't convert value to real");                     break; }
        }

        return 0.0;
    }

    // Variables usage information:
    //
    // varsUsed: returns a sparse matrix where elements corresponding to
    //           variables used in this equation are set 1 (all others not
    //           present, so nominally 0).  This will not work for unresolved
    //           variables (eg var(.,var(.,.))) or vectorial/matrix
    //           dereferences of variables (eg var([1 2 9],0)).
    // rowsUse:  like varsUsed, but reflects only which rows i of var(i,...)
    //           are used in this equation

    SparseVector<SparseVector<int> > varsUsed(void) const { SparseVector<SparseVector<int> > temp; return varsUsed(temp); }
    SparseVector<int>                rowsUsed(void) const { SparseVector<int> temp;                return rowsUsed(temp); }

    // Evaluate equation
    //
    // Note that x = var(0,0), y = var(0,1) ...

    gentype operator()(void) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs;

        return (*this)(evalargs); 
    }

    gentype operator()(const gentype &x) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x;

        return (*this)(evalargs);
    }

    gentype operator()(const gentype &x, const gentype &y) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y;

        return (*this)(evalargs);
    }

    gentype evalyonly(const gentype &y) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",1) = y;

        return (*this)(evalargs);
    }

    gentype operator()(const gentype &x, const gentype &y, const gentype &z) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z;

        return (*this)(evalargs);
    }

    gentype operator()(const gentype &x, const gentype &y, const gentype &z, const gentype &v) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z; 
        evalargs("&",0)("&",3) = v;

        return (*this)(evalargs);
    }

    gentype operator()(const gentype &x, const gentype &y, const gentype &z, const gentype &v, const gentype &w) const 
    {
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z; 
        evalargs("&",0)("&",3) = v; evalargs("&",0)("&",4) = w;

        return (*this)(evalargs); 
    }

    gentype operator()(const gentype &x, const gentype &y, const gentype &z, const gentype &v, const gentype &w, const gentype &g) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z; 
        evalargs("&",0)("&",3) = v; evalargs("&",0)("&",4) = w; evalargs("&",0)("&",5) = g; 

        return (*this)(evalargs);
    }

    gentype operator()(const char *, const gentype &h) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",2)("&",0) = h; 

        return (*this)(evalargs); 
    }

    gentype operator()(const gentype &x, const char *, const gentype &h) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; 
        evalargs("&",2)("&",0) = h; 

        return (*this)(evalargs); 
    }

    gentype operator()(const gentype &x, const gentype &y, const char *, const gentype &h) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; 
        evalargs("&",2)("&",0) = h; 

        return (*this)(evalargs); 
    }

    gentype operator()(const gentype &x, const gentype &y, const gentype &z, const char *, const gentype &h) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z; 
        evalargs("&",2)("&",0) = h; 

        return (*this)(evalargs); 
    }

    gentype operator()(const gentype &x, const gentype &y, const gentype &z, const gentype &v, const char *, const gentype &h) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z; 
        evalargs("&",0)("&",3) = v; 
        evalargs("&",2)("&",0) = h; 

        return (*this)(evalargs); 
    }

    gentype operator()(const gentype &x, const gentype &y, const gentype &z, const gentype &v, const gentype &w, const char *, const gentype &h) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z; 
        evalargs("&",0)("&",3) = v; evalargs("&",0)("&",4) = w; 
        evalargs("&",2)("&",0) = h; 

        return (*this)(evalargs); 
    }

    gentype operator()(const gentype &x, const gentype &y, const gentype &z, const gentype &v, const gentype &w, const gentype &g, const char *, const gentype &h) const 
    { 
        SparseVector<SparseVector<gentype> > evalargs; 

        evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z; 
        evalargs("&",0)("&",3) = v; evalargs("&",0)("&",4) = w; evalargs("&",0)("&",5) = g; 
        evalargs("&",2)("&",0) = h; 

        return (*this)(evalargs); 
    }

    gentype operator()(const SparseVector<SparseVector<gentype> > &evalargs) const 
    { 
        gentype res(*this); 

        res.substitute(evalargs);

        return res;
    }

    // Dereference operators:
    //
    // These act like the dereference operators in vector, matrix, set and
    // dgraph.  Note that some forms of matrix dereference are not
    // implemented as these would conflict ambiguously with the vector
    // dereferences.  Non-const operations first morph, then dereference,
    // const operations use cast_....

    gentype         &operator()(const char *dummy, int i)                                           { return dir_vector()(dummy,i);            }
    Vector<gentype> &operator()(const char *dummy, int ib, int is, int im, retVector<gentype> &tmp) { return dir_vector()(dummy,ib,is,im,tmp); }
    Vector<gentype> &operator()(const char *dummy, const Vector<int> &i, retVector<gentype> &tmp)   { return dir_vector()(dummy,i,tmp);        }

    const gentype         &operator()(int i)                                           const { return cast_vector(0)(i);            }
    const Vector<gentype> &operator()(int ib, int is, int im, retVector<gentype> &tmp) const { return cast_vector(0)(ib,is,im,tmp); }
    const Vector<gentype> &operator()(const Vector<int> &i, retVector<gentype> &tmp)   const { return cast_vector(0)(i,tmp);        }

    gentype         &operator()(const char *dummy, int i, int j)                                                                   { return dir_matrix()(dummy,i,j);                   }
    Vector<gentype> &operator()(const char *dummy, int i, const Vector<int> &j, retVector<gentype> &tmp, retVector<gentype> &tmpb) { return dir_matrix()(dummy,i,j,tmp,tmpb);          }
    Vector<gentype> &operator()(const char *dummy, int i, int jb, int js, int jm, retVector<gentype> &tmp)                         { return dir_matrix()(dummy,i,jb,js,jm,tmp);        }
    Matrix<gentype> &operator()(const char *dummy, const Vector<int> &i, int j, retMatrix<gentype> &tmp)                           { return dir_matrix()(dummy,i,j,tmp);               }
    Matrix<gentype> &operator()(const char *dummy, const Vector<int> &i, const Vector<int> &j, retMatrix<gentype> &tmp)            { return dir_matrix()(dummy,i,j,tmp);               }
    Matrix<gentype> &operator()(const char *dummy, const Vector<int> &i, int jb, int js, int jm, retMatrix<gentype> &tmp)          { return dir_matrix()(dummy,i,jb,js,jm,tmp);        }
    Matrix<gentype> &operator()(const char *dummy, int ib, int is, int im, int j, retMatrix<gentype> &tmp, const char *dummyb)     { return dir_matrix()(dummy,ib,is,im,j,tmp,dummyb); }
    Matrix<gentype> &operator()(const char *dummy, int ib, int is, int im, const Vector<int> &j, retMatrix<gentype> &tmp)          { return dir_matrix()(dummy,ib,is,im,j,tmp);        }
    Matrix<gentype> &operator()(const char *dummy, int ib, int is, int im, int jb, int js, int jm, retMatrix<gentype> &tmp)        { return dir_matrix()(dummy,ib,is,im,jb,js,jm,tmp); }

    const gentype         &operator()(int i, int j)                                                                   const { return cast_matrix(0)(i,j);                   }
    const Vector<gentype> &operator()(int i, const Vector<int> &j, retVector<gentype> &tmp, retVector<gentype> &tmpb) const { return cast_matrix(0)(i,j,tmp,tmpb);          }
    const Vector<gentype> &operator()(int i, int jb, int js, int jm, retVector<gentype> &tmp)                         const { return cast_matrix(0)(i,jb,js,jm,tmp);        }
    const Matrix<gentype> &operator()(const Vector<int> &i, int j, retMatrix<gentype> &tmp)                           const { return cast_matrix(0)(i,j,tmp);               }
    const Matrix<gentype> &operator()(const Vector<int> &i, const Vector<int> &j, retMatrix<gentype> &tmp)            const { return cast_matrix(0)(i,j,tmp);               }
    const Matrix<gentype> &operator()(const Vector<int> &i, int jb, int js, int jm, retMatrix<gentype> &tmp)          const { return cast_matrix(0)(i,jb,js,jm,tmp);        }
    const Matrix<gentype> &operator()(int ib, int is, int im, int j, const char *dummyb, retMatrix<gentype> &tmp)     const { return cast_matrix(0)(ib,is,im,j,tmp,dummyb); }
    const Matrix<gentype> &operator()(int ib, int is, int im, const Vector<int> &j, retMatrix<gentype> &tmp)          const { return cast_matrix(0)(ib,is,im,j,tmp);        }
    const Matrix<gentype> &operator()(int ib, int is, int im, int jb, int js, int jm, retMatrix<gentype> &tmp)        const { return cast_matrix(0)(ib,is,im,jb,js,jm,tmp); }

    const Vector<gentype> &all(void) const { if ( isValSet() ) { return cast_set(0).all();       } return cast_dgraph(0).all();       }
    int contains(const gentype &x)   const { if ( isValSet() ) { return cast_set(0).contains(x); } return cast_dgraph(0).contains(x); }

    const Matrix<double> &edgeWeights(void)                const { return cast_dgraph(0).edgeWeights();    }
    double edgeWeights(const gentype &x, const gentype &y) const { return cast_dgraph(0).edgeWeights(x,y); }

    // Other stuff
    //
    // ident:   set this = 1 (real)
    // zero:    set this = 0 (real)
    // zeropassive: sets zero without changing type
    // posate:  does nothing
    // negate:  negates this (negating a string reverse it)
    // transpose: transpose this if vector, matrix or set (transpose also
    //          applied to the elements of the vector/matrix/set)
    // conj:    conjugate this (does nothing to strings)
    // inverse: inverts this (converts string/set/dgraph/vector to error
    //
    // substitute: substitute into equation.
    //
    // realDeriv: derivative with respect to real variable (i,j)
    //        - if i,j are integers then this works as expected
    //        - if i is an integer and j a vector then the result is:
    //          [ df/dvar(i(0),j) ]
    //          [ df/dvar(i(1),j) ]
    //          [       ...       ]
    //        - if i is a vector and j an integer then the result is:
    //          [ df/dvar(i,j(0)) ]
    //          [ df/dvar(i,j(1)) ]
    //          [       ...       ]
    //        - if i,j are vectors then the result is:
    //          [ df/dvar(i(0),j(0)) df/dvar(i(0),j(1)) ... ]
    //          [ df/dvar(i(1),j(0)) df(dvar(i(1),j(1)) ... ]
    //          [       ...                ...          ... ]
    //        - these definitions work recursively
    //        - other cases not defined
    //        - return >0 if anything changed

    gentype &ident(void) 
    {
        return ( *this = 1 );
    }

    gentype &zero(void)
    {
        return (*this = 0 );
    }

    gentype &zeropassive(void)
    {
             if ( isValNull()    ) { ;                          }
        else if ( isValInteger() ) { intval = 0; doubleval = 0; }
        else if ( isValReal()    ) { doubleval = 0;             }
        else if ( isValAnion()   ) { (*anionval) = 0.0;         }
        else if ( isValVector()  ) { (*vectorval).zero();       }
        else if ( isValMatrix()  ) { (*matrixval).zero();       }
        else if ( isValSet()     ) { (*setval).zero();          }
        else if ( isValDgraph()  ) { (*dgraphval).zero();       }
        else if ( isValString()  ) { (*stringval) = "";         }
        else if ( isValError()   ) { ;                          }
        else                       { *this = zeroint();         }

        return *this;
    }

    gentype &posate(void)
    {
        const static int posInd = getfnind("pos");

             if ( isValNull()    ) { ; }
        else if ( isValInteger() ) { ; }
        else if ( isValReal()    ) { ; }
        else if ( isValAnion()   ) { ; }
        else if ( isValVector()  ) { (*vectorval).posate(); }
        else if ( isValMatrix()  ) { (*matrixval).posate(); }
        else if ( isValSet()     ) { (*setval).posate();    }
        else if ( isValDgraph()  ) { (*dgraphval).posate(); }
        else if ( isValString()  ) { ; }
        else if ( isValError()   ) { ; }
        else
        {
            // Method: construct a new gentype, qswap (fast) this
            // new gentype with the present equation, then set
            // the present equation as neg with the argument
            // being the newly created gentype.

            Vector<gentype> *tempeqnargs;

            MEMNEW(tempeqnargs,Vector<gentype>(1));

            (*tempeqnargs)("&",zeroint()) = zeroint();
            qswap(*this,(*tempeqnargs)("&",zeroint()));

            typeis     = 'F';
            eqnargs    = tempeqnargs;
            fnnameind  = posInd;
            thisfninfo = getfninfo(fnnameind);
        }

        return *this;
    }

    gentype &negate(void)
    {
        const static int negInd = getfnind("neg");

             if ( isValNull()    ) { ;                                 }
        else if ( isValInteger() ) { intval *= -1; doubleval = intval; }
        else if ( isValReal()    ) { doubleval *= -1;                  }
        else if ( isValAnion()   ) { (*anionval) *= -1.0;              }
        else if ( isValVector()  ) { (*vectorval).negate();            }
        else if ( isValMatrix()  ) { (*matrixval).negate();            }
        else if ( isValSet()     ) { (*setval).negate();               }
        else if ( isValDgraph()  ) { (*dgraphval).negate();            }
        else if ( isValString()  ) { reversestring();                  }
        else if ( isValError()   ) { ;                                 }
        else
        {
            // Method: construct a new gentype, qswap (fast) this
            // new gentype with the present equation, then set
            // the present equation as neg with the argument
            // being the newly created gentype.

            Vector<gentype> *tempeqnargs;

            MEMNEW(tempeqnargs,Vector<gentype>(1));

            (*tempeqnargs)("&",zeroint()) = zeroint();
            qswap(*this,(*tempeqnargs)("&",zeroint()));

            typeis     = 'F';
            eqnargs    = tempeqnargs;
            fnnameind  = negInd;
            thisfninfo = getfninfo(fnnameind);
        }

        return *this;
    }

    gentype &transpose(void)
    {
        const static int transInd = getfnind("trans");

             if ( isValNull()    ) { ;                                                      }
        else if ( isValInteger() ) { ;                                                      }
        else if ( isValReal()    ) { ;                                                      }
        else if ( isValAnion()   ) { ;                                                      }
        else if ( isValVector()  ) { if ( !((*vectorval).infsize()) ) { (*vectorval).applyon(trans); } }
        else if ( isValMatrix()  ) { (*matrixval).applyon(trans); (*matrixval).transpose(); }
        else if ( isValSet()     ) { (*setval).applyon(trans);                              }
        else if ( isValDgraph()  ) { ;                                                      }
        else if ( isValString()  ) { ;                                                      }
        else if ( isValError()   ) { ;                                                      }
        else
        {
            // Method: construct a new gentype, qswap (fast) this
            // new gentype with the present equation, then set
            // the present equation as neg with the argument
            // being the newly created gentype.

            Vector<gentype> *tempeqnargs;

            MEMNEW(tempeqnargs,Vector<gentype>(1));

            (*tempeqnargs)("&",zeroint()) = zeroint();
            qswap(*this,(*tempeqnargs)("&",zeroint()));

            typeis     = 'F';
            eqnargs    = tempeqnargs;
            fnnameind  = transInd;
            thisfninfo = getfninfo(fnnameind);
        }

        return *this;
    }

    gentype &conj(void)
    {
             if ( isValNull()    ) { ;                      }
        else if ( isValInteger() ) { ;                      }
        else if ( isValReal()    ) { ;                      }
        else if ( isValAnion()   ) { setconj(*anionval);    }
        else if ( isValVector()  ) { (*vectorval).conj();   }
        else if ( isValMatrix()  ) { (*matrixval).conj();   }
        else if ( isValSet()     ) { (*setval).conj();      }
        else if ( isValDgraph()  ) { (*dgraphval).conj();   }
        else if ( isValString()  ) { ;                      }
        else if ( isValError()   ) { ;                      }
        else
        {
            int namechange = 0;
            int revorder = (*thisfninfo).conjargmod < 0;
            int conjdef = revorder ? -(*thisfninfo).conjargmod : (*thisfninfo).conjargmod;

            if ( strcmp((*thisfninfo).conjfnname,"~") )
            {
                namechange = 1;
            }

            int i,j;

            if ( (*thisfninfo).numargs )
            {
                i = 1;

                for ( j = 0 ; j < (*thisfninfo).numargs ; j++ )
                {
                    if ( conjdef & i )
                    {
                        (*eqnargs)("&",j).conj();
                    }

                    i *= 2;
                }
            }

            if ( revorder && ((*thisfninfo).numargs)/2 )
            {
                for ( j = 0 ; j < ((*thisfninfo).numargs)/2 ; j++ )
                {
                    qswap((*eqnargs)("&",j),(*eqnargs)("&",((*thisfninfo).numargs)-1-j));
                }
            }

            if ( namechange )
            {
                fnnameind  = getfnindConj(fnnameind);
                thisfninfo = getfninfo(fnnameind);
            }
        }

        return *this;
    }

    gentype &rand(void)
    {
             if ( isValNull()    ) { ;                                    }
        else if ( isValInteger() ) { setrand(intval); doubleval = intval; }
        else if ( isValReal()    ) { setrand(doubleval);                  }
        else if ( isValAnion()   ) { setrand(*anionval);                  }
        else if ( isValVector()  ) { (*vectorval).rand();                 }
        else if ( isValMatrix()  ) { (*matrixval).rand();                 }
        else if ( isValSet()     ) { (*setval).rand();                    }
        else if ( isValDgraph()  ) { (*dgraphval).rand();                 }
        else
        {
            throw("Randomisation of functions, strings and errors not implemented");
        }

        return *this;
    }

    gentype &inverse(void)
    {
        if ( isValEqnDir() )
        {
            const static gentype res("inv(x)");

            return ( *this = res(*this) );
        }

        // NB: inv returns pseudo-inverse for non-square matrices

             if ( isValNull()   ) { ;                                                          }
        else if ( isValStrErr() ) { constructError(*this,*this,"inv ill-defined for string");  }
        else if ( isValSet()    ) { constructError(*this,*this,"inv ill-defined for sets");    }
        else if ( isValDgraph() ) { constructError(*this,*this,"inv ill-defined for dgraphs"); }
        else if ( isValMatrix() ) { *matrixval = inv(*matrixval);                              }
        else if ( isValVector() ) { constructError(*this,*this,"inv ill-defined for vector");  }
        else if ( isValAnion()  ) { *anionval  = inv(*anionval);                               }
        else if ( isValReal()   ) { doubleval  = inv(doubleval);                               }
        else if ( intval != 1   ) { doubleval  = inv(intval); typeis = 'R';                    }

        return *this;
    }

    int substitute(const gentype &x)                                                                         { SparseVector<SparseVector<gentype> > evalargs; evalargs("&",0)("&",0) = x;                                                                                                                 return substitute(evalargs); }
    int substitute(const gentype &x, const gentype &y)                                                       { SparseVector<SparseVector<gentype> > evalargs; evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y;                                                                                     return substitute(evalargs); }
    int substitute(const gentype &x, const gentype &y, const gentype &z)                                     { SparseVector<SparseVector<gentype> > evalargs; evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z;                                                         return substitute(evalargs); }
    int substitute(const gentype &x, const gentype &y, const gentype &z, const gentype &v)                   { SparseVector<SparseVector<gentype> > evalargs; evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z; evalargs("&",0)("&",3) = v;                             return substitute(evalargs); }
    int substitute(const gentype &x, const gentype &y, const gentype &z, const gentype &v, const gentype &w) { SparseVector<SparseVector<gentype> > evalargs; evalargs("&",0)("&",0) = x; evalargs("&",0)("&",1) = y; evalargs("&",0)("&",2) = z; evalargs("&",0)("&",3) = v; evalargs("&",0)("&",4) = w; return substitute(evalargs); }
    int substitute(const SparseVector<SparseVector<gentype> > &evalargs) { return evaluate(evalargs); }

    int realDeriv(const int &i,     const int &j    ) { gentype ii(i); gentype jj(j); return realDeriv(ii,jj); }
    int realDeriv(const gentype &i, const gentype &j);

    gentype &leftmult(const gentype &right_op);
    gentype &leftdiv (const gentype &right_op);
    gentype &leftadd (const gentype &right_op);
    gentype &leftsub (const gentype &right_op);

    gentype &rightmult(const gentype &left_op);
    gentype &rightdiv (const gentype &left_op);

    gentype &leftmult (const double &right_op);
    gentype &rightmult(const double &left_op );

    char gettypeis(void) const { return typeis; }

    void resize(int nsize)
    {
        NiceAssert( ( nsize == 1 ) || ( !nsize && isValNull() ) || isValVector() );

        if ( isValVector() && ( size() != nsize ) )
        {
            (*vectorval).resize(nsize);
        }

        return;
    }

    // Is value treated as scalar function?  By treating a gentype as a
    // scalar function the treatment is modified in some contexts, namely:
    //
    // - inner product becomes an approx integral over var(i,j) = 0->1
    // - norms and abs become the appropriate lp measure over "
    //
    // The general formatting for these functions is:
    //
    // @(i,j,n):fn
    //
    // where i,j is the variable to be integrated over and n is the number
    // of steps in the integral.  All of these are optional and defaults are:
    //
    // i = 0
    // j = 0
    // n = 100
    //
    // If i,j,n are vectors on size > 1 then this is a scalar function over
    // all variables var(i(k),j(k)) in the vectors.
    //
    // For example @():sin(x)
    //
    // Notes:
    //
    // varid_isscalar: 0 normally, 1 if treated as scalar
    // varid_xi,varid_xj: identify which variable is integrated over (real)
    // varid_numpts: number of steps in integral

    int scalarfn_isscalarfn(void) const { return varid_isscalar; }
    int scalarfn_numpts    (void) const { return varid_numpts;   }

    const Vector<int> &scalarfn_i(void) const { return varid_xi;       }
    const Vector<int> &scalarfn_j(void) const { return varid_xj;       }

    void scalarfn_setisscalarfn(int nv);
    void scalarfn_setnumpts    (int nv) { varid_numpts = nv; return; }

    void scalarfn_seti(const Vector<int> &nv) { varid_xi = nv; return; }
    void scalarfn_setj(const Vector<int> &nv) { varid_xj = nv; return; }

private:

    // types: N == null (nothing, can be cast to anything)
    //        Z == int
    //        R == double
    //        A == anion
    //        V == vector
    //        M == matrix
    //        X == set
    //        G == dgraph
    //        S == string
    //        E == error string
    //        F == equation (function)

    char typeis;

    int intval;
    double doubleval;
    d_anion *anionval;
    Vector<gentype> *vectorval;
    Matrix<gentype> *matrixval;
    Set<gentype> *setval;
    Dgraph<gentype,double> *dgraphval;
    std::string *stringval;
    int fnnameind; // var is evalargs
    Vector<gentype> *eqnargs;
    const fninfoblock *thisfninfo;

    Vector<double> *vectorvalreal;
    Matrix<double> *matrixvalreal;

    // Is value treated as scalar function?  By treating a gentype as a
    // scalar function the treatment is modified in some contexts, namely:
    //
    // - inner product becomes an approx integral over var(i,j) = 0->1
    // - norms and abs become the appropriate lp measure over "
    //
    // Notes:
    //
    // varid_isscalar: 0 normally, 1 if treated as scalar
    // varid_xi,varid_xj: identify which variable is integrated over (real)
    // varid_numpts: number of steps in integral

    int varid_isscalar;
    int varid_numpts;

    Vector<int> varid_xi;
    Vector<int> varid_xj;





    // Is value equation (extended version)

    int isValEqnFull(void) const
    {
        int res = 0;
        int i,j;

        if ( typeis == 'F' )
        {
            const static int varInd = getfnind("var");
            const static int VarInd = getfnind("Var");
            const static int gvarInd = getfnind("gvar");
            const static int gVarInd = getfnind("gVar");

            if ( ( fnnameind == varInd ) || ( fnnameind == VarInd ) || ( fnnameind == gvarInd ) || ( fnnameind == gVarInd ) )
            {
                res |= 16;
            }

            else
            {
                for ( i = 0 ; i < (*eqnargs).size() ; i++ )
                {
                    res |= ((*eqnargs)(i)).isValEqnFull();
                }

                // 1 contains indeterminant random or global parts
                // 2 contains variable
                // 4 contains indeterminant random parts
                // 8 contains indeterminant global parts

                res |= ( ( (*thisfninfo).isInDetermin ? 1 : 2 ) | ( ( (*thisfninfo).isInDetermin & 2 ) ? 4 : 0 ) | ( ( (*thisfninfo).isInDetermin & 1 ) ? 8 : 0 ) );
                //res |= ( ( (*thisfninfo).isInDetermin ? 1 : 0 ) | ( ( (*thisfninfo).isInDetermin & 2 ) ? 4 : 0 ) | ( ( (*thisfninfo).isInDetermin & 1 ) ? 8 : 0 ) );
            }
        }

        else if ( isValVector() && size() )
        {
            for ( i = 0 ; i < size() ; i++ )
            {
                res = res | ((*vectorval)(i)).isValEqn();
            }
        }

        else if ( isValMatrix() && numRows() && numCols() )
        {
            for ( i = 0 ; i < numRows() ; i++ )
            {
                for ( j = 0 ; j < numCols() ; j++ )
                {
                    res = res | ((*matrixval)(i,j)).isValEqn();
                }
            }
        }

        else if ( isValSet() && size() )
        {
            for ( i = 0 ; i < size() ; i++ )
            {
                res = res | (((*setval).all())(i)).isValEqn();
            }
        }

        else if ( isValDgraph() && size() )
        {
            for ( i = 0 ; i < size() ; i++ )
            {
                res = res | (((*dgraphval).all())(i)).isValEqn();
            }
        }

        return res;
    }

    // Const-cheating access functions

    int                     *NVintval       (void) const { return &((**thisthisthis).intval);        }
    double                  *NVdoubleval    (void) const { return &((**thisthisthis).doubleval);     }
    d_anion                **NVanionval     (void) const { return &((**thisthisthis).anionval);      }
    Vector<gentype>        **NVvectorval    (void) const { return &((**thisthisthis).vectorval);     }
    Vector<double>         **NVvectorvalreal(void) const { return &((**thisthisthis).vectorvalreal); }
    Matrix<gentype>        **NVmatrixval    (void) const { return &((**thisthisthis).matrixval);     }
    Matrix<double>         **NVmatrixvalreal(void) const { return &((**thisthisthis).matrixvalreal); }
    Set<gentype>           **NVsetval       (void) const { return &((**thisthisthis).setval);        }
    Dgraph<gentype,double> **NVdgraphval    (void) const { return &((**thisthisthis).dgraphval);     }
    std::string            **NVstringval    (void) const { return &((**thisthisthis).stringval);     }


    // Internal versions of casting operators.  These return the iserr value
    // that the cast would cause (so if 1 is returned then the cast is invalid
    // and errstr holds the relevant error message).

    int loctoNull   (                             std::string &errstr) const;
    int loctoInteger(int                    &res, std::string &errstr) const;
    int loctoReal   (double                 &res, std::string &errstr) const;
    int loctoAnion  (d_anion                &res, std::string &errstr) const;
    int loctoVector (Vector<gentype>        &res, std::string &errstr) const;
    int loctoMatrix (Matrix<gentype>        &res, std::string &errstr) const;
    int loctoSet    (Set<gentype>           &res, std::string &errstr) const;
    int loctoDgraph (Dgraph<gentype,double> &res, std::string &errstr) const;
    int loctoString (std::string            &res, std::string &errstr) const;

    // Internal version of substitute - function evaluation

    int evaluate(void) { const SparseVector<SparseVector<gentype> > evalargs; return evaluate(evalargs); }
    int evaluate(const SparseVector<SparseVector<gentype> > &evalargs) { return fastevaluate(evalargs,0); }

    // Helper functions:
    //
    // nearcopy:         internal equation evaluation function
    // deleteVectMatMem: memory deletion function.  The optional argument
    //                   also allocates (or keeps) allocated memory for the
    //                   type specified.  Function type not allowed.  Target
    //                   sizes are used for allocation but not guaranteed if
    //                   allocation not required.
    // reversestring:    reverse order of string
    // invertstringcase: invert case of string

    void nearcopy(gentype &res, const Vector<gentype> &locargres) const
    {
        res.deleteVectMatMem();

        res.typeis     = typeis;
        res.intval     = intval;
        res.doubleval  = doubleval;
        res.fnnameind  = fnnameind;
        res.thisfninfo = thisfninfo;

        if ( anionval  != NULL ) { MEMNEW(res.anionval ,d_anion(*anionval));                 }
        if ( vectorval != NULL ) { MEMNEW(res.vectorval,Vector<gentype>(*vectorval));        }
        if ( matrixval != NULL ) { MEMNEW(res.matrixval,Matrix<gentype>(*matrixval));        }
        if ( setval    != NULL ) { MEMNEW(res.setval   ,Set<gentype>(*setval));              }
        if ( dgraphval != NULL ) { MEMNEW(res.dgraphval,xDgraph(*dgraphval));                }
        if ( stringval != NULL ) { MEMNEW(res.stringval,std::string(*stringval));            }

        MEMNEW(res.eqnargs,Vector<gentype>(locargres));

        return;
    }

    void deleteVectMatMem(char targtype = 'R', int sizenRows = 0, int nCols = 0)
    {
        NiceAssert( targtype != 'F' );

        if ( ( anionval != NULL ) && ( targtype != 'A' ) )
        {
            MEMDEL(anionval);
            anionval = NULL;
        }

        if ( ( vectorval != NULL ) && ( ( targtype != 'V' ) || ( (*vectorval).infsize() ) ) )
        {
            MEMDEL(vectorval);
	    vectorval = NULL;
        }

        if ( vectorvalreal != NULL )
        {
            MEMDEL(vectorvalreal);
            vectorvalreal = NULL;
        }

        if ( ( matrixval != NULL ) && ( targtype != 'M' ) )
        {
            MEMDEL(matrixval);
	    matrixval = NULL;
        }

        if ( matrixvalreal != NULL )
        {
            MEMDEL(matrixvalreal);
            matrixvalreal = NULL;
        }

        if ( ( setval != NULL ) && ( targtype != 'X' ) )
        {
            MEMDEL(setval);
            setval = NULL;
        }

        if ( ( dgraphval != NULL ) && ( targtype != 'G' ) )
        {
            MEMDEL(dgraphval);
            dgraphval = NULL;
        }

        if ( ( stringval != NULL ) && ( targtype != 'S' ) && ( targtype != 'E' ) )
        {
            MEMDEL(stringval);
            stringval = NULL;
        }

        if ( eqnargs != NULL )
        {
            MEMDEL(eqnargs);
            eqnargs = NULL;
        }

        typeis     = targtype;
        intval     = 0;
        doubleval  = 0.0;
        thisfninfo = NULL;

        if ( ( anionval == NULL ) && ( targtype == 'A' ) )
        {
            MEMNEW(anionval,d_anion);
        }

        if ( targtype == 'V' )
        {
            if ( vectorval == NULL )
            {
                MEMNEW(vectorval,Vector<gentype>(sizenRows));
            }

            else
            {
                (*vectorval).resize(sizenRows);
            }
        }

        if ( targtype == 'M' )
        {
            if ( matrixval == NULL )
            {
                MEMNEW(matrixval,Matrix<gentype>(sizenRows,nCols));
            }

            else
            {
                (*matrixval).resize(sizenRows,nCols);
            }
        }

        if ( ( setval == NULL ) && ( targtype == 'X' ) )
        {
            MEMNEW(setval,Set<gentype>);
        }

        if ( ( dgraphval == NULL ) && ( targtype == 'G' ) )
        {
            MEMNEW(dgraphval,xDgraph);
        }

        if ( ( stringval == NULL ) && ( ( targtype == 'E' ) || ( targtype == 'S' ) ) )
        {
            MEMNEW(stringval,std::string);
        }

        return;
    }

    void reversestring(void);
    void invertstringcase(void);

    // Maths parsing.  Note that both of these assume that the
    // equation has been "made nice" prior to calling using the
    // function makeMathsStringNice.  This ensures that + is only
    // used as binary addition, - only as unary negation, and there
    // are only +, - and +- combos in the equation.
    //
    // - mathsparse converts src to srcx by replacing all operators
    //   (+-* etc) with their functional equivalents.  Operators are,
    //   in order of precedence (direction of evaluation is either
    //   left to right (LtoR) or right to left (RtoL)):
    //
    //   Factorial (RtoL):                      a!    -> fact(a)
    // 	 Unary negation and logical not (LtoR): -a    -> neg(a)
    //                                          ~a    -> lnot(a)
    //	 Power of (RtoL):                       a^b   -> pow(a,b)
    //                                          a.^b  -> epow(a,b)
    //	 Multiplication and division (LtoR):    a*b   -> mul(a,b)
    //                                          a/b   -> div(a,b)
    //                                          a\b   -> rdiv(a,b)
    //                                          a%b   -> mod(a,b)
    //                                          a.*b  -> emul(a,b)
    //                                          a./b  -> ediv(a,b)
    //                                          a.\b  -> erdiv(a,b)
    //                                          a.%b  -> emod(a,b)
    //	 Addition (LtoR):                       a+b   -> add(a,b)
    //	 Cayley-Dickson construction (LtoR):    a|b   -> cayleyDickson(a,b)
    //   Logical comparison (LtoR):             a==b  -> eq(a,b)
    //                                          a~=b  -> ne(a,b)
    //                                          a>b   -> gt(a,b)
    //                                          a>=b  -> ge(a,b)
    //                                          a<=b  -> le(a,b)
    //                                          a<b   -> lt(a,b)
    //   Logical combination (LtoR):            a||b  -> lor(a,b)
    //                                          a&&b  -> land(a,b)
    //
    // - makeEqnInternal processes the fully functional expression
    //   into an equation tree for use here.  Does not simplify.

    int mathsparse(std::string &srcx, const std::string &src);
    int makeEqnInternal(const std::string &src);

    // Variable evaluation function - replace this with variable, if available
    //
    // OP_var: return 0 if nothing happened, >0 otherwise

    gentype var(const SparseVector<SparseVector<gentype> > &evalargs, const Vector<gentype> &locargres) const;
    int OP_var(const SparseVector<SparseVector<gentype> > &evalargs);

    SparseVector<SparseVector<int> > &varsUsed(SparseVector<SparseVector<int> > &res) const;
    SparseVector<int> &rowsUsed(SparseVector<int> &res) const;

    // Fastcopy: like operator=, but if you set areDistinct then will assume
    // that the source is completely independent of the destination which
    // can speed things up significantly.
    //
    // fastevaluate: returns 0 if nothing changes, >0 otherwise
    // finalise: evaluate non-deterministic functions to get a final result
    //           1 means finalise randoms (indetermin 2)
    //           2 means finalise globals (indetermin 1)
    //           3 means both
    //
    // switcheroo: shorthand for gentype temp, qswap(temp,src),
    // qswap(temp,*this).  Use this for fast swapping in the case where you
    // want to replace this with one of its children (grandchildren etc).
    // If you use operator= then the operation will be slow.  If you use
    // qswap then you will end up with an orphan that never gets deleted.
    // So you use switheroo.

    gentype &fastcopy(const gentype &src, int areDistinct = 0);
    int fastevaluate(const SparseVector<SparseVector<gentype> > &evalargs, int finalise = 0);
    void switcheroo(gentype &src);

    // initBase: set variables, double type 0.  Only to be called by
    //           constructors

    void initBase(void)
    {
        thisthis = this;
        thisthisthis = &thisthis;

        typeis = 'Z';

        intval    = 0;
        doubleval = 0.0;

        anionval      = NULL;
        vectorval     = NULL;
        vectorvalreal = NULL;
        matrixval     = NULL;
        matrixvalreal = NULL;
        setval        = NULL;
        dgraphval     = NULL;
        stringval     = NULL;

        fnnameind = 0;

        eqnargs    = NULL;
        thisfninfo = NULL;

        varid_isscalar = 0;
        varid_numpts   = DEFAULT_INTEGRAL_SLICES;

        varid_xi.resize(1) = (int) DEFAULTVARI;
        varid_xj.resize(1) = (int) DEFAULTVARJ;

        return;
    }

    gentype *thisthis;
    gentype **thisthisthis;

public:
    int getfnnameind(void) const { return fnnameind; }
};

inline void qswap(gentype &a, gentype &b)
{
    qswap(a.typeis   ,b.typeis   );
    qswap(a.intval   ,b.intval   );
    qswap(a.doubleval,b.doubleval);

    qswap(a.varid_isscalar,b.varid_isscalar);
    qswap(a.varid_numpts  ,b.varid_numpts  );

    qswap(a.varid_xi,b.varid_xi);
    qswap(a.varid_xj,b.varid_xj);

    d_anion *anionval;
    Vector<gentype> *vectorval;
    Vector<double> *vectorvalreal;
    Matrix<gentype> *matrixval;
    Matrix<double> *matrixvalreal;
    Set<gentype> *setval;
    Dgraph<gentype,double> *dgraphval;
    std::string *stringval;
    int fnnameind;
    Vector<gentype> *eqnargs;
    const fninfoblock *thisfninfo;

    anionval      = a.anionval;      a.anionval      = b.anionval;      b.anionval      = anionval;
    vectorval     = a.vectorval;     a.vectorval     = b.vectorval;     b.vectorval     = vectorval;
    vectorvalreal = a.vectorvalreal; a.vectorvalreal = b.vectorvalreal; b.vectorvalreal = vectorvalreal;
    matrixval     = a.matrixval;     a.matrixval     = b.matrixval;     b.matrixval     = matrixval;
    matrixvalreal = a.matrixvalreal; a.matrixvalreal = b.matrixvalreal; b.matrixvalreal = matrixvalreal;
    setval        = a.setval;        a.setval        = b.setval;        b.setval        = setval;
    dgraphval     = a.dgraphval;     a.dgraphval     = b.dgraphval;     b.dgraphval     = dgraphval;
    stringval     = a.stringval;     a.stringval     = b.stringval;     b.stringval     = stringval;
    fnnameind     = a.fnnameind;     a.fnnameind     = b.fnnameind;     b.fnnameind     = fnnameind;
    eqnargs       = a.eqnargs;       a.eqnargs       = b.eqnargs;       b.eqnargs       = eqnargs;
    thisfninfo    = a.thisfninfo;    a.thisfninfo    = b.thisfninfo;    b.thisfninfo    = thisfninfo;

    return;
}

inline void qswap(const gentype *&a, const gentype *&b)
{
    const gentype *c = NULL;

    c = a;
    a = b;
    b = c;

    return;
}

inline void qswap(gentype *&a, gentype *&b)
{
    gentype *c = NULL;

    c = a;
    a = b;
    b = c;

    return;
}

inline void qswap(gentype **&a, gentype **&b)
{
    gentype **c = NULL;

    c = a;
    a = b;
    b = c;

    return;
}



inline const gentype &nullgentype(void); 
inline const gentype &nullgentype(void) 
{
    const static gentype nullval('N');

    return nullval;
}

inline const gentype &intgentype(void); 
inline const gentype &intgentype(void) 
{
    const static gentype nullval('Z');

    return nullval;
}

inline const gentype &realgentype(void); 
inline const gentype &realgentype(void) 
{
    const static gentype nullval('R');

    return nullval;
}

inline const gentype &zerogentype(void); 
inline const gentype &zerogentype(void) 
{
    const static gentype zeroval(0.0);

    return zeroval;
}

inline const gentype &zerointgentype(void); 
inline const gentype &zerointgentype(void) 
{
    const static gentype zeroval(zeroint());

    return zeroval;
}

inline const gentype &oneintgentype(void);
inline const gentype &oneintgentype(void)
{
    const static gentype oneval(1);

    return oneval;
}

inline const gentype &negoneintgentype(void);
inline const gentype &negoneintgentype(void)
{
    const static gentype oneval(-1);

    return oneval;
}

inline const gentype &twointgentype(void);
inline const gentype &twointgentype(void)
{
    const static gentype twoval(2);

    return twoval;
}

inline const gentype &threeintgentype(void);
inline const gentype &threeintgentype(void)
{
    const static gentype threeval(3);

    return threeval;
}

inline const gentype &onedblgentype(void);
inline const gentype &onedblgentype(void)
{
    const static gentype oneval(1.0);

    return oneval;
}

inline const gentype &negonedblgentype(void);
inline const gentype &negonedblgentype(void)
{
    const static gentype negoneval(-1.0);

    return negoneval;
}

inline const gentype &halfdblgentype(void);
inline const gentype &halfdblgentype(void)
{
    const static gentype halfval(0.5);

    return halfval;
}

gentype acos(const gentype &a);

inline const gentype &acosonedblgentype(void);
inline const gentype &acosonedblgentype(void)
{
    const static gentype acosoneval(acos(onedblgentype()));

    return acosoneval;
}

inline const gentype &defaultgentype(void);
inline const gentype &defaultgentype(void)
{
    const static gentype defval;

    return defval;
}




inline gentype &randfill(gentype &res);
inline gentype &randnfill(gentype &res);



inline gentype &leftmult (gentype &left_op, const gentype &right_op) { return left_op.leftmult (right_op); }
inline gentype &rightmult(const gentype &left_op, gentype &right_op) { return right_op.rightmult(left_op); }

inline gentype &leftmult (gentype &left_op, const double &right_op) { return left_op.leftmult (right_op); }
inline gentype &rightmult(const double &left_op, gentype &right_op) { return right_op.rightmult(left_op); }



const gentype &twoProduct       (gentype &res, const gentype &a, const gentype &b);
const gentype &twoProductNoConj (gentype &res, const gentype &a, const gentype &b);
const gentype &twoProductRevConj(gentype &res, const gentype &a, const gentype &b);

const gentype &fourProduct(gentype &res, const gentype &a, const gentype &b, const gentype &c, const gentype &d);



inline int operator==(const gentype &leftop, const gentype &rightop);
       int operator!=(const gentype &leftop, const gentype &rightop);
       int operator< (const gentype &leftop, const gentype &rightop);
       int operator<=(const gentype &leftop, const gentype &rightop);
       int operator> (const gentype &leftop, const gentype &rightop);
       int operator>=(const gentype &leftop, const gentype &rightop);

gentype operator+(const gentype &left_op);
gentype operator-(const gentype &left_op);

inline gentype operator+(const gentype &left_op, const gentype &right_op);
inline gentype operator-(const gentype &left_op, const gentype &right_op);
inline gentype operator*(const gentype &left_op, const gentype &right_op);
inline gentype operator/(const gentype &left_op, const gentype &right_op);
       gentype operator%(const gentype &left_op, const gentype &right_op);

inline gentype operator+(const gentype &left_op, const double  &right_op);
inline gentype operator-(const gentype &left_op, const double  &right_op);
inline gentype operator*(const gentype &left_op, const double  &right_op);
inline gentype operator/(const gentype &left_op, const double  &right_op);

inline gentype operator+(const double  &left_op, const gentype &right_op);
inline gentype operator-(const double  &left_op, const gentype &right_op);
inline gentype operator*(const double  &left_op, const gentype &right_op);
inline gentype operator/(const double  &left_op, const gentype &right_op);


// Speed hint: always put the more complicated argument on the left in these

inline gentype &operator+=(gentype &left_op, const gentype &right_op);
inline gentype &operator-=(gentype &left_op, const gentype &right_op);
inline gentype &operator*=(gentype &left_op, const gentype &right_op);
inline gentype &operator/=(gentype &left_op, const gentype &right_op);

       gentype &operator+=(gentype &left_op, const int &right_op);
       gentype &operator-=(gentype &left_op, const int &right_op);
       gentype &operator*=(gentype &left_op, const int &right_op);
inline gentype &operator/=(gentype &left_op, const int &right_op);

       gentype &operator+=(gentype &left_op, const double &right_op);
       gentype &operator-=(gentype &left_op, const double &right_op);
       gentype &operator*=(gentype &left_op, const double &right_op);
inline gentype &operator/=(gentype &left_op, const double &right_op);

       gentype &operator+=(gentype &left_op, const d_anion &right_op);
       gentype &operator-=(gentype &left_op, const d_anion &right_op);
       gentype &operator*=(gentype &left_op, const d_anion &right_op);
inline gentype &operator/=(gentype &left_op, const d_anion &right_op);

template <class T> gentype &operator+=(gentype &left_op, const Vector<T> &right_op);
template <class T> gentype &operator-=(gentype &left_op, const Vector<T> &right_op);
template <class T> gentype &operator*=(gentype &left_op, const Vector<T> &right_op);
template <class T> gentype &operator/=(gentype &left_op, const Vector<T> &right_op);

template <class T> gentype &operator+=(gentype &left_op, const Matrix<T> &right_op);
template <class T> gentype &operator-=(gentype &left_op, const Matrix<T> &right_op);
template <class T> gentype &operator*=(gentype &left_op, const Matrix<T> &right_op);
template <class T> gentype &operator/=(gentype &left_op, const Matrix<T> &right_op);

template <class T> gentype &operator+=(gentype &left_op, const Set<T> &right_op);
template <class T> gentype &operator-=(gentype &left_op, const Set<T> &right_op);
template <class T> gentype &operator*=(gentype &left_op, const Set<T> &right_op);
template <class T> gentype &operator/=(gentype &left_op, const Set<T> &right_op);

template <class T> gentype &operator+=(gentype &left_op, const Dgraph<T,double> &right_op);
template <class T> gentype &operator-=(gentype &left_op, const Dgraph<T,double> &right_op);
template <class T> gentype &operator*=(gentype &left_op, const Dgraph<T,double> &right_op);
template <class T> gentype &operator/=(gentype &left_op, const Dgraph<T,double> &right_op);

inline gentype &operator+=(gentype &left_op, const std::string &right_op);
inline gentype &operator-=(gentype &left_op, const std::string &right_op);
inline gentype &operator*=(gentype &left_op, const std::string &right_op);
inline gentype &operator/=(gentype &left_op, const std::string &right_op);

double gentypeToMatrixRep(const gentype &src, int dim, int iq, int jq);
Matrix<double> &gentypeToMatrixRep(Matrix<double> &dest, const gentype &src, int dim);


gentype &raiseto(gentype &a, int b); // a := a^b
inline void raiseto(double &a, int b) { a = pow(a,b); }




// postProInnerProd: post-process an inner product.  If this is a scalar
//   function then this integrates the function over var(i,j) = 0->1.

gentype &postProInnerProd(gentype &x);

inline gentype &setident        (gentype &a);
inline gentype &setzero         (gentype &a);
inline gentype &setzeropassive  (gentype &a);
inline gentype &setposate       (gentype &a);
inline gentype &setnegate       (gentype &a);
inline gentype &setconj         (gentype &a);
inline gentype &setrand         (gentype &a);
inline gentype &settranspose    (gentype &a);

inline const gentype *&setident (const gentype *&a);
inline const gentype *&setzero  (const gentype *&a);
inline const gentype *&setposate(const gentype *&a);
inline const gentype *&setnegate(const gentype *&a);
inline const gentype *&setconj  (const gentype *&a);
inline const gentype *&setrand  (const gentype *&a);

inline gentype *&setident (gentype *&a);
inline gentype *&setzero  (gentype *&a);
inline gentype *&setposate(gentype *&a);
inline gentype *&setnegate(gentype *&a);
inline gentype *&setconj  (gentype *&a);
inline gentype *&setrand  (gentype *&a);

inline gentype **&setident (gentype **&a) { throw("no"); return a; }
inline gentype **&setzero  (gentype **&a) { return a = NULL; }
inline gentype **&setposate(gentype **&a) { return a; }
inline gentype **&setnegate(gentype **&a) { throw("bleh"); return a; }
inline gentype **&setconj  (gentype **&a) { throw("blit"); return a; }
inline gentype **&setrand  (gentype **&a) { throw("OK, rand"); return a; }




// NaN and inf tests

int testisvnan(const gentype &x);
int testisinf (const gentype &x);
int testispinf(const gentype &x);
int testisninf(const gentype &x);





// make res = var(i,j)

inline gentype makeVar(int i, int j);
inline gentype makeVar(int i, int j)
{
    gentype tmp("var(var(1,0),var(1,1))");
    SparseVector<SparseVector<gentype> > tmpv;

    tmpv("&",1)("&",0) = i;
    tmpv("&",1)("&",1) = j;

    gentype res;

    res = tmp(tmpv);
    res.finalise();

    return res;
}























// Mathematical functions begin here
//
// Functions with a capital first letter are paired with non-capitalised
// functions in those cases where there is ambiguity where real arguments
// give non-real results.  Specifically, the capitalised version uses -i
// as a default unit pure imaginary, whereas the non-capitalised version
// uses i.  So for example:
//
// sqrt(-1) = i
// Sqrt(-1) = -i

// Operator forms:
//
// OP_... forms overwrite the first argument with the result, which is
// generally faster than the alternative form.

// Basic maths functions
//
// pos(a)     = a
// neg(a)     = -a
// add(a,b)   = a+b
// sub(a,b)   = a-b
// mul(a,b)   = a*b (multiplying two vectors will compute the inner product)
//                  (multiplying two strings will be 1 if equivalent, 0 otherwise)
// div(a,b)   = a/b (if a and b are integer then they get promoted to real before division)
// idiv(a,b)  = a/b (if a and b are integer then integer division is performed, otherwise standard division)
// rdiv(a,b)  = a\b
// mod(a,b)   = a%b
// pow(a,b)   = a^b (for anions, (powl(a,b)+powr(a,b))/2)
// emul(a,b)  = a.*b (elementwise version of *)
// ediv(a,b)  = a./b (elementwise version of /)
// eidiv(a,b) = a./b (elementwise version of integer division)
// erdiv(a,b) = a.\b (elementwise version of \)
// emod(a,b)  = a.%b (elementwise version of %)
// epow(a,b)  = a.^b (elementwise version of ^)
//
// emaxv(a,b) = max value
// eminv(a,b) = min value
//
// NB: - the elementwise operations will iterate until a non matrix/vector is found

gentype pos  (const gentype &a);
gentype neg  (const gentype &a);

gentype add  (const gentype &a, const gentype &b);
gentype sub  (const gentype &a, const gentype &b);
gentype mul  (const gentype &a, const gentype &b);
gentype div  (const gentype &a, const gentype &b);
gentype idiv (const gentype &a, const gentype &b);
gentype rdiv (const gentype &a, const gentype &b);
gentype mod  (const gentype &a, const gentype &b);
gentype pow  (const gentype &a, const gentype &b);
gentype Pow  (const gentype &a, const gentype &b);
gentype powl (const gentype &a, const gentype &b);
gentype Powl (const gentype &a, const gentype &b);
gentype powr (const gentype &a, const gentype &b);
gentype Powr (const gentype &a, const gentype &b);

gentype emul (const gentype &a, const gentype &b);
gentype ediv (const gentype &a, const gentype &b);
gentype eidiv(const gentype &a, const gentype &b);
gentype erdiv(const gentype &a, const gentype &b);
gentype emod (const gentype &a, const gentype &b);
gentype epow (const gentype &a, const gentype &b);
gentype Epow (const gentype &a, const gentype &b);
gentype epowl(const gentype &a, const gentype &b);
gentype Epowl(const gentype &a, const gentype &b);
gentype epowr(const gentype &a, const gentype &b);
gentype Epowr(const gentype &a, const gentype &b);

gentype &OP_pos  (gentype &a);
gentype &OP_neg  (gentype &a);

gentype &OP_add  (gentype &a, const gentype &b);
gentype &OP_sub  (gentype &a, const gentype &b);
gentype &OP_mul  (gentype &a, const gentype &b);
gentype &OP_div  (gentype &a, const gentype &b);
gentype &OP_idiv (gentype &a, const gentype &b);
gentype &OP_mod  (gentype &a, const gentype &b);

//gentype emaxv(const gentype &a, const gentype &b);
//gentype eminv(const gentype &a, const gentype &b);


// Basic logic functions
//
// There return 1 if true, 0 if not false or ill-defined
//
// eq(a,b):  a == b
// ne(a,b):  a != b
// gt(a,b):  a >  b
// ge(a,b):  a >= b
// le(a,b):  a <= b
// lt(a,b):  a <  b
//
// eeq(a,b):  elementwise a == b
// ene(a,b):  elementwise a != b
// egt(a,b):  elementwise a >  b
// ege(a,b):  elementwise a >= b
// ele(a,b):  elementwise a <= b
// elt(a,b):  elementwise a <  b
//
// not(a):   ~a
// or(a,b):  a || b
// and(a,b): a && b
//
// ifthenelse(a,b,c): if a then b, else c.  Works elementwise for
// vectors/matrices

gentype eq(const gentype &a, const gentype &b);
gentype ne(const gentype &a, const gentype &b);
gentype gt(const gentype &a, const gentype &b);
gentype ge(const gentype &a, const gentype &b);
gentype le(const gentype &a, const gentype &b);
gentype lt(const gentype &a, const gentype &b);

gentype eeq(const gentype &a, const gentype &b);
gentype ene(const gentype &a, const gentype &b);
gentype egt(const gentype &a, const gentype &b);
gentype ege(const gentype &a, const gentype &b);
gentype ele(const gentype &a, const gentype &b);
gentype elt(const gentype &a, const gentype &b);

gentype lnot(const gentype &a);
gentype lor (const gentype &a, const gentype &b);
gentype land(const gentype &a, const gentype &b);

gentype ifthenelse(const gentype &a, const gentype &b, const gentype &c);

// Basic typing and information functions
//
// is...(a): return 1 if true, 0 if false
//
// size(a):    returns size of a (1 unless vector, matrix or string)
// numRows(a): returns number of rows in a matrix
// numCols(a): returns number of columns in a matrix

gentype isnull  (const gentype &a);
gentype isint   (const gentype &a);
gentype isreal  (const gentype &a);
gentype isanion (const gentype &a);
gentype isvector(const gentype &a);
gentype ismatrix(const gentype &a);
gentype isset   (const gentype &a);
gentype isdgraph(const gentype &a);
gentype isstring(const gentype &a);
gentype iserror (const gentype &a);

gentype isvnan(const gentype &a);
gentype isinf (const gentype &a);
gentype ispinf(const gentype &a);
gentype isninf(const gentype &a);

gentype size   (const gentype &a);
gentype numRows(const gentype &a);
gentype numCols(const gentype &a);

// Some handy constants
//
// pi():     3.14...
// euler():  euler's constant
// pinf():   positive infinity
// ninf():   negative infinity
// vnan():   not a number
// eye(i,j): identity matrix i*j

inline gentype pi   (void);
inline gentype euler(void);
inline gentype pinf (void);
inline gentype ninf (void);
inline gentype vnan (void);
inline gentype null (void);
       gentype eye  (const gentype &i, const gentype &j);

// Modifiers/operators:
//
// conj(a):          returns the conjugate R-M of a.
// realDeriv(i,j,a): returns the partial derivative of a wrt var(i,j)

inline gentype conj     (const gentype &a);
       gentype realDeriv(const gentype &i, const gentype &j, const gentype &x);

// Basic vector construction
//
// vect_const(n,x): vector of size n, all elements equal to x
// vect_unit(n,i):  vector of size n with element i equal to 1 (all others zero)
//
// ivect(a,b,c):    a copy of Matlab's a:b:c notation
//
// funcv(f):     return functional vector for function f(x)
// rkhsv(k,x,a): return RKHS vector with kernel K (string - see mercer.h for format), centres (vectors) x and weights a
// bernv(w):     return the Bernstein polynomial *vector* of order size(w) (w is a weight vector)

gentype vect_const(const gentype &n, const gentype &x);
gentype vect_unit (const gentype &n, const gentype &i);

gentype ivect(const gentype &a, const gentype &b, const gentype &c);

gentype funcv(const gentype &f);
gentype rkhsv(const gentype &k, const gentype &x, const gentype &a);
gentype bernv(const gentype &w);

// Commutators and Associators
//
// commutate(x,y):   (xy - yx)/2
// associate(x,y,z): (x(yz) - (xy)z)/2
//
// anticommutate(x,y):   (xy + yx)/2
// antiassociate(x,y,z): (x(yz) + (xy)z)/2

gentype commutate(const gentype &x, const gentype &y);
gentype associate(const gentype &x, const gentype &y, const gentype &z);

gentype anticommutate(const gentype &x, const gentype &y);
gentype antiassociate(const gentype &x, const gentype &y, const gentype &z);

// Basic anion construction
//
// eps_comm(n,q,r,s):    commutator structure constant, order n, element q,r,s (0 real, 1,2,... imag)
// eps_assoc(n,q,r,s,t): associator structure constant, order n, element q,r,s,t (0 real, 1,2,... imag)
//
// im_complex(i): complex unit 1 (i=0) or i (i=1)
// im_quat(i):    quaternion unit 1 (i=0), I (i=1), J (i=2), K (i=3)
// im_octo(i):    octonion unit 1 (i=0), l (i=1), m (i=2), n (i=3), o (i=4), p (i=5), q (i=6), r (i=7)
// im_anion(n,i): anion (order n) unit 1 (i=0), imaginary i (0<i<2^n)
//
// cayleyDickson(x,y): returns the Cayley-Dickson construct (x|y)
// CayleyDickson(x,y): returns the Cayley-Dickson construct (conj(x)|-y)

gentype eps_comm (const gentype &n, const gentype &q, const gentype &r, const gentype &s);
gentype eps_assoc(const gentype &n, const gentype &q, const gentype &r, const gentype &s, const gentype &t);

gentype im_complex(const gentype &i);
gentype im_quat   (const gentype &i);
gentype im_octo   (const gentype &i);
gentype im_anion  (const gentype &n, const gentype &i);

gentype Im_complex(const gentype &i);
gentype Im_quat   (const gentype &i);
gentype Im_octo   (const gentype &i);
gentype Im_anion  (const gentype &n, const gentype &i);

gentype cayleyDickson(const gentype &x, const gentype &y);
gentype CayleyDickson(const gentype &x, const gentype &y);

// Permutations, combinations, factorials and deltas
//
// kronDelta(i,j):   - kronecker delta function
// diracDelta(x):    - dirac delta function
// ekronDelta(i,j):  - elementwise kronecker delta function
// ediracDelta(x):   - elementwise dirac delta function
// perm(i,j):        - number of permutations of j objects selectable from i
// comb(i,j):        - number of combinations of j objects selectable from i
// fact(i):          - i!

gentype kronDelta  (const gentype &i, const gentype &j);
gentype diracDelta (const gentype &x);
gentype ekronDelta (const gentype &i, const gentype &j);
gentype ediracDelta(const gentype &x);
gentype perm       (const gentype &i, const gentype &j);
gentype comb       (const gentype &i, const gentype &j);
gentype fact       (const gentype &i);

// Non-elementwise maths functions
//
// abs1(a):      returns the 1-norm |a|_1 of a.
// abs2(a):      returns the absolute value (magnitude) |a| of a.
// absp(a,p):    returns the p-norm |a|_p of a.
// absinf(a):    returns the inf-norn |a|inf if a
// norm1(a):     returns the 1-norm |a|_1 of a.
// norm2(a):     returns the square of the magnitude |a|^2 of a.
// normp(a,p):   returns the p-norm raised to power q |a|_p^p of a.
// angle(a):     returns the direction a/|a| (or 0 if a = 0) of a.
// inv(a):       returns the inverse of a (actually the pseudo-inverse for non-square matrices).

       gentype abs2  (const gentype &a);
       gentype abs1  (const gentype &a);
       gentype absp  (const gentype &a, const gentype &q);
inline gentype absp  (const gentype &a, const double &qx);
       gentype absinf(const gentype &a);
       gentype norm2 (const gentype &a);
       gentype norm1 (const gentype &a);
       gentype normp (const gentype &a, const gentype &q);
inline gentype normp (const gentype &a, const double &qx);
       gentype angle (const gentype &a);
       gentype inv   (const gentype &a);

// Elementwise maths functions
//
// eabs1(a):     returns the 1-norm |a|_1 of a.
// eabs2(a):     returns the absolute value (magnitude) |a| of a.
// eabsp(a,p):   returns the p-norm |a|_p of a.
// eabsinf(a):   returns the inf-norn |a|inf if a
// enorm1(a):    returns the 1-norm |a|_1 of a.
// enorm2(a):    returns the square of the magnitude |a|^2 of a.
// enormp(a,p):  returns the p-norm raised to power q |a|_p^p of a.
// eangle(a):    returns the direction a/|a| (or 0 if a = 0) of a.
// einv(a):      returns the inverse cong(a)/norm2(a) of a.
//
// real(a):      returns the real part of a.
// imag(a):      returns the imaginary sign-corrected magnitude |imagx(a)|*sgn(a[1]) (or |imagx(a)| if sgn(a[1]) = 0) of a.
// imagd(a):     returns the unit imaginary imagx(a)/imag(a) (or (0,1) if imag(a) = 0) of a (equivalent to argd(a)).*
// imagx(a):     returns the complete imaginary part of a.
// arg(a):       returns the argument |imagx(log(a))|*sgn(a[1]) (or |imagx(log(a))| if sgn(a[1]) = 0) of a.
// argd(a):      returns the unit imaginary imagx(log(a))/arg(a) (or (0,1) if arg(a) = 0) of a.*
// argx(a):      returns the complete argument imagx(log(a)) of a.*
// polar(x,y,q): returns the value x*exp(y*q) defined by the polar form (x,y*q).
// polard(x,y,q):returns the value x*exp(y*q) defined by the polar form (x,y*q).
// polarx(x,a):  returns the value x*exp(a) defined by the polar form (x,a).
// sgn(a):       returns the elementwise sign a.
//
// sqrt(a):      returns the square root exp(log(a)/2) of a.*
//
// exp(a):       returns the natural exponent exp(R)*cos(I) + q*exp(R)*sin(I) of a.
// tenup(a):     returns 10^a.
// log(a):       returns the natural logarithm log(abs2(a)) + q*atan2(I,R) of a.*,***
// log10(a):     returns the base 10 logarithm log(a)/2.3025851 of a.*,***
// logb(a,b):    returns the symmetrised base b logarithm (logbl(a,b)+logbr(a,b))/2 of a.*,***
// logbl(a,b):   returns the left base b logarithm log(a)*inv(log(b)) of a.*,***,****
// logbr(a,b):   returns the right base b logarithm inv(log(b))*log(a) of a.*,***,****
// 
// sin(a):       returns the sine sin(R)*cosh(I) + q*cos(R)*sinh(I) of a.
// cos(a):       returns the cosine cos(R)*cosh(I) - q*sin(R)*sinh(I) of a.
// tan(a):       returns the tangent sin(a)*inv(cos(a)) of a.
// cosec(a):     returns the cosecant inv(sin(a)) of a.
// sec(a):       returns the secant inv(cos(a)) of a.
// cot(a):       returns the cotangent inv(tan(a)) of a.
// asin(a):      returns the inverse sine of a.*,*****
// acos(a):      returns the inverse cosine of a.*,*****
// atan(a):      returns the inverse tangent of a.*,*****
// acosec(a):    returns the inverse cosecant asin(inv(a)) of a.*
// asec(a):      returns the inverse secant acos(inv(a)) of a.*
// acot(a):      returns the inverse cotangent atan(inv(a)) of a.*
// sinc(a):      returns the sinc sin(a)*inv(a) if |a| != 0 (1 if |a| = 0) of a.
// cosc(a):      returns the sinc cos(a)*inv(a) (no overflow checking)
// tanc(a):      returns the tanc tan(a)*inv(a) if |a| != 0 (1 if |a| = 0) of a.
// vers(a):      returns the versed sine 1-cos(a) of a.
// covers(a):    returns the coversed sine 1-sin(a) of a.
// hav(a):       returns the half versed sine vers(a)/2 of a.
// excosec(a):   returns the external cosecant cosec(a)-1 of a.
// exsec(a):     returns the exsecant sec(a)-1 of a.
// avers(a):     returns the inverse versed sine acos(a+1) of a.*
// acovers(a):   returns the inverse coversed sine asin(a+1) of a.*
// ahav(a):      returns the inverse half versed sine avers(2*a) of a.*
// aexcosec(a):  returns the inverse external cosecant acosec(a+1) of a.*
// aexsec(a):    returns the inverse external secant asec(a+1) of a.*
// 
// sinh(a):      returns the hyperbolic sine sinh(R)*cos(I) + q*cosh(R)*sin(I) of a.
// cosh(a):      returns the hyperbolic cosine cosh(R)*cos(I) + q*sinh(R)*sin(I) of a.
// tanh(a):      returns the hyperbolic tangent sinh(a)*inv(cosh(a)) of a.
// cosech(a):    returns the hyperbolic cosecant inv(sinh(a)) of a.
// sech(a):      returns the hyperbolic secant inv(cosh(a)) of a.
// coth(a):      returns the hyperbolic cotangent inv(tanh(a)) of a.
// asinh(a):     returns the inverse hyperbolic sine of a.*,******
// acosh(a):     returns the inverse hyperbolic cosine of a.*,******
// atanh(a):     returns the inverse hyperbolic tangent of a.*,******
// acosech(a):   returns the inverse hyperbolic cosecant asinh(inv(a)) of a.*
// asech(a):     returns the inverse hyperbolic secant acosh(inv(a)) of a.*
// acoth(a):     returns the inverse hyperbolic cotangent atanh(inv(a)) of a.*
// sinhc(a):     returns the hyperbolic sinc sinh(a)*inv(a) if |a| != 0 (1 if |a| = 0) of a.
// coshc(a):     returns the hyperbolic cosc cosh(a)*inv(a) (no overflow checking)
// tanhc(a):     returns the hyperbolic tanc tanh(a)*inv(a) if |a| != 0 (1 if |a| = 0) of a.
// versh(a):     returns the hyperbolic versed sine 1-cosh(a) of a.
// coversh(a):   returns the hyperbolic coversed sine 1-sinh(a) of a.
// havh(a):      returns the hyperbolic half versed sine versh(a)/2 of a.
// excosech(a):  returns the hyperbolic external cosecant cosech(a)-1 of a.
// exsech(a):    returns the hyperbolic external secant sech(a) - 1 of a.
// aversh(a):    returns the inverse hyperbolic versed sine acosh(a+1) of a.*
// acovrsh(a):   returns the inverse hyperbolic coversed sine asinh(a+1) of a.*
// ahavh(a):     returns the inverse hyperbolic half versed sine aversh(2*a) of a.*
// aexcosech(a): returns the inverse hyperbolic external cosecant acosech(a+1) of a.*
// aexsech(a):   returns the inverse hyperbolic exsecant asech(a+1) of a.*
// 
// sigm(a):      returns the sigmoid inv(1+exp(a)) of a.*
// gd(a):        returns the gudermannian 2*atan(tanh(a/2)) of a.*
// asigm(a):     returns the inverse sigmoid log(inv(a)-1) of a.*
// agd(a):       returns the inverse gudermannian 2*atanh(tan(a/2)) of a.*
//
// bern(w,x):    returns the Bernstein polynomial of order size(w) (w is a weight vector) evaluated at x
//
// normDistr(x):   Normal distribution (0 mean, unit variance)
// polyDistr(x,n): Polynomial distribution (0 mean, unit variance)

gentype eabs1  (const gentype &a);
gentype eabs2  (const gentype &a);
gentype eabsp  (const gentype &a, const gentype &p);
gentype eabsinf(const gentype &a);
gentype enorm1 (const gentype &a);
gentype enorm2 (const gentype &a);
gentype enormp (const gentype &a, const gentype &p);
gentype eangle (const gentype &a);
gentype einv   (const gentype &a);

gentype real  (const gentype &a);
gentype imag  (const gentype &a);
gentype imagd (const gentype &a);
gentype Imagd (const gentype &a);
gentype imagx (const gentype &a);
gentype arg   (const gentype &a);
gentype argd  (const gentype &a);
gentype Argd  (const gentype &a);
gentype argx  (const gentype &a);
gentype Argx  (const gentype &a);
gentype polar (const gentype &x, const gentype &y, const gentype &a);
gentype polard(const gentype &x, const gentype &y, const gentype &a);
gentype polarx(const gentype &x, const gentype &a);
gentype sgn   (const gentype &a);

gentype sqrt(const gentype &a);
gentype Sqrt(const gentype &a);

gentype exp  (const gentype &a);
gentype tenup(const gentype &a);
gentype log  (const gentype &a);
gentype Log  (const gentype &a);
gentype log10(const gentype &a);
gentype Log10(const gentype &a);
gentype logb (const gentype &a, const gentype &b);
gentype Logb (const gentype &a, const gentype &b);
gentype logbl(const gentype &a, const gentype &b);
gentype Logbl(const gentype &a, const gentype &b);
gentype logbr(const gentype &a, const gentype &b);
gentype Logbr(const gentype &a, const gentype &b);

gentype sin     (const gentype &a);
gentype cos     (const gentype &a);
gentype tan     (const gentype &a);
gentype cosec   (const gentype &a);
gentype sec     (const gentype &a);
gentype cot     (const gentype &a);
gentype asin    (const gentype &a);
gentype Asin    (const gentype &a);
//gentype acos    (const gentype &a);
gentype Acos    (const gentype &a);
gentype atan    (const gentype &a);
gentype acosec  (const gentype &a);
gentype Acosec  (const gentype &a);
gentype asec    (const gentype &a);
gentype Asec    (const gentype &a);
gentype acot    (const gentype &a);
gentype sinc    (const gentype &a);
gentype cosc    (const gentype &a);
gentype tanc    (const gentype &a);
gentype vers    (const gentype &a);
gentype covers  (const gentype &a);
gentype hav     (const gentype &a);
gentype excosec (const gentype &a);
gentype exsec   (const gentype &a);
gentype avers   (const gentype &a);
gentype Avers   (const gentype &a);
gentype acovers (const gentype &a);
gentype Acovers (const gentype &a);
gentype ahav    (const gentype &a);
gentype Ahav    (const gentype &a);
gentype aexcosec(const gentype &a);
gentype Aexcosec(const gentype &a);
gentype aexsec  (const gentype &a);
gentype Aexsec  (const gentype &a);

gentype sinh     (const gentype &a);
gentype cosh     (const gentype &a);
gentype tanh     (const gentype &a);
gentype cosech   (const gentype &a);
gentype sech     (const gentype &a);
gentype coth     (const gentype &a);
gentype asinh    (const gentype &a);
gentype acosh    (const gentype &a);
gentype Acosh    (const gentype &a);
gentype atanh    (const gentype &a);
gentype Atanh    (const gentype &a);
gentype acosech  (const gentype &a);
gentype asech    (const gentype &a);
gentype Asech    (const gentype &a);
gentype acoth    (const gentype &a);
gentype Acoth    (const gentype &a);
gentype sinhc    (const gentype &a);
gentype coshc    (const gentype &a);
gentype tanhc    (const gentype &a);
gentype versh    (const gentype &a);
gentype coversh  (const gentype &a);
gentype havh     (const gentype &a);
gentype excosech (const gentype &a);
gentype exsech   (const gentype &a);
gentype aversh   (const gentype &a);
gentype Aversh   (const gentype &a);
gentype acovrsh  (const gentype &a);
gentype ahavh    (const gentype &a);
gentype Ahavh    (const gentype &a);
gentype aexcosech(const gentype &a);
gentype aexsech  (const gentype &a);
gentype Aexsech  (const gentype &a);

gentype sigm (const gentype &a);
gentype gd   (const gentype &a);
gentype asigm(const gentype &a);
gentype Asigm(const gentype &a);
gentype agd  (const gentype &a);
gentype Agd  (const gentype &a);

gentype bern(const gentype &w, const gentype &x);

gentype normDistr(const gentype &x);
gentype polyDistr(const gentype &x, const gentype &n);
gentype PolyDistr(const gentype &x, const gentype &n);

// Various elementwise maths functions that do not yet have complex/anionic
// implementations
//
// gamma(a)     - gamma function = integ(0 to inf) (t^(a-1) * e^-t) dt
// lngamma(a)   - log gamma function = log(|gamma(x)|)
// psi(x)       - Digamma function = (deriv gamma)(x) / gamma(x)
// psi_n(i,x)   - Polygamma function = ith derivative of psi(x)
// gami(a,x)    - incomplete gamma fn = integ(0 to x) (t^(a-1) * e^-t) dt
// gamic(a,x)   - inverse "  gamma fn = integ(x to inf) (t^(a-1) * e^-t) dt
// zeta(a)      - returns the Reimann zeta function a, sum_{n=1}^\infty n^{-a}
// lambertW(a)  - Lambert W function main branch W0 (W>-1)
// lambertWx(a) - Lambert W function lower branch W1 (W<-1)
// 
// erf(x)         - error function
// erfc(x)        - complementary error function
// dawson(x)      - Dawson's fn: D(x) = e^(-x^2) * integ(0 to x)(e^(-t^2))dt

       gentype gamma    (const gentype &a);
       gentype lngamma  (const gentype &a);
       gentype psi      (const gentype &x);
       gentype psi_n    (const gentype &n, const gentype &x);
inline gentype gami     (const gentype &a, const gentype &x);
       gentype gamic    (const gentype &a, const gentype &x);
       gentype zeta     (const gentype &a);
       gentype lambertW (const gentype &a);
       gentype lambertWx(const gentype &a);

gentype erf      (const gentype &x);
gentype erfc     (const gentype &x);
gentype dawson   (const gentype &x);

// Type conversion functions
//
// rint(a):  a converted to nearest integer
// ceil(a):  smallest integer larger than rint(a)
// floor(a): largest integer smaller than rint(a)
//
// NB: rint, ceil and floor work elementwise on vectors and matrices, and
//     throw an error for anions

gentype rint (const gentype &a);
gentype ceil (const gentype &a);
gentype floor(const gentype &a);

// Vector and matrix functions
//
// outerProd(a,b):    outer product
// fourProd(a,b,c,d): 4-inner product
//
// trans(a):        transpose matrix (no change for vector)
// det(a):          calculate determinant
// trace(a):        calculate trace of a matrix
// miner(a,i,j):    calculate miner i,j of matrix a
// cofactor(a,i,j): calculate cofactor i,j of matrix a
// adj(a):          calculate adjoint of a
//
// max(a): maximum element of a
// min(a): minimum element of a
// maxdiag(a): maximum element of a
// mindiag(a): minimum element of a
// argmax(a): first maximum element of a
// argmin(a): first minimum element of a
// argmaxdiag(a): first maximum element of a
// argmindiag(a): first minimum element of a
// allargmax(a): all maximum elements of a
// allargmin(a): all minimum elements of a
// allargmaxdiag(a): all maximum elements of a
// allargmindiag(a): all minimum elements of a
//
// maxabs(a): maximum absolute element of a
// minabs(a): minimum absolute element of a
// maxabsdiag(a): maximum absolute element of a
// minabsdiag(a): minimum absolute element of a
// argmaxabs(a): first maximum absolute element of a
// argminabs(a): first minimum absolute element of a
// argmaxabsdiag(a): first maximum absolute element of a
// argminabsdiag(a): first minimum absolute element of a
// allargmaxabs(a): all maximum absolute elements of a
// allargminabs(a): all minimum absolute elements of a
// allargmaxabsdiag(a): all maximum absolute elements of a
// allargminabsdiag(a): all minimum absolute elements of a
//
// sum(a):          sum of elements in vector/matrix
// prod(a):         product of elements in vector
// Prod(a):         reversed order product of elements in vector
// mean(a):         mean of elements in vector/matrix
// median(a):       median of elements in vector/matrix
// argmedian(a):    median of elements in vector/matrix
//
// deref(a,x):    returns element x in a
//                scalar: x = [ ]     -> a
//                vector: x = [ i ]   -> derefv(a,i)
//                matrix: x = [ i j ] -> derefm(a,i,j)
// derefv(a,i):   returns element i in vector a
// derefm(a,i,j): returns element i,j in matrix a
// derefa(a,i):   for anions, returns a(i)
//
// collapse(a): given a vector with vectorial elements to a "flat" vector, or a matrix with matrix elements to a "flat" matrix
//              eg collapse([[1 2] [3 4]]) = [ 1 2 3 4 ]
//                 collapse([M: [1 2 ; 3 4 ] M: [ 5 6 ; 7 8 ] ; M: [ 9 10 ] M: [ 11 12 ]]) = [ 1 2 5 6 ; 3 4 7 8 ; 9 10 11 12 ]
//              NB: scalars (int, real, anion, string, error) are treated as vectors/matrices with 1 element
//                  vectors are always assumed to be column vectors (cast to matrix and transpose if this is not desired)

       gentype outerProd(const gentype &a, const gentype &b);
inline gentype OuterProd(const gentype &a, const gentype &b);
       gentype fourProd (const gentype &a, const gentype &b, const gentype &c, const gentype &d);

//gentype trans    (const gentype &a);
gentype det      (const gentype &a);
gentype trace    (const gentype &a);
gentype miner    (const gentype &a, const gentype &i, const gentype &j);
gentype cofactor (const gentype &a, const gentype &i, const gentype &j);
gentype adj      (const gentype &a);

gentype max          (const gentype &a);
gentype min          (const gentype &a);
gentype maxdiag      (const gentype &a);
gentype mindiag      (const gentype &a);
gentype argmax       (const gentype &a);
gentype argmin       (const gentype &a);
gentype argmaxdiag   (const gentype &a);
gentype argmindiag   (const gentype &a);
gentype allargmax    (const gentype &a);
gentype allargmin    (const gentype &a);
gentype allargmaxdiag(const gentype &a);
gentype allargmindiag(const gentype &a);

gentype maxabs          (const gentype &a);
gentype minabs          (const gentype &a);
gentype maxabsdiag      (const gentype &a);
gentype minabsdiag      (const gentype &a);
gentype argmaxabs       (const gentype &a);
gentype argminabs       (const gentype &a);
gentype argmaxabsdiag   (const gentype &a);
gentype argminabsdiag   (const gentype &a);
gentype allargmaxabs    (const gentype &a);
gentype allargminabs    (const gentype &a);
gentype allargmaxabsdiag(const gentype &a);
gentype allargminabsdiag(const gentype &a);

gentype sum      (const gentype &a);
gentype prod     (const gentype &a);
gentype Prod     (const gentype &a);
gentype mean     (const gentype &a);
gentype median   (const gentype &a);
gentype argmedian(const gentype &a);

gentype deref (const gentype &a, const gentype &i);
gentype derefv(const gentype &a, const gentype &i);
gentype derefm(const gentype &a, const gentype &i, const gentype &j);
gentype derefa(const gentype &a, const gentype &i);

gentype collapse(const gentype &a);


// Calls to global function pointer table, and non-deterministic functions, test functions
//
// fnA: evaluate global function i,j (0 arguments)
// fnB: evaluate global function i,j (1 arguments)
// fnC: evaluate global function i,j (2 arguments)
//
// dfnB: evaluate global function i,j derivative (1 arguments)
// dfnC: evaluate global function i,j derivative (2 arguments)
//
// efnB: elementwise fnB
// efnC: elementwise fnC
//
// edfnB: elementwise dfnB
// edfnC: elementwise dfnC
//
// irand: uniform positive random integer [0,i-1]
// urand: uniform random double [x,y]
// grand: gaussian random double from N(m,c)
//
// testfn:  evaluate test function i with input vector x
// testfnA: evaluate test function i with input vector x, using matrix A (see opttest.h)
//
// partestfn:  evaluate multi-objective test function i, target dim M, with input vector x
// partestfnA: evaluate multi-objective test function i, target dim M, with input vector x, using alpha given (see paretotest.h)

inline gentype fnA(const gentype &i, const gentype &j);
inline gentype fnB(const gentype &i, const gentype &j, const gentype &xa);
inline gentype fnC(const gentype &i, const gentype &j, const gentype &xa, const gentype &xb);

inline gentype dfnB(const gentype &i, const gentype &j, const gentype &xa, const gentype &ia);
inline gentype dfnC(const gentype &i, const gentype &j, const gentype &xa, const gentype &ia, const gentype &xb, const gentype &ib);

inline gentype efnB(const gentype &i, const gentype &j, const gentype &xa);
inline gentype efnC(const gentype &i, const gentype &j, const gentype &xa, const gentype &xb);

inline gentype edfnB(const gentype &i, const gentype &j, const gentype &xa, const gentype &ia);
inline gentype edfnC(const gentype &i, const gentype &j, const gentype &xa, const gentype &ia, const gentype &xb, const gentype &ib);

gentype irand(const gentype &i);
gentype urand(const gentype &x, const gentype &y);
gentype grand(const gentype &m, const gentype &c);

gentype testfn (const gentype &i, const gentype &x);
gentype testfnA(const gentype &i, const gentype &x, const gentype &A);

gentype partestfn (const gentype &i, const gentype &M, const gentype &x);
gentype partestfnA(const gentype &i, const gentype &M, const gentype &x, const gentype &A);


























// Design decision: the result is always written to the left-hand
// argument, even if this isn't technically where it should go
// (eg rdiv).



gentype &OP_add  (gentype &a, const gentype &b);
gentype &OP_sub  (gentype &a, const gentype &b);
gentype &OP_mul  (gentype &a, const gentype &b);
gentype &OP_div  (gentype &a, const gentype &b);
gentype &OP_idiv (gentype &a, const gentype &b);
gentype &OP_rdiv (gentype &a, const gentype &b);
gentype &OP_mod  (gentype &a, const gentype &b);

gentype &OP_pos  (gentype &a);
gentype &OP_neg  (gentype &a);

gentype &OP_lnot(gentype &a);

gentype &OP_isnull  (gentype &a);
gentype &OP_isint   (gentype &a);
gentype &OP_isreal  (gentype &a);
gentype &OP_isanion (gentype &a);
gentype &OP_isvector(gentype &a);
gentype &OP_ismatrix(gentype &a);
gentype &OP_isset   (gentype &a);
gentype &OP_isdgraph(gentype &a);
gentype &OP_isstring(gentype &a);
gentype &OP_iserror (gentype &a);

gentype &OP_isvnan(gentype &a);
gentype &OP_isinf (gentype &a);
gentype &OP_ispinf(gentype &a);
gentype &OP_isninf(gentype &a);

gentype &OP_size   (gentype &a);
gentype &OP_numRows(gentype &a);
gentype &OP_numCols(gentype &a);

gentype &OP_eabs1  (gentype &a);
gentype &OP_eabs2  (gentype &a);
gentype &OP_eabsinf(gentype &a);
gentype &OP_enorm1 (gentype &a);
gentype &OP_enorm2 (gentype &a);
gentype &OP_eangle (gentype &a);
gentype &OP_einv   (gentype &a);

gentype &OP_real  (gentype &a);
gentype &OP_imag  (gentype &a);
gentype &OP_imagd (gentype &a);
gentype &OP_Imagd (gentype &a);
gentype &OP_imagx (gentype &a);
gentype &OP_arg   (gentype &a);
gentype &OP_argd  (gentype &a);
gentype &OP_Argd  (gentype &a);
gentype &OP_argx  (gentype &a);
gentype &OP_Argx  (gentype &a);
gentype &OP_sgn   (gentype &a);

gentype &OP_sqrt(gentype &a);
gentype &OP_Sqrt(gentype &a);

gentype &OP_exp  (gentype &a);
gentype &OP_tenup(gentype &a);
gentype &OP_log  (gentype &a);
gentype &OP_Log  (gentype &a);
gentype &OP_log10(gentype &a);
gentype &OP_Log10(gentype &a);

gentype &OP_sin     (gentype &a);
gentype &OP_cos     (gentype &a);
gentype &OP_tan     (gentype &a);
gentype &OP_cosec   (gentype &a);
gentype &OP_sec     (gentype &a);
gentype &OP_cot     (gentype &a);
gentype &OP_asin    (gentype &a);
gentype &OP_Asin    (gentype &a);
gentype &OP_acos    (gentype &a);
gentype &OP_Acos    (gentype &a);
gentype &OP_atan    (gentype &a);
gentype &OP_acosec  (gentype &a);
gentype &OP_Acosec  (gentype &a);
gentype &OP_asec    (gentype &a);
gentype &OP_Asec    (gentype &a);
gentype &OP_acot    (gentype &a);
gentype &OP_sinc    (gentype &a);
gentype &OP_cosc    (gentype &a);
gentype &OP_tanc    (gentype &a);
gentype &OP_vers    (gentype &a);
gentype &OP_covers  (gentype &a);
gentype &OP_hav     (gentype &a);
gentype &OP_excosec (gentype &a);
gentype &OP_exsec   (gentype &a);
gentype &OP_avers   (gentype &a);
gentype &OP_Avers   (gentype &a);
gentype &OP_acovers (gentype &a);
gentype &OP_Acovers (gentype &a);
gentype &OP_ahav    (gentype &a);
gentype &OP_Ahav    (gentype &a);
gentype &OP_aexcosec(gentype &a);
gentype &OP_Aexcosec(gentype &a);
gentype &OP_aexsec  (gentype &a);
gentype &OP_Aexsec  (gentype &a);

gentype &OP_sinh     (gentype &a);
gentype &OP_cosh     (gentype &a);
gentype &OP_tanh     (gentype &a);
gentype &OP_cosech   (gentype &a);
gentype &OP_sech     (gentype &a);
gentype &OP_coth     (gentype &a);
gentype &OP_asinh    (gentype &a);
gentype &OP_acosh    (gentype &a);
gentype &OP_Acosh    (gentype &a);
gentype &OP_atanh    (gentype &a);
gentype &OP_Atanh    (gentype &a);
gentype &OP_acosech  (gentype &a);
gentype &OP_asech    (gentype &a);
gentype &OP_Asech    (gentype &a);
gentype &OP_acoth    (gentype &a);
gentype &OP_Acoth    (gentype &a);
gentype &OP_sinhc    (gentype &a);
gentype &OP_coshc    (gentype &a);
gentype &OP_tanhc    (gentype &a);
gentype &OP_versh    (gentype &a);
gentype &OP_coversh  (gentype &a);
gentype &OP_havh     (gentype &a);
gentype &OP_excosech (gentype &a);
gentype &OP_exsech   (gentype &a);
gentype &OP_aversh   (gentype &a);
gentype &OP_Aversh   (gentype &a);
gentype &OP_acovrsh  (gentype &a);
gentype &OP_ahavh    (gentype &a);
gentype &OP_Ahavh    (gentype &a);
gentype &OP_aexcosech(gentype &a);
gentype &OP_aexsech  (gentype &a);
gentype &OP_Aexsech  (gentype &a);

gentype &OP_sigm (gentype &a);
gentype &OP_gd   (gentype &a);
gentype &OP_asigm(gentype &a);
gentype &OP_Asigm(gentype &a);
gentype &OP_agd  (gentype &a);
gentype &OP_Agd  (gentype &a);





//gentype pow  (const gentype &a, const gentype &b);
//gentype Pow  (const gentype &a, const gentype &b);
//gentype powl (const gentype &a, const gentype &b);
//gentype Powl (const gentype &a, const gentype &b);
//gentype powr (const gentype &a, const gentype &b);
//gentype Powr (const gentype &a, const gentype &b);

//gentype emul (const gentype &a, const gentype &b);
//gentype ediv (const gentype &a, const gentype &b);
//gentype eidiv(const gentype &a, const gentype &b);
//gentype erdiv(const gentype &a, const gentype &b);
//gentype emod (const gentype &a, const gentype &b);
//gentype epow (const gentype &a, const gentype &b);
//gentype Epow (const gentype &a, const gentype &b);
//gentype epowl(const gentype &a, const gentype &b);
//gentype Epowl(const gentype &a, const gentype &b);
//gentype epowr(const gentype &a, const gentype &b);
//gentype Epowr(const gentype &a, const gentype &b);

//gentype eq(const gentype &a, const gentype &b);
//gentype ne(const gentype &a, const gentype &b);
//gentype gt(const gentype &a, const gentype &b);
//gentype ge(const gentype &a, const gentype &b);
//gentype le(const gentype &a, const gentype &b);
//gentype lt(const gentype &a, const gentype &b);

//gentype eeq(const gentype &a, const gentype &b);
//gentype ene(const gentype &a, const gentype &b);
//gentype egt(const gentype &a, const gentype &b);
//gentype ege(const gentype &a, const gentype &b);
//gentype ele(const gentype &a, const gentype &b);
//gentype elt(const gentype &a, const gentype &b);

//gentype lor (const gentype &a, const gentype &b);
//gentype land(const gentype &a, const gentype &b);

//gentype ifthenelse(const gentype &a, const gentype &b, const gentype &c);

//gentype conj     (const gentype &a);
//gentype realDeriv(const gentype &i, const gentype &j, const gentype &x);

//gentype vect_const(const gentype &n, const gentype &x);
//gentype vect_unit (const gentype &n, const gentype &i);

//gentype ivect(const gentype &a, const gentype &b, const gentype &c);

//gentype commutate(const gentype &x, const gentype &y);
//gentype associate(const gentype &x, const gentype &y, const gentype &z);

//gentype anticommutate(const gentype &x, const gentype &y);
//gentype antiassociate(const gentype &x, const gentype &y, const gentype &z);

//gentype eps_comm (const gentype &n, const gentype &q, const gentype &r, const gentype &s);
//gentype eps_assoc(const gentype &n, const gentype &q, const gentype &r, const gentype &s, const gentype &t);

//gentype im_complex(const gentype &i);
//gentype im_quat   (const gentype &i);
//gentype im_octo   (const gentype &i);
//gentype im_anion  (const gentype &n, const gentype &i);

//gentype Im_complex(const gentype &i);
//gentype Im_quat   (const gentype &i);
//gentype Im_octo   (const gentype &i);
//gentype Im_anion  (const gentype &n, const gentype &i);

//gentype cayleyDickson(const gentype &x, const gentype &y);
//gentype CayleyDickson(const gentype &x, const gentype &y);

//gentype kronDelta  (const gentype &i, const gentype &j);
//gentype diracDelta (const gentype &x);
//gentype ekronDelta (const gentype &i, const gentype &j);
//gentype ediracDelta(const gentype &x);
//gentype perm       (const gentype &i, const gentype &j);
//gentype comb       (const gentype &i, const gentype &j);
//gentype fact       (const gentype &i);

//gentype abs1  (const gentype &a);
//gentype abs2  (const gentype &a);
//gentype absp  (const gentype &a, const gentype &q);
//gentype absinf(const gentype &a);
//gentype norm1 (const gentype &a);
//gentype norm2 (const gentype &a);
//gentype normp (const gentype &a, const gentype &q);
//gentype angle (const gentype &a);
//gentype inv   (const gentype &a);

//gentype eabsp  (const gentype &a, const gentype &p);
//gentype enormp (const gentype &a, const gentype &p);
//gentype polar (const gentype &x, const gentype &y, const gentype &a);
//gentype polard(const gentype &x, const gentype &y, const gentype &a);
//gentype polarx(const gentype &x, const gentype &a);
//gentype logb (const gentype &a, const gentype &b);
//gentype Logb (const gentype &a, const gentype &b);
//gentype logbl(const gentype &a, const gentype &b);
//gentype Logbl(const gentype &a, const gentype &b);
//gentype logbr(const gentype &a, const gentype &b);
//gentype Logbr(const gentype &a, const gentype &b);

//gentype normDistr(const gentype &x);
//gentype polyDistr(const gentype &x, const gentype &n);
//gentype PolyDistr(const gentype &x, const gentype &n);

//gentype gamma  (const gentype &a);
//gentype lngamma(const gentype &a);
//gentype psi    (const gentype &x);
//gentype psi_n  (const gentype &n, const gentype &x);
//gentype gami   (const gentype &a, const gentype &x);
//gentype gamic  (const gentype &a, const gentype &x);

//gentype erf      (const gentype &x);
//gentype erfc     (const gentype &x);
//gentype dawson   (const gentype &x);

//gentype rint (const gentype &a);
//gentype ceil (const gentype &a);
//gentype floor(const gentype &a);

//gentype outerProd(const gentype &a, const gentype &b);
//gentype OuterProd(const gentype &a, const gentype &b);
//gentype fourProd (const gentype &a, const gentype &b, const gentype &c, const gentype &d);

//gentype trans    (const gentype &a);
//gentype det      (const gentype &a);
//gentype trace    (const gentype &a);
//gentype miner    (const gentype &a, const gentype &i, const gentype &j);
//gentype cofactor (const gentype &a, const gentype &i, const gentype &j);
//gentype adj      (const gentype &a);

//gentype max          (const gentype &a);
//gentype min          (const gentype &a);
//gentype maxdiag      (const gentype &a);
//gentype mindiag      (const gentype &a);
//gentype argmax       (const gentype &a);
//gentype argmin       (const gentype &a);
//gentype argmaxdiag   (const gentype &a);
//gentype argmindiag   (const gentype &a);
//gentype allargmax    (const gentype &a);
//gentype allargmin    (const gentype &a);
//gentype allargmaxdiag(const gentype &a);
//gentype allargmindiag(const gentype &a);

//gentype maxabs          (const gentype &a);
//gentype minabs          (const gentype &a);
//gentype maxabsdiag      (const gentype &a);
//gentype minabsdiag      (const gentype &a);
//gentype argmaxabs       (const gentype &a);
//gentype argminabs       (const gentype &a);
//gentype argmaxabsdiag   (const gentype &a);
//gentype argminabsdiag   (const gentype &a);
//gentype allargmaxabs    (const gentype &a);
//gentype allargminabs    (const gentype &a);
//gentype allargmaxabsdiag(const gentype &a);
//gentype allargminabsdiag(const gentype &a);

//gentype sum      (const gentype &a);
//gentype prod     (const gentype &a);
//gentype Prod     (const gentype &a);
//gentype mean     (const gentype &a);
//gentype median   (const gentype &a);
//gentype argmedian(const gentype &a);

//gentype deref (const gentype &a, const gentype &i);
//gentype derefv(const gentype &a, const gentype &i);
//gentype derefm(const gentype &a, const gentype &i, const gentype &j);
//gentype derefa(const gentype &a, const gentype &i);

//gentype collapse(const gentype &a);



























































template <class T> 
gentype &operator+=(gentype &a, const Vector<T> &bb)
{
    gentype b;

    b = bb;

    return a += b;
}

template <class T> 
gentype &operator-=(gentype &a, const Vector<T> &bb)
{
    gentype b;

    b = bb;

    return a -= b;
}

template <class T> 
gentype &operator*=(gentype &a, const Vector<T> &bb)
{
    gentype b;

    b = bb;

    return a *= b;
}

template <class T> 
gentype &operator/=(gentype &a, const Vector<T> &bb)
{
    gentype b;

    b = bb;

    return a /= b;
}

template <class T> 
gentype &operator+=(gentype &a, const Matrix<T> &bb)
{
    gentype b;

    b = bb;

    return a += b;
}

template <class T> 
gentype &operator-=(gentype &a, const Matrix<T> &bb)
{
    gentype b;

    b = bb;

    return a -= b;
}

template <class T> 
gentype &operator*=(gentype &a, const Matrix<T> &bb)
{
    gentype b;

    b = bb;

    return a *= b;
}

template <class T> 
gentype &operator/=(gentype &a, const Matrix<T> &bb)
{
    gentype b;

    b = bb;

    return a /= b;
}

template <class T> 
gentype &operator+=(gentype &a, const Set<T> &bb)
{
    gentype b;

    b = bb;

    return a += b;
}

template <class T> 
gentype &operator-=(gentype &a, const Set<T> &bb)
{
    gentype b;

    b = bb;

    return a -= b;
}

template <class T> 
gentype &operator*=(gentype &a, const Set<T> &bb)
{
    gentype b;

    b = bb;

    return a *= b;
}

template <class T> 
gentype &operator/=(gentype &a, const Set<T> &bb)
{
    gentype b;

    b = bb;

    return a /= b;
}

template <class T> 
gentype &operator+=(gentype &a, const Dgraph<T,double> &bb)
{
    gentype b;

    b = bb;

    return a += b;
}

template <class T> 
gentype &operator-=(gentype &a, const Dgraph<T,double> &bb)
{
    gentype b;

    b = bb;

    return a -= b;
}

template <class T> 
gentype &operator*=(gentype &a, const Dgraph<T,double> &bb)
{
    gentype b;

    b = bb;

    return a *= b;
}

template <class T> 
gentype &operator/=(gentype &a, const Dgraph<T,double> &bb)
{
    gentype b;

    b = bb;

    return a /= b;
}

inline double &scaladd(double &a, const gentype &b)
{
    return a += (b.quickrealpart());
}

inline double &scaladd(double &a, const gentype &b, const gentype &c)
{
    return a += (b.quickrealpart())*(c.quickrealpart());
}

inline double &scaladd(double &a, const gentype &b, const gentype &c, const gentype &d)
{
    return a += (b.quickrealpart())*(c.quickrealpart())*(d.quickrealpart());
}

inline double &scaladd(double &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e)
{
    return a += (b.quickrealpart())*(c.quickrealpart())*(d.quickrealpart())*(e.quickrealpart());
}

inline double &scalsub(double &a, const gentype &b)
{
    return a -= (b.quickrealpart());
}

inline double &scalmul(double &a, const gentype &b)
{
    return a *= (b.quickrealpart());
}

inline double &scaldiv(double &a, const gentype &b)
{
    return a /= (b.quickrealpart());
}

/*
inline double &scaladd(double &a, const gentype &b)
{
    return a += (b.doubleval);
}

inline double &scaladd(double &a, const gentype &b, const gentype &c)
{
    return a += (b.doubleval)*(c.doubleval);
}

inline double &scaladd(double &a, const gentype &b, const gentype &c, const gentype &d)
{
    return a += (b.doubleval)*(c.doubleval)*(d.doubleval);
}

inline double &scaladd(double &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e)
{
    return a += (b.doubleval)*(c.doubleval)*(d.doubleval)*(e.doubleval);
}

inline double &scalsub(double &a, const gentype &b)
{
    return a -= (b.doubleval);
}

inline double &scalmul(double &a, const gentype &b)
{
    return a *= (b.doubleval);
}

inline double &scaldiv(double &a, const gentype &b)
{
    return a /= (b.doubleval);
}
*/

inline gentype &scaladd(gentype &a, const gentype &b)
{
    return a += b;
}

inline gentype &scaladd(gentype &a, const gentype &b, const gentype &c)
{
    return a += b*c;
}

inline gentype &scaladd(gentype &a, const gentype &b, const gentype &c, const gentype &d)
{
    return a += b*c*d;
}

inline gentype &scaladd(gentype &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e)
{
    return a += b*c*d*e;
}

inline gentype &scalsub(gentype &a, const gentype &b)
{
    return a -= b;
}

inline gentype &scalmul(gentype &a, const gentype &b)
{
    return a *= b;
}

inline gentype &scaldiv(gentype &a, const gentype &b)
{
    return a /= b;
}

inline gentype &randfill (gentype &res)
{
    res = 0.0;
    randfill(res.dir_double());
    return res;
}

inline gentype &randnfill(gentype &res)
{
    res = 0.0;
    randnfill(res.dir_double());
    return res;
}

inline gentype &setident(gentype &a)
{
    a.ident();

    return a;
}

inline gentype &setzero(gentype &a)
{
    a.zero();

    return a;
}

inline gentype &setzeropassive(gentype &a)
{
    a.zeropassive();

    return a;
}

inline gentype &setposate(gentype &a)
{
    a.posate();

    return a;
}

inline gentype &setnegate(gentype &a)
{
    a.negate();

    return a;
}

inline gentype &setconj(gentype &a)
{
    a.conj();

    return a;
}

inline gentype &setrand(gentype &a)
{
    a.rand();

    return a;
}

inline gentype &settranspose(gentype &a)
{
    a.transpose();

    return a;
}

inline const gentype *&setident (const gentype *&a) { throw("no"); return a; }
inline const gentype *&setzero  (const gentype *&a) { return a = NULL; }
inline const gentype *&setposate(const gentype *&a) { return a; }
inline const gentype *&setnegate(const gentype *&a) { throw("bleh"); return a; }
inline const gentype *&setconj  (const gentype *&a) { throw("blit"); return a; }
inline const gentype *&setrand  (const gentype *&a) { throw("OK, rand"); return a; }

inline gentype *&setident (gentype *&a) { throw("no"); return a; }
inline gentype *&setzero  (gentype *&a) { return a = NULL; }
inline gentype *&setposate(gentype *&a) { return a; }
inline gentype *&setnegate(gentype *&a) { throw("bleh"); return a; }
inline gentype *&setconj  (gentype *&a) { throw("blit"); return a; }
inline gentype *&setrand  (gentype *&a) { throw("OK, rand"); return a; }



inline gentype conj(const gentype &a) { gentype res = a; res.conj(); return res; }

inline gentype null (void) { const static gentype temp('N');           return temp;  }
inline gentype pi   (void) { const static gentype temp(NUMBASE_PI);    return temp;  }
inline gentype euler(void) { const static gentype temp(NUMBASE_EULER); return temp;  }
inline gentype pinf (void) { const static gentype temp(valpinf());     return temp;  }
inline gentype ninf (void) { const static gentype temp(valninf());     return temp;  }
inline gentype vnan (void) { const static gentype temp(valvnan());     return temp;  }

inline gentype operator+(const gentype &left_op, const gentype &right_op) { gentype res(left_op); return res += right_op; }
inline gentype operator-(const gentype &left_op, const gentype &right_op) { gentype res(left_op); return res -= right_op; }
inline gentype operator*(const gentype &left_op, const gentype &right_op) { gentype res(left_op); return res *= right_op; }
inline gentype operator/(const gentype &left_op, const gentype &right_op) { gentype res(left_op); return res /= right_op; }
       gentype operator%(const gentype &left_op, const gentype &right_op);

inline gentype operator+(const gentype &left_op, const double  &right_op) { gentype res(left_op); return res += right_op; }
inline gentype operator-(const gentype &left_op, const double  &right_op) { gentype res(left_op); return res -= right_op; }
inline gentype operator*(const gentype &left_op, const double  &right_op) { gentype res(left_op); return res *= right_op; }
inline gentype operator/(const gentype &left_op, const double  &right_op) { gentype res(left_op); return res /= right_op; }

inline gentype operator+(const double  &left_op, const gentype &right_op) { gentype res(left_op); return res += right_op; }
inline gentype operator-(const double  &left_op, const gentype &right_op) { gentype res(left_op); return res -= right_op; }
inline gentype operator*(const double  &left_op, const gentype &right_op) { gentype res(left_op); return res *= right_op; }
inline gentype operator/(const double  &left_op, const gentype &right_op) { gentype res(left_op); return res /= right_op; }

inline int operator==(const gentype &leftop, const gentype &rightop) { return leftop.iseq(rightop); }

inline gentype &operator+=(gentype &left_op, const gentype &right_op) { return left_op.leftadd (right_op);       }
inline gentype &operator-=(gentype &left_op, const gentype &right_op) { return left_op.leftsub (right_op);       }
inline gentype &operator*=(gentype &left_op, const gentype &right_op) { return left_op.leftmult(right_op);       }
inline gentype &operator/=(gentype &left_op, const gentype &right_op) { return left_op.leftdiv (right_op);       }

inline gentype &operator/=(gentype &left_op, const int &right_op) { return left_op *= (1/((double) right_op)); }
inline gentype &operator/=(gentype &left_op, const double &right_op) { return left_op *= (1/right_op); }
inline gentype &operator/=(gentype &left_op, const d_anion &right_op) { return left_op *= inv(right_op); }

inline gentype &operator+=(gentype &left_op, const std::string &right_op) { gentype b(right_op); return left_op += b; }
inline gentype &operator-=(gentype &left_op, const std::string &right_op) { gentype b(right_op); return left_op -= b; }
inline gentype &operator*=(gentype &left_op, const std::string &right_op) { gentype b(right_op); return left_op *= b; }
inline gentype &operator/=(gentype &left_op, const std::string &right_op) { gentype b(right_op); return left_op /= b; }

inline gentype absp (const gentype &a, const double &qx) { gentype q(qx); return absp(a,q); }
inline gentype normp(const gentype &a, const double &qx) { gentype q(qx); return pow(absp(a,q),q); }

inline gentype gami(const gentype &a, const gentype &x) { return add(gamma(a),neg(gamic(a,x))); }

inline gentype OuterProd(const gentype &a, const gentype &b) { return trans(outerProd(a,b)); }

























// Specialisation to sparsevector for speed

template <> void SparseVector<gentype>::makealtcontent(void);

template <> gentype &oneProduct  (gentype &res, const SparseVector<gentype> &a);
template <> gentype &twoProduct  (gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int afaroff, int bfaroff);
template <> gentype &threeProduct(gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c);
template <> gentype &fourProduct (gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d);

template <> gentype &twoProductNoConj (gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int afaroff, int bfaroff);
template <> gentype &twoProductRevConj(gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int afaroff, int bfaroff);

template <> gentype &oneProductScaled  (gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &scale);
template <> gentype &twoProductScaled  (gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &scale, int afaroff, int bfaroff);
template <> gentype &threeProductScaled(gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &scale);
template <> gentype &fourProductScaled (gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d, const SparseVector<gentype> &scale);

template <> gentype &twoProductScaledNoConj (gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &scale, int afaroff, int bfaroff);
template <> gentype &twoProductScaledRevConj(gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &scale, int afaroff, int bfaroff);

template <> double &oneProductAssumeReal  (double &res, const SparseVector<gentype> &a);
template <> double &twoProductAssumeReal  (double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b);
template <> double &threeProductAssumeReal(double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c);
template <> double &fourProductAssumeReal (double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d);

template <> double absinf(const SparseVector<gentype> &a);
template <> double abs0  (const SparseVector<gentype> &a);
template <> double norm1 (const SparseVector<gentype> &a);
template <> double norm2 (const SparseVector<gentype> &a);
template <> double normp (const SparseVector<gentype> &a, double p);

template <> SparseVector<gentype> &operator*=(SparseVector<gentype> &left_op, const SparseVector<gentype> &right_op);





/*

Do this lot at some point if it becomes relevant

template <class S, class T, class U> T &twoProductLeftScaled               (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductLeftScaledNoConj         (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductLeftScaledRevConj        (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductRightScaled              (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductRightScaledNoConj        (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductRightScaledRevConj       (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T         > T &indexedtwoProduct                  (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class S, class T         > T &indexedtwoProductNoConj            (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class S, class T         > T &indexedtwoProductRevConj           (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductScaled            (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductScaledNoConj      (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductScaledRevConj     (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductLeftScaled        (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductLeftScaledNoConj  (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductLeftScaledRevConj (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductRightScaled       (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductRightScaledNoConj (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductRightScaledRevConj(T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);

template <class T> T &mProduct(T &res, const Vector<const SparseVector <T> *> &a);
template <class T> T &mProductScaled(T &res, const Vector<const SparseVector <T> *> &a, const SparseVector<T> &scale);
template <class T> double &mProductAssumeReal    (double &res, const Vector<const SparseVector <T> *> &a);

template <class T> T &indexedoneProduct  (T &res, const Vector<int> &n, const SparseVector<T> &a);
template <class T> T &indexedthreeProduct(T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c);
template <class T> T &indexedfourProduct (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d);
template <class T> T &indexedmProduct    (T &res, const Vector<int> &n, const Vector<const SparseVector <T> *> &a);

template <class S, class T         > T &twoProductTanh                         (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class S, class T         > T &twoProductNoConjTanh                   (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class S, class T         > T &twoProductRevConjTanh                  (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductScaledTanh                   (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductScaledNoConjTanh             (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductScaledRevConjTanh            (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductLeftScaledTanh               (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductLeftScaledNoConjTanh         (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductLeftScaledRevConjTanh        (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductRightScaledTanh              (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductRightScaledNoConjTanh        (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &twoProductRightScaledRevConjTanh       (T &res,                       const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T         > T &indexedtwoProductTanh                  (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class S, class T         > T &indexedtwoProductNoConjTanh            (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class S, class T         > T &indexedtwoProductRevConjTanh           (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b,                               int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductScaledTanh            (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductScaledNoConjTanh      (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductScaledRevConjTanh     (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductLeftScaledTanh        (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductLeftScaledNoConjTanh  (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductLeftScaledRevConjTanh (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductRightScaledTanh       (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductRightScaledNoConjTanh (T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);
template <class S, class T, class U> T &indexedtwoProductRightScaledRevConjTanh(T &res, const Vector<int> &n, const SparseVector<S> &a, const SparseVector<T> &b, const SparseVector<U> &scale, int afaroff = 0, int bfaroff = 0);

template <class T> double &oneProductAssumeRealTanh  (double &res, const SparseVector<T> &a);
template <class T> double &twoProductAssumeRealTanh  (double &res, const SparseVector<T> &a, const SparseVector<T> &b);
template <class T> double &threeProductAssumeRealTanh(double &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c);
template <class T> double &fourProductAssumeRealTanh (double &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d);
template <class T> double &mProductAssumeRealTanh    (double &res, const Vector<const SparseVector <T> *> &a);

template <class T> T &oneProductTanh  (T &res, const SparseVector<T> &a);
template <class T> T &threeProductTanh(T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c);
template <class T> T &fourProductTanh (T &res, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d);
template <class T> T &mProductTanh    (T &res, const Vector<const SparseVector <T> *> &a);

template <class T> T &indexedoneProductTanh  (T &res, const Vector<int> &n, const SparseVector<T> &a);
template <class T> T &indexedthreeProductTanh(T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c);
template <class T> T &indexedfourProductTanh (T &res, const Vector<int> &n, const SparseVector<T> &a, const SparseVector<T> &b, const SparseVector<T> &c, const SparseVector<T> &d);
template <class T> T &indexedmProductTanh    (T &res, const Vector<int> &n, const Vector<const SparseVector <T> *> &a);

template <class S, class T> T &fourProductPow(T &res, const SparseVector<S> &a, int ia, const SparseVector<S> &b, int ib, const SparseVector<S> &c, int ic, const SparseVector<S> &d, int id);
template <class S, class T> T &threeProductPow(T &res, const SparseVector<S> &a, int ia, const SparseVector<S> &b, int ib, const SparseVector<S> &c, int ic);

*/
















// These specialisations relate to matrix.h

template <> template <> inline Vector<gentype> &Matrix<double>::forwardElim(Vector<gentype> &y, const Vector<gentype> &b, int implicitTranspose) const;
template <> template <> inline Vector<gentype> &Matrix<double>::forwardElim(Vector<gentype> &y, const Vector<gentype> &b, int implicitTranspose) const
{
    // G.y = b  know G and b, calculate y

    NiceAssert( isSquare() );
    NiceAssert( ( dnumRows == b.size() ) || isEmpty() );

    if ( y.shareBase(b) && ( &y != &b ) )
    {
	Vector<gentype> bb(b);

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
                gentype temp;

                if ( &y != &b )
                {
                    y = b;
                }

                retVector<double>  tmpva;
                retVector<gentype> tmpvb;

                for ( i = 0 ; i < dnumRows ; i++ )
                {
                    y("&",i) -= sum(temp,y(zer,1,i-1,tmpvb),(*this)(i,zer,1,i-1,tmpva));

                    double thisii = ( (*this)(i,i) < MINDIAGLOC ) ? MINDIAGLOC : (*this)(i,i);
                    gentype ywas = y(i);

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
                    gentype ywas = y(i);

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

template <> template <> inline Vector<gentype> &Matrix<double>::backwardSubst(Vector<gentype> &y, const Vector<gentype> &b, int implicitTranspose) const;
template <> template <> inline Vector<gentype> &Matrix<double>::backwardSubst(Vector<gentype> &y, const Vector<gentype> &b, int implicitTranspose) const
{
    // G.y = b  know G and b, calculate y

    NiceAssert( isSquare() );
    NiceAssert( ( dnumRows == b.size() ) || isEmpty() );

    if ( y.shareBase(b) && ( &y != &b ) )
    {
	Vector<gentype> bb(b);

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
                gentype temp;

                if ( &y != &b )
                {
                    y = b;
                }

                retVector<double>  tmpva;
                retVector<gentype> tmpvb;

                for ( i = dnumRows-1 ; i >= 0 ; i-- )
                {
                    y("&",i) -= sum(temp,y(i+1,1,dnumRows-1,tmpvb),(*this)(i,i+1,1,dnumRows-1,tmpva));

                    double thisii = ( (*this)(i,i) < MINDIAGLOC ) ? MINDIAGLOC : (*this)(i,i);
                    gentype ywas = y(i);

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
                    gentype ywas = y(i);

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


template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<double> &src);

template <>
template <>
Vector<double> &Vector<double>::castassign(const Vector<gentype> &src);

template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<int> &src);


template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<gentype> &src);

template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<Vector<int> > &src);

template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<SparseVector<gentype> > &src);

template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<Vector<gentype> > &src);

template <>
template <>
Vector<SparseVector<gentype> >& Vector<SparseVector<gentype> >::castassign(const Vector<SparseVector<gentype> > &src);


#endif

