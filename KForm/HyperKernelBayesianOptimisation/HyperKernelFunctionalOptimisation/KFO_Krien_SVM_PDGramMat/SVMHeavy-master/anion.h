
//
// Complex, quaternion, octonion and anionic class.
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//
// This is a rather basic 2^n-ion (anionic) class written to take advantage
// of the Cayley-Dickson construction of the complex, quaternionic, octonionic
// and higher 2^n-ion algrebras.  It is designed to be largely compatible with
// the complex class of C++, and also combinable with same.
//


#ifndef _anion_h
#define _anion_h

#include <iostream>
#include <math.h>
#include <complex>
#include <stdlib.h>
#include "basefn.h"


class d_anion;

#define COMMUTATOR(x,y)         (((x)*(y))-((y)*(x)))/2
#define ANTICOMMUTATOR(x,y)     (((x)*(y))+((y)*(x)))/2

#define ASSOCIATOR(x,y,z)       (((x)*((y)*(z)))-(((x)*(y))*(z)))/2
#define ANTIASSOCIATOR(x,y,z)   (((x)*((y)*(z)))+(((x)*(y))*(z)))/2


// Calculate structure constants

int epsilon(int order, int q, int r, int s);
int epsilon(int order, int q, int r, int s, int t);

inline void qswap(d_anion &a, d_anion &b);

// Conversion: atod_anion is like atof.
//             atod_anion_safe is like atod_anion, except returns a non-zero error code on fail (0 on success) rather than asserting and throwing

d_anion atod_anion(const char *qwerty, int len = -1);
int atod_anion_safe(d_anion &result, const char *qwerty, int len = -1);

// streams

std::ostream &operator<<(std::ostream &output, const d_anion &source);
std::istream &operator>>(std::istream &input ,       d_anion &destin);


class d_anion
{
    /*
       Friend functions.
    */

    friend d_anion operator+(const d_anion &left_op);
    friend d_anion operator-(const d_anion &left_op);
    friend d_anion &operator+=(d_anion  &left_op, const double               &right_op);
    friend d_anion &operator+=(d_anion  &left_op, const std::complex<double> &right_op);
    friend d_anion &operator+=(d_anion  &left_op, const d_anion              &right_op);
    friend d_anion &operator-=(d_anion  &left_op, const double               &right_op);
    friend d_anion &operator-=(d_anion  &left_op, const std::complex<double> &right_op);
    friend d_anion &operator-=(d_anion  &left_op, const d_anion              &right_op);
    friend d_anion &leftmult(  d_anion  &left_op, const double               &right_op);
    friend d_anion &leftmult(  d_anion  &left_op, const std::complex<double> &right_op);
    friend d_anion &leftmult(  d_anion  &left_op, const d_anion              &right_op);
    friend d_anion &rightmult(const double               &left_op, d_anion &right_op);
    friend d_anion &rightmult(const std::complex<double> &left_op, d_anion &right_op);
    friend d_anion &rightmult(const d_anion              &left_op, d_anion &right_op);
    friend int operator==(const d_anion &left_op, const d_anion &right_op);
    friend int operator< (const d_anion &left_op, const d_anion &right_op);
    friend int operator<=(const d_anion &left_op, const d_anion &right_op);
    friend int operator> (const d_anion &left_op, const d_anion &right_op);
    friend int operator>=(const d_anion &left_op, const d_anion &right_op);
    friend std::ostream &operator<<(std::ostream &output, const d_anion &source);
    friend d_anion atod_anion(const char *qwerty, int len);
    friend d_anion &setconj(d_anion &a);

    friend void qswap(d_anion &a, d_anion &b);

    public:

    /*
       Constructors and destructors.
    */

    svm_explicit d_anion()                                { is_im = 0; value_real = 0.0; value_inf = NULL; value_0 = NULL;                          return; }
    svm_explicit d_anion(const double &src)               { is_im = 0; value_real = 0.0; value_inf = NULL; value_0 = NULL; *this = src;             return; }
    svm_explicit d_anion(const std::complex<double> &src) { is_im = 0; value_real = 0.0; value_inf = NULL; value_0 = NULL; *this = src;             return; }
                 d_anion(const d_anion &src)              { is_im = 0; value_real = 0.0; value_inf = NULL; value_0 = NULL; *this = src;             return; }
    svm_explicit d_anion(const char *src)                 { is_im = 0; value_real = 0.0; value_inf = NULL; value_0 = NULL; *this = atod_anion(src); return; }

    svm_explicit d_anion(const double &left_op, const double &right_op)                             { is_im = 1; value_real = 0.0; MEMNEW(value_inf,d_anion(left_op)); MEMNEW(value_0,d_anion(right_op)); return; }
    svm_explicit d_anion(const std::complex<double> &left_op, const double &right_op)               { is_im = 1; value_real = 0.0; MEMNEW(value_inf,d_anion(left_op)); MEMNEW(value_0,d_anion(right_op)); return; }
    svm_explicit d_anion(const d_anion &left_op, const double &right_op)                            { is_im = 1; value_real = 0.0; MEMNEW(value_inf,d_anion(left_op)); MEMNEW(value_0,d_anion(right_op)); return; }
    svm_explicit d_anion(const double &left_op, const std::complex<double> &right_op)               { is_im = 1; value_real = 0.0; MEMNEW(value_inf,d_anion(left_op)); MEMNEW(value_0,d_anion(right_op)); return; }
    svm_explicit d_anion(const std::complex<double> &left_op, const std::complex<double> &right_op) { is_im = 1; value_real = 0.0; MEMNEW(value_inf,d_anion(left_op)); MEMNEW(value_0,d_anion(right_op)); return; }
    svm_explicit d_anion(const d_anion &left_op, const std::complex<double> &right_op)              { is_im = 1; value_real = 0.0; MEMNEW(value_inf,d_anion(left_op)); MEMNEW(value_0,d_anion(right_op)); return; }
    svm_explicit d_anion(const double &left_op, const d_anion &right_op)                            { is_im = 1; value_real = 0.0; MEMNEW(value_inf,d_anion(left_op)); MEMNEW(value_0,d_anion(right_op)); return; }
    svm_explicit d_anion(const std::complex<double> &left_op, const d_anion &right_op)              { is_im = 1; value_real = 0.0; MEMNEW(value_inf,d_anion(left_op)); MEMNEW(value_0,d_anion(right_op)); return; }
    svm_explicit d_anion(const d_anion &left_op, const d_anion &right_op)                           { is_im = 1; value_real = 0.0; MEMNEW(value_inf,d_anion(left_op)); MEMNEW(value_0,d_anion(right_op)); return; }

    svm_explicit d_anion(int order);

    ~d_anion() { if ( is_im ) { MEMDEL(value_inf); MEMDEL(value_0); } return; }

    /*
       Assignment operators
    */

    d_anion &operator=(const double &val);
    d_anion &operator=(const std::complex<double> &val);
    d_anion &operator=(const d_anion &val);
    d_anion &operator=(const char *val);

    /*
       Back casting operators.

       For an anion a = (x,y) the cast (std::complex<double>) a will create a
       complex number (real(x),real(y)).

       For an anion the cast (double) will return the real part
    */

    operator std::complex<double>() const;

    /*
       Information functions.

       The function order gives the order of the anion, which is the maximum
       depth of the tree - that is, 0 for a real, 1 for C, 2 for H etc.

       The function iscomplex returns 0 if the object is real, or 1 otherwise.

       The functions leftpart and rightpart will first extend the tree so that
       order is at least 1 (by 0 filling if necessary, so the actual number
       represented is unchanged) and then return a reference to either the
       left or right component of the Cayley-Dickson construction (that is, if
       a = (x,y) then leftpart will return x and rightpart will return y).

       Finally, the () operator is overloaded so that a(i) returns a reference
       to the i^th component of an anion, after first zero filling if it does
       not actually exist in the present tree.  For example, if a=(1,(2,3))
       then a(1) will extend this to ((1,0),(2,3)) and return a reference to
       the 0 element.

       (i)      - get const reference to element i
       ("&",i) - get reference to element i
       (i,x)    - set element i to value x

       isindet - returns true of any part of this number is inf or nan
    */

    int order(void)              const;
    int size(void)               const { return 1 << order();     }
    int isreal(void)             const { return !iscomplex();     }
    int iscomplex(void)          const { return is_im;            }
    int iscommutative(void)      const { return ( order() <= 1 ); }
    int isassociative(void)      const { return ( order() <= 2 ); }
    int ispowerassociative(void) const { return ( order() <= 3 ); }
    int isindet(void)            const;

    d_anion &leftpart(void);
    d_anion &rightpart(void);

    const double &operator()(int i) const;
    double &operator()(const char *dummy, int i);
    double operator()(int i, double x);

    double realpart(void) const;

    /*
       simplify: minimise the order of the anion by collapsing any tree
                 elements of the form (x,0), where x is real, to the real form
                 (ie. is_im = 0) x.

       setorder: set the order, dropping non-real parts or padding with
                 zero if required.
    */

    d_anion &simplify(void);
    d_anion &setorder(int neworder);
    d_anion &resize(int newdim);

    /*
       String casting operator
    */

    std::string &tostring(std::string &dest) const;

    private:

    /*
       The number may be represented in two ways, as specified by the is_im
       flag.  If is_im == 0 then the anion is real, and will be given by
       value_real.  Otherwise the Cayley-Dickson construction is used, ie:

       a = (x,y)

       where x is an anion pointed to by value_inf and y an anion pointed to
       by value_o.  Thus the structure is, in essence, a binary tree.
    */

    int is_im;
    double value_real;
    d_anion *value_inf;
    d_anion *value_0;

    /*
       Internal functions.

       setorderge: set order to greater than or equal to the required number.
                 This is achieved by extending any branches shorter than this
                 to the required order by padding with zeros.
    */

    void setorderge(int n);

    /*
       Local dereferencing function
    */

    double &getref(int i);

    /*
       Output stream in strict Cayley-Dickson notation
    */

    std::ostream &forced_cayley_ostream(std::ostream &output) const;
};

void qswap(d_anion &a, d_anion &b)
{
    // NOTE: we are not dealing with a single element here but rather a node
    // in a larger tree of elements (potentially).  Therefore we cannot simply
    // use the naive swap-all-elements approach.  Better to simply use value
    // copying and hope that this never gets used too often.

    d_anion x(a); a = b; b = x;

    return;
}


inline const d_anion &defaultanion(void);
inline const d_anion &defaultanion(void)
{
    const static d_anion defval;

    return defval;
}

inline const d_anion &zeroanion(void);
inline const d_anion &zeroanion(void)
{
    const static d_anion zeroval(0.0);

    return zeroval;
}


inline d_anion &oneProduct  (d_anion &res, const d_anion &a);
inline d_anion &twoProduct  (d_anion &res, const d_anion &a, const d_anion &b);
inline d_anion &threeProduct(d_anion &res, const d_anion &a, const d_anion &b, const d_anion &c);
inline d_anion &fourProduct (d_anion &res, const d_anion &a, const d_anion &b, const d_anion &c, const d_anion &d);
inline d_anion &mProduct    (d_anion &res, int m, const d_anion *a);

inline d_anion &twoProductNoConj (d_anion &res, const d_anion &a, const d_anion &b);
inline d_anion &twoProductRevConj(d_anion &res, const d_anion &a, const d_anion &b);

// Mode functions: these functions control the form of Cayley-Dickson construct
// used, either eyes-left or eyes-right (see Conway and Smith, page 80)
//
// eyes-left:  (a,b).(c,d) = ( a.c - d.conj(b) , c.b + conj(a).d )
// eyes-right: (a,b).(c,d) = ( a.c - conj(d).b , d.a + b.conj(c) )
//
// By default eyes-right is used, as this emulates the behaviour of the standard
// ijk basis for quaternions.

int isAnionEyesLeft();
int isAnionEyesRight();
void setAnionEyesLeft();
void setAnionEyesRight();

// + posation - unary, return rvalue
// - negation - unary, return rvalue

inline d_anion  operator+(const d_anion &left_op);
inline d_anion  operator-(const d_anion &left_op);

// + addition       - binary, return rvalue
// - subtraction    - binary, return rvalue

inline d_anion  operator+(const double               &left_op, const d_anion              &right_op);
inline d_anion  operator+(const std::complex<double> &left_op, const d_anion              &right_op);
inline d_anion  operator+(const d_anion              &left_op, const double               &right_op);
inline d_anion  operator+(const d_anion              &left_op, const std::complex<double> &right_op);
inline d_anion  operator+(const d_anion              &left_op, const d_anion              &right_op);

inline d_anion  operator-(const double               &left_op, const d_anion              &right_op);
inline d_anion  operator-(const std::complex<double> &left_op, const d_anion              &right_op);
inline d_anion  operator-(const d_anion              &left_op, const double               &right_op);
inline d_anion  operator-(const d_anion              &left_op, const std::complex<double> &right_op);
inline d_anion  operator-(const d_anion              &left_op, const d_anion              &right_op);

// += additive       assignment - binary, return lvalue
// -= subtractive    assignment - binary, return lvalue

inline d_anion &operator+=(d_anion  &left_op, const double               &right_op);
inline d_anion &operator+=(d_anion  &left_op, const std::complex<double> &right_op);
       d_anion &operator+=(d_anion  &left_op, const d_anion              &right_op);

inline d_anion &operator-=(d_anion  &left_op, const double               &right_op);
inline d_anion &operator-=(d_anion  &left_op, const std::complex<double> &right_op);
       d_anion &operator-=(d_anion  &left_op, const d_anion              &right_op);

// * multiplication - binary, return rvalue
// / division       - binary, return rvalue

inline d_anion  operator*(const double               &left_op, const d_anion              &right_op);
inline d_anion  operator*(const std::complex<double> &left_op, const d_anion              &right_op);
inline d_anion  operator*(const d_anion              &left_op, const double               &right_op);
inline d_anion  operator*(const d_anion              &left_op, const std::complex<double> &right_op);
inline d_anion  operator*(const d_anion              &left_op, const d_anion              &right_op);

inline d_anion  operator/(const d_anion  &left_op, const double &right_op);

// *= multiplicative assignment - binary, return lvalue
// /= divisive       assignment - binary, return lvalue
//
// leftmult:  overwrite left_op with left_op*right_op
// rightmult: overwrite right_op with left_op*right_op

inline d_anion &operator*=(d_anion  &left_op, const double               &right_op);
inline d_anion &operator*=(d_anion  &left_op, const std::complex<double> &right_op);
inline d_anion &operator*=(d_anion  &left_op, const d_anion              &right_op);

inline d_anion &operator/=(d_anion  &left_op, const double &right_op);

d_anion &leftmult( d_anion &left_op, const double               &right_op);
d_anion &leftmult( d_anion &left_op, const std::complex<double> &right_op);
d_anion &leftmult( d_anion &left_op, const d_anion              &right_op);
d_anion &rightmult(const double               &left_op, d_anion &right_op);
d_anion &rightmult(const std::complex<double> &left_op, d_anion &right_op);
d_anion &rightmult(const d_anion              &left_op, d_anion &right_op);

// == equivalence
//
// NB: we also overload <, <=, >, >= as operators in the real part, otherwise
// gentype ends up in funky loops when it encounters <= for anions: specifically,
// d_anion <= d_anion is automagically interpretted as gentype <= gentype,
// which then loops back on itself.

inline int operator==(const double               &left_op, const d_anion              &right_op);
inline int operator==(const std::complex<double> &left_op, const d_anion              &right_op);
inline int operator==(const d_anion              &left_op, const double               &right_op);
inline int operator==(const d_anion              &left_op, const std::complex<double> &right_op);
       int operator==(const d_anion              &left_op, const d_anion              &right_op);
       int operator< (const d_anion              &left_op, const d_anion              &right_op);
       int operator<=(const d_anion              &left_op, const d_anion              &right_op);
       int operator> (const d_anion              &left_op, const d_anion              &right_op);
       int operator>=(const d_anion              &left_op, const d_anion              &right_op);

// != inequivalence

inline int operator!=(const double               &left_op, const d_anion              &right_op);
inline int operator!=(const std::complex<double> &left_op, const d_anion              &right_op);
inline int operator!=(const d_anion              &left_op, const double               &right_op);
inline int operator!=(const d_anion              &left_op, const std::complex<double> &right_op);
inline int operator!=(const d_anion              &left_op, const d_anion              &right_op);



// NaN and inf tests

inline int testisvnan(const d_anion &x);
inline int testisinf (const d_anion &x);
inline int testispinf(const d_anion &x);
inline int testisninf(const d_anion &x);

// Non-member functions:
//
// Extending complex functions: complex functions of a single variable
// can be extended to the quaternion case by noting that the
// set of numbers of the form R + q.I, where R and I are real and q is an
// imaginary unit anion are isomorphic to complex numbers by simple
// interchange of q and i (noting that q commutes with reals, qq = -1, and
// the conjugate of q is -q).
// 
// So, given a quaternion a, write a = R + M, where R is real and M is
// entirely imaginary (ie. R = real(a) and M = imagx(a)).  If we further define
// I = abs2(M) and q = angle(M) we may write x as:
// 
// a = R + q.I
// 
// R = real(a)
// I = abs2(imagx(a))
// q = angle(imagx(a))
// 
// where R and I are real and q is an imaginary unit anion.  If f is an
// analytic complex valued function of a single complex variable, ie.:
// 
// f(z) = f_R(real(z),imagx(z)) + i.f_I(real(z),imagx(z))
// 
// then, given that q is indistinguishable from the complex imaginery
// element i so long as q is the only imaginery present, the natural
// extension of f to the anions is:
// 
// f(a) = f_R(R,I) + q.f_I(R,I)
// 
// For functions of more than one variable, however, there are difficulties
// due to the presence of multiple unit imaginary anions which do not in
// general act like i.  In particular, they will not in general commute or
// multiply to give -1 in general.
// 
// In some cases where functions are well defined for the complex case, for
// example acosh(-2), we could define an infinite number of results, as we
// are going from real to complex without a well defined unit anion q to base
// our result.  If we assume q is complex (and we do) then q is defined up to
// sign and the sign can be taken care of.  If, however, we don't so restrict
// q then we have an infinite family to chose from, and they all give valid
// answers.  We use two approaches to this problem.  By default, we give a
// complex result with q = i (the complex imaginary unit).  Alternatively we
// provide an additional form of these function with a new argument that is
// used as the unit imaginary q.  If the unit given is 0, however, this version
// reverts to q = i.
// 
// The following functions are provided:
// 
// Let: a = R + M = R + q.I   (where R,I are real and q is unit)
//      b = S + N = S + r.T   (where S,T are real and r is unit)
//      x = real
//      y = real
// 
// abs1(a):      returns the 1-norm |a|_1 of a.
// abs2(a):      returns the absolute value (magnitude) |a| of a.
// absp(a,p):    returns the p-norm |a|_p of a.
// absinf(a):    returns the inf-norm |a|_inf of a.
// absd(a):      returns the default absolute value |a|.
// arg(a):       returns the argument |imagx(log(a))|*sgn(a(1)) (or |imagx(log(a))| if sgn(a(1)) = 0) of a.
// argd(a):      returns the unit imaginary imagx(log(a))/arg(a) (or (0,1) if arg(a) = 0) of a.*
// argx(a):      returns the complete argument imagx(log(a)) of a.*
// norm1(a):     returns the 1-norm |a|_1 of a.
// norm2(a):     returns the square of the magnitude |a|^2 of a.
// normp(a,p):   returns the p-norm raised to power q |a|_p^p of a.
// normd(a):     returns the square of the default magnitude |a|^2 of a.
// angle(a):     returns the direction a/|a| (or 0 if a = 0) of a.
// polar(x,y,q): returns the value x*exp(y*q) defined by the polar form (x,y*q).
// polard(x,y,q):returns the value x*exp(y*q) defined by the polar form (x,y*q).
// polarx(x,a):  returns the value x*exp(a) defined by the polar form (x,a).
// sgn(a):       returns the elementwise sign a.
// 
// real(a):      returns the real part of a.
// imag(a):      returns the imaginary sign-corrected magnitude |imagx(a)|*sgn(a(1)) (or |imagx(a)| if sgn(a(1)) = 0) of a.
// imagd(a):     returns the unit imaginary imagx(a)/imag(a) (or (0,1) if imag(a) = 0) of a (equivalent to argd(a)).*
// imagx(a):     returns the complete imaginary part of a.
// conj(a):      returns the conjugate R-M of a.
// inv(a):       returns the inverse cong(a)/norm2(a) of a.
// 
// powl(a,b):    returns the left power a^b - that is, exp(b*log(a)).*,****
// powr(a,b):    returns the right power a^b - that is, exp(log(a)*b).*,****
// pow(a,b):     returns the symmetrised power (powl(a,b)+powr(a,b))/2.*
// sqrt(a):      returns the square root exp(log(a)/2) of a.*
// 
// exp(a):       returns the natural exponent exp(R)*cos(I) + q*exp(R)*sin(I) of a.
// tenup(a):     returns the 10^a.
// log(a):       returns the natural logarithm log(abs2(a)) + q*atan2(I,R) of a.*,***
// log10(a):     returns the base 10 logarithm log(a)/2.3025851 of a.*,***
// logbl(a,b):   returns the left base b logarithm log(a)*inv(log(b)) of a.*,***,****
// logbr(a,b):   returns the right base b logarithm inv(log(b))*log(a) of a.*,***,****
// logb(a,b):    returns the symmetrised base b logarithm (logbl(a,b)+logbr(a,b))/2 of a.*,***
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
// 
// Notes: * for these functions a second version is provided where the complex
//          imaginary q is explicitly defined as a final argument.
//      *** log(a) is calculated assuming a branch-cut on the negative real
//          axis of the imaginary component.
//     **** for octonions and below, powl(b,logbl(a,b)) = a (likewise powr).
//    ***** asin(a), acos(a) and atan(a) follow Abramowitz and Stegun
//   ****** asinh(a), acosh(a) and atanh(a) are calculated using the formula:
//             asinh(a) = q*asin(conj(q)*a)    if I != 0
//                      = log(R+sqrt((R*R)+1)) if I = 0
//             acosh(a) = q*acos(a)            if I != 0
//                      = log(R+sqrt((R*R)-1)) if I = 0 and R >= 1
//                      = (0,1)*acos(a)        if I = 0 and R < 1
//             atanh(a) = q*atan(conj(q)*a)    if I != 0
//                      = 0.5*log((1+R)/(1-R)) if I = 0 and -1 < R < 1
//                      = (0,1)*atan((0,-1)*a) if I = 0 and ( R <= -1 or R >= 1 )
//          where of course (0,1) and (0,-1) may be replaced with a more
//          general unit imaginary anion if the function form where an explicit
//          imaginary q argument is present.  In this case (0,1) -> q and
//          (0,-1) -> conj(q).
//
// Rules of conjugation
// ====================
//
// Generally speaking, for a function f(a):
//
// conj(f(a)) = f(conj(a))
//
// with the exceptions:
//
// conj(powl(a))    = Powr(conj(a))
// conj(powr(a))    = Powl(conj(a))
// conj(logbl(a,b)) = Logbr(conj(a),conj(b))
// conj(logbr(a,b)) = Logbl(conj(a),conj(b))
// conj(Powl(a))    = powr(conj(a))
// conj(Powr(a))    = powl(conj(a))
// conj(Logbl(a,b)) = logbr(conj(a),conj(b))
// conj(Logbr(a,b)) = logbl(conj(a),conj(b))
//
// Furthermore in cases where there is a complex-imaginary explicit function
// there is a second function with a capitalised first letter.  This function
// differs insofar as the "default" anion result is conjugated, so for
// example:
//
// sqrt(-1) = i
// Sqrt(-1) = -i
//
// So for these functions:
//
// conj(function(a)) = Function(conj(a))

double  abs1(const d_anion &a);
double  abs2(const d_anion &a);
double  absp(const d_anion &a, const double &q);
double  absinf(const d_anion &a);
double  absd(const d_anion &a);
double  arg(const d_anion &a);
d_anion argd(const d_anion &a);
d_anion argx(const d_anion &a);
double  norm1(const d_anion &a);
double  norm2(const d_anion &a);
double  normp(const d_anion &a, const double &q);
double  normd(const d_anion &a);
d_anion angle(const d_anion &a);
d_anion vangle(const d_anion &a, const d_anion &defsign);
d_anion polar(const double &x, const double &y, const d_anion &a);
d_anion polard(const double &x, const double &y, const d_anion &a);
d_anion polarx(const double &x, const d_anion &a);
d_anion sgn(const d_anion &a);

double  real(const d_anion &a);
double  imag(const d_anion &a);
d_anion imagd(const d_anion &a);
d_anion imagx(const d_anion &a);
d_anion conj(const d_anion &a);
d_anion inv(const d_anion &a);

d_anion pow(const long &a, const d_anion &b);
d_anion pow(const double &a, const d_anion &b);
d_anion pow(const std::complex<double> &a, const d_anion &b);
d_anion powl(const std::complex<double> &a, const d_anion &b);
d_anion powr(const std::complex<double> &a, const d_anion &b);
d_anion pow(const d_anion &a, const long &b);
d_anion pow(const d_anion &a, const double &b);
d_anion pow(const d_anion &a, const std::complex<double> &b);
d_anion powl(const d_anion &a, const std::complex<double> &b);
d_anion powr(const d_anion &a, const std::complex<double> &b);
d_anion pow(const d_anion &a, const d_anion &b);
d_anion powl(const d_anion &a, const d_anion &b);
d_anion powr(const d_anion &a, const d_anion &b);
d_anion sqrt(const d_anion &a);

d_anion exp(const d_anion &a);
d_anion tenup(const d_anion &a);
d_anion log(const d_anion &a);
d_anion log10(const d_anion &a);
d_anion logb(const long &a, const d_anion &b);
d_anion logb(const double &a, const d_anion &b);
d_anion logb(const std::complex<double> &a, const d_anion &b);
d_anion logbl(const std::complex<double> &a, const d_anion &b);
d_anion logbr(const std::complex<double> &a, const d_anion &b);
d_anion logb(const d_anion &a, const long &b);
d_anion logb(const d_anion &a, const double &b);
d_anion logb(const d_anion &a, const std::complex<double> &b);
d_anion logbl(const d_anion &a, const std::complex<double> &b);
d_anion logbr(const d_anion &a, const std::complex<double> &b);
d_anion logb(const d_anion &a, const d_anion &b);
d_anion logbl(const d_anion &a, const d_anion &b);
d_anion logbr(const d_anion &a, const d_anion &b);

d_anion sin(const d_anion &a);
d_anion cos(const d_anion &a);
d_anion tan(const d_anion &a);
d_anion cosec(const d_anion &a);
d_anion sec(const d_anion &a);
d_anion cot(const d_anion &a);
d_anion asin(const d_anion &a);
d_anion acos(const d_anion &a);
d_anion atan(const d_anion &a);
d_anion acosec(const d_anion &a);
d_anion asec(const d_anion &a);
d_anion acot(const d_anion &a);
d_anion sinc(const d_anion &a);
d_anion cosc(const d_anion &a);
d_anion tanc(const d_anion &a);
d_anion vers(const d_anion &a);
d_anion covers(const d_anion &a);
d_anion hav(const d_anion &a);
d_anion excosec(const d_anion &a);
d_anion exsec(const d_anion &a);
d_anion avers(const d_anion &a);
d_anion acovers(const d_anion &a);
d_anion ahav(const d_anion &a);
d_anion aexcosec(const d_anion &a);
d_anion aexsec(const d_anion &a);

d_anion sinh(const d_anion &a);
d_anion cosh(const d_anion &a);
d_anion tanh(const d_anion &a);
d_anion cosech(const d_anion &a);
d_anion sech(const d_anion &a);
d_anion coth(const d_anion &a);
d_anion asinh(const d_anion &a);
d_anion acosh(const d_anion &a);
d_anion atanh(const d_anion &a);
d_anion acosech(const d_anion &a);
d_anion asech(const d_anion &a);
d_anion acoth(const d_anion &a);
d_anion sinhc(const d_anion &a);
d_anion coshc(const d_anion &a);
d_anion tanhc(const d_anion &a);
d_anion versh(const d_anion &a);
d_anion coversh(const d_anion &a);
d_anion havh(const d_anion &a);
d_anion excosech(const d_anion &a);
d_anion exsech(const d_anion &a);
d_anion aversh(const d_anion &a);
d_anion acovrsh(const d_anion &a);
d_anion ahavh(const d_anion &a);
d_anion aexcosech(const d_anion &a);
d_anion aexsech(const d_anion &a);

d_anion sigm(const d_anion &a);
d_anion gd(const d_anion &a);
d_anion asigm(const d_anion &a);
d_anion agd(const d_anion &a);



d_anion Argd(const d_anion &a);
d_anion Argx(const d_anion &a);
d_anion Imagd(const d_anion &a);

d_anion Pow(const long &a, const d_anion &b);
d_anion Pow(const double &a, const d_anion &b);
d_anion Pow(const std::complex<double> &a, const d_anion &b);
d_anion Powl(const std::complex<double> &a, const d_anion &b);
d_anion Powr(const std::complex<double> &a, const d_anion &b);
d_anion Pow(const d_anion &a, const long &b);
d_anion Pow(const d_anion &a, const double &b);
d_anion Pow(const d_anion &a, const std::complex<double> &b);
d_anion Powl(const d_anion &a, const std::complex<double> &b);
d_anion Powr(const d_anion &a, const std::complex<double> &b);
d_anion Pow(const d_anion &a, const d_anion &b);
d_anion Powl(const d_anion &a, const d_anion &b);
d_anion Powr(const d_anion &a, const d_anion &b);
d_anion Sqrt(const d_anion &a);

d_anion Log(const d_anion &a);
d_anion Log10(const d_anion &a);
d_anion Logb(const long &a, const d_anion &b);
d_anion Logb(const double &a, const d_anion &b);
d_anion Logb(const std::complex<double> &a, const d_anion &b);
d_anion Logbl(const std::complex<double> &a, const d_anion &b);
d_anion Logbr(const std::complex<double> &a, const d_anion &b);
d_anion Logb(const d_anion &a, const long &b);
d_anion Logb(const d_anion &a, const double &b);
d_anion Logb(const d_anion &a, const std::complex<double> &b);
d_anion Logbl(const d_anion &a, const std::complex<double> &b);
d_anion Logbr(const d_anion &a, const std::complex<double> &b);
d_anion Logb(const d_anion &a, const d_anion &b);
d_anion Logbl(const d_anion &a, const d_anion &b);
d_anion Logbr(const d_anion &a, const d_anion &b);

d_anion Asin(const d_anion &a);
d_anion Acos(const d_anion &a);
d_anion Acosec(const d_anion &a);
d_anion Asec(const d_anion &a);
d_anion Avers(const d_anion &a);
d_anion Acovers(const d_anion &a);
d_anion Ahav(const d_anion &a);
d_anion Aexcosec(const d_anion &a);
d_anion Aexsec(const d_anion &a);

d_anion Acosh(const d_anion &a);
d_anion Atanh(const d_anion &a);
d_anion Asech(const d_anion &a);
d_anion Acoth(const d_anion &a);
d_anion Aversh(const d_anion &a);
d_anion Ahavh(const d_anion &a);
d_anion Aexsech(const d_anion &a);

d_anion Asigm(const d_anion &a);
d_anion Agd(const d_anion &a);



d_anion argd(const d_anion &a, const d_anion &q_default);
d_anion argx(const d_anion &a, const d_anion &q_default);
d_anion imagd(const d_anion &a, const d_anion &q_default);

d_anion pow(const long &a, const d_anion &b, const d_anion &q_default);
d_anion pow(const double &a, const d_anion &b, const d_anion &q_default);
d_anion pow(const std::complex<double> &a, const d_anion &b, const d_anion &q_default);
d_anion powl(const std::complex<double> &a, const d_anion &b, const d_anion &q_default);
d_anion powr(const std::complex<double> &a, const d_anion &b, const d_anion &q_default);
d_anion pow(const d_anion &a, const long &b, const d_anion &q_default);
d_anion pow(const d_anion &a, const double &b, const d_anion &q_default);
d_anion pow(const d_anion &a, const std::complex<double> &b, const d_anion &q_default);
d_anion powl(const d_anion &a, const std::complex<double> &b, const d_anion &q_default);
d_anion powr(const d_anion &a, const std::complex<double> &b, const d_anion &q_default);
d_anion pow(const d_anion &a, const d_anion &b, const d_anion &q_default);
d_anion powl(const d_anion &a, const d_anion &b, const d_anion &q_default);
d_anion powr(const d_anion &a, const d_anion &b, const d_anion &q_default);
d_anion sqrt(const d_anion &a, const d_anion &q_default);

d_anion log(const d_anion &a, const d_anion &q_default);
d_anion log10(const d_anion &a, const d_anion &q_default);
d_anion logb(const long &a, const d_anion &b, const d_anion &q_default);
d_anion logb(const double &a, const d_anion &b, const d_anion &q_default);
d_anion logb(const std::complex<double> &a, const d_anion &b, const d_anion &q_default);
d_anion logbl(const std::complex<double> &a, const d_anion &b, const d_anion &q_default);
d_anion logbr(const std::complex<double> &a, const d_anion &b, const d_anion &q_default);
d_anion logb(const d_anion &a, const long &b, const d_anion &q_default);
d_anion logb(const d_anion &a, const double &b, const d_anion &q_default);
d_anion logb(const d_anion &a, const std::complex<double> &b, const d_anion &q_default);
d_anion logbl(const d_anion &a, const std::complex<double> &b, const d_anion &q_default);
d_anion logbr(const d_anion &a, const std::complex<double> &b, const d_anion &q_default);
d_anion logb(const d_anion &a, const d_anion &b, const d_anion &q_default);
d_anion logbl(const d_anion &a, const d_anion &b, const d_anion &q_default);
d_anion logbr(const d_anion &a, const d_anion &b, const d_anion &q_default);

d_anion asin(const d_anion &a, const d_anion &q_default);
d_anion acos(const d_anion &a, const d_anion &q_default);
d_anion acosec(const d_anion &a, const d_anion &q_default);
d_anion asec(const d_anion &a, const d_anion &q_default);
d_anion avers(const d_anion &a, const d_anion &q_default);
d_anion acovers(const d_anion &a, const d_anion &q_default);
d_anion ahav(const d_anion &a, const d_anion &q_default);
d_anion aexcosec(const d_anion &a, const d_anion &q_default);
d_anion aexsec(const d_anion &a, const d_anion &q_default);

d_anion acosh(const d_anion &a, const d_anion &q_default);
d_anion atanh(const d_anion &a, const d_anion &q_default);
d_anion asech(const d_anion &a, const d_anion &q_default);
d_anion acoth(const d_anion &a, const d_anion &q_default);
d_anion aversh(const d_anion &a, const d_anion &q_default);
d_anion ahavh(const d_anion &a, const d_anion &q_default);
d_anion aexsech(const d_anion &a, const d_anion &q_default);

d_anion asigm(const d_anion &a, const d_anion &q_default);
d_anion agd(const d_anion &a, const d_anion &q_default);



// Other functions for anions

inline d_anion &setident(d_anion &a);
inline d_anion &setzero(d_anion &a);
inline d_anion &setposate(d_anion &a);
inline d_anion &setnegate(d_anion &a);
inline d_anion &setconj(d_anion &a);
inline d_anion &setrand(d_anion &a);
inline d_anion &postProInnerProd(d_anion &a) { return a; }

// Non-standard functions for complex

inline std::complex<double> angle(const std::complex<double> &a);
inline std::complex<double> vangle(const std::complex<double> &a, const std::complex<double> &defsign);
inline std::complex<double> inv(const std::complex<double> &a);
inline std::complex<double> sgn(const std::complex<double> &a);

inline std::complex<double> &setident(std::complex<double> &a);
inline std::complex<double> &setzero(std::complex<double> &a);
inline std::complex<double> &setposate(std::complex<double> &a);
inline std::complex<double> &setnegate(std::complex<double> &a);
inline std::complex<double> &setconj(std::complex<double> &a);
inline std::complex<double> &setrand(std::complex<double> &a);
inline std::complex<double> &postProInnerProd(std::complex<double> &a) { return a; }

inline std::complex<double> &leftmult (std::complex<double>  &a, const std::complex<double>  &b);
inline std::complex<double> &leftmult (std::complex<double>  &a, const double                &b);
inline std::complex<double> &rightmult(const std::complex<double>  &a, std::complex<double>  &b);
inline std::complex<double> &rightmult(const double                &a, std::complex<double>  &b);





// inlines

inline d_anion &operator*=(d_anion &left_op, const double &right_op)
{
    return leftmult(left_op,right_op);
}

inline d_anion &operator*=(d_anion &left_op, const std::complex<double> &right_op)
{
    return leftmult(left_op,right_op);
}

inline d_anion &operator*=(d_anion &left_op, const d_anion &right_op)
{
    return leftmult(left_op,right_op);
}

inline d_anion &operator/=(d_anion &left_op, const double &right_op)
{
    return leftmult(left_op,1/right_op);
}

inline d_anion &setident(d_anion &a)
{
    a = 1.0;

    return a;
}

inline d_anion &setzero(d_anion &a)
{
    a = 0.0;

    return a;
}

inline d_anion &setposate(d_anion &a)
{
    return a;
}

inline d_anion &setnegate(d_anion &a)
{
    a *= -1.0;

    return a;
}

inline d_anion &setconj(d_anion &a)
{
    if ( a.iscomplex() )
    {
        setconj(a.leftpart());
        setnegate(a.rightpart());
    }

    return a;
}

inline d_anion &setrand(d_anion &a)
{
    if ( a.size() )
    {
        int i;

        for ( i = 0 ; i < a.size() ; i++ )
        {
            a("&",i) = ((double) svm_rand())/SVM_RAND_MAX;
        }
    }

    return a;
}

inline std::complex<double> angle(const std::complex<double> &a)
{
    double absa = abs(a);

    if ( absa == 0.0 )
    {
        std::complex<double> resultzero = 0.0;

        return resultzero;
    }

    return a/absa;
}

inline std::complex<double> vangle(const std::complex<double> &a, const std::complex<double> &defsign)
{
    double absa = abs(a);

    if ( absa == 0.0 )
    {
        return defsign;
    }

    return a/absa;
}

inline std::complex<double> inv(const std::complex<double> &a)
{
    return conj(a)/norm(a);
}

inline std::complex<double> sgn(const std::complex<double> &a)
{
    d_anion tempa(a);

    return ((std::complex<double>) sgn(tempa));
}

inline std::complex<double> &setident(std::complex<double> &a)
{
    return ( a = 1 );
}

inline std::complex<double> &setzero(std::complex<double> &a)
{
    return ( a = 0 );
}

inline std::complex<double> &setposate(std::complex<double> &a)
{
    return a;
}

inline std::complex<double> &setnegate(std::complex<double> &a)
{
    return ( a *= -1 );
}

inline std::complex<double> &setconj(std::complex<double> &a)
{
    return ( a = conj(a) );
}

inline std::complex<double> &setrand(std::complex<double> &a)
{
    a.real(((double) svm_rand())/SVM_RAND_MAX);
    a.imag(((double) svm_rand())/SVM_RAND_MAX);

    return a;
}

inline std::complex<double>  &leftmult (std::complex<double>  &a, const std::complex<double>  &b)
{
    return a *= b;
}

inline std::complex<double>  &leftmult (std::complex<double>  &a, const double                &b)
{
    return a *= b;
}

inline std::complex<double>  &rightmult(const std::complex<double>  &a, std::complex<double>  &b)
{
    return b *= a;
}

inline std::complex<double>  &rightmult(const double                &a, std::complex<double>  &b)
{
    return b *= a;
}

d_anion operator+(const d_anion &left_op)
{
    return left_op;
}

d_anion operator-(const d_anion &left_op)
{
    d_anion result(left_op);

    return result *= -1.0;
}

d_anion operator+(const double &left_op, const d_anion &right_op)
{
    d_anion temp(left_op);

    return temp += right_op;
}

d_anion operator+(const std::complex<double> &left_op, const d_anion &right_op)
{
    d_anion temp(left_op);

    return temp += right_op;
}

d_anion operator+(const d_anion &left_op, const double &right_op)
{
    d_anion temp(left_op);

    return temp += right_op;
}

d_anion operator+(const d_anion &left_op, const std::complex<double> &right_op)
{
    d_anion temp(left_op);

    return temp += right_op;
}

d_anion operator+(const d_anion &left_op, const d_anion &right_op)
{
    d_anion temp(left_op);

    return temp += right_op;
}

d_anion operator-(const double &left_op, const d_anion &right_op)
{
    d_anion temp(left_op);

    return temp -= right_op;
}

d_anion operator-(const std::complex<double> &left_op, const d_anion &right_op)
{
    d_anion temp(left_op);

    return temp -= right_op;
}

d_anion operator-(const d_anion &left_op, const double &right_op)
{
    d_anion temp(left_op);

    return temp -= right_op;
}

d_anion operator-(const d_anion &left_op, const std::complex<double> &right_op)
{
    d_anion temp(left_op);

    return temp -= right_op;
}

d_anion operator-(const d_anion &left_op, const d_anion &right_op)
{
    d_anion temp(left_op);

    return temp -= right_op;
}

inline d_anion &operator+=(d_anion &left_op, const double &right_op)
{
    d_anion temp(right_op);

    return ( left_op += temp );
}

inline d_anion &operator+=(d_anion &left_op, const std::complex<double> &right_op)
{
    d_anion temp(right_op);

    return ( left_op += temp );
}

inline d_anion &operator-=(d_anion &left_op, const double &right_op)
{
    d_anion temp(right_op);

    return ( left_op -= temp );
}

inline d_anion &operator-=(d_anion &left_op, const std::complex<double> &right_op)
{
    d_anion temp(right_op);

    return ( left_op -= temp );
}

d_anion operator*(const double &left_op, const d_anion &right_op)
{
    d_anion result(right_op);

    return rightmult(left_op,result);
}

d_anion operator*(const std::complex<double> &left_op, const d_anion &right_op)
{
    d_anion result(right_op);

    return rightmult(left_op,result);
}

d_anion operator*(const d_anion &left_op, const double &right_op)
{
    d_anion result(left_op);

    return leftmult(result,right_op);
}

d_anion operator*(const d_anion &left_op, const std::complex<double> &right_op)
{
    d_anion result(left_op);

    return leftmult(result,right_op);
}

d_anion operator*(const d_anion &left_op, const d_anion &right_op)
{
    d_anion result(left_op);

    return leftmult(result,right_op);
}

d_anion operator/(const d_anion &left_op, const double &right_op)
{
    d_anion result(left_op);

    return ( result *= (1/right_op) );
}

int operator==(const double &left_op, const d_anion &right_op)
{
    d_anion temp(left_op);

    return temp == right_op;
}

int operator==(const std::complex<double> &left_op, const d_anion &right_op)
{
    d_anion temp(left_op);

    return temp == right_op;
}

int operator==(const d_anion &left_op, const double &right_op)
{
    d_anion temp(right_op);

    return left_op == temp;
}

int operator==(const d_anion &left_op, const std::complex<double> &right_op)
{
    d_anion temp(right_op);

    return left_op == temp;
}

int operator!=(const double &left_op, const d_anion &right_op)
{
    return !(left_op == right_op);
}

int operator!=(const std::complex<double> &left_op, const d_anion &right_op)
{
    return !(left_op == right_op);
}

int operator!=(const d_anion &left_op, const double &right_op)
{
    return !(left_op == right_op);
}

int operator!=(const d_anion &left_op, const std::complex<double> &right_op)
{
    return !(left_op == right_op);
}

int operator!=(const d_anion &left_op, const d_anion &right_op)
{
    return !(left_op == right_op);
}

inline d_anion &oneProduct(d_anion &res, const d_anion &a)
{
    return res = a;
}

inline d_anion &twoProduct(d_anion &res, const d_anion &a, const d_anion &b)
{
    res = a;
    setconj(res);
    res *= b;

    return res;
}

inline d_anion &threeProduct(d_anion &res, const d_anion &a, const d_anion &b, const d_anion &c)
{
    res = a;
    res *= b;
    res *= c;

    return res;
}

inline d_anion &fourProduct(d_anion &res, const d_anion &a, const d_anion &b, const d_anion &c, const d_anion &d)
{
    res = a;
    res *= b;
    res *= c;
    res *= d;

    return res;
}

inline d_anion &mProduct(d_anion &res, int m, const d_anion *a)
{
    NiceAssert( m >= 0 );

    setident(res);

    if ( m > 0 )
    {
        int i;

        for ( i = 0 ; i < m ; i++ )
        {
            res *= a[i];
        }
    }

    return res;
}

inline d_anion &twoProductNoConj(d_anion &res, const d_anion &a, const d_anion &b)
{
    res = a;
    res *= b;

    return res;
}

inline d_anion &twoProductRevConj(d_anion &res, const d_anion &a, const d_anion &b)
{
    res = b;
    setconj(res);
    rightmult(a,res);

    return res;
}


inline int testisvnan(const d_anion &x)
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

inline int testisinf (const d_anion &x)
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

inline int testispinf(const d_anion &x)
{
    int res = 0;

    if ( ( x.size() == 1 ) && ( testispinf(x(zeroint())) ) )
    {
        res = 1;
    }

    return res;
}

inline int testisninf(const d_anion &x)
{
    int res = 0;

    if ( ( x.size() == 1 ) && ( testisninf(x(zeroint())) ) )
    {
        res = 1;
    }

    return res;
}





#endif
