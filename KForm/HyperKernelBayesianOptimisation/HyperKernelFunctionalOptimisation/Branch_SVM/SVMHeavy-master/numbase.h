
//
// Non-standard functions for reals
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <math.h>
#include <stdlib.h>
#include "basefn.h"
#include "gslrefs.h"

#ifndef _numbase_h
#define _numbase_h




// Constants:
//
// NUMBASE_EULER:          euler's constant
// NUMBASE_E:              e
// NUMBASE_PI:             pi
// NUMBASE_PION2:          pi/2
// NUMBASE_PION4:          pi/4
// NUMBASE_1ONPI:          1/pi
// NUMBASE_2ONPI:          2/pi
// NUMBASE_SQRT2ONPI:      sqrt(2/pi)
// NUMBASE_SQRTPI:         sqrt(pi)
// NUMBASE_SQRTSQRTPI:     sqrt(sqrt(pi))
// NUMBASE_2ONSQRTPI:      2/sqrt(pi)
// NUMBASE_1ONSQRT2PI:     1/sqrt(2pi)
// NUMBASE_1ONSQRTSQRT2PI: 1/sqrt(sqrt(2pi))
// NUMBASE_LN2:            ln(2)
// NUMBASE_LNPI:           ln(pi)
// NUMBASE_LN10:           ln(10)
// NUMBASE_LOG2E:          log2(e)
// NUMBASE_LOG10E:         log10(e)
// NUMBASE_SQRT2:          sqrt(2)
// NUMBASE_SQRT1ON2:       sqrt(1/2)
// NUMBASE_SQRT3:          sqrt(3)

#define NUMBASE_EULER       0.57721566490153286060651209008
#define NUMBASE_E           2.71828182845904523536028747135
#define NUMBASE_PI          3.14159265358979323846264338328
#define NUMBASE_PION2       1.57079632679489661923132169164
#define NUMBASE_PION4       0.78539816339744830966156608458
#define NUMBASE_1ONPI       0.31830988618379067153776752675
#define NUMBASE_2ONPI       0.63661977236758134307553505349
#define NUMBASE_SQRT2ONPI   0.7978845608
#define NUMBASE_SQRTPI      1.77245385090551602729816748334
#define NUMBASE_SQRTSQRTPI  1.3313353638
#define NUMBASE_2ONSQRTPI   1.12837916709551257389615890312
#define NUMBASE_1ONSQRT2PI  0.39894228040143267793994605993
#define NUMBASE_1ONSQRTSQRT2PI  0.63161877774
#define NUMBASE_LN2         0.69314718055994530941723212146
#define NUMBASE_LNPI        1.14472988584940017414342735135
#define NUMBASE_LN10        2.30258509299404568401799145468
#define NUMBASE_LN2PI       1.83787706641
#define NUMBASE_LOG2E       1.44269504088896340735992468100
#define NUMBASE_LOG10E      0.43429448190325182765112891892
#define NUMBASE_SQRT2       1.41421356237309504880168872421
#define NUMBASE_SQRT1ON2    0.70710678118654752440084436210
#define NUMBASE_SQRT3       1.73205080756887729352744634151
#define NUMBASE_HALFLOG2PI  0.91893853320467274178032973640562










// Non-standard functions for real/int

inline double arg   (double a); // apparently needs to be out for visual studio (with c++11?)
inline double abs1  (double a);
inline double abs2  (double a);
inline double absp  (double a, double p);
inline double absinf(double a);
inline double absd  (double a);
inline double angle (double a);
inline double vangle(double a, double res = 0.0);
inline double conj  (double a);
inline double real  (double a);
inline double imag  (double a);
inline double inv   (double a);
inline double norm1 (double a);
inline double norm2 (double a);
inline double normp (double a, double p);
inline double normd (double a);
inline double sgn   (double a);
inline double tenup (double a);

//inline int hypot (int a, int b);
//inline int arg   (int a);
inline int abs1  (int a);
inline int abs2  (int a);
inline int absp  (int a, double p);
inline int absd  (int a);
inline int angle (int a);
inline int vangle(int a, double res = 0);
inline int absinf(int a);
inline int conj  (int a);
inline int real  (int a);
inline int imag  (int a);
inline double inv(int a);
inline int norm1 (int a);
inline int norm2 (int a);
inline int normp (int a, double p);
inline int normd (int a);
inline int sgn   (int a);
inline int tenup (int a);




inline double &scaladd(double &a, const double &b); 
inline double &scaladd(double &a, const double &b, const double &c); 
inline double &scaladd(double &a, const double &b, const double &c, const double &d); 
inline double &scaladd(double &a, const double &b, const double &c, const double &d, const double &e); 
inline double &scalsub(double &a, const double &b); 
inline double &scalmul(double &a, const double &b); 
inline double &scaldiv(double &a, const double &b);

inline double &scaladd(double &a, const double &b) { return a += b; }
inline double &scaladd(double &a, const double &b, const double &c) { return a += (b*c); }
inline double &scaladd(double &a, const double &b, const double &c, const double &d) { return a += (b*c*d); }
inline double &scaladd(double &a, const double &b, const double &c, const double &d, const double &e) { return a += (b*c*d*e); }
inline double &scalsub(double &a, const double &b) { return a -= b; }
inline double &scalmul(double &a, const double &b) { return a *= b; }
inline double &scaldiv(double &a, const double &b) { return a /= b; }








// Borrowed from Fortran
//
// dsign returns a with sign of b (if b == 0 returns a)
//
// sppythag finds sqrt(a**2+b**2) without overflow or destructive underflow
// (translated from eispack function pythag.f)

inline double dsign(double a, double b);
inline double dsign(double a, double b)
{
    return ( b >= 0.0 ) ? a : -a;
}

inline double sppythag(double a, double b);
inline double sppythag(double a, double b)
{
    double p,r,s,t,u;

    a = fabs(a);
    b = fabs(b);

    p = ( a > b ) ? a : b;

    if ( p > 0.0 )
    {
        r  = ( ( a < b ) ? a : b );
        r /= p;
        r *= r;
        t = 4.0 + r;

        while ( t > 4.0 )
        {
            s = r/t;
            u = 1.0 + 2.0*s;
            p = u*p;
            t = (s/u);
            r *= t*t;
            t = 4.0 + r;
        }
    }

    return p;
}









// Manual operations for double/int

inline double &setident (double &a);
inline double &setzero  (double &a);
inline double &setposate(double &a);
inline double &setnegate(double &a);
inline double &setconj  (double &a);
inline double &setrand  (double &a); // uniform 0 to 1
inline double &leftmult (double &a, double  b);
inline double &rightmult(double  a, double &b);
inline double &postProInnerProd(double &a);

inline int &setident (int &a);
inline int &setzero  (int &a);
inline int &setposate(int &a);
inline int &setnegate(int &a);
inline int &setconj  (int &a);
inline int &setrand  (int &a); // random -1 or 1
inline int &leftmult (int &a, int  b);
inline int &rightmult(int  a, int &b);
inline int &postProInnerProd(int &a);

inline unsigned int &setident (unsigned int &a);
inline unsigned int &setzero  (unsigned int &a);
inline unsigned int &setposate(unsigned int &a);
inline unsigned int &setnegate(unsigned int &a);
inline unsigned int &setconj  (unsigned int &a);
inline unsigned int &setrand  (unsigned int &a); // random -1 or 1
inline unsigned int &leftmult (unsigned int &a, unsigned int  b);
inline unsigned int &rightmult(unsigned int  a, unsigned int &b);
inline unsigned int &postProInnerProd(unsigned int &a);

inline char &setident (char &a);
inline char &setzero  (char &a);
inline char &setposate(char &a);
inline char &setnegate(char &a);
inline char &setconj  (char &a);
inline char &setrand  (char &a); // random -1 or 1
inline char &leftmult (char &a, char  b);
inline char &rightmult(char  a, char &b);
inline char &postProInnerProd(char &a);

inline std::string &setident (std::string &a); // throw
inline std::string &setzero  (std::string &a); // empty string
inline std::string &setposate(std::string &a);
inline std::string &setnegate(std::string &a); // throw
inline std::string &setconj  (std::string &a); // throw
inline std::string &setrand  (std::string &a); // throw
inline std::string &leftmult (std::string &a, std::string  b); // throw
inline std::string &rightmult(std::string  a, std::string &b); // throw
inline std::string &postProInnerProd(std::string &a);

inline double *&setident (double *&a) { throw("mummble"); return a; }
inline double *&setzero  (double *&a) { return a = NULL; }
inline double *&setposate(double *&a) { return a; }
inline double *&setnegate(double *&a) { throw("mummble"); return a; }
inline double *&setconj  (double *&a) { throw("mummble"); return a; }
inline double *&setrand  (double *&a) { throw("mummble"); return a; }
inline double *&postProInnerProd(double *&a) { return a; }

inline const double *&setident (const double *&a) { throw("mummble"); return a; }
inline const double *&setzero  (const double *&a) { return a = NULL; }
inline const double *&setposate(const double *&a) { return a; }
inline const double *&setnegate(const double *&a) { throw("mummble"); return a; }
inline const double *&setconj  (const double *&a) { throw("mummble"); return a; }
inline const double *&setrand  (const double *&a) { throw("mummble"); return a; }
inline const double *&postProInnerProd(const double *&a) { return a; }












// Inner products etc (used by vectors in template form)

inline double &oneProduct  (double &res, const double &a);
inline double &twoProduct  (double &res, const double &a, const double &b);
inline double &threeProduct(double &res, const double &a, const double &b, const double &c);
inline double &fourProduct (double &res, const double &a, const double &b, const double &c, const double &d);
inline double &mProduct    (double &res, int m, const double *a);

inline double &twoProductNoConj (double &res, const double &a, const double &b);
inline double &twoProductRevConj(double &res, const double &a, const double &b);

inline int &oneProduct  (int &res, const int &a);
inline int &twoProduct  (int &res, const int &a, const int &b);
inline int &threeProduct(int &res, const int &a, const int &b, const int &c);
inline int &fourProduct (int &res, const int &a, const int &b, const int &c, const int &d);
inline int &mProduct    (int &res, int m, const int *a);

inline int &twoProductNoConj (int &res, const int &a, const int &b);
inline int &twoProductRevConj(int &res, const int &a, const int &b);











// Trigonometric and other special functions

inline double cosec    (double a);
inline double sec      (double a);
inline double cot      (double a);
inline double acosec   (double a);
inline double asec     (double a);
inline double acot     (double a);
inline double sinc     (double a);
inline double cosc     (double a);
inline double tanc     (double a);
inline double vers     (double a);
inline double covers   (double a);
inline double hav      (double a);
inline double excosec  (double a);
inline double exsec    (double a);
inline double avers    (double a);
inline double acovers  (double a);
inline double ahav     (double a);
inline double aexcosec (double a);
inline double aexsec   (double a);
inline double cosech   (double a);
inline double sech     (double a);
inline double coth     (double a);
inline double acosech  (double a);
inline double asech    (double a);
inline double acoth    (double a);
inline double sinhc    (double a);
inline double coshc    (double a);
inline double tanhc    (double a);
inline double versh    (double a);
inline double coversh  (double a);
inline double havh     (double a);
inline double excosech (double a);
inline double exsech   (double a);
inline double aversh   (double a);
inline double acovrsh  (double a);
inline double ahavh    (double a);
inline double aexcosech(double a);
inline double aexsech  (double a);
inline double sigm     (double a);
inline double gd       (double a);
inline double asigm    (double a);
inline double agd      (double a);













// Hypervolume Calculation Functions
//
// spherevol(rsq,n): calculate the volume of a hypersphere of radius sqrt(rsq)
//                   in n-dimensional space.  Note that the volume of a
//                   sphere in zero-dimensional space is defined as 1.

double spherevol(double rsq, unsigned int n);








// Other obscure functions
//
// return value: non-zero if error, zero if good
// result: placed in res
//
// dawson:      Dawson function
// gamma:       Gamma function
// lngamma:     Log gamma function
// gamma_inc:   Incomplete gamma function
// psi, psn_n:  Psi functions
// erf,erfc:    Error and complementary error functions
// erfinv:      Inverse of the erf function
// zeta:        Reimann zeta function
// lambertW:    standard W0 branch (W>-1) of the Lambert W function
// lambertWx:   W1 branch (W<-1) of the Lambert W function
// j0,j1,jn:    Bessel functions of the first kind
// k0,k1,kn:    Bessel functions of the second kind
// i0,i1,in:    Modified Bessel functions of the first kind
// k0,k1,kn:    Modified Bessel functions of the second kind
// probit:      Probit function


inline int numbase_dawson   (double &res,           double x);
       int numbase_gamma    (double &res,           double x);
       int numbase_lngamma  (double &res,           double x);
inline int numbase_gamma_inc(double &res, double a, double x);
inline int numbase_psi      (double &res,           double x);
inline int numbase_psi_n    (double &res, int n,    double x);
inline int numbase_erf      (double &res,           double x);
inline int numbase_erfc     (double &res,           double x);
inline int numbase_Phi      (double &res,           double x);
inline int numbase_phi      (double &res,           double x);
       int numbase_erfinv   (double &res,           double x);
inline int numbase_probit   (double &res,           double x);
inline int numbase_zeta     (double &res,           double x);
inline int numbase_lambertW (double &res,           double x);
inline int numbase_lambertWx(double &res,           double x);
       int numbase_j0       (double &res,           double x);
       int numbase_j1       (double &res,           double x);
       int numbase_jn       (double &res, int n,    double x);
       int numbase_y0       (double &res,           double x);
       int numbase_y1       (double &res,           double x);
       int numbase_yn       (double &res, int n,    double x);
       int numbase_i0       (double &res,           double x);
       int numbase_i1       (double &res,           double x);
       int numbase_in       (double &res, int n,    double x);
       int numbase_k0       (double &res,           double x);
       int numbase_k1       (double &res,           double x);
       int numbase_kn       (double &res, int n,    double x);

// For functions that never fail

inline double normPhi(double x);
inline double normphi(double x);






































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================


inline double abs1  (double a          ) { return fabs(a); }
inline double abs2  (double a          ) { return fabs(a); }
inline double absp  (double a, double p) { (void) p; return fabs(a); }
inline double absinf(double a          ) { return fabs(a); }
inline double absd  (double a          ) { return abs2(a); }
inline double angle (double a          ) { return ( a > 0 ) ? +1 : ( ( a < 0 ) ? -1 : 0 ); }
inline double vangle(double a, double p) { (void) a; return p; }
inline double arg   (double a          ) { return ( a >= 0 ) ? 0 : NUMBASE_PI; }
inline double conj  (double a          ) { return a; }
inline double imag  (double a          ) { (void) a; return 0; }
inline double inv   (double a          ) { return 1/a; }
inline double inv   (int a             ) { return inv((double) a); }
inline double norm1 (double a          ) { return fabs(a); }
inline double norm2 (double a          ) { return a*a; }
inline double normp (double a, double p) { return pow(fabs(a),p); }
inline double normd (double a          ) { return norm2(a); }
inline double real  (double a          ) { return a; }
inline double sgn   (double a          ) { return ( a > 0 ) ? +1 : ( ( a < 0 ) ? -1 : 0 ); }
inline double tenup (double a          ) { return pow(10,a); }

//inline double arg (int    a          ) { return ( a >= 0 ) ? 0 : NUMBASE_PI; }

inline int    abs1  (int    a          ) { return abs(a); }
inline int    abs2  (int    a          ) { return abs(a); }
inline int    absp  (int    a, double p) { (void) p; return abs(a); }
inline int    absinf(int    a          ) { return abs(a); }
inline int    absd  (int    a          ) { return abs2(a); }
inline int    angle (int    a          ) { return ( a > 0 ) ? +1 : ( ( a < 0 ) ? -1 : 0 ); }
inline int    vangle(int    a, double p) { (void) a; int pp = (int) p; NiceAssert( p == pp ); return pp; }
inline int    conj  (int    a          ) { return a; }
inline int    imag  (int    a          ) { (void) a; return 0; }
inline int    norm1 (int    a          ) { return ( a < 0 ) ? -a : a; }
inline int    norm2 (int    a          ) { return a*a; }
inline int    normp (int    a, double p) { int pp = (int) p; NiceAssert( p == pp ); NiceAssert( pp >= 0 ); return ( a < 0 ) ? normp(-a,pp) : ( ( pp > 0 ) ? a*normp(a,pp-1) : 1 ); }
inline int    normd (int    a          ) { return norm2(a); }
inline int    real  (int    a          ) { return a; }
inline int    sgn   (int    a          ) { return ( a > 0 ) ? +1 : ( ( a < 0 ) ? -1 : 0 ); }
inline int    tenup (int    a          ) { return (int) pow(10.0,(double) a); }

inline double &setident (double &a           ) { return ( a = 1 ); }
inline double &setzero  (double &a           ) { return ( a = 0 ); }
inline double &setposate(double &a           ) { return a; }
inline double &setnegate(double &a           ) { return ( a *= -1 ); }
inline double &setconj  (double &a           ) { return a; }
inline double &setrand  (double &a           ) { return ( a = ((double) svm_rand())/SVM_RAND_MAX ); }
inline double &leftmult (double &a, double  b) { return a *= b; }
inline double &rightmult(double  a, double &b) { return b *= a; }
inline double &postProInnerProd(double &a) { return a; }

inline double &oneProduct  (double &res, const double &a)                                                    { res = a;       return res; }
inline double &twoProduct  (double &res, const double &a, const double &b)                                   { res = a*b;     return res; }
inline double &threeProduct(double &res, const double &a, const double &b, const double &c)                  { res = a*b*c;   return res; }
inline double &fourProduct (double &res, const double &a, const double &b, const double &c, const double &d) { res = a*b*c*d; return res; }
inline double &mProduct    (double &res, int m, const double *a) { res = 1.0; NiceAssert( m >= 0 ); if ( m ) { int i; for ( i = 0 ; i < m ; i++ ) { res = a[i]; } } return res; }

inline double &twoProductNoConj (double &res, const double &a, const double &b) { res = a*b; return res; }
inline double &twoProductRevConj(double &res, const double &a, const double &b) { res = a*b; return res; }

inline int &setident (int &a        ) { return ( a = 1 ); }
inline int &setzero  (int &a        ) { return ( a = 0 ); }
inline int &setposate(int &a        ) { return a; }
inline int &setnegate(int &a        ) { return ( a *= -1 ); }
inline int &setconj  (int &a        ) { return a; }
inline int &setrand  (int &a        ) { return ( a = (2*(svm_rand()%2))-1 ); }
inline int &leftmult (int &a, int  b) { return a *= b; }
inline int &rightmult(int  a, int &b) { return b *= a; }
inline int &postProInnerProd(int &a) { return a; }

inline unsigned int &setident (unsigned int &a        ) { return ( a = 1 ); }
inline unsigned int &setzero  (unsigned int &a        ) { return ( a = 0 ); }
inline unsigned int &setposate(unsigned int &a        ) { return a; }
inline unsigned int &setnegate(unsigned int &a        ) { NiceAssert( a == 0 ); return a = 0; }
inline unsigned int &setconj  (unsigned int &a        ) { return a; }
inline unsigned int &setrand  (unsigned int &a        ) { return ( a = (svm_rand()%2) ); }
inline unsigned int &leftmult (unsigned int &a, unsigned int  b) { return a *= b; }
inline unsigned int &rightmult(unsigned int  a, unsigned int &b) { return b *= a; }
inline unsigned int &postProInnerProd(unsigned int &a) { return a; }

inline char &setident (char &a         ) { return ( a = 1 ); }
inline char &setzero  (char &a         ) { return ( a = 0 ); }
inline char &setposate(char &a         ) { return a; }
inline char &setnegate(char &a         ) { return ( a *= -1 ); }
inline char &setconj  (char &a         ) { return a; }
inline char &setrand  (char &a         ) { return ( a = (2*(svm_rand()%2))-1 ); }
inline char &leftmult (char &a, char  b) { return a *= b; }
inline char &rightmult(char  a, char &b) { return b *= a; }
inline char &postProInnerProd(char &a) { return a; }

inline std::string &setident (std::string &a) { throw("setident string is meaningless"); return a;}
inline std::string &setzero  (std::string &a) { a = ""; return a; }
inline std::string &setposate(std::string &a) { return a; }
inline std::string &setnegate(std::string &a) { throw("setnegate string is meaningless"); return a;}
inline std::string &setconj  (std::string &a) { throw("setconj string is meaningless"); return a;}
inline std::string &setrand  (std::string &a) { throw("setrand string is meaningless"); return a;}
inline std::string &leftmult (std::string &a, std::string  b) { (void) b; throw("leftmult string is meaningless"); return a;}
inline std::string &rightmult(std::string  a, std::string &b) { (void) a; throw("rightmult string is meaningless"); return b;}
inline std::string &postProInnerProd(std::string &a) { return a; }


inline int &oneProduct  (int &res, const int &a)                                           { res = a;       return res; }
inline int &twoProduct  (int &res, const int &a, const int &b)                             { res = a*b;     return res; }
inline int &threeProduct(int &res, const int &a, const int &b, const int &c)               { res = a*b*c;   return res; }
inline int &fourProduct (int &res, const int &a, const int &b, const int &c, const int &d) { res = a*b*c*d; return res; }
inline int &mProduct    (int &res, int m, const int *a) { res = 1; NiceAssert( m >= 0 ); if ( m ) { int i; for ( i = 0 ; i < m ; i++ ) { res = a[i]; } } return res; }

inline int &twoProductNoConj (int &res, const int &a, const int &b) { res = a*b; return res; }
inline int &twoProductRevConj(int &res, const int &a, const int &b) { res = a*b; return res; }

// yes sin(a)*inv(a) = inv(a)*sin(a), even for anions

inline double cosec    (double a) { return inv(sin(a)); }
inline double sec      (double a) { return inv(cos(a)); }
inline double cot      (double a) { return cos(a)*inv(sin(a)); }
inline double acosec   (double a) { return asin(inv(a)); }
inline double asec     (double a) { return acos(inv(a)); }
inline double acot     (double a) { return atan(inv(a)); }
inline double sinc     (double a) { return ( fabs(a) <= 1e-7 ) ? 1 : sin(a)*inv(a); }
inline double cosc     (double a) { return cos(a)*inv(a); }
inline double tanc     (double a) { return ( fabs(a) <= 1e-7 ) ? 1 : tan(a)*inv(a); }
inline double vers     (double a) { return 1-cos(a); }
inline double covers   (double a) { return 1-sin(a); }
inline double hav      (double a) { return vers(a)/2.0; }
inline double excosec  (double a) { return cosec(a)-1; }
inline double exsec    (double a) { return sec(a)-1; }
inline double avers    (double a) { return acos(a+1); }
inline double acovers  (double a) { return asin(a+1); }
inline double ahav     (double a) { return avers(2*a); }
inline double aexcosec (double a) { return acosec(a+1); }
inline double aexsec   (double a) { return asec(a+1); }
inline double cosech   (double a) { return inv(sinh(a)); }
inline double sech     (double a) { return inv(cosh(a)); }
inline double coth     (double a) { return cosh(a)*inv(sinh(a)); }
inline double acosech  (double a) { return asinh(inv(a)); }
inline double asech    (double a) { return acosh(inv(a)); }
inline double acoth    (double a) { return atanh(inv(a)); }
inline double sinhc    (double a) { return ( fabs(a) <= 1e-7 ) ? 1 : sinh(a)*inv(a); }
inline double coshc    (double a) { return cosh(a)*inv(a); }
inline double tanhc    (double a) { return ( fabs(a) <= 1e-7 ) ? 1 : tanh(a)*inv(a); }
inline double versh    (double a) { return 1-cosh(a); }
inline double coversh  (double a) { return 1-sinh(a); }
inline double havh     (double a) { return versh(a)/2.0; }
inline double excosech (double a) { return cosech(a)-1; }
inline double exsech   (double a) { return sech(a)-1; }
inline double aversh   (double a) { return acosh(a+1); }
inline double acovrsh  (double a) { return asinh(a+1); }
inline double ahavh    (double a) { return aversh(2*a); }
inline double aexcosech(double a) { return acosech(a+1); }
inline double aexsech  (double a) { return asech(a+1); }
inline double sigm     (double a) { return inv(1+exp(a)); }
inline double gd       (double a) { return 2*atan(tanh(a/2.0)); }
inline double asigm    (double a) { return log(inv(a)-1.0); }
inline double agd      (double a) { return 2*atanh(tan(a/2.0)); }




inline int numbase_dawson   (double &res,           double x) { return gsl_dawson(res,x);      }
inline int numbase_gamma_inc(double &res, double a, double x) { return gsl_gamma_inc(res,a,x); }
inline int numbase_psi      (double &res,           double x) { return gsl_psi(res,x);         }
inline int numbase_psi_n    (double &res, int n,    double x) { return gsl_psi_n(res,n,x);     }
inline int numbase_zeta     (double &res,           double x) { return gsl_zeta(res,x);        }
inline int numbase_lambertW (double &res,           double x) { return gsl_lambertW0(res,x);   }
inline int numbase_lambertWx(double &res,           double x) { return gsl_lambertW1(res,x);   }
inline int numbase_erf      (double &res,           double x) { res = erf(x);                                                 return 0;    }
inline int numbase_erfc     (double &res,           double x) { res = 1-erf(x);                                               return 0;    }
inline int numbase_Phi      (double &res,           double x) { res = 0.5 + (0.5*erf(x*NUMBASE_SQRT1ON2));                    return 0;    }
inline int numbase_phi      (double &res,           double x) { res = NUMBASE_1ONSQRT2PI*exp(-x*x/2);                         return 0;    }
inline int numbase_probit   (double &res,           double x) { int ires = numbase_erfinv(res,(2*x)-1); res *= NUMBASE_SQRT2; return ires; }

inline double normPhi(double x)
{
    return 0.5 + (0.5*erf(x*NUMBASE_SQRT1ON2));
}

inline double normphi(double x)
{
    return NUMBASE_1ONSQRT2PI*exp(-x*x/2);
}




#endif
