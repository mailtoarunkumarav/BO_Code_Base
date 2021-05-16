
//
// Integer class that defaults to -1
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _nonzeroint_h
#define _nonzeroint_h

#include "basefn.h"

class nzint;

inline std::ostream &operator<<(std::ostream &output, const nzint &src );
inline std::istream &operator>>(std::istream &input,        nzint &dest);

inline std::istream &streamItIn(std::istream &input, nzint& dest, int processxyzvw = 1);
inline std::ostream &streamItOut(std::ostream &output, const nzint& src, int retainTypeMarker = 0);

inline void qswap(nzint &a, nzint &b);

class nzint
{
    friend std::ostream &operator<<(std::ostream &output, const nzint &src );
    friend std::istream &operator>>(std::istream &input,        nzint &dest);

    friend void qswap(nzint &a, nzint &b);

public:

    svm_explicit nzint()                            { val = -1;      return;       }
    svm_explicit nzint(int src)                     { val = src;     return;       }
                 nzint(const nzint &src)            { val = src.val; return;       }
    
    nzint &operator=(int src)          { val = src;     return *this; }
    nzint &operator=(const nzint &src) { val = src.val; return *this; }
    operator int() const               {                return val;   }

private:

    int val; // -1 if not set, >=0 otherwise
};

inline void qswap(nzint &a, nzint &b)
{
    int x = a.val; a.val = b.val; b.val = x;

    return;
}

inline nzint &setident (nzint &a          );
inline nzint &setzero  (nzint &a          );
inline nzint &setposate(nzint &a          );
inline nzint &setnegate(nzint &a          );
inline nzint &setconj  (nzint &a          );
inline nzint &setrand  (nzint &a          );
inline nzint &leftmult (nzint &a, nzint  b);
inline nzint &rightmult(nzint  a, nzint &b);
inline nzint &postProInnerProd(nzint &a);


inline nzint &setident (nzint &a          ) { a = 1;                           return a; }
inline nzint &setzero  (nzint &a          ) { a = -1;                          return a; }
inline nzint &setposate(nzint &a          ) {                                  return a; }
inline nzint &setnegate(nzint &a          ) { a = -(int) a;                    return a; }
inline nzint &setconj  (nzint &a          ) {                                  return a; }
inline nzint &setrand  (nzint &a          ) { a = (2*(svm_rand()%2))-1;        return a; }
inline nzint &leftmult (nzint &a, nzint  b) { a = ( ((int) a) * ((int) b) );   return a; }
inline nzint &rightmult(nzint  a, nzint &b) { b = ( ((int) a) * ((int) b) );   return b; }
inline nzint &postProInnerProd(nzint &a) { return a; }

inline std::ostream &operator<<(std::ostream &output, const nzint &src )
{
    output << (int) src;

    return output;
}

inline std::istream &operator>>(std::istream &input, nzint &dest)
{
    int temp;
    input >> temp;
    dest = temp;

    return input;
}

inline std::istream &streamItIn(std::istream &input, nzint& dest, int processxyzvw)
{
    (void) processxyzvw;

    input >> dest;

    return input;
}

inline std::ostream &streamItOut(std::ostream &output, const nzint& src, int retainTypeMarker)
{
    (void) retainTypeMarker;

    output << src;

    return output;
}

#endif
