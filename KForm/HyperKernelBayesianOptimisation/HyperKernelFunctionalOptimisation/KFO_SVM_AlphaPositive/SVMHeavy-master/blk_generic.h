
//
// Functional block base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_generic_h
#define _blk_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.h"



// Defines a very basic set of blocks for use in machine learning.


class BLK_Generic;


typedef int (*gcallback)(gentype &res, const SparseVector<gentype> &x, void *fndata);


// Swap and zeroing (restarting) functions

inline void qswap(BLK_Generic &a, BLK_Generic &b);
inline BLK_Generic &setzero(BLK_Generic &a);

class BLK_Generic : public ML_Base
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Generic(int isIndPrune = 0);
    BLK_Generic(const BLK_Generic &src, int isIndPrune = 0);
    BLK_Generic(const BLK_Generic &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_Generic &operator=(const BLK_Generic &src) { assign(src); return *this; }
    virtual ~BLK_Generic();





    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const;

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual       ML_Base &getML     (void)       { return static_cast<      ML_Base &>(getBLK()     ); }
    virtual const ML_Base &getMLconst(void) const { return static_cast<const ML_Base &>(getBLKconst()); }

    virtual int isSampleMode(void) const { return xissample; }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp = DEFAULT_SAMPLES_SAMPLE)
    {
        int res = ( xissample != nv ) ? 1 : 0; 

        if ( ( xissample = nv ) ) 
        { 
            doutfn.finalise(); 
        } 

        return res | ML_Base::setSampleMode(nv,xmin,xmax,Nsamp); 
    }


    // ================================================================
    //     BLK Specific functions
    // ================================================================

    virtual       BLK_Generic &getBLK     (void)       { return *this; }
    virtual const BLK_Generic &getBLKconst(void) const { return *this; }

    // Information functions (training data):

    virtual const gentype &outfn    (void) const { return doutfn; }
    virtual const gentype &outfngrad(void) const { (**thisthisthis).outgrad = outfn(); (**thisthisthis).outgrad.realDeriv(0,0); return (**thisthisthis).outgrad; }

    // General modification and autoset functions

    virtual int setoutfn(const gentype &newoutfn) { doutfn = newoutfn; return 1; }
    virtual int setoutfn(const std::string &newoutfn) { doutfn = newoutfn; return 1; }

    // Streams used by userio

    virtual int setuseristream(std::istream &src) { xuseristream = &src; return 1; }
    virtual int setuserostream(std::ostream &dst) { xuserostream = &dst; return 1; }

    virtual std::istream &useristream(void) const { return *xuseristream; }
    virtual std::ostream &userostream(void) const { return *xuserostream; }

    // Callback function used by calbak

    virtual int setcallback(gcallback ncallback, void *ncallbackfndata) { xcallback = ncallback; xcallbackfndata = ncallbackfndata; return 1; }
    virtual gcallback callback(void)   const { return xcallback; }
    virtual void *callbackfndata(void) const { return xcallbackfndata; }

    // Callback string used by MEX interface

    virtual int setmexcall  (const std::string &xmexfn) { mexfn   = xmexfn;   return 1; }
    virtual int setmexcallid(int xmexfnid)              { mexfnid = xmexfnid; return 1; }
    virtual const std::string &getmexcall  (void) const { return mexfn;                 }
    virtual int                getmexcallid(void) const { return mexfnid;               }

    // Callback string used by sytem call interface
    //
    // This is cast as a (gentype) function and then evaluated given x
    //
    // xfilename:  datafile containing x data (not written if string empty)
    // yfilename:  datafile containing y data (not written if string empty)
    // xyfilename: datafile containing xy (target at end) data (not written if string empty)
    // yxfilename: datafile containing yx (target at start) data (not written if string empty)
    // rfilename:  name of file where result is retrieved (NULL if string empty)

    virtual int setsyscall(const std::string &xsysfn)   { sysfn   = xsysfn; return 1; }
    virtual int setxfilename(const std::string &fname)  { xfname  = fname;  return 1; }
    virtual int setyfilename(const std::string &fname)  { yfname  = fname;  return 1; }
    virtual int setxyfilename(const std::string &fname) { xyfname = fname;  return 1; }
    virtual int setyxfilename(const std::string &fname) { yxfname = fname;  return 1; }
    virtual int setrfilename(const std::string &fname)  { rfname  = fname;  return 1; }

    virtual const std::string &getsyscall(void)    const { return sysfn;   }
    virtual const std::string &getxfilename(void)  const { return xfname;  }
    virtual const std::string &getyfilename(void)  const { return yfname;  }
    virtual const std::string &getxyfilename(void) const { return xyfname; }
    virtual const std::string &getyxfilename(void) const { return yxfname; }
    virtual const std::string &getrfilename(void)  const { return rfname;  }

    // MEX function: mex does not actually exist as far as this code is concerned.
    // Hence for the mex callback blocks you need to give it a funciton to call.
    // You'll need to set the following globally here to access it.  Operation is
    // assumed to be:
    //
    // getsetExtVar: - get or set external (typically mex) variable.
    //               - if num >= 0 then loads extvar num into res.  If extvar is
    //                 a function handle then src acts as an argument (optional,
    //                 not used if null, multiple arguments if set).
    //               - if num == -1 then loads external variable named in res
    //                 (res must be string) into res before returning.  In this
    //                 case src gives preferred type if result interpretation is
    //                 ambiguous (type of res will attempt to copy gentype of
    //                 src).
    //               - if num == -2 then loads contents of src into external
    //                 variable named in res before returning.
    //               - if num == -3 then evaluates fn(v) where fn is a matlab
    //                 function named by res, v is the set of arguments (see
    //                 num >= 0) and the result is stored in res.
    //               - returns 0 on success, -1 on failure.
    //
    // Call is (*getsetExtVar)(res,src,mexfnid), where res = mexfn is set prior
    // to call.

    // Mercer cache size: set -1 for no cache, N >= 0 for cache of size N
    //
    // fill cache: pre-calculates all elements in cache for later use
    // norm cache: normalise cache so that diagonals are all 1 in K2

    virtual int mercachesize(void) const { return xmercachesize; }
    virtual int setmercachesize(int nv) { NiceAssert( nv >= -1 ); xmercachesize = nv; return 1; }

    virtual int mercachenorm(void) const { return xmercachenorm; }
    virtual int setmercachenorm(int nv) { xmercachenorm = nv; return 1; }

    // ML block averaging: set/remove element in list of ML blocks being averaged

    virtual int setmlqlist(int i, ML_Base &src)          { xmlqlist("[]",i) = &src; xmlqweight("[]",i) = 1.0;                           return 1; }
    virtual int setmlqlist(const Vector<ML_Base *> &src) { xmlqlist = src; xmlqweight.indalign(xmlqlist); xmlqweight = onedblgentype(); return 1; }

    virtual int setmlqweight(int i, const gentype &w)  { xmlqweight("[]",i) = w; return 1; }
    virtual int setmlqweight(const Vector<gentype> &w) { xmlqweight = w;         return 1; }

    virtual int removemlqlist(int i) { xmlqlist.remove(i); xmlqweight.remove(i); return 1; }

    const SparseVector<ML_Base *> mlqlist(void) const { return xmlqlist; }
    const SparseVector<gentype>   mlqweight(void) const { return xmlqweight; }

    // Kernel training:
    //
    // K(..m.) = sum_{i_0,...,i_{m-1}} lambda_{i_0} ... lambda_{i_{m-1}} K(x_{i_0},...,x_{i_{m-1}},...)

    virtual const Vector<double> &lambdaKB(void) const { return KBlambda; }
    virtual int setlambdaKB(const Vector<double> &nv) { KBlambda = nv; return 1; }

    // Bernstein polynomials
    //
    // degree and index can be either null, int or Vector<int>.

    virtual const gentype &bernDegree(void) const { return berndeg; }
    virtual const gentype &bernIndex(void)  const { return bernind; }

    virtual int setBernDegree(const gentype &nv) { berndeg = nv; return 1; }
    virtual int setBernIndex(const gentype &nv)  { bernind = nv; return 1; }

    typedef int (*mexcallsyn)(gentype &, const gentype &, int);
    static mexcallsyn getsetExtVar;

    // Battery modelling parameters
    //
    // battparam: 21-d vector of battery parameters
    // batttmax: total simulation time (sec)
    // battImax: max charge/discharge current (amps)
    // batttdelta: time granunaliry (sec)
    // battVstart: start voltage (V)
    // battthetaStart: start temperature (deg)
    // battneglectParasitic: neglect parasitic branch if set
    // battfixedTheta: if >-1000 then use this fixed theta

    virtual const Vector<double> &battparam(void)            const { return xbattParam;            }
    virtual const double         &batttmax(void)             const { return xbatttmax;             }
    virtual const double         &battImax(void)             const { return xbattImax;             }
    virtual const double         &batttdelta(void)           const { return xbatttdelta;           }
    virtual const double         &battVstart(void)           const { return xbattVstart;           }
    virtual const double         &battthetaStart(void)       const { return xbattthetaStart;       }
    virtual const int            &battneglectParasitic(void) const { return xbattneglectParasitic; }
    virtual const double         &battfixedTheta(void)       const { return xbattfixedTheta;       }

    virtual int setbattparam(const Vector<gentype> &nv)
    {
        Vector<double> nnv(xbattParam);

        NiceAssert( nv.size() == nnv.size() );

        int i;

        for ( i = 0 ; i < nv.size() ; i++ )
        {
            if ( !nv(i).isValNull() )
            {
                nnv("&",i) = (double) nv(i);
            }
        }

        xbattParam = nnv;

        return 1;
    }

    virtual int setbatttmax(double nv)          { xbatttmax             = nv; return 1; }
    virtual int setbattImax(double nv)          { xbattImax             = nv; return 1; }
    virtual int setbatttdelta(double nv)        { xbatttdelta           = nv; return 1; }
    virtual int setbattVstart(double nv)        { xbattVstart           = nv; return 1; }
    virtual int setbattthetaStart(double nv)    { xbattthetaStart       = nv; return 1; }
    virtual int setbattneglectParasitic(int nv) { xbattneglectParasitic = nv; return 1; }
    virtual int setbattfixedTheta(double nv)    { xbattfixedTheta       = nv; return 1; }

private:

    int xissample;

    int xmercachesize;
    int xmercachenorm;

    gentype doutfn;
    gentype outgrad; // only defined or calculated when required

    std::istream *xuseristream;
    std::ostream *xuserostream;

    gcallback xcallback;
    void *xcallbackfndata;

    std::string mexfn;
    int mexfnid;

    std::string sysfn;
    std::string xfname;
    std::string yfname;
    std::string xyfname;
    std::string yxfname;
    std::string rfname;

    // ML block averaging

    SparseVector<ML_Base *> xmlqlist;
    SparseVector<gentype> xmlqweight;

    // Kernel training

    Vector<double> KBlambda;

    // Bernstein

    gentype berndeg;
    gentype bernind;

    // Battery sims

    Vector<double> xbattParam;
    double xbatttmax;
    double xbattImax;
    double xbatttdelta;
    double xbattVstart;
    double xbattthetaStart;
    int xbattneglectParasitic;
    double xbattfixedTheta;

    BLK_Generic *thisthis;
    BLK_Generic **thisthisthis;
};

inline void qswap(BLK_Generic &a, BLK_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline BLK_Generic &setzero(BLK_Generic &a)
{
    a.restart();

    return a;
}

inline void BLK_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Generic &b = dynamic_cast<BLK_Generic &>(bb.getML());

    qswap(xissample,b.xissample);

    qswap(doutfn       ,b.doutfn       );
    qswap(xmercachesize,b.xmercachesize);
    qswap(xmercachenorm,b.xmercachenorm);

    std::istream *xistream;
    std::ostream *xostream;

    xistream = xuseristream; xuseristream = b.xuseristream; b.xuseristream = xistream;
    xostream = xuserostream; xuserostream = b.xuserostream; b.xuserostream = xostream;

    gcallback qcallback;
    void *qcallbackfndata;

    qcallback       = xcallback;       xcallback       = b.xcallback;       b.xcallback       = qcallback;
    qcallbackfndata = xcallbackfndata; xcallbackfndata = b.xcallbackfndata; b.xcallbackfndata = qcallbackfndata;

    qswap(mexfn  ,b.mexfn  );
    qswap(mexfnid,b.mexfnid);

    qswap(sysfn  ,b.sysfn  );
    qswap(xfname ,b.xfname );
    qswap(yfname ,b.yfname );
    qswap(xyfname,b.xyfname);
    qswap(yxfname,b.yxfname);
    qswap(rfname ,b.rfname );

    qswap(KBlambda,b.KBlambda);

    qswap(xmlqlist  ,b.xmlqlist  );
    qswap(xmlqweight,b.xmlqweight);

    qswap(berndeg,b.berndeg);
    qswap(bernind,b.bernind);

    qswap(xbattParam           ,b.xbattParam           );
    qswap(xbatttmax            ,b.xbatttmax            );
    qswap(xbattImax            ,b.xbattImax            );
    qswap(xbatttdelta          ,b.xbatttdelta          );
    qswap(xbattVstart          ,b.xbattVstart          );
    qswap(xbattthetaStart      ,b.xbattthetaStart      );
    qswap(xbattneglectParasitic,b.xbattneglectParasitic);
    qswap(xbattfixedTheta      ,b.xbattfixedTheta      );

    ML_Base::qswapinternal(b);

    return;
}

inline void BLK_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Generic &b = dynamic_cast<const BLK_Generic &>(bb.getMLconst());

    xissample = b.xissample;

    doutfn        = b.doutfn;
    xmercachesize = b.xmercachesize;
    xmercachenorm = b.xmercachenorm;

    xuseristream = b.xuseristream;
    xuserostream = b.xuserostream;

    xcallback       = b.xcallback;
    xcallbackfndata = b.xcallbackfndata;

    mexfn   = b.mexfn;
    mexfnid = b.mexfnid;

    sysfn   = b.sysfn;
    xfname  = b.xfname;
    yfname  = b.yfname;
    xyfname = b.xyfname;
    yxfname = b.yxfname;
    rfname  = b.rfname;

    KBlambda = b.KBlambda;

    xmlqlist   = b.xmlqlist;
    xmlqweight = b.xmlqweight;

    berndeg = b.berndeg;
    bernind = b.bernind;

    xbattParam            = b.xbattParam;
    xbatttmax             = b.xbatttmax;
    xbattImax             = b.xbattImax;
    xbatttdelta           = b.xbatttdelta;
    xbattVstart           = b.xbattVstart;
    xbattthetaStart       = b.xbattthetaStart;
    xbattneglectParasitic = b.xbattneglectParasitic;
    xbattfixedTheta       = b.xbattfixedTheta;

    ML_Base::semicopy(b);

    return;
}

inline void BLK_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Generic &src = dynamic_cast<const BLK_Generic &>(bb.getMLconst());

    xissample = src.xissample;

    doutfn        = src.doutfn;
    xmercachesize = src.xmercachesize;
    xmercachenorm = src.xmercachenorm;

    xuseristream = src.xuseristream;
    xuserostream = src.xuserostream;

    xcallback       = src.xcallback;
    xcallbackfndata = src.xcallbackfndata;

    mexfn   = src.mexfn;
    mexfnid = src.mexfnid;

    sysfn   = src.sysfn;
    xfname  = src.xfname;
    yfname  = src.yfname;
    xyfname = src.xyfname;
    yxfname = src.yxfname;
    rfname  = src.rfname;

    KBlambda = src.KBlambda;

    xmlqlist   = src.xmlqlist;
    xmlqweight = src.xmlqweight;

    berndeg = src.berndeg;
    bernind = src.bernind;

    xbattParam            = src.xbattParam;
    xbatttmax             = src.xbatttmax;
    xbattImax             = src.xbattImax;
    xbatttdelta           = src.xbatttdelta;
    xbattVstart           = src.xbattVstart;
    xbattthetaStart       = src.xbattthetaStart;
    xbattneglectParasitic = src.xbattneglectParasitic;
    xbattfixedTheta       = src.xbattfixedTheta;

    ML_Base::assign(src,onlySemiCopy);

    return;
}

#endif
