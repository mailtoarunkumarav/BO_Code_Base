
//
// SVMHeavyv6 abstracted interface
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _mlinter_h
#define _mlinter_h

#include <iostream>
#include <fstream>
#include "ml_base.h"
#include "mlcommon.h"
#include "ml_mutable.h"
#include "gentype.h"
#include "ofiletype.h"
#include "vecstack.h"
#include "awarestream.h"



class SVMThreadContext;


// Core function
// =============
//
// runsvm: This is the core of svmheavy.  Given various arguments it interprets
//         commands given and runs operations on the given SVM (svmbase) as
//         requested.  This is abstracted so that it can be called recursively
//         (eg in gridsearch).
//
// threadInd  - thread index (used during multi-threaded operation)
// svmContext - running data for all threads.  svmContext(thread) contains
//              data for this thread.
// commStack  - input (command) stream for this thread.
// globalvariables - data shared between all threads
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
//               - if num == -4 then returns 0 if this function actually
//                 does anything (eg if mex is present) or -1 if this is
//                 just a dummy function that does nothing.
//               - returns 0 on success, -1 on failure.
//
// Return value:
//
// 0:       commStack exhausted, ML still running
// -1:      -ZZZZ encoutered
// 1-99:    syntax error
// 101-199: file error
// 201-299: thread error
// 301-399: unknown throw

int runsvm(int threadInd,
           SparseVector<SVMThreadContext *> &svmContext,
           SparseVector<ML_Mutable *> &svmbase,
           SparseVector<int> &svmThreadOwner,
           Stack<awarestream *> *commstack,
           svmvolatile SparseVector<SparseVector<gentype> > &globargvariables,
           int (*getsetExtVar)(gentype &res, const gentype &src, int num),
           SparseVector<SparseVector<int> > &returntag);

// Kill all threads, including main (0) thread if killmain is set.

void killallthreads(SparseVector<SVMThreadContext *> &svmContext, int killmain = 0);

// Delete all MLs

void deleteMLs(SparseVector<ML_Mutable *> &svmbase);


const gentype &getmacro(int i);

class SVMThreadContext;

inline SVMThreadContext *&setzero(SVMThreadContext *&x);
inline void qswap(SVMThreadContext *&a, SVMThreadContext *&b);
inline ML_Mutable *&setzero(ML_Mutable *&x);

inline SVMThreadContext *&setident (SVMThreadContext *&a) { throw("Whatever"); return a; }
inline SVMThreadContext *&setposate(SVMThreadContext *&a) { return a; }
inline SVMThreadContext *&setnegate(SVMThreadContext *&a) { throw("I reject your reality and substitute my own"); return a; }
inline SVMThreadContext *&setconj  (SVMThreadContext *&a) { throw("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
inline SVMThreadContext *&setrand  (SVMThreadContext *&a) { throw("Blippity Blappity Blue"); return a; }
inline SVMThreadContext *&postProInnerProd(SVMThreadContext *&x);


class SVMThreadContext
{
public:
    SVMThreadContext(int xsvmInd = 1, int xthreadInd = -1)
    {
        verblevel        = 1;
        finalresult      = 0.0;
        svmInd           = xsvmInd;
        biasdefault      = 0.0;
        filevariables.zero();
        xtemplate.zero();
        depthin          = 0;
        bgTrainOn        = 0;
        logfile          = "";
        binaryRelabel    = 0;
        singleDrop       = 0;
        updateargvars    = 1;
        killmethod       = 1;
        killswitch       = 0;
        controlThreadInd = xthreadInd;

        argvariables("&",1)("&",45) = svmInd;

        argvariables("&",130)("&",0)  = getmacro(0);
        argvariables("&",130)("&",1)  = getmacro(1);
        argvariables("&",130)("&",2)  = getmacro(2);
        argvariables("&",130)("&",3)  = getmacro(3);
        argvariables("&",130)("&",4)  = getmacro(4);
        argvariables("&",130)("&",5)  = getmacro(5);
        argvariables("&",130)("&",6)  = getmacro(6);
        argvariables("&",130)("&",7)  = getmacro(7);
        argvariables("&",130)("&",8)  = getmacro(8);
        argvariables("&",130)("&",9)  = getmacro(9);
        argvariables("&",130)("&",10) = getmacro(10);
        argvariables("&",130)("&",11) = getmacro(11);

        return;
    }

    SVMThreadContext(const SVMThreadContext &src, int xthreadInd = -1)
    {
        *this = src;

        controlThreadInd = xthreadInd;

        return;
    }

    SVMThreadContext &operator=(const SVMThreadContext &src)
    {
        verblevel        = src.verblevel;
        finalresult      = src.finalresult;
        svmInd           = src.svmInd;
        biasdefault      = src.biasdefault;
        argvariables     = src.argvariables;
        filevariables    = src.filevariables;
        xtemplate        = src.xtemplate;
        depthin          = 0;
        bgTrainOn        = 0;
        logfile          = "";
        binaryRelabel    = src.binaryRelabel;
        singleDrop       = src.singleDrop;
        updateargvars    = src.updateargvars;
        killmethod       = 1;
        MLindstack       = src.MLindstack;
        killswitch       = 0;
        controlThreadInd = -1;

        return *this;
    }

    // verblevel:     verbosity level when writing to logfile.
    //                0: minimal - only send feedback to errstream()
    //                1: normal - above, plus write details to logfiles
    // finalresult:   during the test phase of a given command block, the result
    //                of the last test run (as defined by the test ordering, not
    //                the order of commands given) is written to this variable.
    //                Generally speaking this is a measure of error, so lower
    //                results mean better performance, though the exact meaning
    //                depends on the SVM type and the resfilter (if any) used.  This
    //                argument is used by gridsearch.
    // svmInd:        decides which ML in svmbase is currently being operated on.
    // biasdefault:   default bias used by the SVM, for use when needed
    // argvariables:  this sparse matrix contains all relevant running variables
    //                for the SVM, which can be directly accessed in commands as
    //                described in the help section at the end.  This is included
    //                as an argument as it is kept and used during gridsearch
    //                recursion.
    // filevariables: for data processing it is possible to extract data from files
    //                at multiple stages in the training and testing process.
    //                Files opened for this purpose are stored in this vector.
    //                (the type ofiletype keeps track of which data entries from
    //                this file have been taken (used) and which are still available
    //                for future extraction).
    // xtemplate:     boilerplate (default) parts of all x vectors
    // depthin:       when recursing (for example) records depth.  1 for top layer
    // bgTrainOn:     0 for normal, 1 to enable background training.
    // logfile:       name of logfile
    // binaryRelabel: see addData.h
    // singleDrop:    see addData.h
    // killmethod:    when attempting to capture an ML for this thread:
    //                0: wait patiently for other thread(s) to finish
    //                1: set killswitch on other threads to terminate them early
    // MLindstack:    used to keep indices when pushing/popping MLs
    // 
    // killswitch:       set this 1 to tell thread to stop.
    // controlThreadInd: if thread is running then this is set to the threadInd of the
    //                   controlling thread.  Otherwise it is set -1.
    // updateargvars:    set 1 if argvariables need to be updated, zero otherwise

    int verblevel;
    gentype finalresult;
    int svmInd;
    gentype biasdefault;
    SparseVector<SparseVector<gentype> > argvariables;
    SparseVector<ofiletype> filevariables;
    SparseVector<gentype> xtemplate;
    int depthin;
    int bgTrainOn;
    std::string logfile;
    int binaryRelabel;
    int singleDrop;
    int updateargvars;
    int killmethod;
    Stack<int> MLindstack;
    svmvolatile int killswitch;
    svmvolatile int controlThreadInd;
};

inline SVMThreadContext *&setzero(SVMThreadContext *&x)
{
    return ( x = NULL ); // Must be deleted elsewhere
}

inline void qswap(SVMThreadContext *&a, SVMThreadContext *&b)
{
    SVMThreadContext *c;

    c = a; a = b; b = c;

    return;
}

inline ML_Mutable *&setzero(ML_Mutable *&x)
{
    x = NULL;

    return x;
}

#endif
