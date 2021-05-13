
// Uncomment this for matlab pre 2019a
#define MEX2019

//THREADS: get/set mex functions should only work for main thread
//THREADS: may need to change in line with svmheavyv7.cc
//THREADS: need way to return list of active threads and list of active MLs and which MLs are currently owned by active threads
//THREADS: move isMainThread function to basefn and use in kbquitdet (only main thread can be interupted and interact in general)


//#define STYPE int
#define STYPE size_t

//
// SVMHeavyv7 Matlab CLI-like Interface
//
// Version: 7
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// Usage: svmmatlab('commands'), where commands are just like the regular CLI
//

#include <string>
#include <sstream>
#include "mlinter.h"
#include "FNVector.h"
#include "mex.h"


// Refresh on streams occurs when any of the following triggers:
//
// - every REFRESHRATE characters
// - every REFRESHTIME seconds
// - whenever a \n is detected
//
// Counters/timers reset on refresh.

#define REFRESHRATE 1024
#define REFRESHTIME 4

#define DEFAULT_OUTPRT 1
//#define DEFAULT_ERRPRT 1
#define DEFAULT_ERRPRT 0

// Print state: 0 means no output
//              1 means standard buffered output
//              2 means no buffering
//
// To set call with desired state as argument.
// Will return previous state (before current change).

int mexAllowPrintOut(int mod = -1);
int mexAllowPrintErr(int mod = -1);

// Stream flush

void mexFlushOut(void);
void mexFlushErr(void);

// Character print for errstream, outstream, and character receive from instream

void mexCharPrintOut(char c);
void mexCharPrintErr(char c);
char mexCharRead(void);

// If these are NZ then streams are also diverted to logfiles on disk (which is helpful for
// when matlab innevitably crashes)

#define LOGOUTTOFILE 1
#define LOGERRTOFILE 1

// Functions to divert character streams to logfiles (if enabled - see above)

void mexPrintToOutLog(char c, int mode = 0);
void mexPrintToErrLog(char c, int mode = 0);

// Mex callback function
//
// mexgetsetExtVar: static store for external variables passed in from matlab
// call with numhand >= 0 to set fnhand pointer, numhand = -1 will return fnhand
// pointer and set numhand to previously assigned value (and ignore xfnhand),
// numhand = -2 will reset to original state.

int mexgetsetExtVar(gentype &res, const gentype &src, int num);
mxArray **getsetfnhand(mxArray **xfnhand, int &numhand);

// Control-c register function

#ifndef MEX2019

#ifdef _MSC_VER
    #pragma comment(lib, "libut.lib")
#endif

#ifdef __cplusplus 
    extern "C" bool utIsInterruptPending();
#else
    extern bool utIsInterruptPending();
#endif

int kbIntDet(void);
int kbIntDet(void)
{
    return utIsInterruptPending() ? 1 : 0;
}

#endif












// Matlab function

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    static int hasbeeninit = 0;
    static int persistenceset = 0;
    static int persistencereq = 0;

    isMainThread(1);

    try
    {
        int argc = nrhs+1;
        std::string commline;

        // Initialisation of static, overall state, set-once type, streamy stuff

        if ( !hasbeeninit )
        {
            outstream() << "Initialising statics and globals...\n";

            initgentype();
            setintercalc(&intercalc);
            #ifndef MEX2019
            setkbcallback(&kbIntDet);
            #endif
            //disablekbquitdet();

            void(*xmexCharPrintErr)(char c) = mexCharPrintErr;
            static LoggingOstream mexcerr(xmexCharPrintErr);
            seterrstream(&mexcerr);

            void(*xmexCharPrintOut)(char c) = mexCharPrintOut;
            static LoggingOstreamOut mexcout(xmexCharPrintOut);
            setoutstream(&mexcout);

            char(*xmexCharRead)(void) = mexCharRead;
            static LoggingIstream mexcin(xmexCharRead);
            setinstream(&mexcin);
        }

        // Print help if no commands given, or too many
        //
        // Assumption: either no arguments or just one

        if ( ( argc == 1 ) || ( nlhs > 0 ) )
        {
        instruct_short:

            outstream() << "SVMheavy 7.0                                                                  \n";
            outstream() << "============                                                                  \n";
            outstream() << "                                                                              \n";
            outstream() << "Copyright: all rights reserved.                                               \n";
            outstream() << "Author: Alistair Shilton                                                      \n";
            outstream() << "                                                                              \n";
            outstream() << "Standard use:  svmmatlab('commands')                                          \n";
            outstream() << "Basic help:    svmmatlab('-?')                                                \n";
            outstream() << "Advanced help: svmmatlab('-??')                                               \n";
            outstream() << "                                                                              \n";
            outstream() << "Persistence: by default calls to svmmatlab are independent of each other. This\n";
            outstream() << "can be changed by turning on  persistence.  When persistence is on all MLs are\n";
            outstream() << "retained in memory between calls, allowing multiple operations on the ML.  For\n";
            outstream() << "example this can be used to test different parameter settings, or retain a    \n";
            outstream() << "trained ML in memory for use.                                                 \n";
            outstream() << "                                                                              \n";
            outstream() << "Turn off persistence: svmmatlab(0)                                            \n";
            outstream() << "Turn on persistence:  svmmatlab(1)                                            \n";
            outstream() << "                                                                              \n";
            outstream() << "Output: by default  both std::cerr and std::cout are  redirected to matlab and\n";
            outstream() << "printed with appropriate prefix (err: or out:).  If persistence is on this can\n";
            outstream() << "be modified as follows:                                                       \n";
            outstream() << "                                                                              \n";
            outstream() << "Turn on std::cerr redirect:  svmmatlab(2)                                     \n";
            outstream() << "Turn off std::cerr redirect: svmmatlab(-2)                                    \n";
            outstream() << "Turn on std::cout redirect:  svmmatlab(3)                                     \n";
            outstream() << "Turn off std::cout redirect: svmmatlab(-3)                                    \n";
            outstream() << "                                                                              \n";
            outstream() << "Multiple MLs: multiple  MLs can  run simultaneously  (see -?? for  details, in\n";
            outstream() << "particular  the  -q...  commands) in  parallel.  To  simplify  operation  when\n";
            outstream() << "multiple MLs are  present an optional  second argument may  be used to specify\n";
            outstream() << "which ML is being addressed by the  command string.  The syntax for this is as\n";
            outstream() << "follows:                                                                      \n";
            outstream() << "                                                                              \n";
            outstream() << "svmmatlab('commands',mlnum)                                                   \n";
            outstream() << "                                                                              \n";
            outstream() << "which runs  the command \"-qw mlnum -Zx\"  before executing the  commands given.\n";
            outstream() << "Use -1 to leave number unchanged.                                             \n";
            outstream() << "                                                                              \n";
            outstream() << "Function handles: to pass function handles to code use the syntax:            \n";
            outstream() << "                                                                              \n";
            outstream() << "svmmatlab('commands',mlnum{,fn1{,fn2{...}}})                                  \n";
            outstream() << "                                                                              \n";
            outstream() << "The functions are assumed to take a single (vector) argument and return a     \n";
            outstream() << "single (vector) result.                                                       \n";

            mexFlushOut();

            return;
        }

        if ( ( argc == 2 ) && mxIsNumeric(prhs[0]) )
        {
            if ( ( mxGetNumberOfElements(prhs[0]) != 1 ) || mxIsComplex(prhs[0]) ) 
            {
                goto instruct_short;
            }

            double temp = mxGetScalar(prhs[0]);

            if ( temp != (int) temp )
            {
                goto instruct_short;
            }

            int tempb = (int) temp;

                 if ( tempb == 0  ) { persistencereq   = 0; }
            else if ( tempb == 1  ) { persistencereq   = 1; }
            else if ( tempb == 2  ) { mexAllowPrintErr(1); }
            else if ( tempb == -2 ) { mexAllowPrintErr(0); }
            else if ( tempb == 3  ) { mexAllowPrintOut(1); }
            else if ( tempb == -3 ) { mexAllowPrintOut(0); }

            argc = 1;
        }

        if ( argc > 3 )
        {
            // Grab function handles

            int numhand = argc-3;
            mxArray **fnhand = new mxArray *[numhand];

            int i;

            for ( i = 0 ; i < numhand ; i++ )
            {
                fnhand[i] = const_cast<mxArray *>(prhs[i+2]);
            }

            getsetfnhand(fnhand,numhand);

            argc = 3;
        }

        if ( argc == 3 )
        {
            // input 0 must be a string

            if ( mxIsChar(prhs[0]) != 1 )
            {
                 goto instruct_short;
            }

            if ( mxGetM(prhs[0]) !=1 )
            {
                 goto instruct_short;
            }

            // input 1 must be a number

            if ( !mxIsNumeric(prhs[1]) )
            {
                goto instruct_short;
            }

            if ( ( mxGetNumberOfElements(prhs[1]) != 1 ) || mxIsComplex(prhs[1]) ) 
            {
                goto instruct_short;
            }

            double temp = mxGetScalar(prhs[1]);

            if ( ( temp < -1 ) && ( temp != (int) temp ) )
            {
                goto instruct_short;
            }

            // Add prefix to command

            int mlnum = (int) temp;

            if ( mlnum >= 0 )
            {
                std::ostringstream oss;

                oss << mlnum;

                commline += "-qw ";
                commline += oss.str();
                commline += " -Zx ";
            }

            argc = 2;
        }

        // If currently not persistent and persistence requested then turn on

        if ( !persistenceset && persistencereq )
        {
            outstream() << "Locking ML stack...\n";

            mexLock();

            persistenceset = 1;
        }

        // Convert single argument to string

        char *input_buf;
        size_t buflen;

        if ( argc == 2 )
        {
            // input must be a string

            if ( mxIsChar(prhs[0]) != 1 )
            {
                 goto instruct_short;
            }

            // input must be a row vector

            if ( mxGetM(prhs[0]) !=1 )
            {
                 goto instruct_short;
            }

            // Get size of input string    

            buflen = (mxGetM(prhs[0])*mxGetN(prhs[0]))+1;

            // Copy command string

            input_buf = mxArrayToString(prhs[0]);
    
            if( !input_buf )
            {
                 goto instruct_short;
            }
        }

        // Make it as C-like, main as possible

        char *argv[2];
        char commname[] = "svmmatlab";
        argv[0] = commname;
        argv[1] = input_buf;

        // Convert the command line arguments into a command string

//        std::string commline;

        int i;
        int isquote;

        if ( argc > 1 )
        {
            for ( i = 1 ; i < argc ; i++ )
            {
                // NB: if string contains spaces but is not enclosed in quotes then
                // dos is being a pita and stripping quotes, so reinstate them.  This
                // will not fix the problem of quotes being stripped from a string
                // not containing spaces, so you need to be wary of that later.  Nor
                // will it fix double-quoted strings

                isquote = 0;

/*
                if ( strlen(argv[i]) )
                {
                    if ( ( argv[i][0] != '\"' ) || ( argv[i][strlen(argv[i])-1] != '\"' ) )
                    {
                        for ( j = 0 ; j < strlen(argv[i]) ; j++ )
                        {
                            if ( argv[i][j] == ' ' )
                            {
                                isquote = 1;
                                break;
                            }
                        }
                    }
                }
*/

                if ( isquote )
                {
                    commline += '\"';
                    commline += argv[i];
                    commline += '\"';
                }

                else
                {
                    commline += argv[i];
                }

                if ( i < argc-1 )
                {
                    commline += " ";
                }
            }
        }

        // Add -Zx to the end of the command string to ensure that the output
        // stream used by -echo will remain available until the end.

        commline += " -Zx";
        outstream() << "Running command: " << commline << "\n";

        // Define global variable store

        static svmvolatile SparseVector<SparseVector<gentype> > globargvariables;

        // Construct command stack.  All commands must be in awarestream, which
        // is similar to a regular stream but can supply commands from a
        // variety of different sources: for example a string (as here), a stream
        // such as standard input, or various ports etc.  You can then open
        // further awarestreams, which are stored on the stack, with the uppermost
        // stream being the active stream from which current commands are sourced.

        Stack<awarestream *> *commstack = new Stack<awarestream *>;
        std::stringstream *commlinestring = new std::stringstream(commline);
        awarestream *commlinestringbox = new awarestream(commlinestring,1);
        commstack->push(commlinestringbox);

        // Threaded data.  Each ML is an element in svmContext, with threadInd
        // specifying which is currently in use.  At this point we only have
        // a single ML with index 0.

        static int threadInd = 0;
        static int svmInd = 0;
        static SparseVector<SVMThreadContext *> svmContext;
        static SparseVector<int> svmThreadOwner;
        static SparseVector<ML_Mutable *> svmbase;
        svmContext("[]",threadInd) = new SVMThreadContext(svmInd,threadInd);
        outstream() << "{";

        // Now that everything has been set up so we can run the actual code.

        SparseVector<SparseVector<int> > returntag;

        runsvm(threadInd,svmContext,svmbase,svmThreadOwner,commstack,globargvariables,mexgetsetExtVar,returntag);

        // Unlock the thread, signalling that the context can be deleted etc

        outstream() << "}\n";

        delete commstack;

        // If currently persistent and persistence not requested then turn off

        if ( persistenceset && !persistencereq )
        {
            outstream() << "Unlocking ML stack...\n";

            mexUnlock();

            persistenceset = 0;
        }

        // Delete everything if not persistent

        int numhandreset = -2;

        getsetfnhand(NULL,numhandreset);

        if ( !persistenceset )
        {
            outstream() << "Removing ML stack...\n";

            // Delete the thread SVM context and remove from vector.

            killallthreads(svmContext,1);

            deleteMLs(svmbase);

            hasbeeninit = 0;
        }

        else
        {
            hasbeeninit = 1;
        }
    }

    catch ( const char *errcode )
    {
        outstream() << "Unknown error: " << errcode << ".\n";
        return;
    }

    catch ( const std::string errcode )
    {
        outstream() << "Unknown error: " << errcode << ".\n";
        return;
    }

    // Clear the buffer

    mexFlushOut();
    mexFlushErr();

    if ( !hasbeeninit )
    {
        mexPrintToOutLog('*',1);
        mexPrintToErrLog('*',1);
    }

    isMainThread(0);

    return;
}





























// Take a gentype object and create a copy that is an mxArray
//
// Returns NULL on failure (actually just throws, null never reached)

mxArray *createmxArray(const gentype &src);

// Take an mxArray and place contents into gentype object
//
// return 0 on success, nz on failure

int retrievemxArray(gentype &dst, const mxArray *src);

int mexgetsetExtVar(gentype &res, const gentype &src, int num)
{
    // Grab function handles and count

    mxArray **fnhand = NULL;
    int numhand = -1;

    fnhand = getsetfnhand(fnhand,numhand);

    // Branch depending on function

    int ires = 0;
    int i;

    if ( num >= numhand )
    {
        return -1;
    }

    else if ( num >= 0 )
    {
        mxArray *matdat = fnhand[num];

        // - if num >= 0 then loads extvar num into res.  If extvar is
        //   a function handle then src acts as an argument (optional,
        //   not used if null, multiple arguments if set).

        if ( mxIsClass(matdat,"function_handle") )
        {
            // Dealing with function handle

            int numargs = src.isValNull() ? 0 : ( src.isValSet() ? src.size() : 1 );

            // Set up arguments

            mxArray **rhs = new mxArray *[1+numargs];

            rhs[0] = fnhand[num];

            if ( src.isValSet() )
            {
                if ( numargs )
                {
                    for ( i = 0 ; i < numargs ; i++ )
                    {
                        rhs[i+1] = createmxArray((src.cast_set().all())(i));

                        if ( !rhs[i+1] )
                        {
                            return -2;
                        }
                    }
                }
            }

            else if ( !src.isValNull() )
            {
                rhs[1] = createmxArray(src);

                if ( !rhs[1] )
                {
                    return -3;
                }
            }

            // Set up return value

            mxArray *lhs[1];

            // Evaluate function

            ires = mexCallMATLAB(1,lhs,1+numargs,rhs,"feval");

            // Grab result

            if ( !ires )
            {
                ires = retrievemxArray(res,lhs[0]);
            }

            // Clear memory (but not rhs[0])

            if ( numargs )
            {
                for ( i = 0 ; i < numargs ; i++ )
                {
                    mxDestroyArray(rhs[i+1]);
                }
            }

            delete[] rhs;

            mxDestroyArray(lhs[0]);
        }

        else
        {
            // Dealing with data - make sure type follows src

            res = src;

            ires = retrievemxArray(res,matdat);
        }
    }

    else if ( num == -1 )
    {
        // - if num == -1 then loads external variable named in res
        //   (res must be string) into res before returning.  In this
        //   case src gives preferred type if result interpretation is
        //   ambiguous (type of res will attempt to copy gentype of
        //   src).

        mxArray *Xdat = mexGetVariable("base",(res.cast_string()).c_str());

        if ( Xdat )
        {
            res = src;

            ires = retrievemxArray(res,Xdat);

            mxDestroyArray(Xdat);
        }

        else
        {
            ires = -42;
        }
    }

    else if ( num == -2 )
    {
        // - if num == -2 then loads contents of src into external
        //   variable named in res before returning.

        mxArray *Xdat = createmxArray(src);

        if ( Xdat )
        {
            ires = mexPutVariable("base",(res.cast_string()).c_str(),Xdat);

            mxDestroyArray(Xdat);
        }

        else
        {
            ires = -52;
        }
    }

    else if ( num == -3 )
    {
        // - if num == -3 then evaluates fn(v) where fn is a matlab
        //   function named by res, v is the set of arguments (see
        //   num >= 0) and the result is stored in res.
        //
        // Basically the num >= 0 function handle case but with named function

        int numargs = src.isValNull() ? 0 : ( src.isValSet() ? src.size() : 1 );

        // Set up arguments

        mxArray **rhs = new mxArray *[numargs ? numargs : 1];

        if ( src.isValSet() )
        {
            if ( numargs )
            {
                for ( i = 0 ; i < numargs ; i++ )
                {
                    rhs[i] = createmxArray((src.cast_set().all())(i));

                    if ( !rhs[i] )
                    {
                        return -100;
                    }
                }
            }
        }

        else if ( !src.isValNull() )
        {
            rhs[0] = createmxArray(src);

            if ( !rhs[0] )
            {
                return -101;
            }
        }

        // Set up return value

        mxArray *lhs[1];

        // Evaluate function

        ires = mexCallMATLAB(1,lhs,numargs,rhs,(res.cast_string()).c_str());

        // Grab result

        if ( !ires )
        {
            ires = retrievemxArray(res,lhs[0]);
        }

        // Clear memory

        if ( numargs )
        {
            for ( i = 0 ; i < numargs ; i++ )
            {
                mxDestroyArray(rhs[i]);
            }
        }

        delete[] rhs;

        mxDestroyArray(lhs[0]);
    }

    else if ( num == -4 )
    {
        ires = 0;
    }

    else
    {
        ires = -1;
    }

    return ires;
}

mxArray *createmxArray(const gentype &src)
{
    int i,j,k,l;

    mxArray *res = NULL;

    if ( src.isValNull() )
    {
        res = mxCreateDoubleMatrix(0,0,mxREAL);
    }

    else if ( src.isValInteger() || src.isValReal() || ( src.isValAnion() && ( src.order() == 0 ) ) )
    {
        const double &srccnt = src.cast_double();

        res = mxCreateDoubleScalar(srccnt);
    }

    else if ( src.isValAnion() && ( src.order() == 1 ) )
    {
        const d_anion &srccnt = src.cast_anion();

        res = mxCreateDoubleMatrix(1,1,mxCOMPLEX);

        double *dstr = mxGetPr(res);
        double *dsti = mxGetPi(res);

        if ( dstr && dsti )
        {
            *dstr = srccnt(0);
            *dsti = srccnt(1);
        }
    }

    else if ( src.isValVector() && !src.infsize() )
    {
        const Vector<gentype> &srccnt = src.cast_vector();

        if ( srccnt.size() == 0 )
        {
            res = mxCreateDoubleMatrix(0,1,mxREAL);
        }

        else
        {
            // Need to first work out if the contents can be fitted.

            int isReal    = 0;
            int isComplex = 0;
            int comOrder  = 0;
            int isVector  = 0;
            int vecSize   = 0;
            int isMatrix  = 0;
            int matRows   = 0;
            int matCols   = 0;
            int isBad     = 0;

            for ( i = 0 ; i < srccnt.size() ; i++ )
            {
                if ( srccnt(i).isValInteger() || srccnt(i).isValReal() )
                {
                    if ( isVector || isMatrix )
                    {
                        isBad = 1;
                    }

                    else if ( !isComplex )
                    {
                        isReal = 1;
                    }
                }

                else if ( srccnt(i).isValAnion() )
                {
                    if ( isVector || isMatrix || ( isComplex && ( comOrder != srccnt(i).order() ) ) )
                    {
                        isBad = 1;
                    }

                    isReal    = 0;
                    isComplex = 1;
                    comOrder  = srccnt(i).order();
                }

                else if ( srccnt(i).isValVector() )
                {
                    if ( isReal || isComplex || isMatrix || ( isVector && ( srccnt(i).size() != vecSize ) ) )
                    {
                        isBad = 1;
                    }

                    isVector = 1;
                    vecSize  = srccnt(i).size();
                }

                else if ( srccnt(i).isValMatrix() )
                {
                    if ( isReal || isComplex || isVector || ( isMatrix && ( ( srccnt(i).numRows() != matRows ) || ( srccnt(i).numCols() != matCols ) ) ) )
                    {
                        isBad = 1;
                    }

                    isMatrix = 1;
                    matRows  = srccnt(i).numRows();
                    matCols  = srccnt(i).numCols();
                }

                else
                {
                    isBad = 1;
                }
            }

            int numRows = srccnt.size();
            int numCols = 1;

            // Check contents themselves are OK

            if ( !isBad && isComplex && ( comOrder > 1 ) )
            {
                isBad = 1;
            }

            else if ( !isBad && isVector )
            {
                for ( i = 0 ; i < numRows ; i++ )
                {
                    const Vector<gentype> &ival = srccnt(i).cast_vector();

                    for ( j = 0 ; j < vecSize ; j++ )
                    {
                        if ( !(ival(j).isValInteger()) && !(ival(j)).isValReal() )
                        {
                            isBad = 1;
                        }
                    }
                }
            }

            else if ( !isBad && isMatrix )
            {
                for ( i = 0 ; i < numRows ; i++ )
                {
                    const Matrix<gentype> &ival = srccnt(i).cast_matrix();

                    for ( k = 0 ; k < matRows ; k++ )
                    {
                        for ( l = 0 ; l < matCols ; l++ )
                        {
                            if ( !(ival(k,l).isValInteger()) && !(ival(k,l).isValReal()) )
                            {
                                isBad = 1;
                            }
                        }
                    }
                }
            }

            // Now make result

            if ( !isBad && isReal )
            {
                res = mxCreateDoubleMatrix(numRows,numCols,mxREAL);

                double *dstr = mxGetPr(res);

                if ( dstr )
                {
                    for ( i = 0 ; i < numRows ; i++ )
                    {
                        const double &ival = srccnt(i).cast_double();

                        dstr[i] = ival;
                    }
                }
            }

            else if ( !isBad && isComplex )
            {
                res = mxCreateDoubleMatrix(numRows,numCols,mxCOMPLEX);

                double *dstr = mxGetPr(res);
                double *dsti = mxGetPi(res);

                if ( dstr && dsti )
                {
                    for ( i = 0 ; i < numRows ; i++ )
                    {
                        const d_anion &ival = srccnt(i).cast_anion();

                        dstr[i] = ival(0);
                        dsti[i] = ival(1);
                    }
                }
            }

            else if ( !isBad && isVector )
            {
// Design decision: originally I had this write a 3-d array.  However that is really
// annoying to deal with in matlab, so this just transposes as a matrix instead

                res = mxCreateDoubleMatrix(numRows,vecSize,mxREAL);

                double *dstr = mxGetPr(res);

                for ( i = 0 ; i < numRows ; i++ )
                {
                    for ( j = 0 ; j < vecSize ; j++ )
                    {
                        const double &ival = ((srccnt(i).cast_vector())(j)).cast_double();

                        dstr[i+(j*numRows)] = ival;
                    }
                }

/*
                int dims[3];

                dims[0] = numRows;
                dims[1] = numCols;
                dims[2] = vecSize;

                res = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);

                double *dstr = mxGetPr(res);

                if ( dstr )
                {
                    for ( i = 0 ; i < numRows ; i++ )
                    {
                        const Vector<gentype> &ival = srccnt(i).cast_vector();

                        j = 0;
    
                        for ( k = 0 ; k < vecSize ; k++ )
                        {
                            dstr[i+(j*numRows)+(k*numRows*numCols)] = (double) ival(k);
                        }
                    }
                }
*/
            }

            else if ( !isBad && isMatrix )
            {
                STYPE dims[4];

                dims[0] = numRows;
                dims[1] = numCols;
                dims[2] = matRows;
                dims[3] = matCols;

                res = mxCreateNumericArray(4,dims,mxDOUBLE_CLASS,mxREAL);

                double *dstr = mxGetPr(res);

                if ( dstr )
                {
                    for ( i = 0 ; i < numRows ; i++ )
                    {
                        const Matrix<gentype> &ival = srccnt(i).cast_matrix();

                        j = 0;

                        for ( k = 0 ; k < matRows ; k++ )
                        {
                            for ( l = 0 ; l < matCols ; l++ )
                            {
                                dstr[i+(j*numRows)+(k*numRows*numCols)+(l*numRows*numCols*matRows)] = (double) ival(k,l);
                            }
                        }
                    }
                }
            }
        }
    }

    else if ( src.isValVector() && src.infsize() )
    {
        retVector<gentype> tmpva; 

        // The following lines grab the (infinite) vector, casts to FuncVector to change definition of operator(), then grabs precalculated vector form (after first unsafeSampling if it doesn't yet exist)

        const Vector<gentype> &srca = src.cast_vector();  // Extract vector from gentype wrapper
        const Vector<gentype> &srcb = srca(tmpva);  // Get the "real" vector (in case imoverhere is set)
        const FuncVector &srcc = dynamic_cast<const FuncVector &>(srcb);  // Cast it to functional vector (possible based on the observation src.infsize())
        const Vector<gentype> &srcd = srcc(tmpva); // Sample (unless already sampled) and get resultant results in vector (*not* infinite size)
        const gentype srce(srcd); // Re-wrap result as gentype for further processing - the only step that allocates new memory
        res = createmxArray(srce); // matlab variable can be extracted from srce
    }

    else if ( src.isValMatrix() )
    {
        const Matrix<gentype> &srccnt = src.cast_matrix();

        if ( ( srccnt.numRows() == 0 ) || ( srccnt.numCols() == 0 ) )
        {
            res = mxCreateDoubleMatrix(srccnt.numRows(),srccnt.numCols(),mxREAL);
        }

        else
        {
            // Need to first work out if the contents can be fitted.

            int isReal    = 0;
            int isComplex = 0;
            int comOrder  = 0;
            int isVector  = 0;
            int vecSize   = 0;
            int isMatrix  = 0;
            int matRows   = 0;
            int matCols   = 0;
            int isBad     = 0;

            for ( i = 0 ; i < srccnt.numRows() ; i++ )
            {
              for ( j = 0 ; j < srccnt.numCols() ; j++ )
              {
                if ( srccnt(i,j).isValInteger() || srccnt(i,j).isValReal() )
                {
                    if ( isVector || isMatrix )
                    {
                        isBad = 1;
                    }

                    else if ( !isComplex )
                    {
                        isReal = 1;
                    }
                }

                else if ( srccnt(i,j).isValAnion() )
                {
                    if ( isVector || isMatrix || ( isComplex && ( comOrder != srccnt(i,j).order() ) ) )
                    {
                        isBad = 1;
                    }

                    isReal    = 0;
                    isComplex = 1;
                    comOrder  = srccnt(i,j).order();
                }

                else if ( srccnt(i,j).isValVector() )
                {
                    if ( isReal || isComplex || isMatrix || ( isVector && ( srccnt(i,j).size() != vecSize ) ) )
                    {
                        isBad = 1;
                    }

                    isVector = 1;
                    vecSize  = srccnt(i,j).size();
                }

                else if ( srccnt(i,j).isValMatrix() )
                {
                    if ( isReal || isComplex || isVector || ( isMatrix && ( ( srccnt(i,j).numRows() != matRows ) || ( srccnt(i,j).numCols() != matCols ) ) ) )
                    {
                        isBad = 1;
                    }

                    isMatrix = 1;
                    matRows  = srccnt(i,j).numRows();
                    matCols  = srccnt(i,j).numCols();
                }

                else
                {
                    isBad = 1;
                }
              }
            }

            int numRows = srccnt.numRows();
            int numCols = srccnt.numCols();

            // Check contents themselves are OK

            if ( !isBad && isComplex && ( comOrder > 1 ) )
            {
                isBad = 1;
            }

            else if ( !isBad && isVector )
            {
                for ( i = 0 ; i < numRows ; i++ )
                {
                  for ( j = 0 ; j < numCols ; j++ )
                  {
                    const Vector<gentype> &ival = srccnt(i,j).cast_vector();

                    for ( k = 0 ; k < vecSize ; k++ )
                    {
                        if ( !(ival(k).isValInteger()) && !(ival(k)).isValReal() )
                        {
                            isBad = 1;
                        }
                    }
                  }
                }
            }

            else if ( !isBad && isMatrix )
            {
                for ( i = 0 ; i < numRows ; i++ )
                {
                  for ( j = 0 ; j < numRows ; j++ )
                  {
                    const Matrix<gentype> &ival = srccnt(i,j).cast_matrix();

                    for ( k = 0 ; k < matRows ; k++ )
                    {
                        for ( l = 0 ; l < matCols ; l++ )
                        {
                            if ( !(ival(k,l).isValInteger()) && !(ival(k,l).isValReal()) )
                            {
                                isBad = 1;
                            }
                        }
                    }
                  }
                }
            }

            if ( !isBad && isReal )
            {
                res = mxCreateDoubleMatrix(numRows,numCols,mxREAL);

                double *dstr = mxGetPr(res);

                for ( i = 0 ; i < numRows ; i++ )
                {
                    for ( j = 0 ; j < numCols ; j++ )
                    {
                        const double &ival = srccnt(i,j).cast_double();

                        dstr[i+(j*numRows)] = ival;
                    }
                }
            }

            else if ( !isBad && isComplex )
            {
                res = mxCreateDoubleMatrix(numRows,numCols,mxCOMPLEX);

                double *dstr = mxGetPr(res);
                double *dsti = mxGetPi(res);

                for ( i = 0 ; i < numRows ; i++ )
                {
                    for ( j = 0 ; j < numCols ; j++ )
                    {
                        const d_anion &ival = srccnt(i,j).cast_anion();

                        dstr[i+(j*numRows)] = ival(0);
                        dsti[i+(j*numRows)] = ival(1);
                     }
                }
            }

            else if ( !isBad && isVector )
            {
                STYPE dims[3];

                dims[0] = numRows;
                dims[1] = numCols;
                dims[2] = vecSize;

                res = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);

                double *dstr = mxGetPr(res);

                for ( i = 0 ; i < numRows ; i++ )
                {
                    for ( j = 0 ; j < numCols ; j++ )
                    {
                        const Vector<gentype> &ival = srccnt(i,j).cast_vector();

                        for ( k = 0 ; k < vecSize ; k++ )
                        {
                            dstr[i+(j*numRows)+(k*numRows*numCols)] = (double) ival(k);
                        }
                    }
                }
            }

            else if ( !isBad && isMatrix )
            {
                STYPE dims[4];

                dims[0] = numRows;
                dims[1] = numCols;
                dims[2] = matRows;
                dims[3] = matCols;

                res = mxCreateNumericArray(4,dims,mxDOUBLE_CLASS,mxREAL);

                double *dstr = mxGetPr(res);

                for ( i = 0 ; i < numRows ; i++ )
                {
                    for ( j = 0 ; j < numCols ; j++ )
                    {
                        const Matrix<gentype> &ival = srccnt(i,j).cast_matrix();

                        for ( k = 0 ; k < matRows ; k++ )
                        {
                            for ( l = 0 ; l < matCols ; l++ )
                            {
                                dstr[i+(j*numRows)+(k*numRows*numCols)+(l*numRows*numCols*matRows)] = (double) ival(k,l);
                            }
                        }
                    }
                }
            }
        }
    }

    else if ( src.scalarfn_isscalarfn() && ( src.scalarfn_i().size() == 1 ) && ( src.scalarfn_j().size() == 1 ) )
    {
        int numpts = src.scalarfn_numpts();
        int ix = src.scalarfn_i()(0);
        int jx = src.scalarfn_j()(0);

        gentype altsrc;
        Vector<gentype> &srccont = altsrc.force_vector(numpts);
        SparseVector<SparseVector<gentype> > x;

        for ( i = 0 ; i < numpts ; i++ )
        {
            x("&",ix)("&",jx) = ((double) i)/((double) numpts-1);
            srccont("&",i) = src(x);
            srccont("&",i).finalise(3);
            srccont("&",i).finalise(2);
            srccont("&",i).finalise(1);
            srccont("&",i).finalise();
        }

        res = createmxArray(altsrc);
    }

    else if ( src.isValString() )
    {
        res = mxCreateString((src.cast_string()).c_str());
    }

//    else if ( src.scalarfn_isscalarfn() )
//    {
//        throw("This is odd - scalar function should have a single argument");
//    }

    return res;
}

            
int retrievemxArray(gentype &dst, const mxArray *src)
{
    NiceAssert( src );

    int i,j; //,k,l;

    int res = 0;

    if ( mxIsNumeric(src) )
    {
        if ( ( mxGetNumberOfElements(src) == 0 ) && !dst.isValVector() && !dst.isValMatrix() )
        {
            // Null

            dst.force_null();
        }

        else if ( ( mxGetNumberOfDimensions(src) == 2 ) && ( mxGetM(src) == 1 ) && ( mxGetN(src) == 1 ) && !dst.isValVector() && !dst.isValMatrix() )
        {
            if ( !mxIsComplex(src) )
            {
                // Real number

                dst.force_double() = mxGetScalar(src);
            }

            else
            {
                // Complex number

                double *srcR = mxGetPr(src);
                double *srcI = mxGetPi(src);

                if ( srcR && srcI )
                {
                    dst.force_anion().setorder(1);

                    dst.dir_anion()("[]",0) = srcR[0];
                    dst.dir_anion()("[]",1) = srcI[0];
                }

                else
                {
                    dst.makeError("retrieve error");
                    res = -1;
                }
            }
        }

        else if ( ( mxGetNumberOfDimensions(src) == 2 ) && ( mxGetN(src) == 1 ) && !dst.isValMatrix() )
        {
            // Vector

            dst.force_vector().resize(mxGetM(src));

            if ( mxGetM(src) > 0 )
            {
                if ( !mxIsComplex(src) )
                {
                    // Real vector

                    double *srcR = mxGetPr(src);

                    if ( srcR )
                    {
                        for ( i = 0 ; i < mxGetM(src) ; i++ )
                        {
                            dst("[]",i).force_double() = srcR[i];
                        }
                    }

                    else
                    {
                        dst.makeError("retrieve error B");
                        res = -2;
                    }
                }

                else
                {
                    // Complex vector

                    double *srcR = mxGetPr(src);
                    double *srcI = mxGetPi(src);

                    if ( srcR && srcI )
                    {
                        for ( i = 0 ; i < mxGetM(src) ; i++ )
                        {
                            dst("[]",i).force_anion().setorder(1);

                            dst("[]",i).dir_anion()("[]",0) = srcR[i];
                            dst("[]",i).dir_anion()("[]",1) = srcI[i];
                        }
                    }

                    else
                    {
                        dst.makeError("retrieve error C");
                        res = -3;
                    }
                }
            }
        }

        else if ( ( mxGetNumberOfDimensions(src) == 2 ) && ( mxGetM(src) == 1 ) && !dst.isValMatrix() )
        {
            // Vector

            dst.force_vector().resize(mxGetN(src));

            if ( mxGetN(src) > 0 )
            {
                if ( !mxIsComplex(src) )
                {
                    // Real vector

                    double *srcR = mxGetPr(src);
 
                    if ( srcR )
                    {
                        for ( i = 0 ; i < mxGetN(src) ; i++ )
                        {
                            dst("[]",i).force_double() = srcR[i];
                        }
                    }

                    else
                    {
                        dst.makeError("retrieve error D");
                        res = -4;
                    }
                }

                else
                {
                    // Complex vector

                    double *srcR = mxGetPr(src);
                    double *srcI = mxGetPi(src);

                    if ( srcR && srcI )
                    {
                        for ( i = 0 ; i < mxGetN(src) ; i++ )
                        {
                            dst("[]",i).force_anion().setorder(1);

                            dst("[]",i).dir_anion()("[]",0) = srcR[i];
                            dst("[]",i).dir_anion()("[]",1) = srcI[i];
                        }
                    }

                    else
                    {
                        dst.makeError("retrieve error E");
                        res = -5;
                    }
                }
            }
        }

        else if ( mxGetNumberOfDimensions(src) == 2 )
        {
            // matrix

            dst.force_matrix().resize(mxGetM(src),mxGetN(src));

            if ( mxGetM(src)*mxGetN(src) > 0 )
            {
                if ( !mxIsComplex(src) )
                {
                    // Real vector

                    double *srcR = mxGetPr(src);

                    if ( srcR )
                    {
                        for ( i = 0 ; i < mxGetM(src) ; i++ )
                        {
                            for ( j = 0 ; j < mxGetN(src) ; j++ )
                            {
                                dst("[]",i,j).force_double() = srcR[i+(j*mxGetM(src))];
                            }
                        }
                    }

                    else
                    {
                        dst.makeError("retrieve error F");
                        res = -6;
                    }
                }

                else
                {
                    // Complex vector

                    double *srcR = mxGetPr(src);
                    double *srcI = mxGetPi(src);

                    if ( srcR && srcI )
                    {
                        for ( i = 0 ; i < mxGetM(src) ; i++ )
                        {
                            for ( j = 0 ; j < mxGetN(src) ; j++ )
                            {
                                dst("[]",i,j).force_anion().setorder(1);

                                dst("[]",i,j).dir_anion()("[]",0) = srcR[i+(j*mxGetM(src))];
                                dst("[]",i,j).dir_anion()("[]",1) = srcI[i+(j*mxGetM(src))];
                            }
                        }
                    }

                    else
                    {
                        dst.makeError("retrieve error G");
                        res = -7;
                    }
                }
            }
        }

        else
        {
            dst.makeError("retrieve error H");
            res = -8;
        }
    }

    else if ( mxIsClass(src,"char") )
    {
        char *stringVal = mxArrayToString(src);

        NiceAssert(stringVal);

        dst.makeString(stringVal);

        mxFree(stringVal);
    }

    else
    {
        dst.makeError("retrieve error I");
        res = -9;
    }

    return res;
}
















int mexAllowPrintOut(int mod)
{
    static int allowprint = DEFAULT_OUTPRT;

    int tempstat = allowprint;

    if ( mod == 0 ) { allowprint = 0; }
    if ( mod == 1 ) { allowprint = 1; }
    if ( mod == 2 ) { allowprint = 2; }

    return tempstat;
}

int mexAllowPrintErr(int mod)
{
    static int allowprint = DEFAULT_ERRPRT;

    int tempstat = allowprint;

    if ( mod == 0 ) { allowprint = 0; }
    if ( mod == 1 ) { allowprint = 1; }
    if ( mod == 2 ) { allowprint = 2; }

    return tempstat;
}

void mexFlushOut(void)
{
    int outstat = mexAllowPrintOut(2);

    outstream() << "\n";

    mexAllowPrintOut(outstat);

    return;
}

void mexFlushErr(void)
{
    int outstat = mexAllowPrintErr(2);

    errstream() << "\n";

    mexAllowPrintErr(outstat);

    return;
}

// Common destination for all streams
//
// mode = mexAllowPrint...()

void mexCharPrint(char c, int mode);
void mexCharPrint(char c, int mode)
{
    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    static std::string outbuff;
    static time_used begintime = TIMECALL;
    static int startline = 1;

    int printbuff = 0;

    if ( startline )
    {
        outbuff += "svmh: ";
        startline = 0;
    }

    if ( isMainThread() )
    {
        if ( c == '\0' )
        {
            printbuff = 1;
        }

        else
        {
            outbuff += c;
        }

        if ( c == '\n' )
        {
            startline = 1;
        }

        if ( mode == 2 )
        {
            printbuff = 1;
        }

        else if ( outbuff.length() >= REFRESHRATE )
        {
            printbuff = 1;
        }

        else
        {
            time_used endtime = TIMECALL;
            double timewait = TIMEDIFFSEC(endtime,begintime);

            if ( timewait > REFRESHTIME )
            {
                printbuff = 1;
            }
        }

        if ( printbuff )
        {
            mexPrintf(outbuff.c_str());
            mexEvalString("pause(.001);");
            mexEvalString("drawnow");
            begintime = TIMECALL;

            outbuff = "";
        }
    }

    svm_mutex_unlock(eyelock);

    return;
}

void mexCharPrintOut(char c)
{
    mexPrintToOutLog(c);

    int mode = mexAllowPrintOut();

    if ( mode )
    {
        mexCharPrint(c,mode);
    }

    return;
}

void mexCharPrintErr(char c)
{
    mexPrintToErrLog(c);

    int mode = mexAllowPrintErr();

    if ( mode )
    {
        mexCharPrint(c,mode);
    }

    return;
}

char mexCharRead(void)
{
    char res = '\0';

    if ( isMainThread() )
    {
        static std::string inbuffer;

        // If buffer is empty then grab a line of text from Matlab

        while ( inbuffer.length() == 0 )
        {
            mxArray *outmat;
            mxArray *str[2];

            str[0] = mxCreateString("");
            str[1] = mxCreateString("s");

            mexCallMATLAB(1,&outmat,2,str,"input");

            char *outstr = mxArrayToString(outmat);

            inbuffer += outstr;
            inbuffer += "\n";

            mxDestroyArray(str[0]);
            mxDestroyArray(str[1]);
        }

        // Record the first character in the buffer, remove it from the buffer and return it

        res = inbuffer[0];

        inbuffer.erase(0,1);
    }

    return res;
}

void mexPrintToOutLog(char c, int mode)
{
    // mode = 0: print char
    //        1: close file for exit

    if ( LOGOUTTOFILE )
    {
        static std::ofstream *outlog = NULL;

        if ( !mode && !outlog )
        {
            outlog = new std::ofstream;

            NiceAssert(outlog);

            std::string outfname("svmheavy.out.log");
            std::string outfnamebase("svmheavy.out.log");

            int fcnt = 1;

            while ( fileExists(outfname) )
            {
                fcnt++;

                std::stringstream ss;

                ss << outfnamebase;
                ss << ".";
                ss << fcnt;

                outfname = ss.str();
            }

            (*outlog).open(outfname);
        }

        if ( mode )
        {
            if ( outlog )
            {
                (*outlog).close();
                delete outlog;
                outlog = NULL;
            }
        }

        else if ( outlog )
        {
            (*outlog) << c;
            (*outlog).flush();
        }
    }    

    return;
}

void mexPrintToErrLog(char c, int mode)
{
    // mode = 0: print char
    //        1: close file for exit

    if ( LOGERRTOFILE )
    {
        static std::ofstream *errlog = NULL;

        if ( !mode && !errlog )
        {
            errlog = new std::ofstream;

            NiceAssert(errlog);

            std::string errfname("svmheavy.err.log");
            std::string errfnamebase("svmheavy.err.log");

            int fcnt = 1;

            while ( fileExists(errfname) )
            {
                fcnt++;

                std::stringstream ss;

                ss << errfnamebase;
                ss << ".";
                ss << fcnt;

                errfname = ss.str();
            }

            (*errlog).open(errfname);
        }

        if ( mode )
        {
            if ( errlog )
            {
                (*errlog).close();
                delete errlog;
                errlog = NULL;
            }
        }

        else
        {
            (*errlog) << c;
            (*errlog).flush();
        }
    }    

    return;
}




mxArray **getsetfnhand(mxArray **xfnhand, int &xnumhand)
{
    static mxArray **fnhand = NULL;
    static int numhand = 0;

    if ( xnumhand >= 0 )
    {
        fnhand  = xfnhand;
        numhand = xnumhand;
    }

    else if ( xnumhand == -2 )
    {
        fnhand = NULL;
        numhand = 0;
    }

    xnumhand = numhand;

    return fnhand;
}
