//TO DO: now that pthread_t has been generalised to svm_pthread_t, do this:
//
// - add unique ID to svm_pthread_t
// - record these IDs to a global variable (linked list?)
// - have option in interactive mode to list active threads, pause, kill, edit variables etc
//
// see for example
//
// void pthread_cleanup_pop(int execute);
// void pthread_cleanup_push(void (*routine)(void*), void *arg); [Option End]

//
// Miscellaneous stuff
// (aka a bunch of random code and ugly hacks gathered together in one place)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _basefn_h
#define _basefn_h

#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <string.h>
#include <string>
#include <time.h>
#include <math.h>
#include <limits.h>



// =======================================================================
//
// Comment these out for different compilation systems.
//
// DJGPP_MATHS:    bessel functions available from DJGPP maths library (and
//                 some other stuff)
// ALLOW_SOCKETS:  sockets are used in awarestream for TCP and UDP streams.
// ENABLE_THREADS: threads are used for background-training and mutexes.
//                 (mutexes are still supported without this, but non-posix)
// CYGWIN_BUILD:   uncomment for cygwin.  Locates various libaries in
//                 different places.  Only required for threads, specifies
//                 location of un.h header file.
// CYGWIN10:       cygwin, in windows 10, has two oddities - no usleep 
//                 function (so use nanosleep instead) and abs *is* defined
//                 for doubles (latter no longer relevant since moving
//                 everything to abs2).
// VISUAL_STU:     visual studio compile.  Disables various features taken
//                 from unistd.h, redefines certain things (eg inf number
//                 macros), allows for "interesting" variations in the maths
//                 library.
// VISUAL_STU_OLD: still more modifications for older versions of visual stu.
// VISUAL_STU_NOERF: because MS 2012 doesn't have erf... obviously :roll:
// HAVE_CONIO:     conio.h is available, so use this for some functions.
// HAVE_TERMIOS:   termios.h is available, so use this for some functions.
// DEBUG_MEM:      new / delete have extra debugging
// DISABLE_KB_BY_DEF: define to disable interactive keyboard by default
// IS_CPP11:       is compiler c++11 compatible
//
// =======================================================================

// =======================================================================
//
// Multithreading note: there is nothing in the C++ standard about whether
// initialisation of static local variables is threadsafe.  The initialisation
// occurs the first time that a segment of code is reached, but if two threads
// hit the static variable at the same time then *in theory* both will call
// the initialisation function.
//
// In practice, gcc does put locks around static variable initialisation, so
// this won't be an issue in unix environments (though I can't speak for clang).
// However visual c++ does not put locks around the same code, so there could
// be an issue here.
//
// UPDATE: for c++11 static local variable initialisation is threadsafe.  If
// the variable is being initialised in one thread and a second thread reaches
// the point where it may initialise it then instead it will wait until the
// first thread has finished initialising and then use the variable so
// initialised.  I assume visual now follows this behaviour (gcc always has).
//
//
// ***************************
// MULTITHREADING AND GENTYPE:
// ***************************
//
// Multithreaded initialisation function: initgentype
//
// Initialises all derivatives in one block.  This is not required for single
// threaded operation, but for multithreaded use call this function first
// before starting any new threads.
//
// =======================================================================

#ifndef SPECIFYSYSVIAMAKE

// dos/djgpp

// #define DJGPP_MATHS
// #define HAVE_CONIO







// cygwin/gcc

// #define ALLOW_SOCKETS
// #define ENABLE_THREADS
// #define CYGWIN_BUILD
// #define HAVE_TERMIOS

// cygwin/gcc modern

// #define ALLOW_SOCKETS
// #define ENABLE_THREADS
// #define CYGWIN_BUILD
// #define HAVE_TERMIOS
// #define IS_CPP11





// linux/gcc

// #define ALLOW_SOCKETS
// #define ENABLE_THREADS
// #define HAVE_TERMIOS

// linux/gcc modern

// #define ALLOW_SOCKETS
// #define ENABLE_THREADS
// #define HAVE_TERMIOS
// #define IS_CPP11

// Linux/gcc unthreaded unthreaded

// #define ALLOW_SOCKETS
// #define ENABLE_THREADS
// #define HAVE_TERMIOS




// Visual Studio

//#define VISUAL_STU
//#define VISUAL_STU_OLD
//#define VISUAL_STU_NOERF
//#define HAVE_CONIO
//#define IGNOREMEM
//#define ALLOW_SOCKETS
//#define _CRT_SECURE_NO_WARNINGS 1

// Visual Studio modern

//#define VISUAL_STU
//#define VISUAL_STUDIO_BESSEL
//#define VISUAL_STU_NOERF
//#define HAVE_CONIO
//#define IGNOREMEM
//#define ALLOW_SOCKETS
//#define ENABLE_THREADS
//#define IS_CPP11
//#define _CRT_SECURE_NO_WARNINGS 1
//#pragma warning(disable:4996)
//#pragma warning(disable:4244)
//#pragma warning(disable:4756)





// Mex old

//#define USE_MEX
//#define VISUAL_STU
//#define VISUAL_STU_NOERF
//#define HAVE_CONIO
//#define IGNOREMEM
//#define ALLOW_SOCKETS




// Mex modern

#define USE_MEX
#define VISUAL_STU
#define VISUAL_STU_NOERF
#define HAVE_CONIO
//#define NDEBUG
#define IS_CPP11

// Uncomment to enable threads (currently not supported)

//#define ENABLE_THREADS

// Comment to debug

//#define ALLOW_SOCKETS
#define IGNOREMEM

// Uncomment to debug memory

//#define DEBUG_MEM




#endif
























// This is handy if available, but not strictly necessary

#ifdef IS_CPP11
#define svm_explicit explicit
#endif
#ifndef IS_CPP11
#define svm_explicit
#endif









// =======================================================================
//
// Alternative optimiser options
// USE_HOPDM:  Use system call to hopdm for linear optimisation rather than
// the default internal optimisation routine
//
// =======================================================================

// #define USE_HOPDM














//#define ALLOW_SOCKETS
//#define ENABLE_THREADS
//#define CYGWIN_BUILD
//#define VISUAL_STU
//#define VISUAL_STU_OLD
//#define USE_MEX







#ifndef VISUAL_STU
#include <unistd.h>
#ifdef IS_CPP11
#include <sys/select.h>
#endif
#endif














// spoilers...

std::ostream &errstream(int i = 0);






// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Replacement for atexit

int svm_atexit(void (*func)(void), const char *desc);

// The above function uses atexit by default (with some modifications).  This
// is not suitable in some environments (eg mex).  The following function 
// can be used to replace atexit (the standard library function
// used to define functions that will be run on exit) with an alternative
// such as mexAtExit.  Returns the atexit function.  If xfn = NULL then will
// keep current function.  Default is atexit.
//
// Note that this must be called prior to *ALL OTHER FUNCTIONS*.

typedef int (*atexitfn)(void (*)(void));

atexitfn svm_setatexitfn(atexitfn xfn = NULL);

















// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// System command
//
// svm_execall: Call executable.  If runbg then attempt to leave it running in the background
// svm_pycall: Call python script.  If runbg then attempt to leave it running in the background

#include <stdlib.h>

inline int svm_system(const char *command);
inline int svm_system(const char *command)
{
    return system(command);
}


int svm_execall(const std::string &command, int runbg);
int svm_pycall(const std::string &command, int runbg);
















// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// NiceAssert: Nicer assert macro.  This one will throw an exception that 
//             can be caught.
// STRTHROW: construct std::string and throw this.
//
// MEMNEW{ARRAY} macro version of new to allow memory debugging
// MEMDEL{ARRAY} macro version of delete to allow memory debugging

#ifdef NDEBUG
#define NiceAssert( cond )
#endif

#ifndef NDEBUG
#define S(x) #x
#define S_(x) S(x)
#define S__LINE__ S_(__LINE__)
#define THROWSTRINGDEF(cond) "Assertion " #cond " failed at line " S__LINE__ " in file " __FILE__
#define NiceAssert( cond ) \
if ( !(cond) ) \
{ \
errstream() << THROWSTRINGDEF(cond) << "\n"; \
    throw(THROWSTRINGDEF(cond)); \
}
#define QuietAssert( cond ) \
if ( !(cond) ) \
{ \
errstream() << THROWSTRINGDEF(cond) << "\n"; \
}
#endif

#define STRTHROW(__errstr__) \
std::string __errstring__ = __errstr__; \
throw __errstring__;


#ifndef DEBUG_MEM
#ifndef DEBUG_MEM_CHEAP

#define MEMNEW(_a_,_b_) _a_ = new _b_
#define MEMNEWARRAY(_a_,_b_,_c_) _a_ = new _b_[_c_]

#define MEMNEWVOID(_a_,_b_) _a_ = (void *) new _b_
#define MEMNEWVOIDARRAY(_a_,_b_,_c_) _a_ = (void *) new _b_[_c_]

#define MEMDEL(_a_) delete _a_
#define MEMDELARRAY(_a_) delete[] _a_

#define MEMDELVOID(_a_) delete _a_
#define MEMDELVOIDARRAY(_a_) delete[] _a_

#endif
#endif

#ifndef DEBUG_MEM
#ifdef DEBUG_MEM_CHEAP

extern unsigned long long _global_alloccnt;
extern unsigned long long _global_maxalloccnt;

#define MEMCUTPNT 16384

#define MEMNEW(_a_,_b_) { _global_alloccnt++; if ( _global_alloccnt/MEMCUTPNT > _global_maxalloccnt ) { _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; errstream() << "!" << _global_maxalloccnt << "!"; } } _a_ = new _b_
#define MEMNEWARRAY(_a_,_b_,_c_) { if ( _global_alloccnt/MEMCUTPNT > _global_maxalloccnt ) { _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; errstream() << "!" << _global_maxalloccnt << "!"; } } _a_ = new _b_[_c_]

#define MEMNEWVOID(_a_,_b_) { _global_alloccnt++; if ( _global_alloccnt/MEMCUTPNT > _global_maxalloccnt ) { _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; errstream() << "!" << _global_maxalloccnt << "!"; } } _a_ = (void *) new _b_
#define MEMNEWVOIDARRAY(_a_,_b_,_c_) { _global_alloccnt++; if ( _global_alloccnt/MEMCUTPNT > _global_maxalloccnt ) { _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; errstream() << "!" << _global_maxalloccnt << "!"; } } _a_ = (void *) new _b_[_c_]

#define MEMDEL(_a_) { if ( _global_alloccnt > 0 ) { _global_alloccnt--; } if ( _global_alloccnt/MEMCUTPNT < _global_maxalloccnt ) { _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; errstream() << "?_" << _global_maxalloccnt << "_?"; } } delete _a_
#define MEMDELARRAY(_a_) { if ( _global_alloccnt > 0 ) { _global_alloccnt--; } if ( _global_alloccnt/MEMCUTPNT < _global_maxalloccnt ) { _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; errstream() << "?_" << _global_maxalloccnt << "_?"; } } delete[] _a_

#define MEMDELVOID(_a_) { if ( _global_alloccnt > 0 ) { _global_alloccnt--; } if ( _global_alloccnt/MEMCUTPNT < _global_maxalloccnt ) { _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; errstream() << "?_" << _global_maxalloccnt << "_?"; } } delete _a_
#define MEMDELVOIDARRAY(_a_) { if ( _global_alloccnt > 0 ) { _global_alloccnt--; } if ( _global_alloccnt/MEMCUTPNT < _global_maxalloccnt ) { _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; errstream() << "?_" << _global_maxalloccnt << "_?"; } } delete[] _a_

#endif
#endif




// Function that keeps track of pointers:
//
// addr is pointer in question
// newdel: 1 if new, 0 of delete
// type: 0 is pointer, 1 is array
// size: size of array
//
// return: 0 = success
//         1 = attempt to delete an unallocated error code
//         2 = attempt to delete array with non-array delete
//         3 = attempt to delete non-array with array delete

int addremoveptr(void *addr, int newdel, int type, int size, const char *desc);

#ifdef DEBUG_MEM

#define xS(x) #x
#define xS_(x) xS(x)
#define xS__LINE__ xS_(__LINE__)
#define PLACEMARK "Line " xS__LINE__ " in file " __FILE__


#define MEMNEW(_a_,_b_) \
{ \
    _a_ = new _b_; \
\
    NiceAssert( _a_ ); \
\
    int _qq_ = addremoveptr((void *) _a_,1,0,1,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
}

#define MEMNEWARRAY(_a_,_b_,_c_) \
{ \
    _a_ = new _b_[_c_]; \
\
    NiceAssert( _a_ ); \
\
    int _qq_ = addremoveptr((void *) _a_,1,1,_c_,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
}

#define MEMNEWVOID(_a_,_b_) \
{ \
    _a_ = (void *) new _b_; \
\
    NiceAssert( _a_ ); \
\
    int _qq_ = addremoveptr((void *) _a_,1,0,1,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
}

#define MEMNEWVOIDARRAY(_a_,_b_,_c_) \
{ \
    _a_ = (void *) new _b_[_c_]; \
\
    QuietAssert( _a_ ); \
\
    int _qq_ = addremoveptr((void *) _a_,1,1,_c_,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
}

#define MEMDEL(_a_) \
{ \
  QuietAssert( _a_ ); \
\
  if ( _a_ ) \
  { \
    int _qq_ = addremoveptr((void *) _a_,0,0,0,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    if ( !_qq_ ) \
    { \
        delete _a_; \
    } \
  } \
}

#define MEMDELARRAY(_a_) \
{ \
  QuietAssert( _a_ ); \
\
  if ( _a_ ) \
  { \
    int _qq_ = addremoveptr((void *) _a_,0,1,0,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    if ( !_qq_ ) \
    { \
        delete[] _a_; \
    } \
 } \
}

//    _a_ = NULL;

#define MEMDELVOID(_a_) \
{ \
  QuietAssert( _a_ ); \
\
  if ( _a_ ) \
  { \
    int _qq_ = addremoveptr((void *) _a_,0,0,0,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    if ( !_qq_ ) \
    { \
        delete _a_; \
    } \
  } \
}

#define MEMDELVOIDARRAY(_a_) \
{ \
  QuietAssert( _a_ ); \
\
  if ( _a_ ) \
  { \
    int _qq_ = addremoveptr((void *) _a_,0,1,0,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    if ( !_qq_ ) \
    { \
        delete[] _a_; \
    } \
  } \
}

#endif






































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Stuff in linux but not windows

#ifndef STDIN_FILENO
#define STDIN_FILENO 1
#endif

#ifndef STDOUT_FILENO
#define STDOUT_FILENO 0
#endif


































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Sockets stuff

// --- If sockets available include relevant libraries ---

#ifdef ALLOW_SOCKETS

#ifdef VISUAL_STU
#include <windows.h>
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
// windows doesn't define this, but inferring from final argument of recvfrom
#define socklen_t int
//class sockaddr_un;
//class sockaddr_un
//{
//    public:
//
//    int sun_family;
//    char *sun_path;
//};
#define UNIX_PATH_MAX 256
#define SHUT_RDWR     2
struct sockaddr_un
{
    int sun_family;
    char *sun_path;

    sockaddr_un()
    {
        sun_family = 0;
        MEMNEWARRAY(sun_path,char,UNIX_PATH_MAX+1);
        sun_path[0] = '\0';
        return;
    }

    ~sockaddr_un()
    {
        MEMDELARRAY(sun_path);
        return;
    }
};
//#define SHUT_RDWR     SD_BOTH
inline int close(int a);
inline int close(int a) { return closesocket(a); }
#pragma comment(lib, "Ws2_32.lib")
#endif

#ifndef VISUAL_STU
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#ifdef CYGWIN_BUILD
#include <sys/un.h>
#endif

#ifndef CYGWIN_BUILD
#ifndef VISUAL_STU
#include <linux/un.h>
#endif
#endif

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#define UDPBUFFERLEN 1024

// Alias everything

#define SVM_SOCK_STREAM   SOCK_STREAM
#define SVM_SOCK_DGRAM    SOCK_DGRAM
#define SVM_MAX_RETRIES   MAX_RETRIES
#define SVM_AF_INET       AF_INET
#define SVM_AF_UNIX       AF_UNIX
#define SVM_UNIX_PATH_MAX UNIX_PATH_MAX
#define SVM_INADDR_ANY    INADDR_ANY
#define SVM_SHUT_WR       SHUT_WR
#define SVM_SHUT_RDWR     SHUT_RDWR
#define SVM_UDPBUFFERLEN  UDPBUFFERLEN
#define svm_socklen_t     socklen_t
#define svm_sockaddr_in   sockaddr_in
#define svm_sockaddr_un   sockaddr_un
#define svm_sockaddr      sockaddr

inline int svm_send(int a, const char *b, int c, int d);
inline int svm_send(int a, const char *b, int c, int d) { return send(a,b,c,d); }

inline int svm_recvfrom(int a, char *b, int c, int d, svm_sockaddr *e, svm_socklen_t *f);
inline int svm_recvfrom(int a, char *b, int c, int d, svm_sockaddr *e, svm_socklen_t *f) { return recvfrom(a,b,c,d,e,f); }

inline int svm_htons(int a);
inline int svm_htons(int a) { return htons(a); }

inline int svm_htonl(int a);
inline int svm_htonl(int a) { return htonl(a); }

inline int svm_inet_addr(const char *a);
inline int svm_inet_addr(const char *a) { return inet_addr(a); }

inline int svm_shutdown(int a, int b);
inline int svm_shutdown(int a, int b) { return shutdown(a,b); }

inline int svm_close(int a);
inline int svm_close(int a) { return close(a); }

inline int svm_socket(int a, int b, int c);
inline int svm_socket(int a, int b, int c) { return socket(a,b,c); }

inline int svm_bind(int a, svm_sockaddr *b, int c);
inline int svm_bind(int a, svm_sockaddr *b, int c) { return bind(a,b,c); }

inline int svm_accept(int a, svm_sockaddr *b, socklen_t *c);
inline int svm_accept(int a, svm_sockaddr *b, socklen_t *c) { return accept(a,b,c); }

inline int svm_connect(int a, svm_sockaddr *b, int c);
inline int svm_connect(int a, svm_sockaddr *b, int c) { return connect(a,b,c); }

inline int svm_listen(int a, int b);
inline int svm_listen(int a, int b) { return listen(a,b); }

#endif




// --- If sockets not possible define stubs and fake classes to allow ---
// --- compilation and return error codes if sockets used.            ---

#ifndef ALLOW_SOCKETS

#define SVM_SOCK_STREAM   0
#define SVM_SOCK_DGRAM    0
#define SVM_MAX_RETRIES   5
#define SVM_AF_INET       0
#define SVM_AF_UNIX       0
#define SVM_UNIX_PATH_MAX 0
#define SVM_INADDR_ANY    0
#define SVM_SHUT_WR       0
#define SVM_SHUT_RDWR     0
#define SVM_UDPBUFFERLEN  1024
#define svm_socklen_t     int

struct svm_saddr;
struct svm_saddr
{
    public:

    int s_addr;
    int S_un; // something windows uses apparently
};

struct svm_sockaddr_in;
struct svm_sockaddr_in
{
    public:

    int sin_family;
    int sin_port;
    struct svm_saddr sin_addr;
};

struct svm_sockaddr_un;
struct svm_sockaddr_un
{
    public:

    int sun_family;
    char *sun_path;
};

struct svm_sockaddr;
struct svm_sockaddr
{
    public:

    int sin_family;
    int sin_port;
    struct svm_saddr sin_addr;
};

inline int svm_send(int a, const char *b, int c, int d);
inline int svm_send(int a, const char *b, int c, int d) { (void) a; (void) b; (void) c; (void) d; return -1; }

inline int svm_recvfrom(int a, char *b, int c, int d, svm_sockaddr *e, unsigned int *f);
inline int svm_recvfrom(int a, char *b, int c, int d, svm_sockaddr *e, unsigned int *f) { (void) a; (void) b; (void) c; (void) d; (void) e; (void) f; return -1; }

inline int svm_htons(int a);
inline int svm_htons(int a) { (void) a; return -1; }

inline int svm_htonl(int a);
inline int svm_htonl(int a) { (void) a; return -1; }

inline int svm_inet_addr(const char *a);
inline int svm_inet_addr(const char *a) { (void) a; return -1; }

inline int svm_shutdown(int a, int b);
inline int svm_shutdown(int a, int b) { (void) a; (void) b; return -1; }

inline int svm_close(int a);
inline int svm_close(int a) { (void) a; return -1; }

inline int svm_socket(int a, int b, int c);
inline int svm_socket(int a, int b, int c) { (void) a; (void) b; (void) c; return -1; }

inline int svm_bind(int a, svm_sockaddr *b, int c);
inline int svm_bind(int a, svm_sockaddr *b, int c) { (void) a; (void) b; (void) c; return -1; }

inline int svm_accept(int a, svm_sockaddr *b, unsigned int *c);
inline int svm_accept(int a, svm_sockaddr *b, unsigned int *c) { (void) a; (void) b; (void) c; return -1; }

inline int svm_connect(int a, svm_sockaddr *b, int c);
inline int svm_connect(int a, svm_sockaddr *b, int c) { (void) a; (void) b; (void) c; return -1; }

inline int svm_listen(int a, int b);
inline int svm_listen(int a, int b) { (void) a; (void) b; return -1; }

#endif








































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// mutex and thread stuff
//
// mutex:   very basic mutex stuff, typically maps to c++11 or pthreads.
// threads: pthreads if available, or similar on windows (if available).
//
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// Mutexes:
//
// - data type is svm_mutex
// - svm_mutex_lock: locks the mutex if available, blocks if not
// - svm_mutex_trylock: if mutex available then locks it and returns true,
//   otherwise returns false
// - svm_mutex_unlock: unlocks mutex.  Mutexes must be unlocked by the
//   same thread that they were locked in.
//
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// Threads:
//
// - implements basic posix functions only.
//
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// threads_enables: returns nz if threads enabled
//
// isMainThread: Thread ID recorder
//
//     Call with val == +1 to set current thread as main
//     Call with val == 0  to test if current thread is main thread (nz is true)
//     Call with val == -1 to set no main thread (so isMainThread will always return true)
//     Call with val == +2 to see if there even is a main thread

inline int isMainThread(int val = 0);




// --- If threads available include relevant libraries and define ---
// --- various functions to allow there (easy) use.               ---

#define svmvolatile volatile

#ifdef ENABLE_THREADS

#ifndef VISUAL_STU
#include <pthread.h>
#endif

inline int threads_enabled(void);
inline int threads_enabled(void)
{
    return 1;
}

// ------------------------------------------------------------------------
// Mutexes
// ------------------------------------------------------------------------

#ifdef IS_CPP11
#include <mutex>

class svm_mutex : public std::mutex
{
public:

    svm_mutex() : std::mutex()
    {
        return;
    }
};

inline int svm_mutex_lock(svmvolatile svm_mutex &temp);
inline int svm_mutex_lock(svmvolatile svm_mutex &temp)
{
    const_cast<svm_mutex &>(temp).lock();
    return 0;
}

inline int svm_mutex_trylock(svmvolatile svm_mutex &temp);
inline int svm_mutex_trylock(svmvolatile svm_mutex &temp)
{
    const_cast<svm_mutex &>(temp).try_lock();
    return 0;
}

inline int svm_mutex_unlock(svmvolatile svm_mutex &temp);
inline int svm_mutex_unlock(svmvolatile svm_mutex &temp)
{
    const_cast<svm_mutex &>(temp).unlock();
    return 0;
}
#endif

// ------------------------------------------------------------------------

#ifndef IS_CPP11
#ifndef VISUAL_STU
#include <pthread.h>

class svm_mutex
{
public:

    svm_mutex()
    {
        int resinit = pthread_mutex_init  (const_cast<pthread_mutex_t *>(&actualMutex),NULL);
        (void) resinit;
        NiceAssert( !resinit );
        int resunlo = pthread_mutex_unlock(const_cast<pthread_mutex_t *>(&actualMutex)     );
        (void) resunlo;
        NiceAssert( !resunlo );

        return;
    }

    pthread_mutex_t actualMutex;
};

inline int svm_mutex_lock(svmvolatile svm_mutex &temp);
inline int svm_mutex_lock(svmvolatile svm_mutex &temp)
{
    return pthread_mutex_lock(const_cast<pthread_mutex_t *>(&(temp.actualMutex)));
}

inline int svm_mutex_trylock(svmvolatile svm_mutex &temp);
inline int svm_mutex_trylock(svmvolatile svm_mutex &temp)
{
    return pthread_mutex_trylock(const_cast<pthread_mutex_t *>(&(temp.actualMutex)));
}

inline int svm_mutex_unlock(svmvolatile svm_mutex &temp);
inline int svm_mutex_unlock(svmvolatile svm_mutex &temp)
{
    return pthread_mutex_unlock(const_cast<pthread_mutex_t *>(&(temp.actualMutex)));
}
#endif

// ------------------------------------------------------------------------

#ifdef VISUAL_STU
// Dubious fallback option
class svm_mutex;
class svm_mutex
{
    public:

    svm_mutex() { lockon = 0; }

    int lockon;
};

inline int svm_mutex_lock(svmvolatile svm_mutex &temp);
inline int svm_mutex_lock(svmvolatile svm_mutex &temp)
{
    while ( temp.lockon )
    {
        ;
    }

    temp.lockon = 1;

    return 0;
}

inline int svm_mutex_trylock(svmvolatile svm_mutex &temp);
inline int svm_mutex_trylock(svmvolatile svm_mutex &temp)
{
    int res = temp.lockon ? 0 : 1;
    temp.lockon = 1;

    return res;
}

inline int svm_mutex_unlock(svmvolatile svm_mutex &temp);
inline int svm_mutex_unlock(svmvolatile svm_mutex &temp)
{
    temp.lockon = 0;

    return 0;
}
#endif
#endif








// ------------------------------------------------------------------------
// Threads
// ------------------------------------------------------------------------

#ifndef VISUAL_STU
#include <pthread.h>

typedef pthread_t      svm_pthread_t;
typedef pthread_attr_t svm_pthread_attr_t;

// local aliases

inline int svm_pthread_create(svm_pthread_t *a, const svm_pthread_attr_t *b, void *(*c)(void *), void *d);
inline int svm_pthread_create(svm_pthread_t *a, const svm_pthread_attr_t *b, void *(*c)(void *), void *d)
{
    return pthread_create(a,b,c,d);
}

inline svm_pthread_t svm_pthread_self(void);
inline svm_pthread_t svm_pthread_self(void)
{
    return pthread_self();
}

inline int svm_pthread_equal(const svm_pthread_t &a, const svm_pthread_t &b);
inline int svm_pthread_equal(const svm_pthread_t &a, const svm_pthread_t &b)
{
    return pthread_equal(a,b);
}

#endif

// ------------------------------------------------------------------------

#ifdef VISUAL_STU
#include <windows.h>
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

class svm_pthread_t;
class svm_pthread_t
{
    public:

    HANDLE hWorkerThread = NULL;
    void *(*fn)(void *) = NULL;
    void *fnarg = NULL;
    DWORD thread_id = NULL;

    void callfn(void)
    {
        if ( fn )
        {
            (*fn)(fnarg);
        }

        return;
    }
};

class svm_pthread_attr_t;
class svm_pthread_attr_t
{
    public: // Never actually used in code, so no need to define

    int dummy;
};

DWORD WINAPI SVMThreadFunction(LPVOID lpParam);

inline int svm_pthread_create(svm_pthread_t *a, const svm_pthread_attr_t *b, void *(*c)(void *), void *d);
inline int svm_pthread_create(svm_pthread_t *a, const svm_pthread_attr_t *b, void *(*c)(void *), void *d)
{
    (void) b; // attributes ignored (actually not even defined here)

    MEMNEW(a,svm_pthread_t);

    if ( !a )
    {
        return -1;
    }

    a->fn = c;
    a->fnarg = d;

    a->hWorkerThread = CreateThread( 
            NULL,                   // default security attributes
            0,                      // use default stack size  
            SVMThreadFunction,      // thread function name
            (LPVOID) a,             // argument to thread function 
            0,                      // use default creation flags 
            &(a->thread_id));       // returns the thread identifier 

    if ( a->thread_id == NULL )
    {
        MEMDEL(a);
        a = NULL;
        return -1;
    }

    return 0;
}

// Only fills out the thread ID - only to be used with pthread_equal, nothing else
inline svm_pthread_t svm_pthread_self(void);
inline svm_pthread_t svm_pthread_self(void)
{
    pthread_t res;

    res.thread_id = GetCurrentThreadId();

    return res;
}

inline int svm_pthread_equal(const svm_pthread_t &a, const svm_pthread_t &b);
inline int svm_pthread_equal(const svm_pthread_t &a, const svm_pthread_t &b)
{
    return ( a.thread_id == b.thread_id );
}
#endif
#endif



















// --- If threads not present then define relevant stubs and funcions ---
// --- to enable compilation and return relevant error codes etc.     ---

#ifndef ENABLE_THREADS
class svm_mutex;
class svm_mutex
{
    public:

    svm_mutex() { lockon = 0; }

    int lockon;
};

inline int svm_mutex_lock(svmvolatile svm_mutex &temp);
inline int svm_mutex_lock(svmvolatile svm_mutex &temp)
{
    while ( temp.lockon )
    {
        ;
    }

    temp.lockon = 1;

    return 0;
}

inline int svm_mutex_trylock(svmvolatile svm_mutex &temp);
inline int svm_mutex_trylock(svmvolatile svm_mutex &temp)
{
    int res = temp.lockon ? 0 : 1;
    temp.lockon = 1;

    return res;
}

inline int svm_mutex_unlock(svmvolatile svm_mutex &temp);
inline int svm_mutex_unlock(svmvolatile svm_mutex &temp)
{
    temp.lockon = 0;

    return 0;
}

typedef int svm_pthread_t;
typedef int svm_pthread_attr_t;

inline int threads_enabled(void);
inline int threads_enabled(void) { return 0; }

inline int svm_pthread_create(svm_pthread_t *a, const svm_pthread_attr_t *b, void *(*c)(void *), void *d);
inline int svm_pthread_create(svm_pthread_t *a, const svm_pthread_attr_t *b, void *(*c)(void *), void *d) { (void) b; (void) c; (void) d; static svm_pthread_t onlythreadid = 0; *a = onlythreadid; return -1; }

inline svm_pthread_t svm_pthread_self(void);
inline svm_pthread_t svm_pthread_self(void) { svm_pthread_t res = 0; return res; }

inline int svm_pthread_equal(const svm_pthread_t &a, const svm_pthread_t &b); 
inline int svm_pthread_equal(const svm_pthread_t &a, const svm_pthread_t &b) { return ( a == b ); }

#endif




inline int isMainThread(int val)
{
    int res = 1;

    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    static int mainThreadSet = 0;
    static svm_pthread_t currthread;

    if ( val == 1 )
    {
        mainThreadSet = 1;
        currthread = svm_pthread_self();
    }

    else if ( val == -1 )
    {
        mainThreadSet = 0;
    }

    if ( mainThreadSet )
    {
        res = ( ( val == 2 ) || svm_pthread_equal(svm_pthread_self(),currthread) );
    }

    svm_mutex_unlock(eyelock);

    return res;
}


























// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Some maths

#ifdef IS_CPP11
#include <cmath>
#endif

// These are used by numbase to define missing functions when required

#ifndef VISUAL_STU
#ifndef IS_CPP11
#define NO_HYPOT
#endif
#endif

#ifndef VISUAL_STU
#ifndef CYGWIN10
#define NO_ABS
#endif
#endif

#ifdef VISUAL_STU
#ifndef IS_CPP11
#define NO_ACOSH_ASINH_ATANH
#endif
#endif

#ifdef VISUAL_STU_NOERF
#define NO_ERF
#endif

#ifdef NO_HYPOT
inline double hypot(double a, double b);
inline double hypot(double a, double b)
{
    double fabsa = ( a < 0 ) ? -a : a;
    double fabsb = ( b < 0 ) ? -b : b;
    double maxab = ( fabsa > fabsb ) ? fabsa : fabsb;
    double minab = ( fabsa > fabsb ) ? fabsb : fabsa;

    double res = 0.0;

    if ( maxab > 0.0 )
    {
        res = maxab*sqrt(1.0+((minab/maxab)*(minab*maxab)));
    }

    return res;
}
#endif


//#ifdef NO_ABS
//inline double abs(double a);
//inline double abs(double a)
//{
//    return fabs(a);
//}
//#endif

#ifdef NO_ACOSH_ASINH_ATANH
inline double asinh(double a);
inline double acosh(double a);
inline double atanh(double a);
inline double asinh(double a) { return log(a+sqrt((a*a)+1)); }
inline double acosh(double a) { return log(a+sqrt((a*a)-1)); }
inline double atanh(double a) { return log((1+a)/(1-a))/2;   }
#endif

#ifdef NO_ERF
inline double erf(double x);
inline double erf(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    // Save the sign of x
    int sign = ( x >= 0 ) ? 1 : -1;
    x = fabs(x);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return sign*y;
}
#endif

inline double erfc(double x);
inline double erfc(double x)
{
    return 1-erf(x);
}

// Various maths functions
//
// roundnearest: round to nearest integer,
// xnfact: factorial function
// xnCr: n choose r using algorithm designed to minimise chances of overflow
// ceilintlog2: ceil(log2(x)) for integers, return 0 if x = 0
// upidiv: integer division that rounds upwards, not downwards

inline int roundnearest(double x);
inline int xnfact(int i);
inline int xnCr(int n, int r);
inline unsigned int ceilintlog2(unsigned int x);
inline int upidiv(int i, int j);

// NaN and inf tests

inline int testisvnan(double x);
inline int testisinf(double x);
inline int testispinf(double x);
inline int testisninf(double x);

// Representations of nan, inf and -inf

inline double valvnan(void);
inline double valpinf(void);
inline double valninf(void);

// zeroint: returns zero.  This is useful when C++ can't decide of 0
// is an integer or a null pointer.

inline int zeroint(void);

// "Make nice" function for maths equations.  Converts E to e and
// fixes + and - to ensure that + only appears as addition, - only
// as unary negation, and no long strings of ++---.  Returns 0 on
// success, nonzero of failure.

int makeMathsStringNice(std::string &dest, const std::string &src);

// Kronecker-Delta function

inline int krondel(int i, int j);







inline int roundnearest(double x)
{
    // round x to nearest integer

    int res = (int) x; // floor
    double rx = x-res; // remainder

    // Initial      Round required      Corrected result
    // 
    // x = 1.1
    // res = 1           0                    1
    // rx = 0.1
    // 
    // x = 1.1
    // res = 2           -1                   1
    // rx = -0.9
    // 
    // x = 1.9
    // res = 1           +1                   2
    // rx = 0.9
    // 
    // x = 1.9
    // res = 2           0                    2
    // rx = -0.1
    // 
    // x = -1.1
    // res = -2          +1                   -1
    // rx = 0.9
    // 
    // x = -1.1
    // res = -1          0                    -1
    // rx = -0.1                              
    // 
    // x = -1.9
    // res = -1          -1                   -2
    // rx = 0.9
    // 
    // x = -1.9
    // res = -2          0                    -2
    // rx = 0.1

    if ( rx < -0.5 )
    {
        res -= 1;
    }

    else if ( rx > 0.5 )
    {
        res += 1;
    }

    return res;
}


inline int xnfact(int i)
{
    if ( i <= 0 )
    {
        return 1;
    }

    int j;
    int res = 1;

    for ( j = 1 ; j <= i ; j++ )
    {
        res *= j;
    }

    return res;
}

inline int xnCr(int n, int r)
{
    int k;
    int result = 1;

    if ( ( n >= 0 ) && ( r >= 0 ) && ( n >= r ) )
    {
        // Recall: comb(0,0) = 1
        //         comb(n,0) = 1
        //         comb(n,n) = 1
        //         comb(n,r) = n!/(r!(n-r)!) (assuming r > n)

        result = 1;

        if ( ( r > 0 ) && ( n != r ) )
        {
            // Want r as large as possible

            if ( r > n/2 )
            {
                r = n-r;
            }

            for ( k = 1 ; k <= r ; k++ )
            {
                result *= (n-r+k);
                result /= k;
            }

//          if ( (n-r) > r )
//          {
//              r = n-r;
//          }
//
//          for ( kk = r+1 ; kk <= n ; kk++ )
//          {
//              result *= kk;
//                result /= r;
//          }
        }
    }

    return result;
}

inline unsigned int ceilintlog2(unsigned int x)
{
    unsigned int zord = 0;
    unsigned int zsize = x;

    if ( x == 0 )
    {
        zord = 0; // special case (the ceil part)
    }

    else
    {
	// Calculate log2(x)

	while ( ( zsize >>= 1 ) ) ++zord;

	// Assert that no rounding has occured

//if ( (1<<zord) != (int) x )
//{
//errstream() << "phantomx 0: zord = " << zord << "\n";
//errstream() << "phantomx 0: x = " << x << "\n";
//errstream() << "phantomx 0: (int) x = " << (int) x << "\n";
//}
//        NiceAssert( (1<<zord) == (int) x );
    }

    // ceilintlog2(0) = 0
    // ceilintlog2(1) = 0
    // ceilintlog2(2) = 1
    // ceilintlog2(4) = 2
    // ceilintlog2(8) = 3
    // ...

    return zord;
}



// Upwards rounding integer division
// (haven't checked what happens with negative arguments)
//
// eg:
//
//  i   | j   | i/j | updiv(i,j)
// -----+-----+-----+------------
//  0   | 2   | 0   | 0
//  1   | 2   | 0   | 1
//  2   | 2   | 1   | 1
//  3   | 2   | 1   | 2
//  4   | 2   | 2   | 2
//  5   | 2   | 2   | 3
//  6   | 2   | 3   | 3
//  7   | 2   | 3   | 4
//  ... | ... | ... | ...


inline int upidiv(int i, int j)
{
    return (i%j) ? ((i/j)+1) : (i/j);
}



#ifdef VISUAL_STU
#ifndef VISUAL_STU_OLD
#define ALT_INF_DEF
#endif
#endif

#ifdef IS_CPP11
#include <cmath>
#endif

#ifndef ALT_INF_DEF
inline int testisvnan(double x)
{
    return ( !( x > 0.0 ) && !( x < 0.0 ) && !( x == 0.0 ) );
}

inline int testisinf(double x)
{
    #ifdef IS_CPP11
    return std::isinf(x); // std::isinf(x);
    #endif
    #ifndef IS_CPP11
    return isinf(x);
    #endif
}

inline int testispinf(double x)
{
    return ( x == valpinf() );
}

inline int testisninf(double x)
{
    return ( x == valninf() );
}

inline double valvnan(void)
{
    double zz = 0.0;

    return 0.0/zz;
}

inline double valpinf(void)
{
    double zz = 0.0;

    return 1.0/zz;
}

inline double valninf(void)
{
    double zz = 0.0;

    return -1.0/zz;
}
#endif

#ifdef ALT_INF_DEF
inline int testisvnan(double x)
{
    return !( x > 0.0 ) && !( x < 0.0 ) && !( x == 0.0 );
}

inline int testisinf(double x)
{
    #ifdef IS_CPP11
    return std::isinf(x); // std::isinf(x);
    #endif
    #ifndef IS_CPP11
    return isinf(x);
    #endif
}

inline int testispinf(double x)
{
    return testisinf(x) && ( x > 0 );
}

inline int testisninf(double x)
{
    return testisinf(x) && ( x < 0 );
}

inline double valvnan(void)
{
    double x = 0.0;

    return 0.0/x;
}

inline double valpinf(void)
{
    double x = 0.0;

    return 1.0/x;
}

inline double valninf(void)
{
    double x = 0.0;

    return -1.0/x;
}
#endif

inline int zeroint(void)
{
    return 0;
}


#ifdef DJGPP_MATHS
inline int base_j0       (double &res,           double x) { res = j0(x);   return 0; }
inline int base_j1       (double &res,           double x) { res = j1(x);   return 0; }
inline int base_jn       (double &res, int n,    double x) { res = jn(n,x); return 0; }
inline int base_y0       (double &res,           double x) { res = y0(x);   return 0; }
inline int base_y1       (double &res,           double x) { res = y1(x);   return 0; }
inline int base_yn       (double &res, int n,    double x) { res = yn(n,x); return 0; }
#endif

#ifndef DJGPP_MATHS
#ifdef VISUAL_STUDIO_BESSEL
inline int base_j0       (double &res,           double x) { res = _j0(x);   return 0; }
inline int base_j1       (double &res,           double x) { res = _j1(x);   return 0; }
inline int base_jn       (double &res, int n,    double x) { res = _jn(n,x); return 0; }
inline int base_y0       (double &res,           double x) { res = _y0(x);   return 0; }
inline int base_y1       (double &res,           double x) { res = _y1(x);   return 0; }
inline int base_yn       (double &res, int n,    double x) { res = _yn(n,x); return 0; }
#endif

#ifndef VISUAL_STUDIO_BESSEL
inline int base_j0       (double &res,        double x) { (void) res; (void) x;           return 1; }
inline int base_j1       (double &res,        double x) { (void) res; (void) x;           return 1; }
inline int base_jn       (double &res, int n, double x) { (void) res; (void) x; (void) n; return 1; }
inline int base_y0       (double &res,        double x) { (void) res; (void) x;           return 1; }
inline int base_y1       (double &res,        double x) { (void) res; (void) x;           return 1; }
inline int base_yn       (double &res, int n, double x) { (void) res; (void) x; (void) n; return 1; }
#endif
#endif

inline int krondel(int i, int j)
{
    return ( i == j ) ? 1 : 0;
}










































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Stream clearing - keeps removing elements from the stream until a ':' is 
// encountered, removes the ':' and return the stream
//
// usage example:
//
// wait_dummy blah;
// int readvar;
// stream >> blah; stream >> readvar;
//
// when given "this is a variable: 10" will ignore the preceeding string and
// set readvar = 10.

class wait_dummy
{
    void *****ha_ha_ha;
};

std::istream &operator>>(std::istream &input, wait_dummy &come_on);

// stream to dev/null

class NullStreamBuf : public std::streambuf
{
    char dummy[128];

protected:

    virtual int overflow(int c)
    {
        setp(dummy,dummy+sizeof(dummy));
        return ( c == traits_type::eof() ) ? '\0' : c;
    }
};

class NullOStream : private NullStreamBuf, public std::ostream
{
public:

    NullOStream() : std::ostream(this) {}
    NullStreamBuf* rdbuf() { return this; }
};



































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Swapping functions

inline void qswap(char &a, char &b);
inline void qswap(char &a, char &b)
{
    char c = a; a = b; b = c;

    return;
}

inline void qswap(int &a, int &b);
inline void qswap(int &a, int &b)
{
    int c = a; a = b; b = c;

    return;
}

inline void qswap(unsigned int &a, unsigned int &b);
inline void qswap(unsigned int &a, unsigned int &b)
{
    int c = a; a = b; b = c;

    return;
}

inline void qswap(volatile int &a, volatile int &b);
inline void qswap(volatile int &a, volatile int &b)
{
    volatile int c = a; a = b; b = c;

    return;
}

inline void qswap(double &a, double &b);
inline void qswap(double &a, double &b)
{
    double c = a; a = b; b = c;

    return;
}

inline void qswap(const double *&a, const double *&b);
inline void qswap(const double *&a, const double *&b)
{
    const double *c = a; a = b; b = c;

    return;
}

inline void qswap(std::string &a, std::string &b);
inline void qswap(std::string &a, std::string &b)
{
    std::string c = a; a = b; b = c;

    return;
}

inline void qswap(std::istream *&a, std::istream *&b);
inline void qswap(std::istream *&a, std::istream *&b)
{
    std::istream *x;

    x = a; a = b; b = x;

    return;
}


























// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Random numbers
//
// randfill:  uniform random number between zero and one
// randnfill: gaussian normal N(0,1) random number
// svm_srand: seed random number generator
//            use -1 for "no seed", -2 for "time seed"
// svm_rand:  generate random integer 0->SVM_RAND_MAX

#define SVM_RAND_MAX RAND_MAX

inline double &randfill (double &res);
inline double &randnfill(double &res);
inline void svm_srand(int sval);
inline int svm_rand(void);

inline double &randfill(double &res)
{
    return res = ((double) svm_rand())/((double) SVM_RAND_MAX);
}

inline double &randnfill(double &res)
{
    // We use the central limit theorem

    res  = ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res += ((double) svm_rand())/((double) SVM_RAND_MAX);
    res -= 6;

    /*
    double probBad = 1;

    do
    {
        randfill(res);
        res = 20*((2*res)-1);
        randfill(probBad);
    }
    while ( probBad >= exp(-res*res) );
    */

    return res;
}

inline void svm_srand(int sval)
{
    if ( sval == -2 )
    {
        srand((int) time(NULL));
    }

    else if ( sval != -1 )
    {
        srand(sval);
    }

    return;
}

inline int svm_rand(void)
{
    return rand();
}






























// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// UUID generator (non-zero unique int)

inline int genUUID(void);

inline int genUUID(void)
{
    //NB zero result not allowed
    svmvolatile static int nextUUID = 1;

    //FIXME: extrememly unlikely bug may result when the UUID wraps back
    // through negatives and reaches zero.

    // Note atomic operation, so no need for locks
    return nextUUID++;
}



































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Binary number support (compile-time conversion)
//
// Converts 8 bit binary representation to int
//
// eg. binary(0110) = 6
//     binary(10000) = 16

#define BIGTYPE  unsigned long

template<BIGTYPE N>
class tobinary
{
public:

    enum
    {
        value = (N % 8) + (tobinary<N / 8>::value << 1)
    };
};

template<>
class tobinary<0>
{
public:

    enum
    {
        value = 0
    };
};

#define binnum(n) tobinary<0##n>::value
































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Time measurement:
//
// The aim is to measure elapsed time in seconds, with roughly ms precision, 
// with result in a double giving seconds elapsed.  Originally I used:
//
// typedef clock_t     time_used;
// typedef long double timediffunits;
//
// #define TIMEDIFFSEC(a,b) (((double) (a-b))/CLOCKS_PER_SEC)
// #define TIMECALL clock();
//
// (with apologies for the bad function naming).  But this fails for long
// itervales as clock_t is typically signed 32 bit and overflows.
//
// Instead we keep both clock ticks and timestamp.  If timestamp difference
// is less than CROSSOVER_TIME_SEC seconds (600 seconds == 10 minutes) then
// clock tick difference is used, which is fine.  Otherwise the more
// coarse-grained but non-overflowing timestamp difference is used.
//
//
// TIMEDIFFSEC: time difference between a and b in seconds
// TIMEABSSEC:  time difference between a and some arbitrary "zero" time
//              (the time this is first called)
// gettimeuse:  grabs timestamp of type time_used
// svm_usleep:  sleep for (possibly rounded off) stm microseconds.
// svm_msleep:  sleep for (possibly rounded off) stm milliseconds.
// svm_sleep:   sleep for (possibly rounded off) stm seconds.
// ZEROTIMEDIFF: set time difference to zero

#ifdef VISUAL_STU
#include <windows.h>
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif

#define CROSSOVER_TIME_SEC 600
#define TIMECALL gettimeuse()

typedef double timediffunits;

#define ZEROTIMEDIFF(x) x = 0

class time_used
{
    public:

    time_used(void)
    {
        fine_grain   = 0;
        coarse_grain = 0;
        return;
    }

    time_used(const time_used &src)
    {
        fine_grain   = src.fine_grain;
        coarse_grain = src.coarse_grain;
        return;
    }

    time_used &operator=(const time_used &src)
    {
        fine_grain   = src.fine_grain;
        coarse_grain = src.coarse_grain;
        return *this;
    }

    clock_t fine_grain;
    time_t  coarse_grain;
};

inline double TIMEDIFFSEC(const time_used &a, const time_used &b);
inline double TIMEABSSEC(const time_used &b);
inline time_used gettimeuse(void);

void svm_usleep(int stm);
void svm_msleep(int stm);
void svm_sleep(int stm);


inline double TIMEDIFFSEC(const time_used &a, const time_used &b)
{
    double coarse_diff = difftime(a.coarse_grain,b.coarse_grain);
    double fine_diff   = ((double) (a.fine_grain-b.fine_grain))/CLOCKS_PER_SEC;

    return ( coarse_diff < CROSSOVER_TIME_SEC ) ? fine_diff : coarse_diff;
}

inline double TIMEABSSEC(const time_used &b)
{
    static time_used a = b; // this will only initialise once, so the value
                            // will be whatever b is on the first call to
                            // this function.

    return TIMEDIFFSEC(a,b);
}

inline time_used gettimeuse(void)
{
    time_used res;

    res.fine_grain   = clock();
    res.coarse_grain = time(&(res.coarse_grain));

    return res;
}








































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Error logging class and stream redirection
//
// errstream: get error stream
// seterrstream: sets alternative error stream to std::cerr
//
// outstream: get standard output stream
// setoutstream: sets alternative output stream to std::cout
//
// instream: get standard input stream
// setinstream: sets alternative input stream to std::cin
//
// NB: This is not necessarily thread-safe in two ways:
//
// - setting the error stream is *NOT* thread-safe.  Do it once at the start
//   of the code *before* you get into threaded code, not later.
// - whatever you put in here must be thread-safe if you want your code to
//   be thread-safe.
//
// LoggingOstream: this class, if constructed with xcallback = NULL, just
// sends whatever is streamed into it to std::cerr.  If xcallback is set
// to a function then anything streamed into this will then be sent to
// that function as a stream of characters.  For example, you could link
// the callback function to printf to make the output stream just go to
// std::cout by a rather convoluated path.  Or you could set the function
// to do nothing to create a stream that acts like /dev/null etc.
//
// LoggingOstreamOut: like LoggingOstream, but defaults to std::cout
//
// LoggingIstream: in this case callback is called when input is
// required by istream.  Usually just gets a char from std::cin, but
// can be used to get input from any source.
//
// Multiple streams: you can set multiple pipes in a stream that can 
// (in principle) go to different targets.  The index i tells you
// which one you're referring to.  For simplicity you can have a 
// maximum of 128 streams (0-127).


class LoggingOstream : public std::ostream, public std::streambuf
{
public:
    LoggingOstream(void (*xcallback)(char c)) : std::ostream(this)
    {
        callback = xcallback;
        //setbuf(0,0);
        return;
    }

    int overflow(int c)
    {
        // This is a virtual function that will be called from ostream

        justprintit(c);
        return 0;
    }

    void justprintit(char c)
    {
        // This splits to either standard printing to cerr or
        // calling the callback function

        if ( callback == NULL )
        {
            std::cerr.put(c);
        }

        else
        {
            (*callback)(c);
        }

        return;
    }

    void setcallback(void (*xcallback)(char c) = NULL)
    {
        // This allows you to set callback

        callback = xcallback;

        return;
    }

private:

    // Function callback to print to alterative destination.
    
    void (*callback)(char c);
};

class LoggingOstreamOut : public std::ostream, public std::streambuf
{
public:
    LoggingOstreamOut(void (*xcallback)(char c)) : std::ostream(this)
    {
        callback = xcallback;
        setbuf(0,0);
        return;
    }

    int overflow(int c)
    {
        // This is a virtual function that will be called from ostream

        justprintit(c);
        return 0;
    }

    void justprintit(char c)
    {
        // This splits to either standard printing to cout (not cerr for this one) or
        // calling the callback function

        if ( callback == NULL )
        {
            std::cout.put(c);
        }

        else
        {
            (*callback)(c);
        }

        return;
    }

    void setcallback(void (*xcallback)(char c) = NULL)
    {
        // This allows you to set callback

        callback = xcallback;

        return;
    }

private:

    // Function callback to print to alterative destination.
    
    void (*callback)(char c);
};

class LoggingIstream : public std::istream, public std::streambuf
{
public:
    LoggingIstream(char (*xcallback)(void)) : std::istream(this)
    {
        callback = xcallback;
        setbuf(0,0);
        return;
    }

    int underflow(void)
    {
        // This is a virtual function that will be called from istream

        setg(buffer,buffer,buffer+1);
        *gptr() = justscanit();
        return *gptr();
    }

    char justscanit()
    {
        // This splits to either standard input from cin or
        // calling the callback function

        char c;

        if ( callback == NULL )
        {
            c = std::cin.get();
        }

        else
        {
            c = (*callback)();
        }

        return c;
    }

    void setcallback(char (*xcallback)(void) = NULL)
    {
        // This allows you to set callback

        callback = xcallback;

        return;
    }

private:

    // Function callback to print to scan alternative input
    
    char (*callback)(void);
    char buffer[1];
};

//std::ostream &errstream(int i = 0);
void seterrstream(LoggingOstream *altdest, int i = 0);

std::ostream &outstream(int i = 0);
void setoutstream(LoggingOstreamOut *altdest, int i = 0);

std::istream &instream(int i = 0);
void setinstream(LoggingIstream *altsrc, int i = 0);


// streamItIn: equivalent to input >> dest.  processxyzvw used elsewhere 
// streamItOut: equivalent to output << src.  if retainTypeMarker set then 
//              src will retain its essential "typeness" when printed (eg
//              double will always contain . or e).

#define STREAMINDUMMY(X) \
inline std::istream &streamItIn(std::istream &input, X& dest, int processxyzvw = 1); \
inline std::istream &streamItIn(std::istream &input, X& dest, int processxyzvw) \
{ \
    (void) dest; \
    (void) processxyzvw; \
    throw("Just no"); \
    return input; \
} \
inline std::ostream &operator<<(std::ostream &output, X& src); \
inline std::ostream &operator<<(std::ostream &output, X& src) \
{ \
    (void) src; \
    throw("Just no"); \
    return output; \
} \
inline std::istream &operator>>(std::istream &input, X& dest); \
inline std::istream &operator>>(std::istream &input, X& dest) \
{ \
    (void) dest; \
    throw("Just no"); \
    return input; \
}

//STREAMINDUMMY(const double *);
//STREAMINDUMMY(const int *);
//STREAMINDUMMY(const std::string *);
	

inline std::istream &streamItIn(std::istream &input, int&         dest, int processxyzvw = 1);
inline std::istream &streamItIn(std::istream &input, double&      dest, int processxyzvw = 1);
inline std::istream &streamItIn(std::istream &input, char&        dest, int processxyzvw = 1);
inline std::istream &streamItIn(std::istream &input, char*        dest, int processxyzvw = 1);
inline std::istream &streamItIn(std::istream &input, std::string& dest, int processxyzvw = 1);

inline std::ostream &streamItOut(std::ostream &output, const int&         src, int retainTypeMarker = 0);
inline std::ostream &streamItOut(std::ostream &output, const double&      src, int retainTypeMarker = 0);
inline std::ostream &streamItOut(std::ostream &output, const char&        src, int retainTypeMarker = 0);
inline std::ostream &streamItOut(std::ostream &output, char* const&       src, int retainTypeMarker = 0);
inline std::ostream &streamItOut(std::ostream &output, const std::string& src, int retainTypeMarker = 0);

inline std::istream &streamItIn(std::istream &input, double &dest, int processxyzvw)
{
    (void) processxyzvw;

    input >> dest;

    return input;
}

inline std::istream &streamItIn(std::istream &input, int &dest, int processxyzvw)
{
    (void) processxyzvw;

    input >> dest;

    return input;
}

inline std::istream &streamItIn(std::istream &input, char &dest, int processxyzvw)
{
    (void) processxyzvw;

    input >> dest;

    return input;
}

inline std::istream &streamItIn(std::istream &input, char *dest, int processxyzvw)
{
    (void) processxyzvw;

    input >> dest;

    return input;
}

inline std::istream &streamItIn(std::istream &input, std::string &dest, int processxyzvw)
{
    (void) processxyzvw;

    input >> dest;

    return input;
}

inline std::ostream &streamItOut(std::ostream &output, const double &src, int retainTypeMarker)
{
    char tempres[100];
    sprintf(tempres,"%.17g",src);
    std::string tempresb(tempres);
    output << tempresb;

    if ( retainTypeMarker && !tempresb.find(".") && !tempresb.find("e") && !tempresb.find("E") )
    {
        output << ".0";
    }

    return output;
}

inline std::ostream &streamItOut(std::ostream &output, const int &src, int retainTypeMarker)
{
    (void) retainTypeMarker;

    output << src;

    return output;
}

inline std::ostream &streamItOut(std::ostream &output, const char &src, int retainTypeMarker)
{
    (void) retainTypeMarker;

    output << src;

    return output;
}

inline std::ostream &streamItOut(std::ostream &output, char* const& src, int retainTypeMarker)
{
    (void) retainTypeMarker;

    output << src;

    return output;
}

inline std::ostream &streamItOut(std::ostream &output, const std::string& src, int retainTypeMarker)
{
    (void) retainTypeMarker;

    output << src;

    return output;
}





































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// nullPrint: Print string to stream, then backspace back to start of string (or a few chars after if offset set positive).  Returns number of characters printed (so you can blackPrint to cover over)
// blankPrint: Print n spaces, then backspace back over them
// repPrint: pring character n times

int nullPrint(std::ostream &dest, const std::string &src, int offset = 0);
int nullPrint(std::ostream &dest, const char *src, int offset = 0);
int nullPrint(std::ostream &dest, const int src, int offset = 0);

std::ostream &blankPrint(std::ostream &dest, int n);

std::ostream &repPrint(std::ostream &dest, const char &c, int n);























































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Asynchronous keyboard quit detection
//
// kbquitdet:
//
// If no keys hit will do nothing and return 0
// If key hit will print prompt and wait for user
// - if user input is quit then returns 1
// - if user input is anything else then returns 0
//
// uservars: either NULL, or NULL-terminated list of adjustable variables
// varnames: names of above
// vardescr: descriptions of above
// goanyhow: if set then assume key pressed
//
// disablekbquitdet: disable function
// enablekbquitdet:  enable function
// triggerkbquitdet: set kbquitdet to act as if key has been pressed on next call
//
// haskeybeenpressed: functionally equivalent to kbhit in conio.h but for
//                    a wider variety of systems (but not all)
//
// interactmenu: what gets called if key is pressed.  Dummy may be modified but
//               just ignore this.
//
// setintercalc: Function to set calculator function callback
// setkbcallback: set callback function to test if interupt key (eg ctrl-c in matlab) has been pressed.
//
// gintercalc: call(back) internal calculator function
// gkbcallback: call(back) interupt key test function

inline int kbquitdet(const char *stateDescr, double **uservars = NULL, const char **varnames = NULL, const char **vardescr = NULL, int goanyhow = 0);
inline void disablekbquitdet(void);
inline void enablekbquitdet(void);
inline void triggerkbquitdet(void);
inline void clearkbquitdet(void);
inline void setintercalc(void (*intercalccall)(std::ostream &, std::istream &) = NULL);
inline void setkbcallback(int (*kbcallback)(void) = NULL);
inline int gkbcallback(int (*kbcallback)(void) = NULL);
inline void gintercalc(std::ostream &output, std::istream &input, void (*intercalccall)(std::ostream &, std::istream &) = NULL);

int interactmenu(int &dummy, const char *stateDescr, double **uservars = NULL, const char **varnames = NULL, const char **vardescr = NULL);
int haskeybeenpressed(void);
const char *randomquote(void);



// Don't use the function retkeygriggerandclear
inline int retkeytriggerandclear(int val = 0);
inline int retkeytriggerandclear(int val)
{
    static int trval = 0; // trigger not set by default
    int retval = trval;   // return is current trigger state

    trval = val; // Set new value

    return retval; // return old value
}

// Don't use setgetkbstate
inline int setgetkbstate(int x = 2);
inline int setgetkbstate(int x)
{
    #ifdef DISABLE_KB_BY_DEF
    svmvolatile static int status = 0; // 1 enabled, 0 disabled
    #endif

    #ifndef DISABLE_KB_BY_DEF
    svmvolatile static int status = 1; // 1 enabled, 0 disabled
    #endif

    if ( x == 0 ) { status = 0; }
    if ( x == 1 ) { status = 1; }

    return status;
}


inline int gkbcallback(int (*kbcallback)(void))
{
    static int (*lockbcallback)(void) = NULL;
    int res = 0;

    if ( kbcallback )
    {
        lockbcallback = kbcallback;
    }

    else if ( lockbcallback )
    {
        res = lockbcallback();
    }

    return res;
}

inline void gintercalc(std::ostream &output, std::istream &input, void (*intercalccall)(std::ostream &, std::istream &))
{
    static void (*loccalc)(std::ostream &, std::istream &) = NULL;

    if ( intercalccall )
    {
        loccalc = intercalccall;
    }

    else if ( loccalc )
    {
        loccalc(output,input);
    }

    else
    {
        output << "Calculator not fitted.\n";
    }

    return;
}

inline void setintercalc(void (*intercalccall)(std::ostream &, std::istream &))
{
    gintercalc(outstream(),instream(),intercalccall);

    return;
}

inline void setkbcallback(int (*kbcallback)(void))
{
    gkbcallback(kbcallback);

    return;
}

inline void disablekbquitdet(void)
{
   setgetkbstate(0);

   return;
}

inline void enablekbquitdet(void)
{
   setgetkbstate(1);

   return;
}

inline void triggerkbquitdet(void)
{
    retkeytriggerandclear(1); // set trigger

    return;
}

inline void clearkbquitdet(void)
{
    retkeytriggerandclear(0); // clear trigger

    return;
}

inline int kbquitdet(const char *stateDescr, double **uservars, const char **varnames, const char **vardescr, int goanyhow)
{
    static int goupone = 0;
    int res = 0;

    // short-circuit logic
    if ( isMainThread() && setgetkbstate() && ( goupone || haskeybeenpressed() || goanyhow || retkeytriggerandclear() || gkbcallback() ) )
    {
        if ( goupone )
        {
            goupone--;
        }

        if ( goupone )
        {
            res = 1;
        }

        else
        {
            res = interactmenu(goupone,stateDescr,uservars,varnames,vardescr);
        }
    }

    return res;
}




























































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// Memory count function
//
// Note that memcount is neither (a) accurate or (b) thread-safe.

inline size_t memcount(size_t incsize = 0, int direction = -1);
inline size_t memcount(size_t incsize, int direction)
{
    svmvolatile static size_t memused = 0;

    if ( direction > 0 )
    {
        memused += incsize;
    }

    else if ( direction < 0 )
    {
        memused -= ( ( incsize < memused ) ? incsize : memused );
    }

    else
    {
        memused = incsize;
    }

    return memused;
}





















// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// "Unsafe" function isolation

inline char *strncpy_safe(char *dest, const char *src, size_t len);
inline char *strncpy_safe(char *dest, const char *src, size_t len)
{
    strncpy(dest,src,len);
    dest[len] = '\0';
    return dest;
}

















// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// File functions: because earlier versions of c++ don't support "does
// this file exist" functionality, we have this
//
// getUniqueFile: construct a filename string pre+UUID+post where the file
// does not exist.  UUID is in hex, because hex is cool.

int fileExists(const std::string &fname);
const std::string &getUniqueFile(std::string &res, const std::string &pre, const std::string &post);
















//------------------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//************************************************************************
//------------------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//************************************************************************
//------------------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//************************************************************************
//------------------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//************************************************************************
//------------------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//******************************************************************

#endif

