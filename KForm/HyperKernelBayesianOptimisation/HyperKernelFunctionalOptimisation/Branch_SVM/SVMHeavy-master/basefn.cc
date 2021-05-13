
//
// Miscellaneous stuff
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//



#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctype.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <sstream>
#include <time.h>
#include <stdlib.h>
//#include <unistd.h>
#include "basefn.h"

#ifdef HAVE_CONIO
#include <conio.h>
#endif

#ifdef HAVE_TERMIOS
#include <termios.h>
#include <fcntl.h>
#endif

#ifdef VISUAL_STU
#include <windows.h>
#endif

#ifdef HAVE_IOCTL
#include <sys/ioctl.h>
#endif



unsigned long long _global_alloccnt = 0;
unsigned long long _global_maxalloccnt = 0;





int fileExists(const std::string &fname)
{
    // We just assume that the user has correct permissions for the file in question.
    // C++17 apparently has a better method, but compatibility blah blah blah

    std::ifstream testfile(fname.c_str());
    int res = testfile.good();
    testfile.close();
    return res;
}

const std::string &getUniqueFile(std::string &res, const std::string &pre, const std::string &post)
{
    char uuidstr[2048];
    int notgotone = 1;

    while ( notgotone )
    {
        int uuidint = genUUID();

        sprintf(uuidstr,"%d",uuidint);

        res  = pre;
        res += uuidstr;
        res += post;

        notgotone = fileExists(res);
    }

    return res;
}

// Call executable.  If runbg then attempt to leave it running in the background

int svm_execall(const std::string &command, int runbg)
{
    std::string fullcommand = "";

    fullcommand += command;

    if ( runbg )
    {
        fullcommand += " &";
    }

    return system(fullcommand.c_str());
}

// Call python script.  If runbg then attempt to leave it running in the background

int svm_pycall(const std::string &command, int runbg)
{
    std::string fullcommand = "python3 ";

    fullcommand += command;

    if ( runbg )
    {
        fullcommand += " &";
    }

errstream() << "full call " << fullcommand << "\n";
    int res = system(fullcommand.c_str());

    return res;
}















void svm_usleep(int stm)
{
    if ( stm >= 1000 )
    {
        svm_msleep(stm/1000);
        svm_usleep(stm%1000);
    }

    else
    {
#ifndef VISUAL_STU
#ifndef CYGWIN10
        // max 1000000
        usleep(stm); // microsecond granularity
#endif
#ifdef CYGWIN10
        // max nanosleep 999999999
        struct timespec tim, tim2;
        tim.tv_sec = 0;
        tim.tv_nsec = 1000*stm;
        nanosleep(&tim,&tim2);
#endif
#endif
#ifdef VISUAL_STU
#ifndef VISUAL_STU_OLD
        //_sleep(stm/1000);
        Sleep(stm/1000); // millisecond granularity
#endif
#endif
    }

    return;
}

void svm_msleep(int stm)
{
    if ( stm > 1000 )
    {
        svm_sleep(stm/1000);
        svm_msleep(stm%1000);
    }

    else
    {
#ifndef VISUAL_STU
#ifndef CYGWIN10
        // max 1000000
        usleep(1000*stm); // microsecond granularity
#endif
#ifdef CYGWIN10
        // max nanosleep 999999999
        struct timespec tim, tim2;
        tim.tv_sec = 0;
        tim.tv_nsec = 1000000*stm;
        nanosleep(&tim,&tim2);
#endif
#endif
#ifdef VISUAL_STU
#ifndef VISUAL_STU_OLD
        Sleep(stm); // millisecond granularity
#endif
#endif
    }

    return;
}

void svm_sleep(int stm)
{
#ifndef VISUAL_STU
#ifndef CYGWIN10
    sleep(stm);
#endif
#ifdef CYGWIN10
    sleep(stm);
#endif
#endif
#ifdef VISUAL_STU
#ifndef VISUAL_STU_OLD
    Sleep(1000*stm);
#endif
#endif

    return;
}


















int nullPrint(std::ostream &dest, int src, int offset)
{
    char buffer[256];

    sprintf(buffer,"%d",src);

    return nullPrint(dest,buffer,offset);
}

int nullPrint(std::ostream &dest, const std::string &src, int offset)
{
    int len = ((int) src.length()); //-offset;
    int blen = 0;

    if ( len > 0 )
    {
        int i;

        for ( i = 0 ; i < len ; i++ )
        {
            if ( src[i] == '\t' )
            {
                dest << ' ';
                blen++;

                while ( blen%8 )
                {
                    dest << ' ';
                    blen++;
                }
            }

            else if ( src[i] == '\b' )
            {
                dest << src[i];
                blen--;
            }

            else
            {
                dest << src[i];
                blen++;
            }
        }

        blen -= offset;

        for ( i = 0 ; i < blen ; i++ )
        {
            dest << "\b";
        }
    }

    return blen;
}

int nullPrint(std::ostream &dest, const char *src, int offset)
{
    std::string altsrc(src);

    return nullPrint(dest,altsrc,offset);
}

std::ostream &blankPrint(std::ostream &dest, int len)
{
    if ( len > 0 )
    {
        int i;

        for ( i = 0 ; i < len ; i++ )
        {
            dest << ' ';
        }

        for ( i = 0 ; i < len ; i++ )
        {
            dest << "\b";
        }
    }

    return dest;
}

std::ostream &repPrint(std::ostream &dest, const char &c, int n)
{
    if ( n > 0 )
    {
        int i;

        for ( i = 0 ; i < n ; i++ )
        {
            dest << c;
        }
    }

    return dest;
}

















std::ostream &errstream(LoggingOstream *newstream, int i);
std::ostream &errstream(LoggingOstream *newstream, int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= 127 );

    svmvolatile static LoggingOstream *altdest[128] = { NULL };

    if ( newstream )
    {
        //Let the caller do deletion (assume only called once anyhow)
        //
        //if ( altdest )
        //{
        //    MEMDEL(altdest);
        //}
        //
        //NiceAssert( !altdest );

        altdest[i] = newstream;
    }

    if ( altdest[i] == NULL )
    {
        return std::cerr;
    }

    return static_cast<std::ostream &>(const_cast<LoggingOstream &>(*(altdest[i])));
}


std::ostream &errstream(int i)
{
    return errstream(NULL,i);
}

void seterrstream(LoggingOstream *altdest, int i)
{
    errstream(altdest,i);

    return;
}

std::ostream &outstream(LoggingOstreamOut *newstream, int i);
std::ostream &outstream(LoggingOstreamOut *newstream, int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= 127 );

    svmvolatile static LoggingOstreamOut *altdest[128] = { NULL };

    if ( newstream )
    {
        //Let the caller do deletion (assume only called once anyhow)
        //
        //if ( altdest )
        //{
        //    MEMDEL(altdest);
        //}
        //
        //NiceAssert( !altdest );

        altdest[i] = newstream;
    }

    if ( altdest[i] == NULL )
    {
        return std::cout;
    }

    return static_cast<std::ostream &>(const_cast<LoggingOstreamOut &>(*(altdest[i])));
}


std::ostream &outstream(int i)
{
    return outstream(NULL,i);
}

void setoutstream(LoggingOstreamOut *altdest, int i)
{
    outstream(altdest,i);

    return;
}

std::istream &instream(LoggingIstream *newstream, int i);
std::istream &instream(LoggingIstream *newstream, int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= 127 );

    svmvolatile static LoggingIstream *altsrc[128] = { NULL };

    if ( newstream )
    {
        //Let the caller do deletion (assume only called once anyhow)
        //
        //if ( altdest )
        //{
        //    MEMDEL(altdest);
        //}
        //
        //NiceAssert( !altdest );

        altsrc[i] = newstream;
    }

    if ( altsrc[i] == NULL )
    {
        return std::cin;
    }

    return static_cast<std::istream &>(const_cast<LoggingIstream &>(*(altsrc[i])));
}


std::istream &instream(int i)
{
    return instream(NULL,i);
}

void setinstream(LoggingIstream *altsrc, int i)
{
    instream(altsrc,i);

    return;
}

std::istream &operator>>(std::istream &input, wait_dummy &)
{
    char scanner = 'z';

    while ( scanner != ':' )
    {
        input >> scanner;
    }

    input.ignore(1);

    return input;
}
























typedef void (*exitptr)(void);

void dumpmemoryleakinfo(void);
void dumpmemoryleakinfo(void)
{
    addremoveptr(NULL,-1,0,0,NULL);

    return;
}

void svm_exitfn(void);
void svm_exitfn(void)
{
    svm_atexit(NULL,NULL);

    return;
}



atexitfn svm_setatexitfn(atexitfn xfn)
{
    static atexitfn fn = atexit;

    if ( xfn )
    {
        fn = xfn;
    }

    return fn;
}

int svm_atexit(void (*func)(void), const char *desc)
{
    static int listsize = 0;
    static int allocsize = 0;
    static char **desclist = NULL;
    static exitptr *exitlist = NULL;

    int i;

    if ( !allocsize )
    {
        //atexit(svm_exitfn);
        svm_setatexitfn()(svm_exitfn);

        allocsize = 100;

        exitlist = new exitptr[allocsize];
        desclist = new char *[allocsize];
    }

    if ( func == NULL )
    {
        if ( desclist )
        {
            for ( i = 0 ; i < listsize ; i++ )
            {
                errstream() << "Exit function " << i << " (" << desclist[i] << ")... ";
                exitlist[i]();
                errstream() << "done.\n";

                delete[] desclist[i];
            }

            dumpmemoryleakinfo();

            delete[] desclist;
            delete[] exitlist;

            desclist = NULL;
            exitlist = NULL;
        }
    }

    else if ( func != dumpmemoryleakinfo )
    {
        if ( allocsize == listsize )
        {
            exitptr *oldexitlist = exitlist;
            char   **olddesclist = desclist;

            allocsize += 100;

            exitlist = new exitptr[allocsize];
            desclist = new char *[allocsize];

            int i;

            for ( i = 0 ; i < listsize ; i++ )
            {
                exitlist[i] = oldexitlist[i];
                desclist[i] = olddesclist[i];
            }

            delete[] oldexitlist;
            delete[] olddesclist;
        }

        exitlist[listsize] = func;
        
        desclist[listsize] = new char[strlen(desc)+1];

        strcpy(desclist[listsize],desc);
        desclist[listsize][strlen(desc)] = '\0';

        listsize++;
    }

    return 0;
}

int addremoveptr(void *addr, int newdel, int type, int size, const char *desc)
{
    static void **addrlist;
    static int *typelist;
    static int *sizelist;
    static char **desclist;
    static int listsize = 0;
    static int allocsize = 0;

    int res = 0;

    if ( newdel == -1 )
    {
        int i;

        for ( i = 0 ; i < listsize ; i++ )
        {
            errstream() << i << " - Memory leak: type " << typelist[i] << ", size " << sizelist[i] << ", description " << desclist[i] << "\n";

            delete[] desclist[i];
        }

        delete[] addrlist;
        delete[] typelist;
        delete[] sizelist;
        delete[] desclist;

        return 0;
    }

    if ( !allocsize )
    {
        svm_atexit(dumpmemoryleakinfo,"Memchecker");

        allocsize = 1024;

        addrlist = new void *[allocsize];
        typelist = new int[allocsize];
        sizelist = new int[allocsize];
        desclist = new char *[allocsize];
    }

    if ( ( allocsize == listsize ) && newdel )
    {
errstream() << "!" << allocsize/1024 << "!";

        void **oldaddrlist = addrlist;
        int *oldtypelist = typelist;
        int *oldsizelist = sizelist;
        char **olddesclist = desclist;

        allocsize += 1024;

        addrlist = new void *[allocsize];
        typelist = new int[allocsize];
        sizelist = new int[allocsize];
        desclist = new char *[allocsize];

        int i;

        for ( i = 0 ; i < listsize ; i++ )
        {
            addrlist[i] = oldaddrlist[i];
            typelist[i] = oldtypelist[i];
            sizelist[i] = oldsizelist[i];
            desclist[i] = olddesclist[i];
        }

        delete[] oldaddrlist;
        delete[] oldtypelist;
        delete[] oldsizelist;
        delete[] olddesclist;
    }

    if ( newdel )
    {
        addrlist[listsize] = addr;
        typelist[listsize] = type;
        sizelist[listsize] = size;
        desclist[listsize] = new char[strlen(desc)+1];

        strcpy(desclist[listsize],desc);
        desclist[listsize][strlen(desc)] = '\0';

        listsize++;
    }

    else
    {
        int i;

        for ( i = 0 ; i < listsize ; i++ )
        {
            if ( addrlist[i] == addr )
            {
                break;
            }
        }

        if ( i >= listsize )
        {
            res = 1;
        }

        else
        {
            if ( ( type == 0 ) && ( typelist[i] == 1 ) )
            {
                res = 2;
            }

            else if ( ( type == 1 ) && ( typelist[i] == 0 ) )
            {
                res = 3;
            }

            listsize--;

            delete[] desclist[i];

            for ( ; i < listsize ; i++ )
            {
                addrlist[i] = addrlist[i+1];
                typelist[i] = typelist[i+1];
                sizelist[i] = sizelist[i+1];
                desclist[i] = desclist[i+1];
            }
        }
    }

    return res;
}






































int makeMathsStringNice(std::string &dest, const std::string &src)
{
  int i,j;
  int isQuote = 0;
  //char prevchar = ' ';

  dest = src;

  if ( dest.length() )
  {
    // Replace any E with e

//    isQuote = 0;
//    prevchar = ' ';
//
//    for ( i = 0 ; i < (int) dest.length() ; i++ )
//    {
//	if ( ( dest[i] == '\"' ) && ( dest[i] != '\\' ) )
//	{
//	    isQuote = isQuote ? 0 : 1;
//	}
//
//	if ( !isQuote && ( dest[i] == 'E' ) )
//	{
//	    dest[i] = 'e';
//	}
//
//        prevchar = dest[i];
//    }

    // Deal with long strings like ++----++++-+

    isQuote = 0;
    //prevchar = ' ';

    if ( dest.length() > 1 )
    {
	j = 0;

	for ( i = 1 ; i < (int) dest.length() ; i++ )
	{
	    if ( ( dest[i] == '\"' ) && ( dest[i] != '\\' ) )
	    {
		isQuote = isQuote ? 0 : 1;
	    }

	    if ( !isQuote && ( ( ( dest[i] == '+' ) && ( dest[j] == '+' ) ) || ( ( dest[i] == '-' ) && ( dest[j] == '-' ) ) ) )
	    {
                if ( dest.length() == 1 ) { return 2; }

                dest.erase(j,1);
		dest[j] = '+';

		i--;
		j--;
	    }

	    else if ( ! isQuote && ( ( ( dest[i] == '+' ) && ( dest[j] == '-' ) ) || ( ( dest[i] == '-' ) && ( dest[j] == '+' ) ) ) )
	    {
                if ( dest.length() == 1 ) { return 3; }

                dest.erase(j,1);
		dest[j] = '-';

		i--;
		j--;
	    }

	    j++;

	    //prevchar = dest[i];
	}
    }

    // Make sure - is used only for negation, so replace any subtraction ( - not at start or preceeded by e(^*/\%:|=><~& ) with +-

    isQuote = 0;
    //prevchar = ' ';

    if ( dest.length() > 1 )
    {
	j = 0;

	for ( i = 1 ; i < (int) dest.length() ; i++ )
	{
	    if ( ( dest[i] == '\"' ) && ( dest[i] != '\\' ) )
	    {
		isQuote = isQuote ? 0 : 1;
	    }

            if ( !isQuote && ( ( dest[i] == '-' ) && ( dest[j] != 'e'  ) && ( dest[j] != 'E' ) && ( dest[j] != '(' ) && ( dest[j] != '^' ) && ( dest[j] != '*' ) && ( dest[j] != '/' )
		                                  && ( dest[j] != '\\' ) && ( dest[j] != '%' ) && ( dest[j] != ':' ) && ( dest[j] != '!' ) && ( dest[j] != '=' )
		                                  && ( dest[j] != ';'  ) && ( dest[j] != '>' ) && ( dest[j] != '<' ) && ( dest[j] != '~' ) && ( dest[j] != '&' )
			                          && ( dest[j] != ','  ) && !isspace(dest[j]) ) )
	    {
                dest.insert(i,1,'+');

		i++;
		j++;
	    }

	    j++;

	    //prevchar = dest[i];
	}
    }

    // Remove any superfluous posations ( + at start or preceeded by e(^*/\%:|=><~& )

    while ( dest[0] == '+' )
    {
	dest.erase(0,1);

	if ( !dest.length() )
	{
            break;
	}
    }

    isQuote = 0;
    //prevchar = ' ';

    if ( dest.length() > 1 )
    {
	j = 0;

	for ( i = 1 ; i < (int) dest.length() ; i++ )
	{
	    if ( ( dest[i] == '\"' ) && ( dest[i] != '\\' ) )
	    {
		isQuote = isQuote ? 0 : 1;
	    }

            if ( !isQuote && ( ( dest[i] == '+' ) && (    ( dest[j] == 'e'  ) || ( dest[j] == 'E'  ) || ( dest[j] == '(' ) || ( dest[j] == '^' ) || ( dest[j] == '*' ) || ( dest[j] == '/' )
		                                       || ( dest[j] == '\\' ) || ( dest[j] == '%' ) || ( dest[j] == ':' ) || ( dest[j] == '!' ) || ( dest[j] == '=' )
		                                       || ( dest[j] == '>'  ) || ( dest[j] == '<' ) || ( dest[j] == '~' ) || ( dest[j] == '&' ) ) ) )
	    {
                if ( dest.length() == 1 ) { return 6; }

                dest.erase(i,1);

		i--;
		j--;
	    }

	    j++;

	    //prevchar = dest[i];
	}
    }
  }

  return 0;
}





































#ifdef ENABLE_THREADS
#ifdef VISUAL_STU
DWORD WINAPI SVMThreadFunction(LPVOID lpParam)
{
    svm_pthread_t &owner = *((svm_pthread_t *) ((void *) lpParam));

    owner.callfn();

    return 0;
}
#endif
#endif





























































/*
 ANSI sequence summary from some webpage or other

- Position the Cursor:
  \033[<L>;<C>H
     Or
  \033[<L>;<C>f
  puts the cursor at line L and column C.
- Move the cursor up N lines:
  \033[<N>A
- Move the cursor down N lines:
  \033[<N>B
- Move the cursor forward N columns:
  \033[<N>C
- Move the cursor backward N columns:
  \033[<N>D

- Clear the screen, move to (0,0):
  \033[2J
- Erase to end of line:
  \033[K

- Save cursor position:
  \033[s
- Restore cursor position:
  \033[u
*/

// Function returns true if a key has been pressed, false otherwise

int haskeybeenpressed(void)
{
    int res = 0;

    // If conio present use kbhit variant

    #ifdef HAVE_CONIO
    #ifdef VISUAL_STU
    #ifdef VISUAL_STU_OLD
    res = kbhit();
    #endif
    #ifndef VISUAL_STU_OLD
    res = _kbhit();
    #endif
    #endif
    #ifndef VISUAL_STU
    res = kbhit();
    #endif
    #endif

    // Fallback method (only detects enter and maybe whitespace, nothing else)

    #ifndef HAVE_CONIO
    struct timeval tv;
    fd_set fds;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO,&fds); //STDIN_FILENO is 0
    select(STDIN_FILENO+1,&fds,NULL,NULL,&tv);
    res = FD_ISSET(STDIN_FILENO,&fds);
    #endif

    return res;
}

void svmclrscr(int fastver);
void svmclrscr(int fastver)
{
    // Not windows, not djgpp, assume ANSI terminal and use escape codes
    // to clear the window and return the cursor to home position

    (void) fastver;

    #ifndef HAVE_CONIO
    //outstream() << "\n";
    //Assuming linux/cygwin
    //system("clear"); - works but v. slow
    //outstream() << 0x0c; - does not work
    //curses.h/ncurses.h - more trouble than it's worth
    // The code below works, assuming and ANSI-compliant terminal
    if ( !fastver )
    {
        outstream() << "\n\033[2J\033[0;0f";
    }
    else
    {
        outstream() << "\033[0;0f";
    }
    //fflush(stdout);
    #endif

    // DJGPP: sanest environment of the lot: call clrscr

    #ifdef HAVE_CONIO
    #ifndef VISUAL_STU
    if ( !fastver )
    {
        clrscr();
    }
    else
    {
        gotoxy(1,1); outstream() << "\n"; gotoxy(1,1);
    }
    #endif
    #endif

    // Windows: short version would be system("cls"), but that is slow,
    // so instead we need the following mess.  This is from the visual
    // studio webpage.

    #ifdef HAVE_CONIO
    #ifdef VISUAL_STU
    //system("cls");
    {
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        COORD coordScreen = {0,0}; // Home cursor position

        if ( !fastver )
        {
            DWORD cCharsWritten;
            CONSOLE_SCREEN_BUFFER_INFO csbi;
            DWORD dwConSize;

            // Get number of char cells in buffer

            if ( !GetConsoleScreenBufferInfo(hConsole,&csbi) )
            {
                return;
            }

            dwConSize = csbi.dwSize.X * csbi.dwSize.Y;

            // Fill entire screen with blanks

            if ( !FillConsoleOutputCharacter(hConsole,(TCHAR) ' ',dwConSize,coordScreen,&cCharsWritten) )
            {
                return;
            }

            // Get the current text attribute

            if ( !GetConsoleScreenBufferInfo(hConsole,&csbi) )
            {
                return;
            }

            // Set the buffer's attributes accordingly

            if ( !FillConsoleOutputAttribute(hConsole,csbi.wAttributes,dwConSize,coordScreen,&cCharsWritten) )
            {
                return;
            }
        }

        // Put the cursor at its home coords

        SetConsoleCursorPosition(hConsole,coordScreen);

        return;
    }
    #endif
    #endif

    return;
}

void getscrrowcol(int &rows, int &cols);
void getscrrowcol(int &rows, int &cols)
{
    // DJGPP (conio, but not windows) version
    #ifdef HAVE_CONIO
    #ifndef VISUAL_STU
    #ifndef VISUAL_STU_OLD
    struct text_info r;
    gettextinfo(&r);
    rows = r.screenheight;
    cols = r.screenwidth;
    #endif
    #endif
    #endif

    // Windows version
    #ifdef VISUAL_STU
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),&csbi);
    cols = csbi.srWindow.Right  - csbi.srWindow.Left + 1;
    rows = csbi.srWindow.Bottom - csbi.srWindow.Top  + 1;
    //cols = csbi.dwSize.X;
    //rows = csbi.dwSize.Y;
    #endif

    // Linux/cygwin version
    #ifndef HAVE_CONIO
    #ifdef HAVE_IOCTL
    struct winsize w;
    ioctl(STDOUT_FILENO,TIOCGWINSZ,&w);
    rows = w.ws_row;
    cols = w.ws_col;
    #endif
    #endif

    return;
}

#ifdef HAVE_TERMIOS

static struct termios newtm, oldtm;
static int oldf;

#endif

void enternonblockmode(void);
void enternonblockmode(void)
{
    #ifdef HAVE_TERMIOS

    fflush(stdout);
#ifndef NOPURGE
#ifndef ALTPURGE
    fpurge(stdin);
#endif
#endif
#ifdef ALTPURGE
    __fpurge(stdin);
#endif
    if(tcgetattr(STDIN_FILENO, &oldtm)<0)
        perror("tcsetattr()");
    newtm = oldtm;
    oldtm.c_lflag&=~(ICANON | ECHO);//~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);//~ICANON;
    if(tcsetattr(STDIN_FILENO,TCSANOW,&oldtm)<0)
        perror("tcsetattr ICANON");
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    #endif

    return;
}

void exitnonblockmode(void);
void exitnonblockmode(void)
{
    #ifdef HAVE_TERMIOS

    fcntl(STDIN_FILENO, F_SETFL, oldf);
    if(tcsetattr(STDIN_FILENO,TCSADRAIN,&newtm)<0)
        perror ("tcsetattr ~ICANON");

    #endif

    return;
}

#define FLUSHINTERVAL 10

char svm_getch_nonblock(void);
char svm_getch_nonblock(void)
{
    char buf = '\0';

    #ifdef HAVE_CONIO
    if ( haskeybeenpressed() )
    {
        #ifdef VISUAL_STU
        #ifdef VISUAL_STU_OLD
        buf = getch();
        #endif
        #ifndef VISUAL_STU_OLD
        buf = _getch();
        #endif
        #endif
        #ifndef VISUAL_STU
        buf = getch();
        #endif
    }
    #endif

    #ifdef HAVE_TERMIOS

    // http://stackoverflow.com/questions/22166074/is-there-a-way-to-detect-if-a-key-has-been-pressed
    // http://stackoverflow.com/questions/22766613/is-there-a-way-to-replace-the-kbhit-and-getch-functions-in-standard-c
    // and others
////    struct termios newtm,oldtm;
////    fflush(stdout);
////    if(tcgetattr(STDIN_FILENO, &oldtm)<0)
////        perror("tcsetattr()");
////    newtm = oldtm;
////    oldtm.c_lflag&=~(ICANON | ECHO);//~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);//~ICANON;
////    //oldtm.c_cc[VMIN]=0;//1;
////    //oldtm.c_cc[VTIME]=0;
////    if(tcsetattr(STDIN_FILENO,TCSANOW,&oldtm)<0)
////        perror("tcsetattr ICANON");
////    //read(0,&buf,1);
////    //if(read(0,&buf,1)<0)
////    //    perror("read()");
////    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
////    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
////    //buf = getchar();
    int ignoreit = read(0,&buf,1);
    (void) ignoreit;
////    if(tcsetattr(STDIN_FILENO,TCSADRAIN,&newtm)<0)
////        perror ("tcsetattr ~ICANON");
////    //printf("%c\n",buf);    

////    //struct termios oldtm, newtm;
////    //tcgetattr(0, &oldtm);          /* grab old terminal i/o settings */
////    //newtm = oldtm;                 /* make new settings same as old settings */
////    //newtm.c_lflag &= ~ICANON;      /* disable buffered i/o */
////    //newtm.c_lflag &= ~ECHO;        /* clear echo mode */
////    //tcsetattr(0, TCSANOW, &newtm); /* use these new terminal i/o settings now */
////    //buf = getchar();               /* getchar should now act like getch */
////    //tcsetattr(0, TCSANOW, &oldtm); /* reset to old terminal style */

    if ( buf != '\0' )
    {
        static int keycnt = 0;

        if ( keycnt++ > FLUSHINTERVAL )
        {
            keycnt = 0;
            exitnonblockmode();
            enternonblockmode();
        }
    }

    #endif

    return buf;
}


int rabbithole(int scrwidth);
void minesweeper(int gamewidth, int gameheight, int nummines);
void snakes(int gamewidth, int gameheight, int numrabbits, int startsnakelen, int addrate, int usleeptime);

#define GAMEWIDE 30
#define GAMEHIGH 16
#define NUMMINES 99

#define SCREENWIDTH  63
#define SCREENHEIGHT 16

void showvars(double **uservars, const char **varnames, const char **vardescr);
void setvar(const std::string &varname, double val, double **uservars, const char **varnames, const char **vardescr);

int interactmenu(int &goupone, const char *stateDescr, double **uservars, const char **varnames, const char **vardescr)
{
    {
        const char credits[] = "TWNIfbwz; b tvqqpsu wfdups nbdijof mjcsbsz )wfstjpo 8/1*/\n"
                               "\n"
                               "Bvuips; Bmjtubjs Tijmupo/\n"
                               "Uibolt; Ebojfm Mbj )E3D bmhpsjuin*\n"
                               "\n"
                               "Cjcufy; Anjtd|twnifbwz-\n"
                               "            bvuips > #Tijmupo- Bmjtubjs#-\n"
                               "            ujumf  > #|TWNIfbwz~; b Tvqqpsu Wfdups Nbdijof Mjcsbsz#-\n"
                               "            zfbs   > #3127#~\n";

        std::string temp;
        double tempval;
        int gamewide = GAMEWIDE;
        int gamehigh = GAMEHIGH;
        int nummines = NUMMINES;
        std::string linebuff;
        int firstrun = 1;

        int scrrows = SCREENHEIGHT;
        int scrcols = SCREENWIDTH;

        getscrrowcol(scrrows,scrcols);

        while ( 1 )
        {
            if ( firstrun )
            {
                outstream() << "\n";
                outstream() << "Optimisation paused during " << stateDescr << ":\n";
                outstream() << "Screen dimensions " << scrrows << "*" << scrcols << ":\n";
                outstream() << " stop       - stop optimisation new\n";
                outstream() << " cont       - continue optimisation\n";
#ifndef USE_MEX
                outstream() << " quit       - hard-exit to command line\n";
                outstream() << " sys        - execute command via system call\n";
#endif
                outstream() << " up         - go up one level in optimisation\n";
                outstream() << " upup n     - go up n levels in optimisation\n";
                outstream() << " mem        - view memory usage in dynarray (vectors/matrices etc)\n";
                outstream() << " srand      - (srand seed) seed random number generator.\n";
                outstream() << " calc       - simple calculator.\n";
                outstream() << " beep       - Ommmmm.\n";
                outstream() << " dis        - disable pause mode.\n";
                outstream() << " en         - enable pause mode.\n";
                outstream() << " show       - show variables.\n";
                outstream() << " set        - (set name val) set named variable to value.\n";
#ifndef USE_MEX
                outstream() << " mines      - minesweeper.\n";
                outstream() << " minewide w - set width of minesweeper.\n";
                outstream() << " minehigh h - set height of minesweeper.\n";
                outstream() << " minenum n  - set number of mines in minesweeper grid.\n";
                outstream() << " snakes     - snakes.\n";
#endif
                outstream() << " help       - this screen.\n";
                outstream() << " ?          - this screen.\n";
                outstream() << " man        - this screen.\n";
            }

            firstrun = 0;

            int showcredits = 0;

            outstream() << "> ";
            outstream() << '\0'; // Forces cache flush in matlab

            instream() >> temp;
            outstream() << temp << "\n";

                 if ( temp == "help"       ) { firstrun = 1; }
            else if ( temp == "man"        ) { firstrun = 1; }
            else if ( temp == "?"          ) { firstrun = 1; }
            else if ( temp == "stop"       ) { return 1; }
            else if ( temp == "cont"       ) { return 0; }
#ifndef USE_MEX
            else if ( temp == "quit"       ) { exit(1);  }
            else if ( temp == "sys"        ) { std::getline(instream(),linebuff); int res = system(linebuff.c_str()); outstream() << "Return value = " << res << "\n"; }
#endif
            else if ( temp == "up"         ) { goupone = 1; return 1;  }
            else if ( temp == "upup"       ) { instream() >> goupone; return 1;  }
            else if ( temp == "mem"        ) { outstream() << "Memory used: " << memcount() << " bytes.\n"; }
            else if ( temp == "srand"      ) { unsigned int sval; instream() >> sval; svm_srand(sval); }
            else if ( temp == "calc"       ) { gintercalc(outstream(),instream()); }
            else if ( temp == "beep"       ) { errstream() << "\a\a\a"; }
            else if ( temp == "dis"        ) { disablekbquitdet(); }
            else if ( temp == "en"         ) { enablekbquitdet(); }
            else if ( temp == "show"       ) { showvars(uservars,varnames,vardescr); }
            else if ( temp == "set"        ) { instream() >> temp; instream() >> tempval; setvar(temp,tempval,uservars,varnames,vardescr); }
#ifndef USE_MEX
            else if ( temp == "mines"      ) { minesweeper(gamewide,gamehigh,nummines); }
            else if ( temp == "minewide"   ) { instream() >> gamewide; }
            else if ( temp == "minehigh"   ) { instream() >> gamehigh; }
            else if ( temp == "minenum"    ) { instream() >> nummines; }
            else if ( temp == "snakes"     ) { snakes(scrcols,scrrows,(20*scrrows)/23,5,20,100000); }
            else if ( temp == "game"       ) { rabbithole(scrcols); }
#endif
            else if ( temp == "towel"      ) { outstream() << "Present.\n"; }
            else if ( temp == "sunglasses" ) { outstream() << "Peril sensitive!\n"; }
            else if ( temp == "help"       ) { outstream() << "Are you drowning or waving?\n"; }
            else if ( temp == "panic"      ) { outstream() << "OK!  *runs around waving hands in air*\n"; }
            else if ( temp == "thesis"     ) { outstream() << "Who dares mention to \"T\" word?\n"; }
            else if ( temp == "row"        ) { outstream() << "...row row your boat, gently up the stream.\n"; }
            else if ( temp == "bort"       ) { outstream() << "Bort bort bort!\n"; }
            else if ( temp == "neddy"      ) { outstream() << "What what what what what?!?!\n"; }
            else if ( temp == "credits"    ) { showcredits = 1; }

            else
            {
                outstream() << randomquote() << "\n";
            }

            if ( showcredits )
            {
                int i = 0;

                while ( credits[i] != '\0' )
                {
                    if ( ( credits[i] == ' ' ) || ( credits[i] == '\n' ) )
                    {
                        outstream() << credits[i];
                    }

                    else
                    {
                        char tempcred = credits[i]-1;

                        outstream() << (char) tempcred;
                    }

                    i++;
                }
            }
        }
    }

    return 0;
}

void showvars(double **uservars, const char **varnames, const char **vardescr)
{
    if ( ( uservars == NULL ) || ( uservars[0] == NULL ) )
    {
        outstream() << "Okay, folks, show’s over. Nothing to see here, show’s... Oh my God! A horrible plane crash! Hey, everybody, get a load of this flaming wreckage! Come on, crowd around, crowd around! Don’t be shy, crowd around!\n";
    }

    else
    {
        NiceAssert( varnames );
        NiceAssert( vardescr );

        int i = 0;

        while ( uservars[i] != NULL )
        {
            outstream() << varnames[i] << "\t = " << *uservars[i] << "\t(" << vardescr[i] << ")\n";

            i++;
        }
    }

    return;
}

void setvar(const std::string &varname, double val, double **uservars, const char **varnames, const char **vardescr)
{
    (void) vardescr;

    int isset = 0;

    if ( uservars == NULL )
    {
        ;
    }

    else if ( uservars[0] == NULL )
    {
        ;
    }

    else
    {
        NiceAssert( varnames );
        NiceAssert( vardescr );

        int i = 0;

        while ( uservars[i] != NULL )
        {
            if ( varname == varnames[i] )
            {
                *uservars[i] = val;
                isset = 1;

                break;
            }

            i++;
        }
    }

    if ( !isset )
    {
        outstream() << "Error: unknown variable\n";
    }

    return;
}

















// Snakes starts here!

//void snakes(int gamewidth, int gameheight, int numrabbits, int startsnakelen, int addrate, int usleeptime)
void snakes(int gameheight, int gamewidth, int numrabbits, int startsnakelen, int addrate, int usleeptime)
{
    static int highscore = 0;

    gamewidth--;
    gameheight--;

    int i,j,k;

    // Memory allocation

    int *upline;
    char **screenmap;

    MEMNEWARRAY(upline,int,gamewidth);
    MEMNEWARRAY(screenmap,char *,gamewidth);

    for ( i = 0 ; i < gamewidth ; i++ )
    {
        MEMNEWARRAY(screenmap[i],char,gameheight+1);
    }

    int maxsnakelen = (gamewidth-2)*(gameheight-2);
    int *snakexpos;
    int *snakeypos;

    MEMNEWARRAY(snakexpos,int,maxsnakelen+1);
    MEMNEWARRAY(snakeypos,int,maxsnakelen+1);

    enternonblockmode();

    while ( 1 )
    {
    int snakelen = startsnakelen;

    // Setup screen map

    for ( i = 0 ; i < gamewidth ; i++ )
    {
        upline[i] = 1;

        for ( j = 0 ; j < gameheight ; j++ )
        {
            if ( ( !i || ( i == gamewidth-1 ) ) && ( !j || ( j == gameheight-1 ) ) )
            {
                screenmap[i][j] = '+';
            }

            else if ( ( !i || ( i == gamewidth-1 ) ) )
            {
                screenmap[i][j] = '-';
            }

            else if ( ( !j || ( j == gameheight-1 ) ) )
            {
                screenmap[i][j] = '|';
            }

            else
            {
                screenmap[i][j] = ' ';
            }
        }

        screenmap[i][gameheight] = '\0';
    }

    // Set up snake

    //int maxsnakelen = (gamewidth-2)*(gameheight-2);
    int snakedir = 0; // 0 = right, 1 = left, 2 = down, 3 = up

    for ( k = 0 ; k < maxsnakelen ; k++ )
    {
        if ( k < snakelen-1 )
        {
            snakexpos[k] = ((gamewidth+snakelen)/2)+k;
            snakeypos[k] = gameheight/2;

            screenmap[snakexpos[k]][snakeypos[k]] = '#';
        }

        else if ( k == snakelen-1 )
        {
            snakexpos[k] = ((gamewidth+snakelen)/2)+k;
            snakeypos[k] = gameheight/2;

            screenmap[snakexpos[k]][snakeypos[k]] = 'o';
        }

        else
        {
            snakexpos[k] = -1;
            snakeypos[k] = -1;
        }
    }

    // Randomly place rabbits

    for ( k = 0 ; k < numrabbits ; k++ )
    {
        i = (svm_rand()%(gamewidth-2))+1;
        j = (svm_rand()%(gameheight-2))+1;

        screenmap[i][j] = '?';
    }

    // Main game loop

    int alive = 1;
    char keypress;
    int itcnt = 0;
    double timetakensec;

    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    int timeout = 0;
    int fastscrn = 0;

    outstream() << std::endl << std::flush;

    while ( alive )
    {
        itcnt++;

        // Redraw

        svmclrscr(fastscrn++);

        for ( i = 0 ; i < gamewidth ; i++ )
        {
            if ( upline[i] )
            {
                //for ( j = 0 ; j < gameheight ; j++ )
                //{
                //    outstream() << screenmap[i][j];
                //}

                outstream() << screenmap[i];

                upline[i] = 0;
            }

            outstream() << "\n";
        }

        outstream() << "Keys: left/right = ,.  up/down = az  quit = q ::: ";
        outstream() << "Score: " << snakelen << " - High Score: " << highscore << std::flush;

        if ( snakelen > highscore )
        {
            highscore = snakelen;
        }

        // Wait
        // (old version used svm_usleep(usleeptime);)

        timeout = 0;

        while ( !timeout )
        {
            curr_time = TIMECALL;

            timetakensec = TIMEDIFFSEC(curr_time,start_time);

            if ( timetakensec > ((double) usleeptime)/1000000 )
            {
                timeout = 1;
            }
            
            else
            {
                svm_usleep((int) timetakensec/1000000);
            }
        }

        start_time = TIMECALL;

        // Read keys to get direction

        keypress = svm_getch_nonblock();

        // Update direction

        if ( keypress == 'x' )
        {
            break;
        }

        else if ( keypress == 'a' )
        {
            if ( snakedir != 0 )
            {
                snakedir = 1;
            }
        }

        else if ( keypress == 'z' )
        {
            if ( snakedir != 1 )
            {
                snakedir = 0;
            }
        }

        else if ( keypress == ',' )
        {
            if ( snakedir != 2 )
            {
                snakedir = 3;
            }
        }

        else if ( keypress == '.' )
        {
            if ( snakedir != 3 )
            {
                snakedir = 2;
            }
        }

        else if ( keypress == 'q' )
        {
            goto exitnow;
        }

        // Calculate next position for snake head

        if ( snakedir == 0 )
        {
            // right

            i = snakexpos[snakelen-1]+1;
            j = snakeypos[snakelen-1];
        }

        else if ( snakedir == 1 )
        {
            // left

            i = snakexpos[snakelen-1]-1;
            j = snakeypos[snakelen-1];
        }

        else if ( snakedir == 2 )
        {
            // up

            i = snakexpos[snakelen-1];
            j = snakeypos[snakelen-1]+1;
        }

        else if ( snakedir == 3 )
        {
            // down

            i = snakexpos[snakelen-1];
            j = snakeypos[snakelen-1]-1;
        }

        // Update state

        if ( ( screenmap[i][j] == '-' ) ||
             ( screenmap[i][j] == '|' ) ||
             ( screenmap[i][j] == '+' ) ||
             ( screenmap[i][j] == '#' ) ||
             ( screenmap[i][j] == 'o' )    )
        {
            alive = 0;
            break;
        }

        else if ( screenmap[i][j] == '?' )
        {
            screenmap[snakexpos[snakelen-1]][snakeypos[snakelen-1]] = '#';
            upline[snakexpos[snakelen-1]] = 1;

            snakelen++;

            snakexpos[snakelen-1] = i;
            snakeypos[snakelen-1] = j;

            screenmap[snakexpos[snakelen-1]][snakeypos[snakelen-1]] = 'o';
            upline[snakexpos[snakelen-1]] = 1;
        }

        else
        {
            if ( screenmap[snakexpos[0]][snakeypos[0]] == '#' )
            {
                screenmap[snakexpos[0]][snakeypos[0]] = ' ';
                upline[snakexpos[0]] = 1;
            }

            for ( k = 0 ; k < snakelen-1 ; k++ )
            {
                snakexpos[k] = snakexpos[k+1];
                snakeypos[k] = snakeypos[k+1];
            }

            snakexpos[snakelen-1] = i;
            snakeypos[snakelen-1] = j;

            screenmap[snakexpos[snakelen-2]][snakeypos[snakelen-2]] = '#';
            screenmap[snakexpos[snakelen-1]][snakeypos[snakelen-1]] = 'o';
            upline[snakexpos[snakelen-2]] = 1;
            upline[snakexpos[snakelen-1]] = 1;
        }

        // Add rabbits

        if ( itcnt >= addrate )
        {
            itcnt = 0;

            do
            {
                i = (svm_rand()%(gamewidth-2))+1;
                j = (svm_rand()%(gameheight-2))+1;

                screenmap[i][j] = '?';
                upline[i] = 1;
            }
            while ( screenmap[i][j] == 'o' );

            screenmap[i][j] = '?';
            upline[i] = 1;
        }
    }
    }

exitnow:

    exitnonblockmode();

    outstream() << "\n";

    for ( i = 0 ; i < gamewidth ; i++ )
    {
        MEMDEL(screenmap[i]);
    }

    MEMDEL(screenmap);

    MEMDEL(snakexpos);
    MEMDEL(snakeypos);

    MEMDEL(upline);

    return;
}





















// Minesweeper starts here!

int clearnear(int i, int j, int **gamegrid, int gamewidth, int gameheight);

void minesweeper(int gamewidth, int gameheight, int nummines)
{
    int **gamegrid;
    int i,j,k;
    std::string usercomm;

    MEMNEWARRAY(gamegrid,int *,gameheight+2);

    for ( i = 0 ; i < gameheight+2 ; i++ )
    {
        MEMNEWARRAY(gamegrid[i],int,gamewidth+2);
    }

    overagain:

    for ( i = 0 ; i < gameheight+2 ; i++ )
    {
        for ( j = 0 ; j < gamewidth+2 ; j++ )
        {
            gamegrid[i][j] = 0;
        }
    }

    for ( k = 0 ; k < nummines ; k++ )
    {
        i = 0;
        j = 0;

        while ( gamegrid[i][j] || !i || !j )
        {
            i = (svm_rand()%gameheight)+1;
            j = (svm_rand()%gamewidth)+1;
        }

        gamegrid[i][j] = 1;
    }

    int alive = 1;
    int ecnt = 0;

    while ( alive )
    {
        svmclrscr(0);

        for ( i = 0 ; i < gameheight+1 ; i++ )
        {
            for ( j = 0 ; j < gamewidth+1 ; j++ )
            {
                if ( !i && !j )
                {
                    outstream() << "  ";
                }

                else if ( !i )
                {
                    if ( j/10 )
                    {
                        outstream() << j;
                    }

                    else
                    {
                        outstream() << " " << j;
                    }
                }

                else if ( !j )
                {
                    if ( i/10 )
                    {
                        outstream() << i;
                    }

                    else
                    {
                        outstream() << " " << i;
                    }
                }

                else if ( !gamegrid[i][j] || ( gamegrid[i][j] == 1 ) )
                {
                    outstream() << " x";
                }

                else if ( gamegrid[i][j] == 2 )
                {
                    ecnt++;

                    k  = ( gamegrid[i-1][j-1] == 1 );
                    k += ( gamegrid[i-1][j  ] == 1 );
                    k += ( gamegrid[i-1][j+1] == 1 );
                    k += ( gamegrid[i  ][j-1] == 1 );
                    k += ( gamegrid[i  ][j+1] == 1 );
                    k += ( gamegrid[i+1][j-1] == 1 );
                    k += ( gamegrid[i+1][j  ] == 1 );
                    k += ( gamegrid[i+1][j+1] == 1 );

                    if ( k )
                    {
                        outstream() << " " << k;
                    }

                    else
                    {
                        outstream() << "  ";
                    }
                }

                //outstream() << " ";
            }

            outstream() << "\n";
        }

        if ( ecnt == (gameheight*gamewidth)-nummines )
        {
            outstream() << "You won!\n";
            alive = 0;
        }

        int goodmove = 0;
        int moveoutcome = 0;

        while ( !goodmove )
        {
            outstream() << "Your move (format i j): ";
            tryagain:
            std::getline(instream(),usercomm);

            for ( i = 0 ; i < (int) usercomm.length() ; i++ )
            {
                if ( ( usercomm[i] != '0' ) && ( usercomm[i] != '1'  ) &&
                     ( usercomm[i] != '2' ) && ( usercomm[i] != '3'  ) &&
                     ( usercomm[i] != '4' ) && ( usercomm[i] != '5'  ) &&
                     ( usercomm[i] != '6' ) && ( usercomm[i] != '7'  ) &&
                     ( usercomm[i] != '8' ) && ( usercomm[i] != '9'  ) &&
                     ( usercomm[i] != ' ' ) && ( usercomm[i] != '\t' )    )
                {
                    outstream() << "Format error - try again\n";
                    goto tryagain;
                }
            }

            std::stringstream resbuffer;
            resbuffer << usercomm;
            resbuffer >> j;
            resbuffer >> i;

            moveoutcome = clearnear(i,j,gamegrid,gamewidth,gameheight);

            if ( moveoutcome != 2 )
            {
                goodmove = 1;
            }
        }

        if ( moveoutcome )
        {
            outstream() << "You hit a mine... play again?\n";
            std::getline(instream(),usercomm);

            if ( ( usercomm[0] != 'y' ) &&
                 ( usercomm[0] != 'Y' )    )
            {
                alive = 0;
            }

            else
            {
                goto overagain;
            }
        }
    }

    for ( i = 0 ; i < gameheight+2 ; i++ )
    {
        MEMDELARRAY(gamegrid[i]);
    }

    MEMDELARRAY(gamegrid);

    return;
}

int clearnear(int i, int j, int **gamegrid, int gamewidth, int gameheight)
{
    int res = 0;

    if ( !i || ( i > gameheight ) || !j || ( j > gamewidth ) )
    {
        outstream() << "Bad move!\n";
        res = 2;
    }

    else if ( !gamegrid[i][j] )
    {
        gamegrid[i][j] = 2;

        int k = 0;

        k  = ( gamegrid[i-1][j-1] == 1 );
        k += ( gamegrid[i-1][j  ] == 1 );
        k += ( gamegrid[i-1][j+1] == 1 );
        k += ( gamegrid[i  ][j-1] == 1 );
        k += ( gamegrid[i  ][j+1] == 1 );
        k += ( gamegrid[i+1][j-1] == 1 );
        k += ( gamegrid[i+1][j  ] == 1 );
        k += ( gamegrid[i+1][j+1] == 1 );

        if ( !k )
        {
            if ( ( i > 1          ) && ( j > 1         ) ) { clearnear(i-1,j-1,gamegrid,gamewidth,gameheight); }
            if ( ( i > 1          )                      ) { clearnear(i-1,j  ,gamegrid,gamewidth,gameheight); }
            if ( ( i > 1          ) && ( j < gamewidth ) ) { clearnear(i-1,j+1,gamegrid,gamewidth,gameheight); }
            if (                       ( j > 1         ) ) { clearnear(i  ,j-1,gamegrid,gamewidth,gameheight); }
            if (                       ( j < gamewidth ) ) { clearnear(i  ,j+1,gamegrid,gamewidth,gameheight); }
            if ( ( i < gameheight ) && ( j > 1         ) ) { clearnear(i+1,j-1,gamegrid,gamewidth,gameheight); }
            if ( ( i < gameheight )                      ) { clearnear(i+1,j  ,gamegrid,gamewidth,gameheight); }
            if ( ( i < gameheight ) && ( j < gamewidth ) ) { clearnear(i+1,j+1,gamegrid,gamewidth,gameheight); }
        }
    }

    else if ( gamegrid[i][j] == 2 )
    {
        ;
    }

    else
    {
        res = 1;
    }

    return res;
}
































// Define number of locations and objects

#define NUMOBJS 4

class localedata
{
public:
    int xn; // index of location north (-1 if none)
    int xs; // index of location south (-1 if none)
    int xe; // index of location east (-1 if none)
    int xw; // index of location west (-1 if none)
    int xu; // index of location up (-1 if none)
    int xd; // index of location down (-1 if none)

    int objhere[NUMOBJS]; // per object.  0 not here, 1 here

    const char *descript; // Location description
};

class objdescr
{
public:
    int portability; // 0 not portable, 1 portable
    int location;    // -1 we have, -2 nowhere, >= 0 that location

    const char *name; // name of object
    const char *descript; // description of object
};

// use location = -1 to see if have object, -2 to see if object nowhere

int nearobject(const char *descr, objdescr &objectdata, int location)
{
    int i;
    int res = 0;

    for ( i = 0 ; i < NUMOBJS ; i++ )
    {
        if ( strcmp(descr,objectdata.name) == 0 )
        {
            if ( location == objectdata.location )
            {
                res = 1;
                break;
            }
        }
    }

    return res;
}


//FIXME need function to return number of words in string
//FIXME need function to copy ith word in string into result










void blockprint(std::string &text, int width);
void blockprint(const char *text, int width);

int makeitemlist(std::string &res, const int *objhere, const objdescr *objectdata, int numobj);
int makeexitlist(std::string &res, const localedata &here);

int rabbithole(int scrwidth)
{
    scrwidth--;

    // NB: we deliberately make this static - potentially multiple users
    // can jump in here and game together.

    svmvolatile static localedata locationdata[] = {
    {  1,-1,-1,-1,-1,-1, { 1,1,1,1 }, "You are sitting at a desk in a antiseptically lit office.  Outside it looks sunny, but inside it feels unnaturally cold." },
    { -1, 0,-1, 0, 0,-1, { 0,0,0,0 }, "You are in the kitchen/dining room.  This kitchen and dining areas are divided by a glass display shelf.  On one side is a laminate table with four chairs, a computer desk, a filing cabinet, a free-standing wooden cupboard and a chest freezer.  On the kitchen stove is an electric oven/cooktop, a kelvinator fridge and many cupboards and drawers." },
    { -1,-1,-1,-1,-1,-1, { 0,0,0,0 }, "You are spiraling in a vortex." }
    };

    svmvolatile static objdescr objectdata[NUMOBJS] = {
    { 0, 1, "coffee",   "cup of cold coffee" },
    { 0, 1, "papers",   "stack of unread papers" },
    { 0, 1, "computer", "old computer" },
    { 0, 1, "button",   "large red button labelled \"do not press\"" }
    };

    // Rest is standard

    int currloc = 0; // current location

    std::string usercomm;
    std::string itemstr;
    std::string exitstr;

    outstream() << "\n\n";

    int looknow = 1;

    while ( 1 )
    {
        if ( looknow )
        {
            looknow = 0;

            makeitemlist(itemstr,(const_cast<localedata &>(locationdata[currloc])).objhere,const_cast<objdescr *>(objectdata),NUMOBJS);
            makeexitlist(exitstr,(const_cast<localedata &>(locationdata[currloc])));

            outstream() << "-------------------------------------------------\n";
            blockprint((locationdata[currloc]).descript,scrwidth);
            outstream() << "\n\n";
            blockprint(itemstr,scrwidth);
            outstream() << "\n\n";
            blockprint(exitstr,scrwidth);
            outstream() << "\n\n";
        }

        outstream() << "> ";
        std::getline(instream(),usercomm);

        if ( ( usercomm == "north" ) || ( usercomm == "n" ) )
        {
            if ( locationdata[currloc].xn == -1 )
            {
                outstream() << "You can't go that way.\n";
            }

            else
            {
                currloc = locationdata[currloc].xn;
                looknow = 1;
            }
        }

        else if ( ( usercomm == "south" ) || ( usercomm == "s" ) )
        {
            if ( locationdata[currloc].xs == -1 )
            {
                outstream() << "You can't go that way.\n";
            }

            else
            {
                currloc = locationdata[currloc].xs;
                looknow = 1;
            }
        }

        else if ( ( usercomm == "east" ) || ( usercomm == "e" ) )
        {
            if ( locationdata[currloc].xe == -1 )
            {
                outstream() << "You can't go that way.\n";
            }

            else
            {
                currloc = locationdata[currloc].xe;
                looknow = 1;
            }
        }

        else if ( ( usercomm == "west" ) || ( usercomm == "w" ) )
        {
            if ( locationdata[currloc].xw == -1 )
            {
                outstream() << "You can't go that way.\n";
            }

            else
            {
                currloc = locationdata[currloc].xw;
                looknow = 1;
            }
        }

        else if ( ( usercomm == "up" ) || ( usercomm == "u" ) )
        {
            if ( locationdata[currloc].xu == -1 )
            {
                outstream() << "You can't go that way.\n";
            }

            else
            {
                currloc = locationdata[currloc].xu;
                looknow = 1;
            }
        }

        else if ( ( usercomm == "down" ) || ( usercomm == "d" ) )
        {
            if ( locationdata[currloc].xd == -1 )
            {
                outstream() << "You can't go that way.\n";
            }

            else
            {
                currloc = locationdata[currloc].xd;
                looknow = 1;
            }
        }

        else if ( usercomm == "drink coffee" )
        {
            outstream() << "Bleh, instant!\n";
        }

        else if ( usercomm == "read papers" )
        {
            outstream() << "You feel suddenly sleepy.\n";
        }

        else if ( usercomm == "use computer" )
        {
            minesweeper(GAMEWIDE,GAMEHIGH,NUMMINES);
            looknow = 1;
        }

        else if ( usercomm == "press button" )
        {
            outstream() << "Your chair explodes suddenly, hurling\n";
            outstream() << "you through the roof and back into reality.";
            outstream() << "Simulation done!\n";
            return 1;
        }

        else if ( usercomm == "quit" )
        {
            outstream() << "Back to reality then?\n";
            return 1;
        }

        else if ( usercomm == "help" )
        {
            outstream() << "Quit to terminate simulation, cont to continue it.\n";
        }

        else if ( usercomm == "look" )
        {
            looknow = 1;
        }

        else
        {
            outstream() << "What's that now?\n";
        }
    }

    return 0;
}



// Find next whitespace, return -1 if no non-whitespace

int findwhite(std::string &text)
{
    int i = 0;

    if ( i >= (int) text.length() )
    {
        return -1;
    }

    while ( isspace(text[i]) )
    {
        if ( ++i >= (int) text.length() )
        {
            return -1;
        }
    }

    while ( !isspace(text[i]) )
    {
        if ( ++i >= (int) text.length() )
        {
            break;
        }
    }

    return i;
}

void blockprint(const char *text, int width)
{
    std::string temp(text);
    blockprint(temp,width);
    return;
}

void blockprint(std::string &text, int width)
{
    int i = 0;
    int j = 0;
    int k;

    while ( 1 )
    {
        while ( isspace(text[i]) )
        {
            outstream() << text[i];

            if ( ++i >= (int) text.length() )
            {
                return;
            }

            ++j;

            if ( j > width )
            {
                outstream() << "\n";
                j = 0;
            }
        }

        text = text.substr(i,text.length()-i);
        i = 0;

        k = findwhite(text);

        if ( !j && ( k > width ) )
        {
            for ( k = 0 ; k < width-1 ; k++ )
            {
                outstream() << text[i];
            }

            outstream() << "-\n";

            text = text.substr(width-1,text.length()-(width-1));
            i = 0;
            j = 0;
        }

        else 
        {
            if ( j+k > width )
            {
                outstream() << "\n";
                j = 0;
            }

            else
            {
                while ( !isspace(text[i]) )
                {
                    outstream() << text[i];

                    if ( ++i >= (int) text.length() )
                    {
                        return;
                    }

                    ++j;
                }


                text = text.substr(i,text.length()-i);
                i = 0;
            }
        }
    }

    return;
}

int makeexitlist(std::string &res, const localedata &here)
{
    int numexits = 0;
    int resint = 0;

    res = "Exits:";

    if ( here.xn != -1 ) { numexits++; }
    if ( here.xs != -1 ) { numexits++; }
    if ( here.xe != -1 ) { numexits++; }
    if ( here.xw != -1 ) { numexits++; }
    if ( here.xu != -1 ) { numexits++; }
    if ( here.xd != -1 ) { numexits++; }

    resint = numexits;

    if ( numexits == 0 )
    {
        res += " none";
    }

    else if ( numexits == 1 )
    {
        if ( here.xn != -1 ) { res += " north"; }
        if ( here.xs != -1 ) { res += " south"; }
        if ( here.xe != -1 ) { res += " east";  }
        if ( here.xw != -1 ) { res += " west";  }
        if ( here.xu != -1 ) { res += " up";    }
        if ( here.xd != -1 ) { res += " down";  }
    }

    else
    {
        if ( here.xn != -1 ) { if ( numexits == resint ) { ; } else if ( numexits > 1 ) { res += ","; } else if ( numexits == 1 ) { res += " and"; } numexits--; res += " north"; }
        if ( here.xs != -1 ) { if ( numexits == resint ) { ; } else if ( numexits > 1 ) { res += ","; } else if ( numexits == 1 ) { res += " and"; } numexits--; res += " south"; }
        if ( here.xe != -1 ) { if ( numexits == resint ) { ; } else if ( numexits > 1 ) { res += ","; } else if ( numexits == 1 ) { res += " and"; } numexits--; res += " east";  }
        if ( here.xw != -1 ) { if ( numexits == resint ) { ; } else if ( numexits > 1 ) { res += ","; } else if ( numexits == 1 ) { res += " and"; } numexits--; res += " west";  }
        if ( here.xu != -1 ) { if ( numexits == resint ) { ; } else if ( numexits > 1 ) { res += ","; } else if ( numexits == 1 ) { res += " and"; } numexits--; res += " up";    }
        if ( here.xd != -1 ) { if ( numexits == resint ) { ; } else if ( numexits > 1 ) { res += ","; } else if ( numexits == 1 ) { res += " and"; } numexits--; res += " down";  }
    }

    res += ".";

    return resint;
}

int makeitemlist(std::string &res, const int *objhere, const objdescr *objectdata, int numobjs)
{
    int numhere = 0;
    int i,j;

    res = "You see: ";

    for ( i = 0 ; i < numobjs ; i++ )
    {
        if ( objhere[i] == 1 )
        {
            numhere++;
        }
    }

    if ( numhere == 0 )
    {
        res += "nothing";
    }

    else
    {
        std::string descprefix;

        j = 0;

        for ( i = 0 ; i < numobjs ; i++ )
        {
            if ( objhere[i] == 1 )
            {
                j++;

                if ( ( objectdata[i].descript[0] == 'a' ) ||
                     ( objectdata[i].descript[0] == 'e' ) ||
                     ( objectdata[i].descript[0] == 'i' ) ||
                     ( objectdata[i].descript[0] == 'o' ) ||
                     ( objectdata[i].descript[0] == 'u' ) ||
                     ( objectdata[i].descript[0] == 'A' ) ||
                     ( objectdata[i].descript[0] == 'E' ) ||
                     ( objectdata[i].descript[0] == 'I' ) ||
                     ( objectdata[i].descript[0] == 'O' ) ||
                     ( objectdata[i].descript[0] == 'U' )    )
                {
                    descprefix = "an ";
                }

                else
                {
                    descprefix = "a ";
                }

                if ( j < numhere )
                {
                    res += descprefix;
                    res += objectdata[i].descript;
                    res += ", ";
                }

                else if ( j > 1 )
                {
                    res += "and ";
                    res += descprefix;
                    res += objectdata[i].descript;
                }

                else
                {
                    res += descprefix;
                    res += objectdata[i].descript;
                }
            }
        }
    }

    res += ".";

    return numhere;
}











const char *randomquotemore(void);

const char *randomquote(void)
{
const static char *goonquotes[] = {
    "Needle nardle noo!",
    "Ying tong iddle i po.",
    "Enter Bluebottle, wearing doublet made from mum's old drawers.",
    "Waits for audience applause, not a sausage. (Applause) Ooh! Sausinges!",
    "You rotten swine, you!",
    "You have deaded me!",
    "Silence! I have drunk my fill of the clapping.",
    "I'm the famous Eccles.",
    "Haaallooooo",
    "Shut up Eccles!",
    "You silly, twisted boy, you.",
    "Here, have a gorilla.",
    "You can't get the wood, you know.",
    "We'll all be murdered in our beds!",
    "He's fallen in the water.",
    "Mr Seagoon? Minnie's been hit with another batter pudding."
    };

    if ( !(svm_rand()%4) )
    {
        return randomquotemore();
    }

    return goonquotes[svm_rand()%16];
}

const char *randomquotemore(void)
{
const static char *goonquotes[] = { "Bluebottle: Me and Eccles know where it's gone, Captain.",
"Eccles: Yeah. We know.",
"Seagoon: Splendid, lads. Tell me where it is and I'll reduce your sentence from two years to four.",
"Bluebottle: Well, it, er, went, um- Thinks: Where did it went? It wented- Eccles?",
"Eccles: Yeah?",
"Bluebottle: Do you remember, Eccles?",
"Eccles: Oh yeah, I remember Eccles.",
"Bluebottle: Well, does he know where it wented?",
"Eccles: I'll ask him: Do you know where it wented?",
"Bluebottle: What does he say, Eccles?",
"Eccles: He hasn't answered yet, I think he's out.",
"Grytpype-Thynne: Oh, Neddie.",
"Seagoon: Curses, I'm spotted.",
"Grytpype-Thynne: Why are you wearing that leopard-skin?",
"Seagoon: So that's why I'm spotted.",
"Grytpype-Thynne: Tell me, where are you taking that gold?",
"Seagoon: I had to think of a good excuse.",
"Grytpype-Thynne: You're stealing it, aren't you, Neddie?",
"Seagoon: Blast! Why didn't I think of that?",
"Grytpype-Thynne: We'll have to give you a week's notice.",
"Seagoon: Why? What have I done?",
"Grytpype-Thynne: Nothing, but we're having to cut down on staff. You see, there's been a robbery. Um, would you get that van started while I get my hat and coat?",
"Seagoon: You coming too?",
"Grytpype-Thynne: There's no point in staying. There's more money in the van than there is in the bank.",
"Seagoon: Very well, we'll be partners.",
"Grytpype-Thynne: Shake.",
"Seagoon: I give you my hand.",
"Grytpype-Thynne: I gave him my foot, it was a fair swap.",
"Announcer: Ten miles he swam. The last three were agony.",
"Seagoon: They were over land. Finally I fell in a heap on the ground. I've no idea who left it there.",
"Bluebottle: Now, man, I was trained in Judo by the great Bert. Using the body as a counter-pivot to displace the opponent, I use the great Bert's method of throwing the opponent to his death! Be warned, Moriarty, one false move and you die by Bert's method!",
"Moriarty: Take that!",
"[thud]",
"Bluebottle: AHOO! Wait till I see that twit Bert.",
"Eccles: You- you hit my friend Bottle again and see what happens!",
"[thud]",
"Bluebottle: AHOO!",
"Eccles: See, that's what happens!",
"Seagoon: We can't stand around here doing nothing. People will think we're workmen!",
"Seagoon: Unexploded German skulls? I hadn't thought of that.",
"Bloodnok: Elephant soup with squodge spuds.",
"Seagoon: I hadn't thought of that either.",
"Bloodnok: Sabrina in the bath.",
"Seagoon: Ha, ha, ha, ha! I do have some spare time.",
"Chisholm: Hairy Scots, tonight we march north to England!",
"Secombe: But England's south!",
"Chisholm: Aye, we're gonna march right round the world and sneak up on them from behind!",
"Seagoon: Well, these earplugs seem to be all right. How much do you want for them?",
"Grytpype-Thynne: 100 pounds.",
"Seagoon: How much do you want for them?",
"Grytpype-Thynne: 100- Aha. Take your earplugs out.",
"Seagoon: Why don't you answer? I asked you, how much do you want for them?",
"Grytpype-Thynne: 100 pounds!",
"Seagoon: That's funny. I can't hear him.",
"Grytpype-Thynne: They cost 100- Look, take out the earplugs.",
"Seagoon: Stop all that silly miming, man! How much?",
"Grytpype-Thynne: 100 pounds!",
"Seagoon: I've had enough of this, Bloodnok. He obviously doesn't want to do business. Come on, get out, get out! Get out! You steaming English idiot.",
"Grytpype-Thynne: No, no, no! Look here-",
"[door slams]",
"Seagoon: 100 pounds for earplugs we can hear through? Ha, ha, ha, ha! Ha, ha! Not likely!",
"Seagoon: Major Bloodnok! I didn't recognise you in that false room!",
"Bloodnok: Well I was only wearing it to keep the rain off. I wouldn't wear it out of doors, of course.",
"Seagoon: Of course. Let me help you off with it.",
"Bloodnok: Thank you. Good heavens! We're outside and it's raining in the direction of down!",
"Seagoon: We'd better put your room on in the direction of on!",
"Bloodnok: That's better. It's much warmer with this direction on. Now Neddie, pull up a chair and sit down.",
"Seagoon: I'd rather stand, if you don't mind.",
"Bloodnok: Well, pull up a floor then.",
"Bloodnok: It's a copper.",
"Spriggs: I'm not a policeman!",
"Bloodnok: I beg your pardon, madam.",
"Spriggs: I'm not a policewoman either!",
"Bloodnok: I say, you're cutting it fine, aren't you?",
"Spriggs: I'll just make out this bill of sale. How do you spell penguin?",
"Seagoon: P-N-Guin.",
"Spriggs: How do you pronounce it?",
"Seagoon: P-E-N-G-U-I-N.",
"Spriggs: I'll write that down. E-Z-L-X-Q. Drat this pen! It can't spell.",
"Seagoon: Wait a minute, perhaps it's the ink that can't spell. Let me taste it.",
"Spriggs: Righto Jim, righto Jim.",
"Seagoon: P-E-N-G- No, no, this ink's all right.",
"Bloodnok: I'll turn a deaf ear.",
"Seagoon: I didn't know you had a deaf ear.",
"Bloodnok: Yes, I found it on the floor of a barber's shop.",
"William: Well, sir, it's like this, see. At 12.30 a monster lorry draws up outside, ten men jump out and wallop me on the head. I turn round to see who it was, and wallop, wallop, on my head again. I stood up, you see, have a quick barder, no-one there and wallop, wallop, wallop, all on my head! As I took out my notebook, all official like, wallop! Wallop, wallop, on my head, all wallops all over my head. And then-",
"Seagoon: Yes, yes, yes, but did you notice anything about these men?",
"William: Yes.",
"Seagoon: What?",
"William: I noticed they kept walloping me on the head.",
"Bloodnok: Still too dark to see a thing. Thurn me blins! Who is it? Hands up!",
"Eccles: I can't put my hands up, I-",
"Bloodnok: Hands up or I fire!",
"Eccles: OK!",
"[crash]",
"Eccles: OWWW!",
"Bloodnok: Now what's happened?",
"Eccles: I was riding a bike!",
"Bloodnok: What's the matter with you this morning, Seagoon? Why have you got such a long face?",
"Seagoon: Heavy dentures, sir.",
"Bloodnok: I see. Well, have you seen a doctor?",
"Seagoon: Yes, I just saw one walking down the road.",
"Seagoon: We've come to disconnect your phone.",
"The Red Bladder: I haven't got one.",
"Seagoon: Don't worry, we've brought one with us.",
"Eccles: Are you Neddie Seagoon?",
"Seagoon: I am.",
"Eccles: Oh, good. You been waiting long?",
"Seagoon: Yes.",
"Eccles: Who for?",
"Seagoon: You, you idiot!",
"Eccles: Oh! Fine.",
"Seagoon: Now, how do I get through the firing line to President Fred's headquarters?",
"Eccles: How do you get there? You go straight up that road there.",
"Seagoon: But they're shooting down it!",
"Eccles: Oh! Don't go that way! You take this road here. They're not shooting up that one.",
"Seagoon: That road doesn't lead to it!",
"Eccles: Oh! Don't take that one!",
"Spriggs: Ding-dong! Clang! Clang! Ding-dong-dang-dang! Hear ye! Ding-dang! Stolen! One bell!.",
"Henry: Here, Minnie, hold my elephant gun.",
"Minnie: I don't know what you brought it for. You can't shoot elephants in England, you know.",
"Henry: And why not?",
"Minnie: They're out of season.",
"Henry: Does this mean we shall have to have pelican for dinner again?",
"Minnie: I fear so, I fear so.",
"Henry: Then I'll risk it. I'll shoot an elephant out of season.",
"Seagoon: We tried using a candle, but it wasn't very bright and we daren't light it.",
"Seagoon: As I swam ashore, I dried myself to save time.",
"Moriarty: I see that ten years in Britain have not changed your Imperial Roman outlook, Caesar.",
"Caesar (G): True, Moriartus, always a Roman eye.",
"Moriarty: Will you take wine?",
"Caesar: No thanks, I think I'll have a half of mild and a packet of crisps.",
"Eccles: I resign! You speak to my secretary! You can't talk to a government minister like that! I won't be out of work long, you'll see! I'll get that Ministry of Fisheries job! You watch! I've kept goldfish!",
"Announcer: Mr. Eccles, Mr. Eccles, we are not for one moment doubting your sincerity. It's just your intelligence that's in question.",
"Eccles: Well, I accept your apology.",
"Moriarty: Be quiet, or I'll tell them who sold those three cardboard tanks.",
"Bloodnok: What? It's all lies! In any case, they never paid me! Is there no honesty? Do you know what happened to me last night?",
"Moriarty: No.",
"Bloodnok: Thank heaven for that!",
"Idiot: Ahar! I've got the pencil.",
"Seagoon: That's a steamroller!",
"Idiot: Is it? I'll kill that blasted store keeper!",
"William: This lamp-post's gonna be a boon. You see, at the present, I have to walk ten miles every night to the one in the village.",
"Seagoon: Eh?",
"William: I keep a dog, you see.",
"Seagoon: One morning in may I was going through an old dustbin, when my valet announced a visitor.",
"Valet: Pardon me sir, there is a visitor to see you.",
"Seagoon: Right. Headstone, put my lunch back in the dustbin and send him in.",
"Valet: This way, sir.",
"Moriarty: Ah, my dear Doctor Seagoon. Allow me. My card.",
"Seagoon: My card.",
"Valet: My card.",
"Moriarty: Snap! And now, my friend, to business. My name is Count Moriarty. Have you ever heard of lurgi?",
"Seagoon: There's no one of that name here.",
"Moriarty: You, you and you alone will go down in history. Think: Louis Pasteur, Madame Curie, and now, you!",
"Seagoon: I agree. But what's Lurgi got to do with me and Pasteur and the other painters?",
"Henry: Do you mind taking those noisy boots off?",
"Eccles: OK!",
"[thud]",
"[thud]",
"Minnie: Ah, that's better.",
"[thud]",
"Minnie: Oh! I didn't know he had three legs, Henry!",
"Henry: He hasn't, Min. He hasn't. He has a one-legged friend. Goodnight, Min.",
"Minnie: Goodnight, buddy.",
"[thud]",
"Minnie: Ooh!",
"Henry: Oh! He's got two one-legged friends!",
"[thud]",
"Minnie: It's that or one three-legged friend.",
"Seagoon: There. I've sawn off all four legs.",
"German 1: Strange. The first time I've known of a piano with four legs.",
"Eccles: Hey! I keep falling down!",
"Seagoon: I'm sorry, I didn't see you standing in that coffee pot.",
"Grytpype-Thynne: I know, we had the lid down.",
"Seagoon: We? Where's your friend?",
"Grytpype-Thynne: He's up the spout.",
"Spriggs: Great Jupiter, mate. Is that thing a flea?",
"William: No, it's an 'orse, mate.",
"Spriggs: A horse?",
"William: Yes.",
"Spriggs: Take his hat off.",
"William: There.",
"Spriggs: You're right, it is a horse.",
"Moriarty: Three days we've stood waist-deep in this ice-bound Loch Lomond. What's the idea, eh?",
"Grytpype-Thynne: Don't you like fishing, Moriarty?",
"Moriarty: Fishing?! Oh-type-oh! We haven't any rods! How do you catch fish like this?",
"Grytpype-Thynne: Well, they've got to die some time. We just wait until then.",
"Interviewer: Get out, you idiot!",
"Eccles: Wait a minute! Wait a minute! But you ain't even heard me speak yet!",
"Interviewer: We'll write to you.",
"Eccles: Well, that's no good, I can't read.",
"Seagoon: Wait! I've got a hunch-",
"Grytpype-Thynne: It suits you!",
"Chisholm: If it's not a rude question, sir, what's it supposed to be?",
"Seagoon: I wish I knew. I'd feel much happier.",
"Chisholm: You said it was to be a mangle.",
"Seagoon: Yes, I know. But I added a bit here and a bit there, and it got completely out of hand.",
"Chisholm: I'll tell you what, mon - you get in the seat and I'll swing the propeller.",
"Bloodnok: Fort Spon will fall any day now.",
"Milligan: But we've just had it wallpapered!",
"Bloodnok: That's no use, I tell you. The defenders are weaponless! Some swine sold the men's rifles to the enemy for ten thousand pounds.",
"Milligan: How much?",
"Bloodnok: Just a minute, I'll count it again.",
"Milligan: You mean-",
"Bloodnok: Yes, ten thousand pounds.",
"Milligan: You mean that those men have only got bullets to defend themselves?",
"Seagoon: Calling, B4. Calling, B4. Hello? Control calling, B4.",
"Bluebottle: Hello, Captain!",
"Seagoon: Is that you, B4?",
"Bluebottle: Yes!",
"Seagoon: Why didn't you answer me, B4?",
"Bluebottle: Because I didn't hear you B4.",
"Seagoon: Listen, warning - do not land at Croydon Airport because it's not there yet.",
"Bluebottle: Righto then.",
"Seagoon: Now, what is your exact position?",
"Bluebottle: I'm lying on my side, with my knees drawn up under my chin.",
"Seagoon: Why?",
"Bluebottle: I'm at home in bed",
"Moriarty: Just get on this bus.",
"Seagoon: Does it go past the house?",
"Moriarty: Yes, but you can jump off.",
"Seagoon: Suddenly, the footman came over and tapped me on the shoulder with his foot.",
"Footman: Pardon me, sir. Colonel Gore would be pleased to see you out on the balcony, sir.",
"Seagoon: Oh, he's out there, is he?",
"Footman: No, he's in here, that's why he'd be pleased to see you out there.",
"Seagoon: Well, I think I'll go out for a breath of fresh air.",
"Footman: Thank you, sir, that'll save us opening the window. Oh, and pardon me, sir, your taxi's outside.",
"Seagoon: I know.",
"Footman: Well please, sir, would you move it on a bit further, sir?",
"Milligan: Someone's coming up the stairs, sir!",
"Bloodnok: What?! Quick, burn this on the fire.",
"Milligan: Right. What is it?",
"Bloodnok: A piece of coal.",
"Seagoon: Here I was, freshly run over with my bagpipes irreparably flattened, and without a remedy. The weight of the steamroller had made a lasting impression on me. I was now two inches thick and twenty-four feet wide. This- this was very awkward. People kept opening and shutting me.",
"Seagoon: Bloodnok, I need your help.",
"Bloodnok: I'm sorry, it's her day off.",
"Officer 2: Gentlemen, I think you should know that we're at war.",
"Grytpype-Thynne: Oh! Was it something we've said?",
"Officer 2: Heavens, no. We want a decent chap to fly to Germany to try and capture one of the enemy... intact.",
"Moriarty: Ah! What's it worth?",
"Officer 2: Well, for the chap who's successful, there'll be a nice little nest-egg waiting for him.",
"Moriarty: Oh? How much in money?",
"Officer 2: No money, I told you. He'll get a nest with an egg in it.",
"Moriarty: I should risk my life for an egg and a nest?",
"Officer 2: Chickens do it all the time!",
"Grytpype-Thynne: The oasis is only ten feet long, they'll never get a battleship in it.",
"Moriarty: They could stand it up on one end.",
"Grytpype-Thynne: The British don't operate that way.",
"Moriarty: Nonsense. I've seen them walking to work like that!",
"Eccles: The wagon train with your wife on board has been attacked by the Indians!",
"Captain: My wife? Is she safe?",
"Eccles: Yeah.",
"Captain: I never did like them Indians.",
"Lt. Hern: Did they follow you?",
"Eccles: Yeah. They were shooting at me all the time. But I just stuck my tongue out at them.",
"Lt. Hern: Get wounded?",
"Eccles: Yeah.",
"Lt. Hern: Where?",
"Eccles: In the tongue.",
"Seagoon: Gad! What will you think of next?",
"Bloodnok: Well, I think I'll say I'm not staying on this ship. I've been beaten, flogged, keel-hauled, mutinied, tarred, hung from the yard-arm, lashed to the mast, and also an unpleasant incident east of the wind.",
"Seagoon: But a sailor must expect these things!",
"Bloodnok: Sailor? I'm a first-class passenger, sir!",
"Seagoon: You're a first-class-",
"Bloodnok: Yes, I know, I know.",
"Seagoon: I need your help!",
"Bloodnok: What? Well, you can stand by me to rely on you.",
"Seagoon: Back in the BBC listening room, I struggled to free myself before the dynamite exploded.",
"Bloodnok: Don't worry, Seagoon.",
"Seagoon: Bloodnok! Eccles!",
"Bloodnok: Quick, untie him.",
"Eccles: OK, I'd better hurry up before the-",
"[explosion]",
"Eccles: That's got his legs free.",
"Seagoon: Yes, but where are they?",
"Eccles: Here they are.",
"Seagoon: Oh, horror of horrors!",
"Eccles: Who, me?",
"Seagoon: Dear faithful old hairy English Tommy! Ten years you waited here rather than disobey that last order I gave you. Stay here till I came back, I said to him. He waited alone in the desert. He never wavered from his duty. He kept the name of servitude shining bright. Eccles - Eccles - you upheld the flag. You never questioned the order. You stayed out here alone. You, without food or water. You, without money. You, without anything to stop you walking away. You! You IDIOT!",
"Seagoon: Now Major, what's all this spaghetti hurling about?",
"Bloodnok: Well, you see, lad, it's the Bloodnok method of ending the war, you see. Each commando is issued with an army sock full of lukewarm spaghetti, and when he meets a Hun full-face, it's WHOOSH - PUTT - NUK - MCNOOL! Right in the square-head's mush. And by the time the Jerries have scraped it off, it's too late! The pubs are all shut, lad!",
"Seagoon: But why use spaghetti?",
"Bloodnok: Don't you see, you military fool? When a German is struck with the full force of spaghetti he'll think the Italians have turned on them, you see!",
"Seagoon: What a brilliantly mediocre idea! You'll get an OBE for this.",
"Bloodnok: Oh good, my last one died.",
"Seagoon: Well, we've all got to go some time.",
"Bloodnok: Yes, I went this morning, it was hell in there.",
"German 1: Montgomery is always flying backwards and forwards between England-",
"German 2: They have planes that fly backwards?",
"German 1: Private Schnutz, I have bad news for you.",
"German 2: Private? I am a general!",
"German 1: That is the bad news.",
"Minnie: Oh, Henry, after all these years, our own piano!",
"Henry: Yes, all our own. At last, we can take a bath.",
"Seagoon: Through the pigeonhole flew a carrier pigeon. There was something attached to its leg. It was a postman.",
"Moriarty: I warn you. I shall count up to a highly skilled forty thousand, and then I'll shoot!",
"Seagoon: Forty thousand?",
"Moriarty: Yes, I've got to go home for my gun.",
"Seagoon: When I saw that he was a dwarf, I was all for attacking him right away, but Bloodnok stopped me.",
"Bloodnok: No, wait till he gets older.",
"Seagoon: Finally, on his ninety-third birthday, we sprang!",
"Henry: Now, let's get some details and documents - we must have documents, you know.",
"Seagoon: Of course.",
"Henry: ... must have documents. Now, what was this all about? Let me - oh yes, yes. Now, your name?",
"Seagoon: Neddie Pugh Seagoon.",
"Henry: N-E-D-D-I-E Neddie. What was next?",
"Seagoon: Neddie Pugh Seagoon.",
"Henry: Pugh, P-H-E-W.",
"Seagoon: No, no, it's pronounced 'Phew' but it's spelt 'Pug'.",
"Henry: Oh, mmmmm, Pug, yes, P-U-G-H.",
"Seagoon: Yes.",
"Henry: There, Neddie Pugh . Seagoon wasn't it?",
"Seagoon: Yes, S-E-A-G-O-O-N.",
"Henry: Could you spell it?",
"Seagoon: Certainly: S-E-A-G-O-O-N.",
"Henry: Seagoon, S-E-A, er, mnkk ... mnkk (goes to sleep.)",
"Seagoon: G-O-O-N, Seagoon.",
"Henry: O-O-N . aahhh good, good, good . there, ZZZZZ (snoring). Oh yes yes yes, good, yes, the full name. Now then, address?",
"Seagoon: No fixed abode.",
"Henry: No ... F-I-X-E-D, fixed ... A-B ...",
"Seagoon: A-B-O-D-E.",
"Henry: O-D-E. There we are, no fixed abode. What number?",
"Seagoon: 29A.",
"Henry: Twenty nine A.",
"Seagoon: Yes.",
"Henry: Twenty nine ... A. District?",
"Seagoon: London, S.W.2.",
"Henry: L-O-N-D-O-N, Southwest. E-S-T ... Two, wasn't it?",
"Seagoon: Yes, two.",
"Henry: T-W- ... It's no good, I'd better get a pencil and paper and write all this down.",
"Announcer: Neddie Seagoon arrived at the port of Guatemala, where he was accorded the typical Latin welcome to an Englishman.",
"Moriarty: Hands up, you pig swine. [spits]",
"Seagoon: Have a care, Latin devil. I am an Englishman. Remember, this rolled umbrella has more uses than one!",
"Moriarty: Ooh!",
"Seagoon: Sorry.",
"Seagoon: Any cases of frozen feet?",
"Eccles: You didn't order any cases of frozen feet!",
"Bloodnok: Mount Everest- it's five miles high, isn't it? Yes?",
"Seagoon: Yes.",
"Bloodnok: But it measures twelve miles across the bottom!",
"Seagoon: Well?",
"Bloodnok: Well, all we need to do is to tip Mount Everest on its side, and we'll have a mountain twelve miles high!",
"Seagoon: How do you intend tipping Mount Everest on its side?",
"Bloodnok: Well, isn't it obvious?",
"Seagoon: No.",
"[pause]",
"Bloodnok: Then I have another idea.",
"Seagoon: He was a tall, vile man, dressed in the naval uniform of a sea-going sailor. Under his left arm he held a neatly rolled anchor, while with his right he scanned the horizon with a pair of powerful kippers.",
"Sellers: Inside, it was pitch black and dark as well. To make it worse, there were no lights on. Luckily, the tunnel was only twenty yards wide, so Ned Seagoon was able to stretch out his arms and feel his way along both sides.",
"Moriarty: I might say whoever planned the robbery must be a man of the highest intelligence, with the courage of a lion!",
"Seagoon: So you suspect me?",
"Moriarty: No.",
"Seagoon: Ole.",
"Moriarty: Ole.",
"Seagoon: Ole.",
"Moriarty: A Britisher has already been encasseroled in the Madrid jail and sentenced to 94 years, Senor.",
"Seagoon: So he was found guilty, eh?",
"Moriarty: I don't know, they haven't tried him yet.",
"Seagoon: Do you think they suspect him?",
"Moriarty: That's difficult to say.",
"Seagoon: Do-you-think-they-suspect-him. Hm, it is a bit difficult to say, yes.",
"Grytpype-Thynne: Nephew Neddie! Enjoying the ball?",
"Seagoon: Immensely! I've danced every dance!",
"Grytpype-Thynne: Oh? Who's the lucky girl?",
"Seagoon: Oh, I don't bother with them! I'm much better on my own!",
"Seagoon: The day before the valuable Westminster Pier sank, it was inspected and certified river-worthy!",
"Milligan: Who was the man who inspected it?",
"Sellers: It was none other than-",
"Seagoon: I resign!",
"Sellers: Resignation accepted... on the grounds of incompetence. Anyone else want the old job there?",
"Seagoon: I'll take it on.",
"Sellers: Right, name?",
"Seagoon: Ned Seagoon.",
"Sellers: Same as the last bloke. All right, carry on.",
"Moriarty: Pardon me, my ami. Mon card.",
"Seagoon: Thank you... but there's nothing on it!",
"Moriarty: Look on the other side.",
"Seagoon: Oh! A silly place to have it printed. On the back. Now what's this? 'Messrs Fred Moriarty Limited, Sunken Westminster Floating Pier Salvage Experts'? Gad! Just the man we want!",
"Moriarty: Sapristi! You mean the Westminster floating pier has sunk?",
"Seagoon: Yes!",
"Moriarty: At last! Employment!",
"Bloodnok: Seagoon! Yes, of course, I remember! Didn't your father have a son?",
"Seagoon: Oh, aha, I never asked him about his private affairs.",
"Bloodnok: Seagoon, of course, of course, yes. I knew your father before you were born!",
"Seagoon: I didn't.",
"Bloodnok: I wish you had, things might have been different.",
"Seagoon: What did this attacker look like?",
"William: I dunno, I dunno, I didn't see him, mate.",
"Seagoon: I see. And would you recognise him if you didn't see him again?",
"William: Straight away! Although you know, sir, I must admit, me eyes ain't what they used to be.",
"Seagoon: No?",
"William: No, they used to be me ears.",
"Seagoon: London, 1901. That was a good year for England. Well, we'd have looked silly without it, wouldn't we?",
"Seagoon: I want to buy a twenty-foot easel.",
"Henry: Twenty-foot? Whatever for?",
"Seagoon: I want people to think I'm tall.",
"Henry: But if you stand by a twenty-foot easel, it'll make you look even shorter.",
"Seagoon: That's just it. I'm not going to stand by it. I'll stand somewhere else. Ha, ha, I'm not a fool, you know!",
"Henry: If you're not going to stand near it, why buy it?",
"Seagoon: I've got to buy it so as to have something tall not to stand by! It's no good not standing by something tall that's not there, is it?",
"Henry: Supposing someone comes in unexpectedly when you're standing near it?",
"Seagoon: Then I shall deny every word of it and stand on a ladder.",
"Fifi: Don't try and fight it, darling. This is bigger than both of us. Look!",
"Seagoon: Gad! A photo of the Eiffel Tower!",
"Seagoon: To try to draw her attention I set fire to myself. It moved her. She fried an egg on me.",
"Seagoon: But wait - there's somebody in the dustbin with me. He's coming over. I'll- I'll pretend I haven't seen him.",
"Chisholm: The black-bearded criminal must have got in through the door or the windows. Everything else was locked.",
"Seagoon: Wait! What's this in the corner?",
"Eccles: Shhh. Don't wake him up.",
"Seagoon: Why not?",
"Eccles: He's asleep.",
"Seagoon: Here, let's have a look... These are the bagpipes!",
"Eccles: Oh! I thought it was a spider in a tartan sweater!",
"Bluebottle: Throws large stone. Forgets to let go, hits head on tree. AHOOO!",
"Grytpype-Thynne: Dear, dear surgeon, you have overlooked one terrifying aspect of the dear Count's condition. This man has the Spon Plague.",
"Seagoon: I've never heard of it.",
"Grytpype-Thynne: That's because the Count is the first man to have caught it.",
"Seagoon: Are you sure?",
"Grytpype-Thynne: He has all the symptoms, namely bare knees.",
"Seagoon: Is it catching?",
"Grytpype-Thynne: Yes, stand back, please- Oh! I'm too late, yes, you've already caught it.",
"Seagoon: What-what-what?",
"Grytpype-Thynne: You have got the bare knees!",
"Seagoon: No, I haven't!",
"Grytpype-Thynne: Roll your trousers up. There! Bare knees!",
"Seagoon: I've got the Spon!",
"Uncle Oscar: Now, I propose- urgh! [thud]",
"Henry: Oh, dear, oh, he's dead, Min.",
"Minnie: What, again?",
"Henry: Gentlemen, the chairman has just died.",
"[polite applause]",
"Henry: We will send a fresh husband to the widow as soon as the weather permits.",
"Henry: Now, as he was saying... Min. Min, hold this chicken. Be caul, she's- What?",
"Minnie: I don't know why you have to carry a chicken around, Henry.",
"Henry: Well it's the fog, Min. I always carry one when there's a fog.",
"Minnie: What- what for?",
"Henry: Because chickens can't see where they're going in a fog. Unless it's a fog chicken, and there's no such thing as a fog chicken.",
"Minnie: What are you talking about? There was no fog today!",
"Henry: Well, this isn't a fog chicken!",
"Minnie: What?",
"Henry: It's not a fog chicken!",
"Bluebottle: 'Ere, why ain't you got no clothes on?",
"Eccles: I've just been making a phone call.",
"Bluebottle: You don't have to undress for that!",
"Eccles: Ha, ha! We learn something new every day!",
"Grytpype-Thynne: What is this place?",
"Seagoon: The Victoria and Albert.",
"Grytpype-Thynne: Oh, really? And which one are you?",
"Seagoon: I'm neither.",
"Grytpype-Thynne: I'm pleased to meet you.",
"Bloodnok: This pound note. What colour was it?",
"Seagoon: Green.",
"Bloodnok: It's mine! Mine was green!",
"Seagoon: I daren't attack now, they're too many. I'll wait till they're both gone, and then I'll spring!",
"Announcer: Erm... yes. Well now, here is an announcement for listeners still wondering why this programme was called 'Drums Along the Mersey'. While the programme was being broadcast, there were in fact several drums beating along the Mersey. Those with their windows open may have heard them.",
"Seagoon: Could... could I do the job?",
"Sellers: Do it? It's right up your street.",
"Seagoon: Well, that'll save bus fares.",
"Eccles: Look up there! There's buzzards circling- there's buzzards circling around!",
"Seagoon: What are they doing up there?",
"Eccles: Flying!",
"Seagoon: Bloodnok- Bloodnok, do you think they're waiting- waiting to eat us?",
"Bloodnok: Not sure, but keep your eyes on the ones carrying knives and forks.",
"Seagoon: Look! We're saved! Look! A house!",
"Eccles: It is! A house! A house!",
"Bloodnok: It's not, it's a mirage.",
"Seagoon: Nonsense, it's a house surrounded by trees. Let's go in.",
"Eccles: Yeah.",
"[sound of door opening]",
"Bloodnok: I still say it's a mirage.",
"Seagoon: Nonsense! Bluebottle, Eccles, search the house for food.",
"Bluebottle: All right, then.",
"Seagoon: So, Bloodnok. You think this house is a mirage, eh? We'll soon see! Wait! It's vanished! Gone! You were right. A mirage.",
"Bloodnok: I told you it was.",
"Eccles: OWWWWWW- [thud] OW!",
"Bloodnok: Eccles! What happened?",
"Eccles: I was upstairs!",
"Seagoon: What? A fake bullet-hole? What does this mean?",
"Grytpype-Thynne: He was murdered by a fake bullet.",
"Seagoon: The ambulance is outside.",
"Eccles: Ambulance? I'm not sick!",
"Seagoon: You will be, it's going to run over you.",
"Seagoon: Mr. Crun! 2 o'clock! Time for your revenge!",
"Henry: You're right, we must save my Modern Min from Ancient Bloodnok.",
"Seagoon: Yes. Here, put this bomb in his coffee.",
"Henry: Won't it keep him awake?",
"Henry: Is this an official visit?",
"Seagoon: I'm afraid you'll have to put your helmet on.",
"Henry: Oh, dear, that'll mean re-potting the geranium.",
"Minnie: And the baby, too.",
"Seagoon: Is this the place where there's been a murder?",
"Grytpype-Thynne: Yes. Which murder are you enquiring about?",
"Seagoon: Which murder? How many have there been?",
"Grytpype-Thynne: One.",
"Seagoon: That's the one.",
"Bluebottle: Corpse? Did you say that was a corpse, my Captain? AHOO! Turn white, ears turn green, hairs fall out, legs drop off, feels faint but manages to hold onto drainpipe.",
"Seagoon: Listen, Auntie Min and Uncle Hen. I know you love children, but isn't it time I was weaned?",
"Henry: Listen, Min, he's trying to talk!",
"Henry: You get on baiting those elephant traps.",
"Minnie: I don't see the point of them, you know.",
"Henry: What?",
"Minnie: We've never caught one.",
"Henry: That doesn't mean we must stop trying, Min of mine. Think of the dangers! Supposing you came down one morning for a cream-strainer, and found an elephant in the larder, eh?",
"Minnie: Well, I've never seen an elephant in the larder.",
"Henry: That is because they're hiding, Min of mine.",
"Minnie: Where do elephants hide? Tell me that! Where do elephants hide, buddy?",
"Henry: Well, I don't know, Saxophone-Min, but it's clear to me that they must hide somewhere. How else could they get away with it for so long?",
"Henry: Let me take your hat and coat.",
"Seagoon: Thank you.",
"Henry: Min - throw these on the fire.",
"Henry: Stop! Stop, stop! This spoon is out of tune, Min. Have you been eating with it again?",
"Minnie: No.",
"Henry: Then what's that you're stirring the soup with?",
"Minnie: A violin.",
"Bloodnok: I claim the South Pole in the name of Gladys Ploog of 13 The Sebastibal Villas, Sutton.",
"Seagoon: Who is she, sir?",
"Bloodnok: I don't know, but obviously we're doing her a big favour.",
"Eccles: Hello?",
"Seagoon: Hello?",
"Eccles: Snap.",
"Seagoon: Splendid, ring again tomorrow and we'll have another game",
"Moriarty: Blat! Thud! Blat! Blam!",
"Bloodnok: Club! Whack! Robin, we can't keep this up much longer. Will they never arrive?",
"Seagoon: Who?",
"Bloodnok: Those blasted sound effects men.",
"Announcer: For years we heard nothing from Neddie. And then, one day...",
"Sellers: We heard nothing from him again.",
"Announcer: We put a light in the window. Nothing much happened, except the house burnt down.",
"Seagoon: Keep your chin up, Major!",
"Bloodnok: Why?",
"Seagoon: It's in the soup.",
"Seagoon: Knock-knock!",
"Henry: Who is it?",
"Seagoon: Short man, can't reach the knocker.",
"Seagoon: What's the date today?",
"Eccles: 24th of December! Christmas Eve!",
"Seagoon: So they've both fallen on the same day! Must be slippery.",
"Henry: Yes, well, I don't think we can wait any longer for any more laughs on that one. Now, back to work or I'll belt your nut in!",
"Seagoon: But Mr. Scrooge, it's Christmas Eve, a time of good will and custard!",
"Henry: So it is. Merry Christmas, Scratchit.",
"Seagoon: Merry Christmas!",
"Henry: Now, get back to your desk or I'll belt your nut in!",
"Seagoon: Please, Mr. Scrooge, can't I go home two seconds early tonight?",
"Henry: Two seconds? You must be mad!",
"Announcer: It was three in the morning and two in the afternoon, making a grand total of five in the evening.",
"Minnie: Come on boy, beg for your supper. Up, up, sit up, sit up. Put this sausage on your nose. There, there's a clever boy.",
"Henry: Minnie.",
"Minnie: What?",
"Henry: I'm fed up having my breakfast like this.",
"Moriarty: OWWW. OWWW.",
"Grytpype-Thynne: That's your pair of OWWWs complete for the day.",
"Seagoon: For an hour we ran in French, which I ran fluently.",
"Minnie: Boil, cauldron, boil. Oh! Eye of newt, leg of toad, eagle's knee, shell of snail. Eeheeheeheehee.",
"Henry: Mistress Bannister. What is that hellish fiend brew?",
"Minnie: It's your laundry, Henry.",
"Servant: Pardon me, would you like your coffee on the balcony?",
"Grytpype-Thynne: Haven't you any cups?",
"Bluebottle: Eccles, save me!",
"Eccles: Where are you?",
"Bluebottle: In the water in 1957!",
"Eccles: Oh, I can't help you then.",
"Bluebottle: Why not?",
"Eccles: I'm in 1600.",
"Bluebottle: You can't be in there in that 1600 there! I can see you quite clearly.",
"Eccles: Ah, but in 1957 you got all them good National Health spectacles.",
"Bluebottle: Well, you can borrow mine, and be the man sees no-one touches them, and then you can pull me up!",
"Eccles: I don't know what he means... but I can't do that. I'm not really... I'm really, I'm really not here.",
"Bluebottle: What do you mean by that, my good man?",
"Eccles: I'll tell you, my good man. If... this is 1957. You said this is 1957? Say yes.",
"Bluebottle: Yes.",
"Eccles: Well if this is 1957, I'm dead.",
"Bluebottle: Then why are you standing up?",
"Eccles: Um. Well, I'm not in- oh, ah! I'll tell you why I'm standing up. 'Cos I'm in 1600 and you're not born yet.",
"Bluebottle: Well, wait till I tell my Mum that. My Dad won't half cop it.",
"Minnie: Henry, the dog wants to come in.",
"Henry: That naughty dog, always forgetting his keys. All right, come in, Psycho.",
"Seagoon: Psycho?",
"Henry: Yes, he's our pet mad dog, you know. Come in, you naughty Psycho.",
"Announcer: Woof, woof.",
"Henry: Where have you been, you mad dog?",
"Announcer: Out in the mid-day sun.",
"Seagoon: AHHH! He talks!",
"Henry: I told you he was mad.",
"Seagoon: But dogs can't talk!",
"Henry: I know, I've told him. He never listens. May as well talk to a brick wall, you know.",
"William: Wait a minute, 'ere! You can't fool me about with all that clever talk, mate. You gotta pay for the ticket. Nah, where did you get on?",
"Seagoon: Curse, the game's up. Well now, er... what was that last station?",
"William: Fun Junction.",
"Seagoon: That's it! That's where I got on!",
"William: But we didn't stop there!",
"Seagoon: Do you think it was easy?",
"William: Look, where are you going to?",
"Seagoon: The next station.",
"William: Right, that'll be 18 shillings and thrupence.",
"Seagoon: Right, there we are.",
"William: Thank you.",
"Seagoon: Fool. Ha, ha. Little does he know that the real fare is not 18 and thrupence but 32 pounds, six shillings.",
"William: Little does he know that I'm nothing to do with the railway at all.",
"Bluebottle: Unscrews false kneecap, takes out secret gun. Am in agony, as I have not got false kneecaps. Puts on bold face. AHEE! It still hurts, though.",
"Seagoon: Rather than surrender we gave ourselves up.",
"Cecile Chevreau: I only had eyes for him and he only had eyes for me.",
"Bloodnok: That explains why we fell over a cliff.",
"Seagoon: And furthermore, there is discontent among the troops.",
"Sellers: Lieutenant Seagoon! You say there is discontent among the troops?",
"Seagoon: Yes, there is discontent among the troops.",
"Sellers: Why do you say there is discontent among the troops?",
"Seagoon: Because there is discontent among the troops.",
"Sellers: I see. You say there is discontent among the troops because there is discontent among the troops.",
"Seagoon: Yes. I say there is discontent among the troops because there is discontent among the troops.",
"Sellers: Yes, well, it all sounds reasonable to me.",
"Henry: I'm washing the dinner plates, Min.",
"Minnie: But we haven't had dinner yet, Henry.",
"Henry: Ah, but I'm washing them now so that we won't have to wash them after.",
"[loud noise]",
"Minnie: Oh! Was that you, Henry?",
"Henry: No, that was the elephant, Min.",
"Minnie: What's the elephant doing in the kitchen?",
"Henry: Helping, Min.",
"Minnie: Is he drying up?",
"Henry: No, he feels quite moist, Min. He's cooking the din, Min.",
"Minnie: I told you not to let him cook the dinner. You know that's the gorilla's job. Shoo, naughty elephant, shoo!",
"Minnie: Come on, put your feet up.",
"[crash]",
"Henry: You shouldn't have done that from the standing position.",
"Grytpype-Thynne: Step on it!",
"Seagoon: So saying, he threw down a dog-end.",
"Seagoon: The door was opened by a heavily strained wreck, wearing the string remains of an ankle-loaf vest, a second-hand trilby and both feet in one sock.",
"Bloodnok: Ooh, I've been through hell to get here.",
"Seagoon: There must be a cooler route?",
"Bloodnok: Yes, I was surrounded by a Jap patrol. But I soon had them crawling for me on their hands and knees.",
"Seagoon: How's that?",
"Bloodnok: I hid in a drainpipe. Shhh! There's someone outside the window. Look out!",
"[window smashing]",
"Seagoon: What is it?",
"Bloodnok: It's a gramophone record!",
"Seagoon: Quick! Put it on!",
"Bloodnok: Right!",
"[record of window smashing]",
"Seagoon: (What is it?)",
"Bloodnok: (It's a gramophone record!)",
"Seagoon: (Quick! Put it on!)",
"Bloodnok: (Right!)",
"[record of record of window smashing]",
"Seagoon: ((What is it?))",
"Bloodnok: ((It's a gramophone record!))",
"Seagoon: ((Quick! Put it on!))",
"Bloodnok: ((Right!))",
"[record of record of record of window smashing]",
"Seagoon: (((What is it?)))",
"Bloodnok: (((It's a gramophone record!)))",
"Seagoon: (((Quick! Put it on!)))",
"Bloodnok: (((Right!)))",
"[record of record of record of record of window smashing]",
"Seagoon: ((((What is it?))))",
"Bloodnok: ((((It's a gramophone record!))))",
"Seagoon: ((((Quick! Put it on!))))",
"Bloodnok: ((((Right!)))) Stretch me scallybonkers and flatten me Doreen Lundies! It's a Japanese mirror trick! We shall have to get out of here.",
"Seagoon: Yes, yes, yes. Now, what about the plane?",
"Bloodnok: The plane, the plane... Ooh! Ooh, heavens!",
"Seagoon: What's up?",
"Bloodnok: Ooh, heavens. Look, if I tell, promise you won't blow up.",
"Seagoon: I promise.",
"Bloodnok: I forgot to order it!",
"[explosion]",
"Bloodnok: You promised!",
"Moriarty: My nerves are strained to breaking point.",
"[twang]",
"Moriarty: There goes one now!",
"William: 'Ello, 'ello, ello! Who's this kipping on the floor? What's this label round his neck say? 'I am the new tenant here.' Oh, are you mate! What's this second label say? 'Yes, I am!' Well, I'll just tie this label saying 'Wake up, mate' round his neck...",
"Bloodnok: I bet you five pounds you'll live forever, starting now!",
"[pause]",
"Bloodnok: You've done it! You've lived forever!",
"Seagoon: From the waist downwards, Bloodnok was tattooed with a pair of false legs... facing the wrong way.",
"Grytpype-Thynne: Yes, that was a brilliant idea of mine that you thought of.",
"Announcer: We included that recording of a cockerel for people who like that sort of thing.",
"Home Secretary: Are those rifles loaded?",
"William: No, but we're not telling you that, mate.",
"Minnie: This bed's all right, Henry. It's still got four legs.",
"Henry: Yes, but two of them are mine.",
"Grytpype-Thynne: Now then, Ned - off with your clothes, Neddie.",
"Seagoon: There - how do I look?",
"Moriarty: OWWW.",
"Grytpype-Thynne: I suppose he makes somebody happy. Hold this rice pudding.",
"Seagoon: What's your name?",
"Eccles: Ah, the hard ones first, eh?",
"Bluebottle: Have you ever been on holiday in Corsica before?",
"Eccles: No, but I once made a dog kennel out of elastic.",
"Bluebottle: Oh. There's something to be said for these premium bonds, then.",
"Eccles: Oh.",
"Bluebottle: I think the government is very clever, you know. I won 25 pounds in a premium bond draw.",
"Eccles: What... what's clever about that?",
"Bluebottle: I never bought any premium bonds!",
"Eccles: Oh! And I made a hole in the front.",
"Bluebottle: What for?",
"Eccles: For the dog to get in and out.",
"Bluebottle: Oh! That's nice for the doggie... that's nice and fine for the doggie. I say, Eccles. Why are you not wearing any trousers?",
"Eccles: Well, it's lunchtime.",
"Bluebottle: Oh! What did you have for lunch?",
"Eccles: My trousers.",
"Grytpype-Thynne: Moriarty - go and slam the door in his face.",
"Moriarty: He hasn't got a door in his face.",
"Grytpype-Thynne: Then he's trapped, and he can't get out!",
"[dong!]",
"William: Listen, mate!",
"[dong!]",
"William: There it is again, mate!",
"[dong!]",
"William: And again, mate. Unless I'm mistaken it's gonna go-",
"[dong!]",
"William: -again, mate!",
"Seagoon: I wonder what it is, mate.",
"William: It's a bell ringing, mate.",
"Seagoon: There you go, jumping to conclusions!",
"Bloodnok: I'm lost, dear fellow, lost, completely lost. Me and the regiment were marching along, you know, and suddenly, quite by accident, me and the regimental funds took the wrong turning.",
"Bloodnok: Do you know what happened to me this morning?",
"Seagoon: Yes, I don't know.",
"Bloodnok: A scruffy little urchin threw a kippered herring at me. He threw it at me!",
"Seagoon: Did you close with him?",
"Bloodnok: Of course not. He was only a kid. I mean, he doesn't know any better. Wasn't meaning any harm. Well, I mean, I'd have done it myself when I was young. He was only having fun.",
"Seagoon: Yes, but what did you do?",
"Bloodnok: I threw him under a steamroller!",
"Seagoon: Ah, you sentimental fool.",
"Bloodnok: Yes! I say, you wouldn't care for a rather unique bookmark, would you?",
"Bluebottle: I shall not speak! No words shall pass my lips! Beat me, torture me, burn me with red-hot irons! I will not speak... until it hurts.",
"Seagoon: Mr. Crun, British Railways want you to grow them six thousand acres of mustard and cress - in the Amazon.",
"Henry: Very well, I'll get my hat. Min!",
"Minnie: What did you say?",
"Henry: I'm just going to the Amazon.",
"Minnie: Be caul.",
"Henry: I'll be away for six years, Min.",
"Minnie: I'll put your dinner in the oven, Henry.",
"Seagoon: This is it - build a full-scale cardboard replica of England, anchor it off the coast of Germany, then, when the Germans have invaded it, we tow it out to sea... and pull the plug out.",
"Seagoon: Now, which one of you two is Mr. Crun?",
"Minnie: I'm Miss Bannister.",
"Seagoon: Never mind who you are. Which one is Henry Crun?",
"Minnie: Don't tell him, Henry!",
"Henry: No! I'm not going to tell him, Min. In any case... in any case, why do you want to know my name?",
"Seagoon: Mr. Crun! You make cardboard models and scenery.",
"Henry: If I was Mr. Crun... which I'm not admitting... yes, I do.",
"Seagoon: How are we to get the waterproof gas stove to the garrison? Drop it by helicopter?",
"Bloodnok: Impossible, sir, impossible. The fort is invisible from the air, and worse still...",
"Seagoon: Yes, yes?",
"Bloodnok: The air is invisible from the fort. Oh!",
"Seagoon: By road, then.",
"Bloodnok: No road.",
"Seagoon: Up the river.",
"Bloodnok: No.",
"Seagoon: Down the river.",
"Bloodnok: No.",
"Seagoon: Across the river, into the trees.",
"Bloodnok: No, no.",
"Seagoon: Why not?",
"Bloodnok: No trees.",
"Seagoon: Then across the trees and into the river!",
"Bloodnok: No river.",
"Seagoon: By train?",
"Bloodnok: Doesn't run.",
"Seagoon: Why not?",
"Bloodnok: No railway.",
"Seagoon: Could we build one?",
"Bloodnok: No, the river would wash it away.",
"Seagoon: You said there was no river!",
"Bloodnok: Ah, it's behind the trees.",
"Seagoon: But a moment ago you said there weren't any trees either!",
"Bloodnok: Ah, but they've grown since then, you know. They just can't stand still for you, you know.",
"Henry: Now, close the oven door from the outside and bring it in after you.",
"Eccles: Just a minute! Close it from the outside and bring it in after me? That would mean climbing through it when it's shut and not opening it until I get through...",
"Seagoon: Eccles, what are you waiting for?",
"Eccles: I dunno how to do it!",
"Grytpype-Thynne: You shouldn't sit so far away, lad.",
"Seagoon: I don't mind, except when it rains.",
"Grytpype-Thynne: Why?",
"Seagoon: I'm outside.",
"Grytpype-Thynne: Don't you find it difficult to follow what the teacher's saying?",
"Seagoon: Oh, no - I can't hear him.",
"Henry: Min... Min's falling to bits. She's a loose woman, you know.",
"Bloodnok: Now, this uniform goes back to Moss Brothers tomorrow.",
"Thing: Yes, sir, there's the deposit on it.",
"Bloodnok: Oh, that'll brush off, don't worry.",
"Bluebottle: Pssst!",
"Eccles: What?",
"Bluebottle: Pssst!",
"Eccles: I haven't touched a drop!",
"Bluebottle: Eccles?",
"Eccles: Yeah?",
"Bluebottle: It's me, Blumebottuns.",
"Eccles: Oh! My friend!",
"Bluebottle: Yes, I'm your friend! You remember me?",
"Eccles: I remember you!",
"Bluebottle: Yes, why do you not open the door?",
"Eccles: Okay, I'll- How do you open a door?",
"Bluebottle: You turn the knob on your side.",
"Eccles: I haven't got a knob on my side!",
"Bluebottle: On the door!",
"Eccles: Oh! I'll soon get the hang of that.",
"Bluebottle: Hello, Little Jim.",
"Little Jim: [incomprehensible jabbering]",
"Bluebottle: Eccles... I do not understand what he is saying.",
"Eccles: Say that again, Little Jim.",
"Little Jim: Okay. [incomprehensible jabbering]",
"Eccles: He says he doesn't understand what he's saying either." };

    return goonquotes[svm_rand()%802];
}
