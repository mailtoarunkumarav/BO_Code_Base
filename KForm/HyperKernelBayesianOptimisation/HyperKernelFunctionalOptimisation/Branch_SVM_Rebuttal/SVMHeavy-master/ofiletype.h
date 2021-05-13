
//
// Statefull file access type
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _ofiletype_h
#define _ofiletype_h

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <string>
#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
#include "vector.h"
#include "basefn.h"




class ofiletype;

// Swap function

inline void qswap(ofiletype &a, ofiletype &b);
inline ofiletype &setzero(ofiletype &a);
inline ofiletype &postProInnerProd(ofiletype &a) { return a; }

class ofiletype
{
    friend void qswap(ofiletype &a, ofiletype &b);
    friend ofiletype &setzero(ofiletype &a);

public:

    // Constructors

    ofiletype()
    {
        filenum  = -1;
        filename = "";
        targpos  = 0;

        return;
    }

    ofiletype(int _filenum, const std::string &_filename, int _targpos, std::ifstream &input)
    {
        filenum  = _filenum;
        filename = _filename;
        targpos  = _targpos;

        getfiledata(input);

        return;
    }

    ofiletype(const ofiletype &src)
    {
        *this = src;

        return;
    }

    // Assignment operators

    ofiletype &operator=(const ofiletype &src)
    {
        filenum  = src.filenum;
        filename = src.filename;
        targpos  = src.targpos;

        lines = src.lines;

        return *this;
    }

    // Information

    const std::string &getfilename(void) const { return filename; }

    int getfilenum(void) const { return filenum;      }
    int getlinecnt(void) const { return lines.size(); }
    int gettargpos(void) const { return targpos;      }

    int pullline(int linenum)
    {
        NiceAssert( ( linenum >= 0 ) && ( linenum < getlinecnt() ) );

	int res = lines(linenum);
	lines.remove(linenum);

	return res;
    }

private:

    // filenum:  file number
    // filename: file name
    // lines:    vector of line numbers that have not yet been read
    // targpos:  0 = target-at-start format
    //           1 = target-at-end format

    int filenum;
    std::string filename;
    Vector<int> lines;
    int targpos;

    void getfiledata(std::ifstream &datfile)
    {
	// Assumed to only be called by the constructor.  This counts the total
	// number of lines containing data in the file and then "fills out" the
	// lines vector so that all lines are included.

	std::string buffer;
        int i,vcnt = 0;

	while ( !datfile.eof() )
	{
	    buffer = "";

	    while ( ( buffer.length() == 0 ) && !datfile.eof() )
	    {
		getline(datfile,buffer);
	    }

	    if ( buffer.length() == 0 )
	    {
		break;
	    }

	    vcnt++;
	}

	if ( vcnt )
	{
	    lines.resize(vcnt);

	    for ( i = 0 ; i < vcnt ; i++ )
	    {
                lines("&",i) = i;
	    }
	}

        return;
    }
};

inline void qswap(ofiletype &a, ofiletype &b)
{
    int x         = a.filenum;  a.filenum  = b.filenum;  b.filenum  = x;
    std::string y = a.filename; a.filename = b.filename; b.filename = y;
    int z         = a.targpos;  a.targpos  = b.targpos;  b.targpos  = z;

    qswap(a.lines,b.lines);

    return;
}

inline ofiletype &setzero(ofiletype &a)
{
    a.filenum = -1;
    a.filename = "";
    a.targpos = 0;

    a.lines.resize(0);

    return a;
}

inline ofiletype &setident (ofiletype &a) { throw("Whatever"); return a; }
inline ofiletype &setposate(ofiletype &a) { return a; }
inline ofiletype &setnegate(ofiletype &a) { throw("I reject your reality and substitute my own"); return a; }
inline ofiletype &setconj  (ofiletype &a) { throw("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
inline ofiletype &setrand  (ofiletype &a) { throw("Blippity Blappity Blue"); return a; }



#endif
