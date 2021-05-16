
//
// Common ML functions and definitions
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "mlcommon.h"
#include "vecstack.h"
#include <iostream>
#include <sstream>
#include <string>

int isZeroString(const std::string &src);
int isZeroString(const std::string &src)
{
    // Test if zero (sparse form) - that is, 0 or n:0 where n is an int

    int res = 0;

    if ( ( src.length() == 1 ) && ( src == "0" ) )
    {
        res = 1;
    }

    else if ( src.length() <= 1 )
    {
        res = 0;
    }

    else if ( ( src[src.length()-1] == '0' ) && ( src[src.length()-2] == ':' ) )
    {
        // Assume n form correct
        res = 1;
    }

    return res;
}

int decomma(std::string &src, int &elmcnt, int countZeros = 0);
int decomma(std::string &src, int &elmcnt, int countZeros)
{
    // It is assumed that there are no preceeding or postceeding whitespace

    elmcnt = 0;

    if ( src.length() )
    {
	unsigned int i;
        Stack<char> parenStack;
        int isQuote = 0;
        int isSpace = 0;
        int valstart = 0;
        int valend = -1;
        int isfinalzero = 0; // final element is included regardless to
                             // preserve dimensionality of vector, if needed.
                             // This is important for per-vector normalsiation
                             // in mercer kernels, where the cim
                             // of the vector must be representative, so
                             // for example mean([ 0 0 1 0 ]) = 0.25, rather
                             // then becoming mean([2:1]) = 1, which would
                             // cause problems.
        int isfirstzero = 1; // So that min/max work well in per-vector
                             // normalisation we need to ensure that if
                             // there are any zeros in the vector then at
                             // least one "representative" zero will be kept.

	for ( i = 0 ; i < src.length() ; i++ )
	{
            // Fix spacing, mark start and end of values

            if ( !(parenStack.size()) )
            {
                if ( ( src[i] == ',' ) || ( src[i] == ' ' ) || ( src[i] == '\t' ) )
                {
                    // Overwrite commas, spacing consistency
                    src[i] = ' ';

                    if ( !isSpace )
                    {
                        valend = i-1;
                        isSpace = 1;
                    }
                }

                else
                {
                    if ( isSpace )
                    {
                        valstart = i;
                        isSpace = 0;
                    }
                }
            }

            if ( i == (src.length())-1 )
            {
                valend = i;
            }

            // Evaluate value - increment if non-zero or countZero

            if ( valend >= valstart )
            {
                isfinalzero = 0;

                if ( countZeros )
                {
                    // counting all, zero or otherwise
                    elmcnt++;
                }

                else if ( !isZeroString(src.substr(valstart,valend-valstart+1)) )
                {
                    // Only counting non-zero, sparsing out others
                    elmcnt++;
                }

                else if ( isfirstzero )
                {
                    // Only counting non-zero, sparsing out others
                    // but keep this one as it may be the only zero, and
                    // we may need it to make max/min work
                    elmcnt++;
                    isfirstzero = 0;
                }

                else
                {
                    isfinalzero = 1;
                }

                valstart = valend+1;
            }

            // Process parenthesis

            if ( !isQuote )
            {
                if ( src[i] == '\"' )
                {
                    isQuote = 1;
                    parenStack.push(src[i]);
                }

                else if ( ( src[i] == '(' ) || ( src[i] == '[' ) || ( src[i] == '{' ) )
                {
                    parenStack.push(src[i]);
                }

                else if ( ( src[i] == ')' ) || ( src[i] == ']' ) || ( src[i] == '}' ) )
                {
                    if ( parenStack.size() )
                    {
                        if ( ( parenStack.accessTop() == '(' ) && ( src[i] == ')' ) )
                        {
                            parenStack.pop();
                        }

                        else if ( ( parenStack.accessTop() == '[' ) && ( src[i] == ']' ) )
                        {
                            parenStack.pop();
                        }

                        else if ( ( parenStack.accessTop() == '{' ) && ( src[i] == '}' ) )
                        {
                            parenStack.pop();
                        }

                        else
                        {
                            return 1;
                        }
                    }
                }
            }

            else if ( src[i] == '\"' )
            {
                NiceAssert( parenStack.size() );
                isQuote = 0;
                parenStack.pop();
            }
        }

        if ( isfinalzero && !countZeros )
        {
            // If vector ends in a zero then this is counted no matter
            // what.  Of course if we are counting zeros then that has
            // already been done, but if we are not counting zeros and
            // the final element was a zero then we need to increment
            // the element count to include it in the count.

            elmcnt++;
        }
    }

    return 0;
}

void parselineML_Single_all(SparseVector<gentype> &x, double &Cweight, double &epsweight, std::string &src, int countZeros = 0);
void parselineML_Single_all(SparseVector<gentype> &x, double &Cweight, double &epsweight, std::string &src, int countZeros)
{
    //std::string src = xsrc;
    int elmcnt = 0;
    decomma(src,elmcnt,countZeros);
    x.zero();
    x.prealloc(elmcnt);

    int i = 0;

    Cweight = 1;
    epsweight = 1;

    NiceAssert( src.length() );

    while ( isspace(src[i]) )
    {
	i++;

        NiceAssert( i < (int) src.length() );
    }

repover:
    if ( i < (int) src.length() )
    {
	if ( ( src[i] == 't' ) || ( src[i] == 'T' ) )
	{
	    i++;

            NiceAssert( i < (int) src.length() );
            NiceAssert( !isspace(src[i]) );

	    std::string dsrcb = src.substr(i,src.length());

	    std::istringstream dbufferb;
	    dbufferb.str(dsrcb);
	    dbufferb >> Cweight;

	    while ( !isspace(src[i]) )
	    {
		i++;

		if ( i == (int) src.length() )
		{
		    break;
		}
	    }

	    if ( i < (int) src.length() )
	    {
		while ( isspace(src[i]) )
		{
		    i++;

		    if ( i == (int) src.length() )
		    {
			break;
		    }
		}
	    }

            goto repover;
	}

	if ( ( src[i] == 's' ) || ( src[i] == 'S' ) )
	{
	    i++;

            NiceAssert( i < (int) src.length() );
            NiceAssert( !isspace(src[i]) );

	    std::string dsrcb = src.substr(i,src.length());

	    std::istringstream dbufferb;
	    dbufferb.str(dsrcb);
	    dbufferb >> Cweight;

            Cweight = ( Cweight < MINSWEIGHT ) ? (1.0/MINSWEIGHT) : (1/Cweight);

	    while ( !isspace(src[i]) )
	    {
		i++;

		if ( i == (int) src.length() )
		{
		    break;
		}
	    }

	    if ( i < (int) src.length() )
	    {
		while ( isspace(src[i]) )
		{
		    i++;

		    if ( i == (int) src.length() )
		    {
			break;
		    }
		}
	    }

            goto repover;
	}

	if ( ( src[i] == 'e' ) || ( src[i] == 'E' ) )
	{
	    i++;

            NiceAssert( i < (int) src.length() );
            NiceAssert( !isspace(src[i]) );

	    std::string dsrcb = src.substr(i,src.length());

	    std::istringstream dbufferb;
	    dbufferb.str(dsrcb);
	    dbufferb >> epsweight;

	    while ( !isspace(src[i]) )
	    {
		i++;

		if ( i == (int) src.length() )
		{
		    break;
		}
	    }

	    if ( i < (int) src.length() )
	    {
		while ( isspace(src[i]) )
		{
		    i++;

		    if ( i == (int) src.length() )
		    {
			break;
		    }
		}
	    }

            goto repover;
	}
    }

    x.zero();

    if ( i < (int) src.length() )
    {
        std::string xsrc(src.substr(i,src.length()-i));
        std::string lbrace("[ ");
        std::string rbrace(" ]");
        std::string ysrc(lbrace+xsrc+rbrace);

        std::istringstream xbufferc;
        xbufferc.str(ysrc);
        streamItInAlt(xbufferc,x,0,!countZeros);
    }

//    if ( x.indsize() )
//    {
//        int iji;
//
//        for ( iji = 0 ; iji < x.indsize() ; iji++ )
//        {
//            if ( (x.direcref(iji)).isValEqnDir() )
//            {
//                (x.direref(iji)).scalarfn_setisscalarfn(1);
//            }
//        }
//    }

    return;
}

void parselineML_Single(SparseVector<gentype> &x, double &Cweight, double &epsweight, std::string &src, int countZeros)
{
    int i = 0;
    int j = (int) src.length()-1;

    while ( isspace(src[i]) )
    {
	i++;

        NiceAssert( i < (int) src.length() );
    }

    while ( isspace(src[j]) )
    {
	j--;

        NiceAssert( ( j >= i ) && ( j >= 0 ) );
    }

    std::string dsrc = src.substr(i,j-i+1);

    parselineML_Single_all(x,Cweight,epsweight,dsrc,countZeros);

    return;
}


void parselineML_Generic(gentype &z, SparseVector<gentype> &x, double &Cweight, double &epsweight, int &r, std::string &src, int reverse, int countZeros)
{
    int dummy;

    decomma(src,dummy);

    int i = 0;
    int j = (int) src.length()-1;

    NiceAssert( src.length() );

    while ( isspace(src[i]) )
    {
	i++;

        NiceAssert( i < (int) src.length() );
    }

    while ( isspace(src[j]) )
    {
	j--;

        NiceAssert( ( j >= i ) && ( j >= 0 ) );
    }

    r = 2;

    if ( src[i] == '>' )
    {
	r = +1;
	i++;

        NiceAssert( i < j+1 );
        NiceAssert( isspace(src[i]) );
    }

    else if ( src[i] == '<' )
    {
	r = -1;
        i++;

        NiceAssert( i < j+1 );
        NiceAssert( isspace(src[i]) );
    }

    else if ( src[i] == '=' )
    {
	r = 2;
	i++;

        NiceAssert( i < j+1 );
        NiceAssert( isspace(src[i]) );
    }

    else if ( src[i] == '!' )
    {
	r = 0;
	i++;

        NiceAssert( i < j+1 );

        if ( src[i] == '=' )
        {
            i++;
        }

        NiceAssert( i < j+1 );
        NiceAssert( isspace(src[i]) );
    }

    while ( isspace(src[i]) )
    {
	i++;

        NiceAssert( i < j+1 );
    }

    if ( !reverse )
    {
	std::string dsrc = src.substr(i,j-i+1);

	std::istringstream dbuffer;
	dbuffer.str(dsrc);
        dbuffer >> z;

        int bracketcnt = 0;
        int sqbracketcnt = 0;
        int curlybracketcnt = 0;
        int inquote = 0;

        while ( !( isspace(src[i]) && !bracketcnt && !sqbracketcnt && !curlybracketcnt && !inquote ) )
	{
            if ( inquote )
            {
                if ( src[i] == '\"' ) { inquote = 0; }
            }

            else
            {
                     if ( src[i] == '('  ) { bracketcnt++;      }
                else if ( src[i] == '['  ) { sqbracketcnt++;    }
                else if ( src[i] == '{'  ) { curlybracketcnt++; }
                else if ( src[i] == ')'  ) { NiceAssert( bracketcnt      ); bracketcnt--;      }
                else if ( src[i] == ']'  ) { NiceAssert( sqbracketcnt    ); sqbracketcnt--;    }
                else if ( src[i] == '}'  ) { NiceAssert( curlybracketcnt ); curlybracketcnt--; }
                else if ( src[i] == '\"' ) { inquote = 1; }
            }

            i++;

	    if ( i == j+1 )
	    {
                NiceAssert( !bracketcnt && !sqbracketcnt && !curlybracketcnt && !inquote );

                break;
	    }
	}

	if ( i < j+1 )
	{
	    while ( isspace(src[i]) )
	    {
		i++;

		if ( i == j+1 )
		{
		    break;
		}
	    }
	}
    }

    else
    {
        int bracketcnt = 0;
        int sqbracketcnt = 0;
        int curlybracketcnt = 0;
        int inquote = 0;

        while ( !( isspace(src[j]) && !bracketcnt && !sqbracketcnt && !curlybracketcnt && !inquote ) )
	{
            if ( inquote )
            {
                if ( src[j] == '\"' ) { inquote = 0; }
            }

            else
            {
                     if ( src[j] == ')'  ) { bracketcnt++;      }
                else if ( src[j] == ']'  ) { sqbracketcnt++;    }
                else if ( src[j] == '}'  ) { curlybracketcnt++; }
                else if ( src[j] == '('  ) { NiceAssert( bracketcnt      ); bracketcnt--;      }
                else if ( src[j] == '['  ) { NiceAssert( sqbracketcnt    ); sqbracketcnt--;    }
                else if ( src[j] == '{'  ) { NiceAssert( curlybracketcnt ); curlybracketcnt--; }
                else if ( src[j] == '\"' ) { inquote = 1; }
            }

	    j--;

	    if ( j == i-1 )
	    {
                NiceAssert( !bracketcnt && !sqbracketcnt && !curlybracketcnt && !inquote );

		break;
	    }
	}

	std::string dsrc = src.substr(j+1,src.length());

	std::istringstream dbuffer;
	dbuffer.str(dsrc);
        dbuffer >> z;

	if ( j > i-1 )
	{
	    while ( isspace(src[j]) )
	    {
		j--;

		if ( j == i-1 )
		{
		    break;
		}
	    }
	}
    }

    std::string dsrc = src.substr(i,j-i+1);

    parselineML_Single_all(x,Cweight,epsweight,dsrc,countZeros);

//    if ( z.isValEqnDir() )
//    {
//        z.scalarfn_setisscalarfn(1);
//    }

    return;
}



