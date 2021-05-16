
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

#include <iostream>
#include <math.h>
#include <complex>
#include <limits.h>
#include <cstring>
#include <cstdlib>
#include <string>

#include "anion.h"
#include "numbase.h"


#define ANIONIOSTREAMBUFSIZE 2048



// Calculate structure constants

int epsilon(int order, int q, int r, int s)
{
    NiceAssert( order >= 0 );

    int res = 0;

    // Machine generated from general case using gencomm.cc (just comment out the else if line below)
    int commutateHR[3][3][3]  = { { {  0, 0, 0 }, {  0, 0,+1 }, {  0,-1, 0 } },
                                  { {  0, 0,-1 }, {  0, 0, 0 }, { +1, 0, 0 } },
                                  { {  0,+1, 0 }, { -1, 0, 0 }, {  0, 0, 0 } } };
    int commutateOR[7][7][7]  = { { {  0, 0, 0, 0, 0, 0, 0 }, {  0, 0,+1, 0, 0, 0, 0 }, {  0,-1, 0, 0, 0, 0, 0 }, {  0, 0, 0, 0,+1, 0, 0 }, {  0, 0, 0,-1, 0, 0, 0 }, {  0, 0, 0, 0, 0, 0,-1 }, {  0, 0, 0, 0, 0,+1, 0 } },
                                  { {  0, 0,-1, 0, 0, 0, 0 }, {  0, 0, 0, 0, 0, 0, 0 }, { +1, 0, 0, 0, 0, 0, 0 }, {  0, 0, 0, 0, 0,+1, 0 }, {  0, 0, 0, 0, 0, 0,+1 }, {  0, 0, 0,-1, 0, 0, 0 }, {  0, 0, 0, 0,-1, 0, 0 } },
                                  { {  0,+1, 0, 0, 0, 0, 0 }, { -1, 0, 0, 0, 0, 0, 0 }, {  0, 0, 0, 0, 0, 0, 0 }, {  0, 0, 0, 0, 0, 0,+1 }, {  0, 0, 0, 0, 0,-1, 0 }, {  0, 0, 0, 0,+1, 0, 0 }, {  0, 0, 0,-1, 0, 0, 0 } },
                                  { {  0, 0, 0, 0,-1, 0, 0 }, {  0, 0, 0, 0, 0,-1, 0 }, {  0, 0, 0, 0, 0, 0,-1 }, {  0, 0, 0, 0, 0, 0, 0 }, { +1, 0, 0, 0, 0, 0, 0 }, {  0,+1, 0, 0, 0, 0, 0 }, {  0, 0,+1, 0, 0, 0, 0 } },
                                  { {  0, 0, 0,+1, 0, 0, 0 }, {  0, 0, 0, 0, 0, 0,-1 }, {  0, 0, 0, 0, 0,+1, 0 }, { -1, 0, 0, 0, 0, 0, 0 }, {  0, 0, 0, 0, 0, 0, 0 }, {  0, 0,-1, 0, 0, 0, 0 }, {  0,+1, 0, 0, 0, 0, 0 } },
                                  { {  0, 0, 0, 0, 0, 0,+1 }, {  0, 0, 0,+1, 0, 0, 0 }, {  0, 0, 0, 0,-1, 0, 0 }, {  0,-1, 0, 0, 0, 0, 0 }, {  0, 0,+1, 0, 0, 0, 0 }, {  0, 0, 0, 0, 0, 0, 0 }, { -1, 0, 0, 0, 0, 0, 0 } },
                                  { {  0, 0, 0, 0, 0,-1, 0 }, {  0, 0, 0, 0,+1, 0, 0 }, {  0, 0, 0,+1, 0, 0, 0 }, {  0, 0,-1, 0, 0, 0, 0 }, {  0,-1, 0, 0, 0, 0, 0 }, { +1, 0, 0, 0, 0, 0, 0 }, {  0, 0, 0, 0, 0, 0, 0 } } };

    // NB: arrays here are machine generated using general (else) case

    if ( order == 2 )
    {
        NiceAssert( q <= 2 );
        NiceAssert( r <= 2 );
        NiceAssert( s <= 2 );

	res = commutateHR[q][r][s];

	if ( isAnionEyesLeft() )
	{
            res *= -1;
	}
    }

    else if ( order == 3 )
    {
        NiceAssert( q <= 6 );
        NiceAssert( r <= 6 );
        NiceAssert( s <= 6 );

	res = commutateOR[q][r][s];

	if ( isAnionEyesLeft() )
	{
            res *= -1;
	}
    }

    else if ( order > 3 )
    {
	d_anion tmpa(order);
	d_anion tmpb(order);
	d_anion tmpc(order);

	tmpb(r+1,1);
	tmpc(s+1,1);

	tmpa = COMMUTATOR(tmpb,tmpc);

	if ( tmpa(q+1) > 0.5 )
	{
	    res = +1;
	}

	else if ( tmpa(q+1) < -0.5 )
	{
	    res = -1;
	}
    }

    return res;
}

int epsilon(int order, int q, int r, int s, int t)
{
    int res = 0;

    // Machine generated from general case using genassoc.cc (just comment out the else if line below)
    int associateOR[7][7][7][7]  = { { { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 0, 0, 1, 0, }, { 0, 0, 0, 0, -1, 0, 0, }, { 0, 0, 0, 1, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 1, 0, }, { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, -1, 0, 0, 0, }, { 0, 0, 0, 0, -1, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 0, 0, -1, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 1, 0, 0, 0, 0, }, { 0, -1, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, -1, 0, }, { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 1, 0, 0, 0, 0, 0, }, { 0, 0, 1, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 1, 0, 0, }, { 0, 0, 0, 1, 0, 0, 0, }, { 0, 0, -1, 0, 0, 0, 0, }, { 0, -1, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, -1, 0, 0, 0, }, { 0, 0, 0, 0, 1, 0, 0, }, { 0, 1, 0, 0, 0, 0, 0, }, { 0, 0, -1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } } },
                                     { { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 0, 0, -1, 0, }, { 0, 0, 0, 0, 1, 0, 0, }, { 0, 0, 0, -1, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, -1, 0, 0, }, { 0, 0, 0, 1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 0, 0, -1, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 1, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, -1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 1, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, -1, 0, 0, 0, }, { 0, 0, 1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, -1, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 1, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 1, 0, }, { -1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, -1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } } },
                                     { { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, -1, 0, }, { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 1, 0, 0, 0, }, { 0, 0, 0, 0, 1, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 1, 0, 0, }, { 0, 0, 0, -1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 0, 0, 1, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 1, 0, }, { 0, 0, 0, 0, -1, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 1, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, -1, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, -1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, -1, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, -1, 0, 0, }, { 0, 0, 0, 0, 0, -1, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, }, { 0, 1, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } } },
                                     { { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 0, 0, 1, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, -1, 0, 0, 0, 0, }, { 0, 1, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, -1, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, -1, 0, }, { 0, 0, 0, 0, 1, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, -1, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, -1, 0, 0, 0, 0, }, { 0, 1, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 0, 0, -1, 0, } },
                                       { { 0, 0, 1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 1, 0, 0, } },
                                       { { 0, -1, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 1, 0, }, { 0, 0, 0, 0, -1, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } } },
                                     { { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 1, 0, }, { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, -1, 0, 0, 0, 0, 0, }, { 0, 0, -1, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, -1, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 1, 0, 0, 0, }, { 0, 0, -1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, -1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 1, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 1, 0, 0, 0, 0, }, { 0, -1, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 0, 0, 1, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 1, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, -1, 0, 0, 0, } },
                                       { { 0, 0, 1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, -1, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } } },
                                     { { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, -1, 0, 0, }, { 0, 0, 0, -1, 0, 0, 0, }, { 0, 0, 1, 0, 0, 0, 0, }, { 0, 1, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 1, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, -1, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 1, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, -1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 1, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, -1, 0, 0, } },
                                       { { 0, -1, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, -1, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 1, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 1, 0, 0, 0, 0, }, { 0, -1, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 1, 0, 0, }, { 0, 0, 0, -1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } } },
                                     { { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 1, 0, 0, 0, }, { 0, 0, 0, 0, -1, 0, 0, }, { 0, -1, 0, 0, 0, 0, 0, }, { 0, 0, 1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, -1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, -1, 0, }, { 1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 1, 0, 0, }, { 0, 0, 0, 0, 0, 1, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, }, { 0, -1, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 1, 0, 0, 0, 0, 0, }, { -1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, -1, 0, }, { 0, 0, 0, 0, 1, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, -1, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 1, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 1, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, -1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, -1, 0, 0, 0, 0, }, { 0, 1, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, -1, 0, 0, }, { 0, 0, 0, 1, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } },
                                       { { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, }, { 0, 0, 0, 0, 0, 0, 0, } } } };

    if ( order == 3 )
    {
        NiceAssert( q <= 6 );
        NiceAssert( r <= 6 );
        NiceAssert( s <= 6 );
        NiceAssert( t <= 6 );

	res = associateOR[q][r][s][t];
    }

    else if ( order > 3 )
    {
	d_anion tmpa(order);
	d_anion tmpb(order);
	d_anion tmpc(order);
	d_anion tmpd(order);

	tmpa(q+1,1);
	tmpb(r+1,1);
	tmpc(s+1,1);

	tmpd = ASSOCIATOR(tmpa,tmpb,tmpc);

	if ( tmpd(t+1) > 0.5 )
	{
	    res = +1;
	}

	else if ( tmpd(t+1) < -0.5 )
	{
	    res = -1;
	}
    }

    return res;
}


int getsetanioneyes(int val);
int getsetanioneyes(int val)
{
    svmvolatile static int setval = 1; // 0 is eyes left, 1 is eyes right

    if ( ( val == 0 ) || ( val == 1 ) )
    {
        setval = val;
    }

    return setval;
}

int isAnionEyesLeft()
{
    return !getsetanioneyes(-1);
}

int isAnionEyesRight()
{
    return getsetanioneyes(-1);
}

void setAnionEyesLeft()
{
    getsetanioneyes(0);
}

void setAnionEyesRight()
{
    getsetanioneyes(1);
}




d_anion::d_anion(int order)
{
    is_im = 0;

    value_real = 0.0;
    value_inf  = NULL;
    value_0    = NULL;

    if ( order )
    {
	is_im = 1;

	value_real = 0.0;

	MEMNEW(value_inf,d_anion(order-1));
        MEMNEW(value_0  ,d_anion(order-1));
    }

    return;
}

d_anion &d_anion::operator=(const double &source)
{
    if ( is_im )
    {
        MEMDEL(value_inf);
        MEMDEL(value_0);
    }

    is_im = 0;

    value_real = source;
    value_inf  = NULL;
    value_0    = NULL;

    return *this;
}

d_anion &d_anion::operator=(const std::complex<double> &source)
{
    if ( !is_im )
    {
	MEMNEW(value_inf,d_anion());
	MEMNEW(value_0  ,d_anion());

	is_im = 1;
    }

    value_real = 0.0;
    *value_inf = real(source);
    *value_0   = imag(source);

    return *this;
}

d_anion &d_anion::operator=(const d_anion &source)
{
    if ( source.is_im )
    {
        if ( !is_im )
        {
            MEMNEW(value_inf,d_anion());
            MEMNEW(value_0  ,d_anion());

            is_im = 1;
        }

        value_real = source.value_real;
        *value_inf = *(source.value_inf);
        *value_0   = *(source.value_0);
    }

    else
    {
        if ( is_im )
        {
            MEMDEL(value_inf);
            MEMDEL(value_0);

            is_im = 0;
        }

        value_real = source.value_real;
        value_inf  = NULL;
        value_0    = NULL;
    }

    return *this;
}

d_anion &d_anion::operator=(const char *src)
{
    *this = atod_anion(src);

    return *this;
}

d_anion::operator std::complex<double>() const
{
    std::complex<double> result(real(*this),imag(*this));
    
    return result;
}


d_anion &d_anion::leftpart(void)
{
    if ( !is_im )
    {
	double temp = value_real;

        MEMNEW(value_inf,d_anion());
        MEMNEW(value_0  ,d_anion());

        is_im = 1;

        *value_inf = temp;
        *value_0   = 0.0;
    }

    return *value_inf;
}

d_anion &d_anion::rightpart(void)
{
    if ( !is_im )
    {
        double temp = value_real;

        MEMNEW(value_inf,d_anion());
        MEMNEW(value_0  ,d_anion());

        is_im = 1;

        *value_inf = temp;
        *value_0   = 0.0;
    }

    return *value_0;
}

int d_anion::order(void) const
{
    int resultleft;
    int resultright;

    if ( !is_im )
    {
        return 0;
    }

    resultleft = value_inf->order();
    resultright = value_0->order();

    if ( resultleft > resultright )
    {
        return 1+resultleft;
    }

    return 1+resultright;
}

int d_anion::isindet(void) const
{
    if ( !is_im )
    {
        if ( testisvnan(value_real) ) { return 2; }
        if ( testisninf(value_real) ) { return -1; }
        if ( testispinf(value_real) ) { return 1; }

        return 0;
    }

    return (value_inf->isindet()) || (value_0->isindet());
}

double d_anion::realpart(void) const
{
    if ( !is_im )
    {
        return value_real;
    }

    return value_inf->realpart();
}

d_anion &d_anion::simplify(void)
{
    if ( is_im )
    {
	value_inf->simplify();
	value_0->simplify();

        if ( *value_0 == 0.0 )
        {
            d_anion temp(*value_inf);

            *this = temp;
        }
    }

    return *this;
}

d_anion &d_anion::setorder(int n)
{
    if ( n <= 0 )
    {
	if ( is_im )
	{
	    value_real = realpart();

	    is_im = 0;

	    MEMDEL(value_inf);
	    MEMDEL(value_0);
	}
    }

    else
    {
	if ( !is_im )
	{
	    d_anion temp(*this);

	    leftpart() = temp;
	    rightpart() *= 0.0;
	}

	leftpart().setorder(n-1);
	rightpart().setorder(n-1);
    }

    return *this;
}

d_anion &d_anion::resize(int newdim)
{
    setorder(ceilintlog2(newdim));
    return *this;
}

void d_anion::setorderge(int n)
{
    if ( n > order() )
    {
        d_anion temp(*this);

	leftpart() = temp;
	rightpart() *= 0.0;

	leftpart().setorderge(n-1);
    }

    return;
}

const double &d_anion::operator()(int i) const
{
    int maxi = 1 << order();
    const static double tempzero = 0.0;

    if ( i >= maxi )
    {
        return tempzero;
    }

    if ( is_im )
    {
	if ( ( i/(maxi/2) ) == 0 )
	{
            return (*value_inf)(i%(maxi/2));
	}

	else
	{
            return (*value_0)(i%(maxi/2));
	}
    }

    return value_real;
}

double &d_anion::operator()(const char *dummy, int i)
{
    (*this)(i,0.0);

    return getref(i);

    if ( dummy[0] == dummy[1] )
    {
        i = 2;
    }
}

double &d_anion::getref(int i)
{
    int maxi = 1 << order();

    NiceAssert( i < maxi );

    if ( is_im )
    {
	if ( ( i/(maxi/2) ) == 0 )
	{
            return (*value_inf).getref(i%(maxi/2));
	}

	else
	{
            return (*value_0).getref(i%(maxi/2));
	}
    }

    return value_real;
}

double d_anion::operator()(int i, double x)
{
    int n = order();

    while ( ( 1 << n ) <= i )
    {
        n++;
    }

    setorderge(n);

    int maxi = 1 << n;

    if ( is_im )
    {
	if ( ( i/(maxi/2) ) == 0 )
	{
	    leftpart().setorderge(n-1);

            return leftpart()(i%(maxi/2),x);
	}

	else
	{
	    rightpart().setorderge(n-1);

            return rightpart()(i%(maxi/2),x);
	}
    }

    return ( value_real = x );
}













// += additive       assignment - binary, return lvalue
// -= subtractive    assignment - binary, return lvalue

d_anion &operator+=(d_anion &left_op, const d_anion &right_op)
{
    if ( left_op.is_im && right_op.is_im )
    {
        *(left_op.value_inf) += *(right_op.value_inf);
        *(left_op.value_0)   += *(right_op.value_0);
    }

    else if ( left_op.is_im )
    {
        *(left_op.value_inf) += right_op.value_real;
    }

    else if ( right_op.is_im )
    {
	double temp = left_op.value_real;

        left_op = right_op;

        *(left_op.value_inf) += temp;
    }

    else
    {
        left_op.value_real += right_op.value_real;
    }

    return left_op;
}

d_anion &operator-=(d_anion &left_op, const d_anion &right_op)
{
    if ( left_op.is_im && right_op.is_im )
    {
        *(left_op.value_inf) -= *(right_op.value_inf);
        *(left_op.value_0)   -= *(right_op.value_0);
    }

    else if ( left_op.is_im )
    {
        *(left_op.value_inf) -= right_op.value_real;
    }

    else if ( right_op.is_im )
    {
	double temp = left_op.value_real;

	left_op = right_op;
        setnegate(left_op);

        *(left_op.value_inf) += temp;
    }

    else
    {
        left_op.value_real -= right_op.value_real;
    }

    return left_op;
}

// *= multiplicative assignment - binary, return lvalue
// /= divisive       assignment - binary, return lvalue

d_anion &leftmult( d_anion &left_op, const double               &right_op)
{
    if ( left_op.is_im )
    {
        *(left_op.value_inf) *= right_op;
        *(left_op.value_0)   *= right_op;
    }

    else
    {
	left_op.value_real *= right_op;
    }

    return left_op;
}

d_anion &leftmult( d_anion &left_op, const std::complex<double> &right_op)
{
    if ( left_op.is_im )
    {
	d_anion result(left_op);

	if ( isAnionEyesLeft() )
	{
            *(result.value_inf) = (      (*( left_op.value_inf))  *      real(right_op)           )
                                - (      imag(right_op)           * conj((*( left_op.value_0  ))) );
            *(result.value_0)   = (      real(right_op)           *      (*( left_op.value_0  ))  )
	                        + ( conj((*( left_op.value_inf))) *      imag(right_op)           );
	}

	else
	{
            *(result.value_inf) = (      (*( left_op.value_inf))  *      real(right_op)  )
                                - (      imag(right_op)           *      (*( left_op.value_0  ))  );
	    *(result.value_0)   = (      imag(right_op)           *      (*( left_op.value_inf))  )
                                + (      (*( left_op.value_0  ))  *      real(right_op)           );
	}

        left_op = result;
    }

    else
    {
	double temp = left_op.value_real;

	left_op = right_op;

        left_op *= temp;
    }

    return left_op;
}

d_anion &leftmult( d_anion &left_op, const d_anion              &right_op)
{
    if ( left_op.is_im && right_op.is_im )
    {
	d_anion result(left_op);

	if ( isAnionEyesLeft() )
	{
            *(result.value_inf) = (      (*( left_op.value_inf))  *      (*(right_op.value_inf))  )
                                - (      (*(right_op.value_0  ))  * conj((*( left_op.value_0  ))) );
            *(result.value_0)   = (      (*(right_op.value_inf))  *      (*( left_op.value_0  ))  )
	                        + ( conj((*( left_op.value_inf))) *      (*(right_op.value_0  ))  );
	}

	else
	{
            *(result.value_inf) = (      (*( left_op.value_inf))  *      (*(right_op.value_inf))  )
                                - ( conj((*(right_op.value_0  ))) *      (*( left_op.value_0  ))  );
	    *(result.value_0)   = (      (*(right_op.value_0  ))  *      (*( left_op.value_inf))  )
                                + (      (*( left_op.value_0  ))  * conj((*(right_op.value_inf))) );
	}

        left_op = result;
    }

    else if ( left_op.is_im )
    {
        *(left_op.value_inf) *= (right_op.value_real);
        *(left_op.value_0)   *= (right_op.value_real);
    }

    else if ( right_op.is_im )
    {
	double temp = left_op.value_real;

	left_op = right_op;

        left_op *= temp;
    }

    else
    {
	left_op.value_real *= right_op.value_real;
    }

    return left_op;
}

d_anion &rightmult(const double               &left_op, d_anion &right_op)
{
    if ( right_op.is_im )
    {
        *(right_op.value_inf) *= left_op;
        *(right_op.value_0)   *= left_op;
    }

    else
    {
	right_op.value_real *= left_op;
    }

    return right_op;
}

d_anion &rightmult(const std::complex<double> &left_op, d_anion &right_op)
{
    if ( right_op.is_im )
    {
	d_anion result(left_op);

	if ( isAnionEyesLeft() )
	{
            *(result.value_inf) = (      real(left_op)            *      (*(right_op.value_inf))  )
                                - (      (*(right_op.value_0  ))  *      imag(left_op)            );
            *(result.value_0)   = (      (*(right_op.value_inf))  *      imag(left_op)            )
	                        + (      real(left_op)            *      (*(right_op.value_0  ))  );
	}

	else
	{
            *(result.value_inf) = (      real(left_op)            *      (*(right_op.value_inf))  )
                                - ( conj((*(right_op.value_0  ))) *      imag(left_op)            );
	    *(result.value_0)   = (      (*(right_op.value_0  ))  *      real(left_op)            )
                                + (      imag(left_op)            * conj((*(right_op.value_inf))) );
	}

        right_op = result;
    }

    else
    {
	double temp = right_op.value_real;

	right_op = left_op;

        right_op *= temp;
    }

    return right_op;
}

d_anion &rightmult(const d_anion              &left_op, d_anion &right_op)
{
    if ( left_op.is_im && right_op.is_im )
    {
	d_anion result(left_op);

	if ( isAnionEyesLeft() )
	{
            *(result.value_inf) = (      (*( left_op.value_inf))  *      (*(right_op.value_inf))  )
                                - (      (*(right_op.value_0  ))  * conj((*( left_op.value_0  ))) );
            *(result.value_0)   = (      (*(right_op.value_inf))  *      (*( left_op.value_0  ))  )
	                        + ( conj((*( left_op.value_inf))) *      (*(right_op.value_0  ))  );
	}

	else
	{
            *(result.value_inf) = (      (*( left_op.value_inf))  *      (*(right_op.value_inf))  )
                                - ( conj((*(right_op.value_0  ))) *      (*( left_op.value_0  ))  );
	    *(result.value_0)   = (      (*(right_op.value_0  ))  *      (*( left_op.value_inf))  )
                                + (      (*( left_op.value_0  ))  * conj((*(right_op.value_inf))) );
	}

        right_op = result;
    }

    else if ( right_op.is_im )
    {
        *(right_op.value_inf) *= (left_op.value_real);
        *(right_op.value_0)   *= (left_op.value_real);
    }

    else if ( left_op.is_im )
    {
	double temp = right_op.value_real;

	right_op = left_op;

        right_op *= temp;
    }

    else
    {
	right_op.value_real *= left_op.value_real;
    }

    return right_op;
}


// == equivalence

int operator==(const d_anion &left_op, const d_anion &right_op)
{
    if ( !(left_op.is_im) && !(right_op.is_im) )
    {
        return left_op.value_real == right_op.value_real;
    }

    if ( !(left_op.is_im) )
    {
        return ( ( left_op.value_real == *(right_op.value_inf) ) &&
                 ( 0.0                == *(right_op.value_0)   )    );
    }

    if ( !(right_op.is_im) )
    {
        return ( ( right_op.value_real == *(left_op.value_inf) ) &&
                 ( 0.0                 == *(left_op.value_0)   )    );
    }

    return ( ( *(left_op.value_inf) == *(right_op.value_inf) ) &&
             ( *(left_op.value_0)   == *(right_op.value_0)   )    );
}







// complex and fpu operations

#define ANION_ZTOL              10*FLT_MIN


double abs1(const d_anion &a)
{
    return norm1(a);
}

double abs2(const d_anion &a)
{
    return sqrt(norm2(a));
}

double absd(const d_anion &a)
{
    return abs2(a);
}

double absp(const d_anion &a, const double &x)
{
    return pow(normp(a,x),1/x);
}

double absinf(const d_anion &a)
{
    if ( a.iscomplex() )
    {
	d_anion temp(a);
	double leftinf = absinf(temp.leftpart());
        double rightinf = absinf(temp.rightpart());

	return ( leftinf > rightinf ) ? leftinf : rightinf;
    }

    return abs2(a.realpart());
}

double arg(const d_anion &a)
{
    double result = abs2(imagx(log(a)));
    d_anion tempx(a);

    if ( tempx(1) < 0 )
    {
        result *= -1;
    }

    return result;
}

d_anion argd(const d_anion &a, const d_anion &q_default)
{
    double a_arg = arg(a);

    if ( a_arg == 0.0 ) // FIXME: suspect need a zero region, not a point
    {
	if ( q_default == 0.0 )
	{
	    d_anion resultdefault(0,1);

	    return resultdefault;
	}

	else
	{
	    return q_default;
	}
    }

    return imagx(log(a))/a_arg;
}

d_anion argd(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return argd(a,resultdefault);
}

d_anion Argd(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return argd(a,resultdefault);
}

d_anion argx(const d_anion &a, const d_anion &q_default)
{
    return imagx(log(a,q_default));
}

d_anion argx(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return argx(a,resultdefault);
}

d_anion Argx(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return argx(a,resultdefault);
}

double norm1(const d_anion &a)
{
    if ( a.iscomplex() )
    {
        d_anion temp(a);

	return norm1(temp.leftpart())+norm1(temp.rightpart());
    }

    return abs2(a.realpart());
}

double norm2(const d_anion &a)
{
    if ( a.iscomplex() )
    {
        d_anion temp(a);

	return norm2(temp.leftpart())+norm2(temp.rightpart());
    }

    return (a.realpart())*(a.realpart());
}

double normd(const d_anion &a)
{
    return norm2(a);
}

double normp(const d_anion &a, const double &x)
{
    if ( a.iscomplex() )
    {
        d_anion temp(a);

	return normp(temp.leftpart(),x)+normp(temp.rightpart(),x);
    }

    return pow(abs2(a.realpart()),x);
}

d_anion angle(const d_anion &a)
{
    double absa = abs2(a);

    if ( absa == 0.0 ) // FIXME: suspect need a zero region, not a point
    {
        d_anion resultzero(0.0);

        return resultzero;
    }

    return a/absa;
}

d_anion vangle(const d_anion &a, const d_anion &defsign)
{
    double absa = abs2(a);

    if ( absa == 0.0 ) // FIXME: suspect need a zero region, not a point
    {
        return defsign;
    }

    return a/absa;
}

d_anion polar(const double &x, const double &y, const d_anion &a)
{
    return x*exp(y*a);
}

d_anion polard(const double &x, const double &y, const d_anion &a)
{
    return x*exp(y*a);
}

d_anion polarx(const double &x, const d_anion &a)
{
    return x*exp(a);
}

d_anion sgn(const d_anion &a)
{
    d_anion result(a);
    
    if ( result.iscomplex() )
    {
        result.leftpart()  = sgn(result.leftpart());
	result.rightpart() = sgn(result.rightpart());
    }

    else
    {
        if ( result.realpart() == 0.0 )
        {
            result = 0.0;
        }
        
        else if ( result.realpart() < 0.0 )
        {
            result = -1.0;
        }
        
        else
        {
            result = 1.0;
	}
    }
    
    return result;
}




double real(const d_anion &a)
{
    if ( a.iscomplex() )
    {
        d_anion temp(a);

	return real(temp.leftpart());
    }

    return a.realpart();
}

double imag(const d_anion &a)
{
    double result = abs2(imagx(a));
    d_anion tempx(a);

    if ( tempx(1) < 0 )
    {
        result *= -1;
    }

    return result;
}

d_anion imagd(const d_anion &a, const d_anion &q_default)
{
    double a_imag = imag(a);

    if ( a_imag == 0.0 ) // FIXME: suspect need a zero region, not a point
    {
	if ( q_default == 0.0 )
	{
	    d_anion resultdefault(0,1);

	    return resultdefault;
	}

	else
	{
	    return q_default;
	}
    }

    return imagx(a)/a_imag;
}

d_anion imagd(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return imagd(a,resultdefault);
}

d_anion Imagd(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return imagd(a,resultdefault);
}

d_anion imagx(const d_anion &a)
{
    d_anion result(a);

    result(0,0.0);

    return result;
}

d_anion conj(const d_anion &a)
{
    d_anion result(a);

    return setconj(result);
}

d_anion inv(const d_anion &a)
{
    return conj(a)/norm2(a);
}




d_anion pow(const long &a, const d_anion &b, const d_anion &q_default)
{
    if ( a == 0 )
    {
	if ( b == 0.0 ) // FIXME: obvious numerical problems here
	{
            d_anion result(1.0);

	    return result; // in line with the c99 standard, 0^0 = 1
	}

	else
	{
            d_anion result(0.0);

	    return result; // 0^b = 0 for all nonzero b
	}
    }

    d_anion aa((double) a);

    return exp(log(aa,q_default)*b);
}

d_anion pow(const double &a, const d_anion &b, const d_anion &q_default)
{
    if ( a == 0.0 ) // FIXME: obvious numerical problems here
    {
	if ( b == 0.0 ) // FIXME: obvious numerical problems here
	{
            d_anion result(1.0);

	    return result; // in line with the c99 standard, 0^0 = 1
	}

	else
	{
            d_anion result(0.0);

	    return result; // 0^b = 0 for all nonzero b
	}
    }

    d_anion aa(a);

    return exp(log(aa,q_default)*b);
}

d_anion pow(const std::complex<double> &a, const d_anion &b, const d_anion &q_default)
{
    d_anion tempa(a);

    return pow(tempa,b,q_default);
}

d_anion powl(const std::complex<double> &a, const d_anion &b, const d_anion &q_default)
{
    d_anion tempa(a);

    return powl(tempa,b,q_default);
}

d_anion powr(const std::complex<double> &a, const d_anion &b, const d_anion &q_default)
{
    d_anion tempa(a);

    return powr(tempa,b,q_default);
}

d_anion pow(const d_anion &a, const long &b, const d_anion &q_default)
{
    if ( a == 0.0 ) // FIXME: obvious numerical problems here
    {
	if ( b == 0 )
	{
            d_anion result(1.0);

	    return result; // in line with the c99 standard, 0^0 = 1
	}

	else
	{
            d_anion result(0.0);

	    return result; // 0^b = 0 for all nonzero b
	}
    }

    d_anion result(1.0);
    long i;

    if ( b > 0 )
    {
        for ( i = 1 ; i <= b ; i++ )
        {
            result *= a;
        }
    }

    else if ( b < 0 )
    {
        result = pow(inv(a),-b,q_default);
    }

    return result;
}

d_anion pow(const d_anion &a, const double &b, const d_anion &q_default)
{
    if ( a == 0.0 ) // FIXME: obvious numerical problems here
    {
	if ( b == 0.0 ) // FIXME: obvious numerical problems here
	{
            d_anion result(1.0);

	    return result; // in line with the c99 standard, 0^0 = 1
	}

	else
	{
            d_anion result(0.0);

	    return result; // 0^b = 0 for all nonzero b
	}
    }

    return exp(log(a,q_default)*b);
}

d_anion pow(const d_anion &a, const std::complex<double> &b, const d_anion &q_default)
{
    d_anion tempb(b);

    return pow(a,tempb,q_default);
}

d_anion powl(const d_anion &a, const std::complex<double> &b, const d_anion &q_default)
{
    d_anion tempb(b);

    return powl(a,tempb,q_default);
}

d_anion powr(const d_anion &a, const std::complex<double> &b, const d_anion &q_default)
{
    d_anion tempb(b);

    return powr(a,tempb,q_default);
}

d_anion pow(const d_anion &a, const d_anion &b, const d_anion &q_default)
{
    return (powl(a,b,q_default)+powr(a,b,q_default))/2.0;
}

d_anion powl(const d_anion &a, const d_anion &b, const d_anion &q_default)
{
    if ( a == 0.0 ) // FIXME: obvious numerical problems here
    {
	if ( b == 0.0 ) // FIXME: obvious numerical problems here
	{
            d_anion result(1.0);

	    return result; // in line with the c99 standard, 0^0 = 1
	}

	else
	{
            d_anion result(0.0);

	    return result; // 0^b = 0 for all nonzero b
	}
    }

    d_anion result(0.0);

    if ( ( !(a.iscomplex()) ) && ( !(b.iscomplex()) ) )
    {
        result = pow(a.realpart(),b.realpart());
    }

    else if ( !(a.iscomplex()) )
    {
	result = pow(a.realpart(),b);
    }

    else if ( !(b.iscomplex()) )
    {
        result = pow(a,b.realpart());
    }

    else
    {
	result = exp(b*log(a,q_default));
    }

    return result;
}

d_anion powr(const d_anion &a, const d_anion &b, const d_anion &q_default)
{
    if ( a == 0.0 ) // FIXME: obvious numerical problems here
    {
	if ( b == 0.0 ) // FIXME: obvious numerical problems here
	{
            d_anion result(1.0);

	    return result; // in line with the c99 standard, 0^0 = 1
	}

	else
	{
            d_anion result(0.0);

	    return result; // 0^b = 0 for all nonzero b
	}
    }

    d_anion result(0.0);

    if ( ( !(a.iscomplex()) ) && ( !(b.iscomplex()) ) )
    {
        result = pow(a.realpart(),b.realpart());
    }

    else if ( !(a.iscomplex()) )
    {
        result = pow(a.realpart(),b);
    }

    else if ( !(b.iscomplex()) )
    {
        result = pow(a,b.realpart());
    }

    else
    {
	result = exp(log(a,q_default)*b);
    }

    return result;
}

d_anion sqrt(const d_anion &a, const d_anion &q_default)
{
    return pow(a,0.5,q_default);
}

































d_anion Pow(const long &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return pow(a,b,resultdefault);
}

d_anion Pow(const double &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return pow(a,b,resultdefault);
}

d_anion Pow(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return pow(a,b,resultdefault);
}

d_anion Powl(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return powl(a,b,resultdefault);
}

d_anion Powr(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return powr(a,b,resultdefault);
}

d_anion Pow(const d_anion &a, const long &b)
{
    d_anion resultdefault(0,-1);

    return pow(a,b,resultdefault);
}

d_anion Pow(const d_anion &a, const double &b)
{
    d_anion resultdefault(0,-1);

    return pow(a,b,resultdefault);
}

d_anion Pow(const d_anion &a, const std::complex<double> &b)
{
    d_anion resultdefault(0,-1);

    return pow(a,b,resultdefault);
}

d_anion Powl(const d_anion &a, const std::complex<double> &b)
{
    d_anion resultdefault(0,-1);

    return powl(a,b,resultdefault);
}

d_anion Powr(const d_anion &a, const std::complex<double> &b)
{
    d_anion resultdefault(0,-1);

    return powr(a,b,resultdefault);
}

d_anion Pow(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return pow(a,b,resultdefault);
}

d_anion Powl(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return powl(a,b,resultdefault);
}

d_anion Powr(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return powr(a,b,resultdefault);
}

d_anion Sqrt(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return sqrt(a,resultdefault);
}





d_anion pow(const long &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return pow(a,b,resultdefault);
}

d_anion pow(const double &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return pow(a,b,resultdefault);
}

d_anion pow(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return pow(a,b,resultdefault);
}

d_anion powl(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return powl(a,b,resultdefault);
}

d_anion powr(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return powr(a,b,resultdefault);
}

d_anion pow(const d_anion &a, const long &b)
{
    d_anion resultdefault(0,1);

    return pow(a,b,resultdefault);
}

d_anion pow(const d_anion &a, const double &b)
{
    d_anion resultdefault(0,1);

    return pow(a,b,resultdefault);
}

d_anion pow(const d_anion &a, const std::complex<double> &b)
{
    d_anion resultdefault(0,1);

    return pow(a,b,resultdefault);
}

d_anion powl(const d_anion &a, const std::complex<double> &b)
{
    d_anion resultdefault(0,1);

    return powl(a,b,resultdefault);
}

d_anion powr(const d_anion &a, const std::complex<double> &b)
{
    d_anion resultdefault(0,1);

    return powr(a,b,resultdefault);
}

d_anion pow(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return pow(a,b,resultdefault);
}

d_anion powl(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return powl(a,b,resultdefault);
}

d_anion powr(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return powr(a,b,resultdefault);
}

d_anion sqrt(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return sqrt(a,resultdefault);
}

























d_anion exp(const d_anion &a)
{
    if ( a == 0.0 ) // FIXME: obvious numerical problems here
    {
	d_anion result(1.0);

	return result; // exp(0) = 1
    }

    double R = real(a);
    d_anion M = imagx(a);
    d_anion q = angle(M);
    double I = abs2(M);

    return (exp(R)*cos(I)) + (q*exp(R)*sin(I));
}

d_anion tenup(const d_anion &a)
{
    long bv = 10;

    return pow(bv,a);
}

d_anion log(const d_anion &a, const d_anion &q_default)
{
    double R = real(a);
    d_anion M = imagx(a);
    d_anion q = angle(M);
    double I = abs2(M);

    if ( ( abs2(q) < 0.1 ) && ( R < 0.0 ) ) // equivalent to q == 0, R < 0, but numerically better
    {
	// Case where a is real and negative

	d_anion qq_def;

	qq_def = angle(q_default);

	if ( abs2(qq_def) < 0.1 )
	{
	    d_anion temp(log(sqrt((R*R)+(I*I))),atan2(I,R));

	    return temp;
	}

	else
	{
	    return log(sqrt((R*R)+(I*I))) + (qq_def*atan2(I,R));
	}
    }

    return log(sqrt((R*R)+(I*I))) + (q*atan2(I,R));
}

d_anion log10(const d_anion &a, const d_anion &q_default)
{
    return log(a,q_default)/NUMBASE_LN10;
}

d_anion logb(const long &a, const d_anion &b, const d_anion &q_default)
{
    if ( a > 0 )
    {
        return log((double) a)*inv(log(b,q_default));
    }

    d_anion aa((double) a);

    return log(aa,q_default)*inv(log(b,q_default));
}

d_anion logb(const double &a, const d_anion &b, const d_anion &q_default)
{
    if ( a > 0.0 )
    {
	return log(a)*inv(log(b,q_default));
    }

    d_anion aa(a);

    return log(aa,q_default)*inv(log(b,q_default));
}

d_anion logb(const std::complex<double> &a, const d_anion &b, const d_anion &q_default)
{
    d_anion tempa(a);

    return logb(tempa,b,q_default);
}

d_anion logbl(const std::complex<double> &a, const d_anion &b, const d_anion &q_default)
{
    d_anion tempa(a);

    return logbl(tempa,b,q_default);
}

d_anion logbr(const std::complex<double> &a, const d_anion &b, const d_anion &q_default)
{
    d_anion tempa(a);

    return logbr(tempa,b,q_default);
}

d_anion logb(const d_anion &a, const long &b, const d_anion &q_default)
{
    if ( b > 0 )
    {
        return log(a,q_default)/log((double) b);
    }

    d_anion bb((double) b);

    return log(a,q_default)*inv(log(bb,q_default));
}

d_anion logb(const d_anion &a, const double &b, const d_anion &q_default)
{
    if ( b > 0 )
    {
        return log(a,q_default)/log(b);
    }

    d_anion bb(b);

    return log(a,q_default)*inv(log(bb,q_default));
}

d_anion logb(const d_anion &a, const std::complex<double> &b, const d_anion &q_default)
{
    d_anion tempb(b);

    return logb(a,tempb,q_default);
}

d_anion logbl(const d_anion &a, const std::complex<double>&b, const d_anion &q_default)
{
    d_anion tempb(b);

    return logbl(a,tempb,q_default);
}

d_anion logbr(const d_anion &a, const std::complex<double> &b, const d_anion &q_default)
{
    d_anion tempb(b);

    return logbr(a,tempb,q_default);
}

d_anion logb(const d_anion &a, const d_anion &b, const d_anion &q_default)
{
    return (logbl(a,b,q_default)+logbr(a,b,q_default))/2.0;
}

d_anion logbl(const d_anion &a, const d_anion &b, const d_anion &q_default)
{
    return log(a,q_default)*inv(log(b,q_default));
}

d_anion logbr(const d_anion &a, const d_anion &b, const d_anion &q_default)
{
    return inv(log(b,q_default))*log(a,q_default);
}













d_anion Log(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return log(a,resultdefault);
}

d_anion Log10(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return log10(a,resultdefault);
}

d_anion Logb(const long &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return logb(a,b,resultdefault);
}

d_anion Logb(const double &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return logb(a,b,resultdefault);
}

d_anion Logb(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return logb(a,b,resultdefault);
}

d_anion Logbl(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return logbl(a,b,resultdefault);
}

d_anion Logbr(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return logbr(a,b,resultdefault);
}

d_anion Logb(const d_anion &a, const long &b)
{
    d_anion resultdefault(0,-1);

    return logb(a,b,resultdefault);
}

d_anion Logb(const d_anion &a, const double &b)
{
    d_anion resultdefault(0,-1);

    return logb(a,b,resultdefault);
}

d_anion Logb(const d_anion &a, const std::complex<double> &b)
{
    d_anion resultdefault(0,-1);

    return logb(a,b,resultdefault);
}

d_anion Logbl(const d_anion &a, const std::complex<double>&b)
{
    d_anion resultdefault(0,-1);

    return logbl(a,b,resultdefault);
}

d_anion Logbr(const d_anion &a, const std::complex<double> &b)
{
    d_anion resultdefault(0,-1);

    return logbr(a,b,resultdefault);
}

d_anion Logb(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return logb(a,b,resultdefault);
}

d_anion Logbl(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return logbl(a,b,resultdefault);
}

d_anion Logbr(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,-1);

    return logbr(a,b,resultdefault);
}



d_anion log(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return log(a,resultdefault);
}

d_anion log10(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return log10(a,resultdefault);
}

d_anion logb(const long &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return logb(a,b,resultdefault);
}

d_anion logb(const double &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return logb(a,b,resultdefault);
}

d_anion logb(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return logb(a,b,resultdefault);
}

d_anion logbl(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return logbl(a,b,resultdefault);
}

d_anion logbr(const std::complex<double> &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return logbr(a,b,resultdefault);
}

d_anion logb(const d_anion &a, const long &b)
{
    d_anion resultdefault(0,1);

    return logb(a,b,resultdefault);
}

d_anion logb(const d_anion &a, const double &b)
{
    d_anion resultdefault(0,1);

    return logb(a,b,resultdefault);
}

d_anion logb(const d_anion &a, const std::complex<double> &b)
{
    d_anion resultdefault(0,1);

    return logb(a,b,resultdefault);
}

d_anion logbl(const d_anion &a, const std::complex<double>&b)
{
    d_anion resultdefault(0,1);

    return logbl(a,b,resultdefault);
}

d_anion logbr(const d_anion &a, const std::complex<double> &b)
{
    d_anion resultdefault(0,1);

    return logbr(a,b,resultdefault);
}

d_anion logb(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return logb(a,b,resultdefault);
}

d_anion logbl(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return logbl(a,b,resultdefault);
}

d_anion logbr(const d_anion &a, const d_anion &b)
{
    d_anion resultdefault(0,1);

    return logbr(a,b,resultdefault);
}








#define REAL_ASIN(_x_)    atan2((_x_),sqrt(1-((_x_)*(_x_))))
#define REAL_ACOS(_x_)    atan2(sqrt(1-((_x_)*(_x_))),(_x_))
#define REAL_ATAN(_x_)    atan(_x_)

#define REAL_ASINH(_x_)    log((_x_)+sqrt(((_x_)*(_x_))+1))
#define REAL_ACOSH(_x_)    log((_x_)+sqrt(((_x_)*(_x_))-1))
#define REAL_ATANH(_x_)    (0.5*log((1+(_x_))/(1-(_x_))))

d_anion sin(const d_anion &a)
{
    double R = real(a);
    d_anion M = imagx(a);
    d_anion q = angle(M);
    double I = abs2(M);

    return (sin(R)*cosh(I))+(q*cos(R)*sinh(I));
}

d_anion cos(const d_anion &a)
{
    double R = real(a);
    d_anion M = imagx(a);
    d_anion q = angle(M);
    double I = abs2(M);

    return (cos(R)*cosh(I))-(q*sin(R)*sinh(I));
}

d_anion tan(const d_anion &a)
{
    return sin(a)*inv(cos(a));
}

d_anion cosec(const d_anion &a)
{
    return inv(sin(a));
}

d_anion sec(const d_anion &a)
{
    return inv(cos(a));
}

d_anion cot(const d_anion &a)
{
    return cos(a)*inv(sin(a));
}

d_anion asin(const d_anion &a, const d_anion &q_default)
{
    // Abramowitz and Stegun

    double R = real(a);
    d_anion M = imagx(a);
    d_anion q = angle(M);

    d_anion result;

    if ( abs2(q) < 0.1 ) // could use == 0, but this is equivalent
    {
	if ( ( R <= 1.0 ) && ( R >= -1.0 ) )
	{
	    result = REAL_ASIN(R);
	}

	else
	{
	    d_anion qq_def;

            qq_def = angle(q_default);

	    if ( abs2(qq_def) < 0.1 )
	    {
		d_anion temp(0,1);

		return asin(a,temp);
	    }

	    if ( R < 0.0 )
	    {
                result = -NUMBASE_PION2;
                result += qq_def*(REAL_ACOSH(-R));
	    }

	    else
	    {
                result = NUMBASE_PION2;
                result -= qq_def*(REAL_ACOSH(-R));
	    }
        }
    }

    else
    {
        // Abramowitz and Stegun

	double I = abs2(M);

	if ( abs2(q) < 0.1 ) // could use == 0, but this is equivalent
	{
	    if ( ( R <= 1.0 ) && ( R >= -1.0 ) )
	    {
		result = REAL_ASIN(R);
	    }

	    else
	    {
		if ( R < 0.0 )
		{
                    d_anion temp(-NUMBASE_PION2,REAL_ACOSH(-R));

		    result = temp;
		}

		else
		{
                    d_anion temp(NUMBASE_PION2,-REAL_ACOSH(R));

		    result = temp;
		}
	    }
	}

	else
	{
	    double x = sqrt(R*R);
	    double y = I;
	    double r = sqrt(((x+1)*(x+1))+(y*y));
	    double s = sqrt(((x-1)*(x-1))+(y*y));
	    double A = 0.5*(r+s);
	    double B = x/A;
	    double y2 = y*y;

	    double rreal,rimag;

	    const double A_crossover = 1.5;
	    const double B_crossover = 0.6417;

	    if ( B <= B_crossover )
	    {
		rreal = REAL_ASIN(B);
	    }

	    else
	    {
		if ( x <= 1 )
		{
		    double D = 0.5 * ( A + x ) * ( y2 / ( r + x + 1 ) + ( s + ( 1 - x ) ) );

		    rreal = atan(x/sqrt(D));
		}

		else
		{
		    double Apx = A + x;
		    double D = 0.5 * ( Apx / ( r + x + 1 ) + Apx / ( s + ( x - 1 ) ) );

		    rreal = atan(x/(y*sqrt(D)));
		}
	    }

	    if ( A <= A_crossover )
	    {
		double Am1;

		if ( x < 1 )
		{
		    Am1 = 0.5 * ( y2 / ( r + ( x + 1 ) ) + y2 / ( s + ( 1 - x ) ) );
		}

		else
		{
		    Am1 = 0.5 * ( y2 / ( r + ( x + 1 ) ) + ( s + ( x - 1 ) ) );
		}

		rimag = log( 1 + Am1 + sqrt( Am1 * ( A + 1 ) ) );
	    }

	    else
	    {
		rimag = log( A + sqrt( A * A - 1 ) );
	    }

	    if ( R >= 0.0 )
	    {
		// NB: q takes care of the sign here.

		result = rreal + (q*rimag);
	    }

	    else
	    {
		// NB: q takes care of the sign here

		result = -rreal + (q*rimag);
	    }
	}
    }

    return result;
}

d_anion acos(const d_anion &a, const d_anion &q_default)
{
    // Abramowitz and Stegun

    double R = real(a);
    d_anion M = imagx(a);
    d_anion q = angle(M);

    d_anion result;

    if ( abs2(q) < 0.1 ) // could use == 0, but this is equivalent
    {
	if ( ( R <= 1.0 ) && ( R >= -1.0 ) )
	{
	    result = REAL_ACOS(R);
	}

	else
	{
	    d_anion qq_def;

            qq_def = angle(q_default);

	    if ( abs2(qq_def) < 0.1 )
	    {
		d_anion temp(0,1);

		return acos(a,temp);
	    }

	    if ( R < 0.0 )
	    {
		result = NUMBASE_PI;
                result -= qq_def*(REAL_ACOSH(-R));
	    }

	    else
	    {
                result = qq_def*(REAL_ACOSH(R));
	    }
        }
    }

    else
    {
        // Abramowitz and Stegun

	double I = abs2(M);

	if ( abs2(q) < 0.1 ) // could use == 0, but this is equivalent
	{
	    if ( ( R <= 1.0 ) && ( R >= -1.0 ) )
	    {
		result = REAL_ACOS(R);
	    }

	    else
	    {
		if ( R < 0.0 )
		{
		    d_anion temp(NUMBASE_PI,-REAL_ACOSH(-R));

		    result = temp;
		}

		else
		{
		    d_anion temp(0.0,REAL_ACOSH(R));

		    result = temp;
		}
	    }
	}

	else
	{
	    double x = sqrt(R*R);
	    double y = I;
	    double r = sqrt(((x+1)*(x+1))+(y*y));
	    double s = sqrt(((x-1)*(x-1))+(y*y));
	    double A = 0.5*(r+s);
	    double B = x/A;
	    double y2 = y*y;

	    double rreal,rimag;

	    const double A_crossover = 1.5;
	    const double B_crossover = 0.6417;

	    if ( B <= B_crossover )
	    {
		rreal = REAL_ACOS(B);
	    }

	    else
	    {
		if ( x <= 1 )
		{
		    double D = 0.5 * ( A + x ) * ( y2 / ( r + x + 1 ) + ( s + ( 1 - x ) ) );

		    rreal = atan(sqrt(D)/x);
		}

		else
		{
		    double Apx = A + x;
		    double D = 0.5 * ( Apx / ( r + x + 1 ) + Apx / ( s + ( x - 1 ) ) );

		    rreal = atan((y*sqrt(D))/x);
		}
	    }

	    if ( A <= A_crossover )
	    {
		double Am1;

		if ( x < 1 )
		{
		    Am1 = 0.5 * ( y2 / ( r + ( x + 1 ) ) + y2 / ( s + ( 1 - x ) ) );
		}

		else
		{
		    Am1 = 0.5 * ( y2 / ( r + ( x + 1 ) ) + ( s + ( x - 1 ) ) );
		}

		rimag = log( 1 + Am1 + sqrt( Am1 * ( A + 1 ) ) );
	    }

	    else
	    {
		rimag = log( A + sqrt( A * A - 1 ) );
	    }

	    if ( R >= 0.0 )
	    {
		// NB: q takes care of the sign here

		result = rreal - (q*rimag);
	    }

	    else
	    {
		// NB: q takes care of the sign here

		result = NUMBASE_PI - rreal - (q*rimag);
	    }
	}
    }

    return result;
}

d_anion atan(const d_anion &a)
{
    // Abramowitz and Stegun

    double R = real(a);
    d_anion M = imagx(a);
    d_anion q = angle(M);
    double I = abs2(M);

    d_anion result;

    if ( abs2(q) < 0.1 ) // could use == 0, but this is equivalent
    {
	result = REAL_ATAN(R);
    }

    else
    {
	double r = sqrt((R*R)+(I*I));
	double rreal,rimag;
        double u = 2 * I / ( 1 + r * r );

        // FIXME: the following cross-over should be optimized but 0.1 seems
        //        to work ok

	if ( ( u < 0.1 ) && ( u > -0.1 ) )
	{
	    rimag = 0.25 * ( log( 1 + u ) - log( 1 - u ) );
	}

	else
	{
            double A = sqrt((R*R)+((I+1)*(I+1)));
	    double B = sqrt((R*R)+((I-1)*(I-1)));

	    rimag = 0.5 * log(A/B);
	}

	if ( R == 0 )
	{
	    if ( I > 1 ) // Mabs > 0 by definition.
	    {
                rreal = NUMBASE_PION2;
	    }

	    else
	    {
		rreal = 0.0;
	    }
	}

	else
	{
            rreal = 0.5 * atan2(2*R,((1+r)*(1-r)));
	}

        result = rreal + (q*rimag);
    }

    return result;
}

d_anion acosec(const d_anion &a, const d_anion &q_default)
{
    return asin(inv(a),q_default);
}

d_anion asec(const d_anion &a, const d_anion &q_default)
{
    return acos(inv(a),q_default);
}

d_anion acot(const d_anion &a)
{
    return atan(inv(a));
}

d_anion sinc(const d_anion &a)
{
    d_anion result;

    if ( abs2(a) <= 1e-7 )
    {
	result = 1.0;
    }

    else
    {
	result = sin(a)*inv(a); // yes sin(a)*inv(a) = inv(a)*sin(a)
    }

    return result;
}

d_anion cosc(const d_anion &a)
{
    return cos(a)*inv(a);
}

d_anion tanc(const d_anion &a)
{
    d_anion result;

    if ( abs2(a) <= 1e-7 )
    {
	result = 1.0;
    }

    else
    {
	result = tan(a)*inv(a);
    }

    return result;
}

d_anion vers(const d_anion &a)
{
    return 1-cos(a);
}

d_anion covers(const d_anion &a)
{
    return 1-sin(a);
}

d_anion hav(const d_anion &a)
{
    return vers(a)/2.0;
}

d_anion excosec(const d_anion &a)
{
    return cosec(a)-1;
}

d_anion exsec(const d_anion &a)
{
    return sec(a)-1;
}

d_anion avers(const d_anion &a, const d_anion &q_default)
{
    return acos(a+1,q_default);
}

d_anion acovers(const d_anion &a, const d_anion &q_default)
{
    return asin(a+1,q_default);
}

d_anion ahav(const d_anion &a, const d_anion &q_default)
{
    return avers(2*a,q_default);
}

d_anion aexcosec(const d_anion &a, const d_anion &q_default)
{
    return acosec(a+1,q_default);
}

d_anion aexsec(const d_anion &a, const d_anion &q_default)
{
    return asec(a+1,q_default);
}


d_anion Asin(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return asin(a,resultdefault);
}

d_anion Acos(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return acos(a,resultdefault);
}

d_anion Acosec(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return acosec(a,resultdefault);
}

d_anion Asec(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return asec(a,resultdefault);
}

d_anion Avers(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return avers(a,resultdefault);
}

d_anion Acovers(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return acovers(a,resultdefault);
}

d_anion Ahav(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return ahav(a,resultdefault);
}

d_anion Aexcosec(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return aexcosec(a,resultdefault);
}

d_anion Aexsec(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return aexsec(a,resultdefault);
}


d_anion asin(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return asin(a,resultdefault);
}

d_anion acos(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return acos(a,resultdefault);
}

d_anion acosec(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return acosec(a,resultdefault);
}

d_anion asec(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return asec(a,resultdefault);
}

d_anion avers(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return avers(a,resultdefault);
}
        
d_anion acovers(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return acovers(a,resultdefault);
}

d_anion ahav(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return ahav(a,resultdefault);
}

d_anion aexcosec(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return aexcosec(a,resultdefault);
}

d_anion aexsec(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return aexsec(a,resultdefault);
}












d_anion sinh(const d_anion &a)
{
    double R = real(a);
    d_anion M = imagx(a);
    d_anion q = angle(M);
    double I = abs2(M);

    return (sinh(R)*cos(I))+(q*cosh(R)*sin(I));
}

d_anion cosh(const d_anion &x)
{
    double R = real(x);
    d_anion M = imagx(x);
    d_anion q = angle(M);
    double I = abs2(M);

    return (cosh(R)*cos(I))+(q*sinh(R)*sin(I));
}

d_anion tanh(const d_anion &a)
{
    return sinh(a)*inv(cosh(a));
}

d_anion cosech(const d_anion &a)
{
    return inv(sinh(a));
}

d_anion sech(const d_anion &a)
{
    return inv(cosh(a));
}

d_anion coth(const d_anion &a)
{
    return cosh(a)*inv(sinh(a));
}

d_anion asinh(const d_anion &a)
{
    d_anion q = angle(imagx(a));

    if ( abs2(q) < 0.1 ) // could use == 0, but this is equivalent
    {
	d_anion result(REAL_ASINH(a.realpart()));

        return result;
    }

    return q*asin(conj(q)*a);
}

d_anion acosh(const d_anion &a, const d_anion &q_default)
{
    d_anion q = angle(imagx(a));

    if ( ( abs2(q) < 0.1 ) ) // could use == 0, but this is equivalent
    {
	if ( a.realpart() >= 1 )
	{
	    d_anion result(REAL_ACOSH(a.realpart()));

	    return result;
	}

	else
	{
	    d_anion qq_def;

            qq_def = angle(q_default);

	    if ( abs2(qq_def) < 0.1 )
	    {
		d_anion temp(0,1);

		return acosh(a,temp);
	    }

	    return qq_def*acos(a);
	}
    }

    return q*acos(a);
}

d_anion atanh(const d_anion &a, const d_anion &q_default)
{
    d_anion q = angle(imagx(a));

    if ( abs2(q) < 0.1 ) // could use == 0, but this is equivalent
    {
	if ( ( a.realpart() > -1 ) && ( a.realpart() < 1 ) )
	{
	    d_anion result(REAL_ATANH(a.realpart()));

            return result;
	}

	else
	{
	    d_anion qq_def;

            qq_def = angle(q_default);

	    if ( abs2(qq_def) < 0.1 )
	    {
		d_anion temp(0,1);

		return atanh(a,temp);
	    }

	    return qq_def*atan(conj(qq_def)*a);
	}
    }

    return q*atan(conj(q)*a);
}

d_anion acosech(const d_anion &a)
{
    return asinh(inv(a));
}

d_anion asech(const d_anion &a, const d_anion &q_default)
{
    return acosh(inv(a),q_default);
}

d_anion acoth(const d_anion &a, const d_anion &q_default)
{
    return atanh(inv(a),q_default);
}

d_anion sinhc(const d_anion &a)
{
    d_anion result;

    if ( abs2(a) <= 1e-7 )
    {
	result = 1.0;
    }

    else
    {
	result = sinh(a)*inv(a);
    }

    return result;
}

d_anion coshc(const d_anion &a)
{
    return cosh(a)*inv(a);
}

d_anion tanhc(const d_anion &a)
{
    d_anion result;

    if ( abs2(a) <= 1e-7 )
    {
	result = 1.0;
    }

    else
    {
	result = tanh(a)*inv(a);
    }

    return result;
}

d_anion versh(const d_anion &a)
{
    return 1-cosh(a);
}

d_anion coversh(const d_anion &a)
{
    return 1-sinh(a);
}

d_anion havh(const d_anion &a)
{
    return versh(a)/2.0;
}

d_anion excosech(const d_anion &a)
{
    return cosech(a)-1;
}

d_anion exsech(const d_anion &a)
{
    return sech(a)-1;
}

d_anion aversh(const d_anion &a, const d_anion &q_default)
{
    return acosh(a+1,q_default);
}

d_anion acovrsh(const d_anion &a)
{
    return asinh(a+1);
}

d_anion ahavh(const d_anion &a, const d_anion &q_default)
{
    return aversh(2*a,q_default);
}

d_anion aexcosech(const d_anion &a)
{
    return acosech(a+1);
}

d_anion aexsech(const d_anion &a, const d_anion &q_default)
{
    return asech(a+1,q_default);
}




d_anion Acosh(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return acosh(a,resultdefault);
}

d_anion Atanh(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return atanh(a,resultdefault);
}

d_anion Asech(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return asech(a,resultdefault);
}

d_anion Acoth(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return acoth(a,resultdefault);
}

d_anion Aversh(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return aversh(a,resultdefault);
}

d_anion Ahavh(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return ahavh(a,resultdefault);
}

d_anion Aexsech(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return aexsech(a,resultdefault);
}




d_anion acosh(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return acosh(a,resultdefault);
}

d_anion atanh(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return atanh(a,resultdefault);
}

d_anion asech(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return asech(a,resultdefault);
}

d_anion acoth(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return acoth(a,resultdefault);
}

d_anion aversh(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return aversh(a,resultdefault);
}

d_anion ahavh(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return ahavh(a,resultdefault);
}

d_anion aexsech(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return aexsech(a,resultdefault);
}





d_anion sigm(const d_anion &a)
{
    return inv(1+exp(a));
}

d_anion gd(const d_anion &a)
{
    return 2.0*atan(tanh(a/2.0));
}

d_anion asigm(const d_anion &a, const d_anion &q_default)
{
    return log(inv(a)-1.0,q_default);
}

d_anion agd(const d_anion &a, const d_anion &q_default)
{
    return 2*atanh(tan(a/2.0),q_default);
}

d_anion Asigm(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return asigm(a,resultdefault);
}

d_anion Agd(const d_anion &a)
{
    d_anion resultdefault(0,-1);

    return agd(a,resultdefault);
}

d_anion asigm(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return asigm(a,resultdefault);
}

d_anion agd(const d_anion &a)
{
    d_anion resultdefault(0,1);

    return agd(a,resultdefault);
}








// streams

std::string &d_anion::tostring(std::string &dest) const
{
    if ( is_im )
    {
	std::string partinf;
	std::string part0;

	(*value_inf).tostring(partinf);
	(*value_0).tostring(part0);

        dest = "("+partinf+"|"+part0+")";
    }

    else
    {
             if ( testisvnan(value_real) ) { dest = "vnan()"; }
        else if ( testispinf(value_real) ) { dest = "pinf()"; }
        else if ( testisninf(value_real) ) { dest = "ninf()"; }

        else
        {
            dest = static_cast<std::ostringstream*>( &(std::ostringstream() << value_real) )->str();
        }
    }

    return dest;
}

std::ostream &d_anion::forced_cayley_ostream(std::ostream &output) const
{
    if ( is_im )
    {
	output << "(";
	(*value_inf).forced_cayley_ostream(output);
	output << "|";
	(*value_0).forced_cayley_ostream(output);
	output << ")";
    }

    else
    {
             if ( testisvnan(value_real) ) { output << "vnan()"; }
        else if ( testispinf(value_real) ) { output << "pinf()"; }
        else if ( testisninf(value_real) ) { output << "ninf()"; }

        else
        {
            output << value_real;
        }
    }

    return output;
}

std::ostream &operator<<(std::ostream &output, const d_anion &source)
{
    // Important note: we specifically do not want to strip the reals
    // from in front of i,I,J,K,l,m,n,o,p,q,r, as this would mess
    // with gentype.  To be precise, gentype interprets:
    //
    // i as "i"
    // 1.0i or 1i as the anion 1.0i

    if ( source.is_im )
    {
        if ( ( source.order() == 1 ) && !(source.isindet()) )
	{
            output << source(0) << "+" << source(1) << "i";
	}

        else if ( ( source.order() == 2 ) && !(source.isindet()) )
	{
            output << source(0) << "+" << source(1) << "I" << "+" << source(2) << "J" << "+" << source(3) << "K";
	}

        else if ( ( source.order() == 3 ) && !(source.isindet()) )
	{
            output << source(0) << "+" << source(1) << "l" << "+" << source(2) << "m" << "+" << source(3) << "n" << "+" << source(4) << "o" << "+" << source(5) << "p" << "+" << source(6) << "q" << "+" << source(7) << "r";
	}

	else
	{
            source.forced_cayley_ostream(output);
	}
    }

    else
    {
             if ( testisvnan(source.value_real) ) { output << "vnan()"; }
        else if ( testispinf(source.value_real) ) { output << "pinf()"; }
        else if ( testisninf(source.value_real) ) { output << "ninf()"; }

        else
        {
            output << source.value_real;
        }
    }

    return output;
}

std::istream &operator>>(std::istream &input, d_anion &destin)
{
    std::string temp;

    input >> temp;

    destin = atod_anion(temp.c_str());

    return input;
}

d_anion atod_anion(const char *qwerty, int len)
{
    d_anion result(0.0);
    int errcde = atod_anion_safe(result,qwerty,len);

    (void) errcde;

    NiceAssert(!errcde);

    return result;
}

int atod_anion_safe(d_anion &result, const char *qwerty, int len)
{
    int deep = 0;
    int comma = 0;
    int i,j,k;

    if ( len == -1 )
    {
        len = (int) strlen(qwerty);
    }

    if ( !len ) { return 1;}

    if ( qwerty[0] != '(' )
    {
      // At this point, check if string is just
      // inf or nan.  If it is then set real appropriately, otherwise
      // this is direct format.

      if ( ( len == 6 ) && !strncmp(qwerty,"vnan()",6) )
      {
        result = valvnan();
      }

      else if ( ( len == 6 ) && !strncmp(qwerty,"pinf()",6) )
      {
        result = valpinf();
      }

      else if ( ( len == 6 ) && !strncmp(qwerty,"ninf()",6) )
      {
        result = valninf();
      }

      else
      {
	char *tempa;

        MEMNEWARRAY(tempa,char,2*(len+1));

	strncpy_safe(tempa,qwerty,len);

	int isdirectformat = 0;

	// Direct format is the standard format for writing complex, quaternion and
	// octonion numbers.
	//
	// Complex eg: 1+2i, 1+i2, -4i
	// Quaternion: 2+4I+3J-2K
	// Octonion: 1+2l+3m+4n+5o+6p+7q+8r (note that we start with l for programmatic simplicity)
                           




	// FIXME: put this in a "make string nice" function

	// Replace any E with e

	for ( i = 0 ; i < len ; i++ )
	{
	    if ( tempa[i] == 'E' )
	    {
                tempa[i] = 'e';
	    }
	}

	// Deal with long strings like ++----++++-+

	if ( len > 1 )
	{
	    j = 0;

	    for ( i = 1 ; i < len ; i++ )
	    {
		if ( ( ( tempa[i] == '+' ) && ( tempa[j] == '+' ) ) || ( ( tempa[i] == '-' ) && ( tempa[j] == '-' ) ) )
		{
		    len--;
		    if ( !len ) { return 2;}

		    if ( j < len )
		    {
			for ( k = j ; k < len ; k++ )
			{
                            tempa[k] = tempa[k+1];
			}
		    }

                    tempa[j] = '+';

		    i--;
                    j--;
		}

		else if ( ( ( tempa[i] == '+' ) && ( tempa[j] == '-' ) ) || ( ( tempa[i] == '-' ) && ( tempa[j] == '+' ) ) )
		{
		    len--;
		    if ( !len ) { return 3;}

		    if ( j < len )
		    {
			for ( k = j ; k < len ; k++ )
			{
                            tempa[k] = tempa[k+1];
			}
		    }

                    tempa[j] = '-';

                    i--;
                    j--;
		}

                j++;
	    }
	}

	// Make sure - is used only for negation, so replace any subtraction (- not at start or preceeded by e ( , * /) with +-

	if ( len > 1 )
	{
            j = 0;

	    for ( i = 1 ; i < len ; i++ )
	    {
		if ( ( tempa[i] == '-' ) && ( tempa[j] != 'e' ) && ( tempa[j] != '(' ) && ( tempa[j] != ':' ) && ( tempa[j] != '*' ) && ( tempa[j] != '/' ) )
		{
		    len++;
		    if ( !len ) { return 4;}

		    for ( k = len-1 ; k >= i+1 ; k-- )
		    {
			tempa[k] = tempa[k-1];
		    }

		    tempa[i] = '+';

                    i++;
                    j++;
		}

                j++;
	    }
	}

	// Remove any remaining superfluous posations

	if ( tempa[0] == '+' )
	{
	    len--;
	    if ( !len ) { return 5;}

	    if ( len )
	    {
		for ( k = 0 ; k < len ; k++ )
		{
                    tempa[k] = tempa[k+1];
		}
	    }
	}

	if ( len > 1 )
	{
            j = 0;

	    for ( i = 1 ; i < len ; i++ )
	    {
		if ( ( tempa[i] == '+' ) && ( ( tempa[j] == 'e' ) || ( tempa[j] == '(' ) || ( tempa[j] == ':' ) || ( tempa[j] == '*' ) || ( tempa[j] == '/' ) ) )
		{
		    len--;
		    if ( !len ) { return 6;}

		    if ( i < len )
		    {
			for ( k = i ; k < len ; k++ )
			{
                            tempa[k] = tempa[k+1];
			}
		    }

                    i--;
                    j--;
		}

                j++;
	    }
	}

	tempa[len] = '\0';

	// NB: at this point, + always represents the addition of separate elements, and - always represents negation, and tempa is null terminated

	// FIXME: "make string nice" function ends here







        // Detect direct format

	for ( i = 0 ; i < len ; i++ )
	{
	    if (    ( tempa[i] == '+' )
                 || ( tempa[i] == 'i' )
                 || ( tempa[i] == 'I' )
                 || ( tempa[i] == 'J' )
                 || ( tempa[i] == 'K' )
                 || ( tempa[i] == 'l' )
                 || ( tempa[i] == 'm' )
                 || ( tempa[i] == 'n' )
                 || ( tempa[i] == 'o' )
                 || ( tempa[i] == 'p' )
                 || ( tempa[i] == 'q' )
		 || ( tempa[i] == 'r' ) )
	    {
		isdirectformat = 1;
                break;
	    }
	}

	if ( !isdirectformat )
	{
	    // If not direct format then revert to atof function

            result = atof(tempa);
	}

	else
	{
	    // It is direct format: break up into substrings and process separately

            int elmcnt = 0;

	    // First count number of elements

	    for ( i = 0 ; i < len ; i++ )
	    {
		if ( !i || ( tempa[i] == '+' ) )
		{
		    elmcnt++;
		}
	    }

	    // Allocate data required

	    char **tempsplit;;
	    int *typesplit; // 0 real, 1 complex, 2 quat, 3 oct
            int *splitdest;
            int overtype = 0;

	    MEMNEWARRAY(tempsplit,char *,elmcnt);
	    MEMNEWARRAY(typesplit,int,elmcnt);
            MEMNEWARRAY(splitdest,int,elmcnt);

            // Split and sprinkle

	    j = 0;
            k = 0;

	    for ( i = 0 ; i < len ; i++ )
	    {
		if ( !i || ( tempa[i] == '+' ) )
		{
		    if ( !i )
		    {
			tempsplit[k] = &tempa[i];
		    }

		    else
		    {
			tempsplit[k] = &tempa[i+1];
                        tempa[i] = '\0';
		    }

		    k++;
		}
	    }

	    // Calculate types

	    for ( k = 0 ; k < elmcnt ; k++ )
	    {
		splitdest[k] = 0;
		typesplit[k] = 0;

                i = (int) strlen(tempsplit[k]);

		if ( !i ) { return 7;}

		if ( i == 1 )
		{
		    if ( tempsplit[k][0] == 'i' )
		    {
			splitdest[k] = 1;
			typesplit[k] = 1;
                        tempsplit[k][0] = '1';
		    }

		    else if ( tempsplit[k][0] == 'I' )
		    {
			splitdest[k] = 1;
			typesplit[k] = 2;
                        tempsplit[k][0] = '1';
		    }

		    else if ( tempsplit[k][0] == 'J' )
		    {
			splitdest[k] = 2;
			typesplit[k] = 2;
                        tempsplit[k][0] = '1';
		    }

		    else if ( tempsplit[k][0] == 'K' )
		    {
			splitdest[k] = 3;
			typesplit[k] = 2;
                        tempsplit[k][0] = '1';
		    }

		    else if ( tempsplit[k][0] == 'l' )
		    {
			splitdest[k] = 1;
			typesplit[k] = 3;
                        tempsplit[k][0] = '1';
		    }

		    else if ( tempsplit[k][0] == 'm' )
		    {
			splitdest[k] = 2;
			typesplit[k] = 3;
                        tempsplit[k][0] = '1';
		    }

		    else if ( tempsplit[k][0] == 'n' )
		    {
			splitdest[k] = 3;
			typesplit[k] = 3;
                        tempsplit[k][0] = '1';
		    }

		    else if ( tempsplit[k][0] == 'o' )
		    {
			splitdest[k] = 4;
			typesplit[k] = 3;
                        tempsplit[k][0] = '1';
		    }

		    else if ( tempsplit[k][0] == 'p' )
		    {
			splitdest[k] = 5;
			typesplit[k] = 3;
                        tempsplit[k][0] = '1';
		    }

		    else if ( tempsplit[k][0] == 'q' )
		    {
			splitdest[k] = 6;
			typesplit[k] = 3;
                        tempsplit[k][0] = '1';
		    }

		    else if ( tempsplit[k][0] == 'r' )
		    {
			splitdest[k] = 7;
			typesplit[k] = 3;
                        tempsplit[k][0] = '1';
		    }

		    else
		    {
			splitdest[k] = 0;
			typesplit[k] = 0;
		    }
		}

		else
		{
		    for ( j = 0 ; j < i ; j = ( !j ? i-1 : i ) )
		    {
			if ( tempsplit[k][j] == 'i' )
			{
			    splitdest[k] = 1;
			    typesplit[k] = 1;

			    if ( !j )
			    {
				tempsplit[k] = &tempsplit[k][1];
			    }

			    else if ( ( j == 1 ) && ( tempsplit[k][0] == '-' ) )
			    {
                                tempsplit[k][1] = '1';
			    }

			    else
			    {
				tempsplit[k][j] = '\0';
			    }
			}

			else if ( ( tempsplit[k][j] == 'I' ) || ( tempsplit[k][j] == 'J' ) || ( tempsplit[k][j] == 'K' ) )
			{
			    if ( tempsplit[k][j] == 'I' )
			    {
				splitdest[k] = 1;
			    }

			    else if ( tempsplit[k][j] == 'J' )
			    {
				splitdest[k] = 2;
			    }

			    else if ( tempsplit[k][j] == 'K' )
			    {
				splitdest[k] = 3;
			    }

			    typesplit[k] = 2;

			    if ( !j )
			    {
				tempsplit[k] = &tempsplit[k][1];
			    }

			    else if ( ( j == 1 ) && ( tempsplit[k][0] == '-' ) )
			    {
                                tempsplit[k][1] = '1';
			    }

			    else
			    {
				tempsplit[k][j] = '\0';
			    }
			}

			else if ( ( tempsplit[k][j] == 'l' ) || ( tempsplit[k][j] == 'm' ) || ( tempsplit[k][j] == 'n' ) || ( tempsplit[k][j] == 'o' ) || ( tempsplit[k][j] == 'p' ) || ( tempsplit[k][j] == 'q' ) || ( tempsplit[k][j] == 'r' ) )
			{
			    if ( tempsplit[k][j] == 'l' )
			    {
				splitdest[k] = 1;
			    }

			    else if ( tempsplit[k][j] == 'm' )
			    {
				splitdest[k] = 2;
			    }

			    else if ( tempsplit[k][j] == 'n' )
			    {
				splitdest[k] = 3;
			    }

			    else if ( tempsplit[k][j] == 'o' )
			    {
				splitdest[k] = 4;
			    }

			    else if ( tempsplit[k][j] == 'p' )
			    {
				splitdest[k] = 5;
			    }

			    else if ( tempsplit[k][j] == 'q' )
			    {
				splitdest[k] = 6;
			    }

			    else if ( tempsplit[k][j] == 'r' )
			    {
				splitdest[k] = 7;
			    }

			    typesplit[k] = 3;

			    if ( !j )
			    {
				tempsplit[k] = &tempsplit[k][1];
			    }

			    else if ( ( j == 1 ) && ( tempsplit[k][0] == '-' ) )
			    {
                                tempsplit[k][1] = '1';
			    }

			    else
			    {
				tempsplit[k][j] = '\0';
			    }
			}
		    }
		}

		if ( !( !overtype || !typesplit[k] || ( typesplit[k] == overtype ) ) ) { return 8;}

		if ( typesplit[k] )
		{
                    overtype = typesplit[k];
		}
	    }

            // Set order

	    result.setorder(overtype);

	    // Read and write final result

	    for ( k = 0 ; k < elmcnt ; k++ )
	    {
                result(splitdest[k],result(splitdest[k])+atof(tempsplit[k]));
	    }
	}

        MEMDELARRAY(tempa);
      }
    }

    else
    {
        for ( i = 1 ; i <= len-2 ; i++ )
        {
            if ( qwerty[i] == '(' )
            {
                deep++;
            }

            else if ( qwerty[i] == ')' )
            {
                deep--;
            }

	    else if ( ( qwerty[i] == '|' ) && ( deep == 0 ) )
            {
                comma = i;
                break;
            }
	}

	if ( comma <= 0 )
        {
	    return 0;
	}

        result.leftpart()  = atod_anion(qwerty+1,comma-1);
        result.rightpart() = atod_anion(qwerty+comma+1,len-comma-2);
    }

    //result.simplify();

    return 0;
}

