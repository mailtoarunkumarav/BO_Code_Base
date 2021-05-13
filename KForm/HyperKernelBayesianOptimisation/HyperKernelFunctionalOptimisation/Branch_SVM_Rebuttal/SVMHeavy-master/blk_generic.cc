
//
// Functional block base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"


int dummygetsetExtVar(gentype &res, const gentype &src, int num);
int dummygetsetExtVar(gentype &res, const gentype &src, int num)
{
    (void) res;
    (void) src;
    (void) num;

    throw("No external variables in this context!");

    return -1;
}

BLK_Generic::mexcallsyn BLK_Generic::getsetExtVar = dummygetsetExtVar;


std::ostream &BLK_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Mercer cache size:     " << xmercachesize         << "\n";
    repPrint(output,'>',dep) << "Mercer cache norm:     " << xmercachenorm         << "\n";
    repPrint(output,'>',dep) << "Output function:       " << doutfn                << "\n";
    repPrint(output,'>',dep) << "Is this a sample:      " << xissample             << "\n";
    repPrint(output,'>',dep) << "MEX callback:          " << mexfn                 << "\n";
    repPrint(output,'>',dep) << "MEX callback id:       " << mexfnid               << "\n";
    repPrint(output,'>',dep) << "SYSTEM callback:       " << sysfn                 << "\n";
    repPrint(output,'>',dep) << "SYSTEM x file:         " << xfname                << "\n";
    repPrint(output,'>',dep) << "SYSTEM y file:         " << yfname                << "\n";
    repPrint(output,'>',dep) << "SYSTEM xy file:        " << xyfname               << "\n";
    repPrint(output,'>',dep) << "SYSTEM yx file:        " << yxfname               << "\n";
    repPrint(output,'>',dep) << "SYSTEM r file:         " << rfname                << "\n";
    repPrint(output,'>',dep) << "KB lambda:             " << KBlambda              << "\n";
    repPrint(output,'>',dep) << "ML weight:             " << xmlqweight            << "\n";
    repPrint(output,'>',dep) << "Bernstein degree:      " << berndeg               << "\n";
    repPrint(output,'>',dep) << "Bernstein index:       " << bernind               << "\n";
    repPrint(output,'>',dep) << "Battery parameters:    " << xbattParam            << "\n";
    repPrint(output,'>',dep) << "Battery t_max:         " << xbatttmax             << "\n";
    repPrint(output,'>',dep) << "Battery I_max:         " << xbattImax             << "\n";
    repPrint(output,'>',dep) << "Battery t_delta:       " << xbatttdelta           << "\n";
    repPrint(output,'>',dep) << "Battery V_start:       " << xbattVstart           << "\n";
    repPrint(output,'>',dep) << "Battery theta_start:   " << xbattthetaStart       << "\n";
    repPrint(output,'>',dep) << "Battery non-parasitic: " << xbattneglectParasitic << "\n";
    repPrint(output,'>',dep) << "Battery fixed-temp:    " << xbattfixedTheta       << "\n";

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &BLK_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> xmercachesize;
    input >> dummy; input >> xmercachenorm;
    input >> dummy; input >> doutfn;
    input >> dummy; input >> xissample;
    input >> dummy; input >> mexfn;
    input >> dummy; input >> mexfnid;
    input >> dummy; input >> sysfn;
    input >> dummy; input >> xfname;
    input >> dummy; input >> yfname;
    input >> dummy; input >> xyfname;
    input >> dummy; input >> yxfname;
    input >> dummy; input >> rfname;
    input >> dummy; input >> KBlambda;
    input >> dummy; input >> xmlqweight;
    input >> dummy; input >> berndeg;
    input >> dummy; input >> bernind;
    input >> dummy; input >> xbattParam;
    input >> dummy; input >> xbatttmax;
    input >> dummy; input >> xbattImax;
    input >> dummy; input >> xbatttdelta;
    input >> dummy; input >> xbattVstart;
    input >> dummy; input >> xbattthetaStart;
    input >> dummy; input >> xbattneglectParasitic;
    input >> dummy; input >> xbattfixedTheta;

    ML_Base::inputstream(input);

    return input;
}

int gcallbackdummy(gentype &res, const SparseVector<gentype> &x, void *fndata);
int gcallbackdummy(gentype &res, const SparseVector<gentype> &x, void *fndata)
{
    (void) res;
    (void) x;
    (void) fndata;

    return 0;
}

BLK_Generic::BLK_Generic(int isIndPrune) : ML_Base(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    xuseristream = &instream();
    xuserostream = &outstream();

    xcallback       = &gcallbackdummy;
    xcallbackfndata = NULL;

    xmercachesize = -1;
    xmercachenorm = 0;

    doutfn.makeNull();

    mexfnid = -3;

    xissample = 0;

    berndeg.force_null();
    bernind.force_null();

    // see blk_batter.cc

    xbattParam.resize(21);

    xbattParam("&",0)  = 261.9; // conversion Ah -> coulombs
    xbattParam("&",1)  = 1.18;
    xbattParam("&",2)  = -40;
    xbattParam("&",3)  = 1.29;
    xbattParam("&",4)  = 1.40;
    xbattParam("&",5)  = 49;
    xbattParam("&",6)  = 2.135;
    xbattParam("&",7)  = 0.58e-3;
    xbattParam("&",8)  = 5000;
    xbattParam("&",9)  = 2e-3;
    xbattParam("&",10) = 0.7e-3;
    xbattParam("&",11) = 15e-3;
    xbattParam("&",12) = -0.3;
    xbattParam("&",13) = -8;
    xbattParam("&",14) = -8.45;
    xbattParam("&",15) = 1.95;
    xbattParam("&",16) = 0.1;
    xbattParam("&",17) = 2;
    xbattParam("&",18) = 2e-12;
    xbattParam("&",19) = 15;
    xbattParam("&",20) = 0.2;

    xbatttmax             = 60*60;
    xbattImax             = 30;
    xbatttdelta           = 0.05;
    xbattVstart           = 2.135;
    xbattthetaStart       = 20;
    xbattneglectParasitic = 0;
    xbattfixedTheta       = -1000;

    return;
}

BLK_Generic::BLK_Generic(const BLK_Generic &src, int isIndPrune) : ML_Base(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    xmercachesize = -1;
    xmercachenorm = 0;

    xissample = 0;

    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_Generic::BLK_Generic(const BLK_Generic &src, const ML_Base *xsrc, int isIndPrune) : ML_Base(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    xmercachesize = -1;
    xmercachenorm = 0;

    xissample = 0;

    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_Generic::~BLK_Generic()
{
    return;
}





int BLK_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
{
    int k,res = 0;

    NiceAssert( xa.size() == xb.size() );

    val.resize(xa.size());

    for ( k = 0 ; k < xa.size() ; k++ )
    {
        res |= getparam(ind,val("&",k),xa(k),ia,xb(k),ib);
    }

    return res;
}



int BLK_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const
{
    int res = 0;

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    int isfallback = 0;

    switch ( ind )
    {
        case 1000: { val = outfn();                     break; }
        case 1001: { val = outfngrad();                 break; }
        case 1002: { val.force_string() = getmexcall(); break; }
        case 1003: { val = getmexcallid();              break; }
        case 1004: { val = mercachesize();              break; }
        case 1005: { val = mercachenorm();              break; }
        case 1006: { val.force_string() = getsyscall(); break; }
        case 1007: { val = bernDegree();                break; }
        case 1008: { val = bernIndex();                 break; }
        case 1009: { val = battparam();                 break; }
        case 1010: { val = batttmax();                  break; }
        case 1011: { val = battImax();                  break; }
        case 1012: { val = batttdelta();                break; }
        case 1013: { val = battVstart();                break; }
        case 1014: { val = battthetaStart();            break; }
        case 1015: { val = battneglectParasitic();      break; }
        case 1016: { val = battfixedTheta();            break; }

        default:
        {
            isfallback = 1;
            res = ML_Base::getparam(ind,val,xa,ia,xb,ib);

            break;
        }
    }

    if ( ( ia || ib ) && !isfallback )
    {
        val.force_null();
    }

    return res;
}





