
//
// 1 layer neural network base class
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
#include <ctime>
#include "onn_generic.h"

#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 1000


ONN_Generic::ONN_Generic() : ML_Base()
{
    getKernel_unsafe().setLeftPlain();
    getKernel_unsafe().setType(201);

    setaltx(NULL);

    dC       = NN_DEFAULT_C;
    dzt      = NN_DEFAULT_ZTOL;
    dot      = NN_DEFAULT_OPTTOL;
    dmitcnt  = NN_DEFAULT_MAXITCNT;
    dmtrtime = NN_DEFAULT_MAXTRAINTIME;
    dlr      = NN_DEFAULT_LR;

    return;
}

ONN_Generic::ONN_Generic(const ONN_Generic &src) : ML_Base()
{
    getKernel_unsafe().setLeftPlain();
    getKernel_unsafe().setType(201);

    setaltx(NULL);

    assign(src,0);

    return;
}

ONN_Generic::ONN_Generic(const ONN_Generic &src, const ML_Base *xsrc) : ML_Base()
{
    getKernel_unsafe().setLeftPlain();
    getKernel_unsafe().setType(201);

    setaltx(xsrc);

    assign(src,0);

    return;
}

ONN_Generic::~ONN_Generic()
{
    return;
}

int ONN_Generic::prealloc(int expectedN)
{
    ML_Base::prealloc(expectedN);

    return 0;
}

int ONN_Generic::preallocsize(void) const
{
    return ML_Base::preallocsize();
}

std::ostream &ONN_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "ONN W:      " << dW     << "\n";
    repPrint(output,'>',dep) << "ONN b:      " << db     << "\n";
    repPrint(output,'>',dep) << "ONN W info: " << dWinfo << "\n";

    repPrint(output,'>',dep) << "ONN C:                      " << dC       << "\n";
    repPrint(output,'>',dep) << "ONN zero tolerance:         " << dzt      << "\n";
    repPrint(output,'>',dep) << "ONN optimality tolerance:   " << dot      << "\n";
    repPrint(output,'>',dep) << "ONN maximum training iters: " << dmitcnt  << "\n";
    repPrint(output,'>',dep) << "ONN maximum training time:  " << dmtrtime << "\n";
    repPrint(output,'>',dep) << "ONN learning rate:          " << dlr      << "\n";

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &ONN_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> dW;
    input >> dummy; input >> db;
    input >> dummy; input >> dWinfo;

    input >> dummy; input >> dC;
    input >> dummy; input >> dzt;
    input >> dummy; input >> dot;
    input >> dummy; input >> dmitcnt;
    input >> dummy; input >> dmtrtime;
    input >> dummy; input >> dlr;

    ML_Base::inputstream(input);

    return input;
}

int ONN_Generic::randomise(double sparsity)
{
    NiceAssert( sparsity >= 0 );
    NiceAssert( sparsity <= 1 );

    int i;
    int res = 0;
    int Nnotz = (int) (xspaceDim()*sparsity);
    int tdim = tspaceDim();
    int tord = order();

    dW.softzero();
    db.zero();

    if ( !Nnotz && xspaceDim() )
    {
        res = 1;
    }

    else if ( Nnotz )
    {
        res = 1;

        Vector<int> canmod(indKey());

        // Observe sparsity

        while ( canmod.size() > Nnotz )
        {
            canmod.remove(svm_rand()%(canmod.size()));
        }

        Vector<gentype> newalpha(N());
        gentype newbias;

        if ( isUnderlyingScalar() )
        {
            gentype zerotemplate(0.0);

            // Set zero

            newalpha = zerotemplate;
            newbias  = zerotemplate;
        }

        else if ( isUnderlyingVector() )
        {
            Vector<double> temp(tdim);
            temp = 0.0;
            gentype zerotemplate(temp);


            // Set zero

            newalpha = zerotemplate;
            newbias  = zerotemplate;
        }

        else if ( isUnderlyingAnions() )
        {
            d_anion temp(tord); // implicitly zeros new variable
            gentype zerotemplate(temp);

            // Set zero

            newalpha = zerotemplate;
            newbias  = zerotemplate;
        }

        else
        {
            newbias  = db;
            newbias.zero();
            newalpha = newbias;
        }

        //newalpha.indalign(canmod);
        newalpha.resize(canmod.size());

        // Next randomise

        for ( i = 0 ; i < canmod.size() ; i++ )
        {
            setrand(newalpha("&",canmod(i)));
        }

        setrand(newbias);

        // Set alpha and bias

        SparseVector<gentype> tmpnewalpha(newalpha);

        setW(tmpnewalpha);
        setB(newbias);
    }

    return res;
}

int ONN_Generic::scale(double a)
{
    dW.scale(a);
    db *= a;

    dC *= a;

    getKernel().getvecInfo(dWinfo,dW);

    return 1;
}

int ONN_Generic::reset(void)
{
    dW.softzero();
    db.zero();

    getKernel().getvecInfo(dWinfo,dW);

    return 1;
}

int ONN_Generic::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    resh = K2(resg,i,-2,db);

    return 1;
}

int ONN_Generic::settspaceDim(int newdim)
{
    NiceAssert( ( ( N() == 0 ) && ( newdim >= -1 ) ) || ( newdim >= 0 ) );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_vector().resize(newdim);
        }
    }

    db.dir_vector().resize(newdim);

    if ( dW.indsize() )
    {
        int ii;

        for ( ii = 0 ; ii < dW.indsize() ; ii++ )
        {
            dW.direref(ii).dir_vector().resize(newdim);
        }

        getKernel().getvecInfo(dWinfo,dW);
    }

    return 1;
}

int ONN_Generic::addtspaceFeat(int i)
{
    NiceAssert( ( i >= 0 ) && ( i <= tspaceDim() ) );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_vector().add(i);
        }
    }

    db.dir_vector().add(i);

    if ( dW.indsize() )
    {
        int ii;

        for ( ii = 0 ; ii < dW.indsize() ; ii++ )
        {
            dW.direref(ii).dir_vector().add(i);
        }

        getKernel().getvecInfo(dWinfo,dW);
    }

    return 1;
}

int ONN_Generic::removetspaceFeat(int i)
{
    NiceAssert( ( i >= 0 ) && ( i < tspaceDim() ) );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_vector().remove(i);
        }
    }

    db.dir_vector().remove(i);

    if ( dW.indsize() )
    {
        int ii;

        for ( ii = 0 ; ii < dW.indsize() ; ii++ )
        {
            dW.direref(ii).dir_vector().remove(i);
        }

        getKernel().getvecInfo(dWinfo,dW);
    }

    return 1;
}

int ONN_Generic::setorder(int neword)
{
    NiceAssert( neword >= 0 );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_anion().setorder(neword);
        }
    }

    db.dir_anion().setorder(neword);

    if ( dW.indsize() )
    {
        int ii;

        for ( ii = 0 ; ii < dW.indsize() ; ii++ )
        {
            dW.direref(ii).dir_anion().setorder(neword);
        }

        getKernel().getvecInfo(dWinfo,dW);
    }

    return 1;
}

int ONN_Generic::addxspaceFeat(int i)
{
    (void) i;

    dW.resize(xspaceDim());

    return 1;
}

int ONN_Generic::removexspaceFeat(int i)
{
    (void) i;

    dW.resize(xspaceDim());

    return 1;
}

int ONN_Generic::train(int &res, svmvolatile int &killSwitch)
{
    (void) res;

    int fres = 0;
//    int dummy;

    if ( N() )
    {
        fres = 1;

        // E = 1/2 ||g(x)-y||^2
        //   = 1/2 ||K(x,W,b)-y||^2
        //
        // dE/dW = (g(x)-y) dK/dy
        // dE/db = (g(x)-y) dK/db

        gentype dyxscaleres(gOutType());
        gentype dyyscaleres(gOutType());

        SparseVector<gentype> Wstep;
        gentype Bstep(gOutType());
        gentype E(gOutType());

        double maxitcnt = dmitcnt;
        double xlr = lr();
        double xmtrtime = maxtraintime();
        double xot = Opttol();
        double *uservars[] = { &maxitcnt, &xmtrtime, &xlr, &xot, NULL };
        const char *varnames[] = { "itercount", "traintime", "learning rate", "optimisation tolerance", NULL };
        const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Learning rate", "Optimality tolerance", NULL };

        int isopt = 0;

        time_used start_time = TIMECALL;
        time_used curr_time = start_time;
        unsigned long long itcnt = 0;
        int timeout = 0;

        int i;

        while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcnt ) || !maxitcnt ) && !timeout )
        {
            isopt = 1;

            for ( i = 0 ; i < N() ; i++ )
            {
                Bstep = 0.0;
                Wstep.softzero();

                if ( d()(i) )
                {
//                    dKdy(dyxscaleres,dyyscaleres,dummy,i,-2,db);

//                    NiceAssert( dummy < 0 );

//                    eTrainingVector(E,i);

                    dyxscaleres *= -xlr*E;
                    dyyscaleres *= -xlr*E;

                    Bstep += dyxscaleres;
                    Bstep += dyyscaleres*db;

                    Wstep.scaleAdd(dyxscaleres,x(i));
                    Wstep.scaleAdd(dyyscaleres-(1.0/dC),dW);

                    dW += Wstep;
                    db += Bstep;

                    if ( (double) abs2(E) > xot )
                    {
                        isopt = 0;
                    }
                }
            }

            if ( !(++itcnt%FEEDBACK_CYCLE) )
            {
                if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
                {
                    errstream() << "|\b";
                }

                else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
                {
                    errstream() << "/\b";
                }

                else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
                {
                    errstream() << "-\b";
                }

                else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
                {
                    errstream() << "\\\b";
                }
            }

            if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
            {
                errstream() << "=" << itcnt << "=  ";
            }

            if ( xmtrtime > 1 )
            {
                curr_time = TIMECALL;

                if ( TIMEDIFFSEC(curr_time,start_time) > xmtrtime )
                {
                    timeout = 1;
                }
            }

            if ( !timeout )
            {
                timeout = kbquitdet("ONN training",uservars,varnames,vardescr);
            }
        }
    }

    incgvernum();







    return fres;
}























int ONN_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
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

int ONN_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const
{
    int res = 0;

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    int isfallback = 0;

    switch ( ind )
    {
        case 7000: { val = lr();                  break; }
        case 7001: { convertSparseToSet(val,W()); break; }
        case 7004: { val = B();                   break; }

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

