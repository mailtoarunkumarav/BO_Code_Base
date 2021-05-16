
//
// Type-II multi-layer kernel-machine base class
//
// Version: 7
// Date: 06/07/2018
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "mlm_generic.h"

MLM_Generic::MLM_Generic() : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    xmlmlr    = DEFAULT_MLMLR;
    xdiffstop = DEFAULT_DIFFSTOP;
    xlsparse  = DEFAULT_LSPARSE;
    xknum     = -1;

    mltree.resize(0);
    xregtype.resize(0);

    return;
}

MLM_Generic::MLM_Generic(const MLM_Generic &src) : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    xmlmlr    = DEFAULT_MLMLR;
    xdiffstop = DEFAULT_DIFFSTOP;
    xlsparse  = DEFAULT_LSPARSE;
    xknum     = -1;

    mltree.resize(0);
    xregtype.resize(0);

    assign(src,0);

    return;
}

MLM_Generic::MLM_Generic(const MLM_Generic &src, const ML_Base *srcx) : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(srcx);

    xmlmlr    = DEFAULT_MLMLR;
    xdiffstop = DEFAULT_DIFFSTOP;
    xlsparse  = DEFAULT_LSPARSE;
    xknum     = -1;

    mltree.resize(0);
    xregtype.resize(0);

    assign(src,1);

    return;
}

std::ostream &MLM_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Regularisation type:    " << xregtype    << "\n";
    repPrint(output,'>',dep) << "Learning rate:          " << xmlmlr      << "\n";
    repPrint(output,'>',dep) << "Difference stop:        " << xdiffstop   << "\n";
    repPrint(output,'>',dep) << "Randomisation sparsity: " << xlsparse    << "\n";
    repPrint(output,'>',dep) << "Kernel number:          " << xknum       << "\n";
    repPrint(output,'>',dep) << "Kernel tree:            " << mltree      << "\n";
    repPrint(output,'>',dep) << "Underlying SVM:         " << getQconst() << "\n";

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &MLM_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> xregtype;
    input >> dummy; input >> xmlmlr;
    input >> dummy; input >> xdiffstop;
    input >> dummy; input >> xlsparse;
    input >> dummy; input >> xknum;
    input >> dummy; input >> mltree;
    input >> dummy; input >> getQ();

    ML_Base::inputstream(input);

    fixMLTree(); // Need this to fix the pointers in the kernel tree

    return input;
}


int MLM_Generic::prealloc(int expectedN)
{
    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).prealloc(expectedN);
        }
    }

    getQ().prealloc(expectedN);

    return 0;
}

int MLM_Generic::preallocsize(void) const
{
    return getQconst().preallocsize();
}

void MLM_Generic::setmemsize(int expectedN)
{
    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).setmemsize(expectedN);
        }
    }

    getQ().setmemsize(expectedN);

    return;
}

int MLM_Generic::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double sigmaweigh, double epsweigh)
{
    if ( tsize() )
    {
        int ii;

        gentype tempy;
        SparseVector<gentype> tempx;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).addTrainingVector(i,tempy,tempx);
        }
    }

    return getQ().addTrainingVector(i,y,x,sigmaweigh,epsweigh);
}

int MLM_Generic::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double sigmaweigh, double epsweigh)
{
    if ( tsize() )
    {
        int ii;

        gentype tempy;
        SparseVector<gentype> tempx;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).addTrainingVector(i,tempy,tempx);
        }
    }

    return getQ().qaddTrainingVector(i,y,x,sigmaweigh,epsweigh);
}

int MLM_Generic::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &sigmaweigh, const Vector<double> &epsweigh)
{
    if ( tsize() )
    {
        int ii;

        Vector<gentype> tempy(y.size());
        Vector<SparseVector<gentype> > tempx(x.size());

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).addTrainingVector(i,tempy,tempx,sigmaweigh,epsweigh);
        }
    }

    return getQ().addTrainingVector(i,y,x,sigmaweigh,epsweigh);
}

int MLM_Generic::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &sigmaweigh, const Vector<double> &epsweigh)
{
    if ( tsize() )
    {
        int ii;

        Vector<gentype> tempy(y.size());

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            // NB: we qadd with x here as we want x to go to mltree(0), and that is the first in line

            mltree("&",ii).qaddTrainingVector(i,tempy,x,sigmaweigh,epsweigh);
        }
    }

    return getQ().qaddTrainingVector(i,y,x,sigmaweigh,epsweigh);
}

int MLM_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    int res = getQ().removeTrainingVector(i,y,x);

    if ( tsize() )
    {
        int ii;

        for ( ii = tsize()-1 ; ii >= 0 ; ii-- )
        {
            mltree("&",ii).removeTrainingVector(i);
        }
    }

    return res;
}

int MLM_Generic::setd(int i, int nd)
{
    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).setd(i,nd);
        }
    }

    return getQ().setd(i,nd);
}

int MLM_Generic::setd(const Vector<int> &i, const Vector<int> &nd)
{
    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).setd(i,nd);
        }
    }

    return getQ().setd(i,nd);
}

int MLM_Generic::setd(const Vector<int> &nd)
{
    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).setd(nd);
        }
    }

    return getQ().setd(nd);
}

int MLM_Generic::disable(int i)
{
    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).disable(i);
        }
    }

    return getQ().disable(i);
}

int MLM_Generic::disable(const Vector<int> &i)
{
    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).disable(i);
        }
    }

    return getQ().disable(i);
}

int MLM_Generic::randomise(double sparsity)
{
    int res = 0;

/*    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            res |= mltree("&",ii).randomise(sparsity);
        }
    }
*/
    res |= getQ().randomise(sparsity);

    return res;
}

int MLM_Generic::realign(void)
{
    int res = 0;

    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            res |= mltree("&",ii).realign();
        }
    }

    res |= getQ().realign();

    return res;
}

int MLM_Generic::scale(double a)
{
    int res = 0;

/*    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            mltree("&",ii).scale(a);
        }
    }
*/

    res |= getQ().scale(a);

    return res;
}

int MLM_Generic::reset(void)
{
    int res = 0;

    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            res |= mltree("&",ii).reset();
        }
    }

    res |= getQ().reset();

    return res;
}

int MLM_Generic::restart(void)
{
    int res = 0;

    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            res |= mltree("&",ii).restart();
        }
    }

    res |= getQ().restart();

    return res;
}

int MLM_Generic::home(void)
{
    int res = 0;

    if ( tsize() )
    {
        int ii;

        for ( ii = 0 ; ii < tsize() ; ii++ )
        {
            res |= mltree("&",ii).home();
        }
    }

    res |= getQ().home();

    return res;
}
















int MLM_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
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

int MLM_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const
{
    int res = 0;

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    int isfallback = 0;

    switch ( ind )
    {
        case 6000: { val = tsize();    break; }
        case 6001: { val = knum();     break; }
        case 6004: { val = mlmlr();    break; }
        case 6005: { val = diffstop(); break; }
        case 6006: { val = lsparse();  break; }

        case 6100: { val = regtype((int) xa); break; }
        case 6101: { val = regC((int) xa);    break; }
        case 6102: { val = GGp((int) xa);     break; }

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

