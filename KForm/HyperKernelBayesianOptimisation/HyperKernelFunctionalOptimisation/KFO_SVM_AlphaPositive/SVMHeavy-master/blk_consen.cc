
//
// Consensus result block
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
#include "blk_consen.h"


BLK_Consen::BLK_Consen(int isIndPrune) : BLK_Generic(isIndPrune)
{
    Nnc.resize(1);
    Nnc = zeroint();

    setaltx(NULL);

    return;
}

BLK_Consen::BLK_Consen(const BLK_Consen &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    Nnc.resize(1);
    Nnc = zeroint();

    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_Consen::BLK_Consen(const BLK_Consen &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    Nnc.resize(1);
    Nnc = zeroint();

    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_Consen::~BLK_Consen()
{
    return;
}

std::ostream &BLK_Consen::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Consensus wrapper block\n";

    repPrint(output,'>',dep) << "Label placeholder storage:       " << label_placeholder << "\n";
    repPrint(output,'>',dep) << "Nnc:                             " << Nnc               << "\n\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Consen::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> label_placeholder;
    input >> dummy; input >> Nnc;

    return BLK_Generic::inputstream(input);
}





























int BLK_Consen::addclass(int label, int epszero)
{
    (void) epszero;

    if ( label )
    {
        if ( label_placeholder.findID(label) == -1 )
	{
	    // Add label to ID store

            label_placeholder.findOrAddID(label);
            Nnc.add(label_placeholder.findID(label)+1);
            Nnc("&",label_placeholder.findID(label)+1) = 0;
        }
    }

    return 1;
}

int BLK_Consen::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xx(x);

    return qaddTrainingVector(i,y,xx,Cweigh,epsweigh);
}

int BLK_Consen::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    int res = addclass((int) y);

    Nnc("&",(label_placeholder.findID((int) y)+1))++;

    return res | BLK_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
}

int BLK_Consen::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size()        );
    NiceAssert( y.size() == Cweigh.size()   );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;
    int j;

    if ( y.size() )
    {
        for ( j = 0 ; j < y.size() ; j++ )
        {
            res |= addTrainingVector(i+j,y(j),x(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int BLK_Consen::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size()        );
    NiceAssert( y.size() == Cweigh.size()   );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;
    int j;

    if ( y.size() )
    {
        for ( j = 0 ; j < y.size() ; j++ )
        {
            res |= qaddTrainingVector(i+j,y(j),x("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int BLK_Consen::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &xx)
{
    Nnc("&",(label_placeholder.findID((int) y(i))+1))--;

    return BLK_Generic::removeTrainingVector(i,yy,xx);
}

int BLK_Consen::sety(int i, const gentype  &nd)
{
    return setd(i,(int) nd);
}

int BLK_Consen::sety(const Vector<int> &i, const Vector<gentype> &nd)
{
    NiceAssert( i.size() == nd.size() );

    int res = 0;
    int j;

    if ( i.size() )
    {
        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= sety(i(j),nd(j));
        }
    }

    return res;
}

int BLK_Consen::sety(const Vector<gentype> &nd)
{
    NiceAssert( N() == nd.size() );

    int res = 0;
    int j;

    if ( N() )
    {
        for ( j = 0 ; j < N() ; j++ )
        {
            res |= sety(j,nd(j));
        }
    }

    return res;
}

int BLK_Consen::setd(int i, int nd)
{
    assert( i >= 0 );
    assert( i < N() );
    assert( nd >= -1 );

    int res = 0;

    if ( nd != (int) y(i) )
    {
        res = 1;

        Nnc("&",(label_placeholder.findID((int) y(i))+1))--;
        gentype yn(i);
        BLK_Generic::sety(i,yn);
        Nnc("&",(label_placeholder.findID((int) y(i))+1))++;
    }

    return res;
}

int BLK_Consen::setd(const Vector<int> &i, const Vector<int> &nd)
{
    NiceAssert( i.size() == nd.size() );

    int res = 0;
    int j;

    if ( i.size() )
    {
        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= setd(i(j),nd(j));
        }
    }

    return res;
}

int BLK_Consen::setd(const Vector<int> &nd)
{
    NiceAssert( N() == nd.size() );

    int res = 0;
    int j;

    if ( N() )
    {
        for ( j = 0 ; j < N() ; j++ )
        {
            res |= setd(j,nd(j));
        }
    }

    return res;
}







































int BLK_Consen::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    int res = 0;

    resg.makeNull();

    if ( xindsize(i) )
    {
        int j;

        // Rules
        //
        // x is pure integer (not castable, actual integer) and method is
        // based on voting.

        int isint = 1;
        int intmin = 0;
        gentype temp;

        for ( j = 0 ; j < xindsize(i) ; j++ )
        {
            if ( !xelm(temp,i,j).isValInteger() )
            {
                isint = 0;
                break;
            }

            if ( (int) xelm(temp,i,j) < intmin )
            {
                intmin = (int) xelm(temp,i,j);
            }
        }

        (void) isint;
        NiceAssert( isint );

        SparseVector<int> votes;

        for ( j = 0 ; j < xindsize(i) ; j++ )
        {
            votes("&",((int) xelm(temp,i,j))-intmin)++;
        }

        int maxvotes = 0;

        for ( j = 0 ; j < votes.indsize() ; j++ )
        {
            if ( votes.direcref(j) > maxvotes )
            {
                res = votes.ind(j)+intmin;
                maxvotes = votes.direcref(j);
            }
        }

        resg = res;
    }

    resh = resg;

    return res;
}


