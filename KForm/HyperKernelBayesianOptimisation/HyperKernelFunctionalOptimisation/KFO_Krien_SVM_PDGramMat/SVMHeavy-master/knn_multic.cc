
//
// k-nearest-neighbour multi-class classifier
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
#include "knn_multic.h"


std::ostream &KNN_MultiC::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Binary KNN\n\n";

    repPrint(output,'>',dep) << "Class labels: " << label_placeholder << "\n";
    repPrint(output,'>',dep) << "Class counts: " << Nnc               << "\n\n";

    KNN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &KNN_MultiC::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> label_placeholder;
    input >> dummy; input >> Nnc;

    KNN_Generic::inputstream(input);

    return input;
}

KNN_MultiC::KNN_MultiC() : KNN_Generic()
{
    setaltx(NULL);

    Nnc.resize(1);
    Nnc = zeroint();

    return;
}

KNN_MultiC::KNN_MultiC(const KNN_MultiC &src) : KNN_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

KNN_MultiC::KNN_MultiC(const KNN_MultiC &src, const ML_Base *xsrc) : KNN_Generic()
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

KNN_MultiC::~KNN_MultiC()
{
    return;
}

double KNN_MultiC::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int KNN_MultiC::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isValInteger() );

    if ( ((int) y) != 0 )
    {
        addclass((int) y);
    }

    Nnc("&",( ((int) y) ? (label_placeholder.findID((int) y)+1) : 0 ))++;

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = (int) y;

    return 1;
}

int KNN_MultiC::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y.isValInteger() );

    if ( ((int) y) != 0 )
    {
        addclass((int) y);
    }

    Nnc("&",( ((int) y) ? (label_placeholder.findID((int) y)+1) : 0 ))++;

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = (int) y;

    return 1;
}

int KNN_MultiC::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            NiceAssert( y(ii).isValInteger() );

            if ( ((int) y(ii)) != 0 )
            {
                addclass((int) y(ii));
            }

            Nnc("&",( ((int) y(ii)) ? (label_placeholder.findID((int) y(ii))+1) : 0 ))++;
        }
    }

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            KNN_Generic::dd("&",i+ii) = (int) y(ii);
        }
    }

    return 1;
}

int KNN_MultiC::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            NiceAssert( y(ii).isValInteger() );

            if ( ((int) y(ii)) != 0 )
            {
                addclass((int) y(ii));
            }

            Nnc("&",( ((int) y(ii)) ? (label_placeholder.findID((int) y(ii))+1) : 0 ))++;
        }
    }

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            KNN_Generic::dd("&",i+ii) = (int) y(ii);
        }
    }

    return 1;
}

int KNN_MultiC::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        Nnc("&",( (int) y()(i) ? (label_placeholder.findID((int) y()(i))+1) : 0 ))--;
    }

    KNN_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

int KNN_MultiC::sety(int i, const gentype &yy)
{
    NiceAssert( yy.isValInteger() );
    NiceAssert( ( (int) yy == +1 ) || ( (int) yy == -1 ) );

    if ( d()(i) )
    {
        Nnc("&",( (int) y()(i) ? (label_placeholder.findID((int) y()(i))+1) : 0 ))--;
        Nnc("&",( (int) yy ? (label_placeholder.findID((int) yy)+1) : 0 ))++;
    }

    KNN_Generic::sety(i,yy);

    if ( d()(i) )
    {
        KNN_Generic::dd("&",i) = (int) yy;
    }

    return 1;
}

int KNN_MultiC::sety(const Vector<int> &i, const Vector<gentype> &y)
{
    NiceAssert( i.size() == y.size() );

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            sety(i(ii),y(ii));
        }
    }

    return 1;
}

int KNN_MultiC::sety(const Vector<gentype> &y)
{
    NiceAssert( N() == y.size() );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            sety(ii,y(ii));
        }
    }

    return 1;
}

int KNN_MultiC::setd(int i, int dd)
{
    NiceAssert( ( dd == +1 ) || ( dd == -1 ) || ( dd == 0 ) );

    KNN_Generic::setd(i,dd);

    if ( dd )
    {
        gentype yy(dd);

        KNN_Generic::sety(i,yy);
    }

    return 1;
}

int KNN_MultiC::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i.size() == d.size() );

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            setd(i(ii),d(ii));
        }
    }

    return 1;
}

int KNN_MultiC::setd(const Vector<int> &d)
{
    NiceAssert( N() == d.size() );

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            setd(ii,d(ii));
        }
    }

    return 1;
}

int KNN_MultiC::addclass(int label, int epszero)
{
    NiceAssert( ( label == -1 ) || ( label >= 1 ) );
    NiceAssert( !epszero );

    (void) epszero;

    if ( label )
    {
        if ( label_placeholder.findID(label) == -1 )
        {
            label_placeholder.findOrAddID(label);
            Nnc.add(label_placeholder.findID(label)+1);
            Nnc("&",label_placeholder.findID(label)+1) = 0;
        }
    }

    return 1;
}

void KNN_MultiC::hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;

    if ( !res.isValInteger() ) { res.force_int(); }
    setzero(res.dir_int());

    // Take 1: simple vote
    // Take 2: which is closest?
    // Take 3: arbitrarily label -1

    if ( effkay )
    {
        Vector<double> vote(numClasses());
        Vector<int> maxset;
        int i = 0;
        double maxvote;
        
        vote = 0.0;
        
        for ( i = 0 ; i < numClasses() ; i++ )
        {
            vote("&",getInternalClass(yk(i))) += weights(i);
        }

        maxvote = max(vote,i);
        
        for ( i = 0 ; i < numClasses() ; i++ )
        {
            if ( vote(i) == maxvote )
            {
                maxset.add(maxset.size());
                maxset("&",maxset.size()-1) = i;
            }
        }
        
        res = yk(maxset(zeroint()));
    }

    else
    {
        res = -1; // This is arbitrary
    }

    return;
}


int KNN_MultiC::randomise(double sparsity)
{
    (void) sparsity;

    int res = 0;
    int Nnotz = N()-NNC(0);

    if ( Nnotz )
    {
        res = 1;

        retVector<int> tmpva;

        Vector<int> canmod(cntintvec(N(),tmpva));

        int i,j;

        for ( i = N()-1 ; i >= 0 ; i-- )
        {
            if ( !d()(i) )
            {
                canmod.remove(i);
            }
        }

        // Randomise

        for ( i = 0 ; i < canmod.size() ; i++ )
        {
            j = canmod(i);

            NiceAssert( d()(j) );

            int &amod = y_unsafe()("&",j).force_int();

            setrand(amod);
        }
    }

    return res;
}
