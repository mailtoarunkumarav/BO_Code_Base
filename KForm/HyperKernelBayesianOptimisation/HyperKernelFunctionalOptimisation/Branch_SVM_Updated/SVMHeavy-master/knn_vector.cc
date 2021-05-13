
//
// k-nearest-neighbour vector classifier
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
#include "knn_vector.h"


std::ostream &KNN_Vector::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector KNN\n\n";

    repPrint(output,'>',dep) << "Class labels:  " << classlabels << "\n";
    repPrint(output,'>',dep) << "Class counts:  " << classcnt    << "\n";
    repPrint(output,'>',dep) << "Dimension:     " << dim         << "\n";
    repPrint(output,'>',dep) << "Class targets: " << z           << "\n\n";

    KNN_Generic::printstream(output,dep+1);

    return output;
}

std::istream &KNN_Vector::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> classlabels;
    input >> dummy; input >> classcnt;
    input >> dummy; input >> dim;
    input >> dummy; input >> z;

    KNN_Generic::inputstream(input);

    return input;
}

KNN_Vector::KNN_Vector() : KNN_Generic()
{
    setaltx(NULL);

    classlabels.resize(1);
    classcnt.resize(2); // includes class 0 (other two don't)

    classlabels("&",0) = +2;
    classcnt = zeroint();

    dim = -1;

    return;
}

KNN_Vector::KNN_Vector(const KNN_Vector &src) : KNN_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

KNN_Vector::KNN_Vector(const KNN_Vector &src, const ML_Base *xsrc) : KNN_Generic()
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

KNN_Vector::~KNN_Vector()
{
    return;
}

double KNN_Vector::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

int KNN_Vector::addTrainingVector (int i, const gentype &_y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( _y.isCastableToVectorWithoutLoss() );
    NiceAssert( ( dim == -1 ) || ( _y.size() == dim ) );

    gentype y(_y);

    y.morph_vector();

    if ( dim == -1 )
    {
        dim = y.size();
    }

    NiceAssert( y.size() == dim );

    classcnt("&",1)++;

    z.add(i); ::assign(z("&",i),(const Vector<gentype> &) y);

    KNN_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = 2;

    return 1;
}

int KNN_Vector::qaddTrainingVector(int i, const gentype &_y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( _y.isCastableToVectorWithoutLoss() );
    NiceAssert( ( dim == -1 ) || ( _y.size() == dim ) );

    gentype y(_y);

    y.morph_vector();

    if ( dim == -1 )
    {
        dim = y.size();
    }

    NiceAssert( y.size() == dim );

    classcnt("&",1)++;

    z.add(i); ::assign(z("&",i),(const Vector<gentype> &) y);

    KNN_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    KNN_Generic::dd("&",i) = 2;

    return 1;
}

int KNN_Vector::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            addTrainingVector(i+ii,y(ii),x(ii),Cweigh(ii),epsweigh(ii));
        }
    }

    return 1;
}

int KNN_Vector::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    if ( y.size() )
    {
        int ii;

        for ( ii = 0 ; ii < y.size() ; ii++ )
        {
            qaddTrainingVector(i+ii,y(ii),x("&",ii),Cweigh(ii),epsweigh(ii));
        }
    }

    return 1;
}

int KNN_Vector::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    if ( d()(i) )
    {
        classcnt("&",dd(i)/2)--;
    }

    z.remove(i);

    KNN_Generic::removeTrainingVector(i,yy,x);

    return 1;
}

int KNN_Vector::sety(int i, const gentype &_yy)
{
    NiceAssert( _yy.isCastableToVectorWithoutLoss() );
    NiceAssert( ( dim == -1 ) || ( _yy.size() == dim ) );

    gentype yy(_yy);

    yy.morph_vector();

    if ( dim == -1 )
    {
        dim = yy.size();
    }

    ::assign(z("&",i),(const Vector<gentype> &) yy);

    KNN_Generic::sety(i,yy);

    return 1;
}

int KNN_Vector::sety(const Vector<int> &i, const Vector<gentype> &y)
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

int KNN_Vector::sety(const Vector<gentype> &y)
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

int KNN_Vector::setd(int i, int dd)
{
    NiceAssert( ( dd == +2 ) || ( dd == 0 ) );

    KNN_Generic::setd(i,dd);

    return 1;
}

int KNN_Vector::setd(const Vector<int> &i, const Vector<int> &d)
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

int KNN_Vector::setd(const Vector<int> &d)
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

void KNN_Vector::hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;
    (void) effkay;

    if ( !res.isValVector() ) { res.force_vector(dim); }
    setzero(res.dir_vector());

    if ( yk.size() )
    {
        mean(res,yk,weights);
    }

    return;
}

void KNN_Vector::hfn(Vector<double> &res, const Vector<Vector<double> > &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
{
    (void) Nnz;
    (void) kdistsq;
    (void) effkay;

    res.resize(dim);
    setzero(res);

    if ( yk.size() )
    {
        mean(res,yk,weights);
    }

    return;
}

int KNN_Vector::randomise(double sparsity)
{
    (void) sparsity;

    int res = 0;
    int Nnotz = N()-NNC(0);

    if ( Nnotz )
    {
        res = 1;

        retVector<int> tmpva;

        Vector<int> canmod(cntintvec(N(),tmpva));

        int i,j,k;

        for ( i = N()-1 ; i >= 0 ; i-- )
        {
            if ( !d()(i) )
            {
                canmod.remove(i);
            }
        }

        // Randomise

        double lbloc = -1.0;
        double ubloc = +1.0;

        for ( i = 0 ; i < canmod.size() ; i++ )
        {
            j = canmod(i);

            NiceAssert( d()(j) );

            Vector<gentype> &bmod = y_unsafe()("&",j).dir_vector();

            if ( bmod.size() )
            {
                for ( k = 0 ; k < bmod.size() ; k++ )
                {
                    double &amod = bmod("&",k).force_double();

                    setrand(amod);
                    amod = lbloc+((ubloc-lbloc)*amod);
                }
            }

            ::assign(z("&",j),bmod);
        }
    }

    return res;
}

int KNN_Vector::settspaceDim(int newdim)
{
    NiceAssert( ( ( N() == 0 ) && ( newdim >= -1 ) ) || ( newdim >= 0 ) );

    dim = newdim;

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_vector().resize(dim);
            z("&",ii).resize(dim);
        }
    }

    return 1;
}

int KNN_Vector::addtspaceFeat(int i)
{
    NiceAssert( ( ( i >= 0 ) && ( i <= dim ) ) || ( dim == -1 ) );

    dim++;

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_vector().add(i);
            z("&",ii).add(i);
        }
    }

    return 1;
}

int KNN_Vector::removetspaceFeat(int i)
{
    NiceAssert( ( i >= 0 ) && ( i < dim ) );

    dim--;

    if ( N() )
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ii++ )
        {
            y_unsafe()("&",ii).dir_vector().remove(i);
            z("&",ii).add(i);
        }
    }

    return 1;
}
