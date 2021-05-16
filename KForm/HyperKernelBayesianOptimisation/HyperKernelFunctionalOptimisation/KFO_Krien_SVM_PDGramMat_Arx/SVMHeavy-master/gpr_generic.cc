
//
// Gaussian Process (GP) base class
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
#include "gpr_generic.h"

GPR_Generic::GPR_Generic() : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    sampleMode = 0;

    setaltx(NULL);

    Nnc.resize(4);
    Nnc = zeroint();

    return;
}

GPR_Generic::GPR_Generic(const GPR_Generic &src) : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    sampleMode = 0;

    setaltx(NULL);

    Nnc.resize(4);
    Nnc = zeroint();

    assign(src,0);

    return;
}

GPR_Generic::GPR_Generic(const GPR_Generic &src, const ML_Base *srcx) : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    sampleMode = 0;

    setaltx(srcx);

    Nnc.resize(4);
    Nnc = zeroint();

    assign(src,1);

    return;
}

int GPR_Generic::prealloc(int expectedN)
{
    dy.prealloc(expectedN);
    dsigmaweight.prealloc(expectedN);
    dCweight.prealloc(expectedN);
    xd.prealloc(expectedN);
    getQ().prealloc(expectedN);

    return 0;
}

std::ostream &GPR_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Measurement sigma:         " << dsigma       << "\n";
    repPrint(output,'>',dep) << "Measurement sigma weights: " << dsigmaweight << "\n";
    repPrint(output,'>',dep) << "Measurement C weights:     " << dCweight     << "\n";
    repPrint(output,'>',dep) << "Local (actual) y:          " << dy           << "\n";
    repPrint(output,'>',dep) << "Local (actual) d:          " << xd           << "\n";
    repPrint(output,'>',dep) << "Class counts Nnc:          " << Nnc          << "\n";
    repPrint(output,'>',dep) << "Sample mode:               " << sampleMode   << "\n";
    repPrint(output,'>',dep) << "Underlying LS-SVM:         " << getQconst()  << "\n\n";

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &GPR_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> dsigma;
    input >> dummy; input >> dsigmaweight;
    input >> dummy; input >> dCweight;
    input >> dummy; input >> dy;
    input >> dummy; input >> xd;
    input >> dummy; input >> Nnc;
    input >> dummy; input >> sampleMode;
    input >> dummy; input >> getQ();

    ML_Base::inputstream(input);

    return input;
}

int GPR_Generic::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    dsigmaweight.add(i);
    dsigmaweight("&",i) = 1/Cweigh;

    dCweight.add(i);
    dCweight("&",i) = Cweigh;

    xd.add(i);
    xd("&",i) = 2;

    dy.add(i);
    dy("&",i) = y;

    Nnc("&",xd(i)+1)++;

    return getQ().addTrainingVector(i,y,x,Cweigh,epsweigh);
}

int GPR_Generic::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    dsigmaweight.add(i);
    dsigmaweight("&",i) = 1/Cweigh;

    dCweight.add(i);
    dCweight("&",i) = Cweigh;

    xd.add(i);
    xd("&",i) = 2;

    dy.add(i);
    dy("&",i) = y;

    Nnc("&",xd(i)+1)++;

    return getQ().qaddTrainingVector(i,y,x,Cweigh,epsweigh);
}

int GPR_Generic::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            res |= addTrainingVector(i+j,y(j),x(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int GPR_Generic::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            res |= qaddTrainingVector(i+j,y(j),x("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int GPR_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    Nnc("&",xd(i)+1)--;

    y = dy(i);

    dsigmaweight.remove(i);
    dCweight.remove(i);
    xd.remove(i);
    dy.remove(i);

    gentype dummy;

    return getQ().removeTrainingVector(i,dummy,x);
}

int GPR_Generic::removeTrainingVector(int i, int num)
{
    int res = 0;
    gentype y;
    SparseVector<gentype> x;

    while ( num )
    {
        num--; 
        res |= removeTrainingVector(i+num,y,x);
    }

    return res;
}

int GPR_Generic::setsigmaweight(int i, double nv)
{
    dsigmaweight("&",i) = nv;
    dCweight("&",i) = 1/nv;

    return getQ().setCweight(i,1/nv);
}

int GPR_Generic::setsigmaweight(const Vector<int> &i, const Vector<double> &nv)
{
    NiceAssert( i.size() == nv.size() );

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            dCweight("&",i(j)) = 1/nv(j);
            dsigmaweight("&",i(j)) = nv(j);
        }
    }

    retVector<double> tmpva;

    return getQ().setCweight(i,dCweight(i,tmpva));
}

int GPR_Generic::setsigmaweight(const Vector<double> &nv)
{
    NiceAssert( N() == nv.size() );

    if ( nv.size() )
    {
        int j;

        for ( j = 0 ; j < nv.size() ; j++ )
        {
            dCweight("&",j) = 1/nv(j);
            dsigmaweight("&",j) = nv(j);
        }
    }

    return getQ().setCweight(dCweight);
}




int GPR_Generic::setCweight(int i, double nv)
{
    dCweight = nv;
    dsigmaweight("&",i) = 1/nv;

    return getQ().setCweight(i,nv);
}

int GPR_Generic::setCweight(const Vector<int> &i, const Vector<double> &nv)
{
    NiceAssert( i.size() == nv.size() );

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            dCweight("&",i(j)) = nv(j);
            dsigmaweight("&",i(j)) = 1/nv(j);
        }
    }

    return getQ().setCweight(i,nv);
}

int GPR_Generic::setCweight(const Vector<double> &nv)
{
    NiceAssert( N() == nv.size() );

    if ( nv.size() )
    {
        int j;

        for ( j = 0 ; j < nv.size() ; j++ )
        {
            dCweight("&",j) = nv(j);
            dsigmaweight("&",j) = 1/nv(j);
        }
    }

    return getQ().setCweight(nv);
}

int GPR_Generic::scaleCweight(double s)
{
    dsigmaweight *= 1/s;
    dCweight *= s;

    return getQ().scaleCweight(s);
}

int GPR_Generic::scalesigmaweight(double s)
{
    dsigmaweight *= s;
    dCweight *= 1/s;

    return getQ().scaleCweight(1/s);
}

int GPR_Generic::setd(int i, int nd) 
{ 
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    // for anions and vectors setting d == -1,+1 makes no sense

    NiceAssert( ( nd == 0 ) || ( nd == 2 ) || ( type() != 401 ) || ( type() != 402 ) );

    Nnc("&",xd(i)+1)--;
    Nnc("&",nd+1)++;

    xd("&",i) = nd;

    // the LSV layer only sees 0 (constrained) or 2 (unconstrained)

    return getQ().setd(i,nd ? 2 : 0); 
}

int GPR_Generic::setd(const Vector<int> &i, const Vector<int> &nd) 
{ 
    NiceAssert( i.size() == nd.size() );

    int res = 0;

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ ) 
        {
            res |= setd(i(ii),nd(ii));
        }
    }

    return res;
}

int GPR_Generic::setd(const Vector<int> &nd)
{ 
    NiceAssert( N() == nd.size() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ ) 
        {
            res |= setd(i,nd(i));
        }
    }

    return res;
}



















int GPR_Generic::cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf, const vecInfo *xbinf, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
    int res = 0;

    if ( isSampleMode() )
    {
        resv = 0.0;
        gg(resmu,xa,xainf,pxyprodi);
    }

    else
    {
        res = getQconst().cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodi,pxyprodj,pxyprodij);
    }

    return res;
}

int GPR_Generic::var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf, gentype ***pxyprodx, gentype **pxyprodxx) const
{
    int res = 0;

    if ( isSampleMode() )
    {
        resv = 0.0;
        gg(resmu,xa,xainf,pxyprodx);
    }

    else
    {
        res = getQconst().var(resv,resmu,xa,xainf,pxyprodx,pxyprodxx);
    }

    return res;
}

int GPR_Generic::covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const
{
    int res = 0;

    if ( isSampleMode() )
    {
        gentype dummy;
        resv.resize(x.size(),x.size()) = ( dummy = 0.0 );
    }

    else
    {
        res = getQconst().covar(resv,x);
    }

    return res;
}

int GPR_Generic::scale(double _sf)
{
    int res = 0;
    gentype sf(_sf);

    dy *= sf;
    res |= sety(dy);

    getKernel_unsafe() *= sf;
    res |= resetKernel(0);

    return res;
}

















int GPR_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
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

int GPR_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const
{
    int res = 0;

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    int isfallback = 0;

    switch ( ind )
    {
        case 2000: { val = muWeight();     break; }
        case 2001: { val = muBias();       break; }
        case 2002: { val = isZeromuBias(); break; }
        case 2003: { val = isVarmuBias();  break; }
        case 2004: { val = isSampleMode(); break; }

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


















//NB ************* THIS IS IMPORTANT ************************
//
// For speed, we need a *consistent*, *linear* sampling for x.  The reason for this
// is that, in smboopt (sequential model-based optimisation) the model must
// compute inner products between sampled models, and, while that can be 
// done autonomously using global functions, is too slow.  However the y()
// vector effectively represents points at which g(x) has been (effectively)
// pre-computed, so if y() corresponds to the same x for two different GPRs
// then we can simply grab y() and use this, which saves a *lot* of time.

int GPR_Generic::setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp)
{
int allowGridSample = 1; //FIXME: need option to do pure random and JIT

        NiceAssert( xmin.size() == xmax.size() );
        NiceAssert( Nsamp > 0 );
        NiceAssert( nv || ( nv == sampleMode ) );

        // IMPORTANT: is this is already a sample then we actually DO NOT want to re-sample as this might
        // subtly change it, and our BO method relies on the assumption that it does not get changed

        if ( nv && !sampleMode )
        {
            int dim = xmin.size();
            int i,j;
            int oldN = N();
            int totsamp = allowGridSample ? ( dim ? (int) pow(Nsamp,dim) : 0 ) : Nsamp;

            gentype xxmin(xmin);
            gentype xxmax(xmax);

//OLD            // Take uniform random samples on x
//OLD
//OLD            Vector<SparseVector<gentype> > x(Nsamp);
//OLD
//OLD            if ( dim )
//OLD            {
//OLD                for ( i = 0 ; i < Nsamp ; i++ )
//OLD                {
//OLD                    gentype xx = urand(xxmin,xxmax);
//OLD
//OLD                    x("&",i) = (const Vector<gentype> &) xx;
//OLD                }
//OLD            }

            // Take uniform samples on x

            Vector<SparseVector<gentype> > x(totsamp);

            if ( allowGridSample && ( dim == 1 ) )
            {
                int jj;

                for ( jj = 0 ; jj < totsamp ; jj++ )
                {
                    SparseVector<gentype> &xq = x("&",jj);

                    xq("&",zeroint())  = (((double) jj)+0.5)/(((double) Nsamp));
                    xq("&",zeroint()) *= (xmax(zeroint())-xmin(zeroint()));
                    xq("&",zeroint()) += xmin(zeroint());
                }
            }

            else if ( allowGridSample )
            {
                int jj,kk;
                Vector<int> jjj(dim);

                jjj = zeroint();

                for ( jj = 0 ; jj < totsamp ; jj++ )
                {
                    SparseVector<gentype> &xq = x("&",jj);

                    for ( kk = 0 ; kk < dim ; kk++ )
                    {
                        xq("&",kk)  = (((double) jjj(kk))+0.5)/(((double) Nsamp));
                        xq("&",kk) *= (xmax(kk)-xmin(kk));
                        xq("&",kk) += xmin(kk);
                    }

                    for ( kk = 0 ; kk < dim ; kk++ )
                    {
                        jjj("&",kk)++;

                        if ( jjj(kk) >= Nsamp )
                        {
                            jjj("&",kk) = 0;
                        }

                        else
                        {
                            break;
                        }
                    }
                }
            }

            else
            {
                for ( i = 0 ; i < totsamp ; i++ )
                {
                    gentype xx = urand(xxmin,xxmax);

                    x("&",i) = (const Vector<gentype> &) xx;
                }
            }

            // Add vectors to model, with d = 0 for now (do this preemptively to avoid double-calculation)

            gentype dummyy(muBias()); // We use bias() to ensure type of y is correct

            for ( i = 0 ; i < totsamp ; i++ )
            {
                qaddTrainingVector(oldN+i,dummyy,x("&",i),sigma()/SIGMA_ADD); // This destroys x(i), but that doesn't matter as we don't need it after this
                setd(oldN+i,0);
            }

            // Calculate means and variances

            Vector<gentype> m(totsamp);
            Matrix<gentype> v(totsamp,totsamp);
            gentype dummymu;

            for ( i = 0 ; i < totsamp ; i++ )
            {
                for ( j = 0 ; j < totsamp ; j++ )
                {
                    if ( i == j )
                    {
                        covTrainingVector(v("&",i,j),m("&",i),oldN+i,oldN+j);
                    }

                    else
                    {
                        covTrainingVector(v("&",i,j),dummymu,oldN+i,oldN+j);
                    }
                }

                // No longer need this fudge factor now that the grand function can deal with semi-definite covariances
                v("&",i,i) += SIGMA_ADD;
            }
            
            // Sample GPR

            gentype mm(m);
            gentype vv(v);

            gentype yy = grand(mm,vv);

            const Vector<gentype> &y = (const Vector<gentype> &) yy;

            // Set y to sampld value, and d = 2 to enable it

            for ( i = 0 ; i < totsamp ; i++ )
            {
                sety(oldN+i,y(i));
                setd(oldN+i,2);
            }

            // Train

            int dummy = 0;
            svmvolatile int voldum = 0;

            train(dummy,voldum);

//            // Calculate mean and variance
//
//            Vector<gentype> m(totsamp);
//            Matrix<gentype> v(totsamp,totsamp);
//            gentype dummymu;
//
//            for ( i = 0 ; i < totsamp ; i++ )
//            {
//                for ( j = 0 ; j < totsamp ; j++ )
//                {
//                    if ( i == j )
//                    {
//                        cov(v("&",i,i),m("&",i),x(i),x(i));
//                    }
//
//                    else
//                    {
//                        cov(v("&",i,j),dummymu,x(i),x(j));
//                    }
//                }
//
//                v("&",i,i) += SIGMA_ADD;
//            }
//
//            // Sample GPR
//
//            gentype mm(m);
//            gentype vv(v);
//
//            gentype yy = grand(mm,vv);
//
//            const Vector<gentype> &y = (const Vector<gentype> &) yy;
//
//            // Add to GPR training set
//
//            for ( i = 0 ; i < totsamp ; i++ )
//            {
//                addTrainingVector(oldN+i,y(i),x(i),sigma()/SIGMA_ADD);
//            }
//
//            // Train
//
//            int dummy = 0;
//            svmvolatile int voldum = 0;
//
//            train(dummy,voldum);
        }

        sampleMode = nv;

        return sampleMode | ML_Base::setSampleMode(nv,xmin,xmax,Nsamp);
}
