
//
// k-nearest-neighbour base class
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
#include "knn_generic.h"




void evalKKNN_dist(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    KNN_Generic *realOwner = (KNN_Generic *) owner;

    NiceAssert( realOwner );

    res = realOwner->distK(i,j);

    return;
}

std::ostream &KNN_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "KNN k:      " << kay       << "\n";
    repPrint(output,'>',dep) << "KNN cache:  " << kerncache << "\n";
    repPrint(output,'>',dep) << "KNN d:      " << dd        << "\n";
    repPrint(output,'>',dep) << "KNN 1:      " << onevec    << "\n";
    repPrint(output,'>',dep) << "KNN wkt:    " << wkt       << "\n\n";

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &KNN_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> kay;
    input >> dummy; input >> kerncache;
    input >> dummy; input >> dd;
    input >> dummy; input >> onevec;
    input >> dummy; input >> wkt;

    kdistscr.resize(dd.size());
    iiscr.resize(dd.size());

    ML_Base::inputstream(input);

    MEMDEL(Gpdist);
    Gpdist = NULL;

    MEMNEW(Gpdist,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache, N(),N()));

    int oldmemsize = kerncache.get_memsize();
    int oldrowdim  = kerncache.get_min_rowdim();

    kerncache.reset(N(),&evalKKNN_dist,this);
    kerncache.setmemsize(oldmemsize,oldrowdim);

    return input;
}

KNN_Generic::KNN_Generic() : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    kerncache.reset(0,&evalKKNN_dist,(void *) this);
    kerncache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    MEMNEW(Gpdist,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,0,0));

    kay = KNN_DEFAULT_KAY;
    wkt = 0;
    getKernel_unsafe().setType(300);

    return;
}

KNN_Generic::KNN_Generic(const KNN_Generic &src) : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    kerncache.reset(0,&evalKKNN_dist,(void *) this);
    kerncache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    MEMNEW(Gpdist,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,0,0));

    assign(src,0);

    return;
}

KNN_Generic::KNN_Generic(const KNN_Generic &src, const ML_Base *xsrc) : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(xsrc);

    kerncache.reset(0,&evalKKNN_dist,(void *) this);
    kerncache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    MEMNEW(Gpdist,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,0,0));

    assign(src,0);

    return;
}

KNN_Generic::~KNN_Generic()
{
    MEMDEL(Gpdist);
    Gpdist = NULL;

    return;
}

int KNN_Generic::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    incgvernum();

    ML_Base::addTrainingVector (i,y,x,Cweigh,epsweigh);

    dd.add(i); dd("&",i) = 2;
    onevec.add(i); onevec("&",i) = 1;
    kdistscr.add(i); kdistscr("&",i) = 0.0;
    iiscr.add(i); iiscr("&",i) = 0;
    Gpdist->addRowCol(i);

    //NiceAssert( y.gettypeis() == targType() );

    if ( kerncache.get_min_rowdim() <= N() )
    {
        kerncache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
    }

    kerncache.add(i);

    return 1;
}

int KNN_Generic::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    incgvernum();

    ML_Base::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    dd.add(i); dd("&",i) = 2;
    onevec.add(i); onevec("&",i) = 1;
    kdistscr.add(i); kdistscr("&",i) = 0.0;
    iiscr.add(i); iiscr("&",i) = 0;
    Gpdist->addRowCol(i);

    //NiceAssert( y.gettypeis() == targType() );

    if ( kerncache.get_min_rowdim() <= N() )
    {
        kerncache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
    }

    kerncache.add(i);

    return 1;
}

int KNN_Generic::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int j;

    incgvernum();

    ML_Base::addTrainingVector (i,y,x,Cweigh,epsweigh);

    for ( j = 0 ; j < y.size() ; j++ )
    {
        //NiceAssert( y(j).gettypeis() == targType() );

        dd.add(i+j); dd("&",i+j) = 2;
        onevec.add(i+j); onevec("&",i+j) = 1;
        kdistscr.add(i+j); kdistscr("&",i+j) = 0.0;
        iiscr.add(i+j); iiscr("&",i+j) = 0;
        Gpdist->addRowCol(i+j);

        if ( kerncache.get_min_rowdim() <= N() )
        {
            kerncache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
        }

        kerncache.add(i+j);
    }

    return 1;
}

int KNN_Generic::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int j;

    incgvernum();

    ML_Base::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    for ( j = 0 ; j < y.size() ; j++ )
    {
        //NiceAssert( y(j).gettypeis() == targType() );

        dd.add(i+j); dd("&",i+j) = 2;
        onevec.add(i+j); onevec("&",i+j) = 1;
        kdistscr.add(i+j); kdistscr("&",i+j) = 0.0;
        iiscr.add(i+j); iiscr("&",i+j) = 0;
        Gpdist->addRowCol(i+j);

        if ( kerncache.get_min_rowdim() <= N() )
        {
            kerncache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
        }

        kerncache.add(i+j);
    }

    return 1;
}

int KNN_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    incgvernum();

    ML_Base::removeTrainingVector(i,y,x);

    dd.remove(i);
    onevec.remove(i);
    kdistscr.remove(i);
    iiscr.remove(i);
    Gpdist->removeRowCol(i);
    kerncache.remove(i);

    // Fix the cache

    if ( ( kerncache.get_min_rowdim() >= (int) (N()*ROWDIMSTEPRATIO) ) && ( N() > MINROWDIM ) )
    {
        kerncache.setmemsize(memsize(),N()-1);
    }

    return 1;
}

int KNN_Generic::setd(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    incgvernum();

    dd("&",i) = d;
    ML_Base::setd(i,d);

    return 1;
}

int KNN_Generic::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( d.size() == i.size() );

    incgvernum();

    retVector<int> tmpva;

    dd("&",i,tmpva) = d;
    ML_Base::setd(i,d);

    return 1;
}

int KNN_Generic::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == dd.size() );

    incgvernum();

    dd = d;
    ML_Base::setd(d);

    return 1;
}

int KNN_Generic::prealloc(int expectedN)
{
    dd.prealloc(expectedN);
    onevec.prealloc(expectedN);
    kerncache.prealloc(expectedN);
    ML_Base::prealloc(expectedN);
    kdistscr.prealloc(expectedN);
    iiscr.prealloc(expectedN);

    return 0;
}

int KNN_Generic::preallocsize(void) const
{
    return ML_Base::preallocsize();
}

int KNN_Generic::distcalcTrainingVector(int effkay, Vector<int> &ii, Vector<double> &kdist, int jj, int &Nnz) const
{
    ii.resize(N());
    kdist.resize(N());

    kdist = 0.0;

    // First calculate all distances

    int i;

    Nnz = 0;
    kdist = 0.0;

    if ( N() )
    {
        for ( i = 0 ; i < N() ; i++ )
        {
            if ( d()(i) )
            {
                // ||x-y||^2 = ||x||^2 + ||y||^2 - 2.<x,y>

                if ( jj >= 0 )
                {
                    kdist("&",i) = (*Gpdist)(i,jj);
                }

                else
                {
                    kdist("&",i) = distK(i,jj);
                }                     

                ii("&",Nnz) = i;
                Nnz++;
            }
        }
    }

    int j;

    ii.resize(Nnz); // This trims off constrained integers

    effkay = ( effkay < Nnz ) ? effkay : Nnz;

    if ( effkay && ( Nnz > 1 ) )
    {
        // Sort smallest to largest

        for ( i = 0 ; i < ( ( effkay < Nnz-1 ) ? effkay : Nnz-1 ) ; i++ )
        {
            for ( j = i+1 ; j < Nnz ; j++ )
            {
                if ( kdist(ii(j)) < kdist(ii(i)) )
                {
                    qswap(ii("&",i),ii("&",j));
                }
            }
        }
    }

    return effkay;
}

void KNN_Generic::ddistdxcalcTrainingVector(Vector<double> &igrad, Vector<double> &jgrad, const Vector<int> &ii, int j) const
{
    igrad.resize(ii.size());
    jgrad.resize(ii.size());

    int i;
    int dummy;

    if ( ii.size() )
    {
        for ( i = 0 ; i < ii.size() ; i++ )
        {
            ddistKdx(igrad("&",ii(i)),jgrad("&",ii(i)),dummy,ii(i),j);

            NiceAssert( dummy < 0 );
        }
    }

    return;
}

int KNN_Generic::numInternalClasses(void) const
{
    int res = numClasses();

    if ( type() == 302 ) { res = 2; }
    if ( type() == 303 ) { res = 1; }
    if ( type() == 304 ) { res = 1; }

    return res;
}

int KNN_Generic::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    int res = 0;

    (void) modind;

    NiceAssert( onlyChangeRowI >= -1 );
    NiceAssert( onlyChangeRowI < N() );

    incgvernum();

    ML_Base::resetKernel(modind,onlyChangeRowI,updateInfo);
    kerncache.setSymmetry(getKernel().getSymmetry());

    if ( N() && ( onlyChangeRowI == -1 ) )
    {
        kerncache.clear();
        res = 1;
    }

    else if ( onlyChangeRowI >= 0 )
    {
        kerncache.recalc(onlyChangeRowI);
        res = 1;
    }

    return res;
}

int KNN_Generic::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    int res = 0;

    (void) modind;

    incgvernum();

    NiceAssert( onlyChangeRowI >= -1 );
    NiceAssert( onlyChangeRowI < N() );

    ML_Base::setKernel(xkernel,modind,onlyChangeRowI);
    kerncache.setSymmetry(getKernel().getSymmetry());

    if ( N() && ( onlyChangeRowI == -1 ) )
    {
        kerncache.clear();
        res = 1;
    }

    else if ( onlyChangeRowI >= 0 )
    {
        kerncache.recalc(onlyChangeRowI);
        res = 1;
    }

    return res;
}

int KNN_Generic::realign(void)
{
    // Block realignment for KNN

    return 0;
}















































double KNN_Generic::calcweight(double dist) const
{
    // Not actual weight - unscaled weight.  Should be
    // K(dist)

    double res = 1;

    // 0: Rectangular:      K(d) = 1/2
    // 1: Triangular:       K(d) = (1-|d|)
    // 2: Epanechnikov:     K(d) = 3/4   (1-d^2)
    // 3: Quartic/biweight: K(d) = 15/16 (1-d^2)^2
    // 4: Triweight:        K(d) = 35/32 (1-d^2)^3
    // 5: Cosine:           K(d) = pi/4 cos(d.pi/2)
    // 6: Gauss:            K(d) = 1/sqrt(2.pi) exp(-d^2/2)
    // 7: Inversion:        K(d) = 1/|d|

    switch ( wkt )
    {
        case 0: { res = 1.0/2.0;                                                     break; }
        case 1: { res = 1-dist;                                                      break; }
        case 2: { res = ( 3.0/4.0 )*(1-(dist*dist));                                 break; }
        case 3: { res = (15.0/16.0)*(1-(dist*dist))*(1-(dist*dist));                 break; }
        case 4: { res = (35.0/32.0)*(1-(dist*dist))*(1-(dist*dist))*(1-(dist*dist)); break; }
        case 5: { res = NUMBASE_PION4*cos(NUMBASE_PION2*dist);                       break; }
        case 6: { res = (1.0/(NUMBASE_SQRT2*NUMBASE_SQRTPI))*exp(-dist*dist/2.0);          break; }
        case 7: { res = 1.0/dist;                                                    break; }

        default:
        {
            break;
        }
    }

    return res;
}

double KNN_Generic::calcweightgrad(double dist) const
{
    // Not actual weight - unscaled weight.  Should be
    // K(dist)

    double res = 0;

    // 0: Rectangular:      K(d) = 1/2
    // 1: Triangular:       K(d) = (1-|d|)
    // 2: Epanechnikov:     K(d) = 3/4   (1-d^2)
    // 3: Quartic/biweight: K(d) = 15/16 (1-d^2)^2
    // 4: Triweight:        K(d) = 35/32 (1-d^2)^3
    // 5: Cosine:           K(d) = pi/4 cos(d.pi/2)
    // 6: Gauss:            K(d) = 1/sqrt(2.pi) exp(-d^2/2)
    // 7: Inversion:        K(d) = 1/d
    //
    // Gradients:
    //
    // 0: Rectangular:      K'(d) = 0
    // 1: Triangular:       K'(d) = -1
    // 2: Epanechnikov:     K'(d) = -3/2 d
    // 3: Quartic/biweight: K'(d) = -15/4 (1-d^2) d
    // 4: Triweight:        K'(d) = -105/16 (1-d^2)^2 d
    // 5: Cosine:           K'(d) = -pi^2/8 sin(d.pi/2)
    // 6: Gauss:            K'(d) = -1/sqrt(2.pi) d exp(-d^2/2)
    // 7: Inversion:        K'(d) = -1/d^2

    switch ( wkt )
    {
        case 0: { res = 0.0;                                                      break; }
        case 1: { res = -1.0;                                                     break; }
        case 2: { res = -(3.0/2.0)*dist;                                          break; }
        case 3: { res = -(15.0/4.0)*(1-(dist*dist))*dist;                         break; }
        case 4: { res = -(105.0/16.0)*(1-(dist*dist))*(1-(dist*dist))*dist;       break; }
        case 5: { res = -NUMBASE_PION4*NUMBASE_PION2*sin(NUMBASE_PION2*dist);     break; }
        case 6: { res = -(1.0/(NUMBASE_SQRT2*NUMBASE_SQRTPI))*exp(-dist*dist/2.0)*dist; break; }
        case 7: { res = -1.0/(dist*dist);                                         break; }

        default:
        {
            break;
        }
    }

    return res;
}

int KNN_Generic::ghTrainingVector(gentype &resh, gentype &resg, int j, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    resg.makeNull();
    resh.makeNull();

    int Nnz = N();
    int effkay = ( kay <= N() ) ? kay : N();

    if ( N() )
    {
        effkay = distcalcTrainingVector(effkay,(**thisthisthis).iiscr,(**thisthisthis).kdistscr,j,Nnz);

        retVector<int>     tmpva;
        retVector<int>     tmpvb;
        retVector<gentype> tmpvc;
        retVector<double>  tmpvd;

        const Vector<gentype> &yk = y()(iiscr(zeroint(),1,effkay-1,tmpva),tmpvc);
        const Vector<double> &kdistk = kdistscr(iiscr(zeroint(),1,effkay-1,tmpvb),tmpvd);

        // knext is the next distance after k set, or final in kset plus a
        // small offset if kset is complete set.

        double knext = ( Nnz > effkay ) ? kdistscr(iiscr(effkay)) : kdistk(effkay-1)+1e-6;
        Vector<double> weights(effkay);

        if ( effkay )
        {
            int i;
            double Ksum = 0;

            for ( i = 0 ; i < effkay ; i++ )
            {
                weights("&",i) = calcweight(kdistk(i)/knext);
                Ksum += weights(i);
            }

            weights /= ( Ksum*Ksum >= KNN_DEFAULT_ZTOL_SQ ) ? Ksum : 1;
        }

        hfn(resg,yk,kdistk,weights,Nnz,effkay);
        resh = resg;
    }

    int res = 0;

    if ( resh.isValInteger() )
    {
        res = (int) resh;
    }

    return res;
}

int KNN_Generic::ggTrainingVectorInt(double &resg, int j, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    resg = 0.0;

    int Nnz = N();
    int effkay = ( kay <= N() ) ? kay : N();

    if ( N() )
    {
        effkay = distcalcTrainingVector(effkay,(**thisthisthis).iiscr,(**thisthisthis).kdistscr,j,Nnz);

        retVector<int>    tmpva;
        retVector<int>    tmpvb;
        retVector<double> tmpvc;
        retVector<double> tmpvd;

        const Vector<double> &yk = yR()(iiscr(zeroint(),1,effkay-1,tmpva),tmpvc);
        const Vector<double> &kdistk = kdistscr(iiscr(zeroint(),1,effkay-1,tmpvb),tmpvd);

        // knext is the next distance after k set, or final in kset plus a
        // small offset if kset is complete set.

        double knext = ( Nnz > effkay ) ? kdistscr(iiscr(effkay)) : kdistk(effkay-1)+1e-6;
        Vector<double> weights(effkay);

        if ( effkay )
        {
            int i;
            double Ksum = 0;

            for ( i = 0 ; i < effkay ; i++ )
            {
                weights("&",i) = calcweight(kdistk(i)/knext);
                Ksum += weights(i);
            }

            weights /= ( Ksum*Ksum >= KNN_DEFAULT_ZTOL_SQ ) ? Ksum : 1;
        }

        hfn(resg,yk,kdistk,weights,Nnz,effkay);
    }

    return ( resg > 0 ) ? +1 : ( ( resg < 0 ) ? -1 : 0 );
}

int KNN_Generic::ggTrainingVectorInt(Vector<double> &resg, int j, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    resg = 0.0;

    int Nnz = N();
    int effkay = ( kay <= N() ) ? kay : N();

    if ( N() )
    {
        effkay = distcalcTrainingVector(effkay,(**thisthisthis).iiscr,(**thisthisthis).kdistscr,j,Nnz);

        retVector<int>             tmpva;
        retVector<int>             tmpvb;
        retVector<Vector<double> > tmpvc;
        retVector<double>          tmpvd;

        const Vector<Vector<double> > &yk = yV()(iiscr(zeroint(),1,effkay-1,tmpva),tmpvc);
        const Vector<double> &kdistk = kdistscr(iiscr(zeroint(),1,effkay-1,tmpvb),tmpvd);

        // knext is the next distance after k set, or final in kset plus a
        // small offset if kset is complete set.

        double knext = ( Nnz > effkay ) ? kdistscr(iiscr(effkay)) : kdistk(effkay-1)+1e-6;
        Vector<double> weights(effkay);

        if ( effkay )
        {
            int i;
            double Ksum = 0;

            for ( i = 0 ; i < effkay ; i++ )
            {
                weights("&",i) = calcweight(kdistk(i)/knext);
                Ksum += weights(i);
            }

            weights /= ( Ksum*Ksum >= KNN_DEFAULT_ZTOL_SQ ) ? Ksum : 1;
        }

        hfn(resg,yk,kdistk,weights,Nnz,effkay);
    }

    return 0;
}

int KNN_Generic::ggTrainingVectorInt(d_anion &resg, int j, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    resg = 0.0;

    int Nnz = N();
    int effkay = ( kay <= N() ) ? kay : N();

    if ( N() )
    {
        effkay = distcalcTrainingVector(effkay,(**thisthisthis).iiscr,(**thisthisthis).kdistscr,j,Nnz);

        retVector<int>     tmpva;
        retVector<int>     tmpvb;
        retVector<d_anion> tmpvc;
        retVector<double>  tmpvd;

        const Vector<d_anion> &yk = yA()(iiscr(zeroint(),1,effkay-1,tmpva),tmpvc);
        const Vector<double> &kdistk = kdistscr(iiscr(zeroint(),1,effkay-1,tmpvb),tmpvd);

        // knext is the next distance after k set, or final in kset plus a
        // small offset if kset is complete set.

        double knext = ( Nnz > effkay ) ? kdistscr(iiscr(effkay)) : kdistk(effkay-1)+1e-6;
        Vector<double> weights(effkay);

        if ( effkay )
        {
            int i;
            double Ksum = 0;

            for ( i = 0 ; i < effkay ; i++ )
            {
                weights("&",i) = calcweight(kdistk(i)/knext);
                Ksum += weights(i);
            }

            weights /= ( Ksum*Ksum >= KNN_DEFAULT_ZTOL_SQ ) ? Ksum : 1;
        }

        hfn(resg,yk,kdistk,weights,Nnz,effkay);
    }

    return 0;
}

void KNN_Generic::dgTrainingVector(Vector<gentype> &res, gentype &resn, int p) const
{
    res.resize(N());

    gentype zerotemplate(gOutType());

    if ( gOutType() == 'V' )
    {
        zerotemplate.dir_vector().resize(tspaceDim());
    }

    else if ( gOutType() == 'A' )
    {
        zerotemplate.dir_anion().setorder(order());
    }

    setzero(zerotemplate);

    res  = zerotemplate;
    resn = zerotemplate;

    // NB: this function is blocked by all classes that do not implement
    // a simple weighted mean function.  So we can safely assume that the
    // function is simply a weighted mean.

    int Nnz = N();
    int effkay = ( kay <= N() ) ? kay : N();

    if ( N() )
    {
        effkay = distcalcTrainingVector(effkay,(**thisthisthis).iiscr,(**thisthisthis).kdistscr,p,Nnz);

        retVector<int>     tmpva;
        retVector<gentype> tmpvb;
        retVector<double>  tmpvc;
        retVector<int>     tmpvd;
        retVector<gentype> tmpve;
        retVector<int>     tmpvf;
        retVector<double>  tmpvg;

        const Vector<gentype> &yk = y()(iiscr(zeroint(),1,effkay-1,tmpvd),tmpve);
        const Vector<double> &kdistk = kdistscr(iiscr(zeroint(),1,effkay-1,tmpvf),tmpvg);

        // knext is the next distance after k set, or final in kset plus a
        // small offset if kset is complete set.

        double knext = ( Nnz > effkay ) ? kdistscr(iiscr(effkay)) : kdistk(effkay-1)+1e-6;
        Vector<double> weights(effkay);
        Vector<double> weightsgrad(effkay);

        if ( effkay )
        {
            int j,l;
            double Ksum = 0;

            for ( j = 0 ; j < effkay ; j++ )
            {
                weights    ("&",j) = calcweight    (kdistk(j)/knext);
                weightsgrad("&",j) = calcweightgrad(kdistk(j)/knext);
                Ksum += weights(j);
            }

            Ksum = ( Ksum*Ksum >= KNN_DEFAULT_ZTOL_SQ ) ? Ksum : 1;

            weights     /= Ksum;
            weightsgrad /= Ksum;

            Vector<double> lambdap;
            Vector<double> betap;

            ddistdxcalcTrainingVector(betap,lambdap,cntintvec(N(),tmpva),p);

            double lambdapq = ( Nnz > effkay ) ? lambdap(iiscr(effkay)) : lambdap(iiscr(effkay-1));
            double betapq   = ( Nnz > effkay ) ? betap  (iiscr(effkay)) : betap  (iiscr(effkay-1));

            Vector<gentype> &xres = res("&",iiscr(zeroint(),1,effkay-1,tmpva),tmpvb);
            gentype &xresq = res("&",iiscr(effkay));
            gentype &xresp = ( p == -1 ) ? resn : res("&",p);

            double scale,scaleb;

            // d/dxp

            for ( j = 0  ; j < effkay ; j++ )
            {
                scale = weightsgrad(j)*(lambdap(iiscr(j))-((kdistk(j)/knext)*lambdapq))/(knext*effkay);

                for ( l = 0 ; l < effkay ; l++ )
                {
                    scaleb = scale*(((j==l)?1:0)-weights(l));

                    if ( gOutType() == 'V' )
                    {
                        xresp.dir_vector().scaleAdd(scaleb,(const Vector<gentype> &) yk(l));
                    }

                    else if ( gOutType() == 'A' )
                    {
                        xresp.dir_anion() += scaleb*((const d_anion &) yk(l));
                    }

                    else
                    {
                        xresp.dir_double() += scaleb*((double) yk(l));
                    }
                }
            }

            // d/dxq

            for ( j = 0  ; j < effkay ; j++ )
            {
                scale = -weightsgrad(j)*((kdistk(j)/knext)*betapq)/(knext*effkay);

                for ( l = 0 ; l < effkay ; l++ )
                {
                    scaleb = scale*(((j==l)?1:0)-weights(l));

                    if ( gOutType() == 'V' )
                    {
                        xresq.dir_vector().scaleAdd(scaleb,(const Vector<gentype> &) yk(l));
                    }

                    else if ( gOutType() == 'A' )
                    {
                        xresq.dir_anion() += scaleb*((const d_anion &) yk(l));
                    }

                    else
                    {
                        xresq.dir_double() += scaleb*((double) yk(l));
                    }
                }
            }

            // d/dxj

            for ( j = 0  ; j < effkay ; j++ )
            {
                scale = weightsgrad(j)*betap(iiscr(j))/(knext*effkay);

                for ( l = 0 ; l < effkay ; l++ )
                {
                    scaleb = scale*(((j==l)?1:0)-weights(l));

                    if ( gOutType() == 'V' )
                    {
                        xres("&",j).dir_vector().scaleAdd(scaleb,(const Vector<gentype> &) yk(l));
                    }

                    else if ( gOutType() == 'A' )
                    {
                        xres("&",j).dir_anion() += scaleb*((const d_anion &) yk(l));
                    }

                    else
                    {
                        xres("&",j).dir_double() += scaleb*((double) yk(l));
                    }
                }
            }
        }
    }

    return;
}























int KNN_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
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



int KNN_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const
{
    int res = 0;

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    int isfallback = 0;

    switch ( ind )
    {
        case 4000: { val = k();   break; }
        case 4001: { val = ktp(); break; }

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


