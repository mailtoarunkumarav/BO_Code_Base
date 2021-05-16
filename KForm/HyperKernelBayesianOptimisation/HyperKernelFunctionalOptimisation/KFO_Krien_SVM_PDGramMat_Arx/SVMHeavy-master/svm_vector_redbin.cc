
//
// Vector regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_vector_redbin.h"
#include "smatrix.h"
#include <iostream>
#include <sstream>
#include <string>


//#define SIGMACUT 1e-6




template <>
int SVM_Vector_redbin<SVM_Scalar>::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh);

template <>
int SVM_Vector_redbin<SVM_Scalar>::qaddTrainingVector(int i, const Vector<double> &z, SparseVector<gentype> &x, double Cweigh, double epsweigh, int d);

template <>
int SVM_Vector_redbin<SVM_Scalar>::qaddTrainingVector(int i, const Vector<gentype> &z, SparseVector<gentype> &x, double Cweigh, double epsweigh, int d);

template <>
int SVM_Vector_redbin<SVM_Scalar>::setd(int i, int d);

template <>
int SVM_Vector_redbin<SVM_Scalar>::setdinternal(int i, int d);

template <>
int SVM_Vector_redbin<SVM_Scalar>::dtrans(int i, int q) const;

template <>
double &SVM_Vector_redbin<SVM_Scalar>::K2(double &res, int i, int j, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const;

template <>
gentype &SVM_Vector_redbin<SVM_Scalar>::K2(gentype &res, int i, int j, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const;

template <>
double &SVM_Vector_redbin<SVM_Scalar>::K2ip(double &res, int i, int j, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const;

template <>
int SVM_Vector_redbin<SVM_Scalar>::setalldifrank(void);

template <>
int SVM_Vector_redbin<SVM_Scalar>::sety(int i, const gentype &z);

template <>
int SVM_Vector_redbin<SVM_Scalar>::sety(int i, const Vector<double> &z);

template <>
int SVM_Vector_redbin<SVM_Scalar>::settspaceDim(int newMinDim);

template <>
int SVM_Vector_redbin<SVM_Scalar>::addtspaceFeat(int iii);

template <>
int SVM_Vector_redbin<SVM_Scalar>::removetspaceFeat(int iii);

template <>
void SVM_Vector_redbin<SVM_Scalar>::locsetGp(int refactsol);

































































// SVM_Scalar specialisations

void evalKSVM_Vector_redbin_SVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    SVM_Vector_redbin<SVM_Scalar> *realOwner = (SVM_Vector_redbin<SVM_Scalar> *) owner;

    NiceAssert( realOwner );

    if ( i != j )
    {
        realOwner->K2(res,i,j,pxyprod);
    }

    else
    {
        res = (realOwner->kerndiagval)(i);
    }

    res += ( ( i == j ) ? (realOwner->diagoff)(i) : 0 );

    return;
}

void evalXYSVM_Vector_redbin_SVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    SVM_Vector_redbin<SVM_Scalar> *realOwner = (SVM_Vector_redbin<SVM_Scalar> *) owner;

    NiceAssert( realOwner );

    realOwner->K2ip(res,i,j,pxyprod);

    return;
}

void evalSigmaSVM_Vector_redbin_SVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    SVM_Vector_redbin<SVM_Scalar> *realOwner = (SVM_Vector_redbin<SVM_Scalar> *) owner;
    NiceAssert( realOwner );
    Matrix<double> *Gpval = realOwner->Gpval;
    NiceAssert( Gpval );

    //if ( i != j )
    //{
        res = (*Gpval)(i,i) + (*Gpval)(j,j) - (2*(*Gpval)(i,j));
    //}
    //
    //else
    //{
    //    res = 0;
    //}
    //
    // NOTE: sigma is positive if the kernel is Mercer and numerical stuff
    //       works out OK.  However note the checks later just in case.
    //
    // FIXME: in the singular case we use the hack of making sigma very small
    //        but positive.  Really what we *should* do is to follow the method
    //        in svoptim.cc in the svmheavy code - ie. follow a direction of
    //        linear descent.  But the maths is tricky and I'm lazy, so here
    //        goes.
    //
    //if ( res < SIGMACUT )
    //{
    //    res = SIGMACUT;
    //}

    return;
}

template <>
SVM_Vector_redbin<SVM_Scalar>::SVM_Vector_redbin() : SVM_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    isFudged   = 0;
    isStateOpt = 1;

    costType  = 0;
    optType   = 0;

    autosetLevel = 0;
    autosetCvalx = 0.0;
    CNval        = DEFAULT_C;

    xycache.reset(0,&evalXYSVM_Vector_redbin_SVM_Scalar,(void *) this);
    xycache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    kerncache.reset(0,&evalKSVM_Vector_redbin_SVM_Scalar,(void *) this);
    kerncache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    sigmacache.reset(0,&evalSigmaSVM_Vector_redbin_SVM_Scalar,(void *) this);
    sigmacache.setmemsize(DEFAULT_MEMSIZE,MINROWDIM);

    Gplocal = 1;
    MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &xycache   ,0,0));
    MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,0,0));
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,0,0));

    Q.useTightAllocation();
    Q.resize(1);

    locsetGp();

    Ns = 0;
    Nnc.resize(2);
    Nnc = zeroint();

    classLabelsval.resize(3);
    classRepval.resize(3);

    classLabelsval("&",0) = -1;
    classLabelsval("&",1) = +1;
    classLabelsval("&",2) = 2;

    classRepval("&",0).resize(1); classRepval("&",0)("&",0) = -1;
    classRepval("&",1).resize(1); classRepval("&",1)("&",0) = +1;
    classRepval("&",2).resize(1); classRepval("&",2)("&",0) = 2;

    return;
}

template <>
std::istream &SVM_Vector_redbin<SVM_Scalar>::inputstream(std::istream &input)
{
    int i,Qsize;
    wait_dummy dummy;

    input >> dummy; input >> costType;
    input >> dummy; input >> optType;

    input >> dummy; input >> CNval;

    input >> dummy; input >> autosetLevel;
    input >> dummy; input >> autosetCvalx;

    input >> dummy; input >> xycache;
    input >> dummy; input >> kerncache;
    input >> dummy; input >> sigmacache;
    input >> dummy; input >> kerndiagval;
    input >> dummy; input >> diagoff;

    input >> dummy; input >> dalpha;
    input >> dummy; input >> dalphaState;
    input >> dummy; input >> db;
    input >> dummy; input >> Ns;
    input >> dummy; input >> Nnc;
    input >> dummy; input >> isStateOpt;

    input >> dummy; input >> isFudged;

    SVM_Generic::inputstream(input);

    input >> dummy; input >> trainclass;
    input >> dummy; input >> traintarg;

    input >> dummy; input >> Qsize;

    (Q).resize(Qsize);

    for ( i = 0 ; i < Qsize ; i++ )
    {
        input >> dummy; input >> (Q)("&",i);
    }

    if ( Gplocal )
    {
        MEMDEL(xyval);
        xyval = NULL;

        MEMDEL(Gpval);
        Gpval = NULL;

        MEMDEL(Gpsigma);
        Gpsigma = NULL;
    }

    Gplocal = 1;

    MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &(xycache)   ,N(),N()));
    MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &(kerncache) ,N(),N()));
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &(sigmacache),N(),N()));

    int oldmemsize = (kerncache).get_memsize();
    int oldrowdim  = (kerncache).get_min_rowdim();

    (xycache).reset(N(),&evalXYSVM_Vector_redbin_SVM_Scalar,this);
    (xycache).setmemsize(oldmemsize,oldrowdim);

    (kerncache).reset(N(),&evalKSVM_Vector_redbin_SVM_Scalar,this);
    (kerncache).setmemsize(oldmemsize,oldrowdim);

    (sigmacache).reset(N(),&evalSigmaSVM_Vector_redbin_SVM_Scalar,this);
    (sigmacache).setmemsize(oldmemsize,oldrowdim);

    locsetGp(0);

    return input;
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::qaddTrainingVector(int i, const Vector<gentype> &z, SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    Vector<double> zz(z.size());

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            zz("&",j) = (double) z(j);
        }
    }

    return SVM_Vector_redbin<SVM_Scalar>::qaddTrainingVector(i,zz,x,Cweigh,epsweigh,d);
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    Vector<double> zd;
    Vector<gentype> zz((const Vector<gentype> &) z);

    zd.resize(zz.size());

    if ( zz.size() )
    {
        int i;

        for ( i = 0 ; i < zz.size() ; i++ )
        {
            zd("&",i) = (double) zz(i);
        }
    }

    return SVM_Vector_redbin<SVM_Scalar>::qaddTrainingVector(i,zd,x,Cweigh,epsweigh,2);
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::qaddTrainingVector(int i, const Vector<double> &z, SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( ( d == 0 ) || ( d == 2 ) );

    isStateOpt = 0;

    int res = 0;

    if ( z.size() > tspaceDim() )
    {
        res |= settspaceDim(z.size());
    }

    Nnc("&",d/2)++;

    if ( kerncache.get_min_rowdim() <= N() )
    {
	xycache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	kerncache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	sigmacache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
    }

    gentype yn;
    yn = z;
    res |= SVM_Generic::qaddTrainingVector(i,yn,x);



    int q;

    retVector<double> tmpva;

    trainclass.add(i);
    trainclass("&",i) = d;
    traintarg.add(i);
    traintarg("&",i).resize(tspaceDim());
    traintarg("&",i)("&",0,1,(z.size())-1,tmpva) = z;
    traintarg("&",i)("&",(z.size()),1,tspaceDim()-1,tmpva) = 0.0;

    diagoff.add(i);
    diagoff("&",i) = QUADCOSTDIAGOFFSETB(trainclass(i),Cweigh,1.0);
    kerndiagval.add(i);
    K2(kerndiagval("&",i),i,i);
    dalpha.add(i);
    dalpha("&",i).resize(tspaceDim());
    dalpha("&",i) = 0.0;
    dalphaState.add(i);
    dalphaState("&",i) = 0;

    SparseVector<gentype> tempvec;

    if ( Gplocal )
    {
        xyval->addRowCol(i);
        Gpval->addRowCol(i);
        Gpsigma->addRowCol(i);
    }

    xycache.add(i);
    kerncache.add(i);
    sigmacache.add(i);

    for ( q = 0 ; q < Q.size() ; q++ )
    {
	if ( q < tspaceDim() )
	{
            res |= Q("&",q).addTrainingVector(i,traintarg(i)(q),tempvec,Cweigh,epsweigh,xisclass(i,trainclass(i),q));
	}

	else
	{
            res |= Q("&",q).addTrainingVector(i,0.0            ,tempvec,Cweigh,epsweigh,xisclass(i,trainclass(i),q));
	}
    }

    res |= fixautosettings(0,1);

    SVM_Generic::basesetalpha(i,alphaV()(i));

    return res;
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::setd(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( ( d == 0 ) || ( d == 2 ) );

    int res = 0;

    if ( d != trainclass(i) )
    {
        int oldd = trainclass(i);

        res |= setdinternal(i,d);

	if ( !d || !oldd )
	{
            res |= fixautosettings(0,1);
	}
    }

    return res;
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::setdinternal(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( ( d == 0 ) || ( d == 2 ) );

    int q;
    int res = 0;

    if ( d != trainclass(i) )
    {
        res = 1;
        isStateOpt = 0;

	Nnc("&",trainclass(i)/2)--;
	Nnc("&",d/2)++;

        trainclass("&",i) = d;

        for ( q = 0 ; q < Q.size() ; q++ )
        {
            res |= Q("&",q).setd(i,xisclass(i,trainclass(i),q));

            if ( q < tspaceDim() )
            {
                dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
	    }
	}

        SVM_Generic::basesetalpha(i,alphaV()(i));
    }

    return res;
}



template <>
int SVM_Vector_redbin<SVM_Scalar>::dtrans(int i, int q) const
{
    (void) q;

    return trainclass(i);
}

template <>
double &SVM_Vector_redbin<SVM_Scalar>::K2(double &res, int i, int j, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    return SVM_Generic::K2(res,i,j,pxyprod,xx,yy,xxinfo,yyinfo);
}

template <>
gentype &SVM_Vector_redbin<SVM_Scalar>::K2(gentype &res, int i, int j, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    return SVM_Generic::K2(res,i,j,pxyprod,xx,yy,xxinfo,yyinfo);
}

template <>
double &SVM_Vector_redbin<SVM_Scalar>::K2ip(double &res, int i, int j, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    return SVM_Generic::K2ip(res,i,j,pxyprod,xx,yy,xxinfo,yyinfo);
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::setalldifrank(void)
{
    return 0;
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::sety(int i, const gentype &z)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( z.isCastableToVectorWithoutLoss() );

    Vector<gentype> zz((const Vector<gentype> &) z);
    Vector<double> zzz(zz.size());

    if ( zz.size() )
    {
        int k;

        for ( k = 0 ; k < zz.size() ; k++ )
        {
            zzz("&",k) = (double) zz(k);
        }
    }

    return sety(i,zzz);
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::sety(int i, const Vector<double> &z)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    if ( z.size() > tspaceDim() )
    {
        res |= settspaceDim(z.size());
    }

    retVector<double> tmpva;

    traintarg("&",i)("&",0,1,z.size()-1,tmpva) = z;
    traintarg("&",i)("&",z.size(),1,tspaceDim()-1,tmpva) = 0.0;

    gentype yn;
    yn = z;
    res |= SVM_Generic::sety(i,yn);

    int q;

    isStateOpt = 0;

    if ( tspaceDim() )
    {
	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
            res |= Q("&",q).sety(i,traintarg(i)(q));
	}
    }

    return res;
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::settspaceDim(int newMinDim)
{
    NiceAssert( newMinDim >= 0 );

    if ( newMinDim != tspaceDim() )
    {
	isStateOpt = 0;

	int tspacedimold = tspaceDim();
	int tspacedimnew = newMinDim;

        int i,q;

	// resize db and dalpha

	db.resize(tspacedimnew);

        retVector<double> tmpva;

	if ( tspacedimnew > tspacedimold )
	{
            db("&",tspacedimold,1,tspacedimnew-1,tmpva) = Q(zeroint()).biasR();
	}

	if ( N() )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
		dalpha("&",i).resize(tspacedimnew);

		if ( tspacedimnew > tspacedimold )
		{
                    dalpha("&",i)("&",tspacedimold,1,tspacedimnew-1,tmpva) = Q(zeroint()).alphaR()(i);
		}
	    }
	}

	// resize traintarg

	if ( N() )
	{
            retVector<double> tmpva;

	    for ( i = 0 ; i < N() ; i++ )
	    {
		traintarg("&",i).resize(tspacedimnew);

		if ( tspacedimnew > tspacedimold )
		{
                    traintarg("&",i)("&",tspacedimold,1,tspacedimnew-1,tmpva) = 0.0;

                    gentype yn;
                    yn = traintarg(i);
                    SVM_Generic::sety(i,yn);
		}
	    }
	}

	// resize Q

	while ( Q.size() > ( tspacedimnew ? tspacedimnew : 1 ) )
	{
	    Q.remove(Q.size()-1);
	}

	while ( Q.size() < tspacedimnew )
	{
	    Q.add(Q.size());
            Q("&",Q.size()-1) = Q(zeroint());

            if ( isFudged )
            {
                Q("&",Q.size()-1).fudgeOn();
            }
	}

	// fix z in new Q elements

	if ( tspacedimnew > tspacedimold )
	{
	    if ( N() )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
		    for ( q = tspacedimold ; q < tspacedimnew ; q++ )
		    {
                        Q("&",q).sety(i,traintarg(i)(q));
		    }
		}
	    }
	}

	// Fix kernel cache

	if ( kerncache.get_min_rowdim() <= N() )
	{
	    xycache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	    kerncache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	    sigmacache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	}

	if ( ( kerncache.get_min_rowdim() >= (int) (N()*ROWDIMSTEPRATIO) ) && ( N() > MINROWDIM ) )
	{
	    xycache.setmemsize(memsize(),N()-1);
	    kerncache.setmemsize(memsize(),N()-1);
	    sigmacache.setmemsize(memsize(),N()-1);
	}

        // Fix Gp pointers

        locsetGp();

        // Need to set d as well to update individual d components if rank

        setalldifrank();
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::addtspaceFeat(int iii)
{
    NiceAssert( iii >= 0 );
    NiceAssert( iii <= tspaceDim() );

    {
	isStateOpt = 0;

	int tspacedimold = tspaceDim();
        int i;

	// resize db and dalpha

        db.add(iii);
        db("&",iii) = db(zeroint()); // this is the value that will be set by the code used below

	if ( N() )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                dalpha("&",i).add(iii);
                traintarg("&",i).add(iii);
                dalpha("&",i)("&",iii) = dalpha(i)(0); // this is the value that will be set by the code used below
                traintarg("&",i)("&",iii) = 0.0;

                gentype yn;
                yn = traintarg(i);
                SVM_Generic::sety(i,yn);
	    }
	}

	// resize Q

        if ( tspacedimold )
        {
            Q.add(iii);
            Q("&",iii) = Q(( iii || !tspacedimold ) ? 0 : 1);
        }

        else
        {
            Q.add(iii);
        }

        if ( isFudged )
        {
            Q("&",iii).fudgeOn();
        }

	// fix z in new Q elements

        if ( N() )
        {
            for ( i = 0 ; i < N() ; i++ )
            {
                Q("&",iii).sety(i,traintarg(i)(iii));
	    }
	}

	// Fix kernel cache

	if ( kerncache.get_min_rowdim() <= N() )
	{
	    xycache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	    kerncache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	    sigmacache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	}

	if ( ( kerncache.get_min_rowdim() >= (int) (N()*ROWDIMSTEPRATIO) ) && ( N() > MINROWDIM ) )
	{
	    xycache.setmemsize(memsize(),N()-1);
	    kerncache.setmemsize(memsize(),N()-1);
	    sigmacache.setmemsize(memsize(),N()-1);
	}

        // Fix Gp pointers

        locsetGp();

        // Need to set d as well to update individual d components

        setalldifrank();
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

template <>
int SVM_Vector_redbin<SVM_Scalar>::removetspaceFeat(int iii)
{
    NiceAssert( iii >= 0 );
    NiceAssert( iii < tspaceDim() );

    {
	isStateOpt = 0;

        int i;

	// resize db and dalpha

        db.remove(iii);

	if ( N() )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                dalpha("&",i).remove(iii);
                traintarg("&",i).remove(iii);

                gentype yn;
                yn = traintarg(i);
                SVM_Generic::sety(i,yn);
	    }
	}

	// resize Q

        Q.remove(iii);

	// Fix kernel cache

	if ( kerncache.get_min_rowdim() <= N() )
	{
	    xycache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	    kerncache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	    sigmacache.setmemsize(memsize(),(int) (N()*ROWDIMSTEPRATIO));
	}

	if ( ( kerncache.get_min_rowdim() >= (int) (N()*ROWDIMSTEPRATIO) ) && ( N() > MINROWDIM ) )
	{
	    xycache.setmemsize(memsize(),N()-1);
	    kerncache.setmemsize(memsize(),N()-1);
	    sigmacache.setmemsize(memsize(),N()-1);
	}

        // Fix Gp pointers

        locsetGp();

        // Need to set d as well to update individual d components

        setalldifrank();
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

template <>
void SVM_Vector_redbin<SVM_Scalar>::locsetGp(int refactsol)
{
    int q;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        Q("&",q).setGp(Gpval,Gpsigma,xyval,refactsol);
    }

    return;
}

































































