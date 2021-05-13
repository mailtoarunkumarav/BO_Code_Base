
//
// Multiclass classification SVM (reduction to binary)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_multic_redbin.h"
#include <iostream>
#include <sstream>
#include <string>

#define D_CALC(_xclass_,_s_)                               ( (_xclass_) ? (classRepval(label_placeholder.findID(_xclass_))(_s_)) : 0 )
#define C_CALC                                             ( isLinearCost() ? CNval : (MAXBOUND) )
#define EPS_CALC                                           (                  epsval             )
#define CWEIGH_CALC(_xclass_,_Cweigh_,_Cweighfuzz_,_s_)    ( (_xclass_) ? (   (_Cweigh_)*(_Cweighfuzz_)*  mulCclass(label_placeholder.findID(_xclass_)) ) : 1.0 )
#define EPSWEIGH_CALC(_xclass_,_epsweigh_,_s_)             ( (_xclass_) ? ( (_epsweigh_)               *mulepsclass(label_placeholder.findID(_xclass_)) ) : 1.0 )
#define QUADCOSTDIAGOFFSET(_xclass_,_Cweigh_,_Cweighfuzz_) ( (_xclass_) ? ( isQuadraticCost() ? (1/(CNval*(_Cweigh_)*(_Cweighfuzz_)*mulCclass(label_placeholder.findID(_xclass_)))) : 0.0 ) : 0.0 )
#define CALC_TSPACEDIM_Q(_numclass_)                       ( ( is1vs1() || isDAGSVM() ) ? numclass1vs1((_numclass_),1) : ( isMOC() ? floorlog2((_numclass_),1) : ( is1vsA() ? ( ( (_numclass_) > 0 ) ? (_numclass_) : 1 ) : 1 ) ) )
#define CALC_TSPACEDIM(_numclass_)                         ( ( is1vs1() || isDAGSVM() ) ? numclass1vs1((_numclass_),0) : ( isMOC() ? floorlog2((_numclass_),0) :                  (_numclass_)                                  ) )
#define SIGMACUT 1e-6


#define MEMSHARE_KCACHE(_totmem_) isOptActive() ? ( ( (_totmem_) > 0 ) ? (_totmem_) : 1 ) : ( ( (_totmem_)/2 > 0 ) ? (_totmem_)/2 : 1 )
#define MEMSHARE_XYCACHE(_totmem_) isOptActive() ? ( ( (_totmem_) > 0 ) ? (_totmem_) : 1 ) : ( ( (_totmem_)/2 > 0 ) ? (_totmem_)/2 : 1 )
#define MEMSHARE_SIGMACACHE(_totmem_) isOptActive() ? 1 : ( ( (_totmem_)/2 > 0 ) ? (_totmem_)/2 : 1 )



// In the individual binary SVMs C and eps are set using C_CALC and EPS_CALC,
// Cclass and epsclass are left as 1, and Cweight and epsweight are set using
// CWEIGH_CALC and EPSWEIGH_CALS.  Finally the actual class (-1,0,+1) used by
// the individual SVMs for a given training vector is set using D_CALC.
//
// Note that in the above _xclass_ is the class number given by the user and
// _s_ is the SVM number.

int floorlog2(int i, int minres = 0);
int floorlog2(int i, int minres)
{
    int k = (int) ceil(log((double) i)/log(2.0));

    return ( ( k >= minres ) ? k : minres );
}

int numclass1vs1(int i, int minres = 0);
int numclass1vs1(int i, int minres)
{
    int k = i*(i-1)/2;

    return ( ( k >= minres ) ? k : minres );
}

void evalKSVM_MultiC_redbin(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    SVM_MultiC_redbin *realOwner = (SVM_MultiC_redbin *) owner;

    NiceAssert( realOwner );

    if ( i != j )
    {
        realOwner->K2(res,i,j,pxyprod);
    }

    else
    {
        res = ((realOwner->xxkerndiag)(i))+((realOwner->diagoff)(i));
    }

    return;
}

void evalXYSVM_MultiC_redbin(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    SVM_MultiC_redbin *realOwner = (SVM_MultiC_redbin *) owner;

    NiceAssert( realOwner );

    realOwner->K2ip(res,i,j,pxyprod);

    return;
}

void evalsigmaSVM_MultiC_redbin(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    SVM_MultiC_redbin *realOwner = (SVM_MultiC_redbin *) owner;
    NiceAssert( realOwner );
    Matrix<double> *Gpval = realOwner->Gpval;
    NiceAssert( Gpval );

    if ( i != j )
    {
        res = (*Gpval)(i,i) + (*Gpval)(j,j) - (2*(*Gpval)(i,j));
    }

    else
    {
	res = 0;
    }

    // NOTE: sigma is positive if the kernel is Mercer and numerical stuff
    //       works out OK.  However note the checks later just in case.
    //
    // FIXME: in the singular case we use the hack of making sigma very small
    //        but positive.  Really what we *should* do is to follow the method
    //        in svoptim.cc in the svmheavy code - ie. follow a direction of
    //        linear descent.  But the maths is tricky and I'm lazy, so here
    //        goes.

    if ( res < SIGMACUT )
    {
	res = SIGMACUT;
    }

    return;
}

SVM_MultiC_redbin::SVM_MultiC_redbin() : SVM_Generic()
{
    setaltx(NULL);

    isStateOpt = 1;

    costType  = 0;
    multitype = 0;
    optType   = 0;

    CNval  = DEFAULT_C;
    epsval = 1.0;

    autosetLevel  = 0;
    autosetnuvalx = 0.0;
    autosetCvalx  = 0.0;

    xycache.reset(0,&evalXYSVM_MultiC_redbin,(void *) this);
    xycache.setmemsize(MEMSHARE_XYCACHE(DEFAULT_MEMSIZE),MINROWDIM);

    kerncache.reset(0,&evalKSVM_MultiC_redbin,(void *) this);
    kerncache.setmemsize(MEMSHARE_KCACHE(DEFAULT_MEMSIZE),MINROWDIM);

    sigmacache.reset(0,&evalsigmaSVM_MultiC_redbin,(void *) this);
    sigmacache.setmemsize(MEMSHARE_SIGMACACHE(DEFAULT_MEMSIZE),MINROWDIM);

    MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &xycache   ,0,0));
    MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,0,0));
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,0,0));

    anomalyd = -3;

    Q.resize(1);

    Q("&",0).setmemsize(1);
    QA.setmemsize(1);

    locsetGp();

    Ns = 0;
    Nnc.resize(1);
    Nnc = zeroint();

    return;
}

SVM_MultiC_redbin::SVM_MultiC_redbin(const SVM_MultiC_redbin &src) : SVM_Generic()
{
    setaltx(NULL);

    xyval   = NULL;
    Gpval   = NULL;
    Gpsigma = NULL;

    assign(src,0);

    return;
}

SVM_MultiC_redbin::SVM_MultiC_redbin(const SVM_MultiC_redbin &src, const ML_Base *xsrc) : SVM_Generic()
{
    setaltx(xsrc);

    xyval   = NULL;
    Gpval   = NULL;
    Gpsigma = NULL;

    assign(src,1);

    return;
}

SVM_MultiC_redbin::~SVM_MultiC_redbin()
{
    if ( Gpval != NULL )
    {
	MEMDEL(xyval);
	MEMDEL(Gpval);
	MEMDEL(Gpsigma);

	xyval   = NULL;
	Gpval   = NULL;
        Gpsigma = NULL;
    }

    return;
}

double SVM_MultiC_redbin::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int SVM_MultiC_redbin::scale(double a)
{
    NiceAssert( a >= 0.0 );
    NiceAssert( a <= 1.0 );

    isStateOpt = 0;

    int i,q;
    int res = 0;

    res |= QA.scale(a);

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).scale(a);
    }

    if ( tspaceDim() && N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
	    dalpha("&",i) *= a;
	}

	if ( a == 0.0 )
	{
            dalphaState("&",i) = 0;
	}

	db *= a;
    }

    SVM_Generic::basescalealpha(a);
    SVM_Generic::basescalebias(a);

    return res;
}

int SVM_MultiC_redbin::reset(void)
{
    int res = QA.reset();

    if ( Q.size() )
    {
	int i;

	for ( i = 0 ; i < Q.size() ; i++ )
	{
            res |= Q("&",i).reset();
	}
    }

    dalpha.zero();
    dalphaState.zero();
    db.zero();

    Ns = 0;
    isStateOpt = 0;

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

int SVM_MultiC_redbin::setAlphaV(const Vector<Vector<double> > &newAlpha)
{
    NiceAssert( newAlpha.size() == N() );

    if ( N() && tspaceDim() )
    {
	isStateOpt = 0;

	int i,q;

	Vector<double> localpha(N());

	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                NiceAssert( newAlpha(i).size() == tspaceDim() );

		localpha("&",i) = newAlpha(i)(q);
	    }

            Q("&",q).setAlphaR(localpha);
	}

	dalphaState = zeroint();

	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                dalpha("&",i)("&",q) = (Q(q).alphaR())(i);

		if ( (Q(q).alphaState())(i) )
		{
		    dalphaState("&",i) = 1;
		}
	    }
	}

	Ns = sum(dalphaState);
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

int SVM_MultiC_redbin::setBiasV(const Vector<double> &newBias)
{
    NiceAssert( newBias.size() == tspaceDim() );

    if ( tspaceDim() )
    {
	isStateOpt = 0;

	int q;

	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
            Q("&",q).setBiasR(newBias(q));
	}

	db = newBias;
    }

    SVM_Generic::basesetbias(biasV());

    return 1;
}

int SVM_MultiC_redbin::setLinearCost(void)
{
    if ( isQuadraticCost() )
    {
	if ( N() )
	{
	    isStateOpt = 0;
	}

	costType = 1;

	recalcdiagoff(-1);
        setC(CNval);
    }

    return 1;
}

int SVM_MultiC_redbin::setQuadraticCost(void)
{
    if ( isLinearCost() )
    {
	if ( N() )
	{
	    isStateOpt = 0;
	}

	costType = 0;

	recalcdiagoff(-1);
        setC(CNval);
    }

    return 1;
}

int SVM_MultiC_redbin::setVarBias(int q)
{
    isStateOpt = 0;

    if ( q == -2 )
    {
	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setVarBias();
	}
    }

    else
    {
	Q("&",q).setVarBias();
    }

    return 1;
}

int SVM_MultiC_redbin::setPosBias(int q)
{
    if ( q == -2 )
    {
	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setPosBias();

	    if ( q < db.size() )
	    {
		if ( db(q) < 0 )
		{
		    db("&",q) = 0.0;
                    isStateOpt = 0;
		}
	    }
	}
    }

    else
    {
	Q("&",q).setPosBias();

	if ( db(q) < 0 )
	{
	    db("&",q) = 0.0;
	    isStateOpt = 0;
	}
    }

    SVM_Generic::basesetbias(biasV());

    return 1;
}

int SVM_MultiC_redbin::setNegBias(int q)
{
    if ( q == -2 )
    {
	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setNegBias();

	    if ( q < db.size() )
	    {
		if ( db(q) > 0 )
		{
		    db("&",q) = 0.0;
                    isStateOpt = 0;
		}
	    }
	}
    }

    else
    {
	Q("&",q).setNegBias();

	if ( db(q) > 0 )
	{
	    db("&",q) = 0.0;
	    isStateOpt = 0;
	}
    }

    SVM_Generic::basesetbias(biasV());

    return 1;
}

int SVM_MultiC_redbin::setFixedBias(int q, double newbias)
{
    isStateOpt = 0;

    if ( q == -2 )
    {
	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    if ( q < db.size() )
	    {
		db("&",q) = newbias;
	    }

	    Q("&",q).setFixedBias(newbias);
	}
    }

    else
    {
	db("&",q) = newbias;
	Q("&",q).setFixedBias(newbias);
    }

    SVM_Generic::basesetbias(biasV());

    return 1;
}

void SVM_MultiC_redbin::setRejectThreshold(double nv)
{
    int q;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        Q("&",q).setRejectThreshold(nv);
    }

    return;
}

int SVM_MultiC_redbin::setC(double xC)
{
    NiceAssert( xC > 0 );

    int res = 0;

    autosetOff();

    if ( N() )
    {
	isStateOpt = 0;
        res = 1;
    }

    int i,q;

    CNval = xC;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    // Need to set C as upper bound on alpha in all cases as this
    // function is also used to move between linear and quadratic
    // cost.

    for ( q = 0 ; q < Q.size() ; q++ )
    {
	Q("&",q).setC(C_CALC);
    }

    if ( N() && tspaceDim() )
    {
	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
	    }
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

int SVM_MultiC_redbin::seteps(double xeps)
{
    NiceAssert( xeps >= 0 );

    int q;
    int res = 0;

    if ( N() )
    {
	isStateOpt = 0;
    }

    epsval = xeps;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).seteps(EPS_CALC);
    }

    return res;
}

int SVM_MultiC_redbin::setCclass(int nc, double xC)
{
    NiceAssert( xC > 0 );
    NiceAssert( nc );
    NiceAssert( nc >= -1 );
    NiceAssert( nc != anomalyd );
    NiceAssert( label_placeholder.findID(nc) >= 0 );

    int res = 0;

    if ( NNC(nc) )
    {
	isStateOpt = 0;
        res = 1;
    }

    int i,q;

    mulCclass("&",label_placeholder.findID(nc)) = xC;

    if ( N() && tspaceDim() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
	    if ( nc == trainclass(i) )
	    {
		if ( isQuadraticCost() )
		{
		    recalcdiagoff(i);
		}

		else
		{
		    for ( q = 0 ; q < tspaceDim() ; q++ )
		    {
                        res |= Q("&",q).setCweight(i,CWEIGH_CALC(trainclass(i),Cweightval(i),Cweightvalfuzz(i),q));

                        dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
		    }
		}
	    }
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

int SVM_MultiC_redbin::setepsclass(int nc, double xeps)
{
    NiceAssert( xeps >= 0 );
    NiceAssert( nc );
    NiceAssert( nc >= -1 );
    NiceAssert( nc != anomalyd );
    NiceAssert( label_placeholder.findID(nc) >= 0 );

    int res = 0;

    if ( NNC(nc) )
    {
	isStateOpt = 0;
        res = 1;
    }

    int i,q;

    mulepsclass("&",label_placeholder.findID(nc)) = xeps;

    if ( N() && tspaceDim() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
	    if ( nc == trainclass(i) )
	    {
		for ( q = 0 ; q < tspaceDim() ; q++ )
		{
                    res |= Q("&",q).setepsweight(i,EPSWEIGH_CALC(trainclass(i),epsweightval(i),q));
		}
	    }
	}
    }

    return res;
}

int SVM_MultiC_redbin::scaleCweight(double scalefactor)
{
    NiceAssert( scalefactor > 0 );

    if ( N() )
    {
	isStateOpt = 0;
    }

    int i,q;

    Cweightval *= scalefactor;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else if ( tspaceDim() )
    {
	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
	    Q("&",q).scaleCweight(scalefactor);
	}

	if ( N() && tspaceDim() )
	{
	    for ( q = 0 ; q < tspaceDim() ; q++ )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
		}
	    }
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

int SVM_MultiC_redbin::scaleCweightfuzz(double scalefactor)
{
    NiceAssert( scalefactor > 0 );

    if ( N() )
    {
	isStateOpt = 0;
    }

    int i,q;

    Cweightvalfuzz *= scalefactor;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else if ( tspaceDim() )
    {
	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
	    Q("&",q).scaleCweight(scalefactor);
	}

	if ( N() && tspaceDim() )
	{
	    for ( q = 0 ; q < tspaceDim() ; q++ )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
		}
	    }
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

int SVM_MultiC_redbin::scaleepsweight(double scalefactor)
{
    int q;
    int res = 0;

    if ( N() )
    {
	isStateOpt = 0;
    }

    epsweightval *= scalefactor;

    if ( Q.size() )
    {
	for ( q = 0 ; q < Q.size() ; q++ )
	{
            res |= Q("&",q).scaleepsweight(scalefactor);
	}
    }

    return res;
}

int SVM_MultiC_redbin::setOptActive(void)
{
    int q;

    if ( !isOptActive() )
    {
	optType = 0;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setOptActive();
	}
    }

    QA.setOptActive();
    setmemsize(memsize());

    return 0;
}

int SVM_MultiC_redbin::setOptSMO(void)
{
    int q;

    if ( !isOptSMO() )
    {
        optType = 1;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setOptSMO();
	}
    }

    QA.setOptSMO();
    setmemsize(memsize());

    return 0;
}

int SVM_MultiC_redbin::setOptD2C(void)
{
    int q;

    if ( !isOptD2C() )
    {
	optType = 2;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setOptD2C();
	}
    }

    QA.setOptD2C();
    setmemsize(memsize());

    return 0;
}

int SVM_MultiC_redbin::setOptGrad(void)
{
    int q;

    if ( !isOptGrad() )
    {
	optType = 3;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
	    Q("&",q).setOptGrad();
	}
    }

    QA.setOptGrad();
    setmemsize(memsize());

    return 0;
}

void SVM_MultiC_redbin::setmemsize(int memsize)
{
    NiceAssert( memsize > 0 );

    int q;

    xycache.setmemsize(MEMSHARE_XYCACHE(memsize)      ,xycache.get_min_rowdim());
    kerncache.setmemsize(MEMSHARE_KCACHE(memsize)     ,kerncache.get_min_rowdim());
    sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize),sigmacache.get_min_rowdim());

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        Q("&",q).setmemsize(1);
    }

    QA.setmemsize(1);

    return;
}

int SVM_MultiC_redbin::setzerotol(double zt)
{
    NiceAssert( zt > 0 );

    isStateOpt = 0;

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setzerotol(zt);
    }

    res |= QA.setzerotol(zt);

    return res;
}

int SVM_MultiC_redbin::setOpttol(double xopttol)
{
    NiceAssert( xopttol > 0 );

    isStateOpt = 0;

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setOpttol(xopttol);
    }

    res |= QA.setOpttol(xopttol);

    return res;
}

int SVM_MultiC_redbin::setmaxitcnt(int maxitcnt)
{
    NiceAssert( maxitcnt >= 0 );

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setmaxitcnt(maxitcnt);
    }

    res |= QA.setmaxitcnt(maxitcnt);

    return res;
}

int SVM_MultiC_redbin::setmaxtraintime(double maxtraintime)
{
    NiceAssert( maxtraintime >= 0 );

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setmaxtraintime(maxtraintime);
    }

    res |= QA.setmaxtraintime(maxtraintime);

    return res;
}

int SVM_MultiC_redbin::setmaxiterfuzzt(int xmaxiterfuzzt)
{
    NiceAssert( xmaxiterfuzzt >= 0 );

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setmaxiterfuzzt(xmaxiterfuzzt);
    }

    return res;
}

int SVM_MultiC_redbin::setusefuzzt(int xusefuzzt)
{
    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setusefuzzt(xusefuzzt);
    }

    return res;
}

int SVM_MultiC_redbin::setlrfuzzt(double xlrfuzzt)
{
    NiceAssert( ( xlrfuzzt >= 0 ) && ( xlrfuzzt <= 1 ) );

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setlrfuzzt(xlrfuzzt);
    }

    return res;
}

int SVM_MultiC_redbin::setztfuzzt(double xztfuzzt)
{
    NiceAssert( xztfuzzt >= 0 );

    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setztfuzzt(xztfuzzt);
    }

    return res;
}

int SVM_MultiC_redbin::setcostfnfuzzt(const gentype &xcostfnfuzzt)
{
    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setcostfnfuzzt(xcostfnfuzzt);
    }

    return res;
}

int SVM_MultiC_redbin::setcostfnfuzzt(const std::string &xcostfnfuzzt)
{
    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setcostfnfuzzt(xcostfnfuzzt);
    }

    return res;
}

int SVM_MultiC_redbin::sety(int i, const gentype &y)
{
    NiceAssert( y.isCastableToIntegerWithoutLoss() );

    return setd(i, (int) y);
}

int SVM_MultiC_redbin::sety(const Vector<int> &j, const Vector<gentype> &yn)
{
    NiceAssert( j.size() == yn.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
        {
            res |= sety(j(i),yn(i));
        }
    }

    return res;
}

int SVM_MultiC_redbin::sety(const Vector<gentype> &yn)
{
    NiceAssert( N() == yn.size() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int SVM_MultiC_redbin::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    (void) onlyChangeRowI;

    int res = 0;

    isStateOpt = 0;

    int fixxycache = getKernel().isIPdiffered();

    res = QA.resetKernel(1,-1,0);
QA.setKernel(getKernel());
    res |= SVM_Generic::resetKernel(modind,onlyChangeRowI,updateInfo);

    if ( fixxycache )
    {
        xycache.setSymmetry(1);
    }

    kerncache.setSymmetry(getKernel().getSymmetry());
    sigmacache.setSymmetry(1);

    if ( N() )
    {
        res |= 1;

        int i;

	for ( i = 0 ; i < N() ; i++ )
	{
            K2(xxkerndiag("&",i),i,i);
	}

        if ( fixxycache )
        {
            xycache.clear();
        }

        kerncache.clear();
        sigmacache.clear();

        locsetGp();

        res |= fixautosettings(1,0);
    }

    getKernel_unsafe().setIPdiffered(0);

    return res;
}

int SVM_MultiC_redbin::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    (void) onlyChangeRowI;

    isStateOpt = 0;

    int res = QA.setKernel(xkernel);

    res |= SVM_Generic::setKernel(xkernel,modind,onlyChangeRowI);

    xycache.setSymmetry(1);
    kerncache.setSymmetry(getKernel().getSymmetry());
    sigmacache.setSymmetry(1);

    if ( N() )
    {
        res |= 1;

        int i;

	for ( i = 0 ; i < N() ; i++ )
	{
            K2(xxkerndiag("&",i),i,i);
	}

        xycache.clear();
        kerncache.clear();
        sigmacache.clear();

        locsetGp();

        res |= fixautosettings(1,0);
    }

    getKernel_unsafe().setIPdiffered(0);

    return res;
}

void SVM_MultiC_redbin::fillCache(void)
{
    QA.fillCache(); 

    if ( Q.size() )
    {
        int i;

        for ( i = 0 ; i < Q.size() ; i++ )
        {
            Q("&",i).fillCache();
        }
    }

    return;
}

int SVM_MultiC_redbin::setd(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( d >= -1 );
    NiceAssert( d != anomalyd ); 

    int res = 0;

    if ( d != trainclass(i) )
    {
	int oldd = trainclass(i);

        res = setdinternal(i,d);

	if ( !d || !oldd )
	{
            res |= fixautosettings(0,1);
	}
    }

    return res;
}

int SVM_MultiC_redbin::setCweight(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( xCweight > 0 );

    int q;

    isStateOpt = 0;

    Cweightval("&",i) = xCweight;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(i);
    }

    else if ( tspaceDim() )
    {
	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
            Q("&",q).setCweight(i,CWEIGH_CALC(trainclass(i),Cweightval(i),Cweightvalfuzz(i),q));

            dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
	}

        SVM_Generic::basesetalpha(i,alphaV()(i));
    }

    return 1;
}

int SVM_MultiC_redbin::setCweightfuzz(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( xCweight > 0 );

    int q;

    isStateOpt = 0;

    Cweightvalfuzz("&",i) = xCweight;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(i);
    }

    else if ( tspaceDim() )
    {
	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
            Q("&",q).setCweight(i,CWEIGH_CALC(trainclass(i),Cweightval(i),Cweightvalfuzz(i),q));

            dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
	}

        SVM_Generic::basesetalpha(i,alphaV()(i));
    }

    return 1;
}

int SVM_MultiC_redbin::setepsweight(int i, double xepsweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( xepsweight >= 0 );

    isStateOpt = 0;

    int q;
    int res = 0;

    epsweightval("&",i) = xepsweight;

    if ( tspaceDim() )
    {
	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
            res |= Q("&",q).setepsweight(i,EPSWEIGH_CALC(trainclass(i),epsweightval(i),q));
	}
    }

    return res;
}

int SVM_MultiC_redbin::setd(const Vector<int> &j, const Vector<int> &d)
{
    NiceAssert( d.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setdinternal(j(i),d(i));
	}

        res |= fixautosettings(0,1);
    }

    return res;
}

int SVM_MultiC_redbin::setCweight(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setCweight(j(i),xCweight(i));
	}
    }

    return res;
}

int SVM_MultiC_redbin::setCweightfuzz(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setCweightfuzz(j(i),xCweight(i));
	}
    }

    return res;
}

int SVM_MultiC_redbin::setepsweight(const Vector<int> &j, const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setepsweight(j(i),xepsweight(i));
	}
    }

    return res;
}

int SVM_MultiC_redbin::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= setdinternal(i,d(i));
	}

        res |= fixautosettings(0,1);
    }

    return res;
}

int SVM_MultiC_redbin::setCweight(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= setCweight(i,xCweight(i));
	}
    }

    return res;
}

int SVM_MultiC_redbin::setCweightfuzz(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= setCweightfuzz(i,xCweight(i));
	}
    }

    return res;
}

int SVM_MultiC_redbin::setepsweight(const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= setepsweight(i,xepsweight(i));
	}
    }

    return res;
}

int SVM_MultiC_redbin::setdinternal(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( d >= -1 );
    NiceAssert( d != anomalyd ); 

    int q;
    int res = 0;

    if ( d != trainclass(i) )
    {
        res = 1;
        isStateOpt = 0;

	Nnc("&",( trainclass(i) ? (label_placeholder.findID(trainclass(i))+1) : 0 ))--;

	if ( d != 0 )
	{
            res |= addclass(d);
	}

	Nnc("&",( d ? (label_placeholder.findID(d)+1) : 0 ))++;

        if ( d == 0 )
        {
            QA.setd(i,0);
        }

        if ( trainclass("&",i) == 0 )
        {
            QA.setd(i,1);
        }

	trainclass("&",i) = d;
        gentype yn(trainclass(i));
        SVM_Generic::sety(i,yn);

	if ( tspaceDim() )
	{
	    for ( q = 0 ; q < tspaceDim() ; q++ )
	    {
		if ( D_CALC(trainclass(i),q) != (Q(q).d())(i) )
		{
		    Q("&",q).setd(i,D_CALC(trainclass(i),q));

                    Q("&",q).setCweight  (i,  CWEIGH_CALC(trainclass(i),  Cweightval(i),Cweightvalfuzz(i),q));
                    Q("&",q).setepsweight(i,EPSWEIGH_CALC(trainclass(i),epsweightval(i)                  ,q));

                    dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
		}
	    }

            SVM_Generic::basesetalpha(i,alphaV()(i));
	}
    }

    return res;
}

int SVM_MultiC_redbin::setLinBiasForce(int q, double newval)
{
    NiceAssert( q != -3 );

    return Q("&",q).setLinBiasForce(newval);
}

int SVM_MultiC_redbin::setQuadBiasForce(int q, double newval)
{
    NiceAssert( q != -3 );

    return Q("&",q).setQuadBiasForce(newval);
}

int SVM_MultiC_redbin::setFixedTube(void)
{
    int res = 0;

    if ( !isFixedTube() )
    {
	int q;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
            res |= Q("&",q).setFixedTube();
	}
    }

    return res;
}

int SVM_MultiC_redbin::setShrinkTube(void)
{
    int res = 0;

    if ( !isShrinkTube() )
    {
	int q;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
            res |= Q("&",q).setShrinkTube();
	}
    }

    return res;
}

int SVM_MultiC_redbin::setnu(double xnuLin)
{
    int q;
    int res = 0;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).setnu(xnuLin);
    }

    return res;
}

int SVM_MultiC_redbin::setClassifyViaSVR(void)
{
    int res = 0;

    if ( !isClassifyViaSVR() && Q.size() )
    {
	isStateOpt = 0;

	int q;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
            res |= Q("&",q).setClassifyViaSVR();
	}
    }

    return res;
}

int SVM_MultiC_redbin::setClassifyViaSVM(void)
{
    int res = 0;

    if ( !isClassifyViaSVM() && Q.size() )
    {
	isStateOpt = 0;

	int q;

	for ( q = 0 ; q < Q.size() ; q++ )
	{
            res |= Q("&",q).setClassifyViaSVM();
	}
    }

    return res;
}

int SVM_MultiC_redbin::set1vsA(void)
{
    if ( !is1vsA() )
    {
	isStateOpt = 0;

	changemultitype(0);
    }

    return 1;
}

int SVM_MultiC_redbin::set1vs1(void)
{
    if ( !is1vs1() )
    {
	isStateOpt = 0;

	changemultitype(1);
    }

    return 1;
}

int SVM_MultiC_redbin::setDAGSVM(void)
{
    if ( !isDAGSVM() )
    {
	isStateOpt = 0;

	changemultitype(2);
    }

    return 1;
}

int SVM_MultiC_redbin::setMOC(void)
{
    if ( !isMOC() )
    {
	isStateOpt = 0;

	changemultitype(3);
    }

    return 1;
}

int SVM_MultiC_redbin::addTrainingVector(int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_MultiC_redbin::addTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int SVM_MultiC_redbin::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_MultiC_redbin::qaddTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int SVM_MultiC_redbin::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<int> zz(z.size());

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            zz("&",j) = (int) z(j);
        }
    }

    return SVM_MultiC_redbin::addTrainingVector(i,zz,x,Cweigh,epsweigh);
}

int SVM_MultiC_redbin::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<int> zz(z.size());

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            zz("&",j) = (int) z(j);
        }
    }

    return SVM_MultiC_redbin::qaddTrainingVector(i,zz,x,Cweigh,epsweigh);
}

int SVM_MultiC_redbin::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    isStateOpt = 0;

    Nnc("&",( trainclass(i) ? (label_placeholder.findID(trainclass(i))+1) : 0 ))--;

    int q;

    res |= setd(i,0);

    res |= SVM_Generic::removeTrainingVector(i,y,x);

    trainclass.remove(i);
    Cweightval.remove(i);
    Cweightvalfuzz.remove(i);
    epsweightval.remove(i);
    diagoff.remove(i);
    xxkerndiag.remove(i);
    dalpha.remove(i);
    dalphaState.remove(i);

    xyval->removeRowCol(i);
    Gpval->removeRowCol(i);
    Gpsigma->removeRowCol(i);

    xycache.remove(i);
    kerncache.remove(i);
    sigmacache.remove(i);

    for ( q = 0 ; q < Q.size() ; q++ )
    {
        res |= Q("&",q).ML_Base::removeTrainingVector(i);
    }

    res |= QA.ML_Base::removeTrainingVector(i);

    // Fix the cache

    if ( ( kerncache.get_min_rowdim() >= (int) (N()*ROWDIMSTEPRATIO) ) && ( N() > MINROWDIM ) )
    {
        xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),N()-1);
        kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),N()-1);
        sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),N()-1);
    }

    res |= fixautosettings(0,1);

    return res;
}

int SVM_MultiC_redbin::addTrainingVector(int i, int y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y != anomalyd );
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( y >= -1 );

    int res = 0;

    if ( kerncache.get_min_rowdim() <= N() )
    {
        xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
        kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
        sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
    }

    if ( y != 0 )
    {
        res |= addclass(y);
    }

    gentype yn(y);
    res |= SVM_Generic::addTrainingVector(i,yn,x);
    res |= qtaddTrainingVector(i,y,Cweigh,epsweigh);
    res |= fixautosettings(0,1);

    return res;
}

int SVM_MultiC_redbin::qaddTrainingVector(int i, int y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( y != anomalyd );
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( y >= -1 );

    int res = 0;

    if ( kerncache.get_min_rowdim() <= N() )
    {
        xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
        kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
        sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
    }

    if ( y != 0 )
    {
        res |= addclass(y);
    }

    gentype yn(y);
    res |= SVM_Generic::qaddTrainingVector(i,yn,x);
    res |= qtaddTrainingVector(i,y,Cweigh,epsweigh);
    res |= fixautosettings(0,1);

    return res;
}

int SVM_MultiC_redbin::addTrainingVector(int i, const Vector<int> &y, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( y.size() == xx.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            NiceAssert( y(j) != anomalyd );
            NiceAssert( y(j) >= -1 );

            if ( kerncache.get_min_rowdim() <= N() )
            {
                xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
                kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
                sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
            }

            if ( y(j) != 0 )
            {
                res |= addclass(y(j));
            }

            gentype tempy(y(j));

            res |= SVM_Generic::addTrainingVector(i+j,tempy,xx(j));
            res |= qtaddTrainingVector(i+j,y(j),Cweigh(j),epsweigh(j));
        }
    }

    res |= fixautosettings(0,1);

    return res;
}

int SVM_MultiC_redbin::qaddTrainingVector(int i, const Vector<int> &y, Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( y.size() == xx.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            NiceAssert( y(j) != anomalyd );
            NiceAssert( y(j) >= -1 );

            if ( kerncache.get_min_rowdim() <= N() )
            {
                xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
                kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
                sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
            }

            if ( y(j) != 0 )
            {
                res |= addclass(y(j));
            }

            gentype tempy(y(j));

            res |= SVM_Generic::qaddTrainingVector(i+j,tempy,xx("&",j));
            res |= qtaddTrainingVector(i+j,y(j),Cweigh(j),epsweigh(j));
        }
    }

    res |= fixautosettings(0,1);

    return res;
}

int SVM_MultiC_redbin::qtaddTrainingVector(int i, int y, double Cweigh, double epsweigh)
{
    int q;
    int res = 0;

    isStateOpt = 0;

    Nnc("&",( y ? (label_placeholder.findID(y)+1) : 0 ))++;

    trainclass.add(i);
    trainclass("&",i) = y;

    Cweightval.add(i);
    Cweightval("&",i) = Cweigh;
    Cweightvalfuzz.add(i);
    Cweightvalfuzz("&",i) = 1.0;
    epsweightval.add(i);
    epsweightval("&",i) = epsweigh;
    xxkerndiag.add(i);
    K2(xxkerndiag("&",i),i,i);
    diagoff.add(i);
    diagoff("&",i) = QUADCOSTDIAGOFFSET(trainclass(i),Cweightval(i),Cweightvalfuzz(i));
    dalpha.add(i);
    dalpha("&",i).resize(tspaceDim());
    dalpha("&",i) = 0.0;
    dalphaState.add(i);
    dalphaState("&",i) = 0;

    SparseVector<gentype> tempvec;

    tempvec.prealloc(1);

    xyval->addRowCol(i);
    Gpval->addRowCol(i);
    Gpsigma->addRowCol(i);

    xycache.add(i);
    kerncache.add(i);
    sigmacache.add(i);

    for ( q = 0 ; q < Q.size() ; q++ )
    {
	if ( q < tspaceDim() )
	{
            res |= Q("&",q).qaddTrainingVector(i,D_CALC(trainclass(i),q),tempvec,CWEIGH_CALC(y,Cweigh,1.0,q),EPSWEIGH_CALC(y,epsweigh,q));
	}

	else
	{
            res |= Q("&",q).qaddTrainingVector(i,0,tempvec);
	}
    }

    res |= QA.qaddTrainingVector(i,tempvec);

    if ( y == 0 )
    {
        res |= QA.setd(i,0);
    }

    SVM_Generic::basesetalpha(i,alphaV()(i));

    return res;
}

int SVM_MultiC_redbin::anomalyOn(int danomalyClass, double danomalyNu)
{
    isStateOpt = 0;

    NiceAssert( ( danomalyClass == -1 ) || ( danomalyClass >= 1 ) );
    NiceAssert( label_placeholder.findID(danomalyClass) == -1 );

    anomalyd = danomalyClass;

    QA.autosetLinBiasForce(danomalyNu);

    return 1;
}

int SVM_MultiC_redbin::anomalyOff(void)
{
    isStateOpt = 0;

    anomalyd = -3;

    return 1;
}

int SVM_MultiC_redbin::addclass(int label, int epszero)
{
    NiceAssert( ( label == -1 ) || ( label >= 1 ) );
    NiceAssert( !epszero );

    (void) epszero;

    if ( label )
    {
	if ( label_placeholder.findID(label) == -1 )
	{
	    isStateOpt = 0;

	    int i,j,k,q;
	    int nold  = numClasses();
	    int nnew  = numClasses()+1;
	    int mold  = CALC_TSPACEDIM(nold);
	    int mnew  = CALC_TSPACEDIM(nnew);
	    int moldq = CALC_TSPACEDIM_Q(nold);
	    int mnewq = CALC_TSPACEDIM_Q(nnew);

	    // Add label to ID store

	    label_placeholder.findOrAddID(label);
	    Nnc.add(label_placeholder.findID(label)+1);
            Nnc("&",label_placeholder.findID(label)+1) = 0;

	    // Add to label-wise variables.

	    mulCclass.add(nnew-1);
	    mulCclass("&",nnew-1) = 1.0;
	    mulepsclass.add(nnew-1);
	    mulepsclass("&",nnew-1) = 1.0;

	    // Add to classlabel-wise variables

	    if ( mnew > mold )
	    {
		for ( j = mold ; j < mnew ; j++ )
		{
		    db.add(j);
		    db("&",j) = 0.0;

		    if ( N() )
		    {
			for ( i = 0 ; i < N() ; i++ )
			{
			    dalpha("&",i).add(j);
			    dalpha("&",i)("&",j) = 0.0;
			}
		    }
		}
	    }

	    // Update representation vectors.

	    if ( nold && ( mnew > mold ) )
	    {
		for ( k = 0 ; k < nold ; k++ )
		{
		    for ( j = mold ; j < mnew ; j++ )
		    {
			classRepval("&",k).add(j);
		    }
		}
	    }

	    classRepval.add(nnew-1);
	    classRepval("&",nnew-1).resize(mnew);

	    if ( is1vsA() )
	    {
		// nnew = nold+1
		// mnew = mold+1

		if ( nold )
		{
		    for ( k = 0 ; k < nold ; k++ )
		    {
			classRepval("&",k)("&",mnew-1) = -1;
		    }
		}

                retVector<int> tmpva;

		classRepval("&",nnew-1)("&",0,1,mold-1,tmpva) = -1;
		classRepval("&",nnew-1)("&",mnew-1)           = +1;
	    }

	    else if ( is1vs1() || isDAGSVM() )
	    {
		// Adding mnew-mold = nold binary SVMs.
		//
		// mold:      0   (-1) vs nnew-1 (+1), 0 all other classes
		// mold+1:    1   (-1) vs nnew-1 (+1), 0 all other classes
		//   :
		// mnew-1: nold-1 (-1) vs nnew-1 (+1), 0 all other classes

		if ( nold && ( mnew > mold ) )
		{
		    for ( k = 0 ; k < nold ; k++ )
		    {
			for ( j = mold ; j < mnew ; j++ )
			{
			    if ( k == j-mold )
			    {
				classRepval("&",k)("&",j) = -1;
			    }

			    else
			    {
				classRepval("&",k)("&",j) = zeroint();
			    }
			}
		    }
		}

                retVector<int> tmpva;

		classRepval("&",nnew-1)("&",0,1,mold-1,tmpva)    = zeroint();
		classRepval("&",nnew-1)("&",mold,1,mnew-1,tmpva) = +1;
	    }

	    else
	    {
                NiceAssert( isMOC() );

		if ( nold && ( mnew > mold ) )
		{
		    for ( k = 0 ; k < nold ; k++ )
		    {
			for ( j = mold ; j < mnew ; j++ )
			{
			    classRepval("&",k)("&",j) = -1;
			}
		    }
		}

		int tdel = nnew;

		for ( j = mnew-1 ; j >= 0 ; j-- )
		{
		    if ( tdel >= ( 1 << j ) )
		    {
			tdel -= ( 1 << j );
			classRepval("&",nnew-1)("&",j) = +1;
		    }

		    else
		    {
			classRepval("&",nnew-1)("&",j) = -1;
		    }
		}
	    }

	    // Add new SVMs and set d and z appropriately.

	    if ( mnewq > moldq )
	    {
		for ( j = moldq ; j < mnewq ; j++ )
		{
                    Q.add(j);
                    Q("&",j) = Q(zeroint());
		}
	    }

            // Change the bias dimension if required

	    if ( mnew > mold )
	    {
		// fix bias

		for ( j = mold ; j < mnew ; j++ )
		{
                    db("&",j) = Q(j).biasR();
		}

		if ( N() )
		{
		    for ( i = 0 ; i < N() ; i++ )
		    {
			for ( q = mold ; q < mnew ; q++ )
			{
			    if ( (Q(q).d())(i) != D_CALC(trainclass(i),q) )
			    {
				Q("&",q).setd(i,D_CALC(trainclass(i),q));
			    }

                            Q("&",q).setCweight  (i,  CWEIGH_CALC(trainclass(i),  Cweightval(i),Cweightvalfuzz(i),q));
                            Q("&",q).setepsweight(i,EPSWEIGH_CALC(trainclass(i),epsweightval(i)                  ,q));

                            dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
			}
		    }
		}
	    }

	    // Fix kernel cache

	    if ( kerncache.get_min_rowdim() <= N() )
	    {
                xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
                kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
                sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
	    }

	    if ( ( kerncache.get_min_rowdim() >= (int) (N()*ROWDIMSTEPRATIO) ) && ( N() > MINROWDIM ) )
	    {
                xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),N());
                kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),N());
                sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),N());
	    }
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

void SVM_MultiC_redbin::fudgeOn(void)
{
    QA.fudgeOn();  

    if ( Q.size() )
    {
        int i;

        for ( i = 0 ; i < Q.size() ; i++ )
        {
            Q("&",i).fudgeOn();
        }
    }

    return; 
}

void SVM_MultiC_redbin::fudgeOff(void)
{ 
    QA.fudgeOff(); 

    if ( Q.size() )
    {
        int i;

        for ( i = 0 ; i < Q.size() ; i++ )
        {
            Q("&",i).fudgeOff();
        }
    }

    return; 
}

int SVM_MultiC_redbin::train(int &res, svmvolatile int &killSwitch)
{
    int i,q,result = 0;

    if ( isanomalyOn() )
    {
        xycache.padCol(4);
        kerncache.padCol(4);
        sigmacache.padCol(4);
        result |= QA.train(res,killSwitch);
        xycache.padCol(0);
        kerncache.padCol(0);
        sigmacache.padCol(0);
    }

    if ( tspaceDim() )
    {
        double dthres = Q(zeroint()).rejectThreshold();

        int dobartlett = ( ( dthres > 0 ) && ( dthres < 0.5 ) ) ? 1 : 0;
        int realN = N();
        int fakeN = 0;

        if ( dobartlett && N() )
        {
            int i;

            SparseVector<gentype> xnew;

            for ( i = 0 ; i < realN ; i++ )
            {
                //if ( d()(i) )
                {
                    xnew.fff("&",0) = i;

                    addTrainingVector(realN+fakeN,d()(i),xnew,1.0,0.0); // eps == 0 for this one, assume z = 0

                    fakeN++;
                }
            }
        }

        xycache.padCol(4);
        kerncache.padCol(4);
        sigmacache.padCol(4);

	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
            result |= Q("&",q).loctrain(res,killSwitch,realN,1);
	}

        xycache.padCol(0);
        kerncache.padCol(0);
        sigmacache.padCol(0);

        if ( dobartlett )
        {
            SVM_Generic::removeTrainingVector(realN,fakeN);
        }

	db = 0.0;
	dalphaState = zeroint();

	if ( N() )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                dalpha("&",i) = 0.0;
	    }
	}

	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
            db("&",q) = Q(q).biasR();

	    if ( N() )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    dalpha("&",i)("&",q) = (Q(q).alphaR())(i);

		    if ( (Q(q).alphaState())(i) )
		    {
			dalphaState("&",i) = 1;
		    }
		}
	    }
	}
    }

    Ns = sum(dalphaState);

    if ( result )
    {
	isStateOpt = 0;
    }

    else
    {
        isStateOpt = 1;
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return result;
}

int SVM_MultiC_redbin::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    Vector<double> gproject;
    int locclassrep = 0;
    int tempresh = 0;

    tempresh = gTrainingVector(gproject,locclassrep,i,retaltg,pxyprodi);
    resh = tempresh;
    resg = gproject;

    if ( retaltg )
    {
        Vector<double> altresg(numClasses());

        if ( numClasses() )
        {
            int i;
            int firstelm;

            for ( locclassrep = 0 ; locclassrep < numClasses() ; locclassrep++ )
            {
                altresg("&",locclassrep) = 0.0;

                // Design decision: 1-norm distance more relevant in this
                // case.

                if ( gproject.size() )
                {
                    firstelm = 1;

                    for ( i = 0 ; i < gproject.size() ; i++ )
                    {
                        if ( classRepval(locclassrep)(i) && ( firstelm || ( classRepval(locclassrep)(i)*gproject(i) < altresg(locclassrep) ) ) )
                        {
                            altresg("&",locclassrep) = classRepval(locclassrep)(i)*gproject(i);
                            firstelm = 0;
                        }
                    }
                }
            }
        }

        resg = altresg;
    }

    return tempresh;
}

int SVM_MultiC_redbin::gTrainingVector(Vector<double> &gproject, int &locclassrep, int i, int raw, gentype ***pxyprodi) const
{
    int dtv = 0;

    (void) raw;

    if ( i >= 0 )
    {
        gproject.resize(tspaceDim());

        int q;

        if ( tspaceDim() )
        {
            for ( q = 0 ; q < tspaceDim() ; q++ )
            {
                (Q(q).gTrainingVector(gproject("&",q),locclassrep,i,raw));
            }

            // Need to subtract diagonal offset here is either case, as Q(q) has no awareness of it.

            gproject.scaleAdd(-diagoff(i),dalpha(i));
        }

        if ( isanomalyOn() )
        {
            double tempres;

            if ( QA.gTrainingVector(tempres,locclassrep,i,raw) == -1 )
            {
                locclassrep = -3;
                return anomalyd;
            }
        }
    }

    else if ( ( dtv = xtang(i) & 7 ) )
    {
        NiceAssert( dtv > 0 );

        gproject.resize(tspaceDim()) = 0.0;

        if ( tspaceDim() && N() )
        {
            int j;
            double Kij;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( alphaState()(j) )
                {
                    K2(Kij,i,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                    gproject.scaleAdd(Kij,dalpha(j));
                }
            }
        }

        if ( isanomalyOn() && N() )
        {
            double temp = 0.0;

            int j;
            double Kij;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( QA.alphaState()(j) )
                {
                    K2(Kij,i,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                    temp += (Kij*(QA.alphaR()(j)));
                }
            }

            if ( temp < 0 )
            {
                locclassrep = -3;
                return anomalyd;
            }
        }
    }

    else
    {
        gproject.resize(tspaceDim()) = db;

        if ( tspaceDim() && N() )
        {
            int j;
            double Kij;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( alphaState()(j) )
                {
                    K2(Kij,i,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                    gproject.scaleAdd(Kij,dalpha(j));
                }
            }
        }

        if ( isanomalyOn() )
        {
            double temp = QA.biasR();

            if ( N() )
            {
                int j;
                double Kij;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( QA.alphaState()(j) )
                    {
                        K2(Kij,i,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                        temp += (Kij*(QA.alphaR()(j)));
                    }
                }
            }

            if ( temp < 0 )
            {
                locclassrep = -3;
                return anomalyd;
            }
        }
    }

    return classify(locclassrep,gproject);
}

int SVM_MultiC_redbin::gTrainingVector(Vector<gentype> &gprojectExt, int &locclassrep, int i, int raw, gentype ***pxyprodi) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    Vector<double> gproject(numClasses());

    int res = gTrainingVector(gproject,locclassrep,i,raw,pxyprodi);

    gprojectExt.castassign(gproject);

    return res;
}

void SVM_MultiC_redbin::locsetGp(int refactsol)
{
    int q;

    for ( q = 0 ; q < Q.size() ; q++ )
    {
	Q("&",q).setGp(Gpval,Gpsigma,xyval,refactsol);
    }

    QA.setGp(Gpval,Gpsigma,xyval,refactsol);

    return;
}

void SVM_MultiC_redbin::recalcdiagoff(int i)
{
    NiceAssert( i >= -1 );
    NiceAssert( i < N() );

    int q;

    // This updates the diagonal offsets.

    if ( N() )
    {
	isStateOpt = 0;

	if ( i == -1 )
	{
	    Vector<double> bp(N());

	    for ( i = 0 ; i < N() ; i++ )
	    {
		bp("&",i) = -diagoff(i);
                diagoff("&",i) = QUADCOSTDIAGOFFSET(trainclass(i),Cweightval(i),Cweightvalfuzz(i));
		bp("&",i) += diagoff(i);
	    }

            xxkerndiag += bp;

	    kerncache.recalcDiag();

	    //int oldmemsize = sigmacache.get_memsize();
	    //int oldrowdim  = sigmacache.get_min_rowdim();

            sigmacache.clear();

	    //sigmacache.reset(N(),&evalsigmaSVM_MultiC_redbin,(void *) this);
            //sigmacache.setmemsize(MEMSHARE_SIGMACACHE(oldmemsize),oldrowdim);

	    for ( q = 0 ; q < Q.size() ; q++ )
	    {
		Q("&",q).recalcdiagoff(bp);
	    }

            QA.recalcdiagoff(bp);
	}

	else
	{
	    double bpoff = 0.0;

	    bpoff = -diagoff(i);
            diagoff("&",i) = QUADCOSTDIAGOFFSET(trainclass(i),Cweightval(i),Cweightvalfuzz(i));
	    bpoff += diagoff(i);

            xxkerndiag("&",i) += bpoff;

	    kerncache.recalcDiag(i);

	    sigmacache.remove(i);
	    sigmacache.add(i);

	    for ( q = 0 ; q < Q.size() ; q++ )
	    {
		Q("&",q).recalcdiagoff(i,bpoff);
	    }

            QA.recalcdiagoff(i,bpoff);
	}
    }

    return;
}

int SVM_MultiC_redbin::classify(int &locclassrep, const Vector<double> &gproject) const
{
    int q,r,s;

    locclassrep = -1;
    int assclass = -1;

    if ( numClasses() )
    {
	locclassrep = 0;

	if ( numClasses() > 1 )
	{
            double dthres = Q(zeroint()).rejectThreshold();

	    if ( is1vsA() )
	    {
		// simple max wins

		max(gproject,locclassrep);

                if ( ( gproject(locclassrep) < dthres ) && ( gproject(locclassrep) > -dthres ) )
                {
 		    locclassrep = -1;

                    return 0;
                }
	    }

	    else if ( is1vs1() )
	    {
		if ( tspaceDim() )
		{
		    // voting

		    Vector<int> votsum(numClasses());

		    votsum = zeroint();

		    // 1vs0, 2vs0, 2vs1, 3vs0, 3vs1, 3vs2, ..., (n-1)vs0, (n-1)vs1, ..., (n-1)vs(n-2)

		    s = 0;

		    for ( q = 1 ; q < numClasses() ; q++ )
		    {
			for ( r = 0 ; r < q ; r++ )
			{
			    if ( gproject(s) <= -dthres )
			    {
				votsum("&",r)++;
			    }

			    else if ( gproject(s) >= dthres )
			    {
				votsum("&",q)++;
			    }

			    s++;
			}
		    }

		    max(votsum,locclassrep);

                    if ( votsum(locclassrep) == 0 )
                    {
 		        locclassrep = -1;

                        return 0;
                    }
		}

		else
		{
                    locclassrep = 0;
		}
	    }

	    else if ( isDAGSVM() )
	    {
		Vector<int> ylist(numClasses());
                Matrix<double> trainres(numClasses(),numClasses());

		ylist = 1;
                trainres = zeroint();

		s = 0;

		for ( q = 1 ; q < numClasses() ; q++ )
		{
		    for ( r = 0 ; r < q ; r++ )
		    {
                        trainres("&",q,r) = gproject(s);
                        trainres("&",r,q) = -gproject(s);

			s++;
		    }
		}

		for ( q = 1 ; q < numClasses() ; q++ )
		{
		    for ( r = 1 ; r < numClasses() ; r++ )
		    {
			if ( ylist(r) )
			{
			    for ( s = 0 ; s < r ; s++ )
			    {
				if ( ylist(s) )
				{
				    if ( trainres(r,s) <= -dthres )
				    {
					ylist("&",r) = 0;

				        goto cntpnt;
				    }

				    else if ( trainres(r,s) >= dthres )
				    {
					ylist("&",s) = 0;

				        goto cntpnt;
				    }

                                    else
                                    {
 		                        locclassrep = -1;

                                        return 0; // Might be a better way to deal with rejects (try a different tree?)
                                    }
				}
			    }
			}
		    }

		    cntpnt:
		    r = 0;
		}

		locclassrep = 0;

		for ( q = 0 ; q < numClasses() ; q++ )
		{
		    if ( ylist(q) )
		    {
			locclassrep = q;
		    }
		}
	    }

	    else
	    {
                NiceAssert( isMOC() );
		if ( tspaceDim() )
		{
		    // binary coding

		    int loc = 1;

		    for ( q = 0 ; q < tspaceDim() ; q++ )
		    {
			if ( gproject(q) >= dthres )
			{
			    locclassrep += loc;
			}

                        else if ( gproject(q) <= -dthres )
                        {
                            ;
                        }

                        else
                        {
    		            locclassrep = -1;

                            return 0;
                        }

			loc *= 2;
		    }

		    // FIXME: should use Hamming distance here.

		    if ( locclassrep >= numClasses() )
		    {
			locclassrep = numClasses()-1;
		    }
		}

		else
		{
		    locclassrep = 0;
		}
	    }
	}

        assclass = label_placeholder.findref(locclassrep);
    }

    return assclass;
}

void SVM_MultiC_redbin::changemultitype(int newmultitype)
{
    int i,j,k,q;
    int oldmultitype = multitype;

    if ( oldmultitype == newmultitype )
    {
	return;
    }

    SparseVector<gentype> tempvec;

    isStateOpt = 0;

    multitype = newmultitype;

    int newtspaceDim  = CALC_TSPACEDIM(numClasses());
    int newtspaceDimQ = CALC_TSPACEDIM_Q(numClasses());

    // Fix class representation

    if ( newtspaceDim )
    {
	classRepval.resize(numClasses());

	for ( q = 0 ; q < numClasses() ; q++ )
	{
	    classRepval("&",q).resize(newtspaceDim);
	}

	if ( is1vsA() )
	{
	    for ( q = 0 ; q < numClasses() ; q++ )
	    {
		for ( k = 0 ; k < newtspaceDim ; k++ )
		{
		    if ( k == q )
		    {
			classRepval("&",q)("&",k) = +1;
		    }

		    else
		    {
			classRepval("&",q)("&",k) = -1;
		    }
		}
	    }
	}

	else if ( is1vs1() || isDAGSVM() )
	{
	    i = 1;
	    j = 0;

	    for ( k = 0 ; k < newtspaceDim ; k++ )
	    {
		for ( q = 0 ; q < numClasses() ; q++ )
		{
		    if ( q == i )
		    {
			classRepval("&",q)("&",k) = +1;
		    }

		    else if ( q == j )
		    {
			classRepval("&",q)("&",k) = -1;
		    }

		    else
		    {
			classRepval("&",q)("&",k) = zeroint();
		    }
		}

		if ( j == i-1 )
		{
		    i++;
		    j = 0;
		}

		else
		{
		    j++;
		}
	    }
	}

	else
	{
            NiceAssert( isMOC() );

	    for ( q = 0 ; q < numClasses() ; q++ )
	    {
		int tdel = q;

		for ( k = newtspaceDim-1 ; k >= 0 ; k-- )
		{
		    classRepval("&",q)("&",k) = -1;

		    if ( tdel >= ( 1 << k ) )
		    {
			tdel -= ( 1 << k );

			classRepval("&",q)("&",k) = +1;
		    }
		}
	    }
	}
    }

    // resize Q

    while ( Q.size() > newtspaceDimQ )
    {
	Q.remove((Q.size())-1);
    }

    while ( Q.size() < newtspaceDimQ )
    {
	Q.add(Q.size());
        Q("&",Q.size()-1) = Q(zeroint());
    }

    // Fix db, and alpha dimension

    db.resize(newtspaceDim);

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
	    dalpha("&",i).resize(newtspaceDim);
	}
    }

    // set d,C,epsilon,z in individual binary SVMs

    if ( tspaceDim() && N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
	    for ( q = 0 ; q < tspaceDim() ; q++ )
	    {
		if ( (Q(q).d())(i) != D_CALC(trainclass(i),q) )
		{
		    Q("&",q).setd(i,D_CALC(trainclass(i),q));
		}

                Q("&",q).setCweight  (i,  CWEIGH_CALC(trainclass(i),  Cweightval(i),Cweightvalfuzz(i),q));
                Q("&",q).setepsweight(i,EPSWEIGH_CALC(trainclass(i),epsweightval(i)                  ,q));

                dalpha("&",i)("&",q) = (Q(q).alphaR())(i);
	    }
	}

	for ( q = 0 ; q < tspaceDim() ; q++ )
	{
            db("&",q) = Q(q).biasR();
	}
    }

    // Fix kernel cache

    if ( kerncache.get_min_rowdim() <= N() )
    {
        xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
        kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
        sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (N()*ROWDIMSTEPRATIO));
    }

    if ( ( kerncache.get_min_rowdim() >= (int) (N()*ROWDIMSTEPRATIO) ) && ( N() > MINROWDIM ) )
    {
        xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),N());
        kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),N());
        sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),N());
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return;
}

int SVM_MultiC_redbin::fixautosettings(int kernchange, int Nchange, int ncut)
{
    int res = 0;

    if ( kernchange || Nchange )
    {
	switch ( autosetLevel )
	{
        case 1: { if ( Nchange ) { res = 1; autosetCscaled(autosetCvalx,ncut);                    } break; }
        case 2: {                  res = 1; autosetCKmean(ncut);                                    break; }
        case 3: {                  res = 1; autosetCKmedian(ncut);                                  break; }
        case 4: {                  res = 1; autosetCNKmean(ncut);                                   break; }
        case 5: {                  res = 1; autosetCNKmedian(ncut);                                 break; }
        case 6: { if ( Nchange ) { res = 1; autosetLinBiasForce(autosetnuvalx,autosetCvalx,ncut); } break; }
	default: { break; }
	}
    }

    return res;
}

double SVM_MultiC_redbin::autosetkerndiagmean(void)
{
    Vector<int> dnonzero;

    if ( N()-NNC(0) )
    {
	int i,j = 0;

	for ( i = 0 ; i < N() ; i++ )
	{
	    if ( trainclass(i) != 0 )
	    {
		dnonzero.add(j);
		dnonzero("&",j) = i;

                j++;
	    }
	}
    }

    retVector<double> tmpva;

    return mean(xxkerndiag(dnonzero,tmpva));
}

double SVM_MultiC_redbin::autosetkerndiagmedian(void)
{
    Vector<int> dnonzero;

    int i,j = 0;

    if ( N()-NNC(0) )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
	    if ( trainclass(i) != 0 )
	    {
		dnonzero.add(j);
		dnonzero("&",j) = i;

                j++;
	    }
	}
    }

    retVector<double> tmpva;

    return median(xxkerndiag(dnonzero,tmpva),i);
}

std::ostream &SVM_MultiC_redbin::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Multiclass Reduction to Binary SVM\n\n";

    repPrint(output,'>',dep) << "Cost type:                       " << costType          << "\n";
    repPrint(output,'>',dep) << "Multiclass type:                 " << multitype         << "\n";
    repPrint(output,'>',dep) << "Opt type (0 act, 1 smo, 2 d2c, 3 grad):  " << optType           << "\n";

    repPrint(output,'>',dep) << "C:                               " << CNval             << "\n";
    repPrint(output,'>',dep) << "eps:                             " << epsval            << "\n";
    repPrint(output,'>',dep) << "C+-:                             " << mulCclass         << "\n";
    repPrint(output,'>',dep) << "eps+-:                           " << mulepsclass       << "\n";

    repPrint(output,'>',dep) << "Parameter autoset level:         " << autosetLevel      << "\n";
    repPrint(output,'>',dep) << "Parameter autoset nu value:      " << autosetnuvalx     << "\n";
    repPrint(output,'>',dep) << "Parameter autoset C value:       " << autosetCvalx      << "\n";

    repPrint(output,'>',dep) << "XY cache details:                " << xycache           << "\n";
    repPrint(output,'>',dep) << "Kernel cache details:            " << kerncache         << "\n";
    repPrint(output,'>',dep) << "Sigma cache details:             " << sigmacache        << "\n";
    repPrint(output,'>',dep) << "Kernel diagonals:                " << xxkerndiag        << "\n";
    repPrint(output,'>',dep) << "Diagonal offsets:                " << diagoff           << "\n";

    repPrint(output,'>',dep) << "Alpha:                           " << dalpha            << "\n";
    repPrint(output,'>',dep) << "Alpha state:                     " << dalphaState       << "\n";
    repPrint(output,'>',dep) << "Bias:                            " << db                << "\n";
    repPrint(output,'>',dep) << "Label placeholder storage:       " << label_placeholder << "\n";
    repPrint(output,'>',dep) << "Class representations:           " << classRepval       << "\n";
    repPrint(output,'>',dep) << "Ns:                              " << Ns                << "\n";
    repPrint(output,'>',dep) << "Nnc:                             " << Nnc               << "\n";
    repPrint(output,'>',dep) << "Is SVM optimal:                  " << isStateOpt        << "\n";

    SVM_Generic::printstream(output,dep+1);
    
    repPrint(output,'>',dep) << "Training classes:                " << trainclass        << "\n";
    repPrint(output,'>',dep) << "Training C weights:              " << Cweightval        << "\n";
    repPrint(output,'>',dep) << "Training C weights (fuzz):       " << Cweightvalfuzz    << "\n";
    repPrint(output,'>',dep) << "Training eps weights:            " << epsweightval      << "\n";

    repPrint(output,'>',dep) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    repPrint(output,'>',dep) << "Optimisation state:              " << (Q).size()        << "\n";
    repPrint(output,'>',dep) << "Anomaly class (-3 if unset):     " << anomalyd          << "\n";
    repPrint(output,'>',dep) << "Anomaly detector:                " << QA                << "\n";

    int i;

    for ( i = 0 ; i < (Q).size() ; i++ )
    {
        repPrint(output,'>',dep) << "Submachine " << i << ": "; (Q)(i).printstream(output,dep+1);
    }

    repPrint(output,'>',dep) << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";

    return output;
}

std::istream &SVM_MultiC_redbin::inputstream(std::istream &input)
{
    int i,Qsize;
    wait_dummy dummy;

    input >> dummy; input >> costType;
    input >> dummy; input >> multitype;
    input >> dummy; input >> optType;

    input >> dummy; input >> CNval;
    input >> dummy; input >> epsval;
    input >> dummy; input >> mulCclass;
    input >> dummy; input >> mulepsclass;

    input >> dummy; input >> autosetLevel;
    input >> dummy; input >> autosetnuvalx;
    input >> dummy; input >> autosetCvalx;

    input >> dummy; input >> xycache;
    input >> dummy; input >> kerncache;
    input >> dummy; input >> sigmacache;
    input >> dummy; input >> xxkerndiag;
    input >> dummy; input >> diagoff;

    input >> dummy; input >> dalpha;
    input >> dummy; input >> dalphaState;
    input >> dummy; input >> db;
    input >> dummy; input >> label_placeholder;
    input >> dummy; input >> classRepval;
    input >> dummy; input >> Ns;
    input >> dummy; input >> Nnc;
    input >> dummy; input >> isStateOpt;

    SVM_Generic::inputstream(input);
    
    input >> dummy; input >> trainclass;
    input >> dummy; input >> Cweightval;
    input >> dummy; input >> Cweightvalfuzz;
    input >> dummy; input >> epsweightval;

    input >> dummy; input >> Qsize;
    input >> dummy; input >> anomalyd;
    input >> dummy; input >> QA;

    (Q).resize(Qsize);

    for ( i = 0 ; i < Qsize ; i++ )
    {
        input >> dummy; (Q)("&",i).inputstream(input);
    }

    if ( Gpval != NULL )
    {
        MEMDEL(xyval);
        MEMDEL(Gpval);
        MEMDEL(Gpsigma);

        xyval   = NULL;
        Gpval   = NULL;
        Gpsigma = NULL;
    }



    MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &(xycache),   N(),N()));
    MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &(kerncache), N(),N()));
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &(sigmacache),N(),N()));

    int oldmemsize = (kerncache).get_memsize();
    int oldrowdim  = (kerncache).get_min_rowdim();

    (xycache).reset(N(),&evalXYSVM_MultiC_redbin,this);
    (xycache).setmemsize(MEMSHARE_XYCACHE(oldmemsize),oldrowdim);

    (kerncache).reset(N(),&evalKSVM_MultiC_redbin,this);
    (kerncache).setmemsize(MEMSHARE_KCACHE(oldmemsize),oldrowdim);

    (sigmacache).reset(N(),&evalsigmaSVM_MultiC_redbin,this);
    (sigmacache).setmemsize(MEMSHARE_SIGMACACHE(oldmemsize),oldrowdim);


    locsetGp(0);

    return input;
}

#define OVERPREALLOC 1.05
int SVM_MultiC_redbin::prealloc(int expectedN)
{
    xxkerndiag.prealloc(expectedN);
    diagoff.prealloc(expectedN);
    dalpha.prealloc(expectedN);
    dalphaState.prealloc(expectedN);
    trainclass.prealloc(expectedN);
    Cweightval.prealloc(expectedN);
    Cweightvalfuzz.prealloc(expectedN);
    epsweightval.prealloc(expectedN);
    SVM_Generic::prealloc(expectedN);

    // NB: can't anticipate the number of points in each class in this case,
    // so we can't prealloc them exactly - need to guess :(
    // FIXME: MOC preallocation equation is obviously wrong
    // NB: actually they all contain all vectors, but the non-relevant ones
    // are marked zero.  This is so that they can share a common root Gp.
    // Really not sure if this is a great idea to be honest.

    int Nguess = expectedN;

//         if ( is1vsA()   ) { Nguess = expectedN;              }
//    else if ( is1vs1()   ) { Nguess = (int) (2*OVERPREALLOC*((double) expectedN)/((double) numClasses())); }
//    else if ( isDAGSVM() ) { Nguess = (int) (2*OVERPREALLOC*((double) expectedN)/((double) numClasses())); }
//    else if ( isMOC()    ) { Nguess = (int) (OVERPREALLOC*expectedN/(1<<(tspaceDim()-1))); }

    xycache.prealloc(Nguess);
    kerncache.prealloc(Nguess);
    sigmacache.prealloc(Nguess);

    if ( Q.size() )
    {
        int i;
    
        for ( i= 0  ; i < Q.size() ; i++ )
        {
            Q("&",i).prealloc(Nguess);
        }
    }

    QA.prealloc(expectedN);

    return 0;
}

int SVM_MultiC_redbin::preallocsize(void) const
{
    return SVM_Generic::preallocsize();
}

int SVM_MultiC_redbin::randomise(double sparsity)
{
    NiceAssert( sparsity >= 0 );
    NiceAssert( sparsity <= 1 );

    int res = 0;
    int i,q;

    if ( tspaceDim() )
    {
        for ( q = 0 ; q < tspaceDim() ; q++ )
        {
            res |= Q("&",q).randomise(sparsity);
        }
    }

    if ( res )
    {
        // NB: FOLLOWING CODE TAKEN FROM TRAINING FUNCTION

        if ( tspaceDim() )
        {
            db = 0.0;
            dalphaState = zeroint();

            if ( N() )
            {
                for ( i = 0 ; i < N() ; i++ )
                {
                    dalpha("&",i) = 0.0;
                }
            }

            for ( q = 0 ; q < tspaceDim() ; q++ )
            {
                db("&",q) = Q(q).biasR();

                if ( N() )
                {
                    for ( i = 0 ; i < N() ; i++ )
                    {
                        dalpha("&",i)("&",q) = (Q(q).alphaR())(i);

                        if ( (Q(q).alphaState())(i) )
                        {
                            dalphaState("&",i) = 1;
                        }
                    }
                }
            }
        }

        Ns = sum(dalphaState);

        isStateOpt = 0;

        SVM_Generic::basesetAlphaBiasFromAlphaBiasV();
    }

    return res;
}
























