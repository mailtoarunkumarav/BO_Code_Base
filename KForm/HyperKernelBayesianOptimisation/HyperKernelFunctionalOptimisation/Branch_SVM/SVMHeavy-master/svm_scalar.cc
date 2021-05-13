
//
// Scalar regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_scalar.h"
#include "sQsLsAsWs.h"
#include "sQsmo.h"
#include "sQd2c.h"
#include "sQgraddesc.h"
#include "linsolve.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

class SVM_Binary;
class SVM_Single;

#define LCALC(_C_,_Cclass_,_xclass_,_Cweigh_,_Cweighfuzz_)              ( !isLinearCost() ? -(MAXBOUND) : -( (_Cclass_)((_xclass_)+1) * (_C_) * (_Cweigh_) * (_Cweighfuzz_) ) )
#define HCALC(_C_,_Cclass_,_xclass_,_Cweigh_,_Cweighfuzz_)              ( !isLinearCost() ?  (MAXBOUND) :  ( (_Cclass_)((_xclass_)+1) * (_C_) * (_Cweigh_) * (_Cweighfuzz_) ) )
#define CRCALC(_C_,_Cclass_,_xclass_,_Cweigh_,_Cweighfuzz_)             ( !is1NormCost() ? 0 : ( (_Cclass_)((_xclass_)+1) * (_C_) * (_Cweigh_) * (_Cweighfuzz_) ) )
#define DRCALC(_C_,_Cclass_,_xclass_,_Cweigh_,_Cweighfuzz_)             ( !is1NormCost() ? 0 : ( (_Cclass_)((_xclass_)+1) * (_C_) * (_Cweigh_) * (_Cweighfuzz_) ) )
#define HPCALC(_E_,_Eclass_,_xclass_,_Eweigh_)                          ( ( (_Eclass_)((_xclass_)+1) * (_E_) * (_Eweigh_) ) )
#define HPSCALECALC(_Eclass_,_xclass_,_Eweigh_)                         ( ( (_Eclass_)((_xclass_)+1)         * (_Eweigh_) ) )
#define QUADCOSTDIAGOFFSET(_C_,_Cclass_,_xclass_,_Cweigh_,_Cweighfuzz_) ( !isQuadraticCost() ? 0 : ( 1 / ( (_Cclass_)((_xclass_)+1) * (_C_) * (_Cweigh_) * (_Cweighfuzz_) ) ) )
#define ALPHARESTRICT(_xclass_)                                         ( ( (_xclass_) == +2 ) ? 0 : ( ( (_xclass_) == +1 ) ? 1 : ( ( (_xclass_) == -1 ) ? 2 : 3 ) ) )
#define GPNWIDTH(_biasdim_)                                             ( ( (_biasdim_) == 0 ) ? 1 : ( ( (_biasdim_) > 0 ) ? (_biasdim_)-1 : -(_biasdim_)   ) )
#define GPNSETHEIGHT(_biasdim_)                                         ( ( (_biasdim_) == 0 ) ? 1 : ( ( (_biasdim_) > 0 ) ? (_biasdim_)-1 : -(_biasdim_)-1 ) )
#define GPNorGPNEXT(_Gpn_,_GpnExt_)                                     ( ( (_GpnExt_) == NULL ) ? (_Gpn_) : *(_GpnExt_ ) )

#define MEMSHARE_KCACHE(_totmem_)     isOptActive() ? ( ( (_totmem_) > 0 ) ? (_totmem_) : 1 ) : ( ( (_totmem_)/2 > 0 ) ? (_totmem_)/2 : 1 )
#define MEMSHARE_XYCACHE(_totmem_)    isOptActive() ? ( ( (_totmem_) > 0 ) ? (_totmem_) : 1 ) : ( ( (_totmem_)/2 > 0 ) ? (_totmem_)/2 : 1 )
#define MEMSHARE_SIGMACACHE(_totmem_) isOptActive() ? 1 : ( ( (_totmem_)/2 > 0 ) ? (_totmem_)/2 : 1 )



//#define K4DIAGOFF 1e-4
#define K4DIAGOFF 0.0
//#define K4DIAGOFF 1.0
//#define K4DIAGOFF 1e-2

void evalKSVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    double tres;

    SVM_Scalar *realOwner = (SVM_Scalar *) owner;

    NiceAssert( realOwner );

    int iskip = realOwner->iskip;
    int emm = realOwner->emm;

    if ( ( realOwner->inEmm4Solve == 1 ) && ( emm == 4 ) )
    {
        NiceAssert( iskip < 0 );

        int k,l;//,t;
        int ii,jj,kk,ll;

        if ( i < j )
        {
            k = i;
            i = j;
            j = k;
        }

        int NZ = realOwner->prevNZ;
        double ****emm4K4cache = realOwner->emm4K4cache;
        Vector<int> &alphaPrevPivNZ = (realOwner->alphaPrevPivNZ);
        Vector<double> &alphaPrev = (realOwner->alphaPrev);

        res = ( !NZ && ( i == j ) ) ? K4DIAGOFF : 0.0;

	if ( NZ )
	{
            int kx,lx;

            for ( kx = 0 ; kx < NZ ; kx++ )
            {
//NB: alphaPrevPivNZ is sorted smallest to largest, hence can just go to kx and double-up via symmetry
//                for ( lx = 0 ; lx < NZ ; lx++ )
                for ( lx = 0 ; lx <= kx ; lx++ )
                {
                    k = alphaPrevPivNZ(kx);
                    l = alphaPrevPivNZ(lx);

//                    if ( k < l )
//                    {
//                        t = k;
//                        k = l;
//                        l = t;
//                    }

//                    if ( emm4K4cache && ( k >= l ) )
                    if ( emm4K4cache )
                    {
                       // We start with the following possible orderings for i,j,k,l, largest to smallest
                       //
                       // ijkl
                       // ijlk
                       // ikjl
                       // iklj
                       // ilkj
                       // iljk
                       //
                       // jikl
                       // jilk
                       // jkil
                       // jkli
                       // jlki
                       // jlik
                       //
                       // kjil
                       // kjli
                       // kijl
                       // kilj
                       // klij
                       // klji
                       //
                       // ljki
                       // ljik
                       // lkji
                       // lkij
                       // likj
                       // lijk
                       //
                       // Can assume that i >= j, so this comes down to
                       //
                       // ijkl
                       // ijlk
                       // ikjl
                       // iklj
                       // ilkj
                       // iljk
                       //
                       // kijl
                       // kilj
                       // klij
                       //
                       // lkij
                       // likj
                       // lijk
                       //
                       // We can further restrict our range to k >= l (we need to make sure we double-add if k > l), so this comes down to:
                       //
                       // ijkl
                       // ikjl
                       // iklj
                       //
                       // kijl
                       // kilj
                       // klij
                       //
                       // which we further reorder to put the simplest tests first.
                       //
                       // ijkl
                       // klij
                       //
                       // iklj
                       // kijl
                       //
                       // ikjl
                       // kilj (which we make the default)
                       //
                       // and hence we need only consider the following options:

                        ii = k;
                        jj = i;
                        kk = l;
                        ll = j;

                             if ( j >= k ) { ii = i; jj = j; kk = k; ll = l; }
                        else if ( l >= i ) { ii = k; jj = l; kk = i; ll = j; }
                        else if ( ( i >= k ) && ( l >= j ) ) { ii = i; jj = k; kk = l; ll = j; }
                        else if ( ( k >= i ) && ( j >= l ) ) { ii = k; jj = i; kk = j; ll = l; }
                        else if ( ( i >= k ) && ( k >= j ) && ( j >= l ) ) { ii = i; jj = k; kk = j; ll = l; }

/* Original full list
                        ii = i; 
                        jj = j; 
                        kk = k; 
                        ll = l;

                             if ( ( i >= j ) && ( j >= k ) && ( k >= l ) ) { ii = i; jj = j; kk = k; ll = l; }
                        else if ( ( i >= j ) && ( j >= l ) && ( l >= k ) ) { ii = i; jj = j; kk = l; ll = k; }
                        else if ( ( i >= k ) && ( k >= j ) && ( j >= l ) ) { ii = i; jj = k; kk = j; ll = l; }
                        else if ( ( i >= k ) && ( k >= l ) && ( l >= j ) ) { ii = i; jj = k; kk = l; ll = j; }
                        else if ( ( i >= l ) && ( l >= k ) && ( k >= j ) ) { ii = i; jj = l; kk = k; ll = j; }
                        else if ( ( i >= l ) && ( l >= j ) && ( j >= k ) ) { ii = i; jj = l; kk = j; ll = k; }

                        else if ( ( j >= i ) && ( i >= k ) && ( k >= l ) ) { ii = j; jj = i; kk = k; ll = l; }
                        else if ( ( j >= i ) && ( i >= l ) && ( l >= k ) ) { ii = j; jj = i; kk = l; ll = k; }
                        else if ( ( j >= k ) && ( k >= i ) && ( i >= l ) ) { ii = j; jj = k; kk = i; ll = l; }
                        else if ( ( j >= k ) && ( k >= l ) && ( l >= i ) ) { ii = j; jj = k; kk = l; ll = i; }
                        else if ( ( j >= l ) && ( l >= k ) && ( k >= i ) ) { ii = j; jj = l; kk = k; ll = i; }
                        else if ( ( j >= l ) && ( l >= i ) && ( i >= k ) ) { ii = j; jj = l; kk = i; ll = k; }

                        else if ( ( k >= j ) && ( j >= i ) && ( i >= l ) ) { ii = k; jj = j; kk = i; ll = l; }
                        else if ( ( k >= j ) && ( j >= l ) && ( l >= i ) ) { ii = k; jj = j; kk = l; ll = i; }
                        else if ( ( k >= i ) && ( i >= j ) && ( j >= l ) ) { ii = k; jj = i; kk = j; ll = l; }
                        else if ( ( k >= i ) && ( i >= l ) && ( l >= j ) ) { ii = k; jj = i; kk = l; ll = j; }
                        else if ( ( k >= l ) && ( l >= i ) && ( i >= j ) ) { ii = k; jj = l; kk = i; ll = j; }
                        else if ( ( k >= l ) && ( l >= j ) && ( j >= i ) ) { ii = k; jj = l; kk = j; ll = i; }

                        else if ( ( l >= j ) && ( j >= k ) && ( k >= i ) ) { ii = l; jj = j; kk = k; ll = i; }
                        else if ( ( l >= j ) && ( j >= i ) && ( i >= k ) ) { ii = l; jj = j; kk = i; ll = k; }
                        else if ( ( l >= k ) && ( k >= j ) && ( j >= i ) ) { ii = l; jj = k; kk = j; ll = i; }
                        else if ( ( l >= k ) && ( k >= i ) && ( i >= j ) ) { ii = l; jj = k; kk = i; ll = j; }
                        else if ( ( l >= i ) && ( i >= k ) && ( k >= j ) ) { ii = l; jj = i; kk = k; ll = j; }
                        else if ( ( l >= i ) && ( i >= j ) && ( j >= k ) ) { ii = l; jj = i; kk = j; ll = k; }
*/

                        res += ( ( k == l ) ? 1 : 2 )*emm4K4cache[ii][jj][kk][ll]*alphaPrev(k)*alphaPrev(l);
                    }

                    else
                    {
                        realOwner->K4(tres,i,j,k,l,pxyprod);
                        res += ( ( k == l ) ? 1 : 2 )*tres*alphaPrev(k)*alphaPrev(l);
                    }
                }
	    }
	}
    }

    else if ( ( realOwner->inEmm4Solve == 1 ) && ( emm > 4 ) )
    {
        NiceAssert( !(emm%2) );
        NiceAssert( iskip < 0 );

        int k;

        int NZ = realOwner->prevNZ;
        Vector<int> &alphaPrevPivNZ = (realOwner->alphaPrevPivNZ);
        Vector<double> &alphaPrev = (realOwner->alphaPrev);

        res = ( !NZ && ( i == j ) ) ? K4DIAGOFF : 0.0;

	if ( NZ )
	{
            retVector<int> tmpva;
            retVector<int> tmpvb;
            retVector<int> tmpvc;
            retVector<double> tmpvd;

            Vector<int> kx(emm);
            Vector<int> kk(emm);
            int z = 0;

            kx("&",z) = i;
            kx("&",1) = j;
            kx("&",2,1,emm-1,tmpva) = z;

            kk("&",z) = i;
            kk("&",1) = j;
            kk("&",2,1,emm-1,tmpva) = alphaPrevPivNZ(kx(2,1,emm-1,tmpvb),tmpvc);

            int isdone = 0;

            while ( !isdone )
            {
                tres  = realOwner->Km(emm,tres,kk,pxyprod);
                tres *= prod(alphaPrev(kk(2,1,emm-1,tmpva),tmpvd));

                res += tres;

                isdone = 1;
                k = 2;

                while ( isdone && ( k < emm ) )
                { 
                    kx("&",k)++;

                    if ( kx(k) < NZ )
                    {
                        kk("&",k) = alphaPrevPivNZ(kx(k));

                        isdone = 0;
                        break;
                    }

                    else
                    {
                        kk("&",k) = alphaPrevPivNZ(z);
                        kx("&",k) = 0;

                        k++;
                    }
                }
            }
	}
    }

    else
    {
	if ( i != j )
	{
            int ix = i;
            int jx = j;

            if ( iskip >= 0 )
            {
                ix = ( ix >= iskip ) ? ix+1 : ix;
                jx = ( jx >= iskip ) ? jx+1 : jx;
            }

            realOwner->K2(res,ix,jx,pxyprod);

            NiceAssert( !testisvnan(res) );
            NiceAssert( !testisinf(res) );
	}

	else
	{
	    res = (realOwner->kerndiagval)(i);

            NiceAssert( !testisvnan(res) );
            NiceAssert( !testisinf(res) );
	}
    }

    res += ( ( i == j ) ? (realOwner->diagoff)(i) : 0 );

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return;
}

void evalxySVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    SVM_Scalar *realOwner = (SVM_Scalar *) owner;

    NiceAssert( realOwner );

    int iskip = realOwner->iskip;

    int ix = i;
    int jx = j;

    if ( iskip >= 0 )
    {
        ix = ( ix >= iskip ) ? ix+1 : ix;
        jx = ( jx >= iskip ) ? jx+1 : jx;
    }

    realOwner->K2ip(res,ix,jx,pxyprod);

    return;
}

void evalSigmaSVM_Scalar(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    NiceAssert( owner );

    const Matrix<double> &Gp = (*(((SVM_Scalar *) owner)->Gpval));

    int N = ( Gp.numRows() < Gp.numCols() ) ? Gp.numRows() : Gp.numCols();

    if ( ( i < N ) && ( j < N ) )
    {
        res = Gp(i,i)+Gp(j,j)-(2.0*Gp(i,j));
    }

    else
    {
        res = 0;
    }

    return;
}





















SVM_Scalar::SVM_Scalar() : SVM_Generic()
{
    diagkernvalcheat = NULL;

    thisthis = this;
    thisthisthis = &thisthis;

    alpharestrictoverride = 0;
    Qconstype = 0;

    setaltx(NULL);

    inEmm4Solve = 0;
    emm4K4cache = NULL;

    GpnExt = NULL;

    costType        = 0;
    biasType        = 0;
    optType         = 0;
    tubeshrink      = 0;
    epsrestrict     = 1;
    maxitcntval     = DEFAULT_MAXITCNT;
    maxtraintimeval = DEFAULT_MAXTRAINTIME;
    opttolval       = DEFAULT_OPTTOL;
    outerlrval      = MULTINORM_OUTERSTEP;
    outermomval     = MULTINORM_OUTERMOMENTUM;
    outermethodval  = MULTINORM_OUTERMETHOD;
    outertolval     = MULTINORM_OUTERACCUR;
    outerovscval    = MULTINORM_OUTEROVSC;
    outermaxits     = MULTINORM_MAXITS;
    outermaxcacheN  = MULTINORM_FULLCACHE_N;
    makeConvex      = 0;

    isStateOpt      = 1;

    emm              = DEFAULT_EMM;
    CNval            = DEFAULT_C;
    epsval           = DEFAULTEPS;
    bfixval          = 0.0;
    nuLin            = 0.0;
    nuQuadv          = 0.0;
    linbiasforceval  = 0.0;
    quadbiasforceval = 0.0;

    autosetLevel  = 0;
    autosetnuvalx = 0.0;
    autosetCvalx  = 0.0;

    xCclass.resize(4);
    xepsclass.resize(4);

    xCclass   = 1.0;
    xepsclass = 1.0;

    maxiterfuzztval = DEFAULT_MAXITERFUZZT;
    usefuzztval     = 0;
    lrfuzztval      = DEFAULT_LRFUZZT;
    ztfuzztval      = DEFAULT_ZTFUZZT;
    costfnfuzztval  = DEFAULT_COSTFNFUZZT;

    xycache.reset(0,&evalxySVM_Scalar,(void *) this);
    xycache.setmemsize(MEMSHARE_XYCACHE(DEFAULT_MEMSIZE),MINROWDIM);

    kerncache.reset(0,&evalKSVM_Scalar,(void *) this);
    kerncache.setmemsize(MEMSHARE_KCACHE(DEFAULT_MEMSIZE),MINROWDIM);
                          
    sigmacache.reset(0,&evalSigmaSVM_Scalar,(void *) this);
    sigmacache.setmemsize(MEMSHARE_SIGMACACHE(DEFAULT_MEMSIZE),MINROWDIM);

    Gplocal = 1;

    MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &xycache   ,0,0));
    MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,0,0));
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,0,0));

    Q.setkeepfact(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),1);

    gn.add(0);
    gn("&",0) = linbiasforceval;

    Gn.addRowCol(0);
    Gn("&",0,0) = quadbiasforceval;

    Gpn.addCol(0);

    Q.addBeta(0,0,0.0);
    Q.setopttol(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,opttolval);

    biasdim = 0;

    Nnc.resize(4);
    Nnc = zeroint();

    classLabelsval.resize(3);
    classRepval.resize(3);
    u.resize(2);

    classLabelsval("&",0) = -1;
    classLabelsval("&",1) = +1;
    classLabelsval("&",2) = 2;

    classRepval("&",0).resize(1); classRepval("&",0)("&",0) = -1;
    classRepval("&",1).resize(1); classRepval("&",1)("&",0) = +1;
    classRepval("&",2).resize(1); classRepval("&",2)("&",0) = 2;

    u("&",0).resize(1); u("&",0)("&",0) = -1.0;
    u("&",1).resize(1); u("&",1)("&",0) = +1.0;

    iskip = -1;

    return;
}

SVM_Scalar::SVM_Scalar(const SVM_Scalar &src) : SVM_Generic()
{
    diagkernvalcheat = NULL;

    thisthis = this;
    thisthisthis = &thisthis;

    alpharestrictoverride = 0;
    Qconstype = 0;

    iskip = -1;

    setaltx(NULL);

    inEmm4Solve = 0;
    emm4K4cache = NULL;

    GpnExt = NULL;

    Gplocal = 0;

    xyval   = NULL;
    Gpval   = NULL;
    Gpsigma = NULL;

    assign(src,0);

    return;
}

SVM_Scalar::SVM_Scalar(const SVM_Scalar &src, const ML_Base *xsrc) : SVM_Generic()
{
    diagkernvalcheat = NULL;

    thisthis = this;
    thisthisthis = &thisthis;

    alpharestrictoverride = 0;
    Qconstype = 0;

    iskip = -1;

    setaltx(xsrc);

    inEmm4Solve = 0;
    emm4K4cache = NULL;

    GpnExt = NULL;

    Gplocal = 0;

    xyval   = NULL;
    Gpval   = NULL;
    Gpsigma = NULL;

    assign(src,1);

    return;
}

SVM_Scalar::~SVM_Scalar()
{
    if ( Gplocal )
    {
	MEMDEL(xyval);
	xyval = NULL;

	MEMDEL(Gpval);
	Gpval = NULL;

        MEMDEL(Gpsigma);
	Gpsigma = NULL;
    }

    return;
}

double SVM_Scalar::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db == +1 )
    {
        // treat as lower bound constraint ha >= hb

        if ( (double) ha < (double) hb )
        {
            res = ( (double) ha ) - ( (double) hb );
            res *= res;
        }
    }

    else if ( db == -1 )
    {
        // treat as upper bound constraint ha <= hb

        if ( (double) ha > (double) hb )
        {
            res = ( (double) ha ) - ( (double) hb );
            res *= res;
        }
    }

    else if ( db )
    {
        res = ( (double) ha ) - ( (double) hb );
        res *= res;
    }

    return res;
}

int SVM_Scalar::setAlphaR(const Vector<double> &newAlpha)
{
    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;
    }

    int z = 0;

    retMatrix<double> tmpma;
    retMatrix<double> tmpmb;

    Q.setAlpha(newAlpha,(*Gpval)(z,1,SVM_Scalar::N()-1,z,1,SVM_Scalar::N()-1,tmpma),(*Gpval)(z,1,SVM_Scalar::N()-1,z,1,SVM_Scalar::N()-1,tmpmb),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,lb,ub);

    // update gentype alpha

    SVM_Generic::basesetAlphaBiasFromAlphaBiasR();

    return 1;
}

int SVM_Scalar::setBiasR(const double &newBias)
{
    NiceAssert( !( ( isPosBias() && ( newBias < 0.0 ) ) || ( isNegBias() && ( newBias > 0.0 ) ) ) );

    isStateOpt = 0;

    if ( !isFixedBias() )
    {
	Vector<double> newBeta(GPNWIDTH(biasdim));

        newBeta = newBias;
	Q.setBeta(newBeta,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    }

    else
    {
        Vector<double> gpnew(traintarg);

	bfixval = newBias;
	gpnew.negate();
	gpnew += bfixval;
	Q.refactgp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gpnew,gn,hp);
	gp = gpnew;
    }

    // update gentype bias

    SVM_Generic::basesetbias(biasR());

    return 1;
}

int SVM_Scalar::scale(double a)
{
    NiceAssert( a >= 0.0 );
    NiceAssert( a <= 1.0 );

    int i,d;

    if ( a == 0.0 )
    {
	isStateOpt = 0;

	// Constrain all alphas to zero (use setd to cheat here)

        if ( SVM_Scalar::N() )
	{
            for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
	    {
		d = trainclass(i);

                SVM_Scalar::setdinternal(i,0);
                SVM_Scalar::setdinternal(i,d);
	    }
	}

	Q.scale(a,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    }

    else if ( a < 1.0 )
    {
	isStateOpt = 0;

	// Free alphas at bounds.

        if ( SVM_Scalar::N() )
	{
            for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
	    {
		if ( alphaState()(i) == -2 )
		{
		    Q.modAlphaLBtoLF(Q.findInAlphaLB(i),*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
		}

		else if ( alphaState()(i) == +2 )
		{
		    Q.modAlphaUBtoUF(Q.findInAlphaUB(i),*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
		}
	    }
	}

	// scale alpha and b

	Q.scale(a,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    }

    SVM_Generic::basescalealpha(a);
    SVM_Generic::basescalebias(a);

    return 1;
}

int SVM_Scalar::reset(void)
{
    Q.reset(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    SVM_Generic::basesetAlphaBiasFromAlphaBiasR();

    isStateOpt = 0;

    return 1;
}

void SVM_Scalar::setGp(Matrix<double> *extGp, Matrix<double> *extGpsigma, Matrix<double> *extxy, int refactsol)
{
    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;
    }

    if ( Gplocal )
    {
	if ( extGp != NULL )
	{
            NiceAssert( extGpsigma != NULL );

	    MEMDEL(xyval);
	    xyval = NULL;

	    MEMDEL(Gpval);
	    Gpval = NULL;

	    MEMDEL(Gpsigma);
	    Gpsigma = NULL;

	    Gplocal = 0;

            xyval   = extxy;
            Gpval   = extGp;
	    Gpsigma = extGpsigma;
	}
    }

    else
    {
        if ( extGp != NULL )
	{
            NiceAssert( extGpsigma != NULL );

	    Gplocal = 0;

	    xyval   = extxy;
	    Gpval   = extGp;
            Gpsigma = extGpsigma;
	}

	else
	{
            NiceAssert( extGpsigma == NULL );

	    Gplocal = 1;

            MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &xycache   ,trainclass.size(),trainclass.size()));
            MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,trainclass.size(),trainclass.size()));
            MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,trainclass.size(),trainclass.size()));
	}
    }

    if ( refactsol )
    {
	Q.refact(*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    }

    return;
}

int SVM_Scalar::setLinearCost(void)
{
    int res = 0;

    if ( isLinearCost() )
    {
        ;
    }

    else if ( isQuadraticCost() )
    {
        res = 1;

        if ( SVM_Scalar::N() )
	{
	    isStateOpt = 0;
	}

	costType = 0;
	recalcdiagoff(-1);
	recalcLUB(-1);
    }

    else if ( is1NormCost() )
    {
        res = 1;

        if ( SVM_Scalar::N() )
	{
	    isStateOpt = 0;
	}

	costType = 0;
        recalcCRDR(-1);
	recalcLUB(-1);
    }

    return res;
}

int SVM_Scalar::setQuadraticCost(void)
{
    int res = 0;

    if ( isLinearCost() )
    {
        res = 1;

        if ( SVM_Scalar::N() )
	{
	    isStateOpt = 0;
	}

	int i;

	costType = 1;

	recalcdiagoff(-1);

	if ( NLB() )
	{
	    for ( i = NLB()-1 ; i >= 0  ; i-- )
	    {
		Q.modAlphaLBtoLF(i,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

	    }
	}

	if ( NUB() )
	{
	    for ( i = NUB()-1 ; i >= 0 ; i-- )
	    {
		Q.modAlphaUBtoUF(i,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

	    }
	}

	recalcLUB(-1);
    }

    else if ( isQuadraticCost() )
    {
        ;
    }

    else if ( is1NormCost() )
    {
        res = 1;

        if ( SVM_Scalar::N() )
	{
	    isStateOpt = 0;
	}

        costType = 1;
        recalcCRDR(-1);
        recalcdiagoff(-1);
    }

    return res;
}

int SVM_Scalar::set1NormCost(void)
{
    int res = 0;

    if ( isLinearCost() )
    {
        res = 1;

        if ( SVM_Scalar::N() )
	{
	    isStateOpt = 0;
	}

	int i;

        costType = 2;

        recalcCRDR(-1);

	if ( NLB() )
	{
	    for ( i = NLB()-1 ; i >= 0  ; i-- )
	    {
		Q.modAlphaLBtoLF(i,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

	    }
	}

	if ( NUB() )
	{
	    for ( i = NUB()-1 ; i >= 0 ; i-- )
	    {
		Q.modAlphaUBtoUF(i,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

	    }
	}

        recalcLUB(-1);
    }

    else if ( isQuadraticCost() )
    {
        res = 1;

        if ( SVM_Scalar::N() )
	{
	    isStateOpt = 0;
	}

        costType = 2;
        recalcdiagoff(-1);
        recalcCRDR(-1);
    }

    else if ( is1NormCost() )
    {
        ;
    }

    return res;
}

int SVM_Scalar::setVarBias(void)
{
    if ( !isVarBias() )
    {
	isStateOpt = 0;

	int wasfixedbias = isFixedBias();

        biasType = 0;

	int i;

	if ( GPNWIDTH(biasdim) )
	{
	    for ( i = 0 ; i < GPNWIDTH(biasdim) ; i++ )
	    {
		Q.changeBetaRestrict(i,0,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
		Q.modBetaCtoF(i,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

		if ( wasfixedbias )
		{
		    Q.betaStep(i,bfixval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
		}
	    }
	}

	if ( wasfixedbias )
	{
            Vector<double> gpnew(traintarg);

	    gpnew.negate();
	    Q.refactgp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gpnew,gn,hp);
	    gp = gpnew;

	    bfixval = 0.0;
	}

	if ( isOptActive() )
	{
	    setOptActive();
	}

	else if ( isOptD2C() )
	{
            setOptD2C();
	}

	else if ( isOptSMO() )
	{
            setOptSMO();
	}

        else
        {
            setOptGrad();
        }

        SVM_Generic::basesetbias(biasR());
    }

    return 1;
}

int SVM_Scalar::setFixedBias(double newbias)
{
    isStateOpt = 0;

    bfixval = newbias;

    if ( !isFixedBias() )
    {
        biasType = 3;

	int i;

	if ( GPNWIDTH(biasdim) )
	{
	    for ( i = 0 ; i < GPNWIDTH(biasdim) ; i++ )
	    {
		Q.changeBetaRestrict(i,3,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp); // this will zero it as well as constrain it
	    }
	}

	if ( isOptActive() )
	{
	    setOptActive();
	}

	else if ( isOptD2C() )
	{
            setOptD2C();
	}

	else if ( isOptSMO() )
	{
            setOptSMO();
	}

        else
        {
            setOptGrad();
        }
    }

    if ( SVM_Scalar::N() )
    {
        Vector<double> gpnew(traintarg);

	gpnew.negate();
	gpnew += bfixval;
	Q.refactgp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gpnew,gn,hp);
	gp = gpnew;
    }

    SVM_Generic::basesetbias(biasR());

    return 1;
}

int SVM_Scalar::setNoMonotonicConstraints(void)
{
    int res = 0;

    if ( makeConvex != 0 )
    {
        isStateOpt = 0;
        makeConvex = 0;

        res = 1;
    }

    return res;
}

int SVM_Scalar::setForcedMonotonicIncreasing(void)
{
    int res = 0;

    if ( makeConvex != 1 )
    {
        isStateOpt = 0;
        makeConvex = 1;

        res = 1;
    }

    return res;
}

int SVM_Scalar::setForcedMonotonicDecreasing(void)
{
    int res = 0;

    if ( makeConvex != 2 )
    {
        isStateOpt = 0;
        makeConvex = 2;

        res = 1;
    }

    return res;
}


int SVM_Scalar::setPosBias(void)
{
    if ( !isPosBias() )
    {
        if ( ( biasR() < 0 ) || isNegBias() )
	{
	    setFixedBias(0);
	}

	isStateOpt = 0;

	int wasfixedbias = isFixedBias();
        double newbias = biasR();

        biasType = 1;

	int i;

	if ( GPNWIDTH(biasdim) )
	{
	    for ( i = 0 ; i < GPNWIDTH(biasdim) ; i++ )
	    {
		Q.changeBetaRestrict(i,1,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
		Q.modBetaCtoF(i,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

		if ( wasfixedbias )
		{
		    Q.betaStep(i,newbias,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
		}
	    }
	}

	if ( wasfixedbias )
	{
            Vector<double> gpnew(traintarg);

	    gpnew.negate();
	    Q.refactgp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gpnew,gn,hp);
	    gp = gpnew;

	    bfixval = 0.0;
	}

	if ( isOptActive() )
	{
	    setOptActive();
	}

	else if ( isOptD2C() )
	{
            setOptD2C();
	}

	else if ( isOptSMO() )
	{
            setOptSMO();
	}

        else
        {
            setOptGrad();
        }

        SVM_Generic::basesetbias(biasR());
    }

    return 1;
}

int SVM_Scalar::setNegBias(void)
{
    if ( !isNegBias() )
    {
        if ( ( biasR() > 0 ) || isPosBias() )
	{
	    setFixedBias(0);
	}

	isStateOpt = 0;

	int wasfixedbias = isFixedBias();
        double newbias = biasR();

        biasType = 2;

	int i;

	if ( GPNWIDTH(biasdim) )
	{
	    for ( i = 0 ; i < GPNWIDTH(biasdim) ; i++ )
	    {
		Q.changeBetaRestrict(i,2,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
		Q.modBetaCtoF(i,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

		if ( wasfixedbias )
		{
		    Q.betaStep(i,newbias,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
		}
	    }
	}

	if ( wasfixedbias )
	{
            Vector<double> gpnew(traintarg);

	    gpnew.negate();
	    Q.refactgp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gpnew,gn,hp);
	    gp = gpnew;

	    bfixval = 0.0;
	}

	if ( isOptActive() )
	{
	    setOptActive();
	}

	else if ( isOptD2C() )
	{
            setOptD2C();
	}

	else if ( isOptSMO() )
	{
            setOptSMO();
	}

        else
        {
            setOptGrad();
        }

        SVM_Generic::basesetbias(biasR());
    }

    return 1;
}

int SVM_Scalar::setVarBias(int q)
{
    NiceAssert( q >= 0 );
    NiceAssert( q < GPNWIDTH(biasdim) );
    NiceAssert( !isFixedBias(q) );

    if ( !isVarBias(q) )
    {
	isStateOpt = 0;
        biasType = 0;

	Q.changeBetaRestrict(q,0,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
	Q.modBetaCtoF(q,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

	if ( isOptActive() )
	{
	    setOptActive();
	}

	else if ( isOptD2C() )
	{
            setOptD2C();
	}

	else if ( isOptSMO() )
	{
            setOptSMO();
	}

        else
        {
            setOptGrad();
        }

        SVM_Generic::basesetbias(biasR());
    }

    return 1;
}

int SVM_Scalar::setPosBias(int q)
{
    NiceAssert( q >= 0 );
    NiceAssert( q < GPNWIDTH(biasdim) );
    NiceAssert( !isFixedBias(q) );

    if ( !isPosBias(q) )
    {
	Vector<double> realbeta(Q.beta());

	if ( isNegBias(q) )
	{
            setVarBias(q);
	}

	if ( realbeta(q) < 0 )
	{
	    realbeta("&",q) = 0.0;
            setBiasVMulti(realbeta);
	}

	isStateOpt = 0;
        biasType = 1;

	Q.changeBetaRestrict(q,1,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
	Q.modBetaCtoF(q,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

	if ( isOptActive() )
	{
	    setOptActive();
	}

	else if ( isOptD2C() )
	{
            setOptD2C();
	}

	else if ( isOptSMO() )
	{
            setOptSMO();
	}

        else
        {
            setOptGrad();
        }

        SVM_Generic::basesetbias(biasR());
    }

    return 1;
}

int SVM_Scalar::setNegBias(int q)
{
    NiceAssert( q >= 0 );
    NiceAssert( q < GPNWIDTH(biasdim) );
    NiceAssert( !isFixedBias(q) );

    if ( !isNegBias(q) )
    {
	Vector<double> realbeta(Q.beta());

	if ( isPosBias(q) )
	{
            setVarBias(q);
	}

	if ( realbeta(q) > 0 )
	{
	    realbeta("&",q) = 0.0;
            setBiasVMulti(realbeta);
	}

	isStateOpt = 0;
        biasType = 2;

	Q.changeBetaRestrict(q,2,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
	Q.modBetaCtoF(q,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

	if ( isOptActive() )
	{
	    setOptActive();
	}

	else if ( isOptD2C() )
	{
            setOptD2C();
	}

	else if ( isOptSMO() )
	{
            setOptSMO();
	}

        else
        {
            setOptGrad();
        }

        SVM_Generic::basesetbias(biasR());
    }

    return 1;
}

int SVM_Scalar::setFixedBias(int q, double newbias)
{
    NiceAssert( q == 0 );
    (void) q;
    setFixedBias(newbias);
    SVM_Generic::basesetbias(biasR());

    return 1;
}

int SVM_Scalar::setFixedTube(void)
{
    if ( !isFixedTube() )
    {
	isStateOpt = 0;
	tubeshrink = 0;

	if ( isOptActive() )
	{
	    setOptActive();
	}

	else if ( isOptD2C() )
	{
	    setOptD2C();
	}

	else if ( isOptSMO() )
	{
	    setOptSMO();
	}

        else
        {
            setOptGrad();
        }
    }

    return 0;
}

int SVM_Scalar::setShrinkTube(void)
{
    if ( !isShrinkTube() )
    {
	isStateOpt = 0;
	tubeshrink = 1;

	if ( isOptActive() )
	{
	    setOptActive();
	}

	else if ( isOptD2C() )
	{
	    setOptD2C();
	}

	else if ( isOptSMO() )
	{
	    setOptSMO();
	}

        else
        {
            setOptGrad();
        }
    }

    return 0;
}

int SVM_Scalar::setRestrictEpsPos(void)
{
    epsrestrict = 1;

    return 0;
}

int SVM_Scalar::setRestrictEpsNeg(void)
{
    epsrestrict = 2;

    return 0;
}

int SVM_Scalar::setOptActive(void)
{
    if ( !isOptActive() )
    {
	optType = 0;
    }

    Q.setkeepfact(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),1);

    // Fix kernel cache memory ratios
    setmemsize(memsize());

    return 0;
}

int SVM_Scalar::setOptSMO(void)
{
    if ( !isOptSMO() )
    {
	optType = 1;
    }

    if ( isPosBias() || isNegBias() || isShrinkTube() )
    {
	Q.setkeepfact(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),1);
    }

    else
    {
	Q.setkeepfact(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),0);
    }

    // Fix kernel cache memory ratios
    setmemsize(memsize());

    return 0;
}

int SVM_Scalar::setOptD2C(void)
{
    if ( !isOptD2C() )
    {
	optType = 2;
    }

    if ( isPosBias() || isNegBias() || isShrinkTube() )
    {
	Q.setkeepfact(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),1);
    }

    else
    {
	Q.setkeepfact(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),0);
    }

    // Fix kernel cache memory ratios
    setmemsize(memsize());

    return 0;
}

int SVM_Scalar::setOptGrad(void)
{
    if ( !isOptGrad() )
    {
	optType = 3;
    }

    if ( isPosBias() || isNegBias() || isShrinkTube() )
    {
	Q.setkeepfact(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),1);
    }

    else
    {
	Q.setkeepfact(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),0);
    }

    // Fix kernel cache memory ratios
    setmemsize(memsize());

    return 0;
}

int SVM_Scalar::setLinBiasForce(double newval)
{
    isStateOpt = 0;
    linbiasforceval = newval;

    retVector<double> tmpva;

    if ( autosetLevel == 6 )
    {
        SVM_Scalar::autosetOff();
    }

    Vector<double> gnnew(gn);

    gnnew("&",0,1,GPNWIDTH(biasdim)-1,tmpva) = newval;
    Q.refactgn(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,gnnew,hp);
    gn("&",0,1,GPNWIDTH(biasdim)-1,tmpva) = newval;

    return 0;
}

int SVM_Scalar::setQuadBiasForce(double newval)
{
    isStateOpt = 0;
    quadbiasforceval = newval;

    int i;

    if ( GPNWIDTH(biasdim) )
    {
	for ( i = 0 ; i < GPNWIDTH(biasdim) ; i++ )
	{
	    Gn("&",i,i) = -newval;
	}

	Q.refact(*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    }

    return 0;
}

int SVM_Scalar::setOpttol(double xopttol)
{
    NiceAssert( xopttol >= 0 );

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;
    }

    opttolval = xopttol;

    Q.setopttol(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,opttolval);

    return 0;
}

int SVM_Scalar::setd(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Scalar::N() );
    NiceAssert( ( d == -1 ) || ( d == +1 ) || ( d == 0 ) || ( d == 2 ) );

    int res = 0;

    if ( d != trainclass(i) )
    {
	int oldd = trainclass(i);

        res |= SVM_Scalar::setdinternal(i,d);

	if ( !d || !oldd )
	{
            res |= SVM_Scalar::fixautosettings(0,1);
	}
    }

    return res;
}

int SVM_Scalar::setdinternal(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Scalar::N() );
    NiceAssert( ( d == -1 ) || ( d == +1 ) || ( d == 0 ) || ( d == 2 ) );

    int res = 0;

    if ( d != trainclass(i) )
    {
        res = 1;

        isStateOpt = 0;

	Nnc("&",trainclass(i)+1)--;
        Nnc("&",d+1)++;

	Q.changeAlphaRestrict(i,3,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp); // this will also zero alpha

	int alphrestrict = ALPHARESTRICT(d);
	trainclass("&",i) = d;

	Q.changeAlphaRestrict(i,alphrestrict,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    }

    return res;
}

int SVM_Scalar::sety(int i, const gentype &zn)
{
    NiceAssert( zn.isCastableToRealWithoutLoss() );

    return sety(i,(double) zn);
}

int SVM_Scalar::sety(const Vector<int> &j, const Vector<gentype> &yn)
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

int SVM_Scalar::sety(const Vector<gentype> &yn)
{
    NiceAssert( SVM_Scalar::N() == yn.size() );

    int res = 0;

    if ( SVM_Scalar::N() )
    {
        int i;

        for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int SVM_Scalar::sety(int i, double zn)
{
    NiceAssert( i >= -1 );
    NiceAssert( i < SVM_Scalar::N() );

    isStateOpt = 0;

    Vector<double> gpnew(gp);

    gentype zng(zn);
    int res = SVM_Generic::sety(i,zng);

    if ( i >= 0 )
    {
        traintarg("&",i) = zn;
        gpnew("&",i) = bfixval-zn;

	Q.refactgp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gpnew,gn,hp,i);

	gp("&",i) = gpnew(i);
    }

    else
    {
        traintarg = zn;
        gpnew = bfixval-zn;

	Q.refactgp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gpnew,gn,hp,i);

	gp = gpnew;
    }

    return res;
}

int SVM_Scalar::setCweight(int i, double xC)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Scalar::N() );
    NiceAssert( xC > 0 );

    isStateOpt = 0;

    Cweightval("&",i) = xC;

    if ( isLinearCost() )
    {
	recalcLUB(i);
    }

    else if ( isQuadraticCost() )
    {
	recalcdiagoff(i);
    }

    else if ( is1NormCost() )
    {
        recalcCRDR(i);
    }

    return 1;
}

int SVM_Scalar::setCweightfuzz(int i, double xC)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Scalar::N() );
    NiceAssert( xC > 0 );

    isStateOpt = 0;

    Cweightfuzzval("&",i) = xC;

    if ( isLinearCost() )
    {
	recalcLUB(i);
    }

    else if ( isQuadraticCost() )
    {
	recalcdiagoff(i);
    }

    else if ( is1NormCost() )
    {
        recalcCRDR(i);
    }

    return 1;
}

int SVM_Scalar::setepsweight(int i, double xepsweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Scalar::N() );

    isStateOpt = 0;

    epsweightval("&",i) = xepsweight;

    Vector<double> hpnew(hp);

    hpnew("&",i)   = HPCALC(epsval,xepsclass,trainclass(i),epsweightval(i));
    hpscale("&",i) = HPSCALECALC(xepsclass,trainclass(i),epsweightval(i));
    Q.refacthp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,hpnew,i);
    hp = hpnew;

    return 0;
}

int SVM_Scalar::setd(const Vector<int> &j, const Vector<int> &dd)
{
    NiceAssert( j.size() == dd.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= SVM_Scalar::setdinternal(j(i),dd(i));
	}

        res |= SVM_Scalar::fixautosettings(0,1);
    }

    return res;
}

int SVM_Scalar::sety(const Vector<int> &j, const Vector<double> &zn)
{
    NiceAssert( j.size() == zn.size() );

    Vector<gentype> zng(zn.size());
    for ( int k = 0 ; k < zn.size() ; k++ )
    {
        zng("&",k) = zn(k);
    }
    int res = SVM_Generic::sety(j,zng);

    if ( j.size() )
    {
        retVector<double> tmpva;
        retVector<double> tmpvb;

        isStateOpt = 0;

        Vector<double> gpnew(gp);

        traintarg("&",j,tmpva) = zn;

        gpnew("&",j,tmpva) = bfixval;
        gpnew("&",j,tmpva) -= zn;

        Q.refactgp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gpnew,gn,hp,-1);

        gp("&",j,tmpva) = gpnew(j,tmpvb);
    }

    return res;
}

int SVM_Scalar::setCweight(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( j.size() == xCweight.size() );

    if ( j.size() )
    {
        retVector<double> tmpva;

        Cweightval("&",j,tmpva) = xCweight;

	isStateOpt = 0;

        if ( isLinearCost() )
        {
            recalcLUB(-1);
        }

        else if ( isQuadraticCost() )
        {
            recalcdiagoff(-1);
        }

        else if ( is1NormCost() )
        {
            recalcCRDR(-1);
        }
    }

    return 1;
}

int SVM_Scalar::setepsweight(const Vector<int> &j, const Vector<double> &xepsweight)
{
    NiceAssert( j.size() == xepsweight.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= SVM_Scalar::setepsweight(j(i),xepsweight(i));
	}
    }

    return res;
}

int SVM_Scalar::setd(const Vector<int> &dd)
{
    NiceAssert( dd.size() == SVM_Scalar::N() );

    int i;
    int res = 0;

    if ( SVM_Scalar::N() )
    {
        for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
	{
            res |= SVM_Scalar::setdinternal(i,dd(i));
	}

        res |= SVM_Scalar::fixautosettings(0,1);
    }

    return res;
}

int SVM_Scalar::sety(const Vector<double> &zn)
{
    NiceAssert( zn.size() == SVM_Scalar::N() );

    isStateOpt = 0;

    Vector<double> gpnew(gp);

    Vector<gentype> zng(zn.size());
    for ( int k = 0 ; k < zn.size() ; k++ )
    {
        zng("&",k) = zn(k);
    }
    int res = SVM_Generic::sety(zng);

    traintarg = zn;

    gpnew = bfixval;
    gpnew -= zn;

    Q.refactgp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gpnew,gn,hp,-1);

    gp = gpnew;

    return res;
}

int SVM_Scalar::setyqnd(const Vector<double> &zn, Vector<double> &gpnew)
{
    NiceAssert( zn.size() == SVM_Scalar::N() );
    NiceAssert( gpnew.size() == SVM_Scalar::N() );

    isStateOpt = 0;

    traintarg = zn;

    gpnew = bfixval;
    gpnew -= zn;

    Q.refactgp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gpnew,gn,hp,-1);

    gp = gpnew;

    return 1;
}

int SVM_Scalar::setCweight(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == SVM_Scalar::N() );

    Cweightval = xCweight;

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;

        if ( isLinearCost() )
        {
            recalcLUB(-1);
        }

        else if ( isQuadraticCost() )
        {
            recalcdiagoff(-1);
        }

        else if ( is1NormCost() )
        {
            recalcCRDR(-1);
        }
    }

    return 1;
}

int SVM_Scalar::setCweightfuzz(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == SVM_Scalar::N() );

    int isdiff = ( Cweightfuzzval != xCweight );

    Cweightfuzzval = xCweight;

    if ( SVM_Scalar::N() && isdiff )
    {
	isStateOpt = 0;

        if ( isLinearCost() )
        {
            recalcLUB(-1);
        }

        else if ( isQuadraticCost() )
        {
            recalcdiagoff(-1);
        }

        else if ( is1NormCost() )
        {
            recalcCRDR(-1);
        }
    }

    return 1;
}

int SVM_Scalar::setCweightfuzz(const Vector<int> &i, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == i.size() );

    retVector<double> tmpva;
 
    Cweightfuzzval("&",i,tmpva) = xCweight;

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;

        if ( isLinearCost() )
        {
            recalcLUB(-1);
        }

        else if ( isQuadraticCost() )
        {
            recalcdiagoff(-1);
        }

        else if ( is1NormCost() )
        {
            recalcCRDR(-1);
        }
    }

    return 1;
}

int SVM_Scalar::setepsweight(const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == SVM_Scalar::N() );

    int i;

    int res = 0;

    if ( SVM_Scalar::N() )
    {
        for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
	{
            res |= SVM_Scalar::setepsweight(i,xepsweight(i));
	}
    }

    return res;
}

int SVM_Scalar::setC(double xC)
{
    NiceAssert( xC > 0 );

    int res = SVM_Scalar::autosetOff();

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;
    }

    if ( isLinearCost() )
    {
        res = 1;

        Q.scale(xC/CNval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
        lb *= xC/CNval;
        ub *= xC/CNval;
        cr *= xC/CNval;
        ddr *= xC/CNval;

        CNval = xC;

        SVM_Generic::basesetAlphaBiasFromAlphaBiasR();

	//recalcLUB(-1);
    }

    else if ( isQuadraticCost() )
    {
        CNval = xC;

	recalcdiagoff(-1);
    }

    else if ( is1NormCost() )
    {
        res = 1;

        CNval = xC;

        recalcCRDR(-1);
    }

    return res;
}

int SVM_Scalar::setCclass(int d, double xC)
{
    NiceAssert( ( d == -1 ) || ( d == 0 ) || ( d == +1 ) || ( d == 2 ) );
    NiceAssert( xC > 0 );

    if ( SVM_Scalar::NNC(d) )
    {
	isStateOpt = 0;
    }

    xCclass("&",d+1) = xC;

    if ( isLinearCost() )
    {
	recalcLUB(-1);
    }

    else if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else if ( is1NormCost() )
    {
        recalcCRDR(-1);
    }

    return 1;
}

int SVM_Scalar::seteps(double xeps)
{
    if ( autosetLevel == 6 )
    {
        SVM_Scalar::autosetOff();
    }

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;
    }

    SVM_Scalar::setalleps(xeps,xepsclass);

    return 0;
}

void SVM_Scalar::setalleps(double xeps, const Vector<double> &qepsclass)
{
    int i;

    epsval    = xeps;
    xepsclass = qepsclass;

    if ( SVM_Scalar::N() )
    {
	Vector<double> hpnew(hp);

        for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
	{
            hpnew("&",i)   = HPCALC(epsval,xepsclass,trainclass(i),epsweightval(i));
	    hpscale("&",i) = HPSCALECALC(xepsclass,trainclass(i),epsweightval(i));
	}

	Q.refacthp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,hpnew);

	hp = hpnew;
    }

    return;
}

int SVM_Scalar::setepsclass(int d, double xeps)
{
    NiceAssert( ( d == -1 ) || ( d == 0 ) || ( d == +1 ) || ( d == 2 ) );

    if ( SVM_Scalar::NNC(d) )
    {
	isStateOpt = 0;
    }

    Vector<double> qepsclass(xepsclass);

    qepsclass("&",d+1) = xeps;

    SVM_Scalar::setalleps(epsval,qepsclass);

    return 0;
}

int SVM_Scalar::autosetCscaled(double Cval)
{
    NiceAssert( Cval > 0 );
    autosetCvalx = Cval;
    int res = SVM_Scalar::setC( (SVM_Scalar::N()-SVM_Scalar::NNC(0)) ? (Cval/((SVM_Scalar::N()-SVM_Scalar::NNC(0)))) : 1.0);
    autosetLevel = 1;
    return res;
}

int SVM_Scalar::autosetCKmean(void)
{
    double diagsum = ( (SVM_Scalar::N()-SVM_Scalar::NNC(0)) ? SVM_Scalar::autosetkerndiagmean() : 1 );
    int res = SVM_Scalar::setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 );
    autosetLevel = 2;
    return res;
}

int SVM_Scalar::autosetCKmedian(void)
{
    double diagsum = ( (SVM_Scalar::N()-SVM_Scalar::NNC(0)) ? SVM_Scalar::autosetkerndiagmedian() : 1 );
    int res = SVM_Scalar::setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 );
    autosetLevel = 3;
    return res;
}

int SVM_Scalar::autosetCNKmean(void)
{
    double diagsum = ( (SVM_Scalar::N()-SVM_Scalar::NNC(0)) ? (SVM_Scalar::N()-SVM_Scalar::NNC(0))*SVM_Scalar::autosetkerndiagmean() : 1 );
    int res = SVM_Scalar::setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 );
    autosetLevel = 4;
    return res;
}

int SVM_Scalar::autosetCNKmedian(void)
{
    double diagsum = ( (SVM_Scalar::N()-SVM_Scalar::NNC(0)) ? (SVM_Scalar::N()-SVM_Scalar::NNC(0))*SVM_Scalar::autosetkerndiagmedian() : 1 );
    int res = SVM_Scalar::setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 );
    autosetLevel = 5;
    return res;
}

int SVM_Scalar::autosetLinBiasForce(double nuval, double Cval)
{
    NiceAssert( ( Cval > 0 ) && ( nuval >= 0.0 ) && ( nuval <= 1.0 ) );
    autosetnuvalx = nuval;
    autosetCvalx = Cval;
    int res = 0;
    res |= SVM_Scalar::setC( (SVM_Scalar::N()-SVM_Scalar::NNC(0)) ? (Cval/((SVM_Scalar::N()-SVM_Scalar::NNC(0))*nuval)) : 1.0);
    res |= SVM_Scalar::setLinBiasForce(-Cval);
    res |= SVM_Scalar::seteps(0.0);
    autosetLevel = 6;
    return res;
}

int SVM_Scalar::scaleCweight(double scalefactor)
{
    NiceAssert( scalefactor > 0 );

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;
    }

    Cweightval *= scalefactor;

    if ( isLinearCost() )
    {
	recalcLUB(-1);
    }

    else if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else if ( is1NormCost() )
    {
        recalcCRDR(-1);
    }

    return 1;
}

int SVM_Scalar::scaleCweightfuzz(double scalefactor)
{
    NiceAssert( scalefactor > 0 );

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;
    }

    Cweightfuzzval *= scalefactor;

    if ( isLinearCost() )
    {
	recalcLUB(-1);
    }

    else if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else if ( is1NormCost() )
    {
        recalcCRDR(-1);
    }

    return 1;
}

int SVM_Scalar::scaleepsweight(double scalefactor)
{
    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;
    }

    epsweightval *= scalefactor;

    Vector<double> hpnew(hp);

    hpnew   *= scalefactor;
    hpscale *= scalefactor;
    Q.refacthp(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,hpnew);
    hp = hpnew;

    return 0;
}

void SVM_Scalar::setmemsize(int memsize)
{
    xycache.setmemsize(MEMSHARE_XYCACHE(memsize)      ,xycache.get_min_rowdim()   );
    kerncache.setmemsize(MEMSHARE_KCACHE(memsize)     ,kerncache.get_min_rowdim() );
    sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize),sigmacache.get_min_rowdim());

    return;
}

int SVM_Scalar::setzerotol(double zt)
{
    isStateOpt = 0;

    Q.setzt(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),zt);

    return 0;
}

void innerCheatFn(double &res, const double &Kalt, void *carg);
void distCheatFn(double &res, const double &Kalt, void *carg);

void innerCheatFn(double &res, const double &Kalt, void *carg)
{
    MercerKernel &lockern = *((MercerKernel *) carg);

    lockern.reverseK(res,Kalt);

    return;
}

void distCheatFn(double &res, const double &Kalt, void *carg)
{
    MercerKernel &lockern = *((MercerKernel *) carg);

    lockern.reverseK(res,Kalt);

    return;
}

void SVM_Scalar::prepareKernel(void)
{
    // If any cheats (taking K(i,j) and getting either <x_i,x_j>+bias or ||x_i-x_j||^2 
    // directly) then set tell the kernel cache to calculate all of these for those
    // kernel evaluations that are in the cache and store them for use shortly, as
    // a kernel parameter is presumably about to change and trigger a recalc from
    // the relevant factor.

    if ( getKernel().isReversible() == 1 )
    {
        kerncache.setInnerCheat(&innerCheatFn,(void *) &(getKernel()));
    }

    else if ( getKernel().isReversible() == 2 )
    {
        kerncache.setDistCheat(&distCheatFn,(void *) &(getKernel()));
    }

    return;
}

int SVM_Scalar::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    NiceAssert( onlyChangeRowI >= -1 );
    NiceAssert( onlyChangeRowI < SVM_Scalar::N() );

    int res = 0;
    int fixxycache = getKernel().isIPdiffered();

    if ( !SVM_Scalar::N() )
    {
        res = SVM_Generic::resetKernel(modind,onlyChangeRowI,updateInfo);
    }

    else if ( SVM_Scalar::N() && ( onlyChangeRowI == -1 ) )
    {
fallbackMethod:
        res = SVM_Generic::resetKernel(modind,onlyChangeRowI,updateInfo);

        if ( fixxycache )
        {
            xycache.setSymmetry(1);
        }

        kerncache.setSymmetry(getKernel().getSymmetry());
        sigmacache.setSymmetry(1);

        res |= 1;

        int i;

	isStateOpt = 0;

        for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
	{
            K2(kerndiagval("&",i),i,i);
	}

        if ( fixxycache )
        {
            xycache.clear();
        }

        kerncache.clear();
        sigmacache.clear();

        Q.refact(*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

        res |= SVM_Scalar::fixautosettings(1,0);
    }

    else if ( onlyChangeRowI >= 0 )
    {
//        NiceAssert( updateInfo ); - I have no idea why this was here, or why the following line is required (if it even is)
updateInfo = 1;

        res |= 1;

        int needGradRecalc = 0;

        if ( alphaState()(onlyChangeRowI) )
        {
            isStateOpt = 0;

            // NB: - removing a row/column from the Cholesky factorisation 
            //       does not require reference to Gp() in 99% of cases -
            //       see chol.h.
            //     - it does however require reference to Gp() if alpha(I)
            //       is non-zero (I = onlyChangeRowI).
            //     - the change required to the alpha gradient is:
            //
            //       alphaGrad := alphaGrad - GpOld(I).alpha(I) + GpNew(I).alpha(I)
            //
            //       so we need to know GpOld(I).  If this row is in the
            //       kernel cache then we're golden, but if it is not then
            //       we can't do this (GpOld(I) is gone and can't be re-
            //       calculated here), so we need to revert to a fallback
            //       method.

            if ( Q.factbad(Gn,GPNorGPNEXT(Gpn,GpnExt)) )
            {
                goto fallbackMethod;
            }

            if ( !(kerncache.isRowInCache(onlyChangeRowI)) && ( abs2(alphaR()(onlyChangeRowI)) > zerotol() ) )
            {
                needGradRecalc = 1;
            }
        }

        int dstval = SVM_Scalar::d()(onlyChangeRowI);

        SVM_Scalar::setdinternal(onlyChangeRowI,0);

        res = SVM_Generic::resetKernel(modind,onlyChangeRowI);

        if ( fixxycache )
        {
            xycache.setSymmetry(1);
        }

        kerncache.setSymmetry(getKernel().getSymmetry());
        sigmacache.setSymmetry(1);

        K2(kerndiagval("&",onlyChangeRowI),onlyChangeRowI,onlyChangeRowI);

        if ( fixxycache )
        {
            xycache.recalc(onlyChangeRowI);
        }

        kerncache.recalc(onlyChangeRowI);
        sigmacache.recalc(onlyChangeRowI);

        SVM_Scalar::setdinternal(onlyChangeRowI,dstval);

        if ( needGradRecalc )
        {
            // Gradient is all wrong - fix it

            Q.refreshGrad_anyhow(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
        }

        else
        {
            // Gradient is part wrong - fix it

            Q.refreshGrad_anyhow(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,onlyChangeRowI);
        }

        res |= SVM_Scalar::fixautosettings(1,0);
    }

    getKernel_unsafe().setIPdiffered(0);

    return res;
}

int SVM_Scalar::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    NiceAssert( onlyChangeRowI >= -1 );
    NiceAssert( onlyChangeRowI < SVM_Scalar::N() );

    int res = 0;

    if ( !SVM_Scalar::N() )
    {
        res = SVM_Generic::setKernel(xkernel,modind,onlyChangeRowI);
    }

    else if ( SVM_Scalar::N() && ( onlyChangeRowI == -1 ) )
    {
        res = SVM_Generic::setKernel(xkernel,modind,onlyChangeRowI);

        xycache.setSymmetry(1);
        kerncache.setSymmetry(getKernel().getSymmetry());
        sigmacache.setSymmetry(1);

        res |= 1;

        int i;

	isStateOpt = 0;

        for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
	{
            K2(kerndiagval("&",i),i,i);
	}

        xycache.clear();
        kerncache.clear();
        sigmacache.clear();

        Q.refact(*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

        res |= SVM_Scalar::fixautosettings(1,0);
    }

    else if ( onlyChangeRowI >= 0 )
    {
        res |= 1;

        if ( alphaState()(onlyChangeRowI) )
        {
            isStateOpt = 0;
        }

        int dstval = SVM_Scalar::d()(onlyChangeRowI);

        SVM_Scalar::setdinternal(onlyChangeRowI,0);

        // DO THIS *NOW* AS WE NEED THE OLD Gp (AND THE OLD KERNEL WHICH IT MAY CALL IN setdinternal ABOVE) TO CORRECTLY PRE-ADJUST THE GRADIENTS!

        res = SVM_Generic::setKernel(xkernel,modind,onlyChangeRowI);

        xycache.setSymmetry(1);
        kerncache.setSymmetry(getKernel().getSymmetry());
        sigmacache.setSymmetry(1);

        K2(kerndiagval("&",onlyChangeRowI),onlyChangeRowI,onlyChangeRowI);

        xycache.recalc(onlyChangeRowI);
        kerncache.recalc(onlyChangeRowI);
        sigmacache.recalc(onlyChangeRowI);

        SVM_Scalar::setdinternal(onlyChangeRowI,dstval);

        // Gradient is part wrong - fix it

        Q.refreshGrad_anyhow(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,onlyChangeRowI);

        res |= SVM_Scalar::fixautosettings(1,0);
    }

    getKernel_unsafe().setIPdiffered(0);

    return res;
}

void SVM_Scalar::fillCache(void)
{
    // This overload is included to make sure the cache Gp is filled!

    if ( N() > 0 )
    {
        int i;
        retVector<double> tmpva;

        for ( i = 0 ; i < N() ; i++ )
        {
            (*Gpval)(i,tmpva);
        }
    }

    return;
}

int SVM_Scalar::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    // NB: GpnExt must be shrunk after calling this

    int res = 0;

    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Scalar::N() );

    if ( alphaState()(i) )
    {
        res = 1;
	isStateOpt = 0;
    }

    Nnc("&",trainclass(i)+1)--;

    if ( Q.alphaRestrict(i) != 3 )
    {
        Q.changeAlphaRestrict(i,3,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    }

    SVM_Generic::removeTrainingVector(i,y,x);

    traintarg.remove(i);
    trainclass.remove(i);
    Cweightval.remove(i);
    Cweightfuzzval.remove(i);
    epsweightval.remove(i);
    kerndiagval.remove(i);

    if ( Gplocal )
    {
	xyval->removeRowCol(i);
	Gpval->removeRowCol(i);
	Gpsigma->removeRowCol(i);
    }

    Gpn.removeRow(i);

    if ( GpnExt != NULL )
    {
	GpnExt->removeRow(i);
    }

    gp.remove(i);
    hp.remove(i);
    hpscale.remove(i);
    lb.remove(i);
    ub.remove(i);
    cr.remove(i);
    ddr.remove(i);
    wr.remove(i);
    diagoff.remove(i);
    xycache.remove(i);
    kerncache.remove(i);
    sigmacache.remove(i);

    Q.removeAlpha(i);

    // Fix the cache

    if ( ( kerncache.get_min_rowdim() >= (int) (SVM_Scalar::N()*ROWDIMSTEPRATIO) ) && ( SVM_Scalar::N() > MINROWDIM ) )
    {
        xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),SVM_Scalar::N()-1);
        kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),SVM_Scalar::N()-1);
        sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),SVM_Scalar::N()-1);
    }

    res |= SVM_Scalar::fixautosettings(0,1);

    return res;
}

int SVM_Scalar::addTrainingVector(int i, double zi, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    // NB: GpnExt must be extended before calling this.

    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Scalar::N() );
    NiceAssert( ( d == -1 ) || ( d == +1 ) || ( d == 0 ) || ( d == 2 ) );

    gentype zig(zi);
    int res = SVM_Generic::addTrainingVector(i,zig,x);

    isStateOpt = 0;

    Nnc("&",d+1)++;

    iskip = i;

    if ( kerncache.get_min_rowdim() <= SVM_Scalar::N() )
    {
        xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
        kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
        sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
    }

    iskip = -1;

    res |= qtaddTrainingVector(i,zi,Cweigh,epsweigh,d);
    res |= SVM_Scalar::fixautosettings(0,1);

    return res;
}

int SVM_Scalar::qaddTrainingVector(int i, double zi, SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    // NB: GpnExt must be extended before calling this.

    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Scalar::N() );
    NiceAssert( ( d == -1 ) || ( d == +1 ) || ( d == 0 ) || ( d == 2 ) );

    gentype zig(zi);
    int res = SVM_Generic::qaddTrainingVector(i,zig,x);

    isStateOpt = 0;

    Nnc("&",d+1)++;

    iskip = i;

    if ( kerncache.get_min_rowdim() <= SVM_Scalar::N() )
    {
        xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
        kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
        sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
    }

    iskip = -1;

    res |= qtaddTrainingVector(i,zi,Cweigh,epsweigh,d);
    res |= SVM_Scalar::fixautosettings(0,1);

    return res;
}

int SVM_Scalar::addTrainingVector(int i, const Vector<double> &zi, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    // NB: GpnExt must be extended before calling this.

    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Scalar::N() );
    NiceAssert( zi.size() == xx.size() );
    NiceAssert( zi.size() == Cweigh.size() );
    NiceAssert( zi.size() == epsweigh.size() );
    NiceAssert( zi.size() == d.size() );

    int res = 0;

    if ( zi.size() )
    {
        int j;

        for ( j = 0 ; j < zi.size() ; j++ )
        {
            gentype zig(zi(j));
            res |= SVM_Generic::addTrainingVector(i+j,zig,xx(j));

            isStateOpt = 0;

            Nnc("&",d(j)+1)++;

            iskip = i+j;

            if ( kerncache.get_min_rowdim() <= SVM_Scalar::N() )
            {
                xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
                kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
                sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
            }

            iskip = -1;

            res |= qtaddTrainingVector(i+j,zi(j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    res |= SVM_Scalar::fixautosettings(0,1);

    return res;
}

int SVM_Scalar::qaddTrainingVector(int i, const Vector<double> &zi, Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    // NB: GpnExt must be extended before calling this.

    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Scalar::N() );
    NiceAssert( zi.size() == xx.size() );
    NiceAssert( zi.size() == Cweigh.size() );
    NiceAssert( zi.size() == epsweigh.size() );
    NiceAssert( zi.size() == d.size() );

    int res = 0;

    if ( zi.size() )
    {
        int j;

        for ( j = 0 ; j < zi.size() ; j++ )
        {
            gentype zig(zi(j));
            res |= SVM_Generic::qaddTrainingVector(i+j,zig,xx("&",j));

            isStateOpt = 0;

            Nnc("&",d(j)+1)++;

            iskip = i+j;

            if ( kerncache.get_min_rowdim() <= SVM_Scalar::N() )
            {
                xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
                kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
                sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (SVM_Scalar::N()*ROWDIMSTEPRATIO));
            }

            iskip = -1;

            res |= qtaddTrainingVector(i+j,zi(j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    res |= SVM_Scalar::fixautosettings(0,1);

    return res;
}

int SVM_Scalar::qtaddTrainingVector(int i, double zi, double Cweigh, double epsweigh, int d)
{
    // NB: GpnExt must be extended before calling this.

    trainclass.add(i); trainclass("&",i) = d;
    traintarg.add(i);  traintarg("&",i) = zi;

    Cweightval.add(i);     Cweightval("&",i) = Cweigh;
    Cweightfuzzval.add(i); Cweightfuzzval("&",i) = 1.0;
    epsweightval.add(i);   epsweightval("&",i) = epsweigh;
    diagoff.add(i);        diagoff("&",i) = QUADCOSTDIAGOFFSET(CNval,xCclass,d,Cweigh,1.0);
    iskip = -2; // disable xymatrix to enable calculation of diagonal value
    kerndiagval.add(i);     
    {
        if ( diagkernvalcheat )
        {
            kerndiagval("&",i) = *diagkernvalcheat;
        }

        else
        {
            K2(kerndiagval("&",i),i,i);
        }
    }

    iskip = -1;

    if ( Gplocal )
    {
	xyval->addRowCol(i);
	Gpval->addRowCol(i);
	Gpsigma->addRowCol(i);
    }

    // Gpn row 1.0 iff not gradient or rank constraint
    // (gradient of constant is zero, rank constants cancel out)

    Gpn.addRow(i); Gpn("&",i,0) = xisrankorgrad(i) ? 0.0 : 1.0;

    gp.add(i);      gp("&",i)      = bfixval-zi;
    hp.add(i);      hp("&",i)      = HPCALC(epsval,xepsclass,d,epsweigh);
    hpscale.add(i); hpscale("&",i) = HPSCALECALC(xepsclass,d,epsweigh);
    lb.add(i);      lb("&",i)      = LCALC(CNval,xCclass,d,Cweigh,1.0);
    ub.add(i);      ub("&",i)      = HCALC(CNval,xCclass,d,Cweigh,1.0);
    cr.add(i);      cr("&",i)      = CRCALC(CNval,xCclass,d,Cweigh,1.0);
    ddr.add(i);     ddr("&",i)     = DRCALC(CNval,xCclass,d,Cweigh,1.0);
    wr.add(i);      wr("&",i)      = 1.0;

    xycache.add(i);
    kerncache.add(i);
    sigmacache.add(i);

    int alphrestrict = ALPHARESTRICT(d);

    Q.addAlpha(i,alphrestrict,0.0);
    Q.fixGrad(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

    SVM_Generic::basesetalpha(i,alphaR()(i));

    return 0;
}

int SVM_Scalar::addTrainingVector(int i, const gentype &zi, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_Scalar::addTrainingVector(i,(double) zi,x,Cweigh,epsweigh,2);
}

int SVM_Scalar::qaddTrainingVector(int i, const gentype &zi, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    int res = SVM_Scalar::qaddTrainingVector(i,(double) zi,x,Cweigh,epsweigh,2);

    return res;
}

int SVM_Scalar::addTrainingVector(int i, const Vector<gentype> &zi, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<double> zzi(zi.size());
    Vector<int> ddd(zi.size());

    ddd = 2;

    if ( zi.size() )
    {
        int j;

        for ( j = 0 ; j < zi.size() ; j++ )
        {
            zzi("&",j) = (double) zi(j);
        }
    }

    return SVM_Scalar::addTrainingVector(i,zzi,x,Cweigh,epsweigh,ddd);
}

int SVM_Scalar::qaddTrainingVector(int i, const Vector<gentype> &zi, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<double> zzi(zi.size());
    Vector<int> ddd(zi.size());

    ddd = 2;

    if ( zi.size() )
    {
        int j;

        for ( j = 0 ; j < zi.size() ; j++ )
        {
            zzi("&",j) = (double) zi(j);
        }
    }

    return SVM_Scalar::qaddTrainingVector(i,zzi,x,Cweigh,epsweigh,ddd);
}


double SVM_Scalar::quasiloglikelihood(void) const
{
    int i;
    double res = 0;

    // Close enough: res = -y.inv(Gp).y - log(det(Gp)) - n/2 log 2pi
    //                   = -y'.alpha - log(det(Gp)) - tspaceDim()/2 log 2.pi
    // (works for fixed-bias GP, not defined well at this level, but meh)

    if ( N() )
    {
        for ( i = 0 ; i < N() ; i++ )
        {
            res -= (double) (y()(i)*alpha()(i));
        }
    }

    // Directly applying Gp().det() appears to whack the stack, so we'll use
    // the Cholesky factorisation instead.

    res -= Q.fact_det();
    res -= NUMBASE_LN2PI*tspaceDim()/2.0;

    return res;
}













int SVM_Scalar::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int tempresh = 0;
    int unusedvar = 0;
    int dtv = xtang(i) & 7;

    if ( !( dtv & 4 ) )
    {
        tempresh = gTrainingVector(resg.force_double(),unusedvar,i,retaltg,pxyprodi);

        resh.force_double() = (double) resg;
    }

    else
    {
        int resbad = 0;

        if ( ( i >= 0 ) && ( emm != 2 ) )
        {
            // Check cache "just in case"

            resbad = Q.unAlphaGradIfPresent(resg.force_double(),i,*Gpval,GPNorGPNEXT(Gpn,GpnExt),gp,hp);
        }

        if ( ( i >= 0 ) && ( emm == 2 ) )
        {
            // Vector is in training set, can use optimised version

            Q.unAlphaGrad(resg.force_double(),i,*Gpval,GPNorGPNEXT(Gpn,GpnExt),gp,hp);

            resg += traintarg(i);
            resg -= diagoff(i)*alphaR()(i);
        }

        else if ( ( i >= 0 ) && !resbad )
        {
            resg += traintarg(i);
            resg -= diagoff(i)*alphaR()(i);
        }

        else if ( emm == 2 )
        {
            int firstval = 1;

            int jP;
            gentype Kxj;

            if ( NLB() )
            {
                for ( jP = 0 ; jP < NLB() ; jP++ )
                {
                    K2(Kxj,i,Q.pivAlphaLB()(jP),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaLB()(jP)] : NULL);

                    if ( firstval )
                    {
                        resg = Kxj*(alpha()(Q.pivAlphaLB()(jP)));

                        firstval = 0;
                    }

                    else
                    {
                        resg += Kxj*(alpha()(Q.pivAlphaLB()(jP)));
                    }
                }
            }

            if ( NUB() )
            {
                for ( jP = 0 ; jP < NUB() ; jP++ )
                {
                    K2(Kxj,i,Q.pivAlphaUB()(jP),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaUB()(jP)] : NULL);

                    if ( firstval )
                    {
                        resg = Kxj*(alpha()(Q.pivAlphaUB()(jP)));

                        firstval = 0;
                    }

                    else
                    {
                        resg += Kxj*(alpha()(Q.pivAlphaUB()(jP)));
                    }
                }
            }

            if ( NF() )
            {
                for ( jP = 0 ; jP < NF() ; jP++ )
                {
                    K2(Kxj,i,Q.pivAlphaF()(jP),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaF()(jP)] : NULL);

                    if ( firstval )
                    {
                        resg = Kxj*(alpha()(Q.pivAlphaF()(jP)));

                        firstval = 0;
                    }

                    else
                    {
                        resg += Kxj*(alpha()(Q.pivAlphaF()(jP)));
                    }
                }
            }
        }

        else
        {
            throw("gh Gradients undefined for n > 2");
        }

        tempresh = +1;
    }

    return tempresh;
}

int SVM_Scalar::gTrainingVector(double &res, int &unusedvar, int i, int raw, gentype ***pxyprodi) const
{
    (void) raw;

    int resbad = 0;

    if ( ( i >= 0 ) && ( emm != 2 ) )
    {
        // Check cache "just in case"

        res = 0;

        resbad = Q.unAlphaGradIfPresent(res,i,*Gpval,GPNorGPNEXT(Gpn,GpnExt),gp,hp);
    }

    if ( ( i >= 0 ) && ( emm == 2 ) )
    {
        // Vector is in training set, can use optimised version

        res = 0;

        Q.unAlphaGrad(res,i,*Gpval,GPNorGPNEXT(Gpn,GpnExt),gp,hp);

        res += traintarg(i);
        res -= diagoff(i)*alphaR()(i);
    }

    else if ( ( i >= 0 ) && !resbad )
    {
        res += traintarg(i);
        res -= diagoff(i)*alphaR()(i);
    }

    else if ( emm == 2 )
    {
        int dtv = 0;

        if ( ( dtv = xtang(i) & 7 ) )
        {
            res = 0.0;

            if ( dtv > 0 )
            {
                int jP;
                double Kxj;

                if ( NLB() )
                {
                    for ( jP = 0 ; jP < NLB() ; jP++ )
                    {
                        K2(Kxj,i,Q.pivAlphaLB()(jP),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaLB()(jP)] : NULL);
                        res += (alphaR()((Q.pivAlphaLB()(jP))))*Kxj;
                    }
                }

                if ( NUB() )
                {
                    for ( jP = 0 ; jP < NUB() ; jP++ )
                    {
                        K2(Kxj,i,Q.pivAlphaUB()(jP),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaUB()(jP)] : NULL);
                        res += (alphaR()((Q.pivAlphaUB()(jP))))*Kxj;
                    }
                }

                if ( NF() )
                {
                    for ( jP = 0 ; jP < NF() ; jP++ )
                    {
                        K2(Kxj,i,Q.pivAlphaF()(jP),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaF()(jP)] : NULL);
                        res += (alphaR()((Q.pivAlphaF()(jP))))*Kxj;
                    }
                }
            }
        }

        else
        {
            res = biasR();

            int jP;
            double Kxj;

            if ( NLB() )
            {
                for ( jP = 0 ; jP < NLB() ; jP++ )
                {
                    K2(Kxj,i,Q.pivAlphaLB()(jP),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaLB()(jP)] : NULL);
                    res += (alphaR()((Q.pivAlphaLB()(jP))))*Kxj;
                }
            }

            if ( NUB() )
            {
                for ( jP = 0 ; jP < NUB() ; jP++ )
                {
                    K2(Kxj,i,Q.pivAlphaUB()(jP),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaUB()(jP)] : NULL);
                    res += (alphaR()((Q.pivAlphaUB()(jP))))*Kxj;
                }
            }

            if ( NF() )
            {
                for ( jP = 0 ; jP < NF() ; jP++ )
                {
                    K2(Kxj,i,Q.pivAlphaF()(jP),pxyprodi ? (const gentype **) pxyprodi[Q.pivAlphaF()(jP)] : NULL);
                    res += (alphaR()((Q.pivAlphaF()(jP))))*Kxj;
                }
            }
        }
    }

    else if ( emm == 4 )
    {
        int dtv = 0;

        if ( ( dtv = xtang(i) & 7 ) )
        {
            res = 0.0;

            if ( dtv > 0 )
            {
                int j,k,l;
                double Kxj;

                // Optimisation:
                //
                // sum_{j,k,l} alpha_j alpha_k alpha_l K(x,xj,xk,xl) = sum_{j<k<l} ( alpha_j alpha_k alpha_l + alpha_j alpha_l alpha_k + alpha_k alpha_j alpha_l + alpha_k alpha_l alpha_j + alpha_l alpha_j alpha_k + alpha_l alpha_k alpha_j ) K(x,xj,xk,xl)
                //                                                   + sum_{j<k}   ( alpha_j alpha_k alpha_k + alpha_k alpha_j alpha_k + alpha_k alpha_k alpha_j + alpha_k alpha_j alpha_j + alpha_j alpha_k alpha_j + alpha_j alpha_j alpha_k ) K(x,xj,xk,xk)
                //                                                   + sum_j alpha_j alpha_j alpha_j K(x,xj,xj,xj)

                if ( NS() )
                {
                    for ( j = 0 ; j < SVM_Scalar::N() ; j++ )
                    {
                        if ( alphaState()(j) )
                        {
                            for ( k = 0 ; k < j ; k++ )
                            {
                                if ( alphaState()(k) )
                                {
                                    for ( l = 0 ; l < k ; l++ )
                                    {
                                        if ( alphaState()(l) )
                                        {
                                            K4(Kxj,i,j,k,l);

                                            res += ( ((alphaR()(j))*(alphaR()(k))*(alphaR()(l))) + 
                                                     ((alphaR()(j))*(alphaR()(l))*(alphaR()(k))) + 
                                                     ((alphaR()(k))*(alphaR()(j))*(alphaR()(l))) + 
                                                     ((alphaR()(k))*(alphaR()(l))*(alphaR()(j))) + 
                                                     ((alphaR()(l))*(alphaR()(j))*(alphaR()(k))) + 
                                                     ((alphaR()(l))*(alphaR()(k))*(alphaR()(j)))    )*Kxj;
                                        }
			            }

                                    K4(Kxj,i,j,k,k);

                                    res += ( ((alphaR()(j))*(alphaR()(k))*(alphaR()(k))) +
                                             ((alphaR()(k))*(alphaR()(j))*(alphaR()(k))) +
                                             ((alphaR()(k))*(alphaR()(k))*(alphaR()(j)))   )*Kxj;

                                    K4(Kxj,i,j,j,k);

                                    res += ( ((alphaR()(k))*(alphaR()(j))*(alphaR()(j))) +
                                             ((alphaR()(j))*(alphaR()(k))*(alphaR()(j))) +
                                             ((alphaR()(j))*(alphaR()(j))*(alphaR()(k)))    )*Kxj;
			        }
		            }

                            K4(Kxj,i,j,j,j);

                            res += ( ((alphaR()(j))*(alphaR()(j))*(alphaR()(j))) )*Kxj;
                        }
		    }
	        }
            }
        }

        else
        {
            res = biasR();

            int j,k,l;
            double Kxj;

            // Optimisation:
            //
            // sum_{j,k,l} alpha_j alpha_k alpha_l K(x,xj,xk,xl) = sum_{j<k<l} ( alpha_j alpha_k alpha_l + alpha_j alpha_l alpha_k + alpha_k alpha_j alpha_l + alpha_k alpha_l alpha_j + alpha_l alpha_j alpha_k + alpha_l alpha_k alpha_j ) K(x,xj,xk,xl)
            //                                                   + sum_{j<k}   ( alpha_j alpha_k alpha_k + alpha_k alpha_j alpha_k + alpha_k alpha_k alpha_j + alpha_k alpha_j alpha_j + alpha_j alpha_k alpha_j + alpha_j alpha_j alpha_k ) K(x,xj,xk,xk)
            //                                                   + sum_j alpha_j alpha_j alpha_j K(x,xj,xj,xj)

            if ( NS() )
            {
                for ( j = 0 ; j < SVM_Scalar::N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < j ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < k ; l++ )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        K4(Kxj,i,j,k,l);

                                        res += ( ((alphaR()(j))*(alphaR()(k))*(alphaR()(l))) + 
                                                 ((alphaR()(j))*(alphaR()(l))*(alphaR()(k))) + 
                                                 ((alphaR()(k))*(alphaR()(j))*(alphaR()(l))) + 
                                                 ((alphaR()(k))*(alphaR()(l))*(alphaR()(j))) + 
                                                 ((alphaR()(l))*(alphaR()(j))*(alphaR()(k))) + 
                                                 ((alphaR()(l))*(alphaR()(k))*(alphaR()(j)))    )*Kxj;
                                    }
			        }

                                K4(Kxj,i,j,k,k);

                                res += ( ((alphaR()(j))*(alphaR()(k))*(alphaR()(k))) +
                                         ((alphaR()(k))*(alphaR()(j))*(alphaR()(k))) +
                                         ((alphaR()(k))*(alphaR()(k))*(alphaR()(j)))   )*Kxj;

                                K4(Kxj,i,j,j,k);

                                res += ( ((alphaR()(k))*(alphaR()(j))*(alphaR()(j))) +
                                         ((alphaR()(j))*(alphaR()(k))*(alphaR()(j))) +
                                         ((alphaR()(j))*(alphaR()(j))*(alphaR()(k)))    )*Kxj;
			    }
		        }

                        K4(Kxj,i,j,j,j);

                        res += ( ((alphaR()(j))*(alphaR()(j))*(alphaR()(j))) )*Kxj;
                    }
		}
	    }
	}
    }

    else
    {
        int dtv = 0;

        if ( ( dtv = xtang(i) & 7 ) )
        {
            res = 0.0;

            if ( dtv > 0 )
            {
                Vector<int> j(emm);
                int jj;
                int z = 0;
                double tres;

                retVector<int>    tmpva;
                retVector<double> tmpvb;

                j("&",z) = i;
                j("&",1,1,emm-1,tmpva) = z;

                int isdone = 0;

                while ( !isdone )
                {
                    if ( prod(alphaState()(j,tmpva)) )
                    {
                        tres  = Km(emm,tres,j);
                        tres *= prod(alphaR()(j(1,1,emm-1,tmpva),tmpvb));

                        res += tres;
                    }

                    isdone = 1;
                    jj = 1;

                    while ( isdone && ( jj < emm ) )
                    {
                        j("&",jj)++;

                        if ( j(jj) < SVM_Scalar::N() )
                        {
                            isdone = 0;
                            break;
                        }

                        else
                        {
                            j("&",jj) = 0;
                            jj++;
                        }
                    }
                }
            }
        }

        else
        {
            res = biasR();

            Vector<int> j(emm);
            int jj;
            int z = 0;
            double tres;

            retVector<int>    tmpva;
            retVector<double> tmpvb;

            j("&",z) = i;
            j("&",1,1,emm-1,tmpva) = z;

            int isdone = 0;

            while ( !isdone )
            {
                if ( prod(alphaState()(j,tmpva)) )
                {
                    tres  = Km(emm,tres,j);
                    tres *= prod(alphaR()(j(1,1,emm-1,tmpva),tmpvb));

                    res += tres;
                }

                isdone = 1;
                jj = 1;

                while ( isdone && ( jj < emm ) )
                {
                    j("&",jj)++;

                    if ( j(jj) < SVM_Scalar::N() )
                    {
                        isdone = 0;
                        break;
                    }

                    else
                    {
                        j("&",jj) = 0;
                        jj++;
                    }
                }
            }
        }
    }

    return ( unusedvar = ( res > 0 ) ? +1 : -1 );
}

void SVM_Scalar::fastg(double &res) const
{
    if ( emm == 2 )
    {
        int jP,k;
        double Kxj;

        res = biasR();

        if ( NLB() )
        {
            for ( jP = 0 ; jP < NLB() ; jP++ )
            {
                k = Q.pivAlphaLB()(jP);

                K1(Kxj,k,NULL,&(x()(k)),&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }

        if ( NUB() )
        {
            for ( jP = 0 ; jP < NUB() ; jP++ )
            {
                k = Q.pivAlphaUB()(jP);

                K1(Kxj,k,NULL,&(x()(k)),&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }

        if ( NF() )
        {
            for ( jP = 0 ; jP < NF() ; jP++ )
            {
                k = Q.pivAlphaF()(jP);

                K1(Kxj,k,NULL,&(x()(k)),&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }
    }

    else
    {
        SVM_Generic::fastg(res);
    }

    return;
}


void SVM_Scalar::fastg(double &res, 
                       int ia, int ib,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                       const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    if ( emm == 2 )
    {
        if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
        if ( ib < 0 ) { setInnerWildpa(&xb,&xbinfo); ib = -2; }

        int jP,k;
        double Kxj;

        res = biasR();

        if ( NLB() )
        {
            for ( jP = 0 ; jP < NLB() ; jP++ )
            {
                k = Q.pivAlphaLB()(jP);

                K3(Kxj,ia,ib,k,NULL,&xa,&xb,&(x()(k)),&xainfo,&xbinfo,&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }

        if ( NUB() )
        {
            for ( jP = 0 ; jP < NUB() ; jP++ )
            {
                k = Q.pivAlphaUB()(jP);

                K3(Kxj,ia,ib,k,NULL,&xa,&xb,&(x()(k)),&xainfo,&xbinfo,&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }

        if ( NF() )
        {
            for ( jP = 0 ; jP < NF() ; jP++ )
            {
                k = Q.pivAlphaF()(jP);

                K3(Kxj,ia,ib,k,NULL,&xa,&xb,&(x()(k)),&xainfo,&xbinfo,&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }

        if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }
    }

    else
    {
        SVM_Generic::fastg(res,ia,ib,xa,xb,xainfo,xbinfo);
    }

    return;
}

void SVM_Scalar::fastg(gentype &res) const
{
    if ( emm == 2 )
    {
        int jP,k;
        double Kxj;

        res = biasR();

        if ( NLB() )
        {
            for ( jP = 0 ; jP < NLB() ; jP++ )
            {
                k = Q.pivAlphaLB()(jP);

                K1(Kxj,k,NULL,&(x()(k)),&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }

        if ( NUB() )
        {
            for ( jP = 0 ; jP < NUB() ; jP++ )
            {
                k = Q.pivAlphaUB()(jP);

                K1(Kxj,k,NULL,&(x()(k)),&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }

        if ( NF() )
        {
            for ( jP = 0 ; jP < NF() ; jP++ )
            {
                k = Q.pivAlphaF()(jP);

                K1(Kxj,k,NULL,&(x()(k)),&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }
    }

    else
    {
        SVM_Generic::fastg(res);
    }

    return;
}

void SVM_Scalar::fastg(gentype &res, 
                       int ia, int ib,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                       const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    if ( emm == 2 )
    {
        if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
        if ( ib < 0 ) { setInnerWildpa(&xb,&xbinfo); ib = -2; }

        int jP,k;
        double Kxj;

        res = biasR();

        if ( NLB() )
        {
            for ( jP = 0 ; jP < NLB() ; jP++ )
            {
                k = Q.pivAlphaLB()(jP);

                K3(Kxj,ia,ib,k,NULL,&xa,&xb,&(x()(k)),&xainfo,&xbinfo,&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }

        if ( NUB() )
        {
            for ( jP = 0 ; jP < NUB() ; jP++ )
            {
                k = Q.pivAlphaUB()(jP);

                K3(Kxj,ia,ib,k,NULL,&xa,&xb,&(x()(k)),&xainfo,&xbinfo,&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }

        if ( NF() )
        {
            for ( jP = 0 ; jP < NF() ; jP++ )
            {
                k = Q.pivAlphaF()(jP);

                K3(Kxj,ia,ib,k,NULL,&xa,&xb,&(x()(k)),&xainfo,&xbinfo,&(xinfo()(k)));

                res += (alphaR()(k))*Kxj;
            }
        }

        if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }
    }

    else
    {
        SVM_Generic::fastg(res,ia,ib,xa,xb,xainfo,xbinfo);
    }

    return;
}

int SVM_Scalar::covTrainingVector(gentype &resv, gentype &resmu, int ia, int ib, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
    int NN = SVM_Scalar::N();

    //NiceAssert( isFixedBias() );
    NiceAssert( NS() == NF() );
    NiceAssert( ( NF() == 0 ) || ( NF() == SVM_Scalar::N()-NNC(0) ) );
    NiceAssert( ( ia >= -3 ) && ( ib != -2 ) );
    NiceAssert( ia < NN );
    NiceAssert( ( ib >= -3 ) && ( ib != -2 ) );
    NiceAssert( ib < NN );
    NiceAssert( emm == 2 );

    // We know that, for the fixed bias LS-SVM:
    //
    // var = K(ia,ib) - K(ia)'.inv(Gp).K(ib)
    //
    // where Gp = Kp + diag(sigma) (though it has a different name here), and
    // can assume the existence of the factorisation of Gp.  In this case
    // we simply calculate the second part for the free part of K(i) and
    // and leave it at that (which will work if this is an LS-SVM and has
    // been trained).
    //
    // As per Bull, Convergence rates of efficient global optimization algorithms,
    // in the variable bias case the prediction is precisely that of the LS-SVR,
    // and moreover the variance can be shown to simply be:
    //
    // K(ia,ib) - [ K(ia) ]' inv([ Gp 1 ]) [ K(ib) ]
    //            [  1    ]      [ 1  0 ]  [  1    ]
    //
    // = K(ia,ib) - K(ia)'.inv(Gp).K(bi) + ( 1 - 1'.inv(Gp).K(ia) )/( 1'.inv(Gp).1 )
    //
    // ... _ 1'.inv(Gp).q ( 1 - bias )
    //
    // and we can mostly use the same code for both.
    //
    // If Ns = 0 then the variable bias case is ill-defined

    // For the non-LS-SVM case (ie regular epsilon-insensitive)
    // then see:
    //
    // Gao, Gunn, Harris - A Probabilistic Framework for SVM Regression and Error Bar Estimation

    int dtva = xtang(ia) & 7;
    int dtvb = xtang(ib) & 7;

    NiceAssert( dtva >= 0 );
    NiceAssert( dtvb >= 0 );

    // This is used elsewhere (ie not scalar), so the following is relevant

//FIXME: resmu
    if ( ( dtva & 4 ) || ( dtvb & 4 ) || !isUnderlyingScalar() )
    {
        if ( NS() )
        {
            int j;

            Vector<gentype> Kia(NN);
            Vector<gentype> Kib(NN);
            Vector<gentype> itsone(1);//isVarBias() ? 1 : 0);
            gentype Kii;

            itsone("&",zeroint()) = 1.0;

            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                retVector<double> tmpva;

                Kii  = Gp()(ia,ib);
                Kii -= ( ia == ib ) ? diagoff(tmpva)(ia) : 0.0;
            }

            else
            {
                K2(Kii,ia,ib,(const gentype **) pxyprodij);
            }

            if ( ia >= 0 )
            {
                for ( j = 0 ; j < NN ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        Kia("&",j) = Gp()(ia,j);
                    }

                    else
                    {
                        Kia("&",j) = 0.0;
                    }
                }

                retVector<double> tmpva;

                Kia("&",ia) -= diagoff(tmpva)(ia);
            }

            else
            {
                for ( j = 0 ; j < NN ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        K2(Kia("&",j),ia,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                    }

                    else
                    {
                        Kia("&",j) = 0.0;
                    }
                }
            }

            if ( ib == ia )
            {
                Kib = Kia;
            }

            else if ( ib >= 0 )
            {
                for ( j = 0 ; j < NN ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        Kib("&",j) = Gp()(j,ib);
                    }

                    else
                    {
                        Kib("&",j) = 0.0;
                    }
                }

                retVector<double> tmpva;

                Kib("&",ib) -= diagoff(tmpva)(ib);
            }

            else
            {
                for ( j = 0 ; j < NN ; j++ )
                {
                    if ( alphaState()(j) || ( ia == j ) || ( ib == j ) )
                    {
                        //K2(Kib("&",j),j,ib,pxyprodj ? (const gentype **) pxyprodj[j] : NULL); - reversed in line with assumptions in Kxfer (unknown "x" comes first)
                        K2(Kib("&",j),ib,j,pxyprodj ? (const gentype **) pxyprodj[j] : NULL);
                        setconj(Kib("&",j));
                    }

                    else
                    {
                        Kib("&",j) = 0.0;
                    }
                }
            }

            Vector<gentype> btemp(1);//isVarBias() ? 1 : 0);
            Vector<gentype> Kres(NN);

            //NB: this will automatically only do part corresponding to pivAlphaF
            fact_minverse(Kres,btemp,Kib,itsone);

            resv = Kii;

            for ( j = 0 ; j < pivAlphaF().size() ; j++ )
            {
                resv -= outerProd(Kia(pivAlphaF()(j)),Kres(pivAlphaF()(j)));
            }

            if ( isVarBias() )
            {
                // This is the additional corrective factor

                resv -= btemp(zeroint());
            }

            // mu calculation

            int firstterm = 1;

            for ( j = 0 ; j < pivAlphaF().size() ; j++ )
            {
                if ( firstterm )
                {
                    resmu = Kia(pivAlphaF()(j))*alpha()(pivAlphaF()(j));

                    firstterm = 0;
                }

                else
                {
                    resmu += Kia(pivAlphaF()(j))*alpha()(pivAlphaF()(j));
                }
            }

            for ( j = 0 ; j < pivAlphaLB().size() ; j++ )
            {
                if ( firstterm )
                {
                    resmu = Kia(pivAlphaLB()(j))*alpha()(pivAlphaLB()(j));

                    firstterm = 0;
                }

                else
                {
                    resmu += Kia(pivAlphaLB()(j))*alpha()(pivAlphaLB()(j));
                }
            }

            for ( j = 0 ; j < pivAlphaUB().size() ; j++ )
            {
                if ( firstterm )
                {
                    resmu = Kia(pivAlphaUB()(j))*alpha()(pivAlphaUB()(j));

                    firstterm = 0;
                }

                else
                {
                    resmu += Kia(pivAlphaUB()(j))*alpha()(pivAlphaUB()(j));
                }
            }

            if ( !( dtva & 7 ) )
            {
                if ( firstterm )
                {
                    resmu = bias();

                    firstterm = 0;
                }

                else
                {
                    resmu += bias();
                }
            }

            else
            {
                if ( firstterm )
                {
                    resmu =  bias();
                    resmu *= 0.0;

                    firstterm = 0;
                }
            }
        }

        else
        {
            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                retVector<double> tmpva;

                resv  = Gp()(ia,ib);
                resv -= ( ia == ib ) ? diagoff(tmpva)(ia) : 0.0;
            }

            else
            {
                K2(resv,ia,ib,(const gentype **) pxyprodij);
            }

            if ( !( dtva & 7 ) )
            {
                resmu = bias();
            }

            else
            {
                resmu  = bias();
                resmu *= 0.0;
            }
        }

        // Additional terms: if the kernel itself is a random process then we need to 
        // allow for variance from this.  This extra term is:
        //
        // variance_k(g(x))
        //
        // which can be calculated using resmode = 0x80

        if ( getKernel().isKVarianceNZ() )
        {
            NiceAssert( ia == ib );

            gentype addres(0.0);
            gentype Kxj;
            int jP;

            if ( NLB() )
            {
                for ( jP = 0 ; jP < NLB() ; jP++ )
                {
                    K2(Kxj,ia,Q.pivAlphaLB()(jP),NULL,NULL,NULL,NULL,NULL,0x80);
                    addres += (alphaR()((Q.pivAlphaLB()(jP))))*(alphaR()((Q.pivAlphaLB()(jP))))*Kxj;
                }
            }

            if ( NUB() )
            {
                for ( jP = 0 ; jP < NUB() ; jP++ )
                {
                    K2(Kxj,ia,Q.pivAlphaUB()(jP),NULL,NULL,NULL,NULL,NULL,0x80);
                    addres += (alphaR()((Q.pivAlphaUB()(jP))))*(alphaR()((Q.pivAlphaUB()(jP))))*Kxj;
                }
            }

            if ( NF() )
            {
                for ( jP = 0 ; jP < NF() ; jP++ )
                {
                    K2(Kxj,ia,Q.pivAlphaF()(jP),NULL,NULL,NULL,NULL,NULL,0x80);
                    addres += (alphaR()((Q.pivAlphaF()(jP))))*(alphaR()((Q.pivAlphaF()(jP))))*Kxj;
                }
            }

            resv += addres;
        }
    }

    else
    {
        double &resvv = resv.force_double();
        double &resgg = resmu.force_double();

        if ( dtva & 7 )
        {
            resgg = 0.0;
        }

        else
        {
            resgg = biasR();
        }

        if ( NS() )
        {
            int j;

            Vector<double> Kia(NN);
            Vector<double> Kib(NN);
            Vector<double> itsone(1);//isVarBias() ? 1 : 0);
            double Kii;

            itsone("&",zeroint()) = 1.0;

            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                retVector<double> tmpva;

                Kii  = Gp()(ia,ib);
                Kii -= ( ia == ib ) ? diagoff(tmpva)(ia) : 0.0;
            }

            else
            {
                K2(Kii,ia,ib,(const gentype **) pxyprodij);
            }

            if ( ia >= 0 )
            {
                for ( j = 0 ; j < NN ; j++ )
                {
                    Kia("&",j) = Gp()(ia,j);
                }

                retVector<double> tmpva;

                Kia("&",ia) -= diagoff(tmpva)(ia);
            }

            else
            {
                for ( j = 0 ; j < NN ; j++ )
                {
                    K2(Kia("&",j),ia,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                }
            }

            if ( ib == ia )
            {
                Kib = Kia;
            }

            else if ( ib >= 0 )
            {
                for ( j = 0 ; j < NN ; j++ )
                {
                    Kib("&",j) = Gp()(j,ib);
                }

                retVector<double> tmpva;

                Kib("&",ib) -= diagoff(tmpva)(ib);
            }

            else
            {
                for ( j = 0 ; j < NN ; j++ )
                {
                    //K2(Kib("&",j),j,ib,pxyprodj ? (const gentype **) pxyprodj[j] : NULL); - see above
                    K2(Kib("&",j),ib,j,pxyprodj ? (const gentype **) pxyprodj[j] : NULL);
                }
            }

            Vector<double> btemp(1);//isVarBias() ? 1 : 0);
            Vector<double> Kres(NN);

            //NB: this will automatically only do part corresponding to pivAlphaF
            fact_minverse(Kres,btemp,Kib,itsone);

            resvv = Kii;

            for ( j = 0 ; j < pivAlphaF().size() ; j++ )
            {
                resvv -= Kia(pivAlphaF()(j))*Kres(pivAlphaF()(j));
            }

            if ( isVarBias() )
            {
                // This is the additional corrective factor

                resvv -= btemp(zeroint());
            }

            // mu calculation

            for ( j = 0 ; j < pivAlphaF().size() ; j++ )
            {
                resgg += Kia(pivAlphaF()(j))*alphaR()(pivAlphaF()(j));
            }

            for ( j = 0 ; j < pivAlphaLB().size() ; j++ )
            {
                resgg += Kia(pivAlphaLB()(j))*alphaR()(pivAlphaLB()(j));
            }

            for ( j = 0 ; j < pivAlphaUB().size() ; j++ )
            {
                resgg += Kia(pivAlphaUB()(j))*alphaR()(pivAlphaUB()(j));
            }
        }

        else
        {
            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                retVector<double> tmpva;

                resvv  = Gp()(ia,ib);
                resvv -= ( ia == ib ) ? diagoff(tmpva)(ia) : 0.0;
            }

            else
            {
                K2(resvv,ia,ib,(const gentype **) pxyprodij);
            }
        }

        // Additional terms: if the kernel itself is a random process then we need to 
        // allow for variance from this.  This extra term is:
        //
        // variance_k(g(x))
        //
        // which can be calculated using resmode = 0x80

        if ( getKernel().isKVarianceNZ() )
        {
            NiceAssert( ia == ib );

            double addres = 0.0;
            double Kxj;
            int jP;

            if ( NLB() )
            {
                for ( jP = 0 ; jP < NLB() ; jP++ )
                {
                    K2(Kxj,ia,Q.pivAlphaLB()(jP),NULL,NULL,NULL,NULL,NULL,0x80);
                    addres += (alphaR()((Q.pivAlphaLB()(jP))))*(alphaR()((Q.pivAlphaLB()(jP))))*Kxj;
                }
            }

            if ( NUB() )
            {
                for ( jP = 0 ; jP < NUB() ; jP++ )
                {
                    K2(Kxj,ia,Q.pivAlphaUB()(jP),NULL,NULL,NULL,NULL,NULL,0x80);
                    addres += (alphaR()((Q.pivAlphaUB()(jP))))*(alphaR()((Q.pivAlphaUB()(jP))))*Kxj;
                }
            }

            if ( NF() )
            {
                for ( jP = 0 ; jP < NF() ; jP++ )
                {
                    K2(Kxj,ia,Q.pivAlphaF()(jP),NULL,NULL,NULL,NULL,NULL,0x80);
                    addres += (alphaR()((Q.pivAlphaF()(jP))))*(alphaR()((Q.pivAlphaF()(jP))))*Kxj;
                }
            }

            resvv += addres;
        }
    }

    return 0;
}









void SVM_Scalar::recalcCRDR(int ival)
{
    // This updates the cr and cd
    //
    // If ival == -1 then all i is scanned.  Otherwise it only does a specific
    // value.

    NiceAssert( ival >= -1 );
    NiceAssert( ival < SVM_Scalar::N() );

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;

	int i,imin,imax;

	if ( ival == -1 )
	{
	    imin = 0;
            imax = SVM_Scalar::N();
	}

	else
	{
	    imin = ival;
	    imax = ival+1;
	}

	if ( imax > imin )
	{
	    for ( i = imin ; i < imax ; i++ )
	    {
                cr("&",i)  = CRCALC(CNval,xCclass,trainclass(i),Cweightval(i),Cweightfuzzval(i));
                ddr("&",i) = DRCALC(CNval,xCclass,trainclass(i),Cweightval(i),Cweightfuzzval(i));
	    }
	}
    }

    return;
}

void SVM_Scalar::recalcLUB(int ival)
{
    // This updates the ub (upper bound on alpha) and lb (lower bound on
    // alpha) vectors.
    //
    // If ival == -1 then all i is scanned.  Otherwise it only does a specific
    // value.

    NiceAssert( ival >= -1 );
    NiceAssert( ival < SVM_Scalar::N() );

    retMatrix<double> tmpma;
    retMatrix<double> tmpmb;

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;

	int i,imin,imax;

	if ( ival == -1 )
	{
	    imin = 0;
            imax = SVM_Scalar::N();
	}

	else
	{
	    imin = ival;
	    imax = ival+1;
	}

	if ( imax > imin )
	{
	    for ( i = imin ; i < imax ; i++ )
	    {
                lb("&",i) = LCALC(CNval,xCclass,trainclass(i),Cweightval(i),Cweightfuzzval(i));
                ub("&",i) = HCALC(CNval,xCclass,trainclass(i),Cweightval(i),Cweightfuzzval(i));

		if ( alphaState()(i) == -2 )
		{
                    if ( alphaR()(i)-lb(i) <= MAXPASSIVECHANGE )
		    {
                        Q.alphaStep(i,-alphaR()(i)+lb(i),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpma),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);

                        SVM_Generic::basesetalpha(i,alphaR()(i));
		    }

		    else
		    {
			Q.modAlphaLBtoLF(Q.findInAlphaLB(i),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpma),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpmb),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

                        SVM_Generic::basesetalpha(i,alphaR()(i));
		    }
		}

		else if ( alphaState()(i) == +2 )
		{
                    if ( ub(i)-alphaR()(i) <= MAXPASSIVECHANGE )
		    {
                        Q.alphaStep(i,-alphaR()(i)+ub(i),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpma),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);

                        SVM_Generic::basesetalpha(i,alphaR()(i));
		    }

		    else
		    {
			Q.modAlphaUBtoUF(Q.findInAlphaUB(i),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpma),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpmb),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

                        SVM_Generic::basesetalpha(i,alphaR()(i));
		    }
		}

		if ( alphaState()(i) == -1 )
		{
                    if ( alphaR()(i) < lb("&",i) )
		    {
                        Q.alphaStep(i,-alphaR()(i)+lb(i),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpma),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);

                        SVM_Generic::basesetalpha(i,alphaR()(i));
		    }
		}

		else if ( alphaState()(i) == +1 )
		{
                    if ( alphaR()(i) > ub("&",i) )
		    {
                        Q.alphaStep(i,-alphaR()(i)+ub(i),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpma),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);

                        SVM_Generic::basesetalpha(i,alphaR()(i));
		    }
		}
	    }
	}
    }

    return;
}

void SVM_Scalar::recalcdiagoff(int i)
{
    NiceAssert( i >= -1 );
    NiceAssert( i < SVM_Scalar::N() );

    // This updates the diagonal offsets.

    retMatrix<double> tmpma;
    retMatrix<double> tmpmb;

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;

	if ( i == -1 )
	{
            Vector<double> bp(SVM_Scalar::N());
	    Vector<double> bn(GPNWIDTH(biasdim));

	    bn.zero();

            for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
	    {
		bp("&",i) = -diagoff(i);
                diagoff("&",i) = QUADCOSTDIAGOFFSET(CNval,xCclass,trainclass(i),Cweightval(i),Cweightfuzzval(i));
		bp("&",i) += diagoff(i);
	    }

	    kerncache.recalcDiag();

            sigmacache.clear();

            //sigmacache.reset(SVM_Scalar::N(),&evalSigmaSVM_Scalar,(void *) this);
            //sigmacache.setmemsize(MEMSHARE_SIGMACACHE(kerncache.get_memsize()),kerncache.get_min_rowdim());

	    Q.diagoffset(bp,bn,bp,bn,(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpma),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpmb),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
	}

	else
	{
	    double bpoff = 0.0;

	    bpoff = -diagoff(i);
            diagoff("&",i) = QUADCOSTDIAGOFFSET(CNval,xCclass,trainclass(i),Cweightval(i),Cweightfuzzval(i));
	    bpoff += diagoff(i);

	    kerncache.recalcDiag(i);

	    sigmacache.remove(i);
	    sigmacache.add(i);

            //sigmacache.reset(SVM_Scalar::N(),&evalSigmaSVM_Scalar,(void *) this);
            //sigmacache.setmemsize(MEMSHARE_SIGMACACHE(kerncache.get_memsize()),kerncache.get_min_rowdim());

	    Q.diagoffset(i,bpoff,bpoff,(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpma),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpmb),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
	}
    }

    return;
}

void SVM_Scalar::recalcdiagoff(const Vector<double> &offset)
{
    NiceAssert( offset.size() == SVM_Scalar::N() );

    if ( SVM_Scalar::N() )
    {
	isStateOpt = 0;

	Vector<double> bn(GPNWIDTH(biasdim));

	bn.zero();

	kerndiagval += offset;
	kerncache.recalcDiag();

        sigmacache.clear();

        //sigmacache.reset(SVM_Scalar::N(),&evalSigmaSVM_Scalar,(void *) this);
        //sigmacache.setmemsize(MEMSHARE_SIGMACACHE(kerncache.get_memsize()),kerncache.get_min_rowdim());

        retMatrix<double> tmpma;
        retMatrix<double> tmpmb;

	Q.diagoffset(offset,bn,offset,bn,(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpma),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpmb),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    }

    return;
}

void SVM_Scalar::recalcdiagoff(int i, double offset)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Scalar::N() );

    isStateOpt = 0;

    kerndiagval("&",i) += offset;
    kerncache.recalcDiag(i);

    sigmacache.clear();

    //sigmacache.reset(SVM_Scalar::N(),&evalSigmaSVM_Scalar,(void *) this);
    //sigmacache.setmemsize(MEMSHARE_SIGMACACHE(kerncache.get_memsize()),kerncache.get_min_rowdim());

    retMatrix<double> tmpma;
    retMatrix<double> tmpmb;

    Q.diagoffset(i,offset,offset,(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpma),(*Gpval)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,SVM_Scalar::N()-1,tmpmb),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

    return;
}

void SVM_Scalar::fudgeOn(void)
{ 
    Q.fact_fudgeOn(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt));  
    return; 
}

void SVM_Scalar::fudgeOff(void) 
{ 
    Q.fact_fudgeOff(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt)); 
    return; 
}


int SVM_Scalar::train(int &res, svmvolatile int &killSwitch)
{
    if ( !SVM_Scalar::N() || isTrained() )
    {
        return 0;
    }

    int isopt = 0;
    int i,j;
    int itcnt = 0;
    gentype fuzzycostgrad(costfnfuzztval);
    double Cwfstep;
    Vector<double> Cweightfuzzset(Cweightfuzzval);
    gentype tempres;

    Cweightfuzzval = 1.0;

    setCweightfuzz(Cweightfuzzset);

    fuzzycostgrad.realDeriv(0,0);

    // This outermost loop takes care of fuzzy weights etc

    while ( !killSwitch && !isopt && ( ( itcnt < maxiterfuzztval ) || !maxiterfuzztval ) )
    {
        res |= intrain(killSwitch);

        isopt = 1;
        itcnt++;

        if ( NS() && usefuzztval )
        {
            Cwfstep = 0.0;

            if ( NF() )
            {
                for ( i = 0 ; i < NF() ; i++ )
                {
                    j = (Q.pivAlphaF())(i);

                    tempres  = abs2(Q.alphaGrad(j));
                    Cwfstep  = (double) fuzzycostgrad(tempres);
                    Cwfstep -= Cweightfuzzval(j);
                    Cwfstep *= lrfuzztval;

                    setCweightfuzz(j,Cweightfuzzval(j)+Cwfstep);

                    if ( abs2(Cwfstep) > ztfuzztval )
                    {
                        isopt = 0;
                    }
                }
            }

            if ( NLB() )
            {
                for ( i = 0 ; i < NLB() ; i++ )
                {
                    j = (Q.pivAlphaLB())(i);

                    tempres  = abs2(Q.alphaGrad(j));
                    Cwfstep  = (double) fuzzycostgrad(tempres);
                    Cwfstep -= Cweightfuzzval(j);
                    Cwfstep *= lrfuzztval;

                    setCweightfuzz(j,Cweightfuzzval(j)+Cwfstep);

                    if ( abs2(Cwfstep) > ztfuzztval )
                    {
                        isopt = 0;
                    }
                }
            }

            if ( NUB() )
            {
                for ( i = 0 ; i < NUB() ; i++ )
                {
                    j = (Q.pivAlphaUB())(i);

                    tempres  = abs2(Q.alphaGrad(j));
                    Cwfstep  = (double) fuzzycostgrad(tempres);
                    Cwfstep -= Cweightfuzzval(j);
                    Cwfstep *= lrfuzztval;

                    setCweightfuzz(j,Cweightfuzzval(j)+Cwfstep);

                    if ( abs2(Cwfstep) > ztfuzztval )
                    {
                        isopt = 0;
                    }
                }
            }
        }
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasR();

    if ( res )
    {
	isStateOpt = 0;
    }

    else
    {
        isStateOpt = 1;
    }

    return 1;
}



// If emm >= 4 then this function is a callback to (a) update all relevant caches 
// and factorisations and (b) calculate the value of the (dual) objective.
//
// diagoffstep: this is added to the diagonal of Gp to allow barrier methods to be used.
//
// Consider the general case:
//
// Q = 1/m sum_{i1,i2,...,im} alpha_i1 alpha_i2 ... alpha_im G_{i1,i2,...,im} + alpha'.Gpn.beta + 1/2 beta'.Gn.beta + gp'.alpha + hp'.|alpha|
//
// Gradient:
//
// dQ/dalpha = sum_{i2,i3,...,im} alpha_i2 alpha_i3 ... alpha_im G_{i1,i2,...,im} + Gpn.beta + gp + hp.*sgn(alpha)
// dQ/dbeta  = Gn.beta + Gpn'.alpha
//
// Hessian:
//
// d2Q/dalpha2     = (m-1) sum_{i3,i4,...,im} alpha_i3 alpha_i4 ... alpha_im G_{i1,i2,...,im}
// d2Q/dalphadbeta = Gpn'
// d2Q/dbetaalpha  = Gpn
// d2Q/dbeta2      = Gn
//
// But we can't use this for Gp, as it would mess up the calculation of the gradient, and that
// is required.  So instead we have...
//
// Gp calculation (as stored here to make the gradient work properly)
//
// Gp = sum_{i3,i4,...,im} alpha_i3 ... alpha_im G_{i1,i2,...,im}
//
// Now, the optimiser requires gradient to be correct and also access to the Hessian.  To achieve
// this gpgnhpGpnGnscalefactor is set to:
//
// gpgnhpGpnGnscalefactor = 1/(m-1)
//
// Noting that the step is calculated as:
//
// [ dalpha ] = inv( [ H    Gpn ] ) [ alphaGrad ]
// [ dbeta  ]        [ Gpn' Gn  ]   [ betaGrad  ]
//        
//            = inv( [ (m-1) Gp   Gpn ] ) [ alphaGrad ]
//                   [ Gpn'       Gn  ]   [ betaGrad  ]
//        
//            = inv( (m-1) [ Gp            1/(m-1) Gpn ] ) [ alphaGrad ]
//                         [ 1/(m-1) Gpn'  1/(m-1) Gn  ]   [ betaGrad  ]
//        
//            = 1/(m-1) inv( [ Gp            1/(m-1) Gpn ] ) [ alphaGrad ]
//                           [ 1/(m-1) Gpn'  1/(m-1) Gn  ]   [ betaGrad  ]
//        
//            = inv( [ Gp            1/(m-1) Gpn ] ) [ 1/(m-1) alphaGrad ]
//                   [ 1/(m-1) Gpn'  1/(m-1) Gn  ]   [ 1/(m-1) betaGrad  ]
//        
//            = inv( [ Gp       sf.Gpn ] ) [ sf.alphaGrad ]   (sf = gpgnhpGpnSnscalefactor = 1/(m-1))
//                   [ sf.Gpn'  sf.Gn  ]   [ sf.betaGrad  ]
//        
// what we need to do is multiply everything - Gpn,Gn,alphaGrad,betaGrad - by gpgnhpGpnscalefactor
// before proceeding with calculation.  Moreover for i : alpha_i = 0 we also need to incorporate
// hp_i != 0, so we must multiply hp by gpgnhpGpnGnscalefactor as well (if alpha_i != 0 then we 
// set hp_i = 0).
//
// Furthermore: if doing Newton descent you need to add diagonal offset to H.
//
// H -> H + diagoffAdd
// 
// and hence:
//
// (m-1) Gp -> (m-1) Gp + diagoffAdd
// Gp -> Gp + 1/(m-1) diagoffAdd
// Gp -> Gp + sf.diagoffAdd
//
// but... we don't want this reflected in the gradient, so
//
// gp := gp - sf.diagoffAdd.*alpha
// 
// We also do allow for offset of gp by alphaGradAdd to make sure that optimality testing is accurate!


double emmupfixer(fullOptState<double,double> &x, void *y, const Vector<double> &diagoffAdd, const Vector<double> &alphaGradAdd, double &gpgnhpGpnGnscalefactor)
{
    (void) diagoffAdd;

    SVM_Scalar &caller = *((SVM_Scalar *) y);
    optState<double,double> &Q = x.x;

    // Reset caches etc.
    //
    // NB: because of the pre-solver the "alpha" from Q may differ from that
    //     of the caller.

    int i,j;
    int &aNNZ = caller.prevNZ;
    int baseN = caller.Cweight().size(); // just the part that is "in" the problem
    static int z = 0;

    double res = 0;

    const Vector<double> &alpha = Q.alpha();

    aNNZ = 0;

    // Note: we only loop over the part "in" the main problem here - can't use
    // .N() as this refers down to Q, so use Cweight().size() instead.

    for ( i = 0 ; i < baseN ; i++ )
    {
        if ( Q.alphaState()(i) )
        {
            caller.alphaPrevPivNZ("&",aNNZ) = i;

            aNNZ++;
        }
    }

//    (caller.alphaPrev)("&",caller.alphaPrevPivNZ(z,1,aNNZ-1)) = alpha(caller.alphaPrevPivNZ(z,1,aNNZ-1));

//    ((caller.diagoff)("&",z,1,baseN-1)) = (caller.diagoffsetBase)(z,1,baseN-1);
//    ((caller.diagoff)("&",z,1,baseN-1)).scaleAdd(1.0/((caller.emm)-1.0),diagoffAdd(z,1,baseN-1));

    retVector<double> tmpva;
    retVector<double> tmpvb;

    caller.alphaPrev("&",z,1,baseN-1,tmpva) = alpha(z,1,baseN-1,tmpvb);

    (caller.kerncache).clear();
    (caller.sigmacache).clear();

    gpgnhpGpnGnscalefactor = 1.0/((caller.emm)-1);

    ((caller.gp)("&",z,1,baseN-1,tmpva)) =  (caller.gpBase)(z,1,baseN-1,tmpvb);
    ((caller.gp)("&",z,1,baseN-1,tmpva)) += alphaGradAdd(z,1,baseN-1,tmpvb);

//    for ( i = 0 ; i < baseN ; i++ )
//    {
//        if ( Q.alphaState()(i) )
//        {
//            caller.gp("&",i) -= (1.0/((caller.emm)-1.0))*diagoffAdd(i)*alpha(i); // This is to stop Hessian-only terms effecting the gradient!
//        }
//    }

//    Q.refact(x.Gp,x.Gp,x.Gn,x.Gpn,x.gp,x.gn,x.hp);
    Q.refact(x.Gp,x.Gp,x.Gn,x.Gpn,x.gp,x.gn,x.hp);

    // Evaluate implied function (Q above)

    if ( aNNZ )
    {
        double Gppart,alphaval,gpval,hpval;

        for ( i = 0 ; i < aNNZ ; i++ )
        {
            j = caller.alphaPrevPivNZ(i);

            Q.unAlphaGrad(Gppart,j,x.Gp,x.Gpn,x.gp,x.hp);

            alphaval = alpha(j);
            gpval    = x.gp(j);
            hpval    = x.hp(j);

            Gppart -= gpval;
//            Gppart -= diagoffAdd(j)/((caller.emm)-1.0); // let the caller take care of the corrections due to barriers in the calculation of the objective

            res += (Gppart*alphaval)/(caller.emm);
            res += (((caller.gpBase)(j))*alphaval); // let caller take care of corrections due to barriers in calculation of objective
            res += (hpval*abs2(alphaval));
        }
    }

    // Part that is not "in" the original (ie. constraints were not met, slacks added)
    // Assume diagonal added to Gp.  Need to add diagonal here.  Note that rather
    // than 1/m we need to use 1/2, as this is presumed to be added to the *hessian*,
    // not Gp.

    if ( baseN < Q.aN() )
    {
        double Gpval,alphaval,gpval,hpval;

        for ( i = baseN ; i < Q.aN() ; i++ )
        {
            Gpval = x.Gp(i,i); // + caller.diagoffsetBase(i);

            alphaval = alpha(i);
            gpval    = x.gp(i);
            hpval    = x.hp(i);

            res += (alphaval*Gpval*alphaval)/2.0;
            res += (gpval*alphaval);
            res += (hpval*abs2(alphaval));
        }
    }

    return res;
}

int SVM_Scalar::intrain(svmvolatile int &killSwitch)
{
    if ( !SVM_Scalar::N() || isTrained() )
    {
        return 0;
    }

    int res = 0;

    if ( emm == 2 )
    {
        res |= inintrain(killSwitch);
    }

    else
    { 
        int allset = inEmm4Solve;

        // Allocate cache, if not too large
        //
        // Cache size is: sum_{i=0,N} sum_{j=0,i} sum_{k=0,j} sum_{l=0,k} 1 = 1/24 N^4 + 1/4 N^3 + 7/12 N^2 + 9/24 N
        // Cache elements are doubles
        // So assuming 8 bytes per double that is: 1/3 N^4 bytes
        //
        // N ~ 40  gives 1MB
        // N ~ 130 gives 100MB
        // N ~ 150 gives 200MB
        // N ~ 234 gives 1GB

        if ( ( emm == 4 ) && !emm4K4cache && ( ( ( N() <= outermaxcacheN ) || !outermaxcacheN ) && N() ) )
        {
            errstream() << " gencache ";

            int i,j,k,l;

            MEMNEWARRAY(emm4K4cache,double ***,N());

            for ( i = 0 ; i < N() ; i++ )
            {
                MEMNEWARRAY(emm4K4cache[i],double **,i+1);

                for ( j = 0 ; j <= i ; j++ )
                {
                    MEMNEWARRAY(emm4K4cache[i][j],double *,j+1);

                    for ( k = 0 ; k <= j ; k++ )
                    {
                        MEMNEWARRAY(emm4K4cache[i][j][k],double,k+1);

                        for ( l = 0 ; l <= k ; l++ )
                        {
                            K4(emm4K4cache[i][j][k][l],i,j,k,l);
                        }
                    }
                }
            }

            errstream() << " done ";
        }

        // Pre-train SVM

//        if ( N() && !( (Q.aNF())+(Q.aNLB())+(Q.aNUB()) ) )
//        {
//            errstream() << " pretrain ";
//
//            res |= inintrain(killSwitch);
//        }
//
//        errstream() << " setup ";

        // Setup

        inEmm4Solve = 1;

        alphaPrev.resize(N());
        alphaPrevPivNZ.resize(N());

        diagoffsetBase = diagoff;
        gpBase = gp;

        res |= inintrain(killSwitch,emmupfixer,(void *) this,outerlrval);

        // NB: sQgraddesc has been modified to remove offsets on exit without
        // destroying caches, so no need to do the following anymore.

//        diagoff = diagoffsetBase;
//        gp = gpBase;
//
//        kerncache.recalcDiag();
//        sigmacache.clear();
//
//        Q.refact(*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);










/*
        int isopt = 0;
        unsigned int itcnt = 0;
        int temp = 0;
        int tempa = 0;
        int tempb = 0;

        double maxstep = 0;
        double maxstepprev = -1;
        double lr = outerlrval;
        double mom = outermomval;
        int method = outermethodval;
        double tol = ( outertolval > Opttol() ) ? outertolval : Opttol();

        alphaPrev.resize(N());
        alphaPrevPivNZ.resize(N());

        //Vector<double> alphaPrev(N());
        Vector<double> alphaStep(N());
        Vector<double> alphaStepPrev(N());

        alphaPrev     = Q.alpha();
        alphaStep     = 0.0;
        alphaStepPrev = 0.0;

        prevNZ = (Q.aNF())+(Q.aNLB())+(Q.aNUB());

        alphaPrevPivNZ("&",0               ,1,Q.aNF()                  -1) = Q.pivAlphaF ();
        alphaPrevPivNZ("&",Q.aNF()         ,1,Q.aNF()+Q.aNLB()         -1) = Q.pivAlphaLB();
        alphaPrevPivNZ("&",Q.aNF()+Q.aNLB(),1,Q.aNF()+Q.aNLB()+Q.aNUB()-1) = Q.pivAlphaUB();

        // Fix Hessian

        kerncache.clear();
        sigmacache.clear();
        Q.refact((*Gpval),(*Gpval),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

        // Prime max error

        isopt = ( ( maxstepprev = maxabs(Q.alphaGrad()(Q.pivAlphaF()),temp) ) > tol ) ? 0 : 1;

        if ( !isopt )
        {
            isopt = Q.maxGradNonOpt(tempa,tempb,temp,maxstepprev,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
            maxstepprev = abs2(maxstepprev);
            isopt = ( !isopt && ( maxstepprev > tol ) ) ? 0 : 1;
        }

        // Optimisation loop

        double maxitcnt = outermaxits;
        double maxtime = maxtraintimeval;
        double ovsc = outerovscval;
        double *uservars[] = { &maxitcnt, &maxtime, &lr, &mom, &tol, &ovsc, NULL };
        const char *varnames[] = { "maxitcnt", "maxtime", "lr", "mom", "tol", "ovsc", NULL };
        const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Learning rate", "Momentum factor", "Zero tolerance", "Overstep scaleback factor", NULL };

        time_used start_time = TIMECALL;
        time_used curr_time = start_time;
        int timeout = 0;

        while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcnt ) || !maxitcnt ) && !timeout )
        {
            errstream() << " $ ";

            // Save state

            prevNZ = (Q.aNF())+(Q.aNLB())+(Q.aNUB());

            alphaPrevPivNZ("&",0               ,1,Q.aNF()                  -1) = Q.pivAlphaF ();
            alphaPrevPivNZ("&",Q.aNF()         ,1,Q.aNF()+Q.aNLB()         -1) = Q.pivAlphaLB();
            alphaPrevPivNZ("&",Q.aNF()+Q.aNLB(),1,Q.aNF()+Q.aNLB()+Q.aNUB()-1) = Q.pivAlphaUB();

//phantomxyzabc - set momentum zero, replace step calculation with simple gradient
//Assumption check: need to assume LS model
            // Call internal training loop

            res |= inintrain(killSwitch);

            // Calculate step

            alphaStep  = Q.alpha();
            alphaStep -= alphaPrev;
//phantomxyzabc

            // Scale step, add momentum

            alphaStep *= lr;

//phantomxyzabc - line-search goes here
            // add momentum

            alphaStep.scaleAdd(mom,alphaStepPrev);

            // Save step

            alphaStepPrev = alphaStep;

            // Take step

            alphaPrev += alphaStep;

            Q.setAlpha(alphaPrev,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,lb,ub);

            // Fix Hessian

            kerncache.clear();
            sigmacache.clear();
            Q.refact((*Gpval),(*Gpval),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

            // Grab max step size and set isopt

            isopt = ( ( maxstep = maxabs(Q.alphaGrad()(Q.pivAlphaF()),temp) ) > tol ) ? 0 : 1;

            if ( !isopt )
            {
                isopt = Q.maxGradNonOpt(tempa,tempb,temp,maxstep,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
                maxstep = abs2(maxstep);
                isopt = ( !isopt && ( maxstep > tol ) ) ? 0 : 1;

                if ( maxstep > maxstepprev )
                {
                    lr  *= ovsc;
                    mom *= ovsc;
                }

                maxstepprev = maxstep;

                errstream() << maxstep;
            }

            itcnt++;

            if ( maxtime > 1 )
            {
                curr_time = TIMECALL;

                if ( TIMEDIFFSEC(curr_time,start_time) > maxtime )
                {
                    timeout = 1;
                }
            }

            if ( !timeout )
            {
                timeout = kbquitdet("m4 optimisation",uservars,varnames,vardescr);
            }
        }

        errstream() << " --- " << maxstep << " $$$ ";

	inEmm4Solve = allset;

        // Fix Hessian

        kerncache.clear();
        Q.refact((*Gpval),(*Gpval),Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
*/












        // Delete cache (if allocated)

        if ( emm4K4cache && !allset )
        {
            errstream() << " delcache ";

            int i,j,k;

            for ( i = 0 ; i < N() ; i++ )
            {
                for ( j = 0 ; j <= i ; j++ )
                {
                    for ( k = 0 ; k <= j ; k++ )
                    {
                        MEMDELARRAY(emm4K4cache[i][j][k]);

                        emm4K4cache[i][j][k] = NULL;
                    }

                    MEMDELARRAY(emm4K4cache[i][j]);

                    emm4K4cache[i][j] = NULL;
                }

                MEMDELARRAY(emm4K4cache[i]);

                emm4K4cache[i] = NULL;
            }

            MEMDELARRAY(emm4K4cache);

            emm4K4cache = NULL;
        }
    }

    return res;
}

int SVM_Scalar::inintrain(svmvolatile int &killSwitch, double (*fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor) 
{ 
    int res = 0;

    if ( !SVM_Scalar::N() || isTrained() )
    {
        return 0;
    }

    int dnorm = GPNorGPNEXT(Gpn,GpnExt).numCols();
    int dconv = fspaceDim();

    int NN = GPNorGPNEXT(Gpn,GpnExt).numRows();

    if ( makeConvex && dconv )
    {
        int i,j;

        NiceAssert( dconv >= 0 );

        Matrix<double> &dGpn = GPNorGPNEXT(Gpn,GpnExt);

        // Extend to include possible additional inequality constraints
        //
        // w'phi2(x) >= 0
        //
        // Assume: phi2(x) = polynomial in x
        //         x >= 0 elementwise
        //
        // So phi2(x) >= 0 elementwise, and all gradients too.
        //
        // So we need w' dphi2/dxk (x) >= 0 for all 0 <= k < xdim, for all x
        //
        // or, equivalently, sum_i alpha_i phi2(x_i) >= 0 in all components k s.t. phi2k(0) == 0

        // Calculations

        SparseVector<gentype> zv;

        Vector<Vector<double> > phi2i(NN); // phi2(x_i)
        Vector<double> phi2z;              // phi2(0)

        phi2(phi2z,zv).size();

        Vector<int> phi2rem(dconv);    // set flags to remove from phi2z and phi2i(i)	
        Vector<double> phi2sum(dconv); // norm of phi2i components

        for ( j = 0 ; j < dconv ; j++ )
        {
            phi2sum("&",j) = 0;
            phi2rem("&",j) = ( norm2(phi2z(j)) < 1e-6 ) ? 0 : 1; // remove if corresponds to constant in map
        }

        for ( i = 0 ; i < NN ; i++ )
        {
            phi2(phi2i("&",i),i);

            for ( j = 0 ; j < dconv ; j++ )
            {
                phi2sum("&",j) += norm2(phi2i(i)(j));
            }
        }

        for ( j = 0 ; j < dconv ; j++ )
        {
            phi2rem("&",j) = phi2rem(j) || ( ( phi2sum(j) == 0 ) ? 1 : 0 ); // remove if not relevant
        }

        for ( i = 0 ; i < NN ; i++ )
        {
            for ( j = dconv-1 ; j >= 0 ; j-- )
            {
                if ( phi2rem(j) )
                {
                    phi2i("&",i).remove(j);
                }
            }
        }

        dconv -= sum(phi2rem);

        // Do extension.

        dGpn.resize(NN,dnorm+dconv);
        Gn.resize(dnorm+dconv,dnorm+dconv);
        gn.resize(dnorm+dconv);

        for ( i = dnorm ; i < dnorm+dconv ; i++ )
        {
            gn("&",i) = 0.0;
        }

        for ( i = 0 ; i < dnorm+dconv ; i++ )
        {
            for ( j = ( i < dnorm ) ? dnorm : 0 ; j < dnorm+dconv ; j++ )
            {
                Gn("&",i,j) = 0.0;
            }
        }

        for ( i = 0 ; i < NN ; i++ )
        {
            retVector<double> tmpva;

            dGpn("&",i,dnorm,1,dnorm+dconv-1,tmpva) = phi2i(i);
        }

        // Add new constraints to Q

        for ( i = dnorm ; i < dnorm+dconv ; i++ )
        {
            Q.addBeta(i,makeConvex,0.0);
        }
    }

  if ( is1NormCost() )
  {
    res = solve_linear_program(Q, wr,gn, cr,ddr, *Gpval,GPNorGPNEXT(Gpn,GpnExt),
                               Qnp,Qn, gp,hp,qn, lb,ub, Gn,gn,
                               alpharestrictoverride,Qconstype,
                               killSwitch,maxitcntval,maxtraintimeval);
  }

  else
  {
    //fixBetaOptimality();

    // First we ensure that the equality constraint is met (if any)
    // We only deal with the most simple case here.  More general non-feasible
    // start is now dealt with by the optimiser code itself.

    if ( ( biasdim == 0 ) && ( GpnExt == NULL ) && Gplocal && !isPosBias() && !isNegBias() )
    {
        double betaGrad = sum(alphaR()) + Gn(zeroint(),zeroint())*(Q.beta(zeroint())) + gn(zeroint());

	//if ( ( ( isVarBias() || isPosBias() ) && ( abs2(betaGrad) > Opttol() ) ) || ( ( isVarBias() || isNegBias() ) && ( betaGrad < -Opttol() ) ) )
	if ( ( isVarBias() && ( abs2(betaGrad) >  Opttol() ) ) || 
             ( isPosBias() && (      betaGrad  >  Opttol() ) ) || 
             ( isNegBias() && (      betaGrad  < -Opttol() ) )    )
	{
nullPrint(errstream(),"^");
            res = presolveit(betaGrad);
if ( res ) { errstream() << "FAIL: No optimal start point.\n"; }
	}
    }

//errstream() << "phantomxyz 0: Gp = " << *Gpval << "\n";
//errstream() << "phantomxyz 1: Gpn = " << GPNorGPNEXT(Gpn,GpnExt) << "\n";
//errstream() << "phantomxyz 2: Gn = " << Gn << "\n";
//errstream() << "phantomxyz 3: gn = " << gn << "\n";
    // Then we train

//phantomxyzabc
    if ( fixHigherOrderTerms || isOptActive() || isPosBias() || isNegBias() || isShrinkTube() )
    {
	res = 1;

        if ( isFixedTube() )
	{
            if ( epsval < zerotol() )
	    {
                int oldGpSize = (*Gpval).numRows();
                int oldxySize = (*xyval).numRows(); 
                int oldGpsigmaSize = (*Gpsigma).numRows();

                kerncache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
                xycache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
                sigmacache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
                (*xyval).resize(oldxySize,oldxySize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
                (*Gpval).resize(oldGpSize,oldGpSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
                (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));

                if ( fixHigherOrderTerms )
                {
                    fullOptStateGradDesc xx(Q,*Gpval,*Gpsigma,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor,
                                            outerovscval,DEFAULTOUTERDELTA,outermethodval,outermaxits,maxtraintimeval,( outertolval > Opttol() ) ? outertolval : Opttol());

                    xx.maxitcnt   = maxitcntval;
                    xx.maxruntime = maxtraintimeval;

                    res = xx.wrapsolve(killSwitch);
                }

                else
                {
                    fullOptStateActive xx(Q,*Gpval,*Gpsigma,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,lb,ub,NULL,NULL,1.0);

                    xx.maxitcnt   = maxitcntval;
                    xx.maxruntime = maxtraintimeval;

                    res = xx.wrapsolve(killSwitch);

                }

                kerncache.padCol(0);
                xycache.padCol(0);
                sigmacache.padCol(0);
                (*xyval).resize(oldxySize,oldxySize);
                (*Gpval).resize(oldGpSize,oldGpSize);
                (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize);
	    }

	    else
	    {
                int oldxySize = (*xyval).numRows();
                int oldGpSize = (*Gpval).numRows();
                int oldGpsigmaSize = (*Gpsigma).numRows();

                kerncache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
                xycache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
                sigmacache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
                (*xyval).resize(oldxySize,oldxySize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
                (*Gpval).resize(oldGpSize,oldGpSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
                (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));

                if ( fixHigherOrderTerms )
                {
                    fullOptStateGradDesc xx(Q,*Gpval,*Gpsigma,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor,
                                            outerovscval,DEFAULTOUTERDELTA,outermethodval,outermaxits,maxtraintimeval,( outertolval > Opttol() ) ? outertolval : Opttol());

                    xx.maxitcnt   = maxitcntval;
                    xx.maxruntime = maxtraintimeval;

                    res = xx.wrapsolve(killSwitch);
                }

                else
                {
                    fullOptStateActive xx(Q,*Gpval,*Gpsigma,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,lb,ub,NULL,NULL,1.0);

                    xx.maxitcnt   = maxitcntval;
                    xx.maxruntime = maxtraintimeval;

                    res = xx.wrapsolve(killSwitch);
                }

                kerncache.padCol(0);
                xycache.padCol(0);
                sigmacache.padCol(0);
                (*xyval).resize(oldxySize,oldxySize);
                (*Gpval).resize(oldGpSize,oldGpSize);
                (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize);
	    }
	}

	else
	{
	    // This is tube shrinking.

            NiceAssert( GpnExt == NULL );
            NiceAssert( biasdim == 0 );

            SVM_Scalar::seteps(0.0);

	    int i;
	    int newGpnRow = isVarBias() ? 1 : 0;

	    Vector<double> zerovect(hp);

	    zerovect = 0.0;

	    Gpn.addCol(newGpnRow);
	    Gn.addRowCol(newGpnRow);
	    gn.add(newGpnRow);

            if ( SVM_Scalar::N() )
	    {
                for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
		{
		    if ( alphaState()(i) < 0 )
		    {
                        Gpn("&",i,newGpnRow) = -hpscale(i);
		    }

		    else if ( alphaState()(i) > 0 )
		    {
                        Gpn("&",i,newGpnRow) = hpscale(i);
		    }

		    else
		    {
                        Gpn("&",i,newGpnRow) = 0.0;
		    }
		}
	    }

	    if ( isVarBias() )
	    {
		Gn("&",0,1) = 0.0;
		Gn("&",1,0) = 0.0;
                Gn("&",1,1) = 0.0;
	    }

	    else
	    {
                Gn("&",0,0) = -(CNval*nuQuadv*(SVM_Scalar::N()-SVM_Scalar::NNC(0)))/2;
	    }

            gn("&",newGpnRow) = -CNval*nuLin*(SVM_Scalar::N()-SVM_Scalar::NNC(0));

	    Q.addBeta(newGpnRow,epsrestrict,0.0);

            {
                int oldxySize = (*xyval).numRows();
                int oldGpSize = (*Gpval).numRows();
                int oldGpsigmaSize = (*Gpsigma).numRows();

                kerncache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
                xycache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
                sigmacache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
                (*xyval).resize(oldxySize,oldxySize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
                (*Gpval).resize(oldGpSize,oldGpSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
                (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));

                if ( fixHigherOrderTerms )
                {
                    fullOptStateGradDesc xx(Q,*Gpval,*Gpsigma,Gn,Gpn,gp,gn,zerovect,lb,ub,hpscale,fixHigherOrderTerms,htArg,stepscalefactor,
                                            outerovscval,DEFAULTOUTERDELTA,outermethodval,outermaxits,maxtraintimeval,( outertolval > Opttol() ) ? outertolval : Opttol());

                    xx.maxitcnt   = maxitcntval;
                    xx.maxruntime = maxtraintimeval;

                    res = xx.wrapsolve(killSwitch);
                }

                else
                {
                    fullOptStateActive xx(Q,*Gpval,*Gpsigma,Gn,Gpn,gp,gn,zerovect,lb,ub,hpscale,NULL,NULL,1.0);

                    xx.maxitcnt   = maxitcntval;
                    xx.maxruntime = maxtraintimeval;

                    res = xx.wrapsolve(killSwitch);
                }

                kerncache.padCol(0);
                xycache.padCol(0);
                sigmacache.padCol(0);
                (*xyval).resize(oldxySize,oldxySize);
                (*Gpval).resize(oldGpSize,oldGpSize);
                (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize);
            }

            double neweps = Q.beta(newGpnRow);

	    if ( Q.betaState(newGpnRow) )
	    {
		Q.betaStep(newGpnRow,-neweps,*Gpval,Gn,Gpn,gp,gn,zerovect);
		Q.modBetaFtoC(Q.findInBetaF(newGpnRow),*Gpval,*Gpval,Gn,Gpn,gp,gn,zerovect);
	    }

	    Q.changeBetaRestrict(newGpnRow,3,*Gpval,*Gpval,Gn,Gpn,gp,gn,zerovect);
	    Q.removeBeta(newGpnRow);

	    gn.remove(newGpnRow);
	    Gn.removeRowCol(newGpnRow);
	    Gpn.removeCol(newGpnRow);

            SVM_Scalar::seteps(neweps);
	}
    }

    else if ( isOptSMO() )
    {
        NiceAssert( !fixHigherOrderTerms );

        int oldxySize = (*xyval).numRows();
        int oldGpSize = (*Gpval).numRows();
        int oldGpsigmaSize = (*Gpsigma).numRows();

        kerncache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
        xycache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
        sigmacache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
        (*xyval).resize(oldxySize,oldxySize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
        (*Gpval).resize(oldGpSize,oldGpSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
        (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));

        {
            fullOptStateSMO xx(Q,*Gpval,*Gpsigma,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,lb,ub,NULL,NULL,1.0);

            xx.maxitcnt   = maxitcntval;
            xx.maxruntime = maxtraintimeval;

            res = xx.wrapsolve(killSwitch);
        }

        kerncache.padCol(0);
        xycache.padCol(0);
        sigmacache.padCol(0);
        (*xyval).resize(oldxySize,oldxySize);
        (*Gpval).resize(oldGpSize,oldGpSize);
        (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize);
    }

    else if ( isOptD2C() )
    {
        NiceAssert( !fixHigherOrderTerms );

        int oldxySize = (*xyval).numRows();
        int oldGpSize = (*Gpval).numRows();
        int oldGpsigmaSize = (*Gpsigma).numRows();

        kerncache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
        xycache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
        sigmacache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
        (*xyval).resize(oldxySize,oldxySize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
        (*Gpval).resize(oldGpSize,oldGpSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
        (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));

        {
            fullOptStateD2C xx(Q,*Gpval,*Gpsigma,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,lb,ub,NULL,NULL,1.0);

            xx.maxitcnt   = maxitcntval;
            xx.maxruntime = maxtraintimeval;

            res = xx.wrapsolve(killSwitch);
        }

        kerncache.padCol(0);
        xycache.padCol(0);
        sigmacache.padCol(0);
        (*xyval).resize(oldxySize,oldxySize);
        (*Gpval).resize(oldGpSize,oldGpSize);
        (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize);
    }

    else
    {
        NiceAssert( !fixHigherOrderTerms );

        int oldxySize = (*xyval).numRows();
        int oldGpSize = (*Gpval).numRows();
        int oldGpsigmaSize = (*Gpsigma).numRows();

        kerncache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
        xycache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
        sigmacache.padCol(4*(GPNorGPNEXT(Gpn,GpnExt).numCols()));
        (*xyval).resize(oldxySize,oldxySize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
        (*Gpval).resize(oldGpSize,oldGpSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));
        (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize+(2*(GPNorGPNEXT(Gpn,GpnExt).numCols())));

        {
            fullOptStateGradDesc xx(Q,*Gpval,*Gpsigma,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,lb,ub,NULL,NULL,1.0);

            xx.maxitcnt   = maxitcntval;
            xx.maxruntime = maxtraintimeval;

            res = xx.wrapsolve(killSwitch);
        }

        kerncache.padCol(0);
        xycache.padCol(0);
        sigmacache.padCol(0);
        (*xyval).resize(oldxySize,oldxySize);
        (*Gpval).resize(oldGpSize,oldGpSize);
        (*Gpsigma).resize(oldGpsigmaSize,oldGpsigmaSize);
    }

    if ( ( !isFixedBias()  ) &&
         ( biasdim == 0    ) &&
         ( GpnExt  == NULL ) &&
         ( Q.aNF() == 0    ) &&
         ( Q.aN()          ) &&
         ( SVM_Scalar::N()-SVM_Scalar::NNC(0) ) )
    {
	int i;

	// centre the bias

        Q.betaStep(0,-Q.beta()(0),*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

	double dbetaDown = -1/Q.zerotol();
	double dbetaUp = 1/Q.zerotol();
	int bdown = 0;
        int bup = 0;

	for ( i = 0 ; i < Q.aN() ; i++ )
	{
	    if ( Q.alphaRestrict()(i) == 0 )
	    {
		if ( ( Q.alphaState()(i) == 0 ) || ( Q.alphaState()(i) == -2 ) )
		{
		    // gradAlpha(i) >= 0

		    dbetaDown = ( ( Q.alphaGrad()(i) + dbetaDown ) < 0 ) ? -Q.alphaGrad()(i) : dbetaDown;
                    bdown = 1;
		}

		if ( ( Q.alphaState()(i) == 0 ) || ( Q.alphaState()(i) == +2 ) )
		{
                    // alphaGrad(i) <= 0

		    dbetaUp = ( ( Q.alphaGrad()(i) + dbetaUp ) > 0 ) ? -Q.alphaGrad()(i) : dbetaUp;
                    bup = 1;
		}
	    }

	    else if ( Q.alphaRestrict(i) == 1 )
	    {
		if ( Q.alphaState()(i) == 0 )
		{
                    // alphaGrad(i) >= 0

		    dbetaDown = ( ( Q.alphaGrad()(i) + dbetaDown ) < 0 ) ? -Q.alphaGrad()(i) : dbetaDown;
                    bdown = 1;
		}

		else if ( Q.alphaState()(i) == +2 )
		{
                    // alphaGrad(i) <= 0

		    dbetaUp = ( ( Q.alphaGrad()(i) + dbetaUp ) > 0 ) ? -Q.alphaGrad()(i) : dbetaUp;
                    bup = 1;
		}
	    }

	    else if ( Q.alphaRestrict(i) == 2 )
	    {
		if ( Q.alphaState()(i) == -2 )
		{
                    // alphaGrad(i) >= 0

		    dbetaDown = ( ( Q.alphaGrad()(i) + dbetaDown ) < 0 ) ? -Q.alphaGrad()(i) : dbetaDown;
                    bdown = 1;
		}

		else if ( Q.alphaState()(i) == 0 )
		{
                    // alphaGrad(i) <= 0

		    dbetaUp = ( ( Q.alphaGrad()(i) + dbetaUp ) > 0 ) ? -Q.alphaGrad()(i) : dbetaUp;
                    bup = 1;
		}
	    }
	}

	double dbeta = ( bdown && bup ) ? (dbetaUp+dbetaDown)/2 : ( !( bdown || bup ) ? 0 : ( bdown ? dbetaDown : dbetaUp ) );

        Q.betaStep(0,dbeta,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    }
  }

    if ( makeConvex && dconv )
    {
        NiceAssert( dconv >= 0 );

        Matrix<double> &dGpn = GPNorGPNEXT(Gpn,GpnExt);

        int i;

        for ( i = dnorm+dconv-1 ; i >= dnorm ; i-- )
        {
            Q.changeBetaRestrict(i,3,*Gpval,*Gpval,Gn,dGpn,gp,gn,hp);

            dconv--;

            Q.removeBeta(i);
            dGpn.removeCol(i);
            Gn.removeRowCol(i);
            gn.remove(i);
        }
    }

  return res;
}

int SVM_Scalar::presolveit(double betaGrad)
{
    int res = 0;

    if ( abs2(Gn(zeroint(),zeroint())) >= zerotol() )
    {
	// Fix with beta step
	//
	// betaGrad + Gn.betastep = 0
	// => betastep = -betaGrad/Gn

	double betaStep = -Q.betaGrad(zeroint())/Gn(zeroint(),zeroint());

	// Free beta if required

	if ( !(Q.betaState(0)) )
	{
	    Q.modBetaCtoF(0,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
	}

	// Take step

	Q.betaStep(0,betaStep,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
    }

    else
    {
	// Fix with alpha steps

	int notopt = 1;
	int convar;
	int iP,i;
	int stateChange;
	Vector<double> stepAlpha(Q.alpha());
	Vector<double> stepBeta(Q.beta());
	double dbetaGrad;
	double gradmag;
	double alphaChange;
	double scale;

	stepBeta = 0.0;

	while ( notopt )
	{
	    if ( NF() )
	    {
		// Calculate required step in free alphas

	    repeatover:
		alphaChange = 0;
		notopt = 1;

		for ( iP = 0 ; iP < NF() ; iP++ )
		{
		    if ( betaGrad*(Q.alphaState(Q.pivAlphaF()(iP))) > 0 )
		    {
                        stepAlpha("&",Q.pivAlphaF()(iP)) = -alphaR()(Q.pivAlphaF()(iP));
		    }

		    else if ( ( Q.alphaState(Q.pivAlphaF()(iP)) < 0 ) && betaGrad > 0 )
		    {
                        stepAlpha("&",Q.pivAlphaF()(iP)) = lb(Q.pivAlphaF()(iP))-alphaR()(Q.pivAlphaF()(iP));
		    }

		    else if ( ( Q.alphaState(Q.pivAlphaF()(iP)) > 0 ) && betaGrad < 0 )
		    {
                        stepAlpha("&",Q.pivAlphaF()(iP)) = ub(Q.pivAlphaF()(iP))-alphaR()(Q.pivAlphaF()(iP));
		    }

		    else
		    {
			stepAlpha("&",Q.pivAlphaF()(iP)) = 0.0;
		    }

		    alphaChange += stepAlpha(Q.pivAlphaF()(iP));
		}

		if ( abs2(alphaChange) != 0 )
		{
		    scale = -betaGrad/alphaChange;

                    retVector<double> tmpva;
                    retVector<double> tmpvb;

		    if ( scale > 1 )
		    {
			dbetaGrad = alphaChange;
		    }

		    else
		    {
			stepAlpha("&",Q.pivAlphaF(),tmpva) *= scale;
			dbetaGrad = -betaGrad;
			notopt = 0;
		    }

		    Q.stepFGeneral(NF(),0,stepAlpha(Q.pivAlphaF(),tmpva),stepBeta(Q.pivBetaF(),tmpvb),*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
		    betaGrad += dbetaGrad;
		}

		if ( notopt )
		{
		    // If possible, switch signs of free but zero alphas to extend step range

		    i = 0;

		    for ( iP = 0 ; iP < NF() ; iP++ )
		    {
			if ( ( Q.alphaState(Q.pivAlphaF()(iP)) == -1 ) && ( betaGrad < 0 ) && ( Q.alphaRestrict(Q.pivAlphaF()(iP)) == 0 ) )
			{
			    Q.modAlphaLFtoUF(iP,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
			    i++;
			}

			else if ( ( Q.alphaState(Q.pivAlphaF()(iP)) == +1 ) && ( betaGrad > 0 ) && ( Q.alphaRestrict(Q.pivAlphaF()(iP)) == 0 ) )
			{
			    Q.modAlphaUFtoLF(iP,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
			    i++;
			}
		    }

		    // If sign switching has occured then repeat step.  Otherwise constrain
		    // variables at bound.

		    if ( i )
		    {
			goto repeatover;
		    }

		    else
		    {
			// Complete failure.  Time to switch to another tactic.  Everything is
			// at bounds, so we will constrain them as such and then look at
			// moving constrained variables

			while ( NF() )
			{
			    iP = NF()-1;

			    if ( ( Q.alphaState(Q.pivAlphaF()(iP)) == -1 ) && ( betaGrad < 0 ) )
			    {
				Q.modAlphaLFtoZ(iP,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
			    }

			    else if ( ( Q.alphaState(Q.pivAlphaF()(iP)) == +1 ) && ( betaGrad > 0 ) )
			    {
				Q.modAlphaUFtoZ(iP,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);
			    }

			    else if ( ( Q.alphaState(Q.pivAlphaF()(iP)) == -1 ) && ( betaGrad > 0 ) )
			    {
				Q.modAlphaLFtoLB(iP,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,lb);
			    }

			    else if ( ( Q.alphaState(Q.pivAlphaF()(iP)) == +1 ) && ( betaGrad < 0 ) )
			    {
				Q.modAlphaUFtoUB(iP,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,ub);
			    }
			}

			//Q.reset(...);
		    }
		}
	    }

	    if ( notopt )
	    {
	    tryanother:
		// Not optimal, run out of steps: start freeing variables

		gradmag     = 0;
		convar      = -1;
		stateChange = 0;

		if ( betaGrad < 0 )
		{
		    if ( NLB() )
		    {
			for ( iP = 0 ; iP < NLB() ; iP++ )
			{
			    if ( -lb(Q.pivAlphaLB()(iP)) > gradmag )
			    {
				gradmag     = -lb(Q.pivAlphaLB()(iP));
				convar      = iP;
				stateChange = -2;
			    }
			}
		    }

		    if ( NZ() )
		    {
			for ( iP = 0 ; iP < NZ() ; iP++ )
			{
			    if ( ( ( Q.alphaRestrict(Q.pivAlphaZ()(iP)) == 0 ) || ( Q.alphaRestrict(Q.pivAlphaZ()(iP)) == 1 ) ) && ( ub(Q.pivAlphaZ()(iP)) > gradmag ) )
			    {
				gradmag     = ub(Q.pivAlphaZ()(iP));;
				convar      = iP;
				stateChange = -1;
			    }
			}
		    }
		}

		else
		{
		    if ( NUB() )
		    {
			for ( iP = 0 ; iP < NUB() ; iP++ )
			{
			    if ( ub(Q.pivAlphaUB()(iP)) > gradmag )
			    {
				gradmag     = ub(Q.pivAlphaUB()(iP));
				convar      = iP;
				stateChange = +2;
			    }
			}
		    }

		    if ( NZ() )
		    {
			for ( iP = 0 ; iP < NZ() ; iP++ )
			{
			    if ( ( ( Q.alphaRestrict(Q.pivAlphaZ()(iP)) == 0 ) || ( Q.alphaRestrict(Q.pivAlphaZ()(iP)) == 2 ) ) && ( -lb(Q.pivAlphaZ()(iP)) > gradmag ) )
			    {
				gradmag     = -lb(Q.pivAlphaZ()(iP));;
				convar      = iP;
				stateChange = +1;
			    }
			}
		    }
		}

		if ( convar == -1 )
		{
		    // Can't achieve success: give up

		    notopt = 0;
		    res    = 5;
		}

		else
		{
		    if ( stateChange == -2 )
		    {
			// betaGrad < 0
			// max change in alpha    = -lb(Q.pivAlphaLB()(convar)) > 0
			// max change in betaGrad = -lb(Q.pivAlphaLB()(convar)) > 0

			if ( betaGrad-lb(Q.pivAlphaLB()(convar)) >= 0 )
			{
			    dbetaGrad   = -betaGrad;
			    alphaChange = dbetaGrad;

			    Q.alphaStep(Q.pivAlphaLB()(convar),alphaChange,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);
			    Q.modAlphaLBtoLF(convar,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

			    betaGrad = 0;
			    notopt = 0;
			}

			else
			{
			    dbetaGrad   = -lb(Q.pivAlphaLB()(convar));
			    alphaChange = dbetaGrad;

			    Q.alphaStep(Q.pivAlphaLB()(convar),alphaChange,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);
			    Q.modAlphaLBtoZ(convar,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

			    betaGrad += dbetaGrad;
			    goto tryanother;
			}
		    }

		    else if ( stateChange == -1 )
		    {
			// betaGrad < 0
			// max change in alpha    = ub(Q.pivAlphaZ()(convar)) > 0
			// max change in betaGrad = ub(Q.pivAlphaZ()(convar)) > 0

			if ( betaGrad+ub(Q.pivAlphaZ()(convar)) >= 0 )
			{
			    dbetaGrad   = -betaGrad;
			    alphaChange = dbetaGrad;

			    Q.alphaStep(Q.pivAlphaZ()(convar),alphaChange,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);
			    Q.modAlphaZtoUF(convar,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

			    betaGrad = 0;
			    notopt = 0;
			}

			else
			{
			    dbetaGrad   = ub(Q.pivAlphaZ()(convar));
			    alphaChange = dbetaGrad;

			    Q.alphaStep(Q.pivAlphaZ()(convar),alphaChange,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);
			    Q.modAlphaZtoUB(convar,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

			    betaGrad += dbetaGrad;
			    goto tryanother;
			}
		    }

		    else if ( stateChange == +1 )
		    {
			// betaGrad > 0
			// max change in alpha    = lb(Q.pivAlphaZ()(convar)) < 0
			// max change in betaGrad = lb(Q.pivAlphaZ()(convar)) < 0

			if ( betaGrad+lb(Q.pivAlphaZ()(convar)) <= 0 )
			{
			    dbetaGrad   = -betaGrad;
			    alphaChange = dbetaGrad;

			    Q.alphaStep(Q.pivAlphaZ()(convar),alphaChange,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);
			    Q.modAlphaZtoLF(convar,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

			    betaGrad = 0;
			    notopt = 0;
			}

			else
			{
			    dbetaGrad   = lb(Q.pivAlphaZ()(convar));
			    alphaChange = dbetaGrad;

			    Q.alphaStep(Q.pivAlphaZ()(convar),alphaChange,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);
			    Q.modAlphaZtoLB(convar,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

			    betaGrad += dbetaGrad;
			    goto tryanother;
			}
		    }

		    else
		    {
			// betaGrad > 0
			// max change in alpha    = -ub(Q.pivAlphaUB()(convar)) < 0
			// max change in betaGrad = -ub(Q.pivAlphaUB()(convar)) < 0

			if ( betaGrad-ub(Q.pivAlphaUB()(convar)) <= 0 )
			{
			    dbetaGrad   = -betaGrad;
			    alphaChange = dbetaGrad;

			    Q.alphaStep(Q.pivAlphaUB()(convar),alphaChange,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);
			    Q.modAlphaUBtoUF(convar,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

			    betaGrad = 0;
			    notopt = 0;
			}

			else
			{
			    dbetaGrad   = -ub(Q.pivAlphaUB()(convar));
			    alphaChange = dbetaGrad;

			    Q.alphaStep(Q.pivAlphaUB()(convar),alphaChange,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp,1);
			    Q.modAlphaUBtoZ(convar,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

			    betaGrad += dbetaGrad;
			    goto tryanother;
			}
		    }
		}
	    }
	}
    }

    return res;
}

void SVM_Scalar::setbiasdim(int xbiasdim, int addpos, double addval, int rempos)
{
    // NB: Prior to calling GpnExt must be extended if needed.
    //     Prior to calling any changes to the value of GpnExt should be made.
    //     After calling GpnExt must be shrunk as required.

    NiceAssert( !( ( biasdim < 0 ) && ( xbiasdim > 0 ) ) );
    NiceAssert( !( ( biasdim > 0 ) && ( xbiasdim < 0 ) ) );
    NiceAssert( GpnExt || ( !biasdim && !xbiasdim ) );
    NiceAssert( xbiasdim >= 0 );
    NiceAssert( ( rempos == -1 ) || ( rempos == 0 ) );

    if ( biasdim != xbiasdim )
    {
	// State is no longer optimal

	isStateOpt = 0;

	// At this point we know that GpnExt != NULL
	// Allow for any changes in the value of existing elements of GpnExt

        retVector<double> tmpva;
        retMatrix<double> tmpma;
        retMatrix<double> tmpmb;

        Q.refact(*Gpval,*Gpval,Gn(zeroint(),1,GPNWIDTH(biasdim)-1,zeroint(),1,GPNWIDTH(biasdim)-1,tmpma),GPNorGPNEXT(Gpn,GpnExt)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,GPNWIDTH(biasdim)-1,tmpmb),gp,gn(zeroint(),1,GPNWIDTH(biasdim)-1,tmpva),hp);

        // Grab relevant values

	if ( biasdim == 0 )
	{
	    // At this point we know that xbiasdim > 0, so to proceed
	    // first remove the only row/column in Gn

            Q.changeBetaRestrict(0,3,*Gpval,*Gpval,Gn(zeroint(),1,GPNWIDTH(biasdim)-1,zeroint(),1,GPNWIDTH(biasdim)-1,tmpma),GPNorGPNEXT(Gpn,GpnExt)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,GPNWIDTH(biasdim)-1,tmpmb),gp,gn(zeroint(),1,GPNWIDTH(biasdim)-1,tmpva),hp);

	    Gn.removeRowCol(0);
	    gn.remove(0);

	    Q.removeBeta(0);

	    biasdim = 1;
	}

	if ( biasdim != xbiasdim )
	{
            // At this point we know that biasdim != 0
	    // ybiasdim is an intermediate target value for biasdim

	    int ybiasdim = ( xbiasdim ? xbiasdim : 1 );

            // we know that both biasdim and ybiasdim are > 0

	    // Correct the size of Gpn Gn

	    if ( biasdim < ybiasdim )
	    {
		// Record bias

		Vector<double> db(gn.size());

		if ( db.size() )
		{
		    int q;

		    for ( q = 0 ; q < db.size() ; q++ )
		    {
                        db("&",q) = biasVMulti(q);
		    }
		}

                // Update biasdim

		int xaddpos;

		while ( biasdim < ybiasdim )
		{
		    // Add row/column to Gn

		    xaddpos = ( addpos == -1 ) ? GPNWIDTH(biasdim) : addpos;

		    db.add(xaddpos);
                    db("&",xaddpos) = addval;

		    Gn.addRowCol(xaddpos);
		    gn.add(xaddpos);

                    retVector<double> tmpva;
                    retMatrix<double> tmpma;

		    Gn("&",xaddpos,0,1,GPNWIDTH(biasdim),tmpva) = 0.0;
		    Gn("&",0,1,Gn.numRows()-1,xaddpos,tmpma,"&") = Gn("&",xaddpos,0,1,GPNWIDTH(biasdim),tmpva);
//		    Gn.setCol(xaddpos,Gn("&",xaddpos,0,1,GPNWIDTH(biasdim)));
		    Gn("&",xaddpos,xaddpos) = quadbiasforceval;
		    gn("&",xaddpos) = linbiasforceval;

		    if ( isVarBias() )
		    {
			Q.addBeta(xaddpos,0,0.0);
		    }

		    else
		    {
			Q.addBeta(xaddpos,3,0.0);
		    }

		    biasdim++;
		}

		// If the target xbiasdim == 0 then we have one step remaining

		if ( xbiasdim == 0 )
		{
                    NiceAssert( biasdim == +1 );

		    // Need to add row/column to Gn

		    Gn.addRowCol(0);
		    gn.add(0);

		    Gn = quadbiasforceval;
		    gn = linbiasforceval;

		    if ( isVarBias() )
		    {
			Q.addBeta(0,0,0.0);
		    }

		    else
		    {
			Q.addBeta(0,3,0.0);
		    }

		    biasdim = 0;
		}

                retVector<double> tmpva;
                retMatrix<double> tmpma;
                retMatrix<double> tmpmb;

		// Refactorise

                Q.refact(*Gpval,*Gpval,Gn(zeroint(),1,GPNWIDTH(biasdim)-1,zeroint(),1,GPNWIDTH(biasdim)-1,tmpma),GPNorGPNEXT(Gpn,GpnExt)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,GPNWIDTH(biasdim)-1,tmpmb),gp,gn(zeroint(),1,GPNWIDTH(biasdim)-1,tmpva),hp);

		// Set bias

		Q.setBeta(db,*Gpval,*Gpval,Gn(zeroint(),1,GPNWIDTH(biasdim)-1,zeroint(),1,GPNWIDTH(biasdim)-1,tmpma),GPNorGPNEXT(Gpn,GpnExt)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,GPNWIDTH(biasdim)-1,tmpmb),gp,gn(zeroint(),1,GPNWIDTH(biasdim)-1,tmpva),hp);
	    }

	    else
	    {
		int iv,xrempos;

		while ( biasdim > ybiasdim )
		{
		    xrempos = ( rempos == -1 ) ? GPNWIDTH(biasdim)-1 : rempos;

		    Vector<int> rowuse(GPNWIDTH(biasdim));

		    for ( iv = 0 ; iv < GPNWIDTH(biasdim) ; iv++ )
		    {
                        rowuse("&",iv) = iv;
		    }

                    rowuse.remove(xrempos);

		    // Remove row/column from Gn

                    retMatrix<double> tmpma;

		    Q.changeBetaRestrict(xrempos,3,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt)(zeroint(),1,SVM_Scalar::N()-1,zeroint(),1,xrempos,tmpma),gp,gn,hp);

		    Gn.removeRowCol(xrempos);
		    gn.remove(xrempos);

		    Q.removeBeta(xrempos);

		    biasdim--;
		}

		// If the target xbiasdim == 0 then we have one step remaining

		if ( xbiasdim == 0 )
		{
                    NiceAssert( biasdim == +1 );

		    // Need to add row/column to Gn

		    Gn.addRowCol(0);
		    gn.add(0);

		    Gn = quadbiasforceval;
		    gn = linbiasforceval;

		    if ( isVarBias() )
		    {
			Q.addBeta(0,0,addval);
		    }

		    else
		    {
			Q.addBeta(0,3,addval);
		    }

		    biasdim = 0;
		}

		// Refactorise

                retMatrix<double> tmpma;

                Q.refact(*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt)(zeroint(),1,SVM_Scalar::N()-1,rempos+1,1,rempos+1+GPNWIDTH(biasdim)-1,tmpma),gp,gn,hp);
	    }
	}
    }

    SVM_Generic::basesetbias(biasR());

    return;
}

int SVM_Scalar::setLinBiasForce(int i, double newval)
{
    int res = 0;

    if ( i == -2 )
    {
        res = setLinBiasForce(newval);
    }

    else
    {
        NiceAssert( i >= 0 );
        NiceAssert( i < GPNWIDTH(biasdim) );

        isStateOpt = 0;

        Vector<double> gnnew(gn);

        gnnew("&",i) = newval;
        Q.refactgn(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,gnnew,hp);
        gn("&",i) = newval;
    }

    return res;
}

void SVM_Scalar::setBiasVMulti(const Vector<double> &newbias)
{
    isStateOpt = 0;

    Q.setBeta(newbias,*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

    SVM_Generic::basesetbias(biasR());

    return;
}

void SVM_Scalar::refactGpnElm(int i, int j, double GpnijNew)
{
    NiceAssert( GpnExt != NULL );
    NiceAssert( Q.alphaState(i) != +1 );
    NiceAssert( Q.alphaState(i) != -1 );

    Q.refactGpnElm(*Gpval,Gn,*GpnExt,GpnijNew,gp,gn,hp,i,j);

    return;
}

void SVM_Scalar::setGpnExt(Matrix<double> *GpnExtOld, Matrix<double> *GpnExtNew)
{
    if ( GpnExt == NULL )
    {
        NiceAssert( GpnExtOld == NULL );

	if ( GpnExtNew != NULL )
	{
	    // Currently: using Gpn default
	    // Subsequently: using GpnExtNew

            NiceAssert( GpnExtNew->numRows() == Gpn.numRows() );
            NiceAssert( GpnExtNew->numCols() == Gpn.numCols() );

	    GpnExt = GpnExtNew;

            Q.refactGpn(*Gpval,*Gpval,Gn,Gpn,*GpnExtNew,gp,gn,hp);
	}
    }

    else
    {
        NiceAssert( GpnExtOld != NULL );

	if ( GpnExtNew == NULL )
	{
	    // Currently: using GpnExtOld
	    // Subsequently: using Gpn default

            NiceAssert( GpnExtOld->numRows() == Gpn.numRows() );
            NiceAssert( GpnExtOld->numCols() == Gpn.numCols() );

	    GpnExt = NULL;

            Q.refactGpn(*Gpval,*Gpval,Gn,*GpnExtOld,Gpn,gp,gn,hp);
	}

	else
	{
	    // Currently: using GpnExtOld
	    // Subsequently: using GpnExtNew

            NiceAssert( GpnExtOld->numRows() == Gpn.numRows() );
            NiceAssert( GpnExtOld->numCols() == Gpn.numCols() );
            NiceAssert( GpnExtNew->numRows() == Gpn.numRows() );
            NiceAssert( GpnExtNew->numCols() == Gpn.numCols() );

	    GpnExt = GpnExtNew;

            Q.refactGpn(*Gpval,*Gpval,Gn,*GpnExtOld,*GpnExtNew,gp,gn,hp);
	}
    }

    return;
}

void SVM_Scalar::setgn(const Vector<double> &gnnew)
{
    NiceAssert( gn.size() == gnnew.size() );

    Q.refactgn(*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,gnnew,hp);
    gn = gnnew;

    return;
}

void SVM_Scalar::setGn(const Matrix<double> &Gnnew)
{
    NiceAssert( Gn.numRows() == Gnnew.numRows() );
    NiceAssert( Gn.numCols() == Gnnew.numCols() );

    Gn = Gnnew;
    //FIXME: should write and use a refactGn function to save cycles recalculating the gradients
    Q.refact(*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn,hp);

    return;
}

int SVM_Scalar::fixautosettings(int kernchange, int Nchange)
{
    int res = 0;

    if ( kernchange || Nchange )
    {
        switch ( autosetLevel )
	{
        case 1: { if ( Nchange ) { res = 1; SVM_Scalar::autosetCscaled(autosetCvalx);                    } break; }
        case 2: {                { res = 1; SVM_Scalar::autosetCKmean();                                 } break; }
        case 3: {                { res = 1; SVM_Scalar::autosetCKmedian();                               } break; }
        case 4: {                { res = 1; SVM_Scalar::autosetCNKmean();                                } break; }
        case 5: {                { res = 1; SVM_Scalar::autosetCNKmedian();                              } break; }
        case 6: { if ( Nchange ) { res = 1; SVM_Scalar::autosetLinBiasForce(autosetnuvalx,autosetCvalx); } break; }
	default: { break; }
	}
    }

    return res;
}

double SVM_Scalar::autosetkerndiagmean(void)
{
    Vector<int> dnonzero;

    if ( SVM_Scalar::N()-SVM_Scalar::NNC(0) )
    {
	int i,j = 0;

        for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
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

    return mean(kerndiagval(dnonzero,tmpva));
}

double SVM_Scalar::autosetkerndiagmedian(void)
{
    Vector<int> dnonzero;

    int i,j = 0;

    if ( SVM_Scalar::N()-SVM_Scalar::NNC(0) )
    {
        for ( i = 0 ; i < SVM_Scalar::N() ; i++ )
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

    return median(kerndiagval(dnonzero,tmpva),i);
}




// Stream operators

std::ostream &SVM_Scalar::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Scalar SVM\n\n";

    repPrint(output,'>',dep) << "Cost type (0 linear, 1 LS):      " << costType               << "\n";
    repPrint(output,'>',dep) << "Bias type (0 fixed, 1 var):      " << biasType               << "\n";
    repPrint(output,'>',dep) << "Opt type (0 act, 1 smo, 2 d2c, 3 grad):  " << optType                << "\n";
    repPrint(output,'>',dep) << "Tube shrinking (0 off, 1 on):    " << tubeshrink             << "\n";
    repPrint(output,'>',dep) << "Epsilon restriction:             " << epsrestrict            << "\n";
    repPrint(output,'>',dep) << "Monotonicity constraint:         " << makeConvex             << "\n";

    repPrint(output,'>',dep) << "Maximum training iterations:     " << maxitcntval            << "\n";
    repPrint(output,'>',dep) << "Maximum training time (sec):     " << maxtraintimeval        << "\n";
    repPrint(output,'>',dep) << "Optimal tolerance:               " << opttolval              << "\n";
    repPrint(output,'>',dep) << "4-norm SVM learning rate:        " << outerlrval             << "\n";
    repPrint(output,'>',dep) << "4-norm SVM momentum:             " << outermomval            << "\n";
    repPrint(output,'>',dep) << "4-norm SVM method:               " << outermethodval         << "\n";
    repPrint(output,'>',dep) << "4-norm SVM zero tolerance:       " << outertolval            << "\n";
    repPrint(output,'>',dep) << "4-norm SVM outer scaleback:      " << outerovscval           << "\n";
    repPrint(output,'>',dep) << "4-norm SVM max iterations:       " << outermaxits            << "\n";
    repPrint(output,'>',dep) << "4-norm SVM max cache:            " << outermaxcacheN         << "\n";

    repPrint(output,'>',dep) << "Default linear bias force:       " << linbiasforceval        << "\n";
    repPrint(output,'>',dep) << "Default quadratic bias force:    " << quadbiasforceval       << "\n";
    repPrint(output,'>',dep) << "nu (linear):                     " << nuLin                  << "\n";
    repPrint(output,'>',dep) << "nu (quadratic):                  " << nuQuadv                << "\n";
    repPrint(output,'>',dep) << "m:                               " << emm                    << "\n";
    repPrint(output,'>',dep) << "C:                               " << CNval                  << "\n";
    repPrint(output,'>',dep) << "eps:                             " << epsval                 << "\n";
    repPrint(output,'>',dep) << "Bias fixed value:                " << bfixval                << "\n";

    repPrint(output,'>',dep) << "C+-:                             " << xCclass                << "\n";
    repPrint(output,'>',dep) << "eps+-:                           " << xepsclass              << "\n";

    repPrint(output,'>',dep) << "Parameter autoset level:         " << autosetLevel           << "\n";
    repPrint(output,'>',dep) << "Parameter autoset nu value:      " << autosetnuvalx          << "\n";
    repPrint(output,'>',dep) << "Parameter autoset C value:       " << autosetCvalx           << "\n";

    repPrint(output,'>',dep) << "XY cache details:                " << xycache                << "\n";
    repPrint(output,'>',dep) << "Kernel cache details:            " << kerncache              << "\n";
    repPrint(output,'>',dep) << "Sigma cache details:             " << sigmacache             << "\n";
    repPrint(output,'>',dep) << "Kernel diagonals:                " << kerndiagval            << "\n";
    repPrint(output,'>',dep) << "Diagonal offsets:                " << diagoff                << "\n";

    repPrint(output,'>',dep) << "Gen cost max iteration count:    " << maxiterfuzztval        << "\n";
    repPrint(output,'>',dep) << "Gen cost mode:                   " << usefuzztval            << "\n";
    repPrint(output,'>',dep) << "Gen cost learning rate:          " << lrfuzztval             << "\n";
    repPrint(output,'>',dep) << "Gen cost zero tolerance:         " << ztfuzztval             << "\n";
    repPrint(output,'>',dep) << "Gen cost fuzzy function:         " << costfnfuzztval         << "\n";

    repPrint(output,'>',dep) << "Nnc:                             " << Nnc                    << "\n";
    repPrint(output,'>',dep) << "Is SVM optimal:                  " << isStateOpt             << "\n";

    SVM_Generic::printstream(output,dep+1);

    repPrint(output,'>',dep) << "Training targets:                " << traintarg              << "\n";
    repPrint(output,'>',dep) << "Training classes:                " << trainclass             << "\n";
    repPrint(output,'>',dep) << "Training C weights:              " << Cweightval             << "\n";
    repPrint(output,'>',dep) << "Training C weights fuzzing:      " << Cweightfuzzval         << "\n";
    repPrint(output,'>',dep) << "Training eps weights:            " << epsweightval           << "\n";

    repPrint(output,'>',dep) << "Gn:                              " << Gn                     << "\n";
    repPrint(output,'>',dep) << "Gpn:                             " << Gpn                    << "\n";
    repPrint(output,'>',dep) << "gp:                              " << gp                     << "\n";
    repPrint(output,'>',dep) << "gn:                              " << gn                     << "\n";
    repPrint(output,'>',dep) << "hp:                              " << hp                     << "\n";
    repPrint(output,'>',dep) << "hpscale:                         " << hpscale                << "\n";
    repPrint(output,'>',dep) << "lb:                              " << lb                     << "\n";
    repPrint(output,'>',dep) << "ub:                              " << ub                     << "\n";
    repPrint(output,'>',dep) << "cr:                              " << cr                     << "\n";
    repPrint(output,'>',dep) << "ddr:                             " << ddr                    << "\n";
    repPrint(output,'>',dep) << "wr:                              " << wr                     << "\n";
    repPrint(output,'>',dep) << "Qnp:                             " << Qnp                    << "\n";
    repPrint(output,'>',dep) << "Qn:                              " << Qn                     << "\n";
    repPrint(output,'>',dep) << "qn:                              " << qn                     << "\n";
    repPrint(output,'>',dep) << "alpharestrictoverride:           " << alpharestrictoverride  << "\n";
    repPrint(output,'>',dep) << "Qconstype                        " << Qconstype              << "\n";

    repPrint(output,'>',dep) << "biasdim:                         " << biasdim                << "\n";

    repPrint(output,'>',dep) << "*********************************************************************\n";
    repPrint(output,'>',dep) << "Optimisation state:              " << Q                      << "\n";
    repPrint(output,'>',dep) << "#####################################################################\n";

    return output;
}

std::istream &SVM_Scalar::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> costType;
    input >> dummy; input >> biasType;
    input >> dummy; input >> optType;
    input >> dummy; input >> tubeshrink;
    input >> dummy; input >> epsrestrict;
    input >> dummy; input >> makeConvex;

    input >> dummy; input >> maxitcntval;
    input >> dummy; input >> maxtraintimeval;
    input >> dummy; input >> opttolval;
    input >> dummy; input >> outerlrval;
    input >> dummy; input >> outermomval;
    input >> dummy; input >> outermethodval;
    input >> dummy; input >> outertolval;
    input >> dummy; input >> outerovscval;
    input >> dummy; input >> outermaxits;
    input >> dummy; input >> outermaxcacheN;

    input >> dummy; input >> linbiasforceval;
    input >> dummy; input >> quadbiasforceval;
    input >> dummy; input >> nuLin;
    input >> dummy; input >> nuQuadv;
    input >> dummy; input >> emm;
    input >> dummy; input >> CNval;
    input >> dummy; input >> epsval;
    input >> dummy; input >> bfixval;

    input >> dummy; input >> xCclass;
    input >> dummy; input >> xepsclass;

    input >> dummy; input >> autosetLevel;
    input >> dummy; input >> autosetnuvalx;
    input >> dummy; input >> autosetCvalx;

    input >> dummy; input >> xycache;
    input >> dummy; input >> kerncache;
    input >> dummy; input >> sigmacache;
    input >> dummy; input >> kerndiagval;
    input >> dummy; input >> diagoff;

    input >> dummy; input >> maxiterfuzztval;
    input >> dummy; input >> usefuzztval;
    input >> dummy; input >> lrfuzztval;
    input >> dummy; input >> ztfuzztval;
    input >> dummy; input >> costfnfuzztval;

    input >> dummy; input >> Nnc;
    input >> dummy; input >> isStateOpt;

    SVM_Generic::inputstream(input);

    input >> dummy; input >> traintarg;
    input >> dummy; input >> trainclass;
    input >> dummy; input >> Cweightval;
    input >> dummy; input >> Cweightfuzzval;
    input >> dummy; input >> epsweightval;

    input >> dummy; input >> Gn;
    input >> dummy; input >> Gpn;
    input >> dummy; input >> gp;
    input >> dummy; input >> gn;
    input >> dummy; input >> hp;
    input >> dummy; input >> hpscale;
    input >> dummy; input >> lb;
    input >> dummy; input >> ub;
    input >> dummy; input >> cr;
    input >> dummy; input >> ddr;
    input >> dummy; input >> wr;
    input >> dummy; input >> Qnp;
    input >> dummy; input >> Qn;
    input >> dummy; input >> qn;
    input >> dummy; input >> alpharestrictoverride;
    input >> dummy; input >> Qconstype;

    input >> dummy; input >> biasdim;

    input >> dummy; input >> Q;

    GpnExt = NULL;

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

    MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &xycache,   SVM_Scalar::N(),SVM_Scalar::N()));
    MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache, SVM_Scalar::N(),SVM_Scalar::N()));
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,SVM_Scalar::N(),SVM_Scalar::N()));

    int oldmemsize = kerncache.get_memsize();
    int oldrowdim  = kerncache.get_min_rowdim();

    xycache.reset(SVM_Scalar::N(),&evalxySVM_Scalar,this);
    xycache.setmemsize(MEMSHARE_XYCACHE(oldmemsize),oldrowdim);

    kerncache.reset(SVM_Scalar::N(),&evalKSVM_Scalar,this);
    kerncache.setmemsize(MEMSHARE_KCACHE(oldmemsize),oldrowdim);

    sigmacache.reset(SVM_Scalar::N(),&evalSigmaSVM_Scalar,this);
    sigmacache.setmemsize(MEMSHARE_SIGMACACHE(oldmemsize),oldrowdim);

    // NOTE: it is up to the user to fix GpnExt using naiveset at this point

    return input;
}

int SVM_Scalar::prealloc(int expectedN)
{
    kerndiagval.prealloc(expectedN);
    diagoff.prealloc(expectedN);
    traintarg.prealloc(expectedN);
    trainclass.prealloc(expectedN);
    Cweightval.prealloc(expectedN);
    Cweightfuzzval.prealloc(expectedN);
    epsweightval.prealloc(expectedN);
    gp.prealloc(expectedN);
    hp.prealloc(expectedN);
    hpscale.prealloc(expectedN);
    lb.prealloc(expectedN);
    ub.prealloc(expectedN);
    cr.prealloc(expectedN);
    ddr.prealloc(expectedN);
    wr.prealloc(expectedN);
    xycache.prealloc(expectedN);
    kerncache.prealloc(expectedN);
    sigmacache.prealloc(expectedN);
    // Gp is a cover (no actual content)
    // GpSigma is a cover (no actual content)
    Gn.prealloc(GPNWIDTH(biasdim),GPNWIDTH(biasdim));
    Gpn.prealloc(expectedN,GPNWIDTH(biasdim));
    SVM_Generic::prealloc(expectedN);

    return 0;
}

int SVM_Scalar::preallocsize(void) const
{
    return SVM_Generic::preallocsize();
}

int SVM_Scalar::randomise(double sparsity)
{
    NiceAssert( sparsity >= 0 );
    NiceAssert( sparsity <= 1 );

    int res = 0;
    int Nnotz = (int) (((double) (SVM_Scalar::N()-NNC(0)))*sparsity);

    if ( Nnotz )
    {
        res = 1;

        retVector<int> tmpva;

        Vector<int> canmod(cntintvec(SVM_Scalar::N(),tmpva));

        int i,j;

        for ( i = SVM_Scalar::N()-1 ; i >= 0 ; i-- )
        {
            if ( !d()(i) )
            {
                canmod.remove(i);
            }
        }

        // Observe sparsity

        while ( canmod.size() > Nnotz )
        {
            canmod.remove(svm_rand()%(canmod.size()));
        }

        // Need to randomise canmod alphas, set rest to zero
        // (need to take care as meaning of zero differs depending on goutType)

        Vector<double> newalpha(SVM_Scalar::N());

        // Set zero

        newalpha = 0.0;

        // Next randomise

        double lbloc;
        double ubloc;

        for ( i = 0 ; i < canmod.size() ; i++ )
        {
            j = canmod(i);

            double &amod = newalpha("&",j);

            lbloc = isLinearCost() ? lb(j) : -1.0;
            ubloc = isLinearCost() ? ub(j) : +1.0;

            NiceAssert( d()(j) );

            if ( d()(j) == +1 )
            {
                lbloc = 0.0;
            }

            if ( d()(j) == -1 )
            {
                ubloc = 0.0;
            }

            setrand(amod);
            amod = lbloc+((ubloc-lbloc)*amod);
        }

        // Lastly set alpha

        setAlphaR(newalpha);
        setBiasR(0);
    }

    return res;
}



// Kernel cache selective access for gradient calculation

double SVM_Scalar::getvalIfPresent(int i, int j, int &isgood) const
{
    double res = 0;

    isgood = 0;

    if ( ( i >= 0 ) && ( j >= 0 ) )
    {
        res = kerncache.getvalIfPresent(i,j,isgood);

        if ( isgood )
        {
            res -= ( ( i == j ) ? diagoff(i) : 0.0 );
        }
    }

    return res;
}
























int SVM_Scalar::maxFreeAlphaBias(void)
{
    return Q.modAllToDesthpzero(*Gpval,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt),gp,gn);
}

int SVM_Scalar::fact_minverse(Vector<double> &aAlpha, Vector<double> &aBeta, const Vector<double> &bAlpha, const Vector<double> &bBeta) const
{
    Q.fact_minverse(aAlpha,aBeta,bAlpha,bBeta,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt));

    return 0;
}

int SVM_Scalar::fact_minverse(Vector<gentype> &aAlpha, Vector<gentype> &aBeta, const Vector<gentype> &bAlpha, const Vector<gentype> &bBeta) const
{
    Q.fact_minverse(aAlpha,aBeta,bAlpha,bBeta,*Gpval,Gn,GPNorGPNEXT(Gpn,GpnExt));

    return 0;
}












