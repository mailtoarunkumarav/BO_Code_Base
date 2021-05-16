
//
// Multiclass classification SVM (all at once)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_multic_atonce.h"
#include "smatrix.h"
#include <iostream>
#include <sstream>
#include <string>

#define D_CALC(_xclass_,_s_)                   ( (_xclass_) ? (classRepval(label_placeholder.findID(_xclass_))(_s_)) : 0 )
#define QFIX(_xclass_,_q_)                     ( ( (_q_) < label_placeholder.findID(_xclass_) ) ? (_q_) : (_q_)-1 )
#define IQFIX(_xclass_,_i_,_q_)                ( ( (_i_) * ( numClasses() - 1 ) ) + ( ( (_q_) < label_placeholder.findID(_xclass_) ) ? (_q_) : (_q_)-1 ) )
#define MODPOS(_xclass_)                       ( (_xclass_) ? label_placeholder.findID(_xclass_) : 0 )

//#define C_CALC                                 ( isLinearCost() ? ( CNval *( ( isrecdiv() && ( numClasses() > 1 ) ) ? ( sqrt(((double) (numClasses()-1))/((double) numClasses())) ) : 1.0 ) ) : (MAXBOUND) )
//#define EPS_CALC                               (                  ( epsval*( ( isrecdiv() && ( numClasses() > 1 ) ) ? ( sqrt(((double) numClasses())/((double) (numClasses()-1))) ) : 1.0 ) )              )
//#define CWEIGH_CALC(_xclass_,_Cweigh_,_s_)     ( (_xclass_) ? (   (_Cweigh_)*  mulCclass(label_placeholder.findID(_xclass_))/(                                       ( isrecdiv() ? ( ( ( (label_placeholder.findID(_xclass_)) != (_s_) ) && ( numClasses() > 1 ) ) ? ((double) (numClasses()-1)) : 1.0 ) : ( ismaxwins() ? ( ( (label_placeholder.findID(_xclass_)) == (_s_) ) ? 1.0 : ( ( numClasses() > 1 ) ? (1.0/(numClasses()-1)) : 1.0 ) ) : 1.0 ) ) ) ) : 1.0 )
//#define EPSWEIGH_CALC(_xclass_,_epsweigh_,_s_) ( (_xclass_) ? ( (_epsweigh_)*mulepsclass(label_placeholder.findID(_xclass_))*( ( ( (_s_) == linbfq ) ? 1.0 : 0.0 ) + ( isrecdiv() ? ( ( ( (label_placeholder.findID(_xclass_)) == (_s_) )                         ) ? 1.0                         : 0.0 ) : ( ismaxwins() ? ( ( (label_placeholder.findID(_xclass_)) == (_s_) ) ? 1.0 : ( ( numClasses() > 1 ) ? (1.0/(numClasses()-1)) : 1.0 ) ) : 1.0 ) ) ) ) : 1.0 )

#define QUADCOSTDIAGOFFSET(_xclass_,_Cweigh_,_Cweighfuzz_)  ( (_xclass_) ? ( isQuadraticCost() ? (1/(CNval*(_Cweigh_)*(_Cweighfuzz_)*mulCclass(label_placeholder.findID(_xclass_)))) : 0.0 ) : 0.0 )
#define CALC_TSPACEDIM(_numclass_)                          ( (_numclass_) )
#define SIGMACUT 1e-6

#define MEMSHARE_KCACHE(_totmem_)     upidiv(( ( isOptActive() ? ( ( (_totmem_) > 0 ) ? (_totmem_) : 1 ) : ( ( upidiv((_totmem_),2) > 0 ) ? upidiv((_totmem_),2) : 1 ) ) ),2)
#define MEMSHARE_XYCACHE(_totmem_)     upidiv(( ( isOptActive() ? ( ( (_totmem_) > 0 ) ? (_totmem_) : 1 ) : ( ( upidiv((_totmem_),2) > 0 ) ? upidiv((_totmem_),2) : 1 ) ) ),2)
#define MEMSHARE_SIGMACACHE(_totmem_) upidiv(( ( isOptActive() ? 1                                       : ( ( upidiv((_totmem_),2) > 0 ) ? upidiv((_totmem_),2) : 1 ) ) ),2)

#define DEFAULT_DTHRES 0.0

double SVM_MultiC_atonce::C_CALC(void)
{
    double res = CNval;

    if ( !isLinearCost() )
    {
        res = MAXBOUND;
    }

    else if ( isrecdiv() && ( numClasses() > 1 ) )
    {
        res *= sqrt( ((double) (numClasses()-1)) / ((double) numClasses()) );
    }

    return res;
}

double SVM_MultiC_atonce::CWEIGH_CALC(int _xclass_, double _Cweigh_, double _Cweighfuzz_, int _s_, double _dthres_)
{
    // _xclass_ = y_i (training class of the vector in question)
    // _s_  = element of epsilon subvector under consideration.

    double res = _Cweigh_*_Cweighfuzz_*mulCclass(label_placeholder.findID(_xclass_));

    if ( _xclass_ && ( label_placeholder.findID(_xclass_) != _s_ ) && ( numClasses() > 1 ) )
    {
        res /= ((double) (numClasses()-1));
    }

    if ( ( _dthres_ > 0 ) && ( _dthres_ < 0.5 ) )
    {
        res *= ((1-_dthres_)/_dthres_);
    }

    return res;
}

/*
double SVM_MultiC_atonce::CWEIGH_CALCBASE(int _xclass_, double _Cweigh_, double _Cweighfuzz_, int _s_, double _dtrehs_)
{
    // _xclass_ = y_i (training class of the vector in question)
    // _s_  = element of epsilon subvector under consideration.

    double res = _Cweigh_*_Cweighfuzz_*mulCclass(label_placeholder.findID(_xclass_));

    if ( _xclass_ && ( label_placeholder.findID(_xclass_) != _s_ ) && ( numClasses() > 1 ) )
    {
        res /= ((double) (numClasses()-1));
    }

    return res;
}

double SVM_MultiC_atonce::CWEIGH_CALCEXTRA(int _xclass_, double _Cweigh_, double _Cweighfuzz_, int _s_, double _dtrehs_)
{
    // _xclass_ = y_i (training class of the vector in question)
    // _s_  = element of epsilon subvector under consideration.

    double res = _Cweigh_*_Cweighfuzz_*mulCclass(label_placeholder.findID(_xclass_));

    if ( _xclass_ && ( label_placeholder.findID(_xclass_) != _s_ ) && ( numClasses() > 1 ) )
    {
        res /= ((double) (numClasses()-1));
    }

    if ( ( _dthres_ > 0 ) && ( _dthres_ < 0.5 ) )
    {
        res *= (((1-_dthres_)/_dthres_)-1);
    }

    return res;
}
*/

double SVM_MultiC_atonce::EPS_CALC(void)
{
    double res = epsval;

    if ( isrecdiv() && ( numClasses() > 1 ) )
    {
        res *= sqrt( ((double) numClasses()) / ((double) (numClasses()-1)) );
    }

    else if ( ismaxwins() && ( numClasses() > 1 ) )
    {
        res *= ( ((double) numClasses()) / ((double) (numClasses()-1)) );
    }

    return res;
}

//#define EPSWEIGH_CALC(_xclass_,_epsweigh_,_s_) ( (_xclass_) ? ( (_epsweigh_)*mulepsclass(label_placeholder.findID(_xclass_))*( ( ( (_s_) == linbfq ) ? 1.0 : 0.0 ) + ( isrecdiv() ? ( ( ( (label_placeholder.findID(_xclass_)) == (_s_) )                         ) ? 1.0                         : 0.0 ) : ( ismaxwins() ? ( ( (label_placeholder.findID(_xclass_)) == (_s_) ) ? 1.0 : ( ( numClasses() > 1 ) ? (1.0/(numClasses()-1)) : 1.0 ) ) : 1.0 ) ) ) ) : 1.0 )
double SVM_MultiC_atonce::EPSWEIGH_CALC(int _xclass_, double _epsweigh_, int _s_)
{
    // _xclass_ = y_i (training class of the vector in question)
    // _s_  = element of epsilon subvector under consideration.
    // linbfq = y_b in the method paper

    double res = 0;

    if ( _xclass_ )
    {
        res += ( ( _s_ == linbfq                             ) ? 1 : 0 );
        res += ( ( _s_ == label_placeholder.findID(_xclass_) ) ? 1 : 0 );
    }

    else
    {
        res += 1;
    }

    res *= _epsweigh_*mulepsclass(label_placeholder.findID(_xclass_));




//    if ( _xclass_ )
//    {
//        res = ( _s_ == linbfq ) ? 1.0 : 0.0;
//
//        if ( ismaxwins() )
//        {
//            if ( label_placeholder.findID(_xclass_) == _s_ )
//            {
//                res += 1;
//            }
//
//            else
//            {
//                res += ( numClasses() > 1 ) ? 1.0/(numClasses()-1) : 1.0;
//            }
//        }
//
//        else
//        {
//            res += 1.0;
//        }
//
//        if ( isrecdiv() )
//        {
//            if ( label_placeholder.findID(_xclass_) == _s_ )
//            {
//                res += 1.0;
//            }
//
//            else
//            {
//                res += 0.0;
//            }
//        }
//
//        res *= _epsweigh_*mulepsclass(label_placeholder.findID(_xclass_));
//    }
//
//    else
//    {
//        res = 1.0;
//    }





//    double res = 0;
//
//
//    if ( _xclass_ && ( label_placeholder.findID(_xclass_) == _s_ ) && ( _s_ != linbfq ) )
//    {
//        res = _epsweigh_*mulepsclass(label_placeholder.findID(_xclass_));
//    }

    return res;
}




// In the individual binary SVMs C and eps are set using C_CALC and EPS_CALC,
// Cclass and epsclass are left as 1, and Cweight and epsweight are set using
// CWEIGH_CALC and EPSWEIGH_CALS.  Finally the actual class (-1,0,+1) used by
// the individual SVMs for a given training vector is set using D_CALC.
//
// Note that in the above _xclass_ is the class number given by the user and
// _s_ is the SVM number.


Vector<Vector<double> > &addclasstou(Vector<Vector<double> > &u);
Vector<Vector<double> > &addclasstou(Vector<Vector<double> > &u)
{
    int q;
    int nold = u.size();
    int nnew = nold+1;

    u.add(0);

    if ( nnew == 1 )
    {
	u("&",0).resize(0);
    }

    else if ( nnew == 2 )
    {
	u("&",0).resize(1); u("&",0)("&",0) = -1;
        u("&",1).resize(1); u("&",1)("&",0) = +1;
    }

    else
    {
        retVector<double> tmpva;

	u("&",0).resize(nnew-1);
	u("&",0)("&",0,1,nnew-3,tmpva).zero();
	u("&",0)("&",nnew-2) = -1;

	for ( q = 0 ; q < nold ; q++ )
	{
	    u("&",q+1) *= sqrt(((double) nnew)*(((double) nnew)-2));
	    u("&",q+1).add(nnew-2);
	    u("&",q+1)("&",nnew-2) = +1;
	    u("&",q+1) *= (1.0/(((double) nnew)-1));
	}
    }

    return u;
}


void evalKSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    SVM_MultiC_atonce *realOwner = (SVM_MultiC_atonce *) owner;

    NiceAssert( realOwner );

    int idiv = i/((realOwner->numClasses())-1);
    int imod = i%((realOwner->numClasses())-1);

    int jdiv = j/((realOwner->numClasses())-1);
    int jmod = j%((realOwner->numClasses())-1);

    if ( realOwner->idivsplit != -1 )
    {
        idiv = i/((realOwner->numclasspre)-1);
        imod = i%((realOwner->numclasspre)-1);

        if ( idiv >= realOwner->idivsplit )
	{
            i -= (realOwner->idivsplit)*((realOwner->numclasspre)-1);

            if ( i < ((realOwner->numclassat)-1) )
	    {
                imod = i%((realOwner->numclassat)-1);
	    }

	    else
	    {
                i -= ((realOwner->numclassat)-1);

                idiv = i/((realOwner->numclasspost)-1);
                imod = i%((realOwner->numclasspost)-1);

                idiv += (realOwner->idivsplit)+1;
	    }
	}

        jmod = j%((realOwner->numclasspre)-1);
        jdiv = j/((realOwner->numclasspre)-1);

        if ( jdiv >= realOwner->idivsplit )
	{
            j -= (realOwner->idivsplit)*((realOwner->numclasspre)-1);

            if ( j < ((realOwner->numclassat)-1) )
	    {
                jmod = j%((realOwner->numclassat)-1);
	    }

	    else
	    {
                j -= ((realOwner->numclassat)-1);

                jdiv = j/((realOwner->numclasspost)-1);
                jmod = j%((realOwner->numclasspost)-1);

                jdiv += (realOwner->idivsplit)+1;
	    }
	}
    }

    if ( realOwner->bartN )
    {
        // Quick redirect to avoid double-calculation when dealing with Bartlett's reject option!

        if ( idiv >= realOwner->bartN )
        {
//errstream() << "phantomxy 0ia: " << idiv << "\n";
            idiv = (realOwner->fakeredir)(idiv-(realOwner->bartN));
//errstream() << "phantomxy 0ib: " << idiv << "\n";
        }

        if ( jdiv >= realOwner->bartN )
        { 
//errstream() << "phantomxy 0ja: " << jdiv << "\n";
            jdiv = (realOwner->fakeredir)(jdiv-(realOwner->bartN));
//errstream() << "phantomxy 0jb: " << jdiv << "\n";
        }
    }

    if ( !(realOwner->pathcorrect) )
    {
	if ( idiv != jdiv )
	{
            //(realOwner->getKernel()).K2(res,(realOwner->x)(idiv),(realOwner->x)(jdiv),realOwner->xinfo()(idiv),realOwner->xinfo()(jdiv),idiv,jdiv,pxyprod);
            //Can run query through QA to minimise kernel calculations
realOwner->K2(res,idiv,jdiv,pxyprod);
            res = ((realOwner->QA).Gp())(idiv,jdiv);
	}

	else
	{
            res = (realOwner->kerndiag()(idiv))+(realOwner->diagoff(idiv));
	}

        res *= (realOwner->CMcl)(((realOwner->label_placeholder).findID(realOwner->trainclass(idiv)))+1,((realOwner->label_placeholder).findID(realOwner->trainclass(jdiv)))+1)(imod,jmod);
    }

    else
    {
        res = (realOwner->CCMcl)(((realOwner->label_placeholder).findID(realOwner->trainclass(idiv)))+1,((realOwner->label_placeholder).findID(realOwner->trainclass(jdiv)))+1)(imod,jmod);
    }

    return;
}



void evalXYSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    SVM_MultiC_atonce *realOwner = (SVM_MultiC_atonce *) owner;

    NiceAssert( realOwner );

    int idiv = i/((realOwner->numClasses())-1);
    int imod = i%((realOwner->numClasses())-1);

    int jdiv = j/((realOwner->numClasses())-1);
    int jmod = j%((realOwner->numClasses())-1);

    if ( realOwner->idivsplit != -1 )
    {
        idiv = i/((realOwner->numclasspre)-1);
        imod = i%((realOwner->numclasspre)-1);

        if ( idiv >= realOwner->idivsplit )
	{
            i -= (realOwner->idivsplit)*((realOwner->numclasspre)-1);

            if ( i < ((realOwner->numclassat)-1) )
	    {
                imod = i%((realOwner->numclassat)-1);
	    }

	    else
	    {
                i -= ((realOwner->numclassat)-1);

                idiv = i/((realOwner->numclasspost)-1);
                imod = i%((realOwner->numclasspost)-1);

                idiv += (realOwner->idivsplit)+1;
	    }
	}

        jmod = j%((realOwner->numclasspre)-1);
        jdiv = j/((realOwner->numclasspre)-1);

        if ( jdiv >= realOwner->idivsplit )
	{
            j -= (realOwner->idivsplit)*((realOwner->numclasspre)-1);

            if ( j < ((realOwner->numclassat)-1) )
	    {
                jmod = j%((realOwner->numclassat)-1);
	    }

	    else
	    {
                j -= ((realOwner->numclassat)-1);

                jdiv = j/((realOwner->numclasspost)-1);
                jmod = j%((realOwner->numclasspost)-1);

                jdiv += (realOwner->idivsplit)+1;
	    }
	}
    }

    if ( realOwner->bartN )
    {
        // Quick redirect to avoid double-calculation when dealing with Bartlett's reject option!

        if ( idiv >= realOwner->bartN )
        {
//errstream() << "phantomxy 0ia: " << idiv << "\n";
            idiv = (realOwner->fakeredir)(idiv-(realOwner->bartN));
//errstream() << "phantomxy 0ib: " << idiv << "\n";
        }

        if ( jdiv >= realOwner->bartN )
        { 
//errstream() << "phantomxy 0ja: " << jdiv << "\n";
            jdiv = (realOwner->fakeredir)(jdiv-(realOwner->bartN));
//errstream() << "phantomxy 0jb: " << jdiv << "\n";
        }
    }

(void) imod;
(void) jmod;

    {
        //(realOwner->getKernel()).K2(res,(realOwner->x)(idiv),(realOwner->x)(jdiv),realOwner->xinfo()(idiv),realOwner->xinfo()(jdiv),idiv,jdiv,pxyprod);
        //Can run query through QA to minimise kernel calculations
//realOwner->K2ip(res,idiv,jdiv,pxyprod);
        res = (((realOwner->QA).xy()))(idiv,jdiv);
    }

    return;
}



void evalsigmaSVM_MultiC_atonce(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    SVM_MultiC_atonce *realOwner = (SVM_MultiC_atonce *) owner;
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

SVM_MultiC_atonce::SVM_MultiC_atonce() : SVM_Generic()
{
    bartfake = 0;
    bartN = 0;

    setaltx(NULL);
    QA.setaltx(this);

    QA.setmemsize(upidiv(DEFAULT_MEMSIZE,2));
    Q.setmemsize(1);

    dthres = DEFAULT_DTHRES;

    anomalyd = -3;

    costType  = 0;
    multitype = 5;
    optType   = 0;

    CNval  = DEFAULT_C;
    epsval = 1.0;

    linbfd          = 0;
    linbfq          = -1;
    linbiasforceset = 0;

    Ns = 0;

    Nnc.resize(1);
    Nnc = zeroint();

    isStateOpt = 1;

    idivsplit    = -1;
    numclasspre  = 0;
    numclassat   = 0;
    numclasspost = 0;

    autosetLevel  = 0;
    autosetnuvalx = 0.0;
    autosetCvalx  = 0.0;

    xycache.reset(0,&evalXYSVM_MultiC_atonce,(void *) this);
    xycache.setmemsize(MEMSHARE_XYCACHE(DEFAULT_MEMSIZE),MINROWDIM);

    kerncache.reset(0,&evalKSVM_MultiC_atonce,(void *) this);
    kerncache.setmemsize(MEMSHARE_KCACHE(DEFAULT_MEMSIZE),MINROWDIM);

    sigmacache.reset(0,&evalsigmaSVM_MultiC_atonce,(void *) this);
    sigmacache.setmemsize(MEMSHARE_SIGMACACHE(DEFAULT_MEMSIZE),MINROWDIM);

    MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &xycache   ,0,0));
    MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,0,0));
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,0,0));

    Gpn.addCol(0);
    locnaivesetGpnExt();
    Q.setbiasdim(1);
    Gpn.removeCol(0);
    locsetGp();

    pathcorrect = 0;

    return;
}

SVM_MultiC_atonce::SVM_MultiC_atonce(const SVM_MultiC_atonce &src) : SVM_Generic()
{
    bartfake = 0;
    bartN = 0;

    setaltx(NULL);
    QA.setaltx(this);

    //Q.assign(src.Q,0);

    xyval   = NULL;
    Gpval   = NULL;
    Gpsigma = NULL;

    pathcorrect = 0;

    assign(src,0);

    return;
}

SVM_MultiC_atonce::SVM_MultiC_atonce(const SVM_MultiC_atonce &src, const ML_Base *xsrc) : SVM_Generic()
{
    bartfake = 0;
    bartN = 0;

    setaltx(xsrc);
    QA.setaltx(this); // NB: QA must point back to here, as *this is overall data store

    //Q.assign(src.Q,1);

    xyval   = NULL;
    Gpval   = NULL;
    Gpsigma = NULL;

    pathcorrect = 0;

    assign(src,1);

    return;
}

SVM_MultiC_atonce::~SVM_MultiC_atonce()
{
    if ( Gpval != NULL )
    {
        MEMDEL(xyval);
        MEMDEL(Gpval);
        MEMDEL(Gpsigma);

        xyval = NULL;
        Gpval = NULL;
        Gpsigma = NULL;
    }

    return;
}

int SVM_MultiC_atonce::scale(double a)
{
    NiceAssert( a >= 0.0 );
    NiceAssert( a <= 1.0 );

    isStateOpt = 0;

    int i;

    int res = 0;

    res |= QA.scale(a);
    res |= Q.scale(a);

    db        *= a;
    dbReduced *= a;

    if ( numClasses() && N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            dalpha("&",i)        *= a;
            dalphaReduced("&",i) *= a;
	}

	if ( a == 0.0 )
	{
            dalphaState("&",i) = 0;
	}
    }

    SVM_Generic::basescalealpha(a);
    SVM_Generic::basescalebias(a);

    return res;
}

double SVM_MultiC_atonce::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int SVM_MultiC_atonce::reset(void)
{
    int res = 0;

    res |= QA.reset();
    res |= Q.reset();

    dalpha.zero();
    dalphaReduced.zero();
    dalphaState.zero();
    db.zero();
    dbReduced.zero();

    Ns = 0;
    isStateOpt = 0;

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

int SVM_MultiC_atonce::setAlphaV(const Vector<Vector<double> > &newAlpha)
{
    NiceAssert( newAlpha.size() == N() );

    if ( N() && numClasses() )
    {
        isStateOpt = 0;

	int i,q;

	Vector<double> localpha(N()*(numClasses()-1));

	for ( q = 0 ; q < numClasses() ; q++ )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                NiceAssert( newAlpha(i).size() == numClasses() );

                if ( q != MODPOS(trainclass(i)) )
		{
                    localpha("&",IQFIX(trainclass(i),i,q)) = newAlpha(i)(q);
		}
	    }
	}

        Q.setAlphaR(localpha);

        dalphaState.zero();
        dalpha.zero();

	for ( q = 0 ; q < numClasses() ; q++ )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                if ( q != MODPOS(trainclass(i)) )
		{
                    dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                    dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);

                    if ( (Q.alphaState())(IQFIX(trainclass(i),i,q)) )
		    {
                        dalphaState("&",i) = 1;
		    }
		}
	    }
	}

        Ns = sum(dalphaState);

        calcAlphaReduced();
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

int SVM_MultiC_atonce::setBiasV(const Vector<double> &newBias)
{
    NiceAssert( newBias.size() == numClasses() );

    if ( numClasses() )
    {
        isStateOpt = 0;

	int q;

	if ( isrecdiv() )
	{
            retVector<double> tmpva;

            Q.setBiasVMulti(newBias(1,1,newBias.size()-1,tmpva));

            db.zero();

	    for ( q = 0 ; q < numClasses() ; q++ )
	    {
		if ( q != 0 )
		{
                    db("&",q) = Q.biasVMulti(q-1);
                    db("&",0) -= db(q);
		}
	    }
	}

	else
	{
            Q.setBiasVMulti(newBias);

            db.zero();

	    for ( q = 0 ; q < numClasses() ; q++ )
	    {
                db("&",q) = Q.biasVMulti(q);
	    }
	}

        calcBiasReduced();
    }

    SVM_Generic::basesetbias(biasV());

    return 1;
}

int SVM_MultiC_atonce::setLinearCost(void)
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

int SVM_MultiC_atonce::setQuadraticCost(void)
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

int SVM_MultiC_atonce::setC(double xC)
{
    NiceAssert( xC > 0 );

    autosetOff();

    int res = 0;

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

    Q.setC(C_CALC());

    if ( N() && numClasses() )
    {
        dalpha.zero();

	for ( q = 0 ; q < numClasses() ; q++ )
	{
	    for ( i = 0 ; i < N() ; i++ )
	    {
                if ( q != MODPOS(trainclass(i)) )
		{
                    dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                    dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);
		}
	    }
	}

        calcAlphaReduced();
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

int SVM_MultiC_atonce::setCclass(int nc, double xC)
{
    NiceAssert( xC > 0 );
    NiceAssert( nc );
    NiceAssert( nc >= -1 );
    NiceAssert( label_placeholder.findID(nc) >= 0 );

    int i,q;
    int res = 0;

    mulCclass("&",label_placeholder.findID(nc)) = xC;

    if ( NNC(nc) && numClasses() )
    {
        isStateOpt = 0;
        res = 1;

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
                    dalpha("&",i).zero();

		    for ( q = 0 ; q < numClasses() ; q++ )
		    {
                        if ( q != MODPOS(trainclass(i)) )
			{
                            Q.setCweight(IQFIX(trainclass(i),i,q),CWEIGH_CALC(trainclass(i),Cweightval(i),Cweightvalfuzz(i),q,dthres));

                            dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                            dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);
			}
		    }
		}
	    }
	}

        calcAlphaReduced();
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

int SVM_MultiC_atonce::scaleCweight(double scalefactor)
{
    NiceAssert( scalefactor > 0 );

    int res = 0;

    if ( N() )
    {
        res = 1;
        isStateOpt = 0;
    }

    int i,q;

    Cweightval *= scalefactor;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else if ( numClasses() )
    {
	for ( q = 0 ; q < numClasses() ; q++ )
	{
            Q.scaleCweight(scalefactor);
	}

	if ( N() && numClasses() )
	{
            dalpha.zero();

	    for ( q = 0 ; q < numClasses() ; q++ )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    if ( q != MODPOS(trainclass(i)) )
		    {
                        dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                        dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);
		    }
		}
	    }

	    calcAlphaReduced();
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

int SVM_MultiC_atonce::scaleCweightfuzz(double scalefactor)
{
    NiceAssert( scalefactor > 0 );

    int res = 0;

    if ( N() )
    {
        res = 1;
        isStateOpt = 0;
    }

    int i,q;

    Cweightvalfuzz *= scalefactor;

    if ( isQuadraticCost() )
    {
	recalcdiagoff(-1);
    }

    else if ( numClasses() )
    {
	for ( q = 0 ; q < numClasses() ; q++ )
	{
            Q.scaleCweightfuzz(scalefactor);
	}

	if ( N() && numClasses() )
	{
            dalpha.zero();

	    for ( q = 0 ; q < numClasses() ; q++ )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    if ( q != MODPOS(trainclass(i)) )
		    {
                        dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                        dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);
		    }
		}
	    }

	    calcAlphaReduced();
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

int SVM_MultiC_atonce::seteps(double xeps)
{
    NiceAssert( xeps >= 0 );

    if ( autosetLevel == 6 )
    {
	autosetOff();
    }

    if ( N() )
    {
        isStateOpt = 0;
    }

    epsval = xeps;



    return Q.seteps(EPS_CALC());
}

int SVM_MultiC_atonce::setepsclass(int nc, double xeps)
{
    NiceAssert( xeps >= 0 );
    NiceAssert( nc );
    NiceAssert( nc >= -1 );
    NiceAssert( label_placeholder.findID(nc) >= 0 );

    int i,q;
    int res = 0;

    mulepsclass("&",label_placeholder.findID(nc)) = xeps;

    if ( NNC(nc) && numClasses() )
    {
        isStateOpt = 0;

	for ( i = 0 ; i < N() ; i++ )
	{
            if ( nc == trainclass(i) )
	    {
		for ( q = 0 ; q < numClasses() ; q++ )
		{
                    if ( q != MODPOS(trainclass(i)) )
		    {
                        res |= Q.setepsweight(IQFIX(trainclass(i),i,q),-D_CALC(trainclass(i),q)*((D_CALC(trainclass(i),q)*EPSWEIGH_CALC(trainclass(i),epsweightval(i),q))+(D_CALC(trainclass(i),MODPOS(trainclass(i)))*EPSWEIGH_CALC(trainclass(i),epsweightval(i),MODPOS(trainclass(i))))));
		    }
		}
	    }
	}
    }

    return res;
}

int SVM_MultiC_atonce::scaleepsweight(double scalefactor)
{
    if ( N() )
    {
        isStateOpt = 0;
    }

    epsweightval *= scalefactor;

    Q.scaleepsweight(scalefactor);

    return 0;
}

void SVM_MultiC_atonce::setmemsize(int memsize)
{
    NiceAssert( memsize > 0 );

    xycache.setmemsize(   MEMSHARE_XYCACHE(   memsize),   xycache.get_min_rowdim());
    kerncache.setmemsize( MEMSHARE_KCACHE(    memsize), kerncache.get_min_rowdim());
    sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize),sigmacache.get_min_rowdim());

    QA.setmemsize(upidiv(memsize,2)); // note that raw Gp stored at this level
    Q.setmemsize(1); // all caching done at this level, not at Q level

    return;
}

int SVM_MultiC_atonce::setOptActive(void)
{
    QA.setOptActive();

    if ( !isOptActive() )
    {
        optType = 0;

        Q.setOptActive();
    }

    setmemsize(memsize());

    return 0;
}

int SVM_MultiC_atonce::setOptSMO(void)
{
    QA.setOptSMO();

    if ( !isOptSMO() )
    {
        optType = 1;

        Q.setOptSMO();
    }

    setmemsize(memsize());

    return 0;
}

int SVM_MultiC_atonce::setOptD2C(void)
{
    QA.setOptD2C();

    if ( !isOptD2C() )
    {
        optType = 2;

        Q.setOptD2C();
    }

    setmemsize(memsize());

    return 0;
}

int SVM_MultiC_atonce::setOptGrad(void)
{
    QA.setOptGrad();

    if ( !isOptGrad() )
    {
        optType = 3;

        Q.setOptGrad();
    }

    setmemsize(memsize());

    return 0;
}

int SVM_MultiC_atonce::setzerotol(double zt)
{
    NiceAssert( zt > 0 );

    isStateOpt = 0;

    QA.setzerotol(zt);
    Q.setzerotol(zt);

    return 0;
}

int SVM_MultiC_atonce::setOpttol(double xopttol)
{
    NiceAssert( xopttol > 0 );

    isStateOpt = 0;

    QA.setOpttol(xopttol);
    Q.setOpttol(xopttol);

    return 0;
}

int SVM_MultiC_atonce::setmaxitcnt(int maxitcnt)
{
    NiceAssert( maxitcnt >= 0 );

    QA.setmaxitcnt(maxitcnt);
    Q.setmaxitcnt(maxitcnt);

    return 0;
}

int SVM_MultiC_atonce::setmaxtraintime(double maxtraintime)
{
    NiceAssert( maxtraintime >= 0 );

    QA.setmaxtraintime(maxtraintime);
    Q.setmaxtraintime(maxtraintime);

    return 0;
}

int SVM_MultiC_atonce::sety(int i, const gentype &y)
{
    NiceAssert( y.isCastableToIntegerWithoutLoss() );



    return setd(i, (int) y);
}

int SVM_MultiC_atonce::sety(const Vector<int> &j, const Vector<gentype> &yn)
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

int SVM_MultiC_atonce::sety(const Vector<gentype> &yn)
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

int SVM_MultiC_atonce::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    int res = 0;
    int fixxycache = getKernel().isIPdiffered();

    QA.setKernel(getKernel()); // Need to set kernel as getKernel only changes kernel in SVM_Generic, not QA
    res |= SVM_Generic::resetKernel(modind,onlyChangeRowI,updateInfo);

    if ( fixxycache )
    {
        xycache.setSymmetry(1);
    }

    kerncache.setSymmetry(getKernel().getSymmetry());
    sigmacache.setSymmetry(1);

    if ( N() && ( onlyChangeRowI == -1 ) )
    {
        res |= 1;
        //res |= QA.resetKernel(modind,onlyChangeRowI);

        isStateOpt = 0;

//	int i;

//	for ( i = 0 ; i < N() ; i++ )
//	{
//            K2(kerndiagval("&",i),i,i);
//	}

        if ( fixxycache )
        {
            xycache.clear();
        }

        kerncache.clear();
        sigmacache.clear();

        locsetGp();

        res |= fixautosettings(1,0);
    }

    else if ( onlyChangeRowI >= 0 )
    {
        res |= 1;
        //res |= QA.resetKernel(modind,onlyChangeRowI);

        int q;

        if ( alphaState()(onlyChangeRowI) )
        {
            isStateOpt = 0;
        }

        int dstval = trainclass(onlyChangeRowI);

        setd(onlyChangeRowI,0);

//        K2(kerndiagval("&",onlyChangeRowI),onlyChangeRowI,onlyChangeRowI);

        if ( fixxycache )
        {
            xycache.recalc(onlyChangeRowI);
        }

        kerncache.recalc(onlyChangeRowI);
        sigmacache.recalc(onlyChangeRowI);

        setd(onlyChangeRowI,dstval);

        if ( numClasses() )
        {
            for ( q = 0 ; q < numClasses() ; q++ )
            {
                if ( q != MODPOS(trainclass(onlyChangeRowI)) )
                {
                    Q.resetKernel(modind,IQFIX(trainclass(onlyChangeRowI),onlyChangeRowI,q),updateInfo);
                }
            }
        }

        res |= fixautosettings(1,0);
    }

    getKernel_unsafe().setIPdiffered(0);

    return res;
}

int SVM_MultiC_atonce::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    int res = 0;

    res |= QA.setKernel(xkernel,modind,onlyChangeRowI);
    res |= SVM_Generic::setKernel(xkernel,modind,onlyChangeRowI);

    xycache.setSymmetry(1);
    kerncache.setSymmetry(getKernel().getSymmetry());
    sigmacache.setSymmetry(1);

    if ( N() && ( onlyChangeRowI == -1 ) )
    {
        res |= 1;
        //res |= QA.setKernel(xkernel,modind,onlyChangeRowI);

        isStateOpt = 0;

//	int i;

//	for ( i = 0 ; i < N() ; i++ )
//	{
//            K2(kerndiagval("&",i),i,i);
//	}

        xycache.clear();
        kerncache.clear();
        sigmacache.clear();

        locsetGp();

        res |= fixautosettings(1,0);
    }

    else if ( onlyChangeRowI >= 0 )
    {
        res |= 1;
        //res |= QA.resetKernel(modind,onlyChangeRowI);

        int q;

        if ( alphaState()(onlyChangeRowI) )
        {
            isStateOpt = 0;
        }

        int dstval = trainclass(onlyChangeRowI);

        setd(onlyChangeRowI,0);

//        K2(kerndiagval("&",onlyChangeRowI),onlyChangeRowI,onlyChangeRowI);

        xycache.recalc(onlyChangeRowI);
        kerncache.recalc(onlyChangeRowI);
        sigmacache.recalc(onlyChangeRowI);

        setd(onlyChangeRowI,dstval);

        if ( numClasses() )
        {
            for ( q = 0 ; q < numClasses() ; q++ )
            {
                if ( q != MODPOS(trainclass(onlyChangeRowI)) )
                {
                    Q.resetKernel(modind,IQFIX(trainclass(onlyChangeRowI),onlyChangeRowI,q));
                }
            }
        }

        res |= fixautosettings(1,0);
    }

    getKernel_unsafe().setIPdiffered(0);

    return res;
}

int SVM_MultiC_atonce::setd(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( d >= -1 );

    int q;
    int res = 0;

    if ( d != trainclass(i) )
    {
        res = 1;

        if ( d )
        {
            isStateOpt = 0;

            // Important: we can't just swap x out of the training set
            // and then qadd it back like the code used to.  This will only
            // work if the relevant components are in the kernel cache: if
            // they are not then they will be recalculated on the fly, which
            // will give the wrong result if the training vector has been
            // swapped out.

            if ( trainclass(i) )
            {
                isStateOpt = 0;
            }

            trainclass("&",i) = d;
            gentype yn(trainclass(i));
            res |= SVM_Generic::sety(i,yn);

            for ( q = numClasses()-1 ; q >= 0 ; q-- )
            {
                if ( q != MODPOS(trainclass(i)) )
                {
                    res |= Q.setd(IQFIX(trainclass(i),i,q),0);
                }
            }

            res |= resetKernel(1,i,0);
        }

        else
        {
            // The first version is tedious but necessary.  The next
            // version is fast but only applicable if d == 0

            res |= naivesetdzero(i);
            res |= fixautosettings(0,1);
        }
    }

    return res;
}

int SVM_MultiC_atonce::setCweight(int i, double xCweight)
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

    else if ( numClasses() )
    {
        dalpha("&",i).zero();

	for ( q = 0 ; q < numClasses() ; q++ )
	{
            if ( q != MODPOS(trainclass(i)) )
	    {
                Q.setCweight(IQFIX(trainclass(i),i,q),CWEIGH_CALC(trainclass(i),Cweightval(i),Cweightvalfuzz(i),q,dthres));

                dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);
	    }
	}

        calcAlphaReduced(i);

        SVM_Generic::basesetalpha(i,alphaV()(i));
    }

    return 1;
}

int SVM_MultiC_atonce::setCweightfuzz(int i, double xCweight)
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

    else if ( numClasses() )
    {
        dalpha("&",i).zero();

	for ( q = 0 ; q < numClasses() ; q++ )
	{
            if ( q != MODPOS(trainclass(i)) )
	    {
                Q.setCweight(IQFIX(trainclass(i),i,q),CWEIGH_CALC(trainclass(i),Cweightval(i),Cweightvalfuzz(i),q,dthres));

                dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);
	    }
	}

        calcAlphaReduced(i);

        SVM_Generic::basesetalpha(i,alphaV()(i));
    }

    return 1;
}

int SVM_MultiC_atonce::setepsweight(int i, double xepsweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( xepsweight >= 0 );

    isStateOpt = 0;

    int q;
    int res = 0;

    epsweightval("&",i) = xepsweight;

    if ( numClasses() )
    {
	for ( q = 0 ; q < numClasses() ; q++ )
	{
            if ( q != MODPOS(trainclass(i)) )
	    {
                res |= Q.setepsweight(IQFIX(trainclass(i),i,q),-D_CALC(trainclass(i),q)*((D_CALC(trainclass(i),q)*EPSWEIGH_CALC(trainclass(i),epsweightval(i),q))+(D_CALC(trainclass(i),MODPOS(trainclass(i)))*EPSWEIGH_CALC(trainclass(i),epsweightval(i),MODPOS(trainclass(i))))));
	    }
	}
    }

    return res;
}

int SVM_MultiC_atonce::setd(const Vector<int> &j, const Vector<int> &d)
{
    NiceAssert( d.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        res = 1;

        if ( d != zeroint() )
        {
            int i;

            for ( i = 0 ; i < j.size() ; i++ )
            {
                res |= setd(j(i),d(i));
            }
        }

        else
        {
            int i;

            for ( i = 0 ; i < j.size() ; i++ )
            {
                res |= naivesetdzero(j(i));
            }

            res |= fixautosettings(0,1);
        }
    }

    return res;
}

int SVM_MultiC_atonce::setCweight(const Vector<int> &j, const Vector<double> &xCweight)
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

int SVM_MultiC_atonce::setCweightfuzz(const Vector<int> &j, const Vector<double> &xCweight)
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

int SVM_MultiC_atonce::setepsweight(const Vector<int> &j, const Vector<double> &xepsweight)
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

int SVM_MultiC_atonce::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == N() );

    int res = 0;

    if ( N() )
    {
        res = 1;

        if ( d != zeroint() )
        {
            int i;

            for ( i = 0 ; i < N() ; i++ )
            {
                res |= setd(i,d(i));
            }
        }

        else
        {
            int i;

            for ( i = 0 ; i < N() ; i++ )
            {
                res |= naivesetdzero(i);
            }

            res |= fixautosettings(0,1);
        }
    }

    return res;
}

int SVM_MultiC_atonce::setCweight(const Vector<double> &xCweight)
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

int SVM_MultiC_atonce::setCweightfuzz(const Vector<double> &xCweight)
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

int SVM_MultiC_atonce::setepsweight(const Vector<double> &xepsweight)
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

int SVM_MultiC_atonce::setLinBiasForce(int d, double newval)
{
    NiceAssert( d == linbfd );

    (void) d;

    if ( autosetLevel == 6 )
    {
	autosetOff();
    }

    NiceAssert( linbfq >= 0 );
    NiceAssert( linbfq < numClasses() );

    linbiasforceval = 0.0;
    linbiasforceval("&",linbfq) = newval;
    linbiasforceset = ( linbiasforceval == 0.0 ) ? 0 : 1;

    Vector<double> gn(linbiasforceval);

    if ( isrecdiv() )
    {
        linbiasforceval("&",linbfq) *= sqrt(((double) numClasses())/((double) (numClasses()-1)));

        retVector<double> tmpva;

        Vector<double> linbiasforcered(linbiasforceval(1,1,linbiasforceval.size()-1,tmpva));

        linbiasforcered -= linbiasforceval(zeroint());

        Q.setgn(linbiasforcered);

        gn = linbiasforcered;
    }

    else
    {
        NiceAssert( ismaxwins() );

        Q.setgn(linbiasforceval);
    }

    return 0;
}

int SVM_MultiC_atonce::setmaxwins(void)
{
    int i;

    if ( !ismaxwins() )
    {
	if ( N() )
	{
            isStateOpt = 0;
	}

        multitype = 4;

        fixCMcl(numClasses(),1);

	if ( numClasses() >= 1 )
	{
	    // Add a column to Gpn

            Gpn.addCol(numClasses()-1);

            // Reset Gpn

	    if ( N() )
	    {
                retMatrix<double> tmpma;
                retMatrix<double> tmpmb;

		for ( i = 0 ; i < N() ; i++ )
		{
                    Gpn("&",(i*(numClasses()-1)),1,((i+1)*(numClasses()-1))-1,0,1,numClasses()-1,tmpma) = CMcl(label_placeholder.findID(trainclass(i))+1,0)(0,1,numClasses()-2,0,1,numClasses()-1,tmpmb);

                    if ( xisrankorgrad(i) )
                    {
                        Gpn("&",(i*(numClasses()-1)),1,((i+1)*(numClasses()-1))-1,0,1,numClasses()-1,tmpma) *= 0.0;
                    }
		}
	    }

            // Set biasdim (this also updates the Gpn factorisation)

            Q.setbiasdim(numClasses()+1,0,db(zeroint()));

	    // Fix bias forcing terms

            if ( numClasses() && linbiasforceset )
	    {
                Q.setgn(linbiasforceval);
	    }
	}

	calcBiasReduced();
        resetCandeps();
    }

    SVM_Generic::basesetbias(biasV());

    return 1;
}

int SVM_MultiC_atonce::setrecdiv(void)
{
    int i;

    if ( !isrecdiv() )
    {
	if ( N() )
	{
            isStateOpt = 0;
	}

        multitype = 5;

        fixCMcl(numClasses(),1);

	if ( numClasses() >= 1 )
	{
            // Reset Gpn (excluding last column)

	    if ( N() )
	    {
                retMatrix<double> tmpma;
                retMatrix<double> tmpmb;

		for ( i = 0 ; i < N() ; i++ )
		{
                    Gpn("&",(i*(numClasses()-1)),1,((i+1)*(numClasses()-1))-1,1,1,numClasses()-1,tmpma) = CMcl(label_placeholder.findID(trainclass(i))+1,1)(0,1,numClasses()-2,0,1,numClasses()-2,tmpmb);

                    if ( xisrankorgrad(i) )
                    {
                        Gpn("&",(i*(numClasses()-1)),1,((i+1)*(numClasses()-1))-1,1,1,numClasses()-1,tmpma)  *= 0.0;
                    }
		}
	    }

	    // Set biasdim (this also updates the Gpn factorisation)

            Q.setbiasdim(numClasses(),-1,0.0,0);

            retVector<double> tmpva;

            db("&",0) = -sum(db(1,1,db.size()-1,tmpva));

            // Remove the column from Gpn

            Gpn.removeCol(0);

	    // Fix bias forcing terms

            if ( ( numClasses() > 1 ) && linbiasforceset )
	    {
                Vector<double> gn(Q.getgn());

                retVector<double> tmpva;

                gn = linbiasforceval(1,1,linbiasforceval.size()-1,tmpva);
                gn -= linbiasforceval(zeroint());

                Q.setgn(gn);
	    }
	}

	calcBiasReduced();
        resetCandeps();
    }

    SVM_Generic::basesetbias(biasV());

    return 1;
}

void SVM_MultiC_atonce::resetCandeps(void)
{
    int i,q;

    // Set C

    if ( isQuadraticCost() )
    {
	// Set C and Cweight quadratic cost

	recalcdiagoff(-1);
    }

    else
    {
	// set C linear cost

        Q.setC(C_CALC());

        // set Cweight linear cost

	if ( N() && numClasses() )
	{
            dalpha.zero();

	    for ( q = 0 ; q < numClasses() ; q++ )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    if ( q != MODPOS(trainclass(i)) )
		    {
                        Q.setCweight(IQFIX(trainclass(i),i,q),CWEIGH_CALC(trainclass(i),Cweightval(i),Cweightvalfuzz(i),q,dthres));

                        dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                        dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);
		    }
		}
	    }

	    calcAlphaReduced();
	}
    }

    // set epsilon

    Q.seteps(EPS_CALC());

    if ( numClasses() && N() )
    {
	// set epsilon weights

	for ( i = 0 ; i < N() ; i++ )
	{
	    for ( q = 0 ; q < numClasses() ; q++ )
	    {
                if ( q != MODPOS(trainclass(i)) )
		{
                    Q.setepsweight(IQFIX(trainclass(i),i,q),-D_CALC(trainclass(i),q)*((D_CALC(trainclass(i),q)*EPSWEIGH_CALC(trainclass(i),epsweightval(i),q))+(D_CALC(trainclass(i),MODPOS(trainclass(i)))*EPSWEIGH_CALC(trainclass(i),epsweightval(i),MODPOS(trainclass(i))))));
		}
	    }
	}
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return;
}

int SVM_MultiC_atonce::addTrainingVector(int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{


    return SVM_MultiC_atonce::addTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int SVM_MultiC_atonce::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{


    return SVM_MultiC_atonce::qaddTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int SVM_MultiC_atonce::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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



    return SVM_MultiC_atonce::addTrainingVector(i,zz,x,Cweigh,epsweigh);
}

int SVM_MultiC_atonce::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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



    return SVM_MultiC_atonce::qaddTrainingVector(i,zz,x,Cweigh,epsweigh);
}

int SVM_MultiC_atonce::removeTrainingVector(int i, gentype &yy, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int q;
    int y = trainclass(i);
    int res = 0;

    if ( y )
    {
        isStateOpt = 0;
    }

    for ( q = numClasses()-1 ; q >= 0 ; q-- )
    {
	if ( q != MODPOS(y) )
	{
            res |= Q.setd(IQFIX(y,i,q),0);
	}
    }

    Nnc("&",(label_placeholder.findID(y)+1))--;

//    kerndiagval.remove(i);
    diagoff.remove(i);

    dalpha.remove(i);
    dalphaReduced.remove(i);
    dalphaState.remove(i);

    if ( !bartN )
    {
        // Bartlett duplicates are aliased back to first bartN points before QA.Gp() is
        // used for lookup, so for optimality we don't need to add such duplicates
        // to QA at all.
        res |= QA.ML_Base::removeTrainingVector(i);
    }
    res |= SVM_Generic::removeTrainingVector(i,yy,x);

    trainclass.remove(i);
    Cweightval.remove(i);
    Cweightvalfuzz.remove(i);
    epsweightval.remove(i);
    onedvec.remove(i);

    for ( q = numClasses()-1 ; q >= 0 ; q-- )
    {
	if ( q != MODPOS(y) )
	{
            idivsplit    = i;
            numclasspre  = numClasses();
            numclassat   = QFIX(y,q);
            numclasspost = numClasses();

            xyval->removeRowCol(IQFIX(y,i,q));
            Gpval->removeRowCol(IQFIX(y,i,q));
            Gpsigma->removeRowCol(IQFIX(y,i,q));

            xycache.remove(IQFIX(y,i,q));
            kerncache.remove(IQFIX(y,i,q));
            sigmacache.remove(IQFIX(y,i,q));

            res |= Q.ML_Base::removeTrainingVector(IQFIX(y,i,q));

            idivsplit    = -1;
            numclasspre  = 0;
            numclassat   = 0;
            numclasspost = 0;
	}
    }

    // Fix the cache

    if ( ( kerncache.get_min_rowdim() >= (int) (N()*(numClasses()-1)*ROWDIMSTEPRATIO) ) && ( N()*(numClasses()-1) > MINROWDIM ) )
    {
        xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(N()-1)*(numClasses()-1));
        kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(N()-1)*(numClasses()-1));
        sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(N()-1)*(numClasses()-1));
    }

    res |= fixautosettings(0,1);

    return res;
}

int SVM_MultiC_atonce::addTrainingVector(int i, int y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xx(x);

    return qaddTrainingVector(i,y,xx,Cweigh,epsweigh);
}

int SVM_MultiC_atonce::qaddTrainingVector(int i, int y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( !linbfd || ( y != linbfd ) );
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( y >= -1 );

    int res = 0;

    if ( kerncache.get_min_rowdim() <= N()*(numClasses()-1) )
    {
        xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (N()*(numClasses()-1)*ROWDIMSTEPRATIO));
        kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (N()*(numClasses()-1)*ROWDIMSTEPRATIO));
        sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (N()*(numClasses()-1)*ROWDIMSTEPRATIO));
    }

    if ( y )
    {
        isStateOpt = 0;
        res |= addclass(y);
    }

    gentype yn(y);
    res |= SVM_Generic::qaddTrainingVector(i,yn,x);
    res |= qtaddTrainingVector(i,y,Cweigh,epsweigh);
    res |= fixautosettings(0,1);

//FIXME: THIS SHOULD NOT BE NECESSARY!
//resetKernel();
    return res;
}

int SVM_MultiC_atonce::addTrainingVector(int i, const Vector<int> &y, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<SparseVector<gentype> > xxx(xx);

    return SVM_MultiC_atonce::qaddTrainingVector(i,y,xxx,Cweigh,epsweigh);
}

int SVM_MultiC_atonce::qaddTrainingVector(int i, const Vector<int> &y, Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( y.size() == xx.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    if ( y.size() )
    {
        int j,k = 0;

        if ( kerncache.get_min_rowdim() < (N()+y.size())*(numClasses()-1) )
        {
            xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) ((N()+y.size()-1)*(numClasses()-1)*ROWDIMSTEPRATIO));
            kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) ((N()+y.size()-1)*(numClasses()-1)*ROWDIMSTEPRATIO));
            sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) ((N()+y.size()-1)*(numClasses()-1)*ROWDIMSTEPRATIO));
        }

        Vector<int> cliny;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            NiceAssert( !linbfd || ( y(j) != linbfd ) );
            NiceAssert( y(j) >= -1 );

            if ( y(j) )
            {
                for ( k = 0 ; k < cliny.size() ; k++ )
                {
                    if ( cliny(k) == y(j) )
                    {
                        break;
                    }
                }

                if ( k == cliny.size() )
                {
                    cliny.add(k);
                    cliny("&",k) = y(j);
                }
            }
        }

        for ( k = 0 ; k < cliny.size() ; k++ )
        {
            isStateOpt = 0;
            res |= addclass(cliny(k));
        }

        for ( j = 0 ; j < y.size() ; j++ )
        {
            gentype tempy(y(j));

            res |= SVM_Generic::qaddTrainingVector(i+j,tempy,xx("&",j));
            res |= qtaddTrainingVector(i+j,y(j),Cweigh(j),epsweigh(j));
        }
    }

    res |= fixautosettings(0,1);

//FIXME: THIS SHOULD NOT BE NECESSARY!
//resetKernel();
    return res;
}

int SVM_MultiC_atonce::qtaddTrainingVector(int i, int y, double Cweigh, double epsweigh)
{
    NiceAssert( !linbfd || ( y != linbfd ) );

    int res = 0;

    SparseVector<gentype> tempvec;
    int q;

    trainclass.add(i); // via multiferous chicanery N() is called, requiring this
    if ( !bartN )
    {
        // Bartlett duplicates are aliased back to first bartN points before QA.Gp() is
        // used for lookup, so for optimality we don't need to add such duplicates
        // to QA at all.
        res |= QA.qaddTrainingVector(i,tempvec);
    }
    trainclass.remove(i); // undo oddity
    
//    double kerndiagis;
    double diagoffis = QUADCOSTDIAGOFFSET(y,Cweigh,1.0);

//    K2(kerndiagis,i,i);

    Nnc("&",label_placeholder.findID(y)+1)++;

    trainclass.add(i);       trainclass("&",i)       = y;
    Cweightval.add(i);       Cweightval("&",i)       = Cweigh;
    Cweightvalfuzz.add(i);   Cweightvalfuzz("&",i)   = 1.0;
    epsweightval.add(i);     epsweightval("&",i)     = epsweigh;
    onedvec.add(i);          onedvec("&",i)          = 1.0;

//    kerndiagval.add(i); kerndiagval("&",i) = kerndiagis;
    diagoff.add(i);     diagoff("&",i)     = diagoffis;

    dalpha.add(i);        dalpha("&",i).resize(numClasses()).zero();
    dalphaReduced.add(i); dalphaReduced("&",i).resize(numClasses() ? numClasses()-1 : 0).zero();
    dalphaState.add(i);   dalphaState("&",i)   = 0;

    tempvec.prealloc(1);

    retVector<double> tmpva;
    retVector<double> tmpvb;

    for ( q = 0 ; q < numClasses() ; q++ )
    {
	if ( q != MODPOS(y) )
	{
            idivsplit    = i;
            numclasspre  = numClasses();
            numclassat   = QFIX(y,q)+2;
            numclasspost = numClasses();

            xyval->addRowCol(IQFIX(y,i,q));
            Gpval->addRowCol(IQFIX(y,i,q));
            Gpsigma->addRowCol(IQFIX(y,i,q));

            xycache.add(IQFIX(y,i,q));
            kerncache.add(IQFIX(y,i,q));
            sigmacache.add(IQFIX(y,i,q));

            Gpn.addRow(IQFIX(y,i,q));

            Gpn("&",IQFIX(y,i,q),tmpva) = CMcl(label_placeholder.findID(y)+1,( isrecdiv() ? 1 : 0 ))(QFIX(y,q),tmpvb);

            if ( xisrankorgrad(i) )
            {
                Gpn("&",IQFIX(y,i,q),tmpva) *= 0.0;
            }

            res |= Q.qaddTrainingVector(
            IQFIX(y,i,q),
            D_CALC(y,q),
            tempvec,
            CWEIGH_CALC(y,Cweigh,1.0,q,dthres),
            -D_CALC(y,q)*((D_CALC(y,q)*EPSWEIGH_CALC(y,epsweightval(i),q))+(D_CALC(y,MODPOS(y))*EPSWEIGH_CALC(y,epsweightval(i),MODPOS(y))))
            );

            idivsplit    = -1;
            numclasspre  = 0;
            numclassat   = 0;
            numclasspost = 0;
	}
    }

    if ( !bartN && ( trainclass(i) == 0 ) )
    {
        // Bartlett duplicates are aliased back to first bartN points before QA.Gp() is
        // used for lookup, so for optimality we don't need to add such duplicates
        // to QA at all.
        res |= QA.setd(i,0);
    }

    SVM_Generic::basesetalpha(i,alphaV()(i));

    return res;
}

int SVM_MultiC_atonce::anomalyOn(int danomalyClass, double danomalyNu)
{
    isStateOpt = 0;

    NiceAssert( ( danomalyClass == -1 ) || ( danomalyClass >= 1 ) );
    NiceAssert( label_placeholder.findID(danomalyClass) == -1 );

    anomalyd = danomalyClass;

    QA.autosetLinBiasForce(danomalyNu);

    return 1;
}

int SVM_MultiC_atonce::anomalyOff(void)
{
    isStateOpt = 0;

    anomalyd = -3;

    return 1;
}

int SVM_MultiC_atonce::addclass(int label, int epszero)
{
    if ( label )
    {
        if ( label_placeholder.findID(label) == -1 )
	{
	    double oldlinbiasforce = 0;

            if ( epszero && ( linbfq >= 0 ) && linbiasforceset )
	    {
                oldlinbiasforce = LinBiasForce(linbfd);
                setLinBiasForce(linbfd,0);
	    }

            isStateOpt = 0;

	    int i,k,q;
	    int nold = numClasses();
	    int nnew = numClasses()+1;

	    // Update "u"

            addclasstou(u);

	    // Update CMcl

	    fixCMcl(nnew);

	    // Extend linear and quadratic bias forcing

            linbiasforceval.add(linbiasforceval.size());
            linbiasforceval("&",linbiasforceval.size()-1) = 0.0;

	    // Add label to ID store

            label_placeholder.findOrAddID(label);
            Nnc.add(label_placeholder.findID(label)+1);
            Nnc("&",label_placeholder.findID(label)+1) = 0;

	    // Add to label-wise variables.

            mulCclass.add(nnew-1);
            mulCclass("&",nnew-1) = 1.0;
            mulepsclass.add(nnew-1);
            mulepsclass("&",nnew-1) = 1.0;

            linbfd = epszero ? label    : linbfd;
            linbfq = epszero ? (nnew-1) : linbfq;

	    // Add to classlabel-wise variables

            db.add(nnew-1);
            db("&",nnew-1) = 0.0;

	    if ( N() )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    dalpha("&",i).add(nnew-1);
                    dalpha("&",i)("&",nnew-1) = 0.0;

		    if ( nnew > 1 )
		    {
                        dalphaReduced("&",i).add(nnew-2);
                        dalphaReduced("&",i)("&",nnew-2) = 0.0;
		    }
		}
	    }

	    // Update representation vectors.

	    if ( nold )
	    {
		for ( k = 0 ; k < nold ; k++ )
		{
                    classRepval("&",k).add(nnew-1);
                    classRepval("&",k)("&",nnew-1) = -1;
		}
	    }

            retVector<int> tmpva;

            classRepval.add(nnew-1);
            classRepval("&",nnew-1).resize(nnew);
            classRepval("&",nnew-1)("&",0,1,nold-1,tmpva) = -1;
            classRepval("&",nnew-1)("&",nnew-1)           = +1;

	    // Change the bias dimension if required

	    if ( isrecdiv() )
	    {
		if ( nnew > 1 )
		{
                    Gpn.addCol(nnew-2);

		    if ( N() )
		    {
                        retMatrix<double> tmpma;
                        retMatrix<double> tmpmb;

			for ( i = 0 ; i < N() ; i++ )
			{
                            Gpn("&",(i*(nold-1)),1,(i*(nold-1))+nold-2,0,1,nnew-2,tmpma) = CMcl(label_placeholder.findID(trainclass(i))+1,1)(0,1,nold-2,0,1,nnew-2,tmpmb);

                            if ( xisrankorgrad(i) )
                            {
                                Gpn("&",(i*(nold-1)),1,(i*(nold-1))+nold-2,0,1,nnew-2,tmpma) *= 0.0;
                            }
			}
		    }

                    Q.setbiasdim(nnew);
		}
	    }

	    else
	    {
                NiceAssert( ismaxwins() );

		if ( nnew )
		{
                    Gpn.addCol(nnew-1);

		    if ( N() )
		    {
                        retMatrix<double> tmpma;
                        retMatrix<double> tmpmb;

			for ( i = 0 ; i < N() ; i++ )
			{
                            Gpn("&",(i*(nold-1)),1,(i*(nold-1))+nold-2,0,1,nnew-1,tmpma) = CMcl(label_placeholder.findID(trainclass(i))+1,0)(0,1,nold-2,0,1,nnew-1,tmpmb);

                            if ( xisrankorgrad(i) )
                            {
                                Gpn("&",(i*(nold-1)),1,(i*(nold-1))+nold-2,0,1,nnew-1,tmpma) *= 0.0;
                            }
			}
		    }

                    Q.setbiasdim(nnew+1);
		}
	    }

	    if ( N() && ( nnew > 1 ) )
	    {
                SparseVector<gentype> tempvec;

                retVector<double> tmpva;
                retVector<double> tmpvb;

		for ( i = 0 ; i < N() ; i++ )
		{
                    if ( numClasses()-1 != MODPOS(trainclass(i)) )
		    {
                        idivsplit    = i;
                        numclasspre  = nnew;
                        numclassat   = nnew;
                        numclasspost = nold;

                        xyval->addRowCol(IQFIX(trainclass(i),i,nnew-1));
                        Gpval->addRowCol(IQFIX(trainclass(i),i,nnew-1));
                        Gpsigma->addRowCol(IQFIX(trainclass(i),i,nnew-1));

                        xycache.add(IQFIX(trainclass(i),i,nnew-1));
                        kerncache.add(IQFIX(trainclass(i),i,nnew-1));
                        sigmacache.add(IQFIX(trainclass(i),i,nnew-1));

                        Gpn.addRow(IQFIX(trainclass(i),i,nnew-1));
                        Gpn("&",IQFIX(trainclass(i),i,nnew-1),tmpva) = CMcl(label_placeholder.findID(trainclass(i))+1,( isrecdiv() ? 1 : 0 ))(nnew-2,tmpvb);

                        if ( xisrankorgrad(i) )
                        {
                            Gpn("&",IQFIX(trainclass(i),i,nnew-1),tmpva) *= 0.0;
                        }

                        Q.qaddTrainingVector(IQFIX(trainclass(i),i,nnew-1),D_CALC(trainclass(i),nnew-1),tempvec,CWEIGH_CALC(trainclass(i),Cweightval(i),Cweightvalfuzz(i),nnew-1,dthres),-D_CALC(trainclass(i),nnew-1)*((D_CALC(trainclass(i),nnew-1)*EPSWEIGH_CALC(trainclass(i),epsweightval(i),nnew-1))+(D_CALC(trainclass(i),MODPOS(trainclass(i)))*EPSWEIGH_CALC(trainclass(i),epsweightval(i),MODPOS(trainclass(i))))));

                        idivsplit    = -1;
                        numclasspre  = 0;
                        numclassat   = 0;
                        numclasspost = 0;
		    }
		}
	    }

	    // Fix Cweight and epsweight, grab alpha

	    if ( N() && ( nnew > 1 ) )
	    {
                dalpha.zero();

		for ( i = 0 ; i < N() ; i++ )
		{
		    for ( q = 0 ; q < nnew ; q++ )
		    {
                        if ( q != MODPOS(trainclass(i)) )
			{
                            if ( (Q.d())(IQFIX(trainclass(i),i,q)) != D_CALC(trainclass(i),q) )
			    {
                                Q.setd(IQFIX(trainclass(i),i,q),D_CALC(trainclass(i),q));
			    }

			    if ( q < nold )
			    {
                                Q.setCweight  (IQFIX(trainclass(i),i,q),CWEIGH_CALC(trainclass(i),Cweightval(i),Cweightvalfuzz(i),q,dthres));
                                Q.setepsweight(IQFIX(trainclass(i),i,q),-D_CALC(trainclass(i),q)*((D_CALC(trainclass(i),q)*EPSWEIGH_CALC(trainclass(i),epsweightval(i),q))+(D_CALC(trainclass(i),MODPOS(trainclass(i)))*EPSWEIGH_CALC(trainclass(i),epsweightval(i),MODPOS(trainclass(i))))));
			    }

                            dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                            dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);
			}
		    }
		}

		calcAlphaReduced();
	    }

	    // Adjust C and epsilon if required

            //NB: I have absolutely no idea why this condition was here
            //if ( isrecdiv() )
	    {
                resetCandeps();
	    }

            // Fix kernel cache

            if ( kerncache.get_min_rowdim() <= N()*(numClasses()-1) )
	    {
                xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),(int) (N()*(numClasses()-1)*ROWDIMSTEPRATIO));
                kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),(int) (N()*(numClasses()-1)*ROWDIMSTEPRATIO));
                sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),(int) (N()*(numClasses()-1)*ROWDIMSTEPRATIO));
	    }

            if ( ( kerncache.get_min_rowdim() >= (int) (N()*(numClasses()-1)*ROWDIMSTEPRATIO) ) && ( N()*(numClasses()-1) > MINROWDIM ) )
            {
                xycache.setmemsize(MEMSHARE_XYCACHE(memsize()),N()*(numClasses()-1));
                kerncache.setmemsize(MEMSHARE_KCACHE(memsize()),N()*(numClasses()-1));
                sigmacache.setmemsize(MEMSHARE_SIGMACACHE(memsize()),N()*(numClasses()-1));
	    }

	    calcBiasReduced();

            if ( linbiasforceset )
	    {
                setLinBiasForce(linbfd,oldlinbiasforce);
	    }
        }
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

void SVM_MultiC_atonce::fudgeOn(void)  
{ 
    QA.fudgeOn();  
    Q.fudgeOn();  

    return; 
}

void SVM_MultiC_atonce::fudgeOff(void) 
{ 
    QA.fudgeOff(); 
    Q.fudgeOff(); 

    return; 
}


int SVM_MultiC_atonce::train(int &res, svmvolatile int &killSwitch)
{
    int result = 0;
    int dobartlett = ( ( dthres > 0 ) && ( dthres < 0.5 ) ) ? 1 : 0;

    if ( isanomalyOn() && !dobartlett )
    {
        errstream() << "Anomaly training...";
        xycache.padCol(4*(Gpn.numCols()));
        kerncache.padCol(4*(Gpn.numCols()));
        sigmacache.padCol(4*(Gpn.numCols()));
        result |= QA.train(res,killSwitch);
        xycache.padCol(0);
        kerncache.padCol(0);
        sigmacache.padCol(0);
        errstream() << " done\n";
    }

    if ( numClasses() )
    {
        int realN = N();
        int &fakeN = bartfake;

        fakeN = 0;

        if ( dobartlett && realN )
        {
            // Set these to indicate Bartlett duplicates present.
            bartN = N();
            bartfake = 0;

            errstream() << "Bartlett duplication...";

            int i;

            SparseVector<gentype> xnew;

            for ( i = 0 ; i < realN ; i++ )
            {
                if ( i < realN-1 )
                {
                    nullPrint(errstream(),i);
                }

                else
                {
                    errstream() << i;
                }

                if ( ( d()(i) == -1 ) || ( d()(i) >= +1 ) )
                {
                    xnew.fff("&",0) = i;

                    fakeredir.add(fakeN);
                    fakeredir("&",fakeN) = i;
                    fakeN++;

                    addTrainingVector(realN+fakeN-1,d()(i),xnew,1.0,0.0); // eps == 0 for this one, assume z = 0
                }
            }

            errstream() << " done\n";
        }

        result = loctrain(res,killSwitch,realN,0);

        if ( dobartlett && realN )
        {
            SparseVector<gentype> dummyx;
            gentype dummyy;

            errstream() << "Bartlett cleanup...";

            while ( fakeN )
            {
                removeTrainingVector(realN,dummyy,dummyx);

                fakeN--;
                fakeredir.remove(fakeN);
            }

            errstream() << " done\n";

            bartN    = 0;
            bartfake = 0;
        } 

        // Calculate reduced version

        calcAlphaReduced();
	calcBiasReduced();
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





int SVM_MultiC_atonce::loctrain(int &res, svmvolatile int &killSwitch, int realN, int assumeDNZ)
{
    int dobartlett = ( ( dthres > 0 ) && ( dthres < 0.5 ) ) ? 1 : 0;

    Vector<double> realCweigh;

    if ( dobartlett && N() )
    {
        // See svm_binary.cc

        realCweigh = Cweightval;

        int fakeN = 0;
        int i;

        SparseVector<gentype> xnew;

        for ( i = 0 ; i < realN ; i++ )
        {
            if ( assumeDNZ || ( d()(i) == -1 ) || ( d()(i) >= +1 ) )
            {
                xnew.fff("&",0) = i;

                setCweight(realN+fakeN,realCweigh(i)*(((1-dthres)/dthres)-1)/((1-dthres)/dthres));
                setCweight(i,          realCweigh(i)                        /((1-dthres)/dthres));

                fakeN++;
            }
        }
    }

    xycache.padCol(4*(Gpn.numCols()));
    kerncache.padCol(4*(Gpn.numCols()));
    sigmacache.padCol(4*(Gpn.numCols()));
    int result = Q.train(res,killSwitch);
    xycache.padCol(0);
    kerncache.padCol(0);
    sigmacache.padCol(0);

    if ( dobartlett )
    {
        Vector<double> Qalpha(Q.alphaR());

        int i,q;
        int fakeN = 0;
        int qN = 0;

        for ( i = 0 ; i < realN ; i++ )
        {
            if ( assumeDNZ || ( d()(i) == -1 ) || ( d()(i) >= +1 ) )
            {
                for ( q = 0 ; q < numClasses() ; q++ )
                {
                    if ( q != MODPOS(trainclass(i)) )
                    {
                        Qalpha("&",IQFIX(trainclass(i),i,q)) += Qalpha(IQFIX(trainclass(i+realN),i+realN,q));

                        qN++;
                    }
                }

                setCweight(realN+fakeN,realCweigh(i));

                fakeN++;
            }

            else
            {
                for ( q = 0 ; q < numClasses() ; q++ )
                {
                    if ( q != MODPOS(trainclass(i)) )
                    {
                        qN++;
                    }
                }
            }
        }

        retVector<double> tmpva;

        Qalpha("&",qN,1,Qalpha.size()-1,tmpva) = 0.0;
        Q.setAlphaR(Qalpha);
    }

    // Record the result

    db.zero();
    dalpha.zero();
    dalphaState.zero();

    int q,i;

    for ( q = 0 ; q < numClasses() ; q++ )
    {
        if ( isrecdiv() )
        {
            if ( q != 0 )
            {
                db("&",q) = Q.biasVMulti(q-1);
                db("&",0) -= db(q);
            }
	}

        else
        {
            NiceAssert( ismaxwins() );

            db("&",q) = Q.biasVMulti(q);
        }

        if ( N() )
        {
            for ( i = 0 ; i < N() ; i++ )
            {
                if ( q != MODPOS(trainclass(i)) )
                {
                    dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                    dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);

                    if ( (Q.alphaState())(IQFIX(trainclass(i),i,q)) )
                    {
                        dalphaState("&",i) = 1;
                    }
                }
            }
        }
    }

    return result;
}


//OLD - int SVM_MultiC_atonce::train(int &res, svmvolatile int &killSwitch)
//OLD - {
//OLD -     int i,q,result = 0;
//OLD - //    int betaopt = 0;
//OLD - 
//OLD -     if ( isanomalyOn() )
//OLD -     {
//OLD -         errstream() << "Anomaly training...";
//OLD -         xycache.padCol(4*(Gpn.numCols()));
//OLD -         kerncache.padCol(4*(Gpn.numCols()));
//OLD -         sigmacache.padCol(4*(Gpn.numCols()));
//OLD -         result |= QA.train(res,killSwitch);
//OLD -         xycache.padCol(0);
//OLD -         kerncache.padCol(0);
//OLD -         sigmacache.padCol(0);
//OLD -         errstream() << " done\n";
//OLD -     }
//OLD - 
//OLD -     if ( numClasses() )
//OLD -     {
//OLD - 	if ( N() && numClasses() > 1 )
//OLD - 	{
////errstream() << "phantomx 0 " << (Q.baseSVM.Q).initGradBetahpzero(*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn) << "\n";
//
//            // Note: hp == 0 at lowest level.
//
//            if ( (Q.baseSVM.Q).initGradBetahpzero(*Gpval,*Gpval,Q.getGn(),Gpn,Q.getgp(),Q.getgn()) == -1 )
//            {
//                betaopt = 1;
//            }
//OLD - 
//OLD - 	    // Fast feasibility finder version 1
//OLD - 	    //
//OLD -             // Works... sometimes, but slow and unpredictable.
//OLD - 
//	    if ( !betaopt )
//	    {
//		int q;
//		int qa,qb;
//		int nostep;
//		int notopt;
//		int isfree;
//		int convar;
//		int convarJ;
//		int convarqa;
//		int convarqb;
//              int stateChangeqa;
//		int stateChangeqb;
//		int qahit;
//              int qbhit;
//
//		double eqaqb;
//		double alphaChange;
//		double scale;
//		double gradmag;
//
//		Vector<double> e(numClasses());
//              Vector<int> qalist((numClasses()*(numClasses()-1))/2);
//              Vector<int> qblist((numClasses()*(numClasses()-1))/2);
//		Vector<double> qabmag((numClasses()*(numClasses()-1))/2);
//		Vector<double> stepAlpha(N()*(numClasses()-1));
//		Vector<double> stepAlphaDir(N()*(numClasses()-1));
//		Vector<int> qhit(N()*(numClasses()-1));
//              Vector<double> stepBeta((Q.baseSVM.Q).bN());
//		Vector<int> J;
//		Vector<int> Jnot;
//		Vector<int> Jtranslate;
//		double alb,aub,blb,bub,mlb,mub;
//
//		// Calculate current infeasibility as per pdf
//
//              e = linbiasforceval;
//
//		for ( q = 0 ; q < numClasses() ; q++ )
//		{
//		    for ( i = 0 ; i < N() ; i++ )
//		    {
//                      if ( q != MODPOS(trainclass(i)) )
//			{
//                          e("&",q)                     += -(Q.baseSVM.Q).alpha(IQFIX(trainclass(i),i,q));
//                          e("&",MODPOS(trainclass(i))) -= -(Q.baseSVM.Q).alpha(IQFIX(trainclass(i),i,q));
//			}
//		    }
//		}
//
//              // Loop while not optimal (unless a break occurs)
//
//		while ( !betaopt )
//		{
//		    // Calculate list of distinct (qa,qb) pairs, calculating |eqa+eqb|
//
//                  q = 0;
//
//		    for ( qa = 0 ; qa < numClasses()-1 ; qa++ )
//		    {
//			for ( qb = qa+1 ; qb < numClasses() ; qb++ )
//			{
//			    qabmag("&",q) = abs2(e(qa)-e(qb));
//                          qalist("&",q) = qa;
//                          qblist("&",q) = qb;
//
//			    q++;
//			}
//		    }
//
//                  // Sort (qa,qb) list into ascending order
//
//		    for ( qa = 0 ; qa < ((numClasses()*(numClasses()-1))/2)-1 ; qa++ )
//		    {
//			for ( qb = qa+1 ; qb < (numClasses()*(numClasses()-1))/2 ; qb++ )
//			{
//			    if ( qabmag("&",qb) > qabmag("&",qa) )
//			    {
//                              qabmag.blockswap(qb,qa);
//                              qalist.blockswap(qb,qa);
//                              qblist.blockswap(qb,qa);
//			    }
//			}
//		    }
//
//		    // Loop through (qa,qb) pairs in ascending order until progress is made
//                  // Break and revert to alternative method if we run out of pairs without making progress
//
//		    nostep = 1;
//  	            q = 0;
//
//		    while ( nostep )
//		    {
//			// Retrieve current list position and feasibility target
//
//			qa = qalist(q);
//			qb = qblist(q);
//
//			eqaqb = (e(qa)-e(qb))/2;
//
//			// Work out J set (relevant free variables)
//			// Also work out Jnot, which is the complement of J
//
//			J.resize(0);
//			Jnot.resize(0);
//
//			for ( i = 0 ; i < N() ; i++ )
//			{
//			    isfree = 1;
//
//                          if ( qa != MODPOS(trainclass(i)) )
//			    {
//                              if ( ( (Q.baseSVM.Q).alphaState(IQFIX(trainclass(i),i,qa)) == 0  ) ||
//                                   ( (Q.baseSVM.Q).alphaState(IQFIX(trainclass(i),i,qa)) == +2 ) ||
//                                   ( (Q.baseSVM.Q).alphaState(IQFIX(trainclass(i),i,qa)) == -2 )    )
//				{
//				    isfree = 0;
//				}
//			    }
//
//                          if ( qb != MODPOS(trainclass(i)) )
//			    {
//                              if ( ( (Q.baseSVM.Q).alphaState(IQFIX(trainclass(i),i,qb)) == 0  ) ||
//                                   ( (Q.baseSVM.Q).alphaState(IQFIX(trainclass(i),i,qb)) == +2 ) ||
//                                   ( (Q.baseSVM.Q).alphaState(IQFIX(trainclass(i),i,qb)) == -2 )    )
//				{
//				    isfree = 0;
//				}
//			    }
//
//			    if ( isfree )
//			    {
//				J.add(J.size());
//				J("&",J.size()-1) = i;
//			    }
//
//			    else
//			    {
//				Jnot.add(Jnot.size());
//				Jnot("&",Jnot.size()-1) = i;
//			    }
//			}
//
//                      // Loop with current (qa,qb) pair until feasibility target reached or progress impossible
//
//			notopt = 1;
//
//			// Attempt step if J is non-empty
//
//			if ( J.size() )
//			{
//			    // Work out maximum step in required direction.
//			    // Work out Jtranslate set, which is the translated positions of alphas which change
//			    // Work out alphaChange, which is the maximum step of eqa in the required direction.
//
//			    Jtranslate.resize(0);
//			    alphaChange = 0;
//
//			    stepAlpha.zero();
//			    stepBeta.zero();
//
//			    for ( i = 0 ; i < J.size() ; i++ )
//			    {
//                              if ( ( qa != MODPOS(trainclass(J(i))) ) && ( qb != MODPOS(trainclass(J(i))) ) )
//				{
//                                  alb = Q.baseSVM.lb(IQFIX(trainclass(J(i)),J(i),qa)) - (Q.baseSVM.Q).alpha(IQFIX(trainclass(J(i)),J(i),qa));
//                                  blb = Q.baseSVM.lb(IQFIX(trainclass(J(i)),J(i),qb)) - (Q.baseSVM.Q).alpha(IQFIX(trainclass(J(i)),J(i),qb));
//
//                                  aub = -(Q.baseSVM.Q).alpha(IQFIX(trainclass(J(i)),J(i),qa));
//                                  bub = -(Q.baseSVM.Q).alpha(IQFIX(trainclass(J(i)),J(i),qb));
//
//				    mlb = ( alb > -bub ) ? alb : -bub;
//				    mub = ( aub < -blb ) ? aub : -blb;
//
//                                  NiceAssert( mlb <= 0 );
//                                  NiceAssert( mub >= 0 );
//
//                                  stepAlphaDir("&",IQFIX(trainclass(J(i)),J(i),qa)) = ( eqaqb > 0 ) ? +1 : -1;
//                                  stepAlphaDir("&",IQFIX(trainclass(J(i)),J(i),qb)) = ( eqaqb > 0 ) ? -1 : +1;
//
//				    if ( eqaqb > 0 )
//				    {
//                                      stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qa)) = mub;
//                                      stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qb)) = -mub;
//
//                                      qhit("&",IQFIX(trainclass(J(i)),J(i),qa)) = ( aub < -blb ) ? +1 :  0;
//                                      qhit("&",IQFIX(trainclass(J(i)),J(i),qb)) = ( aub < -blb ) ?  0 : -1;
//				    }
//
//				    else
//				    {
//                                      stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qa)) = mlb;
//                                      stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qb)) = -mlb;
//
//                                      qhit("&",IQFIX(trainclass(J(i)),J(i),qa)) = ( aub < -blb ) ? -1 :  0;
//                                      qhit("&",IQFIX(trainclass(J(i)),J(i),qb)) = ( aub < -blb ) ?  0 : +1;
//				    }
//
//                                  alphaChange += stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qa));
//
//				    Jtranslate.add(Jtranslate.size());
//                                  Jtranslate("&",Jtranslate.size()-1) = IQFIX(trainclass(J(i)),J(i),qa);
//
//				    Jtranslate.add(Jtranslate.size());
//                                  Jtranslate("&",Jtranslate.size()-1) = IQFIX(trainclass(J(i)),J(i),qb);
//				}
//
//                              else if ( qa != MODPOS(trainclass(J(i))) )
//				{
//                                  alb = Q.baseSVM.lb(IQFIX(trainclass(J(i)),J(i),qa)) - (Q.baseSVM.Q).alpha(IQFIX(trainclass(J(i)),J(i),qa));
//                                  aub = -(Q.baseSVM.Q).alpha(IQFIX(trainclass(J(i)),J(i),qa));
//
//				    mlb = alb;
//				    mub = aub;
//
//                                  NiceAssert( mlb <= 0 );
//                                  NiceAssert( mub >= 0 );
//
//                                  stepAlphaDir("&",IQFIX(trainclass(J(i)),J(i),qa)) = ( eqaqb > 0 ) ? +1 : -1;
//
//				    if ( eqaqb > 0 )
//				    {
//                                      stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qa)) = mub;
//
//                                      qhit("&",IQFIX(trainclass(J(i)),J(i),qa)) = +1;
//				    }
//
//				    else
//				    {
//                                      stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qa)) = mlb;
//
//                                      qhit("&",IQFIX(trainclass(J(i)),J(i),qa)) = -1;
//				    }
//
//                                  alphaChange += stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qa));
//
//				    Jtranslate.add(Jtranslate.size());
//                                  Jtranslate("&",Jtranslate.size()-1) = IQFIX(trainclass(J(i)),J(i),qa);
//				}
//
//				else
//				{
//                                  blb = Q.baseSVM.lb(IQFIX(trainclass(J(i)),J(i),qb)) - (Q.baseSVM.Q).alpha(IQFIX(trainclass(J(i)),J(i),qb));
//                                  bub = -(Q.baseSVM.Q).alpha(IQFIX(trainclass(J(i)),J(i),qb));
//
//				    mlb = -bub;
//				    mub = -blb;
//
//                                  NiceAssert( mlb <= 0 );
//                                  NiceAssert( mub >= 0 );
//
//                                  stepAlphaDir("&",IQFIX(trainclass(J(i)),J(i),qb)) = ( eqaqb > 0 ) ? -1 : +1;
//
//				    if ( eqaqb > 0 )
//				    {
//                                      stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qb)) = -mub;
//
//                                      qhit("&",IQFIX(trainclass(J(i)),J(i),qb)) = -1;
//				    }
//
//				    else
//				    {
//                                      stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qb)) = -mlb;
//
//                                      qhit("&",IQFIX(trainclass(J(i)),J(i),qb)) = +1;
//				    }
//
//                                  alphaChange -= stepAlpha("&",IQFIX(trainclass(J(i)),J(i),qb));
//
//				    Jtranslate.add(Jtranslate.size());
//                                  Jtranslate("&",Jtranslate.size()-1) = IQFIX(trainclass(J(i)),J(i),qb);
//				}
//			    }
//
//			    // If a step is possible then scale and take it.
//
//			    if ( abs2(alphaChange) != 0 )
//			    {
//				// Work out step to reach feasibility target
//
//				scale  = eqaqb/alphaChange;
//				nostep = 0;
//
//				// If feasible then scale to reach target
//
//				if ( scale <= 1 )
//				{
//				    stepAlpha("&",Jtranslate) *= scale;
//				    alphaChange *= scale;
//				    notopt = 0;
//				}
//
//				// Take step
//
//                              (Q.baseSVM.Q).stepFGeneralhpzero((Q.baseSVM.Q).aNF(),0,stepAlpha((Q.baseSVM.Q).pivAlphaF()),stepBeta((Q.baseSVM.Q).pivBetaF()),*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//
//				// Update feasibility targets
//
//				e("&",qa) -= alphaChange;
//				e("&",qb) += alphaChange;
//				eqaqb      -= alphaChange;
//			    }
//
//			    if ( notopt )
//			    {
//				// Constraint relevant variables at bounds
//
//				if ( Jtranslate.size() )
//				{
//				    for ( i = 0 ; i < Jtranslate.size() ; i++ )
//				    {
//                                      if ( ( qhit(Jtranslate(i)) == -1 ) && ( (Q.baseSVM.Q).alphaState(Jtranslate(i)) == -1 ) )
//					{
//                                          (Q.baseSVM.Q).modAlphaLFtoLBhpzero((Q.baseSVM.Q).findInAlphaF(Jtranslate(i)),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn,Q.baseSVM.lb);
//					}
//
//                                      else if ( ( qhit(Jtranslate(i)) == -1 ) && ( (Q.baseSVM.Q).alphaState(Jtranslate(i)) == +1 ) )
//					{
//                                            (Q.baseSVM.Q).modAlphaUFtoZhpzero ((Q.baseSVM.Q).findInAlphaF(Jtranslate(i)),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					}
//
//                                      else if ( ( qhit(Jtranslate(i)) == +1 ) && ( (Q.baseSVM.Q).alphaState(Jtranslate(i)) == -1 ) )
//					{
//                                            (Q.baseSVM.Q).modAlphaLFtoZhpzero ((Q.baseSVM.Q).findInAlphaF(Jtranslate(i)),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					}
//
//                                      else if ( ( qhit(Jtranslate(i)) == +1 ) && ( (Q.baseSVM.Q).alphaState(Jtranslate(i)) == +1 ) )
//					{
//                                            (Q.baseSVM.Q).modAlphaUFtoUBhpzero((Q.baseSVM.Q).findInAlphaF(Jtranslate(i)),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn,Q.baseSVM.ub);
//					}
//				    }
//				}
//			    }
//			}
//
//			    // If we have not reached feasibility target for this (qa,qb) pair then attempt to
//			    // change to active set to do so.  Break if we run out of changes without reaching
//                            // feasibility target.
//
//			while ( notopt )
//			{
//			    {
//				gradmag       = 0;
//				convar        = -1;
//                              convarJ       = -1;
//				convarqa      = -1;
//				convarqb      = -1;
//				stateChangeqa = 0;
//				stateChangeqb = 0;
//
//                                Jtranslate.resize(0);
//
//                                // Search for the change that will take us furthest toward our target
//
//				if ( Jnot.size() )
//				{
//				    for ( i = 0 ; i < Jnot.size() ; i++ )
//				    {
//                                      if ( ( qa != MODPOS(trainclass(Jnot(i))) ) && ( qb != MODPOS(trainclass(Jnot(i))) ) )
//					{
//                                          alb = Q.baseSVM.lb(IQFIX(trainclass(Jnot(i)),Jnot(i),qa)) - (Q.baseSVM.Q).alpha(IQFIX(trainclass(Jnot(i)),Jnot(i),qa));
//                                          blb = Q.baseSVM.lb(IQFIX(trainclass(Jnot(i)),Jnot(i),qb)) - (Q.baseSVM.Q).alpha(IQFIX(trainclass(Jnot(i)),Jnot(i),qb));
//
//                                          aub = -(Q.baseSVM.Q).alpha(IQFIX(trainclass(Jnot(i)),Jnot(i),qa));
//                                          bub = -(Q.baseSVM.Q).alpha(IQFIX(trainclass(Jnot(i)),Jnot(i),qb));
//
//					    mlb = ( alb > -bub ) ? alb : -bub;
//					    mub = ( aub < -blb ) ? aub : -blb;
//
//                                          NiceAssert( mlb <= 0 );
//                                          NiceAssert( mub >= 0 );
//
//					    if ( eqaqb > 0 )
//					    {
//						if ( mub > gradmag )
//						{
//						    gradmag       = mub;
//						    convar        = Jnot(i);
//						    convarJ       = i;
//                                                  convarqa      = IQFIX(trainclass(Jnot(i)),Jnot(i),qa);
//                                                  convarqb      = IQFIX(trainclass(Jnot(i)),Jnot(i),qb);
//                                                  stateChangeqa = ( (Q.baseSVM.Q).alphaState(convarqa) == -2 ) ? -2 : ( ( ( Q.baseSVM.Q).alphaState(convarqa) == 0 ) ? -1 : 0 );
//                                                  stateChangeqb = ( (Q.baseSVM.Q).alphaState(convarqb) == +2 ) ? +2 : ( ( ( Q.baseSVM.Q).alphaState(convarqb) == 0 ) ? +1 : 0 );
//						    qahit         = ( aub <= -blb ) ? +1 :  0;
//						    qbhit         = ( aub <= -blb ) ?  0 : -1;
//						}
//					    }
//
//					    else
//					    {
//						if ( -mlb > gradmag )
//						{
//						    gradmag       = -mlb;
//						    convar        = Jnot(i);
//						    convarJ       = i;
//                                                  convarqa      = IQFIX(trainclass(Jnot(i)),Jnot(i),qa);
//                                                  convarqb      = IQFIX(trainclass(Jnot(i)),Jnot(i),qb);
//                                                  stateChangeqa = ( (Q.baseSVM.Q).alphaState(convarqa) == +2 ) ? +2 : ( ( ( Q.baseSVM.Q).alphaState(convarqa) == 0 ) ? +1 : 0 );
//                                                  stateChangeqb = ( (Q.baseSVM.Q).alphaState(convarqb) == -2 ) ? -2 : ( ( ( Q.baseSVM.Q).alphaState(convarqb) == 0 ) ? -1 : 0 );
//						    qahit         = ( alb >= -bub ) ? -1 :  0;
//						    qbhit         = ( alb >= -bub ) ?  0 : +1;
//						}
//					    }
//					}
//
//                                      else if ( qa != MODPOS(trainclass(Jnot(i))) )
//					{
//                                          alb = Q.baseSVM.lb(IQFIX(trainclass(Jnot(i)),Jnot(i),qa)) - (Q.baseSVM.Q).alpha(IQFIX(trainclass(Jnot(i)),Jnot(i),qa));
//                                          aub = -(Q.baseSVM.Q).alpha(IQFIX(trainclass(Jnot(i)),Jnot(i),qa));
//
//					    mlb = alb;
//					    mub = aub;
//
//                                          NiceAssert( mlb <= 0 );
//                                          NiceAssert( mub >= 0 );
//
//					    if ( eqaqb > 0 )
//					    {
//						if ( mub > gradmag )
//						{
//						    gradmag       = mub;
//						    convar        = Jnot(i);
//						    convarJ       = i;
//                                                  convarqa      = IQFIX(trainclass(Jnot(i)),Jnot(i),qa);
//						    convarqb      = -1;
//                                                  stateChangeqa = ( (Q.baseSVM.Q).alphaState(convarqa) == -2 ) ? -2 : ( ( ( Q.baseSVM.Q).alphaState(convarqa) == 0 ) ? -1 : 0 );
//						    stateChangeqb = 0;
//						    qahit         = +1;
//						    qbhit         = 0;
//						}
//					    }
//
//					    else
//					    {
//						if ( -mlb > gradmag )
//						{
//						    gradmag       = -mlb;
//						    convar        = Jnot(i);
//						    convarJ       = i;
//                                                  convarqa      = IQFIX(trainclass(Jnot(i)),Jnot(i),qa);
//						    convarqb      = -1;
//                                                  stateChangeqa = ( (Q.baseSVM.Q).alphaState(convarqa) == +2 ) ? +2 : ( ( ( Q.baseSVM.Q).alphaState(convarqa) == 0 ) ? +1 : 0 );
//						    stateChangeqb = 0;
//						    qahit         = -1;
//						    qbhit         = 0;
//						}
//					    }
//					}
//
//					else
//					{
//                                          blb = Q.baseSVM.lb(IQFIX(trainclass(Jnot(i)),Jnot(i),qb)) - (Q.baseSVM.Q).alpha(IQFIX(trainclass(Jnot(i)),Jnot(i),qb));
//                                          bub = -(Q.baseSVM.Q).alpha(IQFIX(trainclass(Jnot(i)),Jnot(i),qb));
//
//					    mlb = -bub;
//					    mub = -blb;
//
//                                          NiceAssert( mlb <= 0 );
//                                          NiceAssert( mub >= 0 );
//
//					    if ( eqaqb > 0 )
//					    {
//						if ( mub > gradmag )
//						{
//						    gradmag       = mub;
//						    convar        = Jnot(i);
//						    convarJ       = i;
//						    convarqa      = -1;
//                                                  convarqb      = IQFIX(trainclass(Jnot(i)),Jnot(i),qb);
//						    stateChangeqa = 0;
//                                                  stateChangeqb = ( (Q.baseSVM.Q).alphaState(convarqb) == +2 ) ? +2 : ( ( ( Q.baseSVM.Q).alphaState(convarqb) == 0 ) ? +1 : 0 );
//						    qahit         = 0;
//						    qbhit         = -1;
//						}
//					    }
//
//					    else
//					    {
//						if ( -mlb > gradmag )
//						{
//						    gradmag       = -mlb;
//						    convar        = Jnot(i);
//						    convarJ       = i;
//						    convarqa      = -1;
//                                                  convarqb      = IQFIX(trainclass(Jnot(i)),Jnot(i),qb);
//						    stateChangeqa = 0;
//                                                  stateChangeqb = ( (Q.baseSVM.Q).alphaState(convarqb) == -2 ) ? -2 : ( ( ( Q.baseSVM.Q).alphaState(convarqb) == 0 ) ? -1 : 0 );
//						    qahit         = 0;
//						    qbhit         = +1;
//						}
//					    }
//					}
//				    }
//				}
//
//				if ( convar == -1 )
//				{
//				    // No change possible, break.
//
//				    break;
//				}
//
//				else
//				{
//                                    nostep = 0;
//				    if ( eqaqb > 0 )
//				    {
//					if ( gradmag > eqaqb )
//					{
//					    gradmag = eqaqb;
//					    qahit   = 0;
//					    qbhit   = 0;
//                                            notopt  = 0;
//					}
//				    }
//
//				    else
//				    {
//					gradmag *= -1;
//
//					if ( gradmag < eqaqb )
//					{
//					    gradmag = eqaqb;
//					    qahit   = 0;
//                                            qbhit   = 0;
//                                            notopt  = 0;
//					}
//				    }
//
//				    e("&",qa) -= gradmag;
//				    e("&",qb) += gradmag;
//				    eqaqb      -= gradmag;
//
//				    if ( convarqa != -1 )
//				    {
//                                      (Q.baseSVM.Q).alphaStephpzero(convarqa,gradmag,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn,1);
//
//					if ( stateChangeqa == -2 )
//					{
//					    if ( qahit )
//					    {
//                                              (Q.baseSVM.Q).modAlphaLBtoZhpzero((Q.baseSVM.Q).findInAlphaLB(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaLBtoLFhpzero((Q.baseSVM.Q).findInAlphaLB(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//					}
//
//					else if ( stateChangeqa == -1 )
//					{
//					    if ( qahit )
//					    {
//                                              (Q.baseSVM.Q).modAlphaZtoUBhpzero((Q.baseSVM.Q).findInAlphaZ(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaZtoUFhpzero((Q.baseSVM.Q).findInAlphaZ(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//					}
//
//					else if ( stateChangeqa == +1 )
//					{
//					    if ( qahit )
//					    {
//                                              (Q.baseSVM.Q).modAlphaZtoLBhpzero((Q.baseSVM.Q).findInAlphaZ(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaZtoLFhpzero((Q.baseSVM.Q).findInAlphaZ(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//					}
//
//					else if ( stateChangeqa == +2 )
//					{
//					    if ( qahit )
//					    {
//                                              (Q.baseSVM.Q).modAlphaUBtoZhpzero((Q.baseSVM.Q).findInAlphaUB(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaUBtoUFhpzero((Q.baseSVM.Q).findInAlphaUB(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//					}
//
//					else if ( qahit == -1 )
//					{
//                                          if ( (Q.baseSVM.Q).alphaState(convarqa) == -1 )
//					    {
//                                              (Q.baseSVM.Q).modAlphaLFtoLBhpzero((Q.baseSVM.Q).findInAlphaF(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn,Q.baseSVM.lb);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaUFtoZhpzero((Q.baseSVM.Q).findInAlphaF(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//					}
//
//					else if ( qahit == +1 )
//					{
//                                          if ( (Q.baseSVM.Q).alphaState(convarqa) == -1 )
//					    {
//                                              (Q.baseSVM.Q).modAlphaLFtoZhpzero((Q.baseSVM.Q).findInAlphaF(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaUFtoUBhpzero((Q.baseSVM.Q).findInAlphaF(convarqa),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn,Q.baseSVM.ub);
//					    }
//					}
//				    }
//
//				    if ( convarqb != -1 )
//				    {
//                                      (Q.baseSVM.Q).alphaStephpzero(convarqb,-gradmag,*Gp,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn,1);
//
//					if ( stateChangeqb == -2 )
//					{
//					    if ( qbhit )
//					    {
//                                              (Q.baseSVM.Q).modAlphaLBtoZhpzero((Q.baseSVM.Q).findInAlphaLB(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaLBtoLFhpzero((Q.baseSVM.Q).findInAlphaLB(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//					}
//
//					else if ( stateChangeqb == -1 )
//					{
//					    if ( qbhit )
//					    {
//                                              (Q.baseSVM.Q).modAlphaZtoUBhpzero((Q.baseSVM.Q).findInAlphaZ(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaZtoUFhpzero((Q.baseSVM.Q).findInAlphaZ(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//					}
//
//					else if ( stateChangeqb == +1 )
//					{
//					    if ( qbhit )
//					    {
//                                              (Q.baseSVM.Q).modAlphaZtoLBhpzero((Q.baseSVM.Q).findInAlphaZ(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaZtoLFhpzero((Q.baseSVM.Q).findInAlphaZ(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//					}
//
//					else if ( stateChangeqb == +2 )
//					{
//					    if ( qbhit )
//					    {
//                                              (Q.baseSVM.Q).modAlphaUBtoZhpzero((Q.baseSVM.Q).findInAlphaUB(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaUBtoUFhpzero((Q.baseSVM.Q).findInAlphaUB(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//					}
//
//					else if ( qbhit == -1 )
//					{
//                                          if ( (Q.baseSVM.Q).alphaState(convarqb) == -1 )
//					    {
//                                              (Q.baseSVM.Q).modAlphaLFtoLBhpzero((Q.baseSVM.Q).findInAlphaF(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn,Q.baseSVM.lb);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaUFtoZhpzero((Q.baseSVM.Q).findInAlphaF(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//					}
//
//					else if ( qbhit == +1 )
//					{
//                                          if ( (Q.baseSVM.Q).alphaState(convarqb) == -1 )
//					    {
//                                              (Q.baseSVM.Q).modAlphaLFtoZhpzero((Q.baseSVM.Q).findInAlphaF(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn);
//					    }
//
//					    else
//					    {
//                                              (Q.baseSVM.Q).modAlphaUFtoUBhpzero((Q.baseSVM.Q).findInAlphaF(convarqb),*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn,Q.baseSVM.ub);
//					    }
//					}
//				    }
//
//                                    Jnot.remove(convarJ);
//				    J.add(J.size());
//				    J("&",J.size()-1) = convar;
//				}
//			    }
//			}
//
//			// If no step taken for current (qa,qb) pair then try the next
//			// pair.  If this is the last (qa,qb) pair in the list then
//                        // resort to alternative feasibility finder.
//
//			if ( nostep )
//			{
//			    if ( ++q >= ((numClasses()*(numClasses()-1))/2) )
//			    {
//                                goto presolver;
//			    }
//			}
//		    }
//
//                    // Test beta optimality, exit if optimal
//
//		    betaopt = 0;
//
//                  if ( (Q.baseSVM.Q).initGradBetahpzero(*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn) == -1 )
//		    {
//			betaopt = 1;
//		    }
//		}
//	    }
//OLD - 
//OLD - 	    // Fallback feasibility finder
//OLD - 	    //
//OLD -             // Not the fastest, but very reliable
//OLD - 
//	    if ( !betaopt && 0 )
//	    {
//	    presolver:
//		// Minimise    (1/2) ( Gpn'.alpha + gn )'( Gpn'.alpha + gn )
//		//          =  (1/2) alpha'.(Gpn.Gpn').alpha + alpha'.(Gpn.gn) + irrelevant constant
//		//
//		// So the approach is this: - fix the bias
//		//                          - modify gp := Gpn.gn
//                //                          - turn on cholesky fudging (minimises active set modification due to boundary hits)
//		//                          - modify Gp := Gpn.Gpn'
//		//                          - train the SVM
//		//                          - turn off cholesky fudging (do before resetting Gp so changes do not propogate back into Gp factorisation)
//		//                          - revert to original Gp
//		//                          - revert to original gp
//                //                          - revert to variable bias
//
//errstream() << "fff";
//                Q.setFixedBias();
//
//                Vector<double> z;
//
//              z = (Q.baseSVM.gp);
//              z -= (Gpn*(Q.baseSVM.gn));
//                Q.setz(z);
//
//                (Q.baseSVM).fudgeOn();
//              pathcorrect = 1;
//
//              kerncache.clear();
//              sigmacache.clear();
//              locsetGp(); // FIXME: THIS IS A SLOW POINT
//
//              result = Q.train(); // FIXME: THIS IS A SLOW POINT
//
//                (Q.baseSVM).fudgeOff();
//              pathcorrect = 0;
//
//              kerncache.clear();
//              sigmacache.clear();
//              locsetGp(); // FIXME: THIS IS A SLOW POINT
//
//		z.zero();
//                Q.setz(z);
//
//                Q.setVarBias();
//errstream() << "d";
//	    }
//OLD - 	}
//OLD - 
//OLD - //errstream() << "phantomx 80: " << (Q.baseSVM.Q).alpha() << "\n";
//OLD - //errstream() << "phantomx 80: " << linbiasforceval << "\n";
//OLD - //errstream() << "phantomx 81 " << (Q.baseSVM.Q).initGradBetahpzero(*Gpval,*Gpval,Q.baseSVM.Gn,Gpn,Q.baseSVM.gp,Q.baseSVM.gn) << "\n";
//OLD - 	// ==============================================================
//OLD - 	// ==============================================================
//OLD - 	// ==============================================================
//OLD -         //
//OLD -         // Train the SVM
//OLD -         //
//OLD - 	// ==============================================================
//OLD - 	// ==============================================================
//OLD - 	// ==============================================================
//OLD - 
//OLD -         xycache.padCol(4*(Gpn.numCols()));
//OLD -         kerncache.padCol(4*(Gpn.numCols()));
//OLD -         sigmacache.padCol(4*(Gpn.numCols()));
//OLD -         result |= Q.train(res,killSwitch);
//OLD -         xycache.padCol(0);
//OLD -         kerncache.padCol(0);
//OLD -         sigmacache.padCol(0);
//OLD - //Codingby A.l.i.s.t.a.i.r...S.h.i.l.t.o.n.
//OLD - //      if ( ( result = Q.train() ) == -1 )
//OLD - //	{
//OLD - //	    goto presolver;
//OLD - //	}
//OLD - 
//OLD -         // Record the result
//OLD - 
//OLD -         db.zero();
//OLD -         dalpha.zero();
//OLD -         dalphaState.zero();
//OLD - 
//OLD - 	for ( q = 0 ; q < numClasses() ; q++ )
//OLD - 	{
//OLD - 	    if ( isrecdiv() )
//OLD - 	    {
//OLD - 		if ( q != 0 )
//OLD - 		{
//OLD -                     db("&",q) = Q.biasVMulti(q-1);
//OLD -                     db("&",0) -= db(q);
//OLD - 		}
//OLD - 	    }
//OLD - 
//OLD - 	    else
//OLD - 	    {
//OLD -                 NiceAssert( ismaxwins() );
//OLD - 
//OLD -                 db("&",q) = Q.biasVMulti(q);
//OLD - 	    }
//OLD - 
//OLD - 	    if ( N() )
//OLD - 	    {
//OLD - 		for ( i = 0 ; i < N() ; i++ )
//OLD - 		{
//OLD -                     if ( q != MODPOS(trainclass(i)) )
//OLD - 		    {
//OLD -                         dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
//OLD -                         dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);
//OLD - 
//OLD -                         if ( (Q.alphaState())(IQFIX(trainclass(i),i,q)) )
//OLD - 			{
//OLD -                             dalphaState("&",i) = 1;
//OLD - 			}
//OLD - 		    }
//OLD - 		}
//OLD - 	    }
//OLD - 	}
//OLD - 
//OLD -         // Calculate reduced version
//OLD - 
//OLD -         calcAlphaReduced();
//OLD - 	calcBiasReduced();
//OLD -     }
//OLD - 
//OLD -     Ns = sum(dalphaState);
//OLD - 
//OLD -     if ( result )
//OLD -     {
//OLD -         isStateOpt = 0;
//OLD -     }
//OLD - 
//OLD -     else
//OLD -     {
//OLD -         isStateOpt = 1;
//OLD -     }
//OLD - 
//OLD -     SVM_Generic::basesetAlphaBiasFromAlphaBiasV();
//OLD - 
//OLD -     return result;
//OLD - }

int SVM_MultiC_atonce::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    Vector<double> gproject;
    int locclassrep = 0;
    int tempresh = 0;

    tempresh = gTrainingVector(gproject,locclassrep,i,retaltg,pxyprodi);
    resh = tempresh;
    resg = gproject;

    return tempresh;
}

int SVM_MultiC_atonce::gTrainingVector(Vector<double> &gprojectExt, int &locclassrep, int i, int raw, gentype ***pxyprodi) const
{
    (void) pxyprodi;

    int dtv = 0;

    Vector<double> gproject(numClasses());

    int dobartlett = ( ( dthres > 0 ) && ( dthres < 0.5 ) ) ? 1 : 0;

    if ( i >= 0 )
    {
        // The result we get from Q is emod and the result we want is f, where:
        //
        // emod = [ I -1 0 ] . f
        //        [ 0 -1 I ]
        //
        // emod being the gradient wrt the reduced dimension alpha, f being the
        // gradient wrt alpha.
        //
        // So we used the generalised inverse:
        //
        // [ I -1 0 ]^+    [  I   0  ]   ( [ I -1 0 ]   [  I   0  ] )^-1
        // [ 0 -1 I ]    = [ -1' -1' ] . ( [ 0 -1 I ] . [ -1' -1' ] )
        //                 [  0   I  ]   (              [  0   I  ] )
        //
        //                 [  I   0  ]
        //               = [ -1' -1' ] . ( I + 1.1' )^-1    (where the second part has dimension n-1 = nunClasses()-1)
        //                 [  0   I  ]
        //
        //                 [  I   0  ]
        //               = [ -1' -1' ] . ( I - (1/n),1.1' )
        //                 [  0   I  ]
        //
        // So:
        //
        //     [  I   0  ]
        // f = [ -1' -1' ] . ( I - (1/n).1.1' ) . emod
        //     [  0   I  ]

        gproject.zero();

        if ( numClasses() > 1 )
        {
            int q;

            Vector<double> gRed(numClasses()-1);
            double gAlt = 0;

            for ( q = 0 ; q < numClasses() ; q++ )
            {
                if ( q != MODPOS(trainclass(i)) )
                {
                    Q.gTrainingVector(gRed("&",QFIX(trainclass(i),q)),locclassrep,IQFIX(trainclass(i),i,q),raw);
                    //gAlt -= gRed(QFIX(trainclass(i),q)); LONG STANDING BUG WAS HERE!!!!
                    gAlt += gRed(QFIX(trainclass(i),q));
                }
            }

            gAlt /= ((double) numClasses());
            gRed -= gAlt;

            for ( q = 0 ; q < numClasses() ; q++ )
            {
                if ( q != MODPOS(trainclass(i)) )
                {
                    gproject("&",q) = gRed(QFIX(trainclass(i),q));
                    gproject("&",MODPOS(trainclass(i))) -= gRed(QFIX(trainclass(i),q));
                }
            }
        }

        if ( ismaxwins() || raw )
        {
            gprojectExt.resize(numClasses());
            gprojectExt = gproject;
        }

        else if ( numClasses() > 1 )
        {
            gprojectExt.resize(numClasses()-1);
            downDim(gproject,gprojectExt);
        }

        else
        {
            gprojectExt.resize(0);
        }

        if ( isanomalyOn() && !dobartlett )
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

        double Kxj;
        int j;

        gproject  = db;
        gproject *= 0.0;

        if ( Ns )
        {
            for ( j = 0 ; j < N() ; j++ )
            {
                if ( dalphaState(j) )
                {
                    K2(Kxj,i,j); 
                    gproject.scaleAdd(Kxj,dalpha(j));
                }
            }
        }

        if ( ismaxwins() || raw )
        {
            gprojectExt.resize(numClasses());
            gprojectExt = gproject;
        }

        else if ( numClasses() > 1 )
        {
            gprojectExt.resize(numClasses()-1);
            downDim(gproject,gprojectExt);
        }

        else
        {
            gprojectExt.resize(0);
        }

        if ( isanomalyOn() && !dobartlett )
        {
            double temp;

            temp = QA.biasR();

            if ( N() )
            {
                int j;
                double Kij;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( QA.alphaState()(j) )
                    {
                        K2(Kij,i,j);
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

    else
    {
        double Kxj;
        int j;

        gproject = db;

        if ( Ns )
        {
            for ( j = 0 ; j < N() ; j++ )
            {
                if ( dalphaState(j) )
                {
                    K2(Kxj,i,j); 
                    gproject.scaleAdd(Kxj,dalpha(j));
                }
            }
        }

        if ( ismaxwins() || raw )
        {
            gprojectExt.resize(numClasses());
            gprojectExt = gproject;
        }

        else if ( numClasses() > 1 )
        {
            gprojectExt.resize(numClasses()-1);
            downDim(gproject,gprojectExt);
        }

        else
        {
            gprojectExt.resize(0);
        }

        if ( isanomalyOn() && !dobartlett )
        {
            double temp;

            temp = QA.biasR();

            if ( N() )
            {
                int j;
                double Kij;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( QA.alphaState()(j) )
                    {
                        K2(Kij,i,j);
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

int SVM_MultiC_atonce::gTrainingVector(Vector<gentype> &gprojectExt, int &locclassrep, int i, int raw, gentype ***pxyprodi) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    Vector<double> gproject(numClasses());

    int res = gTrainingVector(gproject,locclassrep,i,raw,pxyprodi);

    gprojectExt.castassign(gproject);

    return res;
}

void SVM_MultiC_atonce::locsetGp(int refactsol)
{
    Q.setGp(Gpval,Gpsigma,xyval,refactsol);

    return;
}

void SVM_MultiC_atonce::locnaivesetGpnExt(void)
{
    Q.naivesetGpnExt(&Gpn);

    return;
}

int SVM_MultiC_atonce::classify(int &locclassrep, const Vector<double> &gproject) const
{
    locclassrep = -1;
    int assclass = -1;

    if ( numClasses() )
    {
        double maxproj = 0;

	locclassrep = 0;

	if ( numClasses() > 1 )
	{
	    maxproj = max(gproject,locclassrep);
	}

        if ( maxproj < dthres )
        {
            // Bartlett's "classify with reject option"

            assclass    = isanomalyOn() ? anomalyd : 0; // reject is reinterpretted as anomaly if anomaly detection is turned on, over-riding QA
            locclassrep = isanomalyOn() ? -3 : -4;
        }

        else
        {
            assclass = label_placeholder.findref(locclassrep);
        }
    }

    return assclass;
}

void SVM_MultiC_atonce::recalcdiagoff(int i)
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
	    Vector<double> bp(N()*(numClasses()-1));

            retVector<double> tmpva;

	    for ( i = 0 ; i < N() ; i++ )
	    {
                bp("&",(i*(numClasses()-1)),1,((i+1)*(numClasses()-1))-1,tmpva) = -diagoff(i);
                diagoff("&",i) = QUADCOSTDIAGOFFSET(trainclass(i),Cweightval(i),Cweightvalfuzz(i));
                bp("&",(i*(numClasses()-1)),1,((i+1)*(numClasses()-1))-1,tmpva) += diagoff(i);
	    }

//            kerndiagval += bp(zeroint(),numClasses()-1,(N()*(numClasses()-1))-1);

            kerncache.recalcDiag();
            Q.recalcdiagoff(bp);

            //int oldmemsize = sigmacache.get_memsize();
            //int oldrowdim  = sigmacache.get_min_rowdim();

            sigmacache.clear();

            //sigmacache.reset(N(),&evalsigmaSVM_MultiC_atonce,(void *) this);
            //sigmacache.setmemsize(MEMSHARE_SIGMACACHE(oldmemsize),oldrowdim);
	}

	else
	{
	    double bpoff;

            bpoff = -diagoff(i);
            diagoff("&",i) = QUADCOSTDIAGOFFSET(trainclass(i),Cweightval(i),Cweightvalfuzz(i));
            bpoff += diagoff(i);

//            kerndiagval("&",i) += bpoff;

            sigmacache.remove(i);
            sigmacache.add(i);

	    for ( q = 0 ; q < numClasses() ; q++ )
	    {
                if ( q != MODPOS(trainclass(i)) )
		{
                    kerncache.recalcDiag(IQFIX(trainclass(i),i,q));
                    Q.recalcdiagoff(IQFIX(trainclass(i),i,q),bpoff);
		}
	    }
	}
    }

    return;
}

void SVM_MultiC_atonce::fixCMcl(int numclassestarg, int fixCCMclOnly)
{
    NiceAssert( numclassestarg >= 0 );

    int s,t;

    if ( !fixCCMclOnly )
    {
        CMcl.resize(numclassestarg+1,numclassestarg+1);

	if ( !numclassestarg )
	{
            CMcl("&",0,0).resize(0,0);
            CMcl("&",0,0).resize(0,0);
	}

	else if ( numclassestarg == 1 )
	{
	    int p = 0;

            CMcl("&",1,1).resize(0,0); CMcl("&",1,p).resize(0,1);
            CMcl("&",p,1).resize(1,0); CMcl("&",p,p).resize(1,1);

            CMcl("&",p,p)("&",p,p) = 1.0;
	}

	else
	{
	    s = -1;
	    t = -1;

	    Matrix<double> *basecutd = smCutM(numclassestarg,&s,&t);

	    {
		{
                    CMcl("&",s+1,t+1) = *basecutd;
		}
	    }

	    s = -1;
	    t = 0;

	    Matrix<double> *basecutb = smCutM(numclassestarg,&s,&t);

	    {
		for ( t = 0 ; t < numclassestarg ; t++ )
		{
                    CMcl("&",s+1,t+1) = *basecutb;
		}
	    }

	    s = 0;
	    t = -1;

	    Matrix<double> *basecutc = smCutM(numclassestarg,&s,&t);

	    for ( s = 0 ; s < numclassestarg ; s++ )
	    {
		{
                    CMcl("&",s+1,t+1) = *basecutc;
		}
	    }

	    s = 0;
	    t = 0;

	    Matrix<double> *basecut = smCutM(numclassestarg,&s,&t);

	    for ( s = 0 ; s < numclassestarg ; s++ )
	    {
		for ( t = 0 ; t < numclassestarg ; t++ )
		{
                    CMcl("&",s+1,t+1) = *basecut;
		}
	    }
	}
    }

    CCMcl.resize(numclassestarg+1,numclassestarg+1);

    if ( !numclassestarg )
    {
        CCMcl("&",0,0).resize(0,0);
    }

    else if ( numclassestarg == 1 )
    {
	int p = 0;

        CCMcl("&",1,1).resize(0,0); CCMcl("&",1,p).resize(0,1);
        CCMcl("&",p,1).resize(1,0); CCMcl("&",p,p).resize(1,1);

        CCMcl("&",p,p)("&",p,p) = 1.0;
    }

    else
    {
	int p = ismaxwins() ? 0 : 1;

        CCMcl("&",0,0) = CMcl(p,p);

	for ( s = 0 ; s < numclassestarg ; s++ )
	{
            CCMcl("&",s+1,0) = CMcl(s+1,p);
            CCMcl("&",0,s+1) = CMcl(p,s+1);
	}

	for ( s = 0 ; s < numclassestarg ; s++ )
	{
	    for ( t = 0 ; t < numclassestarg ; t++ )
	    {
                CCMcl("&",s+1,t+1) = CMcl(s+1,p)*CMcl(p,t+1);
	    }
	}
    }

    return;
}

int SVM_MultiC_atonce::naivesetdzero(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    isStateOpt = 0;

    int res = QA.setd(i,0);

    int q;

    for ( q = 0 ; q < numClasses() ; q++ )
    {
        if ( q != MODPOS(trainclass(i)) )
	{
            res |= Q.setd(IQFIX(trainclass(i),i,q),0);
	}
    }

    Nnc("&",(label_placeholder.findID(trainclass(i))+1))--;
    Nnc("&",(label_placeholder.findID(0)+1))++;

    return res;
}

void SVM_MultiC_atonce::calcAlphaReduced(int i)
{
    NiceAssert( i >= -1 );
    NiceAssert( i < N() );

    if ( numClasses() > 1 )
    {
	if ( i == -1 )
	{
            downDim(dalpha,dalphaReduced);
	}

	else
	{
            downDim(dalpha(i),dalphaReduced("&",i));
	}
    }

    return;
}

void SVM_MultiC_atonce::calcBiasReduced(void)
{
    dbReduced.resize( numClasses() ? numClasses()-1 : 0 );

    if ( isrecdiv() )
    {
        downDim(db,dbReduced);
    }

    else
    {
        dbReduced = db;
    }

    return;
}

int SVM_MultiC_atonce::fixautosettings(int kernchange, int Nchange, int ncut)
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

double SVM_MultiC_atonce::autosetkerndiagmean(void)
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

    return mean(kerndiag()(dnonzero,tmpva));
}

double SVM_MultiC_atonce::autosetkerndiagmedian(void)
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

    return median(kerndiag()(dnonzero,tmpva),i);
}

std::ostream &SVM_MultiC_atonce::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Multiclass All At Once SVM\n\n";

    repPrint(output,'>',dep) << "Cost type:                       " << costType          << "\n";
    repPrint(output,'>',dep) << "Multiclass type:                 " << multitype         << "\n";
    repPrint(output,'>',dep) << "Opt type (0 act, 1 smo, 2 d2c, 3 grad):  " << optType           << "\n";

    repPrint(output,'>',dep) << "C:                               " << CNval             << "\n";
    repPrint(output,'>',dep) << "eps:                             " << epsval            << "\n";
    repPrint(output,'>',dep) << "C+-:                             " << mulCclass         << "\n";
    repPrint(output,'>',dep) << "eps+-:                           " << mulepsclass       << "\n";
    repPrint(output,'>',dep) << "Bartlett threshold:              " << dthres            << "\n";

    repPrint(output,'>',dep) << "Parameter autoset level:         " << autosetLevel      << "\n";
    repPrint(output,'>',dep) << "Parameter autoset nu value:      " << autosetnuvalx     << "\n";
    repPrint(output,'>',dep) << "Parameter autoset C value:       " << autosetCvalx      << "\n";

    repPrint(output,'>',dep) << "XY cache details:                " << xycache           << "\n";
    repPrint(output,'>',dep) << "Kernel cache details:            " << kerncache         << "\n";
    repPrint(output,'>',dep) << "Sigma cache details:             " << sigmacache        << "\n";
//    repPrint(output,'>',dep) << "Kernel diagonals:                " << kerndiagval       << "\n";
    repPrint(output,'>',dep) << "Diagonal offsets:                " << diagoff           << "\n";

    repPrint(output,'>',dep) << "Linear bias forcing:             " << linbiasforceval   << "\n";
    repPrint(output,'>',dep) << "Linear bias forcing class:       " << linbfd            << "\n";
    repPrint(output,'>',dep) << "zero bound class:                " << linbfq            << "\n";
    repPrint(output,'>',dep) << "Linear bias forcing nonzero?:    " << linbiasforceset   << "\n";

    repPrint(output,'>',dep) << "Alpha:                           " << dalpha            << "\n";
    repPrint(output,'>',dep) << "Alpha reduced:                   " << dalphaReduced     << "\n";
    repPrint(output,'>',dep) << "Alpha state:                     " << dalphaState       << "\n";
    repPrint(output,'>',dep) << "Bias:                            " << db                << "\n";
    repPrint(output,'>',dep) << "Bias reduced:                    " << dbReduced         << "\n";
    repPrint(output,'>',dep) << "Label placeholder storage:       " << label_placeholder << "\n";
    repPrint(output,'>',dep) << "Class representations:           " << classRepval       << "\n";
    repPrint(output,'>',dep) << "Ns:                              " << Ns                << "\n";
    repPrint(output,'>',dep) << "Nnc:                             " << Nnc               << "\n";
    repPrint(output,'>',dep) << "u vectors:                       " << u                 << "\n";
    repPrint(output,'>',dep) << "Is SVM optimal:                  " << isStateOpt        << "\n";

    SVM_Generic::printstream(output,dep+1);
    
    repPrint(output,'>',dep) << "Training classes:                " << trainclass        << "\n";
    repPrint(output,'>',dep) << "Training C weights:              " << Cweightval        << "\n";
    repPrint(output,'>',dep) << "Training C weights:              " << Cweightvalfuzz    << "\n";
    repPrint(output,'>',dep) << "Training eps weights:            " << epsweightval      << "\n";
    repPrint(output,'>',dep) << "Training 1:                      " << onedvec           << "\n";

    repPrint(output,'>',dep) << "Gpn:                             " << Gpn               << "\n";

    repPrint(output,'>',dep) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    repPrint(output,'>',dep) << "Base SVM:                        "; Q.printstream(output,dep+1);
    repPrint(output,'>',dep) << "Base anomaly:                    "; QA.printstream(output,dep+1);
    repPrint(output,'>',dep) << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";

    return output;
}

std::istream &SVM_MultiC_atonce::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> costType;
    input >> dummy; input >> multitype;
    input >> dummy; input >> optType;

    input >> dummy; input >> CNval;
    input >> dummy; input >> epsval;
    input >> dummy; input >> mulCclass;
    input >> dummy; input >> mulepsclass;
    input >> dummy; input >> dthres;

    input >> dummy; input >> autosetLevel;
    input >> dummy; input >> autosetnuvalx;
    input >> dummy; input >> autosetCvalx;

    input >> dummy; input >> xycache;
    input >> dummy; input >> kerncache;
    input >> dummy; input >> sigmacache;
//    input >> dummy; input >> kerndiagval;
    input >> dummy; input >> diagoff;

    input >> dummy; input >> linbiasforceval;
    input >> dummy; input >> linbfd;
    input >> dummy; input >> linbfq;
    input >> dummy; input >> linbiasforceset;

    input >> dummy; input >> dalpha;
    input >> dummy; input >> dalphaReduced;
    input >> dummy; input >> dalphaState;
    input >> dummy; input >> db;
    input >> dummy; input >> dbReduced;
    input >> dummy; input >> label_placeholder;
    input >> dummy; input >> classRepval;
    input >> dummy; input >> Ns;
    input >> dummy; input >> Nnc;
    input >> dummy; input >> u;
    input >> dummy; input >> isStateOpt;

    SVM_Generic::inputstream(input);
    
    input >> dummy; input >> trainclass;
    input >> dummy; input >> Cweightval;
    input >> dummy; input >> Cweightvalfuzz;
    input >> dummy; input >> epsweightval;
    input >> dummy; input >> onedvec;

    input >> dummy; input >> Gpn;
    input >> dummy; Q.inputstream(input);
    input >> dummy; QA.inputstream(input);

    QA.setaltx(this);

    fixCMcl(numClasses());

    if ( Gpval != NULL )
    {
        MEMDEL(xyval);
        MEMDEL(Gpval);
        MEMDEL(Gpsigma);

        xyval = NULL;
        Gpval = NULL;
        Gpsigma = NULL;
    }

    MEMNEW(xyval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &(xycache),   (N())*((numClasses()-1)),(N())*((numClasses()-1))));
    MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &(kerncache), (N())*((numClasses()-1)),(N())*((numClasses()-1))));
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) &(sigmacache),(N())*((numClasses()-1)),(N())*((numClasses()-1))));

    int oldmemsize = (kerncache).get_memsize();
    int oldrowdim  = (kerncache).get_min_rowdim();

    (xycache).reset((N())*(numClasses()-1),&evalXYSVM_MultiC_atonce,this);
    (xycache).setmemsize(MEMSHARE_KCACHE(oldmemsize),oldrowdim);

    (kerncache).reset((N())*(numClasses()-1),&evalKSVM_MultiC_atonce,this);
    (kerncache).setmemsize(MEMSHARE_KCACHE(oldmemsize),oldrowdim);

    (sigmacache).reset((N())*(numClasses()-1),&evalsigmaSVM_MultiC_atonce,this);
    (sigmacache).setmemsize(MEMSHARE_SIGMACACHE(oldmemsize),oldrowdim);

    locnaivesetGpnExt();
    locsetGp(0);

    return input;
}

Vector<double> &updim(const Vector<Vector<double> > &u, const Vector<double> &dTDimArg, Vector<double> &nDimArg)
{
    int s;
    double dT = (u.size())-1;;
    double temp;

    nDimArg.resize((dTDimArg.size())+1);
    nDimArg.zero();

    for ( s = 0 ; s < u.size() ; s++ )
    {
        nDimArg("&",s) = twoProductNoConj(temp,u(s),dTDimArg);
    }

    nDimArg *= sqrt(dT/(dT+1));

    return nDimArg;
}

Vector<double> &downdim(const Vector<Vector<double> > &u, const Vector<double> &nDimArg, Vector<double> &dTDimArg)
{
    int s;
    double dT = (u.size())-1;;

    dTDimArg.resize((nDimArg.size())-1);
    dTDimArg.zero();

    for ( s = 0 ; s < u.size() ; s++ )
    {
	dTDimArg.scaleAdd(nDimArg(s),u(s));
    }

    dTDimArg *= sqrt(dT/(dT+1));

    return dTDimArg;
}

Vector<Vector<double> > &updim(const Vector<Vector<double> > &u, const Vector<Vector<double> > &dTDimArg, Vector<Vector<double> > &nDimArg)
{
    int i;

    nDimArg.resize(dTDimArg.size());

    for ( i = 0 ; i < nDimArg.size() ; i++ )
    {
        updim(u,dTDimArg(i),nDimArg("&",i));
    }

    return nDimArg;
}

Vector<Vector<double> > &downdim(const Vector<Vector<double> > &u, const Vector<Vector<double> > &nDimArg, Vector<Vector<double> > &dTDimArg)
{
    int i;

    dTDimArg.resize(nDimArg.size());

    for ( i = 0 ; i < dTDimArg.size() ; i++ )
    {
        downdim(u,nDimArg(i),dTDimArg("&",i));
    }

    return dTDimArg;
}

Vector<double> &SVM_MultiC_atonce::upDim(const Vector<double> &dTDimArg, Vector<double> &nDimArg) const
{
    if ( isrecdiv() )
    {
	return updim(getu(),dTDimArg,nDimArg);
    }

    nDimArg = dTDimArg;

    return nDimArg;
}

Vector<double> &SVM_MultiC_atonce::downDim(const Vector<double> &nDimArg, Vector<double> &dTDimArg) const
{
    if ( isrecdiv() )
    {
	return downdim(getu(),nDimArg,dTDimArg);
    }

    dTDimArg = nDimArg;

    return dTDimArg;
}

Vector<Vector<double> > &SVM_MultiC_atonce::upDim(const Vector<Vector<double> > &dTDimArg, Vector<Vector<double> > &nDimArg) const
{
    if ( isrecdiv() )
    {
	return updim(getu(),dTDimArg,nDimArg);
    }

    nDimArg = dTDimArg;

    return nDimArg;
}

Vector<Vector<double> > &SVM_MultiC_atonce::downDim(const Vector<Vector<double> > &nDimArg, Vector<Vector<double> > &dTDimArg) const
{
    if ( isrecdiv() )
    {
	return downdim(getu(),nDimArg,dTDimArg);
    }

    dTDimArg = nDimArg;

    return dTDimArg;
}

int SVM_MultiC_atonce::autosetLinBiasForce(double nuval, double Cval, int ncut)
{
    NiceAssert(Cval > 0);
    NiceAssert(nuval >= 0.0);
    NiceAssert(nuval <= 1.0);
    NiceAssert(linbfq >= 0);

    autosetCvalx = Cval;
    autosetnuvalx = nuval;

    int res = setC(( (N()-NNC(0)-ncut) ) ? ((((((double) numClasses())-1))/nuval)*(Cval/((double) ((N()-NNC(0)-ncut))))) : 1.0);
    res |= seteps(( numClasses() >= 3 ) ? (sqrt((((double) numClasses())-1)/(2*(((double) numClasses())-2)))) : 0.0);
    res |= setLinBiasForce(linbfd,Cval); 
    autosetLevel = 6;

    return res;
}


//void construct_u(Vector<Vector<double> > &u, int n);
//void construct_u(Vector<Vector<double> > &u, int n)
//{
//    int s,t,q;
//    Vector<double> utemplate(dT(n));
//
//    u.resize(0);
//
//    for ( s = 0 ; s < n ; s++ )
//    {
//	u.add(s,utemplate);
//
//	for ( q = 0 ; q < dT(n) ; q++ )
//	{
//	    if ( ( s <= q ) && ( q < dT(n)-1 ) )
//	    {
//		u("&",s)("&",q) = 1/(((double) q) + 1);
//
//		for ( t = q+2 ; t < n ; t++ )
//		{
//		    u("&",s)("&",q) *= sqrt((t*t)-1)/((double) t);
//		}
//	    }
//
//	    else if ( ( s <= q ) && ( q == dT(n)-1 ) )
//	    {
//		u("&",s)("&",q) = 1/(((double) q) + 1);
//	    }
//
//	    else if ( ( q == s-1 ) && ( q < dT(n)-1 ) )
//	    {
//		u("&",s)("&",q) = -1;
//
//		for ( t = q+2 ; t < n ; t++ )
//		{
//		    u("&",s)("&",q) *= sqrt((t*t)-1)/((double) t);
//		}
//	    }
//
//	    else if ( ( q == s-1 ) && ( q == n-2 ) )
//	    {
//		u("&",s)("&",q) = -1;
//	    }
//
//	    else
//	    {
//		u("&",s)("&",q) = 0;
//	    }
//	}
//    }
//
//    return;
//}

int SVM_MultiC_atonce::prealloc(int expectedN)
{
//    kerndiagval.prealloc(expectedN);
    diagoff.prealloc(expectedN);
    dalpha.prealloc(expectedN);
    dalphaReduced.prealloc(expectedN);
    dalphaState.prealloc(expectedN);
    trainclass.prealloc(expectedN);
    Cweightval.prealloc(expectedN);
    Cweightvalfuzz.prealloc(expectedN);
    epsweightval.prealloc(expectedN);
    onedvec.prealloc(expectedN);
    xycache.prealloc(expectedN);
    kerncache.prealloc(expectedN);
    sigmacache.prealloc(expectedN);
    Gpn.prealloc(numClasses()*expectedN,Gpn.numCols());
    SVM_Generic::prealloc(expectedN);

    QA.prealloc(expectedN);
    Q.prealloc(( numClasses() > 1 ) ? (numClasses()-1)*expectedN : 1);

    return 0;
}

int SVM_MultiC_atonce::preallocsize(void) const
{
    return SVM_Generic::preallocsize();
}

int SVM_MultiC_atonce::randomise(double sparsity)
{
    NiceAssert( sparsity >= 0 );
    NiceAssert( sparsity <= 1 );

    int res = Q.randomise(sparsity);
    int i,q;

    if ( res )
    {
        // Record the result
        // NB: FOLLOWING CODE TAKEN FROM TRAINING FUNCTION

        db.zero();
        dalpha.zero();
        dalphaState.zero();

	for ( q = 0 ; q < numClasses() ; q++ )
	{
	    if ( isrecdiv() )
	    {
		if ( q != 0 )
		{
                    db("&",q) = Q.biasVMulti(q-1);
                    db("&",0) -= db(q);
		}
	    }

	    else
	    {
                NiceAssert( ismaxwins() );

                db("&",q) = Q.biasVMulti(q);
	    }

	    if ( N() )
	    {
		for ( i = 0 ; i < N() ; i++ )
		{
                    if ( q != MODPOS(trainclass(i)) )
		    {
                        dalpha("&",i)("&",q) = (Q.alphaR())(IQFIX(trainclass(i),i,q));
                        dalpha("&",i)("&",MODPOS(trainclass(i))) -= dalpha(i)(q);

                        if ( (Q.alphaState())(IQFIX(trainclass(i),i,q)) )
			{
                            dalphaState("&",i) = 1;
			}
		    }
		}
	    }
	}

        // Calculate reduced version

        calcAlphaReduced();
	calcBiasReduced();

        Ns = sum(dalphaState);

        isStateOpt = 0;

        SVM_Generic::basesetAlphaBiasFromAlphaBiasV();
    }

    return res;
}
































