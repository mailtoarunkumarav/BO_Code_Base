\
//
// Sparse gradient descent
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#define FEEDBACK_CYCLE 1
#define MAJOR_FEEDBACK_CYCLE 1000

#include "sQgraddesc.h"
#include "sQsLsAsWs.h"
#include "sQd2c.h"
#include "smatrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime>


//FIXME: betaRestrict can be safely assumed 0 or 3!

//FIXME: post-calculate beta, use penalty functions to enforce beta-related inequalities!


////#define BOUND_STRETCH 1e-2
////#define BOUND_COST 1e-5
//#define BOUND_STRETCH 1e-2
//#define BOUND_COST 10

#define MINLR 1e-20
//#define ASSUMEDM 4
//#define MINBNDDIST 1e-15
//#define MINBNDDIST 1e-10
#define MINBNDDIST 1e-7
#define MAXGRADCORR 10
#define DEFAULTOFFSET 0.1
#define DEFAULTOFFSETSTEP 1.0

//#define MAXCALCBND 100.0
#define MAXCALCBND 10.0
//#define MAXCALCBND 0.01
//#define MAXCALCBND 0.0

//#define MUSTART 1e-1
//#define MUSTART 1.0e-2
#define MUSTART 10
//#define MUEND 1.0e12
#define MUEND 1.0e5
#define MUDIV 2.0

void recentreBias(int aN, int bN, optState<double,double> &x, Vector<double> &beta, 
                  const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp);

double calc_diagoff(Vector<double> &gradoff, Vector<double> &diagoff, const Vector<double> &efflb, const Vector<double> &effub, const Vector<double> &alpha, double mu, const Vector<int> &alphaRestrict);

void updateOffsets(Vector<double> &locgp, Vector<double> &lochp, Vector<double> &locgn, Matrix<double> &locGp, Matrix<double> &locGpsigma, const optState<double,double> &x, 
                  const Matrix<double> *locGpBase, const Matrix<double> *locGpsigmaBase, int useGpBase, double gphpgnGpnGnscalefact, const Vector<double> &diagoff, int aN);

int calcStep(int aN, int bN, int hpzero, double outertol, int useactive, int maxitcnt, double maxtraintime,
             optState<double,double> &x, optState<double,double> &xxx, double &gphpgnGpnGnscalefact,
             const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn,
             const Vector<double> &gp, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, const Vector<double> &gn, const Vector<double> &gradoff, 
             Matrix<double> &locGp, const Matrix<double> &locGpsigma, const Matrix<double> &locGn, Matrix<double> &locGpn,
             Vector<double> &locgp, Vector<double> &lochp, Vector<double> &loclb, Vector<double> &locub, Vector<double> &locgn,
             Vector<int> &alphares, Vector<int> &betares, svmvolatile int &killSwitch, int &useGpBase, double currlr);

int fullOptStateGradDesc::solve(svmvolatile int &killSwitch)
{
    NiceAssert( !GpnRowTwoSigned );
    NiceAssert( fixHigherOrderTerms );

    retMatrix<double> tmpma;
    retMatrix<double> tmpmb;

    int i;
    int z = 0;

    int aN = x.aN();
    int bN = x.bN();

//    if ( fixHigherOrderTerms == fullfixbasea )
//    {
//        usels |= 2;
//    }

    // Use active set or D2C?

    int useactive = 1; // ( ( bN == 1 ) && ( x.betaRestrict(0) == 0 ) ) ? 0 : 1;    - for some reason on some problems D2C does not work (no step, just return)
    int useGpBase = 1; // ( usels & 2 ) ? 1 : 0;    - for some reason not using GpBase is a bad idea
    int isBiased = ( x.betaRestrict() == 0 ) ? 1 : 0;





    // ------------------------------------------------------------------------------------------------------------------------------------

    int chinum = ( chistart >= 0 ) ? chistart : aN;

    if ( ( chinum < aN ) && useGpBase )
    {
        int res = 0;

        // Because you can't recurse with wrapsolve (unless it's just passthrough because we have a feasible solution),
        // and because wrapsolve has, in this case, been non-feasible, we need a preliminary solve to "clear" the 
        // non-feasibility before proceeding with actual gradient descent.  The exception occurs when we are creating
        // a local version of Gp etc, as the fault lies in the recursive use of matrix extension (of these matrices
        // and their underlying caches).

        // Optimisation state for step calculation

        if ( x.keepfact() )
        {
            // Memory intensive, keeps factorisation (slow update), but can do arbitrary constraints

            fullOptStateActive xxxx(x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,NULL,NULL,1.0);

            xxxx.maxitcnt   = maxitcnt;
            xxxx.maxruntime = maxruntime;

            res = xxxx.wrapsolve(killSwitch);
        }

        else
        {
            // Minimal memory, no factorisation, but limited to simple constraints

            fullOptStateD2C xxxx(x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,NULL,NULL,1.0);

            xxxx.maxitcnt   = maxitcnt;
            xxxx.maxruntime = maxruntime;

            res = xxxx.wrapsolve(killSwitch);
        }

        repover = 1;

        return res;
    }



    // ------------------------------------------------------------------------------------------------------------------------------------
    //
    // - Work out effective lower and upper bounds
    // - Count number of alphas that are negative, positive or free
    // - Step toward centre

    Vector<int> alphares(aN); // alpha restriction for step
    Vector<int> betares(x.betaRestrict());

    alphares = z;
    betares  = isBiased ? 0 : 3;

    Vector<double> efflb(lb);
    Vector<double> effub(ub);

    int Nn = 0; // Number of alphas that are strictly negative
    int Np = 0; // Number of alphas that are strictly positive
    int Nf = 0; // Number of alphas that can be negative or positive

    for ( i = 0 ; i < efflb.size() ; i++ )
    {
        if ( x.alphaRestrict()(i) == 0 )
        {
            if ( ( efflb(i) < -x.zerotol() ) && ( effub(i) > x.zerotol() ) )
            {
                Nf++;
            }

            else if ( ( efflb(i) >= -x.zerotol() ) && ( effub(i) > x.zerotol() ) )
            {
                Np++;
            }

            else if ( ( efflb(i) < -x.zerotol() ) && ( effub(i) <= x.zerotol() ) )
            {
                Nn++;
            }

            else
            {
                alphares("&",i) = 3;
            }
        }

        else if ( x.alphaRestrict()(i) == 1 )
        {
            efflb("&",i) = 0.0;

            if ( effub(i) > x.zerotol() )
            {
                Np++;
            }

            else
            {
                alphares("&",i) = 3;
            }
        }

        else if ( x.alphaRestrict()(i) == 2 )
        {
            effub("&",i) = 0.0;

            if ( efflb(i) < -x.zerotol() )
            {
                Nn++;
            }

            else
            {
                alphares("&",i) = 3;
            }
        }

        else
        {
            efflb("&",i) = 0.0;
            effub("&",i) = 0.0;

            alphares("&",i) = 3;
        }
    }
//errstream() << "efflb = " << efflb << "\n";
//errstream() << "effub = " << effub << "\n";



    // ------------------------------------------------------------------------------------------------------------------------------------
    //
    // Initialise x for optimisation

    Vector<double> zalpha(aN);
    Vector<double> zbeta(bN);

    zalpha = 0.0;
    zbeta  = 0.0;

    x.setBeta(zbeta,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp);
    x.initGradBeta(Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp);
    x.refreshGrad(Gp(0,1,aN-1,0,1,aN-1,tmpma),Gn,Gpn,gp,gn,hp);




    // ------------------------------------------------------------------------------------------------------------------------------------
    //
    // Basic matrices and vectors for step calculation
    //
    // idmat = identity matrix
    // odmat = sigma matrix deriving from idmat

    Matrix<double> idmat(Gpfull.numRows(),Gpfull.numCols());
    Matrix<double> odmat(Gpsigmafull.numRows(),Gpsigmafull.numCols());

    idmat = 0.0;
    odmat = 0.0;

    idmat.diagoffset(1.0);

    // "Base" Gp matrices (not including penalties)

    const Matrix<double> *locGpBase      = ( useGpBase ? &Gpfull      : &idmat );
    const Matrix<double> *locGpsigmaBase = ( useGpBase ? &Gpsigmafull : &odmat );

    // Local "x" definition matrices/vectors

    Matrix<double> locGp(locGpBase->numRows(),locGpBase->numCols());
    Matrix<double> locGpsigma(locGpsigmaBase->numRows(),locGpsigmaBase->numCols());
    Matrix<double> locGn(Gn);
    Matrix<double> locGpn(Gpn);

    Vector<double> locgp(aN);
    Vector<double> lochp(aN);
    Vector<double> locgn(bN);

    Vector<double> loclb(aN);
    Vector<double> locub(aN);

    locGp      = 0.0;
    locGpsigma = 0.0;

    locgp = 0.0;
    lochp = 0.0;
    locgn = 0.0;

    loclb = -MAXBOUND;
    locub =  MAXBOUND;





    // ------------------------------------------------------------------------------------------------------------------------------------
    //
    // Optimisation state for step calculation

    optState<double,double> xxx;

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retVector<double> tmpvc;

    xxx.setopttol(locGp(z,1,-1,z,1,-1,tmpma),locGn(z,1,-1,z,1,-1,tmpmb),Gpn(z,1,-1,z,1,-1,tmpma),locgp(z,1,-1,tmpva),locgn(z,1,-1,tmpvb),lochp(z,1,-1,tmpvc),1e-12);
    xxx.setkeepfact(locGp(z,1,-1,z,1,-1,tmpma),locGn(z,1,-1,z,1,-1,tmpmb),locGpn(z,1,-1,z,1,-1,tmpma),useactive);

    if ( aN )
    {
        for ( i = 0 ; i < aN ; i++ )
        {
            xxx.addAlpha(i,alphares(i),0.0);
        }
    }

    if ( bN )
    {
        for ( i = 0 ; i < bN ; i++ )
        {
            xxx.addBeta(i,betares(i),0.0);
        }
    }

    // Don't call initGradBeta or refreshGrad until locGp etc are calculated







    // ------------------------------------------------------------------------------------------------------------------------------------
    //
    // Vectors controlling barriers
    //
    // Hessian  = 1/gphpgnGpnGnscalefact Gp + diagoff
    // Gradient = (Gp.alpha + gp + sign(alpha).*hp) + gradoff
    //
    // diagoff = term resulting from log barriers
    // gradoff = term resulting from log barriers

    Vector<double> gradoff(aN);
    Vector<double> diagoff(aN);
    Vector<double> zerooff(aN);

    gradoff = 0.0;
    diagoff = 0.0;
    zerooff = 0.0;

    double gphpgnGpnGnscalefact = 1.0;






    // ------------------------------------------------------------------------------------------------------------------------------------
    //
    // mustart,muend,mudiv: these control the weight of the log-barrier

    double mustart = MUSTART; //1*aN;
    double muend = MUEND; //1e12; //100*aN;
    double mudiv = MUDIV; //2;

    double mu = mustart;





    // ------------------------------------------------------------------------------------------------------------------------------------
    //
    // Starting point stuff

    Vector<double> alpha(aN);
    Vector<double> beta(bN);

    Vector<double> oldalpha(aN);
    Vector<double> oldbeta(bN);

    alpha = 0.0;
    beta  = 0.0;

    oldalpha = 0.0;
    oldbeta  = 0.0;

    double limefflb,limeffub;

    for ( i = 0 ; i < aN ; i++ )
    {
        limefflb = ( efflb(i) < -MAXCALCBND ) ? -MAXCALCBND : efflb(i);
        limeffub = ( effub(i) >  MAXCALCBND ) ?  MAXCALCBND : effub(i);

        alpha("&",i) = (limefflb+limeffub)/2;
    }

    if ( isBiased && ( sum(alpha) > 0.0 ) )
    {
        double alphaCorrect = sum(alpha)/Np;

        for ( i = 0 ; i < aN ; i++ )
        {
            if ( alpha(i) > 0.0 )
            {
                alpha("&",i) -= alphaCorrect;
            }
        }
    }

    else if ( isBiased && ( sum(alpha) < 0.0 ) )
    {
        double alphaCorrect = -sum(alpha)/Nn;

        for ( i = 0 ; i < aN ; i++ )
        {
            if ( alpha(i) < 0.0 )
            {
                alpha("&",i) += alphaCorrect;
            }
        }
    }

//errstream() << "alpha start = " << alpha << "\n";
//errstream() << "sum alpha start = " << sum(alpha) << "\n";
    x.setAlpha(alpha,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp,lb,ub);
    x.setBeta(beta,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp);
    fixHigherOrderTerms(*this,htArg,zerooff,zerooff,gphpgnGpnGnscalefact);








    // ------------------------------------------------------------------------------------------------------------------------------------
    //
    // Work out various offsets and current objective value

    double fnnext  = 0.0;
    double fnprev  = 0.0;
    double modstep = 0.0;

    locGpBase      = ( useGpBase ? &Gpfull      : &idmat );
    locGpsigmaBase = ( useGpBase ? &Gpsigmafull : &odmat );

    fnnext  = calc_diagoff(gradoff,diagoff,efflb,effub,alpha,mu,x.alphaRestrict());
    fnnext += fixHigherOrderTerms(*this,htArg,zerooff,gradoff,gphpgnGpnGnscalefact);
    updateOffsets(locgp,lochp,locgn,locGp,locGpsigma,x,locGpBase,locGpsigmaBase,useGpBase,gphpgnGpnGnscalefact,diagoff,aN);

    // Initialisation of step calculator

    xxx.initGradBeta(locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpma),locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpmb),locGn,locGpn,locgp,locgn,lochp);
    xxx.refreshGrad(locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpma),locGn,locGpn,locgp,locgn,lochp);
//errstream() << "fn start = " << fnnext << "\n";





    // ------------------------------------------------------------------------------------------------------------------------------------
    //
    // Optimisation stuff

    double outmaxitcnt = outermaxitcnt;
    double outmaxtraintime = outermaxtraintime;
    double outlr = lr;
    double outlrback = lrback;
    double outdelta = delta;
    double *uservars[] = { &outmaxitcnt, &outmaxtraintime, &outlr, &outlrback, &outdelta, NULL };
    const char *varnames[] = { "outermaxitcnt", "outermaxtraintime", "outerlr", "outerlrback", "outerdelta", NULL };
    const char *vardescr[] = { "Outer maximum iteration count (0 for unlimited)", "Outer maximum training time (seconds, 0 for unlimited)", "Outer learning rate", "Outer learning rate backpull", "delta", NULL };

    int isopt = 0;
    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    unsigned long long itcnt = 0;
    int timeout = 0;
    int bailout = 0;

    int firststep   = 1;
    int badstep     = 0;
    int alphaCIndex = 0;
    int betaCIndex  = 0;
    int stateChange = 0;
    double gradmag  = 0;

    // Obscure note: in c++, if maxitcnt is a double then !maxitcnt is
    // true if maxitcnt == 0, false otherwise.  This is defined in the
    // standard, and the reason the following while statement will work.

    errstream() << " #\b\b";

    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) outmaxitcnt ) || !outmaxitcnt ) && !timeout && !bailout )
    {
        // Bounding needs to be done here for numerical reasons (otherwise the
        // slack gradients tend to swamp the problem)

        if ( ( chistart >= 0 ) && ( chistart < aN ) )
        {
            for ( i = chistart ; i < aN ; i++ )
            {
                if ( x.alphaState()(i) == 0 )
                {
                    x.changeAlphaRestrict(i,3,Gp,Gp,Gn,Gpn,gp,gn,hp);
                }
            }
        }

        // Save current state

        fnprev = fnnext;

        oldalpha = x.alpha();
        oldbeta  = x.beta();

        // Work out optimality

        alphaCIndex = -1;
        betaCIndex  = -1;
        stateChange = 0;
        gradmag     = 0;
        badstep     = 0;

        if ( firststep )
        {
            isopt     = 0;
            firststep = 0;
        }

        else
        {
            // Note that ignorebeta = 1 as we are doing that inequality via a penalty function

            if ( isBiased )
            {
                recentreBias(aN,bN,x,beta,Gp,Gn,Gpn,gp,gn,hp);
            }

            retMatrix<double> tmpma;

            isopt = x.maxGradNonOpt(alphaCIndex,betaCIndex,stateChange,gradmag,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gn,Gpn,gp,gn,hp,outertol,1,1);
        }

        if ( !isopt )
        {
            if ( usels & 0x01 )
            {
                // Line-search step

                oldalpha = alpha;
                oldbeta  = beta;

                calcStep(aN,bN,hpzero,outertol,useactive,maxitcnt,maxruntime,
                         x,xxx,gphpgnGpnGnscalefact,
                         Gp,Gn,Gpn,gp,hp,efflb,effub,gn,gradoff,
                         locGp,locGpsigma,locGn,locGpn,locgp,lochp,loclb,locub,locgn,
                         alphares,betares,killSwitch,useGpBase,outlr);

                int isstep = 0;

                double currlr = outlr;

                while ( !isstep && !badstep )
                {
                    alpha = oldalpha;
                    beta  = oldbeta;

                    alpha.scaleAdd(currlr,xxx.alpha());
                    //beta.scaleAdd(currlr,xxx.beta());

                    retMatrix<double> tmpma;
                    retMatrix<double> tmpmb;

                    x.setAlpha(alpha,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp,lb,ub,0);
                    //x.setBeta(beta,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp,0);

                    locGpBase      = ( useGpBase ? &Gpfull      : &idmat );
                    locGpsigmaBase = ( useGpBase ? &Gpsigmafull : &odmat );

                    fnnext  = calc_diagoff(gradoff,diagoff,efflb,effub,alpha,mu,x.alphaRestrict());
                    double fnextra = fixHigherOrderTerms(*this,htArg,zerooff,gradoff,gphpgnGpnGnscalefact);
                    updateOffsets(locgp,lochp,locgn,locGp,locGpsigma, x,locGpBase,locGpsigmaBase,useGpBase,gphpgnGpnGnscalefact,diagoff,aN);

                    modstep = currlr*currlr*norm2(xxx.alpha());
//errstream() << "fnant = " << fnnext << " + " << fnextra << " = " << fnnext+fnextra << " at " << modstep << "\n";
                    fnnext += fnextra;

                    isstep = 1;

                    // reltol was x.zerotol()

                    if ( (fnprev-fnnext) < (1-outdelta)*modstep )
                    {
                        errstream() << "\\/";

                        isstep = 0;

                        currlr *= outlrback;
                    }

                    if ( !isstep && ( currlr < MINLR ) )
                    {
                        badstep = 1;
                        isopt   = 1;

                        alpha = oldalpha;
                        beta  = oldbeta;

                        errstream() << "??" << gradmag << "," << beta << "??\n";

                        retMatrix<double> tmpma;
                        retMatrix<double> tmpmb;

                        x.setAlpha(alpha,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp,lb,ub,0);
                        //x.setBeta(beta,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp,0);

                        locGpBase      = ( useGpBase ? &Gpfull      : &idmat );
                        locGpsigmaBase = ( useGpBase ? &Gpsigmafull : &odmat );

                        fnnext  = calc_diagoff(gradoff,diagoff,efflb,effub,alpha,mu,x.alphaRestrict());
                        fnnext += fixHigherOrderTerms(*this,htArg,zerooff,gradoff,gphpgnGpnGnscalefact);
                        updateOffsets(locgp,lochp,locgn,locGp,locGpsigma, x,locGpBase,locGpsigmaBase,useGpBase,gphpgnGpnGnscalefact,diagoff,aN);
                    }
                }
            }

            else
            {
                // Simple gradient descent

                calcStep(aN,bN,hpzero,outertol,useactive,maxitcnt,maxruntime,
                         x,xxx,gphpgnGpnGnscalefact,
                         Gp,Gn,Gpn,gp,hp,efflb,effub,gn,gradoff,
                         locGp,locGpsigma,locGn,locGpn,locgp,lochp,loclb,locub,locgn,
                         alphares,betares,killSwitch,useGpBase,outlr);

                alpha = oldalpha;
                beta  = oldbeta;

                alpha.scaleAdd(outlr,xxx.alpha());
                //beta.scaleAdd(outlr,xxx.beta());

                retMatrix<double> tmpma;
                retMatrix<double> tmpmb;

                x.setAlpha(alpha,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp,lb,ub,0);
                //x.setBeta(beta,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp,0);

                locGpBase      = ( useGpBase ? &Gpfull      : &idmat );
                locGpsigmaBase = ( useGpBase ? &Gpsigmafull : &odmat );

                fnnext  = calc_diagoff(gradoff,diagoff,efflb,effub,alpha,mu,x.alphaRestrict());
                fnnext += fixHigherOrderTerms(*this,htArg,zerooff,gradoff,gphpgnGpnGnscalefact);
                updateOffsets(locgp,lochp,locgn,locGp,locGpsigma, x,locGpBase,locGpsigmaBase,useGpBase,gphpgnGpnGnscalefact,diagoff,aN);

                modstep = outlr*outlr*norm2(xxx.alpha());

                if ( modstep <= reltol )
                {
                    isopt = 1;
                }
            }
        }

        retMatrix<double> tmpma;

        int aerr,berr;
        errstream() << fnnext << "(" << x.testGradInt(aerr,berr,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gn,Gpn,gp,gn,hp) << ")";

        if ( badstep )
        {
            errstream() << "mu";

            if ( mu < muend )
            {
                // Step failed, try to increase mu

                mu *= mudiv;

                badstep = 0;
                isopt   = 0;

                fnnext  = calc_diagoff(gradoff,diagoff,efflb,effub,alpha,mu,x.alphaRestrict());
                fnnext += fixHigherOrderTerms(*this,htArg,zerooff,gradoff,gphpgnGpnGnscalefact);
                updateOffsets(locgp,lochp,locgn,locGp,locGpsigma, x,locGpBase,locGpsigmaBase,useGpBase,gphpgnGpnGnscalefact,diagoff,aN);

                errstream() << "<<<bad" << mu << ">>> ";
            }
        }

        if ( isopt && !badstep )
        {
            // Need to test log-barrier optimality

            for ( i = 0 ; isopt && !badstep && ( i < aN ) ; i++ )
            {
                if ( ( alphares(i) != 3 ) && ( x.alpha()(i) >= efflb(i)+(2*(x.zerotol())) ) && ( x.alpha()(i) <= effub(i)-(2*(x.zerotol())) ) )
                {
                     // OK: alpha is not at a boundary - so is offset gradient "within range"

                     if ( ( gradoff(i) > (x.opttol()) ) || ( gradoff(i) < -(x.opttol()) ) )
                     {
                         isopt = 0;
                     }
                }
            }

            if ( !isopt )
            {
                errstream() << "mu";

                // Barrier not "sharp" enough to ensure sensible gradient corrections in non-bound region, so need to increase sharpness and try again.

                badstep = 1;

                if ( mu < muend )
                {
                    mu *= mudiv;

                    badstep = 0;
                    isopt   = 0;

                    fnnext  = calc_diagoff(gradoff,diagoff,efflb,effub,alpha,mu,x.alphaRestrict());
                    fnnext += fixHigherOrderTerms(*this,htArg,zerooff,gradoff,gphpgnGpnGnscalefact);
                    updateOffsets(locgp,lochp,locgn,locGp,locGpsigma, x,locGpBase,locGpsigmaBase,useGpBase,gphpgnGpnGnscalefact,diagoff,aN);

                    errstream() << "<<<" << mu << ">>> ";
                }
            }
        }

        if ( badstep )
        {
            errstream() << "{{{failed to find sharp barriers}}} ";
            isopt = 1;
        }



//FIXME: add terminate on modstep too small, repeated lack of progress in fnnext

        // timeouts and interupts

        if ( !(++itcnt%FEEDBACK_CYCLE) )
        {
            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
            {
                errstream() << " |\b\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
            {
                errstream() << " /\b\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
            {
                errstream() << " -\b\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
            {
                errstream() << " \\\b\b";
            }
        }

        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
        {
            errstream() << "===" << itcnt << "===  ";
        }

        if ( outmaxtraintime > 1 )
        {
            curr_time = TIMECALL;

            if ( TIMEDIFFSEC(curr_time,start_time) > outmaxtraintime )
            {
                timeout = 1;
            }
        }

        if ( !timeout )
        {
            timeout = kbquitdet("quadratic (sQsLsAsWs) optimisation",uservars,varnames,vardescr);
        }
    }

/*
    if ( x.aNF() )
    {
        for ( i = x.aNF()-1 ; i >= 0 ; i-- )
        {
            if ( x.alphaRestrict(x.pivAlphaF()(i)) == 0 )
            {
                if ( x.alphaState()(x.pivAlphaF()(i)) == -1 )
                {
                    if ( x.alpha()(x.pivAlphaF()(i)) >= -2*(x.zerotol()) )
                    {
                        x.modAlphaLFtoZ(i,Gp,Gp,Gn,Gpn,gp,gn,hp);
                    }
          
                    else if ( x.alpha()(x.pivAlphaF()(i)) <= lb(i)+((x.zerotol())) )
                    {
                        x.modAlphaLFtoLB(i,Gp,Gp,Gn,Gpn,gp,gn,hp,lb);
                    }
                }
          
                else if ( x.alphaState()(x.pivAlphaF()(i)) == +1 )
                {
                    if ( x.alpha()(x.pivAlphaF()(i)) <= 2*(x.zerotol()) )
                    {
                        x.modAlphaUFtoZ(i,Gp,Gp,Gn,Gpn,gp,gn,hp);
                    }
      
                    else if ( x.alpha()(x.pivAlphaF()(i)) >= ub(i)-((x.zerotol())) )
                    {
                        x.modAlphaUFtoUB(i,Gp,Gp,Gn,Gpn,gp,gn,hp,ub);
                    }
                }
            }
          
            else if ( x.alphaRestrict(x.pivAlphaF()(i)) == 1 )
            {
                if ( x.alphaState()(x.pivAlphaF()(i)) == +1 )
                {
                    if ( x.alpha()(x.pivAlphaF()(i)) <= 2*(x.zerotol()) )
                    {
                        x.modAlphaUFtoZ(i,Gp,Gp,Gn,Gpn,gp,gn,hp);
                    }
            
                    else if ( x.alpha()(x.pivAlphaF()(i)) >= ub(i)-((x.zerotol())) )
                    {
                        x.modAlphaUFtoUB(i,Gp,Gp,Gn,Gpn,gp,gn,hp,ub);
                    }
                }
            }
            
            else if ( x.alphaRestrict(x.pivAlphaF()(i)) == 2 )
            {
                if ( x.alphaState()(x.pivAlphaF()(i)) == -1 )
                {
                    if ( x.alpha()(x.pivAlphaF()(i)) >= -2*(x.zerotol()) )
                    {
                        x.modAlphaLFtoZ(i,Gp,Gp,Gn,Gpn,gp,gn,hp);
                    }
            
                    else if ( x.alpha()(x.pivAlphaF()(i)) <= lb(i)+((x.zerotol())) )
                    {
                        x.modAlphaLFtoLB(i,Gp,Gp,Gn,Gpn,gp,gn,hp,lb);
                    }
                }
            }
        }
    }
*/

    fixHigherOrderTerms(*this,htArg,zerooff,zerooff,gphpgnGpnGnscalefact);
    x.refreshGrad(Gp(0,1,aN-1,0,1,aN-1,tmpma),Gn,Gpn,gp,gn,hp);

    if ( isBiased )
    {
        recentreBias(aN,bN,x,beta,Gp,Gn,Gpn,gp,gn,hp);
    }

    return isopt ? 0 : 1;
}








void recentreBias(int aN, int bN, optState<double,double> &x, Vector<double> &beta, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp)
{
    int i,j,k;

    if ( bN && aN && ( x.betaRestrict() == zeroint() ) )
    {
        // (Note lazy assumption above)

        Matrix<double> A(bN,bN);

        A = 0.0;

        for ( i = 0 ; i < bN ; i++ )
        {
            for ( j = 0 ; j < bN ; j++ )
            {
                for ( k = 0 ; k < aN ; k++ )
                {
                    if ( ( ( x.alphaState()(k) != -2 ) && ( x.alphaGrad()(k) < 0.0    )                                                                       ) ||
                         ( ( x.alphaState()(k) != -1 )                                                                                                        ) ||
                         ( ( x.alphaState()(k) != 0  ) && ( x.alphaGrad()(k) < -hp(k) ) && ( ( x.alphaRestrict()(k) == 0 ) || ( x.alphaRestrict()(k) == 1 ) ) ) ||
                         ( ( x.alphaState()(k) != 0  ) && ( x.alphaGrad()(k) >  hp(k) ) && ( ( x.alphaRestrict()(k) == 0 ) || ( x.alphaRestrict()(k) == 2 ) ) ) ||
                         ( ( x.alphaState()(k) != +1 )                                                                                                        ) ||
                         ( ( x.alphaState()(k) != +2 ) && ( x.alphaGrad()(k) > 0.0    )                                                                       )    )
                    {
                        A("&",i,j) += Gpn(k,i)*Gpn(k,j);
                    }
                }
            }
        }

        if ( abs2(A.det()) > x.zerotol() )
        {
            Vector<double> b(bN);

            b = 0.0;

            for ( i = 0 ; i < bN ; i++ )
            {
                for ( k = 0 ; k < aN ; k++ )
                {
                    if ( ( ( x.alphaState()(k) != -2 ) && ( x.alphaGrad()(k) < 0.0    )                                                                       ) ||
                         ( ( x.alphaState()(k) != -1 )                                                                                                        ) ||
                         ( ( x.alphaState()(k) != 0  ) && ( x.alphaGrad()(k) < -hp(k) ) && ( ( x.alphaRestrict()(k) == 0 ) || ( x.alphaRestrict()(k) == 1 ) ) ) ||
                         ( ( x.alphaState()(k) != 0  ) && ( x.alphaGrad()(k) >  hp(k) ) && ( ( x.alphaRestrict()(k) == 0 ) || ( x.alphaRestrict()(k) == 2 ) ) ) ||
                         ( ( x.alphaState()(k) != +1 )                                                                                                        ) ||
                         ( ( x.alphaState()(k) != +2 ) && ( x.alphaGrad()(k) > 0.0    )                                                                       )    )
                    {
                        b("&",i) += Gpn(k,i)*x.alphaGrad()(k);
                    }
                }
            }

            beta -= ((A.inve())*b);
//errstream() << "recentre:" << beta << "...";

            retMatrix<double> tmpma;
            retMatrix<double> tmpmb;

            x.setBeta(beta,Gp(0,1,aN-1,0,1,aN-1,tmpma),Gp(0,1,aN-1,0,1,aN-1,tmpmb),Gn,Gpn,gp,gn,hp,0);
        }
    }

    return;
}



double calc_diagoff(Vector<double> &gradoff, Vector<double> &diagoff, const Vector<double> &efflb, const Vector<double> &effub, const Vector<double> &alpha, double mu, const Vector<int> &alphaRestrict)
{
    NiceAssert( diagoff.size() == efflb.size() );
    NiceAssert( diagoff.size() == effub.size() );

    int aN = diagoff.size();
    int i;

    double res = 0;

    // x(i) >= lb(i)
    // x(i) <= ub(i)
    //
    // or:
    //
    //  x(i) - lb(i) >= 0
    // -x(i) + ub(i) >= 0
    //
    // Using log-barrier
    //
    // cost    = -1/mu sum_i ( log( ub(i) - alpha(i) ) + log( alpha(i) - lb(i) ) )
    // grad_i  = -1/mu ( -1/(ub(i)-alpha(i)) + 1/(alpha(i)-lb(i)) )
    // diag_ii = -1/mu ( -1/(ub(i)-alpha(i))^2 + -1/(alpha(i)-lb(i))^2 )
    //
    // cost    = -1/mu sum_i ( log(ub(i)-alpha(i)) + log(-(lb(i)-alpha(i))) )
    // grad_i  = 1/mu ( 1/(ub(i)-alpha(i))   + 1/(lb(i)-alpha(i))   )
    // diag_ii = 1/mu ( 1/(ub(i)-alpha(i))^2 + 1/(lb(i)-alpha(i))^2 )

    double alphaLD,alphaUD;

    for ( i = 0 ; i < aN ; i++ )
    {
        if ( alphaRestrict(i) != 3 )
        {
            alphaUD = effub(i)-alpha(i)+MINBNDDIST;
            alphaLD = efflb(i)-alpha(i)-MINBNDDIST;

            res -= (log(alphaUD)+log(-alphaLD))/mu;

            gradoff("&",i) = ( (1/alphaUD) + (1/alphaLD) )/mu;
            diagoff("&",i) = ( (1/(alphaUD*alphaUD)) + (1/(alphaLD*alphaLD)) )/mu;
        }

        else
        {
            gradoff("&",i) = 0.0;
            diagoff("&",i) = 0.0;
        }
    }

    diagoff += DEFAULTOFFSET;

    return res;
}

void updateOffsets(Vector<double> &locgp, Vector<double> &lochp, Vector<double> &locgn, Matrix<double> &locGp, Matrix<double> &locGpsigma, 
                  const optState<double,double> &x, 
                  const Matrix<double> *locGpBase, const Matrix<double> *locGpsigmaBase, int useGpBase, double gphpgnGpnGnscalefact, const Vector<double> &diagoff, int aN)
{
    (void) locgn;

    locgp = x.alphaGrad();
    lochp = 0.0;

    locGp      = *locGpBase;
    locGpsigma = *locGpsigmaBase;

    if ( useGpBase )
    {
        retMatrix<double> tmpma;

        locGp("&",0,1,aN-1,0,1,aN-1,tmpma)      *= 1/gphpgnGpnGnscalefact; // *= (m-1)
        locGpsigma("&",0,1,aN-1,0,1,aN-1,tmpma) *= 1/gphpgnGpnGnscalefact;
    }

    locGp.diagoffset(diagoff);

    return;
}









int calcStep(int aN, int bN, int hpzero, double outertol, int useactive, int maxitcnt, double maxtraintime,
              optState<double,double> &x, optState<double,double> &xxx, double &gphpgnGpnGnscalefact,
              const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn,
              const Vector<double> &gp, const Vector<double> &hp, const Vector<double> &efflb, const Vector<double> &effub, const Vector<double> &gn, const Vector<double> &gradoff,
              Matrix<double> &locGp, const Matrix<double> &locGpsigma, const Matrix<double> &locGn, Matrix<double> &locGpn,
              Vector<double> &locgp, Vector<double> &lochp, Vector<double> &loclb, Vector<double> &locub, Vector<double> &locgn,
              Vector<int> &alphares, Vector<int> &betares, svmvolatile int &killSwitch, int &useGpBase, double outlr)
{
    (void) gn;
    (void) hpzero;
    (void) outertol;
    (void) Gn;
    (void) gradoff;
    (void) gphpgnGpnGnscalefact;
    (void) bN;
    (void) alphares;
    (void) betares;
    (void) useGpBase;

//errstream() << "x = " << x << "\n";
//errstream() << "Gp = " << Gp << "\n";
//errstream() << "Gn = " << Gn << "\n";
//errstream() << "Gpn = " << Gpn << "\n";
//errstream() << "gp = " << gp << "\n";
//errstream() << "hp = " << hp << "\n";
//errstream() << "gn = " << gn << "\n";
//
//errstream() << "xxx = " << xxx << "\n";
//errstream() << "locGp = " << locGp << "\n";
//errstream() << "locGn = " << locGn << "\n";
//errstream() << "locGpn = " << locGpn << "\n";
//errstream() << "locgp = " << locgp << "\n";
//errstream() << "lochp = " << lochp << "\n";
//errstream() << "locgn = " << locgn << "\n";

    int i;
    int badres = 1;
    int res = 0;

    while ( badres )
    {
        for ( i = 0 ; i < aN ; i++ )
        {
            if ( xxx.alphaState()(i) == -2 )
            {
                retMatrix<double> tmpma;
                retMatrix<double> tmpmb;

                xxx.modAlphaLBtoLF(xxx.findInAlphaLB(i),locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpma),locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpmb),locGn,locGpn,locgp,locgn,lochp);
            }

            else if ( xxx.alphaState()(i) == +2 )
            {
                retMatrix<double> tmpma;
                retMatrix<double> tmpmb;

                xxx.modAlphaUBtoUF(xxx.findInAlphaUB(i),locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpma),locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpmb),locGn,locGpn,locgp,locgn,lochp);
            }

            if ( xxx.alphaState()(i) )
            {
                double astep = -xxx.alpha()(i);

                retMatrix<double> tmpma;

                xxx.alphaStep(i,astep,locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpma),locGn,locGpn,locgp,locgn,lochp);
            }
        }

        for ( i = 0 ; i < bN ; i++ )
        {
            if ( xxx.betaState()(i) )
            {
                double bstep = -xxx.beta()(i);

                retMatrix<double> tmpma;

                xxx.betaStep(i,bstep,locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpma),locGn,locGpn,locgp,locgn,lochp);
            }
        }

        locgp = x.alphaGrad(); // This includes log-barrier terms already
        lochp = 0.0;

        for ( i = 0 ; i < aN ; i++ )
        {
            // Work out "effective" state of alpha and fix gradient if required

            int effstate = x.alphaState()(i);

            if ( ( effstate == -1 ) && ( x.alpha()(i) >= 0 ) )
            {
                effstate = 0;

                x.unAlphaGrad(locgp("&",i),i,Gp,Gpn,gp,hp);
            }

            if ( ( effstate == +1 ) && ( x.alpha()(i) <= 0 ) )
            {
                effstate = 0;

                x.unAlphaGrad(locgp("&",i),i,Gp,Gpn,gp,hp);
            }

            lochp("&",i) = ( effstate == 0 ) ? hp(i) : 0.0;
        }

        // Calculate step

        int calcres = 0;

        {           
            retMatrix<double> tmpma;
            retMatrix<double> tmpmb;

            xxx.refact(locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpma),locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpmb),locGn,locGpn,locgp,locgn,lochp);

            if ( useactive )
            {
                // Memory intensive, keeps factorisation (slow update), but can do arbitrary constraints

                fullOptStateActive xxxx(xxx,locGp,locGpsigma,locGn,locGpn,locgp,locgn,lochp,loclb,locub,NULL,NULL,1.0);

                xxxx.maxitcnt   = maxitcnt;
                xxxx.maxruntime = maxtraintime;
                xxxx.linbreak   = 1;

                calcres = xxxx.wrapsolve(killSwitch);
            }

            else
            {
                // Minimal memory, no factorisation, but limited to simple constraints

                fullOptStateD2C xxxx(xxx,locGp,locGpsigma,locGn,locGpn,locgp,locgn,lochp,loclb,locub,NULL,NULL,1.0);

                xxxx.maxitcnt   = maxitcnt;
                xxxx.maxruntime = maxtraintime;
                //xxxx.linbreak   = 1;

                calcres = xxxx.wrapsolve(killSwitch);
           }

            // Bounds fit for step calculation, diagonally perturb and repeat if bounds hit (as it isn't a true inversion)

            badres = ( calcres == 500 ) ? 1 : 0;

            for ( i = 0 ; !badres && ( i < aN ) ; i++ )
            {
                if ( ( xxx.alphaState()(i) == +2 ) || ( xxx.alphaState()(i) == -2 ) )
                {
                    badres = 1;
                }
            }

            if ( badres )
            {
                res++;

                locGp.diagoffset(exp(log(DEFAULTOFFSETSTEP)*res));
                errstream() << "~~~";
            }
        }
    }

    // Overall step scale to ensure x bounds met

    double scale = 1.0;

    for ( i = 0 ; i < aN ; i++ )
    {
        if ( xxx.alphaState()(i) )
        {
            if ( ( outlr*(xxx.alpha()(i)) > 0 ) && ( x.alpha()(i) + scale*(outlr*(xxx.alpha()(i))) > effub(i) ) )
            {
                scale = ( effub(i) - x.alpha()(i) ) / (outlr*(xxx.alpha()(i)));
            }

            else if ( ( outlr*(xxx.alpha()(i)) < 0 ) && ( x.alpha()(i) + scale*(outlr*(xxx.alpha()(i))) < efflb(i) ) )
            {
                scale = ( efflb(i) - x.alpha()(i) ) / (outlr*(xxx.alpha()(i)));
            }
        }
    }

    if ( scale < 1.0 )
    {
        retMatrix<double> tmpma;

        xxx.scale(scale,locGp(zeroint(),1,aN-1,zeroint(),1,aN-1,tmpma),locGn,locGpn,locgp,locgn,lochp);
    }

//errstream() << "xxx step = " << xxx << "\n";
    return res;
}

