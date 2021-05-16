//TRAINING: why not start with all u the same, then let them drift apart only as far as required to achieved optimality?  Maybe start with all equal or something?
//REASON: if all u same then step is zero
//ALT METHOD: lasso-regularisation.  Gradient components are decreased by beta in whichever direction they are defined (positive or negative)

//
// Multi-dimensional ranking SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_mvrank.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>











SVM_MvRank::SVM_MvRank() : SVM_Planar()
{
    thisthis = this;
    thisthisthis = &thisthis;

    xmaxitermvrank = DEFAULT_MAXITERMVRANK;
    xlrmvrank      = DEFAULT_LRMVRANK;
    xztmvrank      = DEFAULT_ZTMVRANK;
    xbetarank      = DEFAULT_BETARANK;

    setaltx(NULL);

    SVM_Planar::setFixedBias();

    return;
}

SVM_MvRank::SVM_MvRank(const SVM_MvRank &src) : SVM_Planar()
{
    thisthis = this;
    thisthisthis = &thisthis;

    xmaxitermvrank = DEFAULT_MAXITERMVRANK;
    xlrmvrank      = DEFAULT_LRMVRANK;
    xztmvrank      = DEFAULT_ZTMVRANK;
    xbetarank      = DEFAULT_BETARANK;

    setaltx(NULL);

    SVM_Planar::setFixedBias();

    assign(src,0);

    return;
}

SVM_MvRank::SVM_MvRank(const SVM_MvRank &src, const ML_Base *xsrc) : SVM_Planar()
{
    thisthis = this;
    thisthisthis = &thisthis;

    xmaxitermvrank = DEFAULT_MAXITERMVRANK;
    xlrmvrank      = DEFAULT_LRMVRANK;
    xztmvrank      = DEFAULT_ZTMVRANK;
    xbetarank      = DEFAULT_BETARANK;

    setaltx(xsrc);

    SVM_Planar::setFixedBias();

    assign(src,1);

    return;
}

SVM_MvRank::~SVM_MvRank()
{
    return;
}

std::ostream &SVM_MvRank::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Multi-Vector Ranking SVM\n\n";

    repPrint(output,'>',dep) << "maxitermvrank: " << xmaxitermvrank << "\n";
    repPrint(output,'>',dep) << "lrmvrank:      " << xlrmvrank      << "\n";
    repPrint(output,'>',dep) << "ztmvrank:      " << xztmvrank      << "\n";
    repPrint(output,'>',dep) << "betarank:      " << xbetarank      << "\n";

    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base SVC: ";
    SVM_Planar::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_MvRank::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> xmaxitermvrank;
    input >> dummy; input >> xlrmvrank;
    input >> dummy; input >> xztmvrank;
    input >> dummy; input >> xbetarank;
    input >> dummy;

    SVM_Planar::inputstream(input);

    return input;
}

void SVM_MvRank::calcLalpha(Matrix<double> &Lalpha)
{
    int M = NbasisVV();
    int i,j,q,r;

    NiceAssert( Lalpha.numRows() == M );
    NiceAssert( Lalpha.numCols() == M );

    Lalpha = 0.0;

    if ( M )
    {
        for ( q = 0 ; q < M ; q++ )
        {
            for ( r = 0 ; r < M ; r++ )
            {
                for ( i = 0 ; i < N() ; i++ )
                {
                    for ( j = 0 ; j < N() ; j++ )
                    {
                        if ( reflocd()(i) && reflocd()(j) && alphaState()(i) && alphaState()(j) )
                        {
                            NiceAssert( x(i).isfarfarfarindpresent(7) );
                            NiceAssert( x(j).isfarfarfarindpresent(7) );

                            if ( ( (int) x(i).fff(7) == q ) && ( (int) x(j).fff(7) == r ) )
                            {
                                Lalpha("&",q,r) += (alphaR()(i))*(alphaR()(j))*(Gp()(i,j));
                            }
                        }
                    }
                }
            }
        }
    }

    return;
}

#define DEFAULT_MVRANK_STEPSCALE 0.3
#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 1000
#define ZERO_ABOVE 0.03

int SVM_MvRank::train(int &res, svmvolatile int &killSwitch)
{
    int locres = 0;
    int isopt = 0;
    unsigned long long itcnt = 0;
    int i,j;
    int timeout = 0;
    int sfirst;
    double s,smod = 1;
    double optdist;
    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    double xmtrtime = maxtraintime();
    double maxitcntval = maxitermvrank();
    double beta = betarank();
    double stepscalefactor = lrmvrank();
    double outertol = ztmvrank();
    double *uservars[] = { &maxitcntval, &xmtrtime, &stepscalefactor, &beta, &outertol, NULL };
    const char *varnames[] = { "itercount", "traintime", "stepscale", "beta", "outertol", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Step scale", "beta factor", "Outer optimality tolerance", NULL };

    // Work out constants

    int M = NbasisVV();
    int d = getbdim();

    if ( !M || !d )
    {
        return 0;
    }

    // Steps and such-like

    Vector<Vector<double> > &ubase = reflocbasis();
    Vector<Vector<double> > ubaseprev(ubase);
    Vector<Vector<double> > ugrad(ubase); // M.ubase
    Vector<Vector<double> > ustep(ubase);
    Vector<double> ubar(d);

    // Starting point

    for ( i = 0 ; i < M ; i++ )
    {
        ubase("&",i) = 0.0;

        while ( sum(ubase(i)) == 0 )
        {
            randfill(ubase("&",i));
        }

        ubase("&",i) /= sum(ubase(i));
        ubaseprev("&",i) = 0.0;
    }

    // Various matrices

    Matrix<double> Lalpha(M,M);
    Matrix<double> Mbold(M,M);
    Matrix<double> Mbeta(M,M);

    for ( i = 0 ; i < M ; i++ )
    {
        for ( j = 0 ; j < M ; j++ )
        {
            Mbeta("&",i,j) = beta;
        }
    }

    Vector<double> temp(d);

    refactorVV();

    // Main training loop

    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcntval ) || !maxitcntval ) && !timeout )
    {
        // Train inner loop for alpha calculation

        locres |= SVM_Planar::train(res,killSwitch);

        // Update matrices

        calcLalpha(Lalpha);

        Mbold = Lalpha;
        Mbold.diagoffset(-M*beta);
        Mbold += Mbeta;

        // Calculate projected gradients

        for ( i = 0 ; i < M ; i++ )
        {
            ugrad("&",i) = 0.0;

            for ( j = 0 ; j < M ; j++ )
            {
                ugrad("&",i).scaleAdd(Mbold(i,j),ubase(j));
            }

            ugrad("&",i) -= sum(ugrad(i))/d;
        }

        // Test optimality

        isopt = 1;
        optdist = outertol;

        optdist = 0;

        for ( i = 0 ; ( i < M ) ; i++ )
        {
            ubaseprev("&",i) -= ubase(i);

            optdist += abs2(ubaseprev(i));
        }

        if ( optdist > outertol )
        {
            isopt = 0;
        }

        // If not optimal then process step

        if ( !isopt )
        {
            ubaseprev = ubase;

            // Calculate and take step

            for ( i = 0 ; i < M ; i++ )
            {
                ustep("&",i)  = ugrad(i);
                ustep("&",i) *= stepscalefactor;
                ubase("&",i) += ustep(i);
            }

            // Scale and anchor

            mean(ubar,ubase);

            s = 1;
            sfirst = 1;

            for ( i = 0 ; i < M ; i++ )
            {
                for ( j = 0 ; j < d ; j++ )
                {
                    // Short-circuit logic: smod not calculated unless first condition met
                    if ( ( ubase(i)(j) < ubar(j) ) && ( ( ( smod = (1.0-(ZERO_ABOVE*d))/(d*(ubar(j)-ubase(i)(j))) ) < s ) || sfirst ) )
                    {
                        s = smod;
                        sfirst = 0;
                    }
                }
            }

            for ( i = 0 ; i < M ; i++ )
            {
                for ( j = 0 ; j < d ; j++ )
                {
                    ubase("&",i)("&",j) = (s*(ubase(i)(j)-ubar(j))) + (1.0/d);
                }
            }

            // Reset relevant parts of kernel

            refactorVV();
        }

        // Termination checks

        itcnt++;

        errstream() << optdist;

        if ( !(++itcnt%FEEDBACK_CYCLE) )
        {
            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
            {
                errstream() << "$$";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
            {
                errstream() << "&&";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
            {
                errstream() << "==";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
            {
                errstream() << "@@";
            }
        }

        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
        {
            errstream() << "=outer=" << itcnt << "=  ";
        }

        if ( xmtrtime > 1 )
        {
             curr_time = TIMECALL;

            if ( TIMEDIFFSEC(curr_time,start_time) > xmtrtime )
            {
                timeout = 1;
            }
        }

        if ( !timeout )
        {
            timeout = kbquitdet("Multi-user rank, outer loop",uservars,varnames,vardescr);
        }
    }

    // Update locbasisgt

    reconstructlocbasisgt();

    return locres;
}















// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================

/*

OLD TRAINING ALGORITHMS

int SVM_MvRank::train(int &res, svmvolatile int &killSwitch)
{
    // This one works but does not have scaling or anchoring

    int locres = 0;
    int isopt = 0;
    unsigned long long itcnt = 0;
    int badstep = 0;
    int i,j;
    int timeout = 0;
    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    double xmtrtime = maxtraintime();
    double maxitcntval = maxitermvrank();
    double beta = betarank();
    double stepscalefactor = lrmvrank();
    double outertol = ztmvrank();
    double *uservars[] = { &maxitcntval, &xmtrtime, &stepscalefactor, &beta, &outertol, NULL };
    const char *varnames[] = { "itercount", "traintime", "stepscale", "beta", "outertol", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Step scale", "beta factor", "Outer optimality tolerance", NULL };

    // Work out constants

    int M = NbasisVV();
    int d = bdim;

    if ( !M || !d )
    {
        return 0;
    }

    // Steps and such-like

    Vector<Vector<double> > &ubase = locbasis;
    Vector<Vector<double> > ugrad(ubase); // M.ubase
    Vector<Vector<double> > ustep(ubase);

    // Randomise/normalise basis

    for ( i = 0 ; i < M ; i++ )
    {
        ubase("&",i) = 0.0;

        while ( sum(ubase(i)) == 0 )
        {
            randfill(ubase("&",i));
        }

        ubase("&",i) /= sum(ubase(i));
    }

    // Various matrices

    Matrix<double> Lalpha(M,M);
    Matrix<double> Mbold(M,M);
    Matrix<double> Mbeta(M,M);

    for ( i = 0 ; i < M ; i++ )
    {
        for ( j = 0 ; j < M ; j++ )
        {
            Mbeta("&",i,j) = beta;
        }
    }

    Vector<double> temp(d);

    refactorVV();

    // Main training loop

    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcntval ) || !maxitcntval ) && !timeout )
    {
        // Train inner loop for alpha calculation

        locres |= SVM_Scalar::train(res,killSwitch);

        // Update matrices

        calcLalpha(Lalpha);

        Mbold = Lalpha;
        Mbold.diagoffset(-M*beta);
        Mbold += Mbeta;

        // Calculate projected gradients

        for ( i = 0 ; i < M ; i++ )
        {
            ugrad("&",i) = 0.0;

            for ( j = 0 ; j < M ; j++ )
            {
                ugrad("&",i).scaleAdd(Mbold(i,j),ubase(j));
            }

            ugrad("&",i) -= sum(ugrad(i))/d;
        }

        // Test optimality

        isopt = 1;

//        for ( i = 0 ; ( i < M ) && isopt ; i++ )
//        {
//            for ( j = 0 ; ( j < d ) && isopt ; j++ )
//            {
//                if ( ( ( ugrad(i)(j) < -outertol ) && ( ubase(i)(j) > 0 ) ) || ( ( ugrad(i)(j) > outertol ) && ( ubase(i)(j) < 1 ) ) )
//                {
//                    isopt = 0;
//                }
//            }
//        }

        double optdist = 0;

        for ( i = 0 ; ( i < M ) ; i++ )
        {
            for ( j = 0 ; ( j < d ) ; j++ )
            {
                if ( ( ( ugrad(i)(j) < -outertol ) && ( ubase(i)(j) > 0 ) ) || ( ( ugrad(i)(j) > outertol ) && ( ubase(i)(j) < 1 ) ) )
                {
                    isopt = 0;
                    if ( ( ugrad(i)(j) < -outertol ) && ( ubase(i)(j) > 0 ) && ( optdist < -ugrad(i)(j) ) ) { optdist = -ugrad(i)(j); }
                    if ( ( ugrad(i)(j) >  outertol ) && ( ubase(i)(j) < 1 ) && ( optdist <  ugrad(i)(j) ) ) { optdist =  ugrad(i)(j); }
                }
            }
        }
errstream() << "Optdist = " << optdist << "\n";

        // If not optimal then process step

        if ( !isopt )
        {
            // Calculate step

            badstep = 1;

            for ( i = 0 ; i < M ; i++ )
            {
                ustep("&",i)  = ugrad(i);
                ustep("&",i) *= stepscalefactor;

                for ( j = 0 ; j < d ; j++ )
                {
                    if ( ubase(i)(j)+ustep(i)(j) < 0 )
                    {
                        ustep("&",i)("&",j) = -ubase(i)(j);
                    }

                    else if ( ubase(i)(j)+ustep(i)(j) > 1 )
                    {
                        ustep("&",i)("&",j) = 1-ubase(i)(j);
                    }

                    if ( ubase(i)(j)+ustep(i)(j) > outertol )
                    {
                        badstep = 0;
                    }
                }
            }

            if ( badstep )
            {
                // Step will take us to zero, so scale to avoid divide by zero error during renormalisation

                for ( i = 0 ; i < M ; i++ )
                {
                    ustep("&",i) /= 2.0;
                }
            }

            // take step and re-normalise

            for ( i = 0 ; i < M ; i++ )
            {
                ubase("&",i) += ustep(i);
                ubase("&",i) /= sum(ubase(i));
            }

            // Reset relevant parts of kernel

            refactorVV();
        }

        // Termination checks

        itcnt++;

        if ( !(++itcnt%FEEDBACK_CYCLE) )
        {
            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
            {
                errstream() << "$$";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
            {
                errstream() << "&&";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
            {
                errstream() << "==";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
            {
                errstream() << "@@";
            }
        }

        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
        {
            errstream() << "=outer=" << itcnt << "=  ";
        }

        if ( xmtrtime > 1 )
        {
             curr_time = TIMECALL;

            if ( TIMEDIFFSEC(curr_time,start_time) > xmtrtime )
            {
                timeout = 1;
            }
        }

        if ( !timeout )
        {
            timeout = kbquitdet("Multi-user rank, outer loop",uservars,varnames,vardescr);
        }
    }

    // Update locbasisgt

    for ( i = 0 ; i < M ; i++ )
    {
        locbasisgt("&",i) = ubase(i);
    }

    return locres;
}




this one does not work at all

int SVM_MvRank::train(int &res, svmvolatile int &killSwitch)
{
    int locres = 0;
    int isopt = 0;
    unsigned long long itcnt = 0;
    int i,j;
    int timeout = 0;
    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    double xmtrtime = maxtraintime();
    double maxitcntval = maxitermvrank();
    double beta = betarank();
    double stepscalefactor = lrmvrank();
    double outertol = ztmvrank();
    double *uservars[] = { &maxitcntval, &xmtrtime, &stepscalefactor, &beta, &outertol, NULL };
    const char *varnames[] = { "itercount", "traintime", "stepscale", "beta", "outertol", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Step scale", "beta factor", "Outer optimality tolerance", NULL };

    // Work out constants

    int M = NbasisVV();
    int d = bdim;
    double grsum;

    if ( !M || !d )
    {
        return 0;
    }

    // Steps and such-like

    Vector<Vector<double> > &ubase = locbasis;
    Vector<Vector<double> > ugrad(ubase); // M.ubase
    Vector<Vector<double> > ustep(ubase);

    // Active sets and pivots

    Vector<Vector<int> > ii(M);
    Vector<int> ni(M);

    for ( i = 0 ; i < M ; i++ )
    {
        retVector<int> tmpva;

        ii("&",i) = cntintvec(d,tmpva);
        ni("&",i) = d;
    }

    // Randomise/normalise basis

    for ( i = 0 ; i < M ; i++ )
    {
        randfill(ubase("&",i));
        ubase("&",i) /= sum(ubase(i));
    }

    // Various matrices

    Matrix<double> Lalpha(M,M);
    Matrix<double> Mbold(M,M);
    Matrix<double> Mbeta(M,M);

    for ( i = 0 ; i < M ; i++ )
    {
        for ( j = 0 ; j < M ; j++ )
        {
            Mbeta("&",i,j) = beta;
        }
    }

    Vector<double> temp(d);

    double rescalefacttemp;
    double rescalefact;
    double rescalei;
    double rescalej;

    refactorVV();

    // Main training loop

    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcntval ) || !maxitcntval ) && !timeout )
    {
        // Train inner loop for alpha calculation

        locres |= SVM_Scalar::train(res,killSwitch);

        // Update matrices

        calcLalpha(Lalpha);

        Mbold = Lalpha;
        Mbold.diagoffset(-M*beta);
        Mbold += Mbeta;

        // Calculate gradients

        for ( i = 0 ; i < M ; i++ )
        {
            ugrad("&",i) = 0.0;

            for ( j = 0 ; j < M ; j++ )
            {
                ugrad("&",i).scaleAdd(Mbold(i,j),ubase(j));
            }
        }

        // Test optimality

        isopt = 1;

        for ( i = 0 ; ( i < M ) && isopt ; i++ )
        {
            if ( ni(i) )
            {
                grsum = sum(ugrad(i)(ii(i))(0,1,ni(i)-1))/ni(i);

                for ( j = 0 ; ( j < ni(i) ) && isopt ; j++ )
                {
                    if ( ( ugrad(i)(ii(i)(j))-grsum > outertol ) || ( ugrad(i)(ii(i)(j))-grsum < -outertol ) )
                    {
                        isopt = 0;
                    }
                }
            }
        }

        // If not optimal then process step

errstream() << "phantomx 0: " << itcnt << "\n";
errstream() << "phantomx 10: " << ubase << "\n";
errstream() << "phantomx 20: " << ugrad << "\n";
errstream() << "phantomx 30: " << Lalpha << "\n";
errstream() << "phantomx 40: " << alphaR() << "\n";
errstream() << "phantomx 50: " << Mbold << "\n";
errstream() << "phantomx 60: " << beta << "\n";
errstream() << "phantomx 70: " << Gp() << "\n";

        if ( !isopt )
        {
            // Reset pivots

            ni = d;

            // Calculate appropriately bounded step

            rescalefact = 0;

            while ( rescalefact == 0 )
            {
                // Calculate unrescaled step

                for ( i = 0 ; i < M ; i++ )
                {
                    if ( ni(i) )
                    {
                        ustep("&",i)("&",ii(i))("&",0,1,ni(i)-1) = ugrad(i)(ii(i))(0,1,ni(i)-1);
                        ustep("&",i)("&",ii(i))("&",0,1,ni(i)-1) -= (sum(ugrad(i)(ii(i))(0,1,ni(i)-1))/ni(i));
                        ustep("&",i)("&",ii(i))("&",0,1,ni(i)-1) *= stepscalefactor;
                    }
                }

                // Rescale step to fit constraints

                rescalefact = 1;
                rescalei = -1;
                rescalej = -1;

                for ( i = 0 ; i < M ; i++ )
                {
                    if ( ni(i) )
                    {
                        for ( j = 0 ; j < ni(i) ; j++ )
                        {
                            if ( (ubase(i)(ii(i)(j)))+(ustep(i)(ii(i)(j))) < 0 )
                            {
                                rescalefacttemp = -(ubase(i)(ii(i)(j)))/(ustep(i)(ii(i)(j)));

                                if ( rescalefacttemp < rescalefact )
                                {
                                    rescalefact = rescalefacttemp;
                                    rescalei = i;
                                    rescalej = j;
                                }
                            }
                        }
                    }
                }

                if ( rescalefact == 0 )
                {
                    ii("&",rescalei).blockswap(rescalej,ni(rescalei)-1);
                    ni("&",rescalei)--;
                    j--;
                }
            }

            // Take step

errstream() << "phantomx 80: " << ustep << "\n";
errstream() << "phantomx 90: " << rescalefact << "\n";

            for ( i = 0 ; i < M ; i++ )
            {
                ubase("&",i)("&",ii(i))("&",0,1,ni(i)-1).scaleAdd(rescalefact,ustep(i)(ii(i))(0,1,ni(i)-1));
            }

            // Reset relevant parts of kernel

            refactorVV();
        }

        // Termination checks

        itcnt++;

        if ( !(++itcnt%FEEDBACK_CYCLE) )
        {
            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
            {
                errstream() << "$$";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
            {
                errstream() << "&&";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
            {
                errstream() << "==";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
            {
                errstream() << "@@";
            }
        }

        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
        {
            errstream() << "=outer=" << itcnt << "=  ";
        }

        if ( xmtrtime > 1 )
        {
             curr_time = TIMECALL;

            if ( TIMEDIFFSEC(curr_time,start_time) > xmtrtime )
            {
                timeout = 1;
            }
        }

        if ( !timeout )
        {
            timeout = kbquitdet("Multi-user rank, outer loop",uservars,varnames,vardescr);
        }
    }

    // Update locbasisgt

    for ( i = 0 ; i < M ; i++ )
    {
        locbasisgt("&",i) = ubase(i);
    }

    return locres;
}

*/

























































































