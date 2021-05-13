
//
// Transfer learning setup
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "xferml.h"
#include "svm_scalar.h"
#include "mlcommon.h"
#include "numbase.h"

#define FEEDBACK_CYCLE 10
#define MAJOR_FEEDBACK_CYCLE 50
#define NEWTSCALE 1.0

Vector<gentype> &randitall(Vector<gentype> &x);
Vector<gentype> &randitall(Vector<gentype> &x)
{
    int i;
    int d = x.size();

    for ( i = 0 ; i < d ; i++ )
    {
        //randnfill(x("&",i).force_double());
        randfill(x("&",i).force_double());
    }

    return x;
}

Vector<double> &calcgrad(Matrix<double> &hess, Vector<double> &res, const SVM_Scalar &core, const Vector<SVM_Generic *> &cases, int n, Vector<double> &outpart, Matrix<double> &Kiipq, double &avediagm1);
Vector<double> &calcgrad(Matrix<double> &hess, Vector<double> &res, const SVM_Scalar &core, const Vector<SVM_Generic *> &cases, int n, Vector<double> &outpart, Matrix<double> &Kiipq, double &avediagm1)
{
    int p,q,i,j,k;
    int M = cases.size();
    int Ntot = 0;

    res = 0.0;
    hess = 0.0;

    Kiipq = 0.0;

// Version 1: want sum_i beta_i = 1
//
//    hess.diagoffset(2*(core.C()));
//
//    double alphaerror = 2*(core.C())*(sum(core.alphaR())-core.eps());
//
////errstream() << "phantomxy 0: " << alphaerror << "\n";
//    res += alphaerror;

    double tempq,tempK;

    // Main error gradient: minimise sum_i C.(g(x_i)-z_i)^2

    for ( k = 0 ; k < M ; k++ )
    {
        const SVM_Generic &kcase = *(cases(k));
        int N = kcase.N();

        Ntot += N;

        double kC = 1; //kcase.C();

        for ( i = 0 ; i < N ; i++ )
        {
            if ( kcase.alphaState()(i) )
            {
//                gentype tempi;
//
//                kcase.ggTrainingVector(tempi,i);
//                tempi -= kcase.y()(i);

                gentype tempi;

                kcase.ggTrainingVector(tempi,i);
                tempi -= kcase.y()(i);

//                double tempi;
//
//                tempi = ( kcase.alphaR()(i) > 0 ) ? 1 : ( ( kcase.alphaR()(i) < 0 ) ? -1 : 0 );

                for ( j = 0 ; j < N ; j++ )
                {
                    if ( kcase.alphaState()(j) )
                    {
//                        gentype tempj(tempi);
//
//                        tempj *= kcase.alpha()(j);

                        double tempj = (double) tempi;

                        tempj *= kcase.alphaR()(j);

//                        double tempj = tempi;
//
//                        tempj *= kcase.alphaR()(j);

                        for ( q = 0 ; q < n ; q++ )
                        {
                            tempq = ((double) tempj)*(core.alphaR()(q));

                            for ( p = 0 ; p < n ; p++ )
                            {
                                double wpg = core.K4(tempK,-42-i,-42-j,p,q,NULL,&(kcase.x(i)),&(kcase.x(j)),&(core.x(p)),&(core.x(q)),&(kcase.xinfo(i)),&(kcase.xinfo(j)),&(core.xinfo(p)),&(core.xinfo(q)));

                                res("&",p) += 4*tempq*wpg*kC*(kcase.Cweight()(i));
                                hess("&",p,q) += 4*wpg*kC*(kcase.Cweight()(i));

                                if ( i == j )
                                {
                                    Kiipq("&",p,q) += wpg;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Kernel diagonal approximate normalisation term
    //
    // Want inheritted kernel to approximately satisfy K(xi,xi) = 1
    //
    // Term: ( 1/N sum_i sum_pq beta_p beta_q K(i,i,p,q) - eps )^2

//    double avediagm1 = 0.0;
    avediagm1 = 0.0;

    for ( p = 0 ; p < n ; p++ )
    {
        for ( q = 0 ; q < n ; q++ )
        {
            avediagm1 += (core.alphaR()(p))*(core.alphaR()(q))*Kiipq(p,q);
            outpart("&",p) += (core.alphaR()(q))*Kiipq(p,q);
        }
    }

    avediagm1 /= (double) Ntot;
    avediagm1 -= core.eps();
//errstream() << "phantomxyz 000: " << avediagm1 << "\n";

    outpart /= (double) Ntot;

//    avediagm1 -= ((double) Ntot)*(core.eps());

    for ( p = 0 ; p < n ; p++ )
    {
//        res("&",p) += 4*(core.C())*avediagm1*outpart(p);
        res("&",p) += 4*(core.C())*avediagm1*outpart(p)*Ntot;

        for ( q = 0 ; q < n ; q++ )
        {
//            hess("&",p,q) += 8*(core.C())*outpart(p)*outpart(q);
//            hess("&",p,q) += 4*(core.C())*avediagm1*Kiipq(p,q);
            hess("&",p,q) += 8*(core.C())*outpart(p)*outpart(q)*Ntot;
            hess("&",p,q) += 4*(core.C())*avediagm1*Kiipq(p,q)*Ntot;
        }
    }

    for ( p = 0 ; p < n ; p++ )
    {
        hess("&",p,p) += ( hess(p,p) < 1e-6 ) ? 1e-6 : hess(p,p);
    }

    return res;
}

int xferMLtrain(svmvolatile int &killSwitch, SVM_Scalar &core, Vector<SVM_Generic *> &cases, int n, int maxitcntint, double xmtrtime, double soltol)
{
    // Rounding

    int res = 0;
    int M = cases.size();

    if ( M == 0 )
    {
        return 1;
    }

    // We just *assume* that each case inherits from core with kernel 801 and
    // is "sensible" (m=2, nothing fancy going on).

    int z = 0;
    int i,k;

    // Initialise core - add random training vectors

    int d = (*(cases(z))).xspaceDim(); // We assume that this is consistent!

    while ( core.NNC(2) > n )
    {
        i = 0;

        while ( core.d()(i) != 2 )
        {
            i++;
        }

        core.removeTrainingVector(i);
    }

    Vector<gentype> randx(d);

    for ( i = core.N()-1 ; i >= 0 ; i-- )
    {
        if ( core.d()(i) == 2 )
        {
            SparseVector<gentype> tempx;

            tempx = randitall(randx);

            core.setx(i,tempx);
        }

        else
        {
            core.removeTrainingVector(i);
        }
    }

    while ( core.NNC(2) < n )
    {
        SparseVector<gentype> tempx;

        tempx = randitall(randx);

        core.qaddTrainingVector(core.N(),0.0,tempx);
    }

    // Initialise core - randomise weights

    core.setFixedBias(0.0); // so it doesn't think it's beta-nonoptimal
    core.setQuadraticCost(); // alpha should be unlimited
    core.randomise(1);

    // Preliminary outer train

    for ( k = 0 ; k < M ; k++ )
    {
        (*(cases("&",k))).resetKernel();
        (*(cases("&",k))).train(res,killSwitch);
    }

    // Some data stores

    Vector<double> alphaold(n);
    Vector<double> alphastep(n);
    Vector<double> alphanew(n);
    Vector<double> alphaGrad(n);

    Matrix<double> alphaHess(n,n);
    Matrix<double> invhess(n,n);

    Vector<double> scratch1(n);
    Matrix<double> scratch2(n,n);

    // Re-weight kernel to ensure weights are sane!  We want the average diagonal kernel to be ~eps

    double avediagm1;

    calcgrad(alphaHess,alphaGrad,core,cases,n,scratch1,scratch2,avediagm1);

errstream() << "diagonal average " << avediagm1 << "\n";
    avediagm1 += core.eps();

    double kweight = (core.eps())/avediagm1;

//FIXME: assuming kernel is at most a linear sum of kernels
    for ( i = 0 ; i < core.getKernel().size() ; i++ )
    {
        core.getKernel_unsafe().setWeight(kweight*(core.getKernel().cWeight(i)),i);
    }

    // Second preliminary outer train

    for ( k = 0 ; k < M ; k++ )
    {
        (*(cases("&",k))).resetKernel();
        (*(cases("&",k))).train(res,killSwitch);
    }

    // Setup for main training loop

    double maxitcnt = maxitcntint;
    double *uservars[] = { &maxitcnt, &xmtrtime, &soltol, NULL };
    const char *varnames[] = { "itercount", "traintime", "soltol", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Solution tolerance", NULL };

    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    unsigned long long itcnt = 0;
    int timeout = 0;
    int isopt = 0;

//errstream() << "phantomx -1: " << core << "\n";
    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcnt ) || !maxitcnt ) && !timeout )
    {
        // State recording

        alphaold = core.alphaR();

        // Gradient calculation (core)

        double avediagm1;

        calcgrad(alphaHess,alphaGrad,core,cases,n,scratch1,scratch2,avediagm1);

//        // Unitize diagonal
//
//        alphaold *= sqrt((core.eps())/avediagm1);
//        alphaGrad *= sqrt((core.eps())/avediagm1);
//        alphaHess *= sqrt((core.eps())/avediagm1);

        // Newton step (important or scaling gets messed up)

//errstream() << "phantomx 0: alpha grad: " << alphaGrad << "\n";
//errstream() << "phantomx 0: alpha hess: " << alphaHess << "\n";
        alphaGrad *= alphaHess.inve(invhess);
//errstream() << "phantomx 0: alpha hess inver: " << invhess << "\n";

        // Step calculation (core)

        alphastep =  alphaGrad;
        alphastep *= -NEWTSCALE;

        // New alpha calculation

        alphanew =  alphaold;
        alphanew += alphastep;

//errstream() << "phantomx 0: alpha old: " << alphaold << "\n";
//errstream() << "phantomx 0: alpha step: " << alphastep << "\n";
//errstream() << "phantomx 0: alpha new: " << alphanew << "\n";
        // Core alpha update

        core.setAlphaR(alphanew);

        // Optimality guessing

        double stepsize = (abs2(alphastep))/n;
errstream() << "Step " << stepsize << "\n";

        isopt = itcnt && ( stepsize <= soltol ) && ( abs2(avediagm1) < 0.1 ) ? 1 : 0;
//isopt = 0;

        // Outer train

        for ( k = 0 ; k < M ; k++ )
        {
            (*(cases("&",k))).resetKernel();
//errstream() << "phantomx 0: " << (*(cases("&",k))).Gp() << "\n";
            (*(cases("&",k))).train(res,killSwitch);
        }

        if ( !(++itcnt%FEEDBACK_CYCLE) )
        {
            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
            {
                errstream() << "|\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
            {
                errstream() << "/\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
            {
                errstream() << "-\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
            {
                errstream() << "\\\b";
            }
        }

        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
        {
            errstream() << "=" << itcnt << "=  ";
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
            timeout = kbquitdet("Transfer learning optimisation",uservars,varnames,vardescr);
        }
    }

//errstream() << "phantomx -10: " << core << "\n";
    return !isopt ? 1 : 0;
}











































/*

Version 1: Based on forming whole problem at once.  Does not work

double calcextGpcore(int ia, int ib, const SVM_Binary &core, const Vector<SVM_Generic *> &cases);
double calcextGpcore(int ia, int ib, const SVM_Binary &core, const Vector<SVM_Generic *> &cases)
{
    int i,j,k;
    int M = cases.size();
    double res = 0;
    double tempres = 0;

    for ( k = 0 ; k < M ; k++ )
    {
        const SVM_Generic &casek = *cases(k);

        int N = casek.N();

        for ( i = 0 ; i < N ; i++ )
        {
            for ( j = 0 ; j < N ; j++ )
            {
                if ( (casek.alphaState())(i) && (casek.alphaState())(j) )
                {
                    res += ((double) ((casek.alpha())(i)*(casek.alpha()(j))))*core.K4(tempres,ia,ib,-42-i,-42-j,NULL,&(core.x(ia)),&(core.x(ib)),&(casek.x(i)),&(casek.x(j)),&(core.xinfo(ia)),&(core.xinfo(ib)),&(casek.xinfo(i)),&(casek.xinfo(j)));
                }
            }
        }
    }

    return res;
}

void calcextGpcore(Matrix<double> &extGp, Matrix<double> &extGpsigma, const SVM_Binary &core, const Vector<SVM_Generic *> &cases);
void calcextGpcore(Matrix<double> &extGp, Matrix<double> &extGpsigma, const SVM_Binary &core, const Vector<SVM_Generic *> &cases)
{
    NiceAssert( extGp.isSquare() );

    int n = extGp.numRows();

    int ia,ib;

    for ( ia = 0 ; ia < n ; ia++ )
    {
        extGp("&",ia,ia) = calcextGpcore(ia,ia,core,cases);
        extGpsigma("&",ia,ia) = 0.0;
    }

    for ( ia = 1 ; ia < n ; ia++ )
    {
        for ( ib = 0 ; ib < ia ; ib++ )
        {
            extGp("&",ia,ib) = calcextGpcore(ia,ib,core,cases);
            extGp("&",ib,ia) = extGp(ia,ib);
        }
    }

    for ( ia = 1 ; ia < n ; ia++ )
    {
        for ( ib = 0 ; ib < ia ; ib++ )
        {
            extGpsigma("&",ia,ib) = extGp(ia,ia)+extGp(ib,ib)-extGp(ia,ib)-extGp(ib,ia);
            extGpsigma("&",ib,ia) = extGpsigma(ia,ib);
        }
    }

    return;
}

int xferMLtrain(svmvolatile int &killSwitch, SVM_Binary &core, Vector<SVM_Generic *> &cases, int n, int maxitcntint, double xmtrtime, double soltol)
{
//core.setFixedBias(0.0);
core.setLinBiasForce(-1.0);

    // Rounding

    n = 2*(n/2);

    int npos = n; //n/2; // n
    int nneg = 0; //n/2; // 0

    int res = 0;
    int M = cases.size();

    if ( M == 0 )
    {
        return 1;
    }

    // We just *assume* that each case inherits from core with kernel 801 and
    // is "sensible" (m=2, nothing fancy going on).

    int z = 0;
    int i,k;

    // Initialise core - want n/2 positive samples and n/2 negative

    int d = (*(cases(z))).xspaceDim(); // We assume that this is consistent!

    while ( core.NNC(-1) > nneg )
    {
        i = 0;

        while ( core.d()(i) != -1 )
        {
            i++;
        }

        core.removeTrainingVector(i);
    }

    while ( core.NNC(+1) > npos )
    {
        i = 0;

        while ( core.d()(i) != +1 )
        {
            i++;
        }

        core.removeTrainingVector(i);
    }

    Vector<gentype> randx(d);

    for ( i = core.N()-1 ; i >= 0 ; i-- )
    {
        if ( core.d()(i) )
        {
            SparseVector<gentype> tempx;

            tempx = randitall(randx);

            core.setx(i,tempx);
        }

        else
        {
            core.removeTrainingVector(i);
        }
    }

    while ( core.NNC(-1) < nneg )
    {
        SparseVector<gentype> tempx;

        tempx = randitall(randx);

        core.qaddTrainingVector(core.N(),-1,tempx);
    }

    while ( core.NNC(+1) < npos )
    {
        SparseVector<gentype> tempx;

        tempx = randitall(randx);

        core.qaddTrainingVector(core.N(),+1,tempx);
    }

    core.randomise(0);

    // Preliminary outer train

    for ( k = 0 ; k < M ; k++ )
    {
        (*(cases("&",k))).resetKernel();
        (*(cases("&",k))).train(res,killSwitch);
    }

    // Setup for main training loop

    Matrix<double> extGp(n,n);
    Matrix<double> extGpsigma(n,n);
    Matrix<double> extGpn(n,1);

    for ( i = 0 ; i < core.N() ; i++ )
    {
        extGpn("&",i,z) = (double) core.d()(i);
    }

    Vector<gentype> alphaold;

    double maxitcnt = maxitcntint;
    double *uservars[] = { &maxitcnt, &xmtrtime, &soltol, NULL };
    const char *varnames[] = { "itercount", "traintime", "soltol", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Solution tolerance", NULL };

    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    unsigned long long itcnt = 0;
    int timeout = 0;
    int isopt = 0;

    core.SVM_Scalar::setGpnExt(NULL,&extGpn);

    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcnt ) || !maxitcnt ) && !timeout )
    {
        // State recording

        alphaold = core.alpha();

        // Inner train

        calcextGpcore(extGp,extGpsigma,core,cases);
        core.SVM_Scalar::setGp(&extGp,&extGpsigma,&extGpsigma);
        core.train(res,killSwitch);
errstream() << "phantomxy 0: " << extGp << "\n";
errstream() << "phantomxy 1: " << extGpn << "\n";
errstream() << "phantomxy 2: " << core << "\n";
        core.SVM_Scalar::setGp(NULL,NULL);

        // Optimality guessing

        double stepsize = (norm2(alphaold-core.alpha()))/n;

        errstream() << "Step " << stepsize << "\n";

        isopt = itcnt && ( stepsize <= soltol ) ? 1 : 0;
isopt = 0;

        // Outer train

        for ( k = 0 ; k < M ; k++ )
        {
            (*(cases("&",k))).resetKernel();
//errstream() << "phantomx 0: " << (*(cases("&",k))).Gp() << "\n";
            (*(cases("&",k))).train(res,killSwitch);
        }

        if ( !(++itcnt%FEEDBACK_CYCLE) )
        {
            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
            {
                errstream() << "|\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
            {
                errstream() << "/\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
            {
                errstream() << "-\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
            {
                errstream() << "\\\b";
            }
        }

        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
        {
            errstream() << "=" << itcnt << "=  ";
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
            timeout = kbquitdet("Transfer learning optimisation",uservars,varnames,vardescr);
        }
    }

    core.SVM_Scalar::setGpnExt(&extGpn,NULL);

    return !isopt ? 1 : 0;
}
*/

