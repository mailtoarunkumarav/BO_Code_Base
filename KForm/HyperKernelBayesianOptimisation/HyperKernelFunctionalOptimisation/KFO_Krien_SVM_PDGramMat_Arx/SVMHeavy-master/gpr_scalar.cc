
//
// Scalar regression GP
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "gpr_scalar.h"

GPR_Scalar::GPR_Scalar() : GPR_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setsigma(DEFAULT_SIGMA);

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(NULL);

    return;
}

GPR_Scalar::GPR_Scalar(const GPR_Scalar &src) : GPR_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setsigma(DEFAULT_SIGMA);

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(NULL);
    assign(src,0);

    return;
}

GPR_Scalar::GPR_Scalar(const GPR_Scalar &src, const ML_Base *srcx) : GPR_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setsigma(DEFAULT_SIGMA);

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(srcx);
    assign(src,1);

    return;
}













// Close-enough-to-infinity for variance

#define CETI 1000.0

int GPR_Scalar::train(int &res, svmvolatile int &killSwitch)
{
    int Nineq = NNC(-1)+NNC(+1);

    int locres = 0;

    if ( !Nineq )
    {
        locres = getQ().train(res,killSwitch); 
    }

    else
    {
        int i,j;

        // Get indices of inequality constraints

        Vector<int> indin(Nineq);

        for ( i = 0, j = 0 ; i < N() ; i++ )
        {
            if ( ( d()(i) == +1 ) || ( d()(i) == -1 ) )
            {
                indin("&",j) = i;

                j++;
            }
        }

        // Start with "base" mean/variance

        retVector<double> tmpva;
        retVector<double> tmpvb;
        retVector<gentype> tmpvc;
        retVector<gentype> tmpvd;

        Vector<gentype> bmean(y()(indin,tmpvc));
        Vector<double> bvar(sigmaweight()(indin,tmpvb));

        bvar *= sigma();

        getQ().sety(indin,bmean);
        getQsetsigmaweight(indin,bvar/sigma());

        locres = getQ().train(res,killSwitch); 

        // Main EP loop

        Vector<gentype> smean(bmean);
        Vector<double> svar(bvar);

        Vector<gentype> pmean(smean);
        Vector<double> pvar(svar);

        int isdone = 0;

        while ( !isdone )
        {
            // Record current mean/variance (for convergence testing)

            pmean = smean;
            pvar  = svar;

            // Loop through constraints

            for ( j = 0 ; j < Nineq ; j++ )
            {
                i = indin(j);

                // Disable vector and update

                getQ().setd(i,0);
                locres |= getQ().train(res,killSwitch); 

                // Get mean and variance ignoring this vector

                gentype gvari,gmeani;

                getQconst().varTrainingVector(gvari,gmeani,i);

                double vari  = (double) gvari;
                double mui   = (double) gmeani;
                double yi    = (double) d()(i);
                double targi = (double) y()(i);

                // Factor out offset

                mui -= targi;

                // Approximate - first (3.58) in Rasmussen

                double nu = sigma()*sigmaweight()(i);
                double zi = yi*mui/sqrt(nu+vari);

                double Phizi = normPhi(zi);
                double phizi = normphi(zi);

                double varhati = vari - (((vari*vari*phizi)/((nu+vari)*Phizi))*(zi+(phizi/Phizi)));
                double muhati  = mui  + ((yi*vari*phizi)/(sqrt(nu+vari)*Phizi));

                // Approximate - next (3.59) in Rasmussen
                //
                // NB: for numerical reasons we limit how bit vartildei can be
                //     (remember that this is the offset on the diagonal of the 
                //      Hessian, so if it's too small then the Cholesky is likely 
                //      to be numerically unstable).

                double vartildei = 1/((1/varhati)-(1/vari));           vartildei = ( vartildei <= CETI ) ? vartildei : CETI;
                double mutildei  = vartildei*((muhati/varhati)-(mui/vari));

                // Factor offset back in

                //vartildei += sigma()*sigmaweight()(i); old version - now put in as logistic variance!
                mutildei  += targi;

                // Update stuff.

                svar("&",j)  = vartildei;
                smean("&",j) = mutildei;

                // Re-enable vector and update (no need to train)

                getQ().setd(i,2);
            }

            // Set new mean/var (no need to train)

            getQ().sety(indin,smean);
            getQsetsigmaweight(indin,svar/sigma());

            // test convergence

            double stepsize = norm2(pmean(indin,tmpvc)-smean(indin,tmpvd))/Nineq;
errstream() << "!" << stepsize << "!";

//FIXME: may need better convergence test.
            isdone = ( stepsize <= 1e-3 ) ? 1 : 0;
        }

        // Final training update

        locres |= getQ().train(res,killSwitch);
    }

    return locres;
}

std::ostream &GPR_Scalar::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "GPR (Scalar/Real)\n";

    GPR_Generic::printstream(output,dep+1);

    return output;
}

std::istream &GPR_Scalar::inputstream(std::istream &input )
{
    GPR_Generic::inputstream(input);

    return input;
}

