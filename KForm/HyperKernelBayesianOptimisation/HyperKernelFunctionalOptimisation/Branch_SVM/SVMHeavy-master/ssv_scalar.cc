
//
// Super-Sparse SVM scalar class
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
#include "ssv_scalar.h"



#define RECINTERVAL 5
#define GAMMA_STV 0.2
#define GAMMA_END 1.2
#define GAMMA_INC 0.2



std::ostream &SSV_Scalar::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Scalar SSV\n\n";

    SSV_Generic::printstream(output,dep+1);

    return output;
}

std::istream &SSV_Scalar::inputstream(std::istream &input)
{
//    wait_dummy dummy;

    SSV_Generic::inputstream(input);

    return input;
}

// Copy src to dest, assuming matching indices

const SparseVector<gentype> &zxfer(SparseVector<gentype> &dest, const SparseVector<double> &src)
{
    int k;

    for ( k = 0 ; k < src.indsize() ; k++ )
    {
        (dest.direref(k)).dir_double() = src.direcref(k);
    }

    return dest;
}

#define OLR 0.3

int SSV_Scalar::train(int &res, svmvolatile int &killSwitch)
{
    // The outer training loop deals with inequalities, specifically:
    // 
    // d = -1: g(x) <= y
    // d = 0:  nothing
    // d = +1: g(x) >= y
    // d = +2: g(x)  = y
    // 
    // d=0,+2 are trivial/done, but consider for example d = +1.  The relevant
    // terms in the objective are (say d_i = +1, incorporating all C in C_i):
    // 
    // C_i c(f+(g(x_i)-y_i))
    // 
    // where c is the usual cost (c(z) = z^2) and:
    // 
    // f+(r) = r if r <= 0, 0 otherwise
    // 
    // (that is, penalise only if g(x_i) <= y).  We could exactly implement
    // this using an active set approach, but that would take a lot of effort,
    // so instead we first note that the term can be replaced by:
    // 
    // C_i s(-d_i.(g(x_i)-y_i)) c(g(x_i)-y_i)
    // 
    // where s is the step function:
    // 
    // s(r) = +1 if r >= 0, 0 if r <= 0
    // 
    // Now we approximate s with the sigmoid, so:
    // 
    // C_i sig(-d_i.(g(x_i)-y_i)) c(g(x_i)-y_i)
    // 
    // where:
    // 
    // sig(r) = 1/(1+exp(-gamma.r))
    // 
    // which becomes a step as gamma -> infty.  This tells us our method:
    // 
    // 1. Start with small gamma
    // 2. Solve with weights:
    // 
    //    C_i -> C_i.sig(-d_i.(g(x_i)-y_i))
    // 
    // 3. Call intrain.
    // 4. Repeat from 2 until g(x_i)-y_i stabilises
    // 5. Increase gamma and repeat from 2 until target gamma reached.
    // 
    // But, because we're lazy and impatient, what we actually do is:
    // 
    // 1. Start with small gamma
    // 2. Solve with weights:
    // 
    //    C_i -> C_i.sig(-d_i.(g(x_i)-y_i))
    // 
    // 3. Call intrain.
    // 4. Increase gamma and repeat from 2 until target gamma reached.
    // 
    // and that's it.

    NiceAssert( zmin().ind() == zmax().ind() );

    incgvernum();

    if ( !N() || !Nzs() || !(zmin().indsize()) )
    {
        return 0;
    }

    SVM_Scalar *zzmodel = &zmodel;

    if ( isLinRegul() )
    {
        MEMNEW(zzmodel,SVM_Scalar(zmodel));

        (*zzmodel).seteps(0);
    }

    SVM_Scalar &betamodel = zmodel;
    SVM_Scalar &zzzmodel  = *zzmodel;

    NiceAssert( zmin() <= zmax() );

    int xres = 0;

    inbypass = 1;

    if ( !(SVM_Scalar::NNC(-1)) && !(SVM_Scalar::NNC(+1)) )
    {
        locCscale = 1.0;

        xres |= entrain(zzzmodel,betamodel,res,killSwitch);
    }

    else
    {
        double gamma;
        int i,j,k;
        Vector<double> gres(N());
        Vector<double> Csn(N());
        Vector<double> Csnstep(N());

        for ( gamma = GAMMA_STV ; gamma <= GAMMA_END ; gamma += GAMMA_INC )
        {
            errstream() << "gamma = " << gamma << " ,,, \n";

            int isopt = 0;

            k = 0;

            while ( !isopt )
            {
//FIXME should use xact() only here
                for ( i = 0 ; i < N() ; i++ )
                {
                    if ( ( d()(i) == -1 ) || ( d()(i) == +1 ) )
                    {
                        gres("&",i) = (double) b();

                        for ( j = 0 ; j < Nzs() ; j++ )
                        {
                            gres("&",i) += ((double) beta()(j))*Gp()(N()+j,i);
                        }

                        Csn("&",i) = 1/(1+exp(gamma*d()(i)*(gres(i)-ZY(i))));
                    }

                    else
                    {
                        Csn("&",i) = 1.0;
                    }
                }

                Csnstep  = Csn;
                Csnstep -= locCscale;

                locCscale.scaleAdd(OLR,Csnstep);

//errstream() << "phantomx 0: g = " << gres << "\n";
//errstream() << "phantomx 1: Cstep = " << Csnstep << "\n";
//errstream() << "phantomx 2: C = " << locCscale << "\n";
//errstream() << "phantomx 3: M = " << zM << "\n";
//errstream() << "phantomx 4: Gp = " << zmodel.Gp() << "\n";
//errstream() << "phantomx 5: n = " << zn << "\n";
//errstream() << "phantomx 6: beta = " << beta() << "\n";
//errstream() << "phantomx 7: b = " << b() << "\n";
                updatez();

                xres |= entrain(zzzmodel,betamodel,res,killSwitch);

                if ( k++ >= 10 ) { isopt = 1; }
            }
        }
    }

    if ( isLinRegul() )
    {
        MEMDEL(zzmodel);
    }

    inbypass = 0;

    return xres;
}

int SSV_Scalar::entrain(SVM_Scalar &zzmodel, SVM_Scalar &betamodel, int &res, svmvolatile int &killSwitch)
{
    int i,j,k,l,m;
    int isopt = 0;
    unsigned int itcnt = 0;

    incgvernum();

    // - Work out random start z vector
    // - Set up step/previous step variables

    int zdim = zmin().indsize();

    double rnval;

    const SparseVector<double> &zxmin = zmin();
    const SparseVector<double> &zxmax = zmax();

    SparseVector<double> zxwid;

    SparseVector<gentype> ztoset;

    zxwid =  zxmax;
    zxwid -= zxmin;

    ztoset.indalign(zxmin);

    Vector<SparseVector<double> > zprev(Nzs());
    Vector<SparseVector<double> > znext(Nzs());
    Vector<SparseVector<double> > zbest(Nzs());
    Vector<SparseVector<double> > zstep(Nzs());

    Vector<SparseVector<double> > zstepprev(Nzs());

    for ( i = 0 ; i < Nzs() ; i++ )
    {
        zprev("&",i) = zxmin;
        zstep("&",i) = zxmin;

        zstepprev("&",i) = zxmin;

        if ( ( z()(i).indsize() != z()(i).size() ) || ( z()(i).indsize() != zdim ) )
        {
            for ( j = 0 ; j < zdim ; j++ )
            {
                randfill(rnval); // uniform random (0,1)

                rnval *= zxwid.direcref(j);
                rnval += zxmin.direcref(j);

                zprev("&",i).direref(j) = rnval;
                zstep("&",i).direref(j) = 0.0;

                zstepprev("&",i).direref(j) = 0.0;
            }

            setz(i,zxfer(ztoset,zprev(i)));
        }

        else
        {
            for ( j = 0 ; j < zdim ; j++ )
            {
                zprev("&",i).direref(j) = (double) z()(i).direcref(j);

                if ( zprev(i).direcref(j) < zxmin.direcref(j) )
                {
                    zprev("&",i).direref(j) = zxmin.direcref(j);
                }

                if ( zprev(i).direcref(j) > zxmax.direcref(j) )
                {
                    zprev("&",i).direref(j) = zxmax.direcref(j);
                }

                zstep("&",i).direref(j) = 0.0;

                zstepprev("&",i).direref(j) = 0.0;
            }

            setz(i,zxfer(ztoset,zprev(i)));
        }

        znext("&",i) = zprev(i);
        zbest("&",i) = zprev(i);
    }

    // Optimisation loop

    Vector<double> zgrad(N()+Nzs());
    Vector<double> xgrad(N()+Nzs());
    Vector<double> rs(Nzs()+1);
    Vector<double> rscratch(Nzs()+1);
    Vector<double> cVal(N());
    Vector<double> g(N());
    Vector<double> gbest(N());
    Vector<double> gprev(N());
    Vector<double> betaR(Nzs());
    Vector<double> betaRprev(Nzs());
    Vector<double> betaRbest(Nzs());
    Vector<double> xfgrad(N());

    double biasR,biasRprev,biasRbest;
    double rscale,cscale;
    double inthgr;
    double objBest = -1;

    for ( i = 0 ; i < Nzs() ; i++ )
    {
        betaR("&",i) = (double) beta()(i);
    }

    biasR = (double) b();

    //setbeta(betaR);
    //setb(biasR);

    betamodel.setyqnd(n(),rscratch);
    betamodel.inintrain(killSwitch);

    // Update beta and b

    retVector<double> tmpva;

    betaR = (betamodel.alphaR())(zeroint(),1,Nzs()-1,tmpva);
    biasR = (betamodel.alphaR())(Nzs());

    betaRprev = betaR;
    biasRprev = biasR;

    betaRbest = betaR;
    biasRbest = biasR;

    for ( l = 0 ; l < xact().size() ; l++ )
    {
        cVal("&",xact()(l)) = calcCval(xact()(l));

        g("&",xact()(l)) = biasR;

        for ( i = 0 ; i < Nzs() ; i++ )
        {
            g("&",xact()(l)) += Gp()(i+N(),xact()(l))*betaR(i);
        }
    }

    gprev = g;
    gbest = g;

//errstream() << "phantomx M++ = " << M()(zeroint(),1,Nzs()-1,zeroint(),1,Nzs()-1) << "\n";
//errstream() << "phantomx M-+ = " << M()(Nzs(),zeroint(),1,Nzs()-1) << "\n";
//errstream() << "phantomx M+- = " << M()(zeroint(),1,Nzs()-1,Nzs(),"&") << "\n";
//errstream() << "phantomx M-- = " << M()(Nzs(),Nzs()) << "\n";
//errstream() << "phantomx n+ = " << n()(zeroint(),1,Nzs()-1) << "\n";
//errstream() << "phantomx n- = " << n()(Nzs()) << "\n";
//errstream() << "phantomx M++ recalc = " << 1000.0*Gp()(N(),1,N()+Nzs()-1,xact())*Gp()(xact(),N(),1,N()+Nzs()-1) << "\n";
//errstream() << "phantomx M-+ recalc = " << 1000.0*onedoublevec(xact().size())*Gp()(xact(),N(),1,N()+Nzs()-1) << "\n";
//errstream() << "phantomx M+- recalc = " << 1000.0*Gp()(N(),1,N()+Nzs()-1,xact())*onedoublevec(xact().size()) << "\n";
//errstream() << "phantomx M-- recalc = " << 1000.0*sum(onedoublevec(xact().size())) << "\n";
//errstream() << "phantomx n+ recalc = " << 1000.0*Gp()(N(),1,N()+Nzs()-1,xact())*ZY(xact()) << "\n";
//errstream() << "phantomx n- recalc = " << 1000.0*sum(ZY(xact())) << "\n";
//errstream() << "phantomx Gp() = " << Gp()(xact(),xact()) << "\n";
//errstream() << "phantomx 0: " << *this << "\n";
//    int dummyind = -1;

    double currobj = 0;
    double prevobj = -1;
    double maxitcnt = ssvmaxitcnt();
    double maxtime  = ssvmaxtime();
    double lr = ssvlr();
    double mom = ssvmom();
    double tol = ( ssvtol() > Opttol() ) ? ssvtol() : Opttol();
    double ovsc = ssvovsc();
    double *uservars[] = { &maxitcnt, &maxtime, &lr, &mom, &tol, &ovsc, NULL };
    double stepscale;
    const char *varnames[] = { "maxitcnt", "maxtime", "lr", "mom", "tol", "ovsc", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Learning rate", "Momentum factor", "Zero tolerance", "Over-step scaleback factor", NULL };

    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    int timeout = 0;
    int printnow = 0;

    j = 0;

    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcnt ) || !maxitcnt ) && !timeout )
    {
        errstream() << ".";

        int dummyind;

        // Work out base z step
        //
        // This involves calculating beta and b gradients, which is done by
        // calculating r/s as per comments at start of ssv_generic (kernel 
        // gradients in zmodel), subbing these back into zmodel temporarily, 
        // optimising to get beta and b gradients, the using eq 5 in Dem2.  

        // dS_{zj,xl}/dz_j = zgrad_l*z_j + xgrad_l*x_l,   where z_j = x_{j+N}

        for ( l = 0 ; l < xact().size() ; l++ )
        {
            SVM_Scalar::dK2delx(zgrad("&",xact()(l)),xgrad("&",xact()(l)),dummyind,j+N(),xact()(l));
        
            NiceAssert( dummyind < 0 );
        }

        // dS_{zj,zi}/dz_j = zgrad_{i+N}*z_j + xgrad_{i+N}*z_i,   where z_i = x_{i+N}

        for ( i = 0 ; i < Nzs() ; i++ )
        {
            SVM_Scalar::dK2delx(zgrad("&",i+N()),xgrad("&",i+N()),dummyind,j+N(),i+N());
        
            NiceAssert( dummyind < 0 );
        }

//errstream() << "\n\n";
//errstream() << "j = " << j << "\n";
//errstream() << "Gp = " << Gp() << "\n";
//errstream() << "zgrad = " << zgrad << "\n";
//errstream() << "xgrad = " << xgrad << "\n";
//errstream() << "\n\n";
        // Set up zmodel to calculate beta and b gradients ((6) in Dem2)

        for ( k = 0 ; k < zdim ; k++ )
        {
            rs = 0.0;

            // r(k) = gradient wrt element k of ..._j

            for ( l = 0 ; l < xact().size() ; l++ )
            {
                // We use this again later, so do it all at once

                xfgrad("&",xact()(l)) = ( ((zgrad(xact()(l)))*(zprev(j).direcref(k))) + ((xgrad(xact()(l)))*( (double) x(xact()(l)).direcref(k) )) );

                // cscale = ( U.dS_{xzj}/d(z_j.direcref(k)) )_l in (6), Dem2 (indices may differ)

                cscale = (cVal(xact()(l)))*xfgrad(xact()(l)); //( ((zgrad(xact()(l)))*( zprev(j).direcref(k) )) + ((xgrad(xact()(l)))*( (double) x(xact()(l)).direcref(k) )) );

                for ( i = 0 ; i < Nzs() ; i++ )
                {
                    // rscale = ( beta_j S_{zx} + V )_{il} in (6), Dem2 (indices may differ)

                    rscale  = ( betaR(j) * (Gp()(i+N(),xact()(l))) ) + ( ( i == j ) ? ( g(xact()(l)) - ZY(xact()(l)) )  : 0.0 );

//errstream() << "i = " << i << ": " << rscale << "*" << cscale << "\n";
                    // NB: we negate this because gp = -z in calculation
                    rs("&",i) -= rscale*cscale;
                }

                rs("&",Nzs()) -= betaR(j)*cscale;
            }

            // Calculate beta and b gradients

            if ( isLinRegul() )
            {
                for ( i = 0 ; i < Nzs() ; i++ )
                {
                    if ( ( betamodel.alphaState()(i) == -2 ) || ( betamodel.alphaState()(i) == +2 ) || ( betamodel.alphaState()(i) == 0 ) )
                    {
                        // If beta is constrained (actively) then the gradient of beta wrt z is 0.
                        // To ensure this we constrain it to 0.  This also ensures a positive
                        // definite Hessian

                        zzmodel.setd(i,0);
                    }

                    else
                    {
                        // Need to (a) make sure this gradient is free (not zero in general)
                        // and (b) correct "z" (rhs of (6) in Dem2) to include sigma.sgn(beta) term

                        zzmodel.setd(i,2);
                        rs("&",i) -= sigma()*(betamodel.alphaState()(i));
                    }
                }
            }

            zzmodel.setyqnd(rs,rscratch);
            zzmodel.inintrain(killSwitch);

            // dbeta/dzj = zzmodel.alphaR()(0,1,Nzs()-1)
            // dbias/dzj = zzmodel.alphaR()(Nzs());

//errstream() << "\n\n";
//errstream() << "cVal = " << cVal << "\n";
//errstream() << "betaR = " << betaR << "\n";
//errstream() << "bias = " << biasR << "\n";
//errstream() << "g = " << g << "\n";
//errstream() << "ZY = " << ZY << "\n";
//errstream() << "xfgrad = " << xfgrad << "\n";
//errstream() << "rs = " << rs << "\n";
//errstream() << "M = " << M() << "\n";
//errstream() << "dbeta/dzj = " << zzmodel.alphaR()(0,1,Nzs()-1) << "\n";
//errstream() << "db/dzj = " << zzmodel.alphaR()(Nzs()) << "\n";
//errstream() << "\n\n";

            // Work out overall z_j gradients

            zstep("&",j).direref(k) = 0;

            for ( i = 0 ; i < Nzs() ; i++ )
            {
                zstep("&",j).direref(k) += 2*betaR(i)*((zzmodel.alphaR())(i)); //dbetadzj(i);
            }

            for ( l = 0 ; l < xact().size() ; l++ )
            {
                inthgr = 0.0;

                for ( m = 0 ; m < xact().size() ; m++ )
                {
                    inthgr += betaR(j)*xfgrad(xact()(m)); //( (zgrad(xact()(m))*zprev(j).direcref(k)) + (xgrad(xact()(m))*( (double) x(xact()(m)).direcref(k) )) );
                }

                for ( i = 0 ; i < Nzs() ; i++ )
                {
                    inthgr += Gp()(i+N(),xact()(l))*((zzmodel.alphaR())(i)); //dbetadzj(i);
                }

                inthgr += (zzmodel.alphaR())(Nzs()); //dbiasdzj;

                zstep("&",j).direref(k) += 2*( g(xact()(l)) - ZY(xact()(l)) )*cVal(xact()(l))*inthgr;
            }
        }

        // step is -lr * grad

        zstep("&",j).scale(-lr);

        // Add scaled penalty to z step - Dem2 is actually ambiguous here, as the
        // penalty given in the paper evaluates as a matrix!  We use the average 
        // similarity gradient to all other templates i != j.

        for ( i = 0 ; i < Nzs() ; i++ )
        {
            if ( i != j )
            {
                zstep("&",j).scaleAdd(-lr*zgrad(i+N())/(((itcnt+1)*(itcnt+1))*(Nzs()-1)),zprev(j));
                zstep("&",j).scaleAdd(-lr*xgrad(i+N())/(((itcnt+1)*(itcnt+1))*(Nzs()-1)),zprev(i)); 
            }
        }

        // Add scaled momentum to z step

        zstep("&",j).scaleAdd(mom,zstepprev("&",j));

        // Project zstep into constrained box

        stepscale = 1;

        for ( k = 0 ; k < zdim ; k++ )
        {
            if ( zprev(j).direcref(k)+stepscale*zstep(j).direcref(k) < zxmin.direcref(k) )
            {
                stepscale = (zxmin.direcref(k)-zprev(j).direcref(k))/zstep(j).direcref(k);
                //zstep("&",j).direref(k) = zxmin.direcref(k)-zprev(j).direcref(k);
            }

            if ( zprev(j).direcref(k)+stepscale*zstep(j).direcref(k) > zxmax.direcref(k) )
            {
                stepscale = (zxmax.direcref(k)-zprev(j).direcref(k))/zstep(j).direcref(k);
                //zstep("&",j).direref(k) = zxmax.direcref(k)-zprev(j).direcref(k);
            }
        }

        zstep("&",j) *= stepscale;

        // Take z step

        znext("&",j)  = zprev(j);
        znext("&",j) += zstep(j);
        zxfer(ztoset,znext(j));
        qswapz(j,ztoset);

        // Work out beta step

        betamodel.setyqnd(n(),rscratch);
        betamodel.inintrain(killSwitch);

        // Update beta and b

        retVector<double> tmpva;

        betaR = (betamodel.alphaR())(zeroint(),1,Nzs()-1,tmpva);
        biasR = (betamodel.alphaR())(Nzs());

        // Re-calculate g

        for ( l = 0 ; l < xact().size() ; l++ )
        {
            g("&",xact()(l)) = biasR;

            for ( i = 0 ; i < Nzs() ; i++ )
            {
                g("&",xact()(l)) += Gp()(i+N(),xact()(l))*betaR(i);
            }
        }

        // Calculate current objective function value
        // and update g

        if ( !(itcnt%RECINTERVAL) )
        {
            printnow = 1;
        }

        isopt = 1;

        if ( j == Nzs()-1 )
        {
            currobj  = sigma() * ( isQuadRegul() ? norm2(betaR) : abs1(betaR) );
            currobj += (sigma()/(betamodel.calcCvalquick(Nzs()))) * ( isQuadRegul() ? biasR*biasR : abs2(biasR) );

            for ( l = 0 ; l < xact().size() ; l++ )
            {
                currobj += cVal(xact()(l))*(g(xact()(l))-ZY(xact()(l)))*(g(xact()(l))-ZY(xact()(l)));
            }

            // Save position if current is best

            if ( ( objBest == -1 ) || ( currobj < objBest ) )
            {
                objBest   = currobj;
                zbest     = znext;
                betaRbest = betaR;
                biasRbest = biasR;
                gbest     = g;
            }

            // Work out optimality, revert if overstep, save zprev if step good

            if ( prevobj != -1 )
            {
                isopt = ( abs2(currobj-prevobj) < tol ) ? 1 : 0;

                if ( !isopt )
                {
                    if ( currobj > prevobj )
                    {
                        // throttle back learning rates

                        lr  *= ovsc;
                        mom *= ovsc;

                        errstream() << "lr = " << lr << " - ";

// Cop-out (it's actually faster)
goto standardmethod;
// This block of code is unreachable

                        // Kill momentum and revert to previous solution

                        for ( i = 0 ; i < Nzs() ; i++ )
                        {
                            zstepprev("&",i) = 0.0;
                            
                            zxfer(ztoset,zprev(j));
                            qswapz(j,ztoset);
                        }

                        // Revert beta and bias

                        betaR = betaRprev;
                        biasR = biasRprev;

                        // Recaculate g

                        g = gprev;
                    }

                    else
                    {
standardmethod:
                        // Acknowledge step

                        zprev     = znext;
                        zstepprev = zstep;

                        betaRprev = betaR;
                        biasRprev = biasR;

                        gprev = g;

                        prevobj = currobj;
                    }
                }
            }

            else
            {
                prevobj = currobj;
                isopt = 0;
            }

            if ( printnow )
            {
                printnow = 0;
                errstream() << currobj;
            }
        }

        else
        {
            isopt = 0;
        }

        if ( isopt )
        {
            if ( objBest < currobj )
            {
                errstream() << " Revert to " << objBest << " ";

                // Current not best, so revert to best solution

                currobj = objBest;
                znext   = zbest;
                betaR   = betaRbest;
                biasR   = biasRbest;
                g       = gbest;

                for ( i = 0 ; i < Nzs() ; i++ )
                {
                    zxfer(ztoset,znext(i));
                    qswapz(i,ztoset);
                }
            }

            // Finalise beta and bias

            setbeta(betaR);
            setb(biasR);

//errstream() << "phantomxyzabc 0: gin = " << g << "\n";
//errstream() << "phantomxyzabc 1: betaR = " << betaR << "\n";
//errstream() << "phantomxyzabc 2: biasR = " << biasR << "\n";
//errstream() << "phantomxyzabc 3: beta = " << beta() << "\n";
//errstream() << "phantomxyzabc 4: b = " << b() << "\n";
            if ( objBest < currobj )
            {
                currobj = objBest;
            }
        }

        // Iteration counters and other termination conditions

        j = ( j+1 < Nzs() ) ? j+1 : 0;

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
            timeout = kbquitdet("ssv optimisation",uservars,varnames,vardescr);
        }
    }

    inbypass = 0;

    res = 0;

    return 0;
}
