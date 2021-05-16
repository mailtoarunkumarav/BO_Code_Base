
//
// Sparse quadratic solver - special case SMO optimiser
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "sQsmo.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 1000


unsigned int solve_SMO(svmvolatile int &killSwitch, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, unsigned int maxitcnt, double maxtraintime, int just_zero_f, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);
unsigned int solve_SMO_fixed_bias(svmvolatile int &killSwitch, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, unsigned int maxitcnt, double maxtraintime, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);

int secondChoice_SMO(int &i1, double &E1, double &E2, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &hp, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);
int takeStep_SMO_f_nonzero(int i2, int tau2, double &e2, int &f_zero, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);
int trial_step_SMO_f_nonzero(int &f_zero_next, int i2, int tau2, double &e2, double &d_alpha2, double &d_b, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gpn, const Vector<double> &lb, const Vector<double> &ub, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);
int actually_take_step_SMO_f_nonzero(int &f_zero, int f_zero_next, int i2, int tau2, double &d_alpha2, double &d_b, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);
int takeStep_SMO_fixed_bias(int i, int tau, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);
//int examineExample_SMO(int i2, int tau2, int &f_zero, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector$
//int takeStep_SMO(int i1, int i2, int tau1, int tau2, double &E1, double &E2, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<do$
//int trial_step_SMO(double &d_J_epart, int i1, int i2, int tau1, int tau2, double &e1, double &e2, double &d_alpha1, double &d_alpha2, double &d_b, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma$
//int actually_take_step_SMO(int i1, int i2, int tau1, int tau2, double &d_alpha1, double &d_alpha2, double &d_b, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const M$

//int solve_quadratic_program_smo(svmvolatile int &killSwitch, optState<double,double> &x, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, 
//const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, int maxitcnt, double maxtraintime)
int fullOptStateSMO::solve(svmvolatile int &killSwitch)
{
    NiceAssert( !GpnRowTwoSigned );
    NiceAssert( x.bN() == 1 );
    NiceAssert( ( x.betaRestrict(0) == 0 ) || ( x.betaRestrict(0) == 3 ) );
    NiceAssert( Gn(zeroint(),0) == 0.0 );
    NiceAssert( maxitcnt >= 0 );
//    NiceAssert( !fixHigherOrderTerms );

    int just_zero_f = 0;

    unsigned int res = 0;
//    double stepscale = stepscalefactor;

    if ( x.betaRestrict(0) == 0 )
    {
	if ( !(x.betaState(0)) )
	{
            x.modBetaCtoF(0,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}

	res = solve_SMO(killSwitch,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,maxitcnt,maxruntime,just_zero_f,NULL,NULL,1.0);
    }

    else
    {
	res = solve_SMO_fixed_bias(killSwitch,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,maxitcnt,maxruntime,NULL,NULL,1.0);
    }

    return res;
}


unsigned int solve_SMO(svmvolatile int &killSwitch, optState<double,double> &probdef, 
                       const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
                       const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
                       const Vector<double> &lb, const Vector<double> &ub, 
                       unsigned int maxitcntint, double maxtraintime, int just_zero_f, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    int f_zero = 0;
    int i;
    int numChanged = 0;
    int examineAll = 1;

    (void) fixHigherOrderTerms;
    (void) htArg;
//    if ( fixHigherOrderTerms )
//    {
//        fixHigherOrderTerms(probdef,htArg);
//    }

    //
    // f_zero = 0 - if f != 0
    //        = 1 - if f == 0
    //

    probdef.refreshGrad(Gp,Gn,Gpn,gp,gn,hp);

    if ( ( probdef.betaGrad(0) >= -(probdef.zerotol()) ) && ( probdef.betaGrad(0) <= (probdef.zerotol()) ) )
    {
	f_zero = 1;

	if ( just_zero_f )
	{
	    return 0;
	}
    }

    //
    // Main algorithmic loop.
    //

    //
    // This is not quite Platt's original implementation, but it is
    // essentially the same.  In the singular case, we move in a
    // direction of linear descent rather then giving up (it can be
    // shown that such a direction will always exist if the error is
    // appropriate and f = 0).
    //

    double maxitcnt = maxitcntint;
    double xmtrtime = maxtraintime;
    double *uservars[] = { &maxitcnt, &xmtrtime, &stepscalefactor, NULL };
    const char *varnames[] = { "itercount", "traintime", "stepscale", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Step scale used for higher-order terms", NULL };

    int res = 1;
    int isopt = 0;
    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    unsigned long long itcnt = 0;
    int timeout = 0;
    int bailout = 0;

    // Obscure note: in c++, if maxitcnt is a double then !maxitcnt is
    // true if maxitcnt == 0, false otherwise.  This is defined in the
    // standard, and the reason the following while statement will work.

    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcnt ) || !maxitcnt ) && !timeout && !bailout )
    {
	numChanged = 0;

        for ( i = 0 ; i < probdef.aN() ; i++ )
	{
	    probdef.refreshGrad(Gp,Gn,Gpn,gp,gn,hp);

	    if ( probdef.alphaRestrict(i) != 3 )
	    {
		if ( probdef.alphaState(i) == 0 )
		{
		    // Point is constrained at zero.  If examineAll then
		    // examine all possible alternative states for this
                    // variable

		    if ( examineAll )
		    {
			if ( probdef.alphaRestrict(i) == 0 )
			{
			    // Unrestricted, so try both positive and negative in random order

			    if ( svm_rand() % 2 )
			    {
				if ( examineExample_SMO(i,-1,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				{
				    numChanged++;
				}

				else
				{
				    numChanged += examineExample_SMO(i,+1,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
				}
			    }

			    else
			    {
				if ( examineExample_SMO(i,+1,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				{
				    numChanged++;
				}

				else
				{
				    numChanged += examineExample_SMO(i,-1,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
				}
			    }
			}

			else if ( probdef.alphaRestrict(i) == 1 )
			{
			    // Restricted positive

			    numChanged += examineExample_SMO(i,+1,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
			}

			else if ( probdef.alphaRestrict(i) == 2 )
			{
			    // Restricted negative

			    numChanged += examineExample_SMO(i,-1,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
			}
                    }
		}

                else if ( probdef.alphaState(i) == +1 )
                {
                    // Variable is free positive.

                    numChanged += examineExample_SMO(i,+1,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
                }

                else if ( probdef.alphaState(i) == -1 )
		{
                    // Variable is free negative.

                    numChanged += examineExample_SMO(i,-1,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
                }

                else if ( probdef.alphaState(i) == +2 )
		{
                    // Variable is constrained at upper bound.

                    if ( examineAll )
                    {
                        numChanged += examineExample_SMO(i,+1,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
                    }
		}

		else if ( probdef.alphaState(i) == -2 )
		{
                    // Variable is constrained at lower bound

                    if ( examineAll )
                    {
                        numChanged += examineExample_SMO(i,-1,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
                    }
                }

                if ( just_zero_f && f_zero )
                {
                    res = 0;
                    isopt = 1;
                }
            }
        }

        itcnt += numChanged-1;

        if ( examineAll )
        {
            if ( numChanged == 0 )
            {
                res = 0;
                isopt = 1;
            }

            examineAll = 0;
        }

        else if ( numChanged == 0 )
        {
            examineAll = 1;
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
            timeout = kbquitdet("SMO (sQsmo) optimisation",uservars,varnames,vardescr);
        }
    }

    return res;
}

unsigned int solve_SMO_fixed_bias(svmvolatile int &killSwitch, optState<double,double> &probdef, 
                                  const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
                                  const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
                                  const Vector<double> &lb, const Vector<double> &ub, 
                                  unsigned int maxitcntint, double maxtraintime, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    int i;
    int numChanged = 0;
    int examineAll = 1;

    (void) fixHigherOrderTerms;
    (void) htArg;
//    if ( fixHigherOrderTerms )
//    {
//        fixHigherOrderTerms(probdef,htArg);
//    }

    //
    // Main algorithmic loop.
    //

    // 
    // The fixed bias version of SMO is essentially trivial.  The active
    // set size is 1, so there is no need for a second choice heuristic
    // and the related complications.  Furthermore, even the singular case
    // becomes trivial to evaluate.
    //

    double maxitcnt = maxitcntint;
    double xmtrtime = maxtraintime;
    double *uservars[] = { &maxitcnt, &xmtrtime, &stepscalefactor, NULL };
    const char *varnames[] = { "itercount", "traintime", "stepscale", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Step scale used for higher-order terms", NULL };

    int res = 1;
    int isopt = 0;
    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    unsigned long long itcnt = 0;
    int timeout = 0;
    int bailout = 0;

    // Obscure note: in c++, if maxitcnt is a double then !maxitcnt is
    // true if maxitcnt == 0, false otherwise.  This is defined in the
    // standard, and the reason the following while statement will work.

    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcnt ) || !maxitcnt ) && !timeout && !bailout )
    {
	numChanged = 0;

        for ( i = 0 ; i < probdef.aN() ; i++ )
	{
	    probdef.refreshGrad(Gp,Gn,Gpn,gp,gn,hp);

	    if ( probdef.alphaRestrict(i) != 3 )
	    {
		if ( probdef.alphaState(i) == 0 )
		{
		    // Point is constrained at zero.  If examineAll then
		    // examine all possible alternative states for this
                    // variable

                    if ( examineAll )
                    {
			if ( probdef.alphaRestrict(i) == 0 )
			{
			    // Unrestricted, so try both positive and negative in random order

			    if ( svm_rand() % 2 )
			    {
				if ( takeStep_SMO_fixed_bias(i,-1,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				{
				    numChanged++;
				}

				else
				{
				    numChanged += takeStep_SMO_fixed_bias(i,+1,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
				}
			    }

			    else
			    {
				if ( takeStep_SMO_fixed_bias(i,+1,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				{
				    numChanged++;
				}

				else
				{
				    numChanged += takeStep_SMO_fixed_bias(i,-1,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
				}
			    }
			}

			else if ( probdef.alphaRestrict(i) == 1 )
			{
			    // Restricted positive

			    numChanged += takeStep_SMO_fixed_bias(i,+1,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
			}

			else
			{
                            NiceAssert( probdef.alphaRestrict(i) == 2 );

			    // Restricted negative

			    numChanged += takeStep_SMO_fixed_bias(i,-1,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
			}
		    }
                }

                else if ( probdef.alphaState(i) == +1 )
                {
                    // Variable is free positive.

		    numChanged += takeStep_SMO_fixed_bias(i,+1,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
		}

                else if ( probdef.alphaState(i) == -1 )
                {
                    // Variable is free negative.

                    numChanged += takeStep_SMO_fixed_bias(i,-1,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
                }

                else if ( probdef.alphaState(i) == +2 )
		{
                    // Variable is constrained at upper bound.

                    if ( examineAll )
                    {
                        numChanged += takeStep_SMO_fixed_bias(i,+1,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
                    }
                }

		else
		{
                    NiceAssert( probdef.alphaState(i) == -2 );

                    // Variable is constrained at lower bound

                    if ( examineAll )
                    {
                        numChanged += takeStep_SMO_fixed_bias(i,-1,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
                    }

                    res = 0;
                    isopt = 1;
                }
            }
        }

        itcnt += numChanged-1;

        if ( examineAll )
        {
            if ( numChanged == 0 )
            {
                res = 0;
                isopt = 1;
            }

            examineAll = 0;
        }

        else if ( numChanged == 0 )
        {
            examineAll = 1;
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
            timeout = kbquitdet("SMO fixed bias (sQsmo) optimisation",uservars,varnames,vardescr);
        }
    }

    return res;
}




int examineExample_SMO(int i2, int tau2, int &f_zero, optState<double,double> &probdef, 
const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
const Vector<double> &lb, const Vector<double> &ub, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    // NB: We make alpha >= 0 for consistency with SMO papers here.  This does
    //     lead to some apparent inconsistencies.

    int firstrunflag;

    int i1start;
    int i1 = 0;
    int tau1 = 0;
    double E1;

    double alph2 = ( tau2 == +1 ) ? (probdef.alpha(i2))                        : -(probdef.alpha(i2));
    double E2    = ( tau2 == +1 ) ? (probdef.posAlphaGrad(E2,i2,Gp,Gpn,gp,hp)) : (probdef.negAlphaGrad(E2,i2,Gp,Gpn,gp,hp));
    double r2    = ( tau2 == +1 ) ? E2                                         : -E2;
    double C2    = ( tau2 == +1 ) ? ub(i2)                                     : -lb(i2);

    if ( ( ( r2 < -probdef.opttol() ) && ( alph2 < C2 ) ) || ( ( r2 > probdef.opttol() ) && ( alph2 > 0.0 ) ) || !f_zero )
    {
        // Second choice heuristic

        if ( !f_zero )
	{
	    // Nonzero f version - take a step with a single free alpha to
	    // minimizef

	    if ( takeStep_SMO_f_nonzero(i2,tau2,E2,f_zero,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
            {
                return 1;
            }
        }

        else
	{
	    // f zero version - need two variables to make a sensible step
            // (note deliberate assignment here)

            if ( ( tau1 = secondChoice_SMO(i1,E1,E2,probdef,Gp,Gpsigma,Gpn,gp,hp,fixHigherOrderTerms,htArg,stepscalefactor) ) )
            {
                if ( takeStep_SMO(i1,i2,tau1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
                {
                    return 1;
                }
	    }

            // Second choice heuristic first fallback

            i1start = ( svm_rand() % (probdef.aN()) );
            firstrunflag = 1;

            for ( i1 = i1start ; ( i1 != i1start ) || firstrunflag ; i1 = (i1+1)%(probdef.aN()) )
	    {
		firstrunflag = 0;

		if ( probdef.alphaRestrict(i1) != 3 )
		{
		    if ( probdef.alphaState(i1) == +1 )
		    {
			E1 = probdef.posAlphaGrad(E1,i1,Gp,Gpn,gp,hp);

			if ( takeStep_SMO(i1,i2,+1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
			{
			    return 1;
			}
		    }

		    else if ( probdef.alphaState(i1) == -1 )
		    {
			E1 = probdef.negAlphaGrad(E1,i1,Gp,Gpn,gp,hp);

			if ( takeStep_SMO(i1,i2,-1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
			{
			    return 1;
			}
		    }
		}
	    }

            // Second choice heuristic second fallback

            i1start = ( svm_rand() % (probdef.aN()) );
            firstrunflag = 1;

            for ( i1 = i1start ; ( i1 != i1start ) || firstrunflag ; i1 = (i1+1)%(probdef.aN()) )
	    {
		firstrunflag = 0;

		if ( probdef.alphaRestrict(i1) != 3 )
		{
		    if ( probdef.alphaState(i1) == 0 )
		    {
			if ( probdef.alphaRestrict(i1) == 0 )
			{
			    if ( svm_rand() % 2 )
			    {
				E1 = probdef.negAlphaGrad(E1,i1,Gp,Gpn,gp,hp);

				if ( takeStep_SMO(i1,i2,-1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				{
				    return 1;
				}

				else
				{
				    E1 = probdef.posAlphaGrad(E1,i1,Gp,Gpn,gp,hp);

				    if ( takeStep_SMO(i1,i2,+1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				    {
					return 1;
				    }
				}
			    }

			    else
			    {
				E1 = probdef.posAlphaGrad(E1,i1,Gp,Gpn,gp,hp);

				if ( takeStep_SMO(i1,i2,+1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				{
				    return 1;
				}

				else
				{
				    E1 = probdef.negAlphaGrad(E1,i1,Gp,Gpn,gp,hp);

				    if ( takeStep_SMO(i1,i2,-1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				    {
					return 1;
				    }
				}
			    }
			}

			else if ( probdef.alphaRestrict(i1) == 1 )
			{
			    E1 = probdef.posAlphaGrad(E1,i1,Gp,Gpn,gp,hp);

			    if ( takeStep_SMO(i1,i2,+1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
			    {
				return 1;
			    }
			}

			else
			{
                            NiceAssert( probdef.alphaRestrict(i1) == 2 );

			    E1 = probdef.negAlphaGrad(E1,i1,Gp,Gpn,gp,hp);

			    if ( takeStep_SMO(i1,i2,-1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
			    {
				return 1;
			    }
			}
		    }

		    else if ( probdef.alphaState(i1) == +2 )
		    {
			E1 = probdef.posAlphaGrad(E1,i1,Gp,Gpn,gp,hp);

			if ( takeStep_SMO(i1,i2,+1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
			{
			    return 1;
			}
		    }

		    else if ( probdef.alphaState(i1) == -2 )
		    {
			E1 = probdef.negAlphaGrad(E1,i1,Gp,Gpn,gp,hp);

			if ( takeStep_SMO(i1,i2,-1,tau2,E1,E2,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
			{
			    return 1;
			}
		    }
		}
            }
        }
    }

    return 0;
}

int secondChoice_SMO(int &i1, double &E1, double &E2, optState<double,double> &probdef, 
const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gpn, 
const Vector<double> &gp, const Vector<double> &hp, 
double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    (void) Gpsigma;
    (void) fixHigherOrderTerms;
    (void) htArg;
    (void) stepscalefactor;

    int firstrunflag = 1;
    int istart;
    int i;
    int firstTry = 1;
    double E;
    int tau1 = 0;

    i1 = 0;
    E1 = 0.0;

    istart = ( svm_rand() % (probdef.aN()) );

    for ( i = istart ; ( i != istart ) || firstrunflag ; i = (i+1)%(probdef.aN()) )
    {
	firstrunflag = 0;

	if ( probdef.alphaRestrict(i) != 3 )
	{
	    if ( probdef.alphaState(i) == +1 )
	    {
		E = probdef.posAlphaGrad(E,i,Gp,Gpn,gp,hp);

		if ( firstTry || ( ( E2 >= 0.0 ) && ( E < E1 ) ) || ( ( E2 <= 0.0 ) && ( E > E1 ) ) )
		{
		    firstTry = 0;

		    i1   = i;
		    E1   = E;
		    tau1 = +1;
		}
	    }

	    else if ( probdef.alphaState(i) == -1 )
	    {
		E = probdef.negAlphaGrad(E,i,Gp,Gpn,gp,hp);

		if ( firstTry || ( ( E2 >= 0.0 ) && ( E < E1 ) ) || ( ( E2 <= 0.0 ) && ( E > E1 ) ) )
		{
		    firstTry = 0;

		    i1   = i;
		    E1   = E;
		    tau1 = -1;
		}
            }
        }
    }

    return tau1;
}

int takeStep_SMO(int i1, int i2, int tau1, int tau2, double &e1, double &e2, optState<double,double> &probdef, 
const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
const Vector<double> &lb, const Vector<double> &ub, 
double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    if ( i1 == i2 )
    {
        return 0;
    }

    double d_alpha1 = 0.0;
    double d_alpha2 = 0.0;
    double d_b = 0.0;
    double dummy = 0.0;

    if ( !trial_step_SMO(dummy,i1,i2,tau1,tau2,e1,e2,d_alpha1,d_alpha2,d_b,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
    {
        return 0;
    }

    return actually_take_step_SMO(i1,i2,tau1,tau2,d_alpha1,d_alpha2,d_b,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
}

int trial_step_SMO(double &d_J_epart, int i1, int i2, int tau1, int tau2, double &e1, double &e2, double &d_alpha1, double &d_alpha2, double &d_b, optState<double,double> &probdef, 
const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
const Vector<double> &lb, const Vector<double> &ub, 
double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    (void) Gn;
    (void) Gpn;
    (void) gp;
    (void) gn;
    (void) hp;
    (void) fixHigherOrderTerms;
    (void) htArg;
    (void) stepscalefactor;

    //
    // It may be observed that the SMO approach is simply a special case of
    // active set, where only 2 variables are free.  To this end, it will
    // be noted that:
    //
    // [ d_b      ]          [ 0  a1  a2  ]   [  f  ]
    // [ d_alpha1 ] = - inv( [ a1 K11 K12 ] ) [ e_1 ]
    // [ d_alpha2 ]          [ a2 K12 K22 ]   [ e_2 ]
    //
    //                   1      [  K11.K22 - K12.K12 ]
    //              = ------- ( [   a2.K12 - a1.K22  ] f
    //                negdetH   [   a1.K12 - a2.K11  ]
    //
    //                  [ a2.K12 - a1.K22 ]       [ a1.K12 - a2.K11 ]
    //                + [    -a2.a2       ] e_1 + [    +a1.a2       ] e_2 )
    //                  [    +a1.a2       ]       [    -a1.a1       ]
    //
    // where: negdetH = a1.a1.K22 + a2.a2.K11 - 2.a1.a2.K12
    //
    // [ d_f  ]   [ -d_f  ]
    // [ d_e1 ] = [ -d_e1 ]
    // [ d_e2 ]   [ -d_e2 ]
    //
    // Of course, this is only true if the 2 hessian is non-singular - ie.
    // negdetH != 0.  If this does not hold, no probdef - just find a
    // direction of linear non-ascent wrt alpha_1 and alpha_2.  Of course,
    // f is zero for this section of the code, which simplifies things
    // somewhat.  Specifically, as f == 0:
    //
    // d_alpha1 = ( -a2.a2.e1 + a1.a2.e2 ) / negdetH
    // d_alpha2 = (  a1.a2.e1 - a1.a1.e2 ) / negdetH
    //          = -(a1/a2).d_alpha1
    //
    // NB: we are ignoring the SMO method here, as it involves evaluating
    //     the entire goddam objective, which is tres-costly one would
    //     think!
    //
    // Specifically, in the singular case:
    //
    // [ d_b      ] = - theta inv( [ 0  a1  ] ) [ a2  ]
    // [ d_alpha1 ]                [ a1 K11 ]   [ K12 ]
    //
    //              = - theta [ -K11/(a1*a1) 1/a1 ] [ a2  ]
    //                        [ 1/a1         0    ] [ K12 ]
    //
    //              = theta [ K11/(a1.a1) -1/a1 ] [ a2  ]
    //                      [ -1/a1        0    ] [ K12 ]
    //
    //              = theta [ (a2/a1).(K11/a1) - K12/a1 ]
    //                      [    -a2/a1                 ]
    //
    // d_alpha1 = -theta.(a2/a1)
    // d_alpha2 = -(a1/a2)*d_alpha1 = theta
    //
    // [ d_f  ]   [ 0 ]
    // [ d_e1 ] = [ 0 ]
    // [ d_e2 ]   [ 0 ]
    //
    // where theta is chosen to ensure linear non-increase (e1.d_alpha1 +
    // e2.d_alpha2 < 0) and also to make sure that a bound is hit.  Now:
    //
    // e1.d_alpha1 + e2.d_alpha2 = d_alpha2.(e2 - (a2/a1).e1)
    //
    // Specifically:
    //
    // theta = -1.1 * |h1| : if e2 >= (a2/a1).e1 and alpha1 is positive
    // theta = +1.1 * |h1| : if e2 <  (a2/a1).e1 and alpha1 is positive
    // theta = -1.1 * |v1| : if e2 >= (a2/a1).e1 and alpha1 is negative
    // theta = +1.1 * |v1| : if e2 <  (a2/a1).e1 and alpha1 is negative
    //

    //double negdetH = (Gpn(i2,0)*Gpn(i2,0)*Gp(i1,i1))+(Gpn(i1,0)*Gpn(i1,0)*Gp(i2,i2))-(2.0*Gpn(i1,0)*Gpn(i2,0)*Gp(i1,i2));
    double negdetH = Gpsigma(i1,i2);

    if ( ( negdetH > (probdef.zerotol()) ) || ( negdetH < -(probdef.zerotol()) ) )
    {
        // Nonsingular case

        //d_b = ((((Gpn(i2,0)*Gp(i1,i2))-(Gpn(i1,0)*Gp(i2,i2)))*e1)+(((Gpn(i1,0)*Gp(i1,i2))-(Gpn(i2,0)*Gp(i1,i1)))*e2))/negdetH;
        //
        //d_alpha1 = (-(Gpn(i2,0)*Gpn(i2,0)*e1)+(Gpn(i1,0)*Gpn(i2,0)*e2))/negdetH;
        //d_alpha2 = ( (Gpn(i1,0)*Gpn(i2,0)*e1)-(Gpn(i1,0)*Gpn(i1,0)*e2))/negdetH;
        d_b = (((Gp(i1,i2)-Gp(i2,i2))*e1)+((Gp(i1,i2)-Gp(i1,i1))*e2))/negdetH;

        d_alpha1 = (-e1+e2)/negdetH;
        d_alpha2 = -d_alpha1;

        /*
           Note: save some time and trust me on this - I've checked it
                 numerous times.  If you must check, you'll notice that I
                 seem to have "missed" a factor of 1/2 here.  It is factored
                 in below, so don't panic.
        */

//FIXME: this is incorrect when Gpn is not 1
        d_J_epart = -((e1-e2)*(e1-e2))/negdetH;
    }

    else
    {
        // Singular case
        
        //if ( Gpn(i1,0)*Gpn(i1,0) <= (probdef.zerotol()) )
        //{
        //    if ( Gpn(i2,0)*Gpn(i2,0) <= (probdef.zerotol()) )
        //    {
        //           Cannot define any sort of inverse in this case.
        //           Effectively, this is just the fixed bias case.
        //
        //        d_b = 0.0;
        //
        //           First try... see if the K matrix is invertible.
        //
        //        negdetH = (Gp(i1,i1)*Gp(i2,i2))-(Gp(i1,i2)*Gp(i1,i2));
        //
        //        if ( negdetH > (probdef.zerotol()) )
        //        {
        //            d_alpha1 = -( (Gp(i2,i2)*e1)-(Gp(i1,i2)*e2))/negdetH;
        //            d_alpha2 = -(-(Gp(i1,i2)*e1)+(Gp(i1,i1)*e2))/negdetH;
        //
	//	    d_J_epart = -((Gp(i2,i2)*e1*e1)+(Gp(i1,i1)*e2*e2)-(2*Gp(i1,i2)*e1*e2))/(2*negdetH);
        //        }
        //
        //        else
        //        {
        //            if ( Gp(i1,i1) > (probdef.zerotol()) )
        //            {
        //                //
        //                // [ d_alpha1 ] = - theta inv( [ K11 ] ) [ K12 ]
        //                //              = - theta [ 1/K11 ] [ K12 ]
        //                //              = - theta K12/K11
        //                //
        //                // d_alpha1 = -theta.(K12/K11)
        //                // d_alpha2 = theta
        //                //
        //                // theta is chosen to ensure linear non-increase 
        //                // (e1.d_alpha1 + e2.d_alpha2 < 0) and also to make sure 
        //                // that a bound is hit.  Now:
        //                //
        //                // e1.d_alpha1 + e2.d_alpha2 = d_alpha2.(e2 - (K12/K11).e1)
        //                //
        //                // Specifically:
        //                //
        //                // theta = -1.1 * |h2| : if e2 >= (K12/K11).e1 & alpha1 > 0
        //                // theta = +1.1 * |h2| : if e2 <  (K12/K11).e1 & alpha1 > 0
        //                // theta = -1.1 * |v2| : if e2 >= (K12/K11).e1 & alpha1 < 0
        //                // theta = +1.1 * |v2| : if e2 <  (K12/K11).e1 & alpha1 < 0
        //                //
        //
        //                double theta;
        //
        //                if ( tau2 == +1 )
        //                {
        //                    if ( e2 >= (Gp(i1,i2)/Gp(i1,i1))*e1 )
        //                    {
        //                        theta = -1.1 * ub(i2);
        //                    }
        //
        //                    else
        //                    {
        //                        theta = 1.1 * ub(i2);
        //                    }
        //                }
        //
        //                else
        //                {
        //                    if ( e2 >= (Gp(i1,i2)/Gp(i1,i1))*e1 )
        //                    {
        //                        theta = -1.1 * -lb(i2);
        //                    }
        //
        //                    else
        //                    {
        //                        theta = 1.1 * -lb(i2);
        //                    }
        //                }
        //
        //                d_alpha1 = -theta*Gp(i1,i2)/Gp(i1,i1);
        //                d_alpha2 = theta;
        //
	//		d_J_epart = (d_alpha1*e1) + (d_alpha2*e2);
        //            }
        //            
        //            else if ( Gp(i2,i2) > (probdef.zerotol()) )
        //            {
        //                return trial_step_SMO(d_J_epart,i2,i1,tau2,tau1,e2,e1,d_alpha2,d_alpha1,d_b,probdef,Gp,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
        //            }
        //            
        //            else
        //            {
        //                // Extremely! unlikely...
        //                
        //                return 0;
        //            }
        //        }
        //    }
        //
        //    else
        //    {
        //        // OK, can get an ok step if we just reverse things a bit.
        //        
        //        return trial_step_SMO(d_J_epart,i2,i1,tau2,tau1,e2,e1,d_alpha2,d_alpha1,d_b,probdef,Gp,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
        //    }
        //}
        //
        //else
        {
            double theta;

            //d_b = ((Gpn(i2,0)/Gpn(i1,0))*(Gp(i1,i1)/Gpn(i1,0)))-(Gp(i1,i2)/Gpn(i1,0));
            //
            //d_alpha1 = -Gpn(i2,0)/Gpn(i1,0);
            //d_alpha2 = +1.0;
            d_b = Gp(i1,i1)-Gp(i1,i2);

            d_alpha1 = -1.0;
            d_alpha2 = +1.0;

            if ( tau2 == +1 )
            {
                //if ( e2 >= (Gpn(i2,0)/Gpn(i1,0))*e1 )
                if ( e2 >= e1 )
                {
                    theta = -1.1 * ub(i2);
                }

                else
                {
                    theta = 1.1 * ub(i2);
                }
            }

            else
            {
                //if ( e2 >= (Gpn(i2,0)/Gpn(i1,0))*e1 )
                if ( e2 >= e1 )
                {
                    theta = -1.1 * -lb(i2);
                }

                else
                {
                    theta = 1.1 * -lb(i2);
                }
            }

            d_b *= theta;

            d_alpha1 *= theta;
            d_alpha2 *= theta;

//FIXME: this is incorrect when b_scales is not 1
            d_J_epart = (d_alpha1*e1) + (d_alpha2*e2);
        }
    }

    // Now let's scale the step

    double step_scale = 1.0;
    double ceta;

    if (    ( d_alpha1 > -(probdef.zerotol())/10 ) && ( d_alpha1 < (probdef.zerotol())/10 )
         && ( d_alpha2 > -(probdef.zerotol())/10 ) && ( d_alpha2 < (probdef.zerotol())/10 ) )
    {
        return 0;
    }

    if ( d_alpha1 < 0.0 )
    {
        if ( tau1 > 0 )
        {
            ceta = -(probdef.alpha(i1)) / d_alpha1;

            // crazy voodoo here, nan
            if ( ( ceta < step_scale ) && !( ceta >= step_scale ) )
            {
                step_scale = ceta;
            }
        }

        else
        {
            ceta = ( lb(i1) - (probdef.alpha(i1)) ) / d_alpha1;

            // crazy voodoo here, nan
            if ( ( ceta < step_scale ) && !( ceta >= step_scale ) )
            {
                step_scale = ceta;
            }
        }
    }
    
    else if ( d_alpha1 > 0.0 )
    {
        if ( tau1 > 0 )
        {
            ceta = ( ub(i1) - (probdef.alpha(i1)) ) / d_alpha1;

            // crazy voodoo here, nan
            if ( ( ceta < step_scale ) && !( ceta >= step_scale ) )
            {
                step_scale = ceta;
            }
        }

        else
        {
            ceta = -(probdef.alpha(i1)) / d_alpha1;

            // crazy voodoo here, nan
            if ( ( ceta < step_scale ) && !( ceta >= step_scale ) )
            {
                step_scale = ceta;
            }
        }
    }

    if ( d_alpha2 > 0.0 )
    {
        if ( tau2 > 0 )
        {
            ceta = ( ub(i2) - (probdef.alpha(i2)) ) / d_alpha2;

            // crazy voodoo here, nan
            if ( ( ceta < step_scale ) && !( ceta >= step_scale ) )
            {
                step_scale = ceta;
            }
        }

        else
        {
            ceta = -(probdef.alpha(i2)) / d_alpha2;

            // crazy voodoo here, nan
            if ( ( ceta < step_scale ) && !( ceta >= step_scale ) )
            {
                step_scale = ceta;
            }
        }
    }

    else if ( d_alpha2 < 0.0 )
    {
        if ( tau2 > 0 )
        {
            ceta = -(probdef.alpha(i2)) / d_alpha2;

            // crazy voodoo here, nan
            if ( ( ceta < step_scale ) && !( ceta >= step_scale ) )
            {
                step_scale = ceta;
            }
        }

        else
        {
            ceta = ( lb(i2) - (probdef.alpha(i2)) ) / d_alpha2;

            // crazy voodoo here, nan
            if ( ( ceta < step_scale ) && !( ceta >= step_scale ) )
            {
                step_scale = ceta;
            }
        }
    }

    d_b *= step_scale;

    d_alpha1 *= step_scale;
    d_alpha2 *= step_scale;

    /*
       Return zero if step is insignificant wrt the current value.
       This is from Platt.
    */

    if ( ( abs2(d_alpha1) < (probdef.opttol())*(abs2((2.0*(probdef.alpha(i1)))+d_alpha1)+(probdef.opttol())) ) &&
         ( abs2(d_alpha2) < (probdef.opttol())*(abs2((2.0*(probdef.alpha(i2)))+d_alpha2)+(probdef.opttol())) )    )
    {
        return 0;
    }

    return 1;
}

int actually_take_step_SMO(int i1, int i2, int tau1, int tau2, double &d_alpha1, double &d_alpha2, double &d_b, optState<double,double> &probdef, 
const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
const Vector<double> &lb, const Vector<double> &ub, 
double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    (void) Gpsigma;
    (void) stepscalefactor;

    int iP1 = 0;
    int iP2 = 0;

    // First we may need to unconstrain alpha1 and/or alpha2

    if      ( probdef.alphaState(i1) == -2 ) { iP1 = probdef.findInAlphaLB(i1); }
    else if ( probdef.alphaState(i1) == -1 ) { iP1 = probdef.findInAlphaF(i1);  }
    else if ( probdef.alphaState(i1) == 0  ) { iP1 = probdef.findInAlphaZ(i1);  }
    else if ( probdef.alphaState(i1) == +1 ) { iP1 = probdef.findInAlphaF(i1);  }
    else if ( probdef.alphaState(i1) == +2 ) { iP1 = probdef.findInAlphaUB(i1); }

    if ( probdef.alphaState(i1) == +2 )
    {
        NiceAssert( tau1 == +1 );

	iP1 = probdef.modAlphaUBtoUF(iP1,Gp,Gp,Gn,Gpn,gp,gn,hp);
    }

    else if ( probdef.alphaState(i1) == -2 )
    {
        NiceAssert( tau1 == -1 );

	iP1 = probdef.modAlphaLBtoLF(iP1,Gp,Gp,Gn,Gpn,gp,gn,hp);
    }

    else if ( probdef.alphaState(i1) == 0 )
    {
	if ( tau1 == +1 )
	{
	    iP1 = probdef.modAlphaZtoUF(iP1,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}

	else
	{
           iP1 = probdef.modAlphaZtoLF(iP1,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}
    }

    if      ( probdef.alphaState(i2) == -2 ) { iP2 = probdef.findInAlphaLB(i2); }
    else if ( probdef.alphaState(i2) == -1 ) { iP2 = probdef.findInAlphaF(i2);  }
    else if ( probdef.alphaState(i2) == 0  ) { iP2 = probdef.findInAlphaZ(i2);  }
    else if ( probdef.alphaState(i2) == +1 ) { iP2 = probdef.findInAlphaF(i2);  }
    else if ( probdef.alphaState(i2) == +2 ) { iP2 = probdef.findInAlphaUB(i2); }

    if ( probdef.alphaState(i2) == +2 )
    {
        NiceAssert( tau2 == +1 );

	iP2 = probdef.modAlphaUBtoUF(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
    }

    else if ( probdef.alphaState(i2) == -2 )
    {
        NiceAssert( tau2 == -1 );

	iP2 = probdef.modAlphaLBtoLF(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
    }

    else if ( probdef.alphaState(i2) == 0 )
    {
	if ( tau2 == +1 )
	{
	    iP2 = probdef.modAlphaZtoUF(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}

	else
	{
            iP2 = probdef.modAlphaZtoLF(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}
    }

    // Tricky part: want to make iP1 > iP2.  This is so that removing
    // iP1 from the free set will not mess up the iP2 index.

    if ( iP1 < iP2 )
    {
	int taux = tau1;
	int iPx  = iP1;
	int ix   = i1;
	double d_alphax = d_alpha1;

	tau1 = tau2;
	iP1  = iP2;
	i1   = i2;
	d_alpha1 = d_alpha2;

	tau2 = taux;
	iP2  = iPx;
	i2   = ix;
        d_alpha2 = d_alphax;
    }

    probdef.alphaStep(i1,d_alpha1,Gp,Gn,Gpn,gp,gn,hp);
    probdef.alphaStep(i2,d_alpha2,Gp,Gn,Gpn,gp,gn,hp);
    probdef.betaStep (0,d_b,Gp,Gn,Gpn,gp,gn,hp);

    (void) fixHigherOrderTerms;
    (void) htArg;
//    if ( fixHigherOrderTerms )
//    {
//        fixHigherOrderTerms(probdef,htArg);
//    }

    // Be careful with constraints.  If alpha wanders too close to the boundary, reel it in and constrain it.

    if ( ( probdef.alpha(i1) <= (probdef.zerotol()) ) && ( probdef.alpha(i1) >= -(probdef.zerotol()) ) )
    {
	if ( probdef.alphaState(i1) == -1 )
	{
	    probdef.modAlphaLFtoZ(iP1,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}

	else
	{
	    probdef.modAlphaUFtoZ(iP1,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}
    }

    else if ( ( probdef.alpha(i1) <= lb(i1)+(probdef.zerotol()) ) && ( probdef.alphaState(i1) == -1 ) )
    {
	probdef.modAlphaLFtoLB(iP1,Gp,Gp,Gn,Gpn,gp,gn,hp,lb);
    }

    else if ( ( probdef.alpha(i1) >= ub(i1)-(probdef.zerotol()) ) && ( probdef.alphaState(i1) == +1 ) )
    {
	probdef.modAlphaUFtoUB(iP1,Gp,Gp,Gn,Gpn,gp,gn,hp,ub);
    }

    if ( ( probdef.alpha(i2) <= (probdef.zerotol()) ) && ( probdef.alpha(i2) >= -(probdef.zerotol()) ) )
    {
	if ( probdef.alphaState(i2) == -1 )
	{
	    probdef.modAlphaLFtoZ(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}

	else
	{
	    probdef.modAlphaUFtoZ(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}
    }

    else if ( ( probdef.alpha(i2) <= lb(i2)+(probdef.zerotol()) ) && ( probdef.alphaState(i2) == -1 ) )
    {
	probdef.modAlphaLFtoLB(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp,lb);
    }

    else if ( ( probdef.alpha(i2) >= ub(i2)-(probdef.zerotol()) ) && ( probdef.alphaState(i2) == +1 ) )
    {
	probdef.modAlphaUFtoUB(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp,ub);
    }

    return 1;
}

int takeStep_SMO_f_nonzero(int i2, int tau2, double &e2, int &f_zero, optState<double,double> &probdef, 
const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
const Vector<double> &lb, const Vector<double> &ub, 
double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    double d_alpha2 = 0.0;
    double d_b = 0.0;
    int f_zero_next = 1;

    if ( !trial_step_SMO_f_nonzero(f_zero_next,i2,tau2,e2,d_alpha2,d_b,probdef,Gp,Gpsigma,Gpn,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
    {
        return 0;
    }

    return actually_take_step_SMO_f_nonzero(f_zero,f_zero_next,i2,tau2,d_alpha2,d_b,probdef,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
}

int trial_step_SMO_f_nonzero(int &f_zero_next, int i2, int tau2, double &e2, double &d_alpha2, double &d_b, optState<double,double> &probdef, 
const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gpn, 
const Vector<double> &lb, const Vector<double> &ub, 
double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    (void) Gpn;
    (void) Gpsigma;
    (void) fixHigherOrderTerms;
    (void) htArg;
    (void) stepscalefactor;

    double f = probdef.betaGrad(0);

    //
    // In this case we use an active set size of 1.
    //
    // [ d_b      ] = - inv( [ 0  a2  ] ) [  f  ]
    // [ d_alpha1 ]          [ a2 K22 ]   [ e_2 ]
    //
    //              = - (1/-(a2.a2)) [ K11 -a2 ] [  f  ]
    //                               [ -a2  0  ] [ e_2 ]
    //
    //              = [ K22/(a2.a2) -1/a2 ] [  f  ]
    //                [ -1/a2        0    ] [ e_2 ]
    //
    //              = [ (K22*f)/(a2/a2) - e_2/a2 ]
    //                [ -f/a2                    ]
    //
    // [ d_f  ] = [  -f  ]
    // [ d_e2 ]   [ -e_2 ]
    //
    // Note that here the Hessian is *guaranteed* to be non-singular.
    //

    //
    // NB: if a2 is zero, as our aim here is *only* to make f zero, then return.
    //
    
    //if ( Gpn(i2,0)*Gpn(i2,0) < (probdef.zerotol()) )
    //{
    //    return 0;
    //}

    //d_b      = ((Gp(i2,i2)*f)/(Gpn(i2,0)*Gpn(i2,0)))-(e2/Gpn(i2,0));
    //d_alpha2 = -f/Gpn(i2,0);
    d_b      = (Gp(i2,i2)*f)-e2;
    d_alpha2 = -f;

    /*
       Now let's scale the step
    */

    double step_scale = 1.0;
    double ceta;

    if ( ( d_alpha2 > -(probdef.zerotol())/10 ) && ( d_alpha2 < (probdef.zerotol())/10 ) )
    {
        return 0;
    }

    if ( d_alpha2 > 0.0 )
    {
        if ( tau2 > 0 )
        {
            ceta = ( ub(i2) - (probdef.alpha(i2)) ) / d_alpha2;

            if ( ceta < step_scale )
            {
                step_scale = ceta;
                f_zero_next = 0;
            }
        }

        else
        {
            ceta = -(probdef.alpha(i2)) / d_alpha2;

            if ( ceta < step_scale )
            {
                step_scale = ceta;
                f_zero_next = 0;
            }
        }
    }

    else
    {
        if ( tau2 > 0 )
        {
            ceta = -(probdef.alpha(i2)) / d_alpha2;

            if ( ceta < step_scale )
            {
                step_scale = ceta;
                f_zero_next = 0;
            }
        }

        else
        {
            ceta = ( lb(i2) - (probdef.alpha(i2)) ) / d_alpha2;

            if ( ceta < step_scale )
            {
                step_scale = ceta;
                f_zero_next = 0;
            }
        }
    }

    d_b      *= step_scale;
    d_alpha2 *= step_scale;

    // Return zero if step is insignificant wrt the current value. This is from Platt.

    if ( abs2(d_alpha2) < (probdef.opttol())*(abs2((2.0*(probdef.alpha(i2)))+d_alpha2)+(probdef.opttol())) )
    {
        return 0;
    }

    return 1;
}

int actually_take_step_SMO_f_nonzero(int &f_zero, int f_zero_next, int i2, int tau2, double &d_alpha2, double &d_b, optState<double,double> &probdef, 
const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
const Vector<double> &lb, const Vector<double> &ub, 
double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    (void) Gpsigma;
    (void) stepscalefactor;

    int iP2 = 0;

    f_zero = f_zero_next;

    // First we may need to unconstrain alpha1 and/or alpha2

    if      ( probdef.alphaState(i2) == -2 ) { iP2 = probdef.findInAlphaLB(i2); }
    else if ( probdef.alphaState(i2) == -1 ) { iP2 = probdef.findInAlphaF(i2);  }
    else if ( probdef.alphaState(i2) == 0  ) { iP2 = probdef.findInAlphaZ(i2);  }
    else if ( probdef.alphaState(i2) == +1 ) { iP2 = probdef.findInAlphaF(i2);  }
    else if ( probdef.alphaState(i2) == +2 ) { iP2 = probdef.findInAlphaUB(i2); }

    if ( probdef.alphaState(i2) == +2 )
    {
        NiceAssert( tau2 == +1 );

	iP2 = probdef.modAlphaUBtoUF(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
    }

    else if ( probdef.alphaState(i2) == -2 )
    {
        NiceAssert( tau2 == -1 );

	iP2 = probdef.modAlphaLBtoLF(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
    }

    else if ( probdef.alphaState(i2) == 0 )
    {
	if ( tau2 == +1 )
	{
	    iP2 = probdef.modAlphaZtoUF(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}

	else
	{
            iP2 = probdef.modAlphaZtoLF(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}
    }

    probdef.alphaStep(i2,d_alpha2,Gp,Gn,Gpn,gp,gn,hp);
    probdef.betaStep (0,d_b,Gp,Gn,Gpn,gp,gn,hp);

    (void) fixHigherOrderTerms;
    (void) htArg;
//    if ( fixHigherOrderTerms )
//    {
//        fixHigherOrderTerms(probdef,htArg);
//    }

    // Be careful with constraints.  If alpha wanders too close to the boundary, reel it in and constrain it.

    if ( ( probdef.alpha(i2) <= (probdef.zerotol()) ) && ( probdef.alpha(i2) >= -(probdef.zerotol()) ) )
    {
	if ( probdef.alphaState(i2) == -1 )
	{
	    probdef.modAlphaLFtoZ(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}

	else
	{
	    probdef.modAlphaUFtoZ(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}
    }

    else if ( ( probdef.alpha(i2) <= lb(i2)+(probdef.zerotol()) ) && ( probdef.alphaState(i2) == -1 ) )
    {
	probdef.modAlphaLFtoLB(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp,lb);
    }

    else if ( ( probdef.alpha(i2) >= ub(i2)-(probdef.zerotol()) ) && ( probdef.alphaState(i2) == +1 ) )
    {
	probdef.modAlphaUFtoUB(iP2,Gp,Gp,Gn,Gpn,gp,gn,hp,ub);
    }

    return 1;
}

int takeStep_SMO_fixed_bias(int i, int tau, optState<double,double> &probdef, 
const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
const Vector<double> &lb, const Vector<double> &ub, 
double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    (void) Gpsigma;
    (void) stepscalefactor;

    int iP = 0;

    double alpha = probdef.alpha(i);
    double e     = ( tau == +1 ) ? (probdef.posAlphaGrad(e,i,Gp,Gpn,gp,hp)) : (probdef.negAlphaGrad(e,i,Gp,Gpn,gp,hp));
    double vbar  = ( tau == +1 ) ? 0.0                                      : lb(i);
    double hbar  = ( tau == +1 ) ? ub(i)                                    : 0.0;

    if ( ( ( e < -probdef.opttol() ) && ( alpha < hbar ) ) || ( ( e > probdef.opttol() ) && ( alpha > vbar ) ) )
    {
        double d_alpha;
        double d_e;

        if ( ( Gp(i,i) > (probdef.zerotol()) ) || ( Gp(i,i) < -(probdef.zerotol()) ) )
        {
            // Nonsingular case.  This is just a newton step.

            d_alpha = -e/Gp(i,i);
            d_e     = -e;
        }

        else
        {
            // Singular case.  Choose linear descent and head to boundary.

            if ( e < 0.0 )
            {
                d_alpha = (hbar-vbar);
            }

            else
            {
                d_alpha = -(hbar-vbar);
            }

            d_e = 0.0;
        }

        // Now let's scale the step

        double step_scale = 1.0;
        double ceta;

        if ( ( d_alpha > -(probdef.zerotol())/10 ) && ( d_alpha < (probdef.zerotol())/10 ) )
        {
            d_alpha = 0.0;
        }

        if ( d_alpha < 0.0 )
        {
            ceta = ( vbar - alpha ) / d_alpha;

            if ( ceta < step_scale )
            {
                step_scale = ceta;
            }
        }

        else if ( d_alpha > 0.0 )
        {
            ceta = ( hbar - alpha ) / d_alpha;

            if ( ceta < step_scale )
            {
                step_scale = ceta;
            }
        }

        d_alpha *= step_scale;
        d_e *= step_scale;

        // Return zero if step too small.  This is from Platt, approximately.

        if ( abs2(d_alpha) < (probdef.opttol())*(abs2((2.0*alpha)+d_alpha)+(probdef.opttol())) )
        {
            return 0;
        }

        // Unconstrain if necessary

	if      ( probdef.alphaState(i) == -2 ) { iP = probdef.findInAlphaLB(i); }
	else if ( probdef.alphaState(i) == -1 ) { iP = probdef.findInAlphaF(i);  }
	else if ( probdef.alphaState(i) == 0  ) { iP = probdef.findInAlphaZ(i);  }
	else if ( probdef.alphaState(i) == +1 ) { iP = probdef.findInAlphaF(i);  }
	else if ( probdef.alphaState(i) == +2 ) { iP = probdef.findInAlphaUB(i); }

	if ( probdef.alphaState(i) == +2 )
	{
            NiceAssert( tau == +1 );

	    iP = probdef.modAlphaUBtoUF(iP,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}

	else if ( probdef.alphaState(i) == -2 )
	{
            NiceAssert( tau == -1 );

	    iP = probdef.modAlphaLBtoLF(iP,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}

	else if ( probdef.alphaState(i) == 0 )
	{
	    if ( tau == +1 )
	    {
		iP = probdef.modAlphaZtoUF(iP,Gp,Gp,Gn,Gpn,gp,gn,hp);
	    }

	    else
	    {
		iP = probdef.modAlphaZtoLF(iP,Gp,Gp,Gn,Gpn,gp,gn,hp);
	    }
        }

        // Take the step;

	probdef.alphaStep(i,d_alpha,Gp,Gn,Gpn,gp,gn,hp);

    (void) fixHigherOrderTerms;
    (void) htArg;
//        if ( fixHigherOrderTerms )
//        {
//            fixHigherOrderTerms(probdef,htArg);
//        }

        // Be careful with constraints.  If alpha wanders too close to the boundary, real it in and constrain it.

	if ( ( probdef.alpha(i) <= (probdef.zerotol()) ) && ( probdef.alpha(i) >= -(probdef.zerotol()) ) )
	{
	    if ( probdef.alphaState(i) == -1 )
	    {
		probdef.modAlphaLFtoZ(iP,Gp,Gp,Gn,Gpn,gp,gn,hp);
	    }

	    else
	    {
		probdef.modAlphaUFtoZ(iP,Gp,Gp,Gn,Gpn,gp,gn,hp);
	    }
	}

	else if ( ( probdef.alpha(i) <= lb(i)+(probdef.zerotol()) ) && ( probdef.alphaState(i) == -1 ) )
	{
	    probdef.modAlphaLFtoLB(iP,Gp,Gp,Gn,Gpn,gp,gn,hp,lb);
	}

	else if ( ( probdef.alpha(i) >= ub(i)-(probdef.zerotol()) ) && ( probdef.alphaState(i) == +1 ) )
	{
	    probdef.modAlphaUFtoUB(iP,Gp,Gp,Gn,Gpn,gp,gn,hp,ub);
	}

        return 1;
    }

    return 0;
}
