
//
// Sparse quadratic solver - large scale, d2c based, warm start
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "sQd2c.h"
#include "sQsmo.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

unsigned int solve_d2c(svmvolatile int &killSwitch, optState<double,double> &x, 
                       const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
                       const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
                       const Vector<double> &lb, const Vector<double> &ub, 
                       int maxitcnt, double maxtraintime, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);

#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 1000



//int solve_quadratic_program_d2c(svmvolatile int &killSwitch, optState<double,double> &x, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, 
//const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, int maxitcnt, double maxtraintime)
int fullOptStateD2C::solve(svmvolatile int &killSwitch)
{
    NiceAssert( x.bN() == 1 );
    NiceAssert( ( x.betaRestrict(0) == 0 ) || ( x.betaRestrict(0) == 3 ) );
    NiceAssert( Gn(zeroint(),0) == 0.0 );
    NiceAssert( maxitcnt >= 0 );

    unsigned int res = 0;

//    double stepscale = stepscalefactor;

    if ( x.betaRestrict(0) == 0 )
    {
	if ( !(x.betaState(0)) )
	{
            x.modBetaCtoF(0,Gp,Gp,Gn,Gpn,gp,gn,hp);
	}

	x.refreshGrad(Gp,Gn,Gpn,gp,gn,hp);

	if ( ( x.betaGrad(0) < -(x.zerotol()) ) || ( x.betaGrad(0) > (x.zerotol()) ) )
	{
            throw("Nonfeasible start not enabled for d2c");

	    //int tmpres;
            //
            //if ( ( tmpres = solve_quadratic_program_smo(killSwitch,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,maxitcnt,maxruntime,1) ) )
	    //{
            //    return tmpres;
	    //}
	}

	res = solve_d2c(killSwitch,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,maxitcnt,maxruntime,NULL,NULL,1.0);
    }

    else
    {
	res = solve_d2c(killSwitch,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,maxitcnt,maxruntime,NULL,NULL,1.0);
    }

    return res;
}




// Copied from the old svmheavy and updated

/***********************************************************************

                            D2C optimiser
                            =============

***********************************************************************/

/*
   Size of buffer used when sorting the most non-optimal points.
*/

/*
   SORTINTOPOS(ee,new_tau,alpha_dist):

   - ee is the gradient at alpha
   - new_tau is what tau will become if we move in the -ve alpha direction.
   - alpha_dist is the distance we can move in the -ve alpha direction
     before hitting a bound.

   This macro will attempt to put variable i into the buffer of maximal
   gradients in the -ve alpha direction - that is, a buffer ordered in
   terms of decreasing gradients, starting with the most positive gradient.


   SORTINTONEG(ee,new_tau,alpha_dist):

   - ee is the gradient at alpha
   - new_tau is what tau will become if we move in the +ve alpha direction.
   - alpha_dist is the distance we can move in the +ve alpha direction
     before hitting a bound.

   This macro will attempt to put variable i into the buffer of maximal
   gradients in the +ve alpha direction - that is, a buffer ordered in
   terms of increasing gradients, starting with the most negative gradient.
*/



#define D2C_BUFF_SIZE 10

// The second condition here overrides the first condition if the
// position in the buffer is empty.

#define SORTINTOPOS(_ee_,_old_tau_,_new_tau_)                           \
{                                                                       \
    j = D2C_BUFF_SIZE-1;                                                \
                                                                        \
    while ( ( ( _ee_ > max_pos_val((j>0)?j:0) ) || !(max_pos_pos((j>0)?j:0)) ) && j+1 ) \
    {                                                                   \
        j--;                                                            \
    }                                                                   \
                                                                        \
    if ( j < D2C_BUFF_SIZE-1 )                                          \
    {                                                                   \
        k = D2C_BUFF_SIZE-2;                                            \
                                                                        \
        while ( k > j )                                                 \
        {                                                               \
            max_pos_pos("&",k+1)        = max_pos_pos(k);              \
            max_pos_val("&",k+1)        = max_pos_val(k);              \
            max_pos_old_tau("&",k+1)    = max_pos_old_tau(k);          \
            max_pos_new_tau("&",k+1)    = max_pos_new_tau(k);          \
                                                                        \
            k--;                                                        \
        }                                                               \
                                                                        \
        max_pos_pos("&",j+1)        = i;                               \
        max_pos_val("&",j+1)        = _ee_;                            \
        max_pos_old_tau("&",j+1)    = _old_tau_;                       \
        max_pos_new_tau("&",j+1)    = _new_tau_;                       \
                                                                        \
        if ( pos_buffer_len < D2C_BUFF_SIZE )                           \
        {                                                               \
            pos_buffer_len++;                                           \
        }                                                               \
    }                                                                   \
}

#define SORTINTONEG(_ee_,_old_tau_,_new_tau_)                           \
{                                                                       \
    j = D2C_BUFF_SIZE-1;                                                \
                                                                        \
    while ( ( ( _ee_ < max_neg_val((j>0)?j:0) ) || !(max_neg_pos((j>0)?j:0)) ) && j+1 ) \
    {                                                                   \
        j--;                                                            \
    }                                                                   \
                                                                        \
    if ( j < D2C_BUFF_SIZE-1 )                                          \
    {                                                                   \
        k = D2C_BUFF_SIZE-2;                                            \
                                                                        \
        while ( k > j )                                                 \
        {                                                               \
            max_neg_pos("&",k+1)        = max_neg_pos(k);              \
            max_neg_val("&",k+1)        = max_neg_val(k);              \
            max_neg_old_tau("&",k+1)    = max_neg_old_tau(k);          \
            max_neg_new_tau("&",k+1)    = max_neg_new_tau(k);          \
                                                                        \
            k--;                                                        \
        }                                                               \
                                                                        \
        max_neg_pos("&",j+1)        = i;                               \
        max_neg_val("&",j+1)        = _ee_;                            \
        max_neg_old_tau("&",j+1)    = _old_tau_;                       \
        max_neg_new_tau("&",j+1)    = _new_tau_;                       \
                                                                        \
        if ( neg_buffer_len < D2C_BUFF_SIZE )                           \
        {                                                               \
            neg_buffer_len++;                                           \
        }                                                               \
    }                                                                   \
}

/*
               Heuristic 1 - if largest violator at bound, find largest
                             complimentary violator at bound and step in
                             both.

                             QUESTION: This is assuming a consistent upper
                                       (lower) bound on alpha.  However, this
                                       won't necessarily hold - what's the
                                       best way to deal with this?  What is
                                       the best method of choosing in this
                                       case?

                             ANSWER?: if the distance alpha_i may travel is
                                      a_i then the change in the objective fn
                                      will be (up to a negative scale
                                      factor):
                                      a_i.e_i + a_j.e_j.
                                      We want to maximise this magnitude, and
                                      to do so we maximise a_i.e_i and
                                      a_j.e_j.  Of course, need to ensure
                                      that the directions of the steps are
                                      complementary - ie.
                                      - if alpha_i = v => either alpha_j = h
                                        or alpha_j = 0 and tau_j = -1
                                      - if alpha_i = h => either alpha_j = v
                                        or alpha_j = 0 and tau_j = +1
                                      - if alpha_i = 0 and tau_i = +1 then
                                        either alpha_j = h or alpha_j = 0
                                        and tau_j = -1
                                      - if alpha_i = 0 and tau_i = -1 then
                                        either alpha_j = v or alpha_j = 0
                                        and tau_j = +1

                             OBVIOUS FOLLOWUP Q: given that we want to max
                                                 a_i.e_i and a_j.e_j, why not
                                                 just do this straight off?
                                                 To put it another way - am I
                                                 looking at the correct
                                                 metric here when ordering
                                                 the e_i values?  Would it
                                                 be better to order the
                                                 a_i.e_i values instead?

                             FOR NOW: just blindly follow the algorithm,
                                      ignore the magnitude of a_i.

                             IMPORTANT: when f != 0 then it is necessary to
                                        check to ensure that a step has
                                        actually occured, or infinite loops
                                        may result.


               Heuristic 2 - if largest violator not at bound, do an
                             exhaustive search to find the largest change in
                             the objective (ASSUMPTION - ignore f component,
                             base only on e) and take this step.

               Heuristic 3 - the first 2 heuristics can fail if f != 0.  In
                             either case, simply revert to SMO approximately
                             when examineall = 1.  This is NOT the optimal
                             thing to do.
*/



unsigned int solve_d2c(svmvolatile int &killSwitch, optState<double,double> &x, 
                       const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
                       const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, 
                       const Vector<double> &lb, const Vector<double> &ub, 
                       int maxitcntint, double maxtraintime, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &), void *htArg, double stepscalefactor)
{
    errstream() << "#\b";

    int step_taken;
    double ee;
    double bj_da_dJ = 0.0;
    double d_J_epart = 0.0;
    int f_zero = 1; /* need for calling SMO stuff */
    int first_try;
    int i,j,k,l,m;
    int i1,i2 = 0;
    int oldtau1;//,oldtau2;
    int newtau1,newtau2 = 0;
    double e1;//,e2;
    int pos_buffer_len;
    int neg_buffer_len;
    Vector<int>    max_pos_pos(D2C_BUFF_SIZE);
    Vector<double> max_pos_val(D2C_BUFF_SIZE);
    Vector<int>    max_pos_old_tau(D2C_BUFF_SIZE);
    Vector<int>    max_pos_new_tau(D2C_BUFF_SIZE);
    Vector<int>    max_neg_pos(D2C_BUFF_SIZE);
    Vector<double> max_neg_val(D2C_BUFF_SIZE);
    Vector<int>    max_neg_old_tau(D2C_BUFF_SIZE);
    Vector<int>    max_neg_new_tau(D2C_BUFF_SIZE);
    Vector<int>    max_sec_pos(D2C_BUFF_SIZE);
    Vector<double> max_sec_val(D2C_BUFF_SIZE);
    Vector<int>    max_sec_old_tau(D2C_BUFF_SIZE);
    Vector<int>    max_sec_new_tau(D2C_BUFF_SIZE);
    double d_alpha1 = 0.0;
    double d_alpha2 = 0.0;
    double d_b = 0.0;
    //double d_f = 0.0;
    double d_alpha1_trial = 0.0;
    double d_alpha2_trial = 0.0;
    double d_b_trial = 0.0;
    //double d_f_trial = 0.0;

    (void) fixHigherOrderTerms;
    (void) htArg;
//    if ( fixHigherOrderTerms )
//    {
//        fixHigherOrderTerms(x,htArg);
//    }

    /*
       max_pos_pos: descent in the +alpha direction
       max_neg_pos: descent in the -alpha direction

       *_new_tau: tau value used to calculate this value.
    */


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
	/*
           First thing - sort the most non-optimal into +ve and -ve buffers.
        */

	max_pos_pos.zero();
	max_neg_pos.zero();
	max_pos_val.zero();
        max_neg_val.zero();

        pos_buffer_len = 0;
        neg_buffer_len = 0;

        for ( i = 0 ; i < x.aN() ; i++ )
	{
	    if ( x.alphaState(i) != 3 )
	    {
                /*
                    tau | contype | alpha can? | gradients
                   -----+---------+------------+-----------
                     +2 | any     | decrease   | +ve
                     -2 | any     | increase   | -ve
                     +1 | any     | either     | +ve/-ve
                     -1 | any     | either     | +ve/-ve
                     0  | 0       | do nothing | none
                        | 1       | decrease   | +ve
                        | 2       | increase   | -ve
                        | 3       | either     | +ve/-ve
                */

		if ( x.alphaState(i) == +2 )
		{
		    ee = x.posAlphaGrad(ee,i,Gp,Gpn,gp,hp);
                    SORTINTOPOS(ee,+2,+1);
		}

		else if ( x.alphaState(i) == -2 )
		{
		    ee = x.negAlphaGrad(ee,i,Gp,Gpn,gp,hp);
                    SORTINTONEG(ee,-2,-1);
		}

		else if ( x.alphaState(i) == +1 )
		{
		    ee = x.posAlphaGrad(ee,i,Gp,Gpn,gp,hp);
		    SORTINTOPOS(ee,+1,+1);
                    SORTINTONEG(ee,+1,+1);
		}

		else if ( x.alphaState(i) == -1 )
		{
		    ee = x.negAlphaGrad(ee,i,Gp,Gpn,gp,hp);
		    SORTINTOPOS(ee,-1,-1);
                    SORTINTONEG(ee,-1,-1);
		}

		else
		{
		    if ( x.alphaRestrict(i) == 0 )
		    {
			ee = x.negAlphaGrad(ee,i,Gp,Gpn,gp,hp);
			SORTINTOPOS(ee,0,-1);
			ee = x.posAlphaGrad(ee,i,Gp,Gpn,gp,hp);
			SORTINTONEG(ee,0,+1);
		    }

		    else if ( x.alphaRestrict(i) == 1 )
		    {
			ee = x.posAlphaGrad(ee,i,Gp,Gpn,gp,hp);
                        SORTINTONEG(ee,0,+1);
		    }

		    else if ( x.alphaRestrict(i) == 2 )
		    {
			ee = x.negAlphaGrad(ee,i,Gp,Gpn,gp,hp);
                        SORTINTOPOS(ee,0,-1);
		    }
                }
            }
        }

        /*
           Is the solution optimal?  This checks the violation gap and also
           (implicitly) the optimality of points not at the boundary.
        */

        if ( ( !(max_pos_pos(zeroint())) || ( max_pos_val(zeroint()) <  x.opttol() ) ) &&
             ( !(max_neg_pos(zeroint())) || ( max_neg_val(zeroint()) > -x.opttol() ) )    )
        {
            break;
        }

        step_taken = 0;

        i = 0;
        j = 0;

        /*
           Heuristics 1 and 2 will be used until either a step has been
           taken or the lists themselves are both exhausted.
        */

        while ( ( !step_taken        ) &&
                ( i < pos_buffer_len ) &&
                ( j < neg_buffer_len )    )
        {
            if ( max_pos_val(i) > -max_neg_val(j) )
            {
                i1      = max_pos_pos(i);
                e1      = max_pos_val(i);
                oldtau1 = max_pos_old_tau(i);
                newtau1 = max_pos_new_tau(i);

                i++;

                max_sec_pos        = max_neg_pos;
                max_sec_val        = max_neg_val;
                max_sec_new_tau    = max_neg_new_tau;
                max_sec_old_tau    = max_neg_new_tau;

                k = j;
                l = neg_buffer_len;
            }

            else
            {
                i1      = max_neg_pos(j);
                e1      = max_neg_val(j);
                oldtau1 = max_neg_old_tau(j);
                newtau1 = max_neg_new_tau(j);

                j++;

                max_sec_pos        = max_pos_pos;
                max_sec_val        = max_pos_val;
                max_sec_new_tau    = max_pos_new_tau;
                max_sec_old_tau    = max_pos_new_tau;

                k = i;
                l = pos_buffer_len;
            }

            /*
               Heuristic 1
            */

            if ( ( oldtau1 == -2 ) ||
                 ( oldtau1 ==  0 ) ||
                 ( oldtau1 == +2 )    )
            {
                /*
                   Scan through the secondary list.  The first to be found
                   with oldtau being one of -2,0,+2 is at a bound and
                   pointed in the correct direction.  Hence attempt to take
                   a step w.r.t. this one.  If this attempted step succeeds
                   then everything is good, we can exit the while loop.  If
                   this step fails, keep looking until another potential is
                   found or fall-through occurs.
                */

                for ( m = k ; ( ( m < l ) && !step_taken ) ; m++ )
                {
                    if ( ( max_sec_old_tau(m) == -2 ) ||
                         ( max_sec_old_tau(m) ==  0 ) ||
                         ( max_sec_old_tau(m) == +2 )    )
                    {
                        step_taken = takeStep_SMO(i1,max_sec_pos(m),newtau1,max_sec_new_tau(m),e1,max_sec_val("&",m),x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
                    }
                }
            }

            /*
               Heuristic 2
            */

            if ( !step_taken )
            {
                /*
                   Find the largest feasible step by searching through the
                   secondary gradient list.
                */

                first_try = 1;

                for ( m = k ; m < l ; m++ )
                {
                    /*
                       Trial the step - this will record the step itself and
                       also calculate the change in the objective function J
                       due to the step (it will return 0 if no step is
                       possible).
                    */

                    if ( i1 != max_sec_pos(m) )
                    {
                        if ( trial_step_SMO(d_J_epart,i1,max_sec_pos(m),newtau1,max_sec_new_tau(m),e1,max_sec_val("&",m),d_alpha1_trial,d_alpha2_trial,d_b_trial,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
                        {
                            /*
                               OK, the step is feasible.  If either this is
                               the first feasible step (first_try = 1) or it
                               results in the largest objective change then
                               record it.  Otherwise, ignore it.
                            */

                            if ( first_try )
                            {
                                bj_da_dJ = d_J_epart;

                                i2      = max_sec_pos(m);
                                //e2      = max_sec_val(m);
                                //oldtau2 = max_sec_old_tau(m);
                                newtau2 = max_sec_new_tau(m);

                                d_alpha1 = d_alpha1_trial;
                                d_alpha2 = d_alpha2_trial;
                                d_b      = d_b_trial;
                                //d_f      = d_f_trial;

                                first_try = 0;
                            }

                            else if ( d_J_epart < bj_da_dJ )
                            {
                                bj_da_dJ = d_J_epart;

                                i2      = max_sec_pos(m);
                                //e2      = max_sec_val(m);
                                //oldtau2 = max_sec_old_tau(m);
                                newtau2 = max_sec_new_tau(m);

                                d_alpha1 = d_alpha1_trial;
                                d_alpha2 = d_alpha2_trial;
                                d_b      = d_b_trial;
                                //d_f      = d_f_trial;
                            }
                        }
                    }
                }

                /*
                   Take the largest feasible step, if there is one.  The
                   function actually_take_step_d2c will always return 1,
                   so this will correctly set step_taken.
                */

                if ( !first_try )
                {
                    step_taken = actually_take_step_SMO(i1,i2,newtau1,newtau2,d_alpha1,d_alpha2,d_b,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
                }
            }
        }

        if ( !step_taken )
        {
            /*
               Heuristic 3
            */

	    for ( i = 0 ; i < x.aN() ; i++ )
	    {
		if ( x.alphaRestrict(i) != 3 )
		{
		    if ( x.alphaState(i) == 0 )
		    {
			if ( x.alphaRestrict(i) == 0 )
			{
			    // Unrestricted, so try both positive and negative in random order

			    if ( svm_rand() % 2 )
			    {
				if ( examineExample_SMO(i,-1,f_zero,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				{
				    step_taken = 1;
				}

				else if ( examineExample_SMO(i,+1,f_zero,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				{
				    step_taken = 1;
				}
			    }

			    else
			    {
				if ( examineExample_SMO(i,+1,f_zero,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				{
				    step_taken = 1;
				}

				else if ( examineExample_SMO(i,-1,f_zero,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor) )
				{
				    step_taken = 1;
				}
			    }
			}

			else if ( x.alphaRestrict(i) == 1 )
			{
			    step_taken += examineExample_SMO(i,+1,f_zero,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
			}

			else
			{
			    step_taken += examineExample_SMO(i,-1,f_zero,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
			}
		    }

		    else if ( x.alphaState(i) == +1 )
		    {
			step_taken += examineExample_SMO(i,+1,f_zero,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
		    }

		    else if ( x.alphaState(i) == -1 )
		    {
			step_taken += examineExample_SMO(i,-1,f_zero,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
		    }

		    else if ( x.alphaState(i) == +2 )
		    {
			step_taken += examineExample_SMO(i,+1,f_zero,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
		    }

		    else if ( x.alphaState(i) == -2 )
		    {
			step_taken += examineExample_SMO(i,-1,f_zero,x,Gp,Gpsigma,Gn,Gpn,gp,gn,hp,lb,ub,fixHigherOrderTerms,htArg,stepscalefactor);
		    }
		}
	    }
        }

        /*
           Fall-back case (numerical weirdness has happened).
        */

        if ( !step_taken )
        {
            isopt = 1;
            res = 0;
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
            timeout = kbquitdet("D2C (sQd2c) optimisation",uservars,varnames,vardescr);
        }
    }

    return res;
}
