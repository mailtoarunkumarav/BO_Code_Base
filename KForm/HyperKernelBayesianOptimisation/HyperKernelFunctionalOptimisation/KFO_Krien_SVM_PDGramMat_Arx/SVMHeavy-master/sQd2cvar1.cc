
//
// Sparse quadratic solver - large scale, d2c variant based, warm start
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "sQd2cvar1.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// STEPSCALETOL: tolerance level before which a constraint is inactivated
// STEPPT: value at which step is considered "significant"
// BOUNDMAXITCNT: maximum number of iterations on the inner loop before we give up
// RANDOM_INNER_STEP: after RANDOM_INNER_STEP times freeing the largest
// Lagrange multiplier we just free a random constraint.  In this way we can
// break cycles which may form due to the degeneracy of the constraint set.

#define STEPSCALETOL 1e-6
#define STEPPT 1e-5
#define BOUNDMAXITCNT 10
//#define BOUNDMAXITCNT 100
#define RANDOM_INNER_STEP 10
#define DOUBLERANDOM_INNER_STEP 50
#define RANDOM_OUTER_STEP 50
#define DOUBLERANDOM_OUTER_STEP 100
#define STEPSPEROUTERBLOCK 1

#define RANDOMLOOPPERIOD 500
#define RANDOMSEEDA time(NULL)
#define RANDOMSEEDB 42

int trainMG2(Vector<optState<double,double> *> &x, const Matrix<double> &Gp, const Matrix<double> &Gpn, const Matrix<double> &Gn, Vector<Vector<double> > &gp, Vector<Vector<double> > &gn, const Vector<Vector<double> > &hp, const Vector<Vector<double> > &lb, const Vector<Vector<double> > &ub, const Matrix<double> &Gpsigma, Vector<double> &mu, double &xi, Vector<Vector<int> > &Sh, Vector<Vector<int> > &Sl, int &itcnt, const Vector<Vector<int> > &ranorder, int Ntrain, int n, int maxitcnt, double maxtraintime);

int solve_quadratic_program_d2cvar1(Vector<optState<double,double> *> &x, Vector<double> &mu, double xi, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<Vector<double> > &gpbase, const Vector<Vector<double> > &gnbase, const Vector<Vector<double> > &hp, const Vector<Vector<double> > &lbbase, const Vector<Vector<double> > &ubbase, int maxitcnt, double maxtraintime)
{
    int n = x.size();
    int Ntrain = mu.size();

    NiceAssert( x.size() == n );
    NiceAssert( mu.size() == Ntrain );
    NiceAssert( Gp.numRows() == Ntrain );
    NiceAssert( Gp.numCols() == Ntrain );
    NiceAssert( Gpsigma.numRows() == Ntrain );
    NiceAssert( Gpsigma.numCols() == Ntrain );
    NiceAssert( Gpn.numRows() == Ntrain );
    NiceAssert( Gpn.numCols() == 1 );
    NiceAssert( Gn.numRows() == 1 );
    NiceAssert( Gn.numCols() == 1 );
    NiceAssert( gpbase.size() == n );
    NiceAssert( gnbase.size() == n );
    NiceAssert( hp.size() == n );
    NiceAssert( lbbase.size() == n );
    NiceAssert( ubbase.size() == n );
    NiceAssert( Gn(zeroint(),0) == 0.0 );
    NiceAssert( maxitcnt >= 0 );

    if ( !n || !Ntrain )
    {
        return 0;
    }

#ifndef NDEBUG
    int sss;

    for ( sss = 0 ; sss < n ; sss++ )
    {
        NiceAssert( !(x(sss)->keepfact()) );
        NiceAssert( x(sss)->betaRestrict(0) == 0 );
        NiceAssert( x(sss)->aN() == Ntrain );
        NiceAssert( x(sss)->bN() == 1 );
        NiceAssert( gpbase(sss).size() == Ntrain );
        NiceAssert( gnbase(sss).size() == 1 );
        NiceAssert( hp(sss).size() == Ntrain );
        NiceAssert( lbbase(sss).size() == Ntrain );
        NiceAssert( ubbase(sss).size() == Ntrain );
    }
#endif

    Vector<Vector<double> > gp(gpbase);
    Vector<Vector<double> > gn(gnbase);
    Vector<Vector<double> > lb(lbbase);
    Vector<Vector<double> > ub(ubbase);

    unsigned int res = 0;
    int i,j,k,kl,ll,s,iP;

    // Construct order randomiser (used to randomise the order in which the
    // training algorithm considers points, lessening the chance of infinite
    // loops).

    srand(RANDOMSEEDB);

    Vector<Vector<int> > ranorder(RANDOMLOOPPERIOD);

    for ( i = 0 ; i < RANDOMLOOPPERIOD ; i++ )
    {
	ranorder("&",i).resize(Ntrain);

	for ( j = 0 ; j < Ntrain ; j++ )
	{
	    ranorder("&",i)("&",j) = j;
	}

	for ( j = 0 ; j < Ntrain ; j++ )
	{
	    k = j+(svm_rand()%(Ntrain-j));

	    kl = ranorder(i)(k);

	    if ( k > j )
	    {
		for ( ll = k ; ll > j ; ll-- )
		{
		    ranorder("&",i)("&",ll) = ranorder(i)(ll-1);
		}
	    }

	    ranorder("&",i)("&",j) = kl;
	}
    }

    // incorporate Lagrange multipliers into gp and gn

    for ( s = 0 ; s < n ; s++ )
    {
	gp("&",s) += mu;
	gn("&",s) += xi;

	x("&",s)->refactgp(Gp,Gn,Gpn,gpbase(s),gp(s),gnbase(s),hp(s));
	x("&",s)->refactgn(Gp,Gn,Gpn,gp(s),gnbase(s),gn(s),hp(s));
    }

    // Free all alphas, fix bounds lb and ub, and construct modified alpha state variables
    //
    // Sh(i)(s) == 1 indicates that alpha(s)(i) is at upper bound ub(s)(i) (ie it can only decrease)
    // Sh(i)(s) == 0 indicates that it is not
    // Sl(i)(s) == 1 indicates that alpha(s)(i) is at lower bound lb(s)(i) (ie it can only increase)
    // Sl(i)(s) == 0 indicates that it is not
    //
    // if both are 1 then it is constrained at zero

    Vector<Vector<int> > Sh(Ntrain);
    Vector<Vector<int> > Sl(Ntrain);

    int Np = 0;
    int Nn = 0;

    double sump = 0.0;
    double sumn = 0.0;

    for ( i = 0 ; i < Ntrain ; i++ )
    {
	Sh("&",i).resize(n);
	Sl("&",i).resize(n);
    }

    for ( s = 0 ; s < n ; s++ )
    {
	if ( x(s)->aNF() )
	{
	    for ( iP = (x(s)->aNF())-1 ; iP >= 0 ; iP-- )
	    {
		// alpha is free

		i = x(s)->pivAlphaF()(iP);

		Sh("&",i)("&",s) = 0;
		Sl("&",i)("&",s) = 0;

		if ( x(s)->alphaRestrict(i) == 1 )
		{
		    // alpha must be positive

		    lb("&",s)("&",i) = 0.0;

		    Np++;
                    sump += x(s)->alpha(i);
		}

		else if ( x(s)->alphaRestrict(i) == 2 )
		{
		    // alpha must be negative

		    ub("&",s)("&",i) = 0.0;

                    Nn++;
                    sumn += x(s)->alpha(i);
		}

		else
		{
		    // otherwise alpha is free, hp is presumed zero

                    NiceAssert( x(s)->alphaRestrict(i) == 0 );

		    if ( x(s)->alphaState(i) > 0 )
		    {
			Np++;
			sump += x(s)->alpha(i);
		    }

		    else
		    {
			Nn++;
			sumn += x(s)->alpha(i);
		    }
		}
	    }
	}

	if ( x(s)->aNLB() )
	{
	    for ( iP = (x(s)->aNLB())-1 ; iP >= 0 ; iP-- )
	    {
		// alpha is at lower bound

		i = x(s)->pivAlphaLB()(iP);

		x("&",s)->modAlphaLBtoLF(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s));

		Sh("&",i)("&",s) = 0;
		Sl("&",i)("&",s) = 1;

		if ( x(s)->alphaRestrict(i) == 2 )
		{
		    // alpha must be negative

		    ub("&",s)("&",i) = 0.0;
		}

		Nn++;
		sumn += x(s)->alpha(i);
	    }
	}

	if ( x(s)->aNUB() )
	{
	    for ( iP = (x(s)->aNUB())-1 ; iP >= 0 ; iP-- )
	    {
		// alpha is at upper bound

		i = x(s)->pivAlphaUB()(iP);

		x("&",s)->modAlphaUBtoUF(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s));

		Sh("&",i)("&",s) = 1;
		Sl("&",i)("&",s) = 0;

		if ( x(s)->alphaRestrict(i) == 1 )
		{
		    // alpha must be positive

		    lb("&",s)("&",i) = 0.0;
		}

                Np++;
		sump += x(s)->alpha(i);
	    }
	}

	if ( x(s)->aNZ() )
	{
	    for ( iP = (x(s)->aNZ())-1 ; iP >= 0 ; iP-- )
	    {
		// alpha is at zero

		i = x(s)->pivAlphaZ()(iP);

		if ( x(s)->alphaRestrict(i) == 0 )
		{
		    // alpha is free, hp assumed zero

		    x("&",s)->modAlphaZtoUF(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s));

		    Sh("&",i)("&",s) = 0;
		    Sl("&",i)("&",s) = 0;

		    Np++;
                    sump += x(s)->alpha(i);
		}

		else if ( x(s)->alphaRestrict(i) == 1 )
		{
		    // alpha must be positive

		    x("&",s)->modAlphaZtoUF(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s));

		    Sh("&",i)("&",s) = 0;
		    Sl("&",i)("&",s) = 1;

		    lb("&",s)("&",i) = 0.0;

                    Np++;
                    sump += x(s)->alpha(i);
		}

		else if ( x(s)->alphaRestrict(i) == 2 )
		{
		    // alpha must be negative

		    x("&",s)->modAlphaZtoLF(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s));

		    Sh("&",i)("&",s) = 1;
		    Sl("&",i)("&",s) = 0;

		    ub("&",s)("&",i) = 0.0;

                    Nn++;
                    sumn += x(s)->alpha(i);
		}

		else if ( x(s)->alphaRestrict(i) == 3 )
		{
		    // alpha is stuck at zero

		    Sh("&",i)("&",s) = 1;
		    Sl("&",i)("&",s) = 1;

		    lb("&",s)("&",i) = 0.0;
		    ub("&",s)("&",i) = 0.0;
		}
	    }
	}
    }

    // ensure that muGrad_i = \sum_s \alpha_is = 0 for all i and xiGrad = \sum_s \beta_s = 0

    double xiGrad;
    Vector<double> muGrad(Ntrain);
    double muscale;

    xiGrad = 0.0;
    muGrad = 0.0;

    for ( s = 0 ; s < n ; s++ )
    {
	xiGrad += x(s)->beta(0);

	if ( x(s)->aNF() )
	{
	    for ( iP = (x(s)->aNF())-1 ; iP >= 0 ; iP-- )
	    {
		i = x(s)->pivAlphaF()(iP);

		muGrad("&",i) += x(s)->alpha(i);
	    }
	}
    }

    if ( fabs(xiGrad) >= x(zeroint())->opttol() )
    {
	xiGrad *= -1.0/((double) n);

	for ( s = 0 ; s < n ; s++ )
	{
	    x(s)->betaStep(0,xiGrad,Gp,Gn,Gpn,gp(s),gn(s),hp(s),1);
	}
    }

    for ( i = 0 ; i < Ntrain ; i++ )
    {
        if ( muGrad(i) >= x(zeroint())->opttol() )
	{
	    muscale = muGrad(i)/sump;

	    for ( s = 0 ; s < n ; s++ )
	    {
		if ( x(s)->alphaState(i) > 0 )
		{
		    x("&",s)->alphaStep(j,-muscale*(x(s)->alpha(i)),Gp,Gn,Gpn,gp(s),gn(s),hp(s),1);

                    Sh("&",i)("&",s) = 0;
		}
	    }
	}

        else if ( muGrad(i) <= -(x(zeroint())->opttol()) )
	{
	    muscale = muGrad(i)/sumn;

	    for ( s = 0 ; s < n ; s++ )
	    {
		if ( x(s)->alphaState(i) < 0 )
		{
		    x("&",s)->alphaStep(j,-muscale*(x(s)->alpha(i)),Gp,Gn,Gpn,gp(s),gn(s),hp(s),1);

                    Sl("&",i)("&",s) = 0;
		}
	    }
	}
    }

    // free bias if required

    for ( s = 0 ; s < n ; s++ )
    {
	if ( !(x(s)->betaState(0)) )
	{
	    x("&",s)->modBetaCtoF(0,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s));
	}

	x("&",s)->refreshGrad(Gp,Gn,Gpn,gp(s),gn(s),hp(s));

	//if ( ( x(s)->betaGrad(0) < -(x(s)->zerotol()) ) || ( x(s)->betaGrad(0) > (x(s)->zerotol()) ) )
	//{
	//    FIXME: need a centering step here
	//}
    }

    // Train the multiclass SVM

    errstream() << "Optimising... \n";

    int itcnt = 0;
    time_used optstarttime,optendtime;

    optstarttime = TIMECALL;

    res = trainMG2(x,Gp,Gpn,Gn,gp,gn,hp,lb,ub,Gpsigma,mu,xi,Sh,Sl,itcnt,ranorder,Ntrain,n,maxitcnt,maxtraintime);

    if ( res )
    {
        errstream() << "...failed with error code " << res << "\n";
    }

    else
    {
        errstream() << "...complete.\n";
    }

    optendtime = TIMECALL;

    errstream() << "Optimisation time: " << TIMEDIFFSEC(optendtime,optstarttime) << " sec.\n";

    // Reconstrain bounded alphas

    for ( s = 0 ; s < n ; s++ )
    {
	if ( x(s)->aNF() )
	{
	    for ( iP = (x(s)->aNF())-1 ; iP >= 0 ; iP-- )
	    {
		i = x(s)->pivAlphaF()(iP);

		if ( !Sl(i)(s) && !Sh(i)(s) )
		{
		    if ( x(s)->alphaRestrict(i) == 0 )
		    {
			// Not at bound, unrestricted

			if ( x(s)->alpha(i) <= -x(s)->zerotol() )
			{
			    // Negative

			    if ( x(s)->alphaState() == +1 )
			    {
				x(s)->modAlphaUFtoLF(iP,Gp,Gn,Gpn,gp(s),gn(s),hp(s));
			    }

                            NiceAssert( x(s)->alphaState() == -1 );
			}

			else if ( x(s)->alpha(i) < x(s)->zerotol() )
			{
			    // Zero

			    if ( x(s)->alphaState() == +1 )
			    {
				x(s)->modAlphaUFtoZ(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s));
			    }

			    else
			    {
                                NiceAssert( x(s)->alphaState() == -1 );

				x(s)->modAlphaLFtoZ(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s));
			    }
			}

			else
			{
                            // Positive

			    if ( x(s)->alphaState() == -1 )
			    {
				x(s)->modAlphaLFtoUF(iP,Gp,Gn,Gpn,gp(s),gn(s),hp(s));
			    }

                            NiceAssert( x(s)->alphaState() == +1 );
			}
		    }
		}

		else if ( Sl(i)(s) && !(Sh(i)(s)) )
		{
		    // At a lower bound (or zero)

		    if ( x(s)->alphaRestrict(i) == 0 )
		    {
                        // At lower bound, unrestricted

			if ( x(s)->alphaState() == +1 )
			{
			    x(s)->modAlphaUFtoLF(iP,Gp,Gn,Gpn,gp(s),gn(s),hp(s));
			}

			x("&",s)->modAlphaLFtoLB(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s),lb(s));
		    }

                    else if ( x(s)->alphaRestrict(i) == 1 )
		    {
			// At zero

			x("&",s)->modAlphaUFtoZ(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s));
		    }

		    else
		    {
			// At lower bound, restricted negative

			x("&",s)->modAlphaLFtoLB(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s),lb(s));
		    }
		}

		else if ( Sh(i)(s) && !(Sl(i)(s)) )
		{
		    // At a upper bound (or zero)

		    if ( x(s)->alphaRestrict(i) == 0 )
		    {
                        // At upper bound, unrestricted

			if ( x(s)->alphaState() == -1 )
			{
			    x(s)->modAlphaLFtoUF(iP,Gp,Gn,Gpn,gp(s),gn(s),hp(s));
			}

			x("&",s)->modAlphaUFtoUB(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s),ub(s));
		    }

		    else if ( x(s)->alphaRestrict(i) == 2 )
		    {
			// At zero

			x("&",s)->modAlphaLFtoZ(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s));
		    }

		    else
		    {
			// At upper bound, restricted possible

			x("&",s)->modAlphaUFtoUB(iP,Gp,Gp,Gn,Gpn,gp(s),gn(s),hp(s),ub(s));
		    }
		}
	    }
	}
    }

    // remove Lagrange multipliers from gp and gn

    for ( s = 0 ; s < n ; s++ )
    {
	x("&",s)->refactgp(Gp,Gn,Gpn,gp(s),gpbase(s),gnbase(s),hp(s));
	x("&",s)->refactgn(Gp,Gn,Gpn,gp(s),gn(s),gnbase(s),hp(s));
    }

    // ...and we're done.

    return 0;
}






int trystep(int i, int j, Vector<int> &Sij, Vector<double> &beta, Vector<double> &STbeta, Vector<double> &betaGrad, Vector<double> &STbetaGrad,
	    Vector<double> &alphai, Vector<double> &alphaj, Vector<double> &STalphai, Vector<double> &STalphaj,
	    Vector<double> &alphaGradi, Vector<double> &alphaGradj, Vector<double> &STalphaGradi, Vector<double> &STalphaGradj,
	    Vector<double> &Dbeta, Vector<double> &DbetaGrad, Vector<double> &Dalphai, Vector<double> &Dalphaj, Vector<double> &DalphaGradi, Vector<double> &DalphaGradj,
	    Vector<int> &nonoptlisti, Vector<int> &nonoptlistj, Vector<int> &nonopttypei, Vector<int> &nonopttypej, Vector<int> &atbndlisti, Vector<int> &atbndlistj,
            Vector<int> &atbndtypei, Vector<int> &atbndtypej, Vector<optState<double,double> *> &x, const Matrix<double> &Gp, const Matrix<double> &Gpn, const Matrix<double> &Gn,
	    Vector<Vector<double> > &gp, Vector<Vector<double> > &gn, const Vector<Vector<double> > &hp,
	    const Vector<Vector<double> > &lb, const Vector<Vector<double> > &ub, const Matrix<double> &Gpsigma, Vector<double> &mu, double xi,
	    Vector<Vector<int> > &Sh, Vector<Vector<int> > &Sl, int n, int isbetaopt);

class listelm;
class listelm
{
public:
    int i;
    int isfixed;
    double emag0norm;
    double emag1norm;
    double emag2norm;
    double emaginfnorm;
    double ediff0norm;
    double ediff1norm;
    double ediff2norm;
    double ediffinfnorm;
    listelm *nextmag;
    listelm *nextdiff;
};

int trainMG2(Vector<optState<double,double> *> &x, const Matrix<double> &Gp, const Matrix<double> &Gpn, const Matrix<double> &Gn, Vector<Vector<double> > &gp, Vector<Vector<double> > &gn, const Vector<Vector<double> > &hp, const Vector<Vector<double> > &lb, const Vector<Vector<double> > &ub, const Matrix<double> &Gpsigma, Vector<double> &mu, double &xi, Vector<Vector<int> > &Sh, Vector<Vector<int> > &Sl, int &itcnt, const Vector<Vector<int> > &ranorder, int Ntrain, int n, int maxitcnt, double maxtraintime)
{
    Vector<int> Sij(n);

    Vector<double> STbeta(n);
    Vector<double> STbetaGrad(n);
    Vector<double> STalphai(n);
    Vector<double> STalphaj(n);
    Vector<double> STalphaGradi(n);
    Vector<double> STalphaGradj(n);

    Vector<double> beta(n);
    Vector<double> betaGrad(n);
    Vector<double> alphai(n);
    Vector<double> alphaj(n);
    Vector<double> alphaGradi(n);
    Vector<double> alphaGradj(n);

    Vector<double> Dbeta(n);
    Vector<double> DbetaGrad(n);
    Vector<double> Dalphai(n);
    Vector<double> Dalphaj(n);
    Vector<double> DalphaGradi(n);
    Vector<double> DalphaGradj(n);

    Vector<int> nonoptlisti(n);
    Vector<int> nonoptlistj(n);
    Vector<int> nonopttypei(n);
    Vector<int> nonopttypej(n);

    Vector<int> atbndlisti(n);
    Vector<int> atbndlistj(n);
    Vector<int> atbndtypei(n);
    Vector<int> atbndtypej(n);

    int isbetaopt = 0;
    int isopt = 0;
    //int exitcode = 0;
    int initcnt = 0;
    int inneritcnt = 0;
    int locstep = 0;
    int stepdone = 0;

    int i,j,ii,jj;
    int s;

    Vector<Vector<double> > fbar(Ntrain);

    for ( i = 0 ; i < Ntrain ; i++ )
    {
        fbar("&",i).resize(n);
    }

    listelm *onelist;
    listelm *onefirstelm;
    listelm *twofirstelm;

    MEMNEWARRAY(onelist,listelm,Ntrain);

    // Initialise numbering in list

    for ( i = 0 ; i < Ntrain ; i++ )
    {
        onelist[i].i = i;
    }

    // Begin optimisation

    double emagsing,emag0norm,emag1norm,emag2norm,emaginfnorm;
    listelm *here;
    listelm *there;
    listelm *onept;
    listelm *twopt;
    int Nnf;
    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    int timeout = 0;

    while ( !isopt && ( ( itcnt < maxitcnt ) || !maxitcnt ) && !timeout )
    {
        if ( maxtraintime > 1 )
	{
            curr_time = TIMECALL;

            if ( TIMEDIFFSEC(curr_time,start_time) > maxtraintime )
	    {
		timeout = 1;
	    }
	}

errstream() << ".";
	// Test optimality and store error magnitudes and corrected errors

	isopt = 1;
	Nnf = 0;

	if ( !isbetaopt )
	{
	    for ( s = 0 ; s < n ; s++ )
	    {
		if ( fabs(x(s)->betaGrad(0)) >= x(s)->opttol() )
		{
		    isopt = 0;
		}
	    }

	    if ( isopt )
	    {
                isbetaopt = 1;
	    }
	}

	for ( ii = 0 ; ii < Ntrain ; ii++ )
	{
	    i = ranorder(itcnt%RANDOMLOOPPERIOD)(ii);

	    // NB: I tried a 1-norm here, but it was significantly slower than the 2-norm

	    emag0norm   = 0; // for some reason this was inside the s loop in the old version.  Was this a bug or a deliberate choice?
	    emag1norm   = 0;
	    emag2norm   = 0;
            emaginfnorm = 0;

            if ( x(zeroint())->alphaRestrict(i) != 3 )
	    {
		Nnf++;

		for ( s = 0 ; s < n ; s++ )
		{
		    emagsing = 0;

		    if ( !Sl(i)(s) && !Sh(i)(s) )
		    {
			// alpha(i)(s) is free

			emagsing = x(s)->alphaGrad(i);
		    }

		    else if ( Sl(i)(s) && !Sh(i)(s) )
		    {
			// alpha(i)(s) is at lower bound

			if ( x(s)->alphaGrad(i) <= -(x(s)->opttol()) )
			{
			    emagsing = x(s)->alphaGrad(i);
			}
		    }

		    else
		    {
                        NiceAssert( !Sl(i)(s) && Sh(i)(s) );

			// alpha(i)(s) is at upper bound

			if ( x(s)->alphaGrad(i) >= (x(s)->opttol()) )
			{
			    emagsing = x(s)->alphaGrad(i);
			}
		    }

                    fbar("&",i)("&",s) = emagsing;

		    emag0norm    = ( ( fabs(emagsing) < emag0norm ) ? fabs(emagsing) : emag0norm );
		    emag1norm   += fabs(emagsing);
		    emag2norm   += (emagsing*emagsing);
		    emaginfnorm  = ( ( fabs(emagsing) > emaginfnorm ) ? fabs(emagsing) : emaginfnorm );
		}

		onelist[i].emag0norm   = emag0norm;
		onelist[i].emag1norm   = emag1norm;
		onelist[i].emag2norm   = emag2norm;
                onelist[i].emaginfnorm = emaginfnorm;
		onelist[i].nextmag     = NULL;
		onelist[i].isfixed     = 0;

                if ( emag2norm > x(zeroint())->opttol() )
		{
		    isopt = 0;
		}
	    }

	    else
	    {
		onelist[i].emag0norm   = 0;
		onelist[i].emag1norm   = 0;
		onelist[i].emag2norm   = 0;
		onelist[i].emaginfnorm = 0;
		onelist[i].nextmag     = NULL;
		onelist[i].isfixed     = 1;
	    }
	}

        // If not optimal attempt a step

	if ( !isopt )
	{
	    // Sort list of error magnitudes

	    i = 0;

	    while ( onelist[i].isfixed )
	    {
                i++;
	    }

	    onefirstelm = &onelist[i];

            i++;

	    for ( ; i < Ntrain ; i++ )
	    {
		if ( !(onelist[i].isfixed) )
		{
		    if ( onelist[i].emag2norm >= onefirstelm->emag2norm )
		    {
			onelist[i].nextmag = onefirstelm;
			onefirstelm = &onelist[i];
		    }

		    else
		    {
			here = onefirstelm;

			while ( ( here->nextmag != NULL ) && ( (here->nextmag)->emag2norm > onelist[i].emag2norm ) )
			{
			    here = here->nextmag;
			}

			if ( here->nextmag == NULL )
			{
			    here->nextmag = &onelist[i];
			}

			else
			{
			    onelist[i].nextmag = here->nextmag;
			    here->nextmag = &onelist[i];
			}
		    }
		}
	    }

	    if ( !(initcnt%DOUBLERANDOM_OUTER_STEP) )
	    {
		// Every now and then it's good to just randomly pick (i,j) to
		// optimise.  This helps to break loops and speed optimisation.
		// As justification, note that the "optimal" choice of (i,j) is
		// highly heuristic - i.e. it is in no way guaranteed to be the
		// best choice.

		i = svm_rand()%Ntrain;

		while ( onelist[i].isfixed )
		{
                    i = svm_rand()%Ntrain;
		}

		j = svm_rand()%Ntrain;

		while ( ( j == i ) || onelist[j].isfixed )
		{
                    j = svm_rand()%Ntrain;
		}

                locstep = trystep(i,j,Sij,beta,STbeta,betaGrad,STbetaGrad,alphai,alphaj,STalphai,STalphaj,alphaGradi,alphaGradj,STalphaGradi,STalphaGradj,Dbeta,DbetaGrad,Dalphai,Dalphaj,DalphaGradi,DalphaGradj,nonoptlisti,nonoptlistj,nonopttypei,nonopttypej,atbndlisti,atbndlistj,atbndtypei,atbndtypej,x,Gp,Gpn,Gn,gp,gn,hp,lb,ub,Gpsigma,mu,xi,Sh,Sl,n,isbetaopt);

		inneritcnt += locstep;
		stepdone += locstep;
                itcnt += locstep;

		if ( inneritcnt >= STEPSPEROUTERBLOCK )
		{
		    goto stepdone;
		}

		initcnt++;
	    }

	    // Run through list

	    onept = onefirstelm;

	    for ( ii = 0 ; ii < Nnf-1 ; ii++ )
	    {
		i = onept->i;

		if ( !(initcnt%RANDOM_OUTER_STEP) )
		{
		    // Every now and then it's good to just randomly pick j to
		    // optimise.  This is like the above one, except that it's
                    // based on the "optimal" first-choice i

		    j = svm_rand()%Ntrain;

		    while ( ( j == i ) || onelist[j].isfixed )
		    {
			j = svm_rand()%Ntrain;
		    }

		    locstep = trystep(i,j,Sij,beta,STbeta,betaGrad,STbetaGrad,alphai,alphaj,STalphai,STalphaj,alphaGradi,alphaGradj,STalphaGradi,STalphaGradj,Dbeta,DbetaGrad,Dalphai,Dalphaj,DalphaGradi,DalphaGradj,nonoptlisti,nonoptlistj,nonopttypei,nonopttypej,atbndlisti,atbndlistj,atbndtypei,atbndtypej,x,Gp,Gpn,Gn,gp,gn,hp,lb,ub,Gpsigma,mu,xi,Sh,Sl,n,isbetaopt);

		    inneritcnt += locstep;
		    stepdone += locstep;
		    itcnt += locstep;

		    if ( inneritcnt >= STEPSPEROUTERBLOCK )
		    {
			goto stepdone;
		    }

                    initcnt++;
		}

		// Calculate magnitude differences

		twopt = onept->nextmag;

//FIXME: phantomx
		for ( jj = ii+1 ; jj < Nnf ; jj++ )
		{
		    j = twopt->i;

                    emag0norm   = 0;
                    emag1norm   = 0;
		    emag2norm   = 0;
                    emaginfnorm = 0;

		    for ( s = 0 ; s < n ; s++ )
		    {
			if ( !Sl(i)(s) && !Sh(i)(s) && !Sl(j)(s) && !Sh(j)(s) )
			{
			    // i free, j free

                            emagsing = x(s)->alphaGrad(i) - x(s)->alphaGrad(j);
			}

			else if (  Sl(i)(s) && !Sh(i)(s) && !Sl(j)(s) && !Sh(j)(s) )
			{
			    // i at lower bound, j free

			    emagsing = x(s)->alphaGrad(i) - x(s)->alphaGrad(j);
			    emagsing = ( emagsing < 0 ) ? 0 : emagsing;
			}

			else if ( !Sl(i)(s) &&  Sh(i)(s) && !Sl(j)(s) && !Sh(j)(s) )
			{
			    // i at upper bound, j free

                            emagsing = x(s)->alphaGrad(i) - x(s)->alphaGrad(j);
			    emagsing = ( emagsing > 0 ) ? 0 : emagsing;
			}

			else if ( !Sl(i)(s) && !Sh(i)(s) &&  Sl(j)(s) && !Sh(j)(s) )
			{
			    // i free, j at lower bound

                            emagsing = x(s)->alphaGrad(i) - x(s)->alphaGrad(j);
			    emagsing = ( emagsing > 0 ) ? 0 : emagsing;
			}

			//else if (  Sl(i)(s) && !Sh(i)(s) &&  Sl(j)(s) && !Sh(j)(s) )
			//{
			//    // i at lower bound, j at lower bound
                        //
                        //    emagsing = 0;
			//}

			else if ( !Sl(i)(s) &&  Sh(i)(s) &&  Sl(j)(s) && !Sh(j)(s) )
			{
			    // i at upper bound, j at lower bound

                            emagsing = x(s)->alphaGrad(i) - x(s)->alphaGrad(j);
			    emagsing = ( emagsing > 0 ) ? 0 : emagsing;
			}

			else if ( !Sl(i)(s) && !Sh(i)(s) && !Sl(j)(s) &&  Sh(j)(s) )
			{
			    // i free, j at upper bound

                            emagsing = x(s)->alphaGrad(i) - x(s)->alphaGrad(j);
			    emagsing = ( emagsing < 0 ) ? 0 : emagsing;
			}

			else if (  Sl(i)(s) && !Sh(i)(s) && !Sl(j)(s) &&  Sh(j)(s) )
			{
			    // i at lower bound, j at upper bound

                            emagsing = x(s)->alphaGrad(i) - x(s)->alphaGrad(j);
			    emagsing = ( emagsing < 0 ) ? 0 : emagsing;
			}

			//else if ( !Sl(i)(s) &&  Sh(i)(s) && !Sl(j)(s) &&  Sh(j)(s) )
			//{
			//    // i at upper bound, j at upper bound
                        //
                        //    emagsing = 0;
			//}

			else
			{
                            emagsing = 0;
			}

			emag0norm    = ( ( fabs(emagsing) < emag0norm ) ? fabs(emagsing) : emag0norm );
			emag1norm   += fabs(emagsing);
			emag2norm   += (emagsing*emagsing);
			emaginfnorm  = ( ( fabs(emagsing) > emaginfnorm ) ? fabs(emagsing) : emaginfnorm );
		    }

// NB originally I used the line with sigma in it.  However in tests
// (train on the first 400 in the vehdata.txt datasets, C = 1, quad
// kernel) I found that the version with sigma took 27050 iterations
// and around 258 seconds whereas the version without (not commented
// out) only took 16735 iterations and around 146 seconds.

//		    twopt->ediff2norm   = emag2norm/(sigma(i,j)*sigma(i,j));
		    twopt->ediff0norm   = emag0norm;
		    twopt->ediff1norm   = emag1norm;
		    twopt->ediff2norm   = emag2norm;
		    twopt->ediffinfnorm = emaginfnorm;
		    twopt->nextdiff     = NULL;

                    twopt = twopt->nextmag;
		}

		// Sort secondary list in order of ||ei-ej||^2
		//
		// EDIT: trying to use 1-norm.  We would like to use the 0-norm
		// as this would guarantee a quick decrease, but the 0-norm is
		// almost always 0, so we use the 1-norm as a compromise between
                // the 0 and 2 norms

		twofirstelm = onept->nextmag;

		if ( ii+2 < Nnf )
		{
		    there = twofirstelm->nextmag;

		    for ( jj = ii+2 ; jj < Nnf ; jj++ )
		    {
			if ( there->ediff1norm >= twofirstelm->ediff1norm )
			{
			    there->nextdiff = twofirstelm;
			    twofirstelm = there;
			}

			else
			{
			    here = twofirstelm;

			    while ( ( here->nextdiff != NULL ) && ( (here->nextdiff)->ediff1norm > there->ediff1norm ) )
			    {
				here = here->nextdiff;
			    }

			    if ( here->nextdiff == NULL )
			    {
				here->nextdiff = there;
			    }

			    else
			    {
				there->nextdiff = here->nextdiff;
				here->nextdiff = there;
			    }
			}

			there = there->nextmag;
		    }
		}

		// Work through sorted secondary list, attempting a step for each possible.

		twopt = twofirstelm;

		for ( jj = ii+1 ; jj < Nnf ; jj++ )
		{
                    j = twopt->i;

		    locstep = trystep(i,j,Sij,beta,STbeta,betaGrad,STbetaGrad,alphai,alphaj,STalphai,STalphaj,alphaGradi,alphaGradj,STalphaGradi,STalphaGradj,Dbeta,DbetaGrad,Dalphai,Dalphaj,DalphaGradi,DalphaGradj,nonoptlisti,nonoptlistj,nonopttypei,nonopttypej,atbndlisti,atbndlistj,atbndtypei,atbndtypej,x,Gp,Gpn,Gn,gp,gn,hp,lb,ub,Gpsigma,mu,xi,Sh,Sl,n,isbetaopt);

		    inneritcnt += locstep;
		    stepdone += locstep;
		    itcnt += locstep;

		    if ( inneritcnt >= STEPSPEROUTERBLOCK )
		    {
			goto stepdone;
		    }

                    twopt = twopt->nextdiff;
		}

                onept = onept->nextmag;
	    }

            errstream() << "Unable to make progress for complete loop\n";

	    //exitcode = 1;

            goto getout;
	}

	// Quick and dirty exit point after

    stepdone:

        inneritcnt = 0;
    }

    if ( !isopt )
    {
        errstream() << "Loop counter overflow.\n";

        //exitcode = 2;
    }

getout:

    MEMDELARRAY(onelist);

    return 0;
}






int trystep(int i, int j, Vector<int> &Sij, Vector<double> &beta, Vector<double> &STbeta, Vector<double> &betaGrad, Vector<double> &STbetaGrad,
	    Vector<double> &alphai, Vector<double> &alphaj, Vector<double> &STalphai, Vector<double> &STalphaj,
	    Vector<double> &alphaGradi, Vector<double> &alphaGradj, Vector<double> &STalphaGradi, Vector<double> &STalphaGradj,
	    Vector<double> &Dbeta, Vector<double> &DbetaGrad, Vector<double> &Dalphai, Vector<double> &Dalphaj, Vector<double> &DalphaGradi, Vector<double> &DalphaGradj,
	    Vector<int> &nonoptlisti, Vector<int> &nonoptlistj, Vector<int> &nonopttypei, Vector<int> &nonopttypej, Vector<int> &atbndlisti, Vector<int> &atbndlistj,
            Vector<int> &atbndtypei, Vector<int> &atbndtypej, Vector<optState<double,double> *> &x, const Matrix<double> &Gp, const Matrix<double> &Gpn, const Matrix<double> &Gn,
	    Vector<Vector<double> > &gp, Vector<Vector<double> > &gn, const Vector<Vector<double> > &hp,
	    const Vector<Vector<double> > &lb, const Vector<Vector<double> > &ub, const Matrix<double> &Gpsigma, Vector<double> &mu, double xi,
	    Vector<Vector<int> > &Sh, Vector<Vector<int> > &Sl, int n, int isbetaopt)
{
    NiceAssert( i >= 0 );
    NiceAssert( j >= 0 );

    // some counter variables for the problem

    int s,t; // range 0 to n-1

    // Record of starting point

    for ( s = 0 ; s < n ; s++ )
    {
	STbeta("&",s)       = x(s)->beta(0);
	STbetaGrad("&",s)   = x(s)->betaGrad(0);
	STalphai("&",s)     = x(s)->alpha(i);
	STalphaj("&",s)     = x(s)->alpha(j);
	STalphaGradi("&",s) = x(s)->alphaGrad(i);
	STalphaGradj("&",s) = x(s)->alphaGrad(j);

	beta("&",s)       = x(s)->beta(0);
	betaGrad("&",s)   = x(s)->betaGrad(0);
	alphai("&",s)     = x(s)->alpha(i);
	alphaj("&",s)     = x(s)->alpha(j);
	alphaGradi("&",s) = x(s)->alphaGrad(i);
        alphaGradj("&",s) = x(s)->alphaGrad(j);
    }

    // Step variables

    double Dmui;
    double Dmuj;
    double Dxi;

    double STmui = mu(i);
    double STmuj = mu(j);
    double STxi  = xi;

    // Now for the inner training algorithm

    int isopt = 0; // optimality flag
    int itcnt = 0; // iteration counter

    int cnstrpi; // element indicator for   activating constraint: alpha[i][s] = h[i][s]
    int cnstrpj; // element indicator for   activating constraint: alpha[j][s] = h[j][s]
    int cnstrni; // element indicator for   activating constraint: alpha[i][s] = l[i][s]
    int cnstrnj; // element indicator for   activating constraint: alpha[j][s] = l[j][s]
    int freepi;  // element indicator for deactivating constraint: alpha[i][s] = h[i][s]
    int freepj;  // element indicator for deactivating constraint: alpha[j][s] = h[j][s]
    int freeni;  // element indicator for deactivating constraint: alpha[i][s] = l[i][s]
    int freenj;  // element indicator for deactivating constraint: alpha[j][s] = l[j][s]

    int nij; // count of variables actively constrained
    int nijold;

    double stepscale; // step scale and general magnitude variable
    double tempvar;

    int numnonopti;
    int numnonoptj;

    int numatbndi;
    int numatbndj;

    int selcnt = 0;

    double itcntbnd = powf(9.0,((float) (n-1))+1); // if itcnt exceeds this then cycling has occured.  This is an upper bound on the
    // number of active sets.  To be precise, each alpha projection can be in 1 of 3
    // states (lower bound, 0 or upper bound) and there are 2n such projections,
    // giving the maximum given in the above equation.

    if ( itcntbnd > BOUNDMAXITCNT )
    {
        itcntbnd = BOUNDMAXITCNT;
    }

    // Work out Sij

    nij = n;

    for ( s = 0 ; s < n ; s++ )
    {
	Sij("&",s) = Sh(i)(s) | Sl(i)(s) | Sh(j)(s) | Sl(j)(s);

	nij -= Sij(s);
    }

    while ( !isopt && ( itcnt <= itcntbnd ) )
    {
	itcnt++;

	// We need not calculate the step if we are fully constrained

	cnstrpi = -1;
	cnstrpj = -1;
	cnstrni = -1;
	cnstrnj = -1;

	if ( nij )
	{
	    // Calculate the step.
	    //
	    // Assumptions: \sum_s \beta_s = \sum_s \alpha_is = \sum_s \alpha_js = 0
            //              Gpsigma(i,j) > 0

	    if ( !isbetaopt )
	    {
		// DbetaGrad != 0

		Dmui = 0;
		Dmuj = 0;

                Dxi = 0;

		for ( s = 0 ; s < n ; s++ )
		{
		    if ( !Sij(s) )
		    {
			Dmui -= alphaGradi(s);
			Dmuj -= alphaGradj(s);

			Dxi -= betaGrad(s);
		    }
		}

		Dmui /= nij;
		Dmuj /= nij;

                Dxi /= nij;

		for ( s = 0 ; s < n ; s++ )
		{
		    if ( !Sij(s) )
		    {
			Dalphai("&",s) = (-1/(Gpsigma(i,j))) * (  (alphaGradi(s)+Dmui) - (alphaGradj(s)+Dmuj) + ((Gp(j,j)-Gp(j,i))*(betaGrad(s)+Dxi)) );
			Dalphaj("&",s) = (-1/(Gpsigma(i,j))) * ( -(alphaGradi(s)+Dmui) + (alphaGradj(s)+Dmuj) + ((Gp(i,i)-Gp(i,j))*(betaGrad(s)+Dxi)) );

			Dbeta("&",s) = (-1/(Gpsigma(i,j))) * ( ((Gp(j,j)-Gp(i,j))*(alphaGradi(s)+Dmui)) + ((Gp(i,i)-Gp(i,j))*(alphaGradj(s)+Dmuj)) - (((Gp(i,i)*Gp(j,j))-(Gp(i,j)*Gp(j,i)))*(betaGrad(s)+Dxi)) );

			DalphaGradi("&",s) = -alphaGradi(s);
			DalphaGradj("&",s) = -alphaGradj(s);

                        DbetaGrad("&",s) = -betaGrad(s);
		    }

		    else
		    {
			Dalphai("&",s) = 0;
			Dalphaj("&",s) = 0;

			Dbeta("&",s) = 0;

			DalphaGradi("&",s) = Dmui;
			DalphaGradj("&",s) = Dmuj;

                        DbetaGrad("&",s) = Dxi;
		    }
		}
	    }

	    else
	    {
		// DbetaGrad == 0

		Dmui = 0;
		Dmuj = 0;

                Dxi = 0;

		for ( s = 0 ; s < n ; s++ )
		{
		    if ( !Sij(s) )
		    {
			Dmui -= alphaGradi(s);
			Dmuj -= alphaGradj(s);
		    }
		}

		Dmui /= nij;
		Dmuj /= nij;

		for ( s = 0 ; s < n ; s++ )
		{
		    if ( !Sij(s) )
		    {
			Dalphai("&",s) = (-1/(Gpsigma(i,j))) * (  (alphaGradi(s)+Dmui) - (alphaGradj(s)+Dmuj) );
			Dalphaj("&",s) = -Dalphai("&",s);

			Dbeta("&",s) = (-1/(Gpsigma(i,j))) * ( ((Gp(j,j)-Gp(i,j))*(alphaGradi(s)+Dmui)) + ((Gp(i,i)-Gp(i,j))*(alphaGradj(s)+Dmuj)) );

			DalphaGradi("&",s) = -alphaGradi(s);
			DalphaGradj("&",s) = -alphaGradj(s);

                        DbetaGrad("&",s) = -betaGrad(s);
		    }

		    else
		    {
			Dalphai("&",s) = 0;
			Dalphaj("&",s) = 0;

			Dbeta("&",s) = 0;

			DalphaGradi("&",s) = Dmui;
			DalphaGradj("&",s) = Dmuj;

			DbetaGrad("&",s) = Dxi;
		    }
		}
	    }

	    // Calculate step scaling for feasibility

	    stepscale = 1;

	    for ( s = 0 ; s < n ; s++ )
	    {
		if ( !Sij(s) )
		{
		    if ( Dalphai(s) >= x(s)->zerotol() )
		    {
			if ( ( tempvar = (ub(s)(i)-alphai(s))/Dalphai(s) ) < stepscale )
			{
			    stepscale = tempvar;

			    cnstrpi = s;
			    cnstrpj = -1;
			    cnstrni = -1;
                            cnstrnj = -1;
			}
		    }

		    if ( Dalphaj(s) >= x(s)->zerotol() )
		    {
			if ( ( tempvar = (ub(s)(j)-alphaj(s))/Dalphaj(s) ) < stepscale )
			{
			    stepscale = tempvar;

			    cnstrpi = -1;
			    cnstrpj = s;
			    cnstrni = -1;
                            cnstrnj = -1;
			}
		    }

		    if ( Dalphai(s) <= -(x(s)->zerotol()) )
		    {
			if ( ( tempvar = (lb(s)(i)-alphai(s))/Dalphai(s) ) < stepscale )
			{
			    stepscale = tempvar;

			    cnstrpi = -1;
			    cnstrpj = -1;
			    cnstrni = s;
                            cnstrnj = -1;
			}
		    }

		    if ( Dalphaj(s) <= -(x(s)->zerotol()) )
		    {
			if ( ( tempvar = (lb(s)(j)-alphaj(s))/Dalphaj(s) ) < stepscale )
			{
			    stepscale = tempvar;

			    cnstrpi = -1;
			    cnstrpj = -1;
			    cnstrni = -1;
                            cnstrnj = s;
			}
		    }
		}
	    }

	    if ( stepscale <= 0 )
	    {
		stepscale = 0;
	    }

	    // Take the scaled step.

            mu("&",i) += stepscale*Dmui;
	    mu("&",j) += stepscale*Dmuj;

            xi += stepscale*Dxi;

	    beta.scaleAdd(stepscale,Dbeta);

	    alphai.scaleAdd(stepscale,Dalphai);
	    alphaj.scaleAdd(stepscale,Dalphaj);

	    alphaGradi.scaleAdd(stepscale,DalphaGradi);
	    alphaGradj.scaleAdd(stepscale,DalphaGradj);

	    if ( !isbetaopt )
	    {
                betaGrad.scaleAdd(stepscale,DbetaGrad);
	    }

	    // If the step was scaled then activate relevant constraints.

	    if ( cnstrpi != -1 )
	    {
		Sh("&",i)("&",cnstrpi) = 1;
		Sij("&",cnstrpi) = 1;
		nij--;
	    }

	    else if ( cnstrpj != -1 )
	    {
                Sh("&",j)("&",cnstrpj) = 1;
		Sij("&",cnstrpj) = 1;
		nij--;
	    }

	    else if ( cnstrni != -1 )
	    {
                Sl("&",i)("&",cnstrni) = 1;
		Sij("&",cnstrni) = 1;
		nij--;
	    }

	    else if ( cnstrnj != -1 )
	    {
                Sl("&",j)("&",cnstrnj) = 1;
		Sij("&",cnstrnj) = 1;
		nij--;
	    }
	}

        if ( ( cnstrpi == -1 ) && ( cnstrpj == -1 ) && ( cnstrni == -1 ) && ( cnstrnj == -1 ) )
	{
	    nijold = nij;

	    // Check gradients and if any are negative then free
	    // the most negative, thereby relaxing the relevant constraint.

            selcnt++;

	    while ( ( nijold == nij ) && !isopt )
	    {
		stepscale = STEPSCALETOL;

		freepi = -1;
		freepj = -1;
		freeni = -1;
		freenj = -1;

		numnonopti = 0;
		numnonoptj = 0;

		numatbndi = 0;
		numatbndj = 0;

		for ( s = 0 ; s < n ; s++ )
		{
		    if ( Sh(i)(s) )
		    {
			atbndlisti("&",numatbndi) = s;
			atbndtypei("&",numatbndi) = +1;

			numatbndi++;

			if ( alphaGradi(s) > STEPSCALETOL )
			{
			    nonoptlisti("&",numnonopti) = s;
			    nonopttypei("&",numnonopti) = +1;

			    numnonopti++;

			    if ( alphaGradi(s) > stepscale )
			    {
				stepscale = alphaGradi(s);

				freepi = s;
				freepj = -1;
				freeni = -1;
				freenj = -1;
			    }
			}
		    }

		    if ( Sh(j)(s) )
		    {
			atbndlistj("&",numatbndj) = s;
			atbndtypej("&",numatbndj) = +1;

			numatbndj++;

			if ( alphaGradj(s) > STEPSCALETOL )
			{
			    nonoptlistj("&",numnonoptj) = s;
			    nonopttypej("&",numnonoptj) = +1;

			    numnonoptj++;

			    if ( alphaGradj(s) > stepscale )
			    {
				stepscale = alphaGradj(s);

				freepi = -1;
				freepj = s;
				freeni = -1;
				freenj = -1;
			    }
			}
		    }

		    if ( Sl(i)(s) )
		    {
			atbndlisti("&",numatbndi) = s;
			atbndtypei("&",numatbndi) = -1;

			numatbndi++;

			if ( -alphaGradi(s) > STEPSCALETOL )
			{
			    nonoptlisti("&",numnonopti) = s;
			    nonopttypei("&",numnonopti) = -1;

			    numnonopti++;

			    if ( -alphaGradi(s) > stepscale )
			    {
				stepscale = -alphaGradi(s);

				freepi = -1;
				freepj = -1;
				freeni = s;
				freenj = -1;
			    }
			}
		    }

		    if ( Sl(j)(s) )
		    {
			atbndlistj("&",numatbndj) = s;
			atbndtypej("&",numatbndj) = -1;

			numatbndj++;

			if ( -alphaGradj(s) > STEPSCALETOL )
			{
			    nonoptlistj("&",numnonoptj) = s;
			    nonopttypej("&",numnonoptj) = -1;

			    numnonoptj++;

			    if ( -alphaGradj(s) > stepscale )
			    {
				stepscale = -alphaGradj(s);

				freepi = -1;
				freepj = -1;
				freeni = -1;
				freenj = s;
			    }
			}
		    }
		}

		// Randomise periodically to avoid cycling

		if ( ( numnonopti+numnonoptj ) && ( !(selcnt%RANDOM_INNER_STEP) ) )
		{
		    // Randomly choose a non-optimal active constraint to inactivate

		    if ( ( svm_rand()%2 || !numnonoptj ) && numnonopti )
		    {
			t = svm_rand()%numnonopti;

			if ( nonopttypei(t) == +1 )
			{
			    freepi = nonoptlisti(t);
			    freepj = -1;
			    freeni = -1;
			    freenj = -1;
			}

			else
			{
			    freepi = -1;
			    freepj = -1;
			    freeni = nonoptlisti(t);
			    freenj = -1;
			}
		    }

		    else
		    {
			t = svm_rand()%numnonoptj;

			if ( nonopttypej(t) == +1 )
			{
			    freepi = -1;
			    freepj = nonoptlistj(t);
			    freeni = -1;
			    freenj = -1;
			}

			else
			{
			    freepi = -1;
			    freepj = -1;
			    freeni = -1;
			    freenj = nonoptlistj(t);
			}
		    }
		}

		if ( ( numatbndi+numatbndj ) && ( !(selcnt%DOUBLERANDOM_INNER_STEP) ) )
		{
		    // Randomly choose any active constraint to inactivate

		    if ( ( svm_rand()%2 || !numatbndj ) && numatbndi )
		    {
			t = svm_rand()%numatbndi;

			if ( atbndtypei(t) == +1 )
			{
			    freepi = atbndlisti(t);
			    freepj = -1;
			    freeni = -1;
			    freenj = -1;
			}

			else
			{
			    freepi = -1;
			    freepj = -1;
			    freeni = atbndlisti(t);
			    freenj = -1;
			}
		    }

		    else
		    {
			t = svm_rand()%numatbndj;

			if ( atbndtypej(t) == +1 )
			{
			    freepi = -1;
			    freepj = atbndlistj(t);
			    freeni = -1;
			    freenj = -1;
			}

			else
			{
			    freepi = -1;
			    freepj = -1;
			    freeni = -1;
			    freenj = atbndlistj(t);
			}
		    }
		}

		if ( freepi != -1 )
		{
		    Sh("&",i)("&",freepi) = 0;

		    nij += Sij("&",freepi);
		    Sij("&",freepi) = Sh(i)(freepi) | Sl(i)(freepi) | Sh(j)(freepi) | Sl(j)(freepi);
		    nij -= Sij("&",freepi);
		}

		else if ( freepj != -1 )
		{
		    Sh("&",j)("&",freepj) = 0;

		    nij += Sij("&",freepj);
		    Sij("&",freepj) = Sh(i)(freepj) | Sl(i)(freepj) | Sh(j)(freepj) | Sl(j)(freepj);
		    nij -= Sij("&",freepj);
		}

		else if ( freeni != -1 )
		{
		    Sl("&",i)("&",freeni) = 0;

		    nij += Sij("&",freeni);
		    Sij("&",freeni) = Sh(i)(freeni) | Sl(i)(freeni) | Sh(j)(freeni) | Sl(j)(freeni);
		    nij -= Sij("&",freeni);
		}

		else if ( freenj != -1 )
		{
		    Sl("&",j)("&",freenj) = 0;

		    nij += Sij("&",freenj);
		    Sij("&",freenj) = Sh(i)(freenj) | Sl(i)(freenj) | Sh(j)(freenj) | Sl(j)(freenj);
		    nij -= Sij("&",freenj);
		}

		else
		{
		    isopt = 1;
		}
	    }
	}
    }

    // Update gradient cache

    for ( s = 0 ; s < n ; s++ )
    {
	x("&",s)->alphaStep(i,alphai(s)-STalphai(s),Gp,Gn,Gpn,gp(s),gn(s),hp(s),1);
	x("&",s)->alphaStep(j,alphaj(s)-STalphaj(s),Gp,Gn,Gpn,gp(s),gn(s),hp(s),1);

	x("&",s)->betaStep(0,beta(s)-STbeta(s),Gp,Gn,Gpn,gp(s),gn(s),hp(s),1);

        x("&",s)->factstepgp(Gp,Gn,Gpn,gp(s),mu(i)-STmui,gn(s),hp(s),i);
	gp("&",s)("&",i) += (mu(i)-STmui);

        x("&",s)->factstepgp(Gp,Gn,Gpn,gp(s),mu(j)-STmuj,gn(s),hp(s),j);
	gp("&",s)("&",j) += (mu(j)-STmuj);

	if ( !isbetaopt )
	{
	    x("&",s)->factstepgn(Gp,Gn,Gpn,gp(s),gn(s),xi-STxi,hp(s),i);
	    gn("&",s) += (xi-STxi);
	}
    }

    // Return 1 only if a "significant" step was taken

    if ( !isbetaopt )
    {
	stepscale = 0;

	for ( s = 0 ; s < n ; s++ )
	{
	    stepscale += fabs(STbetaGrad(s)-(x(s)->betaGrad(i)));

	    if ( stepscale > 2*n*STEPPT )
	    {
		return 1;
	    }
	}
    }

    stepscale = 0;

    for ( s = 0 ; s < n ; s++ )
    {
	stepscale += fabs(STalphai(s)-(x(s)->alpha(i)));
	stepscale += fabs(STalphaj(s)-(x(s)->alpha(j)));

	if ( stepscale > 2*n*STEPPT )
	{
	    return 1;
	}
    }

    return 0;
}

