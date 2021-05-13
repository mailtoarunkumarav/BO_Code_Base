
//
// Linear programming solver - quick and dirty hopdm front-end
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "linsolve.h"
#include "optlinstate.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 1000


#define FEEDBACKBLOCK                                      \
{                                                          \
            if ( !(++itcnt%FEEDBACK_CYCLE) )               \
            {                                              \
                if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )       \
                {                                          \
                    errstream() << "|\b";                  \
                }                                          \
                                                           \
                else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )  \
                {                                          \
                    errstream() << "/\b";                  \
                }                                          \
                                                           \
                else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )  \
                {                                          \
                    errstream() << "-\b";                  \
                }                                          \
                                                           \
                else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )  \
                {                                          \
                    errstream() << "\\\b";                 \
                }                                          \
            }                                              \
                                                           \
            if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )           \
            {                                              \
                errstream() << "=" << itcnt << "=  ";      \
            }                                              \
}


const char *etoDconv(char *flrep);
const char *Dtoeconv(char *flrep);
const char *makeVarName(char *dest, const char *basename, int i);
const char *makeCostString(char *dest, double costval);

int linsolvebase(svmvolatile int &killSwitch, optLinState &x,
                 const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g,
                 int maxitcntint, double xmtrtime);

int linsolvetrans(svmvolatile int &killSwitch, Vector<double> &alphabchi, const Vector<int> &contype,
                  const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g,
                  int maxitcntint, double xmtrtime);


int solve_linear_program(optState<double,double> &x,
        const Vector<double> &w, const Vector<double> &v,
        const Vector<double> &c, const Vector<double> &d,
        const Matrix<double> &Gp, const Matrix<double> &Gpn,
        const Matrix<double> &Qnp, const Matrix<double> &Qn,
        const Vector<double> &gp, const Vector<double> &hp,
        const Vector<double> &qn,
        const Vector<double> &lb, const Vector<double> &ub,
        const Matrix<double> &Gn, const Vector<double> &gn,
        int alpharestrictoverride, int Qconstype,
        svmvolatile int &killSwitch, int maxitcntint, double xmtrtime)
{
    int aresover = 0;

    if ( alpharestrictoverride == 5 )
    {
        alpharestrictoverride = 0;
        aresover = 1;
    }

    int i,j;

    NiceAssert( x.aN() == w.size()      );
    NiceAssert( x.aN() == c.size()      );
    NiceAssert( x.aN() == d.size()      );
    NiceAssert( x.aN() == Gp.numRows()  );
    NiceAssert( x.aN() == Gp.numCols()  );
    NiceAssert( x.aN() == Gpn.numRows() );
    NiceAssert( x.bN() == Gpn.numCols() );
    NiceAssert( x.aN() == gp.size()     );
    NiceAssert( x.aN() == hp.size()     );
    NiceAssert( x.aN() == lb.size()     );
    NiceAssert( x.aN() == ub.size()     );
    NiceAssert( ( x.aN() == Qnp.numCols() ) || !(Qn.numRows()) );
    NiceAssert( ( x.bN() == Qn.numCols()  ) || !(Qn.numRows()) );
    NiceAssert( ( Qn.numRows() == Qnp.numRows() ) || !(Qn.numRows()) );
    NiceAssert( ( Qn.numRows() == qn.size()     ) || !(Qn.numRows()) );



    // apiv: alpha pivot for dissected alpha
    // asgn: signs of alpha dissection
    // bpiv: beta pivot for dissected alpha
    // asgn: signs of beta dissection

    Vector<int> apiv;
    Vector<double> asgn;
    Vector<double> abnd;
    Vector<double> cd;
    Vector<int> bpiv;
    Vector<double> bsgn;

    if ( x.aN() )
    {
        for ( i = 0 ; i < x.aN() ; i++ )
        {
            if ( !aresover && ( ( x.alphaRestrict()(i) == 0 ) || ( x.alphaRestrict()(i) == 1 ) ) )
            {
                apiv.add(apiv.size());
                apiv("&",apiv.size()-1) = i;
                asgn.add(asgn.size());
                asgn("&",asgn.size()-1) = +1;
                cd.add(cd.size());
                cd("&",cd.size()-1) = c(i);
            }

            if ( ( ( x.alphaRestrict()(i) == 0 ) || ( x.alphaRestrict()(i) == 2 ) ) )
            {
                apiv.add(apiv.size());
                apiv("&",apiv.size()-1) = i;
                asgn.add(asgn.size());
                asgn("&",asgn.size()-1) = -1;
                cd.add(cd.size());
                cd("&",cd.size()-1) = d(i);
            }
        }
    }

    if ( x.bN() )
    {
        for ( i = 0 ; i < x.bN() ; i++ )
        {
            if ( ( x.betaRestrict()(i) == 0 ) || ( x.betaRestrict()(i) == 1 ) )
            {
                bpiv.add(bpiv.size());
                bpiv("&",bpiv.size()-1) = i;
                bsgn.add(bsgn.size());
                bsgn("&",bsgn.size()-1) = +1;
            }

            if ( ( x.betaRestrict()(i) == 0 ) || ( x.betaRestrict()(i) == 2 ) )
            {
                bpiv.add(bpiv.size());
                bpiv("&",bpiv.size()-1) = i;
                bsgn.add(bsgn.size());
                bsgn("&",bsgn.size()-1) = -1;
            }
        }
    }

//errstream() << "phantomx 0: " << Gp << "\n";
//errstream() << "phantomx 1: " << apiv << "\n";
//errstream() << "phantomx 1: " << asgn << "\n";
//errstream() << "phantomx 2: " << Gp(apiv,apiv) << "\n";
    // Problem is now a lot simpler: we are solving in terms of the new
    // variables:
    //
    // min xw'xalpha + xv'.xbeta + cd'.chi
    //
    // s.t. xGp .xalpha + xGpn.xbeta + xchi >= -xgp
    //      xQnp.xalpha + xQn .xbeta         = -qn
    //      xalpha >= 0
    //      xbeta  >= 0
    //
    // where:
    //
    // xw   = [ w_ {apiv_i} ]
    // xv   = [ v_ {bpiv_i} ]
    // xcd  = [ cd_{apiv_i} ]
    // xGp  = [ Gp _{apiv_i,apiv_j}.asgn_j ]
    // xGpn = [ Gpn_{apiv_i,bpiv_j}.bsgn_j ]
    // xQnp = [ Qnp_{     i,apiv_j}.asgn_j ]
    // xQn  = [ Qn _{     i,bpiv_j}.bsgn_j ]
    // xgp  = [ gp_{apiv_i}*asgn_i + hp_{apiv_i} ]

    int xaN = apiv.size();
    int xbN = bpiv.size();
    int xxN = qn.size();

    Vector<double> alpha(x.aN());
    Vector<double> beta(x.bN());
















// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

#ifdef USE_HOPDM
    int Asize = 0;
    int k;

    // Construct problem file

    Vector<double> xalpha(xaN);
    Vector<double> xbeta(xbN);

    {
        std::ofstream modfile(PROBFILE);

	char modbuffer[256];
	char mudboffer[256];
	char wodpuffer[256];
        char wumputter[256];
	int altbit;
        int ummm,well,yeah;

	modfile << "NAME          SVMLIN\n";
	modfile << "ROWS\n";
	modfile << " N  COST\n";

        /*
           ===============================================================

           Name all constraints (but note that variables are positive by
           default for mps format files)

           ===============================================================
        */

        for ( i = 0 ; i < xaN ; i++ )
	{
            modfile << " G  AC" << i << "\n";
	}

        for ( i = 0 ; i < xxN ; i++ )
	{
            if ( Qconstype == 0 )
            {
                modfile << " G  BC" << i << "\n";
            }

            else
            {
                modfile << " E  BC" << i << "\n";
            }
	}


        /*
           ===============================================================

           Name variables, put values in COST and also constraint matrix

           ===============================================================
        */

	modfile << "COLUMNS\n";

        // xapha first

        if ( xaN )
        {
            for ( i = 0 ; i < xaN ; i++ )
            {
              // Take into account additional restrictions on the sign of alpha

              if ( ( ( asgn(i) == +1 ) && ( ( alpharestrictoverride == 0 ) || ( alpharestrictoverride == 1 ) ) ) ||
                   ( ( asgn(i) == -1 ) && ( ( alpharestrictoverride == 0 ) || ( alpharestrictoverride == 2 ) ) )    )
              {
                // Behold the wonders of MPS formatting!
                //
                // Next comes the values in the various matrix parts.  This
                // must be in the format of a left-justified row name
                // followed by a right justified floating point number where
                // D (not e) acts as the exponent leader.  And it must be
                // exactly 22 letters long.
                //
                // Obviously.

                // FIXME: rewrite and modularise this spaghetti code

                modfile << makeVarName(mudboffer,"AA",i);
                modfile << makeCostString(wumputter,w(apiv(i)));

		altbit = 1;

                if ( xaN )
                {
                    for ( j = 0 ; j < xaN ; j++ )
                    {
                        // Gpelm is the relevant matrix value in xGp
                        // (note reversal of i and j)
                        double Gpelm = asgn(j)*Gp(apiv(j),apiv(i))*asgn(i);

                        if ( abs2(Gpelm) > ZERO_PT )
                        {
                            sprintf(modbuffer,"AC%d",j);
                            ummm = strlen(modbuffer);

                            sprintf(modbuffer,"%lf",Gpelm);
                            well = strlen(modbuffer);

                            yeah = 22-ummm-well;

                            for ( k = 0 ; k < yeah ; k++ )
                            {
                                wodpuffer[k] = ' ';
                            }

                            wodpuffer[k] = '\0';

                            sprintf(modbuffer,"AC%d%s%lf",j,wodpuffer,Gpelm);
                            etoDconv(modbuffer);

                            // MPS format says: precisely two per line

                            if ( altbit ) { altbit = 0; modfile << "   " << modbuffer << "\n"; }
                            else          { altbit = 1; modfile << mudboffer << modbuffer;     }

                            Asize++;
                        }
                    }
		}

                if ( xxN )
                {
                    for ( j = 0 ; j < xxN ; j++ )
                    {
                        // Gnpelm is the relevant matrix value in xGnp
                        // (note reversal of i and j)
                        double Qnpelm = Qnp(j,apiv(i))*asgn(i);

                        if ( aresover )
                        {
                            Qnpelm *= -1.0;
                        }

                        if ( abs2(Qnpelm) > ZERO_PT )
                        {
                            sprintf(modbuffer,"BC%d",j);
                            ummm = strlen(modbuffer);

                            sprintf(modbuffer,"%lf",Qnpelm);
                            well = strlen(modbuffer);

                            yeah = 22-ummm-well;

                            for ( k = 0 ; k < yeah ; k++ )
                            {
                                wodpuffer[k] = ' ';
                            }

                            wodpuffer[k] = '\0';

                            sprintf(modbuffer,"BC%d%s%lf",j,wodpuffer,Qnpelm);
                            etoDconv(modbuffer);

                            // MPS format says: precisely two per line

                            if ( altbit ) { altbit = 0; modfile << "   " << modbuffer << "\n"; }
                            else          { altbit = 1; modfile << mudboffer << modbuffer;     }

                            Asize++;
                        }
                    }
                }

		if ( altbit )
		{
                    // MPS format: always end with a newline

		    modfile << "\n";
		}
              }
            }
	}

        // xbeta next

        if ( xbN )
        {
            for ( i = 0 ; i < xbN ; i++ )
            {
                // See comments in xalpha section

                modfile << makeVarName(mudboffer,"BB",i);
                modfile << makeCostString(wumputter,v(bpiv(i)));

		altbit = 1;

                if ( xaN )
                {
                    for ( j = 0 ; j < xaN ; j++ )
                    {
                        // Gpnelm is the relevant matrix value in xGpn
                        // (note reversal of i and j)
                        double Gpnelm = asgn(j)*Gpn(apiv(j),bpiv(i))*bsgn(i);

                        if ( abs2(Gpnelm) > ZERO_PT )
                        {
                            sprintf(modbuffer,"AC%d",j);
                            ummm = strlen(modbuffer);

                            sprintf(modbuffer,"%lf",Gpnelm);
                            well = strlen(modbuffer);

                            yeah = 22-ummm-well;

                            for ( k = 0 ; k < yeah ; k++ )
                            {
                                wodpuffer[k] = ' ';
                            }

                            wodpuffer[k] = '\0';

                            sprintf(modbuffer,"AC%d%s%lf",j,wodpuffer,Gpnelm);
                            etoDconv(modbuffer);

                            // MPS format says: precisely two per line

                            if ( altbit ) { altbit = 0; modfile << "   " << modbuffer << "\n"; }
                            else          { altbit = 1; modfile << mudboffer << modbuffer;     }

                            Asize++;
                        }
                    }
		}

                if ( xxN )
                {
                    for ( j = 0 ; j < xxN ; j++ )
                    {
                        // Gnelm is the relevant matrix value in xGn
                        // (note reversal of i and j)
                        double Qnelm = Qn(j,bpiv(i))*bsgn(i);

                        if ( abs2(Qnelm) > ZERO_PT )
                        {
                            sprintf(modbuffer,"BC%d",j);
                            ummm = strlen(modbuffer);

                            sprintf(modbuffer,"%lf",Qnelm);
                            well = strlen(modbuffer);

                            yeah = 22-ummm-well;

                            for ( k = 0 ; k < yeah ; k++ )
                            {
                                wodpuffer[k] = ' ';
                            }

                            wodpuffer[k] = '\0';

                            sprintf(modbuffer,"BC%d%s%lf",j,wodpuffer,Qnelm);
                            etoDconv(modbuffer);

                            // MPS format says: precisely two per line

                            if ( altbit ) { altbit = 0; modfile << "   " << modbuffer << "\n"; }
                            else          { altbit = 1; modfile << mudboffer << modbuffer;     }

                            Asize++;
                        }
                    }
                }

		if ( altbit )
		{
                    // MPS format: always end with a newline

		    modfile << "\n";
		}
            }
	}

        // and lastly chi (slack variables)

        if ( xaN )
        {
            for ( i = 0 ; i < xaN ; i++ )
            {
                modfile << makeVarName(mudboffer,"XI",i);
                modfile << makeCostString(wumputter,cd(i));

                altbit = 1;

                j = i;
                {
                    sprintf(modbuffer,"AC%d",j);
                    ummm = strlen(modbuffer);
                    well = 1;

                    yeah = 22-ummm-well;

                    for ( k = 0 ; k < yeah ; k++ )
                    {
                        wodpuffer[k] = ' ';
                    }

                    wodpuffer[k] = '\0';

                    sprintf(modbuffer,"AC%d%s1",j,wodpuffer);
                    etoDconv(modbuffer);

                    if ( altbit ) { altbit = 0; modfile << "   " << modbuffer << "\n"; }
                    else          { altbit = 1; modfile << mudboffer << modbuffer;     }

                    Asize++;
                }

                if ( altbit )
                {
                    modfile << "\n";
                }
            }
	}

        /*
           ===============================================================

           Define the actual constraint values here

           ===============================================================
        */

	modfile << "RHS\n";

        altbit = 0;

        if ( xaN )
        {
            for ( i = 0 ; i < xaN ; i++ )
            {
                sprintf(modbuffer,"AC%d",i);
                ummm = strlen(modbuffer);

                sprintf(modbuffer,"%lf",-gp(apiv(i))*asgn(i));
                well = strlen(modbuffer);

                yeah = 22-ummm-well;

                for ( k = 0 ; k < yeah ; k++ )
                {
                    wodpuffer[k] = ' ';
                }

                wodpuffer[k] = '\0';

                sprintf(modbuffer,"AC%d%s%lf",i,wodpuffer,-gp(apiv(i))*asgn(i));
                etoDconv(modbuffer);

                if ( altbit ) { altbit = 0; modfile << "   " << modbuffer << "\n";    }
                else          { altbit = 1; modfile << "    B         " << modbuffer; }
            }
        }

        if ( xxN )
        {
            for ( i = 0 ; i < xxN ; i++ )
            {
                sprintf(modbuffer,"BC%d",i);
                ummm = strlen(modbuffer);

                sprintf(modbuffer,"%lf",-qn(i));
                well = strlen(modbuffer);

                yeah = 22-ummm-well;

                for ( k = 0 ; k < yeah ; k++ )
                {
                    wodpuffer[k] = ' ';
                }

                wodpuffer[k] = '\0';

                sprintf(modbuffer,"BC%d%s%lf",i,wodpuffer,-qn(i));
                etoDconv(modbuffer);

                if ( altbit ) { altbit = 0; modfile << "   " << modbuffer << "\n";    }
                else          { altbit = 1; modfile << "    B         " << modbuffer; }
            }
        }

        if ( altbit )
        {
            modfile << "\n";
        }

        modfile << "ENDATA\n";

	modfile.close();
    }

    // Call optimiser routine

//    system("del log");
//    system("del sol");
    int ignore_res = system("./hopdm");

    (void) ignore_res;

    // Grab result from optimiser file

    xalpha = 0.0;
    xbeta  = 0.0;

    std::ifstream solfile(SOLFILE);

    char tempbuf[256];
    char tumpbef[256];

    solfile >> tempbuf;

    while ( !(solfile.eof()) )
    {
        strcpy(tumpbef,tempbuf);

        tumpbef[2] = '\0';

        if ( strcmp(tumpbef,"AA") == 0 )
        {
            i = atoi(tempbuf+2);

            solfile >> tempbuf;
            solfile >> tempbuf;
            solfile >> tempbuf;

            xalpha("&",i) = atof(Dtoeconv(tempbuf));
        }

        else if ( strcmp(tumpbef,"BB") == 0 )
        {
            i = atoi(tempbuf+2);

            solfile >> tempbuf;
            solfile >> tempbuf;
            solfile >> tempbuf;

            xbeta("&",i) = atof(Dtoeconv(tempbuf));
        }

        solfile >> tempbuf;
    }

    solfile.close();

    // Undo pivotting and retrieve result

    alpha = 0.0;
    beta  = 0.0;

    for ( i = 0 ; i < apiv.size() ; i++ )
    {
        alpha("&",apiv(i)) += xalpha(i)*asgn(i);
    }

    for ( i = 0 ; i < bpiv.size() ; i++ )
    {
        beta("&",bpiv(i)) += xbeta(i)*bsgn(i);
    }

//errstream() << "phantomxy 0: " << alpha << "\n";
//errstream() << "phantomxy 1: " << beta << "\n";
#endif





























// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

#ifndef USE_HOPDM
    // Problem is now a lot simpler: we are solving in terms of the new
    // variables:
    //
    // min xw'xalpha + xv'.xbeta + cd'.chi
    //
    // s.t. xGp .xalpha + xGpn.xbeta + xchi >= -xgp
    //      xQnp.xalpha + xQn .xbeta         = -qn
    //      xalpha >= 0
    //      xbeta  >= 0
    //      xchi   >= 0
    //
    // where:
    //
    // xw   = [ w_ {apiv_i} ]
    // xv   = [ v_ {bpiv_i} ]
    // xcd  = [ cd_{apiv_i} ]
    // xGp  = [ Gp _{apiv_i,apiv_j}.asgn_j ]
    // xGpn = [ Gpn_{apiv_i,bpiv_j}.bsgn_j ]
    // xQnp = [ Qnp_{     i,apiv_j}.asgn_j ]
    // xQn  = [ Qn _{     i,bpiv_j}.bsgn_j ]
    // xgp  = [ gp_{apiv_i}*asgn_i + hp_{apiv_i} ]
    //
    //
    // For the purposes of MEX and matlab:
    //
    // min xf'xx
    //
    // A.xx >= b
    // Aeq.xx = beq
    //
    // A   = [ xGp xGpn I ]
    // b   = [ -xgp ]
    // Aeq = [ xQnp xQn 0 ]
    // beq = [ -xqn ]
    // lb  = [ 0 ]
    // ub  = [ bignum ]
    //
    //      [ xalpha ]
    // xx = [ xbeta  ]
    //      [ xxi    ]
    //
    //      [ xw ]
    // xf = [ xv ]
    //      [ cd ]
    //
    //      [ xalpha_current  ]
    // x0 = [ xbeta_current   ]
    //      [ xxi (calculate) ]
    //
    // We can further simplify this to:
    //
    // min xf'.xx
    //
    // A.xx + b ?= 0
    //
    // A   = [ xGp   xGpn I ]    ?= -> >=
    //       [ xQnp  xQn  0 ]    ?= -> ==
    //
    // b   = [ xgp ]
    //       [ xqn ]
    //
    //      [ xalpha ]
    // xx = [ xbeta  ]
    //      [ xxi    ]
    //
    //      [ xw     ]
    // xf = [ xv     ]
    //      [ cd     ]
    //
    //      [ xalpha_current   ]
    // x0 = [ xbeta_current    ]
    //      [ xxi (calculate)  ]
    //
    // Up to non-negativity:
    //
    // [ xxi  ] = -( xgp + [  xGp   xGpn ] [ xalpha_current ] )
    //
    //
    // Pre-solver:
    //
    // min xfalt'.xxalt
    //
    // Aalt.xxalt + balt == 0
    //
    // Aalt = [ xGp   xGpn I 0 ]  ?= -> >=
    //        [ xQnp  xQn  0 I ]  ?= -> ==
    //
    // balt = [ xgp ]
    //        [ xqn ]
    //
    //         [ xalpha ]
    // xxalt = [ xbeta  ]
    //         [ xchi   ]
    //         [ xsup   ]
    //
    //         [ 0 ]
    // xfalt = [ 0 ]
    //         [ 0 ]
    //         [ 1 ]
    //
    //
    //
    // Hard-margin version of pre-solver:
    //
    // If cd >= HARD_MARGIN_LINCUT_VAL then pre-solver uses:
    //
    //         [ 0 ]
    // xfalt = [ 0 ]
    //         [ 1 ]
    //         [ 1 ]
    //
    // to enforce xi = 0.  Then xi part is removed prior to optimisation.

    // Construct problem variables (initially in pre-solver form)

    int itcnt = 0;

    errstream() << "Setup problem...";

    Vector<double> xx(        xaN+xbN+xaN+xxN);
    Vector<double> xf(        xaN+xbN+xaN+xxN);
    Vector<double> xfalt(     xaN+xbN+xaN+xxN);
    Matrix<double> AA(xaN+xxN,xaN+xbN+xaN+xxN);
    Vector<double> bb(xaN+xxN                );
    Vector<int> constrtype(xaN+xxN);

    int hardmargincondmet = 1;

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retMatrix<double> tmpma;

    // Setup problem definition
    {
        xx = 0.0;
        xf = 0.0;
        AA = 0.0;
        bb = 0.0;

        xfalt = 0.0;

        constrtype = 1;

        int ibase = 0;

        if ( xaN )
        {
            // alpha parts

            ibase = 0;

            for ( i = 0 ; i < xaN ; i++ )
            {
                if ( ( ( asgn(i) == +1 ) && ( ( alpharestrictoverride == 0 ) || ( alpharestrictoverride == 1 ) ) ) ||
                     ( ( asgn(i) == -1 ) && ( ( alpharestrictoverride == 0 ) || ( alpharestrictoverride == 2 ) ) )    )
                {
                    xf("&",ibase+i) = w(apiv(i));
                    xfalt("&",ibase+i) = 0.0;

                    if ( xaN )
                    {
                        for ( j = 0 ; j < xaN ; j++ )
                        {
                            AA("&",j,ibase+i) = asgn(j)*Gp(apiv(j),apiv(i))*asgn(i);
                            FEEDBACKBLOCK;
                        }
                    }

                    if ( xxN )
                    {
                        for ( j = 0 ; j < xxN ; j++ )
                        {
                            AA("&",xaN+j,ibase+i) = Qnp(j,apiv(i))*asgn(i) * ( ( aresover ) ? -1.0 : 1.0 );
                            FEEDBACKBLOCK;
                        }
                    }
                }
            }
	}

        if ( xbN )
        {
            // beta parts

            ibase = xaN;

            for ( i = 0 ; i < xbN ; i++ )
            {
                xf("&",ibase+i) = v(bpiv(i));
                xfalt("&",ibase+i) = 0.0;

                if ( xaN )
                {
                    for ( j = 0 ; j < xaN ; j++ )
                    {
                        AA("&",j,ibase+i) = asgn(j)*Gpn(apiv(j),bpiv(i))*bsgn(i);
                        FEEDBACKBLOCK;
                    }
		}

                if ( xxN )
                {
                    for ( j = 0 ; j < xxN ; j++ )
                    {
                        AA("&",xaN+j,ibase+i) = Qn(j,bpiv(i))*bsgn(i);
                        FEEDBACKBLOCK;
                    }
                }
            }
	}

        if ( xaN )
        {
            // chi parts

            ibase = xaN+xbN;

            for ( i = 0 ; i < xaN ; i++ )
            {
                xf("&",ibase+i) = cd(i);
                xfalt("&",ibase+i) = 1.0;

                if ( cd(i) < HARD_MARGIN_LINCUT_VAL )
                {
                    hardmargincondmet = 0;
                }

                if ( xaN )
                {
                    for ( j = 0 ; j < xaN ; j++ )
                    {
                        AA("&",j,ibase+i) = ( i == j ) ? 1.0 : 0.0;
                        FEEDBACKBLOCK;
                    }
                }

                if ( xxN )
                {
                    for ( j = 0 ; j < xxN ; j++ )
                    {
                        AA("&",xaN+j,ibase+i) = 0.0;
                        FEEDBACKBLOCK;
                    }
                }
            }
	}

        if ( xxN )
        {
            // xxN parts

            ibase = xaN+xbN+xaN;

            for ( i = 0 ; i < xxN ; i++ )
            {
                xf("&",ibase+i) = 0.0;
                xfalt("&",ibase+i) = 1.0;

                if ( xaN )
                {
                    for ( j = 0 ; j < xaN ; j++ )
                    {
                        AA("&",j,ibase+i) = 0.0;
                        FEEDBACKBLOCK;
                    }
                }

                if ( xxN )
                {
                    for ( j = 0 ; j < xxN ; j++ )
                    {
                        AA("&",xaN+j,ibase+i) = ( i == j ) ? 1.0 : 0.0;
                        FEEDBACKBLOCK;
                    }
                }
            }
	}

        if ( xaN )
        {
            for ( i = 0 ; i < xaN ; i++ )
            {
                bb("&",i) = gp(apiv(i))*asgn(i);
                constrtype("&",i) = 1;
                FEEDBACKBLOCK;
            }
	}

        if ( xxN )
        {
            for ( i = 0 ; i < xxN ; i++ )
            {
                bb("&",i+xaN) = qn(i);
                constrtype("&",i) = Qconstype ? 1 : 0;
                FEEDBACKBLOCK;
            }
	}
    }

    errstream() << "Grab state...";

    // Recall current state and fix non-negativity

    xx("&",0,1,xaN-1,tmpva) = (x.alpha())(apiv,tmpvb);
    xx("&",0,1,xaN-1,tmpva) *= asgn;

    xx("&",xaN,1,xaN+xbN-1,tmpva) = (x.beta())(bpiv,tmpvb);
    xx("&",xaN,1,xaN+xbN-1,tmpva) *= bsgn;

    if ( xaN+xbN )
    {
        for ( i = 0 ; i < xaN+xbN ; i++ )
        {
            if ( xx(i) < 0.0 )
            {
                xx("&",i) = 0.0;
            }
        }
    }

    errstream() << "Precalculate slacks...";

    // Work out slacks

    mult(xx("&",xaN+xbN,1,xaN+xbN+xaN+xxN-1,tmpva),AA(zeroint(),1,xaN+xxN-1,zeroint(),1,xaN+xbN-1,tmpma),xx(zeroint(),1,xaN+xbN-1,tmpvb));
    xx("&",xaN+xbN,1,xaN+xbN+xaN+xxN-1,tmpva) += bb(zeroint(),1,xaN+xxN-1,tmpvb);
    xx("&",xaN+xbN,1,xaN+xbN+xaN+xxN-1,tmpva).negate();

    if ( xaN )
    {
        for ( i = 0 ; i < xaN ; i++ )
        {
            if ( xx(xaN+xbN+i) < 0.0 )
            {
                xx("&",xaN+xbN+i) = 0.0;
            }
        }
    }

    int need_presolve = 0;

    if ( xaN && hardmargincondmet )
    {
        for ( i = 0 ; i < xaN ; i++ )
        {
            if ( abs2(xx(xaN+xbN+i)) >= DEFAULT_ZTOL )
            {
                need_presolve = 1;
            }
        }
    }

    if ( xxN )
    {
        for ( i = 0 ; i < xxN ; i++ )
        {
            if ( abs2(xx(xaN+xbN+xaN+i)) >= DEFAULT_ZTOL )
            {
                need_presolve = 1;
            }
        }
    }

    // Run presolver if needed

    errstream() << "Presolve...";

    if ( need_presolve )
    {
errstream() << "pre;;;";
        // Note use of xfalt here.  If soft-margin optimisation set then
        // need to mod xfalt to:
        //
        //         [ 0 ]
        // xfalt = [ 0 ]
        //         [ 0 ]
        //         [ 1 ]
        //
        // In any case aim is to enforce xi = chi = 0

        if ( !hardmargincondmet )
        {
            xfalt("&",xaN+xbN,1,xaN+xbN+xaN-1,tmpva).zero();
        }

        linsolvetrans(killSwitch,xx,constrtype,AA,xfalt,bb,maxitcntint,xmtrtime);
    }

    // Call optimiser routine

    errstream() << "Resize...";

    if ( hardmargincondmet )
    {
        // Need to remove both xi and chi

        xx.resize(xaN+xbN);
        xf.resize(xaN+xbN);
        AA.resize(xaN+xxN,xaN+xbN);
        bb.resize(xaN+xxN);
    }

    else
    {
        // Need only remove chi

        xx.resize(xaN+xbN+xaN);
        xf.resize(xaN+xbN+xaN);
        AA.resize(xaN+xxN,xaN+xbN+xaN);
        bb.resize(xaN+xxN);
    }

    errstream() << "Optimise...";

    linsolvetrans(killSwitch,xx,constrtype,AA,xf,bb,maxitcntint,xmtrtime);

    // Undo pivotting and retrieve result

    alpha = 0.0;
    beta  = 0.0;

    for ( i = 0 ; i < apiv.size() ; i++ )
    {
        alpha("&",apiv(i)) += xx(i)*asgn(i);
    }

    for ( i = 0 ; i < bpiv.size() ; i++ )
    {
        beta("&",bpiv(i)) += xx(i+xaN)*bsgn(i);
    }

//errstream() << "phantomxy -1: " << xx << "\n";
//errstream() << "phantomxy 2: " << alpha << "\n";
//errstream() << "phantomxy 3: " << beta << "\n";
#endif





















// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

    // Overwrite solution in x

    if ( xaN )
    {
        x.setAlpha(alpha,Gp,Gp,Gn,Gpn,gp,gn,hp,lb,ub);
    }

    if ( xbN )
    {
        x.setBeta(beta,Gp,Gp,Gn,Gpn,gp,gn,hp);
    }

    // exit

    return 0;
}





// Convert C standard float string to fortran standard float string

const char *etoDconv(char *flrep)
{
    int k = (int) strlen(flrep)-1;

    while ( flrep[k] != ' ' )
    {
        if ( flrep[k] == 'e' )
        {
            flrep[k] = 'D';
        }

        k--;
    }

    return flrep;
}

// Converts fortran standard float string to C standard float string

const char *Dtoeconv(char *flrep)
{
    int j,k;

    j = (int) strlen(flrep);

    for ( k = 0 ; k < j ; k++ )
    {
        if ( ( flrep[k] == 'E' ) || ( flrep[k] == 'D' ) )
        {
            flrep[k] = 'e';
        }
    }

    return flrep;
}



// Make standard-width MPS cost string

const char *makeCostString(char *dest, double costval)
{
    int yeah;
    int well;
    int k;
    char spacebuffer[22];

    sprintf(dest,"%lf",costval);
    well = (int) strlen(dest);

    yeah = 22-4-well;

    for ( k = 0 ; k < yeah ; k++ )
    {
        spacebuffer[k] = ' ';
    }

    spacebuffer[k] = '\0';

    sprintf(dest,"COST%s%lf",spacebuffer,costval);

    return etoDconv(dest);
}

const char *makeVarName(char *dest, const char *basename, int i)
{
    int yeah;
    int well;
    int k;
    char spacebuffer[22];

    sprintf(dest,"    %s%d",basename,i);
    well = (int) strlen(dest);

    yeah = 14-well;

    NiceAssert( yeah > 0 );

    for ( k = 0 ; k < yeah ; k++ )
    {
        spacebuffer[k] = ' ';
    }

    spacebuffer[k] = '\0';

    sprintf(dest,"    %s%d%s",basename,i,spacebuffer);

    return dest;
}














int linsolvetrans(svmvolatile int &killSwitch, Vector<double> &alphabchi, const Vector<int> &contype,
                  const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g,
                  int maxitcntint, double xmtrtime)
{
    NiceAssert( G.numRows() == g.size() );
    NiceAssert( G.numCols() == c.size() );
    NiceAssert( alphabchi.size() == c.size() );

    optLinState x;

//errstream() << "Solver start: G = " << G << "\n";
//errstream() << "Solver start: g = " << g << "\n";
//errstream() << "Solver start: c = " << c << "\n";
//errstream() << "Solver start: x = " << alphabchi << "\n";
//errstream() << "Solver start: ct = " << contype << "\n";

    // Populate x

errstream() << "Populate vars...";
    if ( G.numCols() )
    {
        int i;

        for ( i = 0 ; i < G.numCols() ; i++ )
        {
            x.addAlpha(i,1);
        }
    }

errstream() << "Populate rows...";
    if ( G.numRows() )
    {
        int i;

        for ( i = 0 ; i < G.numRows() ; i++ )
        {
            x.addRow(i,contype(i));
        }
    }

errstream() << "Warm-start...";
    x.setAlpha(alphabchi,G,c,g);

    // Call optimiser

errstream() << "Core entry...";
    int res = linsolvebase(killSwitch,x,G,c,g,maxitcntint,xmtrtime);

    // Extract result

errstream() << "Result retrieval...";
    if ( alphabchi.size() )
    {
        int i;

        for ( i = 0 ; i < alphabchi.size() ; i++ )
        {
            alphabchi("&",i) = x.alpha()(i);
        }
    }
//errstream() << "Solver end: x = " << alphabchi << "\n";

    return res;
}



int linsolvebase(svmvolatile int &killSwitch, optLinState &x,
                 const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g,
                 int maxitcntint, double xmtrtime)
{
    int res = 0;

    double maxitcnt = maxitcntint;
    double *uservars[] = { &maxitcnt, &xmtrtime, NULL };
    const char *varnames[] = { "itercount", "traintime", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", NULL };

    if ( x.aN() )
    {
        retVector<double> tmpa;
        retVector<double> tmpb;

	int isopt = 0;

        Vector<double> stepAlpha(x.aN());
        Vector<double> stepe(x.eN());

	Vector<double> combStepAlpha(x.alpha());
        Vector<double> combStepe(x.e());
        Vector<int> FnF;
        Vector<int> startPivAlphaF;

        double scale;
	double gradmag;

	int alphaFIndex;
        int alphaZIndex;
        int eFIndex;
        int eZIndex;

        time_used start_time = TIMECALL;
        time_used curr_time = start_time;
        unsigned long long itcnt = 0;
        int timeout = 0;

        while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcnt ) || !maxitcnt ) && !timeout )
	{
            isopt = 1;

            // Calculate step

//errstream() << "Calculating step:\n";
            x.calcStep(stepAlpha,stepe,G,c,g);
//errstream() << "Alpha step = " << stepAlpha(x.pivAlphaF()) << "\n";
//errstream() << "e step = " << stepe(x.pivRowZ()) << "\n";
//errstream() << "x = " << x << "\n\n\n\n";

            if ( x.aNF() && ( norm2(stepAlpha(x.pivAlphaF(),tmpa)) >= x.zerotol() ) )
            {
                isopt = 0;

                // Scale step

//errstream() << "Scaling step:\n";
                x.scaleFStep(scale,alphaFIndex,eZIndex,stepAlpha(x.pivAlphaF(),tmpa),stepe(x.pivRowZ(),tmpb),G,c,g);
//errstream() << "Scale = " << scale << "\n";
//errstream() << "alpha index = " << alphaFIndex << "\n";
//errstream() << "e index = " << eZIndex << "\n";
//errstream() << "x = " << x << "\n\n\n\n";

                stepAlpha("&",x.pivAlphaF(),tmpa) *= scale;
                stepe("&",x.pivRowZ(),tmpa) *= scale;

                // Take scaled step

//errstream() << "Taking scaled step:\n";
//errstream() << "alpha step = " << stepAlpha(x.pivAlphaF()) << "\n";
//errstream() << "e step = " << stepe(x.pivRowZ()) << "\n";
                x.stepFGeneral(stepAlpha(x.pivAlphaF(),tmpa),stepe(x.pivRowZ(),tmpb),G,c,g);
//errstream() << "x = " << x << "\n\n\n\n";

                // Fix active sets

                if ( alphaFIndex >= 0 )
                {
//errstream() << "Constraining alpha: " << alphaFIndex << " (absolute " << x.pivAlphaF()(alphaFIndex) << ")\n";
                    x.modAlphaFtoZ(alphaFIndex,G,c,g);
//errstream() << "x = " << x << "\n\n\n\n";
                }

                else
                {
                    NiceAssert( eZIndex >= 0 );

//errstream() << "Activating row constraint: " << eZIndex << " (absolute " << x.pivRowZ()(eZIndex) << ")\n";
                    x.modRowZtoF(eZIndex,G,c,g);
//errstream() << "x = " << x << "\n\n\n\n";
                }
            }

            else
            {
                // Find least optimal constraint

//errstream() << "Calculating maximal non-optimality\n";
                if ( !(x.maxGradNonOpt(alphaZIndex,eFIndex,gradmag,G,c,g)) )
                {
                    if ( alphaZIndex >= 0 )
                    {
                        isopt = 0;

//errstream() << "Freeing alpha " << alphaZIndex << " (absolute " << x.pivAlphaZ()(alphaZIndex) << ")\n";
                        x.modAlphaZtoF(alphaZIndex,G,c,g);
//errstream() << "x = " << x << "\n\n\n\n";
                    }

                    else
                    {
                        NiceAssert( eFIndex >= 0 );

                        isopt = 0;

//errstream() << "Inactivating row constraint " << eFIndex << " (absolute " << x.pivRowF()(eFIndex) << ")\n";
                        x.modRowFtoZ(eFIndex,G,c,g);
//errstream() << "x = " << x << "\n\n\n\n";
                    }
                }
            }

            FEEDBACKBLOCK;

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
                timeout = kbquitdet("linear optimisation",uservars,varnames,vardescr);
            }
	}

	if ( !isopt )
	{
	    res = 1;
	}
    }

    return res;
}


