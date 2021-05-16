
//
// Line minimiser
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include <math.h>
#include <iostream>
#include "zerocross.h"

#define QSGN(a)  ( (a) > 0 ) ? +1 : ( ( (a) < 0 ) ? -1 : 0 )


double findZeroCross(double etaij, double eij, double epsiloni, double epsilonj, double thetai, double normalphai, double absalphai, double thetaj, double normalphaj, double absalphaj, double iota, double smin, double smax, int maxitcnt, double kappa, int isidiscont, int isjdiscont, double CCNi, double CCNj, double t, double ztol)
{
    (void) iota;

    double sL = smin;
    double sH = smax;
    double sM = sH; // NB this is deliberate as the upper bound is checked first
    int itcnt = 0;
    double Qgrad;
    //double absalphai = sqrt(normalphai);
    //double absalphaj = sqrt(normalphaj);
    //int isidiscont = ( thetai < -absalphai+iota );
    //int isjdiscont = ( thetaj >  absalphaj-iota );
    double qi,qj;
    double ri,rj;
    double tti,ttj;
    double Qorig,Qs;
    int resbetter = 0;

    //Qorig = eij + (epsiloni*absalphai) + (epsilonj*absalphaj);

    Qorig = (epsiloni*absalphai) - ((1/t)*log(CCNi-absalphai))
          + (epsilonj*absalphaj) - ((1/t)*log(CCNj-absalphaj));

    // Check end of range

    tti = sqrt((sM*sM)+(2*thetai*sM)+normalphai);
    ttj = sqrt((sM*sM)-(2*thetaj*sM)+normalphaj);

    ri = epsiloni + (1/(t*(CCNi-tti)));
    rj = epsilonj + (1/(t*(CCNj-ttj)));

    if ( isidiscont )
    {
        qi = QSGN(sM-absalphai);
    }

    else
    {
        qi = (sM+thetai)/tti;
    }

    if ( isjdiscont )
    {
        qj = QSGN(sM-absalphaj);
    }

    else
    {
        qj = (sM-thetaj)/ttj;
    }

    Qgrad = (etaij*sM) + eij + (ri*qi) + (rj*qj);

    if ( Qgrad <= kappa )
    {
        sM = sH;
    }

    else
    {
        // Find gradient zero-crossing 
    
        while ( ( ( itcnt < maxitcnt ) || !resbetter ) && ( sH-sL > ztol/2 ) )
        {
            sM = (sL+sH)/2;

            tti = sqrt((sM*sM)+(2*thetai*sM)+normalphai);
            ttj = sqrt((sM*sM)-(2*thetaj*sM)+normalphaj);

            ri = epsiloni + (1/(t*(CCNi-tti)));
            rj = epsilonj + (1/(t*(CCNj-ttj)));

            if ( isidiscont )
            {
                qi = QSGN(sM-absalphai);
            }

            else
            {
                qi = (sM+thetai)/tti;
            }

            if ( isjdiscont )
            {
                qj = QSGN(sM-absalphaj);
            }

            else
            {
                qj = (sM-thetaj)/ttj;
            }

            Qgrad = (etaij*sM) + eij + (ri*qi) + (rj*qj);
//errstream() << "phantomxxjj " << Qgrad << "\n";

            if ( Qgrad > kappa )
            {
                sH = sM;
            }

            else if ( Qgrad < -kappa )
            {
                sL = sM;
            }

            else
            {
                break;
            }

            if ( !resbetter )
            {
                tti = sqrt((sH*sH)+(2*thetai*sH)+normalphai);
                ttj = sqrt((sH*sH)-(2*thetaj*sH)+normalphaj);

                Qs = (etaij*sH*sH/2) + (eij*sH)
                   + (epsiloni*tti) - ((1/t)*log(CCNi-tti))
                   + (epsilonj*ttj) - ((1/t)*log(CCNj-ttj));

                if ( Qs < Qorig )
                {
                    resbetter = 1;
                }
            }

            itcnt++;
        }
    }

//errstream() << "phantomxxtr\n";
    return sM;
}

