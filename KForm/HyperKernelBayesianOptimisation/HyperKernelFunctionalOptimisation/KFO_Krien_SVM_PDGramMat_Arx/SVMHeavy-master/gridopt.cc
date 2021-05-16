
//
// Grid-based Optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "gridopt.h"

#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 1000
#define LOGZTOL 1e-8

int GridOptions::optim(int dim,
                       Vector<gentype> &xres,
                       gentype &fres,
                       int &ires,
                       Vector<Vector<gentype> > &allxres,
                       Vector<gentype> &allfres,
                       Vector<gentype> &allfresmod,
                       Vector<gentype> &allsupres,
                       Vector<double> &sscore,
                       const Vector<gentype> &xmin,
                       const Vector<gentype> &xmax,
                       void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                       void *fnarg,
                       svmvolatile int &killSwitch)
{
    int isalone = 0;
    int issolo  = 0;

    if ( locdistMode.size() == 0 )
    {
        isalone = 1;

        locdistMode.resize(dim);
        locvarsType.resize(dim);

        locdistMode = zeroint();
        locvarsType = 1;
    }

    if ( numPts.size() == 0 )
    {
        issolo = 1;

        numPts.resize(dim);

        numPts = DEFAULT_NUMOPTS;
    }

    NiceAssert( dim > 0 );
    NiceAssert( xmin.size() == dim );
    NiceAssert( xmax.size() == dim );
    NiceAssert( numPts.size() == dim );
    NiceAssert( locdistMode.size() == dim );
    NiceAssert( locvarsType.size() == dim );
    NiceAssert( numPts >= zeroint() );
    NiceAssert( locdistMode >= zeroint() );
    NiceAssert( locdistMode <= 4 );
    NiceAssert( locvarsType >= zeroint() );
    NiceAssert( locvarsType <= 1 );

    int res = 0;
    int i,j,k;

    double dres = 0;
    double s,t;

    xres.resize(dim);
    fres = 0.0;
    ires = 0;

    allxres.resize(0);
    allfres.resize(0);
    allfresmod.resize(0);
    allsupres.resize(0);

    // Enumerate testpoints on all axis

    Vector<Vector<gentype> > gridmarks(dim);
    Vector<int> isAxisRandom(locvarsType); // This will be zero for all exis except the random ones

    isAxisRandom = zeroint();

    for ( i = 0 ; i < dim ; ++i )
    {
        gridmarks("&",i).resize(numPts(i));

        isAxisRandom("&",i) = locvarsType(i) ? ( ( locdistMode(i) == 4 ) ? 1 : 2 ) : 0;

        t = ( numPts(i) == 1 ) ? 0.5 : 0;

        for ( j = 0 ; j < numPts(i) ; ++j,(t+=(1/((double) numPts(i)-1))) )
        {
            randfill(s);
                
            gridmarks("&",i)("&",j) = ( locdistMode(i) == 3 ) ? s : t;
        }

        // For randomised we sort ascending to make inter-grid jumps
        // as small as possible

        if ( locdistMode(i) == 3 )
        {
            int kmin = 0;

            for ( j = 0 ; j < gridmarks(i).size()-1 ; ++j )
            {
                kmin = j;

                for ( k = j+1 ; k < gridmarks(i).size() ; ++k )
                {
                    if ( gridmarks(i)(k) < gridmarks(i)(kmin) )
                    {
                        kmin = k;
                    }
                }

                gridmarks("&",i).squareswap(j,kmin);
            }
        }
    }

    // Grab centre vector.  This will act as a secondary selector if we
    // have multiple results that have the same target (assuming centrality
    // is good, which it generally is).

    Vector<gentype> xcentre(dim);

    for ( i = 0 ; i < dim ; ++i )
    {
        xcentre("&",i) = gridmarks(i)((gridmarks(i).size()-1)/2);
    }

    // Main loop.

    Vector<int> ii(dim);
    Vector<int> idir(dim);

    Vector<gentype> locxres(dim);
    gentype locfres;
    gentype locsupres;
    double locdres;

    // Initialise loop counters.  Direction is kept to minimise jump between
    // loop rows - basically we only want to change one variable at a time,
    // so count goes from 0 to N-1, then N-1 to 0, and so on

    for ( i = 0 ; i < dim ; ++i )
    {
        ii("&",i) = 0;
        idir("&",i) = +1;
    }

    int isopt = 0;
    int isfirst = 1;

    j = 0;

    int timeout = 0;
    double *uservars[] = { &maxtraintime, &zoomFact, NULL };
    const char *varnames[] = { "traintime", "zoomFact", NULL };
    const char *vardescr[] = { "Maximum training time (seconds, 0 for unlimited)", "Zoom factor", NULL };

    while ( !killSwitch && !isopt && !timeout )
    {
        // Find position

        for ( i = 0 ; i < dim ; ++i )
        {
            locxres("&",i) = gridmarks(i)(ii(i));

            if ( isAxisRandom(i) == 1 )
            {
                randfill(s);
                
                locxres("&",i) = s;
            }
        }

        // Evaluate function

        locfres.force_int() = 0;

        (*fn)(locfres,locxres,fnarg);

        locsupres.force_vector().resize(2);
        locsupres("&",zeroint()) = nullgentype();

        if ( locfres.isValSet() )
        {
            // Variance is also here, so grab it

            gentype mures((locfres.all())(zeroint()));
            gentype varres((locfres.all())(1));

            locfres = mures;
            locsupres("&",zeroint()) = varres;
        }

        // Find distance to centre

        locxres -= xcentre;
        locdres = norm2(locxres);
        locxres += xcentre;

        locsupres("&",1) = locdres;

        // Record if required

        if ( 1 )
        {
            allxres.append(allxres.size(),locxres);
            allfres.append(allfres.size(),locfres);
            allfresmod.append(allfresmod.size(),locfres);
            allsupres.append(allsupres.size(),locsupres);
            sscore.append(sscore.size(),1.0);
        }

        // Test optimality

        if ( isfirst || ( locfres < fres ) || ( ( locfres == fres ) && ( locdres < dres ) ) )
        {
            isfirst = 0;

            xres = locxres;
            fres = locfres;
            dres = locdres;
            ires = j;
        }

        // Terminate if hardmin/max found, increment otherwise

        if ( (double) locfres <= hardmin )
        {
            killSwitch = 1;
        }

        else if ( (double) locfres >= hardmax )
        {
            killSwitch = 1;
        }

        else
        {
            isopt = 1;

            for ( i = 0 ; i < dim ; ++i )
            {
                if ( idir(i) == +1 )
                {
                    if ( ii(i) == gridmarks(i).size()-1 )
                    {
                        idir("&",i) = -1;
                    }

                    else
                    {
                        ++(ii("&",i));
                        isopt = 0;
                        break;
                    }
                }

                else
                {
                    if ( ii(i) == 0 )
                    {
                        idir("&",i) = +1;
                    }

                    else
                    {
                        --(ii("&",i));
                        isopt = 0;
                        break;
                    }
                }
            }

            if ( !timeout )
            {
                timeout = kbquitdet("Grid optimisation",uservars,varnames,vardescr);
            }
        }

        j++;
    }

    NiceAssert( !isfirst );

    // If requested zoom, recurse and incorporate new results

    if ( numZooms && !killSwitch )
    {
        Vector<gentype> subxres;
        gentype subfres(zeroint());
        int subires = 0;
        int submres = 0;
        int submuInd = 0;
        int subsigInd = 0;
        int subsrcmodInd = 0;
        int subdiffmodInd = 0;
        Vector<Vector<gentype> > suballxres;
        Vector<gentype> suballfres;
        Vector<gentype> suballfresmod;
        Vector<gentype> suballsupres;
        Vector<double> subsscore;
        Vector<gentype> subxmin(xmin);
        Vector<gentype> subxmax(xmax);

        double width;
        double centre;

        // First up we need to zoom and 

        for ( i = 0 ; i < dim ; ++i )
        {
            // real zoom always

            width  = ( (double) xmax(i) ) - ( (double) xmin(i) );
            centre = (double) xres(i);
            width *= zoomFact;

            subxmin("&",i) = centre - width/2;
            subxmax("&",i) = centre + width/2;

            // Need to make sure we stay within the feasible region

            subxmin("&",i) = ( subxmin(i) > xmin(i) ) ? subxmin(i) : xmin(i);
            subxmax("&",i) = ( subxmax(i) < xmax(i) ) ? subxmax(i) : xmax(i);
        }

        // Recurse

        GridOptions locgopts = *this;
        --(locgopts.numZooms);

        Vector<int> &subdistMode(locdistMode);
        Vector<int> &subvarsType(locvarsType);

        subdistMode = zeroint(); // So we don't get twice-applied whatever scales.
        subvarsType = 1;         // So we only round to integer once.

        Vector<gentype> subxignore;             // We only really care about "raw" here
        Vector<Vector<gentype> > suballxignore; // We only really care about "raw" here

        gentype dummymeanfres, dummyvarfres;
        gentype dummymeanires, dummyvarires;
        gentype dummymeantres, dummyvartres;
        gentype dummymeanTres, dummyvarTres;

        Vector<gentype> dummymeanallfres, dummyvarallfres;
        Vector<gentype> dummymeanallmres, dummyvarallmres;

        res |= locgopts.optim(dim,subxignore,subxres,subfres,subires,submres,submuInd,subsigInd,subsrcmodInd,subdiffmodInd,suballxignore,suballxres,suballfres,suballfresmod,suballsupres,subsscore,
                              subxmin,subxmax,subdistMode,subvarsType,fn,fnarg,killSwitch,1,
                              dummymeanfres,dummyvarfres,dummymeanires,dummyvarires,dummymeantres,dummyvartres,dummymeanTres,dummyvarTres,dummymeanallfres,dummyvarallfres,dummymeanallmres,dummyvarallmres);

        // Incorporate results

        if ( subfres < fres )
        {
            xres   = subxres;
            fres   = subfres;
            ires   = subires+allfres.size();
//            mres   = submres;
//            muInd  = submuInd;
//            sigInd = subsigInd;
//            srcmodInd = subsrcmodInd;
//            diffmodInd = subdiffmodInd;
        }

        if ( 1 )
        {
            allxres.append(allxres.size(),suballxres);
            allfres.append(allfres.size(),suballfres);
            allfresmod.append(allfresmod.size(),suballfresmod);
            allsupres.append(allsupres.size(),suballsupres);
            sscore.append(sscore.size(),subsscore);
        }
    }

    if ( isalone )
    {
        locdistMode.resize(zeroint());
        locvarsType.resize(zeroint());
    }

    if ( issolo )
    {
        numPts.resize(zeroint());
    }

    return res;
}

