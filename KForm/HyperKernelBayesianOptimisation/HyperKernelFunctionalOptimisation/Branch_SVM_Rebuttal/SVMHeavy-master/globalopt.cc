
//
// Global optimisation options base-class
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "globalopt.h"
#include "hyper_base.h"

int GlobalOptions::optim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &Xres,
                      gentype &fres,
                      int &ires,
                      int &mInd,
                      int &muInd,
                      int &sigInd,
                      int &srcmodInd,
                      int &diffmodInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allXres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allmres,
                      Vector<gentype> &allsres,
                      Vector<double>  &s_score,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch,
                      unsigned int numReps, 
                      gentype &meanfres, gentype &varfres,
                      gentype &meanires, gentype &varires,
                      gentype &meantres, gentype &vartres,
                      gentype &meanTres, gentype &varTres,
                      Vector<gentype> &meanallfres, Vector<gentype> &varallfres,
                      Vector<gentype> &meanallmres, Vector<gentype> &varallmres)
    {
        // numReps:  1 = normal behaviour
        //          >1 = do multiple runs.  Everything returned is for the final run, except
        //               allfres and allmres are now vectors, where the first element is
        //               an average and the second element is a variance.

        int res = 0;

        if ( numReps == 1 )
        {
            int k;

            gentype nullval;
            nullval.makeNull();

            res = realOptim(dim,xres,Xres,fres,ires,mInd,muInd,sigInd,srcmodInd,diffmodInd,allxres,allXres,allfres,allmres,allsres,s_score,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch);

            // Sort allmres to be strictly decreasing!

            for ( k = 0 ; k < allmres.size() ; k++ )
            {
                if ( k )
                {
                    allmres("&",k) = ( allmres(k) < allmres(k-1) ) ? allmres(k) : allmres(k-1);
                }
            }

            meanfres = fres; varfres = nullval;
            meanires = ires; varires = nullval;

            meanallfres = allfres; varallfres.resize(allfres.size()) = nullval;
            meanallmres = allmres; varallmres.resize(allmres.size()) = nullval;

            findfirstLT(meantres,allmres,softmin); vartres = nullval;
            findfirstLT(meanTres,allmres,hardmin); vartres = nullval;
        }

        else if ( numReps > 1 )
        {
            unsigned int j;
            int k;

            Vector<gentype> vecfres(numReps);
            Vector<gentype> vecires(numReps);

            Vector<Vector<gentype> > vecallfres(numReps);
            Vector<Vector<gentype> > vecallmres(numReps);

            Vector<gentype> vectres(numReps);
            Vector<gentype> vecTres(numReps);

            int maxxlen = 0;

            for ( j = 0 ; j < numReps ; j++ )
            {
                GlobalOptions *locopt = makeDup();

                allxres.resize(0);
                allXres.resize(0);
                allsres.resize(0);
                s_score.resize(0);

                res += (*locopt).realOptim(dim,xres,Xres,vecfres("&",j),vecires("&",j).force_int(),mInd,muInd,sigInd,srcmodInd,diffmodInd,allxres,allXres,vecallfres("&",j),vecallmres("&",j),allsres,s_score,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch);

                if ( j == numReps-1 )
                {
                    fres = vecfres(j);
                    ires = (int) vecires(j);

                    allfres = vecallfres(j);
                    allmres = vecallmres(j);
                }

                // Sort vecallmres(j) to be strictly decreasing!

                for ( k = 0 ; k < vecallfres(j).size() ; k++ )
                {
                    if ( k )
                    {
                        vecallmres("&",j)("&",k) = ( vecallmres(j)(k) < vecallmres(j)(k-1) ) ? vecallmres(j)(k) : vecallmres(j)(k-1);
                    }
                }

                maxxlen = ( vecallmres(j).size() > maxxlen ) ? vecallmres(j).size() : maxxlen; 

                findfirstLT(vectres("&",j),vecallmres(j),softmin);
                findfirstLT(vecTres("&",j),vecallmres(j),hardmin);

                MEMDEL(locopt);
            }

            retVector<gentype> tmpva;

            for ( j = 0 ; j < numReps ; j++ )
            {
                if ( ( k = vecallmres(j).size() ) < maxxlen )
                {
                    (vecallfres("&",j).resize(maxxlen))("&",k,1,maxxlen-1,tmpva) = vecallfres(j)(k-1);
                    (vecallmres("&",j).resize(maxxlen))("&",k,1,maxxlen-1,tmpva) = vecallmres(j)(k-1);
                }
            }

            calcmeanvar(meanfres,varfres,vecfres);
            calcmeanvar(meanires,varires,vecires);
            calcmeanvar(meantres,vartres,vectres);
            calcmeanvar(meanTres,varTres,vecTres);

            calcmeanvar(meanallfres,varallfres,vecallfres);
            calcmeanvar(meanallmres,varallmres,vecallmres);
        }

        return res;
    }

int GlobalOptions::realOptim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &Xres,
                      gentype &fres,
                      int &ires,
                      int &mInd,
                      int &muInd,
                      int &sigInd,
                      int &srcmodInd,
                      int &diffmodInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allXres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allmres,
                      Vector<gentype> &allsres,
                      Vector<double>  &s_score,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch)
    {
        (void) muInd;
        (void) sigInd;
        (void) srcmodInd;
        (void) diffmodInd;

        locfnarg = fnarg;

        // dim: dimension of problem
        // xres: x at minimum
        // Xres: xres in format seen by specific method, without projection or distortion
        // fres: f(x) at minimum
        // ires: index of optimal solution in allres
        // allxres: vector of all vectors evaluated
        // allXres: vector of all vectors evaluated, in raw (unprojected/undistorted) format
        // allfres: vector of all f(x) evaluations
        // xmin: lower bound on variables
        // xmax: upper bound on variables
        // distMode: 0 = linear distribution of points
        //           1 = logarithmic distribution of points
        //           2 = anti-logarithmic distribution of points
        //           3 = uniform random distribution of points (grid only)
        //           4 = inverse logistic distribution of points
        //           5 = REMBO
        // varsType: 0 = integer
        //           1 = double
        // fn: function being optimised
        // fnarg: argument passed to fn
        // killSwitch: set nonzero to force stop

        NiceAssert( dim >= 0 );
        NiceAssert( optdefed() );
        NiceAssert( distMode.size() == dim );
        NiceAssert( varsType.size() == dim );

        addSubDim = 0;

        // For schedules Bernstein projection, effdim is the (reduced) dimension for this "round"

        int effdim = dim;

        stopearly  = 0;
        firsttest  = 1;
        spOverride = 0;
        bestyet    = 0.0;

        // The following variables are used to normalise the axis to [0,1].
        // Their exact meaning depends on distMode.

        Vector<gentype> allfakexmin(dim);
        Vector<gentype> allfakexmax(dim);

        xwidth = calcabc(dim,allfakexmin,allfakexmax,a,b,c,xmin,xmax,distMode);

        Vector<gentype> fakexmin(allfakexmin);
        Vector<gentype> fakexmax(allfakexmax);

        locdistMode = distMode;
        locvarsType = varsType;

        // These need to be passed back

        Vector<int> dummyMLnumbers(6);
        Vector<int> &MLnumbers = MLdefined ? (*((Vector<int> *) ((void **) fnarg)[15])) : dummyMLnumbers;
        Vector<int> locMLnumbers(6); // See mlinter definition of MLnumbers

        locMLnumbers = -1;

        // overfn setup

        Vector<gentype> xmod(dim);

        void *overfnargs[16]; // It is very important that overfnargs[15] is defined here!

        overfnargs[0] = (void *)  fnarg;
        overfnargs[1] = (void *) &dim;
        overfnargs[2] = (void *) &xmod;
        overfnargs[3] = (void *)  this;
        overfnargs[4] = (void *) &allxres;
        overfnargs[5] = (void *) &penterm;
        overfnargs[15] = (void *) &locMLnumbers; // if the optimiser gets recursed (which only happens in gridOpt) then this is required to ensure thet MLnumbers is defined and doesn't overlap!

        overfnargs[6] = (void *) fn;

        // Projection templates

        MLnumbers("&",3) = -1;

        if ( isProjection == 0 )
        {
            // No vector/function projection, so do nothing

            ;
        }

        else if ( isProjection == 1 )
        {
            // Create random subspace - vector-valued subspace

            ML_Mutable *temprandDirtemplate;

            MEMNEW(temprandDirtemplate,ML_Mutable);
            (*temprandDirtemplate).setMLTypeClean(defrandDirtemplateVec.type());

            (*temprandDirtemplate).getML() = defrandDirtemplateVec;
            randDirtemplateInd = regML(temprandDirtemplate,fnarg,0);

            randDirtemplate = &((*temprandDirtemplate).getML());

            MLnumbers("&",3) = randDirtemplateInd;
        }

        else if ( isProjection == 2 )
        {
            // Create random subspace - function-valued subspace

            ML_Mutable *temprandDirtemplate;

            MEMNEW(temprandDirtemplate,ML_Mutable);
            (*temprandDirtemplate).setMLTypeClean(defrandDirtemplateFnGP.type());

            (*temprandDirtemplate).getML() = defrandDirtemplateFnGP;
            randDirtemplateInd = regML(temprandDirtemplate,fnarg,1);

            randDirtemplate = &((*temprandDirtemplate).getML());
//errstream() << "phantomx 0: randDirtemplate = " << *randDirtemplate << "\n";

            MLnumbers("&",3) = randDirtemplateInd;

            // Need to fill in x distributions.

            Vector<int> sampleInd(fnDim);
            Vector<gentype> sampleDist(fnDim);

            int j;

            for ( j = 0 ; j < fnDim ; j++ )
            {
                sampleInd("&",j)  = j;
                sampleDist("&",j) = "urand(x,y)";

                SparseVector<SparseVector<gentype> > xy;

                xy("&",zeroint())("&",zeroint()) = 0.0;
                xy("&",zeroint())("&",1)         = 1.0;

                sampleDist("&",j).substitute(xy);
            }

            initModelDistr(sampleInd,sampleDist);
        }

        else if ( ( isProjection == 3 ) || ( isProjection == 4 ) )
        {
            // Create non-random subspace - function-valued subspace, Bernstein basis

            ML_Mutable *temprandDirtemplate;

            MEMNEW(temprandDirtemplate,ML_Mutable);
            (*temprandDirtemplate).setMLTypeClean(defrandDirtemplateFnBern.type());

            (*temprandDirtemplate).getML() = defrandDirtemplateFnBern;
            randDirtemplateInd = regML(temprandDirtemplate,fnarg,1);

            randDirtemplate = &((*temprandDirtemplate).getML());

            MLnumbers("&",3) = randDirtemplateInd;

            if ( isProjection == 4 )
            {
                effdim = bernstart;
            }
        }

        else if ( isProjection == 5 )
        {
//RKHSFIXME
            // Projection via RKHSVector, so do nothing

            ;
        }

        // Optimisation loop
        //
        // substatus: 0 = this is the first optimisation
        //            1 = not the first optimisation, but the "best" subspace wasn't changed by the most recent optimisation
        //            2 = not the first optimisation, "best" subspace given by subsequent x vector

        int res = 0;

        int ii,jj;
        int substatus = 0;

        // We can't afford to call prelim operations between optimisation and
        // making a subspace, we we pre-make it here, then after each optim

        // Trigger preliminary "setup" operations

        ii = 0;
        {
            gentype trigval;
            Vector<gentype> xdummy;
            SparseVector<gentype> altlocxres;

            trigval.force_int() = (-ii*10000)-1; // -1,-10001,-20001,-30001,...
            (*fn)(trigval,xdummy,fnarg);

            MLnumbers("&",2) = makeSubspace(dim,fnarg,substatus,altlocxres); // until we actually have a projection, at least

            trigval.force_int() = (-ii*10000)-2; // -2,-10002,-20002,-30002,...
            (*fn)(trigval,xdummy,fnarg);
        }

        int locrandReproject = randReproject;

        for ( ii = 0 ; ii <= locrandReproject ; ii++ )
        {
            outstream() << "---\n";

            Vector<gentype> locxres;
            Vector<gentype> locXres;
            SparseVector<gentype> altlocxres;
            SparseVector<gentype> altlocXres;
            gentype locfres;
            int locires;

            Vector<Vector<gentype> > locallxres;
            Vector<Vector<gentype> > locallXres;
            Vector<gentype> locallfres;
            Vector<gentype> locallmres;
            Vector<gentype> locallsres;
            Vector<double>  locs_score;

            stopearly = 0;

            if ( isProjection == 4 )
            {
                spOverride = ii; // only want one initial random batch

                // Update bernstein schedule (note that if isProjection == 3 this intentionally does nothing, and that effdim is bounded by dim)

                if ( ii && ( effdim < dim ) )
                {
                    effdim++;
                }

                // Fake-out unused dimensions for scheduled Bernstein optimisation

                if ( effdim )
                {
                    for ( jj = 0 ; jj < effdim ; jj++ )
                    {
                        fakexmin("&",jj) = allfakexmin(jj);
                        fakexmax("&",jj) = allfakexmax(jj);
                    }
                }

                if ( effdim < dim )
                {
                    for ( jj = effdim ; jj < dim ; jj++ )
                    {
                        fakexmin("&",jj) = allfakexmin(jj);
                        fakexmax("&",jj) = allfakexmin(jj);
                    }
                }
            }

            else
            {
                effdim = dim;
            }

            berndim = effdim;

            // Optimisation

            res = optim(dim,locXres,locfres,locires,locallXres,locallfres,locallmres,locallsres,locs_score,
                        fakexmin,fakexmax,overfn,overfnargs,killSwitch);

            // Update subspace if/as required

            if ( substatus == 0 )
            {
                // Grab both x result *and* put subspace in optimal state

                convertx(dim,locxres,locXres);
                model_convertx(altlocxres,( altlocXres = locXres ));

                // Just completed the first optimisation, so the solution must be the best so far by definition

                xres = locxres;
                Xres = locXres;
                fres = locfres;
                ires = locires+(allfres.size());
                mInd = projOpInd;

                allxres.append(allxres.size(),locallxres);
                allXres.append(allXres.size(),locallXres);
                allfres.append(allfres.size(),locallfres);
                allmres.append(allmres.size(),locallmres);
                allsres.append(allsres.size(),locallsres);
                s_score.append(s_score.size(),locs_score);

                substatus = 2;
            }

            else if ( locfres < fres )
            {
                // Grab both x result *and* put subspace in optimal state

                convertx(dim,locxres,locXres);
                model_convertx(altlocxres,( altlocXres = locXres ));

                // Not first optimisation, but there is a new best solution

                xres = locxres;
                Xres = locXres;
                fres = locfres;
                ires = locires+(allfres.size());
                mInd = projOpInd;

                allxres.append(allxres.size(),locallxres);
                allXres.append(allXres.size(),locallXres);
                allfres.append(allfres.size(),locallfres);
                allmres.append(allmres.size(),locallmres);
                allsres.append(allsres.size(),locallsres);
                s_score.append(s_score.size(),locs_score);

                substatus = 2;
            }

            else
            {
                // Set "x" to zero so that it just reverts to previous best

                locXres = zerogentype();

                // Grab both x result *and* put subspace in optimal state

                convertx(dim,locxres,locXres,1); // Note that useOrigin is set here, so locXres is not actually used!
                model_convertx(altlocxres,( altlocXres = locXres ),1); // Note that useOrigin is set here, so locXres is not actually used!

                // Not first optimisation, no improvement found (but zero set anyhow)

                //xres = locxres;
                //Xres = locXres;
                //fres = locfres;
                //ires = locires+(allfres.size());
                //mInd = projOpInd;

                allxres.append(allxres.size(),locallxres);
                allXres.append(allXres.size(),locallXres);
                allfres.append(allfres.size(),locallfres);
                allmres.append(allmres.size(),locallmres);
                allsres.append(allsres.size(),locallsres);
                s_score.append(s_score.size(),locs_score);

                substatus = 1;
            }

            // Increase iterations if stopearly set (bernstein scheduled stopearly)

            if ( stopearly )
            {
                NiceAssert( isProjection == 4 );

                locrandReproject++;
            }

            // Construct random subspace if required (but not for scheduled Bernstein)

            if ( ii < locrandReproject )
            {
                if ( isProjection != 4 )
                {
                    MLnumbers("&",2) = makeSubspace(dim,fnarg,substatus,altlocxres);
                }

                // Trigger intermediate (hyperparameter tuning) operations

                gentype trigval;
                Vector<gentype> xdummy;

                trigval.force_int() = (-(ii+1)*10000)-2; // -2,-10002,-20002,-30002,...
                (*fn)(trigval,xdummy,fnarg);
            }
        }

        // And we're done

        a.resize(0);
        b.resize(0);
        c.resize(0);

        return res;
    }

int GlobalOptions::makeSubspace(int dim, void *fnarg, int substatus, const SparseVector<gentype> &locxres)
    {
        // substatus zero on first call, 1 or 2 otherwise

        // Note we don't delete any projections blocks as they may be
        // required in the functional optimisation case where return
        // is of form fnB(projOpInd,500,x).  We do register them though
        // so they will be deleted on exit.

        if ( substatus )
        {
            // This function is used by smboopt to clear models
            // if the model is built on x rather than model_convertx(x)

            model_clear();
        }

        if ( isProjection == 0 )
        {
            // No projection, nothing to see here

            addSubDim = 0;

            projOpInd = -1;
        }

        else if ( ( isProjection == 1 ) || ( isProjection == 2 ) )
        {
            // Projection onto either random vectors or random functions.  Both of
            // these are distributions, so we can clone and finalise them to get a
            // random projection.

            // If not first run then we add a dimension (zero point), permanently
            // weighted to 1, that represents the current best solution.

            addSubDim = substatus ? 1 : 0;

            errstream() << "Constructing random subspace... (" << dim+addSubDim << ") ";

            subDef.resize(dim+addSubDim);
            subDefInd.resize(dim+addSubDim);

            int i,j;

            ML_Mutable *randDir;

            errstream() << "creating random direction prototypes... ";

            int rdim = addSubDim ? dim+1 : dim;

            for ( i = 0 ; i < dim ; i++ )
            {
//errstream() << "phantomxaadd 6: "  << *randDirtemplate << "\n";
                if ( ( i < dim-1 ) || !( !addSubDim && includeConst ) )
                {
                    MEMNEW(randDir,ML_Mutable);
                    (*randDir).setMLTypeClean((*randDirtemplate).type());

                    (*randDir).getML() = (*randDirtemplate);
                    subDefInd("&",i) = regML(randDir,fnarg,2);

                    subDef("&",i) = &((*randDir).getML());

                    if ( substatus )
                    {
                        consultTheOracle(*randDir,dim,locxres,!i);
                    }
//errstream() << "phantomx 1: randDir (" << i << ") = " << *randDir << "\n";
                }

                else
                {
                    errstream() << "include constant term (weighted)... ";

                    // First round, so final variable controls a *constant* offset

                    gentype outfnhere(whatConst);

                    MEMNEW(randDir,ML_Mutable);
                    (*randDir).setMLTypeClean(207);  // BLK_UsrFnb

                    (*randDir).setoutfn(outfnhere);
                    subDefInd("&",i) = regML(randDir,fnarg,2);

                    subDef("&",i) = &((*randDir).getML());
                }
            }

            if ( addSubDim )
            {
                errstream() << "include previous best term (fixed zero point)... ";

                // Put previous best as "fixed" point in new subspace.  projOp
                // has been made the best result from the most recent sub-optimisation.

                randDir = projOpRaw;
                subDefInd("&",dim) = projOpInd;
                subDef("&",dim) = &((*randDir).getML());
            }

            // Make projOp a projection onto this subspace

            MEMNEW(projOpRaw,ML_Mutable);
            (*projOpRaw).setMLTypeClean(projOptemplate.type());

            (*projOpRaw).getML() = projOptemplate;
            projOpInd = regML(projOpRaw,fnarg,3);

            projOp = &(dynamic_cast<BLK_Conect &>((*projOpRaw).getBLK()));
//errstream() << "phantomx 2: projOp = " << *projOp << "\n";

            errstream() << "sampling....";

            Vector<gentype> subWeight(dim+addSubDim);

            gentype biffer;

            subWeight = ( biffer = 0.0 );

            if ( addSubDim )
            {
                subWeight("&",dim) = 1.0;
            }

            (*projOp).setmlqlist(subDef);
            (*projOp).setmlqweight(subWeight);

//FIXMEFIXME fnDim
            errstream() << "combining... ";

            Vector<gentype> xmin(fnDim);
            Vector<gentype> xmax(fnDim);

            gentype buffer;

            xmin = ( buffer = 0.0 );
            xmax = ( buffer = 1.0 );

            (*projOp).setSampleMode(1,xmin,xmax,xNsamp); // Need this even when subDef has been sampled to set variables used to construct y in projOp, when required
//errstream() << "phantomx 3: subDef (0) = " << *subDef(0) << "\n";
//errstream() << "phantomx 4: projOp = " << (*projOp).y() << "\n";

            if ( useScalarFn && !includeConst && ( isProjection == 2 ) )
            {
                xbasis.resize(rdim);

                for ( i = 0 ; i < rdim ; i++ )
                {
                    xbasis("&",i) = (*subDef(i)).y();
                    xbasis("&",i).scale(sqrt(1/((double) (*subDef(i)).y().size())));
                }
//errstream() << "phantomx 5: xbasis = " << xbasis << "\n";

                gentype tempxp;

                xbasisprod.resize(rdim,rdim);

                for ( i = 0 ; i < rdim ; i++ )
                {
                    for ( j = 0 ; j < rdim ; j++ )
                    {
                        twoProduct(xbasisprod("&",i,j),xbasis(i),xbasis(j));
                    }
                }
//errstream() << "phantomx 6: xbasisprod = " << xbasisprod << "\n";

                model_update();
            }

            errstream() << "done with weight " << ((*projOp).mlqweight()) << "\n";
        }

        else if ( ( isProjection == 3 ) || ( isProjection == 4 ) )
        {
            // We set up the same dimension, even though we don't use some of it

            NiceAssert( !includeConst );

            addSubDim = 0;

            errstream() << "Constructing Bernstein basis... ";

            subDef.resize(dim);
            subDefInd.resize(dim);

            int i,j,k;
            int axisdim = (int) pow((double) dim+1,1/((double) fnDim))-1; // floor

            Vector<double> berndim(fnDim);
            Vector<double> bernind(fnDim);

            berndim = (double) axisdim;
            bernind = 0.0;

            ML_Mutable *randDir;

            errstream() << "setting parameters... ";

            for ( i = 0 ; i < dim ; i++ )
            {
                MEMNEW(randDir,ML_Mutable);
                (*randDir).setMLTypeClean((*randDirtemplate).type());

                (*randDir).getML() = (*randDirtemplate);
                subDefInd("&",i) = regML(randDir,fnarg,8);

                subDef("&",i) = &((*randDir).getML());

                k = i;

                for ( j = 0 ; j < fnDim ; j++ )
                {
                    bernind("&",j) = k%(((int) berndim(j))+1);
                    k /= ((int) berndim(j))+1;
                }

                gentype gberndim(berndim);
                gentype gbernind(bernind);

//FIXMEFIXME fnDim
                dynamic_cast<BLK_Bernst &>((*(subDef("&",i)))).setBernDegree(gberndim);
                dynamic_cast<BLK_Bernst &>((*(subDef("&",i)))).setBernIndex(gbernind);
            }

            errstream() << "combining... ";

            // Make projOp a projection onto this subspace

            MEMNEW(projOpRaw,ML_Mutable);
            (*projOpRaw).setMLTypeClean(projOptemplate.type());

            (*projOpRaw).getML() = projOptemplate;
            projOpInd = regML(projOpRaw,fnarg,9);

            projOp = &(dynamic_cast<BLK_Conect &>((*projOpRaw).getBLK()));

            errstream() << "sampling....";

            Vector<gentype> subWeight(dim);

            gentype biffer;

            subWeight = ( biffer = 0.0 );

            (*projOp).setmlqlist(subDef);
            (*projOp).setmlqweight(subWeight);

//FIXMEFIXME fnDim
            Vector<gentype> xmin(fnDim);
            Vector<gentype> xmax(fnDim);

            gentype buffer;

            xmin = ( buffer = 0.0 );
            xmax = ( buffer = 1.0 );

            (*projOp).setSampleMode(1,xmin,xmax,xNsamp); // Need this even when subDef has been sampled to set variables used to construct y in projOp, when required

            errstream() << "done with weight " << ((*projOp).mlqweight()) << "\n";
        }

        else if ( isProjection == 5 )
        {
//RKHSFIXME
            // Projection via RKHSVector, nothing to see here

            addSubDim = 0;

            defrandDirtemplateFnRKHS.resizeN(dim);

            int i,j;

            for ( i = 0 ; i < dim ; i++ )
            {
                defrandDirtemplateFnRKHS.x("&",i).zero();

                for ( j = 0 ; j < fnDim ; j++ )
                {
                    randfill(defrandDirtemplateFnRKHS.x("&",i)("&",j).force_double());
                }
            }

            projOpInd = -1;
        }

        return projOpInd;
    }











int godebug = 0; // to debug vector.h

void overfn(gentype &res, Vector<gentype> &x, void *arg)
{
    void *fnarg                       = ((void *)                      ((void **) arg)[0]);
    int &dim                          = *((int *)                      ((void **) arg)[1]);
    Vector<gentype> &xmod             = *((Vector<gentype> *)          ((void **) arg)[2]);
    GlobalOptions &gopts              = *((GlobalOptions *)            ((void **) arg)[3]);
    Vector<Vector<gentype> > &allxres = *((Vector<Vector<gentype> > *) ((void **) arg)[4]);
    gentype &penterm                  = *((gentype *)                  ((void **) arg)[5]);

    void (*fn)(gentype &, Vector<gentype> &, void *) = ((void (*)(gentype &, Vector<gentype> &, void *)) ((void **) arg)[6]);

    gopts.convertx(dim,xmod,x,0,1);
//errstream() << "phantomxxxxx 0 convert\n";
//gopts.convertx(dim,xmod,x,0,1);
//errstream() << "phantomxxxxx 1 model_convert\n";
//SparseVector<gentype> xalt(x);
//SparseVector<gentype> xmoddummy(x);
//gopts.model_convertx(xmoddummy,xalt,0,0,1);
//errstream() << "phantomxxxxx 2 done\n";

    (*fn)(res,xmod,fnarg);
//errstream() << "f(g(" << x << ")) = f(" << xmod << ") = " << res << "\n";
errstream() << "Global optimiser: f(g(.))" << xmod << ") = " << res << "\n";

    if ( !(penterm.isCastableToRealWithoutLoss()) || ( ( (double) penterm ) != 0.0 ) )
    {
        gentype xmodvecform(xmod);

        res += penterm(xmodvecform);
errstream() << "Global optimiser: f(g(.)) + penterm(" << xmodvecform << ") = " << res << "\n";
    }

    // Clip result at softmax

    if ( ( res.isValInteger() || res.isValReal() ) && ( (double) res > gopts.softmax ) )
    {
errstream() << "Result clipped at softmax: " << res << " -> " << gopts.softmax << "\n";
        res = gopts.softmax;
    }

    // Save to allxres if this is an actual evaluation ( x.size() == 0 if intermediate operation )

    if ( x.size() )
    {
        allxres.append(allxres.size(),xmod);

        if ( ( gopts.isProjection == 4 ) && ( gopts.stopearly == 0 ) && ( gopts.berndim > 1 ) && ( gopts.firsttest || ( res < gopts.bestyet ) ) )
        {
            int i,j;
            int effdim = gopts.berndim;

            gopts.stopearly = 0;
            gopts.firsttest = 0;
            gopts.bestyet   = res;

            for ( i = 0 ; i < effdim-1 ; i++ )
            {
                for ( j = i+1 ; j < effdim ; j++ )
                {
                    if ( (double) abs2(x(i)-x(j)) > 0.95 )
                    {
                        gopts.stopearly = 1;
                        break;
                    }
                }
            }
        }
    }

    errstream() << ".";

    return;
}

double calcabc(int dim, 
               Vector<gentype> &fakexmin, Vector<gentype> &fakexmax, 
               Vector<double> &a, Vector<double> &b, Vector<double> &c,
               const Vector<gentype> &xmin, const Vector<gentype> &xmax, 
               const Vector<int> &distMode)
{
    double xwidth = 1;

    a.resize(dim+1);
    b.resize(dim+1);
    c.resize(dim+1);

    if ( dim )
    {
        int i;

        for ( i = 0 ; i < dim ; i++ )
        {
            double lb = (double) xmin(i);
            double ub = (double) xmax(i);

            int zerowidth = ( lb == ub ) ? 1 : 0;

            xwidth = ( !i || ( (ub-lb) > xwidth ) ) ? (ub-lb) : xwidth;

            fakexmin("&",i) = 0.0;
            fakexmax("&",i) = zerowidth ? 0.0 : 1.0;

            if ( distMode(i) == 1 )
            {
                // Logarithmic grid
                //
                // v = a + e^(b+c.t)
                // 
                // c = log(10)
                // 
                // t = 0 => v = lb
                //       => a+e^b = lb
                //       => b = log(lb-a)
                // t = 1 => v = ub
                //       => a+e^(b+c) = ub
                //       => e^(b+c) = ub-a
                //       => b+c = log(ub-a)
                //       => b = log(ub-a)-c
                // 
                // => log(lb-a) = lob(ub-a)-c
                // => log((ub-a)/(lb-a)) = log(e^c)
                // => (ub-a)/(lb-a) = e^c
                // => ub - a = (e^c)lb - (e^c)a
                // => ((e^c)-1).a = (e^c)lb - ub
                // => a = ((e^c)lb - ub)/((e^c)-1)

                c("&",i) = log(10.0);
                a("&",i) = ( (exp(c(i))*lb) - ub ) / ( exp(c(i)) - 1.0 );
                b("&",i) = log( lb - a(i) );
            }

            else if ( distMode(i) == 2 )
            {
                // Anti-logarithmic grid (inverse of logarithmic grid)
                //
                // v = (1/c) log(t-a) - (b/c)
                //
                // c = log(10)
                // 
                // t = 0 => v = lb
                //       => 0 = a + e^(b+c.lb)
                //       => -a = e^(b+c.lb)
                //       => b+c.lb = log(-a)
                //       => b = log(-a) - c.lb
                // t = 0 => t = ub
                //       => 1 = a + e^(b+c.ub)
                //       => 1-a = e^(b+c.ub)
                //       => b+c.ub = log(1-a)
                //       => b = log(1-a) - c.ub
                //
                // => log(-a) - c.lb = log(1-a) - c.ub
                // => c.(ub-lb) = log((a-1)/a)
                // => (a-1)/a = exp(c.(ub-lb))
                // => 1 - 1/a = exp(c.(ub-lb))
                // => 1/a = 1 - exp(c.(ub-lb))
                // => a = 1/(1 - exp(c.(ub-lb)))

                c("&",i) = log(10);
                a("&",i) = 1/(1-exp(c(i)*(ub-lb)));
                b("&",i) = log(-a(i))-(c(i)*lb);
            }

            else if ( distMode(i) == 4 )
            {
                // Anti-logistic grid
                //
                // v = a - (1/b) log( 1/(0.5+(c*(t-0.5))) - 1 )
                //   = a - (1/b) log(0.5-(c*(t-0.5))) + (1/b) log(0.5+(c*(t-0.5)))
                //   = a - (1/b) log(1/(0.5+(c*(t-0.5)))) + (1/b) log(1/(0.5-(c*(t-0.5))))
                //
                // a = lb+((ub-lb)/2)
                // a = (ub+lb)/2
                //
                // m = small number < 1/2
                // B = big number
                // D = (ub-lb)/2
                //
                // t = 0 => v = a-B
                //       => a - (1/b) log( 1/(0.5*(1-c)) - 1 ) = a-B
                //       => a - (1/b) log( 2/(1-c) - 1 ) = a-B
                //       => a - (1/b) log( (1+c)/(1-c) ) = a-B
                //       => a - (1/b) log(1+c) + (1/b) log(1-c) = a-B
                //       => (1/b) log(1+c) - (1/b) log(1-c) = B
                //       => log(1+c) - log(1-c) = b.B
                // t = 1 => v >= a+B
                //       => a - (1/b) log( 1/(0.5*(1+c)) - 1 ) = a+B
                //       => a - (1/b) log( 2/(1+c) - 1 ) = a+B
                //       => a - (1/b) log( (1-c)/(1+c) ) = a+B
                //       => a - (1/b) log(1-c) + (1/b) log(1+c) = a+B
                //       => (1/b) log(1-c) - (1/b) log(1+c) = -B
                //       => (1/b) log(1+c) - (1/b) log(1-c) = B
                //       => log(1+c) - log(1-c) = b.B
                // t = m => v = lb
                //       => a - (1/b) log( 1/(0.5+(c*(m-0.5))) - 1 ) = lb
                //       => a - (1/b) log( 1/(0.5*(1-c)+cm) - 1 ) = lb
                //       => (1/b) log( 1/(0.5*(1-c)+cm) - 1 ) = a-lb
                //       => (1/b) log( 1/(0.5*(1-c)+cm) - 1 ) = (ub-lb)/2
                //       => (1/b) log( 2/(1-c+2cm) - 1 ) = (ub-lb)/2
                //       => (1/b) log( (2-(1-c+2cm))/(1-c+2cm) ) = (ub-lb)/2
                //       => (1/b) log( (1+c-2cm)/(1-c+2cm) ) = (ub-lb)/2
                //       => log(1+c-2cm) - log(1-c+2cm) = b.(ub-lb)/2
                // t = 1-m => v = ub
                //       => a - (1/b) log( 1/(0.5+(c*(1-m-0.5))) - 1 ) = lb
                //       => a - (1/b) log( 1/(0.5*(1+c)-cm) - 1 ) = ub
                //       => (1/b) log( 1/(0.5*(1+c)-cm) - 1 ) = a-ub
                //       => (1/b) log( 1/(0.5*(1+c)-cm) - 1 ) = -(ub-lb)/2
                //       => (1/b) log( 2/(1+c-2cm) - 1 ) = -(ub-lb)/2
                //       => (1/b) log( (2-(1+c-2cm))/(1+c-2cm) ) = -(ub-lb)/2
                //       => (1/b) log( (1-c+2cm)/(1+c-2cm) ) = -(ub-lb)/2
                //       => log(1-c+2cm) - log(1+c-2cm) = -b.(ub-lb)/2
                //       => log(1+c-2cm) - log(1-c+2cm) = b.(ub-lb)/2
                //
                // NB: - symmetry means we need only consider one of t=m,1-m 
                //       (note that the final two expressions are identical)
                //       where both imply that:
                //
                //       b = (log(1+c-2cm)-log(1-c+2cm))/D
                //       b = (log(1+(c*(1-2m)))-log(1-(c*(1-2m))))/D   (*)
                //
                //       where we have defined D = (ub-lb)/2
                //
                //     - likewise symmetry means we only need consider one
                //       of t=0,1 (see final expressions), which using (*) imply:
                //
                //       q = B.(log(1+(c*(1-2m)))-log(1-(c*(1-2m)))) - D.(log(1+c)-log(1-c)) = 0    (**)
                //
                //     - note the derivative of (**) wrt c
                //
                //       B.(1-2m).( 1/(1+(c*(1-2m))) + 1/(1-(c*(1-2m))) ) - D.( 1/(1+c) + 1/(1-c) ) = 0
                //
                //       Assuming B sufficiently large this will be increasing.

                double aval = (ub+lb)/2;
                double qval,bval,cval;

                double cmin = 1e-6;
                double cmax = 0.5;

                double m = NEARZEROVAL;
                double B = NEARINFVAL;
                double D = (ub-lb)/2;
                double ratmax = RATIOMAX;

                cmin = 0;
                cmax = 1;

                while ( 2*(cmax-cmin)/(cmax+cmin) > ratmax )
                {
                    cval  = (cmin+cmax)/2;
                    //bval  = 2*((1/(1+(cval*(1-(2*m)))))-(1/(1-(cval*(1-(2*m))))))/D;
                    bval  = 2*((1/(1-(cval*(1-(2*m)))))-(1/(1+(cval*(1-(2*m))))))/D;
                    qval  = 2*((1.0/(1-cval))-(1.0/(1+cval)))/bval;
                    if ( qval == B )
                    {
                        cmax = cval;
                        cmin = cval;
                    }

                    else if ( qval > B )
                    {
                        cmax = cval;
                    }

                    else
                    {
                        cmin = cval;
                    }
                }

                //cval = cmax;
                //bval = 2*((1/(1+(cval*(1-(2*m)))))-(1/(1-(cval*(1-(2*m))))))/D;
                bval = 2*((1/(1-(cval*(1-(2*m)))))-(1/(1+(cval*(1-(2*m))))))/D;

                a("&",i) = aval;
                b("&",i) = bval;
                c("&",i) = cval;
            }

            else
            {
                // Linear (potentially random) grid
                //
                // v = a + b.t
                //
                // t = 0 => v = lb
                //       => a = lb
                // t = 1 => v = ub
                //       => lb+b = ub
                //       => b = ub-lb

                a("&",i) = lb;
                b("&",i) = ub-a(i);
                c("&",i) = 0;
            }
        }
    }

    return xwidth;
}





int GlobalOptions::analyse(const Vector<Vector<gentype> > &allxres,
                           const Vector<gentype> &allfresmod,
                           Vector<double> &hypervol,
                           Vector<int> &parind,
                           int calchypervol) const
{
    NiceAssert( allxres.size() == allfresmod.size() );

    int N = allxres.size();

    parind.resize(N);
    hypervol.resize(N);

    hypervol = 0.0;

    if ( N )
    {
        // Work out hypervolume sequence

        if ( calchypervol )
        {
            if ( allfresmod(zeroint()).isValReal() )
            {
                int i;

                for ( i = 0 ; i < N ; i++ )
                {
                    if ( !i || ( (double) allfresmod(i) < -(hypervol(i-1)) ) )
                    {
                        hypervol("&",i) = -((double) allfresmod(i));
                    }

                    else
                    {
                        hypervol("&",i) = hypervol(i-1);
                    }
                }
            }

            else
            {
                int i,j;

                int M = (allfresmod(zeroint())).size();
                double **X;

                MEMNEWARRAY(X,double *,N);

                NiceAssert(X);

                for ( i = 0 ; i < N ; i++ )
                {
                    MEMNEWARRAY(X[i],double,M);

                    NiceAssert( X[i] );

                    for ( j = 0 ; j < M ; j++ )
                    {
                        X[i][j] = -((double) ((allfresmod(i))(j)));
                    }

                    hypervol("&",i) = h(X,i+1,M);
                }

                for ( i = 0 ; i < N ; i++ )
                {
                    MEMDELARRAY(X[i]);
                }

                MEMDELARRAY(X);
            }
        }

        // Work out Pareto set

        int m = allfresmod(zeroint()).size();

        retVector<int> tmpva;

        parind = cntintvec(N,tmpva);

        int pos,i,j,isdom;

        for ( pos = parind.size()-1 ; pos >= 0 ; pos-- )
        {
            NiceAssert( allfresmod(pos).size() == m );

            // Test if allfres(parind(pos)) is dominated by any points
            // in parind != pos.  If it is then remove it from parind.

            isdom = 0;

            for ( i = 0 ; i < parind.size() ; i++ )
            {
                if ( parind(i) != parind(pos) )
                {
                    // Test if allfres(i) dominates allfres(pos) - that is,
                    // if allfres(pos)(j) >= allfres(i)(j) for all j

                    isdom = 1;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        if ( allfresmod(parind(pos))(j) < allfresmod(parind(i))(j) )
                        {
                            isdom = 0;
                            break;
                        }
                    }

                    if ( isdom )
                    {
                        break;
                    }
                }
            }

            if ( isdom )
            {
                parind.remove(pos);
            }
        }
    }

    return parind.size();
}
