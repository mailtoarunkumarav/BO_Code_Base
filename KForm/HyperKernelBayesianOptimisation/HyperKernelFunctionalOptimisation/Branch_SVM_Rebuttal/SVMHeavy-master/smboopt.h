
//
// Sequential model-based optimisation base class
//
// Date: 2/12/2019
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//#define TURNOFFSHORTCUT

#include "ml_base.h"
#include "ml_mutable.h"
#include "gpr_scalar.h"
#include "gpr_vector.h"
#include "globalopt.h"
#include "gridopt.h"
#include "directopt.h"
#include "nelderopt.h"
#include "errortest.h"
#include "addData.h"

#ifndef _smboopt_h
#define _smboopt_h

class SMBOOptions : public GlobalOptions
{
public:
    // Models (moo == multi-objective)

    ML_Base *fnapprox;
    ML_Base *fnapproxmoo;

    int fnapproxInd;
    int fnapproxmooInd;

    // sigmuseparate: for multi-recommendation by default both sigma and by
    //         are approximated by the same ML.  Alternatively you can do
    //         them separately: mu is updated for each batch, and sigma
    //         independently for each point selected.
    //         0 = use the same ML.
    //         1 = use separate MLs.
    // ismoo: set 1 for multi-objective optimisation
    //
    // modeltype: 0 = model f(p(x)) using p(x)
    //            1 = model f(p(x)) using p(x), model_clear resets model
    //            2 = model f(p(x)) using x, model_clear resets model
    //            3 = model f(p(x)) using x
    // oracleMode: 0 = oracle uses GP model derivative to find mean/variance
    //                 descent direction at current best solution, samples that
    //             1 = oracle uses GP model derivative to find mean descent
    //                 direction, uses that
    //             2 = fallback (purely random oracle)
    //             3 = use mode 0 for primary axis, mode 2 for the rest
    //             4 = use mode 1 for primary axis, mode 2 for the rest
    //
    // Transfer learning:
    //
    // tranmeth: 0 - content of muModel treated as data from current model
    //           1 - transfer learning as per Joy1 (Shi21 Env-GP)
    //           2 - transfer learning as per Shi21, Diff-GP
    // alpha0:   starting alpha value
    // beta0:    starting beta value
    //
    // Kernel transfer learning:
    //
    // kernapprox: pointer to ML to copy kernel from
    // kxfnum:     kernel transfer type (-kt)
    // kxfnorm:    0 - no normalisation
    //             1 - yes
    //
    // Model tuning:
    //
    // tunemu:      set to tune muapprox at every step (default 1)
    // tunesigma:   set to tune sigmaapprox at every step if sigmuseparate (default 1)
    // tunesrcmod:  set to tune srcmodel at start (default 1)
    // tunediffmod: set to tune diffapprox at every step (default 1)
    //
    // 0: don't tune
    // 1: tune for max-likelihood
    // 2: tune for leave-one-out (default)
    // 3: tune for recall
    //
    // xtemplate: "background" template for x data.  To construct data to be added
    //            to models we start with this template and over-write parts with x.
    //            For example if xtemplate = [ ~ xa ] then data x -> [ x ~ xa ].

    int sigmuseparate;
    int ismoo;
    int modeltype;
    int oracleMode;

    int tranmeth;
    double alpha0;
    double beta0;

    ML_Base *kernapprox;
    int kxfnum;
    int kxfnorm;

    int tunemu;
    int tunesigma;
    int tunesrcmod;
    int tunediffmod;

    SparseVector<gentype> xtemplate;

    // Constructors and assignment operators

    SMBOOptions() : GlobalOptions()
    {
        thisthis = this;
        thisthisthis = &thisthis;

        locxres.useTightAllocation();

        fnapproxInd    = -1;
        fnapproxmooInd = -1;

        sigmuseparate = 0;
        ismoo         = 0;
        modeltype     = 0;
        oracleMode    = 0;

        tranmeth = 0;
        alpha0   = 0.1;
        beta0    = 1;

        kernapprox = NULL;
        kxfnum     = 801;
        kxfnorm    = 1;

        tunemu      = 2;
        tunesigma   = 2;
        tunesrcmod  = 2;
        tunediffmod = 2;

        srcmodel  = NULL;
        diffmodel = NULL;

        srcmodelInd  = -1;
        diffmodelInd = -1;

        altfnapproxmoo.settspaceDim(2);

        fnapprox    = NULL;
        fnapproxmoo = NULL;

        muapprox    = NULL;
        sigmaapprox = NULL;

        modelErrOptim   = NULL;
        ismodelErrLocal = 1;

        altfnapproxFNapprox.getKernel_unsafe().resize(2);
        altfnapproxFNapprox.getKernel_unsafe().setType(1,0);
        altfnapproxFNapprox.getKernel_unsafe().setType(1,1);
        altfnapproxFNapprox.getKernel_unsafe().setMagTerm(1);
        altfnapproxFNapprox.resetKernel();

        xshortcutenabled = 0;

        return;
    }

    SMBOOptions(const SMBOOptions &src) : GlobalOptions(src)
    {
        thisthis = this;
        thisthisthis = &thisthis;

        modelErrOptim   = NULL;
        ismodelErrLocal = 1;

        locxres.useTightAllocation();

        *this = src;

        return;
    }

    SMBOOptions &operator=(const SMBOOptions &src)
    {
        GlobalOptions::operator=(src);

        killModelErrOptim();

        fnapprox    = src.fnapprox;
        fnapproxmoo = src.fnapproxmoo;

        fnapproxInd    = src.fnapproxInd;
        fnapproxmooInd = src.fnapproxmooInd;

        sigmuseparate = src.sigmuseparate;
        ismoo         = src.ismoo;
        modeltype     = src.modeltype;
        oracleMode    = src.oracleMode;

        tranmeth = src.tranmeth;
        alpha0   = src.alpha0;
        beta0    = src.beta0;

        kernapprox = src.kernapprox;
        kxfnum     = src.kxfnum;
        kxfnorm    = src.kxfnorm;

        tunemu      = src.tunemu;
        tunesigma   = src.tunesigma;
        tunesrcmod  = src.tunesrcmod;
        tunediffmod = src.tunediffmod;

        xtemplate = src.xtemplate;

        // ======================================

        xmodprod         = src.xmodprod;
        xshortcutenabled = src.xshortcutenabled;
        xsp              = src.xsp;
        xspp             = src.xspp;
        indpremu         = src.indpremu;
        presigweightmu   = src.presigweightmu;

        Nbasemu = src.Nbasemu;
        resdiff = src.resdiff;

        alpha = src.alpha;
        beta  = src.beta;

        srcmodel  = src.srcmodel;
        diffmodel = src.diffmodel;

        srcmodelInd  = src.srcmodelInd;
        diffmodelInd = src.diffmodelInd;

        diffval  = src.diffval;
        predval  = src.predval;
        storevar = src.storevar;

        firsttrain = src.firsttrain;

        xx = src.xx;

        muapprox    = src.muapprox;
        sigmaapprox = src.sigmaapprox;

        altfnapprox         = src.altfnapprox;
        altfnapproxFNapprox = src.altfnapproxFNapprox;
        altfnapproxmoo      = src.altfnapproxmoo;

        locxres = src.locxres;
        locires = src.locires;

        if ( src.modelErrOptim )
        {
            modelErrOptim   = (*(src.modelErrOptim)).makeDup();
            ismodelErrLocal = 1;
        }

        return *this;
    }

    // Generate a copy of the relevant optimisation class.

    virtual GlobalOptions *makeDup(void) const
    {
        SMBOOptions *newver;

        MEMNEW(newver,SMBOOptions(*this));

        return newver;
    }

    // virtual Destructor to get rid of annoying warnings

    virtual ~SMBOOptions() 
    {
        killModelErrOptim();

        return;
    }

    // Local or global models?

    template <class S>
    const SparseVector<gentype> &model_convertx(SparseVector<gentype> &res, const SparseVector<S> &x, int useOrigin = 0, int useShortcut = 0) const
    {
//errstream() << "phantomxrr 0\n";
        if ( ( modeltype == 2 ) || ( modeltype == 3 ) )
        {
//errstream() << "phantomxrr 1\n";
             return res.castassign(x);
//errstream() << "phantomxrr 2\n";
        }

//errstream() << "phantomxrr 3\n";
        return GlobalOptions::model_convertx(res,x,useOrigin,useShortcut);
    }

    template <class S>
    const Vector<SparseVector<gentype> > &model_convertx(Vector<SparseVector<gentype> > &res, const Vector<SparseVector<S> > &x) const
    {
        if ( ( modeltype == 2 ) || ( modeltype == 3 ) )
        {
            return res.castassign(x);
        }

        return GlobalOptions::model_convertx(res,x);
    }

    virtual void model_clear(void)
    {
        if ( ( modeltype == 1 ) || ( modeltype == 2 ) )
        {
            (*muapprox).removeTrainingVector(0,(*muapprox).N());
            (*sigmaapprox).removeTrainingVector(0,(*sigmaapprox).N());
        }

        return;
    }

    Matrix<gentype> xmodprod; // xmodprod(i,j) is inner products between x(i) and xbasisj
    int xshortcutenabled; // set 1 if we can do fast calculation of inner product using xmodprod and xbasisprod
    Vector<gentype> xsp; // vector used to record inner product of xi and suggested vector
    Vector<gentype **> xspp; // pointery stuff for some reason

    virtual void model_update(void)
    {
#ifndef TURNOFFSHORTCUT
        if ( ( modeltype == 0 ) || ( modeltype == 1 ) )
        {
            NiceAssert( muapprox == sigmaapprox );

            int N = (*muapprox).N();
            int m = xbasis.size();

            xmodprod.resize(N,m);

            int i,j;

            for ( i = 0 ; i < N ; i++ )
            {
                for ( j = 0 ; j < m ; j++ )
                {
                    twoProduct(xmodprod("&",i,j),(*muapprox).x(i),xbasis(j));
                }
            }

            xshortcutenabled = 1;
        }
#endif

        return;
    }

    virtual void consultTheOracle(ML_Mutable &randDir, int dim, const SparseVector<gentype> &locxres, int isFirstAxis)
    {
        if ( ( ( oracleMode == 0 ) || ( ( oracleMode == 3 ) && isFirstAxis ) ) && ( isProjection == 2 ) && useScalarFn && isGPRScalar(randDir) && ( ( modeltype == 0 ) || ( modeltype == 1 ) ) && !sigmuseparate && !ismoo )
        {
            // Construct vector to calculate gradient at locxres

            gentype Ns(xNsamp);

            SparseVector<gentype> czm(locxres);
            czm.fff("&",6) = 1; // This tells models to evaluate gradients

            // Find mean and variance of gradient at czm

            gentype mvec;
            gentype vmat;

            (*muapprox).var(vmat,mvec,czm);

            // Sample gradient GP to get direction

            gentype yvec;

            yvec = grand(mvec,vmat);
            yvec *= -1.0; // gradient *descent*

            // Pre-sample GPR (we will overwrite the result)

//FIXMEFIXME fnDim
            Vector<gentype> xmin(fnDim);
            Vector<gentype> xmax(fnDim);

            gentype buffer;

            xmin = ( buffer = 0.0 );
            xmax = ( buffer = 1.0 );

            randDir.getGPR().setSampleMode(1,xmin,xmax,xNsamp);

            // Overwrite y to finalise sample

            int dummy = 0;

            randDir.getGPR().sety((const Vector<gentype> &) yvec);
            randDir.getGPR().train(dummy);
        }

        else if ( ( ( oracleMode == 1 ) || ( ( oracleMode == 4 ) && isFirstAxis ) ) && ( isProjection == 2 ) && useScalarFn && isGPRScalar(randDir) && ( ( modeltype == 0 ) || ( modeltype == 1 ) ) && !sigmuseparate && !ismoo )
        {
            // Construct vector to calculate gradient at locxres

            gentype Ns(xNsamp);

            SparseVector<gentype> czm(locxres);
            czm.fff("&",6) = 1; // This tells models to evaluate gradients

            // Find mean and variance of gradient at czm

            gentype mvec;

            (*muapprox).gg(mvec,czm);

            // We don't sample here, just take the direction

            gentype yvec;

            yvec = mvec;
            yvec *= -1.0; // gradient *descent*

            // Pre-sample GPR (we will overwrite the result)

//FIXMEFIXME fnDim
            Vector<gentype> xmin(fnDim);
            Vector<gentype> xmax(fnDim);

            gentype buffer;

            xmin = ( buffer = 0.0 );
            xmax = ( buffer = 1.0 );

            randDir.getGPR().setSampleMode(1,xmin,xmax,xNsamp);

            // Overwrite y to finalise sample

            int dummy = 0;

            randDir.getGPR().sety((const Vector<gentype> &) yvec);
            randDir.getGPR().train(dummy);
        }

        else
        {
//RKHSFIXME
            GlobalOptions::consultTheOracle(randDir,dim,locxres,isFirstAxis);
        }

        return;
    }

    // Optimisation functions etc fall back to GlobalOptions

    virtual int optim(int dim,
                      Vector<gentype> &rawxres,
                      gentype &fres,
                      int &ires,
                      Vector<Vector<gentype> > &allrawxres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allfresmod,
                      Vector<gentype> &supres,
                      Vector<double> &sscore,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch)
    {
        return GlobalOptions::optim(dim,rawxres,fres,ires,allrawxres,allfres,allfresmod,supres,sscore,xmin,xmax,fn,fnarg,killSwitch);
    }

    virtual int optim(int dim,
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
        int res = GlobalOptions::optim(dim,xres,Xres,fres,ires,mInd,muInd,sigInd,srcmodInd,diffmodInd,allxres,allXres,allfres,allmres,allsres,s_score,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch,numReps,meanfres,varfres,meanires,varires,meantres,vartres,meanTres,varTres,meanallfres,varallfres,meanallmres,varallmres);

        return res;
    }

    // Variables for env-GP (Joy1,Shi21) and diff-GP (Shi21)
    //
    // indpremu: indices of pre-loaded samples (transfer learning) for env-GP
    // presigweightmu: sigma "stretch" for env-GP
    //
    // Nbasemu: number of base (transfer) variables for env-GP
    // resdiff: helper variable for env-GP
    //
    // alpha,beta: see Joy1, env-GP/diff-GP
    //
    // srcmodel:  for diff-GP, this is used to store the source and model data in "raw form"
    // diffmodel: for diff-GP, this is used to model the difference between source and target models
    //
    // srcmodelInd:  index for srcmodel
    // diffmodelInd: index for diffmodel
    //
    // diffval: used for diff-GP
    // predval: used for diff-GP
    //
    // firsttrain: set 1 by optim, then 0 again by train

    Vector<int> indpremu;
    Vector<double> presigweightmu;

    int Nbasemu;
    gentype resdiff;

    double alpha;
    double beta;

    ML_Mutable *srcmodel;
    ML_Mutable *diffmodel;

    int srcmodelInd;
    int diffmodelInd;

    gentype diffval;
    gentype predval;
    gentype storevar;

    int firsttrain;

    virtual int realOptim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &rawxres,
                      gentype &fres,
                      int &ires,
                      int &mres,
                      int &muInd,
                      int &sigInd,
                      int &srcmodInd,
                      int &diffmodInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allrawxres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allfresmod,
                      Vector<gentype> &supres,
                      Vector<double> &sscore,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch)
    {
        // Create and register models

        //int fnapproxInd    = -1;
        //int fnapproxmooInd = -1;

        // These need to be passed back

        Vector<int> dummyMLnumbers(6);
        Vector<int> &MLnumbers = MLdefined ? (*((Vector<int> *) ((void **) fnarg)[15])) : dummyMLnumbers;

        // Construct models and register them

        if ( !ismoo )
        {
            if ( fnapprox )
            {
                ML_Mutable *fnapproxRaw;

                MEMNEW(fnapproxRaw,ML_Mutable);
                (*fnapproxRaw).setMLTypeClean((*fnapprox).type());

                (*fnapproxRaw).getML() = *fnapprox;
                fnapproxInd = regML(fnapproxRaw,fnarg,5);

                fnapprox = &((*fnapproxRaw).getML());
            }

            else if ( isProjection != 2 )
            {
//RKHSFIXME
                ML_Mutable *fnapproxRaw;

                MEMNEW(fnapproxRaw,ML_Mutable);
                (*fnapproxRaw).setMLTypeClean(altfnapprox.type());

                (*fnapproxRaw).getML() = altfnapprox;
                fnapproxInd = regML(fnapproxRaw,fnarg,5);

                fnapprox = &((*fnapproxRaw).getML());
//RKHSFIXME
                //(*fnapprox).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );
            }

            else
            {
                ML_Mutable *fnapproxRaw;

                MEMNEW(fnapproxRaw,ML_Mutable);
                (*fnapproxRaw).setMLTypeClean(altfnapproxFNapprox.type());

                (*fnapproxRaw).getML() = altfnapproxFNapprox;
                fnapproxInd = regML(fnapproxRaw,fnarg,5);

                fnapprox = &((*fnapproxRaw).getML());
            }

            MLnumbers("&",zeroint()) = fnapproxInd;

            muapprox = fnapprox;
            muInd    = fnapproxInd;
//errstream() << "phantomx model = " << *muapprox << "\n";
        }

        else
        {
            if ( fnapproxmoo == NULL )
            {
                ML_Mutable *fnapproxmooRaw;

                MEMNEW(fnapproxmooRaw,ML_Mutable);
                (*fnapproxmooRaw).setMLTypeClean((*fnapproxmoo).type());

                (*fnapproxmooRaw).getML() = *fnapproxmoo;
                fnapproxmooInd = regML(fnapproxmooRaw,fnarg,6);

                fnapproxmoo = &((*fnapproxmooRaw).getML());
            }

            else
            {
                ML_Mutable *fnapproxmooRaw;

                MEMNEW(fnapproxmooRaw,ML_Mutable);
                (*fnapproxmooRaw).setMLTypeClean(altfnapproxmoo.type());

                (*fnapproxmooRaw).getML() = altfnapproxmoo;
                fnapproxmooInd = regML(fnapproxmooRaw,fnarg,6);

                fnapproxmoo = &((*fnapproxmooRaw).getML());
//RKHSFIXME
                //(*fnapproxmoo).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );
            }

            MLnumbers("&",zeroint()) = fnapproxInd;

            muapprox = fnapproxmoo; 
            muInd    = fnapproxmooInd;
        }

//RKHSFIXME
        //(*muapprox).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );

        if ( sigmuseparate )
        {
            ML_Mutable *sigmaapproxRaw;

            MEMNEW(sigmaapproxRaw ,ML_Mutable);
            (*sigmaapproxRaw).setMLTypeClean((*muapprox).type());

            (*sigmaapproxRaw).getML() = *muapprox;
            sigInd = regML(sigmaapproxRaw,fnarg,7);

            sigmaapprox = &((*sigmaapproxRaw).getML());

            MLnumbers("&",1) = sigInd;

//RKHSFIXME
            //(*sigmaapprox).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );
        }

        else
        {
            sigmaapprox = muapprox;
            sigInd      = muInd;

            MLnumbers("&",1) = -1; //MLnumbers(zeroint());
        }

        // Kernel transfer (must occur before we make copies etc)

//errstream() << "phantomx: about to transfer kernel: " << kernapprox << "\n";
        if ( kernapprox )
        {
            MercerKernel newkern;

            newkern.setAltCall((*kernapprox).MLid());
            newkern.setType(kxfnum);
//RKHSFIXME
            //newkern.setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );

            if ( kxfnorm )
            {
                newkern.setNormalised();
            }

//errstream() << "phantomx: kernel transfered: " << newkern << "\n";
            (*muapprox).setKernel(newkern);

            if ( sigmuseparate )
            {
                (*sigmaapprox).setKernel(newkern);
            }
        }

        // Record number of points in model for transfer (default: just assume they're from the model we're trying to learn)

        Nbasemu = modelmu_N();
//errstream() << "phantomx 500: Nbasemu = " << Nbasemu << "\n";

        MLnumbers("&",4) = -1;
        MLnumbers("&",5) = -1;

        // Setup helper variables for env-GP transfer learning

        if ( ( tranmeth == 1 ) && Nbasemu )
        {
            retVector<int> tmpva;

//errstream() << "phantomx 501 env-GP?\n";
            indpremu = cntintvec(Nbasemu,tmpva);
            presigweightmu.resize(Nbasemu) = 1.0;

            alpha = alpha0;
            beta  = beta0;

            MEMNEW(srcmodel,ML_Mutable);
            (*srcmodel).setMLTypeClean((*muapprox).type());
            (*srcmodel).getML() = (*muapprox);
            srcmodelInd = regML(srcmodel ,fnarg,10);

//RKHSFIXME
            //(*srcmodel).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );

//errstream() << "phantomx 501 env-GP: type = " << (*srcmodel).type() << "\n";
            MLnumbers("&",4) = srcmodelInd;
        }

        // Setup helper variables for diff-GP transfer learning

        if ( ( tranmeth == 2 ) && Nbasemu )
        {
//errstream() << "phantomx 502 GP-UCB?\n";
            MEMNEW(srcmodel ,ML_Mutable);
            MEMNEW(diffmodel,ML_Mutable);

            (*srcmodel ).setMLTypeClean((*muapprox).type());
            (*diffmodel).setMLTypeClean((*muapprox).type());

            (*srcmodel).getML() = (*muapprox);

//RKHSFIXME
            //(*srcmodel).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );
            //(*diffmodel).getKernel_unsafe().setAssumeReal( ( isProjection == 5 ) ? 0 : 1 );

            srcmodelInd  = regML(srcmodel ,fnarg,10);
            diffmodelInd = regML(diffmodel,fnarg,11);

            MLnumbers("&",4) = srcmodelInd;
            MLnumbers("&",5) = diffmodelInd;
        }

        srcmodInd  = srcmodelInd;
        diffmodInd = diffmodelInd;

        firsttrain = 1;

        // Optimise

//errstream() << "phantomx 1: sigma in muapprox = " << (*muapprox).sigma() << "\n";
        int res = GlobalOptions::realOptim(dim,xres,rawxres,fres,ires,mres,muInd,sigInd,srcmodInd,diffmodInd,allxres,allrawxres,allfres,allfresmod,supres,sscore,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch);

        return res;
    }

    // Model control and use functionality

    virtual int initModelDistr(const Vector<int> &sampleInd, const Vector<gentype> &sampleDist)
    {
        (*muapprox).getKernel_unsafe().setSampleDistribution(sampleDist);
        (*muapprox).getKernel_unsafe().setSampleIndices(sampleInd);
        (*muapprox).resetKernel();

        if ( sigmuseparate )
        {
            (*sigmaapprox).getKernel_unsafe().setSampleDistribution(sampleDist);
            (*sigmaapprox).getKernel_unsafe().setSampleIndices(sampleInd);
            (*sigmaapprox).resetKernel();
        }

        return 1;
    }

    // Model use:
    //
    // model_*: use given function in ML_Base.
    // modelmu_*: specify where differences may exist between mu and sigma models (typically indexing only)
    // modelsigma_*: specify where differences may exist between mu and sigma models (typically indexing only)
    //
    // NB: x() refers to real x, not the "fake" x seen by eg the Bayesian optimiser using a log-scale transform

    SparseVector<gentype> xx; // just use a global here rather than constant calls to constructors and destructors

    double model_sigma(void) const { return (*muapprox).sigma(); }

    const MercerKernel &model_getKernel(void) const { return (*muapprox).getKernel(); }

    template <class S> int model_gg(gentype &resg, const SparseVector<S> &x, const vecInfo *xing = NULL) const 
    { 
//errstream() << "phantomxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 0\n";
        (void) xing; 

        gentype ***pxyprodx = NULL;
        vecInfo *xinf = NULL;
        vecInfo xinfloc;
        gentype xxp;

        if ( xshortcutenabled )
        {
            model_convertx((**thisthisthis).xx,x,0,1);

            if ( !xpweightIsWeight )
            {
                goto bailout;
            }

            NiceAssert( xpweight.size() == xbasis.size() );

            int N = (*muapprox).N();

            (**thisthisthis).xsp.resize(N);

            int i,j;

            while ( xmodprod.numRows() < N )
            {
                i = xmodprod.numRows();

                (**thisthisthis).xmodprod.addRow(i);

                for ( j = 0 ; j < xbasis.size() ; j++ )
                {
                    twoProduct(((**thisthisthis).xmodprod)("&",i,j),(*muapprox).x(i),xbasis(j));
                }
            }

            retVector<gentype> tmpva;

            for ( i = 0 ; i < N ; i++ )
            {
                twoProduct(((**thisthisthis).xsp)("&",i),xmodprod(i,tmpva),xpweight);
            }

            while ( xspp.size() > N )
            {
                i = xspp.size()-1;

                MEMDELARRAY(((**thisthisthis).xspp)("&",i));
                ((**thisthisthis).xspp).remove(i);
            }

            while ( xspp.size() < N )
            {
                i = xspp.size();

                ((**thisthisthis).xspp).add(i);
                MEMNEWARRAY(((**thisthisthis).xspp)("&",i),gentype *,2);
            }

            for ( i = 0 ; i < N ; i++ )
            {
                ((**thisthisthis).xspp)("&",i)[0] = &(((**thisthisthis).xsp)("&",i));
                ((**thisthisthis).xspp)("&",i)[1] = NULL;
            }

            pxyprodx = N ? &(((**thisthisthis).xspp)("&",zeroint())) : NULL;

            for ( i = 0 ; i < xbasis.size() ; i++ )
            {
                for ( j = 0 ; j < xbasis.size() ; j++ )
                {
                    xxp += xpweight(i)*xpweight(j)*xbasisprod(i,j);
                }
            }

            xinf = &((*muapprox).getKernel().getvecInfo(xinfloc,xx,&xxp)); // Can't calculate the inner-product of vectors that aren't actually formed!
        }

        else
        {
bailout:
            model_convertx((**thisthisthis).xx,x);
        }

        ((**thisthisthis).xx).makealtcontent();

        return (*muapprox).gg(resg,(**thisthisthis).xx,xinf,pxyprodx);
    }

    template <class S> int model_gg(Vector<double> &resg, const SparseVector<S> &x, const vecInfo *xing = NULL) const
    {
//errstream() << "phantomxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 1\n";
        (void) xing; 

        gentype ***pxyprodx = NULL;
        vecInfo *xinf = NULL;
        vecInfo xinfloc;
        gentype xxp;

        if ( xshortcutenabled )
        {
            model_convertx((**thisthisthis).xx,x,0,1);

            if ( !xpweightIsWeight )
            {
                goto bailout;
            }

            NiceAssert( xpweight.size() == xbasis.size() );

            int N = (*muapprox).N();

            ((**thisthisthis).xsp).resize(N);

            int i,j;

            while ( xmodprod.numRows() < N )
            {
                i = xmodprod.numRows();

                (**thisthisthis).xmodprod.addRow(i);

                for ( j = 0 ; j < xbasis.size() ; j++ )
                {
                    twoProduct(((**thisthisthis).xmodprod)("&",i,j),(*muapprox).x(i),xbasis(j));
                }
            }

            retVector<gentype> tmpva;

            for ( i = 0 ; i < N ; i++ )
            {
                twoProduct(((**thisthisthis).xsp)("&",i),xmodprod(i,tmpva),xpweight);
            }

            while ( xspp.size() > N )
            {
                i = xspp.size()-1;

                MEMDELARRAY(((**thisthisthis).xspp)("&",i));
                ((**thisthisthis).xspp).remove(i);
            }

            while ( xspp.size() < N )
            {
                i = xspp.size();

                ((**thisthisthis).xspp).add(i);
                MEMNEWARRAY(((**thisthisthis).xspp)("&",i),gentype *,2);
            }

            for ( i = 0 ; i < N ; i++ )
            {
                ((**thisthisthis).xspp)("&",i)[0] = &(((**thisthisthis).xsp)("&",i));
                ((**thisthisthis).xspp)("&",i)[1] = NULL;
            }

            pxyprodx = N ? &(((**thisthisthis).xspp)("&",zeroint())) : NULL;

            for ( i = 0 ; i < xbasis.size() ; i++ )
            {
                for ( j = 0 ; j < xbasis.size() ; j++ )
                {
                    xxp += xpweight(i)*xpweight(j)*xbasisprod(i,j);
                }
            }

            xinf = &((*muapprox).getKernel().getvecInfo(xinfloc,xx,&xxp)); // Can't calculate the inner-product of vectors that aren't actually formed!
        }

        else
        {
bailout:
            model_convertx((**thisthisthis).xx,x);
        }

        ((**thisthisthis).xx).makealtcontent();

        return (*muapprox).gg(resg,(**thisthisthis).xx,0,xinf,pxyprodx);
    }

    template <class S> int model_muvar(gentype &resv, gentype &resmu, const SparseVector<S> &x, const vecInfo *xing = NULL) const
    {
        (void) xing; 

//errstream() << "phantomxqq 0\n";
        gentype ***pxyprodx = NULL;
        gentype **pxyprodxx = NULL;
        vecInfo *xinf = NULL;
        vecInfo xinfloc;
        gentype xxp;

        if ( xshortcutenabled )
        {
//errstream() << "phantomxqq 1\n";
            model_convertx((**thisthisthis).xx,x,0,1);

//errstream() << "phantomxqq 2\n";
            if ( !xpweightIsWeight )
            {
                goto bailout;
            }

            NiceAssert( xpweight.size() == xbasis.size() );

//errstream() << "phantomxqq 3\n";
            int N = (*muapprox).N();

            ((**thisthisthis).xsp).resize(N);

            int i,j;

//errstream() << "phantomxqq 4\n";
            while ( xmodprod.numRows() < N )
            {
                i = xmodprod.numRows();

                (**thisthisthis).xmodprod.addRow(i);

                for ( j = 0 ; j < xbasis.size() ; j++ )
                {
                    twoProduct(((**thisthisthis).xmodprod)("&",i,j),(*muapprox).x(i),xbasis(j));
                }
            }

//errstream() << "phantomxqq 5\n";
            retVector<gentype> tmpva;

            for ( i = 0 ; i < N ; i++ )
            {
                twoProduct(((**thisthisthis).xsp)("&",i),xmodprod(i,tmpva),xpweight);
            }

//errstream() << "phantomxqq 6\n";
            while ( xspp.size() > N )
            {
                i = xspp.size()-1;

                MEMDELARRAY(((**thisthisthis).xspp)("&",i));
                ((**thisthisthis).xspp).remove(i);
            }

//errstream() << "phantomxqq 7\n";
            while ( xspp.size() < N )
            {
                i = xspp.size();

                ((**thisthisthis).xspp).add(i);
                MEMNEWARRAY(((**thisthisthis).xspp)("&",i),gentype *,2);
            }

//errstream() << "phantomxqq 8\n";
            for ( i = 0 ; i < N ; i++ )
            {
                ((**thisthisthis).xspp)("&",i)[0] = &(((**thisthisthis).xsp)("&",i));
                ((**thisthisthis).xspp)("&",i)[1] = NULL;
            }

//errstream() << "phantomxqq 9\n";
            pxyprodx = N ? &(((**thisthisthis).xspp)("&",zeroint())) : NULL;

//errstream() << "phantomxqq 10\n";
            for ( i = 0 ; i < xbasis.size() ; i++ )
            {
                for ( j = 0 ; j < xbasis.size() ; j++ )
                {
                    xxp += xpweight(i)*xpweight(j)*xbasisprod(i,j);
                }
            }

//errstream() << "phantomxqq 11\n";
            xinf = &((*muapprox).getKernel().getvecInfo(xinfloc,xx,&xxp)); // Can't calculate the inner-product of vectors that aren't actually formed!

//errstream() << "phantomxqq 12\n";
            MEMNEWARRAY(pxyprodxx,gentype *,2);
            pxyprodxx[0] = &xxp;
            pxyprodxx[1] = NULL;
//errstream() << "phantomxqq 13\n";
        }

        else
        {
bailout:
//errstream() << "phantomxqq 14: " << x << "\n";
            model_convertx((**thisthisthis).xx,x);
//errstream() << "phantomxqq 15: " << (**thisthisthis).xx << "\n";
        }

        int ires = 0;

//errstream() << "phantomxqq 16\n";
        ((**thisthisthis).xx).makealtcontent();

        if ( !sigmuseparate )
        {
//errstream() << "phantomxqq 17: " << *muapprox << "\n";
            ires = (*muapprox).var(resv,resmu,(**thisthisthis).xx,xinf,pxyprodx,pxyprodxx);
//errstream() << "phantomxqq 18\n";
        }

        else
        {
//errstream() << "phantomxqq 19\n";
            gentype dummy;

//errstream() << "phantomxsmbo 0\n";
            ires  = (*muapprox).gg(resmu,(**thisthisthis).xx,xinf,pxyprodx);
//errstream() << "phantomxqq 20\n";
            ires |= (*sigmaapprox).var(resv,dummy,(**thisthisthis).xx,xinf,pxyprodx,pxyprodxx);
//errstream() << "phantomxqq 21\n";
        }

//errstream() << "phantomxqq 22\n";
        if ( pxyprodxx )
        {
//errstream() << "phantomxqq 23\n";
            MEMDELARRAY(pxyprodxx);
        }
//errstream() << "phantomxqq 24\n";

        return ires;
    }

    template <class S> int model_covar(Matrix<gentype> &resv, const Vector<SparseVector<S> > &x) const 
    {
        Vector<SparseVector<gentype> > xxx; 

        model_convertx(xxx,x); 

        return (*sigmaapprox).covar(resv,xxx);
    }

    template <class S> void model_stabProb(double &res, const SparseVector<S> &x, int p, double pnrm, int rot, double mu, double B) const { model_convertx((**thisthisthis).xx,x); (*muapprox).stabProb(res,(**thisthisthis).xx,p,pnrm,rot,mu,B); return; }

    int model_ggTrainingVector(gentype &resg, int i)        const { return (*muapprox).ggTrainingVector(resg,i); }
    int model_ggTrainingVector(Vector<double> &resg, int i) const { return (*muapprox).ggTrainingVector(resg,i); }

    int model_muvarTrainingVector(gentype &resvar, gentype &resmu, int ivar, int imu) const 
    { 
        int resi = 0;
        gentype dummy;

        if ( !sigmuseparate )
        {
            resi = (*muapprox).varTrainingVector(resvar,resmu,imu);
        }

        else
        {
            gentype dummy;

            resi =  (*muapprox).ggTrainingVector(resmu,imu);
            resi |= (*sigmaapprox).varTrainingVector(resvar,dummy,ivar);
        }

        return resi;
    }

    int model_varTrainingVector(gentype &resv, int i) const 
    { 
        int resi = 0;
        gentype dummy;

        resi |= (*sigmaapprox).varTrainingVector(resv,dummy,i);

        return resi;
    }

    int model_covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const { return (*sigmaapprox).covarTrainingVector(resv,i); }

    void model_stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const { (*muapprox).stabProbTrainingVector(res,i,p,pnrm,rot,mu,B); return; }

    int model_train(int &res, svmvolatile int &killSwitch) 
    {
        int ires = 0;

        ires |= modeldiff_train(res,killSwitch);

        if ( tunediffmod && diffmodel ) 
        {
            tuneKernel((*diffmodel).getML(),xwidth,tunediffmod);
        }

        ires |= modelmu_train(res,killSwitch);
        ires |= modelsigma_train(res,killSwitch);

        if ( tunemu ) 
        { 
            tuneKernel(*muapprox,xwidth,tunemu);
        }

        if ( tunesigma && sigmuseparate ) 
        { 
            tuneKernel(*sigmaapprox,xwidth,tunesigma); 
        }

        return ires;
    }

    int modelsigma_train(int &res, svmvolatile int &killSwitch) 
    {
        int ires = 0;

        if ( sigmuseparate )
        {
            ires = (*sigmaapprox).train(res,killSwitch);
        }

        return ires;
    }

    int model_setd(int imu, int isigma, int nd)
    { 
        int res = (*muapprox).setd(imu,nd);

        if ( sigmuseparate )
        {
            res += (*sigmaapprox).setd(isigma,nd);
        }

        return res;
    }

    int model_setsigmaweight(int imu, int isigma, double nv) 
    { 
        int res = (*muapprox).setsigmaweight(imu,nv);

        if ( sigmuseparate )
        {
            res += (*sigmaapprox).setsigmaweight(isigma,nv);
        }

        return res;
    }

    int modelmu_N(void)    const { return (*muapprox).N();    }
    int modelmu_NNCz(void) const { return (*muapprox).NNC(0); }

    const Vector<gentype> &modelmu_y(void) const
    {
        return (*muapprox).y();
    }

    const Vector<double> &modelmu_xcopy(Vector<double> &resx, int i) const 
    {
        int j,k;

        for ( j = 0 ; j < locires.size() ; j++ )
        {
            if ( i == locires(j) )
            {
                resx.resize(locxres(j).indsize());

                for ( k = 0 ; k < locxres(j).indsize() ; k++ )
                {
                    resx("&",k) = (double) locxres(j).direcref(k);
                }

                return resx;
            }
        }

        //return backconvertx(resx,((*muapprox).x())(i));
        NiceAssert( !isXconvertNonTrivial() );

        const SparseVector<gentype> &tempresx = ((*muapprox).x())(i);

        resx.resize(tempresx.indsize());

        for ( j = 0 ; j < tempresx.indsize() ; j++ )
        {
            resx("&",j) = (double) tempresx.direcref(j);
        }

        return resx;
    }

    const Vector<int> &modelmu_d(void) const { return (*muapprox).d(); }

    int modelmu_setd(int i, int nd) { return (*muapprox).setd(i,nd); }

    int modelmu_setsigmaweight(             int   i,              double   nv) { return (*muapprox).setsigmaweight(i,nv); }
    int modelmu_setsigmaweight(const Vector<int> &i, const Vector<double> &nv) { return (*muapprox).setsigmaweight(i,nv); }

    int modelmu_addTrainingVector(const gentype &y, const gentype &ypred, const SparseVector<gentype> &x)
    {
        int ires = 0;

        //SparseVector<gentype> xx; 

        model_convertx(xx,x);

        ires |= modeldiff_int_addTrainingVector(y,ypred,xx);
        ires |= modelmu_int_addTrainingVector(y,x,xx);

        return ires;
    }

    int modelsigma_N(void) const { return (*sigmaapprox).N(); }

    int modelsigma_setd(int i, int nd) { return (*sigmaapprox).setd(i,nd); }

    int modelsigma_setsigmaweight(             int   i,              double   nv) { return (*sigmaapprox).setsigmaweight(i,nv); }
    int modelsigma_setsigmaweight(const Vector<int> &i, const Vector<double> &nv) { return (*sigmaapprox).setsigmaweight(i,nv); }

    int modelsigma_addTrainingVector(const gentype &y, const SparseVector<gentype> &x)
    {
        //SparseVector<gentype> xx; 

        model_convertx(xx,x);

        return modelsigma_int_addTrainingVector(y,xx);
    }



    // Simple default-model adjustments

    int default_model_settspaceDim(int nv) { return altfnapproxmoo.settspaceDim(nv); }
    int default_model_setsigma(double nv)  { 
return altfnapprox.setsigma(nv) | altfnapproxmoo.setsigma(nv); }

    int default_model_setkernelg(const gentype &nv)
    {
        int res = 0;
        int lockernnum = 0;

        Vector<gentype> kernRealConstsa(altfnapprox.getKernel().cRealConstants(lockernnum));
        Vector<gentype> kernRealConstsb(altfnapproxmoo.getKernel().cRealConstants(lockernnum));

        if ( kernRealConstsa(zeroint()) != nv )
	{
            kernRealConstsa("&",zeroint()) = nv;

            altfnapprox.getKernel_unsafe().setRealConstants(kernRealConstsa,lockernnum);
            altfnapprox.resetKernel(0);
	}

        if ( kernRealConstsb(zeroint()) != nv )
	{
            kernRealConstsb("&",zeroint()) = nv;

            altfnapproxmoo.getKernel_unsafe().setRealConstants(kernRealConstsb,lockernnum);
            altfnapproxmoo.resetKernel(0);
	}

        return res;
    }

    int default_model_setkernelgg(const SparseVector<gentype> &nv)
    {
        int res = 0;

        altfnapprox.getKernel_unsafe().setScale(nv);
        altfnapprox.resetKernel(0);

        altfnapproxmoo.getKernel_unsafe().setScale(nv);
        altfnapproxmoo.resetKernel(0);
//errstream() << "phantomx 0: " << nv << "\n";

        return res;
    }





    // Work out frequentist certainty - see "Adaptive and Safe Bayesian Optimization in High Dimensions via One-Dimentional Subspaces"

    double model_err(int dim, const Vector<double> &xmin, const Vector<double> &xmax, svmvolatile int &killSwitch);

private:

    int modelmu_int_addTrainingVector(const gentype &y, const SparseVector<gentype> &x, const SparseVector<gentype> &xx)
    {
        locires.add(locires.size()); locires("&",locires.size()-1) = (*muapprox).N();
        locxres.add(locxres.size()); locxres("&",locxres.size()-1) = x;

        SparseVector<gentype> xxx(xx);

        addtemptox(xxx,xtemplate);

        return (*muapprox).qaddTrainingVector((*muapprox).N(),y,xxx);
    }

    int modelsigma_int_addTrainingVector(const gentype &y, const SparseVector<gentype> &xx)
    {
        NiceAssert( sigmuseparate );

        SparseVector<gentype> xxx(xx);

        addtemptox(xxx,xtemplate);

        return (*sigmaapprox).addTrainingVector((*sigmaapprox).N(),y,xxx); 
    }

    int modeldiff_int_addTrainingVector(const gentype &y, const gentype &ypred, const SparseVector<gentype> &xx)
    {
        (void) ypred;

        int ires = 0;

        if ( ( tranmeth == 1 ) && Nbasemu )
        {
            if ( firsttrain && srcmodel && tunesrcmod ) 
            { 
outstream() << "Tuning source model\n";
                tuneKernel((*srcmodel).getML(),xwidth,tunesrcmod); 
                firsttrain = 0; 
                int dummy = 0;
                (*srcmodel).train(dummy);
            }

            (const_cast<SparseVector<gentype> &>(xx)).makealtcontent();

            (*srcmodel).gg(predval,xx);

            resdiff =  y;
            resdiff -= predval;

outstream() << "beta := " << beta << " + ( " << y << " - " << predval << " )^2 = ";
            beta += ((double) norm2(resdiff))/2.0;
outstream() << beta << "\n";
        }

        if ( ( tranmeth == 2 ) && Nbasemu )
        {
            if ( firsttrain && srcmodel && tunesrcmod ) 
            { 
                tuneKernel((*srcmodel).getML(),xwidth,tunesrcmod); 
                firsttrain = 0; 
                int dummy = 0;
                (*srcmodel).train(dummy);
            }

            // Predict y based on target (and noise)

            (const_cast<SparseVector<gentype> &>(xx)).makealtcontent();

            //(*srcmodel).gg(predval,xx);
            (*srcmodel).var(storevar,predval,xx);

            // Calculate difference between reality and model

            diffval =  y;
            diffval -= predval;

            // How noisy is the difference?

            double sigmaval;

            sigmaval =  ((*muapprox).sigma()); // variance of observation
            sigmaval += (double) storevar; // variance from source model

            // Add to difference model

            double sigmaweight = sigmaval/((*diffmodel).sigma());
            double Cweight = 1/sigmaweight;

            SparseVector<gentype> xxx(xx);

            addtemptox(xxx,xtemplate);

            ires |= (*diffmodel).addTrainingVector((*diffmodel).N(),diffval,xxx,Cweight);
        }

        return ires;
    }

    int modeldiff_train(int &res, svmvolatile int &killSwitch) 
    {
        int ires = 0;

        if ( ( tranmeth == 1 ) && Nbasemu )
        {
            if ( firsttrain && srcmodel && tunesrcmod ) 
            { 
                tuneKernel((*srcmodel).getML(),xwidth,tunesrcmod); 
                firsttrain = 0; 
                int dummy = 0;
                (*srcmodel).train(dummy);
            }

            int Nmodel = (*muapprox).N();

            alpha = alpha0 + ((Nmodel-Nbasemu)/2.0);
            //beta updated incrementally

            modelmu_setsigmaweight(indpremu,( presigweightmu = beta/(alpha+1) ));
outstream() << "sigma = (" << beta << "/(" << alpha << "+1)) = " << beta/(alpha+1) << "\n";
        }

        if ( ( tranmeth == 2 ) && Nbasemu )
        {
            if ( firsttrain && srcmodel && tunesrcmod ) 
            { 
                tuneKernel((*srcmodel).getML(),xwidth,tunesrcmod); 
                firsttrain = 0; 
                int dummy = 0;
                (*srcmodel).train(dummy);
            }

            int i;

            // Train difference model

            ires = (*diffmodel).train(res,killSwitch);

            // Update y and sigma in muapprox

            for ( i = 0 ; i < Nbasemu ; i++ )
            {
                // Calc predicted difference and difference varian

                (*diffmodel).var(storevar,diffval,(*srcmodel).x(i));

                // Calculate bias corrected source y

                predval =  (*srcmodel).y()(i);
                predval += diffval;

                // Calculate bias corrected source y variance

                double sigmaval;

                sigmaval =  ((*srcmodel).sigma())*((*srcmodel).sigmaweight()(i));
                sigmaval += (double) storevar;

                // Set bias corrected source y and variance in muapprox

                (*muapprox).sety(i,predval);
                (*muapprox).setsigmaweight(i,(sigmaval/((*muapprox).sigma())));
            }
        }

        return ires;
    }

    int modelmu_train(int &res, svmvolatile int &killSwitch) 
    {
        return (*muapprox).train(res,killSwitch);
    }

    // Generic kernel tuning function
    //
    // xwidth: maximum length-scale
    // method: 1 = max-likelihood
    //         2 = loo error
    //         3 = recall
    //
    // Note that this is very basic.  It tunes continuous variables
    // only, parameter bounds are arbitrary, and grid sizes fixed
    //
    // Returns best error

    double tuneKernel(ML_Base &model, double xwidth, int method);

    // Models in use

public:
    ML_Base *muapprox;
    ML_Base *sigmaapprox;
private:

    // Default models

    GPR_Scalar altfnapprox;
    GPR_Scalar altfnapproxFNapprox;
    GPR_Vector altfnapproxmoo;

    // Local store for x vectors

    Vector<int> locires;
    Vector<SparseVector<gentype> > locxres;

    // Optimiser for model_err calculation
    //
    // JIT allocation, DIRect by default, set src to use alternative

    GlobalOptions *modelErrOptim;
    int ismodelErrLocal;

    GlobalOptions &getModelErrOptim(GlobalOptions *src = NULL) const
    {
        if ( src )
        {
            (**thisthisthis).killModelErrOptim();
        }

        if ( !modelErrOptim )
        {
            if ( !src )
            {
                (**thisthisthis).ismodelErrLocal = 1;

                MEMNEW((**thisthisthis).modelErrOptim,DIRectOptions);

                NiceAssert( modelErrOptim );

                *((**thisthisthis).modelErrOptim) = static_cast<const GlobalOptions &>(*this);
            }

            else
            {
                (**thisthisthis).ismodelErrLocal = 0;
                (**thisthisthis).modelErrOptim   = src;
            }
        }

        return *((**thisthisthis).modelErrOptim);
    }

    void killModelErrOptim(void)
    {
        if ( modelErrOptim )
        {
            if ( ismodelErrLocal )
            {
                MEMDEL(modelErrOptim);
            }

            modelErrOptim   = NULL;
            ismodelErrLocal = 1;
        }

        return;
    }

    SMBOOptions *thisthis;
    SMBOOptions **thisthisthis;
};

#endif



