
//
// Global optimisation setup/run helper functions
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "uniopt.h"
#include "gpr_vector.h"
#include "imp_expect.h"
#include "imp_parsvm.h"

GlobalOptions *setupGridOpt(int numZooms,
                            double zoomFact,
                            double maxtraintime)
{
    GridOptions *locres;

    MEMNEW(locres,GridOptions);

    locres->numZooms = numZooms;
    locres->zoomFact = zoomFact;

    GlobalOptions *res = static_cast<GlobalOptions *>(locres);

    res->maxtraintime = maxtraintime;
    res->storeRes     = 1; // currently this is required

    return res;
}

GlobalOptions *setupDirectOpt(long int maxits,
                              long int maxevals,
                              double eps,
                              direct_algorithm algorithm,
                              double maxtraintime)
{
    DIRectOptions *locres;

    MEMNEW(locres,DIRectOptions);

    locres->maxits    = maxits;
    locres->maxevals  = maxevals;
    locres->eps       = eps;
    locres->algorithm = algorithm;

    GlobalOptions *res = static_cast<GlobalOptions *>(locres);

    res->maxtraintime = maxtraintime;
    res->storeRes     = 1; // currently this is required

    return res;
}

GlobalOptions *setupBayesOptI(int method,
                              ML_Base *fnapprox,
                              IMP_Generic *impmeasu,
                              int startpoints,
                              unsigned int totiters,
                              double ztol,
                              double delta,
                              double nu,
                              double maxtraintime)
{
    BayesOptions *locres;

//    MEMNEW(locres,BayesOptions(fnapprox,impmeasu));
//FIXME
MEMNEW(locres,BayesOptions());

    // Bayes

    locres->method      = method;
    locres->startpoints = startpoints;
    locres->totiters    = totiters;
    locres->ztol        = ztol;
    locres->delta       = delta;
    locres->nu          = nu;

    GlobalOptions *res = static_cast<GlobalOptions *>(locres);

    // Global

    res->maxtraintime = maxtraintime;
    res->storeRes     = 1; // currently this is required

    return res;
}


GlobalOptions *setupBayesOptII(int method,
                               double fnsigma,
                               double fnkernr,
                               int startpoints,
                               unsigned int totiters,
                               double ztol,
                               double delta,
                               double nu,
                               long int maxits,
                               long int maxevals,
                               double eps,
                               direct_algorithm algorithm,
                               double maxtraintime)
{
    GPR_Scalar *fnapprox;

    MEMNEW(fnapprox,GPR_Scalar);

    fnapprox->setsigma(fnsigma);
    fnapprox->getKernel_unsafe().setRealConstZero(fnkernr);
    fnapprox->resetKernel();

    return setupBayesOptI(method,fnapprox,NULL,
                          startpoints,totiters,
                          ztol,delta,nu,
                          maxits,maxevals,eps,algorithm,
                          maxtraintime);
}

GlobalOptions *setupBayesOptIII(int odim,
                                double fnsigma,
                                double fnkernr,
                                int ehimethod,
                                int startpoints,
                                unsigned int totiters,
                                double ztol,
                                double delta,
                                double nu,
                                long int maxits,
                                long int maxevals,
                                double eps,
                                direct_algorithm algorithm,
                                double maxtraintime,
                                double traintimeoverride)
{
    GPR_Vector *fnapprox;

    MEMNEW(fnapprox,GPR_Vector);

    fnapprox->settspaceDim(odim);
    fnapprox->setsigma(fnsigma);
    fnapprox->getKernel_unsafe().setRealConstZero(fnkernr);
    fnapprox->resetKernel();

    IMP_Expect *impmeasu;

    MEMNEW(impmeasu,IMP_Expect);

    impmeasu->setehimethod(ehimethod);

    return setupBayesOptI(0,
                          fnapprox,
                          impmeasu,
                          startpoints,
                          totiters,
                          ztol,
                          delta,
                          nu,
                          maxits,
                          maxevals,
                          eps,
                          algorithm,
                          maxtraintime,
                          traintimeoverride);
}

GlobalOptions *setupBayesOptIV(int odim,
                               double fnsigma,
                               double fnkernr,
                               double impkernr,
                               int svmmethod,
                               int startpoints,
                               unsigned int totiters,
                               double ztol,
                               double delta,
                               double nu,
                               long int maxits,
                               long int maxevals,
                               double eps,
                               direct_algorithm algorithm,
                               double maxtraintime)
{
    GPR_Vector *fnapprox;

    MEMNEW(fnapprox,GPR_Vector);

    fnapprox->settspaceDim(odim);
    fnapprox->setsigma(fnsigma);
    fnapprox->getKernel_unsafe().setRealConstZero(fnkernr);
    fnapprox->resetKernel();

    IMP_ParSVM *impmeasu;

    MEMNEW(impmeasu,IMP_ParSVM);

    impmeasu->getKernel_unsafe().setRealConstZero(impkernr);
    impmeasu->resetKernel();
//    impmeasu->setsvmmethod(svmmethod);
(void) svmmethod;

    return setupBayesOptI(0,fnapprox,impmeasu,
                          startpoints,totiters,
                          ztol,delta,nu,
                          maxits,maxevals,eps,algorithm,
                          maxtraintime);
}

void deleteOptim(GlobalOptions *opt)
{
    if ( opt->optdefed() == 3 )
    {
        // This is Bayesian optimisation, so we may need to delete fnapprox
        // and impmeasu if they are non-NULL

        BayesOptions *locopt = dynamic_cast<BayesOptions *>(opt);

//FIXME
//        if ( locopt->fnapproxNonLocal() )
//        {
//            MEMDEL(locopt->fnapprox);
//        }

        if ( locopt->impmeasuNonLocal() )
        {
            MEMDEL(locopt->impmeasu);
        }
    }

    MEMDEL(opt);

    return;
}




void locfneval(gentype &res, Vector<gentype> &x, void *arg)
{
    double *locx                                    =  ((double *)                                   (((void **) arg)[0]) );
    double *locf                                    =  ((double *)                                   (((void **) arg)[1]) );
    void (*locfn)(double *, const double *, void *) =  ((void (*)(double *, const double *, void *)) (((void **) arg)[2]) );
    void *locarg                                    =  ((void *)                                     (((void **) arg)[3]) );
    int &idim                                       = *((int *)                                      (((void **) arg)[4]) );
    int &odim                                       = *((int *)                                      (((void **) arg)[5]) );

    int j;

    NiceAssert( x.size() == idim );

    for ( j = 0 ; j < idim ; j++ )
    {
        locx[j] = (double) x(j);
    }

    (*locfn)(locf,locx,locarg);

    if ( odim )
    {
        res.force_vector(odim);

        for ( j = 0 ; j < odim ; j++ )
        {
            (res.dir_vector())("&",j) = locf[j];
        }
    }

    else
    {
        res.force_double() = locf[0];
    }

    return;
}

int uniOpt(GlobalOptions *optDef,
           OptProb *probDef,
           double *xres,
           double *fres,
           double ***allxres,
           double ***allfres,
           int &Nres,
           svmvolatile int &killSwitch)
{
    NiceAssert( optDef );
    NiceAssert( probDef );
    NiceAssert( xres );
    NiceAssert( fres );

    Vector<gentype> locxres(probDef->idim);
    gentype locfres;
    int locires = 0;

    Vector<Vector<gentype> > locallxres;
    Vector<gentype> locallfres;
    Vector<gentype> locallfresmod;

    double *locx;
    double *locf;

    MEMNEWARRAY(locx,double,probDef->idim);
    MEMNEWARRAY(locf,double,(probDef->odim ? probDef->odim : 1));

    void **loclocfnarg;

    MEMNEWARRAY(loclocfnarg,void *,6);

    loclocfnarg[0] = (void *) locx;
    loclocfnarg[1] = (void *) locf;
    loclocfnarg[2] = (void *) probDef->fn;
    loclocfnarg[3] = (void *) probDef->fnarg;
    loclocfnarg[4] = (void *) &(probDef->idim);
    loclocfnarg[5] = (void *) &(probDef->odim);

    void *locfnarg = (void *) loclocfnarg;

    Vector<gentype> supresdummy;
    Vector<double> locsscoredummy;

    int res = optDef->optim(probDef->idim,
                            locxres,
                            locfres,
                            locires,
                            locallxres,
                            locallfres,
                            locallfresmod,
                            supresdummy,
                            locsscoredummy,
                            probDef->xmin,
                            probDef->xmax,
                            probDef->numpts,
                            probDef->distMode,
                            probDef->varsType,
                            locfneval,
                            locfnarg,
                            killSwitch);

    MEMDELARRAY(locx);
    MEMDELARRAY(locf);

    MEMDELARRAY(loclocfnarg);

    int i,j;

    for ( j = 0 ; j < probDef->idim ; j++ )
    {
        xres[j] = (double) locxres(j);
    }

    if ( probDef->odim == 0 )
    {
        fres[0] = (double) locfres;
    }

    NiceAssert( locallxres.size() == locallfres.size() );

    Nres = locallxres.size();

    if ( allxres )
    {
        MEMNEWARRAY(*allxres,double *,Nres);

        for ( i = 0 ; i < Nres ; i++ )
        {
            MEMNEWARRAY((*allxres)[i],double,probDef->idim);

            for ( j = 0 ; j < probDef->idim ; j++ )
            {
                (*allxres)[i][j] = (double) locallxres(i)(j);
            }
        }
    }

    if ( allfres )
    {
        MEMNEWARRAY(*allfres,double *,Nres);

        for ( i = 0 ; i < Nres ; i++ )
        {
            MEMNEWARRAY((*allfres)[i],double,((probDef->odim) ? (probDef->odim) : 1) + (probDef->idim));

            if ( probDef->odim )
            {
                NiceAssert( locallfres(i).isValVector() );

                for ( j = 0 ; j < probDef->odim ; j++ )
                {
                    (*allfres)[i][j] = (double) ((locallfres(i).cast_vector())(j));
                }
            }

            else
            {
                (*allfres)[i][0] = (double) locallfres(i);
            }
        }
    }

    return res;
}

