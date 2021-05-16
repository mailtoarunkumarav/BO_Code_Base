
//
// Sequential model-based optimisation base class
//
// Date: 12/02/2019
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "smboopt.h"


double calcLCB(int dim, const double *x, void *arg);
double calcUCB(int dim, const double *x, void *arg);

double calcLCB(int dim, const double *x, void *arg)
{
    SMBOOptions &caller      = *((SMBOOptions *)          (((void **) arg)[0]));
    SparseVector<double> &xx = *((SparseVector<double> *) (((void **) arg)[1]));
    gentype &mu              = *((gentype *)              (((void **) arg)[2]));
    gentype &sigmasq         = *((gentype *)              (((void **) arg)[3]));

    int i;

    for ( i = 0 ; i < dim ; i++ )
    {
        xx("&",i) = x[i];
    }

    caller.model_muvar(sigmasq,mu,xx);

    return -((double) mu) - sqrt((double) sigmasq);
}

double calcUCB(int dim, const double *x, void *arg)
{
    SMBOOptions &caller      = *((SMBOOptions *)          (((void **) arg)[0]));
    SparseVector<double> &xx = *((SparseVector<double> *) (((void **) arg)[1]));
    gentype &mu              = *((gentype *)              (((void **) arg)[2]));
    gentype &sigmasq         = *((gentype *)              (((void **) arg)[3]));

    int i;

    for ( i = 0 ; i < dim ; i++ )
    {
        xx("&",i) = x[i];
    }

    caller.model_muvar(sigmasq,mu,xx);

    return -((double) mu) + sqrt((double) sigmasq);
}

void altcalcLCB(gentype &res, Vector<gentype> &x, void *arg);
void altcalcUCB(gentype &res, Vector<gentype> &x, void *arg);

void altcalcLCB(gentype &res, Vector<gentype> &x, void *arg)
{
    SMBOOptions &caller      = *((SMBOOptions *)          (((void **) arg)[0]));
    SparseVector<double> &xx = *((SparseVector<double> *) (((void **) arg)[1]));
    gentype &mu              = *((gentype *)              (((void **) arg)[2]));
    gentype &sigmasq         = *((gentype *)              (((void **) arg)[3]));

    int i;
    int dim = x.size();

    for ( i = 0 ; i < dim ; i++ )
    {
        xx("&",i) = (double) x(i);
    }

    caller.model_muvar(sigmasq,mu,xx);

    res.force_double() = -((double) mu) - sqrt((double) sigmasq);

    return;
}

void altcalcUCB(gentype &res, Vector<gentype> &x, void *arg)
{
    SMBOOptions &caller      = *((SMBOOptions *)          (((void **) arg)[0]));
    SparseVector<double> &xx = *((SparseVector<double> *) (((void **) arg)[1]));
    gentype &mu              = *((gentype *)              (((void **) arg)[2]));
    gentype &sigmasq         = *((gentype *)              (((void **) arg)[3]));

    int i;
    int dim = x.size();

    for ( i = 0 ; i < dim ; i++ )
    {
        xx("&",i) = (double) x(i);
    }

    caller.model_muvar(sigmasq,mu,xx);

    res.force_double() = -((double) mu) + sqrt((double) sigmasq);

    return;
}

double SMBOOptions::model_err(int dim, const Vector<double> &xmin, const Vector<double> &xmax, svmvolatile int &killSwitch)
{
    SparseVector<double> xx;
    gentype mu;
    gentype sigmasq;

    void *modelarg[4];

    modelarg[0] = (void *) this;
    modelarg[1] = (void *) &xx;
    modelarg[2] = (void *) &mu;
    modelarg[3] = (void *) &sigmasq;

    gentype minLCB(0.0); // min_x mu(x) - sigma(x)
    gentype minUCB(0.0); // min_x mu(x) + sigma(x)

    Vector<gentype> xdummy;
    int idummy;
    Vector<Vector<gentype> > allxdummy;
    Vector<gentype> allfdummy;
    Vector<gentype> allfmoddummy;
    Vector<gentype> allsupdummy;
    Vector<double> allsdummy;
    Vector<gentype> altxmin;
    Vector<gentype> altxmax;

    altxmin.castassign(xmin);
    altxmax.castassign(xmax);

//    DIRectOptions &dopts = getModelErrOptim();
//    static_cast<GlobalOptions &>(dopts) = static_cast<const GlobalOptions &>(*this);
//
//    int dresa = directOpt(dim,xdummy,minLCB,xmin,xmax,calcLCB,(void *) modelarg,dopts,killSwitch);
//    int dresb = directOpt(dim,xdummy,minUCB,xmin,xmax,calcUCB,(void *) modelarg,dopts,killSwitch);

    GlobalOptions &dopts = getModelErrOptim();

    int dresa = dopts.optim(dim,xdummy,minLCB,idummy,allxdummy,allfdummy,allfmoddummy,allsupdummy,allsdummy,altxmin,altxmax,altcalcLCB,modelarg,killSwitch);
    int dresb = dopts.optim(dim,xdummy,minUCB,idummy,allxdummy,allfdummy,allfmoddummy,allsupdummy,allsdummy,altxmin,altxmax,altcalcUCB,modelarg,killSwitch);

    errstream() << "Model error calculation using DIRect: " << dresa << "," << dresb << "\n";

    return ((double) minUCB)-((double) minLCB);
}


double SMBOOptions::tuneKernel(ML_Base &model, double xwidth, int method)
{
//errstream() << "phantomx 000: " << model << "\n";
    if ( !model.N() )
    {
        return 1;
    }

    MercerKernel &kernel = model.getKernel_unsafe();
    int kdim = kernel.size();

    int i,j;

    // Gather data on kernel

    int ddim = 0;
    int adim = 1;

    Vector<int> kind; // which element in kernel dictionary
    Vector<int> kelm; // which element in cRealConstants
    Vector<double> kmin; // range minimum
    Vector<double> kmax; // range maximum
    Vector<int> kstp; // number of steps over range
    Vector<Vector<gentype> > constVecs(kdim);

    if ( kdim )
    {
        for ( i = 0 ; i < kdim ; i++ )
        {
            constVecs("&",i) = kernel.cRealConstants(i);

            if ( constVecs(i).size() )
            {
                for ( j = 0 ; j < constVecs(i).size() ; j++ )
                {
                    double lb;
                    double ub;
                    int steps;
                    int addit = 0;

                    // Fixme: currently basically do lengthscale (r0) for "normal" kernels, need to extend

                    if ( ( kernel.cType(i) == 5 ) && j )
                    {
                        // This is norm order

                        lb    = 1;
                        ub    = 5;
                        steps = 6;
                        addit = 1;
                    }

                    else if ( ( kernel.cType(i) == 48 ) && j )
                    {
                        // This is inter-task relatedness

                        lb    = 0;
                        ub    = 1;
                        steps = 20;
                        addit = 1;
                    }

                    else if ( ( kernel.cType(i) < 800 ) && ( kernel.cType(i) != 0 ) && ( kernel.cType(i) != 48 ) && !j )
                    {
                        // This is length-scale, always, with the single exception of kernels 0 and 48 where lengthscale is meaningless

                        lb    = 1e-2*xwidth; 
                        ub    = 0.7*xwidth; 
                        steps = 20; 
                        addit = 1;
                    }

                    if ( addit )
                    {
                        kind.add(ddim); kind("&",ddim) = i;
                        kelm.add(ddim); kelm("&",ddim) = j;
                        kmin.add(ddim); kmin("&",ddim) = lb;
                        kmax.add(ddim); kmax("&",ddim) = ub;
                        kstp.add(ddim); kstp("&",ddim) = steps;

                        ddim++;
                        adim *= steps;
                    }
                }
            }
        }
    }

    double bestres = 1;

    if ( ddim )
    {
        // setup step grid

        Vector<int> pointspec(ddim);
        Vector<Vector<int> > stepgrid(adim);

        stepgrid = pointspec;

        for ( i = 0 ; i < adim ; i++ )
        {
            if ( !i )
            {
                stepgrid("&",i) = zeroint();
            }

            else
            {
                stepgrid("&",i) = stepgrid(i-1);

                for ( j = 0 ; j < ddim ; j++ )
                {
                    stepgrid("&",i)("&",j)++;

                    if ( stepgrid(i)(j) < kstp(j) )
                    {
                        break;
                    }

                    else
                    {
                        stepgrid("&",i)("&",j) = zeroint();
                    }
                }
            }
        }

        // Work out results on all of grid

        Vector<double> gridres(adim);

        gridres = 0.0;

        int dummy;

        for ( i = 0 ; i < adim ; i++ )
        {
            for ( j = 0 ; j < ddim ; j++ )
            {
                constVecs("&",kind(j))("&",kelm(j)) = kmin(kind(j))+((kmax(kind(j))-kmin(kind(j)))*stepgrid(i)(j)/((double) kstp(kind(j))-1));
            }

            for ( j = 0 ; j < kdim ; j++ )
            {
                kernel.setRealConstants(constVecs(j),j);
            }

            model.resetKernel();
//errstream() << "phantomxxxqqq\n";
            model.train(dummy);

            if ( method == 1 )
            {
                gridres("&",i) = calcnegloglikelihood(model,1);
//errstream() << "NLL";
            }

            else if ( method == 2 )
            {
                gridres("&",i) = calcLOO(model,0,1);
//errstream() << "LOO = " << gridres(i) << " for legnthscale " << constVecs << "\n";
            }

            else if ( method == 3 )
            {
                gridres("&",i) = calcRecall(model,0,1);
//errstream() << "REC";
            }
        }

        // Find best result index

        int bestind;

        bestres = min(gridres,bestind);

        // Set kernel params to best result

errstream() << "tune " << bestind << "," << bestres << ": ";
        i = bestind;
        {
            for ( j = 0 ; j < ddim ; j++ )
            {
                constVecs("&",kind(j))("&",kelm(j)) = kmin(kind(j))+((kmax(kind(j))-kmin(kind(j)))*stepgrid(i)(j)/((double) kstp(kind(j))-1));
            }

errstream() << "LOO goodset " << constVecs << "\n";
            for ( j = 0 ; j < kdim ; j++ )
            {
                kernel.setRealConstants(constVecs(j),j);
            }

            model.resetKernel();
            model.train(dummy);
        }
    }

    return bestres;
}
