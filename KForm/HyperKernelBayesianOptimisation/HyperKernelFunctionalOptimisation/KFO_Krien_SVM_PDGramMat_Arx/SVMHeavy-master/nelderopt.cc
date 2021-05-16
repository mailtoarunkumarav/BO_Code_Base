
//
// Nelder-Mead optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "nelderopt.h"
#include <math.h>
#include <stddef.h>
#include <iostream>
#include "neldermead.h"



int nelderOpt(int dim,
              Vector<gentype> &xres,
              gentype &fres,
              int &ires,
              Vector<Vector<gentype> > &allxres,
              Vector<gentype> &allfres,
              Vector<gentype> &supres,
              const Vector<gentype> &xmin,
              const Vector<gentype> &xmax,
              void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
              void *fnarg,
              const NelderOptions &nopts,
              svmvolatile int &killSwitch);

int nelderOpt(int dim,
              Vector<double> &xres,
              gentype &fres,
              const Vector<double> &xmin,
              const Vector<double> &xmax,
              double (*fn)(int n, const double *x, void *arg),
              void *fnarg,
              const NelderOptions &nopts,
              svmvolatile int &force_stop);





double tst_obj(unsigned int n, const double *x, double *undefined_flag, void *fnarginnerdr);
double tst_obj(unsigned int n, const double *x, double *undefined_flag, void *fnarginnerdr)
{
    int    &force_stop = *((int    *) (((void **) fnarginnerdr)[2]));
    double &hardmin    = *((double *) (((void **) fnarginnerdr)[3]));
    double *xx         =   (double *)  ((void **) fnarginnerdr)[4];
    double &hardmax    = *((double *) (((void **) fnarginnerdr)[5]));

    unsigned int i;

    for ( i = 0 ; i < n ; i++ )
    {
        xx[i] = x[i];
    }

    (void) undefined_flag;

    void **fnarginner = (void **) fnarginnerdr;

    double (*fn)(int, const double *, void *) = (double (*)(int, const double *, void *)) fnarginner[0];
    void *fnarg = fnarginner[1];

    double res = (*fn)(n,xx,fnarg);

    if ( res <= hardmin )
    {
        // Trigger early termination if hardmin reached

        force_stop = 1;
    }

    if ( res >= hardmax )
    {
        // Trigger early termination if hardmax reached

        force_stop = 1;
    }

    return res;
}


int nelderOpt(int dim,
              Vector<double> &xres,
              gentype &fres,
              const Vector<double> &xmin,
              const Vector<double> &xmax,
              double (*fn)(int n, const double *x, void *arg),
              void *fnarg,
              const NelderOptions &nopts,
              svmvolatile int &force_stop)
{
    NiceAssert( dim > 0 );
    NiceAssert( xmin.size() == dim );
    NiceAssert( xmax.size() == dim );
    NiceAssert( xmax >= xmin );

    xres.resize(dim);

    double *x = &xres("&",zeroint());
    const double *l = &xmin(zeroint());
    const double *u = &xmax(zeroint());

    double *xx;

    MEMNEWARRAY(xx,double,dim);

    void *fnarginner[6];

    fnarginner[0] = (void *) fn;
    fnarginner[1] = (void *) fnarg;
    fnarginner[4] = (void *) xx;

    void *fnarginnerdr = (void *) fnarginner;

    nlopt_stopping optcond;
    double *xtol_abs;

    MEMNEWARRAY(xtol_abs,double,dim);

    optcond.n          = dim;
    optcond.minf_max   = nopts.minf_max;
    optcond.ftol_rel   = nopts.ftol_rel;
    optcond.ftol_abs   = nopts.ftol_abs;
    optcond.xtol_rel   = nopts.xtol_rel;
    optcond.xtol_abs   = xtol_abs;
    optcond.nevals     = 0;
    optcond.maxeval    = nopts.maxeval;
    optcond.maxtime    = nopts.maxtraintime;
    optcond.force_stop = &force_stop;
    double hardmin     = nopts.hardmin;
    double hardmax     = nopts.hardmax;

    fnarginner[2] = (void *) &force_stop;
    fnarginner[3] = (void *) &hardmin;
    fnarginner[5] = (void *) &hardmax;

    // Assume same xtol_abs for all components (if you want different you can rescale x components externally)

    int i;

    for ( i = 0 ; i < dim ; i++ ) 
    {
        xtol_abs[i] = nopts.xtol_abs;
    }

    // Create initial step using nlopt method

    double *xstep;

    MEMNEWARRAY(xstep,double,dim);

    for ( i = 0 ; i < dim ; i++ ) 
    {
         double step = HUGE_VAL;

        if ( !testisinf(u[i]) && !testisinf(l[i]) && ( (u[i]-l[i])*0.25 < step ) && ( u[i] > l[i] ) )
        {
            step = (u[i]-l[i])*0.25;
        }

        if (!testisinf(u[i]) && ( u[i]-x[i] < step ) && ( u[i] > x[i] ) )
        {
            step = (u[i]-x[i])*0.75;
        }

        if ( !testisinf(l[i]) && ( x[i]-l[i] < step ) && ( x[i] > l[i] ) )
        {
            step = (x[i]-l[i])*0.75;
        }

        if ( testisinf(step) ) 
        {
            if ( !testisinf(u[i]) && ( abs2(u[i]-x[i]) < abs2(step) ) )
            {
                step = (u[i]-x[i])*1.1;
            }

            if ( !testisinf(l[i]) && ( abs2(x[i]-l[i]) < abs2(step) ) )
            {
                step = (x[i]-l[i])*1.1;
            }
        }

        if ( testisinf(step) || ( step == 0 ) ) 
        {
            step = x[i];
        }

        if ( testisinf(step) || ( step == 0 ) )
        {
            step = 1;
        }

        xstep[i] = step;
    }

    int intres;

    if ( nopts.method )
    {
        errstream() << "Nelder-Mead Optimisation Initiated:\n";

        intres = nldrmd_minimize(dim,tst_obj,fnarginnerdr,
                                 l,u,
                                 x,&(fres.force_double()),
                                 xstep,
                                 &optcond);

        errstream() << "Nelder-Mead Optimisation Ended.\n";
    }

    else
    {
        errstream() << "Subplex Optimisation Initiated:\n";

        intres = sbplx_minimize(dim,tst_obj,fnarginnerdr,
                                l,u,
                                x,&(fres.force_double()),
                                xstep,
                                &optcond);

        errstream() << "Subplex Optimisation Ended.\n";
    }

    MEMDELARRAY(xstep);
    MEMDELARRAY(xtol_abs);
    MEMDELARRAY(xx);

    return intres;
}


double fninnerdd(int dim, const double *x, void *arg);
double fninnerdd(int dim, const double *x, void *arg)
{
    Vector<gentype> &xx                                    = *((Vector<gentype> *)          (((void **) arg)[0]));
    void (*fn)(gentype &res, Vector<gentype> &, void *arg) = ( (void (*)(gentype &, Vector<gentype> &, void *arg))  (((void **) arg)[1]) );
    void *arginner                                         = ((void *)                      (((void **) arg)[2]));
    int &ires                                              = *((int *)                      (((void **) arg)[3]));
    Vector<Vector<gentype> > &allxres                      = *((Vector<Vector<gentype> > *) (((void **) arg)[4]));
    Vector<gentype> &allfres                               = *((Vector<gentype> *)          (((void **) arg)[5]));
    gentype &fres                                          = *((gentype *)                  (((void **) arg)[7]));
    Vector<gentype> &xres                                  = *((Vector<gentype> *)          (((void **) arg)[8]));
    gentype &tempres                                       = *((gentype *)                  (((void **) arg)[9]));

    if ( dim )
    {
        int i;

        for ( i = 0 ; i < dim ; i++ )
        {
            xx("&",i) = x[i];
        }
    }

    tempres.force_int() = 0;
    (*fn)(tempres,xx,arginner);

    if ( ( allfres.size() == 0 ) || ( tempres < fres ) )
    {
        ires = allfres.size();
        fres = tempres;
        xres = xx;
    }

    if ( 1 )
    {
        allfres.append(allfres.size(),tempres);
        allxres.append(allxres.size(),xx);
    }

    return (double) tempres;
}

int nelderOpt(int dim,
              Vector<gentype> &xres,
              gentype &fres,
              int &ires,
              Vector<Vector<gentype> > &allxres,
              Vector<gentype> &allfres,
              Vector<gentype> &supres,
              const Vector<gentype> &xmin,
              const Vector<gentype> &xmax,
              void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
              void *fnarg,
              const NelderOptions &dopts,
              svmvolatile int &force_stop)
{
    NiceAssert( dim > 0 );
    NiceAssert( xmin.size() == dim );
    NiceAssert( xmax.size() == dim );

    allxres.resize(0);
    allfres.resize(0);
    supres.resize(0);

    Vector<double> locxres;

    Vector<double> locxmin(dim);
    Vector<double> locxmax(dim);

    int i;
    gentype locfres(0.0);

    for ( i = 0 ; i < dim ; i++ )
    {
        locxmin("&",i) = (double) xmin(i);
        locxmax("&",i) = (double) xmax(i);
    }

    void *fnarginner[10];
    Vector<gentype> xx(dim);
    gentype tempres;

    fnarginner[0] = (void *) &xx;
    fnarginner[1] = (void *) fn;
    fnarginner[2] = (void *) fnarg;
    fnarginner[3] = (void *) &ires;
    fnarginner[4] = (void *) &allxres;
    fnarginner[5] = (void *) &allfres;
    fnarginner[7] = (void *) &fres;
    fnarginner[8] = (void *) &xres;
    fnarginner[9] = (void *) &tempres;

    xres.resize(dim);
    locxres.resize(dim);

    for ( i = 0 ; i < dim ; i++ )
    {
        locxres("&",i) = (double) xres(i);
    }

    int res = nelderOpt(dim,locxres,locfres,locxmin,locxmax,fninnerdd,(void *) fnarginner,dopts,force_stop);

    supres.resize(allxres.size());
    gentype dummynull;
    dummynull.force_null();
    supres = dummynull;

    for ( i = 0 ; i < dim ; i++ )
    {
        xres("&",i) = locxres(i);
    }

    return res;
}




int NelderOptions::optim(int dim,
                      Vector<gentype> &xres,
                      gentype &fres,
                      int &ires,
                      Vector<Vector<gentype> > &allxres,
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
    int res = nelderOpt(dim,xres,fres,ires,allxres,allfres,supres,
                         xmin,xmax,fn,fnarg,*this,killSwitch);

    allfresmod = allfres;

    sscore.resize(allfres.size());
    sscore = 1.0;

    return res;
}
