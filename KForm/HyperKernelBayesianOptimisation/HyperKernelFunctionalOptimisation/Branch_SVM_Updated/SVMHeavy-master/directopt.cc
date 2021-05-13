
//
// DIRect global optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "directopt.h"
#include <math.h>
#include <stddef.h>
#include <iostream>


int directOpt(int dim,
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
              const DIRectOptions &dopts,
              svmvolatile int &killSwitch);





double tst_obj(int n, const double *x, int *undefined_flag, void *fnarginnerdr);
double tst_obj(int n, const double *x, int *undefined_flag, void *fnarginnerdr)
{
    (void) n;
 
    void **fnarginner = (void **) fnarginnerdr;

    double (*fn)(int, const double *, void *) = (double (*)(int, const double *, void *)) fnarginner[0];
    void *fnarg = fnarginner[1];

    int            &force_stop = *((int            *) (((void **) fnarginnerdr)[2]));
    double         &hardmin    = *((double         *) (((void **) fnarginnerdr)[3]));
    Vector<double> &l          = *((Vector<double> *) (((void **) fnarginnerdr)[4]));
    Vector<double> &u          = *((Vector<double> *) (((void **) fnarginnerdr)[5]));
    double         *xx         =  ((double         *) (((void **) fnarginnerdr)[6]));
    int            &dim        = *((int            *) (((void **) fnarginnerdr)[7]));
    int            &effdim     = *((int            *) (((void **) fnarginnerdr)[8]));
    double         &hardmax    = *((double         *) (((void **) fnarginnerdr)[9]));

    NiceAssert( n == effdim );

    int i;

    for ( i = 0 ; i < dim ; i++ )
    {
        if ( i < effdim )
        {
            xx[i] = (x[i]*(u(i)-l(i)))+l(i);
        }

        else
        {
            xx[i] = l(i);
        }
    }

    (void) undefined_flag;

    double res = (*fn)(dim,xx,fnarg);

    if ( res <= hardmin )
    {
        // Trigger early termination if hardmin reached

        force_stop = 1;
    }

    else if ( res >= hardmax )
    {
        // Trigger early termination if hardmax reached

        force_stop = 1;
    }

    return res;
}



int directOpt(int dim,
              Vector<double> &xres,
              gentype &fres,
              const Vector<double> &xmin,
              const Vector<double> &xmax,
              double (*fn)(int n, const double *x, void *arg),
              void *fnarg,
              const DIRectOptions &dopts,
              svmvolatile int &force_stop)
{
    int i;

    // WARNING: THERE IS A SERIOUS BUG IN THE NL_OPT VERSION OF DIRECT USED HERE!
    //
    // When the range [l,u] is anything other than [0,1] there is a good chance that the
    // result x will lie outside of [l,u].  For example I was using [-0.5,0.5] and found
    // that *if* the error code was -4 (that is, NLOPT_ROUNDOFF_LIMITED, which is supposed
    // to be a non-serious bug probably giving a typically useful result) x = 0.83333 was
    // a common result!  I suspect that the range is normalised to [0,1] inside of DIRect,
    // but return code -4 bypasses some sort of range correction to give the erroneous
    // result.
    //
    // WORKAROUND: give DIRect bounds [0,1] and re-scale for callback to *fn and when
    // the result is returned.

    NiceAssert( dim > 0 );
    NiceAssert( xmin.size() == dim );
    NiceAssert( xmax.size() == dim );
    NiceAssert( xmax >= xmin );

    int effdim = 0;

    for ( i = 0 ; i < dim ; i++ )
    {
        if ( xmin(i) < xmax(i) )
        {
            effdim = i+1;
        }
    }

    NiceAssert( effdim > 0 );

    long int maxits          = dopts.maxits;
    long int maxevals        = dopts.maxevals;
    double eps               = dopts.eps;
    double maxtraintime      = dopts.maxtraintime;
    double traintimeoverride = dopts.traintimeoverride;
    double hardmin           = dopts.hardmin;
    double hardmax           = dopts.hardmax;

    xres.resize(dim);

    double *x = &xres("&",zeroint());
    double *l;
    double *u;
    double *xx;
    double magic_eps_abs = 0;
    double volume_reltol = 0.0;
    double sigma_reltol = -1.0;

    MEMNEWARRAY(l ,double,dim);
    MEMNEWARRAY(u ,double,dim);
    MEMNEWARRAY(xx,double,dim);

    for ( i = 0 ; i < dim ; i++ )
    {
        l[i] = 0;
        u[i] = +1;
    }

    void *fnarginner[10];

    fnarginner[0] = (void *)  fn;
    fnarginner[1] = (void *)  fnarg;
    fnarginner[2] = (void *) &force_stop;
    fnarginner[3] = (void *) &hardmin;
    fnarginner[4] = (void *) &xmin;
    fnarginner[5] = (void *) &xmax;
    fnarginner[6] = (void *)  xx;
    fnarginner[7] = (void *) &dim;
    fnarginner[8] = (void *) &effdim;
    fnarginner[9] = (void *) &hardmax;

    void *fnarginnerdr = (void *) fnarginner;

    errstream() << "DIRect Optimisation Initiated:\n";

    // Note use of effdim here!

    int intres = direct_optimize(tst_obj,fnarginnerdr,
                           effdim,
                           l,u,
                           x,&(fres.force_double()),
                           maxevals,maxits,
                           ( traintimeoverride == 0 ) ? maxtraintime : traintimeoverride,
                           eps,magic_eps_abs,
                           volume_reltol,sigma_reltol,
                           force_stop, 
                           DIRECT_UNKNOWN_FGLOBAL,
                           0,
                           dopts.algorithm);

    // Need to re-scale result

    for ( i = 0 ; i < dim ; i++ )
    {
        x[i] = (x[i]*(xmax(i)-xmin(i)))+xmin(i);
    }

    MEMDELARRAY(l);
    MEMDELARRAY(u);
    MEMDELARRAY(xx);

    errstream() << "DIRect Optimisation Ended\n";

    return intres;
}


double fninnerd(int dim, const double *x, void *arg);
double fninnerd(int dim, const double *x, void *arg)
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

int directOpt(int dim,
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
              const DIRectOptions &dopts,
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

    int res = directOpt(dim,locxres,locfres,locxmin,locxmax,fninnerd,(void *) fnarginner,dopts,force_stop);

    supres.resize(allxres.size());
    gentype dummynull;
    dummynull.force_null();
    supres = dummynull;

//    xres.resize(dim);
//
//    for ( i = 0 ; i < dim ; i++ )
//    {
//        xres("&",i) = locxres(i);
//    }

    return res;
}







int DIRectOptions::optim(int dim,
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
    int res = directOpt(dim,xres,fres,ires,allxres,allfres,supres,
                        xmin,xmax,fn,fnarg,*this,killSwitch);

    allfresmod = allfres;

    sscore.resize(allfres.size());
    sscore = 1.0;

    return res;
}
