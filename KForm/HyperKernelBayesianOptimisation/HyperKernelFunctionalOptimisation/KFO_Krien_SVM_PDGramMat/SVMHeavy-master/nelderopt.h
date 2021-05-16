
//
// Nelder-Mead optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "globalopt.h"

#ifndef _nelderopt_h
#define _nelderopt_h

//
// Uses Nelder-Mead optimisation to minimise a function.
// Can be warm-started.
//
// Return value:
//
// -1: generic fail
// -2: invalid args
// -3: out of memory
// -4: roundoff limited
// -5: forced stop
// 1: success
// 2: stopval reached
// 3: ftol reached
// 4: xtol reached
// 5: maxeval reached
// 6: maxtime reached
//

class NelderOptions : public GlobalOptions
{
public:

     // minf_max: maximum f value (-HUGE_VAL)
     // ftol_rel: relative tolerance of function value (0)
     // ftol_abs: absolute tolerance of function value (0)
     // xtol_rel: relative tolerance of x value (0)
     // xtol_abs: absolute tolerance of x value (0)
     // maxeval: max number of f evaluations (1000)
     // method: 0 is subplex, 1 is original Nelder-Mead

     double minf_max;
     double ftol_rel;
     double ftol_abs;
     double xtol_rel;
     double xtol_abs;
     int maxeval;
     int method;

    NelderOptions() : GlobalOptions()
    {
        minf_max = -HUGE_VAL;
        ftol_rel = 0;
        ftol_abs = 0;
        xtol_rel = 0;
        xtol_abs = 0;
        maxeval  = 1000;
        method   = 0;

        return;
    }

    NelderOptions(const NelderOptions &src) : GlobalOptions(src)
    {
        *this = src;

        return;
    }

    NelderOptions &operator=(const NelderOptions &src)
    {
        GlobalOptions::operator=(src);

        minf_max = src.minf_max;
        ftol_rel = src.ftol_rel;
        ftol_abs = src.ftol_abs;
        xtol_rel = src.xtol_rel;
        xtol_abs = src.xtol_abs;
        maxeval  = src.maxeval;
        method   = src.method;

        return *this;
    }

    // Generate a copy of the relevant optimisation class.

    virtual GlobalOptions *makeDup(void) const
    {
        NelderOptions *newver;

        MEMNEW(newver,NelderOptions(*this));

        return newver;
    }

    virtual int optim(int dim,
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
                      svmvolatile int &killSwitch);

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
        return GlobalOptions::optim(dim,xres,Xres,fres,ires,mInd,muInd,sigInd,srcmodInd,diffmodInd,allxres,allXres,allfres,allmres,allsres,s_score,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch,numReps,meanfres,varfres,meanires,varires,meantres,vartres,meanTres,varTres,meanallfres,varallfres,meanallmres,varallmres);
    }

    virtual int optdefed(void)
    {
        return 2;
    }
};

#endif
