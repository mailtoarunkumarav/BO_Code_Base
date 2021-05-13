
//
// Grid-based Optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "globalopt.h"

//
// Does a simple grid-search to minimise function
//

#ifndef _gridopt_h
#define _gridopt_h

class GridOptions : public GlobalOptions
{
public:

    // numZooms: number of "zoom and repeat" operations (find optimal on
    //           grid, then zoom in and repeat on smaller scale.
    // zoomFact: zoom factor in above operations (range reduced by this
    //           factor around optimal, with cutoff for grid edges).
    // numPts:   number of grid points for each "axis"

    // NB: we use the default assignment operator in the code, so if anything
    //     tricky gets added you'll need to define an assignment operator.

    int numZooms;
    double zoomFact;
    Vector<int> numPts;

    GridOptions() : GlobalOptions()
    {
        numZooms = 0;
        zoomFact = 0.333333333333;

        return;
    }

    GridOptions(const GridOptions &src) : GlobalOptions(src)
    {
        *this = src;

        return;
    }

    GridOptions &operator=(const GridOptions &src)
    {
        GlobalOptions::operator=(src);

        numZooms = src.numZooms;
        zoomFact = src.zoomFact;
        numPts   = src.numPts;

        return *this;
    }

    // Generate a copy of the relevant optimisation class.

    virtual GlobalOptions *makeDup(void) const
    {
        GridOptions *newver;

        MEMNEW(newver,GridOptions(*this));

        return newver;
    }

    // supres: none

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
        return 1;
    }
};

#endif
