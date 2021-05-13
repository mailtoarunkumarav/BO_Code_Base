
//
// Data loading functions
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <string>
#include <math.h>
#include "addData.h"
#include "ml_mutable.h"
#include "basefn.h"



int genericMLDataLoad(int binaryRelabel, 
                      int singleDrop, 
                      const ML_Base &mlbase, 
                      const std::string &trainfile, 
                      int reverse, 
                      int ignoreStart, 
                      int imax, 
                      Vector<SparseVector<gentype> > &xtest, 
                      Vector<gentype> &ytest, 
                      Vector<int> &outkernind,
                      int ibase, 
                      int coercetosingle, 
                      int coercefromsingle, 
                      const gentype &fromsingletarget, 
                      int uselinesvector, 
                      Vector<int> &linesread, 
                      Vector<gentype> &ytestresh, 
                      Vector<gentype> &ytestresg, 
                      Vector<gentype> &gvarres, 
                      const SparseVector<gentype> &xtemplate,
                      int dovartest = 0, 
                      int runtestonly = 0, 
                      ML_Base *mldest = NULL);

int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemplate, const std::string &trainfile, int reverse, int ignoreStart, int imax, int ibase)
{
    // These just need to be empty

    gentype fromsingletarget;
    Vector<int> linesread;

    return addtrainingdata(mlbase,
                           xtemplate,
                           trainfile,
                           reverse,
                           ignoreStart,
                           imax,
                           ibase,
                           0,
                           0,
                           fromsingletarget,
                           0,
                           0,
                           0,
                           linesread);
}

int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemplate, const std::string &trainfile, int reverse, int ignoreStart, int imax, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget, int binaryRelabel, int singleDrop, int uselinesvector, Vector<int> &linesread)
{
    Vector<SparseVector<gentype> > xtest;
    Vector<gentype> ytest;
    Vector<gentype> ytestresh;
    Vector<gentype> ytestresg;
    Vector<gentype> gvarres;
    Vector<int> outkernind;

    return genericMLDataLoad(binaryRelabel,
                             singleDrop,
                             mlbase,
                             trainfile,
                             reverse,
                             ignoreStart,
                             imax,
                             xtest,
                             ytest,
                             outkernind,
                             ibase,
                             coercetosingle,
                             coercefromsingle,
                             fromsingletarget,
                             uselinesvector,
                             linesread,
                             ytestresh,
                             ytestresg,
                             gvarres,
                             xtemplate,
                             0,
                             0,
                             static_cast<ML_Base *>(&mlbase));
}

int loadFileForHillClimb(const ML_Base &mlbase, const SparseVector<gentype> &xtemplate, const std::string &trainfile, int reverse, int ignoreStart, int imax, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget, int binaryRelabel, int singleDrop, int uselinesvector, Vector<int> &linesread, Vector<SparseVector<gentype> > &xtest, Vector<gentype> &ytest)
{
    Vector<gentype> ytestresh;
    Vector<gentype> ytestresg;
    Vector<gentype> gvarres;
    int ibase = -1;
    Vector<int> outkernind;

    return genericMLDataLoad(binaryRelabel,
                             singleDrop,
                             mlbase,
                             trainfile,
                             reverse,
                             ignoreStart,
                             imax,
                             xtest,
                             ytest,
                             outkernind,
                             ibase,
                             coercetosingle,
                             coercefromsingle,
                             fromsingletarget,
                             uselinesvector,
                             linesread,
                             ytestresh,
                             ytestresg,
                             gvarres,
                             xtemplate,
                             0,
                             0,
                             NULL);
}

int loadFileAndTest(const ML_Base &mlbase, const SparseVector<gentype> &xtemplate, const std::string &trainfile, int reverse, int ignoreStart, int imax, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget, int binaryRelabel, int singleDrop, int uselinesvector, Vector<int> &linesread, Vector<gentype> &ytest, Vector<gentype> &ytestresh, Vector<gentype> &ytestresg, Vector<gentype> &gvarres, int dovartest, Vector<int> &outkernind)
{
    Vector<SparseVector<gentype> > xtest;

    return genericMLDataLoad(binaryRelabel,
                             singleDrop,
                             mlbase,
                             trainfile,
                             reverse,
                             ignoreStart,
                             imax,
                             xtest,
                             ytest,
                             outkernind,
                             ibase,
                             coercetosingle,
                             coercefromsingle,
                             fromsingletarget,
                             uselinesvector,
                             linesread,
                             ytestresh,
                             ytestresg,
                             gvarres,
                             xtemplate,
                             dovartest,
                             1,
                             NULL);
}



int genericMLDataLoad(int binaryRelabel,
                      int singleDrop, 
                      const ML_Base &mlbase,
                      const std::string &trainfile,
                      int reverse,
                      int ignoreStart,
                      int imax,
                      Vector<SparseVector<gentype> > &xtest,
                      Vector<gentype> &ytest,
                      Vector<int> &outkernind,
                      int ibase,
                      int coercetosingle,
                      int coercefromsingle,
                      const gentype &fromsingletarget,
                      int uselinesvector,
                      Vector<int> &linesread,
                      Vector<gentype> &ytestresh,
                      Vector<gentype> &ytestresg,
                      Vector<gentype> &gvarres,
                      const SparseVector<gentype> &xtemplate,
                      int dovartest,
                      int runtestonly,
                      ML_Base *mldest)
{
    char realtargtype = 'N';

    Vector<int> linesreadatstart(linesread);

    if ( !mldest ) { realtargtype = mlbase.hOutType();    }
    else           { realtargtype = (*mldest).targType(); }

    if ( coercetosingle && !( realtargtype == 'N' ) )
    {
        STRTHROW("Can't use u suffix as ML is not single class.");
    }

    if ( coercefromsingle && ( realtargtype == 'N' ) )
    {
        STRTHROW("Can't use l suffix as ML is single class.");
    }

    int pointsadded = 0;

    if ( ibase == -1 )
    {
        ibase = mlbase.N();
    }

    std::ifstream datfile(trainfile.c_str());

    if ( !datfile.is_open() )
    {
        STRTHROW("Unable to open training file "+trainfile);
    }

    SparseVector<gentype> x,y;
    gentype z;
    Vector<double> xCweigh;
    Vector<double> xepsweigh;
    Vector<int> xd;
    Vector<int> xi;
    int xk;

    double Cweight;
    double epsweight;
    int d;
    std::string buffer;
    int ij = 0;
    int goahead;
    int i = ibase;

    while ( !datfile.eof() && ( ( ij < imax+ignoreStart ) || imax == -1 ) )
    {
        goover:

        if ( uselinesvector && ( linesread.size() == 0 ) )
	{
	    break;
	}

	buffer = "";

	while ( ( buffer.length() == 0 ) && !datfile.eof() )
	{
	    getline(datfile,buffer);
	}

	if ( buffer.length() == 0 )
	{
	    break;
	}

        if ( ( buffer.length() >= 2 ) && ( buffer[0] == '/' ) && ( buffer[1] == '/' ) )
        {
            goto goover;
        }

	goahead = 1;

	if ( uselinesvector )
	{
	    if ( ij < linesread(zeroint()) )
	    {
		goahead = 0;
	    }

	    else
	    {
                NiceAssert( linesread(zeroint()) == ij );
		linesread.remove(0);
	    }
	}

	if ( goahead && ( ij >= ignoreStart ) )
	{
            // Give feedback on progress

//            if ( !(pointsadded%1000) ) { errstream() << "." << pointsadded; }
//            else                       { errstream() << ".";                }

            if ( !(pointsadded%1000) ) { errstream() << "." << pointsadded; }
            else                       { errstream() << "."; nullPrint(errstream(),pointsadded,-1); }

            // Load training vector from file

            if ( coercefromsingle || ( !coercetosingle && ( realtargtype == 'N' ) ) )
            {
                // No target given in file

                parselineML_Single(x,Cweight,epsweight,buffer,1); //!(mlbase.xspaceSparse()));

                if ( ( realtargtype == 'N' ) && !isSVMAutoEn(mlbase) && !isKNNAutoEn(mlbase) )
                {
                    // No target for this type, so make target NULL

                    z.makeNull();
                }

                else if ( ( realtargtype == 'N' ) && isSVMAutoEn(mlbase) && isKNNAutoEn(mlbase) )
                {
                    // For the auto-encoder, target is always input (even
                    // when target given)

                    Vector<gentype> temp;

                    (*mldest).xlateFromSparse(temp,x);

                    z = temp;
                }

                else
                {
                    // Target not given by file but given by user

                    z = fromsingletarget;
                }

                if ( z.isValEqnDir() )
                {
                    z.scalarfn_setisscalarfn(1);
                }
            }

            else
            {
                // Target given in file - may or may not actually be used

                parselineML_Generic(z,x,Cweight,epsweight,d,buffer,reverse,1); //!(mlbase.xspaceSparse()));

                if ( coercetosingle )
                {
                    // No target for this type, so make target NULL

                    z.makeNull();
                }
            }

            addtemptox(x,xtemplate);

            // Binary relabelling (if any)

            if ( binaryRelabel )
            {
                if ( z.isValInteger() )
                {
                    if ( (int) z == binaryRelabel )
                    {
                        z = +1;
                    }

                    else
                    {
                        z = -1;
                    }
                }
            }

            // Class skipping (if any)

            if ( singleDrop )
            {
                if ( z.isValInteger() )
                {
                    if ( (int) z == singleDrop )
                    {
                        // There is surely a nicer way to do this

                        goto goover;
                    }
                }
            }

            // Use for training vector depends on task

            if ( mldest )
            {
                // Task is to add training vectors to machine

                xtest.add(xtest.size());
                ytest.add(ytest.size());
                xCweigh.add(xCweigh.size());
                xepsweigh.add(xepsweigh.size());
                xd.add(xd.size());
                xi.add(xi.size());

                qswap(xtest("&",xtest.size()-1),x);
                ytest("&",ytest.size()-1) = z;
                xCweigh("&",xCweigh.size()-1) = Cweight;
                xepsweigh("&",xepsweigh.size()-1) = epsweight;
                xd("&",xd.size()-1) = d;
                xi("&",xi.size()-1) = i;
            }

            else if ( !runtestonly )
            {
                // Task is to load data into (x,y) vectors

                xtest.add(xtest.size());
                ytest.add(ytest.size());

                qswap(xtest("&",xtest.size()-1),x);
                ytest("&",ytest.size()-1) = z;
            }

            else
            {
                // Task is to run tests.  Do not save x as testing file may
                // be very large, but do keep everything else for reporting.

                ytest.add(ytest.size());
                ytestresh.add(ytestresh.size());
                ytestresg.add(ytestresg.size());
                outkernind.add(outkernind.size());

                outkernind("&",outkernind.size()-1) = x.isfarfarfarindpresent(3) ? (int) x.fff(3) : -1;

                if ( dovartest )
                {
                    gvarres.add(gvarres.size());
                }

                ytest("&",ytest.size()-1) = z;

                mlbase.gh(ytestresh("&",ytestresh.size()-1),ytestresg("&",ytestresg.size()-1),x);

                if ( dovartest )
                {
                    gentype dummy;

                    mlbase.var(gvarres("&",gvarres.size()-1),dummy,x);
                }
            }

            // Update counters and such

            pointsadded++;
	    i++;
	}

        // Update counters and such

	ij++;
    }

    datfile.close();

    errstream() << "." << pointsadded << "...";

    // If task is to add training data to ML then do this now blockwise
    // (blockwise preferable as it is much faster if there is variable
    // autosetting setup in the ML)

    if ( mldest && xi.size() )
    {
        if ( isSVMMvRank(mlbase) )
        {
            errstream() << "*";

            // Need to have d set immediately, so use alternative method

            int Nnew = (dynamic_cast<SVM_MvRank &>(mldest->getML())).N() + xi.size();

            if ( (dynamic_cast<SVM_MvRank &>(mldest->getML())).preallocsize() < Nnew )
            {
                (dynamic_cast<SVM_MvRank &>(mldest->getML())).prealloc(Nnew+1);
            }

            for ( i = 0 ; i < xi.size() ; i++ )
            {
                (dynamic_cast<SVM_MvRank &>(mldest->getML())).qaddTrainingVector(xi(i),(double) ytest(i),xtest("&",i),xCweigh(i),xepsweigh(i),xd(i));
            }

            errstream() << "#";
        }

        else if ( isLSVMvRank(mlbase) )
        {
            errstream() << "*";

            // Need to have d set immediately, so use alternative method

            int Nnew = (dynamic_cast<LSV_MvRank &>(mldest->getML())).N() + xi.size();

            if ( (dynamic_cast<LSV_MvRank &>(mldest->getML())).preallocsize() < Nnew )
            {
                (dynamic_cast<LSV_MvRank &>(mldest->getML())).prealloc(Nnew+1);
            }

            for ( i = 0 ; i < xi.size() ; i++ )
            {
                (dynamic_cast<LSV_MvRank &>(mldest->getML())).qaddTrainingVector(xi(i),(double) ytest(i),xtest("&",i),xCweigh(i),xepsweigh(i),xd(i));
            }

            errstream() << "#";
        }

        else
        {
            errstream() << "*";

            xk = xi(zeroint());

            int Nnew = (*mldest).N() + xi.size();

            if ( (*mldest).preallocsize() < Nnew )
            {
                (*mldest).prealloc(Nnew+1);
            }

// The following was a test I added while debugging the altcontent acceleration of inner product calculation (see sparsevector.h and gentype.cc).
// It is no longer necessary but I retained it (commented out for speed) as a backstop just in case.  All things being equal it does precisely nothing.
//errstream() << "phantomx fscking start\n";
//for ( i = 0 ; i < xi.size() ; i++ )
//{
//xtest("&",i).makealtcontent();
//}
//errstream() << "phantomx fscking add\n";
            (*mldest).qaddTrainingVector(xk,ytest,xtest,xCweigh,xepsweigh);
//errstream() << "phantomx fscking stop\n";

            errstream() << "#";

            if ( isSVMScalar(mlbase) || isSVMVector(mlbase) || isSVMPlanar(mlbase) || isLSVScalar(mlbase) || isGPRScalar(mlbase) || isSSVScalar(mlbase) || isSVMSimLrn(mlbase) )
            {
                for ( i = 0 ; i < xi.size() ; i++ )
                {
                    xk = xi(i);

                    if ( xd(i) != 2 )
                    {
                        (*mldest).setd(xk,xd(i));
                    }
                }
            }
        }
    }

    errstream() << "\n";

    if ( uselinesvector == 2 )
    {
        linesread = linesreadatstart;
    }

    return pointsadded;
}

int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemplate, Vector<SparseVector<gentype> > &x, const Vector<gentype> &yy, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget)
{
    Vector<gentype> sigmaweight(x.size());

    sigmaweight = onedblgentype();

    return addtrainingdata(mlbase,xtemplate,x,yy,sigmaweight,ibase,coercetosingle,coercefromsingle,fromsingletarget);
}

int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemplate, Vector<SparseVector<gentype> > &x, const Vector<gentype> &yy, const Vector<gentype> &sigmaweight, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget)
{
    Vector<gentype> y(yy);

    if ( coercetosingle && !( mlbase.targType() == 'N' ) )
    {
        STRTHROW("Can't use u suffix as ML is not single class.");
    }

    if ( coercefromsingle && ( mlbase.targType() == 'N' ) )
    {
        STRTHROW("Can't use l suffix as ML is single class.");
    }

    if ( coercefromsingle )
    {
        y.resize(x.size());
        y.zero();
        y = fromsingletarget;
    }

    NiceAssert( x.size() == y.size() );

    if ( ibase == -1 )
    {
        ibase = mlbase.N();
    }

    Vector<double> Cweight(sigmaweight.size());
    Vector<double> weightdummy(y.size());

    weightdummy = 1.0;

    if ( x.size() )
    {
        int i;

        for ( i = 0 ; i < x.size() ; i++ )
        {
            if ( x("&",i).indsize() )
            {
                int iji;
        
                for ( iji = 0 ; iji < x("&",i).indsize() ; iji++ )
                {
                    if ( (x(i).direcref(iji)).isValEqnDir() )
                    {
                        (x("&",i).direref(iji)).scalarfn_setisscalarfn(1);
                    }
                }
            }
        }
    }

    if ( y.size() )
    {
        int i;

        for ( i = 0 ; i < y.size() ; i++ )
        {
            if ( y(i).isValEqnDir() )
            {
                y("&",i).scalarfn_setisscalarfn(1);
            }
        }
    }

    if ( sigmaweight.size() )
    {
        int i;

        for ( i = 0 ; i < sigmaweight.size() ; i++ )
        {
            Cweight("&",i) = 1.0/( ( ((double) sigmaweight(i)) < MINSWEIGHT ) ? MINSWEIGHT : ((double) sigmaweight(i)) );
        }
    }

    int Nnew = mlbase.N() + x.size();

    if ( mlbase.preallocsize() < Nnew )
    {
        mlbase.prealloc(Nnew+1);
    }

    addtemptox(x,xtemplate);

    mlbase.qaddTrainingVector(ibase,y,x,Cweight,weightdummy);

    return x.size();
}




int addbasisdataUU(ML_Base &dest, const std::string &fname)
{
    int pointsadded = 0;

    std::ifstream  srcfile;

    srcfile.open(fname.c_str(),std::ofstream::in);

    if ( !srcfile.is_open() )
    {
        std::string errstring;
        errstring = "Unable to open basis file "+fname;
        throw errstring;
    }

    std::string buffer;
    gentype tempbasevec;

    while ( !srcfile.eof() )
    {
        buffer = "";

        while ( ( buffer.length() == 0 ) && !srcfile.eof() )
        {
            getline(srcfile,buffer);
        }

        if ( buffer.length() == 0 )
        {
            break;
        }

        std::stringstream transit;

        transit << buffer;
        transit >> tempbasevec;

        if ( tempbasevec.isValEqnDir() )
        {
            tempbasevec.scalarfn_setisscalarfn(1);
        }

        dest.addToBasisUU(dest.NbasisUU(),tempbasevec);
        pointsadded++;
    }

    srcfile.close();

    return pointsadded;
}

int addbasisdataVV(ML_Base &dest, const std::string &fname)
{
    int pointsadded = 0;

    std::ifstream  srcfile;

    srcfile.open(fname.c_str(),std::ofstream::in);

    if ( !srcfile.is_open() )
    {
        std::string errstring;
        errstring = "Unable to open basis file "+fname;
        throw errstring;
    }

    std::string buffer;
    gentype tempbasevec;

    while ( !srcfile.eof() )
    {
        buffer = "";

        while ( ( buffer.length() == 0 ) && !srcfile.eof() )
        {
            getline(srcfile,buffer);
        }

        if ( buffer.length() == 0 )
        {
            break;
        }

        std::stringstream transit;

        transit << buffer;
        transit >> tempbasevec;

        if ( tempbasevec.isValEqnDir() )
        {
            tempbasevec.scalarfn_setisscalarfn(1);
        }

        dest.addToBasisVV(dest.NbasisVV(),tempbasevec);
        pointsadded++;
    }

    srcfile.close();

    return pointsadded;
}




SparseVector<gentype> &addtemptox(SparseVector<gentype> &x, const SparseVector<gentype> &xtemp)
{
    if ( xtemp.indsize() )
    {
        int i,ii;

        for ( i = 0 ; i < xtemp.indsize() ; i++ )
        {
            ii = xtemp.ind(i);

            if ( !(x.isindpresent(ii)) )
            {
                x("&",ii) = xtemp.direcref(i);
            }
        }
    }

    return x;
}

Vector<SparseVector<gentype> > &addtemptox(Vector<SparseVector<gentype> > &x, const SparseVector<gentype> &xtemp)
{
    if ( x.size() )
    {
        int i;

        for ( i = 0 ; i < x.size() ; i++ )
        {
            addtemptox(x("&",i),xtemp);
        }
    }

    return x;
}
