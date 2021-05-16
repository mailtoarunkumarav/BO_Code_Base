
//
// Performance/error testing routines
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "errortest.h"
#include "ml_mutable.h"

double calcLOORecall(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int startpoint, int isLOO, int calcgvarres, int suppressfb = 0);

void disableVector(int i, ML_Base *activeML);
void disableVector(const Vector<int> &i, ML_Base *activeML);
void resetGlobal(ML_Base *activeML);
void semicopyML(ML_Base *activeML, const ML_Base *srcML);
void copyML(ML_Base *activeML, const ML_Base *srcML);
int trainGlobal(ML_Base *activeML, int islastopt);
int isVectorActive(int i, ML_Base *activeML);
int isVectorEnabled(int i, ML_Base *activeML);
int isTrainedML(ML_Base *activeML);

double calcnegloglikelihood(const ML_Base &baseML, int suppressfb)
{
    (void) suppressfb;

    double res = 0;

    if ( isSVM(baseML) )
    {
        res = (dynamic_cast<const SVM_Generic &>(baseML)).quasiloglikelihood();
    }

    else if ( isGPR(baseML) )
    {
        res = (dynamic_cast<const GPR_Generic &>(baseML)).loglikelihood();
    }

    else if ( isLSV(baseML) )
    {
        res = (dynamic_cast<const LSV_Generic &>(baseML)).lsvloglikelihood();
    }

    else
    {
        throw("Log-likelihood not defined for this ML type");
    }

    return -res;
}

double calcLOO(const ML_Base &baseML, int startpoint, int suppressfb)
{
    Vector<int> cnt;
    Matrix<int> cfm;

    return calcLOO(baseML,cnt,cfm,startpoint,suppressfb);
}

double calcRecall(const ML_Base &baseML, int startpoint, int suppressfb)
{
    Vector<int> cnt;
    Matrix<int> cfm;

    return calcRecall(baseML,cnt,cfm,startpoint,suppressfb);
}

double calcCross(const ML_Base &baseML, int m, int rndit, int numreps, int startpoint, int suppressfb)
{
    Vector<double> repres;
    Vector<double> cnt;
    Matrix<double> cfm;

    return calcCross(baseML,m,rndit,repres,cnt,cfm,numreps,startpoint,suppressfb);
}

double CalcSparSens(const ML_Base &baseML, int minbad, int maxbad, double noisemean, double noisevar, int startpoint, int suppressfb)
{
    Vector<double> repres;
    Vector<double> cnt;
    Matrix<double> cfm;

    return calcSparSens(baseML,repres,cnt,cfm,minbad,maxbad,noisemean,noisevar,startpoint,suppressfb);
}

double calcLOO(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, int startpoint, int suppressfb)
{
    Vector<gentype> resh;
    Vector<gentype> resg;
    Vector<gentype> gvarres;

    return calcLOO(baseML,cnt,cfm,resh,resg,gvarres,startpoint,0,suppressfb);
}

double calcRecall(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, int startpoint, int suppressfb)
{
    Vector<gentype> resh;
    Vector<gentype> resg;
    Vector<gentype> gvarres;

    return calcRecall(baseML,cnt,cfm,resh,resg,gvarres,startpoint,0,suppressfb);
}

double calcCross(const ML_Base &baseML, int m, int rndit, Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, int numreps, int startpoint, int suppressfb)
{
    Vector<Vector<gentype> > resh;
    Vector<Vector<gentype> > resg;
    Vector<Vector<gentype> > gvarres;

    return calcCross(baseML,m,rndit,repres,cnt,cfm,resh,resg,gvarres,numreps,startpoint,0,suppressfb);
}

double calcSparSens(const ML_Base &baseML, Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, int minbad, int maxbad, double noisemean, double noisevar, int startpoint, int suppressfb)
{
    Vector<Vector<gentype> > resh;
    Vector<Vector<gentype> > resg;
    Vector<Vector<gentype> > gvarres;

    return calcSparSens(baseML,repres,cnt,cfm,resh,resg,gvarres,minbad,maxbad,noisemean,noisevar,startpoint,0,suppressfb);
}

double calcTest(const ML_Base &baseML, const Vector<SparseVector<gentype> > &xtest, const Vector<gentype> &ytest, int suppressfb)
{
    Vector<int> cnt;
    Matrix<int> cfm;

    return calcTest(baseML,xtest,ytest,cnt,cfm,suppressfb);
}

double calcTest(const ML_Base &baseML, const Vector<SparseVector<gentype> > &xtest, const Vector<gentype> &ytest, Vector<int> &cnt, Matrix<int> &cfm, int suppressfb)
{
    Vector<gentype> resh;
    Vector<gentype> resg;
    Vector<gentype> gvarres;

    return calcTest(baseML,xtest,ytest,cnt,cfm,resh,resg,gvarres,0,suppressfb);
}

double calcLOO(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int startpoint, int calcvarres, int suppressfb)
{
    return calcLOORecall(baseML,cnt,cfm,resh,resg,gvarres,startpoint,1,calcvarres,suppressfb);
}

double calcRecall(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int startpoint, int calcvarres, int suppressfb)
{
    return calcLOORecall(baseML,cnt,cfm,resh,resg,gvarres,startpoint,0,calcvarres,suppressfb);
}




int ishzero(const gentype &h)
{
    int res = 0;

    if ( h.isValInteger() && ( 0 == (int) h ) )
    {
        res = 1;
    }

    return res;
}





double calcCross(const ML_Base &baseML, int m, int rndit, Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, Vector<Vector<gentype> > &resh, Vector<Vector<gentype> > &resg, Vector<Vector<gentype> > &gvarres, int numreps, int startpoint, int calcgvarres, int suppressfb)
{
    NiceAssert( numreps > 0 );

    int oneoff = startpoint & 2;
    startpoint &= 1;

    if ( oneoff )
    {
        // This option is intended to simulate a single random validation set over multiple runs, so it is important that it be the *same* random sequence every time

        svm_srand(42);
    }

    startpoint = startpoint || isONN(baseML);

    int i,j,k,l,rndsel;
    double res = 0;
    int N = baseML.N();
    int Nnz = baseML.N()-baseML.NNC(0);
    int Nclasses = baseML.numInternalClasses();

    if ( m > Nnz )
    {
	m = Nnz;
    }

    cnt.resize(Nclasses);
    cfm.resize(Nclasses,Nclasses+1);

    cnt.zero();
    cfm.zero();

    repres.resize(numreps);
    repres.zero();

    resg.resize(numreps);
    resh.resize(numreps);

    if ( calcgvarres )
    {
        gvarres.resize(numreps);
    }

    for ( i = 0 ; i < numreps ; i++ )
    {
        resg("&",i).resize(N);
        resg("&",i).zero();

        resh("&",i).resize(N);
        resh("&",i).zero();

        if ( calcgvarres )
        {
            gvarres("&",i).resize(N);
            gvarres("&",i).zero();
        }
    }

    if ( Nnz )
    {
	Vector<double> loccnt(cnt);
	Matrix<double> loccfm(cfm);
        double locres = 0.0;

        ML_Base *locML = (ML_Base *) &baseML;
        ML_Base *srcML = (ML_Base *) &baseML;

        locML = makeNewML(baseML.type(),baseML.subtype());
        NiceAssert( locML );
        semicopyML(locML,&baseML);

        ML_Base *temp = srcML; srcML = locML; locML = temp;

        Vector<int> locisenabled(N);
        Vector<gentype> locy = locML->y();

        for ( i = 0 ; i < N ; i++ )
        {
            locisenabled("&",i) = isVectorEnabled(i,locML);
        }

	for ( k = 0 ; k < numreps ; k++ )
	{
	    // Work out cross-fold groups

            loccnt.zero();
            loccfm.zero();
            locres = 0.0;

	    Vector<int> countvect(Nnz);

            j = 0;

            for ( i = 0 ; i < N ; i++ )
	    {
                if ( locisenabled(i) )
		{
                    countvect("&",j++) = i;
		}
	    }

	    Vector<Vector<int> > blockidvect(m);

            int Nnzleft = Nnz;

	    for ( i = 0 ; i < m ; i++ )
	    {
		blockidvect("&",i).resize(Nnzleft/(m-i));

		for ( j = 0 ; j < Nnzleft/(m-i) ; j++ )
		{
		    rndsel = 0;

		    if ( rndit )
		    {
			rndsel = svm_rand()%(countvect.size());
		    }

		    (blockidvect("&",i))("&",j) = countvect(rndsel);
		    countvect.remove(rndsel);
		}

                Nnzleft -= (Nnzleft/(m-i));
	    }

            NiceAssert( !(countvect.size()) );

	    Vector<double> tmpcnt(cnt);
	    Matrix<double> tmpcfm(cfm);
	    double tmpres = 0.0;
            int cla,clb;
            int Nblock;
            int Nnonz = 0;

	    for ( i = 0 ; ( i < m ) && ( !i || !oneoff ) ; i++ )
	    {
                Nblock = (blockidvect(i)).size();

                disableVector(blockidvect(i),locML);

                if ( startpoint )
                {
                    resetGlobal(locML);
                }

if ( !suppressfb )
{
nullPrint(errstream(),"~~~~~",5);
nullPrint(errstream(),i,-5); 
}
                trainGlobal(locML,( ( i == m-1 ) && ( k == numreps-1 ) ));

		tmpcnt.zero();
		tmpcfm.zero();
		tmpres = 0;

                for ( j = 0 ; j < Nblock ; j++ )
		{
                    l = (blockidvect(i))(j);

                    locML->ghTrainingVector(resh("&",k)("&",l),resg("&",k)("&",l),l);

                    if ( calcgvarres )
                    {
                        gentype dummy;

                        locML->varTrainingVector(gvarres("&",k)("&",l),dummy,l);
                    }

                    // NB: isenabled actually returns d for MLs.  For ML_Scalar,
                    //     d sets whether a point is an upper bound, a lower bound
                    //     or a standard target type.  This information is needed
                    //     to calculate distance.  The isenabled(i) actually returns
                    //     d for MLs (d = 0 means disabled).  Hence the following
                    //     line passes isenabled(l) to calcDist.

                    if ( (*locML).isClassifier() && ishzero(resh(k)(l)) )
                    {
                        tmpres += 1;

                        cla = (*locML).getInternalClass(locy(l));
                        clb = Nclasses;
                    }

                    else
                    {
                        tmpres += locML->calcDist(resh(k)(l),locy(l),l,locisenabled(l));

                        cla = (*locML).getInternalClass(locy(l));
                        clb = (*locML).getInternalClass((resh(k))(l));
                    }

                    ++(tmpcnt("&",cla));
                    ++(tmpcfm("&",cla,clb));

                    ++Nnonz;
		}

		loccnt += tmpcnt;
		loccfm += tmpcfm;
                locres += tmpres;

                semicopyML(locML,srcML);
	    }

            locres = Nnonz ? locres/Nnonz : res;

            if ( baseML.isRegression() )
            {
                locres = sqrt(locres);
            }

	    cnt += loccnt;
	    cfm += loccfm;
	    res += locres;

            repres("&",k) = locres;
	}

        if ( !oneoff )
        {
            cnt *= 1/((double) numreps);
            cfm *= 1/((double) numreps);
            res *= 1/((double) numreps);
        }

        { ML_Base *temp = srcML; srcML = locML; locML = temp; }

        MEMDEL(locML);
    }

//    if ( baseML.isRegression() )
//    {
//        res = sqrt(res);
//    }

if ( !suppressfb )
{
errstream() << "\n";
}
    return res;
}



double calcLOORecall(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int startpoint, int isLOO, int calcgvarres, int suppressfb)
{
    int oneoff = startpoint & 2;
    startpoint &= 1;

    NiceAssert( !oneoff );
    (void) oneoff;

    startpoint = startpoint || isONN(baseML);

    int i,cla,clb,isReTrain;
    double res = 0.0;
    int N = baseML.N();
    int Nnz = baseML.N()-baseML.NNC(0);
    int Nclasses = baseML.numInternalClasses();
    int astategood;

    cnt.resize(Nclasses);
    cfm.resize(Nclasses,Nclasses+1);

    cnt.zero();
    cfm.zero();

    resh.resize(N);
    resg.resize(N);

    resh.zero();
    resg.zero();

    if ( calcgvarres )
    {
        gvarres.resize(N);
        gvarres.zero();
    }

    if ( N )
    {
        ML_Base *locML = (ML_Base *) &baseML;
        ML_Base *srcML = (ML_Base *) &baseML;

        if ( isLOO )
	{
            locML = makeNewML(baseML.type(),baseML.subtype());
            NiceAssert( locML );
            semicopyML(locML,&baseML);
	}

        ML_Base *temp = srcML; srcML = locML; locML = temp;

        int locisenable;
        gentype locy;

        for ( i = 0 ; i < N ; i++ )
	{
	    isReTrain = 0;

            locisenable = isVectorEnabled(i,locML);
            locy        = (*locML).y()(i);

            if ( locisenable )
	    {
                astategood = isVectorActive(i,locML);

if ( !suppressfb )
{
nullPrint(errstream(),"~~~~~",5);
nullPrint(errstream(),i,-5);
}
                if ( ( astategood || !isTrainedML(locML) ) && isLOO )
		{
                    disableVector(i,locML);

                    if ( startpoint )
                    {
                        resetGlobal(locML);
                    }

                    trainGlobal(locML,( i == N-1 ));

                    isReTrain = 1;
		}
	    }

            else if ( isLOO && ( i == N-1 ) )
            {
                trainGlobal(locML,( i == N-1 ));
            }
//if ( !suppressfb )
//{
//nullPrint(errstream(),".");
//}

            (*locML).ghTrainingVector(resh("&",i),resg("&",i),i);

            if ( calcgvarres )
            {
                gentype dummy;

                (*locML).varTrainingVector(gvarres("&",i),dummy,i);
            }

            if ( locisenable )
            {
                if ( (*locML).isClassifier() && ishzero(resh(i)) )
                {
                    res += 1;

                    cla = (*locML).getInternalClass(locy);
                    clb = Nclasses;
                }

                else
                {
                    res += (*locML).calcDist(resh(i),locy,i,locisenable);

                    cla = (*locML).getInternalClass(locy);
                    clb = (*locML).getInternalClass((resh)(i));
                }

                ++(cnt("&",cla));
                ++(cfm("&",cla,clb));

                if ( isReTrain && isLOO )
                {
                    semicopyML(locML,srcML);
                }
            }
        }

        if ( isLOO )
        {
            { ML_Base *temp = srcML; srcML = locML; locML = temp; }

            MEMDEL(locML);
        }
    }

    res = Nnz ? res/Nnz : res;

    if ( baseML.isRegression() )
    {
        res = sqrt(res);
    }

if ( !suppressfb )
{
errstream() << "\n";
}
    return res;
}

double calcTest(const ML_Base &baseML, const Vector<SparseVector<gentype> > &xtest, const Vector<gentype> &ytest, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int calcgvarres, int suppressfb)
{
    (void) suppressfb;

    NiceAssert( xtest.size() == ytest.size() );

    int i,cla,clb;
    double res = 0.0;
    int N = xtest.size();
    int Nnz = N;
    int Nclasses = baseML.numInternalClasses();

    cnt.resize(Nclasses);
    cfm.resize(Nclasses,Nclasses+1);

    cnt.zero();
    cfm.zero();

    resh.resize(N);
    resg.resize(N);

    resh.zero();
    resg.zero();

    if ( calcgvarres )
    {
        gvarres.resize(N);
        gvarres.zero();
    }

    if ( N )
    {
        for ( i = 0 ; i < N ; i++ )
	{
            baseML.gh(resh("&",i),resg("&",i),xtest(i));

            if ( calcgvarres )
            {
                gentype dummy;

                baseML.var(gvarres("&",i),dummy,xtest(i));
            }

            // NB: for svm_planar if fff(3) present then this is expert fff(3).  Otherwise vector output

            if ( baseML.isClassifier() && ishzero(resh(i)) )
            {
                res += 1;

                cla = baseML.getInternalClass(ytest(i));
                clb = Nclasses;
            }

            else
            {
                res += baseML.calcDist(resh(i),ytest(i),xtest(i).isfarfarfarindpresent(3) ? (int) xtest(i).fff(3) : -1);

                cla = baseML.getInternalClass(ytest(i));
                clb = baseML.getInternalClass(resh (i));
            }

            ++(cnt("&",cla));
            ++(cfm("&",cla,clb));
        }
    }

    res = Nnz ? res/Nnz : res;

    if ( baseML.isRegression() )
    {
        res = sqrt(res);
    }

    return res;
}

double calcSparSens(const ML_Base &baseML, Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, Vector<Vector<gentype> > &resh, Vector<Vector<gentype> > &resg, Vector<Vector<gentype> > &gvarres, int minbad, int maxbad, double noisemean, double noisevar, int startpoint, int calcgvarres, int suppressfb)
{
    NiceAssert( maxbad >= minbad );
    NiceAssert( minbad >= 0 );
    NiceAssert( noisevar >= 0 );

    int numreps = maxbad-minbad+1;

    startpoint = startpoint || isONN(baseML);

    int i,j,k;
    double res = 0;
    int N = baseML.N();
    int Nnz = baseML.N()-baseML.NNC(0);
    int Nclasses = baseML.numInternalClasses();
    int xdim = baseML.xspaceDim();

    cnt.resize(Nclasses);
    cfm.resize(Nclasses,Nclasses+1);

    cnt.zero();
    cfm.zero();

    repres.resize(numreps);
    repres.zero();

    resg.resize(numreps);
    resh.resize(numreps);

    if ( calcgvarres )
    {
        gvarres.resize(numreps);
    }

    for ( i = 0 ; i < numreps ; i++ )
    {
        resg("&",i).resize(N);
        resg("&",i).zero();

        resh("&",i).resize(N);
        resh("&",i).zero();

        if ( calcgvarres )
        {
            gvarres("&",i).resize(N);
            gvarres("&",i).zero();
        }
    }

    if ( Nnz )
    {
	Vector<double> loccnt(cnt);
	Matrix<double> loccfm(cfm);
        double locres = 0.0;

        ML_Base *locML = (ML_Base *) &baseML;
        ML_Base *srcML = (ML_Base *) &baseML;

        locML = makeNewML(baseML.type(),baseML.subtype());
        NiceAssert( locML );
        copyML(locML,&baseML);

        int locisenabled;
        gentype locy;

        for ( k = 0 ; k < minbad ; k++ )
        {
            for ( j = 0 ; j < N ; j++ )
            {
                SparseVector<gentype> xj = ((*locML).x())(j);

                double temp;

                randnfill(temp);

                xj("[]",xdim+1) = noisemean+(temp*noisevar);

                (*locML).setx(j,xj);
            }

            xdim++;
        }

        for ( k = 0 ; k < numreps ; k++ )
        {
            loccnt.zero();
            loccfm.zero();
            locres = 0.0;

            int cla,clb;
            int Nnonz = 0;

            if ( startpoint )
            {
                resetGlobal(locML);
            }

if ( !suppressfb )
{
nullPrint(errstream(),"~~~~~",5);
nullPrint(errstream(),k,-5); 
}
            trainGlobal(locML,( k == numreps-1 ));

            for ( j = 0 ; j < N ; j++ )
            {
                locisenabled = isVectorEnabled(j,locML);
                locy         = (*locML).y()(j);

                if ( locisenabled )
                {
                    locML->ghTrainingVector(resh("&",k)("&",j),resg("&",k)("&",j),j);

                    if ( calcgvarres )
                    {
                        gentype dummy;

                        locML->varTrainingVector(gvarres("&",k)("&",j),dummy,j);
                    }

                    // NB: isenabled actually returns d for MLs.  For ML_Scalar,
                    //     d sets whether a point is an upper bound, a lower bound
                    //     or a standard target type.  This information is needed
                    //     to calculate distance.  The isenabled(i) actually returns
                    //     d for MLs (d = 0 means disabled).  Hence the following
                    //     line passes isenabled(l) to calcDist.

                    if ( (*locML).isClassifier() && ishzero(resh(k)(j)) )
                    {
                        locres += 1;

                        cla = (*locML).getInternalClass(locy);
                        clb = Nclasses;
                    }

                    else
                    {
                        locres += locML->calcDist(resh(k)(j),locy,j,locisenabled);

                        cla = (*locML).getInternalClass(locy);
                        clb = (*locML).getInternalClass((resh(k))(j));
                    }

                    ++(loccnt("&",cla));
                    ++(loccfm("&",cla,clb));

                    ++Nnonz;
                }
            }

            semicopyML(locML,srcML);

            locres = Nnonz ? locres/Nnonz : res;

            if ( baseML.isRegression() )
            {
                locres = sqrt(locres);
            }

            cnt += loccnt;
            cfm += loccfm;
            res += locres;

            repres("&",k) = locres;

            for ( j = 0 ; j < N ; j++ )
            {
                SparseVector<gentype> xj = ((*locML).x())(j);

                double temp;

                randnfill(temp);

                xj("[]",xdim+1) = noisemean+(temp*noisevar);

                (*locML).setx(j,xj);
            }

            xdim++;
	}

	cnt *= 1/((double) numreps);
	cfm *= 1/((double) numreps);
        res *= 1/((double) numreps);

        MEMDEL(locML);
    }

//    if ( baseML.isRegression() )
//    {
//        res = sqrt(res);
//    }

if ( !suppressfb )
{
errstream() << "\n";
}
    return res;
}


double assessResult(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, const Vector<gentype> &ytestresh, const Vector<gentype> &ytest, const Vector<int> &outkernind)
{
    NiceAssert( ytest.size() == ytestresh.size() );

    double res = 0;
    int i;
    int N = ytest.size();
    int Nclasses = baseML.numInternalClasses();
    int cla,clb;

    cnt.resize(Nclasses);
    cfm.resize(Nclasses,Nclasses+1);

    cnt = zeroint();
    cfm = zeroint();

    if ( N )
    {
        for ( i = 0 ; i < N ; i++ )
        {
            if ( baseML.isClassifier() && ishzero(ytestresh(i)) )
            {
                res += 1;

                cla = baseML.getInternalClass(ytest(i));
                clb = Nclasses;
            }

            else
            {
                res += baseML.calcDist(ytestresh(i),ytest(i),outkernind(i));

                cla = baseML.getInternalClass(ytest(i));
                clb = baseML.getInternalClass(ytestresh(i));
            }

            if ( ( cla >= 0 ) && ( clb >= 0 ) && ( cla < Nclasses ) && ( clb < Nclasses+1 ) )
            {
                ++(cnt("&",cla));
                ++(cfm("&",cla,clb));
            }
        }
    }

    res = N ? res/N : res;

    if ( baseML.isRegression() )
    {
        res = sqrt(res);
    }

    return res;
}








void disableVector(int i, ML_Base *activeML)
{
    activeML->disable(i);

    return;
}

void disableVector(const Vector<int> &i, ML_Base *activeML)
{
    activeML->disable(i);

    return;
}

void resetGlobal(ML_Base *activeML)
{
    activeML->reset();

    return;
}

int trainGlobal(ML_Base *activeML, int islastopt)
{
    int dummyres = 0; // FIXME: should check return value

    if ( activeML->type() == 0 )
    {
        (dynamic_cast<SVM_Scalar &>(activeML->getML())).inEmm4Solve = islastopt ? 0 : 2;
    }

    else if ( activeML->type() == 1 )
    {
        (dynamic_cast<SVM_Binary &>(activeML->getML())).inEmm4Solve = islastopt ? 0 : 2;
    }

    else if ( activeML->type() == 2 )
    {
        (dynamic_cast<SVM_Single &>(activeML->getML())).inEmm4Solve = islastopt ? 0 : 2;
    }

//phantomxyz
    return activeML->train(dummyres);
}

void semicopyML(ML_Base *activeML, const ML_Base *srcML)
{
    activeML->semicopy(*srcML);

    return;
}

void copyML(ML_Base *activeML, const ML_Base *srcML)
{
    *activeML = *srcML;

    return;
}

int isVectorActive(int i, ML_Base *activeML)
{
    return (*activeML).alphaState()(i);
}

int isVectorEnabled(int i, ML_Base *activeML)
{
    return (*activeML).isenabled(i);
}

int isTrainedML(ML_Base *activeML)
{
    return (*activeML).isTrained();
}












void measureAccuracy(Vector<double> &res, const Vector<gentype> &resg, const Vector<gentype> &resh, const Vector<gentype> &ytarg, const Vector<int> &dstat, const ML_Base &ml)
{
    NiceAssert( resg.size() == ytarg.size() );
    NiceAssert( resg.size() == dstat.size() );
    
    res.resize(7);

    double &accuracy  = res("&",0);
    double &precision = res("&",1);
    double &recall    = res("&",2);
    double &f1score   = res("&",3);
    double &auc       = res("&",4);
    double &sparsity  = res("&",5);
    double &error     = res("&",6);

    accuracy  = 1;
    precision = 1;
    recall    = 1;
    f1score   = 1;
    auc       = 1;
    sparsity  = ml.sparlvl();
    error     = 0;

    int numClasses = ml.numClasses();

    NiceAssert( numClasses = 2 );

    int clvala = ( numClasses >= 1 ) ? (ml.ClassLabels())(zeroint()) : -1;
    int clvalb = ( numClasses >= 2 ) ? (ml.ClassLabels())(1)         : clvala+2;

    NiceAssert( clvala != clvalb );

    if ( clvalb < clvala )
    {
        int clvalx;

        clvalx = clvala;
        clvala = clvalb;
        clvalb = clvalx;
    }

    int Nnz = 0;
    int i,j;
    int N = resg.size();

    for ( i = 0 ; i < N ; i++ )
    {
        if ( dstat(i) )
        {
            Nnz++;
        }
    }

    Vector<double> g(N);
    Vector<int> d(N);
    Vector<int> dpred(N);
    Vector<int> dstatp(dstat);

    int isAUCcalcable = ( numClasses == 2 ) ? 1 : 0;

    if ( N > 1 )
    {
        // Grab data in useable form

        for ( i = N-1 ; i >= 0 ; i-- )
        {
            if ( !dstatp(i) )
            {
                g.remove(i);
                d.remove(i);
                dpred.remove(i);
                dstatp.remove(i);
                N--;
            }

            else
            {
                g("&",i) =  resg(i).isCastableToRealWithoutLoss() ? (double) resg(i) : 0;
                d("&",i) = (int) ytarg(i);

                dpred("&",i) = (int) resh(i);

                isAUCcalcable = resg(i).isCastableToRealWithoutLoss() ? isAUCcalcable : 0;
            }
        }

        NiceAssert( N == Nnz );
    }




    if ( N > 1 )
    {
        // Accuracy Calculation

        accuracy = 0;

        for ( i = 0 ; i < N ; i++ )
        {
            if ( dpred(i) == d(i) )
            {
                accuracy++;
            }
        }

        accuracy /= N;




        // Precision / recall / f1 score Calculation

        if ( numClasses == 2 )
        {
            // Precision/recall Calculation

            int trpos = 0; // true positive
            int fapos = 0; // false positive
            int faneg = 0; // false negative

            for ( i = 0 ; i < N ; i++ )
            {
                if ( ( dpred(i) == clvalb ) && ( d(i) == clvalb ) )
                {
                    trpos++;
                }

                else if ( ( dpred(i) == clvala ) && ( d(i) == clvalb ) )
                {
                    fapos++;
                }

                else if ( ( dpred(i) == clvalb ) && ( d(i) == clvala ) )
                {
                    faneg++;
                }
            }

            precision = trpos+fapos ? ((double) trpos)/((double) trpos+fapos) : 1;
            recall    = trpos+faneg ? ((double) trpos)/((double) trpos+faneg) : 1;

            // f1Score Calculation

            f1score = 2*precision*recall/(precision+recall+1e-10);
        }

        else
        {
            precision = -1;
            recall    = -1;
            f1score   = -1;
        }





        // AUC Calculation

        if ( isAUCcalcable )
        {
            // Sort data from smallest g to largest

            for ( i = 0 ; i < N-1 ; i++ )
            {
                for ( j = 1 ; j < N ; j++ )
                {
                    if ( g(j) < g(i) )
                    {
                        qswap(g("&",j),g("&",i));
                        qswap(dpred("&",j),dpred("&",i));
                        qswap(d("&",j),d("&",i));
                        qswap(dstatp("&",j),dstatp("&",i));
                    }
                }
            }

            // calculate tp/fp

            Vector<int> tp(N);
            Vector<int> fp(N);

            int NP = 0;
            int NN = 0;

            tp("&",zeroint()) = 0;
            fp("&",zeroint()) = 0;

            for ( i = 1 ; i < N ; i++ )
            {
                tp("&",i) = tp("&",i-1);
                fp("&",i) = fp("&",i-1);

                if ( d(i) == clvalb )
                {
                    tp("&",i)++;
                    NP++;
                }

                else
                {
                    fp("&",i)++;
                    NN++;
                }
            }

            // Calculate AUC

            auc = 0;

            for ( i = 1 ; i < N ; i++ )
            {
                if ( fp(i) > fp(i-1) )
                {
                    auc += ( (2*(fp(i)-fp(i-1))*NN*NP) != 0 ) ? ((double) (tp(i)+tp(i-1)))/((double) (2*(fp(i)-fp(i-1))*NN*NP)) : 0;
                }
            }
        }

        else
        {
            auc = -1;
        }
    }

    error = 1-accuracy;

    return;
}




void bootstrapML(ML_Base &ml)
{
    if ( ml.N() )
    {
        int j;

        // Start with all indices

        retVector<int> tmpva;

        Vector<int> dnzind(cntintvec(ml.N(),tmpva));

        // Remove all with d == 0

        for ( j = dnzind.size()-1 ; j >= 0 ; j-- )
        {
            if ( !((ml.d())(dnzind(j))) )
            {
                dnzind.remove(j);
            }
        }

        if ( dnzind.size() )
        {
            // Construct tally vector

            retVector<int> tmpva;

            Vector<int> dnztally(zerointvec(dnzind.size(),tmpva));

            // Randomly "select" elements from dnzind by incrementing the relevant element in tally

            for ( j = 0 ; j < dnzind.size() ; j++ )
            {
                dnztally("&",svm_rand()%(dnzind.size()))++;
            }

            // Remove elements *selected* (that way we have the indices of those for which we wish to set d = 0 left)

            for ( j = dnzind.size()-1 ; j >= 0 ; j-- )
            {
                if ( dnztally(j) )
                {
                    dnzind.remove(j);
                    dnztally.remove(j);
                }
            }

            // Set d = 0 for unselected elements (remaining dnztally = 0, so just use this as "d")

            ml.setd(dnzind,dnztally);
        }
    }

    return;
}

