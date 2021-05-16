
//
// Quadratic solver base
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "sQbase.h"
#include "vector.h"
#include "matrix.h"
#include "smatrix.h"
#include "optstate.h"

void makeeye(double &x, double diagval);
void makeeye(Matrix<double> &x, double diagval);

double getdiagelm(const double &x);
double getdiagelm(const Matrix<double> &x);

void makeeye(double &x, double diagval)
{
    x = diagval;

    return;
}

void makeeye(Matrix<double> &x, double diagval)
{
    NiceAssert( x.numRows() == x.numCols() );

    x = 0.0;

    if ( x.numRows() )
    {
        int i;

        for ( i = 0 ; i < x.numRows() ; i++ )
        {
            x("&",i,i) = diagval;
        }
    }

    return;
}

double getdiagelm(const double &x)
{
    return x;
}

double getdiagelm(const Matrix<double> &x)
{
    return x(zeroint(),zeroint());
}

const double &celmcalc(int i, int j, const void *Gphid);
const double &celmcalc(int i, int j, const void *Gphid)
{
    //Matrix<double> &Gp = *((Matrix<double> *) ((void **) Gphid)[0]);
    //Vector<double> &res = *((Vector<double> *) ((void **) Gphid)[1]);
    Matrix<double> &newGpsigma = *((Matrix<double> *) ((void **) Gphid)[2]);
    //int aN = *((int *) ((void **) Gphid)[3]);
    //int Naug = *((int *) ((void **) Gphid)[4]);
    //retVector<double> &tmpres = *((retVector<double> *) ((void **) Gphid)[5]);

    // Make sure we never sample from top right block (which is zero, though it should not be)

    return ( i < j ) ? newGpsigma(j,i) : newGpsigma(i,j);
}

double &elmcalc(int i, int j, void *Gphid);
double &elmcalc(int i, int j, void *Gphid)
{
    //Matrix<double> &Gp = *((Matrix<double> *) ((void **) Gphid)[0]);
    //Vector<double> &res = *((Vector<double> *) ((void **) Gphid)[1]);
    Matrix<double> &newGpsigma = *((Matrix<double> *) ((void **) Gphid)[2]);
    //int aN = *((int *) ((void **) Gphid)[3]);
    //int Naug = *((int *) ((void **) Gphid)[4]);
    //retVector<double> &tmpres = *((retVector<double> *) ((void **) Gphid)[5]);

    // Make sure we never sample from top right block (which is zero, though it should not be)

    return ( i < j ) ? newGpsigma("&",j,i) : newGpsigma("&",i,j);
}

const Vector<double> &crowcalc(int i, const void *Gphid);
const Vector<double> &crowcalc(int i, const void *Gphid)
{
    //Matrix<double> &Gp = *((Matrix<double> *) ((void **) Gphid)[0]);
    Vector<double> &res = *((Vector<double> *) ((void **) Gphid)[1]);
    Matrix<double> &newGpsigma = *((Matrix<double> *) ((void **) Gphid)[2]);
    int aN = *((int *) ((void **) Gphid)[3]);
    int Naug = *((int *) ((void **) Gphid)[4]);
    retVector<double> &tmpres = *((retVector<double> *) ((void **) Gphid)[5]);

    if ( i >= aN )
    {
        return newGpsigma(i,tmpres);
    }

    int j;

    for ( j = 0 ; j < aN+Naug ; j++ )
    {
        res("&",j) = newGpsigma(j,i); // Note order of indices here!
    }

    return res;
}

Vector<double> &rowcalc(int i, void *Gphid);
Vector<double> &rowcalc(int i, void *Gphid)
{
    //Matrix<double> &Gp = *((Matrix<double> *) ((void **) Gphid)[0]);
    Vector<double> &res = *((Vector<double> *) ((void **) Gphid)[1]);
    Matrix<double> &newGpsigma = *((Matrix<double> *) ((void **) Gphid)[2]);
    int aN = *((int *) ((void **) Gphid)[3]);
    int Naug = *((int *) ((void **) Gphid)[4]);
    retVector<double> &tmpres = *((retVector<double> *) ((void **) Gphid)[5]);

    if ( i >= aN )
    {
        return newGpsigma("&",i,tmpres);
    }

    int j;

    for ( j = 0 ; j < aN+Naug ; j++ )
    {
        res("&",j) = newGpsigma("&",j,i); // Note order of indices here!
    }

    return res;
}

// First two functions are basically identical
//
// I tried partial specialisation, but apparently partial specialisation of
// member functions isn't supported in C++, so redundant full specialisations
// it is.

template <>
int fullOptState<Vector<double>,Matrix<double> >::wrapsolve(svmvolatile int &killSwitch)
{
    int i,j,jP;
    int aN = x.aN();
    int bN = x.bN();
    int res = 0;
    Vector<double> zeroeg;

    NiceAssert( !GpnRowTwoSigned );

    x.initGradBetahpzero(Gp,Gp,Gn,Gpn,gp,gn);

    int Naug = 0;

    repover = 0;

    if ( bN )
    {
        for ( i = 0 ; i < bN ; i++ )
        {
            if ( abs2(x.betaGrad()(i)) > 0 )
            {
                Naug++;
            }
        }
    }

    if ( !Naug )
    {
doitagain:
        fullOptState *altxx = gencopy(aN,Gpfull,Gpsigmafull,Gn,Gpn,gp,gn,hp,lb,ub,GpnRowTwoMag);

        res = altxx->solve(killSwitch);

        MEMDEL(altxx);
    }

    else
    {
        zeroeg.resize(x.betaGrad()(zeroint()).size()) = 0.0;

        // Work out slacks to add

        Vector<int> chiadd;
        Vector<Vector<double> > chivec;

        for ( i = 0 ; i < bN ; i++ )
        {
            if ( abs2(x.betaGrad()(i)) > 0 )
            {
                chiadd.add(chiadd.size());
                chiadd("&",chiadd.size()-1) = i;

                chivec.add(chivec.size());
                chivec("&",chivec.size()-1) = x.betaGrad()(i);
            }
        }

        // Setup augmented problem

        Matrix<double> Gpnloc(Gpn);

        Vector<Vector<double> > gploc(gp);
        Vector<double> hploc(hp);

        Vector<double> lbloc(lb);
        Vector<double> ubloc(ub);

        Vector<double> GpnRowTwoMagloc(GpnRowTwoMag);

        Gpnloc.padRow(Naug);

        gploc.pad(Naug);
        hploc.pad(Naug);

        lbloc.pad(Naug);
        ubloc.pad(Naug);

        GpnRowTwoMagloc.pad(Naug);

        Matrix<Matrix<double> > GpExtend(Naug,aN+Naug);
        Matrix<double> GpsigmaExtend(Naug,aN+Naug);

        Matrix<double> GpExtendTemplateOffDiag = Gp(zeroint(),zeroint());
        Matrix<double> GpExtendTemplateOnDiag  = Gp(zeroint(),zeroint());

        makeeye(GpExtendTemplateOffDiag,0.0);
        makeeye(GpExtendTemplateOnDiag,CHIDIAGOFFSET);

        GpExtend = GpExtendTemplateOffDiag;
        GpsigmaExtend = 2*CHIDIAGOFFSET; // Gpii+Gpjj-2.Gpij

        for ( i = 0 ; i < Naug ; i++ )
        {
            Gpnloc("&",aN+i,chiadd(i)) = 1.0;
            GpExtend("&",i,aN+i) = GpExtendTemplateOnDiag;
            GpsigmaExtend("&",i,aN+i) = 0.0;

            for ( j = 0 ; j < aN ; j++ )
            {
                GpsigmaExtend("&",i,j) = getdiagelm(Gp(j,j))+CHIDIAGOFFSET;
            }

            gploc("&",aN+i) = zeroeg;
            hploc("&",aN+i) = DVALDEFAULT; // We use hp here as the sign is not defined!
            lbloc("&",aN+i) = -BETASLACKMAX; //0.0;
            ubloc("&",aN+i) = BETASLACKMAX; // 2*abs2(x.betaGrad()(i));
            GpnRowTwoMagloc("&",aN+i) = 0.0;
        }

        // Incorporate slacks

        for ( i = 0 ; i < Naug ; i++ )
        {
            x.addAlpha(i+aN,0,zeroeg);
        }

        Matrix<Matrix<double> > *newGp = smStack(&Gpfull,&GpExtend);
        Matrix<double> *newGpsigma = smStack(&Gpsigmafull,&GpsigmaExtend);
        Matrix<Matrix<double> > &Gploc = *newGp;
        //Matrix<double> &Gpsigmaloc = *newGpsigma;

        Vector<double> crowres(aN+Naug);
        retVector<double> crowresb;
        void *Gphid[6] = { (void *) &Gploc, (void *) &crowres, (void *) newGpsigma, (void *) &aN, (void *) &Naug, (void *) &crowresb };

        Matrix<double> Gpsigmaloc(celmcalc,crowcalc,Gphid,aN+Naug,aN+Naug);

        Gploc.resize(aN+Naug,aN+Naug);

        x.fixGradhpzero(Gploc,Gn,Gpnloc,gploc,gn);

        // Step chi to make all gradients zero

        for ( i = 0 ; i < Naug ; i++ )
        {
            j = i+aN;
            jP = x.findInAlphaZ(j);

            x.modAlphaZtoUFhpzero(jP,Gploc,Gploc,Gn,Gpnloc,gploc,gn);
            x.alphaStephpzero(j,chivec(i),Gploc,Gn,Gpnloc,gploc,gn);
        }

        // Solve

        int Nbad = 0;
        int Nreps = 0;

        do
        {
            {
                fullOptState *altxx = gencopy(aN,Gploc,Gpsigmaloc,Gn,Gpnloc,gploc,gn,hploc,lbloc,ubloc,GpnRowTwoMagloc);

                res = altxx->solve(killSwitch);

                repover |= altxx->repover;

                MEMDEL(altxx);
            }

            Nbad = 0;

            for ( i = Naug-1 ; i >= 0 ; i-- )
            {
                if ( ( (x.alphaRestrict())(i+aN) != 3 ) && ( abs2(x.alpha()(i+aN)) < (x.zerotol()) ) )
                {
                    if ( ( ( x.alphaRestrict(i+aN) == 0 ) && ( ( x.alphaState()(i+aN) == -1 ) || ( x.alphaState()(i+aN) == +1 ) ) ) ||
                         ( ( x.alphaRestrict(i+aN) == 1 ) && (                                   ( x.alphaState()(i+aN) == +1 ) ) ) ||
                         ( ( x.alphaRestrict(i+aN) == 2 ) && ( ( x.alphaState()(i+aN) == -1 )                                   ) )    )
                    {
                        x.changeAlphaRestricthpzero(i+aN,3,Gploc,Gploc,Gn,Gpnloc,gploc,gn);
                    }
                }

                if ( (x.alphaRestrict())(i+aN) != 3 )
                {
                    Nbad++;
                    hploc("&",aN+i) += DVALDEFAULT;
                }
            }

            if ( Nbad )
            {
                x.fixGradhpzero(Gploc,Gn,Gpnloc,gploc,gn);
            }

            Nreps++;
        }
        while ( Nbad && ( Nreps < MAXFEASREPS ) );

        // Zero, constrain and remove slacks

        for ( i = Naug-1 ; i >= 0 ; i-- )
        {
            x.changeAlphaRestricthpzero(i+aN,3,Gploc,Gploc,Gn,Gpnloc,gploc,gn);
        }

        for ( i = Naug-1 ; i >= 0 ; i-- )
        {
            x.removeAlpha(i+aN);
        }

        if ( Nbad )
        {
            x.fixGradhpzero(Gp,Gn,Gpn,gp,gn);
        }

        MEMDEL(newGp);
        MEMDEL(newGpsigma);

        if ( repover )
        {
            goto doitagain;
        }
    }

    return res;
}

template <>
int fullOptState<Vector<double>,double>::wrapsolve(svmvolatile int &killSwitch)
{
    int i,j,jP;
    int aN = x.aN();
    int bN = x.bN();
    int res = 0;
    Vector<double> zeroeg;

    NiceAssert( !GpnRowTwoSigned );

    x.initGradBetahpzero(Gp,Gp,Gn,Gpn,gp,gn);

    int Naug = 0;

    repover = 0;

    if ( bN )
    {
        for ( i = 0 ; i < bN ; i++ )
        {
            if ( abs2(x.betaGrad()(i)) > 0 )
            {
                Naug++;
            }
        }
    }

    if ( !Naug )
    {
doitagain:
        fullOptState *altxx = gencopy(aN,Gpfull,Gpsigmafull,Gn,Gpn,gp,gn,hp,lb,ub,GpnRowTwoMag);

        res = altxx->solve(killSwitch);

        MEMDEL(altxx);
    }

    else
    {
        zeroeg.resize(x.betaGrad()(zeroint()).size()) = 0.0;

        // Work out slacks to add

        Vector<int> chiadd;
        Vector<Vector<double> > chivec;

        for ( i = 0 ; i < bN ; i++ )
        {
            if ( abs2(x.betaGrad()(i)) > 0 )
            {
                chiadd.add(chiadd.size());
                chiadd("&",chiadd.size()-1) = i;

                chivec.add(chivec.size());
                chivec("&",chivec.size()-1) = x.betaGrad()(i);
            }
        }

        // Setup augmented problem

        Matrix<double> Gpnloc(Gpn);

        Vector<Vector<double> > gploc(gp);
        Vector<double> hploc(hp);

        Vector<double> lbloc(lb);
        Vector<double> ubloc(ub);

        Vector<double> GpnRowTwoMagloc(GpnRowTwoMag);

        Gpnloc.padRow(Naug);

        gploc.pad(Naug);
        hploc.pad(Naug);

        lbloc.pad(Naug);
        ubloc.pad(Naug);

        GpnRowTwoMagloc.pad(Naug);

        Matrix<double> GpExtend(Naug,aN+Naug);
        Matrix<double> GpsigmaExtend(Naug,aN+Naug);

        double GpExtendTemplateOffDiag = Gp(zeroint(),zeroint());
        double GpExtendTemplateOnDiag  = Gp(zeroint(),zeroint());

        makeeye(GpExtendTemplateOffDiag,0.0);
        makeeye(GpExtendTemplateOnDiag,CHIDIAGOFFSET);

        GpExtend = GpExtendTemplateOffDiag;
        GpsigmaExtend = 2*CHIDIAGOFFSET; // Gpii+Gpjj-2.Gpij

        for ( i = 0 ; i < Naug ; i++ )
        {
            Gpnloc("&",aN+i,chiadd(i)) = 1.0;
            GpExtend("&",i,aN+i) = GpExtendTemplateOnDiag;
            GpsigmaExtend("&",i,aN+i) = 0.0;

            for ( j = 0 ; j < aN ; j++ )
            {
                GpsigmaExtend("&",i,j) = getdiagelm(Gp(j,j))+CHIDIAGOFFSET;
            }

            gploc("&",aN+i) = zeroeg;
            hploc("&",aN+i) = DVALDEFAULT; // We use hp here as the sign is not defined!
            lbloc("&",aN+i) = -BETASLACKMAX; //0.0;
            ubloc("&",aN+i) = BETASLACKMAX; // 2*abs2(x.betaGrad()(i));
            GpnRowTwoMagloc("&",aN+i) = 0.0;
        }

        // Incorporate slacks

        for ( i = 0 ; i < Naug ; i++ )
        {
            x.addAlpha(i+aN,0,zeroeg);
        }

        Matrix<double> *newGp = smStack(&Gpfull,&GpExtend);
        Matrix<double> *newGpsigma = smStack(&Gpsigmafull,&GpsigmaExtend);
        Matrix<double> &Gploc = *newGp;
        //Matrix<double> &Gpsigmaloc = *newGpsigma;

        Vector<double> crowres(aN+Naug);
        retVector<double> crowresb;
        void *Gphid[6] = { (void *) &Gploc, (void *) &crowres, (void *) newGpsigma, (void *) &aN, (void *) &Naug, (void *) &crowresb };

        Matrix<double> Gpsigmaloc(celmcalc,crowcalc,Gphid,aN+Naug,aN+Naug);

        Gploc.resize(aN+Naug,aN+Naug);

        x.fixGradhpzero(Gploc,Gn,Gpnloc,gploc,gn);

        // Step chi to make all gradients zero

        for ( i = 0 ; i < Naug ; i++ )
        {
            j = i+aN;
            jP = x.findInAlphaZ(j);

            x.modAlphaZtoUFhpzero(jP,Gploc,Gploc,Gn,Gpnloc,gploc,gn);
            x.alphaStephpzero(j,chivec(i),Gploc,Gn,Gpnloc,gploc,gn);
        }

        // Solve

        int Nbad = 0;
        int Nreps = 0;

        do
        {
            {
                fullOptState *altxx = gencopy(aN,Gploc,Gpsigmaloc,Gn,Gpnloc,gploc,gn,hploc,lbloc,ubloc,GpnRowTwoMagloc);

                res = altxx->solve(killSwitch);

                repover |= altxx->repover;

                MEMDEL(altxx);
            }

            Nbad = 0;

            for ( i = Naug-1 ; i >= 0 ; i-- )
            {
                if ( ( (x.alphaRestrict())(i+aN) != 3 ) && ( abs2(x.alpha()(i+aN)) < (x.zerotol()) ) )
                {
                    if ( ( ( x.alphaRestrict(i+aN) == 0 ) && ( ( x.alphaState()(i+aN) == -1 ) || ( x.alphaState()(i+aN) == +1 ) ) ) ||
                         ( ( x.alphaRestrict(i+aN) == 1 ) && (                                   ( x.alphaState()(i+aN) == +1 ) ) ) ||
                         ( ( x.alphaRestrict(i+aN) == 2 ) && ( ( x.alphaState()(i+aN) == -1 )                                   ) )    )
                    {
                        x.changeAlphaRestricthpzero(i+aN,3,Gploc,Gploc,Gn,Gpnloc,gploc,gn);
                    }
                }

                if ( (x.alphaRestrict())(i+aN) != 3 )
                {
                    Nbad++;
                    hploc("&",aN+i) += DVALDEFAULT;
                }
            }

            if ( Nbad )
            {
                x.fixGradhpzero(Gploc,Gn,Gpnloc,gploc,gn);
            }

            Nreps++;
        }
        while ( Nbad && ( Nreps < MAXFEASREPS ) );

        // Zero, constrain and remove slacks

        for ( i = Naug-1 ; i >= 0 ; i-- )
        {
            x.changeAlphaRestricthpzero(i+aN,3,Gploc,Gploc,Gn,Gpnloc,gploc,gn);
        }

        for ( i = Naug-1 ; i >= 0 ; i-- )
        {
            x.removeAlpha(i+aN);
        }

        if ( Nbad )
        {
            x.fixGradhpzero(Gp,Gn,Gpn,gp,gn);
        }

        MEMDEL(newGp);
        MEMDEL(newGpsigma);

        if ( repover )
        {
            goto doitagain;
        }
    }

    return res;
}



int istrig = 0;

//int fullOptState<double,double>::wrapsolve(svmvolatile int &killSwitch);
template<>
int fullOptState<double,double>::wrapsolve(svmvolatile int &killSwitch)
{
    int i,j,jP;
    int aN = x.aN();
    int bN = x.bN();
    int res = 0;
    double zeroeg = 0.0;

    x.initGradBeta(Gp,Gp,Gn,Gpn,gp,gn,hp);

    int Naug = 0;

    repover = 0;

    if ( bN )
    {
        for ( i = 0 ; i < bN ; i++ )
        {
            if ( ( ( abs2(x.betaGrad()(i)) > 0 ) && ( x.betaRestrict()(i) == 0 ) && ( ( i < bN-1 ) || !GpnRowTwoSigned ) ) ||
                 ( ( x.betaGrad()(i)       > 0 ) && ( x.betaRestrict()(i) == 1 ) && ( ( i < bN-1 ) || !GpnRowTwoSigned ) ) ||
                 ( ( x.betaGrad()(i)       < 0 ) && ( x.betaRestrict()(i) == 2 ) && ( ( i < bN-1 ) || !GpnRowTwoSigned ) )     )
            {
                Naug++;
            }
        }
    }

    if ( !Naug )
    {
doitagain:
        fullOptState<double,double> *altxx = gencopy(aN,Gpfull,Gpsigmafull,Gn,Gpn,gp,gn,hp,lb,ub,GpnRowTwoMag);

        res = altxx->solve(killSwitch);

        MEMDEL(altxx);
    }

    else
    {
        // Work out slacks to add

        Vector<int> chiadd;
        Vector<double> chisign;
        Vector<double> chivec;

        for ( i = 0 ; i < bN ; i++ )
        {
            if ( ( abs2(x.betaGrad()(i)) > 0 ) && ( x.betaRestrict()(i) == 0 ) && ( ( i < bN-1 ) || !GpnRowTwoSigned ) )
            {
                chiadd.add(chiadd.size());
                chiadd("&",chiadd.size()-1) = i;

                chisign.add(chisign.size());
                chisign("&",chisign.size()-1) = ( x.betaGrad()(i) < 0 ) ? +1 : -1;

                chivec.add(chivec.size());
                chivec("&",chivec.size()-1) = abs2(x.betaGrad()(i));
            }

            else if ( ( x.betaGrad()(i) > 0 ) && ( x.betaRestrict()(i) == 1 ) && ( ( i < bN-1 ) || !GpnRowTwoSigned ) )
            {
                chiadd.add(chiadd.size());
                chiadd("&",chiadd.size()-1) = i;

                chisign.add(chisign.size());
                chisign("&",chisign.size()-1) = -1;

                chivec.add(chivec.size());
                chivec("&",chivec.size()-1) = x.betaGrad()(i);
            }

            else if ( ( x.betaGrad()(i) < 0 ) && ( x.betaRestrict()(i) == 2 ) && ( ( i < bN-1 ) || !GpnRowTwoSigned ) )
            {
                chiadd.add(chiadd.size());
                chiadd("&",chiadd.size()-1) = i;

                chisign.add(chisign.size());
                chisign("&",chisign.size()-1) = +1;

                chivec.add(chivec.size());
                chivec("&",chivec.size()-1) = -x.betaGrad()(i);
            }
        }

        // Setup augmented problem

        Matrix<double> Gpnloc(Gpn);

        Vector<double> gploc(gp);
        Vector<double> hploc(hp);

        Vector<double> lbloc(lb);
        Vector<double> ubloc(ub);

        Vector<double> GpnRowTwoMagloc(GpnRowTwoMag);

        Gpnloc.padRow(Naug);

        gploc.pad(Naug);
        hploc.pad(Naug);

        lbloc.pad(Naug);
        ubloc.pad(Naug);

        GpnRowTwoMagloc.pad(Naug);

        Matrix<double> GpExtend(Naug,aN+Naug);
        Matrix<double> GpsigmaExtend(Naug,aN+Naug);

        GpExtend = 0.0;
        GpsigmaExtend = 2*CHIDIAGOFFSET; // Gpii+Gpjj-2.Gpij

        for ( i = 0 ; i < Naug ; i++ )
        {
            Gpnloc("&",aN+i,chiadd(i)) = chisign(i);
            GpExtend("&",i,aN+i) = CHIDIAGOFFSET;
            GpsigmaExtend("&",i,aN+i) = 0.0;

            for ( j = 0 ; j < aN ; j++ )
            {
                GpsigmaExtend("&",i,j) = Gp(j,j)+GpExtend(i,aN+i);
            }

            gploc("&",aN+i) = DVALDEFAULT;
            hploc("&",aN+i) = 0.0;
            lbloc("&",aN+i) = -BETASLACKMAX; // 0.0;
            ubloc("&",aN+i) = BETASLACKMAX; // 2*abs2(x.betaGrad()(i));
            GpnRowTwoMagloc("&",aN+i) = 0.0;
        }

        for ( i = 0 ; i < Naug ; i++ )
        {
            for ( j = 0 ; j < Naug ; j++ )
            {
                if ( i != j )
                {
                    GpsigmaExtend("&",i,j+aN) = GpExtend(i,i+aN)+GpExtend(j,j+aN);
                }
            }
        }

        // Incorporate slacks

        for ( i = 0 ; i < Naug ; i++ )
        {
            x.addAlpha(i+aN,1,zeroeg);
        }

        Matrix<double> *newGp = smStack(&Gpfull,&GpExtend);
        Matrix<double> *newGpsigma = smStack(&Gpsigmafull,&GpsigmaExtend);
        Matrix<double> &Gploc = *newGp;
        //Matrix<double> &Gpsigmaloc = *newGpsigma;

        Vector<double> crowres(aN+Naug);
        retVector<double> crowresb;
        void *Gphid[6] = { (void *) &Gploc, (void *) &crowres, (void *) newGpsigma, (void *) &aN, (void *) &Naug, (void *) &crowresb };

        Matrix<double> Gpsigmaloc(elmcalc,rowcalc,Gphid,celmcalc,crowcalc,Gphid,aN+Naug,aN+Naug);

        Gploc.resize(aN+Naug,aN+Naug);
        Gpsigmaloc.resize(aN+Naug,aN+Naug);

//errstream() << "phantomx 0: Gp = " << Gploc << "\n";
//errstream() << "phantomx 0: Gpsigma = " << Gpsigmaloc << "\n";
//errstream() << "phantomx 0: Gn = " << Gn << "\n";
//errstream() << "phantomx 0: Gpn = " << Gpnloc << "\n";
//errstream() << "phantomx 0: gp = " << gploc << "\n";
//errstream() << "phantomx 0: gn = " << gn << "\n";
//errstream() << "phantomx 0: hp = " << hploc << "\n";
//errstream() << "phantomx 0: lb = " << lbloc << "\n";
//errstream() << "phantomx 0: ub = " << ubloc << "\n";
//errstream() << "phantomx 0: magloc = " << GpnRowTwoMagloc << "\n";
//errstream() << "phantomx 0: x = " << x << "\n";
//istrig = 1;
        x.fixGrad(Gploc,Gn,Gpnloc,gploc,gn,hploc);

        // Step chi to make all gradients zero

        for ( i = 0 ; i < Naug ; i++ )
        {
            j = i+aN;
            jP = x.findInAlphaZ(j);

            x.modAlphaZtoUF(jP,Gploc,Gploc,Gn,Gpnloc,gploc,gn,hploc);
            x.alphaStep(j,chivec(i),Gploc,Gn,Gpnloc,gploc,gn,hploc);
        }

        // Solve

        int Nbad = 0;
        int Nreps = 0;

//errstream() << "phantomx 0: Gp = " << Gploc << "\n";
//errstream() << "phantomx 0: Gpsigma = " << Gpsigmaloc << "\n";
//errstream() << "phantomx 0: Gn = " << Gn << "\n";
//errstream() << "phantomx 0: Gpn = " << Gpnloc << "\n";
//errstream() << "phantomx 0: gp = " << gploc << "\n";
//errstream() << "phantomx 0: gn = " << gn << "\n";
//errstream() << "phantomx 0: hp = " << hploc << "\n";
//errstream() << "phantomx 0: lb = " << lbloc << "\n";
//errstream() << "phantomx 0: ub = " << ubloc << "\n";
//errstream() << "phantomx 0: magloc = " << GpnRowTwoMagloc << "\n";
//errstream() << "phantomx 0: x = " << x << "\n";
        do
        {
            {
                fullOptState<double,double> *altxx = gencopy(aN,Gploc,Gpsigmaloc,Gn,Gpnloc,gploc,gn,hploc,lbloc,ubloc,GpnRowTwoMagloc);

                res = altxx->solve(killSwitch);

                repover |= altxx->repover;

                MEMDEL(altxx);
            }

            Nbad = 0;

            for ( i = Naug-1 ; i >= 0 ; i-- )
            {
                if ( ( (x.alphaRestrict())(i+aN) != 3 ) && ( x.alpha()(i+aN) >= -(x.zerotol()) ) && ( x.alpha()(i+aN) <= (x.zerotol()) ) )
                {
                    if ( ( ( x.alphaRestrict(i+aN) == 0 ) && ( ( x.alphaState()(i+aN) == -1 ) || ( x.alphaState()(i+aN) == +1 ) ) ) ||
                         ( ( x.alphaRestrict(i+aN) == 1 ) && (                                   ( x.alphaState()(i+aN) == +1 ) ) ) ||
                         ( ( x.alphaRestrict(i+aN) == 2 ) && ( ( x.alphaState()(i+aN) == -1 )                                   ) )    )
                    {
                        x.changeAlphaRestrict(i+aN,3,Gploc,Gploc,Gn,Gpnloc,gploc,gn,hploc);
                    }
                }

                if ( (x.alphaRestrict())(i+aN) != 3 )
                {
                    Nbad++;
                    gploc("&",aN+i) += DVALDEFAULT;
                }
            }

            if ( Nbad )
            {
                x.fixGrad(Gploc,Gn,Gpnloc,gploc,gn,hploc);
            }

            Nreps++;
        }
        while ( Nbad && ( Nreps < MAXFEASREPS ) );

        // Zero, constrain and remove slacks

        for ( i = Naug-1 ; i >= 0 ; i-- )
        {
            x.changeAlphaRestrict(i+aN,3,Gploc,Gploc,Gn,Gpnloc,gploc,gn,hploc);
        }

        for ( i = Naug-1 ; i >= 0 ; i-- )
        {
            x.removeAlpha(i+aN);
        }

        if ( Nbad )
        {
            x.fixGrad(Gp,Gn,Gpn,gp,gn,hp);
        }

        MEMDEL(newGp);
        MEMDEL(newGpsigma);

        if ( repover )
        {
            goto doitagain;
        }
    }

    return res;
}


















double fullfixbasea(fullOptState<double,double> &x, void *p, const Vector<double> &diagoff, const Vector<double> &gradoff, double &sf)
{
    return x.intfullfixbasea(x,p,diagoff,gradoff,sf);
}

double fullfixbaseb(fullOptState<Vector<double>,double> &x, void *p, const Vector<double> &diagoff, const Vector<double> &gradoff, double &sf)
{
    return x.intfullfixbaseb(x,p,diagoff,gradoff,sf);
}


