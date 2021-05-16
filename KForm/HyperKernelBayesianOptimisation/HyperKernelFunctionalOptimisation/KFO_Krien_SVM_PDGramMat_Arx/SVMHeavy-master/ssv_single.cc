
//
// 1-class Classification SSV
//
// Version: 7
// Date: 06/12/2017
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "ssv_single.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

SSV_Single::SSV_Single() : SSV_Binary()
{
    dclass = +1;
    setaltx(NULL);
    return;
}

SSV_Single::SSV_Single(const SSV_Single &src) : SSV_Binary(src)
{
    dclass = +1;
    setaltx(NULL);

    return;
}

SSV_Single::SSV_Single(const SSV_Single &src, const ML_Base *xsrc) : SSV_Binary(src,xsrc)
{
    dclass = +1;
    setaltx(xsrc);

    return;
}

SSV_Single::~SSV_Single()
{
    return;
}

double SSV_Single::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int SSV_Single::setanomalclass(int n)
{
    NiceAssert( ( -1 == n ) || ( +1 == n ) );

    if ( n != dclass )
    {
        // Need to store force value before change and set after so that
        // sign is reversed at classifier level as required.

        double biasforceval = biasForce();
        dclass = n;

        if ( N() )
        {
            Vector<int> dval(N());

            dval = dclass;

            SSV_Binary::setd(dval);
        }

        setBiasForce(biasforceval);
    }

    return 1;
}

int SSV_Single::addTrainingVector(int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    (void) z;

    gentype az(dclass);

    return SSV_Binary::addTrainingVector(i,az,x,Cweigh,epsweigh);
}

int SSV_Single::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    (void) z;

    gentype az(dclass);

    return SSV_Binary::qaddTrainingVector(i,az,x,Cweigh,epsweigh);
}

int SSV_Single::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    (void) z;

    Vector<gentype> az(z.size());
    gentype ad(dclass);

    az = ad;

    return SSV_Binary::addTrainingVector(i,az,x,Cweigh,epsweigh);
}

int SSV_Single::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    (void) z;

    Vector<gentype> az(z.size());
    gentype ad(dclass);

    az = ad;

    return SSV_Binary::qaddTrainingVector(i,az,x,Cweigh,epsweigh);
}

int SSV_Single::train(int &res, svmvolatile int &killSwitch)
{
    //int Nnz = N()-NNC(0);
    int Nnz = sum(xstate());
    int ares = 0;

    if ( Nnz )
    {
        double baseBiasForce = biasForce();
(void) baseBiasForce;

/*
        int i;
        int Nerr;

        double bflb = 0;
        double bfub = baseBiasForce;
        double bfval = baseBiasForce;
        gentype dummya,dummyb;

        // Work out ub

errstream() << "Calculating UB\n";
        while ( 1 )
        {
            ares |= setBiasForce(bfub);
            ares |= setsigma(bfub);
            ares |= SSV_Binary::train(res,killSwitch);
            ares |= setb(b()-1.0);

            Nerr = 0;
            
            for ( i = 0 ; i < N() ; i++ )
            {
                if ( ghTrainingVector(dummya,dummyb,i) != dclass )
                {
                    Nerr++;
                }
//errstream() << "\n\n\n\nphantomx 0: " << dummya << "\n";
//errstream() << "phantomx 1: " << dummyb << "\n\n\n\n\n";
            }
            
errstream() << "\n... error = " << ((double) Nerr)/((double) Nnz) << " (bf = " << bfub << ")\n\n\n";
            if ( ((double) Nerr)/((double) Nnz) >= baseBiasForce )
            {
                break;
            }

            bflb  = bfub;
            bfub *= 2;
        }
errstream() << "\n... done\n\n\n\n";

        // Work out lb if required

        if ( bflb == 0 )
        {
errstream() << "Calculating LB\n";
            bflb = bfub/2;

            while ( 1 )
            {
                ares |= setBiasForce(bflb);
                ares |= setsigma(bflb);
                ares |= SSV_Binary::train(res,killSwitch);
                ares |= setb(b()-1.0);

                Nerr = 0;
            
                for ( i = 0 ; i < N() ; i++ )
                {
                    if ( ghTrainingVector(dummya,dummyb,i) != dclass )
                    {
                        Nerr++;
                    }
                }
            
errstream() << "\n... error = " << ((double) Nerr)/((double) Nnz) << " (bf = " << bflb << ")\n\n\n";
                if ( ((double) Nerr)/((double) Nnz) <= baseBiasForce )
                {
                    break;
                }

                bflb  /= 2;
            }
errstream() << "\n... done\n\n\n\n";
        }

        // Solve

errstream() << "Calculating final result\n";
        while ( 1 )
        {
            bfval = (bflb+bfub)/2;

            ares |= setBiasForce(bfval);
            ares |= setC(bfval);
            ares |= SSV_Binary::train(res,killSwitch);
            ares |= setb(b()-1.0);

            Nerr = 0;
            
            for ( i = 0 ; i < N() ; i++ )
            {
                if ( ghTrainingVector(dummya,dummyb,i) != dclass )
                {
                    Nerr++;
                }
            }
            
errstream() << "\n... error = " << ((double) Nerr)/((double) Nnz) << " (bf = " << bfval << ")\n\n\n";
            if ( ( abs2((((double) Nerr)/((double) N()))-baseBiasForce) < ssvtol() ) || ( abs2(bflb-bfub) < ssvtol() ) )
            {
                break;
            }

            else if ( ((double) Nerr)/((double) N()) < baseBiasForce )
            {
                bflb = bfval;
            }

            else
            {
                bfub = bfval;
            }
errstream() << "\n... done\n\n\n\n";
        }

        ares |= setBiasForce(baseBiasForce);
*/
        //ares |= setBiasForce(Nnz*baseBiasForce);
        //ares |= zmodel.setsigmaweight(Nzs(),Nnz*(zmodel.C())*baseBiasForce);
        ares |= SSV_Binary::train(res,killSwitch);
        //ares |= setBiasForce(baseBiasForce);
    }

    return ares;
}

std::ostream &SSV_Single::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Single (unary) SSV\n\n";

    repPrint(output,'>',dep) << "1-class SSV class: " << dclass << "\n";
    repPrint(output,'>',dep) << "Base SSV: ";
    SSV_Binary::printstream(output,dep+1);

    return output;
}

std::istream &SSV_Single::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> dclass;
    input >> dummy;
    SSV_Binary::inputstream(input);

    return input;
}

