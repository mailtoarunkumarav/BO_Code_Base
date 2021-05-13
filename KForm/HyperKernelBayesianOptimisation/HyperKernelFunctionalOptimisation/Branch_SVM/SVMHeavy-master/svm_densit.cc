
//
// Density estimation SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_densit.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

SVM_Densit::SVM_Densit() : SVM_Scalar()
{
    (SVM_Scalar::getKernel_unsafe()).setType(400);
    SVM_Scalar::resetKernel();
    set1NormCost();
    return;
}

SVM_Densit::SVM_Densit(const SVM_Densit &src) : SVM_Scalar(src)
{
    set1NormCost();
    return;

}
SVM_Densit::SVM_Densit(const SVM_Densit &src, const ML_Base *xsrc) : SVM_Scalar(src,xsrc)
{
    set1NormCost();
    return;
}

std::ostream &SVM_Densit::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Density estimation SVM\n\n";

    return SVM_Scalar::printstream(output,dep+1);
}

std::istream &SVM_Densit::inputstream(std::istream &input)
{
    return SVM_Scalar::inputstream(input);
}

// Modification and autoset functions

int SVM_Densit::setd(int i, int d)
{
    int res = SVM_Scalar::setd(i,d);
    fixz();
    return res;
}

int SVM_Densit::setd(const Vector<int> &i, const Vector<int> &d)
{
    int res = SVM_Scalar::setd(i,d);
    fixz();
    return res;
}

int SVM_Densit::setd(const Vector<int> &d)
{
    int res = SVM_Scalar::setd(d);
    fixz();
    return res;
}

// Training set control

int SVM_Densit::addTrainingVector (int i, double z, const SparseVector<gentype> &xx, double Cweigh, double epsweigh, int d)
{
    (void) i;
    (void) z;
    (void) xx;
    (void) Cweigh;
    (void) epsweigh;
    (void) d;

    throw("addTrainingVector with z not defined for density estimation");

    return 0;
}

int SVM_Densit::qaddTrainingVector(int i, double z,       SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    (void) x;
    (void) i;
    (void) z;
    (void) Cweigh;
    (void) epsweigh;
    (void) d;

    throw("qaddTrainingVector with z not defined for density estimation");

    return 0;
}

int SVM_Densit::addTrainingVector (int i, const Vector<double> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    (void) x;
    (void) i;
    (void) z;
    (void) Cweigh;
    (void) epsweigh;
    (void) d;

    throw("addTrainingVector with z not defined for density estimation");

    return 0;
}

int SVM_Densit::qaddTrainingVector(int i, const Vector<double> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    (void) x;
    (void) i;
    (void) z;
    (void) Cweigh;
    (void) epsweigh;
    (void) d;

    throw("qaddTrainingVector with z not defined for density estimation");

    return 0;
}

int SVM_Densit::addTrainingVector (int i, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    int res = SVM_Scalar::addTrainingVector(i,0.0,x,Cweigh,epsweigh,d);
    fixz();
    return res;
}

int SVM_Densit::qaddTrainingVector(int i,       SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    int res = SVM_Scalar::qaddTrainingVector(i,0.0,x,Cweigh,epsweigh,d);
    fixz();
    return res;
}

int SVM_Densit::addTrainingVector (int i, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    Vector<double> z(x.size());
    z = 0.0;

    int res = SVM_Scalar::addTrainingVector(i,z,x,Cweigh,epsweigh,d);
    fixz();
    return res;
}

int SVM_Densit::qaddTrainingVector(int i,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    Vector<double> z(x.size());
    z = 0.0;

    int res = SVM_Scalar::qaddTrainingVector(i,z,x,Cweigh,epsweigh,d);
    fixz();
    return res;
}

int SVM_Densit::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    int res = SVM_Scalar::removeTrainingVector(i,y,x);
    fixz();
    return res;
}

int SVM_Densit::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    (void) z;

    NiceAssert( z.isValNull() );

    double zz = 0.0;

    int res = SVM_Scalar::addTrainingVector(i,zz,x,Cweigh,epsweigh);
    fixz();

    return res;
}

int SVM_Densit::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    (void) z;

    NiceAssert( z.isValNull() );

    double zz = 0.0;

    int res = SVM_Scalar::qaddTrainingVector(i,zz,x,Cweigh,epsweigh);
    fixz();

    return res;
}

int SVM_Densit::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<double> zz(z.size());
    Vector<int> dd(z.size());

    zz = 0.0;
    dd = 2;

    int res = SVM_Scalar::addTrainingVector(i,zz,x,Cweigh,epsweigh,dd);
    fixz();

    return res;
}

int SVM_Densit::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<double> zz(z.size());
    Vector<int> dd(z.size());

    zz = 0.0;
    dd = 2;

    int res = SVM_Scalar::qaddTrainingVector(i,zz,x,Cweigh,epsweigh,dd);
    fixz();

    return res;
}

// Evaluation:

int SVM_Densit::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int unusedvar = 0;
    int res = gTrainingVector(resg.force_double(),unusedvar,i,retaltg,pxyprodi);
    resh = resg;

    return res;
}

int SVM_Densit::gTrainingVector(double &res, int &unusedvar, int i, int raw, gentype ***pxyprodi) const
{
    (void) raw;
    (void) pxyprodi;

    int iP;
    double Kxi = 0;

    NiceAssert( emm == 2 );

    int dtv = 0;

    if ( ( dtv = xtang(i) & 7 ) )
    {
        res = 0.0;

        if ( dtv > 0 )
        {
            if ( NLB() )
            {
                for ( iP = 0 ; iP < NLB() ; iP++ )
                {
                    densedKdx(Kxi,i,Q.pivAlphaLB()(iP));

                    res += (alphaR()((Q.pivAlphaLB()(iP))))*Kxi;
                }
            }

            if ( NUB() )
            {
                for ( iP = 0 ; iP < NUB() ; iP++ )
                {
                    densedKdx(Kxi,i,Q.pivAlphaUB()(iP));

                    res += (alphaR()((Q.pivAlphaUB()(iP))))*Kxi;
                }
            }

            if ( NF() )
            {
                for ( iP = 0 ; iP < NF() ; iP++ )
                {
                    densedKdx(Kxi,i,Q.pivAlphaF()(iP));

                    res += (alphaR()((Q.pivAlphaF()(iP))))*Kxi;
                }
            }
        }
    }

    else
    {
        res = biasR();

        if ( NLB() )
        {
            for ( iP = 0 ; iP < NLB() ; iP++ )
            {
                densedKdx(Kxi,i,Q.pivAlphaLB()(iP));

                res += (alphaR()((Q.pivAlphaLB()(iP))))*Kxi;
            }
        }

        if ( NUB() )
        {
            for ( iP = 0 ; iP < NUB() ; iP++ )
            {
                densedKdx(Kxi,i,Q.pivAlphaUB()(iP));

                res += (alphaR()((Q.pivAlphaUB()(iP))))*Kxi;
            }
        }

        if ( NF() )
        {
            for ( iP = 0 ; iP < NF() ; iP++ )
            {
                densedKdx(Kxi,i,Q.pivAlphaF()(iP));

                res += (alphaR()((Q.pivAlphaF()(iP))))*Kxi;
            }
        }
    }

    unusedvar = -1;

    if ( res > 0 )
    {
        unusedvar = +1;
    }

    return unusedvar;
}

int SVM_Densit::train(int &res, svmvolatile int &killSwitch)
{
    // This is a bit of a hack.  We bypass a whole stack of stuff in SVM_Scalar
    // and basically enforce the problem that we want

    SVM_Scalar::Qnp.resize(1,N());
    SVM_Scalar::Qn.resize(1,1);
    SVM_Scalar::qn.resize(1);
    SVM_Scalar::Qconstype = 0; // enforce lower bound not equality

    SVM_Scalar::Qnp = 1.0;
    SVM_Scalar::Qn = 0.0;
    SVM_Scalar::qn = -1.0;

    SVM_Scalar::alpharestrictoverride = 1;
    // The above forces alpha to be strictly positive without changing constraints

    return SVM_Scalar::train(res,killSwitch);
}


void SVM_Densit::fixz(void)
{
    gentype dummy(0.0);

    if ( N() )
    {
        int i,j,k;

        // This pivot vector will be used for ordering features smallest->largest

        retVector<int> tmpva;

        Vector<int> ableind(cntintvec(N(),tmpva));

        // Remove disabled variables

        for ( i = N()-1 ; i >= 0 ; i-- )
        {
            if ( !d()(i) )
            {
                ableind.remove(i);
            }
        }

        // xN is the number of disabled vectors
        // xm is the dimension of the training data

        int xN = ableind.size();
        int xm = xspaceDim();

        // ztot will be the target vector
        // sortpiv used to sort vectors

        Vector<double> ztot(N());
        Vector<int> sortpiv(ableind);

        ztot = 1.0;

        if ( xN && xm )
        {
            // Lots of assumptions here.  All features have values (and if
            // not then they default to zero).  All features are real.
            // All training vectors are nonsparse

            for ( k = 0 ; k < xm ; k++ )
            {
errstream() << "@" << k;
                if ( xN > 1 )
                {
                    for ( i = 0 ; i < xN-1 ; i++ )
                    {
                        for ( j = i+1 ; j < xN ; j++ )
                        {
                            if ( xelm(dummy,sortpiv(j),k) < xelm(dummy,sortpiv(i),k) )
                            {
                                sortpiv.squareswap(i,j);
                            }
                        }
                    }
                }

errstream() << ";";
                for ( i = 0 ; i < xN ; i++ )
                {
                    ztot("&",sortpiv(i)) *= ((double) i+1)/((double) xN);
                }
            }
errstream() << ".";
        }

        else if ( xN )
        {
            // Based on nothing use an arbitrary ordering

            for ( i = 0 ; i < xN ; i++ )
            {
                ztot("&",ableind(i)) *= ((double) i+1)/((double) xN);
            }
        }

        SVM_Scalar::sety(ztot);
    }

    return;
}
