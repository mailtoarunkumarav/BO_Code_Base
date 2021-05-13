
//
// 1-class Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//



// 
// Notes on Tax and Duin
//
// By default this code implements the Scholkopf variant:
//
//@techreport{Sch16,
//    author      = "Sch{\"o}lkopf, Bernhard and Platt, John~C. and Shawe-Taylor, John~C. and Smola, Alexander~J. and Williamson, Robert~C.",
//    title       = "Estimating the Support of a High-Dimensional Distribution",
//    institution = "Microsoft Research",
//    address     = "Redmond",
//    number      = "{MSR-TR-99-87}",
//    year        = "1999"}
// 
// but if xsingmethod == 1 then instead it implements Tax and Duin:
//
//@article{Tax1,
//    author      = "Tax, David~M.~J. and Duin, Robert~P.~W.",
//    title       = "Support Vector Data Description",
//    journal     = "Machine Learning",
//    volume      = "54",
//    number      = "1",
//    year        = "2004",
//    issn        = "0885--6125",
//    pages       = "45--66",
//    doi         = {http://dx.doi.org/10.1023/B:MACH.0000008084.60811.49},
//    publisher   = "Kluwer Academic Publishers",
//    address     = "Hingham, MA, USA"}
// 
// Under this variant we want to minimise:
// 
// 1/2 sum_ij alpha_i alpha_j K_ij + sum_i alpha_i g_i
// 
// such that: 0 <= alpha_i <= C  ( = 1/nu.N )
//            sum_i alpha_i = 1
// 
// where g_i = -1/2 K_ii
// 
// This is actually precisely SVM_Scalar with eps = 0 and 
// LinBiasForce = -1 (so gn = -1), so nothing special - this 
// is what we set up for Scholkopf in any case, plus the gp bit.
// However the interpretation/use of alpha_i is different.  In
// particular:
// 
// g(x) = || varphi(x) - a ||^2 - R^2    
// h(x) = -sign(g(x))                        (NOTE REVERSED SIGN CONVENTION - CLASS +1 if g(x) < 0)
// 
// where: a = sum_i alpha_i varphi(x_i)
// 
// Hence
// 
// g(x) = K(x,x) - 2 sum_j alpha_j K(x,x_j) + Q - R^2
// 
// where: Q = sum_ij alpha_i alpha_j K_ij
//        g(x_i) = 0 for i: 0 < alpha_i < C (free supports)
// 
// We also have that gBinary(x_i) = -g_i for i: 0 < alpha_i < C (this
// is because it's the optimal solution to SVM_Scalar with hp = 0, so
// by optimality gBinary, which is the gradient, is zero), where:
// 
// gBinary(x_i) = sum_j alpha_j K_ij + b
// 
// so, consequently:
// 
// 2 sum_j alpha_j K_ij = 2.gBinary(x_i) - 2.b
//                      = K_ii - 2.b   for i: 0 < alpha_i < C
// 
// and hence:
// 
// g(x) = K(x,x) - 2.gBinary(x) + 2.b + Q - R^2
// 
// g(x_i) = K_ii - 2.gBinary_i + 2.b + Q - R^2
//        = K_ii + 2.g_i + 2.b + Q - R^2 = 2.b + Q - R^2 = 0 for i : 0 < alpha_i < C
// 
// which tells us that:
// 
// 2.b + Q - R^2 = 0
// 
// This simplifies things quite a bit:
//
// g(x)   = K(x,x)     - 2 gBinary(x)
// g(x_i) = K(x_i,x_i) - 2 gBinary(x_i)
// 
// which is trivially.  So, to summarise:
//
// - Minimise with variable bias, gn = -1 (as normal here), gp_i = -1/2 K_ii
// - g(x) = K(x,x) - 2.gBinary(x)
// 
// That was for dclass == +1.  For dclass == +-1 we get:
//
// - Minimise with variable bias, gn = -dclass (as normal here), gp_i = -dclass/2 K_ii
// - g(x) = dclass.K(x,x) - 2.gBinary(x)
// 
// Final modification (IMPORTANT):
// ==============================
//
// Tax and Duin uses the reverse sign convention on g than we do, therefore we
// modify by instead using (bringing it back to our preferred sign convention and
// dividing by 2):
//
// - Minimise with variable bias, gn = -dclass (as normal here), gp_i = -dclass/2 K_ii
// - g(x) = gBinary(x) - dclass.K(x,x)/2 (that way g(x) >= 0 for class +1 (normal), <= 0 for class -1 (anomaly))
//



#include "svm_single.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

SVM_Single::SVM_Single() : SVM_Binary()
{
    diagkernvalcheat = NULL;
    dclass = +1;
    xsingmethod = 0;
    setaltx(NULL);
    seteps(0); // This is correct for Tax and Duin as well
    setLinBiasForce(-1); // This is correct for Tax and Duin as well
    return;
}

SVM_Single::SVM_Single(const SVM_Single &src) : SVM_Binary(src)
{
    diagkernvalcheat = NULL;
    dclass = src.dclass;
    xsingmethod = src.xsingmethod;
    setaltx(NULL);

    return;
}

SVM_Single::SVM_Single(const SVM_Single &src, const ML_Base *xsrc) : SVM_Binary(src,xsrc)
{
    diagkernvalcheat = NULL;
    dclass = src.dclass;
    xsingmethod = src.xsingmethod;
    setaltx(xsrc);

    return;
}

SVM_Single::~SVM_Single()
{
    return;
}

double SVM_Single::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int SVM_Single::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    int res = SVM_Binary::resetKernel(modind,onlyChangeRowI,updateInfo);

    if ( xsingmethod )
    {
        fixz();
    }

    return res;
}

int SVM_Single::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    int res = SVM_Binary::setKernel(xkernel,modind,onlyChangeRowI);

    if ( xsingmethod )
    {
        fixz();
    }

    return res;
}

void SVM_Single::fixz(void)
{
    int j;

    Vector<double> zz(N());

    for ( j = 0 ; j < N() ; j++ )
    {
        zz("&",j) = calcz(j);

        sety(zz);
    } 

    return;
}

void SVM_Single::setanomalyclass(int n)
{
    NiceAssert( ( -1 == n ) || ( +1 == n ) );

    if ( n != dclass )
    {
        // Need to store force value before change and set after so that
        // sign is reversed at classifier level as required.

        double biasforceval = LinBiasForce();
        dclass = n;

        if ( N() )
        {
            Vector<int> dval(N());

            dval = dclass;

            SVM_Binary::setd(dval);
        }

        setLinBiasForce(biasforceval);

        if ( xsingmethod )
        {
            fixz();
        }
    }

    return;
}

int SVM_Single::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    //int unusedvar = 0;
    int tempresh = 0;
    double tempresg = 0;

// - g(x) = gBinary(x) - dclass.K(x,x)/2 (that way g(x) >= 0 for class +1 (normal), <= 0 for class -1 (anomaly))

    gentype tempsomeg;
    gentype tempsomeh;

    //gTrainingVector(tempresg,unusedvar,i,retaltg);
    int temptemph = SVM_Scalar::ghTrainingVector(tempsomeh,tempsomeg,i,retaltg,pxyprodi);

    tempresg = (double) tempsomeg;

    tempresg -= dclass*( xsingmethod ? kerndiag()(i) : 0.0 )/2;
    tempresh = temptemph ? ( ( tempresg > 0 ) ? +1 : -1 ) : 0; // temptemph == 0 implies Bartlett's reject

    resh = tempresh;

    if ( retaltg )
    {
        gentype negres(-tempresg);
        gentype posres(tempresg);

        Vector<gentype> tempresg(2);

        tempresg("&",0) = negres;
        tempresg("&",1) = posres;

        resg = tempresg;
    }

    else
    {
        resg = tempresg;
    }

    return tempresh;
}

int SVM_Single::addTrainingVector(int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    (void) z;

    return SVM_Single::addTrainingVector(i,x,Cweigh,epsweigh);
}

int SVM_Single::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    (void) z;

    return SVM_Single::qaddTrainingVector(i,x,Cweigh,epsweigh);
}

int SVM_Single::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    (void) z;

    Vector<double> zz(z.size());

    zz = 0.0;

    return SVM_Single::addTrainingVector(i,x,Cweigh,epsweigh,zz);
}

int SVM_Single::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    (void) z;

    Vector<double> zz(z.size());

    zz = 0.0;

    return SVM_Single::qaddTrainingVector(i,x,Cweigh,epsweigh,zz);
}

int SVM_Single::addTrainingVector(int i, const SparseVector<gentype> &x, double Cweigh, double epsweigh, double z)
{
    SVM_Scalar::diagkernvalcheat = diagkernvalcheat;

    int res = SVM_Binary::addTrainingVector(i,dclass,x,Cweigh,epsweigh,diagkernvalcheat ? calcz(z,*diagkernvalcheat) : calcz(x,z));

    SVM_Scalar::diagkernvalcheat = NULL;

    return res;
}

int SVM_Single::qaddTrainingVector(int i, SparseVector<gentype> &x, double Cweigh , double epsweigh, double z)
{
    SVM_Scalar::diagkernvalcheat = diagkernvalcheat;

    int res = SVM_Binary::qaddTrainingVector(i,dclass,x,Cweigh,epsweigh,diagkernvalcheat ? calcz(z,*diagkernvalcheat) : calcz(x,z));

    SVM_Scalar::diagkernvalcheat = NULL;

    return res;
}

int SVM_Single::addTrainingVector(int i, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z)
{
    Vector<int> dd(x.size());

    dd = dclass;

    Vector<double> zz(z);

    int j;

    for ( j = 0 ; j < x.size() ; j++ )
    {
        zz("&",j) = calcz(x(j),z(j));
    }

    return SVM_Binary::addTrainingVector(i,dd,x,Cweigh,epsweigh,zz);
}

int SVM_Single::qaddTrainingVector(int i, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z)
{
    Vector<int> dd(x.size());

    dd = dclass;

    Vector<double> zz(z);

    int j;

    for ( j = 0 ; j < x.size() ; j++ )
    {
        zz("&",j) = calcz(x(j),z(j));
    }

//errstream() << "phantomx 0: " << zz << "\n";
    return SVM_Binary::qaddTrainingVector(i,dd,x,Cweigh,epsweigh,zz);
}

std::ostream &SVM_Single::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Single (unary) SVM\n\n";

    repPrint(output,'>',dep) << "1-class SVM class:  " << dclass      << "\n";
    repPrint(output,'>',dep) << "1-class SVM method: " << xsingmethod << "\n";
    repPrint(output,'>',dep) << "Base SVM: ";
    SVM_Binary::printstream(output, dep+1);
    repPrint(output,'>',dep) << "\n";

    return output;
}

std::istream &SVM_Single::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> dclass;
    input >> dummy; input >> xsingmethod;
    input >> dummy;
    SVM_Binary::inputstream(input);

    return input;
}
