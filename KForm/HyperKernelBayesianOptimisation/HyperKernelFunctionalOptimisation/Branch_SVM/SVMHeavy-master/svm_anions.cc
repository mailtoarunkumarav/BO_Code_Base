
//
// Anionic regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "svm_anions.h"
#include <iostream>
#include <sstream>
#include <string>



void AtoV(const d_anion &src, Vector<double> &dest);
void VtoA(const Vector<double> &src, d_anion &dest);
void VtoA(const Vector<gentype> &src, d_anion &dest);

void AtoVvect(const Vector<d_anion> &src, Vector<Vector<double> > &dest);
void VtoAvect(const Vector<Vector<double> > &src, Vector<d_anion> &dest);

void AtoV(const d_anion &src, Vector<double> &dest)
{
    dest.resize(src.size());

    if ( src.size() )
    {
	int j;

        for ( j = 0 ; j < src.size() ; j++ )
	{
            dest("&",j) = src(j);
	}
    }

    return;
}

void VtoA(const Vector<double> &src, d_anion &dest)
{
    int neword = ceilintlog2(src.size());

    NiceAssert( src.size() == 1<<neword );

    dest.setorder(neword);

    if ( src.size() )
    {
	int j;

        for ( j = 0 ; j < src.size() ; j++ )
	{
            dest("&",j) = src(j);
	}
    }

    return;
}

void VtoA(const Vector<gentype> &src, d_anion &dest)
{
    int neword = ceilintlog2(src.size());

    NiceAssert( src.size() == 1<<neword );

    dest.setorder(neword);

    if ( src.size() )
    {
	int j;

        for ( j = 0 ; j < src.size() ; j++ )
	{
            dest("&",j) = (double) src(j);
	}
    }

    return;
}

void AtoVvect(const Vector<d_anion> &src, Vector<Vector<double> > &dest)
{
    dest.resize(src.size());

    if ( dest.size() )
    {
        int i;

        for ( i = 0 ; i < dest.size() ; i++ )
        {
            AtoV(src(i),dest("&",i));
        }
    }

    return;
}

void VtoAvect(const Vector<Vector<double> > &src, Vector<d_anion> &dest)
{
    dest.resize(src.size());

    if ( dest.size() )
    {
        int i;

        for ( i = 0 ; i < dest.size() ; i++ )
        {
            VtoA(src(i),dest("&",i));
        }
    }

    return;
}



SVM_Anions::SVM_Anions() : SVM_Vector()
{
    setaltx(NULL);

    return;
}

SVM_Anions::SVM_Anions(const SVM_Anions &src) : SVM_Vector()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

SVM_Anions::SVM_Anions(const SVM_Anions &src, const ML_Base *xsrc) : SVM_Vector()
{
    setaltx(xsrc);

    assign(src,1);

    return;
}

SVM_Anions::~SVM_Anions()
{
    return;
}

double SVM_Anions::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

int SVM_Anions::scale(double a)
{
    NiceAssert( a >= 0.0 );
    NiceAssert( a <= 1.0 );

    int res = SVM_Vector::scale(a);

    grabalpha();
    grabdb();

    return res;
}

int SVM_Anions::reset(void)
{
    int res = SVM_Vector::reset();

    setzero(dalpha);
    setzero(db);

    SVM_Generic::basesetAlphaBiasFromAlphaBiasA();

    return res;
}

int SVM_Anions::setAlphaA(const Vector<d_anion> &newAlpha)
{
    Vector<Vector<double> > newAlphaV;

    AtoVvect(newAlpha,newAlphaV);

    int res = SVM_Vector::setAlphaV(newAlphaV);

    dalpha = newAlpha;

    SVM_Generic::basesetAlphaBiasFromAlphaBiasA();

    return res;
}

int SVM_Anions::setBiasA(const d_anion &newBias)
{
    Vector<double> newBiasV;

    AtoV(newBias,newBiasV);

    int res = SVM_Vector::setBiasV(newBiasV);

    db = newBias;

    SVM_Generic::basesetbias(biasA());

    return res;
}

int SVM_Anions::sety(int i, const gentype &z)
{
    NiceAssert( z.isCastableToAnionWithoutLoss() );

    return sety(i,(const d_anion &) z);
}

int SVM_Anions::sety(int i, const d_anion &z)
{
    Vector<double> zV;

    AtoV(z,zV);

    int res = SVM_Vector::sety(i,zV);

    traintarg("&",i) = z;

    return res;
}

int SVM_Anions::sety(const Vector<int> &j, const Vector<gentype> &z)
{
    NiceAssert( z.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= sety(j(i),z(i));
	}
    }

    return res;
}

int SVM_Anions::sety(const Vector<int> &j, const Vector<d_anion> &z)
{
    NiceAssert( z.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= sety(j(i),z(i));
	}
    }

    return res;
}

int SVM_Anions::sety(const Vector<gentype> &z)
{
    NiceAssert( z.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= sety(i,z(i));
	}
    }

    return res;
}

int SVM_Anions::sety(const Vector<d_anion> &z)
{
    NiceAssert( z.size() == N() );

    int i;
    int res = 0;

    if ( N() )
    {
	for ( i = 0 ; i < N() ; i++ )
	{
            res |= sety(i,z(i));
	}
    }

    return res;
}


int SVM_Anions::settspaceDim(int newdim)
{
    NiceAssert( newdim == 1<<ceilintlog2(newdim) );

    int res = SVM_Vector::settspaceDim(newdim);

    grabalpha();
    grabdb();
    grabtraintarg();

    return res;
}

int SVM_Anions::addtspaceFeat(int iii)
{
    (void) iii;

    throw("Function addtspaceFeat has no meaning in anionic context.");

    return 0;
}

int SVM_Anions::removetspaceFeat(int iii)
{
    (void) iii;

    throw("Function removetspaceFeat has no meaning in anionic context.");

    return 0;
}

int SVM_Anions::setorder(int neword)
{
    NiceAssert( neword >= 0 );

    int res = SVM_Vector::settspaceDim(1<<neword);

    grabalpha();
    grabdb();
    grabtraintarg();

    return res;
}

int SVM_Anions::train(int &res, svmvolatile int &killSwitch)
{
    int modmod = SVM_Vector::train(res,killSwitch);

    grabalpha();
    grabdb();

    return modmod;
}

int SVM_Anions::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int res;

    gentype tempresh;
    gentype tempresg;

    res = SVM_Vector::ghTrainingVector(tempresh,tempresh,i,retaltg,pxyprodi);

    VtoA(tempresh.dir_vector(),resh.force_anion());
    VtoA(tempresg.dir_vector(),resg.force_anion());

    return res;
}

int SVM_Anions::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_Anions::addTrainingVector(i,(const d_anion &) z,x,Cweigh,epsweigh,2);
}

int SVM_Anions::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_Anions::qaddTrainingVector(i,(const d_anion &) z,x,Cweigh,epsweigh,2);
}

int SVM_Anions::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Anions::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_Anions::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Anions::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_Anions::addTrainingVector( int i, const d_anion &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    Vector<double> zv;

    AtoV(z,zv);

    return SVM_Vector::addTrainingVector(i,zv,x,Cweigh,epsweigh,d);
}

int SVM_Anions::qaddTrainingVector(int i, const d_anion &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    Vector<double> zv;

    AtoV(z,zv);

    return SVM_Vector::qaddTrainingVector(i,zv,x,Cweigh,epsweigh,d);
}

int SVM_Anions::addTrainingVector( int i, const Vector<d_anion> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );
    NiceAssert( z.size() == d.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Anions::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    return res;
}

int SVM_Anions::qaddTrainingVector(int i, const Vector<d_anion> &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );
    NiceAssert( z.size() == d.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Anions::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    return res;
}

void SVM_Anions::grabalpha(void)
{
    VtoAvect(alphaV(),dalpha);
    SVM_Generic::basesetAlphaBiasFromAlphaBiasA();

    return;
}

void SVM_Anions::grabdb(void)
{
    VtoA(biasV(),db);
    SVM_Generic::basesetbias(biasA());

    return;
}

void SVM_Anions::grabtraintarg(void)
{
    VtoAvect(zV(),traintarg);

    return;
}



int SVM_Anions::setLinearCost(void)
{
    int res = SVM_Vector::setLinearCost();
    grabalpha();
    return res;
}

int SVM_Anions::setQuadraticCost(void)
{
    int res = SVM_Vector::setQuadraticCost();
    grabalpha();
    return res;
}

int SVM_Anions::setC(double xC)
{
    int res = SVM_Vector::setC(xC);
    grabalpha();
    return res;
}

int SVM_Anions::scaleCweight(double scaleFactor)
{
    int res = SVM_Vector::scaleCweight(scaleFactor);
    grabalpha();
    return res;
}

int SVM_Anions::scaleCweightfuzz(double scaleFactor)
{
    int res = SVM_Vector::scaleCweightfuzz(scaleFactor);
    grabalpha();
    return res;
}

int SVM_Anions::setd(int i, int d)
{
    int res = SVM_Vector::setd(i,d);
    grabalpha();
    return res;
}

int SVM_Anions::setCweight(int i, double xCweight)
{
    int res = SVM_Vector::setCweight(i,xCweight);
    grabalpha();
    return res;
}

int SVM_Anions::setCweight(const Vector<int> &i, const Vector<double> &xCweight)
{
    int res = SVM_Vector::setCweight(i,xCweight);
    grabalpha();
    return res;
}

int SVM_Anions::setCweight(const Vector<double> &xCweight)
{
    int res = SVM_Vector::setCweight(xCweight);
    grabalpha();
    return res;
}

int SVM_Anions::setCweightfuzz(int i, double xCweight)
{
    int res = SVM_Vector::setCweightfuzz(i,xCweight);
    grabalpha();
    return res;
}

int SVM_Anions::setCweightfuzz(const Vector<int> &i, const Vector<double> &xCweight)
{
    int res = SVM_Vector::setCweightfuzz(i,xCweight);
    grabalpha();
    return res;
}

int SVM_Anions::setCweightfuzz(const Vector<double> &xCweight)
{
    int res = SVM_Vector::setCweightfuzz(xCweight);
    grabalpha();
    return res;
}








std::ostream &SVM_Anions::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Anionic SVM\n\n";

    repPrint(output,'>',dep) << "Alpha:   " << dalpha    << "\n";
    repPrint(output,'>',dep) << "Bias:    " << db        << "\n\n";
    repPrint(output,'>',dep) << "Targets: " << traintarg << "\n";

    repPrint(output,'>',dep) << "*********************************************************************\n";
    repPrint(output,'>',dep) << "Optimisation state:              ";
    SVM_Vector::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "#####################################################################\n";

    return output;
}

std::istream &SVM_Anions::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> dalpha;
    input >> dummy; input >> db;
    input >> dummy; input >> traintarg;

    input >> dummy; SVM_Vector::inputstream(input);

    return input;
}

















int SVM_Anions::setAlphaV(const Vector<Vector<double> > &newAlpha)
{
    (void) newAlpha;

    throw("Function setAlphaV not available for this SVM type.");

    return 1;
}

int SVM_Anions::setBiasV(const Vector<double> &newBias)
{
    (void) newBias;

    throw("Function setBiasV not available for this SVM type.");

    return 1;
}

int SVM_Anions::sety(int i, const Vector<double> &z)
{
    (void) i;
    (void) z;

    throw("Function sety not available for this SVM type.");

    return 1;
}

int SVM_Anions::sety(const Vector<int> &i, const Vector<Vector<double> > &z)
{
    (void) i;
    (void) z;

    throw("Function sety not available for this SVM type.");

    return 1;
}

int SVM_Anions::sety(const Vector<Vector<double> > &z)
{
    (void) z;

    throw("Function sety not available for this SVM type.");

    return 1;
}

int SVM_Anions::addTrainingVector( int i, const Vector<double> &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    (void) i;
    (void) z;
    (void) x;
    (void) Cweigh;
    (void) epsweigh;
    (void) d;

    throw("Function addTrainingVector not available for this SVM type.");

    return 0;
}

int SVM_Anions::qaddTrainingVector(int i, const Vector<double> &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    (void) i;
    (void) z;
    (void) x;
    (void) Cweigh;
    (void) epsweigh;
    (void) d;

    throw("Function qaddTrainingVector not available for this SVM type.");

    return 0;
}

int SVM_Anions::prealloc(int expectedN)
{
    dalpha.prealloc(expectedN);
    traintarg.prealloc(expectedN);
    SVM_Generic::prealloc(expectedN);

    return 0;
}

int SVM_Anions::preallocsize(void) const
{
    return SVM_Generic::preallocsize();
}

int SVM_Anions::randomise(double sparsity)
{
    int res = SVM_Vector::randomise(sparsity);

    grabalpha();
    grabdb();

    return res;
}

