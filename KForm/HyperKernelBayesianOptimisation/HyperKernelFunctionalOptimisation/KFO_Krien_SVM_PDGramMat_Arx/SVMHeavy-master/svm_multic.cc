
//
// Multiclass classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_multic.h"
#include <iostream>
#include <sstream>
#include <string>


SVM_MultiC::SVM_MultiC() : SVM_Generic()
{
    setaltx(NULL);

    isQatonce = 1;

    return;
}

SVM_MultiC::~SVM_MultiC()
{
    return;
}

SVM_MultiC::SVM_MultiC(const SVM_MultiC &src) : SVM_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

SVM_MultiC::SVM_MultiC(const SVM_MultiC &src, const ML_Base *xsrc) : SVM_Generic()
{
    setaltx(xsrc);

    assign(src,1);

    return;
}

int SVM_MultiC::scale(double a)
{
    int res = 0;

    res |= Qredbin.scale(a);
    res |= Qatonce.scale(a);

    return res;
}

int SVM_MultiC::reset(void)
{
    int res = 0;

    res |= Qredbin.reset();
    res |= Qatonce.reset();

    return res;
}

int SVM_MultiC::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_MultiC::addTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int SVM_MultiC::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_MultiC::qaddTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int SVM_MultiC::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<int> zz(z.size());

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            zz("&",j) = (int) z(j);
        }
    }

    return SVM_MultiC::addTrainingVector(i,zz,x,Cweigh,epsweigh);
}

int SVM_MultiC::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<int> zz(z.size());

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            zz("&",j) = (int) z(j);
        }
    }

    return SVM_MultiC::qaddTrainingVector(i,zz,x,Cweigh,epsweigh);
}

int SVM_MultiC::addTrainingVector( int i, int d, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    int res = 0;

    if ( d )
    {
        res |= Qredbin.addclass(d);
        res |= Qatonce.addclass(d);
    }

    if ( isQatonce ) { res |= Qatonce.addTrainingVector(i,d,x,Cweigh,epsweigh); }
    else             { res |= Qredbin.addTrainingVector(i,d,x,Cweigh,epsweigh); }

    return res;
}

int SVM_MultiC::qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    int res = 0;

    if ( d )
    {
        res |= Qredbin.addclass(d);
        res |= Qatonce.addclass(d);
    }

    if ( isQatonce ) { res |= Qatonce.qaddTrainingVector(i,d,x,Cweigh,epsweigh); }
    else             { res |= Qredbin.qaddTrainingVector(i,d,x,Cweigh,epsweigh); }

    return res;
}

int SVM_MultiC::addTrainingVector( int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int res = 0;

    if ( d.size() )
    {
        int j;

        for ( j = 0 ; j < d.size() ; j++ )
        {
            if ( d(j) )
            {
                res |= Qredbin.addclass(d(j));
                res |= Qatonce.addclass(d(j));
            }
        }
    }

    if ( isQatonce ) { res |= Qatonce.addTrainingVector(i,d,x,Cweigh,epsweigh); }
    else             { res |= Qredbin.addTrainingVector(i,d,x,Cweigh,epsweigh); }

    return res;
}

int SVM_MultiC::qaddTrainingVector(int i, const Vector<int> &d,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int res = 0;

    if ( d.size() )
    {
        int j;

        for ( j = 0 ; j < d.size() ; j++ )
        {
            if ( d(j) )
            {
                res |= Qredbin.addclass(d(j));
                res |= Qatonce.addclass(d(j));
            }
        }
    }

    if ( isQatonce ) { res |= Qatonce.qaddTrainingVector(i,d,x,Cweigh,epsweigh); }
    else             { res |= Qredbin.qaddTrainingVector(i,d,x,Cweigh,epsweigh); }

    return res;
}

int SVM_MultiC::setsubtype(int i)
{
    NiceAssert( ( i >= 0 ) && ( i <= 1 ) );

    int res = 0;

    if ( i != subtype() )
    {
        switch ( i )
        {
            case 0: { res |= setatonce(); break; }
            case 1: { res |= setredbin(); break; }
            default: { throw("Unknown subtype in SVM_MultiC"); break; }
        }
    }

    return 1;
}

int SVM_MultiC::setatonce(void)
{
    int res = 0;

    if ( isredbin() )
    {
	if ( isShrinkTube() )
	{
            res |= Qatonce.setShrinkTube();
	}

        if ( !isVarBias() )
	{
            res |= Qatonce.setPosBias();
	}

        if ( usefuzzt() )
	{
            res |= Qatonce.setusefuzzt(1);
	}

        // This is needed in case getKernel_unsafe() resetKernel() method has been
        // used to adjust the kernel in Qredbin.

        res |= Qatonce.setKernel(Qredbin.getKernel());

        // NB: quad bias force is ignored.
        //     bias force is ignored

        // Transfer problem data

        SparseVector<gentype> xtemp;
        gentype ytemp;
        int dtemp;
        double Cweighttemp;
        double epsweighttemp;

        while ( Qredbin.N() )
        {
            dtemp         = (Qredbin.d())((Qredbin.N())-1);
            epsweighttemp = (Qredbin.epsweight())((Qredbin.N())-1);
            Cweighttemp   = (Qredbin.Cweight())((Qredbin.N())-1);

            res |= Qredbin.removeTrainingVector((Qredbin.N())-1,ytemp,xtemp);
            res |= Qatonce.qaddTrainingVector(Qatonce.N(),dtemp,xtemp,Cweighttemp,epsweighttemp);
        }

        isQatonce = 1;
    }

    return res;
}

int SVM_MultiC::setredbin(void)
{
    int res = 0;

    if ( isQatonce )
    {
        // This is needed in case getKernel_unsafe() resetKernel() method has been
        // used to adjust the kernel in Qredbin.

        res |= Qredbin.setKernel(Qatonce.getKernel());

        // Transfer problem data

        SparseVector<gentype> xtemp;
        gentype ytemp;
        int dtemp;
        double Cweighttemp;
        double epsweighttemp;

        while ( Qatonce.N() )
        {
            dtemp         = (Qatonce.d())((Qatonce.N())-1);
            epsweighttemp = (Qatonce.epsweight())((Qatonce.N())-1);
            Cweighttemp   = (Qatonce.Cweight())((Qatonce.N())-1);

            res |= Qatonce.removeTrainingVector((Qatonce.N())-1,ytemp,xtemp);
            res |= Qredbin.qaddTrainingVector(Qredbin.N(),dtemp,xtemp,Cweighttemp,epsweighttemp);
        }

        isQatonce = 0;
    }

    return res;
}

std::ostream &SVM_MultiC::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Multiclass SVM\n\n";

    repPrint(output,'>',dep) << "SVM mode:   " << isQatonce     << "\n";
    repPrint(output,'>',dep) << "SVM atonce: "; Qatonce.printstream(output,dep+1);
    repPrint(output,'>',dep) << "SVM redbin: "; Qredbin.printstream(output,dep+1);

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &SVM_MultiC::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> isQatonce;
    input >> dummy; Qatonce.inputstream(input);
    input >> dummy; Qredbin.inputstream(input);

    ML_Base::inputstream(input);

    return input;
}

int SVM_MultiC::prealloc(int expectedN)
{
    if ( isQatonce )
    {
        Qatonce.prealloc(expectedN);
    }

    else
    {
        Qredbin.prealloc(expectedN);
    }

    //SVM_Generic::prealloc(expectedN);

    return 0;
}

int SVM_MultiC::preallocsize(void) const
{
    if ( isQatonce )
    {
        return Qatonce.preallocsize();
    }

    return Qredbin.preallocsize();
}
