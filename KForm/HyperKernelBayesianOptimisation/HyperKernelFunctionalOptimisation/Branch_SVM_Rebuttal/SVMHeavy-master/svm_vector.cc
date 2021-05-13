
//
// Vector regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "svm_vector.h"
#include <iostream>
#include <sstream>
#include <string>


SVM_Vector::SVM_Vector() : SVM_Generic()
{
    setaltx(NULL);

    isQatonce = 0;
    isQreal   = 1;

    return;
}

SVM_Vector::SVM_Vector(const SVM_Vector &src) : SVM_Generic()
{
    setaltx(NULL);

    assign(src,0);

    return;
}

SVM_Vector::SVM_Vector(const SVM_Vector &src, const ML_Base *xsrc) : SVM_Generic()
{
    setaltx(xsrc);

    assign(src,1);

    return;
}

SVM_Vector::~SVM_Vector()
{
    return;
}

int SVM_Vector::setsubtype(int i)
{
    NiceAssert( ( i >= 0 ) && ( i <= 3 ) );

    int res = 0;

    if ( i != subtype() )
    {
        switch ( i )
        {
            case 0: { res |= setatonce(); setKreal();   break; }
            case 1: { res |= setredbin(); setKreal();   break; }
            case 2: { res |= setatonce(); setKunreal(); break; }
            case 3: { res |= setredbin(); setKunreal(); break; }
            default: { throw("Unknown subtype in SVM_Vector"); break; }
        }
    }

    return res;
}

int SVM_Vector::scale(double a)
{
    int res = 0;

    res |= Qatonce.scale(a);
    res |= Qredbin.scale(a);
    res |= QMatonce.scale(a);
    res |= QMredbin.scale(a);

    return res;
}

int SVM_Vector::reset(void)
{
    int res = 0;

    res |= Qatonce.reset();
    res |= Qredbin.reset();
    res |= QMatonce.reset();
    res |= QMredbin.reset();

    return res;
}

int SVM_Vector::setatonce(void)
{
    int res = 0;

    if ( SVM_Vector::isKreal() )
    {
        if ( SVM_Vector::isredbin() )
        {
            res |= Qatonce.settspaceDim(Qredbin.tspaceDim());

            // This is needed in case getKernel_unsafe() resetKernel() method has been
            // used to adjust the kernel in Qredbin.

            res |= Qatonce.setKernel(Qredbin.getKernel());

            // Transfer problem data
        
            SparseVector<gentype> xtemp;
            gentype ytemp;
            Vector<double> zVtemp;
            int dtemp;
            double Cweighttemp;
            double epsweighttemp;

            while ( Qredbin.N() )
            {
                dtemp         = (Qredbin.d()        )((Qredbin.N())-1);
                zVtemp        = (Qredbin.zV()       )((Qredbin.N())-1);
                epsweighttemp = (Qredbin.epsweight())((Qredbin.N())-1);
                Cweighttemp   = (Qredbin.Cweight()  )((Qredbin.N())-1);

                res |= Qredbin.removeTrainingVector((Qredbin.N())-1,ytemp,xtemp);
                res |= Qatonce.qaddTrainingVector(Qatonce.N(),zVtemp,xtemp,Cweighttemp,epsweighttemp,dtemp);
            }

            isQatonce = 1;
        }
    }

    else
    {
        if ( SVM_Vector::isredbin() )
        {
            res |= QMatonce.settspaceDim(QMredbin.tspaceDim());

            // This is needed in case getKernel_unsafe() resetKernel() method has been
            // used to adjust the kernel in Qredbin.

            res |= QMatonce.setKernel(QMredbin.getKernel());

            // Transfer problem data
        
            SparseVector<gentype> xtemp;
            gentype ytemp;
            Vector<double> zVtemp;
            int dtemp;
            double Cweighttemp;
            double epsweighttemp;

            while ( QMredbin.N() )
            {
                dtemp         = (QMredbin.d()        )((QMredbin.N())-1);
                zVtemp        = (QMredbin.zV()       )((QMredbin.N())-1);
                epsweighttemp = (QMredbin.epsweight())((QMredbin.N())-1);
                Cweighttemp   = (QMredbin.Cweight()  )((QMredbin.N())-1);

                res |= QMredbin.removeTrainingVector((QMredbin.N())-1,ytemp,xtemp);
                res |= QMatonce.qaddTrainingVector(QMatonce.N(),zVtemp,xtemp,Cweighttemp,epsweighttemp,dtemp);
            }

            isQatonce = 1;
        }
    }

    return res;
}

int SVM_Vector::setredbin(void)
{
    int res = 0;

    if ( SVM_Vector::isKreal() )
    {
        if ( SVM_Vector::isatonce() )
        {
            res |= Qredbin.settspaceDim(Qatonce.tspaceDim());

            // This is needed in case getKernel_unsafe() resetKernel() method has been
            // used to adjust the kernel in Qredbin.

            res |= Qredbin.setKernel(Qatonce.getKernel());

            // Transfer problem data
        
            SparseVector<gentype> xtemp;
            gentype ytemp;
            Vector<double> zVtemp;
            int dtemp;
            double Cweighttemp;
            double epsweighttemp;

            while ( Qatonce.N() )
            {
                dtemp         = (Qatonce.d()        )((Qatonce.N())-1);
                zVtemp        = (Qatonce.zV()       )((Qatonce.N())-1);
                epsweighttemp = (Qatonce.epsweight())((Qatonce.N())-1);
                Cweighttemp   = (Qatonce.Cweight()  )((Qatonce.N())-1);

                res |= Qatonce.removeTrainingVector((Qatonce.N())-1,ytemp,xtemp);
                res |= Qredbin.qaddTrainingVector(Qredbin.N(),zVtemp,xtemp,Cweighttemp,epsweighttemp,dtemp);
            }

            isQatonce = 0;
        }
    }

    else
    {
        if ( SVM_Vector::isatonce() )
        {
            res |= QMredbin.settspaceDim(QMatonce.tspaceDim());

            // This is needed in case getKernel_unsafe() resetKernel() method has been
            // used to adjust the kernel in Qredbin.

            res |= QMredbin.setKernel(QMatonce.getKernel());

            // Transfer problem data
        
            SparseVector<gentype> xtemp;
            gentype ytemp;
            Vector<double> zVtemp;
            int dtemp;
            double Cweighttemp;
            double epsweighttemp;

            while ( QMatonce.N() )
            {
                dtemp         = (QMatonce.d()        )((QMatonce.N())-1);
                zVtemp        = (QMatonce.zV()       )((QMatonce.N())-1);
                epsweighttemp = (QMatonce.epsweight())((QMatonce.N())-1);
                Cweighttemp   = (QMatonce.Cweight()  )((QMatonce.N())-1);

                res |= QMatonce.removeTrainingVector((QMatonce.N())-1,ytemp,xtemp);
                res |= QMredbin.qaddTrainingVector(QMredbin.N(),zVtemp,xtemp,Cweighttemp,epsweighttemp,dtemp);
            }

            isQatonce = 0;
        }
    }

    return res;
}

int SVM_Vector::setKreal(void)
{
    int res = 0;

    if ( SVM_Vector::isatonce() )
    {
        if ( SVM_Vector::isKunreal() )
        {
            res |= Qatonce.settspaceDim(QMatonce.tspaceDim());

            // This is needed in case getKernel_unsafe() resetKernel() method has been
            // used to adjust the kernel in Qredbin.

            res |= Qatonce.setKernel(QMatonce.getKernel());

            // Transfer problem data
        
            SparseVector<gentype> xtemp;
            gentype ytemp;
            Vector<double> zVtemp;
            int dtemp;
            double Cweighttemp;
            double epsweighttemp;

            while ( QMatonce.N() )
            {
                dtemp         = (QMatonce.d()        )((QMatonce.N())-1);
                zVtemp        = (QMatonce.zV()       )((QMatonce.N())-1);
                epsweighttemp = (QMatonce.epsweight())((QMatonce.N())-1);
                Cweighttemp   = (QMatonce.Cweight()  )((QMatonce.N())-1);

                res |= QMatonce.removeTrainingVector((QMatonce.N())-1,ytemp,xtemp);
                res |= Qatonce.qaddTrainingVector(Qatonce.N(),zVtemp,xtemp,Cweighttemp,epsweighttemp,dtemp);
            }

            isQreal = 1;
        }
    }

    else
    {
        if ( SVM_Vector::isKunreal() )
        {
            res |= Qredbin.settspaceDim(QMredbin.tspaceDim());

            // This is needed in case getKernel_unsafe() resetKernel() method has been
            // used to adjust the kernel in Qredbin.

            res |= Qredbin.setKernel(QMredbin.getKernel());

            // Transfer problem data
        
            SparseVector<gentype> xtemp;
            gentype ytemp;
            Vector<double> zVtemp;
            int dtemp;
            double Cweighttemp;
            double epsweighttemp;

            while ( QMredbin.N() )
            {
                dtemp         = (QMredbin.d()        )((QMredbin.N())-1);
                zVtemp        = (QMredbin.zV()       )((QMredbin.N())-1);
                epsweighttemp = (QMredbin.epsweight())((QMredbin.N())-1);
                Cweighttemp   = (QMredbin.Cweight()  )((QMredbin.N())-1);

                res |= QMredbin.removeTrainingVector((QMredbin.N())-1,ytemp,xtemp);
                res |= Qredbin.qaddTrainingVector(Qredbin.N(),zVtemp,xtemp,Cweighttemp,epsweighttemp,dtemp);
            }

            isQreal = 1;
        }
    }

    return res;
}

int SVM_Vector::setKunreal(void)
{
    int res = 0;

    if ( SVM_Vector::isatonce() )
    {
        if ( SVM_Vector::isKreal() )
        {
            res |= QMatonce.settspaceDim(Qatonce.tspaceDim());

            // This is needed in case getKernel_unsafe() resetKernel() method has been
            // used to adjust the kernel in Qredbin.

            res |= QMatonce.setKernel(Qatonce.getKernel());

            // Transfer problem data
        
            SparseVector<gentype> xtemp;
            gentype ytemp;
            Vector<double> zVtemp;
            int dtemp;
            double Cweighttemp;
            double epsweighttemp;

            while ( Qatonce.N() )
            {
                dtemp         = (Qatonce.d()        )((Qatonce.N())-1);
                zVtemp        = (Qatonce.zV()       )((Qatonce.N())-1);
                epsweighttemp = (Qatonce.epsweight())((Qatonce.N())-1);
                Cweighttemp   = (Qatonce.Cweight()  )((Qatonce.N())-1);

                res |= Qatonce.removeTrainingVector((Qatonce.N())-1,ytemp,xtemp);
                res |= QMatonce.qaddTrainingVector(QMatonce.N(),zVtemp,xtemp,Cweighttemp,epsweighttemp,dtemp);
            }

            isQreal = 0;
        }
    }

    else
    {
        if ( SVM_Vector::isKreal() )
        {
            res |= QMredbin.settspaceDim(Qredbin.tspaceDim());

            // This is needed in case getKernel_unsafe() resetKernel() method has been
            // used to adjust the kernel in Qredbin.

            res |= QMredbin.setKernel(Qredbin.getKernel());

            // Transfer problem data
        
            SparseVector<gentype> xtemp;
            gentype ytemp;
            Vector<double> zVtemp;
            int dtemp;
            double Cweighttemp;
            double epsweighttemp;

            while ( Qredbin.N() )
            {
                dtemp         = (Qredbin.d()        )((Qredbin.N())-1);
                zVtemp        = (Qredbin.zV()       )((Qredbin.N())-1);
                epsweighttemp = (Qredbin.epsweight())((Qredbin.N())-1);
                Cweighttemp   = (Qredbin.Cweight()  )((Qredbin.N())-1);

                res |= Qredbin.removeTrainingVector((Qredbin.N())-1,ytemp,xtemp);
                res |= QMredbin.qaddTrainingVector(QMredbin.N(),zVtemp,xtemp,Cweighttemp,epsweighttemp,dtemp);
            }

            isQreal = 0;
        }
    }

    return res;
}


int SVM_Vector::findID(int ref) const
{
    NiceAssert( ref == 2 );

    (void) ref;

    return 2;
}

void addclass(int label, int epszero)
{
    (void) label;
    (void) epszero;
    throw("Function addclass not available for SVM_Vector");
    return;
}

int SVM_Vector::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    Vector<double> zd;
    Vector<gentype> zz((const Vector<gentype> &) z);

    zd.resize(zz.size());

    if ( zz.size() )
    {
        int i;

        for ( i = 0 ; i < zz.size() ; i++ )
        {
            zd("&",i) = (double) zz(i);
        }
    }

    return SVM_Vector::addTrainingVector(i,zd,x,Cweigh,epsweigh);
}

int SVM_Vector::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    Vector<double> zd(z.size());

    if ( z.size() )
    {
        int i;

        const Vector<gentype> &zz = (const Vector<gentype> &) z;

        for ( i = 0 ; i < z.size() ; i++ )
        {
            zd("&",i) = (double) zz(i);
        }
    }

    return SVM_Vector::qaddTrainingVector(i,zd,x,Cweigh,epsweigh);
}

int SVM_Vector::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == xx.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Vector::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_Vector::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == xx.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Vector::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_Vector::addTrainingVector( int i, const Vector<double> &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    int res = 0;

         if ( isatonce() && isKreal()   ) { res |=  Qatonce.addTrainingVector( i,z,x,Cweigh,epsweigh,d); }
    else if ( isatonce() && isKunreal() ) { res |= QMatonce.addTrainingVector( i,z,x,Cweigh,epsweigh,d); }
    else if ( isredbin() && isKreal()   ) { res |=  Qredbin.addTrainingVector( i,z,x,Cweigh,epsweigh,d); }
    else                                  { res |= QMredbin.addTrainingVector( i,z,x,Cweigh,epsweigh,d); }

    return res;
}

int SVM_Vector::qaddTrainingVector(int i, const Vector<double> &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    int res = 0;

         if ( isatonce() && isKreal()   ) { res |=  Qatonce.qaddTrainingVector(i,z,x,Cweigh,epsweigh,d); }
    else if ( isatonce() && isKunreal() ) { res |= QMatonce.qaddTrainingVector(i,z,x,Cweigh,epsweigh,d); }
    else if ( isredbin() && isKreal()   ) { res |=  Qredbin.qaddTrainingVector(i,z,x,Cweigh,epsweigh,d); }
    else                                  { res |= QMredbin.qaddTrainingVector(i,z,x,Cweigh,epsweigh,d); }

    return res;
}

int SVM_Vector::addTrainingVector( int i, const Vector<Vector<double> > &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    int res = 0;

         if ( isatonce() && isKreal()   ) { res |=  Qatonce.addTrainingVector( i,z,x,Cweigh,epsweigh,d); }
    else if ( isatonce() && isKunreal() ) { res |= QMatonce.addTrainingVector( i,z,x,Cweigh,epsweigh,d); }
    else if ( isredbin() && isKreal()   ) { res |=  Qredbin.addTrainingVector( i,z,x,Cweigh,epsweigh,d); }
    else                                  { res |= QMredbin.addTrainingVector( i,z,x,Cweigh,epsweigh,d); }

    return res;
}

int SVM_Vector::qaddTrainingVector(int i, const Vector<Vector<double> > &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    int res = 0;

         if ( isatonce() && isKreal()   ) { res |=  Qatonce.qaddTrainingVector(i,z,x,Cweigh,epsweigh,d); }
    else if ( isatonce() && isKunreal() ) { res |= QMatonce.qaddTrainingVector(i,z,x,Cweigh,epsweigh,d); }
    else if ( isredbin() && isKreal()   ) { res |=  Qredbin.qaddTrainingVector(i,z,x,Cweigh,epsweigh,d); }
    else                                  { res |= QMredbin.qaddTrainingVector(i,z,x,Cweigh,epsweigh,d); }

    return res;
}



std::ostream &SVM_Vector::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector SVM\n\n";

    repPrint(output,'>',dep) << "SVM mode:    " << isQatonce << "\n";
    repPrint(output,'>',dep) << "SVM K mode:  " << isQreal   << "\n";
    repPrint(output,'>',dep) << "SVM atonce:  " << Qatonce   << "\n";
    repPrint(output,'>',dep) << "SVM redbin:  " << Qredbin   << "\n";
    repPrint(output,'>',dep) << "SVM Matonce: "; QMatonce.printstream(output,dep+1);
    repPrint(output,'>',dep) << "SVM Mredbin: "; QMredbin.printstream(output,dep+1);

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &SVM_Vector::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> isQatonce;
    input >> dummy; input >> isQreal;
    input >> dummy; input >> Qatonce;
    input >> dummy; input >> Qredbin;
    input >> dummy; QMatonce.inputstream(input);
    input >> dummy; QMredbin.inputstream(input);

    ML_Base::inputstream(input);

    return input;
}

int SVM_Vector::prealloc(int expectedN)
{
    if ( isQatonce && isQreal )
    {
        Qatonce.prealloc(expectedN);
    }

    else if ( isQatonce && !isQreal )
    {
        QMatonce.prealloc(expectedN);
    }

    else if ( !isQatonce && isQreal )
    {
        Qredbin.prealloc(expectedN);
    }

    else
    {
        QMredbin.prealloc(expectedN);
    }

    //SVM_Generic::prealloc(expectedN);

    return 0;
}

int SVM_Vector::preallocsize(void) const
{
    if ( isQatonce && isQreal )
    {
        return Qatonce.preallocsize();
    }

    else if ( isQatonce && !isQreal )
    {
        return QMatonce.preallocsize();
    }

    else if ( !isQatonce && isQreal )
    {
        return Qredbin.preallocsize();
    }

    return QMredbin.preallocsize();
}
