
//
// Super-Sparse SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "ssv_generic.h"
#include <iostream>
#include <sstream>
#include <string>

SSV_Generic::SSV_Generic() : SVM_Scalar()
{
    setaltx(NULL);

    gentype ydummy;
    SparseVector<gentype> xdummy;

    setQuadraticCost();

    zM.resize(1,1).zero();
    zn.resize(1).zero();

    zmodel.setC(1.0);
    zmodel.seteps(0.0);
    zmodel.setFixedBias();
    zmodel.setQuadraticCost();
    zmodel.addTrainingVector(0,ydummy,xdummy);
    zmodel.setGp(&zM,&zM,0);
    zmodel.sety(zn);

    inbypass = 0;

    return;
}

SSV_Generic::SSV_Generic(const SSV_Generic &src) : SVM_Scalar()
{
    setaltx(NULL);

    gentype ydummy;
    SparseVector<gentype> xdummy;

    setQuadraticCost();

    zM.resize(1,1).zero();
    zn.resize(1).zero();

    zmodel.setC(1.0);
    zmodel.seteps(0.0);
    zmodel.setFixedBias();
    zmodel.setQuadraticCost();
    zmodel.addTrainingVector(0,ydummy,xdummy);
    zmodel.setGp(&zM,&zM,0);
    zmodel.sety(zn);

    inbypass = 0;

    assign(src,0);

    return;
}

SSV_Generic::SSV_Generic(const SSV_Generic &src, const ML_Base *srcx) : SVM_Scalar()
{
    setaltx(srcx);

    gentype ydummy;
    SparseVector<gentype> xdummy;

    setQuadraticCost();

    zM.resize(1,1).zero();
    zn.resize(1).zero();

    zmodel.setC(1.0);
    zmodel.seteps(0.0);
    zmodel.setFixedBias();
    zmodel.setQuadraticCost();
    zmodel.addTrainingVector(0,ydummy,xdummy);
    zmodel.setGp(&zM,&zM,0);
    zmodel.sety(zn);

    inbypass = 0;

    assign(src,0);

    return;
}


int SSV_Generic::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    int res = 0;

    if ( !inbypass )
    {
        res |= deactivate(onlyChangeRowI);
        res |= SVM_Scalar::setKernel(xkernel,modind,onlyChangeRowI);
        res |= activate(onlyChangeRowI);
    }

    else
    {
        inbypass++;
        res |= SVM_Scalar::setKernel(xkernel,modind,onlyChangeRowI);
        inbypass--;
    }

    return res;
}

int SSV_Generic::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    int res = 0;

    zxstate.add(i);
    zxstate("[]",i) = 0;

    zy.add(i);
    zy("[]",i) = y;

    res |= SVM_Scalar::addTrainingVector(i,y,x,Cweigh,epsweigh);
    res |= activate(i,activateonadd());

    return res;
}

int SSV_Generic::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    int res = 0;

    zxstate.add(i);
    zxstate("[]",i) = 0;

    zy.add(i);
    zy("[]",i) = y;

    res |= SVM_Scalar::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    res |= activate(i,activateonadd());

    return res;
}

int SSV_Generic::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( i >= zeroint() );
    NiceAssert( i <= N() );

    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    zxstate.addpad(i,y.size());
    zxstate("[]",i,1,i+y.size()-1) = zeroint();

    zy.addpad(i,y.size());
    zy("[]",i,1,i+y.size()-1) = y;

    res |= SVM_Scalar::addTrainingVector(i,y,x,Cweigh,epsweigh);
    res |= activate(i,activateonadd(),y.size());

    return res;
}

int SSV_Generic::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( i >= zeroint() );
    NiceAssert( i <= N() );

    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = 0;

    zxstate.addpad(i,y.size());
    zxstate("[]",i,1,i+y.size()-1) = zeroint();

    zy.addpad(i,y.size());
    zy("[]",i,1,i+y.size()-1) = y;

    res |= SVM_Scalar::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    res |= activate(i,activateonadd(),y.size());

    return res;
}

int SSV_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    gentype ydummy;

    y = zy(i);

    int res = 0;

    zxstate.remove(i);
    zy.remove(i);

    res |= deactivate(i);
    res |= SVM_Scalar::removeTrainingVector(i,ydummy,x);

    return res;
}

int SSV_Generic::setx(int i, const SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    
    int res = 0;
    int isact = zxstate(i);

    res |= deactivate(i);
    res |= SVM_Scalar::setx(i,x);
    res |= activate(i,isact);

    return res;
}

int SSV_Generic::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x)
{
    NiceAssert( i >= zeroint() );
    NiceAssert( i < N() );

    NiceAssert( i.size() == x.size() );

    int res = 0;
    Vector<int> isact(zxstate(i));

    res |= deactivate(i);
    res |= SVM_Scalar::setx(i,x);
    res |= activate(i,isact);

    return res;
}

int SSV_Generic::setd(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;
    int isact = zxstate(i);

    res |= ( isact && !d ) ? deactivate(i) : 0; // short-circuit
    res |= SVM_Scalar::setd(i,d);
    res |= activate(i,isact|(d*activateonadd()));

    return res;
}

int SSV_Generic::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    NiceAssert( i.size() == d.size() );

    int res = 0;
    Vector<int> isact(zxstate(i));

    res |= deactivate(i); // lazy
    res |= SVM_Scalar::setd(i,d);
    res |= activate(i,isact);

    return res;
}

int SSV_Generic::setCweight(int i, double nv)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    
    int res = 0;
    int isact = zxstate(i);

    res |= deactivate(i);
    res |= SVM_Scalar::setCweight(i,nv);
    res |= activate(i,isact);

    return res;
}

int SSV_Generic::setCweight(const Vector<int> &i, const Vector<double> &nv)
{
    NiceAssert( i >= zeroint() );
    NiceAssert( i < N() );

    NiceAssert( i.size() == nv.size() );

    int res = 0;
    Vector<int> isact(zxstate(i));

    res |= deactivate(i);
    res |= SVM_Scalar::setCweight(i,nv);
    res |= activate(i,isact);

    return res;
}

int SSV_Generic::setsigmaweight(int i, double nv)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    
    int res = 0;
    int isact = zxstate(i);

    res |= deactivate(i);
    res |= SVM_Scalar::setsigmaweight(i,nv);
    res |= activate(i,isact);

    return res;
}

int SSV_Generic::setsigmaweight(const Vector<int> &i, const Vector<double> &nv)
{
    NiceAssert( i >= zeroint() );
    NiceAssert( i < N() );

    NiceAssert( i.size() == nv.size() );

    int res = 0;
    Vector<int> isact(zxstate);

    res |= deactivate(i);
    res |= SVM_Scalar::setsigmaweight(i,nv);
    res |= activate(i,isact);

    return res;
}

int SSV_Generic::setC(double xC)
{
    int res = 0;

    res |= SVM_Scalar::setC(xC);
    res |= updatez();

    return res;
}

int SSV_Generic::setsigma(double xC)
{
    int res = 0;

    res |= SVM_Scalar::setsigma(xC);
    res |= updatez();

    return res;
}

int SSV_Generic::setCclass(int d, double xC)
{
    int res = 0;

    res |= SVM_Scalar::setCclass(d,xC);
    res |= updatez();

    return res;
}

int SSV_Generic::scale(double a)
{
    int res = 0;

    res |= SVM_Scalar::scale(a);
    res |= updatez();

    return res;
}

int SSV_Generic::reset(void)
{
    int res = 0;

    res |= SVM_Scalar::reset();
    res |= updatez();

    setzero(zbeta);
    setzero(zb);

    return res;
}


































int SSV_Generic::setz(int j, const SparseVector<gentype> &newz)
{
    NiceAssert( j >= zeroint() );
    NiceAssert( j < Nzs() );

    int res = 0;

    inbypass = 1;

    res |= SVM_Scalar::setx(j+N(),newz);
    res |= updatez(j);

    inbypass = 0;

    return res;
}

int SSV_Generic::setz(const Vector<int> &j, const Vector<SparseVector<gentype> > &newz)
{
    NiceAssert( j >= zeroint() );
    NiceAssert( j < Nzs() );

    NiceAssert( j.size() == newz.size() );

    int res = 0;

    inbypass = 1;

    res |= SVM_Scalar::setx(j+N(),newz);
    res |= updatez(j);

    inbypass = 0;

    return res;
}

int SSV_Generic::setz(const Vector<SparseVector<gentype> > &newz)
{
    NiceAssert( Nzs() == newz.size() );

    int res = 0;

    inbypass = 1;

    res |= SVM_Scalar::setx(cntintvec(Nzs())+N(),newz);
    res |= updatez();

    inbypass = 0;

    return res;
}

int SSV_Generic::setNzs(int nv)
{
    int res = 0;
    int i,j,k;

    while ( nv < Nzs() )
    {
        res = 1;

        i = Nzs()-1;
        j = N();

        SVM_Scalar::removeTrainingVector(i+j);
 
        zbeta.remove(i);
        zn.remove(i);

        updateNzs(i,i,i-1);

        // Need to be done post update!
        zM.removeRowCol(i);
    }

    Vector<int> newind( ( nv > Nzs() ) ? nv-Nzs() : 0 );

    k = 0;

    while ( nv > Nzs() )
    {
        gentype dummytarg(targType());

        res = 2;

        i = Nzs();
        j = N();
 
        zbeta.add(i);
        zM.addRowCol(i);
        zn.add(i);

        SparseVector<gentype> newz;

        SVM_Scalar::qaddTrainingVector(i+j,dummytarg,newz);

        updatez(i,0); // don't allow call to updateM here, just calculate M
        updateNzs(i,i,i+1);

        newind("[]",k++) = i;
    }

    if ( res == 2 )
    {
        // Propogate M changes from size increase

        updateM(newind);
        updaten(newind);
    }

    return res;
}

int SSV_Generic::activate(int i, int isact)
{
    int res = 0;

    if ( isact )
    {
        res |= activate(i);
    }

    return res;
}

int SSV_Generic::activate(int i, int isact, int num)
{
    NiceAssert( num >= 0 );

    int res = 0;

    if ( num && isact )
    {
        int j;

        for ( j = 0 ; j < num ; j++ )
        {
            res |= activate(i+j);
        }
    }

    return res;
}

int SSV_Generic::activate(const Vector<int> &i)
{
    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= activate(i(j));
        }
    }

    return res;
}

int SSV_Generic::activate(const Vector<int> &i, int isact)
{
    int res = 0;

    if ( i.size() && isact )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= activate(i(j));
        }
    }

    return res;
}

int SSV_Generic::activate(const Vector<int> &i, const Vector<int> &isact)
{
    NiceAssert( i.size() == isact.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            if ( isact(j) )
            {
                res |= activate(i(j));
            }
        }
    }

    return res;
}

int SSV_Generic::deactivate(const Vector<int> &i)
{
    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= deactivate(i(j));
        }
    }

    return res;
}

int SSV_Generic::setbeta(const Vector<gentype> &newBeta) 
{ 
    zbeta = newBeta; 

    if ( isUnderlyingScalar() )
    {
        Vector<gentype> newalpha(N());

        newalpha.zero();
        newalpha.append(N(),newBeta);

        SVM_Scalar::setAlpha(newalpha);
    }

    return 1; 
}

int SSV_Generic::setb(const gentype &newb) 
{ 
    zb = newb; 

    if ( isUnderlyingScalar() )
    {
        SVM_Scalar::setBias(newb);
    }

    return 1; 
}













































int SSV_Generic::sety(int i, const gentype &y) 
{ 
    NiceAssert( i >= 0 );
    NiceAssert( i < N() ); 

    int res = 0;

    double Cval = 0.0;

    if ( zxstate(i) )
    {
        Cval = calcCval(i);

        if ( Nzs() )
        {
            int j;

            for ( j = 0 ; j < Nzs() ; j++ )
            {
                zn("[]",j) -= Gp()(j+N(),i)*Cval*zy(i);
            }
        }

        zn("[]",Nzs()) -= zy(i)*Cval;
    }

    zy("[]",i) = y;

    if ( zxstate(i) )
    {
        if ( Nzs() )
        {
            int j;

            for ( j = 0 ; j < Nzs() ; j++ )
            {
                zn("[]",j) += Gp()(j+N(),i)*Cval*zy(i);
            }
        }

        zn("[]",Nzs()) += zy(i)*Cval;

        res |= updaten(-1);
    }

    res |= SVM_Scalar::sety(i,y); 

    return res;
}

int SSV_Generic::sety(const Vector<int> &i, const Vector<gentype> &y)
{ 
    NiceAssert( i >= zeroint() );
    NiceAssert( i < N() ); 

    NiceAssert( i.size() == y.size() );

    int res = 0;

    Vector<double> Cval(i.size());

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            Cval("[]",ii) = calcCval(i(ii));
        }
    }

    zy("[]",i) = y;
    zn.zero();

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            if ( zxstate(i(ii)) )
            {
                int j;

                for ( j = 0 ; j < Nzs() ; j++ )
                {
                    zn("[]",j) += Gp()(j+N(),i(ii))*Cval(ii)*zy(i(ii));
                }

                zn("[]",Nzs()) += Cval(ii)*zy(i(ii));
            }
        }

        res |= updaten(-1);
    }

    res |= SVM_Scalar::sety(i,y); 

    return res;
}

int SSV_Generic::activate(int jj)
{
    if ( jj == -1 )
    {
        return activate(cntintvec(N()));
    }

    NiceAssert( jj >= 0 );
    NiceAssert( jj < N() );

    int res = 0;

    if ( !zxstate(jj) )
    {
        int i,j;

        zxstate("[]",jj) = 1;

        i = 0;

        for ( j = 0 ; j < Nzs() ; j++ )
        {
            if ( zxact(j) > jj )
            {
                i = jj;
                break;
            }
        }

        zxact.add(i);
        zxact("[]",i) = jj;

        // see updatez functions

        if ( Nzs() )
        {
            double Cval = calcCval(jj);

            for ( i = 0 ; i < Nzs() ; i++ )
            {
                for ( j = 0 ; j < Nzs() ; j++ )
                {
                    zM("[]",i,j) += Gp()(i+N(),jj)*Cval*Gp()(j+N(),jj);
                }

                zM("[]",i,Nzs()) += Gp()(i+N(),jj)*Cval;
                zM("[]",Nzs(),i) =  zM(i,Nzs());

                zn("[]",j) += Gp()(i+N(),jj)*Cval*zy(jj);
            }

            zM("[]",Nzs(),Nzs()) += Cval;

            zn("[]",Nzs()) += Cval*zy(jj);

            res |= 1;
            res |= updateM(-1);
            res |= updaten(-1);
        }
    }

    return res;
}

int SSV_Generic::deactivate(int jj)
{
    if ( jj == -1 )
    {
        return deactivate(cntintvec(N()));
    }

    NiceAssert( jj >= 0 );
    NiceAssert( jj < N() );

    int res = 0;

    if ( zxstate(jj) )
    {
        int i,j;

        zxstate("[]",jj) = 0;

        i = 0;

        for ( j = 0 ; j < Nzs() ; j++ )
        {
            if ( zxact(j) == jj )
            {
                zxact.remove(j);
                break;
            }

            else if ( zxact(j) > jj )
            {
                break;
            }
        }

        // see updatez functions

        if ( Nzs() )
        {
            double Cval = calcCval(jj);

            for ( i = 0 ; i < Nzs() ; i++ )
            {
                for ( j = 0 ; j < Nzs() ; j++ )
                {
                    zM("[]",i,j) -= Gp()(i+N(),jj)*Cval*Gp()(j+N(),jj);
                }

                zM("[]",i,Nzs()) -= Gp()(i+N(),jj)*Cval;
                zM("[]",Nzs(),i) =  zM(i,Nzs());

                zn("[]",j) -= Gp()(i+N(),jj)*Cval*zy(jj);
            }

            zM("[]",Nzs(),Nzs()) -= Cval;

            zn("[]",Nzs()) -= Cval*zy(jj);

            res |= 1;
            res |= updateM(-1);
            res |= updaten(-1);
        }
    }

    return res;
}

int SSV_Generic::updatez(void)
{
    // [ Mp   Mpn ]  = [ S_{zx}.U.S_{xz}    S_{zx}.U.1 ]
    // [ Mpn' Mn  ]    [    1'.U.S_{xz}       1'.U.1   ]
    //
    // By design, Gp() will return:
    //
    // [ S_{xx} S_{xz} ]
    // [ S_{zx} S_{zz} ]
    //
    // (all vectors held in SVM_Scalar parent cache), so we rely on
    // Gp() to calculate S_{xz}, S_{zx}.  However, because the cache
    // is stored row-wise, and because we want to minimise the total
    // memory used for this, we always access S_{zx} and use inferred
    // symmetry to derive S_{xz} from this (that is, S_{xz} = S_{zx}').

    int res = 0;
    int i;

    zM.zero();
    zn.zero();

    if ( Nzs() )
    {
        if ( zxact.size() )
        {
            int j,k;

            Vector<double> Cval(N());

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    Cval("[]",zxact(k)) = calcCval(zxact(k));
                }
            }

            for ( j = 0 ; j < Nzs() ; j++ )
            {
                for ( i = 0 ; i <= j ; i++ )
                {
                    {
                        for ( k = 0 ; k < zxact.size() ; k++ )
                        {
                            // NB: always access Gp() row-wise to minimise cache use

                            zM("[]",i,j) += Gp()(i+N(),zxact(k))*Cval(zxact(k))*Gp()(j+N(),zxact(k));
                        }
                    }

                    if ( i == j )
                    {
                        zM("[]",i,j) += 1.0;
                    }

                    else
                    {
                        zM("[]",j,i) = zM(i,j);
                    }
                }

                {
                    for ( k = 0 ; k < zxact.size() ; k++ )
                    {
                        // NB: always access Gp() row-wise to minimise cache use

                        zM("[]",j,Nzs()) += Gp()(j+N(),zxact(k))*Cval(zxact(k));

                        zn("[]",j) += Gp()(j+N(),zxact(k))*Cval(zxact(k))*zy(zxact(k));
                    }

                    zM("[]",Nzs(),j) = zM(j,Nzs());
                }
            }

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    // NB: always access Gp() row-wise to minimise cache use

                    zM("[]",Nzs(),Nzs()) += Cval(zxact(k));
                }
            }

            res |= 1;
            res |= updateM(-1);
            res |= updaten(-1);
        }
    }

    return res;
}

int SSV_Generic::updatez(const Vector<int> &jj)
{
    NiceAssert( jj >= zeroint() );
    NiceAssert( jj <  Nzs() );

    // Like above, but with extra indexing.

    int res = 0;
    int i,j;

    for ( j = 0 ; j < jj.size() ; j++ )
    {
        for ( i = 0 ; i < Nzs() ; i++ )
        {
            zM("[]",jj(j),i) = 0.0;
            zM("[]",i,jj(j)) = 0.0;
        }

        zM("[]",jj(j),Nzs()) = 0.0;
        zM("[]",Nzs(),jj(j)) = 0.0;

        zn("[]",jj(j)).zero();
    }

    zM("[]",Nzs(),Nzs()) = 0.0;

    if ( Nzs() && jj.size() )
    {
        if ( zxact.size() )
        {
            int k;

            Vector<double> Cval(N());

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    Cval("[]",zxact(k)) = calcCval(zxact(k));
                }
            }

            for ( j = 0 ; j < jj.size() ; j++ )
            {
                for ( i = 0 ; i < Nzs() ; i++ )
                {
                    {
                        for ( k = 0 ; k < zxact.size() ; k++ )
                        {
                            // NB: always access Gp() row-wise to minimise cache use

                            zM("[]",i,jj(j)) += Gp()(i+N(),zxact(k))*Cval(zxact(k))*Gp()(jj(j)+N(),zxact(k));
                        }
                    }

                    if ( i == jj(j) )
                    {
                        zM("[]",i,jj(j)) += 1.0;
                    }

                    else
                    {
                        zM("[]",jj(j),i) = zM(i,jj(j));
                    }
                }

                {
                    for ( k = 0 ; k < zxact.size() ; k++ )
                    {
                        // NB: always access Gp() row-wise to minimise cache use

                        zM("[]",jj(j),Nzs()) += Gp()(jj(j)+N(),zxact(k))*Cval(zxact(k));

                        zn("[]",jj(j)) += Gp()(jj(j)+N(),zxact(k))*Cval(zxact(k))*zy(zxact(k));
                    }

                    zM("[]",Nzs(),jj(j)) = zM(jj(j),Nzs());
                }
            }

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    // NB: always access Gp() row-wise to minimise cache use

                    zM("[]",Nzs(),Nzs()) += Cval(zxact(k));
                }
            }

            res |= 1;
            res |= updateM(jj);
            res |= updaten(jj);
        }
    }

    return res;
}

int SSV_Generic::updatez(int jj, int alsoupdateM)
{
    NiceAssert( jj >= 0     );
    NiceAssert( jj <  Nzs() );

    // Like above, no indexing

    int res = 0;
    int i;

    {
        for ( i = 0 ; i < Nzs() ; i++ )
        {
            zM("[]",jj,i) = 0.0;
            zM("[]",i,jj) = 0.0;
        }

        zM("[]",jj,Nzs()) = 0.0;
        zM("[]",Nzs(),jj) = 0.0;

        zn("[]",jj).zero();
    }

    zM("[]",Nzs(),Nzs()) = 0.0;

    if ( Nzs() )
    {
        if ( zxact.size() )
        {
            int k;

            Vector<double> Cval(N());

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    Cval("[]",zxact(k)) = calcCval(zxact(k));
                }
            }

            {
                for ( i = 0 ; i < Nzs() ; i++ )
                {
                    {
                        for ( k = 0 ; k < zxact.size() ; k++ )
                        {
                            // NB: always access Gp() row-wise to minimise cache use

                            zM("[]",i,jj) += Gp()(i+N(),zxact(k))*Cval(zxact(k))*Gp()(jj+N(),zxact(k));
                        }
                    }

                    if ( i == jj )
                    {
                        zM("[]",i,jj) += 1.0;
                    }

                    zM("[]",jj,i) = zM(i,jj);
                }

                {
                    for ( k = 0 ; k < zxact.size() ; k++ )
                    {
                        // NB: always access Gp() row-wise to minimise cache use

                        zM("[]",jj,Nzs()) += Gp()(jj+N(),zxact(k))*Cval(zxact(k));

                        zn("[]",jj) += Gp()(jj+N(),zxact(k))*Cval(zxact(k))*zy(zxact(k));
                    }

                    zM("[]",Nzs(),jj) = zM(jj,Nzs());
                }
            }

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    // NB: always access Gp() row-wise to minimise cache use

                    zM("[]",Nzs(),Nzs()) += Cval(zxact(k));
                }
            }

            res |= 1;

            if ( alsoupdateM )
            {
                res |= updateM(jj);
                res |= updaten(jj);
            }
        }
    }

    return res;
}























int SSV_Generic::updateM(int j)
{
/*    (void) j;

    //zmodel.setGp(&zM,&zM);
    zmodel.resetKernel();*/

    if ( j >= 0 )
    {
        zmodel.resetKernel(1,j);
        zmodel.resetKernel(1,Nzs());
    }

    else
    {
        zmodel.resetKernel();
    }

    return 1;
}

int SSV_Generic::updateM(const Vector<int> &j)
{
    if ( j.size() )
    {
        int jj;

        for ( jj = 0 ; jj < j.size() ; jj++ )
        {
            //zmodel.setGp(&zM,&zM);
            zmodel.resetKernel(1,j(jj));
        }

        zmodel.resetKernel(1,Nzs());
    }

    return 1;
}

int SSV_Generic::updaten(int j)
{
    (void) j;

    zmodel.sety(zn);

    return 1;
}

int SSV_Generic::updaten(const Vector<int> &j)
{
    (void) j;

    zmodel.sety(zn);

    return 1;
}

int SSV_Generic::updateNzs(int i, int oldNzs, int newNzs)
{
    if ( newNzs > oldNzs )
    {
        zmodel.addTrainingVector(i,zn(i),z()(i));
    }

    else
    {
        zmodel.removeTrainingVector(i);
    }

    return 1;
}


























int SSV_Generic::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg) const
{
    int res;

    if ( ( gOutType() == 'R' ) && isUnderlyingScalar() )
    {
        res = SVM_Scalar::ghTrainingVector(resh,resg,i,retaltg);
    }

    else
    {
        resg = zb;

        if ( Nzs() )
        {
            int j;
            gentype temp;

            for ( j = 0 ; j < Nzs() ; j++ )
            {
                resg += K(temp,i,j+N())*zbeta(j);
            }
        }

        resh = resg;
    }

    return res;
}

void SSV_Generic::eTrainingVector(gentype &res, int i) const
{
    if ( ( gOutType() == 'R' ) && ( hOutType() == 'R' ) && ( targType() == 'R' ) && isUnderlyingScalar() )
    {
        SVM_Scalar::eTrainingVector(res,i);
    }

    else
    {
        hhTrainingVector(res,i);
        res -= y()(i);
    }

    return;
}


















































std::ostream &operator<<(std::ostream &output, const SSV_Generic &src)
{
    return src.printstream(output);
}

std::istream &operator>>(std::istream &input, SSV_Generic &dest)
{
    return dest.inputstream(input);
}

std::ostream &SSV_Generic::printstream(std::ostream &output) const
{
    SVM_Scalar::printstream(output);

    output << "Base training targets: " << zy      << "\n";
    output << "Base training alpha:   " << zbeta   << "\n";
    output << "Base training bias:    " << zb      << "\n";
    output << "Base training state:   " << zxstate << "\n";
    output << "Base training act:     " << zxact   << "\n";
    output << "Base training M:       " << zM      << "\n";
    output << "Base training n:       " << zn      << "\n\n";

    output << "SVM Training Block: " << zmodel << "\n\n";

    output << "lr:       " << zssvlr       << "\n";
    output << "mom:      " << zssvmom      << "\n";
    output << "tol:      " << zssvtol      << "\n";
    output << "ovsc:     " << zssvovsc     << "\n";
    output << "maxitcnt: " << zssvmaxitcnt << "\n";
    output << "maxtime:  " << zssvmaxtime  << "\n\n";

    return output;
}

std::istream &SSV_Generic::inputstream(std::istream &input )
{
    SVM_Scalar::inputstream(input);

    wait_dummy dummy;

    input >> dummy; input >> zy;
    input >> dummy; input >> zbeta;
    input >> dummy; input >> zb;
    input >> dummy; input >> zxstate;
    input >> dummy; input >> zxact;
    input >> dummy; input >> zM;
    input >> dummy; input >> zn;

    input >> dummy; input >> zmodel;

    input >> dummy; input >> zssvlr;
    input >> dummy; input >> zssvmom;
    input >> dummy; input >> zssvtol;
    input >> dummy; input >> zssvovsc;
    input >> dummy; input >> zssvmaxitcnt;
    input >> dummy; input >> zssvmaxtime;

    zmodel.setGp(&zM,&zM,0);

    return input;
}

int SSV_Generic::getparam(int ind, Vector<gentype> &val) const
{
    int res = 0;
    int q = 0;
    double qq = 0;

    if ( val.size() )
    {
        if ( val(zeroint()).isCastableToRealWithoutLoss() )
        {
            q  = (int) val(zeroint());
            qq = (double) val(zeroint());
        }
    }

    (void) qq;
    (void) q;

    switch ( ind )
    {
        case  605: { val.resize(1); val("[]",0) = b(); break; }

        case 2018: { val.resize(1); val("[]",0) = ssvlr     (); break; }
        case 2019: { val.resize(1); val("[]",0) = ssvmom    (); break; }
        case 2020: { val.resize(1); val("[]",0) = ssvtol    (); break; }
        case 2021: { val.resize(1); val("[]",0) = ssvovsc   (); break; }
        case 2022: { val.resize(1); val("[]",0) = ssvmaxtime(); break; }

        case 2114: { val.resize(1); val("[]",0) = ssvmaxitcnt(); break; }

        case 3103: { val = beta(); break; }

        default:
        {
            return ML_Base::getparam(ind,val);
            break;
        }
    }

    return res;
}

int SSV_Generic::setparam(int ind, const Vector<gentype> &val)
{
    int res = 0;

    switch ( ind )
    {
        case  605: { res = setb(val(0)); break; }

        case 2018: { res = setssvlr     ((double) val(0)); break; }
        case 2019: { res = setssvmom    ((double) val(0)); break; }
        case 2020: { res = setssvtol    ((double) val(0)); break; }
        case 2021: { res = setssvovsc   ((double) val(0)); break; }
        case 2022: { res = setssvmaxtime((double) val(0)); break; }

        case 2114: { res = setssvmaxitcnt((int) val(0)); break; }

        case 3103: { res = setbeta(val); break; }

        default:
        {
            return ML_Base::setparam(ind,val);
            break;
        }
    }

    return res;
}

const SparseVector<std::string> &SSV_Generic::infparam(void) const
{
    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    svmvolatile static int isfirst = 1;
    svmvolatile static SparseVector<std::string> locinfstore;
    SparseVector<std::string> &locinf = const_cast<SparseVector<std::string> &>(locinfstore);

    if ( isfirst )
    {
        {
            // Now we're locked we need to check again to see if this is
            // the only call to get locinf (as multiple threads may have
            // tried to grab the lock and only needs to setup locinf.

            if ( isfirst )
            {
                // Clear isfirst flag

                isfirst = 0;

                // Set locinf sparsevector as required.

                locinf = ML_Base::infparam();

                locinf("[]",  605) = "SSV: bias";

                locinf("[]", 2018) = "SSV: learning rate";
                locinf("[]", 2019) = "SSV: momentum factor";
                locinf("[]", 2020) = "SSV: tolerance factor";
                locinf("[]", 2021) = "SSV: overrun scaleback";
                locinf("[]", 2022) = "SSV: max training time";

                locinf("[]", 2114) = "SSV: max training iterations";

                locinf("[]", 3103) = "SSV: beta vector";
            }

        }
    }

    svm_mutex_unlock(eyelock);

    return locinf;
}

int SSV_Generic::getallparam(SparseVector<gentype> &val) const
{
    ML_Base::getallparam(val);

//    int q;

    val("[]", 605) = b();

    val("[]",2018) = ssvlr     ();
    val("[]",2019) = ssvmom    ();
    val("[]",2020) = ssvtol    ();
    val("[]",2021) = ssvovsc   ();
    val("[]",2022) = ssvmaxtime();

    val("[]",2114) = ssvmaxitcnt();

    return 0;
}

