
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

    Kcall.realOwner = this;

    gentype ydummy;
    SparseVector<gentype> xdummy;

    setQuadraticCost();
    assumeConsistentX();

    zM.resize(1,1).zero();
    zn.resize(1).zero();

    zmodel.setFixedBias(); // Bias is within model (see addTrainingVector below)
    zmodel.setQuadraticCost(); // Default to quadratic beta regulation
    zmodel.setsigma(1);
    zmodel.seteps(0);
    zmodel.addTrainingVector(0,ydummy,xdummy,1e6); // Note Cweight set very large here - this makes diagonal offset negligable (quadratic) and limits on bias large (linear)
    zmodel.sety(zn);

    inbypass   = 0;
    xbiasForce = 0;

    zssvlr       = DEFAULT_SSV_LR;
    zssvmom      = DEFAULT_SSV_MOM;
    zssvtol      = DEFAULT_SSV_TOL;
    zssvovsc     = DEFAULT_SSV_OUTEROVSC;
    zssvmaxitcnt = DEFAULT_SSV_MAXITS;
    zssvmaxtime  = 0;

    Nnz = 0;

    fixKcallback();

    return;
}

int SSV_Generic::setQuadRegul(void)
{
    int res = 0;

    if ( isLinRegul() )
    {
        double sigval = zmodel.eps();

        res |= zmodel.setQuadraticCost();
        res |= zmodel.setsigma(sigval);
        res |= zmodel.seteps(0);
    }

    return res;
}

int SSV_Generic::setLinRegul(void)
{
    int res = 0;

    if ( isQuadRegul() )
    {
        double sigval = zmodel.sigma();

        res |= zmodel.setLinearCost();
        res |= zmodel.setsigma(1e-6);
        res |= zmodel.seteps(sigval);
    }

    return res;
}

int SSV_Generic::setsigma(double xC)
{
    int res = 0;

    if ( isQuadRegul() )
    {
        res |= zmodel.setsigma(xC);
    }

    else
    {
        res |= zmodel.seteps(xC);
    }

    return res;
}


SSV_Generic::SSV_Generic(const SSV_Generic &src) : SVM_Scalar()
{
    setaltx(NULL);

    Kcall.realOwner = this;
    inbypass = 0;

    assign(src,0);

    return;
}

SSV_Generic::SSV_Generic(const SSV_Generic &src, const ML_Base *srcx) : SVM_Scalar()
{
    setaltx(srcx);

    Kcall.realOwner = this;
    inbypass = 0;

    assign(src,0);

    return;
}

void SSV_Generic::fixKcallback(void)
{
    (zmodel.getKernel_unsafe()).setType(800);
    (zmodel.getKernel_unsafe()).setAltCall(Kcall.MLid());

    zmodel.resetKernel();

    return;
}































int kssv_callback::isKVarianceNZ(void) const
{
    return 0;
}


void kssv_callback::Kmxfer(gentype &res, int &minmaxind, int typeis,
                            Vector<const SparseVector<gentype> *> &x,
                            Vector<const vecInfo *> &xinfo,
                            Vector<int> &ii,
                            int xdim, int m, int densetype, int resmode, int mlid) const
{
    if ( ( m == 0 ) || ( m == 1 ) || ( m == 2 ) || ( m == 3 ) || ( m == 4 ) )
    {
        kernPrecursor::Kmxfer(res,minmaxind,typeis,x,xinfo,ii,xdim,m,densetype,resmode,mlid);
        return;
    }

    throw("Km (m>4) not defined for kssv_callback");

    return;
}

void kssv_callback::K4xfer(gentype &res, int &minmaxind, int typeis,
                            const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                            const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                            int ia, int ib, int ic, int id,
                            int xdim, int densetype, int resmode, int mlid) const
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xa;
    (void) xb;
    (void) xc;
    (void) xd;
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;
    (void) xdinfo;
    (void) ia;
    (void) ib;
    (void) ic;
    (void) id;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;

    throw("K4 not defined for kssv_callback");

    return;
}

void kssv_callback::K3xfer(gentype &res, int &minmaxind, int typeis,
                            const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                            const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                            int ia, int ib, int ic, 
                            int xdim, int densetype, int resmode, int mlid) const
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xa;
    (void) xb;
    (void) xc;
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;
    (void) ia;
    (void) ib;
    (void) ic;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;

    throw("K3 not defined for kssv_callback");

    return;
}

void kssv_callback::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                            const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                            const vecInfo &xainfo, const vecInfo &xbinfo,
                            int i, int j,
                            int xdim, int densetype, int resmode, int mlid) const
{
    (void) minmaxind;
    (void) typeis;
    (void) xa;
    (void) xb;
    (void) xainfo;
    (void) xbinfo;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) dxyprod;
    (void) ddiffis;
    (void) mlid;

    NiceAssert( i >= 0 );
    NiceAssert( j >= 0 );

    res = (realOwner->zM)(i,j);

    return;
}

void kssv_callback::K1xfer(gentype &res, int &minmaxind, int typeis,
                            const SparseVector<gentype> &xa, 
                            const vecInfo &xainfo, 
                            int ia, 
                            int xdim, int densetype, int resmode, int mlid) const
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xa;
    (void) xainfo;
    (void) ia;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;

    throw("K1 not defined for kssv_callback");

    return;
}

void kssv_callback::K0xfer(gentype &res, int &minmaxind, int typeis,
                            int xdim, int densetype, int resmode, int mlid) const
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;

    throw("K0 not defined for kssv_callback");

    return;
}




void kssv_callback::Kmxfer(double &res, int &minmaxind, int typeis,
                            Vector<const SparseVector<gentype> *> &x,
                            Vector<const vecInfo *> &xinfo,
                            Vector<int> &ii,
                            int xdim, int m, int densetype, int resmode, int mlid) const
{
    gentype temp(res);

    Kmxfer(temp,minmaxind,typeis,x,xinfo,ii,xdim,m,densetype,resmode,mlid);

    res = (double) temp;

    return;
}

void kssv_callback::K4xfer(double &res, int &minmaxind, int typeis,
                            const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                            const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                            int ia, int ib, int ic, int id,
                            int xdim, int densetype, int resmode, int mlid) const
{
    gentype temp(res);

    K4xfer(temp,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

    res = (double) temp;

    return;
}

void kssv_callback::K3xfer(double &res, int &minmaxind, int typeis,
                            const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                            const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                            int ia, int ib, int ic, 
                            int xdim, int densetype, int resmode, int mlid) const
{
    gentype temp(res);

    K3xfer(temp,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

    res = (double) temp;

    return;
}

void kssv_callback::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                             const vecInfo &xainfo, const vecInfo &xbinfo,
                             int i, int j,
                             int xdim, int densetype, int resmode, int mlid) const
{
    gentype temp(res);
    gentype tempa(dxyprod);
    gentype tempb(ddiffis);

    K2xfer(tempa,tempb,temp,minmaxind,typeis,xa,xb,xainfo,xbinfo,i,j,xdim,densetype,resmode,mlid);

    res = (double) temp;
    dxyprod = (double) tempa;
    ddiffis = (double) tempb;

    return;
}

void kssv_callback::K1xfer(double &res, int &minmaxind, int typeis,
                            const SparseVector<gentype> &xa, 
                            const vecInfo &xainfo, 
                            int ia, 
                            int xdim, int densetype, int resmode, int mlid) const
{
    gentype temp(res);

    K1xfer(temp,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

    res = (double) temp;

    return;
}

void kssv_callback::K0xfer(double &res, int &minmaxind, int typeis,
                            int xdim, int densetype, int resmode, int mlid) const
{
    gentype temp(res);

    K0xfer(temp,minmaxind,typeis,xdim,densetype,resmode,mlid);

    res = (double) temp;

    return;
}







































int SSV_Generic::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    int res = 0;

    if ( !inbypass )
    {
        res |= deactivate(onlyChangeRowI);
        res |= SVM_Scalar::resetKernel(modind,onlyChangeRowI,updateInfo);
        res |= activate(onlyChangeRowI);
    }

    else
    {
        inbypass++;
        res |= SVM_Scalar::resetKernel(modind,onlyChangeRowI,updateInfo);
        inbypass--;
    }

    return res;
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

    SparseVector<gentype> locx(x);

    return qaddTrainingVector(i,y,locx,Cweigh,epsweigh);
}

int SSV_Generic::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    int res = 0;

    zxstate.add(i);    zxstate("&",i) = 0;
    zy.add(i);         zy("&",i) = (double) y;
    locCscale.add(i);  locCscale("&",i) = 1;

    if ( zxact.size() )
    {
        int j;

        for ( j = 0 ; j < zxact.size() ; j++ )
        {
            if ( zxact(j) >= i )
            {
                zxact("&",j)++;
            }
        }
    }

    res |= SVM_Scalar::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    res |= activate(i);

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

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            res |= addTrainingVector(i+j,y(j),x(j),Cweigh(j),epsweigh(j));
        }
    }

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

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            res |= qaddTrainingVector(i+j,y(j),x("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SSV_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    zxstate.remove(i);
    zy.remove(i);

    res |= deactivate(i);

    if ( zxact.size() )
    {
        int j;

        for ( j = 0 ; j < zxact.size() ; j++ )
        {
            if ( zxact(j) > i )
            {
                zxact("&",j)--;
            }
        }
    }

    locCscale.remove(i);
    res |= SVM_Scalar::removeTrainingVector(i,y,x);

    return res;
}

int SSV_Generic::setx(int i, const SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    
    int res = 0;

    res |= deactivate(i);
    res |= SVM_Scalar::setx(i,x);
    res |= activate(i);

    return res;
}

int SSV_Generic::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x)
{
    NiceAssert( i >= zeroint() );
    NiceAssert( i < N() );

    NiceAssert( i.size() == x.size() );

    int res = 0;

    res |= deactivate(i);
    res |= SVM_Scalar::setx(i,x);
    res |= activate(i);

    return res;
}

int SSV_Generic::qswapx(int i, SparseVector<gentype> &x, int dontupdate)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    
    int res = 0;

    if ( dontupdate )
    {
        res |= SVM_Scalar::qswapx(i,x,dontupdate);
    }

    else
    {
        res |= deactivate(i);
        res |= SVM_Scalar::qswapx(i,x,dontupdate);
        res |= activate(i);
    }

    return res;
}

int SSV_Generic::qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate)
{
    NiceAssert( i >= zeroint() );
    NiceAssert( i < N() );

    NiceAssert( i.size() == x.size() );

    int res = 0;

    if ( dontupdate )
    {
        res |= SVM_Scalar::qswapx(i,x,dontupdate);
    }

    else
    {
        res |= deactivate(i);
        res |= SVM_Scalar::qswapx(i,x,dontupdate);
        res |= activate(i);
    }

    return res;
}

int SSV_Generic::setd(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    res |= deactivate(i);
    res |= SVM_Scalar::setd(i,d);
    res |= activate(i);

    return res;
}

int SSV_Generic::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    NiceAssert( i.size() == d.size() );

    int res = 0;

    res |= deactivate(i);
    res |= SVM_Scalar::setd(i,d);
    res |= activate(i);

    return res;
}

int SSV_Generic::setCweight(int i, double nv)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    
    int res = 0;

    res |= deactivate(i);
    res |= SVM_Scalar::setCweight(i,nv);
    res |= activate(i);

    return res;
}

int SSV_Generic::setCweight(const Vector<int> &i, const Vector<double> &nv)
{
    NiceAssert( i >= zeroint() );
    NiceAssert( i < N() );

    NiceAssert( i.size() == nv.size() );

    int res = 0;

    res |= deactivate(i);
    res |= SVM_Scalar::setCweight(i,nv);
    res |= activate(i);

    return res;
}

int SSV_Generic::setCweightfuzz(int i, double nv)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    
    int res = 0;

    res |= deactivate(i);
    res |= SVM_Scalar::setCweightfuzz(i,nv);
    res |= activate(i);

    return res;
}

int SSV_Generic::setCweightfuzz(const Vector<int> &i, const Vector<double> &nv)
{
    NiceAssert( i >= zeroint() );
    NiceAssert( i < N() );

    NiceAssert( i.size() == nv.size() );

    int res = 0;

    res |= deactivate(i);
    res |= SVM_Scalar::setCweightfuzz(i,nv);
    res |= activate(i);

    return res;
}

int SSV_Generic::setsigmaweight(int i, double nv)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    
    int res = 0;

    res |= deactivate(i);
    res |= SVM_Scalar::setsigmaweight(i,nv);
    res |= activate(i);

    return res;
}

int SSV_Generic::setsigmaweight(const Vector<int> &i, const Vector<double> &nv)
{
    NiceAssert( i >= zeroint() );
    NiceAssert( i < N() );

    NiceAssert( i.size() == nv.size() );

    int res = 0;

    res |= deactivate(i);
    res |= SVM_Scalar::setsigmaweight(i,nv);
    res |= activate(i);

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
    res |= zmodel.scale(a);
    res |= updatez();

    return res;
}

int SSV_Generic::reset(void)
{
    int res = 0;

    res |= SVM_Scalar::reset();
    res |= zmodel.reset();

    if ( Nzs() )
    {
        int j;

        Vector<SparseVector<gentype> > newz(Nzs());

        qswapz(newz,1);

        for ( j = 0 ; j < Nzs() ; j++ )
        {
            newz("&",j).softzero();
        }
        
        qswapz(newz,0);
    }

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

    inbypass++;

    res |= SVM_Scalar::setx(j+N(),newz);
    res |= updatez(j);

    inbypass--;

    return res;
}

int SSV_Generic::setz(const Vector<int> &j, const Vector<SparseVector<gentype> > &newz)
{
    NiceAssert( j >= zeroint() );
    NiceAssert( j < Nzs() );

    NiceAssert( j.size() == newz.size() );

    int res = 0;

    inbypass++;

    res |= SVM_Scalar::setx(j+N(),newz);
    res |= updatez(j);

    inbypass--;

    return res;
}

int SSV_Generic::setz(const Vector<SparseVector<gentype> > &newz)
{
    NiceAssert( Nzs() == newz.size() );

    int res = 0;

    inbypass++;

    retVector<int> tmpva;

    res |= SVM_Scalar::setx(cntintvec(Nzs(),tmpva)+N(),newz);
    res |= updatez();

    inbypass--;

    return res;
}

int SSV_Generic::qswapz(int j, SparseVector<gentype> &newz, int dontupdate)
{
    NiceAssert( j >= zeroint() );
    NiceAssert( j < Nzs() );

    int res = 0;

    inbypass++;

    res |= SVM_Scalar::qswapx(j+N(),newz,dontupdate);

    if ( !dontupdate )
    {
        res |= updatez(j);
    }

    inbypass--;

    return res;
}

int SSV_Generic::qswapz(const Vector<int> &j, Vector<SparseVector<gentype> > &newz, int dontupdate)
{
    NiceAssert( j >= zeroint() );
    NiceAssert( j < Nzs() );

    NiceAssert( j.size() == newz.size() );

    int res = 0;

    inbypass++;

    res |= SVM_Scalar::qswapx(j+N(),newz,dontupdate);

    if ( !dontupdate )
    {
        res |= updatez(j);
    }

    inbypass--;

    return res;
}

int SSV_Generic::qswapz(Vector<SparseVector<gentype> > &newz, int dontupdate)
{
    NiceAssert( Nzs() == newz.size() );

    int res = 0;

    inbypass++;

    retVector<int> tmpva;

    res |= SVM_Scalar::qswapx(cntintvec(Nzs(),tmpva)+N(),newz,dontupdate);

    if ( !dontupdate )
    {
        res |= updatez();
    }

    inbypass--;

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

        newind("&",k++) = i;
    }

    if ( res == 2 )
    {
        // Propogate M changes from size increase

        updateM(newind);
        updaten(newind);
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

int SSV_Generic::setbeta(const Vector<double> &newBeta) 
{ 
    NiceAssert( zbeta.size() == newBeta.size() );

    if ( zbeta.size() )
    {
        int i;

        for ( i = 0 ; i < zbeta.size() ; i++ )
        {
            zbeta("&",i) = newBeta(i);
        }
    }

    if ( isUnderlyingScalar() )
    {
        Vector<gentype> newalpha(N());

        newalpha.zero();
        newalpha.append(N(),zbeta);

        SVM_Scalar::setAlpha(newalpha);
    }

    return 1; 
}

int SSV_Generic::setb(const double &newb) 
{ 
    zb = newb; 

    if ( isUnderlyingScalar() )
    {
        SVM_Scalar::setBias(zb);
    }

    return 1; 
}










































int SSV_Generic::setBiasForce(double nv)
{
    zn("&",Nzs()) += xbiasForce;
    xbiasForce = nv;
    zn("&",Nzs()) -= xbiasForce;
    updaten(Nzs());

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
                zn("&",j) -= Gp()(j+N(),i)*Cval*ZY(i);
            }
        }

        zn("&",Nzs()) -= Cval*ZY(i);
    }

    zy("&",i) = (double) y;

    if ( zxstate(i) )
    {
        if ( Nzs() )
        {
            int j;

            for ( j = 0 ; j < Nzs() ; j++ )
            {
                zn("&",j) += Gp()(j+N(),i)*Cval*ZY(i);
            }
        }

        zn("&",Nzs()) += Cval*ZY(i);

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
            Cval("&",ii) = calcCval(i(ii));
        }

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            if ( zxstate(i(ii)) )
            {
                int j;

                for ( j = 0 ; j < Nzs() ; j++ )
                {
                    zn("&",j) -= Gp()(j+N(),i(ii))*Cval(ii)*ZY(i(ii));
                }

                zn("&",Nzs()) -= Cval(ii)*ZY(i(ii));
            }
        }
    }

    if ( i.size() )
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            zy("&",i(ii)) = (double) y(ii);
        }
    }

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
                    zn("&",j) += Gp()(j+N(),i(ii))*Cval(ii)*ZY(i(ii));
                }

                zn("&",Nzs()) += Cval(ii)*ZY(i(ii));
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
        retVector<int> tmpva;

        return activate(cntintvec(N(),tmpva));
    }

    NiceAssert( jj >= 0 );
    NiceAssert( jj < N() );

    int res = 0;

    if ( !zxstate(jj) && d()(jj) )
    {
        int i,j;

        zxstate("&",jj) = 1;

        zxact.add(zxact.size());
        zxact("&",zxact.size()-1) = jj;

        Nnz++;

        zM *= ( ( Nnz > 1 ) ? ((double) (Nnz-1))/((double) Nnz) : 1.0 );
        zn *= ( ( Nnz > 1 ) ? ((double) (Nnz-1))/((double) Nnz) : 1.0 );

        // see updatez functions

        double Cval = calcCval(jj);

        if ( Nzs() )
        {
            for ( i = 0 ; i < Nzs() ; i++ )
            {
                for ( j = 0 ; j < Nzs() ; j++ )
                {
                    zM("&",i,j) += Gp()(i+N(),jj)*Cval*Gp()(j+N(),jj);
                }

                zM("&",i,Nzs()) += Gp()(i+N(),jj)*Cval;
                zM("&",Nzs(),i) =  zM(i,Nzs());

                zn("&",j) += Gp()(i+N(),jj)*Cval*ZY(jj);
            }
        }

        zM("&",Nzs(),Nzs()) += Cval;

        zn("&",Nzs()) += Cval*ZY(jj);

        res |= 1;
        res |= updateM(-1);
        res |= updaten(-1);
    }

    return res;
}

int SSV_Generic::deactivate(int jj)
{
    if ( jj == -1 )
    {
        retVector<int> tmpva;

        return deactivate(cntintvec(N(),tmpva));
    }

    NiceAssert( jj >= 0 );
    NiceAssert( jj < N() );

    int res = 0;

    if ( zxstate(jj) )
    {
        int i,j;

        zxstate("&",jj) = 0;

        i = 0;

        for ( j = 0 ; j < zxact.size() ; j++ )
        {
            if ( zxact(j) == jj )
            {
                zxact.remove(j);
                break;
            }
        }

        Nnz--;

        zM *= ( ( Nnz > 0 ) ? ((double) (Nnz+1))/((double) Nnz) : 1.0 );
        zn *= ( ( Nnz > 0 ) ? ((double) (Nnz+1))/((double) Nnz) : 1.0 );

        // see updatez functions

        double Cval = calcCval(jj);

        if ( Nzs() )
        {
            for ( i = 0 ; i < Nzs() ; i++ )
            {
                for ( j = 0 ; j < Nzs() ; j++ )
                {
                    zM("&",i,j) -= Gp()(i+N(),jj)*Cval*Gp()(j+N(),jj);
                }

                zM("&",i,Nzs()) -= Gp()(i+N(),jj)*Cval;
                zM("&",Nzs(),i) =  zM(i,Nzs());

                zn("&",j) -= Gp()(i+N(),jj)*Cval*ZY(jj);
            }
        }

        zM("&",Nzs(),Nzs()) -= Cval;

        zn("&",Nzs()) -= Cval*ZY(jj);

        res |= 1;
        res |= updateM(-1);
        res |= updaten(-1);
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

    zn("&",Nzs()) = -xbiasForce;

    if ( Nzs() )
    {
        if ( zxact.size() )
        {
            int j,k;

            Vector<double> Cval(N());

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    Cval("&",zxact(k)) = calcCval(zxact(k));
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

                            zM("&",i,j) += Gp()(i+N(),zxact(k))*Cval(zxact(k))*Gp()(j+N(),zxact(k));
                        }
                    }

                    zM("&",j,i) = zM(i,j);
                }

                {
                    for ( k = 0 ; k < zxact.size() ; k++ )
                    {
                        // NB: always access Gp() row-wise to minimise cache use

                        zM("&",j,Nzs()) += Gp()(j+N(),zxact(k))*Cval(zxact(k));

                        zn("&",j) += Gp()(j+N(),zxact(k))*Cval(zxact(k))*ZY(zxact(k));
                    }

                    zM("&",Nzs(),j) = zM(j,Nzs());
                }
            }

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    // NB: always access Gp() row-wise to minimise cache use

                    zM("&",Nzs(),Nzs()) += Cval(zxact(k));
                    zn("&",Nzs()) += Cval(zxact(k))*ZY(zxact(k));
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
            zM("&",jj(j),i) = 0.0;
            zM("&",i,jj(j)) = 0.0;
        }

        zM("&",jj(j),Nzs()) = 0.0;
        zM("&",Nzs(),jj(j)) = 0.0;

        zn("&",jj(j)) = 0.0;
    }

    zM("&",Nzs(),Nzs()) = 0.0;
    zn("&",Nzs()) = -xbiasForce;

    if ( Nzs() && jj.size() )
    {
        if ( zxact.size() )
        {
            int k;

            Vector<double> Cval(N());

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    Cval("&",zxact(k)) = calcCval(zxact(k));
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

                            zM("&",i,jj(j)) += Gp()(i+N(),zxact(k))*Cval(zxact(k))*Gp()(jj(j)+N(),zxact(k));
                        }
                    }

                    zM("&",jj(j),i) = zM(i,jj(j));
                }

                {
                    for ( k = 0 ; k < zxact.size() ; k++ )
                    {
                        // NB: always access Gp() row-wise to minimise cache use

                        zM("&",jj(j),Nzs()) += Gp()(jj(j)+N(),zxact(k))*Cval(zxact(k));

                        zn("&",jj(j)) += Gp()(jj(j)+N(),zxact(k))*Cval(zxact(k))*ZY(zxact(k));
                    }

                    zM("&",Nzs(),jj(j)) = zM(jj(j),Nzs());
                }
            }

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    // NB: always access Gp() row-wise to minimise cache use

                    zM("&",Nzs(),Nzs()) += Cval(zxact(k));
                    zn("&",Nzs()) += Cval(zxact(k))*ZY(zxact(k));
                }
            }

            res |= 1;
            res |= updateM(jj);
            res |= updateM(Nzs());
            res |= updaten(jj);
            res |= updaten(Nzs());
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
            zM("&",jj,i) = 0.0;
            zM("&",i,jj) = 0.0;
        }

        zM("&",jj,Nzs()) = 0.0;
        zM("&",Nzs(),jj) = 0.0;

        zn("&",jj) = 0.0;
    }

    zM("&",Nzs(),Nzs()) = 0.0;
    zn("&",Nzs()) = -xbiasForce;

    if ( Nzs() )
    {
        if ( zxact.size() )
        {
            int k;

            Vector<double> Cval(N());

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    Cval("&",zxact(k)) = calcCval(zxact(k));
                }
            }

            {
                for ( i = 0 ; i < Nzs() ; i++ )
                {
                    {
                        for ( k = 0 ; k < zxact.size() ; k++ )
                        {
                            // NB: always access Gp() row-wise to minimise cache use

                            zM("&",i,jj) += Gp()(i+N(),zxact(k))*Cval(zxact(k))*Gp()(jj+N(),zxact(k));
                        }
                    }

                    zM("&",jj,i) = zM(i,jj);
                }

                {
                    for ( k = 0 ; k < zxact.size() ; k++ )
                    {
                        // NB: always access Gp() row-wise to minimise cache use

                        zM("&",jj,Nzs()) += Gp()(jj+N(),zxact(k))*Cval(zxact(k));

                        zn("&",jj) += Gp()(jj+N(),zxact(k))*Cval(zxact(k))*ZY(zxact(k));
                    }

                    zM("&",Nzs(),jj) = zM(jj,Nzs());
                }
            }

            {
                for ( k = 0 ; k < zxact.size() ; k++ )
                {
                    // NB: always access Gp() row-wise to minimise cache use

                    zM("&",Nzs(),Nzs()) += Cval(zxact(k));
                    zn("&",Nzs()) += Cval(zxact(k))*ZY(zxact(k));
                }
            }

            res |= 1;

            if ( alsoupdateM )
            {
                res |= updateM(jj);
                res |= updateM(Nzs());
                res |= updaten(jj);
                res |= updaten(Nzs());
            }
        }
    }

    return res;
}























int SSV_Generic::updateM(int j)
{
    zmodel.resetKernel(1,j);

    return 1;
}

int SSV_Generic::updateM(const Vector<int> &j)
{
    (void) j;

    // Can't update just relevant columns as gradient errors will creep in for *all* alphas

    zmodel.resetKernel();

    return 1;
}

int SSV_Generic::updaten(int j)
{
    if ( j == -1 )
    {
        zmodel.sety(zn);
    }

    else
    {
        zmodel.sety(j,zn(j));
    }

    return 1;
}

int SSV_Generic::updaten(const Vector<int> &j)
{
    retVector<double> tmpva;

    zmodel.sety(j,zn(j,tmpva));

    return 1;
}

int SSV_Generic::updateNzs(int i, int oldNzs, int newNzs)
{
    if ( newNzs > oldNzs )
    {
        gentype temp(zn(i));

        zmodel.addTrainingVector(i,temp,z()(i));
    }

    else
    {
        zmodel.removeTrainingVector(i);
    }

    return 1;
}


























int SSV_Generic::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) pxyprodi;

    int dtv = 0;
    int res = 1;

    if ( ( gOutType() == 'R' ) && isUnderlyingScalar() )
    {
        res = SVM_Scalar::ghTrainingVector(resh,resg,i,retaltg);
    }

    else if ( ( dtv = xtang(i) & 7 ) )
    {
        NiceAssert( !( dtv & 4 ) );

        resg  = zb;
        resg *= 0.0;

        if ( ( dtv > 0 ) && Nzs() )
        {
            int j;
            gentype temp;

            for ( j = 0 ; j < Nzs() ; j++ )
            {
                resg += K2(temp,i,j+N())*zbeta(j);
            }
        }

        resh = resg;
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
                resg += K2(temp,i,j+N())*zbeta(j);
            }
        }

        resh = resg;
    }

    return res;
}













































std::ostream &SSV_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Base training targets:    " << zy         << "\n";
    repPrint(output,'>',dep) << "Base training alpha:      " << zbeta      << "\n";
    repPrint(output,'>',dep) << "Base training bias:       " << zb         << "\n";
    repPrint(output,'>',dep) << "Base training state:      " << zxstate    << "\n";
    repPrint(output,'>',dep) << "Base training act:        " << zxact      << "\n";
    repPrint(output,'>',dep) << "Base training M:          " << zM         << "\n";
    repPrint(output,'>',dep) << "Base training n:          " << zn         << "\n";
    repPrint(output,'>',dep) << "Base training C rescale:  " << locCscale  << "\n";
    repPrint(output,'>',dep) << "Base training bias force: " << xbiasForce << "\n";
    repPrint(output,'>',dep) << "Base training Nnz:        " << Nnz        << "\n\n";

    repPrint(output,'>',dep) << "lr:       " << zssvlr       << "\n";
    repPrint(output,'>',dep) << "mom:      " << zssvmom      << "\n";
    repPrint(output,'>',dep) << "tol:      " << zssvtol      << "\n";
    repPrint(output,'>',dep) << "ovsc:     " << zssvovsc     << "\n";
    repPrint(output,'>',dep) << "maxitcnt: " << zssvmaxitcnt << "\n";
    repPrint(output,'>',dep) << "maxtime:  " << zssvmaxtime  << "\n\n";

    repPrint(output,'>',dep) << "SVM Training Block: " << zmodel << "\n\n";

    SVM_Scalar::printstream(output,dep+1);

    return output;
}

std::istream &SSV_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> zy;
    input >> dummy; input >> zbeta;
    input >> dummy; input >> zb;
    input >> dummy; input >> zxstate;
    input >> dummy; input >> zxact;
    input >> dummy; input >> zM;
    input >> dummy; input >> zn;
    input >> dummy; input >> locCscale;
    input >> dummy; input >> xbiasForce;
    input >> dummy; input >> Nnz;

    input >> dummy; input >> zssvlr;
    input >> dummy; input >> zssvmom;
    input >> dummy; input >> zssvtol;
    input >> dummy; input >> zssvovsc;
    input >> dummy; input >> zssvmaxitcnt;
    input >> dummy; input >> zssvmaxtime;

    input >> dummy; input >> zmodel;

    SVM_Scalar::inputstream(input);

    fixKcallback();

    return input;
}

int SSV_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
{
    int k,res = 0;

    NiceAssert( xa.size() == xb.size() );

    val.resize(xa.size());

    for ( k = 0 ; k < xa.size() ; k++ )
    {
        res |= getparam(ind,val("&",k),xa(k),ia,xb(k),ib);
    }

    return res;
}

int SSV_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const
{
    int res = 0;

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    int isfallback = 0;

    retVector<double> tmpva;
    retVector<gentype> tmpvb;

    switch ( ind )
    {
        case 8000: { val = Nzs();                    break; }
        case 8001: { val = beta();                   break; }
        case 8002: { val = b();                      break; }
        case 8003: { val = zmin()(tmpva);            break; }
        case 8004: { val = zmax()(tmpva);            break; }
        case 8005: { val = xstate();                 break; }
        case 8006: { val = xact();                   break; }
        case 8007: { val = M();                      break; }
        case 8008: { val = n();                      break; }
        case 8009: { val = isQuadRegul();            break; }
        case 8010: { val = isLinRegul();             break; }
        case 8011: { val = biasForce();              break; }
        case 8012: { val = anomalclass();            break; }
        case 8013: { val = ssvlr();                  break; }
        case 8014: { val = ssvmom();                 break; }
        case 8015: { val = ssvtol();                 break; }
        case 8016: { val = ssvovsc();                break; }
        case 8017: { val = ssvmaxitcnt();            break; }
        case 8018: { val = ssvmaxtime();             break; }

        case 8100: { val = z()((int) xa)(tmpvb); break; }

        default:
        {
            isfallback = 1;
            res = ML_Base::getparam(ind,val,xa,ia,xb,ib);

            break;
        }
    }

    if ( ( ia || ib ) && !isfallback )
    {
        val.force_null();
    }

    return res;
}

