
//
// Vector regression SVM (matrix reduction to binary)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "svm_vector_mredbin.h"
#include <iostream>
#include <sstream>
#include <string>



int scalar_callback::isKVarianceNZ(void) const
{
    return 0;
}

void scalar_callback::Kmxfer(gentype &res, int &minmaxind, int typeis,
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

    throw("Km (m>4) not defined for scalar_callback");

    return;
}

void scalar_callback::K4xfer(gentype &res, int &minmaxind, int typeis,
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

    throw("K4 not defined for scalar_callback");

    return;
}

void scalar_callback::K3xfer(gentype &res, int &minmaxind, int typeis,
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

    throw("K3 not defined for scalar_callback");

    return;
}

void scalar_callback::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
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

    int ix = 0;
    int jx = 0;
    int iq = 0;
    int jq = 0;

    int ixsplit = realOwner->ixsplit;
    int iqsplit = realOwner->iqsplit;
    int ixskip  = realOwner->ixskip;
    int ixskipc = realOwner->ixskipc;

    int tspaceDim = realOwner->tspaceDim();

    gentype temp;

    NiceAssert( ( ixsplit == -1 ) || ( ixskip == -1 ) );
    NiceAssert( ( i >= 0 ) && ( j >= 0 ) );

    res = 0.0;

    {
        ix = i/tspaceDim;
        iq = i%tspaceDim;

        if ( !( ( ixsplit == -1 ) || ( ix < ixsplit ) || ( ( ix == ixsplit ) && ( iq < iqsplit ) ) ) )
        {
            i += tspaceDim-iqsplit;
        }

        ix = i/tspaceDim;
        iq = i%tspaceDim;

        if ( !( ( ixskip == -1 ) || ( ix <= ixskip ) ) )
        {
            ix  = ixskip;
            i  -= ixskip*tspaceDim;

            ix += i/(tspaceDim-ixskipc);
            iq  = i%(tspaceDim-ixskipc);
        }
    }

    {
        jx = j/tspaceDim;
        jq = j%tspaceDim;

        if ( !( ( ixsplit == -1 ) || ( jx < ixsplit ) || ( ( jx == ixsplit ) && ( jq < iqsplit ) ) ) )
        {
            j += tspaceDim-iqsplit;
        }

        jx = j/tspaceDim;
        jq = j%tspaceDim;

        if ( !( ( ixskip == -1 ) || ( jx <= ixskip ) ) )
        {
            jx  = ixskip;
            j  -= ixskip*tspaceDim;

            jx += j/(tspaceDim-ixskipc);
            jq  = j%(tspaceDim-ixskipc);
        }
    }

    realOwner->K2(temp,ix,jx);

    res = gentypeToMatrixRep(temp,tspaceDim,iq,jq);

    return;
}

void scalar_callback::K1xfer(gentype &res, int &minmaxind, int typeis,
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

    throw("K1 not defined for scalar_callback");

    return;
}

void scalar_callback::K0xfer(gentype &res, int &minmaxind, int typeis,
                            int xdim, int densetype, int resmode, int mlid) const
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;

    throw("K0 not defined for scalar_callback");

    return;
}





void scalar_callback::Kmxfer(double &res, int &minmaxind, int typeis,
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

void scalar_callback::K4xfer(double &res, int &minmaxind, int typeis,
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

void scalar_callback::K3xfer(double &res, int &minmaxind, int typeis,
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

void scalar_callback::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
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

void scalar_callback::K1xfer(double &res, int &minmaxind, int typeis,
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

void scalar_callback::K0xfer(double &res, int &minmaxind, int typeis,
                            int xdim, int densetype, int resmode, int mlid) const
{
    gentype temp(res);

    K0xfer(temp,minmaxind,typeis,xdim,densetype,resmode,mlid);

    res = (double) temp;

    return;
}









SVM_Vector_Mredbin::SVM_Vector_Mredbin() : SVM_Generic()
{
    setaltx(NULL);

    Kcall.realOwner = this;

    aN  = 0;
    aNS = 0;
    aNZ = 0;
    aNF = 0;
    aNC = 0;

    setzero(dbiasA);
    ixsplit = -1;
    iqsplit = -1;
    ixskip  = -1;
    ixskipc = -1;
    fixKcallback();

    Gpn.addCol(0);
    locnaivesetGpnExt();
    Q.setbiasdim(1+1);
    Q.setbiasdim(1+0);
    Gpn.removeCol(0);
    locnaivesetGpnExt();

    return;
}

SVM_Vector_Mredbin::SVM_Vector_Mredbin(const SVM_Vector_Mredbin &src) : SVM_Generic()
{
    setaltx(NULL);

    Kcall.realOwner = this;

    assign(src,0);

    return;
}

SVM_Vector_Mredbin::SVM_Vector_Mredbin(const SVM_Vector_Mredbin &src, const ML_Base *xsrc) : SVM_Generic()
{
    setaltx(xsrc);

    Kcall.realOwner = this;

    assign(src,1);

    return;
}

SVM_Vector_Mredbin::~SVM_Vector_Mredbin()
{
    return;
}

double SVM_Vector_Mredbin::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm2(ha-hb);
    }

    return res;
}

int SVM_Vector_Mredbin::scale(double a)
{
    NiceAssert( a >= 0.0 );
    NiceAssert( a <= 1.0 );

    int res = Q.scale(a);

    dbiasA *= a;
    dalphaA.scale(a);

    if ( a == 0.0 )
    {
        aNS = 0;
        aNZ = aN;
        aNF = 0;
        aNC = aN;

        xalphaState = zeroint();
    }

    SVM_Generic::basescalealpha(a);
    SVM_Generic::basescalebias(a);

    return res;
}

int SVM_Vector_Mredbin::reset(void)
{
    int res = Q.reset();

    setzero(dbiasA);  // multiply to preserve order
    setzero(dalphaA); // scale to preserve order

    aNS = 0;
    aNZ = aN;
    aNF = 0;
    aNC = aN;

    xalphaState = zeroint();

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return res;
}

int SVM_Vector_Mredbin::setAlphaV(const Vector<Vector<double> > &newAlpha)
{
    NiceAssert( newAlpha.size() == N() );

    Vector<double> newAlphaV(N()*tspaceDim());

    if ( N() )
    {
        int i,j,k;

        for ( i = 0 ; i < N() ; i++ )
	{
            NiceAssert( newAlpha(i).size() == tspaceDim() );

            for ( j = 0 ; j < tspaceDim() ; j++ )
            {
                k = (i*tspaceDim())+j;

                newAlphaV("&",k) = newAlpha(i)(j);
	    }
	}
    }

    Q.setAlphaR(newAlphaV);

    updateAlpha();

    return 1;
}

int SVM_Vector_Mredbin::setBiasV(const Vector<double>  &newBias)
{
    if ( !N() )
    {
        settspaceDim(newBias.size());
    }

    NiceAssert( newBias.size() == tspaceDim() );

    Vector<double> newBiasV(tspaceDim());

    int j;

    for ( j = 0 ; j < tspaceDim() ; j++ )
    {
        newBiasV("&",j) = newBias(j);
    }

    Q.setBiasVMulti(newBiasV);

    updateBias();

    return 1;
}



int SVM_Vector_Mredbin::setd(int i, int d)
{
    int j,k;

    if ( xalphaState(i) == 1 )
    {
        aNS--;
        aNF--;
        aNZ++;
    }

    else if ( xalphaState(i) == 2 )
    {
        aNS--;
        aNC--;
        aNZ++;
    }

    int res = 0;

    for ( j = 0 ; j < tspaceDim() ; j++ )
    {
        k = (i*tspaceDim())+j;

        res |= Q.setd(k,xisclass(i,d,j));
    }

    return res;
}

int SVM_Vector_Mredbin::sety(int i, const gentype &z)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( z.isCastableToVectorWithoutLoss() );

    Vector<gentype> zz((const Vector<gentype> &) z);
    Vector<double> zzz(zz.size());

    if ( zz.size() )
    {
        int k;

        for ( k = 0 ; k < zz.size() ; k++ )
        {
            zzz("&",k) = (double) zz(k);
        }
    }

    return sety(i,zzz);
}

int SVM_Vector_Mredbin::sety(const Vector<int> &j, const Vector<gentype> &yn)
{
    NiceAssert( j.size() == yn.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
        {
            res |= sety(j(i),yn(i));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::sety(const Vector<gentype> &yn)
{
    NiceAssert( N() == yn.size() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    int res = SVM_Generic::resetKernel(modind,onlyChangeRowI,updateInfo);
    res |= Q.resetKernel(modind,onlyChangeRowI,updateInfo);

    return res;
}

int SVM_Vector_Mredbin::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    int res = SVM_Generic::setKernel(xkernel,modind,onlyChangeRowI);
    res |= Q.resetKernel(modind,onlyChangeRowI);

    return res;
}

int SVM_Vector_Mredbin::sety(int i, const Vector<double>  &z)
{
    NiceAssert( z.size() == tspaceDim() );

    int j,k;
    int res = 0;

    traintarg("&",i) = z;

    gentype yn;
    yn = z;
    res |= SVM_Generic::sety(i,yn);

    for ( j = 0 ; j < tspaceDim() ; j++ )
    {
        k = (i*tspaceDim())+j;

        res |= Q.sety(k,z(j));
    }

    return res;
}

int SVM_Vector_Mredbin::setCweight(int i, double xCweight)
{
    int j,k;
    int res = 0;

    for ( j = 0 ; j < tspaceDim() ; j++ )
    {
        k = (i*tspaceDim())+j;

        res |= Q.setCweight(k,xCweight);
    }

    return res;
}

int SVM_Vector_Mredbin::setCweightfuzz(int i, double xCweight)
{
    int j,k;
    int res = 0;

    for ( j = 0 ; j < tspaceDim() ; j++ )
    {
        k = (i*tspaceDim())+j;

        res |= Q.setCweightfuzz(k,xCweight);
    }

    return res;
}

int SVM_Vector_Mredbin::setepsweight(int i, double xepsweight)
{
    int j,k;
    int res = 0;

    for ( j = 0 ; j < tspaceDim() ; j++ )
    {
        k = (i*tspaceDim())+j;

        res |= Q.setepsweight(k,xepsweight);
    }

    return res;
}

int SVM_Vector_Mredbin::setd(const Vector<int> &j, const Vector<int> &d)
{
    NiceAssert( d.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
        {
            res |= setd(j(i),d(i));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::sety(const Vector<int> &j, const Vector<Vector<double> > &z)
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

int SVM_Vector_Mredbin::setCweight(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
        {
            res |= setCweight(j(i),xCweight(i));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::setCweightfuzz(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
        {
            res |= setCweightfuzz(j(i),xCweight(i));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::setepsweight(const Vector<int> &j, const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; i++ )
        {
            res |= setepsweight(j(i),xepsweight(i));
        }
    }

    return res;
}


int SVM_Vector_Mredbin::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == N() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            res |= setd(i,d(i));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::sety(const Vector<Vector<double> > &z)
{
    NiceAssert( z.size() == N() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            res |= sety(i,z(i));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::setCweight(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == N() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            res |= setCweight(i,xCweight(i));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::setCweightfuzz(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == N() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            res |= setCweightfuzz(i,xCweight(i));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::setepsweight(const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == N() );

    int res = 0;

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; i++ )
        {
            res |= setepsweight(i,xepsweight(i));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::settspaceDim(int newdim)
{
    while ( tspaceDim() < newdim )
    {
        addtspaceFeat(tspaceDim());
    }

    while ( tspaceDim() > newdim )
    {
        removetspaceFeat(tspaceDim()-1);
    }

    return 1;
}

int SVM_Vector_Mredbin::addtspaceFeat(int iii)
{
    NiceAssert( iii >= 0 );
    NiceAssert( iii <= tspaceDim() );

    int newtspacedim = tspaceDim()+1;

    dbiasA.add(iii);
    dbiasA("&",iii) = 0.0;

    Gpn.addCol(iii);

    if ( Gpn.numRows() )
    {
        int i;

        for ( i = 0 ; i < Gpn.numRows() ; i++ )
        {
            Gpn("&",i,iii) = 0.0;
        }
    }

    Q.setbiasdim(newtspacedim+1,iii,dbiasA(iii));

    if ( N() )
    {
        int i,k;

        retVector<double> tmpva;
        retVector<int>    tmpvb;

        for ( i = 0 ; i < N() ; i++ )
        {
            interlace("&",i+1,1,N()-1,tmpvb) += 1;

            traintarg("&",i).add(iii);
            traintarg("&",i)("&",iii) = 0.0;

            gentype yn;
            yn = traintarg(i);
            SVM_Generic::sety(i,yn);

            dalphaA("&",i).add(iii);
            dalphaA("&",i)("&",iii) = 0.0;

            SparseVector<gentype> dummy;

            ixskip  = i;
            ixskipc = 1;

            k = (i*newtspacedim)+iii;

            Gpn.addRow(k);
            Gpn("&",k,tmpva) = 0.0;
            Gpn("&",k,iii) = xisrankorgrad(i) ? 0.0 : 1.0;

            Q.qaddTrainingVector(k,traintarg(i)(iii),dummy,Cweight()(i),epsweight()(i),xisclass(i,d()(i),iii));

            ixskip  = -1;
            ixskipc = -1;
        }
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

int SVM_Vector_Mredbin::removetspaceFeat(int iii)
{
    NiceAssert( iii >= 0 );
    NiceAssert( iii < tspaceDim() );

    int newtspacedim = tspaceDim()-1;

    dbiasA.remove(iii);

    Q.setbiasdim(newtspacedim+1,-1,dbiasA(iii),iii);

    Gpn.removeCol(iii);

    if ( N() )
    {
        int i,k;

        retVector<int> tmpva;

        for ( i = 0 ; i < N() ; i++ )
        {
            interlace("&",i+1,1,N()-1,tmpva) -= 1;

            traintarg("&",i).remove(iii);
            dalphaA("&",i).remove(iii);

            gentype yn;
            yn = traintarg(i);
            SVM_Generic::sety(i,yn);

            ixskip  = i;
            ixskipc = -1;

            k = (i*newtspacedim)+iii;

            Gpn.removeRow(k);

            Q.ML_Base::removeTrainingVector(k);

            ixskip  = -1;
            ixskipc = -1;
        }
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return 1;
}

int SVM_Vector_Mredbin::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    int j,k;

    int res = setd(i,0);

    for ( j = tspaceDim()-1 ; j >= 0 ; j-- )
    {
        ixsplit = i;
        iqsplit = j+1; // Must act like change occurs *after* call

        k = (i*tspaceDim())+j;

        // Gpn is taken care of by Q.removeTrainingVector call

        res |= Q.ML_Base::removeTrainingVector(k);

        ixsplit = -1;
        iqsplit = -1;
    }

    retVector<int> tmpva;

    interlace.remove(i);
    interlace("&",i+1,1,N()-1,tmpva) -= tspaceDim();

    res |= SVM_Generic::removeTrainingVector(i,y,x);
    
    traintarg.remove(i);
    xalphaState.remove(i);
    onedvec.remove(i);
    dalphaA.remove(i);

    res |= Q.ML_Base::removeTrainingVector(i);

    aNZ--;
    aNC--;
    aN--;

    return res;
}

int SVM_Vector_Mredbin::inintrain(int &res, svmvolatile int &killSwitch)
{
    return Q.train(res,killSwitch);
}

int SVM_Vector_Mredbin::intrain(int &res, svmvolatile int &killSwitch)
{
    return inintrain(res,killSwitch);
}

int SVM_Vector_Mredbin::train(int &res,svmvolatile int &killSwitch)
{
    int result = intrain(res,killSwitch);

    updateAlpha();
    updateBias();

    return result;
}

int SVM_Vector_Mredbin::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int unusedvar = 0;
    int tempresh = 0;
    Vector<double> tempresg;

    tempresh = gTrainingVector(tempresg,unusedvar,i,retaltg,pxyprodi);
    resh = tempresg; // processed output of regressor is scalar
    resg = tempresg;

    return tempresh;
}

int SVM_Vector_Mredbin::gTrainingVector(Vector<double>  &res, int &dummy, int i, int raw, gentype ***pxyprodi) const
{
    (void) pxyprodi;

    int dtv = 0;

    (void) raw;
    dummy = 0;

    if ( i >= 0 )
    {
        int j,k;

        res.resize(tspaceDim());

        for ( j = 0 ; j < tspaceDim() ; j++ )
        {
            k = (i*tspaceDim())+j;

            gentype tempsomeh,tempsomeg;

            //Q.gTrainingVector(res("&",j),dummy,k,raw);
            Q.ghTrainingVector(tempsomeh,tempsomeg,k,raw);

            res("&",j) = (double) tempsomeg;
        }
    }

    else if ( ( dtv = xtang(i) & 7 ) )
    {
        res.resize(tspaceDim()) = 0.0;

        if ( ( dtv > 0 ) && N() )
        {
            int j;
            Matrix<double>  Kij;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( alphaState()(j) )
                {
                    K2(tspaceDim(),Kij,i,j);
                    res += (Kij*dalphaA(j));
                }
            }
        }
    }

    else
    {
        res.resize(tspaceDim()) = dbiasA;

        if ( N() )
        {
            int j;
            Matrix<double>  Kij;

            for ( j = 0 ; j < N() ; j++ )
            {
                if ( alphaState()(j) )
                {
                    K2(tspaceDim(),Kij,i,j);
                    res += (Kij*dalphaA(j));
                }
            }
        }
    }

    return 0;
}

int SVM_Vector_Mredbin::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
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

    return SVM_Vector_Mredbin::addTrainingVector(i,zd,x,Cweigh,epsweigh,2);
}

int SVM_Vector_Mredbin::qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
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

    return SVM_Vector_Mredbin::qaddTrainingVector(i,zd,x,Cweigh,epsweigh,2);
}

int SVM_Vector_Mredbin::addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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
            res |= SVM_Vector_Mredbin::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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
            res |= SVM_Vector_Mredbin::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::addTrainingVector(int i, const Vector<double>  &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( z.size() == tspaceDim() );

    gentype yn;
    yn = z;
    int res = SVM_Generic::addTrainingVector(i,yn,x);
    res |= qtaddTrainingVector(i,z,Cweigh,epsweigh,d);

    return res;
}

int SVM_Vector_Mredbin::qaddTrainingVector(int i, const Vector<double>  &z, SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( z.size() == tspaceDim() );

    gentype yn;
    yn = z;
    int res = SVM_Generic::qaddTrainingVector(i,yn,x);
    res |= qtaddTrainingVector(i,z,Cweigh,epsweigh,d);

    return res;
}

int SVM_Vector_Mredbin::addTrainingVector (int i, const Vector<Vector<double> > &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
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
            res |= SVM_Vector_Mredbin::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::qaddTrainingVector(int i, const Vector<Vector<double> > &z,       Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
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
            res |= SVM_Vector_Mredbin::qaddTrainingVector(i+j,z(j),xx("&",j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    return res;
}

int SVM_Vector_Mredbin::setFixedBias(const Vector<double>  &newbias)
{
    NiceAssert( newbias.size() == tspaceDim() );

    int i;

    for ( i = 0 ; i < tspaceDim() ; i++ )
    {
        Q.setFixedBias(i,newbias(i));
    }

    return 1;
}

void SVM_Vector_Mredbin::updateBias(void)
{
    int j;

    for ( j = 0 ; j < tspaceDim() ; j++ )
    {
        dbiasA("&",j) = Q.biasVMulti(j);
    }

    SVM_Generic::basesetbias(biasV());

    return;
}

void SVM_Vector_Mredbin::updateAlpha(void)
{
    int i,j,k;

    if ( N() )
    {
        for ( i = 0 ; i < N() ; i++ )
        {
            xalphaState("&",i) = 0;

            for ( j = 0 ; j < tspaceDim() ; j++ )
            {
                k = (i*tspaceDim())+j;

                dalphaA("&",i)("&",j) = (Q.alphaR())(k);

                if (      ( (Q.alphaState())(k) == -1 ) || ( (Q.alphaState())(k) == +1 ) )
                {
                    xalphaState("&",i) = 1;
                }

                else if ( ( (Q.alphaState())(k) == -2 ) || ( (Q.alphaState())(k) == +2 ) )
                {
                    xalphaState("&",i) = 2;
                }
            }

            if ( xalphaState(i) == 0 )
            {
                aNZ++;
                aNC++;
            }

            else if ( xalphaState(i) == 1 )
            {
                aNS++;
                aNF++;
            }

            else
            {
                aNS++;
                aNC++;
            }
        }
    }

    SVM_Generic::basesetAlphaBiasFromAlphaBiasV();

    return;
}

void SVM_Vector_Mredbin::fixKcallback(void)
{
    (Q.getKernel_unsafe()).setType(800);
    (Q.getKernel_unsafe()).setAltCall(Kcall.MLid());

    Q.resetKernel();

    return;
}

int SVM_Vector_Mredbin::qtaddTrainingVector(int i, const Vector<double>  &z, double Cweigh, double epsweigh, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( z.size() == tspaceDim() );

    int res = 0;

    retVector<double> tmpva;
    retVector<int>    tmpvb;

    interlace.add(i);
    interlace("&",i) = i*tspaceDim();
    interlace("&",i+1,1,N()-1,tmpvb) += tspaceDim();

    traintarg.add(i);   traintarg("&",i)   = z;
    xalphaState.add(i); xalphaState("&",i) = 0;
    onedvec.add(i);     onedvec("&",i)     = 1.0;
    dalphaA.add(i);     dalphaA("&",i)     = 0.0;

    dalphaA("&",i).resize(tspaceDim());

    int j,k;
    SparseVector<gentype> dummy;

    for ( j = 0 ; j < tspaceDim() ; j++ )
    {
        ixsplit = i;
        iqsplit = j+1; // Must act like change occurs *before* call

        k = (i*tspaceDim())+j;

        Gpn.addRow(k);
        Gpn("&",k,tmpva) = 0.0;
        Gpn("&",k,j) = xisrankorgrad(i) ? 0.0 : 1.0;

        res |= Q.qaddTrainingVector(k,z(j),dummy,Cweigh,epsweigh,xisclass(i,d,j));

        ixsplit = -1;
        iqsplit = -1;
    }

    aN++;
    aNZ++;
    aNC++;

    SVM_Generic::basesetalpha(i,dalphaA(i));

    return res;
}

void SVM_Vector_Mredbin::locnaivesetGpnExt(void)
{
    Q.naivesetGpnExt(&Gpn);

    return;
}





std::ostream &SVM_Vector_Mredbin::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector Mredbin SVM\n\n";

    SVM_Generic::printstream(output,dep+1);

    repPrint(output,'>',dep) << "Targets:               " << traintarg   << "\n";
    repPrint(output,'>',dep) << "Alpha:                 " << dalphaA     << "\n";
    repPrint(output,'>',dep) << "Bias:                  " << dbiasA      << "\n";
    repPrint(output,'>',dep) << "Alpha state:           " << xalphaState << "\n";
    repPrint(output,'>',dep) << "1:                     " << onedvec     << "\n";
    repPrint(output,'>',dep) << "Interlacing:           " << interlace   << "\n";
    repPrint(output,'>',dep) << "Gpn:                   " << Gpn         << "\n\n";

    repPrint(output,'>',dep) << "*********************************************************************\n";
    repPrint(output,'>',dep) << "Optimisation state:    ";
    Q.printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "#####################################################################\n";

    return output;
}

std::istream &SVM_Vector_Mredbin::inputstream(std::istream &input)
{
    wait_dummy dummy;

    SVM_Generic::inputstream(input);

    input >> dummy; input >> traintarg;
    input >> dummy; input >> dalphaA;
    input >> dummy; input >> dbiasA;
    input >> dummy; input >> xalphaState;
    input >> dummy; input >> onedvec;
    input >> dummy; input >> interlace;
    input >> dummy; input >> Gpn;

    input >> dummy; Q.inputstream(input);

    ixsplit = -1;
    iqsplit = -1;
    ixskip  = -1;
    ixskipc = -1;

    locnaivesetGpnExt();
    fixKcallback();

    return input;
}

int SVM_Vector_Mredbin::prealloc(int expectedN)
{
    interlace.prealloc(expectedN);
    traintarg.prealloc(expectedN);
    xalphaState.prealloc(expectedN);
    onedvec.prealloc(expectedN);
    dalphaA.prealloc(expectedN);
    Gpn.prealloc(expectedN,Gpn.numCols());
    SVM_Generic::prealloc(expectedN);

    Q.prealloc(tspaceDim() ? expectedN*tspaceDim() : 1);

    return 0;
}

int SVM_Vector_Mredbin::preallocsize(void) const
{
    return SVM_Generic::preallocsize();
}

int SVM_Vector_Mredbin::randomise(double sparsity)
{
    NiceAssert( sparsity >= 0 );
    NiceAssert( sparsity <= 1 );

    int res = Q.randomise(sparsity);

    updateAlpha();
    updateBias();

    return res;
}


