
//
// Similarity Learning SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_simlrn.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>



SVM_SimLrn::SVM_SimLrn() : SVM_Scalar()
{
    setaltx(NULL);

    xtheta   = DEFAULT_THETA;
    xsimnorm = 1;

    SVM_Scalar::setFixedBias();
    //SVM_Scalar::setPosBias();

    return;
}

SVM_SimLrn::SVM_SimLrn(const SVM_SimLrn &src) : SVM_Scalar()
{
    setaltx(NULL);

    xtheta   = DEFAULT_THETA;
    xsimnorm = 1;

    SVM_Scalar::setFixedBias();
    //SVM_Scalar::setPosBias();

    assign(src,0);

    return;
}

SVM_SimLrn::SVM_SimLrn(const SVM_SimLrn &src, const ML_Base *xsrc) : SVM_Scalar()
{
    setaltx(xsrc);

    xtheta   = DEFAULT_THETA;
    xsimnorm = 1;

    SVM_Scalar::setFixedBias();
    //SVM_Scalar::setPosBias();

    assign(src,1);

    return;
}

SVM_SimLrn::~SVM_SimLrn()
{
    return;
}

std::ostream &SVM_SimLrn::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Similarity Learning SVM\n\n";

    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base SVR:                        ";
    SVM_Scalar::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_SimLrn::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy;
    SVM_Scalar::inputstream(input);

    return input;
}


void evalKSVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    NiceAssert( owner );

    SVM_SimLrn &realOwner = *((SVM_SimLrn *) owner);
    SVM_Scalar &realParent = static_cast<SVM_Scalar &>(realOwner);

    double theta = realOwner.xtheta;

    res = (realParent.kerncache).getval(i,j) + ( ( i == j ) ? theta : 0 );

    return;
}

void evalxySVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    NiceAssert( owner );

    SVM_SimLrn &realOwner = *((SVM_SimLrn *) owner);
    SVM_Scalar &realParent = static_cast<SVM_Scalar &>(realOwner);

    res = (realParent.xycache).getval(i,j);

    return;
}

void evalSigmaSVM_SimLrn(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    NiceAssert( owner );

    SVM_SimLrn &realOwner = *((SVM_SimLrn *) owner);

    res = (*(realOwner.GpOuter))(i,i)+(*(realOwner.GpOuter))(j,j)-(2.0*(*(realOwner.GpOuter))(i,j));

    return;
}

void SVM_SimLrn::K0xfer(gentype &res, int &minmaxind, int typeis,
                       int xdim, int densetype, int resmode, int mlid) const
{
    SVM_Scalar::K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid);

    return;
}

void SVM_SimLrn::K1xfer(gentype &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    SVM_Scalar::K1xfer(res,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

    if ( simnorm() )
    {
        gentype tempa;

        SVM_Scalar::K1xfer(tempa,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

        res /= tempa;
    }

    return;
}

void SVM_SimLrn::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
{
//FIXME: gradients don't work here

    SVM_Scalar::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

    if ( simnorm() )
    {
        gentype tempa,tempb,dummy;

        SVM_Scalar::K2xfer(dummy,dummy,tempa,minmaxind,typeis,xa,xa,xainfo,xainfo,ia,ia,xdim,densetype,resmode,mlid);
        SVM_Scalar::K2xfer(dummy,dummy,tempb,minmaxind,typeis,xb,xb,xbinfo,xbinfo,ib,ib,xdim,densetype,resmode,mlid);

        OP_sqrt(tempa);
        OP_sqrt(tempb);

        res /= tempa;
        res /= tempb;
    }

    return;
}

void SVM_SimLrn::K3xfer(gentype &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    SVM_Scalar::K3xfer(res,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

    if ( simnorm() )
    {
        gentype tempa,tempb,tempc;

        SVM_Scalar::K3xfer(tempa,minmaxind,typeis,xa,xa,xa,xainfo,xainfo,xainfo,ia,ia,ia,xdim,densetype,resmode,mlid);
        SVM_Scalar::K3xfer(tempb,minmaxind,typeis,xb,xb,xb,xbinfo,xbinfo,xbinfo,ib,ib,ib,xdim,densetype,resmode,mlid);
        SVM_Scalar::K3xfer(tempc,minmaxind,typeis,xc,xc,xc,xcinfo,xcinfo,xcinfo,ic,ic,ic,xdim,densetype,resmode,mlid);

        gentype oneonm(1.0/3.0);

        res /= pow(tempa,oneonm);
        res /= pow(tempb,oneonm);
        res /= pow(tempc,oneonm);
    }

    return;
}

void SVM_SimLrn::K4xfer(gentype &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const
{
    SVM_Scalar::K4xfer(res,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

    if ( simnorm() )
    {
        gentype tempa,tempb,tempc,tempd;

        SVM_Scalar::K4xfer(tempa,minmaxind,typeis,xa,xa,xa,xa,xainfo,xainfo,xainfo,xainfo,ia,ia,ia,ia,xdim,densetype,resmode,mlid);
        SVM_Scalar::K4xfer(tempb,minmaxind,typeis,xb,xb,xb,xb,xbinfo,xbinfo,xbinfo,xbinfo,ib,ib,ib,ib,xdim,densetype,resmode,mlid);
        SVM_Scalar::K4xfer(tempc,minmaxind,typeis,xc,xc,xc,xc,xcinfo,xcinfo,xcinfo,xcinfo,ic,ic,ic,ic,xdim,densetype,resmode,mlid);
        SVM_Scalar::K4xfer(tempd,minmaxind,typeis,xd,xd,xd,xd,xdinfo,xdinfo,xdinfo,xdinfo,id,id,id,id,xdim,densetype,resmode,mlid);

        OP_sqrt(tempa);
        OP_sqrt(tempb);
        OP_sqrt(tempc);
        OP_sqrt(tempd);

        OP_sqrt(tempa);
        OP_sqrt(tempb);
        OP_sqrt(tempc);
        OP_sqrt(tempd);

        res /= tempa;
        res /= tempb;
        res /= tempc;
        res /= tempd;
    }

    return;
}

void SVM_SimLrn::Kmxfer(gentype &res, int &minmaxind, int typeis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xinfo,
                        Vector<int> &i,
                        int xdim, int m, int densetype, int resmode, int mlid) const
{
    if ( ( m == 0 ) || ( m == 1 ) || ( m == 2 ) || ( m == 3 ) || ( m == 4 ) )
    {
        kernPrecursor::Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid);
        return;
    }

    SVM_Scalar::Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

    if ( simnorm() )
    {
        int ii;
        gentype temp;
        gentype invm(1.0/m);

        Vector<const SparseVector<gentype> *> y(x);
        Vector<const vecInfo *> yinfo(xinfo);
        Vector<int> j(i);

        for ( ii = 0 ; ii < m ; ii++ )
        {
            y = x(ii);
            yinfo = xinfo(ii);
            j = i(ii);

            SVM_Scalar::Kmxfer(temp,minmaxind,typeis,y,yinfo,j,xdim,m,densetype,resmode,mlid);

            res /= epow(temp,invm);
        }
    }

    return;
}

void SVM_SimLrn::K0xfer(double &res, int &minmaxind, int typeis,
                       int xdim, int densetype, int resmode, int mlid) const
{
    SVM_Scalar::K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid);

    return;
}

void SVM_SimLrn::K1xfer(double &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    SVM_Scalar::K1xfer(res,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

    if ( simnorm() )
    {
        double tempa;

        SVM_Scalar::K1xfer(tempa,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

        res /= tempa;
    }

    return;
}

void SVM_SimLrn::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
{
//FIXME: gradients don't work here

    SVM_Scalar::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

    if ( simnorm() )
    {
        double tempa,tempb,dummy;

        SVM_Scalar::K2xfer(dummy,dummy,tempa,minmaxind,typeis,xa,xa,xainfo,xainfo,ia,ia,xdim,densetype,resmode,mlid);
        SVM_Scalar::K2xfer(dummy,dummy,tempb,minmaxind,typeis,xb,xb,xbinfo,xbinfo,ib,ib,xdim,densetype,resmode,mlid);

        OP_sqrt(tempa);
        OP_sqrt(tempb);

        res /= tempa;
        res /= tempb;
    }

    return;
}

void SVM_SimLrn::K3xfer(double &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    SVM_Scalar::K3xfer(res,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

    if ( simnorm() )
    {
        double tempa,tempb,tempc;

        SVM_Scalar::K3xfer(tempa,minmaxind,typeis,xa,xa,xa,xainfo,xainfo,xainfo,ia,ia,ia,xdim,densetype,resmode,mlid);
        SVM_Scalar::K3xfer(tempb,minmaxind,typeis,xb,xb,xb,xbinfo,xbinfo,xbinfo,ib,ib,ib,xdim,densetype,resmode,mlid);
        SVM_Scalar::K3xfer(tempc,minmaxind,typeis,xc,xc,xc,xcinfo,xcinfo,xcinfo,ic,ic,ic,xdim,densetype,resmode,mlid);

        res /= pow(tempa,1.0/3.0);
        res /= pow(tempb,1.0/3.0);
        res /= pow(tempc,1.0/3.0);
    }

    return;
}

void SVM_SimLrn::K4xfer(double &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const
{
    SVM_Scalar::K4xfer(res,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

    if ( simnorm() )
    {
        double tempa,tempb,tempc,tempd;

        SVM_Scalar::K4xfer(tempa,minmaxind,typeis,xa,xa,xa,xa,xainfo,xainfo,xainfo,xainfo,ia,ia,ia,ia,xdim,densetype,resmode,mlid);
        SVM_Scalar::K4xfer(tempb,minmaxind,typeis,xb,xb,xb,xb,xbinfo,xbinfo,xbinfo,xbinfo,ib,ib,ib,ib,xdim,densetype,resmode,mlid);
        SVM_Scalar::K4xfer(tempc,minmaxind,typeis,xc,xc,xc,xc,xcinfo,xcinfo,xcinfo,xcinfo,ic,ic,ic,ic,xdim,densetype,resmode,mlid);
        SVM_Scalar::K4xfer(tempd,minmaxind,typeis,xd,xd,xd,xd,xdinfo,xdinfo,xdinfo,xdinfo,id,id,id,id,xdim,densetype,resmode,mlid);

        OP_sqrt(tempa);
        OP_sqrt(tempb);
        OP_sqrt(tempc);
        OP_sqrt(tempd);

        OP_sqrt(tempa);
        OP_sqrt(tempb);
        OP_sqrt(tempc);
        OP_sqrt(tempd);

        res /= tempa;
        res /= tempb;
        res /= tempc;
        res /= tempd;
    }

    return;
}

void SVM_SimLrn::Kmxfer(double &res, int &minmaxind, int typeis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xinfo,
                        Vector<int> &i,
                        int xdim, int m, int densetype, int resmode, int mlid) const
{
    if ( ( m == 0 ) || ( m == 1 ) || ( m == 2 ) || ( m == 3 ) || ( m == 4 ) )
    {
        kernPrecursor::Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid);
        return;
    }

    SVM_Scalar::Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

    if ( simnorm() )
    {
        int ii;
        double temp;
        double invm = 1.0/m;

        Vector<const SparseVector<gentype> *> y(x);
        Vector<const vecInfo *> yinfo(xinfo);
        Vector<int> j(i);

        for ( ii = 0 ; ii < m ; ii++ )
        {
            y = x(ii);
            yinfo = xinfo(ii);
            j = i(ii);

            SVM_Scalar::Kmxfer(temp,minmaxind,typeis,y,yinfo,j,xdim,m,densetype,resmode,mlid);

            res /= pow(temp,invm);
         }
    }

    return;
}





















int SVM_SimLrn::train(int &res, svmvolatile int &killSwitch)
{
    int intres = 0;

    res = 0;

    // Work out size of underlying training set - basically count the number of "not far" vectors

    int i,j;
    int Nb = 0;

    for ( i = 0 ; i < N() ; i++ )
    {
        const SparseVector<gentype> &xref = x()(i);

        if ( xref.isnofaroffindpresent() )
        {
            Nb++;
        }
    }

    Vector<int> ia(N());
    Vector<int> ib(N());

    ia = -1;
    ib = -1;

    Vector<int> ja(N());
    Vector<int> jb(N());

    ja = -1;
    jb = -1;

//errstream() << "phantomx 0 Nb = " << Nb << "\n";
    for ( i = 0 ; i < N() ; i++ )
    {
        const SparseVector<gentype> &xref = x()(i);

        NiceAssert( !(xref.isfarfarfarindpresent(2)) );
        NiceAssert( !(xref.isfarfarfarindpresent(3)) );
        NiceAssert( !(xref.isfarfarfarindpresent(4)) );

        if ( xref.isfarfarfarindpresent(zeroint()) && xref.fff(0).isValVector() )
        {
            const Vector<gentype> &xvecref = (const Vector<gentype> &) xref.fff(0);

            NiceAssert( xvecref.size() == 2 );

            ia("&",i) = (int) xvecref(zeroint());
            ib("&",i) = (int) xvecref(1);

            NiceAssert( ia(i) >= 0 );
            NiceAssert( ib(i) >= 0 );

            NiceAssert( ia(i) < Nb );
            NiceAssert( ib(i) < Nb );
        }

        if ( xref.isfarfarfarindpresent(1) && xref.fff(1).isValVector() )
        {
            const Vector<gentype> &xvecref = (const Vector<gentype> &) xref.fff(1);

            NiceAssert( xvecref.size() == 2 );

            ja("&",i) = (int) xvecref(zeroint());
            jb("&",i) = (int) xvecref(1);

            NiceAssert( ja(i) >= 0 );
            NiceAssert( jb(i) >= 0 );

            NiceAssert( ja(i) < Nb );
            NiceAssert( jb(i) < Nb );
        }
    }

//errstream() << "phantomx 1 ia = " << ia << "\n";
//errstream() << "phantomx 2 ib = " << ib << "\n";
//errstream() << "phantomx 1 ja = " << ja << "\n";
//errstream() << "phantomx 2 jb = " << jb << "\n";
    Matrix<double> beta(Nb,Nb);
    Matrix<double> R(Nb,Nb);
    Matrix<double> dR(Nb,Nb);

    Vector<double> yOuter(N());
    Vector<double> yInner(N());

    beta = 0.0;
    R    = 0.0;
    dR   = 0.0;

    for ( i = 0 ; i < N() ; i++ )
    {
        yOuter("&",i) = (double) y()(i);
        yInner("&",i) = (double) y()(i);
    }

    MEMNEW(xycachesim,   Kcache<double>);
    MEMNEW(kerncachesim, Kcache<double>);
    MEMNEW(sigmacachesim,Kcache<double>);

    (*xycachesim).reset(0,&evalxySVM_SimLrn,(void *) this);
    (*xycachesim).setmemsize(memsize(),N());

    (*kerncachesim).reset(0,&evalKSVM_SimLrn,(void *) this);
    (*kerncachesim).setmemsize(memsize(),N());
                          
    (*sigmacachesim).reset(0,&evalSigmaSVM_SimLrn,(void *) this);
    (*sigmacachesim).setmemsize(memsize(),N());

    MEMNEW(xyOuter,     Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) xycachesim   ,N(),N()));
    MEMNEW(GpOuter,     Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) kerncachesim ,N(),N()));
    MEMNEW(GpSigmaOuter,Matrix<double>(Kcache_celm_double,Kcache_crow_double,(void *) sigmacachesim,N(),N()));

    for ( i = 0 ; i < N() ; i++ )
    {
        (*xycachesim).add(i);
        (*kerncachesim).add(i);
        (*sigmacachesim).add(i);
    }

    Matrix<int> betascale(Nb,Nb);

    betascale = 0;

    for ( i = 0 ; i < N() ; i++ )
    {
        if ( ( ia(i) >= 0 ) && ( ib(i) >= 0 ) )
        {
            betascale("&",ia(i),ib(i))++;
            betascale("&",ib(i),ia(i))++;
        }

        if ( ( ja(i) >= 0 ) && ( jb(i) >= 0 ) )
        {
            betascale("&",ja(i),jb(i))++;
            betascale("&",jb(i),ja(i))++;
        }
    }

    for ( i = 0 ; i < Nb ; i++ )
    {
        for ( j = 0 ; j < Nb ; j++ )
        {
            if ( !betascale("&",i,j) )
            {
                betascale("&",i,j) = 1;
            }
        }
    }







    Vector<double> fv1;
    Vector<double> fv2;
    Matrix<double> fm1;

    double dstepmag;

    int isopt = 0;

//errstream() << "phantomx 3 GpOuter = " << *GpOuter << "\n";
    setGp(GpOuter,GpSigmaOuter,xyOuter);

    while ( !isopt )
    {
//errstream() << "phantomx 5\n";
        kerncachesim->padCol(4);
        xycachesim->padCol(4);
        sigmacachesim->padCol(4);

        //(*xyOuter).resize(N(),N()+2);
        //(*GpOuter).resize(N(),N()+2);
        //(*GpSigmaOuter).resize(N(),N()+2);

        intres = inintrain(killSwitch);

        kerncachesim->padCol(0);
        xycachesim->padCol(0);
        sigmacachesim->padCol(0);

        //(*xyOuter).resize(N(),N());
        //(*GpOuter).resize(N(),N());
        //(*GpSigmaOuter).resize(N(),N());

        // Fill beta matrix
        //
        // Note that we must ensure symmetry and allow
        // for possible missing values.

        beta = 0.0;

        for ( i = 0 ; i < N() ; i++ )
        {
            if ( ( ia(i) >= 0 ) && ( ib(i) >= 0 ) )
            {
                beta("&",ia(i),ib(i)) += alphaR()(i);
                beta("&",ib(i),ia(i)) += alphaR()(i);
            }

            if ( ( ja(i) >= 0 ) && ( jb(i) >= 0 ) )
            {
                beta("&",ja(i),jb(i)) -= alphaR()(i);
                beta("&",jb(i),ja(i)) -= alphaR()(i);
            }
        }

        for ( i = 0 ; i < Nb ; i++ )
        {
            for ( j = 0 ; j < Nb ; j++ )
            {
                beta("&",i,j) /= betascale("&",i,j);
            }
        }

        // Project to nearest positive semi-definite beta matrix

//errstream() << "phantomxyz 5 alpha = " << alphaR() << "\n";
//errstream() << "phantomxyz 5 beta = " << beta << "\n";
        beta.projpsd(dR,fv1,fm1,fv2);

        // Work out R step

        dR -= R;
        dR *= 0.7; //FIXME lr;
//errstream() << "phantomxyz 5 dR = " << dR << "\n";

        // Take R step

        R += dR;
//errstream() << "phantomxyz 5 R = " << R << "\n";

        // Backmodify yOuter for new beta value

        for ( i = 0 ; i < N() ; i++ )
        {
            yOuter("&",i) = yInner("&",i);

            if ( ( ia(i) >= 0 ) && ( ib(i) >= 0 ) )
            {
                yOuter("&",i) += xtheta*R(ia(i),ib(i));
            }

            if ( ( ja(i) >= 0 ) && ( jb(i) >= 0 ) )
            {
                yOuter("&",i) -= xtheta*R(ja(i),jb(i));
            }
        }

        sety(yOuter);

        dstepmag = absF(dR);

        errstream() << ".|" << dstepmag << "|.";

        isopt = ( dstepmag <= 1e-2 ) ? 1 : 0; //FIXME
    }

    // Need to ensure that alpha follows beta

    Vector<double> alphap(N());

    alphap = 0.0;

    for ( i = 0 ; i < N() ; i++ )
    {
        alphap("&",i) = 0.0;

        if ( ( ia(i) >= 0 ) && ( ib(i) >= 0 ) )
        {
            alphap("&",i) += beta(ia(i),ib(i));
        }

        if ( ( ja(i) >= 0 ) && ( jb(i) >= 0 ) )
        {
            alphap("&",i) -= beta(ja(i),jb(i));
        }
    }

    setAlphaR(alphap);
    sety(yInner);
    SVM_Scalar::setGp(NULL,NULL,NULL);
//errstream() << "phantomx 8: " << beta << "\n";
//errstream() << "phantomx 8: " << R << "\n";






    MEMDEL(xyOuter);
    MEMDEL(GpOuter);
    MEMDEL(GpSigmaOuter);

    MEMDEL(xycachesim);
    MEMDEL(kerncachesim);
    MEMDEL(sigmacachesim);

    return intres;
}





























