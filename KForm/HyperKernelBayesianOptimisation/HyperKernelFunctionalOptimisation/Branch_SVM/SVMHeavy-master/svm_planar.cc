
//
// Planar SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_planar.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>






const gentype &VVcallbackdef(gentype &res, const gentype &kval, const ML_Base &caller, int iplanr, int jplanr, int iplan, int jplan, const gentype &xb, const gentype &yb, int defbasis);
const gentype &VVcallbackdef(gentype &res, const gentype &kval, const ML_Base &caller, int iplanr, int jplanr, int iplan, int jplan, const gentype &xb, const gentype &yb, int defbasis)
{
    Vector<int> iokr(2);
    Vector<int> iok(2);
    Vector<const gentype *> xalt(2);

    int z = 0;

    iokr("&",z) = iplanr;
    iokr("&",1) = jplanr;

    iok("&",z) = iplan;
    iok("&",1) = jplan;

    xalt("&",z) = &xb;
    xalt("&",1) = &yb;

    return VVcallbackdef(res,2,kval,caller,iokr,iok,xalt,defbasis);
}






// Constructors and destructors

SVM_Planar::SVM_Planar() : SVM_Scalar()
{
    ghEvalFull = 0;

    thisthis = this;
    thisthisthis = &thisthis;

    ML_Base::VVcallback = &VVcallbacknon;
    cheatscache.ML_Base::VVcallback = &VVcallbacknon;

    bdim    = 0;
    midadd  = 0;
    defproj = 0;

    setaltx(NULL);

    inGpn.resize(0,1);

    SVM_Scalar::setGpnExt(NULL,&inGpn);
    SVM_Scalar::setbiasdim(NbasisVV()+1);
    inGpn.resize(0,NbasisVV());

    return;
}

SVM_Planar::SVM_Planar(const SVM_Planar &src) : SVM_Scalar()
{
    ghEvalFull = 0;

    thisthis = this;
    thisthisthis = &thisthis;

    ML_Base::VVcallback = &VVcallbacknon;
    cheatscache.ML_Base::VVcallback = &VVcallbacknon;

    bdim    = 0;
    midadd  = 0;
    defproj = 0;

    setaltx(NULL);

    inGpn.resize(0,1);

    SVM_Scalar::setGpnExt(NULL,&inGpn);
    SVM_Scalar::setbiasdim(NbasisVV()+1);
    inGpn.resize(0,NbasisVV());

    assign(src,0);

    return;
}

SVM_Planar::SVM_Planar(const SVM_Planar &src, const ML_Base *xsrc) : SVM_Scalar()
{
    ghEvalFull = 0;

    thisthis = this;
    thisthisthis = &thisthis;

    ML_Base::VVcallback = &VVcallbacknon;
    cheatscache.ML_Base::VVcallback = &VVcallbacknon;

    bdim    = 0;
    midadd  = 0;
    defproj = 0;

    setaltx(xsrc);

    inGpn.resize(0,1);

    SVM_Scalar::setGpnExt(NULL,&inGpn);
    SVM_Scalar::setbiasdim(NbasisVV()+1);
    inGpn.resize(0,NbasisVV());

    assign(src,1);

    return;
}

SVM_Planar::~SVM_Planar()
{
    return;
}















void SVM_Planar::setmemsize(int memsize)
{
    cheatscache.setmemsize(memsize/2);
    SVM_Scalar::setmemsize(memsize/2);

    return;
}

int SVM_Planar::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    NiceAssert( onlyChangeRowI >= -1 );
    NiceAssert( onlyChangeRowI < SVM_Planar::N() );

    int res = 0;

    res |= cheatscache.setKernel(SVM_Planar::getKernel()); // Note we use setKernel, *not* resetKernel, and getKernel only changes ML_Base, not cheat branch
    res |= SVM_Scalar::resetKernel(modind,onlyChangeRowI,updateInfo);

    return res;
}

int SVM_Planar::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    NiceAssert( onlyChangeRowI >= -1 );
    NiceAssert( onlyChangeRowI < SVM_Planar::N() );

    int res = 0;

    res |= cheatscache.setKernel(xkernel,modind,onlyChangeRowI);
    res |= SVM_Scalar::setKernel(xkernel,modind,onlyChangeRowI);

    return res;
}

std::ostream &SVM_Planar::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Planar SVM\n\n";

    repPrint(output,'>',dep) << "override Gpn:  " << inGpn          << "\n";
    repPrint(output,'>',dep) << "local d:       " << locd           << "\n";
    repPrint(output,'>',dep) << "y basis:       " << locbasis       << "\n";
    repPrint(output,'>',dep) << "y basis:       " << locbasisgt     << "\n";
    repPrint(output,'>',dep) << "VV:            " << VV             << "\n";
    repPrint(output,'>',dep) << "cheatscache:   " << cheatscache    << "\n";
    repPrint(output,'>',dep) << "midadd:        " << midadd         << "\n";
    repPrint(output,'>',dep) << "default proj:  " << defproj        << "\n";

    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base SVC: ";
    SVM_Scalar::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_Planar::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> inGpn;
    input >> dummy; input >> locd;
    input >> dummy; input >> locbasis;
    input >> dummy; input >> locbasisgt;
    input >> dummy; input >> VV;
    input >> dummy; input >> cheatscache;
    input >> dummy; input >> midadd;
    input >> dummy; input >> defproj;
    input >> dummy;

    ML_Base::VVcallback = &VVcallbacknon;

    SVM_Scalar::inputstream(input);
    SVM_Scalar::naivesetGpnExt(&inGpn);

    return input;
}

int SVM_Planar::prealloc(int expectedN)
{
    locd.prealloc(expectedN);
    inGpn.prealloc(expectedN,NbasisVV());
    cheatscache.prealloc(expectedN);
    SVM_Scalar::prealloc(expectedN);

    return 0;
}

int SVM_Planar::preallocsize(void) const
{
    return SVM_Scalar::preallocsize();
}

double &SVM_Planar::K2(double &res, int i, int j, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo, int resmode) const
{
    if ( midadd || ( i < 0 ) || ( j < 0 ) || resmode )
    {
        // Cache not relevant this time, so recalculate directly

        cheatscache.K2(res,i,j,pxyprod,xx,yy,xxinfo,yyinfo,resmode);
    }

    else
    {
        // Look in cache - if not in cache this will bounce back to ML_Base::K via evalKSVM_Planar

        res = (cheatscache.Gp())(i,j);
    }

    // Factor in expert agreement factor if relevant

    const SparseVector<gentype> *xinear   = NULL;
    const SparseVector<gentype> *xifar    = NULL;
    const SparseVector<gentype> *xifarfar = NULL;
    const vecInfo *xinearinfo = NULL;
    const vecInfo *xifarinfo  = NULL;
    const gentype *ineartup = NULL;
    const gentype *ifartup  = NULL;

    int inear,ifar,iokr,iok,idiagr,igradOrder,iplanr,iplan,iset,alr,arr,agr;

    const SparseVector<gentype> *xjnear   = NULL;
    const SparseVector<gentype> *xjfar    = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;
    const vecInfo *xjnearinfo = NULL;
    const vecInfo *xjfarinfo  = NULL;
    const gentype *jneartup = NULL;
    const gentype *jfartup  = NULL;

    int jnear,jfar,jokr,jok,jdiagr,jgradOrder,jplanr,jplan,jset,blr,brr,bgr;

    SparseVector<gentype> *xauntang = NULL; 
    SparseVector<gentype> *xbuntang = NULL;

    vecInfo *xainfountang = NULL;
    vecInfo *xbinfountang = NULL;

    detangle_x(xauntang,xainfountang,xinear,xifar,xifarfar,xinearinfo,xifarinfo,inear,ifar,ineartup,ifartup,alr,arr,agr,iokr,iok,i,idiagr,&x(i),xxinfo,igradOrder,iplanr,iplan,iset,1,0);
    detangle_x(xbuntang,xbinfountang,xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,jnear,jfar,jneartup,jfartup,blr,brr,bgr,jokr,jok,j,jdiagr,&x(j),yyinfo,jgradOrder,jplanr,jplan,jset,1,0);

    if ( iplanr || jplanr )
    {
        if ( ( iplan >= 0 ) && ( jplan >= 0 ) )
        {
            res *= VV(iplan,jplan); // Pre-calculated for speed
        }

        else
        {
            gentype VVres;
            gentype kval(res);

            gentype gdum('N');
            VVcallbackdef(VVres,kval,*this,iplanr,jplanr,iplan,jplan,x(i).isfarfarfarindpresent(7) ? x(i).fff(7) : gdum,x(j).isfarfarfarindpresent(7) ? x(j).fff(7) : gdum,defProjVV());

            res = (double) VVres;
        }
    }

    return res;
}


gentype &SVM_Planar::K2(gentype &res, int i, int j, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo, int resmode) const
{
    if ( midadd || ( i < 0 ) || ( j < 0 ) || !resmode )
    {
        // Cache not relevant this time, so recalculate directly

        cheatscache.K2(res,i,j,pxyprod,xx,yy,xxinfo,yyinfo,resmode);
    }

    else
    {
        // Look in cache - if not in cache this will bounce back to ML_Base::K via evalKSVM_Planar

        res = (cheatscache.Gp())(i,j);
    }

    // Factor in expert agreement factor if relevant

    const SparseVector<gentype> *xinear   = NULL;
    const SparseVector<gentype> *xifar    = NULL;
    const SparseVector<gentype> *xifarfar = NULL;
    const vecInfo *xinearinfo = NULL;
    const vecInfo *xifarinfo  = NULL;
    const gentype *ineartup = NULL;
    const gentype *ifartup  = NULL;

    int inear,ifar,iokr,iok,idiagr,igradOrder,iplanr,iplan,iset,alr,arr,agr;

    const SparseVector<gentype> *xjnear   = NULL;
    const SparseVector<gentype> *xjfar    = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;
    const vecInfo *xjnearinfo = NULL;
    const vecInfo *xjfarinfo  = NULL;
    const gentype *jneartup = NULL;
    const gentype *jfartup  = NULL;

    int jnear,jfar,jokr,jok,jdiagr,jgradOrder,jplanr,jplan,jset,blr,brr,bgr;

    SparseVector<gentype> *xauntang = NULL; 
    SparseVector<gentype> *xbuntang = NULL;

    vecInfo *xainfountang = NULL;
    vecInfo *xbinfountang = NULL;

    detangle_x(xauntang,xainfountang,xinear,xifar,xifarfar,xinearinfo,xifarinfo,inear,ifar,ineartup,ifartup,alr,arr,agr,iokr,iok,i,idiagr,&x(i),xxinfo,igradOrder,iplanr,iplan,iset,1,0);
    detangle_x(xbuntang,xbinfountang,xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,jnear,jfar,jneartup,jfartup,blr,brr,bgr,jokr,jok,j,jdiagr,&x(j),yyinfo,jgradOrder,jplanr,jplan,jset,1,0);

    if ( iplanr || jplanr )
    {
        if ( ( iplan >= 0 ) && ( jplan >= 0 ) )
        {
            res *= VV(iplan,jplan); // Pre-calculated for speed
        }

        else
        {
            gentype VVres;
            gentype kval(res);

            gentype gdum('N');
            VVcallbackdef(VVres,kval,*this,iplanr,jplanr,iplan,jplan,x(i).isfarfarfarindpresent(7) ? x(i).fff(7) : gdum,x(j).isfarfarfarindpresent(7) ? x(j).fff(7) : gdum,defProjVV());

            res = VVres;
        }
    }

    return res;
}


gentype &SVM_Planar::K2(gentype &res, int i, int j, const gentype &bias, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo, int resmode) const
{
    if ( midadd || ( i < 0 ) || ( j < 0 ) || !resmode )
    {
        // Cache not relevant this time, so recalculate directly

        cheatscache.K2(res,i,j,bias,pxyprod,xx,yy,xxinfo,yyinfo,resmode);
    }

    else
    {
        // Look in cache - if not in cache this will bounce back to ML_Base::K via evalKSVM_Planar

        res = (cheatscache.Gp())(i,j);
    }

    // Factor in expert agreement factor if relevant

    const SparseVector<gentype> *xinear   = NULL;
    const SparseVector<gentype> *xifar    = NULL;
    const SparseVector<gentype> *xifarfar = NULL;
    const vecInfo *xinearinfo = NULL;
    const vecInfo *xifarinfo  = NULL;
    const gentype *ineartup = NULL;
    const gentype *ifartup  = NULL;

    int inear,ifar,iokr,iok,idiagr,igradOrder,iplanr,iplan,iset,alr,arr,agr;

    const SparseVector<gentype> *xjnear   = NULL;
    const SparseVector<gentype> *xjfar    = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;
    const vecInfo *xjnearinfo = NULL;
    const vecInfo *xjfarinfo  = NULL;
    const gentype *jneartup = NULL;
    const gentype *jfartup  = NULL;

    int jnear,jfar,jokr,jok,jdiagr,jgradOrder,jplanr,jplan,jset,blr,brr,bgr;

    SparseVector<gentype> *xauntang = NULL; 
    SparseVector<gentype> *xbuntang = NULL;

    vecInfo *xainfountang = NULL;
    vecInfo *xbinfountang = NULL;

    detangle_x(xauntang,xainfountang,xinear,xifar,xifarfar,xinearinfo,xifarinfo,inear,ifar,ineartup,ifartup,alr,arr,agr,iokr,iok,i,idiagr,&x(i),xxinfo,igradOrder,iplanr,iplan,iset,1,0);
    detangle_x(xbuntang,xbinfountang,xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,jnear,jfar,jneartup,jfartup,blr,brr,bgr,jokr,jok,j,jdiagr,&x(j),yyinfo,jgradOrder,jplanr,jplan,jset,1,0);

    if ( iplanr || jplanr )
    {
        if ( ( iplan >= 0 ) && ( jplan >= 0 ) )
        {
            res *= VV(iplan,jplan); // Pre-calculated for speed
        }

        else
        {
            gentype VVres;
            gentype kval(res);

            gentype gdum('N');
            VVcallbackdef(VVres,kval,*this,iplanr,jplanr,iplan,jplan,x(i).isfarfarfarindpresent(7) ? x(i).fff(7) : gdum,x(j).isfarfarfarindpresent(7) ? x(j).fff(7) : gdum,defProjVV());

            res = VVres;
        }
    }

    return res;
}


gentype &SVM_Planar::K2(gentype &res, int i, int j, const MercerKernel &altK, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo, int resmode) const
{
    if ( midadd || ( i < 0 ) || ( j < 0 ) || !resmode )
    {
        // Cache not relevant this time, so recalculate directly

        cheatscache.K2(res,i,j,altK,pxyprod,xx,yy,xxinfo,yyinfo,resmode);
    }

    else
    {
        // Look in cache - if not in cache this will bounce back to ML_Base::K via evalKSVM_Planar

        res = (cheatscache.Gp())(i,j);
    }

    // Factor in expert agreement factor if relevant

    const SparseVector<gentype> *xinear   = NULL;
    const SparseVector<gentype> *xifar    = NULL;
    const SparseVector<gentype> *xifarfar = NULL;
    const vecInfo *xinearinfo = NULL;
    const vecInfo *xifarinfo  = NULL;
    const gentype *ineartup = NULL;
    const gentype *ifartup  = NULL;

    int inear,ifar,iokr,iok,idiagr,igradOrder,iplanr,iplan,iset,alr,arr,agr;

    const SparseVector<gentype> *xjnear   = NULL;
    const SparseVector<gentype> *xjfar    = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;
    const vecInfo *xjnearinfo = NULL;
    const vecInfo *xjfarinfo  = NULL;
    const gentype *jneartup = NULL;
    const gentype *jfartup  = NULL;

    int jnear,jfar,jokr,jok,jdiagr,jgradOrder,jplanr,jplan,jset,blr,brr,bgr;

    SparseVector<gentype> *xauntang = NULL; 
    SparseVector<gentype> *xbuntang = NULL;

    vecInfo *xainfountang = NULL;
    vecInfo *xbinfountang = NULL;

    detangle_x(xauntang,xainfountang,xinear,xifar,xifarfar,xinearinfo,xifarinfo,inear,ifar,ineartup,ifartup,alr,arr,agr,iokr,iok,i,idiagr,&x(i),xxinfo,igradOrder,iplanr,iplan,iset,1,0);
    detangle_x(xbuntang,xbinfountang,xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,jnear,jfar,jneartup,jfartup,blr,brr,bgr,jokr,jok,j,jdiagr,&x(j),yyinfo,jgradOrder,jplanr,jplan,jset,1,0);

    if ( iplanr || jplanr )
    {
        if ( ( iplan >= 0 ) && ( jplan >= 0 ) )
        {
            res *= VV(iplan,jplan); // Pre-calculated for speed
        }

        else
        {
            gentype VVres;
            gentype kval(res);

            gentype gdum('N');
            VVcallbackdef(VVres,kval,*this,iplanr,jplanr,iplan,jplan,x(i).isfarfarfarindpresent(7) ? x(i).fff(7) : gdum,x(j).isfarfarfarindpresent(7) ? x(j).fff(7) : gdum,defProjVV());

            res = VVres;
        }
    }

    return res;
}


Matrix<double> &SVM_Planar::K2(int spaceDim, Matrix<double> &res, int i, int j, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo, int resmode) const
{
    if ( midadd || ( i < 0 ) || ( j < 0 ) || !resmode )
    {
        // Cache not relevant this time, so recalculate directly

        cheatscache.K2(spaceDim,res,i,j,pxyprod,xx,yy,xxinfo,yyinfo,resmode);
    }

    else
    {
        // Look in cache - if not in cache this will bounce back to ML_Base::K via evalKSVM_Planar

        res = (cheatscache.Gp())(i,j);
    }

    // Factor in expert agreement factor if relevant

    const SparseVector<gentype> *xinear   = NULL;
    const SparseVector<gentype> *xifar    = NULL;
    const SparseVector<gentype> *xifarfar = NULL;
    const vecInfo *xinearinfo = NULL;
    const vecInfo *xifarinfo  = NULL;
    const gentype *ineartup = NULL;
    const gentype *ifartup  = NULL;

    int inear,ifar,iokr,iok,idiagr,igradOrder,iplanr,iplan,iset,alr,arr,agr;

    const SparseVector<gentype> *xjnear   = NULL;
    const SparseVector<gentype> *xjfar    = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;
    const vecInfo *xjnearinfo = NULL;
    const vecInfo *xjfarinfo  = NULL;
    const gentype *jneartup = NULL;
    const gentype *jfartup  = NULL;

    int jnear,jfar,jokr,jok,jdiagr,jgradOrder,jplanr,jplan,jset,blr,brr,bgr;

    SparseVector<gentype> *xauntang = NULL; 
    SparseVector<gentype> *xbuntang = NULL;

    vecInfo *xainfountang = NULL;
    vecInfo *xbinfountang = NULL;

    detangle_x(xauntang,xainfountang,xinear,xifar,xifarfar,xinearinfo,xifarinfo,inear,ifar,ineartup,ifartup,alr,arr,agr,iokr,iok,i,idiagr,&x(i),xxinfo,igradOrder,iplanr,iplan,iset,1,0);
    detangle_x(xbuntang,xbinfountang,xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,jnear,jfar,jneartup,jfartup,blr,brr,bgr,jokr,jok,j,jdiagr,&x(j),yyinfo,jgradOrder,jplanr,jplan,jset,1,0);

    if ( iplanr || jplanr )
    {
        if ( ( iplan >= 0 ) && ( jplan >= 0 ) )
        {
            res *= VV(iplan,jplan); // Pre-calculated for speed
        }

        else
        {
            gentype VVres;
            gentype kval(res);

            gentype gdum('N');
            VVcallbackdef(VVres,kval,*this,iplanr,jplanr,iplan,jplan,x(i).isfarfarfarindpresent(7) ? x(i).fff(7) : gdum,x(j).isfarfarfarindpresent(7) ? x(j).fff(7) : gdum,defProjVV());

            res = (const Matrix<double> &) VVres;
        }
    }

    return res;
}


d_anion &SVM_Planar::K2(int spaceDim, d_anion &res, int i, int j, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo, int resmode) const
{
    if ( midadd || ( i < 0 ) || ( j < 0 ) || !resmode )
    {
        // Cache not relevant this time, so recalculate directly

        cheatscache.K2(spaceDim,res,i,j,pxyprod,xx,yy,xxinfo,yyinfo,resmode);
    }

    else
    {
        // Look in cache - if not in cache this will bounce back to ML_Base::K via evalKSVM_Planar

        res = (cheatscache.Gp())(i,j);
    }

    // Factor in expert agreement factor if relevant

    const SparseVector<gentype> *xinear   = NULL;
    const SparseVector<gentype> *xifar    = NULL;
    const SparseVector<gentype> *xifarfar = NULL;
    const vecInfo *xinearinfo = NULL;
    const vecInfo *xifarinfo  = NULL;
    const gentype *ineartup = NULL;
    const gentype *ifartup  = NULL;

    int inear,ifar,iokr,iok,idiagr,igradOrder,iplanr,iplan,iset,alr,arr,agr;

    const SparseVector<gentype> *xjnear   = NULL;
    const SparseVector<gentype> *xjfar    = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;
    const vecInfo *xjnearinfo = NULL;
    const vecInfo *xjfarinfo  = NULL;
    const gentype *jneartup = NULL;
    const gentype *jfartup  = NULL;

    int jnear,jfar,jokr,jok,jdiagr,jgradOrder,jplanr,jplan,jset,blr,brr,bgr;

    SparseVector<gentype> *xauntang = NULL; 
    SparseVector<gentype> *xbuntang = NULL;

    vecInfo *xainfountang = NULL;
    vecInfo *xbinfountang = NULL;

    detangle_x(xauntang,xainfountang,xinear,xifar,xifarfar,xinearinfo,xifarinfo,inear,ifar,ineartup,ifartup,alr,arr,agr,iokr,iok,i,idiagr,&x(i),xxinfo,igradOrder,iplanr,iplan,iset,1,0);
    detangle_x(xbuntang,xbinfountang,xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,jnear,jfar,jneartup,jfartup,blr,brr,bgr,jokr,jok,j,jdiagr,&x(j),yyinfo,jgradOrder,jplanr,jplan,jset,1,0);

    if ( iplanr || jplanr )
    {
        if ( ( iplan >= 0 ) && ( jplan >= 0 ) )
        {
            res *= VV(iplan,jplan); // Pre-calculated for speed
        }

        else
        {
            gentype VVres;
            gentype kval(res);

            gentype gdum('N');
            VVcallbackdef(VVres,kval,*this,iplanr,jplanr,iplan,jplan,x(i).isfarfarfarindpresent(7) ? x(i).fff(7) : gdum,x(j).isfarfarfarindpresent(7) ? x(j).fff(7) : gdum,defProjVV());

            res = (const d_anion &) VVres;
        }
    }

    return res;
}










// The Gpn vector (actually a single-column matrix) controls whether the
// bias is included with the relevant parts of the Hessian.  This calculates
// the relevant part of Gpn.  bias is stored using U as basis.

int SVM_Planar::rankcalcGpn(Vector<double> &res, int d, const SparseVector<gentype> &x, int i)
{
    (void) i;
    (void) d;

    NiceAssert( res.size() == NbasisVV() );

    int nzres = 0;

    if ( x.isfaroffindpresent() || x.isfarfarfarindpresent(1) || !(x.isfarfarfarindpresent(7)) )
    {
        nzres = 0;
        res = 0.0;
    }

    else if ( (x.fff(7)).isValInteger() )
    {
        retVector<double> tmpva;

        nzres = 1;
        res = VV((int) x.fff(7),tmpva);
    }

    else
    {
        nzres = 1;

        res.resize(bdim);

        int j;

        for ( j = 0 ; j < bdim ; j++ )
        {
            res("&",j) = (double) (VbasisVV()(j)*(x.fff(7)));
        }
    }

    return nzres;
}

void SVM_Planar::calcVVij(double &res, int i, int j) const
{
    twoProductNoConj(res,locbasis(i),locbasis(j));

    return;
}

void SVM_Planar::refactorVV(int updateGpn)
{
    int M = NbasisVV();
    int i,q,r;

    if ( M )
    {
        for ( q = 0 ; q < M ; q++ )
        {
            for ( r = 0 ; r < M ; r++ )
            {
                calcVVij(VV("&",q,r),q,r);
            }
        }
    }

    if ( SVM_Planar::N() )
    {
        for ( i = 0 ; i < SVM_Planar::N() ; i++ )
        {
            if ( locd(i) )
            { 
                resetKernel(0,i,0);
            }
        }
    }

    if ( updateGpn && !isFixedBias() && SVM_Planar::N() && NbasisVV() )
    {
        // setbiasdim requires Gpn to be updated *after* calling if
        // biasdim decrease and *before* calling if biasdim increases,
        // so we can simply decrase biasdim to 0(+1), update Gpn and
        // then increase it.

        SVM_Scalar::setbiasdim(1);

        inGpn.resize(SVM_Planar::N(),NbasisVV());

        retVector<double> tmpva;

        for ( i = 0 ; i < SVM_Planar::N() ; i++ )
        {
            rankcalcGpn(inGpn("&",i,tmpva),locd(i),x(i),i);
        }

        SVM_Scalar::setbiasdim(NbasisVV()+1);
    }

    return;
}



// Evaluation:

int SVM_Planar::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
//    ghEvalFull = 0; - not actually needed here, but might be in future.  Meant to force
// full evaluation of g(x_i) even if i >= 0 (ie don't just use error), but that appears
// to be default behaviour in any case.

    (void) retaltg;

    int res = 1;

    {
        resg = 0.0;

        int j;

        gentype temp;

        if ( NS() )
        {
            for ( j = 0 ; j < SVM_Planar::N() ; j++ )
            {
                if ( alphaState()(j) )
                {
                    K2(temp,i,j,pxyprodi ? (const gentype **) pxyprodi[j] : NULL);
                    temp *= alphaR()(j);

                    resg += temp;
                }
            }
        }

        if ( !isFixedBias() && NbasisVV() )
        {
            for ( j = 0 ; j < NbasisVV() ; j++ )
            {
                if ( x(i).isfarfarfarindpresent(7) && x(i).fff(7).isValInteger() )
                {
                    temp =  VbasisVV()(j)*VbasisVV()((int) x(i).fff(7));
                    temp *= SVM_Scalar::biasVMulti(j);

                    resg += temp;
                }

                else if ( x(i).isfarfarfarindpresent(7) )
                {
                    temp =  VbasisVV()(j)*(x(i).fff(7));
                    temp *= SVM_Scalar::biasVMulti(j);

                    resg += temp;
                }

                else
                {
                    temp =  VbasisVV()(j);
                    temp *= SVM_Scalar::biasVMulti(j);

                    resg += temp;
                }
            }
        }

        if ( defProjVV() == -1 )
        {
            resh = resg;
        }

        else
        {
            resh = VbasisVV()(defProjVV())*resg;
        }
    }

    return res;
}

double SVM_Planar::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    double res = 0;

    if ( ia >= SVM_Planar::N() )
    {
        // ha is vector, hb is scalar

        gentype haa(ha);

        haa *= VbasisVV()(ia-SVM_Planar::N());

        res = SVM_Scalar::calcDist(haa,hb,db);
    }

    else if ( ia >= 0 )
    {
        // ha is vector, hb is scalar
        // x(ia) gives u vector as a7 element

        gentype haa(ha);

        haa *= x(ia).fff(7).isValInteger() ? VbasisVV()((int) x(ia).fff(7)) : x(ia).fff(7);

        res = SVM_Scalar::calcDist(haa,hb,db);
    }

    else
    {
        res = db ? abs2((double) (ha-hb)) : 0.0;
    }

    return res;
}














int SVM_Planar::qaddTrainingVector(int i, double z, SparseVector<gentype> &xx, double Cweigh, double epsweigh, int d)
{ 
    cheatscache.addTrainingVector(i,z,xx,Cweigh,epsweigh,d);

    locd.add(i);
    locd("&",i) = d;

    retVector<double> tmpva;

    inGpn.addRow(i);
    rankcalcGpn(inGpn("&",i,tmpva),d,xx,i);

    midadd++;
    int res = SVM_Scalar::qaddTrainingVector(i,z,xx,Cweigh,epsweigh,d);
    midadd--;

    return res;
}

int SVM_Planar::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    int res = SVM_Scalar::removeTrainingVector(i,y,x);

    inGpn.removeRow(i);
    locd.remove(i);
    cheatscache.removeTrainingVector(i,y,x);

    return res;
}

int SVM_Planar::setx(int i, const SparseVector<gentype> &x)
{
    int res = 0;

    res =  cheatscache.setx(i,x);
    res |= SVM_Scalar::setx(i,x);

    return res;
}

int SVM_Planar::setd(int i, int nd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Planar::N() );

    int res = 0;

    // This is slightly complicated.  We need to update
    //
    // - Gpn(i,0)
    // - Gp(i,:) and Gp(:,i)

    if ( SVM_Planar::d()(i) && nd && ( SVM_Planar::d()(i) != nd ) )
    {
        res |= SVM_Planar::setd(i,0);
        res |= SVM_Planar::setd(i,nd);
    }

    else if ( SVM_Planar::d()(i) && !nd )
    {
        // Make d zero first so that refactGpnElm works properly
        res |= SVM_Scalar::setd(i,nd);
        locd("&",i) = nd;

        // Update Gpn
        Vector<double> newGpnVal(NbasisVV());
        rankcalcGpn(newGpnVal,nd,x(i),i);

        if ( NbasisVV() )
        {
            int j;

            for ( j = 0 ; j < NbasisVV() ; j++ )
            {
                SVM_Scalar::refactGpnElm(i,j,newGpnVal(j));
                inGpn("&",i,j) = newGpnVal(j);
            }
        }

        // Update Gp
        SVM_Planar::resetKernel(1,i,0);
    }

    else if ( !SVM_Planar::d()(i) && nd )
    {
        // Update Gpn
        Vector<double> newGpnVal(NbasisVV());
        rankcalcGpn(newGpnVal,nd,x(i),i);

        if ( NbasisVV() )
        {
            int j;

            for ( j = 0 ; j < NbasisVV() ; j++ )
            {
                SVM_Scalar::refactGpnElm(i,j,newGpnVal(j));
                inGpn("&",i,j) = newGpnVal(j);
            }
        }

        // Update d
        res |= SVM_Scalar::setd(i,nd);
        locd("&",i) = nd;

        // Update Gp
        SVM_Planar::resetKernel(1,i,0);
    }

    return res;
}

int SVM_Planar::train(int &res, svmvolatile int &killSwitch)
{
    return SVM_Scalar::train(res,killSwitch);
}





































int SVM_Planar::addToBasisVV(int i, const gentype &o)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= NbasisVV() );
    NiceAssert( o.isValVector() );
    NiceAssert( ( !bdim && !NbasisVV() ) || ( bdim == o.size() ) );

    if ( !bdim )
    {
        bdim = o.size();
    }

    // Add to basis set

    locbasisgt.add(i);
    locbasisgt("&",i) = o;
    locbasis.add(i); // This update Nbasis
    locbasis("&",i).resize(o.size());

    int j;

    const Vector<gentype> &ghgh = (const Vector<gentype> &) locbasisgt(i);

    for ( j = 0 ; j < o.size() ; j++ )
    {
        locbasis("&",i)("&",j) = (double) (ghgh(j));
    }

    // Update VV matrix

    VV.addRowCol(i);

    for ( j = 0 ; j < NbasisVV() ; j++ )
    {
        calcVVij(VV("&",i,j),i,j);
        calcVVij(VV("&",j,i),j,i);
    }

    // Update Gpn matrix

    inGpn.addCol(i);

    if ( SVM_Planar::N() )
    {
        int j;

        retVector<double> tmpva;

        for ( j = 0 ; j < SVM_Planar::N() ; j++ )
        {
            rankcalcGpn(inGpn("&",j,tmpva),locd(j),x(j),j);
        }
    }

    SVM_Scalar::setbiasdim(NbasisVV()+1,i);

    return 1;
}

int SVM_Planar::removeFromBasisVV(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisVV() );

    // Remove from basis set

    locbasisgt.remove(i);
    locbasis.remove(i);

    // Update VV matrix

    VV.removeRowCol(i);

    // Update Gpn matrix

    SVM_Scalar::setbiasdim(NbasisVV()+1);

    inGpn.removeCol(i);

    if ( NbasisVV() == 0 )
    {
        bdim = 0;
    }

    return 1;
}

int SVM_Planar::setBasisVV(int i, const gentype &o)
{
    return setBasisVV(i,o,1);
}

int SVM_Planar::setBasisVV(int i, const gentype &o, int updateU)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisVV() );
    NiceAssert( ( NbasisVV() == 1 ) || ( bdim == o.size() ) );

    // Add to basis set

    locbasisgt("&",i) = o;
    locbasis("&",i).resize(o.size());

    int j;

    const Vector<gentype> &ghgh = (const Vector<gentype> &) locbasisgt(i);

    for ( j = 0 ; j < o.size() ; j++ )
    {
        locbasis("&",i)("&",j) = (double) ghgh(j);
    }

    // Update VV matrix

    if ( updateU )
    {
        refactorVV(0);
    }

    // Update Gpn matrix

    inGpn.resize(SVM_Planar::N(),NbasisVV());

    if ( SVM_Planar::N() )
    {
        int j;

        retVector<double> tmpva;

        for ( j = 0 ; j < SVM_Planar::N() ; j++ )
        {
            rankcalcGpn(inGpn("&",j,tmpva),locd(j),x(j),j);
        }
    }

    SVM_Scalar::setbiasdim(NbasisVV()+1);

    return 1;
}

int SVM_Planar::setBasisVV(const Vector<gentype> &o)
{
    int i;
    int res = 0;

    while ( NbasisVV() > o.size() )
    {
        res = 1;

        locbasisgt.remove(locbasisgt.size()-1);
        locbasis.remove(locbasisgt.size()-1);
    }

    if ( NbasisVV() )
    {
        for ( i = 0 ; i < NbasisVV() ; i++ )
        {
            res |= setBasisVV(i,o(i),(i==NbasisVV()-1)?1:0);
        }
    }

    while ( NbasisVV() < o.size() )
    {
        res |= addToBasisVV(NbasisVV(),o(NbasisVV()));
    }

    return res;
}

void SVM_Planar::reconstructlocbasisgt(void)
{
    int i;
    int M = NbasisVV();

    for ( i = 0 ; i < M ; i++ )
    {
        locbasisgt("&",i) = locbasis(i);
    }

    return;
}

int SVM_Planar::setVarBias(void)
{
    int doup = isFixedBias();
    int res = SVM_Scalar::setVarBias();

    if ( doup )
    {
        res |= 1;
        refactorVV(1);
    }

    return res;
}

int SVM_Planar::setPosBias(void)
{
    int doup = isFixedBias();
    int res = SVM_Scalar::setPosBias();

    if ( doup )
    {
        res |= 1;
        refactorVV(1);
    }

    return res;
}

int SVM_Planar::setNegBias(void)
{
    int doup = isFixedBias();
    int res = SVM_Scalar::setNegBias();

    if ( doup )
    {
        res |= 1;
        refactorVV(1);
    }

    return res;
}

int SVM_Planar::setFixedBias(double newbias)
{
    return SVM_Scalar::setFixedBias(newbias);
}

int SVM_Planar::setVarBias(int q)
{
    int doup = isFixedBias();
    int res = SVM_Scalar::setVarBias(q);

    if ( doup )
    {
        res |= 1;
        refactorVV(1);
    }

    return res;
}

int SVM_Planar::setPosBias(int q)
{
    int doup = isFixedBias();
    int res = SVM_Scalar::setPosBias(q);

    if ( doup )
    {
        res |= 1;
        refactorVV(1);
    }

    return res;
}

int SVM_Planar::setNegBias(int q)
{
    int doup = isFixedBias();
    int res = SVM_Scalar::setNegBias(q);

    if ( doup )
    {
        res |= 1;
        refactorVV(1);
    }

    return res;
}

int SVM_Planar::setFixedBias(int q, double newbias)
{
    int doup = !isFixedBias();
    int res = SVM_Scalar::setFixedBias(q,newbias);

    if ( doup )
    {
        res |= 1;
        refactorVV(1);
    }

    return res;
}









































































































int SVM_Planar::removeTrainingVector(int i, int num)
{
    NiceAssert( i < ML_Base::N() );
    NiceAssert( num >= 0 );
    NiceAssert( num <= ML_Base::N()-i );

    int res = 0;
    gentype y;
    SparseVector<gentype> x;

    while ( num )
    {
        res |= removeTrainingVector(i+num,y,x);
        num--;
    }

    return res;
}

int SVM_Planar::addTrainingVector (int i, double z, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d)
{
    SparseVector<gentype> xxx(x);

    return SVM_Planar::qaddTrainingVector(i,z,xxx,Cweigh,epsweigh,d);
}

int SVM_Planar::addTrainingVector (int i, const Vector<double> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &xxd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Planar::N() );
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );
    NiceAssert( z.size() == xxd.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Planar::addTrainingVector(i+j,z(j),x(j),Cweigh(j),epsweigh(j),xxd(j));
        }
    }

    return res;
}

int SVM_Planar::qaddTrainingVector(int i, const Vector<double> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &xxd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Planar::N() );
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );
    NiceAssert( z.size() == xxd.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Planar::qaddTrainingVector(i+j,z(j),x("&",j),Cweigh(j),epsweigh(j),xxd(j));
        }
    }

    return res;
}

int SVM_Planar::addTrainingVector(int i, const gentype &zi, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_Planar::addTrainingVector(i,(double) zi,x,Cweigh,epsweigh,2);
}

int SVM_Planar::qaddTrainingVector(int i, const gentype &zi, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_Planar::qaddTrainingVector(i,(double) zi,x,Cweigh,epsweigh,2);
}

int SVM_Planar::addTrainingVector(int i, const Vector<gentype> &zi, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<double> zzi(zi.size());
    Vector<int> ddd(zi.size());

    ddd = 2;

    if ( zi.size() )
    {
        int j;

        for ( j = 0 ; j < zi.size() ; j++ )
        {
            zzi("&",j) = (double) zi(j);
        }
    }

    return SVM_Planar::addTrainingVector(i,zzi,x,Cweigh,epsweigh,ddd);
}

int SVM_Planar::qaddTrainingVector(int i, const Vector<gentype> &zi, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<double> zzi(zi.size());
    Vector<int> ddd(zi.size());

    ddd = 2;

    if ( zi.size() )
    {
        int j;

        for ( j = 0 ; j < zi.size() ; j++ )
        {
            zzi("&",j) = (double) zi(j);
        }
    }

    return SVM_Planar::qaddTrainingVector(i,zzi,x,Cweigh,epsweigh,ddd);
}

int SVM_Planar::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x)
{
    NiceAssert( i.size() == x.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= SVM_Planar::setx(i(j),x(j));
        }
    }

    return res;
}

int SVM_Planar::setx(const Vector<SparseVector<gentype> > &x)
{
    NiceAssert( x.size() == SVM_Planar::N() );

    int res = 0;

    if ( SVM_Planar::N() )
    {
        int j;

        for ( j = 0 ; j < SVM_Planar::N() ; j++ )
        {
            res |= SVM_Planar::setx(j,x(j));
        }
    }

    return res;
}

int SVM_Planar::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i.size() == d.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= SVM_Planar::setd(i(j),d(j));
        }
    }

    return res;
}

int SVM_Planar::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == SVM_Planar::N() );

    int res = 0;

    if ( SVM_Planar::N() )
    {
        int j;

        for ( j = 0 ; j < SVM_Planar::N() ; j++ )
        {
            res |= SVM_Planar::setd(j,d(j));
        }
    }

    return res;
}






















































