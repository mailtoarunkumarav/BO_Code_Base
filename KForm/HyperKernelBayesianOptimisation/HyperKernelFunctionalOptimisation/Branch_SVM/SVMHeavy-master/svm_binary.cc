
//
// Binary Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_binary.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


class SVM_Single;

// the zcalc macro effectively calculates what we want gp to be to take into
// account both z and tube factors, allowing hp == 0 which results in faster
// optimisation.  The functions below are those that need to use this translation
// macro.

#define DCALC(_d_)                                             ( ( isClassifyViaSVR() && (_d_) ) ? 2 : (_d_) )
#define ZCALC(_z_,_d_,_E_,_Eclass_,_xclass_,_Eweigh_)          ( ( isFixedTube() ? ( (_z_) + ( (_d_) * (_Eclass_)(((_xclass_)+1)/2) * (_E_) * (_Eweigh_) ) ) : (_z_) + ( isClassifyViaSVR() ? (_d_) : 0 ) ) )
#define CWCALC(_xclass_,_Cweight_,_Cweightfuzz_,_dthres_)      ( (_Cweight_) * (_Cweightfuzz_) * (binCclass((_xclass_)+1)) * ( ( ( (_dthres_) > 0 ) && ( (_dthres_) < 0.5 ) ) ? ( ( 1 - (_dthres_) ) / (_dthres_) ) : 1 ) )
#define CWCALCBASE(_xclass_,_Cweight_,_Cweightfuzz_,_dthres_)  ( (_Cweight_) * (_Cweightfuzz_) * (binCclass((_xclass_)+1)) )
#define CWCALCEXTRA(_xclass_,_Cweight_,_Cweightfuzz_,_dthres_) ( (_Cweight_) * (_Cweightfuzz_) * (binCclass((_xclass_)+1)) * ( ( ( (_dthres_) > 0 ) && ( (_dthres_) < 0.5 ) ) ? ( ( 1 - (_dthres_) ) / (_dthres_) ) - 1 : 0 ) )

#define DEFAULT_DTHRES 0.0

SVM_Binary::SVM_Binary() : SVM_Scalar()
{
    setaltx(NULL);

    SVM_Scalar::seteps(0);
    bineps = 1;
    binepsclass.resize(3);
    binepsclass = 1.0;
    binCclass.resize(3);
    binCclass = 1.0;
    isSVMviaSVR = 0;
    binNnc.resize(4);
    binNnc = zeroint();
    SVM_Scalar::setRestrictEpsNeg();

    binautosetLevel = 0;
    binautosetnuval = 0.0;
    binautosetCval  = 0.0;

    dthres = DEFAULT_DTHRES;

    return;
}

SVM_Binary::SVM_Binary(const SVM_Binary &src) : SVM_Scalar()
{
    setaltx(NULL);

    SVM_Scalar::seteps(0);
    bineps = 1;
    binepsclass.resize(3);
    binepsclass = 1.0;
    binCclass.resize(3);
    binCclass = 1.0;
    isSVMviaSVR = 0;
    binNnc.resize(4);
    binNnc = zeroint();
    SVM_Scalar::setRestrictEpsNeg();

    binautosetLevel = 0;
    binautosetnuval = 0.0;
    binautosetCval  = 0.0;

    dthres = DEFAULT_DTHRES;

    assign(src,0);

    return;
}

SVM_Binary::SVM_Binary(const SVM_Binary &src, const ML_Base *xsrc) : SVM_Scalar()
{
    setaltx(xsrc);

    SVM_Scalar::seteps(0);
    bineps = 1;
    binepsclass.resize(3);
    binepsclass = 1.0;
    binCclass.resize(3);
    binCclass = 1.0;
    isSVMviaSVR = 0;
    binNnc.resize(4);
    binNnc = zeroint();
    SVM_Scalar::setRestrictEpsNeg();

    binautosetLevel = 0;
    binautosetnuval = 0.0;
    binautosetCval  = 0.0;

    dthres = DEFAULT_DTHRES;

    assign(src,1);

    return;
}

SVM_Binary::~SVM_Binary()
{
    return;
}

double SVM_Binary::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int SVM_Binary::setC(double xC)
{
    autosetOff();
    return SVM_Scalar::setC(xC);
}

int SVM_Binary::seteps(double xeps)
{
    int i;
    int res = 0;

    if ( binautosetLevel == 6 )
    {
	autosetOff();
    }

    bineps = xeps;

    if ( isFixedTube() )
    {
        if ( SVM_Binary::N() )
	{
            for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	    {
                res |= SVM_Scalar::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
                gentype yn(bintrainclass(i));
                res |= SVM_Generic::sety(i,yn);
	    }
	}
    }

    else
    {
	if ( isClassifyViaSVR() )
	{
            res |= SVM_Scalar::seteps(bineps);
	}

	else
	{
            res |= SVM_Scalar::seteps(-bineps);
	}
    }

    return res;
}

int SVM_Binary::setepsclass(int d, double xeps)
{
    NiceAssert( ( d == -1 ) || ( d == 0 ) || ( d == +1 ) );

    int i;

    binepsclass("&",d+1) = xeps;

    int res = SVM_Scalar::setepsclass(d,xeps);

    if ( isFixedTube() )
    {
        if ( SVM_Binary::N() )
	{
            for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	    {
                if ( bintrainclass(i) == d )
		{
                    res |= SVM_Scalar::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
                    gentype yn(bintrainclass(i));
                    res |= SVM_Generic::sety(i,yn);
		}
	    }
	}
    }

    return res;
}

void SVM_Binary::setalleps(double xeps, const Vector<double> &xepsclass)
{
    NiceAssert( xepsclass.size() == 3 );

    int i;

    seteps(xeps);

    for ( i = 0 ; i < 3 ; i++ )
    {
        setepsclass(i,xepsclass(i));
    }

    return;
}

int SVM_Binary::scaleepsweight(double scalefactor)
{
    int res = 0;

    if ( SVM_Binary::N() )
    {
	int i;

        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            res |= setepsweight(i,binepsweight(i)*scalefactor);
	}
    }

    return res;
}

int SVM_Binary::scaleCweight(double scalefactor)
{
    int res = 0;

    if ( SVM_Binary::N() )
    {
	int i;

        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            res |= setCweight(i,binCweight(i)*scalefactor);
	}
    }

    return res;
}

int SVM_Binary::scaleCweightfuzz(double scalefactor)
{
    int res = 0;

    if ( SVM_Binary::N() )
    {
	int i;

        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            res |= setCweightfuzz(i,binCweightfuzz(i)*scalefactor);
	}
    }

    return res;
}

int SVM_Binary::setCclass(int d, double xC)
{
    NiceAssert( ( d == -1 ) || ( d == 0 ) || ( d == +1 ) );

    int i;
    int res = 0;

    binCclass("&",d+1) = xC;

    if ( SVM_Binary::N() )
    {
        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            if ( bintrainclass(i) == d )
	    {
                res |= SVM_Scalar::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));
	    }
	}
    }

    return res;
}

void SVM_Binary::prepareKernel(void)
{
    SVM_Scalar::prepareKernel();

    return;
}

int SVM_Binary::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    int res = SVM_Scalar::resetKernel(modind,onlyChangeRowI,updateInfo);
    SVM_Binary::fixautosettings(1,0);

    return res;
}

int SVM_Binary::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    int res = SVM_Scalar::setKernel(xkernel,modind,onlyChangeRowI);
    SVM_Binary::fixautosettings(1,0);

    return res;
}

int SVM_Binary::sety(int i, const gentype &zn)
{
    NiceAssert( zn.isCastableToIntegerWithoutLoss() );

    return setd(i,(int) zn);
}

int SVM_Binary::sety(const Vector<int> &j, const Vector<gentype> &yn)
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

int SVM_Binary::sety(const Vector<gentype> &yn)
{
    NiceAssert( SVM_Binary::N() == yn.size() );

    int res = 0;

    if ( SVM_Binary::N() )
    {
        int i;

        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int SVM_Binary::setd(int i, int xd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary::N() );
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) );

    int res = 0;

    if ( xd != bintrainclass(i) )
    {
        res = 1;

        int oldd = bintrainclass(i);

        setdinternal(i,xd);

	if ( !xd || !oldd )
	{
            res |= SVM_Binary::fixautosettings(0,1);
	}
    }

    return res;
}

int SVM_Binary::sety(int i, double xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary::N() );

    int res = 0;

    bintraintarg("&",i) = xz;

    res |= SVM_Scalar::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
    gentype yn(bintrainclass(i));
    res |= SVM_Generic::sety(i,yn);

    return res;
}

int SVM_Binary::setCweight(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary::N() );

    binCweight("&",i) = xCweight;

    return SVM_Scalar::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));
}

int SVM_Binary::setCweightfuzz(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary::N() );

    binCweightfuzz("&",i) = xCweight;

    return SVM_Scalar::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));
}

int SVM_Binary::setepsweight(int i, double xepsweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary::N() );

    binepsweight("&",i) = xepsweight;

    int res = SVM_Scalar::setepsweight(i,xepsweight);

    if ( isFixedTube() )
    {
        res |= SVM_Scalar::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
        gentype yn(bintrainclass(i));
        res |= SVM_Generic::sety(i,yn);
    }

    return res;
}

int SVM_Binary::setd(const Vector<int> &j, const Vector<int> &d)
{
    NiceAssert( j.size() == d.size() );

    int res = 0;

    if ( j.size() )
    {
        res = 1;

        int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= setdinternal(j(i),d(i));
	}

        res |= SVM_Binary::fixautosettings(0,1);
    }

    return res;
}

int SVM_Binary::sety(const Vector<int> &j, const Vector<double> &xz)
{
    NiceAssert( j.size() == xz.size() );

    int res = 0;

    retVector<double> tmpva;

    bintraintarg("&",j,tmpva) = xz;

    if ( j.size() )
    {
	int i;

        for ( i = 0 ; i < j.size() ; i++ )
	{
            res |= SVM_Scalar::sety(j(i),ZCALC(bintraintarg(j(i)),bintrainclass(j(i)),bineps,binepsclass,bintrainclass(j(i)),binepsweight(j(i))));
            gentype yn(bintrainclass(j(i)));
            res |= SVM_Generic::sety(j(i),yn);
	}
    }

    return res;
}

int SVM_Binary::setCweight(const Vector<int> &j, const Vector<double> &xCweight)
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

int SVM_Binary::setCweightfuzz(const Vector<int> &j, const Vector<double> &xCweight)
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

int SVM_Binary::setepsweight(const Vector<int> &j, const Vector<double> &xepsweight)
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

int SVM_Binary::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == SVM_Binary::N() );

    int i;
    int res = 0;

    if ( SVM_Binary::N() )
    {
        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            res |= setdinternal(i,d(i));
	}

        res |= SVM_Binary::fixautosettings(0,1);
    }

    return res;
}

int SVM_Binary::sety(const Vector<double> &xz)
{
    NiceAssert( bintraintarg.size() == xz.size() );

    int res = 0;

    bintraintarg = xz;

    if ( SVM_Binary::N() )
    {
	int i;

        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            res |= SVM_Scalar::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
            gentype yn(bintrainclass(i));
            res |= SVM_Generic::sety(i,yn);
	}
    }

    return res;
}

int SVM_Binary::setCweight(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == SVM_Binary::N() );

    int res = 0;

    int i;

    if ( SVM_Binary::N() )
    {
        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            res |= setCweight(i,xCweight(i));
	}
    }

    return res;
}

int SVM_Binary::setCweightfuzz(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == SVM_Binary::N() );

    int res = 0;

    int i;

    if ( SVM_Binary::N() )
    {
        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            res |= setCweightfuzz(i,xCweight(i));
	}
    }

    return res;
}

int SVM_Binary::setepsweight(const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == SVM_Binary::N() );

    int i;
    int res = 0;

    if ( SVM_Binary::N() )
    {
        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            res |= setepsweight(i,xepsweight(i));
	}
    }

    return res;
}

int SVM_Binary::setdinternal(int i, int xd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary::N() );
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) );

    int res = 0;

    if ( xd != bintrainclass(i) )
    {
        res = 1;

        binNnc("&",bintrainclass(i)+1)--;
        binNnc("&",xd+1)++;

        bintrainclass("&",i) = xd;

        res |= SVM_Scalar::setd(i,DCALC(xd));
        res |= SVM_Scalar::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
        gentype yn(bintrainclass(i));
        res |= SVM_Generic::sety(i,yn);
        res |= SVM_Scalar::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));
    }

    return res;
}

int SVM_Binary::setLinBiasForce(double newval)
{
    if ( binautosetLevel == 6 )
    {
        autosetOff();
    }

    return SVM_Scalar::setLinBiasForce(newval);
}

int SVM_Binary::setFixedTube(void)
{
    int i;
    int res = 0;

    if ( !isFixedTube() )
    {
        res |= SVM_Scalar::setFixedTube();

        if ( SVM_Binary::N() )
	{
            for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	    {
                res |= SVM_Scalar::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
                gentype yn(bintrainclass(i));
                res |= SVM_Generic::sety(i,yn);
	    }
	}

        res |= SVM_Scalar::seteps(0);
    }

    return res;
}

int SVM_Binary::setShrinkTube(void)
{
    int i;
    int res = 0;

    if ( !isShrinkTube() )
    {
        res |= SVM_Scalar::setShrinkTube();

        if ( SVM_Binary::N() )
	{
            for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	    {
                res |= SVM_Scalar::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
                gentype yn(bintrainclass(i));
                res |= SVM_Generic::sety(i,yn);
	    }
	}

	if ( isClassifyViaSVR() )
	{
            res |= SVM_Scalar::seteps(bineps);
	}

	else
	{
            res |= SVM_Scalar::seteps(-bineps);
	}
    }

    return res;
}

int SVM_Binary::addTrainingVector(int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_Binary::addTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int SVM_Binary::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_Binary::qaddTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int SVM_Binary::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<int> zz(z.size());
    Vector<double> xz(z.size());

    xz = 0.0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            zz("&",j) = (int) z(j);
        }
    }

    return SVM_Binary::addTrainingVector(i,zz,x,Cweigh,epsweigh,xz);
}

int SVM_Binary::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<int> zz(z.size());
    Vector<double> xz(z.size());

    xz = 0.0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            zz("&",j) = (int) z(j);
        }
    }

    return SVM_Binary::qaddTrainingVector(i,zz,x,Cweigh,epsweigh,xz);
}

int SVM_Binary::addTrainingVector(int i, int xd, const SparseVector<gentype> &x, double xCweigh, double xepsweigh, double xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Binary::N() );

    binNnc("&",xd+1)++;

    bintrainclass.add(i);
    bintraintarg.add(i);
    binepsweight.add(i);
    binCweight.add(i);
    binCweightfuzz.add(i);

    bintrainclass("&",i)  = xd;
    bintraintarg("&",i)   = xz;
    binepsweight("&",i)   = xepsweigh;
    binCweight("&",i)     = xCweigh;
    binCweightfuzz("&",i) = 1.0;

    int res = SVM_Scalar::addTrainingVector(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)),x,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres),binepsweight(i),DCALC(bintrainclass(i)));

    res |= SVM_Binary::fixautosettings(0,1);

    gentype yn(bintrainclass(i));
    res |= SVM_Generic::sety(i,yn);

    return res;
}

int SVM_Binary::qaddTrainingVector(int i, int xd, SparseVector<gentype> &x, double xCweigh, double xepsweigh, double xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Binary::N() );

    binNnc("&",xd+1)++;

    bintrainclass.add(i);
    bintraintarg.add(i);
    binepsweight.add(i);
    binCweight.add(i);
    binCweightfuzz.add(i);

    bintrainclass("&",i)  = xd;
    bintraintarg("&",i)   = xz;
    binepsweight("&",i)   = xepsweigh;
    binCweight("&",i)     = xCweigh;
    binCweightfuzz("&",i) = 1.0;

    int res = SVM_Scalar::qaddTrainingVector(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)),x,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres),binepsweight(i),DCALC(bintrainclass(i)));

    res |= SVM_Binary::fixautosettings(0,1);

    gentype yn(bintrainclass(i));
    res |= SVM_Generic::sety(i,yn);

    return res;
}

int SVM_Binary::addTrainingVector(int i, const Vector<int> &xd, const Vector<SparseVector<gentype> > &xx, const Vector<double> &xCweigh, const Vector<double> &xepsweigh, const Vector<double> &xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Binary::N() );
    NiceAssert( xd.size() == xx.size() );
    NiceAssert( xd.size() == xCweigh.size() );
    NiceAssert( xd.size() == xepsweigh.size() );
    NiceAssert( xd.size() == xz.size() );

    int res = 0;

    if ( xd.size() )
    {
        int j;

        for ( j = 0 ; j < xd.size() ; j++ )
        {
            binNnc("&",xd(j)+1)++;

            bintrainclass.add(i+j);
            bintraintarg.add(i+j);
            binepsweight.add(i+j);
            binCweight.add(i+j);
            binCweightfuzz.add(i+j);

            bintrainclass("&",i+j)  = xd(j);
            bintraintarg("&",i+j)   = xz(j);
            binepsweight("&",i+j)   = xepsweigh(j);
            binCweight("&",i+j)     = xCweigh(j);
            binCweightfuzz("&",i+j) = 1.0;

            res |= SVM_Scalar::addTrainingVector(i+j,ZCALC(bintraintarg(i+j),bintrainclass(i+j),bineps,binepsclass,bintrainclass(i+j),binepsweight(i+j)),xx(j),CWCALC(bintrainclass(i+j),binCweight(i+j),binCweightfuzz(i+j),dthres),binepsweight(i+j),DCALC(bintrainclass(i+j)));

            gentype yn(bintrainclass(i+j));
            res |= SVM_Generic::sety(i+j,yn);
        }
    }

    res |= SVM_Binary::fixautosettings(0,1);

    return res;
}

int SVM_Binary::qaddTrainingVector(int i, const Vector<int> &xd, Vector<SparseVector<gentype> > &xx, const Vector<double> &xCweigh, const Vector<double> &xepsweigh, const Vector<double> &xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Binary::N() );
    NiceAssert( xd.size() == xx.size() );
    NiceAssert( xd.size() == xCweigh.size() );
    NiceAssert( xd.size() == xepsweigh.size() );
    NiceAssert( xd.size() == xz.size() );

    int res = 0;

    if ( xd.size() )
    {
        int j;

        for ( j = 0 ; j < xd.size() ; j++ )
        {
            binNnc("&",xd(j)+1)++;

            bintrainclass.add(i+j);
            bintraintarg.add(i+j);
            binepsweight.add(i+j);
            binCweight.add(i+j);
            binCweightfuzz.add(i+j);

            bintrainclass("&",i+j)  = xd(j);
            bintraintarg("&",i+j)   = xz(j);
            binepsweight("&",i+j)   = xepsweigh(j);
            binCweight("&",i+j)     = xCweigh(j);
            binCweightfuzz("&",i+j) = 1.0;

            res |= SVM_Scalar::qaddTrainingVector(i+j,ZCALC(bintraintarg(i+j),bintrainclass(i+j),bineps,binepsclass,bintrainclass(i+j),binepsweight(i+j)),xx("&",j),CWCALC(bintrainclass(i+j),binCweight(i+j),binCweightfuzz(i+j),dthres),binepsweight(i+j),DCALC(bintrainclass(i+j)));

            gentype yn(bintrainclass(i+j));
            res |= SVM_Generic::sety(i+j,yn);
        }
    }

    res |= SVM_Binary::fixautosettings(0,1);

    return res;
}

int SVM_Binary::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary::N() );

    binNnc("&",bintrainclass(i)+1)--;

    bintrainclass.remove(i);
    bintraintarg.remove(i);
    binepsweight.remove(i);
    binCweight.remove(i);
    binCweightfuzz.remove(i);

    int res = SVM_Scalar::removeTrainingVector(i,y,x);

    res |= SVM_Binary::fixautosettings(0,1);

    return res;
}

int SVM_Binary::setClassifyViaSVR(void)
{
    int res = 0;

    if ( !isClassifyViaSVR() )
    {
	isSVMviaSVR = 1;
        res |= SVM_Scalar::setRestrictEpsPos();

	int i;

        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            res |= SVM_Scalar::setd(i,DCALC(bintrainclass(i))); // No point using setdinternal here, as autosets in the scalar are not used
            res |= SVM_Scalar::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
            gentype yn(bintrainclass(i));
            res |= SVM_Generic::sety(i,yn);
	}

	if ( isShrinkTube() )
	{
            res |= SVM_Scalar::seteps(bineps);
	}
    }

    return res;
}

int SVM_Binary::setClassifyViaSVM(void)
{
    int res = 0;
    
    if ( !isClassifyViaSVM() )
    {
        isSVMviaSVR = 0;
        res |= SVM_Scalar::setRestrictEpsNeg();

	int i;

        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            res |= SVM_Scalar::setd(i,DCALC(bintrainclass(i))); // No point using setdinternal here, as autosets in the scalar base are not used
            res |= SVM_Scalar::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
            gentype yn(bintrainclass(i));
            res |= SVM_Generic::sety(i,yn);
	}

	if ( isShrinkTube() )
	{
            res |= SVM_Scalar::seteps(-bineps);
	}
    }

    return res;
}

int SVM_Binary::autosetOff(void)
{
    binautosetLevel = 0;

    return 0;
}

int SVM_Binary::autosetCscaled(double Cval)
{
    NiceAssert( Cval > 0 );
    binautosetCval = Cval;
    int res = setC( (SVM_Binary::N()-NNC(0)) ? (Cval/((SVM_Binary::N()-NNC(0)))) : 1.0);
    binautosetLevel = 1;

    return res;
}

int SVM_Binary::autosetCKmean(void)
{
    double diagsum = ( (SVM_Binary::N()-NNC(0)) ? autosetkerndiagmean() : 1 );
    int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 );
    binautosetLevel = 2;

    return res;
}

int SVM_Binary::autosetCKmedian(void)
{
    double diagsum = ( (SVM_Binary::N()-NNC(0)) ? autosetkerndiagmedian() : 1 );
    int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 );
    binautosetLevel = 3;

    return res;
}

int SVM_Binary::autosetCNKmean(void)
{
    double diagsum = ( (SVM_Binary::N()-NNC(0)) ? (SVM_Binary::N()-NNC(0))*autosetkerndiagmean() : 1 );
    int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 );
    binautosetLevel = 4;

    return res;
}

int SVM_Binary::autosetCNKmedian(void)
{
    double diagsum = ( (SVM_Binary::N()-NNC(0)) ? (SVM_Binary::N()-NNC(0))*autosetkerndiagmedian() : 1 );
    int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 );
    binautosetLevel = 5;

    return res;
}

int SVM_Binary::autosetLinBiasForce(double nuval, double Cval)
{
    NiceAssert( ( Cval > 0 ) && ( nuval >= 0.0 ) && ( nuval <= 1.0 ) );
    binautosetnuval = nuval;
    binautosetCval = Cval;
    int res = setC( (SVM_Binary::N()-NNC(0)) ? (Cval/((SVM_Binary::N()-NNC(0))*nuval)) : 1.0);
    // NB: It is important to leave this as setLinBiasForce rather than for
    //     example SVM_Scalar::setLinBiasForce of SVM_Binary::SetLinBiasForce
    //     as SVM_Single polymorphs the setLinBiasForce function to do sign
    //     correction, so all calls for this function must pass through the
    //     relevant polymorph.
    res |= setLinBiasForce(-Cval);
    res |= seteps(0.0);
    binautosetLevel = 6;

    return res;
}

int SVM_Binary::train(int &res, svmvolatile int &killSwitch)
{
    int dobartlett = ( ( dthres > 0 ) && ( dthres < 0.5 ) ) ? 1 : 0;
    int realN = N();
    int fakeN = 0;

    if ( dobartlett && N() )
    {
        int i;

        SparseVector<gentype> xnew;

        for ( i = 0 ; i < realN ; i++ )
        {
            if ( ( d()(i) == -1 ) || ( d()(i) == +1 ) )
            {
                xnew.fff("&",0) = i;

                addTrainingVector(realN+fakeN,d()(i),xnew,1.0,0.0); // eps == 0 for this one, assume z = 0

                fakeN++;
            }
        }
    }

    int modmod = loctrain(res,killSwitch,realN);

    if ( dobartlett )
    {
        SVM_Generic::removeTrainingVector(realN,fakeN);
    }

    return modmod;
}

int SVM_Binary::loctrain(int &res, svmvolatile int &killSwitch, int realN, int assumeDNZ)
{
    int dobartlett = ( ( dthres > 0 ) && ( dthres < 0.5 ) ) ? 1 : 0;

    Vector<double> altalpha;

    if ( dobartlett && N() )
    {
        int fakeN = 0;

        altalpha = alphaR();

        // Need to enforce usual cost on 0 <= y_i.g(x_i) <= 1
        // plus additional cost (Cweightmult) on y_i.g(x_i) <= 0
        //
        // That is (Bar4, section 2, not including existing C and Cweight):
        //
        // phi(z) = 1-z   if 0 <= y_i.g(x_i) <= 1
        //          1-az  if y_i.g(x_i) <= 0
        //        = standard_phi(z) + extra_phi(z)
        //
        // standard_phi(z) = 1-z      if y_i.g(x_i) <= 1
        // extra_phi   (z) = (1-a).z  if y_i.g(x_i) <= 0
        //
        // where: a = (1-d)/d
        //        a-1 = (1-2d)/d
        //
        // We assume that epsilon == 1 here for simplicity

        int i;

        SparseVector<gentype> xnew;

        for ( i = 0 ; i < realN ; i++ )
        {
            if ( assumeDNZ || ( d()(i) == -1 ) || ( d()(i) == +1 ) )
            {
                xnew.fff("&",0) = i;

                SVM_Scalar::setCweight(realN+fakeN,CWCALCEXTRA(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));
                SVM_Scalar::setCweight(i,          CWCALCBASE( bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));

                altalpha("&",realN+fakeN) = altalpha(i)-alphaR()(i);
                altalpha("&",i)           = alphaR()(i);

                fakeN++;
            }
        }
    }

    int modmod = SVM_Scalar::train(res,killSwitch);

    if ( dobartlett )
    {
        int fakeN = 0;

        altalpha = alphaR();

        int i;

        for ( i = 0 ; i < realN ; i++ )
        {
            if ( assumeDNZ || ( d()(i) == -1 ) || ( d()(i) == +1 ) )
            {
                altalpha("&",i) = (SVM_Scalar::alphaR())(i) + (SVM_Scalar::alphaR())(realN+fakeN);

                SVM_Scalar::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));

                fakeN++;
            }
        }

        SVM_Scalar::setAlphaR(altalpha);

        SVM_Scalar::isStateOpt = 1;
    }

    if ( isShrinkTube() )
    {
	if ( isClassifyViaSVR() )
	{
            bineps = SVM_Scalar::eps();
	}

	else
	{
            bineps = -(SVM_Scalar::eps());
	}
    }

    return modmod;
}

int SVM_Binary::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int unusedvar = 0;
    int tempresh = 0;
    double tempresg = 0;

    tempresh = SVM_Scalar::gTrainingVector(tempresg,unusedvar,i,retaltg,pxyprodi);

    if ( ( tempresg > -dthres ) && ( tempresg < dthres ) )
    {
        // Bartlett's "classify with reject option"

        tempresh = 0;
    }

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


int SVM_Binary::fixautosettings(int kernchange, int Nchange)
{
    int res = 0;

    if ( kernchange || Nchange )
    {
        switch ( binautosetLevel )
	{
        case 1: { if ( Nchange ) { res = 1; autosetCscaled(binautosetCval);                      } break; }
        case 2: {                  res = 1; autosetCKmean();                                       break; }
        case 3: {                  res = 1; autosetCKmedian();                                     break; }
        case 4: {                  res = 1; autosetCNKmean();                                      break; }
        case 5: {                  res = 1; autosetCNKmedian();                                    break; }
        case 6: { if ( Nchange ) { res = 1; autosetLinBiasForce(binautosetnuval,binautosetCval); } break; }
	default: { break; }
	}
    }

    return res;
}

double SVM_Binary::autosetkerndiagmean(void)
{
    Vector<int> dnonzero;

    if ( SVM_Binary::N()-NNC(0) )
    {
	int i,j = 0;

        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            if ( bintrainclass(i) != 0 )
	    {
		dnonzero.add(j);
		dnonzero("&",j) = i;

                j++;
	    }
	}
    }

    retVector<double> tmpva;

    return mean((SVM_Scalar::kerndiag())(dnonzero,tmpva));
}

double SVM_Binary::autosetkerndiagmedian(void)
{
    Vector<int> dnonzero;

    int i,j = 0;

    if ( SVM_Binary::N()-NNC(0) )
    {
        for ( i = 0 ; i < SVM_Binary::N() ; i++ )
	{
            if ( bintrainclass(i) != 0 )
	    {
		dnonzero.add(j);
		dnonzero("&",j) = i;

                j++;
	    }
	}
    }

    retVector<double> tmpva;

    return median((SVM_Scalar::kerndiag())(dnonzero,tmpva),i);
}


std::ostream &SVM_Binary::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Binary SVM\n\n";

    repPrint(output,'>',dep) << "epsilon:                         " << bineps                 << "\n";
    repPrint(output,'>',dep) << "d:                               " << bintrainclass          << "\n";
    repPrint(output,'>',dep) << "z:                               " << bintraintarg           << "\n";
    repPrint(output,'>',dep) << "classwise epsilon:               " << binepsclass            << "\n";
    repPrint(output,'>',dep) << "elementwise epsilon:             " << binepsweight           << "\n";
    repPrint(output,'>',dep) << "classwise C:                     " << binCclass              << "\n";
    repPrint(output,'>',dep) << "elementwise C:                   " << binCweight             << "\n";
    repPrint(output,'>',dep) << "elementwise C (fuzz):            " << binCweightfuzz         << "\n";
    repPrint(output,'>',dep) << "SVM as SVR:                      " << isSVMviaSVR            << "\n";
    repPrint(output,'>',dep) << "Nnc:                             " << binNnc                 << "\n";
    repPrint(output,'>',dep) << "Parameter autoset level:         " << binautosetLevel        << "\n";
    repPrint(output,'>',dep) << "Parameter autoset nu value:      " << binautosetnuval        << "\n";
    repPrint(output,'>',dep) << "Parameter autoset C value:       " << binautosetCval         << "\n";
    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base SVR:                        ";
    SVM_Scalar::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_Binary::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> bineps;
    input >> dummy; input >> bintrainclass;
    input >> dummy; input >> bintraintarg;
    input >> dummy; input >> binepsclass;
    input >> dummy; input >> binepsweight;
    input >> dummy; input >> binCclass;
    input >> dummy; input >> binCweight;
    input >> dummy; input >> binCweightfuzz;
    input >> dummy; input >> isSVMviaSVR;
    input >> dummy; input >> binNnc;
    input >> dummy; input >> binautosetLevel;
    input >> dummy; input >> binautosetnuval;
    input >> dummy; input >> binautosetCval;
    input >> dummy;
    SVM_Scalar::inputstream(input);

    return input;
}

int SVM_Binary::prealloc(int expectedN)
{
    bintrainclass.prealloc(expectedN);
    bintraintarg.prealloc(expectedN);
    binepsclass.prealloc(expectedN);
    binepsweight.prealloc(expectedN);
    binCclass.prealloc(expectedN);
    binCweight.prealloc(expectedN);
    binCweightfuzz.prealloc(expectedN);
    SVM_Scalar::prealloc(expectedN);

    return 0;
}

int SVM_Binary::preallocsize(void) const
{
    return SVM_Scalar::preallocsize();
}


