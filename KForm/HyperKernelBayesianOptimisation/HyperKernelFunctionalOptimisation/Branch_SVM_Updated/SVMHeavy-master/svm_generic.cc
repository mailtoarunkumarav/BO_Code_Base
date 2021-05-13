
//
// SVM base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_generic.h"

std::ostream &SVM_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Base training alpha:       " << dalpha      << "\n";
    repPrint(output,'>',dep) << "Base training bias:        " << dbias       << "\n";
    repPrint(output,'>',dep) << "Base training alpha state: " << xalphaState << "\n";

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &SVM_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> dalpha;
    input >> dummy; input >> dbias;
    input >> dummy; input >> xalphaState;

    ML_Base::inputstream(input);

    return input;
}

int SVM_Generic::getInternalClass(const gentype &y) const
{
    int res = 0;

    if ( type() == 0 )
    {
        res = ( ( (double) y ) < 0 ) ? 0 : 1;
    }

    else if ( isClassifier() )
    {
        if ( isanomalyOn() )
        {
            if ( (int) y == anomalyClass() )
            {
                res = numClasses();
            }

            else
            {
                res = findID((int) y);
            }
        }

        else
        {
            res = findID((int) y);
        }

        if ( res == -1 )
        {
            if ( isanomalyOn() || ( ( type() == 3 ) && ( subtype() == 0 ) ) )
            {
                // In this case if we don't recognise a class *but* there is
                // an anomaly class defined *then* register this as an anomaly

                //res = dynamic_cast<const SVM_MultiC_atonce &>(*this).grablinbfq();
                res = grablinbfq();
            }
        }
    }

    return res;
}

int SVM_Generic::setFixedBias(const gentype &newBias)
{
    if ( isUnderlyingVector() )
    {
	Vector<gentype> simplebias((const Vector<gentype> &) newBias);

	int i;

	if ( simplebias.size() )
	{
	    for ( i = 0 ; i < simplebias.size() ; i++ )
	    {
                setFixedBias(i,(double) simplebias(i));
	    }
	}
    }

    else if ( isUnderlyingAnions() )
    {
        d_anion simplebias((const d_anion &) newBias);

        setBiasA(simplebias);
        setFixedBias((double) simplebias(0));
    }

    else
    {
        setFixedBias((double) newBias);
    }

    return 1;
}

int SVM_Generic::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    incgvernum();

    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Generic::N() );

    int res = ML_Base::addTrainingVector(i,y,x,Cweigh,epsweigh);
    dalpha.add(i);
    xalphaState.add(i);

    dalpha("&",i) = 0;
    xalphaState("&",i) = 1;

    return res;
}

int SVM_Generic::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    incgvernum();

    NiceAssert( i >= 0 );
//    NiceAssert( i <= SVM_Generic::N() );

    int res = ML_Base::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    dalpha.add(i);
    xalphaState.add(i);

    dalpha("&",i) = 0;
    xalphaState("&",i) = 1;

    return res;
}

int SVM_Generic::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    incgvernum();

    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Generic::N() );
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = ML_Base::addTrainingVector(i,y,x,Cweigh,epsweigh);

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            dalpha.add(i+j);
            xalphaState.add(i+j);

            dalpha("&",i+j) = 0;
            xalphaState("&",i+j) = 1;
        }
    }

    return res;
}

int SVM_Generic::qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    incgvernum();

    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Generic::N() );
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    int res = ML_Base::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            dalpha.add(i+j);
            xalphaState.add(i+j);

            dalpha("&",i+j) = 0;
            xalphaState("&",i+j) = 1;
        }
    }

    return res;
}

int SVM_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    incgvernum();

    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Generic::N() );

    int res = ML_Base::removeTrainingVector(i,y,x);
    dalpha.remove(i);
    xalphaState.remove(i);

    return res;
}

int SVM_Generic::removeTrainingVector(int i, int num)
{
    int res = 0;
    gentype y;
    SparseVector<gentype> x;

    while ( num )
    {
        num--; 
        res |= removeTrainingVector(i+num,y,x);
    }

    return res;
}







// Kernel transfer

int SVM_Generic::isKVarianceNZ(void) const
{
    return getKernel().isKVarianceNZ();
}

void SVM_Generic::fastg(double &res) const
{
    SparseVector<gentype> x;

    gg(res,x);

    return;
}

void SVM_Generic::fastg(double &res, 
                        int ia, 
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo) const
{
    (void) ia;
    (void) xainfo;

    gg(res,xa);

    return;
}

void SVM_Generic::fastg(double &res, 
                        int ia, int ib, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                        const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    (void) xainfo;
    (void) xbinfo;

    (void) ia;
    (void) ib;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; i++ )
        {
            x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(double &res, 
                        int ia, int ib, int ic, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;

    (void) ia;
    (void) ib;
    (void) ic;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; i++ )
        {
            x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
        }
    }

    if ( xc.indsize() )
    {
        int i;

        for ( i = 0 ; i < xc.indsize() ; i++ )
        {
            x("&",xc.ind(i)+(2*DEFAULT_TUPLE_INDEX_STEP)) = xc.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(double &res, 
                        int ia, int ib, int ic, int id,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;
    (void) xdinfo;

    (void) ia;
    (void) ib;
    (void) ic;
    (void) id;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; i++ )
        {
            x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
        }
    }

    if ( xc.indsize() )
    {
        int i;

        for ( i = 0 ; i < xc.indsize() ; i++ )
        {
            x("&",xc.ind(i)+(2*DEFAULT_TUPLE_INDEX_STEP)) = xc.direcref(i);
        }
    }

    if ( xd.indsize() )
    {
        int i;

        for ( i = 0 ; i < xd.indsize() ; i++ )
        {
            x("&",xd.ind(i)+(3*DEFAULT_TUPLE_INDEX_STEP)) = xd.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(double &res, 
                        Vector<int> &ia, 
                        Vector<const SparseVector<gentype> *> &xa,
                        Vector<const vecInfo *> &xainfo) const
{
    (void) xainfo;
    (void) ia;

    SparseVector<gentype> x;

    if ( xa.size() )
    {
        int i,j;

        for ( j = 0 ; j < xa.size() ; j++ )
        {
            const SparseVector<gentype> &xb = (*(xa(j)));

            if ( xb.indsize() )
            {
                for ( i = 0 ; i < xb.indsize() ; i++ )
                {
                    x("&",xb.ind(i)+(j*DEFAULT_TUPLE_INDEX_STEP)) = xb.direcref(i);
                }
            }
        }
    }

    gg(res,x);

    return;
}


void SVM_Generic::fastg(gentype &res) const
{
    SparseVector<gentype> x;

    gg(res,x);

    return;
}

void SVM_Generic::fastg(gentype &res, 
                        int ia, 
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo) const
{
    (void) ia;
    (void) xainfo;

    gg(res,xa);

    return;
}

void SVM_Generic::fastg(gentype &res, 
                        int ia, int ib, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                        const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    (void) xainfo;
    (void) xbinfo;

    (void) ia;
    (void) ib;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; i++ )
        {
            x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(gentype &res, 
                        int ia, int ib, int ic, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;

    (void) ia;
    (void) ib;
    (void) ic;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; i++ )
        {
            x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
        }
    }

    if ( xc.indsize() )
    {
        int i;

        for ( i = 0 ; i < xc.indsize() ; i++ )
        {
            x("&",xc.ind(i)+(2*DEFAULT_TUPLE_INDEX_STEP)) = xc.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(gentype &res, 
                        int ia, int ib, int ic, int id,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;
    (void) xdinfo;

    (void) ia;
    (void) ib;
    (void) ic;
    (void) id;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; i++ )
        {
            x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
        }
    }

    if ( xc.indsize() )
    {
        int i;

        for ( i = 0 ; i < xc.indsize() ; i++ )
        {
            x("&",xc.ind(i)+(2*DEFAULT_TUPLE_INDEX_STEP)) = xc.direcref(i);
        }
    }

    if ( xd.indsize() )
    {
        int i;

        for ( i = 0 ; i < xd.indsize() ; i++ )
        {
            x("&",xd.ind(i)+(3*DEFAULT_TUPLE_INDEX_STEP)) = xd.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(gentype &res, 
                        Vector<int> &ia, 
                        Vector<const SparseVector<gentype> *> &xa,
                        Vector<const vecInfo *> &xainfo) const
{
    (void) xainfo;
    (void) ia;

    SparseVector<gentype> x;

    if ( xa.size() )
    {
        int i,j;

        for ( j = 0 ; j < xa.size() ; j++ )
        {
            const SparseVector<gentype> &xb = (*(xa(j)));

            if ( xb.indsize() )
            {
                for ( i = 0 ; i < xb.indsize() ; i++ )
                {
                    x("&",xb.ind(i)+(j*DEFAULT_TUPLE_INDEX_STEP)) = xb.direcref(i);
                }
            }
        }
    }

    gg(res,x);

    return;
}




void SVM_Generic::K0xfer(gentype &res, int &minmaxind, int typeis,
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    gentype dummy;
    double dummyR = 0.0;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j,k;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        K2(Kij,j,k,NULL,NULL,NULL,NULL,NULL,resmode);

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        throw("K0xfer 801 not supported.");

                                        break;
                                    }

                                    default:
                                    {
                                        throw("K0xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kij *= alphaR()(j)*alphaR()(k);
                                }
                                        
                                else if ( isUnderlyingVector() )
                                {
                                    Kij *= twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
                                }
                                        
                                else
                                {
                                    Kij *= (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
                                }

                                if ( docheat )
                                {
                                    if ( k < j )
                                    {
                                        res += Kij;
                                        res += Kij;
                                    }

                                    else if ( k == j )
                                    {
                                        res += Kij;
                                    }
                                }

                                else
                                {
                                    res += Kij;
                                }
                            }
                        }
                    }
                }
            }

            break;
        }

        case 802:
        case 812:
        {
            NiceAssert( !(resmode & 0x80 ) );

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kija,Kijb;

                int j,k;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        K1(Kija,j,NULL,NULL,NULL,resmode);
                                        K1(Kijb,k,NULL,NULL,NULL,resmode);

                                        Kija *= Kijb;

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        throw("K0xfer 802 not supported.");

                                        break;
                                    }

                                    case 64:
                                    {
                                        throw("K0xfer precursor second order derivatives not yet implemented.");

                                        break;
                                    }

                                    default:
                                    {
                                        throw("K0xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kija *= alphaR()(j)*alphaR()(k);
                                }
                                        
                                else if ( isUnderlyingVector() )
                                {
                                    Kija *= twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
                                }
                                       
                                else
                                {
                                    Kija *= (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
                                }

                                res += Kija;
                                res += Kija;
                            }
                        }
                    }
                }
            }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K0xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K0xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void SVM_Generic::K0xfer(double &res, int &minmaxind, int typeis,
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype tempa,tempb,tempc;

        K0xfer(tempc,minmaxind,typeis,xdim,densetype,resmode,mlid);

        res = (double) tempc;

        return;
    }

    gentype dummy;
    double dummyR = 0.0;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kij;

                int j,k;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        K2(Kij,j,k,NULL,NULL,NULL,NULL,NULL,resmode);

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        throw("K0xfer 801 not supported.");

                                        break;
                                    }

                                    default:
                                    {
                                        throw("K0xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kij *= alphaR()(j)*alphaR()(k);
                                }
                                        
                                else if ( isUnderlyingVector() )
                                {
                                    Kij *= twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
                                }
                                        
                                else
                                {
                                    Kij *= (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
                                }

                                if ( docheat )
                                {
                                    if ( k < j )
                                    {
                                        res += Kij;
                                        res += Kij;
                                    }

                                    else if ( k == j )
                                    {
                                        res += Kij;
                                    }
                                }

                                else
                                {
                                    res += Kij;
                                }
                            }
                        }
                    }
                }
            }

            break;
        }

        case 802:
        case 812:
        {
            NiceAssert( !(resmode & 0x80 ) );

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kija,Kijb;

                int j,k;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        K1(Kija,j,NULL,NULL,NULL,resmode);
                                        K1(Kijb,k,NULL,NULL,NULL,resmode);

                                        Kija *= Kijb;

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        throw("K0xfer 802 not supported.");

                                        break;
                                    }

                                    case 64:
                                    {
                                        throw("K0xfer precursor second order derivatives not yet implemented.");

                                        break;
                                    }

                                    default:
                                    {
                                        throw("K0xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kija *= alphaR()(j)*alphaR()(k);
                                }
                                        
                                else if ( isUnderlyingVector() )
                                {
                                    Kija *= twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
                                }
                                       
                                else
                                {
                                    Kija *= (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
                                }

                                res += Kija;
                                res += Kija;
                            }
                        }
                    }
                }
            }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K0xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K0xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}






void SVM_Generic::K1xfer(gentype &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, 
                       const vecInfo &xainfo, 
                       int ia, 
                       int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    gentype dummy;
    double dummyR = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }

            Vector<const SparseVector<gentype> *> x(2);
            Vector<const vecInfo *> xinfo(2);
            Vector<int> i(2);

            x("&",zeroint()) = &xa;
            x("&",1)         = NULL;

            xinfo("&",zeroint()) = &xainfo;
            xinfo("&",1)         = NULL;

            i("&",zeroint()) = ia;
            i("&",1)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j;


                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                                                i("&",1) = j;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Km(2,Kij,i,NULL,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        throw("K1xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        throw("K1xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= oneProduct(dummyR,alphaV()(j));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(alpha()(j));
                                                }
                    }
                }
            }

            if ( ia < 0 ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,xa,xainfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K1xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,xa,xainfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K1xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::K1xfer(res,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void SVM_Generic::K1xfer(double &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, 
                       const vecInfo &xainfo, 
                       int ia, 
                       int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype temp;

        K1xfer(temp,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

        return;
    }

    gentype dummy;
    double dummyR = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }

            Vector<const SparseVector<gentype> *> x(2);
            Vector<const vecInfo *> xinfo(2);
            Vector<int> i(2);

            x("&",zeroint()) = &xa;
            x("&",1)         = NULL;

            xinfo("&",zeroint()) = &xainfo;
            xinfo("&",1)         = NULL;

            i("&",zeroint()) = ia;
            i("&",1)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j;


                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                                                i("&",1) = j;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Km(2,Kij,i,NULL,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        throw("K1xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        throw("K1xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= oneProduct(dummyR,alphaV()(j));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(alpha()(j));
                                                }
                    }
                }
            }

            if ( ia < 0 ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,xa,xainfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K1xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,xa,xainfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K1xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::K1xfer(res,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}









void SVM_Generic::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                         const vecInfo &xainfo, const vecInfo &xbinfo,
                         int ia, int ib,
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    gentype dummy;
    double dummyR = 0.0;

    int iacall = ia;
    int ibcall = ib;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42; // ia if type == x1x,x2x,... (ie assuming shared dataset), -42 otherwise
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43; // ib if type == x1x,x2x,... (ie assuming shared dataset), -43 otherwise

    switch ( typeis )
    {
        case 800:
        case 810:
        {
//errstream() << "phantomx 0: " << xa << "," << xb << "\n";
            if ( !resmode && isUnderlyingScalar() && ( ia >= 0 ) )
            {
                res = ( ia == ib ) ? kerndiag()(ia) : Gp()(ia,ib);
            }

            ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            break;
        }

        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                switch ( resmode & 0x7f )
                {
                    case 0:  case 1:  case 2:  case 3:
                    case 4:  case 5:  case 6:  case 7:
                    case 8:  case 9:  case 10: case 11:
                    case 12: case 13: case 14: case 15:
                    {
                        break;
                    }

                    case 16: case 32: case 48:
                    {
                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                        throw("K2xfer 801 not supported.");

                        break;
                    }

                    default:
                    {
                        throw("K2xfer precursor specified resmode undefined at this level.");

                        break;
                    }
                }

                svmvolatile static svm_mutex eyelock;
                svm_mutex_lock(eyelock);

//errstream() << "phantomx 0 gentype (" << mlid << ")\n";
                // This bit of code is speed-critical.  To optimise we cache the
                // direct products of <xa,X,X> (xa with dataset here, twice)
                // as we note that, if for example we are calculating g(x) on an
                // inheritted kernel, at least one of these if practically fixed.
                // Then we can use standard inner-products, which are much quicker,
                // to calculate the K4 evaluation.  We use xa for our cache as we
                // note that g(x) almost always takes the form sum_i alpha_i K(x,xi),
                // so in the inheritance it is xa that gets repeated for each part
                // of this sum and xb that keeps on changing.

                // Cache shared by all MLs

                Matrix<SparseVector<gentype> > &xvdirectProdsFull = (**thisthisthis).gxvdirectProdsFull;
                Matrix<double> &aadirectProdsFull                 = (**thisthisthis).gaadirectProdsFull;
                Vector<int> &prevalphaState                       = (**thisthisthis).gprevalphaState;
                Vector<int> &prevbalphaState                      = (**thisthisthis).gprevbalphaState;

                int &prevxvernum = (**thisthisthis).gprevxvernum;
                int &prevgvernum = (**thisthisthis).gprevgvernum;
                int &prevN       = (**thisthisthis).gprevN;
                int &prevNb      = (**thisthisthis).gprevNb;

                // Initialise variables

                if ( !(allprevxbvernum.isindpresent(mlid)) )
                {
                    ((**thisthisthis).allprevxbvernum)("&",mlid) = -1;
                }

                // Cached values relating to the caller (mlid)

                SparseVector<gentype> &xaprev                     = ((**thisthisthis).allxaprev)("&",mlid);            // previous xa vector (if different then need to recalculate)
                Matrix<SparseVector<gentype> > &xadirectProdsFull = ((**thisthisthis).allxadirectProdsFull)("&",mlid); // pre-calculated direct products between xa, x(j) and x(k)
                SparseVector<gentype> &xnorms                     = ((**thisthisthis).allxnormsgentype)("&",mlid);     // cached evaluations of K(ia,ia), ia >= 0, if done
                int &prevxbvernum                                 = ((**thisthisthis).allprevxbvernum)("&",mlid);      // version number of x

                // Have relevant state, can now continue

                int j,k;

                int detchange = 0;
                int alchange  = 0;
                int xchange   = 0;

                if ( ( prevxbvernum == -1 ) || ( prevxbvernum != xvernum(mlid) ) )
                {
                    // Caller's X has changed, so flush relevant cache

                    xnorms.zero();

                    xadirectProdsFull.resize(N(),N());
                    prevbalphaState = zeroint();

                    prevxbvernum = xvernum(mlid);
//errstream() << "K2xfer: Caller X reset.\n";
                }

                if ( ( ( prevxvernum == -1 ) && ( prevxvernum != xvernum() ) ) || ( ( prevgvernum == -1 ) && ( prevgvernum != gvernum() ) ) )
                {
                    // First call, so everything needs to be calculated

                    prevN  = N();
                    prevNb = N();

                    xvdirectProdsFull.resize(N(),N());
                    xadirectProdsFull.resize(N(),N());

                    aadirectProdsFull.resize(N(),N());

                    prevalphaState.resize(N());
                    prevalphaState = zeroint();

                    prevbalphaState.resize(N());
                    prevbalphaState = zeroint();

                    prevxvernum = xvernum();
                    prevgvernum = gvernum();

                    xnorms.zero();

                    detchange = 1;
                    alchange  = 1;
//errstream() << "K2xfer: First call setup.\n";
                }

                else if ( prevxvernum != xvernum() )
                {
                    // X has changed, so everything needs to be calculated

                    prevN  = N();
                    prevNb = N();

                    xvdirectProdsFull.resize(N(),N());
                    xadirectProdsFull.resize(N(),N());

                    aadirectProdsFull.resize(N(),N());

                    prevalphaState.resize(N());
                    prevalphaState = zeroint();

                    prevbalphaState.resize(N());
                    prevbalphaState = zeroint();

                    prevxvernum = xvernum();
                    prevgvernum = gvernum();

                    xnorms.zero();

                    detchange = 1;
                    alchange  = 1;
//errstream() << "K2xfer: Call to modified X.\n";
                }

                else if ( prevgvernum != gvernum() )
                {
                    // alpha has changed but not X, so alphaState is still the same size, but we need to recalculate some bits

                    prevgvernum = gvernum();

                    xnorms.zero();

                    detchange = 1;
                    alchange  = 1;
//errstream() << "K2xfer: Call to modified alpha.\n";
                }

                xchange = 1;

                //if ( xa == xaprev ) // The following is a marginal speedup for var() calculation in gpr, noting
                // that (a) assignment copies vecID and (b) most often it will be called b.N times for each vector
                if ( ( xa.vecID() == xaprev.vecID() ) || ( ( xa.size() < 10 ) && ( xa == xaprev ) ) )
                {
                    // xa has not changed

                    xchange = 0;
//errstream() << "K2xfer: New x.\n";
                }

                if ( detchange )
                {
//errstream() << "K2xfer: Setup X otimes X product cache\n";
                    // X changed or first call

                    for ( j = 0 ; j < prevN ; j++ )
                    {
                        if ( alphaState()(j) && !prevalphaState(j) )
                        {
                            // compute direct product xa,x(j) on outer loop to avoid repetition

                            for ( k = 0 ; k <= j ; k++ )
                            {
                                if ( alphaState()(k) && !prevalphaState(k) )
                                {
                                    xvdirectProdsFull("&",j,k) =  x(j);
                                    xvdirectProdsFull("&",j,k) *= x(k);

                                    xvdirectProdsFull("&",j,k).makealtcontent();
                                }
                            }
                        }
                    }
                }

                if ( alchange )
                {
//errstream() << "K2xfer: Setup alpha otimes alpha product cache\n";
                    // alpha changed or first call

                    for ( j = 0 ; j < prevN ; j++ )
                    {
                        for ( k = 0 ; k <= j ; k++ )
                        {
                            if ( isUnderlyingScalar() )
                            {
                                aadirectProdsFull("&",j,k) = alphaR()(j)*alphaR()(k);
                            }
                                        
                            else if ( isUnderlyingVector() )
                            {
                                aadirectProdsFull("&",j,k) = twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
                            }

                            else
                            {
                                aadirectProdsFull("&",j,k) = (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
                            }
                        }
                    }
                }

                if ( detchange )
                {
                    retVector<int> tmpva;

                    prevalphaState = alphaState()(0,1,prevN-1,tmpva);
                }

                if ( prevN != N() )
                {
//errstream() << "K2xfer: Extend X otimes X product cache\n";
                    // extend cache if required.

                    xvdirectProdsFull.resize(N(),N());
                    aadirectProdsFull.resize(N(),N());

                    for ( j = prevN ; j < N() ; j++ )
                    {
                        for ( k = 0 ; k <= j ; k++ )
                        {
                            xvdirectProdsFull("&",j,k) =  x(j);
                            xvdirectProdsFull("&",j,k) *= x(k);

                            xvdirectProdsFull("&",j,k).makealtcontent();

                            if ( isUnderlyingScalar() )
                            {
                                aadirectProdsFull("&",j,k) = alphaR()(j)*alphaR()(k);
                            }
                                        
                            else if ( isUnderlyingVector() )
                            {
                                aadirectProdsFull("&",j,k) = twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
                            }

                            else
                            {
                                aadirectProdsFull("&",j,k) = (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
                            }
                        }
                    }

                    retVector<int> tmpva;
                    retVector<int> tmpvb;

                    prevalphaState.resize(N());
                    prevalphaState("&",prevN,1,N()-1,tmpva) = alphaState()(prevN,1,N()-1,tmpvb);

                    prevN = N();
                }

                if ( ( iacall == ibcall ) && ( iacall >= 0 ) && xnorms.isindpresent(iacall) )
                {
                    // This is K(x(ia),x(ib)), and we have calculated it before, so just use cached version

                    res = xnorms(iacall);

                    NiceAssert( !testisvnan(res) );
                    NiceAssert( !testisinf(res) );
                }

                else if ( iacall < 0 )
                {
                    // This is K(x,x(ib)), which is called for all x(ib) in g(x) calculation, so cache xadirectProds on first hit

                    int upastate = 0;
//int actdone = 0;

                    //if ( xchange || detchange ) - commented, as prevbalphaState is also a factor, and the comparision is done in the first loop anyhow
                    {
                        // Expensive, but only required *once* for each g(xa) evaluation!

                        for ( j = 0 ; j < prevNb ; j++ )
                        {
                            upastate |= ( alphaState()(j) != prevbalphaState(j) );

                            if ( alphaState()(j) && ( xchange || !prevbalphaState(j) ) )
                            {
                                // compute direct product xa,x(j) on outer loop to avoid repetition

                                for ( k = 0 ; k <= j ; k++ )
                                {
                                    if ( alphaState()(k) && ( xchange || !prevbalphaState(k) ) )
                                    {
                                        xadirectProdsFull("&",j,k) =  xa;
                                        xadirectProdsFull("&",j,k) *= xvdirectProdsFull(j,k);

                                        xadirectProdsFull("&",j,k).makealtcontent();
//actdone = 1;
                                    }
                                }
                            }
                        }

//if ( actdone ) { errstream() << "!"; }
                        if ( upastate )
                        {
                            retVector<int> tmpva;

                            prevbalphaState = alphaState()(0,1,prevNb-1,tmpva);
                        }

                        if ( xchange )
                        {
                            xaprev = xa; // this will make the vecIDs the same, so the next comparison may be short-circuited
                        }
                    }

                    if ( prevNb != N() )
                    {
//errstream() << "@";
                        // extend cache if required.

                        xadirectProdsFull.resize(N(),N());

                        for ( j = prevNb ; j < N() ; j++ )
                        {
                            for ( k = 0 ; k <= j ; k++ )
                            {
                                xadirectProdsFull("&",j,k) =  xa;
                                xadirectProdsFull("&",j,k) *= xvdirectProdsFull(j,k);

                                xadirectProdsFull("&",j,k).makealtcontent();
                            }
                        }

                        retVector<int> tmpva;
                        retVector<int> tmpvb;

                        prevbalphaState.resize(N());
                        prevbalphaState("&",prevNb,1,N()-1,tmpva) = alphaState()(prevNb,1,N()-1,tmpvb);

                        prevNb = N();
                    }

                    gentype ipres(0.0);
                    gentype kres(0.0);
                    const gentype *pxyprod[2];

                    pxyprod[0] = &ipres; // we will use this to pass the pre-calculated 4-product in directly
                    pxyprod[1] = NULL;

                    for ( j = 0 ; j < N() ; j++ )
                    {
                        // Not cheap, but can't see a way to avoid this

                        if ( alphaState()(j) )
                        {
                            for ( k = 0 ; k <= j ; k++ )
                            {
                                if ( alphaState()(k) )
                                {
                                    twoProductNoConj(ipres,xadirectProdsFull(j,k),xb); // Assume no conjugation for speed here

                                    NiceAssert( !testisvnan(ipres) );
                                    NiceAssert( !testisinf(ipres) );

                                    // We have xyprod, and other norms will be inferred from xinfo.  Note that 
                                    // NULLs are filled at ML_Base level as required, and relevant norms are 
                                    // always cached in xinfo (so only calc once).
                                    //
                                    // We can safely assume that xainfo and xbinfo exist.  xbinfo is probably the
                                    // xinfo() from the calling class, and xainfo is created by g(x).

                                    K4(kres,ia,ib,j,k,pxyprod,&xa,&xb,NULL,NULL,&xainfo,&xbinfo,NULL,NULL,resmode);

                                    if ( j != k )
                                    {
                                        kres *= 2; // there are two identical instances by symmetry, real by assumption and for speed
                                    }

                                    // Scale by alpha and add to result

                                    kres *= aadirectProdsFull(j,k);

                                    NiceAssert( !testisvnan(kres) );
                                    NiceAssert( !testisinf(kres) );

                                    res += kres;
                                }
                            }
                        }
                    }
                }

                else
                {
//errstream() << "*(" << iacall << "," << ibcall << ")*";
                    // This calculation is one-hit, so for speed don't modify caches etc

                    gentype ipres(0.0);
                    gentype kres(0.0);
                    const gentype *pxyprod[2];

                    pxyprod[0] = &ipres; // we will use this to pass the pre-calculated 4-product in directly
                    pxyprod[1] = NULL;

                    static SparseVector<gentype> xout;

                    xout  = xa;
                    xout *= xb;

                    for ( j = 0 ; j < N() ; j++ )
                    {
                        if ( alphaState()(j) )
                        {
                            for ( k = 0 ; k <= j ; k++ )
                            {
                                if ( alphaState()(k) )
                                {
                                    twoProductNoConj(ipres,xvdirectProdsFull(j,k),xout); // Again we assume commutativity etc

                                    NiceAssert( !testisvnan(ipres) );
                                    NiceAssert( !testisinf(ipres) );

                                    K4(kres,ia,ib,j,k,pxyprod,&xa,&xb,NULL,NULL,&xainfo,&xbinfo,NULL,NULL,resmode);

                                    if ( j != k )
                                    {
                                        kres *= 2; // there are two identical instances by symmetry, real by assumption and for speed
                                    }

                                    // Scale by alpha and add to result

                                    kres *= aadirectProdsFull(j,k);

                                    NiceAssert( !testisvnan(kres) );
                                    NiceAssert( !testisinf(kres) );

                                    res += kres;
                                }
                            }
                        }
                    }

                    if ( iacall == ibcall )
                    {
                        xnorms("&",iacall) = res;
                    }

                    NiceAssert( !testisvnan(res) );
                    NiceAssert( !testisinf(res) );
                }

//errstream() << "phantomx 0 gentype res = " << res << "\n";
                svm_mutex_unlock(eyelock);

//                gentype Kij;
//
//                int j,k;
//                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;
//
//                for ( j = 0 ; j < N() ; j++ )
//                {
//                    if ( alphaState()(j) )
//                    {
//                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; k++ )
//                        {
//                            if ( alphaState()(k) )
//                            {
//                                K4(Kij,ia,ib,j,k,NULL,&xa,&xb,NULL,NULL,&xainfo,&xbinfo,NULL,NULL,resmode);
//
//                                if ( isUnderlyingScalar() )
//                                {
//                                    Kij *= alphaR()(j)*alphaR()(k);
//                                }
//                                        
//                                else if ( isUnderlyingVector() )
//                                {
//                                    Kij *= twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
//                                }
//                                        
//                                else
//                                {
//                                    Kij *= (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
//                                }
//
//                                if ( docheat )
//                                {
//                                    if ( k < j )
//                                    {
//                                        res += Kij;
//                                        res += Kij;
//                                    }
//
//                                    else if ( k == j )
//                                    {
//                                        res += Kij;
//                                    }
//                                }
//
//                                else
//                                {
//                                    res += Kij;
//                                }
//                            }
//                        }
//                    }
//                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 802:
        case 812:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            NiceAssert( !(resmode & 0x80 ) );

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kija,Kijb;

                int j,k;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        K2(Kija,ia,j,NULL,&xa,NULL,&xainfo,NULL,resmode);
                                        K2(Kijb,ib,k,NULL,&xb,NULL,&xbinfo,NULL,resmode);

                                        Kija *= Kijb;

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        throw("K2xfer 802 not supported.");

                                        break;
                                    }

                                    case 64:
                                    {
                                        throw("K2xfer precursor second order derivatives not yet implemented.");

                                        break;
                                    }

                                    default:
                                    {
                                        throw("K2xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kija *= alphaR()(j)*alphaR()(k);
                                }
                                        
                                else if ( isUnderlyingVector() )
                                {
                                    Kija *= twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
                                }
                                       
                                else
                                {
                                    Kija *= (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
                                }

                                res += Kija;
                                res += Kija;
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,xa,xb,xainfo,xbinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K2xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,xa,xb,xainfo,xbinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K2xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            break;
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return;
}

void SVM_Generic::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                         const vecInfo &xainfo, const vecInfo &xbinfo,
                         int ia, int ib,
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype tempa,tempb,tempc;

        K2xfer(tempa,tempb,tempc,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

        dxyprod = (double) tempa;
        ddiffis = (double) tempb;

        res = (double) tempc;

        NiceAssert( !testisvnan(res) );
        NiceAssert( !testisinf(res) );

        return;
    }

    gentype dummy;
    double dummyR = 0.0;

    if ( ( ia >= 0 ) && ( ib < 0 ) )
    {
        // To satisfy later assumptions

        K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xb,xa,xbinfo,xainfo,ib,ia,xdim,densetype,resmode,mlid);

        NiceAssert( !testisvnan(res) );
        NiceAssert( !testisinf(res) );

        return;
    }

    int iacall = ia;
    int ibcall = ib;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( !resmode && isUnderlyingScalar() && ( ia >= 0 ) )
            {
                res = ( ia == ib ) ? kerndiag()(ia) : Gp()(ia,ib);
            }

//errstream() << "phantomx 1\n";
            ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            break;
        }

        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                switch ( resmode & 0x7f )
                {
                    case 0:  case 1:  case 2:  case 3:
                    case 4:  case 5:  case 6:  case 7:
                    case 8:  case 9:  case 10: case 11:
                    case 12: case 13: case 14: case 15:
                    {
                        break;
                    }

                    case 16: case 32: case 48:
                    {
                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                        throw("K2xfer 801 not supported.");

                        break;
                    }

                    default:
                    {
                        throw("K2xfer precursor specified resmode undefined at this level.");

                        break;
                    }
                }

                svmvolatile static svm_mutex eyelock;
                svm_mutex_lock(eyelock);

                // duplicate of gentype, with gentype -> double

//errstream() << "phantomx K2xfer 0 double (" << mlid << ") = ";
                // This bit of code is speed-critical.  To optimise we cache the
                // direct products of <xa,X,X> (xa with dataset here, twice)
                // as we note that, if for example we are calculating g(x) on an
                // inheritted kernel, at least one of these if practically fixed.
                // Then we can use standard inner-products, which are much quicker,
                // to calculate the K4 evaluation.  We use xa for our cache as we
                // note that g(x) almost always takes the form sum_i alpha_i K(x,xi),
                // so in the inheritance it is xa that gets repeated for each part
                // of this sum and xb that keeps on changing.

                // Cache shared by all MLs

                Matrix<SparseVector<gentype> > &xvdirectProdsFull = (**thisthisthis).gxvdirectProdsFull;
                Matrix<double> &aadirectProdsFull                 = (**thisthisthis).gaadirectProdsFull;
                Vector<int> &prevalphaState                       = (**thisthisthis).gprevalphaState;
                Vector<int> &prevbalphaState                      = (**thisthisthis).gprevbalphaState;

                int &prevxvernum = (**thisthisthis).gprevxvernum;
                int &prevgvernum = (**thisthisthis).gprevgvernum;
                int &prevN       = (**thisthisthis).gprevN;
                int &prevNb      = (**thisthisthis).gprevNb;

                // Initialise variables

                if ( !(allprevxbvernum.isindpresent(mlid)) )
                {
                    ((**thisthisthis).allprevxbvernum)("&",mlid) = -1;
                }

                // Cached values relating to the caller (mlid)

                SparseVector<gentype> &xaprev                     = ((**thisthisthis).allxaprev)("&",mlid);            // previous xa vector (if different then need to recalculate)
                Matrix<SparseVector<gentype> > &xadirectProdsFull = ((**thisthisthis).allxadirectProdsFull)("&",mlid); // pre-calculated direct products between xa, x(j) and x(k)
                SparseVector<double> &xnorms                      = ((**thisthisthis).allxnormsdouble)("&",mlid);      // cached evaluations of K(ia,ia), ia >= 0, if done
                int &prevxbvernum                                 = ((**thisthisthis).allprevxbvernum)("&",mlid);      // version number of x

                // Setup, let's go

                int j,k;

                int detchange = 0;
                int alchange  = 0;
                int xchange   = 0;
//errstream() << "phantomx K2xfer 4\n";

                if ( ( prevxbvernum == -1 ) || ( prevxbvernum != xvernum(mlid) ) )
                {
//errstream() << "phantomx K2xfer 5\n";
                    // Caller's X has changed, so flush relevant cache

                    xnorms.zero();

                    xadirectProdsFull.resize(N(),N());
                    prevbalphaState = zeroint();

                    prevxbvernum = xvernum(mlid);
//errstream() << "K2xfer: Caller X reset.\n";
                }

                if ( ( ( prevxvernum == -1 ) && ( prevxvernum != xvernum() ) ) || ( ( prevgvernum == -1 ) && ( prevgvernum != gvernum() ) ) )
                {
//errstream() << "phantomx K2xfer 6\n";
                    // First call, so everything needs to be calculated

                    prevN  = N();
                    prevNb = N();

                    xvdirectProdsFull.resize(N(),N());
                    xadirectProdsFull.resize(N(),N());

                    aadirectProdsFull.resize(N(),N());

                    prevalphaState.resize(N());
                    prevalphaState = zeroint();

                    prevbalphaState.resize(N());
                    prevbalphaState = zeroint();

                    prevxvernum = xvernum();
                    prevgvernum = gvernum();

                    xnorms.zero();

                    detchange = 1;
                    alchange  = 1;
//errstream() << "K2xfer: First call setup.\n";
                }

                else if ( prevxvernum != xvernum() )
                {
//errstream() << "phantomx K2xfer 7\n";
                    // X has changed, so everything needs to be calculated

                    prevN  = N();
                    prevNb = N();

                    xvdirectProdsFull.resize(N(),N());
                    xadirectProdsFull.resize(N(),N());

                    aadirectProdsFull.resize(N(),N());

                    prevalphaState.resize(N());
                    prevalphaState = zeroint();

                    prevbalphaState.resize(N());
                    prevbalphaState = zeroint();

                    prevxvernum = xvernum();
                    prevgvernum = gvernum();

                    xnorms.zero();

                    detchange = 1;
                    alchange  = 1;
//errstream() << "K2xfer: Call to modified X.\n";
                }

                else if ( prevgvernum != gvernum() )
                {
//errstream() << "phantomx K2xfer 8\n";
                    // alpha has changed but not X, so alphaState is still the same size, but we need to recalculate some bits

                    prevgvernum = gvernum();

                    xnorms.zero();

                    detchange = 1;
                    alchange  = 1;
//errstream() << "K2xfer: Call to modified alpha.\n";
                }

                xchange = 1;

                //if ( xa == xaprev ) // The following is a marginal speedup for var() calculation in gpr, noting
                // that (a) assignment copies vecID and (b) most often it will be called b.N times for each vector
                if ( ( xa.vecID() == xaprev.vecID() ) || ( ( xa.size() < 10 ) && ( xa == xaprev ) ) )
                {
                    // xa has not changed

                    xchange = 0;
//errstream() << "K2xfer: New x.\n";
                }

                if ( detchange )
                {
//errstream() << "phantomx K2xfer 10\n";
//errstream() << "K2xfer: Setup X otimes X product cache\n";
                    // X changed or first call

                    for ( j = 0 ; j < prevN ; j++ )
                    {
                        if ( alphaState()(j) && !prevalphaState(j) )
                        {
                            // compute direct product xa,x(j) on outer loop to avoid repetition

                            for ( k = 0 ; k <= j ; k++ )
                            {
                                if ( alphaState()(k) && !prevalphaState(k) )
                                {
                                    xvdirectProdsFull("&",j,k) =  x(j);
                                    xvdirectProdsFull("&",j,k) *= x(k);

                                    xvdirectProdsFull("&",j,k).makealtcontent();
                                }
                            }
                        }
                    }
                }

                if ( alchange )
                {
//errstream() << "K2xfer: Setup alpha otimes alpha product cache\n";
                    // alpha changed or first call

                    for ( j = 0 ; j < prevN ; j++ )
                    {
                        for ( k = 0 ; k <= j ; k++ )
                        {
                            if ( isUnderlyingScalar() )
                            {
                                aadirectProdsFull("&",j,k) = alphaR()(j)*alphaR()(k);
                            }
                                        
                            else if ( isUnderlyingVector() )
                            {
                                aadirectProdsFull("&",j,k) = twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
                            }

                            else
                            {
                                aadirectProdsFull("&",j,k) = (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
                            }
                        }
                    }
                }

                if ( detchange )
                {
                    retVector<int> tmpva;

                    prevalphaState = alphaState()(0,1,prevN-1,tmpva);
                }

                if ( prevN != N() )
                {
//errstream() << "phantomx K2xfer 11\n";
//errstream() << "K2xfer: Extend X otimes X product cache\n";
                    // extend cache if required.

                    xvdirectProdsFull.resize(N(),N());
                    aadirectProdsFull.resize(N(),N());

                    for ( j = prevN ; j < N() ; j++ )
                    {
                        for ( k = 0 ; k <= j ; k++ )
                        {
                            xvdirectProdsFull("&",j,k) =  x(j);
                            xvdirectProdsFull("&",j,k) *= x(k);

                            xvdirectProdsFull("&",j,k).makealtcontent();

                            if ( isUnderlyingScalar() )
                            {
                                aadirectProdsFull("&",j,k) = alphaR()(j)*alphaR()(k);
                            }
                                        
                            else if ( isUnderlyingVector() )
                            {
                                aadirectProdsFull("&",j,k) = twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
                            }

                            else
                            {
                                aadirectProdsFull("&",j,k) = (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
                            }
                        }
                    }

                    retVector<int> tmpva;
                    retVector<int> tmpvb;

                    prevalphaState.resize(N());
                    prevalphaState("&",prevN,1,N()-1,tmpva) = alphaState()(prevN,1,N()-1,tmpvb);

                    prevN = N();
                }

                if ( ( iacall == ibcall ) && ( iacall >= 0 ) && xnorms.isindpresent(iacall) )
                {
//errstream() << "phantomx K2xfer 12\n";
                    // This is K(x(ia),x(ib)), and we have calculated it before, so just use cached version

                    res = xnorms(iacall);

                    NiceAssert( !testisvnan(res) );
                    NiceAssert( !testisinf(res) );
                }

                else if ( iacall < 0 )
                {
                    // This is K(x,x(ib)), which is called for all x(ib) in g(x) calculation, so cache xadirectProds on first hit
//errstream() << "phantomx K2xfer 13\n";

                    int upastate = 0;
//int actdone = 0;

                    //if ( xchange || detchange ) - commented, as prevbalphaState is also a factor, and the comparision is done in the first loop anyhow
                    {
                        // Expensive, but only required *once* for each g(xa) evaluation!

//errstream() << "phantomx K2xfer 13a: " << prevNb << "\n";
                        for ( j = 0 ; j < prevNb ; j++ )
                        {
//errstream() << "phantomx K2xfer 13b: " << alphaState().size() << "\n";
//errstream() << "phantomx K2xfer 13c: " << prevbalphaState.size() << "\n";
                            upastate |= ( alphaState()(j) != prevbalphaState(j) );

//errstream() << "phantomx K2xfer 13d: " << upastate << "\n";
//errstream() << "phantomx K2xfer 13e: " << xchange << "\n";
                            if ( alphaState()(j) && ( xchange || !prevbalphaState(j) ) )
                            {
                                // compute direct product xa,x(j) on outer loop to avoid repetition

//errstream() << "phantomx K2xfer 13f\n";
                                for ( k = 0 ; k <= j ; k++ )
                                {
//errstream() << "phantomx K2xfer 13g: " << j << "," << k << "\n";
                                    if ( alphaState()(k) && ( xchange || !prevbalphaState(k) ) )
                                    { 
//errstream() << "phantomx K2xfer 13h: " << xadirectProdsFull.numRows() << "\n";
//errstream() << "phantomx K2xfer 13i: " << xadirectProdsFull.numCols() << "\n";
//errstream() << "phantomx K2xfer 13h.2: " << xvdirectProdsFull.numRows() << "\n";
//errstream() << "phantomx K2xfer 13i.2: " << xvdirectProdsFull.numCols() << "\n";
                                        xadirectProdsFull("&",j,k) =  xa;
//errstream() << "phantomx K2xfer 13j\n";
                                        xadirectProdsFull("&",j,k) *= xvdirectProdsFull(j,k);

                                        xadirectProdsFull("&",j,k).makealtcontent();
//errstream() << "phantomx K2xfer 13l\n";
//actdone = 1;
                                    }
//errstream() << "phantomx K2xfer 13m\n";
                                }
//errstream() << "phantomx K2xfer 13n\n";
                            }
//errstream() << "phantomx K2xfer 13o\n";
                        }
//errstream() << "phantomx K2xfer 13p\n";

//if ( actdone ) { errstream() << "!"; }
                        if ( upastate )
                        {
//errstream() << "phantomx K2xfer 13q\n";
                            retVector<int> tmpva;

                            prevbalphaState = alphaState()(0,1,prevNb-1,tmpva);
//errstream() << "phantomx K2xfer 13r\n";
                        }

                        if ( xchange )
                        {
                            xaprev = xa; // this will make the vecIDs the same, so the next comparison may be short-circuited
                        }
                    }

                    if ( prevNb != N() )
                    {
//errstream() << "@ blahto";
                        // extend cache if required.

                        xadirectProdsFull.resize(N(),N());

                        for ( j = prevNb ; j < N() ; j++ )
                        {
                            for ( k = 0 ; k <= j ; k++ )
                            {
                                xadirectProdsFull("&",j,k) =  xa;
                                xadirectProdsFull("&",j,k) *= xvdirectProdsFull(j,k);

                                xadirectProdsFull("&",j,k).makealtcontent();
                            }
                        }

                        retVector<int> tmpva;
                        retVector<int> tmpvb;

                        prevbalphaState.resize(N());
                        prevbalphaState("&",prevNb,1,N()-1,tmpva) = alphaState()(prevNb,1,N()-1,tmpvb);

                        prevNb = N();
                    }

//errstream() << "phantomx K2xfer 20\n";
                    gentype ipres(0.0);
                    double kres = 0.0;
                    const gentype *pxyprod[2];

                    pxyprod[0] = &ipres; // we will use this to pass the pre-calculated 4-product in directly
                    pxyprod[1] = NULL;

//errstream() << "phantomx K2xfer 21\n";
                    for ( j = 0 ; j < N() ; j++ )
                    {
                        // Not cheap, but can't see a way to avoid this

                        if ( alphaState()(j) )
                        {
                            for ( k = 0 ; k <= j ; k++ )
                            {
                                if ( alphaState()(k) )
                                {
                                    twoProductNoConj(ipres,xadirectProdsFull(j,k),xb); // Assume no conjugation for speed here

                                    NiceAssert( !testisvnan(ipres) );
                                    NiceAssert( !testisinf(ipres) );

                                    // We have xyprod, and other norms will be inferred from xinfo.  Note that 
                                    // NULLs are filled at ML_Base level as required, and relevant norms are 
                                    // always cached in xinfo (so only calc once).
                                    //
                                    // We can safely assume that xainfo and xbinfo exist.  xbinfo is probably the
                                    // xinfo() from the calling class, and xainfo is created by g(x).

                                    K4(kres,ia,ib,j,k,pxyprod,&xa,&xb,NULL,NULL,&xainfo,&xbinfo,NULL,NULL,resmode);

                                    NiceAssert( !testisvnan(kres) );
                                    NiceAssert( !testisinf(kres) );

                                    if ( j != k )
                                    {
                                        kres *= 2; // there are two identical instances by symmetry, real by assumption and for speed
                                    }

                                    // Scale by alpha and add to result

                                    kres *= aadirectProdsFull(j,k);

                                    NiceAssert( !testisvnan(kres) );
                                    NiceAssert( !testisinf(kres) );

                                    res += kres;
                                }
                            }
                        }
                    }

                    NiceAssert( !testisvnan(res) );
                    NiceAssert( !testisinf(res) );
//errstream() << "phantomx K2xfer 22\n";
                }

                else
                {
//errstream() << "*(" << iacall << "," << ibcall << ")*";
                    // This calculation is one-hit, so for speed don't modify caches etc

                    gentype ipres(0.0);
                    double kres = 0.0;
                    const gentype *pxyprod[2];

                    pxyprod[0] = &ipres; // we will use this to pass the pre-calculated 4-product in directly
                    pxyprod[1] = NULL;

                    static SparseVector<gentype> xout;

                    xout  = xa;
                    xout *= xb;

                    for ( j = 0 ; j < N() ; j++ )
                    {
                        if ( alphaState()(j) )
                        {
                            for ( k = 0 ; k <= j ; k++ )
                            {
                                if ( alphaState()(k) )
                                {
                                    twoProductNoConj(ipres,xvdirectProdsFull(j,k),xout); // Again we assume commutativity etc

                                    NiceAssert( !testisvnan(ipres) );
                                    NiceAssert( !testisinf(ipres) );

                                    K4(kres,ia,ib,j,k,pxyprod,&xa,&xb,NULL,NULL,&xainfo,&xbinfo,NULL,NULL,resmode);

                                    NiceAssert( !testisvnan(kres) );
                                    NiceAssert( !testisinf(kres) );

                                    if ( j != k )
                                    {
                                        kres *= 2; // there are two identical instances by symmetry, real by assumption and for speed
                                    }

                                    // Scale by alpha and add to result

                                    kres *= aadirectProdsFull(j,k);

                                    NiceAssert( !testisvnan(kres) );
                                    NiceAssert( !testisinf(kres) );

                                    res += kres;
                                }
                            }
                        }
                    }

                    NiceAssert( !testisvnan(res) );
                    NiceAssert( !testisinf(res) );

                    if ( iacall == ibcall )
                    {
                        xnorms("&",iacall) = res;
                    }
                }

//errstream() << "phantomx 999 K(" << xa << "," << xb << ") = " << res << "\n";
                svm_mutex_unlock(eyelock);

//                double Kij;
//
//                int j,k;
//                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;
//
//                for ( j = 0 ; j < N() ; j++ )
//                {
//                    if ( alphaState()(j) )
//                    {
//                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; k++ )
//                        {
//                            if ( alphaState()(k) )
//                            {
//                                switch ( resmode & 0x7f )
//                                {
//                                    case 0:  case 1:  case 2:  case 3:
//                                    case 4:  case 5:  case 6:  case 7:
//                                    case 8:  case 9:  case 10: case 11:
//                                    case 12: case 13: case 14: case 15:
//                                    {
//                                        K4(Kij,ia,ib,j,k,NULL,&xa,&xb,NULL,NULL,&xainfo,&xbinfo,NULL,NULL,resmode);
//
//                                        break;
//                                    }
//
//                                    case 16: case 32: case 48:
//                                    {
//                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return
//
//                                        throw("K2xfer 801 not supported.");
//
//                                        break;
//                                    }
//
//                                    default:
//                                    {
//                                        throw("K2xfer precursor specified resmode undefined at this level.");
//
//                                        break;
//                                    }
//                                }
//
//                                if ( isUnderlyingScalar() )
//                                {
//                                    Kij *= alphaR()(j)*alphaR()(k);
//                                }
//                                        
//                                else if ( isUnderlyingVector() )
//                                {
//                                    Kij *= twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
//                                }
//                                        
//                                else
//                                {
//                                    Kij *= (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
//                                }
//
//                                if ( docheat )
//                                {
//                                    if ( k < j )
//                                    {
//                                        res += Kij;
//                                        res += Kij;
//                                    }
//
//                                    else if ( k == j )
//                                    {
//                                        res += Kij;
//                                    }
//                                }
//
//                                else
//                                {
//                                    res += Kij;
//                                }
//                            }
//                        }
//                    }
//                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 802:
        case 812:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            NiceAssert( !(resmode & 0x80 ) );

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kija,Kijb;

                int j,k;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        K2(Kija,ia,j,NULL,&xa,NULL,&xainfo,NULL,resmode);
                                        K2(Kijb,ib,k,NULL,&xb,NULL,&xbinfo,NULL,resmode);

                                        Kija *= Kijb;

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        throw("K2xfer 802 not supported.");

                                        break;
                                    }

                                    case 64:
                                    {
                                        throw("K2xfer precursor second order derivatives not yet implemented.");

                                        break;
                                    }

                                    default:
                                    {
                                        throw("K2xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kija *= alphaR()(j)*alphaR()(k);
                                }
                                        
                                else if ( isUnderlyingVector() )
                                {
                                    Kija *= twoProductNoConj(dummyR,alphaV()(j),alphaV()(k));
                                }
                                       
                                else
                                {
                                    Kija *= (double) real(twoProduct(dummy,alpha()(j),alpha()(k)));
                                }

                                res += Kija;
                                res += Kija;
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,xa,xb,xainfo,xbinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K2xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,xa,xb,xainfo,xbinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K2xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            break;
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return;
}






















void SVM_Generic::K3xfer(gentype &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                       int ia, int ib, int ic, 
                       int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    gentype dummy;
    double dummyR = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }

            Vector<const SparseVector<gentype> *> x(6);
            Vector<const vecInfo *> xinfo(6);
            Vector<int> i(6);

            x("&",zeroint()) = &xa;
            x("&",1)         = &xb;
            x("&",2)         = &xc;
            x("&",3)         = NULL;
            x("&",4)         = NULL;
            x("&",5)         = NULL;

            xinfo("&",zeroint()) = &xainfo;
            xinfo("&",1)         = &xbinfo;
            xinfo("&",2)         = &xcinfo;
            xinfo("&",3)         = NULL;
            xinfo("&",4)         = NULL;
            xinfo("&",5)         = NULL;

            i("&",zeroint()) = ia;
            i("&",1)         = ib;
            i("&",2)         = ic;
            i("&",3)         = 0;
            i("&",4)         = 0;
            i("&",5)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j,k,l;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < ( docheat ? k+1 : N() ) ; l++ )
                                {
                                    if ( alphaState()(l) )
                                    {
                                                i("&",3) = j;
                                                i("&",4) = k;
                                                i("&",5) = l;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Km(6,Kij,i,NULL,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        throw("K3xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        throw("K3xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= threeProduct(dummyR,alphaV()(j),alphaV()(k),alphaV()(l));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(threeProduct(dummy,((const Vector<gentype> &) alpha()(j)),((const Vector<gentype> &) alpha()(k)),((const Vector<gentype> &) alpha()(l))));
                                                }

                                                if ( docheat )
                                                {
                                                    if ( ( j == k ) && ( j == l ) )
                                                    {
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( j != l ) ) ||
                                                              ( ( j == l ) && ( j != k ) ) ||
                                                              ( ( k == l ) && ( k != j ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( j != k ) && ( j != l ) && ( k != l ) )
                                                    {
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                    }
                                                }

                                                else
                                                {
                                                    res += Kij;
                                                }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K3xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K3xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::K3xfer(res,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void SVM_Generic::K3xfer(double &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                       int ia, int ib, int ic, 
                       int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype temp;

        K3xfer(temp,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

        return;
    }

    gentype dummy;
    double dummyR = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }

            Vector<const SparseVector<gentype> *> x(6);
            Vector<const vecInfo *> xinfo(6);
            Vector<int> i(6);

            x("&",zeroint()) = &xa;
            x("&",1)         = &xb;
            x("&",2)         = &xc;
            x("&",3)         = NULL;
            x("&",4)         = NULL;
            x("&",5)         = NULL;

            xinfo("&",zeroint()) = &xainfo;
            xinfo("&",1)         = &xbinfo;
            xinfo("&",2)         = &xcinfo;
            xinfo("&",3)         = NULL;
            xinfo("&",4)         = NULL;
            xinfo("&",5)         = NULL;

            i("&",zeroint()) = ia;
            i("&",1)         = ib;
            i("&",2)         = ic;
            i("&",3)         = 0;
            i("&",4)         = 0;
            i("&",5)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kij;

                int j,k,l;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < ( docheat ? k+1 : N() ) ; l++ )
                                {
                                    if ( alphaState()(l) )
                                    {
                                                i("&",3) = j;
                                                i("&",4) = k;
                                                i("&",5) = l;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Km(6,Kij,i,NULL,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        throw("K3xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        throw("K3xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= threeProduct(dummyR,alphaV()(j),alphaV()(k),alphaV()(l));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(threeProduct(dummy,((const Vector<gentype> &) alpha()(j)),((const Vector<gentype> &) alpha()(k)),((const Vector<gentype> &) alpha()(l))));
                                                }

                                                if ( docheat )
                                                {
                                                    if ( ( j == k ) && ( j == l ) )
                                                    {
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( j != l ) ) ||
                                                              ( ( j == l ) && ( j != k ) ) ||
                                                              ( ( k == l ) && ( k != j ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( j != k ) && ( j != l ) && ( k != l ) )
                                                    {
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                    }
                                                }

                                                else
                                                {
                                                    res += Kij;
                                                }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K3xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K3xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::K3xfer(res,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}






















void SVM_Generic::K4xfer(gentype &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                       int ia, int ib, int ic, int id,
                       int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    gentype dummy;
    double dummyR = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;
    id = (typeis-(100*(typeis/100)))/10 ? id : -45;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            Vector<const SparseVector<gentype> *> x(8);
            Vector<const vecInfo *> xinfo(8);
            Vector<int> i(8);

            x("&",zeroint()) = &xa;
            x("&",1)         = &xb;
            x("&",2)         = &xc;
            x("&",3)         = &xd;
            x("&",4)         = NULL;
            x("&",5)         = NULL;
            x("&",6)         = NULL;
            x("&",7)         = NULL;

            xinfo("&",zeroint()) = &xainfo;
            xinfo("&",1)         = &xbinfo;
            xinfo("&",2)         = &xcinfo;
            xinfo("&",3)         = &xdinfo;
            xinfo("&",4)         = NULL;
            xinfo("&",5)         = NULL;
            xinfo("&",6)         = NULL;
            xinfo("&",7)         = NULL;

            i("&",zeroint()) = ia;
            i("&",1)         = ib;
            i("&",2)         = ic;
            i("&",3)         = id;
            i("&",4)         = 0;
            i("&",5)         = 0;
            i("&",6)         = 0;
            i("&",7)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j,k,l,m;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < ( docheat ? k+1 : N() ) ; l++ )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        for ( m = 0 ; m < ( docheat ? l+1 : N() ) ; m++ )
                                        {
                                            if ( alphaState()(m) )
                                            {
                                                i("&",4) = j;
                                                i("&",5) = k;
                                                i("&",6) = l;
                                                i("&",7) = m;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Km(8,Kij,i,NULL,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        throw("K4xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        throw("K4xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l)*alphaR()(m);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= fourProduct(dummyR,alphaV()(j),alphaV()(k),alphaV()(l),alphaV()(m));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(fourProduct(dummy,alpha()(j),alpha()(k),alpha()(l),alpha()(m)));
                                                }

                                                if ( docheat )
                                                {
                                                    if ( ( j == k ) && ( j == l ) && ( j == m ) )
                                                    {
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( j == l ) && ( j != m ) ) ||
                                                              ( ( k == l ) && ( k == m ) && ( k != j ) ) ||
                                                              ( ( l == m ) && ( l == j ) && ( l != k ) ) ||
                                                              ( ( m == j ) && ( m == k ) && ( m != l ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( l != m ) && ( j != l ) ) ||
                                                              ( ( j == l ) && ( k != m ) && ( j != k ) ) ||
                                                              ( ( j == m ) && ( k != l ) && ( j != k ) ) ||
                                                              ( ( k == l ) && ( j != m ) && ( k != j ) ) ||
                                                              ( ( k == m ) && ( j != l ) && ( k != j ) ) ||
                                                              ( ( l == m ) && ( j != k ) && ( l != j ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( j != k ) && ( j != l ) && ( j != m ) && ( k != l ) && ( k != m ) && ( l != m ) )
                                                    {
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                    }
                                                }

                                                else
                                                {
                                                    res += Kij;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 802:
        case 812:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            NiceAssert( !( resmode & 0x80 ) );

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                gentype Kijaa,Kijab;
                gentype Kijba,Kijbb;
                gentype Kijca,Kijcb;
                gentype Kijda,Kijdb;
                gentype Kijea,Kijeb;
                gentype Kijfa,Kijfb;

                int j,k,l,m;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < N() ; l++ )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        for ( m = 0 ; m < N() ; m++ )
                                        {
                                            if ( alphaState()(m) )
                                            {
                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        K4(Kijaa,ia,ib,j,k,NULL,&xa,&xb,NULL,NULL,&xainfo,&xbinfo,NULL,NULL); K4(Kijab,ic,id,l,m,NULL,&xa,&xb,NULL,NULL,&xainfo,&xbinfo,NULL,NULL);
                                                        K4(Kijba,ia,ic,j,k,NULL,&xa,&xc,NULL,NULL,&xainfo,&xcinfo,NULL,NULL); K4(Kijbb,ib,id,l,m,NULL,&xa,&xc,NULL,NULL,&xainfo,&xcinfo,NULL,NULL);
                                                        K4(Kijca,ia,id,j,k,NULL,&xa,&xd,NULL,NULL,&xainfo,&xdinfo,NULL,NULL); K4(Kijcb,ib,ic,l,m,NULL,&xa,&xd,NULL,NULL,&xainfo,&xdinfo,NULL,NULL);
                                                        K4(Kijda,ib,ic,j,k,NULL,&xb,&xc,NULL,NULL,&xbinfo,&xcinfo,NULL,NULL); K4(Kijdb,ia,id,l,m,NULL,&xb,&xc,NULL,NULL,&xbinfo,&xcinfo,NULL,NULL);
                                                        K4(Kijea,ib,id,j,k,NULL,&xb,&xd,NULL,NULL,&xbinfo,&xdinfo,NULL,NULL); K4(Kijeb,ia,id,l,m,NULL,&xb,&xd,NULL,NULL,&xbinfo,&xdinfo,NULL,NULL);
                                                        K4(Kijfa,ic,id,j,k,NULL,&xc,&xd,NULL,NULL,&xcinfo,&xdinfo,NULL,NULL); K4(Kijfb,ia,ib,l,m,NULL,&xc,&xd,NULL,NULL,&xcinfo,&xdinfo,NULL,NULL);

                                                        Kijaa *= Kijab;
                                                        Kijba *= Kijbb;
                                                        Kijca *= Kijcb;
                                                        Kijda *= Kijdb;
                                                        Kijea *= Kijeb;
                                                        Kijfa *= Kijfb;

                                                        Kij  = Kijaa;
                                                        Kij += Kijba;
                                                        Kij += Kijca;
                                                        Kij += Kijda;
                                                        Kij += Kijea;
                                                        Kij += Kijfa;

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        throw("K4xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        throw("K4xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l)*alphaR()(m);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= fourProduct(dummyR,alphaV()(j),alphaV()(k),alphaV()(l),alphaV()(m));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(fourProduct(dummy,alpha()(j),alpha()(k),alpha()(l),alpha()(m)));
                                                }

                                                res += Kij;
                                                res += Kij;
                                                res += Kij;
                                                res += Kij;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K4xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K4xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::K4xfer(res,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void SVM_Generic::K4xfer(double &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                       int ia, int ib, int ic, int id,
                       int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype temp;

        K4xfer(temp,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

        return;
    }

    gentype dummy;
    double dummyR = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;
    id = (typeis-(100*(typeis/100)))/10 ? id : -45;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            Vector<const SparseVector<gentype> *> x(8);
            Vector<const vecInfo *> xinfo(8);
            Vector<int> i(8);

            x("&",zeroint()) = &xa;
            x("&",1)         = &xb;
            x("&",2)         = &xc;
            x("&",3)         = &xd;
            x("&",4)         = NULL;
            x("&",5)         = NULL;
            x("&",6)         = NULL;
            x("&",7)         = NULL;

            xinfo("&",zeroint()) = &xainfo;
            xinfo("&",1)         = &xbinfo;
            xinfo("&",2)         = &xcinfo;
            xinfo("&",3)         = &xdinfo;
            xinfo("&",4)         = NULL;
            xinfo("&",5)         = NULL;
            xinfo("&",6)         = NULL;
            xinfo("&",7)         = NULL;

            i("&",zeroint()) = ia;
            i("&",1)         = ib;
            i("&",2)         = ic;
            i("&",3)         = id;
            i("&",4)         = 0;
            i("&",5)         = 0;
            i("&",6)         = 0;
            i("&",7)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kij;

                int j,k,l,m;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < ( docheat ? k+1 : N() ) ; l++ )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        for ( m = 0 ; m < ( docheat ? l+1 : N() ) ; m++ )
                                        {
                                            if ( alphaState()(m) )
                                            {
                                                i("&",4) = j;
                                                i("&",5) = k;
                                                i("&",6) = l;
                                                i("&",7) = m;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Km(8,Kij,i,NULL,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        throw("K4xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        throw("K4xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l)*alphaR()(m);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= fourProduct(dummyR,alphaV()(j),alphaV()(k),alphaV()(l),alphaV()(m));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(fourProduct(dummy,alpha()(j),alpha()(k),alpha()(l),alpha()(m)));
                                                }

                                                if ( docheat )
                                                {
                                                    if ( ( j == k ) && ( j == l ) && ( j == m ) )
                                                    {
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( j == l ) && ( j != m ) ) ||
                                                              ( ( k == l ) && ( k == m ) && ( k != j ) ) ||
                                                              ( ( l == m ) && ( l == j ) && ( l != k ) ) ||
                                                              ( ( m == j ) && ( m == k ) && ( m != l ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( l != m ) && ( j != l ) ) ||
                                                              ( ( j == l ) && ( k != m ) && ( j != k ) ) ||
                                                              ( ( j == m ) && ( k != l ) && ( j != k ) ) ||
                                                              ( ( k == l ) && ( j != m ) && ( k != j ) ) ||
                                                              ( ( k == m ) && ( j != l ) && ( k != j ) ) ||
                                                              ( ( l == m ) && ( j != k ) && ( l != j ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( j != k ) && ( j != l ) && ( j != m ) && ( k != l ) && ( k != m ) && ( l != m ) )
                                                    {
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                    }
                                                }

                                                else
                                                {
                                                    res += Kij;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 802:
        case 812:
        {
            NiceAssert( !( resmode & 0x80 ) );

            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kij;

                double Kijaa,Kijab;
                double Kijba,Kijbb;
                double Kijca,Kijcb;
                double Kijda,Kijdb;
                double Kijea,Kijeb;
                double Kijfa,Kijfb;

                int j,k,l,m;

                for ( j = 0 ; j < N() ; j++ )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; k++ )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < N() ; l++ )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        for ( m = 0 ; m < N() ; m++ )
                                        {
                                            if ( alphaState()(m) )
                                            {
                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        K4(Kijaa,ia,ib,j,k,NULL,&xa,&xb,NULL,NULL,&xainfo,&xbinfo,NULL,NULL); K4(Kijab,ic,id,l,m,NULL,&xa,&xb,NULL,NULL,&xainfo,&xbinfo,NULL,NULL);
                                                        K4(Kijba,ia,ic,j,k,NULL,&xa,&xc,NULL,NULL,&xainfo,&xcinfo,NULL,NULL); K4(Kijbb,ib,id,l,m,NULL,&xa,&xc,NULL,NULL,&xainfo,&xcinfo,NULL,NULL);
                                                        K4(Kijca,ia,id,j,k,NULL,&xa,&xd,NULL,NULL,&xainfo,&xdinfo,NULL,NULL); K4(Kijcb,ib,ic,l,m,NULL,&xa,&xd,NULL,NULL,&xainfo,&xdinfo,NULL,NULL);
                                                        K4(Kijda,ib,ic,j,k,NULL,&xb,&xc,NULL,NULL,&xbinfo,&xcinfo,NULL,NULL); K4(Kijdb,ia,id,l,m,NULL,&xb,&xc,NULL,NULL,&xbinfo,&xcinfo,NULL,NULL);
                                                        K4(Kijea,ib,id,j,k,NULL,&xb,&xd,NULL,NULL,&xbinfo,&xdinfo,NULL,NULL); K4(Kijeb,ia,id,l,m,NULL,&xb,&xd,NULL,NULL,&xbinfo,&xdinfo,NULL,NULL);
                                                        K4(Kijfa,ic,id,j,k,NULL,&xc,&xd,NULL,NULL,&xcinfo,&xdinfo,NULL,NULL); K4(Kijfb,ia,ib,l,m,NULL,&xc,&xd,NULL,NULL,&xcinfo,&xdinfo,NULL,NULL);

                                                        Kijaa *= Kijab;
                                                        Kijba *= Kijbb;
                                                        Kijca *= Kijcb;
                                                        Kijda *= Kijdb;
                                                        Kijea *= Kijeb;
                                                        Kijfa *= Kijfb;

                                                        Kij  = Kijaa;
                                                        Kij += Kijba;
                                                        Kij += Kijca;
                                                        Kij += Kijda;
                                                        Kij += Kijea;
                                                        Kij += Kijfa;

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        throw("K4xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        throw("K4xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l)*alphaR()(m);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= fourProduct(dummyR,alphaV()(j),alphaV()(k),alphaV()(l),alphaV()(m));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(fourProduct(dummy,alpha()(j),alpha()(k),alpha()(l),alpha()(m)));
                                                }

                                                res += Kij;
                                                res += Kij;
                                                res += Kij;
                                                res += Kij;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K4xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("K4xfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::K4xfer(res,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}


















void SVM_Generic::Kmxfer(gentype &res, int &minmaxind, int typeis,
                       Vector<const SparseVector<gentype> *> &x,
                       Vector<const vecInfo *> &xinfo,
                       Vector<int> &iii,
                       int xdim, int m, int densetype, int resmode, int mlid) const
{
    if ( ( m == 0 ) || ( m == 1 ) || ( m == 2 ) || ( m == 3 ) || ( m == 4 ) )
    {
        kernPrecursor::Kmxfer(res,minmaxind,typeis,x,xinfo,iii,xdim,m,densetype,resmode,mlid);
        return;
    }

    NiceAssert( !densetype );

    gentype dummy;

    Vector<int> i(iii);

    int iq;

    for ( iq = 0 ; iq < m ; iq++ )
    {
        i("&",iq) = (typeis-(100*(typeis/100)))/10 ? i(iq) : -42-iq;
    }

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            Vector<SparseVector<gentype> > *xx = NULL;

            if ( !( i >= zeroint() ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ir++ )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva; 

                setInnerWildpx(xx); 
                i = cntintvec(m,tmpva);
                i += 1; 
                i *= -100;
            }

            Vector<const SparseVector<gentype> *> xa(2*m);
            Vector<const vecInfo *> xainfo(2*m);
            Vector<int> ia(2*m);

            retVector<const SparseVector<gentype> *> tmpva;
            retVector<const vecInfo *>               tmpvb;
            retVector<int>                           tmpvc;

            xa("&",0,1,m-1,tmpva)     = x;
            xa("&",m,1,(2*m)-1,tmpva) = NULL;

            xainfo("&",0,1,m-1,tmpvb)     = xinfo;
            xainfo("&",m,1,(2*m)-1,tmpvb) = NULL;

            ia("&",0,1,m-1,tmpvc)     = i;
            ia("&",m,1,(2*m)-1,tmpvc) = zeroint();

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j;
                int isdone = 0;
                int isnz;

                while ( !isdone )
                {
                    isnz = 1;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        if ( !alphaState()(ia(m+j)) )
                        {
                            isnz = 0;
                            break;
                        }
                    }

                    if ( isnz )
                    {
                        switch ( resmode & 0x7f )
                        {
                            case 0:  case 1:  case 2:  case 3:
                            case 4:  case 5:  case 6:  case 7:
                            case 8:  case 9:  case 10: case 11:
                            case 12: case 13: case 14: case 15:
                            {
                                Km(2*m,Kij,ia,NULL,&xa,&xainfo,resmode);

                                break;
                            }

                            case 64:
                            {
                                throw("Kmxfer precursor second order derivatives not yet implemented.");

                                break;
                            }

                            default:
                            {
                                throw("Kmxfer precursor specified resmode undefined at this level.");

                                break;
                            }
                        }

                        if ( isUnderlyingScalar() )
                        {
                            for ( j = 0 ; j < m ; j++ )
                            {
                                Kij *= alphaR()(ia(j));
                            }
                        }

                        else
                        {
                            throw("m-products for vectors non pointer blah something.");
                        }

                        res += Kij;
                    }

                    isdone = 1;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        ia("&",m+j)++;

                        if ( ia(m+j) < N() )
                        {
                            isdone = 0;

                            break;
                        }

                        ia("&",m+j) = 0;
                    }
                }
            }

            if ( !( i >= zeroint() ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 802:
        case 812:
        {
            NiceAssert( !( resmode & 0x80 ) );
            NiceAssert( !(m%2) );

            Vector<SparseVector<gentype> > *xx = NULL;

            if ( !( i >= zeroint() ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ir++ )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva; 

                setInnerWildpx(xx); 
                i = cntintvec(m,tmpva);
                i += 1; 
                i *= -100;
            }

            Vector<const SparseVector<gentype> *> xa(2*m);
            Vector<const vecInfo *> xainfo(2*m);
            Vector<int> ia(2*m);

            Vector<const SparseVector<gentype> *> xb(m);
            Vector<const vecInfo *> xbinfo(m);
            Vector<int> ib(m);

            Vector<const SparseVector<gentype> *> xc(m);
            Vector<const vecInfo *> xcinfo(m);
            Vector<int> ic(m);

            Vector<int> k(2*m);

            retVector<const SparseVector<gentype> *> tmpva;
            retVector<const vecInfo *>               tmpvb;
            retVector<int>                           tmpvc;

            xa("&",0,1,m-1,tmpva)     = x;
            xa("&",m,1,(2*m)-1,tmpva) = NULL;

            xainfo("&",0,1,m-1,tmpvb)     = xinfo;
            xainfo("&",m,1,(2*m)-1,tmpvb) = NULL;

            ia("&",0,1,m-1,tmpvc)     = i;
            ia("&",m,1,(2*m)-1,tmpvc) = zeroint();

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kija,Kijb;

                gentype Kij;

                int j,l;
                int isdone = 0;
                int isfill = 0;
//                int ummbongo;
                int isnz,isun;

                while ( !isdone )
                {
                    isnz = 1;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        if ( !alphaState()(ia(m+j)) )
                        {
                            isnz = 0;
                            break;
                        }
                    }

                    if ( isnz )
                    {
                        switch ( resmode & 0x7f )
                        {
                            case 0:  case 1:  case 2:  case 3:
                            case 4:  case 5:  case 6:  case 7:
                            case 8:  case 9:  case 10: case 11:
                            case 12: case 13: case 14: case 15:
                            {
                                k = zeroint();

                                isfill = 0;

                                // Outer loop over all combinations.
                                // Inner loop over all unique combinations

                                while ( !isfill )
                                {
                                    isun = 1;

                                    for ( j = 0 ; j < (2*m)-1 ; j++ )
                                    {
                                        for ( l = j+1 ; l < 2*m ; l++ )
                                        {
                                            if ( k(j) == k(l) )
                                            {
                                                isun = 0;
                                                break;
                                            }
                                        }
                                    }

                                    // Skip repetitions that dont put k halves in order.  This will also ensure NULLs go to right

                                    if ( isun )
                                    {
                                        for ( j = 0 ; j < m-1 ; j++ )
                                        {
                                            if ( ( ib(j) >= ib(j+1) ) || ( ic(j) >= ic(j+1) ) )
                                            {
                                                isun = 0;
                                                break;
                                            }
                                        }
                                    }

                                    // Skip repetitions that don't evenly split NULLs and non-NULLs

                                    if ( isun && ( !xb((m/2)-1) || xb(m/2) || !xc((m/2)-1) || xc(m/2) ) )
                                    {
                                        isun = 0;
                                    }

                                    if ( isun )
                                    {
                                        for ( j = 0 ; j < m ; j++ )
                                        {
                                            xb("&",j)     = xa(k(j));
                                            xbinfo("&",j) = xainfo(k(j));
                                            ib("&",j)     = ia(k(j));

                                            xc("&",j)     = xa(k(j+m));
                                            xcinfo("&",j) = xainfo(k(j+m));
                                            ic("&",j)     = ia(k(j+m));
                                        }

                                        Km(m,Kija,ib,NULL,&xb,&xbinfo,resmode);
                                        Km(m,Kijb,ic,NULL,&xc,&xcinfo,resmode);

                                        Kija *= Kijb;

                                        Kij += Kija;
                                    }

                                    isfill = 1;

                                    for ( j = 0 ; j < 2*m ; j++ )
                                    {
                                        k("&",j)++;

                                        if ( k(j) < 2*m )
                                        {
                                            isfill = 0;

                                            break;
                                        }

                                        k("&",j) = 0;
                                    }
                                }

                                break;
                            }

                            case 64:
                            {
                                throw("Kmxfer precursor second order derivatives not yet implemented.");

                                break;
                            }

                            default:
                            {
                                throw("Kmxfer precursor specified resmode undefined at this level.");

                                break;
                            }
                        }

                        if ( isUnderlyingScalar() )
                        {
                            for ( j = 0 ; j < m ; j++ )
                            {
                                Kij *= alphaR()(ia(j));
                            }
                        }

                        else
                        {
                            throw("m-products for vectors non pointer blah something.");
                        }

                        res += Kij;
                    }

                    isdone = 1;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        ia("&",m+j)++;

                        if ( ia(m+j) < N() )
                        {
                            isdone = 0;

                            break;
                        }

                        ia("&",m+j) = 0;
                    }
                }
            }

            if ( !( i >= zeroint() ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,i,x,xinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("Kmxfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,i,x,xinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("Kmxfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void SVM_Generic::Kmxfer(double &res, int &minmaxind, int typeis,
                       Vector<const SparseVector<gentype> *> &x,
                       Vector<const vecInfo *> &xinfo,
                       Vector<int> &iii,
                       int xdim, int m, int densetype, int resmode, int mlid) const
{
    if ( ( m == 0 ) || ( m == 1 ) || ( m == 2 ) || ( m == 3 ) || ( m == 4 ) )
    {
        kernPrecursor::Kmxfer(res,minmaxind,typeis,x,xinfo,iii,xdim,m,densetype,resmode,mlid);
        return;
    }

    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype temp;

        Kmxfer(temp,minmaxind,typeis,x,xinfo,iii,xdim,m,densetype,resmode,mlid);

        return;
    }

    gentype dummy;

    Vector<int> i(iii);

    int iq;

    for ( iq = 0 ; iq < m ; iq++ )
    {
        i("&",iq) = (typeis-(100*(typeis/100)))/10 ? i(iq) : -42-iq;
    }

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            Vector<SparseVector<gentype> > *xx = NULL;

            if ( !( i >= zeroint() ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ir++ )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva; 

                setInnerWildpx(xx); 
                i = cntintvec(m,tmpva);
                i += 1; 
                i *= -100;
            }

            Vector<const SparseVector<gentype> *> xa(2*m);
            Vector<const vecInfo *> xainfo(2*m);
            Vector<int> ia(2*m);

            retVector<const SparseVector<gentype> *> tmpva;
            retVector<const vecInfo *>               tmpvb;
            retVector<int>                           tmpvc;

            xa("&",0,1,m-1,tmpva)     = x;
            xa("&",m,1,(2*m)-1,tmpva) = NULL;

            xainfo("&",0,1,m-1,tmpvb)     = xinfo;
            xainfo("&",m,1,(2*m)-1,tmpvb) = NULL;

            ia("&",0,1,m-1,tmpvc)     = i;
            ia("&",m,1,(2*m)-1,tmpvc) = zeroint();

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kij = 0.0;

                int j;
                int isdone = 0;
                int isnz;

                while ( !isdone )
                {
                    isnz = 1;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        if ( !alphaState()(ia(m+j)) )
                        {
                            isnz = 0;
                            break;
                        }
                    }

                    if ( isnz )
                    {
                        switch ( resmode & 0x7f )
                        {
                            case 0:  case 1:  case 2:  case 3:
                            case 4:  case 5:  case 6:  case 7:
                            case 8:  case 9:  case 10: case 11:
                            case 12: case 13: case 14: case 15:
                            {
                                Km(2*m,Kij,ia,NULL,&xa,&xainfo,resmode);

                                break;
                            }

                            case 64:
                            {
                                throw("Kmxfer precursor second order derivatives not yet implemented.");

                                break;
                            }

                            default:
                            {
                                throw("Kmxfer precursor specified resmode undefined at this level.");

                                break;
                            }
                        }

                        if ( isUnderlyingScalar() )
                        {
                            for ( j = 0 ; j < m ; j++ )
                            {
                                Kij *= alphaR()(ia(j));
                            }
                        }

                        else
                        {
                            throw("m-products for vectors non pointer blah something.");
                        }

                        res += Kij;
                    }

                    isdone = 1;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        ia("&",m+j)++;

                        if ( ia(m+j) < N() )
                        {
                            isdone = 0;

                            break;
                        }

                        ia("&",m+j) = 0;
                    }
                }
            }

            if ( !( i >= zeroint() ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 802:
        case 812:
        {
            NiceAssert( !( resmode & 0x80 ) );
            NiceAssert( !(m%2) );

            Vector<SparseVector<gentype> > *xx = NULL;

            if ( !( i >= zeroint() ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ir++ )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva; 

                setInnerWildpx(xx); 
                i = cntintvec(m,tmpva);
                i += 1; 
                i *= -100;
            }

            Vector<const SparseVector<gentype> *> xa(2*m);
            Vector<const vecInfo *> xainfo(2*m);
            Vector<int> ia(2*m);

            Vector<const SparseVector<gentype> *> xb(m);
            Vector<const vecInfo *> xbinfo(m);
            Vector<int> ib(m);

            Vector<const SparseVector<gentype> *> xc(m);
            Vector<const vecInfo *> xcinfo(m);
            Vector<int> ic(m);

            Vector<int> k(2*m);

            retVector<const SparseVector<gentype> *> tmpva;
            retVector<const vecInfo *>               tmpvb;
            retVector<int>                           tmpvc;

            xa("&",0,1,m-1,tmpva)     = x;
            xa("&",m,1,(2*m)-1,tmpva) = NULL;

            xainfo("&",0,1,m-1,tmpvb)     = xinfo;
            xainfo("&",m,1,(2*m)-1,tmpvb) = NULL;

            ia("&",0,1,m-1,tmpvc)     = i;
            ia("&",m,1,(2*m)-1,tmpvc) = zeroint();

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kija,Kijb;

                double Kij = 0.0;

                int j,l;
                int isdone = 0;
                int isfill = 0;
                int isnz,isun;

                while ( !isdone )
                {
                    isnz = 1;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        if ( !alphaState()(ia(m+j)) )
                        {
                            isnz = 0;
                            break;
                        }
                    }

                    if ( isnz )
                    {
                        switch ( resmode & 0x7f )
                        {
                            case 0:  case 1:  case 2:  case 3:
                            case 4:  case 5:  case 6:  case 7:
                            case 8:  case 9:  case 10: case 11:
                            case 12: case 13: case 14: case 15:
                            {
                                k = zeroint();

                                isfill = 0;

                                // Outer loop over all combinations.
                                // Inner loop over all unique combinations

                                while ( !isfill )
                                {
                                    isun = 1;

                                    for ( j = 0 ; j < (2*m)-1 ; j++ )
                                    {
                                        for ( l = j+1 ; l < 2*m ; l++ )
                                        {
                                            if ( k(j) == k(l) )
                                            {
                                                isun = 0;
                                                break;
                                            }
                                        }
                                    }

                                    if ( isun )
                                    {
                                        for ( j = 0 ; j < m ; j++ )
                                        {
                                            xb("&",j)     = xa(k(j));
                                            xbinfo("&",j) = xainfo(k(j));
                                            ib("&",j)     = ia(k(j));

                                            xc("&",j)     = xa(k(j+m));
                                            xcinfo("&",j) = xainfo(k(j+m));
                                            ic("&",j)     = ia(k(j+m));
                                        }

                                        Km(m,Kija,ib,NULL,&xb,&xbinfo,resmode);
                                        Km(m,Kijb,ic,NULL,&xc,&xcinfo,resmode);

                                        Kija *= Kijb;

                                        Kij += Kija;
                                    }

                                    isfill = 1;

                                    for ( j = 0 ; j < 2*m ; j++ )
                                    {
                                        k("&",j)++;

                                        if ( k(j) < 2*m )
                                        {
                                            isfill = 0;

                                            break;
                                        }

                                        k("&",j) = 0;
                                    }
                                }

                                break;
                            }

                            case 64:
                            {
                                throw("Kmxfer precursor second order derivatives not yet implemented.");

                                break;
                            }

                            default:
                            {
                                throw("Kmxfer precursor specified resmode undefined at this level.");

                                break;
                            }
                        }

                        if ( isUnderlyingScalar() )
                        {
                            for ( j = 0 ; j < m ; j++ )
                            {
                                Kij *= alphaR()(ia(j));
                            }
                        }

                        else
                        {
                            throw("m-products for vectors non pointer blah something.");
                        }

                        res += Kij;
                    }

                    isdone = 1;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        ia("&",m+j)++;

                        if ( ia(m+j) < N() )
                        {
                            isdone = 0;

                            break;
                        }

                        ia("&",m+j) = 0;
                    }
                }
            }

            if ( !( i >= zeroint() ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 804:
        case 814:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,i,x,xinfo);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("Kmxfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 805:
        case 815:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fastg(res,i,x,xinfo);
                    res *= res;

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    throw("Kmxfer gradient undefined for this resmode.");

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            ML_Base::Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

















void SVM_Generic::basesetAlphaBiasFromAlphaBiasR(void)
{
    incgvernum();

    dalpha.resize(SVM_Generic::N());
    xalphaState.resize(SVM_Generic::N());

    xalphaState = 1;

    if ( SVM_Generic::N() )
    {
        int i;

        for ( i = 0 ; i < SVM_Generic::N() ; i++ )
        {
            dalpha("&",i) = alphaR()(i);
        }
    }

    dbias = biasR();

    return;
}

void SVM_Generic::basesetAlphaBiasFromAlphaBiasV(void)
{
    incgvernum();

    dalpha.resize(SVM_Generic::N());
    xalphaState.resize(SVM_Generic::N());

    xalphaState = 1;

    if ( SVM_Generic::N() )
    {
        int i;

        for ( i = 0 ; i < SVM_Generic::N() ; i++ )
        {
            dalpha("&",i) = alphaV()(i);
        }
    }

    dbias = biasV();

    return;
}

void SVM_Generic::basesetAlphaBiasFromAlphaBiasA(void)
{
    incgvernum();

    dalpha.resize(SVM_Generic::N());
    xalphaState.resize(SVM_Generic::N());

    xalphaState = 1;

    if ( SVM_Generic::N() )
    {
        int i;

        for ( i = 0 ; i < SVM_Generic::N() ; i++ )
        {
            dalpha("&",i) = alphaA()(i);
        }
    }

    dbias = biasA();

    return;
}

int SVM_Generic::removeNonSupports(void)
{
    int i;
    int res = 0;

    gentype y;
    SparseVector<gentype> x;

    while ( NZ() )
    {
	minabs(alphaState(),i);
        res |= removeTrainingVector(i,y,x);
    }

    return res;
}

int SVM_Generic::trimTrainingSet(int maxsize)
{
    NiceAssert( maxsize >= 0 );

    int i;
    int res = 0;

    gentype y;
    SparseVector<gentype> x;

    while ( SVM_Generic::N() > maxsize )
    {
	if ( NZ() )
	{
	    minabs(alphaState(),i);
            res |= removeTrainingVector(i,y,x);
	}

	else
	{
            minabs(alpha(),i);
            res |= removeTrainingVector(i,y,x);
	}
    }

    return res;
}

void SVM_Generic::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    res.resize(SVM_Generic::N());

    if ( gOutType() == 'V' )
    {
        gentype zerotemplate('V');

        zerotemplate.dir_vector().resize(tspaceDim());

        setzero(zerotemplate);

        res  = zerotemplate;
        resn = zerotemplate;

        if ( SVM_Generic::N() )
        {
            int j;
            int dummyind;

            gentype xscale('R');
            gentype yscale('R');

            for ( j = 0 ; j < SVM_Generic::N() ; j++ )
            {
                if ( d()(j) )
                {
                    dK2delx(xscale,yscale,dummyind,i,j);

                    NiceAssert( dummyind < 0 );

                    if ( i >= 0 )
                    {
                        res("&",i).dir_vector().scaleAdd((double) xscale,(const Vector<gentype> &) dalpha(j));
                        res("&",j).dir_vector().scaleAdd((double) yscale,(const Vector<gentype> &) dalpha(j));
                    }

                    else
                    {
                        resn      .dir_vector().scaleAdd((double) xscale,(const Vector<gentype> &) dalpha(j));
                        res("&",j).dir_vector().scaleAdd((double) yscale,(const Vector<gentype> &) dalpha(j));
                    }
                }
            }
        }
    }

    else if ( gOutType() == 'A' )
    {
        gentype zerotemplate('A');

        zerotemplate.dir_anion().setorder(order());

        setzero(zerotemplate);

        res  = zerotemplate;
        resn = zerotemplate;

        if ( SVM_Generic::N() )
        {
            int j;
            int dummyind;

            gentype xscale('R');
            gentype yscale('R');

            for ( j = 0 ; j < SVM_Generic::N() ; j++ )
            {
                if ( d()(j) )
                {
                    dK2delx(xscale,yscale,dummyind,i,j);
        
                    NiceAssert( dummyind < 0 );

                    if ( i >= 0 )
                    {
                        res("&",i).dir_anion() += ((double) xscale)*((const d_anion &) dalpha(j));
                        res("&",j).dir_anion() += ((double) yscale)*((const d_anion &) dalpha(j));
                    }

                    else
                    {
                        resn      .dir_anion() += ((double) xscale)*((const d_anion &) dalpha(j));
                        res("&",j).dir_anion() += ((double) yscale)*((const d_anion &) dalpha(j));
                    }
                }
            }
        }
    }

    else
    {
        gentype zerotemplate(0.0);

        res  = zerotemplate;
        resn = zerotemplate;

        if ( SVM_Generic::N() )
        {
            int j;
            int dummyind = -1;

            gentype xscale('R');
            gentype yscale('R');

            for ( j = 0 ; j < SVM_Generic::N() ; j++ )
            {
                if ( d()(j) )
                {
                    dK2delx(xscale,yscale,dummyind,i,j);
        
                    NiceAssert( dummyind < 0 );

                    if ( i >= 0 )
                    {
                        res("&",i).dir_double() += ((double) xscale)*((double) dalpha(j));
                        res("&",j).dir_double() += ((double) yscale)*((double) dalpha(j));
                    }

                    else
                    {
                        resn      .dir_double() += ((double) xscale)*((double) dalpha(j));
                        res("&",j).dir_double() += ((double) yscale)*((double) dalpha(j));
                    }
                }
            }
        }
    }

    return;
}

int SVM_Generic::setAlpha(const Vector<gentype> &newAlpha)
{
    incgvernum();

    NiceAssert( newAlpha.size() == SVM_Generic::N() );

    // Does not set alpha, relies on function callback to internalsetAlpha

    int res = 0;

    if ( SVM_Generic::N() )
    {
        int i,j;

        if ( isUnderlyingScalar() )
        {
            Vector<double> nAlpha(newAlpha.size());

            for ( i = 0 ; i < SVM_Generic::N() ; i++ )
            {
                nAlpha("&",i) = ((double) newAlpha(i));
            }

            res |= setAlphaR(nAlpha);
        }

        else if ( isUnderlyingVector() )
        {
            Vector<Vector<double> > nAlpha(newAlpha.size());

            for ( i = 0 ; i < SVM_Generic::N() ; i++ )
            {
                nAlpha("&",i).resize(newAlpha(i).size());

                if ( newAlpha(i).size() )
                {
                    const Vector<gentype> &ghgh = (const Vector<gentype> &) newAlpha(i);

                    for ( j = 0 ; j < newAlpha(i).size() ; j++ )
                    {
                        nAlpha("&",i)("&",j) = ghgh(j);
                    }
                }
            }

            res |= setAlphaV(nAlpha);
        }   

        else
        {
            Vector<d_anion> nAlpha(newAlpha.size());

            for ( i = 0 ; i < SVM_Generic::N() ; i++ )
            {
                nAlpha("&",i) = ((const d_anion &) newAlpha(i));
            }

            res |= setAlphaA(nAlpha);
        }
    }

    return res;
}

int SVM_Generic::setBias(const gentype &newBias)
{
    incgvernum();

    // Does not set alpha, relies on function callback to internalsetBias

    int res = 0;

    if ( isUnderlyingScalar() )
    {
        res |= setBiasR(((double) newBias));
    }

    else if ( isUnderlyingVector() )
    {
        int i;

        Vector<double> temp(newBias.size());

        if ( newBias.size() )
        {
            const Vector<gentype> &ghgh = (const Vector<gentype> &) newBias;

            for ( i = 0 ; i < newBias.size() ; i++ )
            {
                temp("&",i) = ghgh(i);
            }
        }

        res |= setBiasV(temp);
    }

    else
    {
        res |= setBiasA((const d_anion &) newBias);
    }

    return res;
}


int SVM_Generic::prealloc(int expectedN)
{
    dalpha.prealloc(expectedN);
    xalphaState.prealloc(expectedN);
    ML_Base::prealloc(expectedN);

    return 0;
}

int SVM_Generic::preallocsize(void) const
{
    return ML_Base::preallocsize();
}




int SVM_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
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

int SVM_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const
{
    int res = 0;

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    int isfallback = 0;

    switch ( ind )
    {
        case 9000: { val = NZ(); break; }
        case 9001: { val = NF(); break; }
        case 9002: { val = NS(); break; }
        case 9003: { val = NC(); break; }
        case 9004: { val = NLB(); break; }
        case 9005: { val = NLF(); break; }
        case 9006: { val = NUF(); break; }
        case 9007: { val = NUB(); break; }
        case 9008: { val = isLinearCost(); break; }
        case 9009: { val = isQuadraticCost(); break; }
        case 9010: { val = is1NormCost(); break; }
        case 9011: { val = isVarBias(); break; }
        case 9012: { val = isPosBias(); break; }
        case 9013: { val = isNegBias(); break; }
        case 9014: { val = isFixedBias(); break; }
        case 9015: { val = isOptActive(); break; }
        case 9016: { val = isOptSMO(); break; }
        case 9017: { val = isOptD2C(); break; }
        case 9018: { val = isOptGrad(); break; }
        case 9019: { val = isFixedTube(); break; }
        case 9020: { val = isShrinkTube(); break; }
        case 9021: { val = isRestrictEpsPos(); break; }
        case 9022: { val = isRestrictEpsNeg(); break; }
        case 9023: { val = isClassifyViaSVR(); break; }
        case 9024: { val = isClassifyViaSVM(); break; }
        case 9025: { val = is1vsA(); break; }
        case 9026: { val = is1vs1(); break; }
        case 9027: { val = isDAGSVM(); break; }
        case 9028: { val = isMOC(); break; }
        case 9029: { val = ismaxwins(); break; }
        case 9030: { val = isrecdiv(); break; }
        case 9031: { val = isatonce(); break; }
        case 9032: { val = isredbin(); break; }
        case 9033: { val = isKreal(); break; }
        case 9034: { val = isKunreal(); break; }
        case 9035: { val = isanomalyOn(); break; }
        case 9036: { val = isanomalyOff(); break; }
        case 9037: { val = isautosetOff(); break; }
        case 9038: { val = isautosetCscaled(); break; }
        case 9039: { val = isautosetCKmean(); break; }
        case 9040: { val = isautosetCKmedian(); break; }
        case 9041: { val = isautosetCNKmean(); break; }
        case 9042: { val = isautosetCNKmedian(); break; }
        case 9043: { val = isautosetLinBiasForce(); break; }
        case 9044: { val = outerlr(); break; }
        case 9045: { val = outermom(); break; }
        case 9046: { val = outermethod(); break; }
        case 9047: { val = outertol(); break; }
        case 9048: { val = outerovsc(); break; }
        case 9049: { val = outermaxitcnt(); break; }
        case 9050: { val = outermaxcache(); break; }
        case 9051: { val = maxiterfuzzt(); break; }
        case 9052: { val = usefuzzt(); break; }
        case 9053: { val = lrfuzzt(); break; }
        case 9054: { val = ztfuzzt(); break; }
        case 9055: { val = costfnfuzzt(); break; }
        case 9056: { val = m(); break; }
        case 9057: { val = LinBiasForce(); break; }
        case 9058: { val = QuadBiasForce(); break; }
        case 9059: { val = nu(); break; }
        case 9060: { val = nuQuad(); break; }
        case 9061: { val = theta(); break; }
        case 9062: { val = simnorm(); break; }
        case 9063: { val = anomalyNu(); break; }
        case 9064: { val = anomalyClass(); break; }
        case 9065: { val = autosetCval(); break; }
        case 9066: { val = autosetnuval(); break; }
        case 9067: { val = anomclass(); break; }
        case 9068: { val = singmethod(); break; }
        case 9069: { val = rejectThreshold(); break; }
        case 9070: { val = Gp(); break; }
        case 9071: { val = XX(); break; }
        case 9072: { val = kerndiag(); break; }
        case 9073: { val = bias(); break; }
        case 9074: { val = alpha(); break; }
        case 9075: { val = quasiloglikelihood(); break; }
        case 9076: { val = isNoMonotonicConstraints(); break; }
        case 9077: { val = isForcedMonotonicIncreasing(); break; }
        case 9078: { val = isForcedMonotonicDecreasing(); break; }

        case 9100: { val = NF((int) xa);  break; }
        case 9101: { val = NZ((int) xa);  break; }
        case 9102: { val = NS((int) xa);  break; }
        case 9103: { val = NC((int) xa);  break; }
        case 9104: { val = NLB((int) xa); break; }
        case 9105: { val = NLF((int) xa); break; }
        case 9106: { val = NUF((int) xa); break; }
        case 9107: { val = NUB((int) xa); break; }
        case 9108: { val = ClassRep()((int) xa); break; }
        case 9109: { val = findID((int) xa); break; }
        case 9110: { val = getu()((int) xa); break; }
        case 9111: { val = isVarBias((int) xa); break; }
        case 9112: { val = isPosBias((int) xa); break; }
        case 9113: { val = isNegBias((int) xa); break; }
        case 9114: { val = isFixedBias((int) xa); break; }
        case 9115: { val = LinBiasForce((int) xa); break; }
        case 9116: { val = QuadBiasForce((int) xa); break; }

        case 9200: { val = Gp()((int) xa, (int) xb); break; }
        case 9201: { val = XX()((int) xa, (int) xb); break; }

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

