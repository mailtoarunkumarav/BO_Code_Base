
//
// Quadratic optimisation context
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "optcontext.h"

#define CALCBETAFIX(iii) ( ( ( GpnFColNorm(iii) <= dzt ) && ( Gn((iii),(iii)) >= -dzt ) ) ? 1 : 0 )

optContext::optContext(void)
{
    thisredirect[0] = this;

    dzt       = DEFAULT_ZTOL;
    dkeepfact = 0;

    daNLF = 0;
    daNUF = 0;

    dnfact = 0;
    dpfact = 0;

    betaFixUpdate = 0;

    return;
}

optContext::optContext(const optContext &src)
{
    thisredirect[0] = this;

    *this = src;

    return;
}

optContext &optContext::operator=(const optContext &src)
{
    pAlphaLB = src.pAlphaLB;
    pAlphaZ  = src.pAlphaZ;
    pAlphaUB = src.pAlphaUB;
    pAlphaF  = src.pAlphaF;

    pBetaC = src.pBetaC;
    pBetaF = src.pBetaF;

    dalphaState = src.dalphaState;
    dbetaState  = src.dbetaState;

    dzt       = src.dzt;
    dkeepfact = src.dkeepfact;

    daNLF = src.daNLF;
    daNUF = src.daNUF;

    GpnFColNorm   = src.GpnFColNorm;
    betaFix       = src.betaFix;
    betaFixUpdate = src.betaFixUpdate;

    dnfact = src.dnfact;
    dpfact = src.dpfact;

    D           = src.D;
    freeVarChol = src.freeVarChol;

    fAlphaF = src.fAlphaF;
    fBetaF  = src.fBetaF;

    return *this;
}

void optContext::refact(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int xkeepfact, double xzt)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( ( xzt > 0 ) || ( xzt == -1 ) );
    NiceAssert( ( xkeepfact == 0 ) || ( xkeepfact == 1 ) || ( xkeepfact == -1 ) );

    if ( dkeepfact && betaFixUpdate )
    {
        fixbetaFix(Gn,Gpn);
    }

    int oldaN = aN();
    int oldbN = bN();
    int oldkeepfact = dkeepfact;

    // Keep current values for keepfact and zt if new values set to -1 ("keep")

    if ( xkeepfact == -1 )
    {
        xkeepfact = dkeepfact;
    }

    if ( xzt < 0 )
    {
        xzt = dzt;
    }

    // Set new values for zt and keepfact

    dzt       = xzt;
    dkeepfact = xkeepfact;

    // Save current pivotting and state

    Vector<int> spAlphaLB(pAlphaLB);
    Vector<int> spAlphaUB(pAlphaUB);
    Vector<int> spAlphaF(pAlphaF);

    Vector<int> spBetaC(pBetaC);
    Vector<int> spBetaF(pBetaF);

    Vector<int> salphaState(dalphaState);

    // Empty pivotting and state.

    pAlphaLB.resize(0);
    pAlphaZ.resize(0);
    pAlphaUB.resize(0);
    pAlphaF.resize(0);

    pBetaC.resize(0);
    pBetaF.resize(0);

    dalphaState.resize(0);
    dbetaState.resize(0);

    // Reset counts of NLF and NUF

    daNLF = 0;
    daNUF = 0;

    // Reset Gpn column totals and beta fixing vector.

    if ( oldkeepfact || dkeepfact )
    {
	GpnFColNorm.resize(0);
	betaFix.resize(0);
    }

    // Reset factorisation counts.

    dnfact = 0;
    dpfact = 0;

    // Reset factorisation

    if ( oldkeepfact || dkeepfact )
    {
	D.resize(0);

        Chol<double> temp(xzt,freeVarChol.fudge());
	freeVarChol = temp;

	fAlphaF.resize(0);
	fBetaF.resize(0);
    }

    int i,iP;

    // Systematically re-add variables

    if ( oldbN )
    {
	for ( i = 0 ; i < oldbN ; i++ )
	{
	    addBeta(i);
	}
    }

    if ( oldaN )
    {
	for ( i = 0 ; i < oldaN ; i++ )
	{
	    addAlpha(i);
	}
    }

    // Restore pivotting and state

    if ( spAlphaLB.size() )
    {
	for ( iP = 0 ; iP < spAlphaLB.size() ; iP++ )
	{
            modAlphaZtoLB(findInAlphaZ(spAlphaLB(iP)));
	}
    }

    if ( spAlphaUB.size() )
    {
	for ( iP = 0 ; iP < spAlphaUB.size() ; iP++ )
	{
            modAlphaZtoUB(findInAlphaZ(spAlphaUB(iP)));
	}
    }

    if ( spAlphaF.size() )
    {
	for ( iP = 0 ; iP < spAlphaF.size() ; iP++ )
	{
	    if ( salphaState(spAlphaF(iP)) < 0 )
	    {
		int aposdummy = 0;
		int bposdummy = 0;

                modAlphaZtoLF(findInAlphaZ(spAlphaF(iP)),Gp,Gn,Gpn,aposdummy,bposdummy);
	    }

	    else
	    {
		int aposdummy = 0;
		int bposdummy = 0;

		modAlphaZtoUF(findInAlphaZ(spAlphaF(iP)),Gp,Gn,Gpn,aposdummy,bposdummy);
	    }
	}
    }

    if ( spBetaF.size() )
    {
	for ( iP = 0 ; iP < spBetaF.size() ; iP++ )
	{
	    int aposdummy = 0;
	    int bposdummy = 0;

            modBetaCtoF(findInBetaC(spBetaF(iP)),Gp,Gn,Gpn,aposdummy,bposdummy);
	}
    }

    return;
}

void optContext::reset(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );

    int apos = 0;
    int bpos = 0;

    while ( aNLB() )
    {
	modAlphaLBtoZ(aNLB()-1); // take from end for computational efficiency
    }

    while ( aNUB() )
    {
	modAlphaUBtoZ(aNUB()-1);
    }

    while ( aNF() )
    {
	if ( dalphaState(pivAlphaF()(aNF()-1)) == -1 )
	{
	    modAlphaLFtoZ(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	}

	else
	{
	    modAlphaUFtoZ(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	}
    }

    while ( bNF() )
    {
	modBetaFtoC(bNF()-1,Gp,Gn,Gpn,apos,bpos);
    }

    return;
}

void optContext::modAlphaFAlltoLowerBound(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );

    while ( aNF() )
    {
	if ( dalphaState(pivAlphaF()(aNF()-1)) == -1 )
	{
	    modAlphaLFtoLB(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	}

	else
	{
	    modAlphaUFtoZ(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	}
    }

    return;
}

void optContext::modAlphaFAlltoUpperBound(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );

    while ( aNF() )
    {
	if ( dalphaState(pivAlphaF()(aNF()-1)) == -1 )
	{
	    modAlphaLFtoZ(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	}

	else
	{
	    modAlphaUFtoUB(aNF()-1,Gp,Gn,Gpn,apos,bpos);
	}
    }

    return;
}

int optContext::addAlpha(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= aN() );

    int j;

    // Fix all pivots to reflect pending increase in number of alphas

    if ( aNLB() )
    {
	for ( j = 0 ; j < aNLB() ; j++ )
	{
	    if ( pAlphaLB(j) >= i )
	    {
                pAlphaLB("&",j)++;
	    }
	}
    }

    if ( aNZ() )
    {
	for ( j = 0 ; j < aNZ() ; j++ )
	{
	    if ( pAlphaZ(j) >= i )
	    {
                pAlphaZ("&",j)++;
	    }
	}
    }

    if ( aNUB() )
    {
	for ( j = 0 ; j < aNUB() ; j++ )
	{
	    if ( pAlphaUB(j) >= i )
	    {
                pAlphaUB("&",j)++;
	    }
	}
    }

    if ( aNF() )
    {
	for ( j = 0 ; j < aNF() ; j++ )
	{
	    if ( pAlphaF(j) >= i )
	    {
                pAlphaF("&",j)++;
	    }
	}
    }

    // Add to zero pivot

    pAlphaZ.add(pAlphaZ.size());
    pAlphaZ("&",pAlphaZ.size()-1) = i;
    dalphaState.add(i);
    dalphaState("&",i) = 0;

    return (pAlphaZ.size())-1;
}

int optContext::addBeta(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= bN() );

    int j;

    // Fix all pivots to reflect pending increase in number of betas

    if ( bNC() )
    {
	for ( j = 0 ; j < bNC() ; j++ )
	{
	    if ( pBetaC(j) >= i )
	    {
                pBetaC("&",j)++;
	    }
	}
    }

    if ( bNF() )
    {
	for ( j = 0 ; j < bNF() ; j++ )
	{
	    if ( pBetaF(j) >= i )
	    {
                pBetaF("&",j)++;
	    }
	}
    }

    // Add to constrained pivot

    pBetaC.add(pBetaC.size());
    pBetaC("&",pBetaC.size()-1) = i;
    dbetaState.add(i);
    dbetaState("&",i) = 0;

    // Update Gpn column norms and betafix

    if ( dkeepfact )
    {
	betaFixUpdate = 1;

	GpnFColNorm.add(i);
	GpnFColNorm("&",i) = 0.0;
	betaFix.add(i);
	betaFix("&",i) = -1;
    }

    return (pBetaC.size())-1;
}

int optContext::removeAlpha(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );

    int iP = -1;
    int j;

    // Find i in pivot vector

    if ( aNZ() )
    {
	for ( j = 0 ; j < aNZ() ; j++ )
	{
	    if ( pAlphaZ(j) == i )
	    {
		iP = j;
	    }

	    else if ( pAlphaZ(j) > i )
	    {
                pAlphaZ("&",j)--;
	    }
	}
    }

    // Sanity check

    NiceAssert( iP >= 0 );

    // Remove i from pivot vector

    pAlphaZ.remove(iP);
    dalphaState.remove(i);

    // Fix all pivots to reflect decrease in number of alphas

    if ( aNLB() )
    {
	for ( j = 0 ; j < aNLB() ; j++ )
	{
	    if ( pAlphaLB(j) > i )
	    {
                pAlphaLB("&",j)--;
	    }
	}
    }

    if ( aNF() )
    {
	for ( j = 0 ; j < aNF() ; j++ )
	{
	    if ( pAlphaF(j) > i )
	    {
                pAlphaF("&",j)--;
	    }
	}
    }

    if ( aNUB() )
    {
	for ( j = 0 ; j < aNUB() ; j++ )
	{
	    if ( pAlphaUB(j) > i )
	    {
                pAlphaUB("&",j)--;
	    }
	}
    }

    return iP;
}

int optContext::removeBeta(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < bN() );

    int iP = -1;
    int j;

    // Find i in pivot vector

    if ( bNC() )
    {
	for ( j = 0 ; j < bNC() ; j++ )
	{
	    if ( pBetaC(j) == i )
	    {
		iP = j;
	    }

	    else if ( pBetaC(j) > i )
	    {
                pBetaC("&",j)--;
	    }
	}
    }

    // Sanity check

    NiceAssert( iP >= 0 );

    // Remove i from pivot vector

    pBetaC.remove(iP);
    dbetaState.remove(i);

    // Fix all pivots to reflect decrease in number of betas

    if ( bNF() )
    {
	for ( j = 0 ; j < bNF() ; j++ )
	{
	    if ( pBetaF(j) > i )
	    {
                pBetaF("&",j)--;
	    }
	}
    }

    if ( dkeepfact )
    {
	GpnFColNorm.remove(i);
	betaFix.remove(i);
    }

    return iP;
}









int optContext::findInAlphaLB(int i) const
{
    int iP = -1;

    if ( aNLB() )
    {
	for ( iP = 0 ; iP < aNLB() ; iP++ )
	{
	    if ( pAlphaLB(iP) == i )
	    {
                break;
	    }
	}
    }

    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );

    return iP;
}

int optContext::findInAlphaZ(int i) const
{
    int iP = -1;

    if ( aNZ() )
    {
	for ( iP = 0 ; iP < aNZ() ; iP++ )
	{
	    if ( pAlphaZ(iP) == i )
	    {
                break;
	    }
	}
    }

    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );

    return iP;
}

int optContext::findInAlphaUB(int i) const
{
    int iP = -1;

    if ( aNUB() )
    {
	for ( iP = 0 ; iP < aNUB() ; iP++ )
	{
	    if ( pAlphaUB(iP) == i )
	    {
                break;
	    }
	}
    }

    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );

    return iP;
}

int optContext::findInAlphaF(int i) const
{
    int iP = -1;

    if ( aNF() )
    {
	for ( iP = 0 ; iP < aNF() ; iP++ )
	{
	    if ( pAlphaF(iP) == i )
	    {
                break;
	    }
	}
    }

    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    return iP;
}

int optContext::findInBetaC(int i) const
{
    int iP = -1;

    if ( bNC() )
    {
	for ( iP = 0 ; iP < bNC() ; iP++ )
	{
	    if ( pBetaC(iP) == i )
	    {
                break;
	    }
	}
    }

    NiceAssert( iP >= 0 );
    NiceAssert( iP < bNC() );

    return iP;
}

int optContext::findInBetaF(int i) const
{
    int iP = -1;

    if ( bNF() )
    {
	for ( iP = 0 ; iP < bNF() ; iP++ )
	{
	    if ( pBetaF(iP) == i )
	    {
                break;
	    }
	}
    }

    NiceAssert( iP >= 0 );
    NiceAssert( iP < bNF() );

    return iP;
}

int optContext::modAlphaLBtoLF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );
    NiceAssert( dalphaState(pAlphaLB(iP)) == -2 );

    dalphaState("&",pAlphaLB(iP)) = -1;

    daNLF++;

    int i = pAlphaLB(iP);
    int insertPos = pAlphaF.size();

    pAlphaLB.remove(iP);
    pAlphaF.add(pAlphaF.size());
    pAlphaF("&",pAlphaF.size()-1) = i;

    if ( dkeepfact )
    {
	insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modAlphaLBtoZ (int iP)
{
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );
    NiceAssert( dalphaState(pAlphaLB(iP)) == -2 );

    dalphaState("&",pAlphaLB(iP)) = 0;

    int i = pAlphaLB(iP);
    int insertPos = pAlphaZ.size();

    pAlphaLB.remove(iP);
    pAlphaZ.add(pAlphaZ.size());
    pAlphaZ("&",pAlphaZ.size()-1) = i;

    return insertPos;
}

int optContext::modAlphaLBtoUF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );
    NiceAssert( dalphaState(pAlphaLB(iP)) == -2 );

    dalphaState("&",pAlphaLB(iP)) = +1;

    daNUF++;

    int i = pAlphaLB(iP);
    int insertPos = pAlphaF.size();

    pAlphaLB.remove(iP);
    pAlphaF.add(pAlphaF.size());
    pAlphaF("&",pAlphaF.size()-1) = i;

    if ( dkeepfact )
    {
	insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modAlphaLBtoUB(int iP)
{
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNLB() );
    NiceAssert( dalphaState(pAlphaLB(iP)) == -2 );

    dalphaState("&",pAlphaLB(iP)) = +2;

    int i = pAlphaLB(iP);
    int insertPos = pAlphaUB.size();

    pAlphaLB.remove(iP);
    pAlphaUB.add(pAlphaUB.size());
    pAlphaUB("&",pAlphaUB.size()-1) = i;

    return insertPos;
}

int optContext::modAlphaLFtoLB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );
    NiceAssert( dalphaState(pAlphaF(iP)) == -1 );

    dalphaState("&",pAlphaF(iP)) = -2;

    daNLF--;

    int i = pAlphaF(iP);
    int insertPos = pAlphaLB.size();

    pAlphaF.remove(iP);
    pAlphaLB.add(pAlphaLB.size());
    pAlphaLB("&",pAlphaLB.size()-1) = i;

    if ( dkeepfact )
    {
	shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modAlphaLFtoZ (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );
    NiceAssert( dalphaState(pAlphaF(iP)) == -1 );

    dalphaState("&",pAlphaF(iP)) = 0;

    daNLF--;

    int i = pAlphaF(iP);
    int insertPos = pAlphaZ.size();

    pAlphaF.remove(iP);
    pAlphaZ.add(pAlphaZ.size());
    pAlphaZ("&",pAlphaZ.size()-1) = i;

    if ( dkeepfact )
    {
	shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modAlphaLFtoUF(int iP)
{
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );
    NiceAssert( dalphaState(pAlphaF(iP)) == -1 );

    daNLF--;
    daNUF++;

    dalphaState("&",pAlphaF(iP)) = +1;

    return iP;
}

int optContext::modAlphaLFtoUB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );
    NiceAssert( dalphaState(pAlphaF(iP)) == -1 );

    dalphaState("&",pAlphaF(iP)) = +2;

    daNLF--;

    int i = pAlphaF(iP);
    int insertPos = pAlphaUB.size();

    pAlphaF.remove(iP);
    pAlphaUB.add(pAlphaUB.size());
    pAlphaUB("&",pAlphaUB.size()-1) = i;

    if ( dkeepfact )
    {
	shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modAlphaZtoLB (int iP)
{
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );
    NiceAssert( dalphaState(pAlphaZ(iP)) == 0 );

    dalphaState("&",pAlphaZ(iP)) = -2;

    int i = pAlphaZ(iP);
    int insertPos = pAlphaLB.size();

    pAlphaZ.remove(iP);
    pAlphaLB.add(pAlphaLB.size());
    pAlphaLB("&",pAlphaLB.size()-1) = i;

    return insertPos;
}

int optContext::modAlphaZtoLF (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );
    NiceAssert( dalphaState(pAlphaZ(iP)) == 0 );

    dalphaState("&",pAlphaZ(iP)) = -1;

    daNLF++;

    int i = pAlphaZ(iP);
    int insertPos = (pAlphaF.size());

    pAlphaZ.remove(iP);
    pAlphaF.add(pAlphaF.size());
    pAlphaF("&",pAlphaF.size()-1) = i;

    if ( dkeepfact )
    {
	insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modAlphaZtoUF (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );
    NiceAssert( dalphaState(pAlphaZ(iP)) == 0 );

    dalphaState("&",pAlphaZ(iP)) = +1;

    daNUF++;

    int i = pAlphaZ(iP);
    int insertPos = (pAlphaF.size());

    pAlphaZ.remove(iP);
    pAlphaF.add(pAlphaF.size());
    pAlphaF("&",pAlphaF.size()-1) = i;

    if ( dkeepfact )
    {
	insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modAlphaZtoUB (int iP)
{
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );
    NiceAssert( dalphaState(pAlphaZ(iP)) == 0 );

    dalphaState("&",pAlphaZ(iP)) = +2;

    int i = pAlphaZ(iP);
    int insertPos = pAlphaUB.size();

    pAlphaZ.remove(iP);
    pAlphaUB.add(pAlphaUB.size());
    pAlphaUB("&",pAlphaUB.size()-1) = i;

    return insertPos;
}

int optContext::modAlphaUFtoLB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );
    NiceAssert( dalphaState(pAlphaF(iP)) == +1 );

    dalphaState("&",pAlphaF(iP)) = -2;

    daNUF--;

    int i = pAlphaF(iP);
    int insertPos = pAlphaLB.size();

    pAlphaF.remove(iP);
    pAlphaLB.add(pAlphaLB.size());
    pAlphaLB("&",pAlphaLB.size()-1) = i;

    if ( dkeepfact )
    {
	shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modAlphaUFtoLF(int iP)
{
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );
    NiceAssert( dalphaState(pAlphaF(iP)) == +1 );

    daNUF--;
    daNLF++;

    dalphaState("&",pAlphaF(iP)) = -1;

    return iP;
}

int optContext::modAlphaUFtoZ (int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );
    NiceAssert( dalphaState(pAlphaF(iP)) == +1 );

    dalphaState("&",pAlphaF(iP)) = 0;

    daNUF--;

    int i = pAlphaF(iP);
    int insertPos = pAlphaZ.size();

    pAlphaF.remove(iP);
    pAlphaZ.add(pAlphaZ.size());
    pAlphaZ("&",pAlphaZ.size()-1) = i;

    if ( dkeepfact )
    {
	shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modAlphaUFtoUB(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );
    NiceAssert( dalphaState(pAlphaF(iP)) == +1 );

    dalphaState("&",pAlphaF(iP)) = +2;

    daNUF--;

    int i = pAlphaF(iP);
    int insertPos = pAlphaUB.size();

    pAlphaF.remove(iP);
    pAlphaUB.add(pAlphaUB.size());
    pAlphaUB("&",pAlphaUB.size()-1) = i;

    if ( dkeepfact )
    {
	shrinkFactAlpha(i,iP,Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modAlphaUBtoLB(int iP)
{
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );
    NiceAssert( dalphaState(pAlphaUB(iP)) == +2 );

    dalphaState("&",pAlphaUB(iP)) = -2;

    int i = pAlphaUB(iP);
    int insertPos = pAlphaLB.size();

    pAlphaUB.remove(iP);
    pAlphaLB.add(pAlphaLB.size());
    pAlphaLB("&",pAlphaLB.size()-1) = i;

    return insertPos;
}

int optContext::modAlphaUBtoLF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );
    NiceAssert( dalphaState(pAlphaUB(iP)) == +2 );

    dalphaState("&",pAlphaUB(iP)) = -1;

    daNLF++;

    int i = pAlphaUB(iP);
    int insertPos = (pAlphaF.size());

    pAlphaUB.remove(iP);
    pAlphaF.add(pAlphaF.size());
    pAlphaF("&",pAlphaF.size()-1) = i;

    if ( dkeepfact )
    {
	insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}


int optContext::modAlphaUBtoZ (int iP)
{
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );
    NiceAssert( dalphaState(pAlphaUB(iP)) == +2 );

    dalphaState("&",pAlphaUB(iP)) = 0;

    int i = pAlphaUB(iP);
    int insertPos = pAlphaZ.size();

    pAlphaUB.remove(iP);
    pAlphaZ.add(pAlphaZ.size());
    pAlphaZ("&",pAlphaZ.size()-1) = i;

    return insertPos;
}

int optContext::modAlphaUBtoUF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNUB() );
    NiceAssert( dalphaState(pAlphaUB(iP)) == +2 );

    dalphaState("&",pAlphaUB(iP)) = +1;

    daNUF++;

    int i = pAlphaUB(iP);
    int insertPos = (pAlphaF.size());

    pAlphaUB.remove(iP);
    pAlphaF.add(pAlphaF.size());
    pAlphaF("&",pAlphaF.size()-1) = i;

    if ( dkeepfact )
    {
	insertPos = extendFactAlpha(Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modBetaCtoF(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < bNC() );
    NiceAssert( dbetaState(pBetaC(iP)) == 0 );

    dbetaState("&",pBetaC(iP)) = 1;

    int i = pBetaC(iP);
    int insertPos = (pBetaF.size());

    pBetaC.remove(iP);
    pBetaF.add(pBetaF.size());
    pBetaF("&",pBetaF.size()-1) = i;

    if ( dkeepfact )
    {
	insertPos = extendFactBeta(Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}

int optContext::modBetaFtoC(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < bNF() );
    NiceAssert( dbetaState(pBetaF(iP)) == 1 );

    dbetaState("&",pBetaF(iP)) = 0;

    int i = pBetaF(iP);
    int insertPos = pBetaC.size();

    pBetaF.remove(iP);
    pBetaC.add(pBetaC.size());
    pBetaC("&",pBetaC.size()-1) = i;

    if ( dkeepfact )
    {
	shrinkFactBeta(iP,Gp,Gn,Gpn,apos,bpos);
    }

    return insertPos;
}










double optContext::fact_det(void) const
{
    double res = freeVarChol.det();

    return res*res;
}

void optContext::fact_rankone(const Vector<double> &bp, const Vector<double> &bn, const double &c, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( bp.size() == aN() );
    NiceAssert( bn.size() == bN() );
    NiceAssert( dkeepfact );

    // No point doing this before the update as we'll need to complete
    // restart afterwards.
    //
    //if ( dkeepfact && betaFixUpdate )
    //{
    //    fixbetaFix(Gn,Gpn);
    //}

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retVector<double> tmpvc;
    retMatrix<double> tmpma;
    retMatrix<double> tmpmb;
    retMatrix<double> tmpmc;
    retMatrix<double> tmpmd;
    retMatrix<double> tmpme;


    freeVarChol.rankone(bp(pAlphaF,tmpva),bn(pBetaF,tmpvb)(0,1,dnfact-1,tmpvc),c,Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D,0,0,0,dnfact);
    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

    if ( dkeepfact )
    {
        betaFix = -1;
        betaFixUpdate = 1;
        fixbetaFix(Gn,Gpn);
    }

    int aposdummy = 0;
    int bposdummy = 0;

    fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);

    return;
}

void optContext::fact_diagmult(const Vector<double> &bp, const Vector<double> &bn, int &apos, int &bpos)
{
    NiceAssert( bp.size() == aN() );
    NiceAssert( bn.size() == bN() );
    NiceAssert( dkeepfact );

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retVector<double> tmpvc;

    freeVarChol.diagmult(bp(pAlphaF,tmpva),bn(pBetaF,tmpvb)(0,1,dnfact-1,tmpvc));

    // Note that multiplying elements symmetrically by +-1 on the off-
    // diagonals makes no difference to the factorisability of a matrix, so
    // there is no need to do anything here.

    return;

    apos = bpos;
}

void optContext::fact_diagoffset(const Vector<double> &bp, const Vector<double> &bn, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( bp.size() == aN() );
    NiceAssert( bn.size() == bN() );
    NiceAssert( dkeepfact );

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retVector<double> tmpvc;
    retMatrix<double> tmpma;
    retMatrix<double> tmpmb;
    retMatrix<double> tmpmc;
    retMatrix<double> tmpmd;
    retMatrix<double> tmpme;

    freeVarChol.diagoffset(bp(pAlphaF,tmpva),bn(pBetaF,tmpvb)(0,1,dnfact-1,tmpvc),Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D,0,0,0,dnfact);
    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

    int aposdummy = 0;
    int bposdummy = 0;

    fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);

    return;
}

double optContext::fact_testFact(Matrix<double> &Gpdest, Matrix<double> &Gndest, Matrix<double> &Gpndest, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    double res = 0;

    Gpdest = Gp;
    Gndest = Gn;
    Gpndest = Gpn;

    Vector<double> Ddest(D);

    retMatrix<double> tmpma;
    retMatrix<double> tmpmb;
    retMatrix<double> tmpmc;
    retMatrix<double> tmpmd;
    retMatrix<double> tmpme;

    freeVarChol.testFact(Gpdest("&",pAlphaF,pAlphaF,tmpma),Gndest("&",pBetaF,pBetaF,tmpmb)("&",0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpndest("&",pAlphaF,pBetaF,tmpmd)("&",0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),Ddest);

    if ( aN() )
    {
	int i,j;

	for ( i = 0 ; i < aN() ; i++ )
	{
	    for ( j = 0 ; j < aN() ; j++ )
	    {
		if ( abs2(Gp(i,j)-Gpdest(i,j)) > res )
		{
                    res = abs2(Gp(i,j)-Gpdest(i,j));
		}
	    }
	}
    }
                          
    if ( bN() )
    {
	int i,j;

	for ( i = 0 ; i < bN() ; i++ )
	{
	    for ( j = 0 ; j < bN() ; j++ )
	    {
		if ( abs2(Gn(i,j)-Gndest(i,j)) > res )
		{
                    res = abs2(Gn(i,j)-Gndest(i,j));
		}
	    }
	}
    }

    if ( aN() && bN() )
    {
	int i,j;

	for ( i = 0 ; i < aN() ; i++ )
	{
	    for ( j = 0 ; j < bN() ; j++ )
	    {
		if ( abs2(Gpn(i,j)-Gpndest(i,j)) > res )
		{
                    res = abs2(Gpn(i,j)-Gpndest(i,j));
		}
	    }
	}
    }

    return res;
}

int optContext::fact_nfact(const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    int retval = dnfact;

    if ( fact_nofact(Gn,Gpn) )
    {
        retval = fact_pfact(Gn,Gpn);
    }

    return retval;
}

int optContext::fact_pfact(const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    int retval = dpfact;

    if ( fact_nofact(Gn,Gpn) )
    {
	int maxsize = ( ( aNF() < bNF() ) ? aNF() : bNF() );
	int nonsingsize = 1;

        retMatrix<double> tmpma;
        retMatrix<double> tmpmb;

        while ( ( (Gpn(pAlphaF,pBetaF,tmpma)(0,1,( ( nonsingsize < maxsize ) ? nonsingsize : maxsize )-1,0,1,( ( nonsingsize < maxsize ) ? nonsingsize : maxsize )-1,tmpmb)).det() > dzt ) && ( nonsingsize <= maxsize ) )
	{
	    nonsingsize++;
	}

        retval = nonsingsize-1;
    }

    return retval;
}

int optContext::fact_nofact(const Matrix<double> &Gn, const Matrix<double> &Gpn) const
{
    NiceAssert( dkeepfact );

    int i;

    if ( betaFixUpdate )
    {
	// OK, here's the trick.  We need to do two things:
	//
	// - fix betaFix, which is the vector of Gpn columns that are nonzero (see header)
	// - respect the const in the function definition.
	//
	// But these are contradictory... fixing betaFix requires changing *this, and
	// *this is const.  To get around this the constructor defines thisredirect so
	// that **thisredirect = *this, but because this is done in a non-const function
	// it is *not* const, even though it is a reference to a nominally const object.
	// So, we fun fixbetaFix(...) on this nonimally const but factually non-const
        // object to adjust things that are const.

        (*(thisredirect[0])).fixbetaFix(Gn,Gpn);
    }

    int bNFgood = 0;

    if ( bNF() )
    {
	for ( i = 0 ; i < bNF() ; i++ )
	{
	    if ( !betaFix(i) )
	    {
		++bNFgood;
                break; // we have the answer, so get out, no need to waste time.
	    }
	}
    }

    return ( !(freeVarChol.ngood()) && aNF() && bNFgood );
}

void optContext::fact_fudgeOn(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    // No point doing this before the update as we'll need to complete
    // restart afterwards.
    //
    //if ( dkeepfact && betaFixUpdate )
    //{
    //    fixbetaFix(Gn,Gpn);
    //}

    freeVarChol.fudgeOn(D);
    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

    if ( dkeepfact )
    {
        betaFix = -1;
        betaFixUpdate = 1;
        fixbetaFix(Gn,Gpn);
    }

    int aposdummy = 0;
    int bposdummy = 0;

    fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);

    return;
}

void optContext::fact_fudgeOff(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    // No point doing this before the update as we'll need to complete
    // restart afterwards.
    //
    //if ( dkeepfact && betaFixUpdate )
    //{
    //    fixbetaFix(Gn,Gpn);
    //}

    freeVarChol.fudgeOff();
    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

    if ( dkeepfact )
    {
        betaFix = -1;
        betaFixUpdate = 1;
        fixbetaFix(Gn,Gpn);
    }

    int aposdummy = 0;
    int bposdummy = 0;

    fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);

    return;
}

int optContext::extendFactAlpha(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    if ( dkeepfact && betaFixUpdate )
    {
        fixbetaFix(Gn,Gpn);
    }

    int retval = (pAlphaF.size())-1;
    int fpos = D.size();
    int i = pAlphaF(retval);
    int j;
    int addpos = 0;
    int betafixchange = 0;
    int betaunfixedextend = 0;
    int oldbetafix;

    fAlphaF.add(fAlphaF.size());
    fAlphaF("&",fAlphaF.size()-1) = fpos;

    // Update betaFix and set betafixchange if any change occurs.
    // betaFix[i] = 0 if the norm of elements in column i exceeds
    // a given threshold (and can therefore be included, potentially,
    // in the factorisation), zero otherwise.

    if ( betaFix.size() )
    {
	for ( j = 0 ; j < betaFix.size() ; j++ )
	{
	    GpnFColNorm("&",j) += (Gpn(i,j)*Gpn(i,j));

            oldbetafix = betaFix(j);

	    betaFix("&",j) = CALCBETAFIX(j);

	    if ( oldbetafix != betaFix(j) )
	    {
                betafixchange = 1;
	    }
	}
    }

    // Run through those columns of Gpn and row/columns of Gn not
    // currently included in the factorisation.  If any have betaFix[j]
    // non-zero then set betaunfixedextend = 1 to indicate that we
    // should try to add these to the factorisation.

    if ( dnfact < pBetaF.size() )
    {
	for ( j = dnfact ; j < pBetaF.size() ; j++ )
	{
	    if ( !betaFix(j) )
	    {
                betaunfixedextend = 1;
	    }
	}
    }

    // Set flag if new diagonal in Hessian is non-zero to within zerotol

    if ( Gp(retval,retval) > dzt )
    {
        addpos = 1;
    }

    // If factoristion non-singular,
    // or if factorisation is completely singular and the new diagonal on
    //   Gp is non-zero,
    // or if any betas that had corresponded to zero columns in Gpn now
    //   correspond to non-zero columns,
    // or there are any parts in the segment Gn not currently included
    //   in the factorisation that correspond to non-zero columns in Gpn
    // then add the row/column to factorisation and try to extend it.
    //
    // Otherwise we know that there is no way to extend the factorisation,
    // so don't even try, just add to the end and leave.

    if ( !(freeVarChol.nbad()) || ( !(freeVarChol.ngood()) && addpos ) || betafixchange || betaunfixedextend )
    {
        retMatrix<double> tmpma;
        retMatrix<double> tmpmb;
        retMatrix<double> tmpmc;
        retMatrix<double> tmpmd;
        retMatrix<double> tmpme;

	D.add(fpos);
	D("&",fpos) = +1;
	freeVarChol.add(fpos,Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D);
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

        int bposdummy = 0;

	fixfact(Gp,Gn,Gpn,apos,bpos,retval,bposdummy);
    }

    else
    {
        retMatrix<double> tmpma;
        retMatrix<double> tmpmb;
        retMatrix<double> tmpmc;
        retMatrix<double> tmpmd;
        retMatrix<double> tmpme;

	D.add(fpos);
	D("&",fpos) = +1;
	freeVarChol.add(fpos,Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D);
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());
    }

    return retval;
}

void optContext::shrinkFactAlpha(int i, int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    if ( dkeepfact && betaFixUpdate )
    {
        fixbetaFix(Gn,Gpn);
    }

    int j,jP;
    int fpos = fAlphaF(iP);

    fAlphaF.remove(iP);

    if ( fAlphaF.size() )
    {
	for ( jP = 0 ; jP < fAlphaF.size() ; jP++ )
	{
	    if ( fAlphaF(jP) > fpos )
	    {
                fAlphaF("&",jP)--;
	    }
	}
    }

    if ( fBetaF.size() )
    {
	for ( jP = 0 ; jP < fBetaF.size() ; jP++ )
	{
	    if ( fBetaF(jP) > fpos )
	    {
                fBetaF("&",jP)--;
	    }
	}
    }

    if ( betaFix.size() )
    {
	for ( j = 0 ; j < betaFix.size() ; j++ )
	{
	    GpnFColNorm("&",j) -= (Gpn(i,j)*Gpn(i,j));

	    betaFix("&",j) = CALCBETAFIX(j);
	}
    }

    if ( fpos > freeVarChol.ngood() )
    {
	// Removing this alpha will have no affect on the singular/nonsingular block structure of the factorisation.

        retMatrix<double> tmpma;
        retMatrix<double> tmpmb;
        retMatrix<double> tmpmc;
        retMatrix<double> tmpmd;
        retMatrix<double> tmpme;

	D.remove(fpos);
        freeVarChol.remove(fpos,Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D);
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());
    }

    else
    {
        retMatrix<double> tmpma;
        retMatrix<double> tmpmb;
        retMatrix<double> tmpmc;
        retMatrix<double> tmpmd;
        retMatrix<double> tmpme;

	D.remove(fpos);
        freeVarChol.remove(fpos,Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D);
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

	int aposdummy = 0;
	int bposdummy = 0;

	fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);
    }

    return;
}

int optContext::extendFactBeta(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    if ( dkeepfact && betaFixUpdate )
    {
        fixbetaFix(Gn,Gpn);
    }

    int retval = (pBetaF.size())-1;
    int i = pBetaF(retval);
    int jP;

    GpnFColNorm("&",i) = 0.0;

    if ( pAlphaF.size() )
    {
	for ( jP = 0 ; jP < pAlphaF.size() ; jP++ )
	{
	    GpnFColNorm("&",i) += (Gpn(pAlphaF(jP),i)*Gpn(pAlphaF(jP),i));
	}
    }

    betaFix("&",i) = CALCBETAFIX(i);

    fBetaF.add(fBetaF.size());
    fBetaF("&",fBetaF.size()-1) = -1;

    if ( !betaFix(i) )
    {
	int aposdummy = 0;

	fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,retval);
    }

    return retval;
}

void optContext::shrinkFactBeta(int iP, const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    if ( dkeepfact && betaFixUpdate )
    {
        fixbetaFix(Gn,Gpn);
    }

    int jP;
    int fpos = fBetaF(iP);

    fBetaF.remove(iP);

    if ( fpos >= 0 )
    {
	if ( fAlphaF.size() )
	{
	    for ( jP = 0 ; jP < fAlphaF.size() ; jP++ )
	    {
		if ( fAlphaF(jP) > fpos )
		{
		    fAlphaF("&",jP)--;
		}
	    }
	}

	if ( fBetaF.size() )
	{
	    for ( jP = 0 ; jP < fBetaF.size() ; jP++ )
	    {
		if ( fBetaF(jP) > fpos )
		{
		    fBetaF("&",jP)--;
		}
	    }
	}

        dnfact--;

        retMatrix<double> tmpma;
        retMatrix<double> tmpmb;
        retMatrix<double> tmpmc;
        retMatrix<double> tmpmd;
        retMatrix<double> tmpme;

	D.remove(fpos);
	freeVarChol.remove(fpos,Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D);
        dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

	int aposdummy = 0;
	int bposdummy = 0;

	fixfact(Gp,Gn,Gpn,apos,bpos,aposdummy,bposdummy);
    }

    return;
}

void optContext::fixfact(const Matrix<double> &Gp, const Matrix<double> &Gn, const Matrix<double> &Gpn, int &apos, int &bpos, int &aposalt, int &bposalt)
{
    NiceAssert( Gp.isSquare() );
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gp.numRows() == Gpn.numRows() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    if ( dkeepfact && betaFixUpdate )
    {
        fixbetaFix(Gn,Gpn);
    }

    int iP,jP,fpos;

    // Clean up the factorisation by removing any betas in the singular
    // part of the factorisation.

    if ( fBetaF.size() )
    {
	while ( ( fpos = max(fBetaF,iP) ) >= freeVarChol.ngood() )
	{
	    if ( iP != (pBetaF.size())-1 )
	    {
		if ( bpos == iP )
		{
		    bpos = pBetaF.size()-1;
		}

		else if ( bpos == pBetaF.size()-1 )
		{
		    bpos = iP;
		}

		if ( bposalt == iP )
		{
		    bposalt = pBetaF.size()-1;
		}

		else if ( bposalt == pBetaF.size()-1 )
		{
		    bposalt = iP;
		}

		pBetaF.squareswap(iP,pBetaF.size()-1);
		fBetaF.squareswap(iP,pBetaF.size()-1);
	    }

	    if ( fAlphaF.size() )
	    {
		for ( jP = 0 ; jP < fAlphaF.size() ; jP++ )
		{
		    if ( fAlphaF(jP) > fpos )
		    {
			fAlphaF("&",jP)--;
		    }
		}
	    }

	    fBetaF("&",pBetaF.size()-1) = -1;
	    dnfact--;

            retMatrix<double> tmpma;
            retMatrix<double> tmpmb;
            retMatrix<double> tmpmc;
            retMatrix<double> tmpmd;
            retMatrix<double> tmpme;

	    D.remove(fpos);
	    freeVarChol.remove(fpos,Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D);
	    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());
	}
    }

    NiceAssert( (freeVarChol.ngood()) >= dnfact );

    int outerdone = 0;
    int innerdone = 0;
    int newpos;
    int oldpos;

    while ( !outerdone )
    {
	outerdone = 1;

	// add as many elements from pBetaF as possible

	innerdone = 0;

	while ( !innerdone && ( dnfact < pBetaF.size() ) )
	{
            innerdone = 1;

	    for ( iP = dnfact ; iP < pBetaF.size() ; iP++ )
	    {
		if ( !betaFix(pBetaF(iP)) )
		{
		    if ( iP > dnfact )
		    {
			if ( bpos == iP )
			{
			    bpos = dnfact;
			}

			else if ( bpos == dnfact )
			{
			    bpos = iP;
			}

			if ( bposalt == iP )
			{
			    bposalt = dnfact;
			}

			else if ( bposalt == dnfact )
			{
			    bposalt = iP;
			}

			pBetaF.squareswap(iP,dnfact);
			fBetaF.squareswap(iP,dnfact);
		    }

		    fBetaF("&",dnfact) = freeVarChol.ngood();
		    dnfact++;

                    retMatrix<double> tmpma;
                    retMatrix<double> tmpmb;
                    retMatrix<double> tmpmc;
                    retMatrix<double> tmpmd;
                    retMatrix<double> tmpme;

		    D.add(freeVarChol.ngood());
		    D("&",freeVarChol.ngood()) = -1;
		    freeVarChol.add(freeVarChol.ngood(),Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D);
		    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

		    if ( freeVarChol.ngood() > fBetaF(dnfact-1) )
		    {
			innerdone = 0;
                        outerdone = 0;

			if ( fAlphaF.size() )
			{
			    for ( jP = 0 ; jP < fAlphaF.size() ; jP++ )
			    {
				if ( fAlphaF(jP) >= fBetaF(dnfact-1) )
				{
				    fAlphaF("&",jP)++;
				}
			    }
			}

			break;
		    }

		    else
		    {
                        NiceAssert( freeVarChol.ngood() == fBetaF(dnfact-1) );

			dnfact--;
			fBetaF("&",dnfact) = -1;

                        retMatrix<double> tmpma;
                        retMatrix<double> tmpmb;
                        retMatrix<double> tmpmc;
                        retMatrix<double> tmpmd;
                        retMatrix<double> tmpme;

			D.remove(freeVarChol.ngood());
			freeVarChol.remove(freeVarChol.ngood(),Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D);
			dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

			if ( iP > dnfact )
			{
			    if ( bpos == iP )
			    {
				bpos = dnfact;
			    }

			    else if ( bpos == dnfact )
			    {
				bpos = iP;
			    }

			    if ( bposalt == iP )
			    {
				bposalt = dnfact;
			    }

			    else if ( bposalt == dnfact )
			    {
				bposalt = iP;
			    }

			    pBetaF.squareswap(iP,dnfact);
			    fBetaF.squareswap(iP,dnfact);
			}
		    }
		}
	    }
	}

        // add as many elements from pAlphaF as possible

	innerdone = 0;

	while ( !innerdone && ( dpfact+1 < pAlphaF.size() ) )
	{
            innerdone = 1;

	    for ( iP = dpfact+1 ; iP < pAlphaF.size() ; iP++ )
	    {
                oldpos = fAlphaF("&",iP);
		newpos = freeVarChol.ngood();

		if ( apos == iP )
		{
                    apos = (pAlphaF.size())-1;
		}

		else if ( apos > iP )
		{
                    apos--;
		}

		if ( aposalt == iP )
		{
                    aposalt = (pAlphaF.size())-1;
		}

		else if ( aposalt > iP )
		{
                    aposalt--;
		}

		pAlphaF.blockswap(iP,(pAlphaF.size())-1);
		fAlphaF.blockswap(iP,(fAlphaF.size())-1);

                retVector<int> tmpva;

		fAlphaF("&",iP,1,(fAlphaF.size())-2,tmpva) -= 1;
		fAlphaF("&",(fAlphaF.size())-1) = (D.size())-1;

                retMatrix<double> tmpma;
                retMatrix<double> tmpmb;
                retMatrix<double> tmpmc;
                retMatrix<double> tmpmd;
                retMatrix<double> tmpme;
                retMatrix<double> tmpmf;

		D.remove(oldpos);
                freeVarChol.remove(oldpos,Gp(pAlphaF,pAlphaF,tmpma)(0,1,(pAlphaF.size())-2,0,1,(pAlphaF.size())-2,tmpmb),Gn(pBetaF,pBetaF,tmpmc)(0,1,dnfact-1,0,1,dnfact-1,tmpmd),Gpn(pAlphaF,pBetaF,tmpme)(0,1,(pAlphaF.size())-2,0,1,dnfact-1,tmpmf),D);

		if ( apos == (pAlphaF.size())-1 )
		{
                    apos = dpfact;
		}

		else if ( apos >= dpfact )
		{
                    apos++;
		}

		if ( aposalt == (pAlphaF.size())-1 )
		{
                    aposalt = dpfact;
		}

		else if ( aposalt >= dpfact )
		{
                    aposalt++;
		}

		pAlphaF.blockswap((pAlphaF.size())-1,dpfact);
		fAlphaF.blockswap((fAlphaF.size())-1,dpfact);

		fAlphaF("&",dpfact) = newpos;
                fAlphaF("&",dpfact+1,1,(fAlphaF.size())-1,tmpva) += 1;

		D.add(newpos);
		D("&",newpos) = +1;
		freeVarChol.add(newpos,Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpme),D);

		if ( freeVarChol.ngood() > newpos )
		{
		    dpfact = (freeVarChol.npos())-(freeVarChol.nbadpos());

		    innerdone = 0;
		    outerdone = 0;

		    break;
		}

		else
		{
                    NiceAssert( freeVarChol.ngood() == newpos );

		    if ( apos == dpfact )
		    {
			apos = (pAlphaF.size())-1;
		    }

		    else if ( apos > dpfact )
		    {
			apos--;
		    }

		    if ( aposalt == dpfact )
		    {
			aposalt = (pAlphaF.size())-1;
		    }

		    else if ( aposalt > dpfact )
		    {
			aposalt--;
		    }

		    pAlphaF.blockswap(dpfact,(pAlphaF.size())-1);
		    fAlphaF.blockswap(dpfact,(fAlphaF.size())-1);

		    fAlphaF("&",dpfact,1,(fAlphaF.size())-2,tmpva) -= 1;
		    fAlphaF("&",(fAlphaF.size())-1) = (D.size())-1;

		    D.remove(newpos);
		    freeVarChol.remove(newpos,Gp(pAlphaF,pAlphaF,tmpma)(0,1,(pAlphaF.size())-2,0,1,(pAlphaF.size())-2,tmpmb),Gn(pBetaF,pBetaF,tmpmc)(0,1,dnfact-1,0,1,dnfact-1,tmpmd),Gpn(pAlphaF,pBetaF,tmpme)(0,1,(pAlphaF.size())-2,0,1,dnfact-1,tmpmf),D);

		    if ( apos == (pAlphaF.size())-1 )
		    {
			apos = iP;
		    }

		    else if ( apos >= iP )
		    {
			apos++;
		    }

		    if ( aposalt == (pAlphaF.size())-1 )
		    {
			aposalt = iP;
		    }

		    else if ( aposalt >= iP )
		    {
			aposalt++;
		    }

		    pAlphaF.blockswap((pAlphaF.size())-1,iP);
		    fAlphaF.blockswap((fAlphaF.size())-1,iP);

		    fAlphaF("&",iP) = oldpos;
		    fAlphaF("&",iP+1,1,(fAlphaF.size())-1,tmpva) += 1;

		    D.add(oldpos);
		    D("&",oldpos) = +1;
		    freeVarChol.add(oldpos,Gp(pAlphaF,pAlphaF,tmpma),Gn(pBetaF,pBetaF,tmpmb)(0,1,dnfact-1,0,1,dnfact-1,tmpmc),Gpn(pAlphaF,pBetaF,tmpmd)(0,1,(pAlphaF.size())-1,0,1,dnfact-1,tmpmf),D);

                    NiceAssert( dpfact == (freeVarChol.npos())-(freeVarChol.nbadpos()) );
		}
	    }
	}
    }

    return;
}


void optContext::fixbetaFix(const Matrix<double> &Gn, const Matrix<double> &Gpn)
{
    NiceAssert( Gn.isSquare() );
    NiceAssert( Gn.numCols() == Gpn.numCols() );
    NiceAssert( Gpn.numRows() == aN() );
    NiceAssert( Gpn.numCols() == bN() );
    NiceAssert( dkeepfact );

    int i,jP;

    if ( betaFixUpdate && bN() )
    {
        for ( i = 0 ; i < bN() ; i++ )
        {
            if ( betaFix(i) == -1 )
            {
                GpnFColNorm("&",i) = 0.0;
                      
                if ( aNF() )
                {
                    for ( jP = 0 ; jP < aNF() ; jP++ )
                    {
                        GpnFColNorm("&",i) += (Gpn(pAlphaF(jP),i)*Gpn(pAlphaF(jP),i));
                    }
                }

                betaFix("&",i) = CALCBETAFIX(i);
            }
        }
    }

    betaFixUpdate = 0;

    return;
}






// Stream operators

std::ostream &operator<<(std::ostream &output, const optContext &src)
{
    output << "LB  pivot:          " << src.pAlphaLB      << "\n";
    output << "Z   pivot:          " << src.pAlphaZ       << "\n";
    output << "UB  pivot:          " << src.pAlphaUB      << "\n";
    output << "F   pivot:          " << src.pAlphaF       << "\n";
    output << "nC  pivot:          " << src.pBetaC        << "\n";
    output << "nF  pivot:          " << src.pBetaF        << "\n";
    output << "Alpha state:        " << src.dalphaState   << "\n";
    output << "Beta state:         " << src.dbetaState    << "\n";
    output << "Zero tolerance:     " << src.dzt           << "\n";
    output << "Keep factorisation: " << src.dkeepfact     << "\n";
    output << "aNLF:               " << src.daNLF         << "\n";
    output << "aNUF:               " << src.daNUF         << "\n";
    output << "Gpn column sums:    " << src.GpnFColNorm   << "\n";
    output << "Beta fixing:        " << src.betaFix       << "\n";
    output << "Beta fixing state:  " << src.betaFixUpdate << "\n";
    output << "nfact:              " << src.dnfact        << "\n";
    output << "pfact:              " << src.dpfact        << "\n";
    output << "D inter:            " << src.D             << "\n";
    output << "F position:         " << src.fAlphaF       << "\n";
    output << "nF position:        " << src.fBetaF        << "\n";
    output << "Factorisation:      " << src.freeVarChol   << "\n";

    return output;
}

std::istream &operator>>(std::istream &input, optContext &dest)
{
    wait_dummy dummy;

    input >> dummy; input >> dest.pAlphaLB;
    input >> dummy; input >> dest.pAlphaZ;
    input >> dummy; input >> dest.pAlphaUB;
    input >> dummy; input >> dest.pAlphaF;
    input >> dummy; input >> dest.pBetaC;
    input >> dummy; input >> dest.pBetaF;
    input >> dummy; input >> dest.dalphaState;
    input >> dummy; input >> dest.dbetaState;
    input >> dummy; input >> dest.dzt;
    input >> dummy; input >> dest.dkeepfact;
    input >> dummy; input >> dest.daNLF;
    input >> dummy; input >> dest.daNUF;
    input >> dummy; input >> dest.GpnFColNorm;
    input >> dummy; input >> dest.betaFix;
    input >> dummy; input >> dest.betaFixUpdate;
    input >> dummy; input >> dest.dnfact;
    input >> dummy; input >> dest.dpfact;
    input >> dummy; input >> dest.D;
    input >> dummy; input >> dest.fAlphaF;
    input >> dummy; input >> dest.fBetaF;
    input >> dummy; input >> dest.freeVarChol;

    return input;
}
