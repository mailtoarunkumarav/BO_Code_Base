
//
// Linear optimisation state
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
#include "optlinstate.h"



std::ostream &operator<<(std::ostream &output, const optLinState &src )
{
    output << "Alpha:                " << src.dalpha         << "\n";
    output << "Alpha gradient:       " << src.dalphaGrad     << "\n";
    output << "e:                    " << src.de             << "\n";
    output << "b:                    " << src.db             << "\n";
    output << "Alpha restriction:    " << src.dalphaRestrict << "\n";
    output << "e restriction:        " << src.deRestrict     << "\n";
    output << "Optimality tolerance: " << src.dopttol        << "\n";
    output << "alphaGradBad:         " << src.alphaGradBad   << "\n";
    output << "eBad:                 " << src.eBad           << "\n";
    output << "Context:              " << src.Q              << "\n";

    return output;
}


std::istream &operator>>(std::istream &input, optLinState &dest)
{
    wait_dummy dummy;

    input >> dummy; input >> dest.dalpha;
    input >> dummy; input >> dest.dalphaGrad;
    input >> dummy; input >> dest.de;
    input >> dummy; input >> dest.db;
    input >> dummy; input >> dest.dalphaRestrict;
    input >> dummy; input >> dest.deRestrict;
    input >> dummy; input >> dest.dopttol;
    input >> dummy; input >> dest.alphaGradBad;
    input >> dummy; input >> dest.eBad;
    input >> dummy; input >> dest.Q;

    dest.btemp = dest.db; // No point saving the scratch pad

    return input;
}


optLinState::optLinState(void)
{
    dopttol = DEFAULT_OPTTOL;

    return;
}

optLinState::optLinState(const optLinState &src)
{
    *this = src;

    return;
}

optLinState &optLinState::operator=(const optLinState &src)
{
    dalpha       = src.dalpha;
    dalphaGrad   = src.dalphaGrad;
    de           = src.de;
    alphaGradBad = src.alphaGradBad;
    eBad         = src.eBad;
    db           = src.db;
    btemp        = src.btemp;

    dalphaRestrict = src.dalphaRestrict;
    deRestrict     = src.deRestrict;

    dopttol = src.dopttol;
    Q       = src.Q;

    return *this;
}

void optLinState::setAlpha(const Vector<double> &newAlpha, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( newAlpha.size() == aN() );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    if ( aN() )
    {
	int i,iP;

	for ( i = 0 ; i < aN() ; i++ )
	{
            NiceAssert( newAlpha(i) >= 0 );

            if ( alphaState()(i) )
	    {
                stepalpha(i,newAlpha(i)-dalpha(i),G,c,g);

		iP = findInAlphaF(i);

                if ( newAlpha(i) < zerotol() )
		{
                    iP = modAlphaFtoZ(iP,G,c,g);
		}
	    }

            else
	    {
		iP = findInAlphaZ(i);

                if ( newAlpha(i) >= zerotol() )
		{
                    NiceAssert( ( dalphaRestrict(i) != 2 ) && ( dalphaRestrict(i) != 3 ) );

                    iP = modAlphaZtoF(iP,G,c,g);

                    stepalpha(i,newAlpha(i)-dalpha(i),G,c,g);
		}
	    }
	}
    }

    return;
}

void optLinState::refact(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    // Need to refactorize Q, recalculate e and then defer to refactlin
    // to finish the job (it will take care of all of alphaGrad)

    Q.refact(G,zerotol());
    refactlin(G,c,g);

    return;
}

void optLinState::refactlin(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    eBad.resize(0);
    alphaGradBad.resize(0);

    // Need to:
    //
    // - recalculate db
    // - recalculate de
    // - recalculate dalphaGrad

    // Recalculate db, de and dalphaGrad

    recalcb(G,c,g);
    recalce(G,c,g);
    recalcAlphaGrad(G,c,g);

    return;
}

void optLinState::reset(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    Q.reset(G);
    dalpha = 0.0;
    refact(G,c,g);

    return;
}

void optLinState::setopttol(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g, double xopttol)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( xopttol > 0 );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    dopttol = xopttol;

    return;
}

void optLinState::setzt(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g, double zt)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( zt > 0 );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    int oldrowpfact = rowpfact();

    Q.refact(G,zt);

    if ( rowpfact() != oldrowpfact )
    {
        refact(G,c,g);
    }

    return;
}

void optLinState::scale(double a, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( a >= 0 );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    dalpha.scale(a);
    de -= g;
    de.scale(a);
    de += g;

    return;
}

int optLinState::addAlpha(int i, int alphrestrict)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= aN() );
    NiceAssert( ( alphrestrict == 1 ) || ( alphrestrict == 3 ) );

    int ucnt = aN()-i;

    // add to context

    int res = Q.addAlpha(i);

    // Add to vectors

    dalpha.add(i);
    dalphaGrad.add(i);
    dalphaRestrict.add(i);

    dalpha("&",i) = 0.0;
    dalphaGrad("&",i) = 0.0;
    dalphaRestrict("&",i) = alphrestrict;

    // mark alpha gradient as bad

    if ( alphaGradBad.size() && ucnt )
    {
        int iP;

        for ( iP = 0 ; ( iP < alphaGradBad.size() ) && ucnt ; iP++ )
        {
            if ( alphaGradBad(iP) >= i )
            {
                alphaGradBad("&",iP)++;
                ucnt--;
            }
        }
    }

    alphaGradBad.add(alphaGradBad.size());
    alphaGradBad("&",alphaGradBad.size()-1) = i;

    return res;
}

int optLinState::removeAlpha(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );

    int ucnt = aN()-i-1;

    // Remove from Vectors

    dalpha.remove(i);
    dalphaGrad.remove(i);
    dalphaRestrict.remove(i);

    // Remove from context

    int res = Q.removeAlpha(i);

    if ( alphaGradBad.size() )
    {
        int iP;

        for ( iP = 0 ; iP < alphaGradBad.size() ; iP++ )
        {
            if ( alphaGradBad(iP) == i )
            {
                break;
            }
        }

        if ( iP < alphaGradBad.size() )
        {
            alphaGradBad.remove(iP);
        }
    }

    if ( alphaGradBad.size() )
    {
        int iP;

        for ( iP = 0 ; ( iP < alphaGradBad.size() ) && ucnt ; iP++ )
        {
            if ( alphaGradBad(iP) > i )
            {
                alphaGradBad("&",iP)--;
                ucnt--;
            }
        }
    }

    return res;
}

int optLinState::addRow(int i, int erestrict)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= eN() );
    NiceAssert( ( erestrict >= 0 ) || ( erestrict <= 3 ) );

    int ucnt = eN()-i;

    // add to context

    int res = Q.addRow(i);

    // Add to vectors

    de.add(i);
    db.add(i);
    btemp.add(i);
    deRestrict.add(i);

    de("&",i) = 0.0;
    db("&",i) = 0.0;
    btemp("&",i) = 0.0;
    deRestrict("&",i) = erestrict;

    // mark e as bad

    if ( eBad.size() && ucnt )
    {
        int iP;

        for ( iP = 0 ; ( iP < eBad.size() ) && ucnt ; iP++ )
        {
            if ( eBad(iP) >= i )
            {
                eBad("&",iP)++;
                ucnt--;
            }
        }
    }

    eBad.add(eBad.size());
    eBad("&",eBad.size()-1) = i;

    return res;
}

int optLinState::removeRow(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < eN() );

    int ucnt = eN()-1-i;

    // Remove from Vectors

    de.remove(i);
    db.remove(i);
    btemp.remove(i);
    deRestrict.remove(i);

    // Remove from context

    int res = Q.removeRow(i);

    if ( eBad.size() )
    {
        int iP;

        for ( iP = 0 ; iP < eBad.size() ; iP++ )
        {
            if ( eBad(iP) == i )
            {
                break;
            }
        }

        if ( iP < eBad.size() )
        {
            eBad.remove(iP);
        }
    }

    if ( eBad.size() && ucnt )
    {
        int iP;

        for ( iP = 0 ; ( iP < eBad.size() ) && ucnt ; iP++ )
        {
            if ( eBad(iP) > i )
            {
                eBad("&",iP)--;
                ucnt--;
            }
        }
    }

    return res;
}

void optLinState::fudgeOn(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    Q.fudgeOn();
    refact(G,c,g);

    return;
}

void optLinState::fudgeOff(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    Q.fudgeOff();
    refact(G,c,g);

    return;
}

void optLinState::changeAlphaRestrict(int i, int alphrestrict, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );
    NiceAssert( ( alphrestrict == 1 ) || ( alphrestrict == 3 ) );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    if ( alphaState()(i) )
    {
        int iP = findInAlphaF(i);
        stepalpha(i,-dalpha(i),G,c,g);
        iP = modAlphaFtoZ(iP,G,c,g);
    }

    dalphaRestrict("&",i) = alphrestrict;

    return;
}

void optLinState::changeeRestrict(int i, int betrestrict, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < eN() );
    NiceAssert( ( betrestrict >= 0 ) || ( betrestrict <= 3 ) );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    if ( rowState()(i) )
    {
        int iP = findInRowF(i);
        iP = modRowFtoZ(iP,G,c,g);
    }

    deRestrict("&",i) = betrestrict;

    return;
}
























// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// Everything after here should be optimised
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------
// -------------------------------------------------------------------

int optLinState::modAlphaZtoF(int iP, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{ 
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    int res = Q.modAlphaZtoF(iP,G);

    // Gra -> [ Gra t ]
    // ca  -> [ ca  v ]
    //
    // Gra.Gra' -> Gra.Gra' + t.t'
    // inv(Gra.Gra') -> inv(Gra.Gra' + t.t')
    //                = inv(Gra.Gra') + s.s' (FIXME USE INVERSE UPDATE LEMMA)
    // b -> ( inv(Gra.Gra') + s.s' ) [ Gra t ] [ ca v ]
    //    = ( inv(Gra.Gra') + s.s' ) ( Gra.ca + v.t )
    //    = inv(Gra.Gra').Gra.ca + s.s'.Gra.ca + v.inv(Gra.Gra').t + v.(s'.t).t
    //    = b + ( bx = (s'.Gra.ca).s + v.(inv(Gra.Gra').t) + (v.(s'.t)).t )
    // 
    // alphaGrad -> alphaGrad - Gr:'.bx
    //
    // FIXME: optimise this

    recalcb(G,c,g);
    recalcAlphaGrad(G,c,g);

    return res;
}

int optLinState::modAlphaFtoZ(int iP, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{ 
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    int res = Q.modAlphaFtoZ(iP,G);

    NiceAssert( abs2(dalpha(pivAlphaZ()(res))) < dopttol );

    dalpha("&",pivAlphaZ()(res)) = 0;

    recalcb(G,c,g);
    recalcAlphaGrad(G,c,g);

    return res;
}

int optLinState::modRowZtoF(int iP, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < eNZ() );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    int res = Q.modRowZtoF(iP,G);

    NiceAssert( abs2(de(pivRowF()(res))) < dopttol );

    recalcb(G,c,g);
    recalcAlphaGrad(G,c,g);

    return res;
}

int optLinState::modRowFtoZ(int iP, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( iP >= 0 );
    NiceAssert( iP < eNF() );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    int res = Q.modRowFtoZ(iP,G);

    recalcb(G,c,g);
    recalcAlphaGrad(G,c,g);

    return res;
}

int optLinState::scaleFStep(double &scale, int &alphaFIndex, int &eFIndex, const Vector<double> &alphaFStep, const Vector<double> &stepde, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( stepde.size() == eNZ() );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    int res = 0;
    double temp;
    scale = 1.0;
    eFIndex = -1;
    alphaFIndex = -1;

    if ( aNF() )
    {
        int iP;

        for ( iP = 0 ; iP < aNF() ; iP++ )
        {
            if ( alphaFStep(iP) < 0 )
            {
                temp = -dalpha(pivAlphaF()(iP))/alphaFStep(iP);

                if ( ( temp < scale )  || !res )
                {
                    scale = temp;
                    res = 1;
                    eFIndex = -1;
                    alphaFIndex = iP;
                }
            }
        }
    }

    if ( eNZ() )
    {
        int iP;

        for ( iP = 0 ; iP < eNZ() ; iP++ )
        {
            if ( ( stepde(iP) < 0 ) && ( ( deRestrict(pivRowZ()(iP)) == 1 ) || ( deRestrict(pivRowZ()(iP)) == 0 ) ) )
            {
                temp = -de(pivRowZ()(iP))/stepde(iP);

                if ( ( temp < scale )  || !res )
                {
                    scale = temp;
                    res = 1;
                    eFIndex = iP;
                    alphaFIndex = -1;
                }
            }

            if ( ( stepde(iP) > 0 ) && ( ( deRestrict(pivRowZ()(iP)) == 2 ) || ( deRestrict(pivRowZ()(iP)) == 0 ) ) )
            {
                temp = de(pivRowZ()(iP))/stepde(iP);

                if ( ( temp < scale )  || !res )
                {
                    scale = temp;
                    res = 1;
                    eFIndex = iP;
                    alphaFIndex = -1;
                }
            }
        }
    }

    return res;
}

void optLinState::stepalpha(int i, double alphaStep, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    dalpha("&",i) += alphaStep;

    if ( aN() )
    {
        int j;

        for ( j = 0 ; j < eN() ; j++ )
        {
            de("&",j) += G(j,i)*alphaStep;
        }
    }

    return;
}

void optLinState::stepFGeneral(const Vector<double> &alphaFStep, const Vector<double> &eZStep, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( alphaFStep.size() == aNF() );
    NiceAssert( eZStep.size() == eNZ() );

    retVector<double> tmpva;

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    dalpha("&",pivAlphaF(),tmpva) += alphaFStep;
    de    ("&",pivRowZ  (),tmpva) += eZStep;

    return;
}

void optLinState::calcStep(Vector<double> &stepAlpha, Vector<double> &stepde, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );
    NiceAssert( stepAlpha.size() == aN() );
    NiceAssert( stepde.size() == eN() );

    if ( eBad.size() || alphaGradBad.size() )
    {
        fixalphaGradBad(G,c,g);
        fixeBad(G,c,g);
    }

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retMatrix<double> tmpma;

    stepAlpha("&",pivAlphaF(),tmpva) = dalphaGrad(pivAlphaF(),tmpvb);
    stepAlpha("&",pivAlphaF(),tmpva).negate();

    mult(stepde("&",pivRowZ(),tmpva),G(pivRowZ(),pivAlphaF(),tmpma),stepAlpha(pivAlphaF(),tmpvb));

    return;
}

int optLinState::maxGradNonOpt(int &alphaZIndex, int &eFIndex, double &gradmag, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    (void) G;
    (void) c;
    (void) g;

    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    int res = 1;
    int iP;

    alphaZIndex = -1;
    eFIndex = -1;
    gradmag = -dopttol;

    if ( aNZ() )
    {
        for ( iP = 0 ; iP < aNZ() ; iP++ )
        {
            if ( dalphaGrad(pivAlphaZ()(iP)) < gradmag )
            {
                alphaZIndex = iP;
                eFIndex = -1;
                gradmag = dalphaGrad(pivAlphaZ()(iP));
                res = 0;
            }
        }
    }

    if ( eNF() )
    {
        for ( iP = 0 ; iP < eNF() ; iP++ )
        {
            if ( (  db(pivRowF()(iP)) < gradmag ) && ( ( deRestrict(pivRowF()(iP)) == 1 ) || ( deRestrict(pivRowF()(iP)) == 3 ) ) )
            {
                alphaZIndex = -1;
                eFIndex = iP;
                gradmag =  db(pivRowF()(iP));
                res = 0;
            }

            if ( ( -db(pivRowF()(iP)) < gradmag ) && ( ( deRestrict(pivRowF()(iP)) == 2 ) || ( deRestrict(pivRowF()(iP)) == 3 ) ) )
            {
                alphaZIndex = -1;
                eFIndex = iP;
                gradmag = -db(pivRowF()(iP));
                res = 0;
            }
        }
    }

    return res;
}











void optLinState::fixalphaGradBad(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    (void) g;

    if ( alphaGradBad.size() )
    {
        retVector<double> tmpva;
        retVector<double> tmpvb;
        retMatrix<double> tmpma;

        mult(dalphaGrad("&",alphaGradBad,tmpva),db(pivRowF(),tmpvb),G(pivRowF(),tmpma));
        dalphaGrad("&",alphaGradBad,tmpva).negate();
        dalphaGrad("&",alphaGradBad,tmpva) += c(alphaGradBad,tmpvb);

        alphaGradBad.resize(0);
    }

    return;
}

void optLinState::fixeBad(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    (void) c;

    if ( eBad.size() )
    {
        retVector<double> tmpva;
        retVector<double> tmpvb;

        mult(de("&",eBad,tmpva),G,dalpha);
        de("&",eBad,tmpva) += g(eBad,tmpvb);

        eBad.resize(0);
    }

    return;
}


void optLinState::recalcAlphaGrad(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    (void) g;

    // Fix gradients
    //
    // - alphaGrad: see above alphaGrad = ( c - Gr:'.br )

    retVector<double> tmpva;
    retMatrix<double> tmpma;

    mult(dalphaGrad,db(pivRowF(),tmpva),G(pivRowF(),tmpma));
    dalphaGrad.negate();
    dalphaGrad += c;

    return;
}

void optLinState::recalce(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    (void) c;

    // Fix gradients
    //
    // - alphaGrad: see above alphaGrad = ( c - Gr:'.br )

    mult(de,G,dalpha);
    de += g;

    return;
}


void optLinState::recalcAlphaGrad(int i, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    (void) g;

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retMatrix<double> tmpma;

    mult(dalphaGrad("&",i,1,i,tmpva),db(pivRowF(),tmpvb),G(pivRowF(),i,tmpma));
    dalphaGrad("&",i) = c(i)-dalphaGrad(i);

    return;
}

void optLinState::recalce(int i, const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    NiceAssert( G.numRows() == eN() );
    NiceAssert( G.numCols() == aN() );
    NiceAssert( c.size() == aN() );
    NiceAssert( g.size() == eN() );

    (void) c;

    retVector<double> tmpva;

    twoProductNoConj(de("&",i),G(i,tmpva),dalpha);
    de("&",i) += g(i);

    return;
}


void optLinState::recalcb(const Matrix<double> &G, const Vector<double> &c, const Vector<double> &g)
{
    (void) g;

    // Recalculate db:
    //
    // - b: br = inv(Gra.Gra').Gra.ca  (r = pivRowF, a = pivAlphaF)
    //      bz = 0                     (z = pivRowZ)

    retVector<double> tmpva;
    retVector<double> tmpvb;
    retMatrix<double> tmpma;

    mult(btemp("&",pivRowF(),tmpva),G(pivRowF(),pivAlphaF(),tmpma),c(pivAlphaF(),tmpvb));
    Q.minverse(db,btemp);
    db("&",pivRowZ(),tmpva).zero();

    return;
}

