
//
// Linear optimisation context
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "optlincontext.h"

// Stream operators

std::ostream &operator<<(std::ostream &output, const optLinContext &src)
{
    output << "H:                   " << src.H           << "\n";
    output << "Free alpha pivot:    " << src.xpivAlphaF  << "\n";
    output << "Zero alpha pivot:    " << src.xpivAlphaZ  << "\n";
    output << "Alpha state:         " << src.xalphaState << "\n";
    output << "Unset H row/columns: " << src.Hblanks     << "\n";
    output << "Q:                   " << src.Q           << "\n";

    return output;
}

std::istream &operator>>(std::istream &input, optLinContext &dest)
{
    wait_dummy dummy;

    input >> dummy; input >> dest.H;
    input >> dummy; input >> dest.xpivAlphaF;
    input >> dummy; input >> dest.xpivAlphaZ;
    input >> dummy; input >> dest.xalphaState;
    input >> dummy; input >> dest.Hblanks;
    input >> dummy; input >> dest.Q;

    (dest.rtempa).resize(dest.rowN());
    (dest.rtempb).resize(dest.rowN());

    return input;
}


optLinContext::optLinContext(void)
{
    return;
}

optLinContext::optLinContext(const optLinContext &src)
{
    *this = src;

    return;
}

optLinContext &optLinContext::operator=(const optLinContext &src)
{
    Q = src.Q;
    H = src.H;

    xpivAlphaF  = src.xpivAlphaF;
    xpivAlphaZ  = src.xpivAlphaZ;
    xalphaState = src.xalphaState;
    rtempa      = src.rtempa;
    rtempb      = src.rtempb;

    Hblanks = src.Hblanks;

    return *this;
}

void optLinContext::refact(const Matrix<double> &G, double xzt)
{
    NiceAssert( G.numRows() == rowN() );
    NiceAssert( G.numCols() == aN() );

    Hblanks.resize(0);

    H.resize(rowN(),rowN());

    if ( rowN() )
    {
        int i,j;

        retVector<double> tmpva;
        retVector<double> tmpvb;
        retVector<double> tmpvc;
        retVector<double> tmpvd;

        for ( i = 0 ; i < rowN() ; i++ )
        {
            for ( j = 0 ; j < rowN() ; j++ )
            {
                twoProduct(H("&",i,j),G(i,pivAlphaF(),tmpva,tmpvb),G(j,pivAlphaF(),tmpvc,tmpvd));
            }
        }
    }

    //xalphaState = zeroint();

    Q.refact(H,xzt);

    return;
}

void optLinContext::reset(const Matrix<double> &G)
{
    (void) G;

    NiceAssert( G.numRows() == rowN() );
    NiceAssert( G.numCols() == aN() );

    Hblanks.resize(0);

    H = 0.0;

    xpivAlphaF.resize(0);
    xpivAlphaZ.resize(aN());

    retVector<int> tmpva;

    xpivAlphaZ = cntintvec(aN(),tmpva);

    xalphaState = zeroint();

    Q.reset(H);

    return;
}

int optLinContext::addRow(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= rowN() );

    int ucnt = rowN()-i;

    int res = Q.add(i); // important to do this first! rowN and aN must be updated.
    H.addRowCol(i);
    rtempa.add(i);
    rtempb.add(i);

    // Cannot actually fill in H at this point, so need to record row added
    // (and update indexing as relevant)

    if ( Hblanks.size() && ucnt )
    {
        int iP;

        for ( iP = 0 ; ( iP < Hblanks.size() ) && ucnt ; iP++ )
        {
            if ( Hblanks(iP) >= i )
            {
                Hblanks("&",iP)++;
                ucnt--;
            }
        }
    }

    Hblanks.add(Hblanks.size());
    Hblanks("&",Hblanks.size()-1) = i;

    return res;
}

int optLinContext::removeRow(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < rowN() );

    int ucnt = rowN()-1-i;

    int res = Q.remove(i);
    H.removeRowCol(i);
    rtempa.remove(i);
    rtempb.remove(i);

    // Need to update Hblanks to fix indexing for next fixH call.  May also
    // need to remove i from Hblanks if present.

    if ( Hblanks.size() )
    {
        int iP;

        for ( iP = 0 ; iP < Hblanks.size() ; iP++ )
        {
            if ( Hblanks(iP) == i )
            {
                break;
            }
        }

        if ( iP < Hblanks.size() )
        {
            Hblanks.remove(iP);
        }
    }

    if ( Hblanks.size() && ucnt )
    {
        int iP;

        for ( iP = 0 ; ( iP < Hblanks.size() ) && ucnt ; iP++ )
        {
            if ( Hblanks(iP) > i )
            {
                Hblanks("&",iP)--;
                ucnt--;
            }
        }
    }

    return res;
}

int optLinContext::addAlpha(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= aN() );

    int ucnt = aN()-i;

    if ( aNZ() && ucnt )
    {
        int iP;

        for ( iP = 0 ; ( iP < aNZ() ) && ucnt ; iP++ )
        {
            if ( xpivAlphaZ(iP) >= i )
            {
                xpivAlphaZ("&",iP)++;
                ucnt--;
            }
        }
    }

    if ( aNF() && ucnt )
    {
        int iP;

        for ( iP = 0 ; ( iP < aNF() ) && ucnt ; iP++ )
        {
            if ( xpivAlphaF(iP) >= i )
            {
                xpivAlphaF("&",iP)++;
                ucnt--;
            }
        }
    }

    int res = xpivAlphaZ.size();

    xpivAlphaZ.add(res);
    xpivAlphaZ("&",res) = i;

    xalphaState.add(i);
    xalphaState("&",i) = 0;

    return res;
}

int optLinContext::removeAlpha(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );

    int ucnt = aN()-1-i;
    int iP = findInAlphaZ(i);

    xpivAlphaZ.remove(iP);
    xalphaState.remove(i);

    if ( aNZ() && ucnt )
    {
        for ( iP = 0 ; ( iP < aNZ() ) && ucnt ; iP++ )
        {
            if ( xpivAlphaZ(iP) > i )
            {
                xpivAlphaZ("&",iP)--;
                ucnt--;
            }
        }
    }

    if ( aNF() )
    {
        for ( iP = 0 ; ( iP < aNF() ) && ucnt ; iP++ )
        {
            if ( xpivAlphaF(iP) > i )
            {
                xpivAlphaF("&",iP)--;
                ucnt--;
            }
        }
    }

    return iP;
}

int optLinContext::findInRowZ(int i) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < rowN() );

    return Q.findInZ(i);
}

int optLinContext::findInRowF(int i) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < rowN() );

    return Q.findInF(i);
}

int optLinContext::findInAlphaZ(int i) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );
    NiceAssert( aNZ() );

    int iP;

    for ( iP = 0 ; iP < aNZ() ; iP++ )
    {
        if ( xpivAlphaZ(iP) == i )
        {
            break;
        }
    }

    NiceAssert( iP < aNZ() );

    return iP;
}

int optLinContext::findInAlphaF(int i) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < aN() );
    NiceAssert( aNF() );

    int iP;

    for ( iP = 0 ; iP < aNF() ; iP++ )
    {
        if ( xpivAlphaF(iP) == i )
        {
            break;
        }
    }

    NiceAssert( iP < aNF() );

    return iP;
}

int optLinContext::modRowZtoF(int iP, const Matrix<double> &G)
{
    if ( Hblanks.size() )
    {
        fixH(G);
    }

    return Q.modZtoF(iP,H);
}

int optLinContext::modRowFtoZ(int iP, const Matrix<double> &G)
{
    if ( Hblanks.size() )
    {
        fixH(G);
    }

    return Q.modFtoZ(iP,H);
}

int optLinContext::modAlphaZtoF(int iP, const Matrix<double> &G)
{
    if ( Hblanks.size() )
    {
        fixH(G);
    }

    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNZ() );

    int res = -1;

    // This involves a positive rank-1 update to H
    // Note that H in call is assumed to have already been updated

    int i = xpivAlphaZ(iP);
    xpivAlphaZ.remove(iP);
    res = xpivAlphaF.size();
    xpivAlphaF.add(res);
    xpivAlphaF("&",res) = i;
    xalphaState("&",i) = 1;

    retMatrix<double> tmpma;

    rtempa = G(0,1,rowN()-1,i,tmpma,"c");

    H.rankone(1.0,rtempa,rtempa);
    Q.rankone(rtempa,1.0,H);

    return res;
}

int optLinContext::modAlphaFtoZ(int iP, const Matrix<double> &G)
{
    if ( Hblanks.size() )
    {
        fixH(G);
    }

    NiceAssert( iP >= 0 );
    NiceAssert( iP < aNF() );

    int res = -1;

    // This involves a negative rank-1 update to H
    // Note that H in call is assumed to have already been updated

    int i = xpivAlphaF(iP);
    xpivAlphaF.remove(iP);
    res = xpivAlphaZ.size();
    xpivAlphaZ.add(res);
    xpivAlphaZ("&",res) = i;
    xalphaState("&",i) = 0;

    retMatrix<double> tmpma;

    rtempa = G(0,1,rowN()-1,i,tmpma,"c");

    H.rankone(-1.0,rtempa,rtempa);
    Q.rankone(rtempa,-1.0,H);

    return res;
}

int optLinContext::minverse(Vector<double> &ap, const Vector<double> &bp) const
{
    NiceAssert( bp.size() == rowN() );

    int res = rowpfact();

    ap.resize(rowN());

    Q.minverse(ap,bp);

    return res;
}

int optLinContext::project(Vector<double> &ap, const Vector<double> &bp, const Matrix<double> &G)
{
    NiceAssert( bp.size() == aN() );

    int res = rowpfact();

    ap.resize(aN());

    // rtempb = G.bp

    retVector<int>    tmpva;
    retVector<int>    tmpvb;
    retVector<double> tmpvc;
    retVector<double> tmpvd;
    retMatrix<double> tmpma;

    mult(rtempb("&",pivRowF()(0,1,res-1,tmpva),tmpvc),G(pivRowF()(0,1,res-1,tmpvb),pivAlphaF(),tmpma),bp(pivAlphaF(),tmpvd));
    rtempb("&",pivRowF()(res,1,rowNF()-1,tmpva),tmpvc) = 0.0; // Not strictly needed I suspect

    // rtempa = inv(H).rtempb

    minverse(rtempa,rtempb);

    // ap = G'.rtempa

    mult(ap("&",pivAlphaF(),tmpvc),rtempa(pivRowF()(0,1,res-1,tmpva),tmpvd),G(pivRowF()(0,1,res-1,tmpvb),pivAlphaF(),tmpma));

    return res;
}


void optLinContext::fixH(const Matrix<double> &G)
{
    if ( Hblanks.size() )
    {
        int i,j;

        retVector<double> tmpva;
        retVector<double> tmpvb;
        retVector<double> tmpvc;
        retVector<double> tmpvd;

        for ( i = Hblanks.size()-1 ; i >= 0 ; i-- )
        {
            for ( j = 0 ; j < rowN() ; j++ )
            {
                twoProduct(H("&",i,j),G(i,pivAlphaF(),tmpva,tmpvb),G(j,pivAlphaF(),tmpvc,tmpvd));
                H("&",j,i) = H(i,j);
            }

            Hblanks.remove(i);
        }

        NiceAssert( !(Hblanks.size()) );
    }

    return;
}

