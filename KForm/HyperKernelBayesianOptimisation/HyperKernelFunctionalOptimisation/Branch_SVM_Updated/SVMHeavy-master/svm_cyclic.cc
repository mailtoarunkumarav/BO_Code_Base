
//
// Cyclic regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_cyclic.h"
#include <iostream>
#include <sstream>
#include <string>











SVM_Cyclic::SVM_Cyclic() : SVM_Planar()
{
    thisthis = this;
    thisthisthis = &thisthis;

    cyceps = DEFAULTCYCEPS;

    setaltx(NULL);

    SVM_Planar::seteps(0.0);
    SVM_Planar::setDefaultProjectionVV(0);

    return;
}

SVM_Cyclic::SVM_Cyclic(const SVM_Cyclic &src) : SVM_Planar()
{
    thisthis = this;
    thisthisthis = &thisthis;

    cyceps = DEFAULTCYCEPS;

    setaltx(NULL);

    SVM_Planar::seteps(0.0);
    SVM_Planar::setDefaultProjectionVV(0);

    assign(src,0);

    return;
}

SVM_Cyclic::SVM_Cyclic(const SVM_Cyclic &src, const ML_Base *xsrc) : SVM_Planar()
{
    thisthis = this;
    thisthisthis = &thisthis;

    cyceps = DEFAULTCYCEPS;

    setaltx(xsrc);

    SVM_Planar::seteps(0.0);
    SVM_Planar::setDefaultProjectionVV(0);

    assign(src,1);

    return;
}

SVM_Cyclic::~SVM_Cyclic()
{
    return;
}

int SVM_Cyclic::prealloc(int expectedN)
{
    locy.prealloc(expectedN);
    locyg.prealloc(expectedN);
    locd.prealloc(expectedN);
    cycepsweight.prealloc(expectedN);

    return SVM_Planar::prealloc(4*expectedN);
}

std::ostream &SVM_Cyclic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Cyclic SVM\n\n";

    repPrint(output,'>',dep) << "y:           " << locy         << "\n";
    repPrint(output,'>',dep) << "y again:     " << locyg        << "\n";
    repPrint(output,'>',dep) << "d:           " << locd         << "\n";
    repPrint(output,'>',dep) << "eps weight:  " << cycepsweight << "\n";
    repPrint(output,'>',dep) << "sigma angle: " << cyceps       << "\n";

    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base planar SVM: ";
    SVM_Planar::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_Cyclic::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> locy;
    input >> dummy; input >> locyg;
    input >> dummy; input >> locd;
    input >> dummy; input >> cycepsweight;
    input >> dummy; input >> cyceps;
    input >> dummy;

    SVM_Planar::inputstream(input);

    return input;
}

int SVM_Cyclic::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Cyclic::N() );
    NiceAssert( !SVM_Cyclic::tspaceDim() || SVM_Cyclic::tspaceDim() == z.size() );
    NiceAssert( epsweigh >= 0 );
    NiceAssert( epsweigh <= 1 );

    int res = 1;

    locy.add(i);  locy("&",i)  = (const Vector<double> &) z;
    locyg.add(i); locyg("&",i) = z;
    locd.add(i);  locd("&",i)  = 2;

    cycepsweight.add(i); cycepsweight("&",i) = epsweigh;

    if ( SVM_Cyclic::N() == 1 )
    {
        Vector<gentype> o(SVM_Cyclic::tspaceDim());
        Vector<double> t(SVM_Cyclic::tspaceDim());

        int j;

        for ( j = 0 ; j < SVM_Cyclic::tspaceDim() ; j++ )
        {
            t = 0.0;
            t("&",j) = 1.0;

            o("&",j) = t;
        }

        SVM_Cyclic::setBasisVV(o);
    }

    res |= SVM_Planar::qaddTrainingVector(i,0.0,x,Cweigh,1.0,0); // Placeholder, includes x (for reference), d = 0

    int j;
    gentype vij;
    double qval;

    for ( j = 0 ; j < SVM_Cyclic::tspaceDim() ; j++ )
    {
        calcvij(vij,qval,i,j);

        SparseVector<gentype> tempx;

        tempx.fff("&",0) = i;
        tempx.fff("&",7) = vij;

        res |= SVM_Planar::qaddTrainingVector(i+((j+1)*(SVM_Cyclic::N())),qval,tempx,Cweigh,1.0);
        res |= SVM_Planar::setd(i+((j+1)*SVM_Cyclic::N()),1);
    }

    return res;    
}

int SVM_Cyclic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Cyclic::N() );

    y = locy(i);

    int res = 1;
    int j;

    gentype ydummy;
    SparseVector<gentype> xdummy;

    for ( j = SVM_Cyclic::tspaceDim()-1 ; j >= 0 ; j-- )
    {
        res |= SVM_Planar::removeTrainingVector(i+((j+1)*SVM_Cyclic::N()),ydummy,xdummy);
    }

    res |= SVM_Planar::removeTrainingVector(i,ydummy,x);

    if ( SVM_Cyclic::N() == 0 )
    {
        Vector<gentype> o; // An empty vector - that is, no basis

        SVM_Cyclic::setBasisVV(o);
    }

    locy.remove(i);
    locyg.remove(i);
    locd.remove(i);
    cycepsweight.remove(i);

    return res;    
}

int SVM_Cyclic::sety(int i, const gentype &z)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Cyclic::N() );

    locy("&",i)  = (const Vector<double> &) z;
    locyg("&",i) = z;

    int res = 0;
    int j;
    gentype vij;

    SparseVector<gentype> tempx;
    double qval;

    for ( j = 0 ; j < SVM_Cyclic::tspaceDim() ; j++ )
    {
        calcvij(vij,qval,i,j);

        tempx.fff("&",0) = i;
        tempx.fff("&",7) = vij;

        res |= SVM_Planar::setx(i+((j+1)*SVM_Cyclic::N()),tempx);
    }

    return res;
}

int SVM_Cyclic::setd(int i, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Cyclic::N() );
    NiceAssert( ( d == 0 ) || ( d == 2 ) );

    locd("&",i) = d;

    int res = 0;
    int j;

    for ( j = 0 ; j < SVM_Cyclic::tspaceDim() ; j++ )
    {
        res |= SVM_Planar::setd(i+((j+1)*SVM_Cyclic::N()),d/2);
    }

    return res;
}

int SVM_Cyclic::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    if ( SVM_Cyclic::N() )
    {
        (**thisthisthis).SVM_Planar::setDefaultProjectionVV(-1);
    }

    int res = SVM_Planar::ghTrainingVector(resh,resg,i,retaltg,pxyprodi);

    double absresg = (double) abs2(resg);

    resh = resg/( ( absresg >= zerotol() ) ? absresg : 1.0 );

    if ( SVM_Cyclic::N() )
    {
        (**thisthisthis).SVM_Planar::setDefaultProjectionVV(0);
    }

    return res;
}

int SVM_Cyclic::disable(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Cyclic::N() );

    int res = 0;
    int j;

    for ( j = 0 ; j < SVM_Cyclic::tspaceDim() ; j++ )
    {
        res |= SVM_Planar::disable(i+((j+1)*SVM_Cyclic::N()));
    }

    return res;
}

int SVM_Cyclic::seteps(double xeps)
{
    NiceAssert( xeps >= 0 );
    NiceAssert( xeps <= 1 );

    cyceps = xeps;

    int res = 0;
    int i,j;
    gentype vij;

    SparseVector<gentype> tempx;
    double qval;

    for ( i = 0 ; i < SVM_Cyclic::N() ; i++ )
    {
        for ( j = 0 ; j < SVM_Cyclic::tspaceDim() ; j++ )
        {
            calcvij(vij,qval,i,j);

            tempx.fff("&",0) = i;
            tempx.fff("&",7) = vij;

            res |= SVM_Planar::setx(i+((j+1)*SVM_Cyclic::N()),tempx);
            res |= SVM_Planar::sety(i+((j+1)*SVM_Cyclic::N()),qval);
        }
    }

    return res;
}

int SVM_Cyclic::setepsweight(int i, double xepsweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Cyclic::N() );
    NiceAssert( xepsweight >= 0 );
    NiceAssert( xepsweight <= 1 );

    cycepsweight("&",i) = xepsweight;

    int res = 0;
    int j;
    gentype vij;

    SparseVector<gentype> tempx;
    double qval;

    for ( j = 0 ; j < SVM_Cyclic::tspaceDim() ; j++ )
    {
        calcvij(vij,qval,i,j);

        tempx.fff("&",0) = i;
        tempx.fff("&",7) = vij;

        res |= SVM_Planar::setx(i+((j+1)*SVM_Cyclic::N()),tempx);
        res |= SVM_Planar::sety(i+((j+1)*SVM_Cyclic::N()),qval);
    }

    return res;
}

gentype &SVM_Cyclic::calcvij(gentype &res, double &qval, int i, int j)
{
    Vector<double> yval(SVM_Cyclic::tspaceDim());
    Vector<double> tval(SVM_Cyclic::tspaceDim());
    Vector<double> delj(SVM_Cyclic::tspaceDim());

    yval = (const Vector<double> &) y()(i); yval /= abs2(yval);
    tval = 1.0/sqrt(SVM_Cyclic::tspaceDim());
    delj = 0.0; delj("&",j) = 1.0;

    Vector<double> ellj(SVM_Cyclic::tspaceDim());

    // l_j = ( delta_j - ((1,1,...).(1,1,...)'delta_j)/||(1,1,...)||_2^2 ) normalised
    //     = ( delta_j - (1,1,...)/n ) normalised

    ellj  = tval;
    ellj /= -sqrt(SVM_Cyclic::tspaceDim());
    ellj += delj;
    ellj /= abs2(ellj);

    Vector<double> vj(SVM_Cyclic::tspaceDim());

    // v_j = ( eps.delta_j + (1-eps).ellj ) normalised
    //     = eps.( delta_j + ((1-eps)/eps).ellj ) normalised

    if ( cyceps < zerotol() )
    {
        vj = ellj;
    }

    else if ( cyceps == 1.0 )
    {
        vj = delj;
    }

    else
    {
        vj  = ellj;
        vj *= (1-cyceps)/cyceps;
        vj += delj;
        vj *= cyceps;
        vj /= abs2(vj);
    }

    // q = sum(vj)/sqrt(n)

    qval = sum(vj)/sqrt(SVM_Cyclic::tspaceDim());

    // v_ij = ( I - 2.( y - (1,1,...,1)/sqrt(n) ).( y - (1,1,...,1)/sqrt(n) )'/|| y - (1,1,...,1)/sqrt(n) ||_2^2 ).v_j (which is already normalised)

    if ( abs2(yval-tval) > zerotol() )
    {
        Vector<double> refij(SVM_Cyclic::tspaceDim());

        refij  = tval;
        refij -= yval;
        refij /= abs2(refij);

        double inprod;

        twoProductNoConj(inprod,refij,vj);

        res  = refij;
        res *= -2*inprod;
        res += vj;
    }

    else
    {
        res = vj;
    }

    return res;
}

double SVM_Cyclic::calcDist(const gentype &ha, const gentype &hb, int i, int db) const
{
    (void) i;

    double res = 0;

    if ( db )
    {
        double haabs = (double) abs2(ha);
        double hbabs = (double) abs2(hb);

        if ( ( haabs >= zerotol() ) && ( hbabs >= zerotol() ) )
        {
            res = (double) norm2((ha/haabs)-(hb/hbabs));
        }

        else if ( haabs >= zerotol() )
        {
            res = haabs*haabs;
        }

        else if ( hbabs >= zerotol() )
        {
            res = hbabs*hbabs;
        }

        else
        {
            res = 0.0;
        }
    }

    return res;
}

int SVM_Cyclic::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    int res = 0;

    if ( onlyChangeRowI < 0 )
    {
        res = SVM_Planar::resetKernel(modind,onlyChangeRowI,updateInfo);
    }

    else
    {
        int i;

        for ( i = 0 ; i <= tspaceDim() ; i++ )
        {
            res |= SVM_Planar::resetKernel(modind,onlyChangeRowI+(i*(SVM_Cyclic::N())),updateInfo);
        }
    }

    return res;
}

int SVM_Cyclic::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    int res = 0;

    if ( onlyChangeRowI < 0 )
    {
        res = SVM_Planar::setKernel(xkernel,modind,onlyChangeRowI);
    }

    else
    {
        int i;

        for ( i = 0 ; i <= tspaceDim() ; i++ )
        {
            res |= SVM_Planar::setKernel(xkernel,modind,onlyChangeRowI+(i*(SVM_Cyclic::N())));
        }
    }

    return res;
}

int SVM_Cyclic::setx(int i, const SparseVector<gentype> &xx)
{
    int res = SVM_Planar::setx(i,xx);
    int j;

    for ( j = 1 ; j <= tspaceDim() ; j++ )
    {
        res |= SVM_Planar::resetKernel(1,i+(j*(SVM_Cyclic::N())));
    }

    return res;
}

int SVM_Cyclic::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &xx)
{
    int j;
    int res = 0;

    for ( j = 0 ; j < i.size() ; j++ )
    {
        res |= SVM_Cyclic::setx(i(j),xx(j));
    }

    return res;
}

int SVM_Cyclic::setx(const Vector<SparseVector<gentype> > &xx)
{
    int j;
    int res = 0;

    for ( j = 0 ; j < SVM_Cyclic::N() ; j++ )
    {
        res |= SVM_Cyclic::setx(j,xx(j));
    }

    return res;
}
























int SVM_Cyclic::sety(const Vector<int> &i, const Vector<gentype> &z)
{
    NiceAssert( i.size() == z.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= SVM_Cyclic::sety(i(j),z(j));
        }
    }

    return res;
}

int SVM_Cyclic::sety(const Vector<gentype> &z)
{
    NiceAssert( z.size() == SVM_Cyclic::N() );

    int res = 0;

    if ( SVM_Cyclic::N() )
    {
        int j;

        for ( j = 0 ; j < SVM_Cyclic::N() ; j++ )
        {
            res |= SVM_Cyclic::sety(j,z(j));
        }
    }

    return res;
}

int SVM_Cyclic::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i.size() == d.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= SVM_Cyclic::setd(i(j),d(j));
        }
    }

    return res;
}

int SVM_Cyclic::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == SVM_Cyclic::N() );

    int res = 0;

    if ( SVM_Cyclic::N() )
    {
        int j;

        for ( j = 0 ; j < SVM_Cyclic::N() ; j++ )
        {
            res |= SVM_Cyclic::setd(j,d(j));
        }
    }

    return res;
}

int SVM_Cyclic::disable(const Vector<int> &i)
{
    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= SVM_Cyclic::disable(i(j));
        }
    }

    return res;
}

int SVM_Cyclic::setepsweight(const Vector<int> &i, const Vector<double> &d)
{
    NiceAssert( i.size() == d.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= SVM_Cyclic::setepsweight(i(j),d(j));
        }
    }

    return res;
}

int SVM_Cyclic::setepsweight(const Vector<double> &d)
{
    NiceAssert( d.size() == SVM_Cyclic::N() );

    int res = 0;

    if ( SVM_Cyclic::N() )
    {
        int j;

        for ( j = 0 ; j < SVM_Cyclic::N() ; j++ )
        {
            res |= SVM_Cyclic::setepsweight(j,d(j));
        }
    }

    return res;
}

int SVM_Cyclic::removeTrainingVector(int i, int num)
{
    NiceAssert( i < SVM_Cyclic::N() );
    NiceAssert( num >= 0 );
    NiceAssert( num <= SVM_Cyclic::N()-i );

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

int SVM_Cyclic::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xx(x);

    return qaddTrainingVector(i,z,xx,Cweigh,epsweigh);
}

int SVM_Cyclic::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &xx, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Cyclic::N() );
    NiceAssert( z.size() == xx.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );


    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Cyclic::addTrainingVector(i+j,z(j),xx(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_Cyclic::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Cyclic::N() );
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );


    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_Cyclic::qaddTrainingVector(i+j,z(j),x("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

