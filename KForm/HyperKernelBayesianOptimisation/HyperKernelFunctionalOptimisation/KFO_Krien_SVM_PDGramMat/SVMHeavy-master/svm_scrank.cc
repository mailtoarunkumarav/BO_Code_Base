
//
// Scalar+Ranking SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_scrank.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


// The Gpn vector (actually a single-column matrix) controls whether the
// bias is included with the relevant parts of the Hessian.  The bias is
// not present if the constraint has the form:
//
// g(xi) ? g(xj)+y
//
// as the biases on either side of the ? cancel out; but is present
// otherwise.
//
// Generally:
//
// g(i) $ g(j) is x = [ i j ] , Gpn = [ 0 ] (no bias)
// g(i) $ 0 is x = [ 0:i ]    , Gpn = [ 1 ] (bias present)
// 0 $ g(j) is x = [ 1:j ]    , Gpn = [ -1 ] (negative bias present)
// 0 $ 0 is x = [ ]           , Gpn = [ 0 ] (no bias)

double calcGpn(int d, const SparseVector<gentype> &x);
double calcGpn(int d, const SparseVector<gentype> &x)
{
    double res = 1;

    if ( d )
    {
        if ( x.isindpresent(0) && x.isindpresent(1) )
        {
            res = 0;
        }

        else if ( !(x.isindpresent(0)) && !(x.isindpresent(1)) )
        {
            res = 0;
        }

        else if ( x.isindpresent(0) )
        {
            // So x.isindpresent(0) && !x.isindpresent(1)

            res = 1;
        }

        else
        {
            // So !x.isindpresent(0) && x.isindpresent(1)

            res = -1;
        }
    }

    else
    {
        res = 1;
    }

    return res;
}




SVM_ScRank::SVM_ScRank() : SVM_Scalar()
{
    thisthis = this;
    thisthisthis = &thisthis;

    evalKReal    = NULL;
    evalKRealarg = NULL;

    setaltx(NULL);

    locTestMode = 1;

    inGpn.resize(0,1);

    SVM_Scalar::setGpnExt(NULL,&inGpn);

    return;
}

SVM_ScRank::SVM_ScRank(const SVM_ScRank &src) : SVM_Scalar()
{
    thisthis = this;
    thisthisthis = &thisthis;

    evalKReal    = NULL;
    evalKRealarg = NULL;

    setaltx(NULL);

    locTestMode = 1;

    inGpn.resize(0,1);

    SVM_Scalar::setGpnExt(NULL,&inGpn);

    assign(src,0);

    return;
}

SVM_ScRank::SVM_ScRank(const SVM_ScRank &src, const ML_Base *xsrc) : SVM_Scalar()
{
    thisthis = this;
    thisthisthis = &thisthis;

    evalKReal    = NULL;
    evalKRealarg = NULL;

    setaltx(xsrc);

    locTestMode = 1;

    inGpn.resize(0,1);

    SVM_Scalar::setGpnExt(NULL,&inGpn);

    assign(src,1);

    return;
}

SVM_ScRank::~SVM_ScRank()
{
    return;
}

std::ostream &operator<<(std::ostream &output, const SVM_ScRank &src)
{
    return src.printstream(output);
}

std::istream &operator>>(std::istream &input, SVM_ScRank &dest)
{
    return dest.inputstream(input);
}

std::ostream &SVM_ScRank::printstream(std::ostream &output) const
{
    output << "Scalar Ranking SVM\n\n";

    output << "locTestMode:  " << locTestMode << "\n";
    output << "override Gpn: " << inGpn       << "\n";
    output << "local d:      " << locd        << "\n";
    output << "=====================================================================\n";
    output << "Base SVC: ";
    SVM_Scalar::printstream(output);
    output << "\n";
    output << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_ScRank::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> locTestMode;
    input >> dummy; input >> inGpn;
    input >> dummy; input >> locd;
    input >> dummy;
    SVM_Scalar::inputstream(input);

    return input;
}

double &SVM_ScRank::KReal(double &res, int i, int j) const
{
    // locTestMode = 1:
    //
    // i   | j   | di  | dj  |  modi  |  modj
    // ----+-----+-----+-----+--------+--------
    // < 0 | < 0 | ... | ... |   0    |   0
    // >=0 | < 0 |  0  | ... |   0    |   0
    // >=0 | < 0 | !0  | ... |   1    |   0
    // < 0 | >=0 | ... |  0  |   0    |   0
    // < 0 | >=0 | ... | !0  |   0    |   1
    // >=0 | >=0 |  0  |  0  |   0    |   0
    // >=0 | >=0 |  0  | !0  |   0    |   1
    // >=0 | >=0 | !0  |  0  |   1    |   0
    // >=0 | >=0 | !0  | !0  |   1    |   1
    //
    // modi = ( i >= 0 ) && d()(i)
    // modj = ( j >= 0 ) && d()(j)
    //
    // locTestMode = 0:
    //
    // i   | j   | di  | dj  |  modi  |  modj
    // ----+-----+-----+-----+--------+--------
    // < 0 | < 0 | ... | ... |   0    |   0
    // >=0 | < 0 |  0  | ... |   0    |   0
    // >=0 | < 0 | !0  | ... |   0    |   0
    // < 0 | >=0 | ... |  0  |   0    |   0
    // < 0 | >=0 | ... | !0  |   0    |   1
    // >=0 | >=0 |  0  |  0  |   0    |   0
    // >=0 | >=0 |  0  | !0  |   0    |   1
    // >=0 | >=0 | !0  |  0  |   0    |   0
    // >=0 | >=0 | !0  | !0  |   0    |   1
    //
    // modi = 0
    // modj = ( j >= 0 ) && d()(j)

    int modi = ( ( i >= 0 ) && locd(i) ) ? locTestMode : 0;
    int modj = ( ( j >= 0 ) && locd(j) ) ? 1           : 0;

    // Addendum: the above assumes that both indices are present - that
    // is, this relates to an inequality of the form
    //
    // g(x(i)(0)) ? g(x(j)(1)) + y
    //
    // this is not always the case - hence the isindpresent stuff below.

    double temp;







    if ( evalKReal )
    {
        if ( modi && modj )
        {
            if ( x(i).isindpresent(0) && x(i).isindpresent(1) )
            {
                if ( x(j).isindpresent(0) && x(j).isindpresent(1) )
                {
                    res = (*evalKReal)(temp,(int) x(i)(0),(int) x(j)(0),evalKRealarg)
                        + (*evalKReal)(temp,(int) x(i)(1),(int) x(j)(1),evalKRealarg)
                        - (*evalKReal)(temp,(int) x(i)(0),(int) x(j)(1),evalKRealarg)
                        - (*evalKReal)(temp,(int) x(i)(1),(int) x(j)(0),evalKRealarg);
                }

                else if ( x(j).isindpresent(0) )
                {
                    res = (*evalKReal)(temp,(int) x(i)(0),(int) x(j)(0),evalKRealarg)
                        - (*evalKReal)(temp,(int) x(i)(1),(int) x(j)(0),evalKRealarg);
                }

                else if ( x(j).isindpresent(1) )
                {
                    res = (*evalKReal)(temp,(int) x(i)(1),(int) x(j)(1),evalKRealarg)
                        - (*evalKReal)(temp,(int) x(i)(0),(int) x(j)(1),evalKRealarg);
                }

                else
                {
                    res = 0;
                }
            }

            else if ( x(i).isindpresent(0) )
            {
                if ( x(j).isindpresent(0) && x(j).isindpresent(1) )
                {
                    res = (*evalKReal)(temp,(int) x(i)(0),(int) x(j)(0),evalKRealarg)
                        - (*evalKReal)(temp,(int) x(i)(0),(int) x(j)(1),evalKRealarg);
                }

                else if ( x(j).isindpresent(0) )
                {
                    res = (*evalKReal)(temp,(int) x(i)(0),(int) x(j)(0),evalKRealarg);
                }

                else if ( x(j).isindpresent(1) )
                {
                    res = -((*evalKReal)(temp,(int) x(i)(0),(int) x(j)(1),evalKRealarg));
                }

                else
                {
                    res = 0;
                }
            }

            else if ( x(i).isindpresent(1) )
            {
                if ( x(j).isindpresent(0) && x(j).isindpresent(1) )
                {
                    res = (*evalKReal)(temp,(int) x(i)(1),(int) x(j)(1),evalKRealarg)
                        - (*evalKReal)(temp,(int) x(i)(1),(int) x(j)(0),evalKRealarg);
                }

                else if ( x(j).isindpresent(0) )
                {
                    res = -((*evalKReal)(temp,(int) x(i)(1),(int) x(j)(0),evalKRealarg));
                }

                else if ( x(j).isindpresent(1) )
                {
                    res = (*evalKReal)(temp,(int) x(i)(1),(int) x(j)(1),evalKRealarg);
                }

                else
                {
                    res = 0;
                }
            }

            else
            {
                res = 0;
            }
        }

        else if ( modi )
        {
            if ( x(i).isindpresent(0) && x(i).isindpresent(1) )
            {
                res = (*evalKReal)(temp,(int) x(i)(0),j,evalKRealarg)
                    - (*evalKReal)(temp,(int) x(i)(1),j,evalKRealarg);
            }

            else if ( x(i).isindpresent(0) )
            {
                res = (*evalKReal)(temp,(int) x(i)(0),j,evalKRealarg);
            }

            else if ( x(i).isindpresent(1) )
            {
                res = -((*evalKReal)(temp,(int) x(i)(1),j,evalKRealarg));
            }

            else
            {
                res = 0;
            }
        }

        else if ( modj )
        {
            if ( x(j).isindpresent(0) && x(j).isindpresent(1) )
            {
                res = (*evalKReal)(temp,i,(int) x(j)(0),evalKRealarg)
                    - (*evalKReal)(temp,i,(int) x(j)(1),evalKRealarg);
            }

            else if ( x(j).isindpresent(0) )
            {
                res = (*evalKReal)(temp,i,(int) x(j)(0),evalKRealarg);
            }

            else if ( x(j).isindpresent(1) )
            {
                res = -((*evalKReal)(temp,i,(int) x(j)(1),evalKRealarg));
            }

            else
            {
                res = 0;
            }
        }

        else
        {
            res = (*evalKReal)(temp,i,j,evalKRealarg);
        }
    }

    else
    {
        if ( modi && modj )
        {
            if ( x(i).isindpresent(0) && x(i).isindpresent(1) )
            {
                if ( x(j).isindpresent(0) && x(j).isindpresent(1) )
                {
                    res = ML_Base::KReal(temp,(int) x(i)(0),(int) x(j)(0))
                        + ML_Base::KReal(temp,(int) x(i)(1),(int) x(j)(1))
                        - ML_Base::KReal(temp,(int) x(i)(0),(int) x(j)(1))
                        - ML_Base::KReal(temp,(int) x(i)(1),(int) x(j)(0));
                }

                else if ( x(j).isindpresent(0) )
                {
                    res = ML_Base::KReal(temp,(int) x(i)(0),(int) x(j)(0))
                        - ML_Base::KReal(temp,(int) x(i)(1),(int) x(j)(0));
                }

                else if ( x(j).isindpresent(1) )
                {
                    res = ML_Base::KReal(temp,(int) x(i)(1),(int) x(j)(1))
                        - ML_Base::KReal(temp,(int) x(i)(0),(int) x(j)(1));
                }

                else
                {
                    res = 0;
                }
            }

            else if ( x(i).isindpresent(0) )
            {
                if ( x(j).isindpresent(0) && x(j).isindpresent(1) )
                {
                    res = ML_Base::KReal(temp,(int) x(i)(0),(int) x(j)(0))
                        - ML_Base::KReal(temp,(int) x(i)(0),(int) x(j)(1));
                }

                else if ( x(j).isindpresent(0) )
                {
                    res = ML_Base::KReal(temp,(int) x(i)(0),(int) x(j)(0));
                }

                else if ( x(j).isindpresent(1) )
                {
                    res = -(ML_Base::KReal(temp,(int) x(i)(0),(int) x(j)(1)));
                }

                else
                {
                    res = 0;
                }
            }

            else if ( x(i).isindpresent(1) )
            {
                if ( x(j).isindpresent(0) && x(j).isindpresent(1) )
                {
                    res = ML_Base::KReal(temp,(int) x(i)(1),(int) x(j)(1))
                        - ML_Base::KReal(temp,(int) x(i)(1),(int) x(j)(0));
                }


                else if ( x(j).isindpresent(0) )
                {
                    res = -(ML_Base::KReal(temp,(int) x(i)(1),(int) x(j)(0)));
                }

                else if ( x(j).isindpresent(1) )
                {
                    res = ML_Base::KReal(temp,(int) x(i)(1),(int) x(j)(1));
                }

                else
                {
                    res = 0;
                }
            }

            else
            {
                res = 0;
            }
        }

        else if ( modi )
        {
            if ( x(i).isindpresent(0) && x(i).isindpresent(1) )
            {
                res = ML_Base::KReal(temp,(int) x(i)(0),j)
                    - ML_Base::KReal(temp,(int) x(i)(1),j);
            }

            else if ( x(i).isindpresent(0) )
            {
                res = ML_Base::KReal(temp,(int) x(i)(0),j);
            }

            else if ( x(i).isindpresent(1) )
            {
                res = -(ML_Base::KReal(temp,(int) x(i)(1),j));
            }

            else
            {
                res = 0;
            }
        }

        else if ( modj )
        {
            if ( x(j).isindpresent(0) && x(j).isindpresent(1) )
            {
                res = ML_Base::KReal(temp,i,(int) x(j)(0))
                    - ML_Base::KReal(temp,i,(int) x(j)(1));
            }

            else if ( x(j).isindpresent(0) )
            {
                res = ML_Base::KReal(temp,i,(int) x(j)(0));
            }

            else if ( x(j).isindpresent(1) )
            {
                res = -(ML_Base::KReal(temp,i,(int) x(j)(1)));
            }

            else
            {
                res = 0;
            }
        }

        else
        {
            res = ML_Base::KReal(temp,i,j);
        }
    }

    return res;
}

// Evaluation:

int SVM_ScRank::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg) const
{
    int res;

    (**thisthisthis).locTestMode = isTestMode();
    res = SVM_Scalar::ghTrainingVector(resh,resg,i,retaltg);
    (**thisthisthis).locTestMode = 1;

    return res;
}

void SVM_ScRank::eTrainingVector(gentype &res, int i) const
{
    (**thisthisthis).locTestMode = isTestMode();
    SVM_Scalar::eTrainingVector(res,i);
    (**thisthisthis).locTestMode = 1;

    return;
}

void SVM_ScRank::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    (**thisthisthis).locTestMode = isTestMode();
    SVM_Scalar::dgTrainingVector(res,resn,i);
    (**thisthisthis).locTestMode = 1;

    return;
}

void SVM_ScRank::dgTrainingVector(SparseVector<gentype> &resx, int i) const
{
    (**thisthisthis).locTestMode = isTestMode();
    SVM_Scalar::dgTrainingVector(resx,i);
    (**thisthisthis).locTestMode = 1;

    return;
}

void SVM_ScRank::drTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    (**thisthisthis).locTestMode = isTestMode();
    SVM_Scalar::drTrainingVector(res,resn,i);
    (**thisthisthis).locTestMode = 1;

    return;
}


















int SVM_ScRank::qaddTrainingVector(int i, double z, SparseVector<gentype> &x, double Cweigh, double epsweigh, int d, double Cweighfuzz)
{
    inGpn.addRow(i);
    locd.add(i);
    inGpn("[]",i,zeroint()) = calcGpn(d,x);
    locd("[]",i) = d;

    return SVM_Scalar::qaddTrainingVector(i,z,x,Cweigh,epsweigh,d,Cweighfuzz);
}

int SVM_ScRank::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    int res = SVM_Scalar::removeTrainingVector(i,y,x);

    inGpn.removeRow(i);
    locd.remove(i);

    return res;
}

int SVM_ScRank::setd(int i, int nd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    // This is slightly complicated.  We need to update
    //
    // - Gpn(i,0)
    // - Gp(i,:) and Gp(:,i)

    if ( d()(i) && nd && ( d()(i) != nd ) )
    {
        res |= setd(i,0);
        res |= setd(i,nd);
    }

    else if ( d()(i) && !nd )
    {
        // Make d zero first so that refactGpnElm works properly
        res |= SVM_Scalar::setd(i,nd);
        locd("[]",i) = nd;

        // Update Gpn
        double newGpnVal = calcGpn(nd,x(i));
        SVM_Scalar::refactGpnElm(i,0,newGpnVal);
        inGpn("[]",i,zeroint()) = newGpnVal;

        // Update Gp
        resetKernel(1,i);
    }

    else if ( !d()(i) && nd )
    {
        // Update Gpn
        double newGpnVal = calcGpn(nd,x(i));
        SVM_Scalar::refactGpnElm(i,0,newGpnVal);
        inGpn("[]",i,zeroint()) = newGpnVal;

        // Update d
        res |= SVM_Scalar::setd(i,nd);
        locd("[]",i) = nd;

        // Update Gp
        resetKernel(1,i);
    }

    return res;
}


















int SVM_ScRank::addTrainingVector (int i, double z, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int d, double Cweighfuzz)
{
    SparseVector<gentype> xxx(x);

    return SVM_ScRank::qaddTrainingVector(i,z,xxx,Cweigh,epsweigh,d,Cweighfuzz);
}

int SVM_ScRank::addTrainingVector (int i, const Vector<double> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &xxd, const Vector<double> &Cweighfuzz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );
    NiceAssert( z.size() == xxd.size() );
    NiceAssert( z.size() == Cweighfuzz.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_ScRank::addTrainingVector(i+j,z(j),x(j),Cweigh(j),epsweigh(j),xxd(j),Cweighfuzz(j));
        }
    }

    return res;
}

int SVM_ScRank::qaddTrainingVector(int i, const Vector<double> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &xxd, const Vector<double> &Cweighfuzz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );
    NiceAssert( z.size() == xxd.size() );
    NiceAssert( z.size() == Cweighfuzz.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= SVM_ScRank::qaddTrainingVector(i+j,z(j),x("[]",j),Cweigh(j),epsweigh(j),xxd(j),Cweighfuzz(j));
        }
    }

    return res;
}

int SVM_ScRank::addTrainingVector(int i, const gentype &zi, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_ScRank::addTrainingVector(i,(double) zi,x,Cweigh,epsweigh,2,Cweigh);
}

int SVM_ScRank::qaddTrainingVector(int i, const gentype &zi, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_ScRank::qaddTrainingVector(i,(double) zi,x,Cweigh,epsweigh,2,Cweigh);
}

int SVM_ScRank::addTrainingVector(int i, const Vector<gentype> &zi, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<double> zzi(zi.size());
    Vector<int> ddd(zi.size());

    ddd = 2;

    if ( zi.size() )
    {
        int j;

        for ( j = 0 ; j < zi.size() ; j++ )
        {
            zzi("[]",j) = (double) zi(j);
        }
    }

    return SVM_ScRank::addTrainingVector(i,zzi,x,Cweigh,epsweigh,ddd,Cweigh);
}

int SVM_ScRank::qaddTrainingVector(int i, const Vector<gentype> &zi, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<double> zzi(zi.size());
    Vector<int> ddd(zi.size());

    ddd = 2;

    if ( zi.size() )
    {
        int j;

        for ( j = 0 ; j < zi.size() ; j++ )
        {
            zzi("[]",j) = (double) zi(j);
        }
    }

    return SVM_ScRank::qaddTrainingVector(i,zzi,x,Cweigh,epsweigh,ddd,Cweigh);
}

int SVM_ScRank::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i.size() == d.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= SVM_ScRank::setd(i(j),d(j));
        }
    }

    return res;
}

int SVM_ScRank::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == N() );

    int res = 0;

    if ( N() )
    {
        int j;

        for ( j = 0 ; j < N() ; j++ )
        {
            res |= SVM_ScRank::setd(j,d(j));
        }
    }

    return res;
}




