
//
// Score SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_biscor.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


SVM_BiScor::SVM_BiScor() : SVM_Binary()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    locN = 0;

    return;
}

SVM_BiScor::SVM_BiScor(const SVM_BiScor &src) : SVM_Binary()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    locN = 0;

    assign(src,0);

    return;
}

SVM_BiScor::SVM_BiScor(const SVM_BiScor &src, const ML_Base *xsrc) : SVM_Binary()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(xsrc);

    locN = 0;

    assign(src,1);

    return;
}

SVM_BiScor::~SVM_BiScor()
{
    return;
}

std::ostream &SVM_BiScor::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Ranking SVM\n\n";

    repPrint(output,'>',dep) << "locN: " << locN << "\n";
    repPrint(output,'>',dep) << "locz: " << locz << "\n";
    repPrint(output,'>',dep) << "locd: " << locd << "\n";
    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base SVC: ";
    SVM_Binary::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_BiScor::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> locN;
    input >> dummy; input >> locz;
    input >> dummy; input >> locd;
    input >> dummy;
    SVM_Binary::inputstream(input);

    return input;
}

int SVM_BiScor::qaddTrainingVector(int i, const gentype &zzz, SparseVector<gentype> &xxxx, double Cweigh, double epsweigh)
{
    int dd = 1;

    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );
    NiceAssert( ( dd == 0 ) || ( dd = 1 ) );
    NiceAssert( zzz.isValVector() );

    locN++;

    int res = 0;

    locz.add(i); locz("&",i) = zzz;
    locd.add(i); locd("&",i) = dd;

    res |= SVM_Binary::qaddTrainingVector(i,zerointgentype(),xxxx,Cweigh,epsweigh);

    // Update current inequalities

    if ( N() < SVM_Binary::N() )
    {
        int j;

        for ( j = SVM_Binary::N()-1 ; j >= N() ; j-- )
        {
            if ( ( (int) x()(j).fff(0) >= i ) || ( (int) x()(j).fff(1) >= i ) )
            {
                SparseVector<gentype> xx(x()(j));

                if ( (int) xx.fff(0) >= i )
                {
                    xx.fff("&",zeroint()) = ((int) xx.fff(0))+1;
                }

                if ( (int) xx.fff(1) >= i )
                {
                    xx.fff("&",1) = ((int) xx.fff(1))+1;
                }

                res |= SVM_Binary::setx(j,xx);
            }
        }
    }

    // Add new inequalities

    res |= processz(i);

    return res;
}

int SVM_BiScor::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &xxxx)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    // Pre-emptively remove related inequalities

    if ( N() < SVM_Binary::N() )
    {
        int j;

        for ( j = SVM_Binary::N()-1 ; j >= N() ; j-- )
        {
            if ( ( (int) x()(j).fff(0) == i ) || ( (int) x()(j).fff(1) == i ) )
            {
                res |= SVM_Binary::removeTrainingVector(j);

                locz.remove(j);
                locd.remove(j);
            }
        }
    }

    // swap relevant vector to end
    // (can't just remove yet, still indexed)

    locz.add(N()); locz("&",N()) = y()(i);
    locd.add(N()); locd("&",N()) = locd(i);

    res |= SVM_Binary::addTrainingVector(N(),zerointgentype(),x()(i));
    res |= SVM_Binary::removeTrainingVector(i,y,xxxx);

    locz.remove(i);
    locd.remove(i);

    // Fix relevant indices

    if ( N() < SVM_Binary::N() )
    {
        int j;

        for ( j = SVM_Binary::N()-1 ; j >= N() ; j-- )
        {
            if ( ( (int) x()(j).fff(0) >= i ) || ( (int) x()(j).fff(1) >= i ) )
            {
                SparseVector<gentype> xx(x()(j));

                if ( (int) xx.fff(0) >= i )
                {
                    xx.fff("&",zeroint()) = ((int) xx.fff(0))-1;
                }

                if ( (int) xx.fff(1) >= i )
                {
                    xx.fff("&",1) = ((int) xx.fff(1))-1;
                }

                res |= SVM_Binary::setx(j,xx);
            }
        }
    }

    // Remove vector

    res |= SVM_Binary::removeTrainingVector(N());

    locz.remove(N());
    locd.remove(N());

    return res;
}

int SVM_BiScor::setx(int i, const SparseVector<gentype> &xxxx)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    // Update Vector

    res |= SVM_Binary::setx(i,xxxx);

    // Update Inequalities

    if ( N() < SVM_Binary::N() )
    {
        int j;

        for ( j = SVM_Binary::N()-1 ; j >= N() ; j-- )
        {
            if ( ( (int) x()(j).fff(0) == i ) || ( (int) x()(j).fff(1) == i ) )
            {
                res |= SVM_Binary::setx(j,x()(j));
            }
        }
    }

    return res;
}

int SVM_BiScor::qswapx(int i, SparseVector<gentype> &xxxx, int dontupdate)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    // Update Vector

    res |= SVM_Binary::qswapx(i,xxxx,dontupdate);

    // Update Inequalities

    if ( N() < SVM_Binary::N() )
    {
        int j;

        for ( j = SVM_Binary::N()-1 ; j >= N() ; j-- )
        {
            if ( ( (int) x()(j).fff(0) == i ) || ( (int) x()(j).fff(1) == i ) )
            {
                res |= SVM_Binary::setx(j,x()(j));
            }
        }
    }

    return res;
}

int SVM_BiScor::sety(int i, const gentype &zz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( zz.isValVector() );

    int res = 1;

    // Update Vector

    locz("&",i) = zz;

    // Remove i-related inequalities

    if ( N() < SVM_Binary::N() )
    {
        int j;

        for ( j = SVM_Binary::N()-1 ; j >= N() ; j-- )
        {
            if ( ( (int) x()(j).fff(0) == i ) || ( (int) x()(j).fff(1) == i ) )
            {
                res |= SVM_Binary::removeTrainingVector(j);

                locz.remove(j);
                locd.remove(j);
            }
        }
    }

    // Add new i-related inequalities

    res |= processz(i);

    return res;
}

int SVM_BiScor::setd(int i, int dd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );
    NiceAssert( ( dd == 0 ) || ( dd = 1 ) );

    int res = 0;

    locd("&",i) = dd;

    if ( N() < SVM_Binary::N() )
    {
        int j;

        for ( j = SVM_Binary::N()-1 ; j >= N() ; j-- )
        {
            if ( ( (int) x()(j).fff(0) == i ) || ( (int) x()(j).fff(1) == i ) )
            {
                int nd = ( locd(i) && locd(j) ) ? 1 : 0;

                res |= SVM_Binary::setd(j,nd);

                locd("&",j) = nd;
            }
        }
    }

    return res;
}

int SVM_BiScor::setCweight(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    res = SVM_Binary::setCweight(i,xCweight);

    if ( N() < SVM_Binary::N() )
    {
        int j;

        for ( j = SVM_Binary::N()-1 ; j >= N() ; j-- )
        {
            if ( ( (int) x()(j).fff(0) == i ) || ( (int) x()(j).fff(1) == i ) )
            {
                res |= SVM_Binary::setCweight(j,Cweight()((int) x()(j).fff(0))*Cweight()((int) x()(j).fff(1)));
            }
        }
    }

    return res;
}

int SVM_BiScor::setCweightfuzz(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    res = SVM_Binary::setCweightfuzz(i,xCweight);

    if ( N() < SVM_Binary::N() )
    {
        int j;

        for ( j = SVM_Binary::N()-1 ; j >= N() ; j-- )
        {
            if ( ( (int) x()(j).fff(0) == i ) || ( (int) x()(j).fff(1) == i ) )
            {
                res |= SVM_Binary::setCweightfuzz(j,Cweightfuzz()((int) x()(j).fff(0))*Cweightfuzz()((int) x()(j).fff(1)));
            }
        }
    }

    return res;
}

int SVM_BiScor::setepsweight(int i, double xepsweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = 0;

    res = SVM_Binary::setepsweight(i,xepsweight);

    if ( N() < SVM_Binary::N() )
    {
        int j;

        for ( j = SVM_Binary::N()-1 ; j >= N() ; j-- )
        {
            if ( ( (int) x()(j).fff(0) == i ) || ( (int) x()(j).fff(1) == i ) )
            {
                res |= SVM_Binary::setepsweight(j,epsweight()((int) x()(j).fff(0))*epsweight()((int) x()(j).fff(1)));
            }
        }
    }

    return res;
}

int SVM_BiScor::processz(int i)
{
    int res = 0;

    if ( N() > 1 )
    {
        int j;
        int isize = locz(i).size();

        for ( j = 0 ; j < N() ; j++ )
        {
            if ( j != i )
            {
                int nd = ( locd(i) && locd(j) ) ? -1 : 0;
                int jsize = locz(j).size();
                int ijsize = ( isize < jsize ) ? isize : jsize;

                if ( ijsize )
                {
                    int m;

                    for ( m = 0 ; m < ijsize ; m++ )
                    {
                        const gentype &ielm = (locz(i))(m);
                        const gentype &jelm = (locz(j))(m);

                        if ( !(ielm.isValNull()) && !(jelm.isValNull()) )
                        {
                            if ( ielm > jelm )
                            {
                                int k = SVM_Binary::N();

                                SparseVector<gentype> xx;

                                xx.fff("&",zeroint()) = j;
                                xx.fff("&",1) = i;

                                locz.add(k); 
                                locd.add(k); locd("&",k) = nd;

                                res |= SVM_Binary::qaddTrainingVector(k,negoneintgentype(),xx);
                                res |= SVM_Binary::setd(k,nd);
                            }

                            else if ( ielm < jelm )
                            {
                                int k = SVM_Binary::N();

                                SparseVector<gentype> xx;

                                xx.fff("&",zeroint()) = i;
                                xx.fff("&",1) = j;

                                locz.add(k);
                                locd.add(k); locd("&",k) = nd;

                                res |= SVM_Binary::qaddTrainingVector(k,negoneintgentype(),xx);
                                res |= SVM_Binary::setd(k,nd);
                            }
                        }
                    }
                }
            }
        }
    }

    return res;
}























































































int SVM_BiScor::addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    SparseVector<gentype> xxx(x);

    return qaddTrainingVector(i,z,xxx,Cweigh,epsweigh);
}

int SVM_BiScor::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= addTrainingVector(i+j,z(j),x(j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_BiScor::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; j++ )
        {
            res |= qaddTrainingVector(i+j,z(j),x("&",j),Cweigh(j),epsweigh(j));
        }
    }

    return res;
}

int SVM_BiScor::removeTrainingVector(int i)
{
    gentype y;
    SparseVector<gentype> x;

    return removeTrainingVector(i,y,x);
}

int SVM_BiScor::removeTrainingVector(int i, int num)
{
    int res = 0;

    if ( num > 0 )
    {
        int j;

        for ( j = num-1 ; j >= 0 ; j-- )
        {
            res |= removeTrainingVector(i+j);
        }
    }

    return res;
}

int SVM_BiScor::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x)
{
    NiceAssert( i.size() == x.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= setx(i(j),x(j));
        }
    }

    return res;
}

int SVM_BiScor::setx(const Vector<SparseVector<gentype> > &x)
{
    retVector<int> tmpva;

    return setx(cntintvec(N(),tmpva),x);
}

int SVM_BiScor::qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate)
{
    NiceAssert( i.size() == x.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= qswapx(i(j),x("&",j),dontupdate);
        }
    }

    return res;
}

int SVM_BiScor::qswapx(Vector<SparseVector<gentype> > &x, int dontupdate)
{
    retVector<int> tmpva;

    return qswapx(cntintvec(N(),tmpva),x,dontupdate);
}

int SVM_BiScor::sety(const Vector<int> &i, const Vector<gentype> &z)
{
    NiceAssert( i.size() == z.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= sety(i(j),z(j));
        }
    }

    return res;
}

int SVM_BiScor::sety(const Vector<gentype> &z)
{
    retVector<int> tmpva;

    return sety(cntintvec(N(),tmpva),z);
}

int SVM_BiScor::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i.size() == d.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= setd(i(j),d(j));
        }
    }

    return res;
}

int SVM_BiScor::setd(const Vector<int> &d)
{
    retVector<int> tmpva;

    return setd(cntintvec(N(),tmpva),d);
}

int SVM_BiScor::setCweight(const Vector<int> &i, const Vector<double> &xCweight)
{
    NiceAssert( i.size() == xCweight.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= setCweight(i(j),xCweight(j));
        }
    }

    return res;
}

int SVM_BiScor::setCweight(const Vector<double> &xCweight)
{
    retVector<int> tmpva;

    return setCweight(cntintvec(N(),tmpva),xCweight);
}

int SVM_BiScor::setCweightfuzz(const Vector<int> &i, const Vector<double> &xCweight)
{
    NiceAssert( i.size() == xCweight.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= setCweightfuzz(i(j),xCweight(j));
        }
    }

    return res;
}

int SVM_BiScor::setCweightfuzz(const Vector<double> &xCweight)
{
    retVector<int> tmpva;

    return setCweightfuzz(cntintvec(N(),tmpva),xCweight);
}

int SVM_BiScor::setepsweight(const Vector<int> &i, const Vector<double> &xepsweight)
{
    NiceAssert( i.size() == xepsweight.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; j++ )
        {
            res |= setepsweight(i(j),xepsweight(j));
        }
    }

    return res;
}

int SVM_BiScor::setepsweight(const Vector<double> &xepsweight)
{
    retVector<int> tmpva;

    return setepsweight(cntintvec(N(),tmpva),xepsweight);
}


