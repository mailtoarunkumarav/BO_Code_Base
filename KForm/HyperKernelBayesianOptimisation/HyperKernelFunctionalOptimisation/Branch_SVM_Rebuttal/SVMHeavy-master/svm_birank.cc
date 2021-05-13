
//
// Ranking SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_birank.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


SVM_BiRank::SVM_BiRank() : SVM_Binary()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    locTestMode = 1;

    SVM_Binary::setFixedBias();

    return;
}

SVM_BiRank::SVM_BiRank(const SVM_BiRank &src) : SVM_Binary()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    locTestMode = 1;

    SVM_Binary::setFixedBias();

    assign(src,0);

    return;
}

SVM_BiRank::SVM_BiRank(const SVM_BiRank &src, const ML_Base *xsrc) : SVM_Binary()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(xsrc);

    locTestMode = 1;

    SVM_Binary::setFixedBias();

    assign(src,1);

    return;
}

SVM_BiRank::~SVM_BiRank()
{
    return;
}

std::ostream &operator<<(std::ostream &output, const SVM_BiRank &src)
{
    return src.printstream(output);
}

std::istream &operator>>(std::istream &input, SVM_BiRank &dest)
{
    return dest.inputstream(input);
}

std::ostream &SVM_BiRank::printstream(std::ostream &output) const
{
    output << "Ranking SVM\n\n";

    output << "locTestMode: " << locTestMode << "\n";
    output << "=====================================================================\n";
    output << "Base SVC: ";
    SVM_Binary::printstream(output);
    output << "\n";
    output << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_BiRank::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> locTestMode;
    input >> dummy;
    SVM_Binary::inputstream(input);

    return input;
}

double &SVM_BiRank::KReal(double &res, int i, int j) const
{
    // locTestMode = 1:
    //
    // i   | j   | di  | dj  |  modi  |  modj
    // ----+-----+-----+-----+--------+--------
    // < 0 | < 0 | ... | ... |   0    |   0
    // >=0 | < 0 |  0  | ... |   0    |   0
    // >=0 | < 0 |  1  | ... |   1    |   0
    // < 0 | >=0 | ... |  0  |   0    |   0
    // < 0 | >=0 | ... |  1  |   0    |   1
    // >=0 | >=0 |  0  |  0  |   0    |   0
    // >=0 | >=0 |  0  |  1  |   0    |   1
    // >=0 | >=0 |  1  |  0  |   1    |   0
    // >=0 | >=0 |  1  |  1  |   1    |   1
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
    // >=0 | < 0 |  1  | ... |   0    |   0
    // < 0 | >=0 | ... |  0  |   0    |   0
    // < 0 | >=0 | ... |  1  |   0    |   1
    // >=0 | >=0 |  0  |  0  |   0    |   0
    // >=0 | >=0 |  0  |  1  |   0    |   1
    // >=0 | >=0 |  1  |  0  |   0    |   0
    // >=0 | >=0 |  1  |  1  |   0    |   1
    //
    // modi = 0
    // modj = ( j >= 0 ) && d()(j)

    int modi = ( ( i >= 0 ) && d()(i) ) ? locTestMode : 0;
    int modj = ( ( j >= 0 ) && d()(j) ) ? 1           : 0;

    double temp;

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

    return res;
}

// Evaluation:

int SVM_BiRank::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg) const
{
    int res;

    (**thisthisthis).locTestMode = isTestMode();
    res = SVM_Binary::ghTrainingVector(resh,resg,i,retaltg);
    (**thisthisthis).locTestMode = 1;

    return res;
}

void SVM_BiRank::eTrainingVector(gentype &res, int i) const
{
    (**thisthisthis).locTestMode = isTestMode();
    SVM_Binary::eTrainingVector(res,i);
    (**thisthisthis).locTestMode = 1;

    return;
}

void SVM_BiRank::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    (**thisthisthis).locTestMode = isTestMode();
    SVM_Binary::dgTrainingVector(res,resn,i);
    (**thisthisthis).locTestMode = 1;

    return;
}

void SVM_BiRank::dgTrainingVector(SparseVector<gentype> &resx, int i) const
{
    (**thisthisthis).locTestMode = isTestMode();
    SVM_Binary::dgTrainingVector(resx,i);
    (**thisthisthis).locTestMode = 1;

    return;
}

void SVM_BiRank::drTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    (**thisthisthis).locTestMode = isTestMode();
    SVM_Binary::drTrainingVector(res,resn,i);
    (**thisthisthis).locTestMode = 1;

    return;
}










