
//
// Vector+Ranking SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_verank.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


SVM_VeRank::SVM_VeRank() : SVM_Vector_redbin<SVM_ScRank>()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    return;
}

SVM_VeRank::SVM_VeRank(const SVM_VeRank &src) : SVM_Vector_redbin<SVM_ScRank>()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    assign(src,0);

    return;
}

SVM_VeRank::SVM_VeRank(const SVM_VeRank &src, const ML_Base *xsrc) : SVM_Vector_redbin<SVM_ScRank>()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(xsrc);

    assign(src,1);

    return;
}

SVM_VeRank::~SVM_VeRank()
{
    return;
}

std::ostream &operator<<(std::ostream &output, const SVM_VeRank &src)
{
    return src.printstream(output);
}

std::istream &operator>>(std::istream &input, SVM_VeRank &dest)
{
    return dest.inputstream(input);
}

std::ostream &SVM_VeRank::printstream(std::ostream &output) const
{
    output << "Vector Ranking SVM\n\n";

    output << "=====================================================================\n";
    output << "Base Vector Regressor:\n";
    SVM_Vector_redbin<SVM_ScRank>::printstream(output);
    output << "\n";
    output << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_VeRank::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy;
    SVM_Vector_redbin<SVM_ScRank>::inputstream(input);

    return input;
}

