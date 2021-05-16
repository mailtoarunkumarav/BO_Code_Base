
//
// Vector+Scoring SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_vescor.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


SVM_VeScor::SVM_VeScor() : SVM_Vector_redbin<SVM_ScScor>()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    return;
}

SVM_VeScor::SVM_VeScor(const SVM_VeScor &src) : SVM_Vector_redbin<SVM_ScScor>()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    assign(src,0);

    return;
}

SVM_VeScor::SVM_VeScor(const SVM_VeScor &src, const ML_Base *xsrc) : SVM_Vector_redbin<SVM_ScScor>()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(xsrc);

    assign(src,1);

    return;
}

SVM_VeScor::~SVM_VeScor()
{
    return;
}

std::ostream &operator<<(std::ostream &output, const SVM_VeScor &src)
{
    return src.printstream(output);
}

std::istream &operator>>(std::istream &input, SVM_VeScor &dest)
{
    return dest.inputstream(input);
}

std::ostream &SVM_VeScor::printstream(std::ostream &output) const
{
    output << "Vector Ranking SVM\n\n";

    output << "=====================================================================\n";
    output << "Base Vector Regressor:\n";
    SVM_Vector_redbin<SVM_ScScor>::printstream(output);
    output << "\n";
    output << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_VeScor::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy;
    SVM_Vector_redbin<SVM_ScScor>::inputstream(input);

    return input;
}

