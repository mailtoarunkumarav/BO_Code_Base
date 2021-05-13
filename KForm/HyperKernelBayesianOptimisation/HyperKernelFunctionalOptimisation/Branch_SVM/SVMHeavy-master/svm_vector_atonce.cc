
//
// Vector (at once) regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_vector_atonce.h"


SVM_Vector_atonce::SVM_Vector_atonce() : SVM_Vector_atonce_temp<double>()
{
    setaltx(NULL);

    return;
}

SVM_Vector_atonce::SVM_Vector_atonce(const SVM_Vector_atonce &src) : SVM_Vector_atonce_temp<double>(static_cast<const SVM_Vector_atonce_temp<double> &>(src))
{
    setaltx(NULL);

    return;
}

SVM_Vector_atonce::SVM_Vector_atonce(const SVM_Vector_atonce &src, const ML_Base *xsrc) : SVM_Vector_atonce_temp<double>(static_cast<const SVM_Vector_atonce_temp<double> &>(src),static_cast<const SVM_Vector_atonce_temp<double> *>(xsrc))
{
    setaltx(xsrc);

    return;
}

SVM_Vector_atonce::~SVM_Vector_atonce()
{
    return;
}

std::ostream &SVM_Vector_atonce::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector atonce SVM (scalar kernel)\n\n";

    (static_cast<const SVM_Vector_atonce_temp<double> &>(*this)).SVM_Vector_atonce_temp<double>::printstream(output,dep+1);

    return output;
}

std::istream &SVM_Vector_atonce::inputstream(std::istream &input)
{
    (static_cast<SVM_Vector_atonce_temp<double> &>(*this)).SVM_Vector_atonce_temp<double>::inputstream(input);

    return input;
}

