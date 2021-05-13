
//
// Vector (at once) regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_vector_matonce.h"


SVM_Vector_Matonce::SVM_Vector_Matonce() : SVM_Vector_atonce_temp<Matrix<double> >()
{
    setaltx(NULL);

    return;
}

SVM_Vector_Matonce::SVM_Vector_Matonce(const SVM_Vector_Matonce &src) : SVM_Vector_atonce_temp<Matrix<double> >(static_cast<const SVM_Vector_atonce_temp<Matrix<double> > &>(src))
{
    setaltx(NULL);

    return;
}

SVM_Vector_Matonce::SVM_Vector_Matonce(const SVM_Vector_Matonce &src, const ML_Base *xsrc) : SVM_Vector_atonce_temp<Matrix<double> >(static_cast<const SVM_Vector_atonce_temp<Matrix<double> > &>(src),static_cast<const SVM_Vector_atonce_temp<Matrix<double> > *>(xsrc))
{
    setaltx(xsrc);

    return;
}

SVM_Vector_Matonce::~SVM_Vector_Matonce()
{
    return;
}

std::ostream &SVM_Vector_Matonce::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector atonce SVM (matrix kernel)\n\n";

    (static_cast<const SVM_Vector_atonce_temp<Matrix<double> > &>(*this)).SVM_Vector_atonce_temp<Matrix<double> >::printstream(output,dep+1);

    return output;
}

std::istream &SVM_Vector_Matonce::inputstream(std::istream &input)
{
    (static_cast<SVM_Vector_atonce_temp<Matrix<double> > &>(*this)).SVM_Vector_atonce_temp<Matrix<double> >::inputstream(input);

    return input;
}

