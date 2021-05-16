
//
// Pareto-frontier style 1-class Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_pfront.h"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

SVM_PFront::SVM_PFront() : SVM_Binary()
{
    SVM_Binary::getKernel_unsafe().setType(402);
    SVM_Binary::resetKernel();
    set1NormCost();
    setFixedBias(0.0);
    seteps(0.5);
    setC(1e20);

    return;
}

SVM_PFront::SVM_PFront(const SVM_PFront &src) : SVM_Binary(src)
{
    setaltx(NULL);

    return;
}

SVM_PFront::SVM_PFront(const SVM_PFront &src, const ML_Base *xsrc) : SVM_Binary(src,xsrc)
{
    setaltx(xsrc);

    return;
}

SVM_PFront::~SVM_PFront()
{
    return;
}

std::ostream &SVM_PFront::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "PFront boundary SVM\n\n";

    repPrint(output,'>',dep) << "Base SVM: ";
    SVM_Binary::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";

    return output;
}

std::istream &SVM_PFront::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy;
    SVM_Binary::inputstream(input);

    return input;
}

int SVM_PFront::train(int &res, svmvolatile int &killSwitch)
{
    return SVM_Binary::train(res,killSwitch);
}

int SVM_PFront::addTrainingVector(int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    (void) z;

    return SVM_PFront::addTrainingVector(i,x,Cweigh,epsweigh);
}

int SVM_PFront::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    (void) z;

    return SVM_PFront::qaddTrainingVector(i,x,Cweigh,epsweigh);
}

int SVM_PFront::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    (void) z;

    return SVM_PFront::addTrainingVector(i,x,Cweigh,epsweigh);
}

int SVM_PFront::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    (void) z;

    return SVM_PFront::qaddTrainingVector(i,x,Cweigh,epsweigh);
}

int SVM_PFront::addTrainingVector(int i, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_Binary::addTrainingVector(i,1,x,Cweigh,epsweigh);
}

int SVM_PFront::qaddTrainingVector(int i, SparseVector<gentype> &x, double Cweigh , double epsweigh)
{
    return SVM_Binary::qaddTrainingVector(i,1,x,Cweigh,epsweigh);
}

int SVM_PFront::addTrainingVector(int i, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<gentype> d(x.size());
    gentype tempz(1);

    d = tempz;

    return SVM_Binary::addTrainingVector(i,d,x,Cweigh,epsweigh);
}

int SVM_PFront::qaddTrainingVector(int i, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<gentype> d(x.size());
    gentype tempz(1);

    d = tempz;

    return SVM_Binary::qaddTrainingVector(i,d,x,Cweigh,epsweigh);
}

double SVM_PFront::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    return db ? ( ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0 ) : 0;
}

int SVM_PFront::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    //int unusedvar = 0;
    double tempresg = 0;
    int tempresh = 1;
    gentype tempsomeh,tempsomeg;

    //gTrainingVector(tempresg,unusedvar,i,retaltg,pxyprodi);
    ghTrainingVector(tempsomeh,tempsomeg,i,retaltg,pxyprodi);

    tempresg = (double) tempsomeg;

    //tempresg = NS()-tempresg;
    tempresg = 1-(2*tempresg);
    tempresh = ( tempresg >= 0.0 ) ? 1 : -1;

    resg = tempresg;
    resh = tempresh;

    return tempresh;
}

