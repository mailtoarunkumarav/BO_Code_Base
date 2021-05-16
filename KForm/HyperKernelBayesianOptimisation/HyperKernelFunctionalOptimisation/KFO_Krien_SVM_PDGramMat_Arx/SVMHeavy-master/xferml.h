
//
// Transfer learning setup
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

/*

Assume that we have a bunch of sensible SVMs all inheritting a kernel from
a common core SVM.  Assume:

- all SVMs are sensible: m=2, outers make sense, inner is binary classifier (nominal)
- outer SVMs are scalar

We can train this as a multi-quadratic optimisation as per paper FIXME

n is the number of vectors in the core

Method outer: all outer mathines inherit kernel from inner machine using kernel 801
Inner method: training Gp is sum of kernels evaluated 801 style from *outer*, so need to calculate and fix these, then train.  This is just svm_scalar where the kernel is fixed before training.



Method:

until finished:
- train outers
- set extGp for inner (calculated as sum of (m+2)-kernels)
- train inner
- unset extGp for inner
- reset outer kernels

*/

#ifndef _xferml_h
#define _xferml_h

#include "svm_generic.h"
#include "svm_binary.h"



int xferMLtrain(svmvolatile int &killSwitch, SVM_Scalar &core, Vector<SVM_Generic *> &cases, int n, int maxiter, double maxtime, double soltol);

#endif

