
//
// Hill-climbing feature selector for SVMs
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


// Given an SVM will use hill-climbing to select the optimal features
// for the training set.  Performance is measured either using n-fold
// cross validation with m randomised repetitions or using leave-one-out
// (no repetitions) if n == -1 (or recall if n == -2, though this will
// lead to overfitting).  The features selected are stored in feats, and
// the optimal error returned.
//
// To test with an external training set use testSVM
//
// If traverse > 1 then this will first do hill climb (descent), then reverse
// direction and do hill descent (climb) and so on traverse times.
//
// If startdirst then will start with whatever indices are already selected
// and build on that rather than start from scratch.

#ifndef _hillclimb_h
#define _hillclimb_h

#include "ml_base.h"

double optFeatHillClimb(ML_Base &svm,
                        int n,
                        int m,
                        int rndit,
                        Vector<int> &usedfeats,
                        std::ostream &logdest,
                        int useDescent,
                        const Vector<SparseVector<gentype> > &xtest,
                        const Vector<gentype> &ytest,
                        int startpoint,
                        int traverse = 0,
                        int startdirty = 0);



#endif

