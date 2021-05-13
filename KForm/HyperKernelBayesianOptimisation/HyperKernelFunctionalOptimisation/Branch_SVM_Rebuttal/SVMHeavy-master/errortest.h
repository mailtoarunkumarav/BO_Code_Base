
//
// Performance/error testing routines
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _errortest_h
#define _errortest_h

#include "ml_base.h"

// Performance
//
// calcLOO(cnt,cfm): calculates leave-one-out RMSE of trained ML.  The
//     total number of lower and upper bound constraints is put in cnt
//     and the confusion matrix for such vectors in cfm.  Alternatively,
//     if there are no such constraints in the training set, cnt and cfm
//     are totals based on the sign of the target of the training vector.
//     Note that training vectors constrained to zero are not included
//     in any of these calculations.
// calcRecall(cnt,cfm): do calc_loo, but doesn't actually leave out.
// calcCross(m,rndit,cnt,cfm): calculate m-fold cross-validation RMSE of
//     trained ML.  The cnt and cfm arguments are averaged, as is the
//     return value.  If rndit then the order of the vectors is
//     randomised, otherwise it is sequential.  If numreps is set to > 1
//     then the process is repeated this number of times and the result is
//     averaged.
// calcTest(xtest,ytest,cnt,cfm): use test dataset provided.  cnt/cfm
//     contain sign-based counts, not error-based counts.
// calcSparTest: calculate performance under varying numbers of "noise"
//     features.
//
// assessResult: given input/output pairs for some testing set, calculate
//     cnt/cfm etc and return classification error.
// calcAUC: calculate the area under the ROC curve.  Only works for binary
//     training sets with classification +1/-1.
//
// NOTE: - LOO and cross of these will train the ML before testing.
//       - startpoint = 0: start training from current state
//                      1: start training from reset state
//                      3: start training from current state, only do one "cross" (effectively use first batch as a single validation set)
//                      4: start training from reset state, only do one "cross" (effectively use first batch as a single validation set)
//       - cla,res: stores the actual results of tests (class and output)
//                  most recent result for multiple reps.  In the case of
//                  cross-fold validation, these are vectors of m results
//                  vectors
//       - suppressfn: nz means give no feedback


double calcLOO     (const ML_Base &baseML,                                                                    int startpoint = 0, int suppressfb = 0);
double calcRecall  (const ML_Base &baseML,                                                                    int startpoint = 0, int suppressfb = 0);
double calcCross   (const ML_Base &baseML, int m, int rndit = 0, int numreps = 1,                             int startpoint = 0, int suppressfb = 0);
double CalcSparSens(const ML_Base &baseML, int minbad, int maxbad, double noisemean = 0, double noisevar = 0, int startpoint = 0, int suppressfb = 0);

double calcLOO     (const ML_Base &baseML,                                           Vector<int>    &cnt, Matrix<int>    &cfm,                                                                    int startpoint = 0, int suppressfb = 0);
double calcRecall  (const ML_Base &baseML,                                           Vector<int>    &cnt, Matrix<int>    &cfm,                                                                    int startpoint = 0, int suppressfb = 0);
double calcCross   (const ML_Base &baseML, int m, int rndit, Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, int numreps = 1,                                                   int startpoint = 0, int suppressfb = 0);
double calcSparSens(const ML_Base &baseML,                   Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, int minbad, int maxbad, double noisemean = 0, double noisevar = 0, int startpoint = 0, int suppressfb = 0);

double calcLOO     (const ML_Base &baseML,                                           Vector<int>    &cnt, Matrix<int>    &cfm,        Vector<gentype>   &resh,        Vector<gentype>   &resg,        Vector<gentype>   &gvarres,                                                                    int startpoint = 0, int calcvarres = 0, int suppressfb = 0);
double calcRecall  (const ML_Base &baseML,                                           Vector<int>    &cnt, Matrix<int>    &cfm,        Vector<gentype>   &resh,        Vector<gentype>   &resg,        Vector<gentype>   &gvarres,                                                                    int startpoint = 0, int calcvarres = 0, int suppressfb = 0);
double calcCross   (const ML_Base &baseML, int m, int rndit, Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, Vector<Vector<gentype> > &resh, Vector<Vector<gentype> > &resg, Vector<Vector<gentype> > &gvarres, int numreps = 1,                                                   int startpoint = 0, int calcvarres = 0, int suppressfb = 0);
double calcSparSens(const ML_Base &baseML,                   Vector<double> &repres, Vector<double> &cnt, Matrix<double> &cfm, Vector<Vector<gentype> > &resh, Vector<Vector<gentype> > &resg, Vector<Vector<gentype> > &gvarres, int minbad, int maxbad, double noisemean = 0, double noisevar = 0, int startpoint = 0, int calcvarres = 0, int suppressfb = 0);

double calcTest(const ML_Base &baseML, const Vector<SparseVector<gentype> > &xtest, const Vector<gentype> &ytest, int suppressfb = 0);
double calcTest(const ML_Base &baseML, const Vector<SparseVector<gentype> > &xtest, const Vector<gentype> &ytest, Vector<int> &cnt, Matrix<int> &cfm, int suppressfb = 0);
double calcTest(const ML_Base &baseML, const Vector<SparseVector<gentype> > &xtest, const Vector<gentype> &ytest, Vector<int> &cnt, Matrix<int> &cfm, Vector<gentype> &resh, Vector<gentype> &resg, Vector<gentype> &gvarres, int calcvarres = 0, int suppressfb = 0);

double calcnegloglikelihood(const ML_Base &baseML, int suppressfb = 0);

double assessResult(const ML_Base &baseML, Vector<int> &cnt, Matrix<int> &cfm, const Vector<gentype> &ytestresh, const Vector<gentype> &ytest, const Vector<int> &outkernind);

// Measure "goodness" for binary result:
//
// res(0) = accuracy
// res(1) = precision
// res(2) = recall
// res(3) = f1 score
// res(4) = AUC
// res(5) = reserved for sparsity
// res(6) = 1-accuracy

void measureAccuracy(Vector<double> &res, const Vector<gentype> &resg, const Vector<gentype> &resh, const Vector<gentype> &ytarg, const Vector<int> &dstat, const ML_Base &ml);

// Bootstrap: this simplified version simply takes the indices of those points d(i) != 0 (of which there are m), randomly
// selects m of these (with repetition allowed), then sets d(i) = 0 for those not in the list so selected.

void bootstrapML(ML_Base &ml);

#endif

