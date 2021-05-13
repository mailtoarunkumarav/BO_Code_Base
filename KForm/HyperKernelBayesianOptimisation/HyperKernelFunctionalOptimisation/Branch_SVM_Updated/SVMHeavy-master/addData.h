
//
// Data loading functions
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _addData_h
#define _addData_h

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <string>
#include <math.h>
#include "ml_base.h"


// Training data extraction
// ========================
//
// addtrainingdata: function to add training data from a file to a ML.
//
// currcommand: command vector, only used when reporting errors.
// mlbase: the actual ML.
// trainfile: name of training file
// reverse: 0 means target at start, 1 means target at end
// ignoreStart: the number of training vectors to ignore before beginning to
//              add training vectors
// imax: maximum number of training vectors to add.  Set -1 for unlimited.  If
//       there are fewer than imax vectors available then add all available.
// ibase: point within the existing training set at which the new training
//        vectors will be inserted.  Set -1 to add to end of existing training
//        set.
// uselinesvector: if non-zero then add only select lines from the training
//                 file are added.  These lines are defined by the linesread
//                 argument.  Effectively the function acts as though only
//                 those lines specified in linesread are present in the 
//                 training file.
//                 If uselinesvector == 2 then linesread is returned to its
//                 original state before return.
// coercetosingle: if non-zero and the ML is of type ML_Single then the
//                 training file is read as a binary/multiclass/scalar training
//                 file and the class/target is disgarded.
// coercefromsingle: if non-zero and the ML is not of type ML_Single then the
//                   training file is read as a single-class (un-labelled)
//                   training file and the class is set to fromsingletarget (in
//                   the case of binary/multiclass classification) or
//                   fromsingletar (in the case of scalar regression).
// fromsingletarget: see coercefromsingle for explanation.
// fromsingletar: see coercefromsingle for explanation.
// fromsinglevec: see coercefromsingle for explanation.
// linesread: see uselinesvector for explanation.
//
// binaryRelabel: 0:      means do nothing
//                d != 0: replace y = d with +1, y != d with -1
//
// singleDrop: 0:  do nothing
//             nz: skip all vectors with this label.
//
// loadFileForHillClimb: this is like addtrainingdata, but rather than adding
//                       data to the ML it instead puts it into vectors for
//                       later use.  Note the lack of coercion arguments.
//
// Return value: number of points added

int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemp, const std::string &trainfile, int reverse, int ignoreStart, int imax, int ibase);

int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemp, Vector<SparseVector<gentype> > &x, const Vector<gentype> &y,                                     int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget);
int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemp, Vector<SparseVector<gentype> > &x, const Vector<gentype> &y, const Vector<gentype> &sigmaweight, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget);
int addtrainingdata(ML_Base &mlbase, const SparseVector<gentype> &xtemp, const std::string &trainfile, int reverse, int ignoreStart, int imax, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget, int binaryRelabel, int singleDrop, int uselinesvector, Vector<int> &linesread);

int loadFileForHillClimb(const ML_Base &mlbase, const SparseVector<gentype> &xtemp, const std::string &trainfile, int reverse, int ignoreStart, int imax,            int coercetosingle, int coercefromsingle, const gentype &fromsingletarget, int binaryRelabel, int singleDrop, int uselinesvector, Vector<int> &linesread, Vector<SparseVector<gentype> > &xtest, Vector<gentype> &ytest);
int loadFileAndTest(     const ML_Base &mlbase, const SparseVector<gentype> &xtemp, const std::string &trainfile, int reverse, int ignoreStart, int imax, int ibase, int coercetosingle, int coercefromsingle, const gentype &fromsingletarget, int binaryRelabel, int singleDrop, int uselinesvector, Vector<int> &linesread,                                        Vector<gentype> &ytest, Vector<gentype> &ytestresh, Vector<gentype> &ytestresg, Vector<gentype> &gvarres, int dovartest, Vector<int> &outkernind);

// Add point to basis

int addbasisdataUU(ML_Base &dest, const std::string &fname);
int addbasisdataVV(ML_Base &dest, const std::string &fname);

// x templating
// ============
//
// Writes (but never over-writes) parts of x with contents of xtemp

SparseVector<gentype> &addtemptox(SparseVector<gentype> &x, const SparseVector<gentype> &xtemp);
Vector<SparseVector<gentype> > &addtemptox(Vector<SparseVector<gentype> > &x, const SparseVector<gentype> &xtemp);



#endif
