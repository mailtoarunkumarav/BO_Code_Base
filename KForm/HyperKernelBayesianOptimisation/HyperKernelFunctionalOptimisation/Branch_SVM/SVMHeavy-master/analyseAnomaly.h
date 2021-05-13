
//
// Anomaly analysis and new class creation
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _analyseAnomaly_h
#define _analyseAnomaly_h

#include "ml_mutable.h"

// Given an ML (assumed classifier with anomaly detection) analyse the set
// of vectors x (unlabelled) to find anonalies:
//
// - Those classified as not anomalous and for whom the distance into the
//   class is greater than (>=) nadist will be assigned the class found.
// - Those classified as not anomalous with distance less than nadist
//   will be assigned anomaly class.
// - Those classified as anomalous with distance into anomaly class less
//   than adist will be assigned anomaly class.
// - Those classified as anomalous with distance into anomaly class greater
//   than (>=) adist *may* be assigned a new class label (Mclass) if the
//   trigger conditions are satisfied, otherwise assigned anomaly class.
//
// Criterium for new class detection are that there must be at least Nnew
// vectors lying at least adist from the boundary of the anomaly region.
// If this trigger condition is met then the new class number created is
// Mclass and those vectors labelled anomalous with distance greater than
// (>=) adist will be assigned class label Mclass.
//
// Arguments are:
//
// ml:           machine.
// x:            unlabelled training data
// zassign:      destination for assigned classes
// trigVect:     indices of training vectors x assigned new label Mclass
//               zassign(trigVect) = Mclass
// anomVect:     indices of training vectors x assigned to the anomaly class
//               zassign(anomVect) = anomalyClass (hence not added to ML)
// addVect:      indices of all training vectors x added to the ML (not anom)
// Nnew:         trigger count for class creation.  Zero to disable.
// adist:        trigger anonaly detection distance
// nadist:       boundary for non-anomaly class assignment.
// Mclass:       new class to be added if detected.
// addVectsToML: set true if vectors are to be incorporated into the ML,
//               false otherwise (1 to add non-anomaly, 2 to add re-labelled
//               anomaly, 3 to add both).
// anomalyClass: label for anomalies in the ml.  If 0 then the label is
//               actually taken from the ml using the anomalyClass() member
//               function (standard behaviour).
//
// Note that the ml is not trained after addition of new vectors.
//
// NB: the vector x will be destroyed by this operation (qadd is used for
//     speed and memory preservation).


int anAnomalyCreate(ML_Mutable &ml, Vector<SparseVector<gentype> > &x, Vector<gentype> &zassign, Vector<int> &trigVect, Vector<int> &anomVect, Vector<int> &addVect, int Nnew, double adist, double nadist, int Mclass, int addVectsToML = 3, int anomalyClass = 0);

#endif

