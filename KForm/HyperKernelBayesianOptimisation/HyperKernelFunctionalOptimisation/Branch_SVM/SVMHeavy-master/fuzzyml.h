
//
// Fuzzy weight selection for MLs
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _fuzzyml_h
#define _fuzzyml_h

#include "ml_base.h"

// Given an ML, a fuzzy function, a kernel function and various parameters,
// this will calculate fuzzy weights for the ML and apply them to it.
//
// So-called fuzzy MLs use functions inspired by fuzzy logic to set the
// individual C and epsilon weights for each training vector based on some
// estimate of how much a particular vector "belongs" to its class.
// The degree of belonging is calculated by the membership function,
// typically based on the relative distances to the class centre of the
// class to which the vector belongs and the distances to other classes.
// The inbuilt membership functions are:
//
// q1 = 0.5+((exp(f*(d_d-d_l)/d)-exp(-f))/(2*(exp(f)-exp(-f))))
// q2 = ((2*(0.5+((exp(f*(d_d-d_l)/d)-exp(-f))/(2*(exp(f)-exp(-f))))))-1)^m
// q3 = 0.5+((1-(d_l/(r_l+f)))/2)
// q4 = 0.5*(1+tanh(f*((2*g_x)+m)))
//
// (Keller and Hunt, modified Keller and Hunt, Lin and Wang, and
// cluster-based).  In all cases, for each training vector pair (x,y):
//
// q   = var(2,0)  = either t (C weight) or s (epsilon weight), pre-fuzzing.
// d_l = var(2,1)  = distance from x to the mean of class y.
// d_d = var(2,2)  = min distance from x to the mean of any other class !y.
// d   = var(2,3)  = distance between the mean of classes y and !y.
// r_l = var(2,4)  = radius of smallest sphere centred at mean of class y.
//                   containing all elements of class y.
// r_d = var(2,5)  = radius of smallest sphere centred at mean of class !y.
//                   containing all elements of class !y.
// g_x = var(2,6)  = output of 1-class SVM trained with all vectors of class y.
// q1  = var(2,7)  = Keller and Hunt membership.
// q2  = var(2,8)  = Modified Keller and Hunt membership.
// q3  = var(2,9)  = Lin and Wang membership.
// q4  = var(2,10) = cluster-based membership.
// f   = var(3,0)  = user parameters set below.
// m   = var(3,1)  = user parameters set below.
// nu  = var(3,2)  = nu value used for clustering.
//
// Returns 0 on success, nz on failure
//
// setCoreps = 1: set C weights, 0 set eps weights


int calcFuzzML(ML_Base &ml, const gentype &fuzzfn, const SparseVector<SparseVector<gentype> > &argvariables, const MercerKernel &distkern, double f, double m, double nu, int setCoreps);

#endif

