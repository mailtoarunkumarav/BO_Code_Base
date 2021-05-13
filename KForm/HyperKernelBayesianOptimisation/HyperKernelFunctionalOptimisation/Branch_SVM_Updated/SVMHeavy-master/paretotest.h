
//
// Pareto Test Functions as per:
//
// Deb, Thiele, Laumanns, Zitzler (DTLZ)
// Scalable Test Problems for Evolutionary Multiobjective Optimisation
//
// Version: 
// Date: 1/12/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _paretotest_h
#define _paretotest_h

#include "vector.h"


// fnnum values (acronyms as per DTLZ)
//
// 1: DTLZ1
// 2: DTLZ2
// 3: DTLZ3
// 4: DTLZ4
// 5: DTLZ5
// 6: DTLZ6
// 7: DTLZ7
// 8: DTLZ8
// 9: DTLZ9
// 10: FON1 (6.1)
// 11: SCH1: f1(x) = x^2, f2(x) = (x-2)^2
// 12: SCH2: f1(x) = -x if x<=1, x-2 if 1<x<=3, 4-x if 3<x<=4, x-4 if x>4, f2(x) = (x-5)^2
//
// - n is the dimension of decision space
// - M is the dimension of target space
//
// - SCH1 has n = 1, M = 2
//
// - FON1 has range x in [-4,4]
// - FON1 has n arbitrary, M = 2
//
// - DTLZn has range x in [0,1]
// - DTLZn has n arbitrary, M <= n
//
// - alpha is used by DTLZ4
//
// - Return value is 0 for feasible in objective space, 1 if non-feasible
//   (non-feasible return is applicable only to DTLZ8 and DTLZ9)
//
//  Deb, Kalyanmoy and Thiele, Lothar and Laumanns, Marco and Zitzler, Eckart
//  - "Scalable Test Problems for Evolutionary Multiobjective Optimization"
//
// TO IMPLEMENT: Kursawe, SCH2, Poloni, Viennet



int evalTestFn(int fnnum, int n, int M, Vector<double> &res, const Vector<double> &x, double alpha = 100);


#endif

