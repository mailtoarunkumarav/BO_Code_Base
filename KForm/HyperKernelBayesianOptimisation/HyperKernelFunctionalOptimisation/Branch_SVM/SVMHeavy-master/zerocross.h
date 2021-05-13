
//
// Line minimiser
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _zerocross_h
#define _zerocross_h

// Find the minimum of the function:
//
// Q = (1/2).etaij.s^2 + eij.s + epsiloni.sqrt(s^2 + 2.thetai.s + normalphai )
//                             + epsilonj.sqrt(s^2 - 2.thetaj.s + normalphaj )
//                             - (1/t).ln( CCNi - sqrt(s^2 + 2.thetai.s + normalphai ) )
//                             - (1/t).ln( CCNj - sqrt(s^2 - 2.thetaj.s + normalphaj ) )
//
// which is equivalent to finding the zero crossing of the function:
//
// etaij.s + eij + ri(s).qi(s) + rj(s).qj(s)
//
// where:
//
// ri(s) = epsiloni + 1/( t.( CCNi - sqrt(s^2 + 2.thetai.s + normalphai ) ) )
// rj(s) = epsilonj + 1/( t.( CCNj - sqrt(s^2 - 2.thetaj.s + normalphaj ) ) )
//
// qi(s) = ( s + thetai ) / sqrt(s^2 + 2.thetai.s + normalphai )
//         sgn( s - absalphai ) if thetai < -absalphai+iota
//
// qj(s) = ( s - thetaj ) / sqrt(s^2 - 2.thetaj.s + normalphaj )
//         sgn( s - absalphaj ) if thetai > absalphaj-iota
//
// to within kappa, where the range is:
//
// smin <= s <= smax
//
// and the algorithm is terminated if more that maxitcnt steps are required.
//
// It is assumed that the minima satisfies:
//
// s >= smin
//
// but not necessarily the upper bound.
//
// NB: will not terminate until it finds a point s such that Q(s) < Q(0)


double findZeroCross(double etaij, double eij, double epsiloni, double epsilonj, double thetai, double normalphai, double absalphai, double thetaj, double normalphaj, double absalphaj, double iota, double smin, double smax, int maxitcnt, double kappa, int isidiscont, int isjdiscont, double CCNi, double CCNj, double t, double ztol);


#endif
